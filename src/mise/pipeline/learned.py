"""Join camera solves of overlapping stretches of one walkthrough into one model.

A learned model (scripts/colab/learned_recon.py) is most accurate on a stretch of a minute or
two; over a whole tour it bends. So the tour is solved in overlapping windows and joined here.
The join does not trust camera centers alone (when the phone turns on the spot they bunch
up): for every frame two windows share, the same pixels are unprojected through each
window's depth and pose, and the similarity (scale, rotation, offset) that brings one cloud
onto the other is fitted to those thousands of point pairs, trimming the outliers.

Each window joins the model built so far. Where windows share a frame, the copy from the
window where the frame sits further from the window's ends is kept (it had more context).
The result has the layout `mise scan --poses` takes: frames.json, sparse/0 (COLMAP text),
depth.npz and meta.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    name: str
    t: float
    c2w: np.ndarray  # 4x4, OpenCV camera axes
    K: np.ndarray  # 3x3 at the frame's size
    size: tuple[int, int]
    depth: np.ndarray | None  # small depth map (z), see Solve.depth_px
    conf: np.ndarray | None
    points: np.ndarray  # Nx3 world points observed in this frame
    uv: np.ndarray  # Nx2 their pixels in the frame
    rgb: np.ndarray  # Nx3 uint8
    edge: float = 0.0  # seconds to the nearer end of its window


@dataclass
class Solve:
    path: Path
    frames: dict[str, Frame]
    depth_px: np.ndarray  # frame pixels per depth-map pixel (x, y)
    meta: dict = field(default_factory=dict)


def load_solve(path: Path) -> Solve:
    import pycolmap

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    times = {k["name"]: k["t"] for k in json.loads((path / "frames.json").read_text())}
    rec = pycolmap.Reconstruction(str(path / "sparse" / "0"))
    dz = np.load(path / "depth.npz") if (path / "depth.npz").exists() else None
    dmap = {str(n): i for i, n in enumerate(dz["names"])} if dz is not None else {}
    # an npz decompresses an array on every access: read each once
    dall, call = (dz["depth"], dz["conf"]) if dz is not None else (None, None)
    t0, t1 = min(times.values()), max(times.values())
    frames = {}
    pts3d = rec.points3D  # one copy: the binding builds the whole map on every access
    cams = rec.cameras
    for im in rec.images.values():
        cam = cams[im.camera_id]
        fx, fy, cx, cy = cam.params[:4] if len(cam.params) >= 4 else (cam.params[0], cam.params[0], *cam.params[1:3])
        cw = im.cam_from_world()
        R, t = cw.rotation.matrix(), np.asarray(cw.translation)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        obs = [(p.xy, pts3d[p.point3D_id]) for p in im.points2D if p.has_point3D()]
        k = dmap.get(im.name)
        frames[im.name] = Frame(
            name=im.name, t=times[im.name], c2w=c2w, K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
            size=(cam.width, cam.height),
            depth=dall[k].astype(np.float32) if k is not None else None,
            conf=call[k].astype(np.float32) if k is not None else None,
            points=np.array([o[1].xyz for o in obs]).reshape(-1, 3), uv=np.array([o[0] for o in obs]).reshape(-1, 2),
            rgb=np.array([o[1].color for o in obs], np.uint8).reshape(-1, 3),
            edge=min(times[im.name] - t0, t1 - times[im.name]))
    px = np.asarray(dz["scale_xy"], float) if dz is not None else np.ones(2)
    return Solve(path=path, frames=frames, depth_px=px, meta=meta)


def unproject(fr: Frame, depth_px: np.ndarray, ij: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """World points of depth-map pixels (rows, cols) of a frame."""
    i, j = ij
    z = fr.depth[i, j]
    u, v = j * depth_px[0], i * depth_px[1]
    cam = np.stack([(u - fr.K[0, 2]) / fr.K[0, 0] * z, (v - fr.K[1, 2]) / fr.K[1, 1] * z, z], 1)
    return cam @ fr.c2w[:3, :3].T + fr.c2w[:3, 3]


def correspondences(a: Frame, pa: np.ndarray, b: Frame, pb: np.ndarray, per_frame: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The same pixels of one frame as two solves place them in their worlds."""
    if a.depth is None or b.depth is None or a.depth.shape != b.depth.shape:
        return np.zeros((0, 3)), np.zeros((0, 3))
    ok = (a.depth > 0) & (b.depth > 0) & np.isfinite(a.depth) & np.isfinite(b.depth)
    if ok.sum() < 50:
        return np.zeros((0, 3)), np.zeros((0, 3))
    ok &= (a.conf >= np.median(a.conf[ok])) & (b.conf >= np.median(b.conf[ok]))
    i, j = np.nonzero(ok)
    if len(i) > per_frame:
        k = rng.choice(len(i), per_frame, replace=False)
        i, j = i[k], j[k]
    return unproject(a, pa, (i, j)), unproject(b, pb, (i, j))


def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimizing |s R x + t - y|."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean())
    return s, R, ym - s * R @ xm


def robust_similarity(X: np.ndarray, Y: np.ndarray, rounds: int = 6) -> tuple[float, np.ndarray, np.ndarray, dict]:
    keep = np.ones(len(X), bool)
    for _ in range(rounds):
        s, R, t = umeyama(X[keep], Y[keep])
        res = np.linalg.norm(s * X @ R.T + t - Y, axis=1)
        keep = res <= max(0.05, 2.5 * np.median(res[keep]))
    return s, R, t, {"pairs": int(len(X)), "used": int(keep.sum()), "scale": round(s, 4),
                     "median_cm": round(100 * float(np.median(res[keep])), 1),
                     "all_median_cm": round(100 * float(np.median(res)), 1)}


def level_rotation(frames: dict[str, Frame], rng: np.random.Generator) -> np.ndarray | None:
    """World -> leveled rotation of one solve: floor normal onto +z, then walls onto x/y."""
    from . import normalize, ply

    pts = np.vstack([f.points for f in frames.values() if len(f.points)])
    if len(pts) < 500:
        return None
    if len(pts) > 40000:
        pts = pts[rng.choice(len(pts), 40000, replace=False)]
    up = -np.mean([f.c2w[:3, 1] for f in frames.values()], axis=0)  # camera y points down
    up /= np.linalg.norm(up)
    try:
        floor = normalize.ransac_plane(pts, iters=600, thresh=0.04, up_hint=up, max_tilt_deg=35, rng=rng)
    except Exception:  # noqa: BLE001
        return None
    n = floor.n if floor.n @ up > 0 else -floor.n
    R1 = ply.rotation_between(n, np.array([0.0, 0, 1]))
    p1 = pts @ R1.T
    h = p1[:, 2] - np.percentile(p1[:, 2], 2)
    band = (h > 0.3) & (h < 2.0)
    if band.sum() < 200:
        return R1
    return ply.rot_z(normalize.manhattan_yaw(p1[band, :2], bin_size=0.03)) @ R1


def square_up(R: np.ndarray, R_ref: np.ndarray, R_new: np.ndarray) -> tuple[np.ndarray, float]:
    """Correct a join's rotation R (new world -> reference world) so the new solve's floor and
    walls line up with the reference's: same up, wall directions a multiple of 90 degrees apart.
    Rooms of one home share a floor and (nearly always) wall directions, and the join's own
    rotation, fitted on a few seconds of shared frames, is its least certain part."""
    from . import ply

    D = R_ref @ R @ R_new.T  # the new solve's level axes, as seen in the reference's level frame
    tilt = ply.rotation_between(D[:, 2], np.array([0.0, 0, 1]))
    D1 = tilt @ D
    yaw = np.degrees(np.arctan2(D1[1, 0], D1[0, 0]))
    snap = ply.rot_z(-(yaw - 90 * np.round(yaw / 90)))
    E = snap @ tilt
    angle = float(np.degrees(np.arccos(np.clip((np.trace(E) - 1) / 2, -1, 1))))
    return R_ref.T @ E @ R_ref @ R, angle


def _apply(fr: Frame, s: float, R: np.ndarray, t: np.ndarray) -> Frame:
    c2w = np.eye(4)
    c2w[:3, :3] = R @ fr.c2w[:3, :3]
    c2w[:3, 3] = s * R @ fr.c2w[:3, 3] + t
    return Frame(name=fr.name, t=fr.t, c2w=c2w, K=fr.K, size=fr.size,
                 depth=None if fr.depth is None else fr.depth * s, conf=fr.conf,
                 points=s * fr.points @ R.T + t, uv=fr.uv, rgb=fr.rgb, edge=fr.edge)


def join_solves(paths: list[Path], per_frame: int = 1500, seed: int = 0, square: bool = False,
                onto: Path | None = None) -> tuple[dict[str, Frame], np.ndarray, list[dict]]:
    """Frames of all solves in one world, and a report per join.

    Without `onto`, solves are chained in time: each joins the ones before it, through the frames
    they share (a few seconds at a doorway), in the first one's world.
    With `onto` (a solve of the whole walk, the skeleton), each solve joins the skeleton alone
    through every frame it covers: the skeleton says where each room is, the room's own solve
    gives its shape. Frames no room solve covers keep the skeleton's pose.
    """
    rng = np.random.default_rng(seed)
    solves = sorted((load_solve(p) for p in paths), key=lambda s: min(f.t for f in s.frames.values()))
    if onto is not None:
        skel = load_solve(onto)
        target, px = skel.frames, skel.depth_px
        merged: dict[str, Frame] = {}
        reports = [{"solve": str(skel.path), "frames": len(skel.frames), "skeleton": True}]
        todo = solves
    else:
        first = solves[0]
        merged = dict(first.frames)
        target, px = merged, first.depth_px
        reports = [{"solve": str(first.path), "frames": len(first.frames), "reference": True}]
        todo = solves[1:]
    for sv in todo:
        shared = sorted(set(target) & set(sv.frames))
        X, Y = [], []
        for n in shared:
            y, x = correspondences(target[n], px, sv.frames[n], sv.depth_px, per_frame, rng)
            X.append(x)
            Y.append(y)
        X, Y = np.vstack(X) if X else np.zeros((0, 3)), np.vstack(Y) if Y else np.zeros((0, 3))
        if len(X) < 100:
            raise RuntimeError(f"{sv.path} shares {len(shared)} frames with the solves before it: too few to join")
        s, R, t, rep = robust_similarity(X, Y)
        if square:
            R_ref, R_new = level_rotation(target, rng), level_rotation(sv.frames, rng)
            if R_ref is not None and R_new is not None:
                R, turned = square_up(R, R_ref, R_new)
                t, keep = np.zeros(3), np.ones(len(X), bool)
                for _ in range(4):  # the offset again, with the squared rotation
                    t = np.median(Y[keep] - s * X[keep] @ R.T, axis=0)
                    res = np.linalg.norm(s * X @ R.T + t - Y, axis=1)
                    keep = res <= max(0.05, 2.5 * np.median(res[keep]))
                rep.update({"squared_deg": round(turned, 2), "median_cm": round(100 * float(np.median(res[keep])), 1)})
        moved = {n: _apply(f, s, R, t) for n, f in sv.frames.items()}
        cams = np.array([[np.linalg.norm(moved[n].c2w[:3, 3] - target[n].c2w[:3, 3]),
                          np.degrees(np.arccos(np.clip((np.trace(moved[n].c2w[:3, :3].T @ target[n].c2w[:3, :3]) - 1) / 2, -1, 1)))]
                         for n in shared])
        rep.update({"solve": str(sv.path), "frames": len(sv.frames), "shared": len(shared),
                    "camera_gap_cm": round(100 * float(np.median(cams[:, 0])), 1),
                    "camera_turn_deg": round(float(np.median(cams[:, 1])), 2)})
        reports.append(rep)
        for n, f in moved.items():
            if n not in merged or f.edge > merged[n].edge:
                merged[n] = f
    if onto is not None:
        gaps = {n: f for n, f in target.items() if n not in merged}
        merged.update(gaps)
        reports[0]["frames_used"] = len(gaps)
    return merged, px, reports


def write_joined(frames: dict[str, Frame], depth_px: np.ndarray, reports: list[dict], out: Path, meta: dict) -> Path:
    out = Path(out)
    model = out / "sparse" / "0"
    model.mkdir(parents=True, exist_ok=True)
    order = sorted(frames.values(), key=lambda f: f.t)
    pid = 1
    with (model / "cameras.txt").open("w") as fc, (model / "images.txt").open("w") as fi, \
            (model / "points3D.txt").open("w") as fp:
        fc.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        fi.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        fp.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for iid, f in enumerate(order, start=1):
            K = f.K
            fc.write(f"{iid} PINHOLE {f.size[0]} {f.size[1]} {K[0, 0]:.4f} {K[1, 1]:.4f} {K[0, 2]:.4f} {K[1, 2]:.4f}\n")
            Rwc = f.c2w[:3, :3].T
            q = _quat_wxyz(Rwc)
            tw = -Rwc @ f.c2w[:3, 3]
            obs = []
            for j, (X, uv, c) in enumerate(zip(f.points, f.uv, f.rgb)):
                fp.write(f"{pid} {X[0]:.5f} {X[1]:.5f} {X[2]:.5f} {c[0]} {c[1]} {c[2]} 1.0 {iid} {j}\n")
                obs.append(f"{uv[0]:.2f} {uv[1]:.2f} {pid}")
                pid += 1
            fi.write(f"{iid} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {tw[0]:.6f} {tw[1]:.6f} {tw[2]:.6f} {iid} {f.name}\n"
                     f"{' '.join(obs)}\n")
    with_depth = [f for f in order if f.depth is not None]
    if with_depth:
        np.savez_compressed(out / "depth.npz", names=np.array([f.name for f in with_depth]),
                            depth=np.stack([f.depth for f in with_depth]).astype(np.float16),
                            conf=np.stack([f.conf for f in with_depth]).astype(np.float16), scale_xy=depth_px)
    (out / "frames.json").write_text(json.dumps([{"name": f.name, "t": f.t} for f in order]))
    (out / "meta.json").write_text(json.dumps({**meta, "frames": len(order), "points": pid - 1, "joins": reports}, indent=1))
    return out


def _quat_wxyz(m: np.ndarray) -> np.ndarray:
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def join(paths: list[Path], out: Path, square: bool = False, onto: Path | None = None) -> dict:
    """Join the solves in `paths` (each a learned_recon.py output) into `out`; returns meta.json.
    square: also line up each solve's floor and walls with the reference's (see square_up).
    onto: a solve of the whole walk to place every solve on (see join_solves)."""
    frames, px, reports = join_solves(paths, square=square, onto=onto)
    metas = [load_solve(p).meta for p in paths]
    meta = {"model": metas[0].get("model"), "model_id": metas[0].get("model_id"),
            "metric": all(m.get("metric") for m in metas), "joined_from": [str(p) for p in paths]}
    if onto is not None:  # every solve took the skeleton's scale
        meta.update({"metric": bool(load_solve(onto).meta.get("metric")), "onto": str(onto)})
    write_joined(frames, px, reports, out, meta)
    return json.loads((Path(out) / "meta.json").read_text())


def calibrate(frames_dir: Path, work: Path, t0: float = 0.0, t1: float = 60.0, min_frames: int = 30) -> dict:
    """The camera's intrinsics from a COLMAP solve (CPU) of one stretch of the picked frames.

    A learned model told the real focal length builds rooms the right size; left to guess, it
    was 7% off on a phone video and the rooms came out 6% small across. Bundle adjustment over
    a well-textured stretch where the phone moves (not just turns) pins the focal length down.
    Uses the scan's own solve (contrast-boosted frames, neighbor + revisit matching). Writes
    frames_dir/intrinsics.json and returns it.
    """
    import shutil

    from .scan import ScanEvents, ScanOptions, _features, _solve

    frames_dir, work = Path(frames_dir), Path(work)
    kept = [k for k in json.loads((frames_dir / "frames.json").read_text()) if t0 <= k["t"] <= t1]
    if len(kept) < min_frames:
        raise RuntimeError(f"only {len(kept)} frames in {t0}-{t1} s to calibrate from")
    if work.exists():
        shutil.rmtree(work)
    (work / "frames").mkdir(parents=True)
    for k in kept:
        shutil.copy2(frames_dir / "frames" / k["name"], work / "frames" / k["name"])
    ev = ScanEvents()
    db = _features(work, kept, ScanOptions(scene_id="calibrate"), ev)
    recs = _solve(work, db, len(kept), ev)
    if not recs or recs[0].num_reg_images() < min_frames:
        n = recs[0].num_reg_images() if recs else 0
        raise RuntimeError(f"only {n} of {len(kept)} frames solved in {t0}-{t1} s: calibrate on a stretch "
                           "where the phone walks past textured things")
    best = recs[0]
    cam = next(iter(best.cameras.values()))
    fx, fy, cx, cy, k1, k2, p1, p2 = (float(v) for v in cam.params[:8])
    out = {"fx": round(fx, 2), "fy": round(fy, 2), "cx": round(cx, 2), "cy": round(cy, 2), "k1": round(k1, 4),
           "k2": round(k2, 4), "p1": round(p1, 4), "p2": round(p2, 4), "width": cam.width, "height": cam.height,
           "source": f"COLMAP self-calibration of {best.num_reg_images()} of {len(kept)} frames, {t0:g}-{t1:g} s"}
    (frames_dir / "intrinsics.json").write_text(json.dumps(out, indent=1))
    return out
