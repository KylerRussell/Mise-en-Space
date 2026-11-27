"""The empty home: a scanned splat without the things in it.

Each Gaussian carries the class the frames that see it voted for (splat_train.py --labels, from
segment_frames.py). The home's shell (walls, floor, ceiling, doors, and what is set into them:
windows, mirrors) stays; beds, clothes, cabinets, bottles... go, and so does anything outside the
traced rooms (what showed through windows and in mirrors). Where things stood on the floor, the
floor is filled by copying the nearest intact floor (decompose.fill_floor_plane, the plane +
texture-copy baseline of the plan). Walls behind furniture are left as they are for now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import decompose, ply
from .structure import SEE_THROUGH, STRUCTURE

KEEP = STRUCTURE | SEE_THROUGH


def _grid_mask(polys: list[np.ndarray], lo: np.ndarray, shape: tuple[int, int], cell: float, grow: float):
    import cv2

    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round((np.asarray(p) - lo) / cell).astype(np.int32)], 1)
    k = max(1, int(round(grow / cell)))
    return cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) if grow > 0 else m


def empty_room(s: ply.Splat, cls: np.ndarray, classes: list[str], rooms: list[np.ndarray], margin: float = 0.3,
               cell: float = 0.05, hole_pad: float = 0.12, floor_z: float = 0.05) -> tuple[ply.Splat, dict]:
    """The splat without things (by class) and outside the rooms; floor holes where things stood filled."""
    import cv2

    xy = s.xyz[:, :2].astype(np.float64)
    keep_ids = np.array([n in KEEP for n in classes] + [True])  # index -1 (never seen) -> last: kept
    thing = ~keep_ids[np.where(cls < 0, len(classes), cls)]
    allp = np.vstack(rooms)
    lo = allp.min(0) - margin - 1.0
    shape = tuple((np.ceil((allp.max(0) + margin + 1.0 - lo) / cell)).astype(int)[::-1])
    home = _grid_mask(rooms, lo, shape, cell, margin)
    ij = np.floor((xy - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    inside = np.zeros(len(s), bool)
    inside[ok] = home[ij[ok, 1], ij[ok, 0]] > 0
    drop = thing | ~inside
    # footprints of what stood on the floor: removed things below 1 m, as cells, padded
    foot = np.zeros(shape, np.uint8)
    low = thing & inside & (s.xyz[:, 2] < 1.0)
    foot[ij[low, 1], ij[low, 0]] = 1
    k = max(1, int(round(hole_pad / cell)))
    foot = cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    foot = cv2.dilate(foot, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) & home
    n, lab, stats, _ = cv2.connectedComponentsWithStats(foot)
    holes = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] * cell * cell < 0.02:
            continue
        cs, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        holes.append((c + 0.5) * cell + lo)
    # the floor under the things goes too (their contact shadows), then is filled from intact floor
    under = np.zeros(len(s), bool)
    under[ok] = (foot[ij[ok, 1], ij[ok, 0]] > 0) & (np.abs(s.xyz[ok, 2]) < floor_z)
    out = s.subset(~(drop | under))
    filled = decompose.fill_floor_plane(out, holes, floor_z=floor_z)
    names, counts = np.unique(np.array(classes + ["unseen"])[np.where(cls < 0, len(classes), cls)][thing], return_counts=True)
    top = np.argsort(-counts)[:10]
    return filled, {"gaussians": len(s), "things": int(thing.sum()), "outside": int((~inside & ~thing).sum()),
                    "floor_holes": len(holes), "floor_hole_m2": round(float(foot.sum()) * cell * cell, 2),
                    "kept": len(out), "filled_with": len(filled) - len(out),
                    "removed_classes": {str(names[i]): int(counts[i]) for i in top}}


def empty_scene(scene_dir: Path, splat_dir: Path, name: str = "empty.ply") -> dict:
    """Write scene_dir/splats/<name> (the empty home) from a trained, labelled splat."""
    from ..scene.models import Scene

    scene_dir, splat_dir = Path(scene_dir), Path(splat_dir)
    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    s = ply.read_ply(splat_dir / "splat.ply")
    z = np.load(splat_dir / "gauss_labels.npz")
    classes = [str(c) for c in z["classes"]]
    rooms = [np.array(sp.polygon, float) for sp in scene.room.spaces]
    out, rep = empty_room(s, z["cls"].astype(int), classes, rooms)
    (scene_dir / "splats").mkdir(exist_ok=True)
    ply.write_ply(scene_dir / "splats" / name, out)
    rep["path"] = f"splats/{name}"
    return rep


def label_gaussians(s: ply.Splat, cameras: list[dict], labels: dict[str, np.ndarray], n_classes: int,
                    grid: tuple[int, int] = (160, 90), tol: float = 0.06, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Each Gaussian's class by vote of the frames that see it (CPU).

    Per frame, every center is projected; on a coarse z-buffer only the centers within `tol`
    (plus 2% of the distance) of the nearest at their cell count as seen, and vote for the
    label under them. Returns (class per Gaussian, -1 when never seen; votes for it).
    """
    X = s.xyz.astype(np.float64)
    keep_o = 1 / (1 + np.exp(-s.opacity)) > 0.2 if s.opacity is not None else np.ones(len(X), bool)
    gw, gh = grid
    votes = np.zeros((len(X), n_classes), np.int32)
    for cam in cameras[::stride]:
        m = labels.get(cam["name"])
        if m is None:
            continue
        V, K = np.array(cam["w2c"]), np.array(cam["K"])
        W, H = cam["size"]
        pc = X @ V[:3, :3].T + V[:3, 3]
        z = pc[:, 2]
        ok = (z > 0.1) & keep_o
        u = K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        gx = (u[idx] * gw / W).astype(int)
        gy = (v[idx] * gh / H).astype(int)
        cell = gy * gw + gx
        zbuf = np.full(gw * gh, np.inf)
        np.minimum.at(zbuf, cell, z[idx])
        seen = z[idx] <= zbuf[cell] + tol + 0.02 * zbuf[cell]
        idx, u_, v_ = idx[seen], u[idx][seen], v[idx][seen]
        cl = m[(v_ * m.shape[0] / H).astype(int).clip(0, m.shape[0] - 1), (u_ * m.shape[1] / W).astype(int).clip(0, m.shape[1] - 1)]
        np.add.at(votes, (idx, cl.astype(int)), 1)
    total = votes.sum(1)
    cls = np.where(total > 0, votes.argmax(1), -1)
    return cls, total
