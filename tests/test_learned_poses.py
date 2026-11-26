"""Poses from a learned model (scripts/colab/learned_recon.py) into the scan pipeline.

A synthetic box room stands in for the model: exact depth maps and cameras, cut into two
chunks the second of which is in its own (scaled, turned, shifted) frame, as a second pass
through a feed-forward model would be. The chunks are joined, written as a COLMAP text model,
imported with `import_poses` and leveled: the cameras must come back at their true height.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")
cv2 = pytest.importorskip("cv2")

from mise.pipeline import sfm_scene  # noqa: E402
from mise.pipeline.scan import import_poses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("learned_recon", ROOT / "scripts" / "colab" / "learned_recon.py")
lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lr)

ROOM = np.array([4.0, 3.0, 2.5])  # box room, z up, floor at 0
H, W = lr.MODEL_H, lr.MODEL_W
K = np.array([[400.0, 0, W / 2], [0, 400.0, H / 2], [0, 0, 1]])


def _c2w(center, yaw_deg, pitch_deg=-25):
    """A handheld phone: turned by yaw, tilted down a little so it sees the floor."""
    cp = np.cos(np.radians(pitch_deg))
    f = np.array([cp * np.cos(np.radians(yaw_deg)), cp * np.sin(np.radians(yaw_deg)), np.sin(np.radians(pitch_deg))])
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(f, right)
    M = np.eye(4)
    M[:3, :3] = np.c_[right, down, f]  # OpenCV camera axes: x right, y down, z forward
    M[:3, 3] = center
    return M


def _render(c2w):
    """z-depth of the box room's walls, floor and ceiling at every pixel."""
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)  # z = 1
    d = rays @ c2w[:3, :3].T
    c = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ts = np.stack([np.where(d[..., i] > 0, (ROOM[i] - c[i]) / d[..., i], -c[i] / d[..., i]) for i in range(3)], -1)
    ts[~np.isfinite(ts) | (ts <= 0)] = np.inf
    return ts.min(-1)  # rays have unit z in the camera, so the ray parameter is the z-depth


def _frame(c2w):
    depth = _render(c2w)
    return {"c2w": c2w, "K": K.copy(), "depth": depth, "conf": np.ones_like(depth), "valid": np.isfinite(depth),
            "rgb": np.full((H, W, 3), 128, np.uint8)}


def _similarity(fr, s, R, t):
    """The same frame as seen by a pass whose world is x' = (R^T (x - t)) / s."""
    M = np.eye(4)
    M[:3, :3] = R.T @ fr["c2w"][:3, :3]
    M[:3, 3] = R.T @ (fr["c2w"][:3, 3] - t) / s
    return {**fr, "c2w": M, "depth": fr["depth"] / s}


def test_learned_chunks_join_import_and_level(tmp_path):
    n, chunk, overlap = 30, 18, 6
    truth = [_c2w([0.8 + 2.4 * i / (n - 1), 1.5 + 0.4 * np.sin(i / 4), 1.45 + 0.05 * np.cos(i / 3)], 20 * i) for i in range(n)]
    frames = [_frame(M) for M in truth]
    # chunk 2 comes out of the model in its own frame: half the scale, turned 40 deg about a tilted axis, shifted
    ax = np.array([0.3, 0.2, 1.0]) / np.linalg.norm([0.3, 0.2, 1.0])
    Rq = cv2.Rodrigues(ax * np.radians(40))[0]
    s_true, t_true = 2.0, np.array([0.5, -1.0, 0.3])
    first = {i: frames[i] for i in range(chunk)}
    second = {i: _similarity(frames[i], s_true, Rq, t_true) for i in range(chunk - overlap, n)}
    s, R, t, rep = lr.join({i: first[i] for i in range(chunk - overlap, chunk)}, second)
    assert s == pytest.approx(s_true, rel=1e-3) and rep["center_residual"] < 1e-3 and rep["rot_spread_deg"] < 0.01
    joined = {**{i: lr.apply(fr, s, R, t) for i, fr in second.items()}, **first}

    # write it as the Colab script does, with frames at 1280 x 720 next to it
    poses = tmp_path / "poses"
    fdir = tmp_path / "picked"
    fdir.mkdir()
    kept = [{"name": f"frame_{i:04d}.jpg", "t": 134 + i / 3} for i in range(n)]
    for k in kept:
        cv2.imwrite(str(fdir / k["name"]), np.full((720, 1280, 3), 90, np.uint8))
    w = lr.ColmapWriter(poses / "sparse" / "0", 1280, 720)
    rng = np.random.default_rng(0)
    for i in range(n):
        w.add(i + 1, kept[i]["name"], joined[i], 300, rng)
    w.close()
    (poses / "frames.json").write_text(json.dumps(kept))
    (poses / "meta.json").write_text(json.dumps({"model": "synthetic", "metric": True, "frames": n}))

    work = tmp_path / "work"
    meta = import_poses(poses, work, fdir)
    assert meta["metric"] and (work / "sparse" / "0" / "images.bin").exists()
    assert len(list((work / "frames").glob("*.jpg"))) == n and len(list((work / "thumbs").glob("*.jpg"))) == n
    rec = pycolmap.Reconstruction(str(work / "sparse" / "0"))
    assert rec.num_reg_images() == n
    ims = sorted(rec.images.values(), key=lambda im: im.name)
    centers = np.array([im.projection_center() for im in ims])
    assert np.abs(centers - np.array([M[:3, 3] for M in truth])).max() < 1e-3

    # every observation re-projects onto its pixel through the frame-size camera
    im = ims[n // 2]
    cam = rec.cameras[im.camera_id]
    assert (cam.width, cam.height) == (1280, 720)
    obs = [p for p in im.points2D if p.has_point3D()][:50]
    pc = np.array([im.cam_from_world() * rec.points3D[p.point3D_id].xyz for p in obs])
    uv = np.asarray(cam.img_from_cam(pc))
    assert np.abs(uv - np.array([p.xy for p in obs])).max() < 0.05

    # and the pipeline's leveling finds the floor: cameras at their true height, sane
    L = sfm_scene.level_reconstruction(rec, 1.0)
    assert np.median(L.cams[:, 2]) == pytest.approx(np.median([M[2, 3] for M in truth]), abs=0.03)
    assert L.pose_sanity()["ok"]
    assert len(L.sights[0]) > 100  # each camera sees the points sampled from its own depth


def test_import_names_missing_frames(tmp_path):
    poses = tmp_path / "poses"
    (poses / "sparse" / "0").mkdir(parents=True)
    (poses / "frames.json").write_text(json.dumps([{"name": "frame_0000.jpg", "t": 0.0}]))
    with pytest.raises(FileNotFoundError, match="not in"):
        import_poses(poses, tmp_path / "work", tmp_path)


def test_quaternion_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        R = cv2.Rodrigues(rng.normal(size=3))[0]
        q = lr.quat_wxyz(R)
        assert np.allclose(pycolmap.Rotation3d(np.array([q[1], q[2], q[3], q[0]])).matrix(), R, atol=1e-6)


def _write_solve(path, frames, kept, idx, metric=True):
    """A learned_recon.py output for frames idx, as the Colab script writes it."""
    w = lr.ColmapWriter(path / "sparse" / "0", 1280, 720)
    rng = np.random.default_rng(1)
    for i in idx:
        w.add(i + 1, kept[i]["name"], frames[i], 200, rng)
    w.close()
    np.savez_compressed(path / "depth.npz", names=np.array([kept[i]["name"] for i in idx]),
                        depth=np.stack([frames[i]["depth"][::2, ::2] for i in idx]).astype(np.float16),
                        conf=np.stack([frames[i]["conf"][::2, ::2] for i in idx]).astype(np.float16),
                        scale_xy=np.array([1280 / (W / 2), 720 / (H / 2)]))
    (path / "frames.json").write_text(json.dumps([kept[i] for i in idx]))
    (path / "meta.json").write_text(json.dumps({"model": "synthetic", "metric": metric}))


def test_join_windows_on_shared_pixels(tmp_path):
    from mise.pipeline.learned import join

    n = 30
    truth = [_c2w([0.8 + 2.4 * i / (n - 1), 1.5 + 0.4 * np.sin(i / 4), 1.45], 25 * i) for i in range(n)]
    frames = [_frame(M) for M in truth]
    frames = [{**f, "depth": np.where(np.isfinite(f["depth"]), f["depth"], 0.0)} for f in frames]
    kept = [{"name": f"frame_{i:04d}.jpg", "t": i / 3} for i in range(n)]
    ax = np.array([0.1, -0.3, 1.0]) / np.linalg.norm([0.1, -0.3, 1.0])
    Rq = cv2.Rodrigues(ax * np.radians(70))[0]
    second = [_similarity(f, 1.7, Rq, np.array([-2.0, 0.4, 0.9])) for f in frames]
    _write_solve(tmp_path / "a", frames, kept, range(0, 20))
    _write_solve(tmp_path / "b", second, kept, range(12, 30))
    meta = join([tmp_path / "b", tmp_path / "a"], tmp_path / "joined")  # order given doesn't matter
    rep = meta["joins"][1]
    assert meta["frames"] == n and meta["metric"]
    assert rep["shared"] == 8 and abs(rep["scale"] - 1.7) < 0.01 and rep["median_cm"] < 1.0 and rep["camera_gap_cm"] < 2.0
    rec = pycolmap.Reconstruction(str(tmp_path / "joined" / "sparse" / "0"))
    got = {im.name: im.projection_center() for im in rec.images.values()}
    assert max(np.linalg.norm(got[k["name"]] - truth[i][:3, 3]) for i, k in enumerate(kept)) < 0.01
    fdir = tmp_path / "picked"
    fdir.mkdir()
    for k in kept:
        cv2.imwrite(str(fdir / k["name"]), np.full((720, 1280, 3), 90, np.uint8))
    assert import_poses(tmp_path / "joined", tmp_path / "work", fdir)["metric"]


def test_box_of_known_size_sets_the_scale(tmp_path):
    """A 0.5 x 0.3048 m rim at 0.25 m, clicked in the frames of a solve that is 1.7x too small."""
    from mise.eval.floorplan import box_check
    from mise.pipeline.scan import ScanOptions, _scale_and_level

    n = 30
    truth = [_c2w([0.8 + 2.4 * i / (n - 1), 1.5 + 0.4 * np.sin(i / 4), 1.45], 25 * i) for i in range(n)]
    frames = [_frame(M) for M in truth]
    frames = [{**f, "depth": np.where(np.isfinite(f["depth"]), f["depth"], 0.0)} for f in frames]
    kept = [{"name": f"frame_{i:04d}.jpg", "t": i / 3} for i in range(n)]
    small = [_similarity(f, 1.7, np.eye(3), np.zeros(3)) for f in frames]
    _write_solve(tmp_path / "s", small, kept, range(n), metric=False)

    c, yaw, a, b = np.array([3.1, 2.5]), np.radians(20), 0.5, 0.3048
    u, v = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rim = {k: np.r_[c + sx * a / 2 * u + sy * b / 2 * v, 0.25]
           for k, (sx, sy) in {"FL": (-1, -1), "FR": (1, -1), "BR": (1, 1), "BL": (-1, 1)}.items()}
    Kf = np.diag([1280 / W, 720 / H, 1.0]) @ K
    clicks = {}
    for i, M in enumerate(truth):
        seen = {}
        for k, X in rim.items():
            x = M[:3, :3].T @ (X - M[:3, 3])
            uv = (Kf @ x)[:2] / x[2]
            if x[2] > 0.2 and 0 <= uv[0] < 1280 and 0 <= uv[1] < 720:
                seen[k] = uv.tolist()
        if len(seen) >= 2:
            clicks[kept[i]["name"]] = seen
    assert sum(len(s) for s in clicks.values()) >= 8

    rec = pycolmap.Reconstruction(str(tmp_path / "s" / "sparse" / "0"))
    L, _ = _scale_and_level(rec, ScanOptions(scene_id="t", scale=1.0))
    got = box_check(rec, L, {"known_m": 0.3048, "known_side": "short", "clicks": clicks})
    assert got["rim_rms_px"] < 1.0
    assert abs(got["factor"] - 1.7) < 0.017  # 1%


def test_rooms_placed_on_a_skeleton(tmp_path):
    """Two room solves, each in its own world, placed on a whole-walk solve through all their frames."""
    from mise.pipeline.learned import join

    n = 30
    truth = [_c2w([0.8 + 2.4 * i / (n - 1), 1.5 + 0.4 * np.sin(i / 4), 1.45], 25 * i) for i in range(n)]
    frames = [_frame(M) for M in truth]
    frames = [{**f, "depth": np.where(np.isfinite(f["depth"]), f["depth"], 0.0)} for f in frames]
    kept = [{"name": f"frame_{i:04d}.jpg", "t": i / 3} for i in range(n)]
    Rq = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(40)]))[0]
    _write_solve(tmp_path / "skel", frames, kept, range(n))
    _write_solve(tmp_path / "a", [_similarity(f, 0.8, Rq, np.array([1.0, 0, 0])) for f in frames], kept, range(0, 16))
    _write_solve(tmp_path / "b", [_similarity(f, 1.3, Rq.T, np.array([0, 2.0, 0])) for f in frames], kept, range(14, 26))
    meta = join([tmp_path / "a", tmp_path / "b"], tmp_path / "j", onto=tmp_path / "skel")
    assert meta["frames"] == n and meta["metric"]
    reps = {Path(r["solve"]).name: r for r in meta["joins"]}
    assert reps["skel"]["frames_used"] == 4  # 26-29: no room solve covers them
    assert abs(reps["a"]["scale"] - 0.8) < 0.01 and abs(reps["b"]["scale"] - 1.3) < 0.01
    rec = pycolmap.Reconstruction(str(tmp_path / "j" / "sparse" / "0"))
    got = {im.name: im.projection_center() for im in rec.images.values()}
    assert max(np.linalg.norm(got[k["name"]] - truth[i][:3, 3]) for i, k in enumerate(kept)) < 0.01


def test_windows_follow_the_rooms():
    from mise.pipeline.learned import plan_windows

    t = np.arange(0, 290, 1 / 3)
    rooms = np.where(t < 137, 0, np.where(t < 190, 1, 2))
    rng = np.random.default_rng(0)
    flicker = (t > 135) & (t < 190) & (rng.uniform(size=len(t)) < 0.1)  # a doorway, seen badly
    rooms = np.where(flicker, 0, rooms)
    noisy = t >= 190  # after 190 s the rough pass can't tell the small rooms apart
    rooms = np.where(noisy, rng.integers(0, 3, len(t)), rooms)
    w = plan_windows(t, rooms, max_frames=450)
    assert len(w) == 3, w
    (a0, a1), (b0, b1), (c0, c1) = w
    assert a0 == 0 and abs(a1 - 142) < 3          # main room, reaching 5 s into the next
    assert abs(b0 - 132) < 3 and abs(b1 - 195) < 4  # bedroom
    assert abs(c0 - 185) < 4 and c1 > 289          # the rest, together
    long = plan_windows(t, np.zeros(len(t), int), max_frames=300)  # one room, 870 frames
    assert len(long) >= 3 and all(b - a <= 300 / 3 + 1 for a, b in long)
    assert all(long[i][1] > long[i + 1][0] for i in range(len(long) - 1))  # they overlap
