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
