"""Floor and wall finishes sampled from the capture, and stored per space."""

import numpy as np
import pytest

from mise.pipeline import finishes as fin

cv2 = pytest.importorskip("cv2")


def pinhole(R, C, f=500.0, cx=320.0, cy=240.0):
    def project(P):
        pc = (P - C) @ R.T
        z = pc[:, 2]
        return np.c_[f * pc[:, 0] / z + cx, f * pc[:, 1] / z + cy], z
    return project


def test_floor_patch_is_metric_and_fronto_parallel():
    # floor painted with 0.25 m stripes along x; camera 1.5 m up, looking down and ahead
    def floor_img(project):
        H, W = 480, 640
        v, u = np.mgrid[0:H, 0:W].astype(float)
        # back-project every pixel to the floor plane to paint the ground truth
        ray = np.stack([(u - 320) / 500, (v - 240) / 500, np.ones_like(u)], -1) @ R
        t = -C[2] / ray[..., 2]
        X = C[0] + t * ray[..., 0]
        return (((np.floor(X / 0.25) % 2) * 200 + 30)[..., None] * np.ones(3)).astype(np.uint8)

    a = np.radians(50)  # pitch down
    R = np.array([[1, 0, 0], [0, -np.sin(a), -np.cos(a)], [0, np.cos(a), -np.sin(a)]])  # rows: cam x, y, z in world
    C = np.array([0.0, -1.0, 1.5])
    img = floor_img(None)
    p = fin.floor_patch(0.0, 0.3, 1.0)
    tex, ok = fin.sample_patch(p, img, pinhole(R, C), px_per_m=200)
    assert tex.shape[:2] == (200, 200) and ok > 0.99
    # stripes are vertical in the texture and 0.25 m = 50 px wide wherever we look
    col = tex[:, :, 0].astype(float)
    assert col.std(axis=0).mean() < 25  # constant down each column
    edges = np.nonzero(np.abs(np.diff(col.mean(0))) > 80)[0]
    assert np.allclose(np.diff(edges), 50, atol=2)


def test_tileable_and_textures():
    # lighting falloff: dark at the top, bright at the bottom, so repeats show a hard seam
    tex = np.repeat(np.linspace(40, 220, 64)[:, None, None], 64, 1).repeat(3, 2).astype(np.uint8)
    t = fin.tileable(tex)
    assert t.shape == tex.shape
    seam = lambda x: np.abs(x[0].astype(int) - x[-1].astype(int)).mean()  # noqa: E731
    assert seam(t) < 0.2 * seam(tex)
    planks, size = fin.plank_texture(np.ones((100, 100, 3), np.float32), "#808080", 0.165, px_per_m=200)
    assert size == pytest.approx(0.99) and planks.shape[:2] == (198, 198)
    blocks, size = fin.block_texture("#a0a0a0", px_per_m=200)
    assert size == pytest.approx(0.812) and blocks.min() < blocks.max()
    assert fin.median_color(np.full((4, 4, 3), (16, 32, 64), np.uint8)) == "#402010"
    assert fin.gain("#404040", 1.5) == "#606060"


def test_space_finish_roundtrip(client):
    from mise.demo import demo_apartment
    client.app.state.store.create(demo_apartment())
    A = "demo-apartment"
    fin_ = {"floor": {"color": "#8c8884", "texture": "finishes/floor.jpg", "size_m": 0.99},
            "walls": {"color": "#aeaca8"}}
    assert client.patch(f"/api/scenes/{A}/spaces/living", json={"finish": fin_}).status_code == 200
    sp = next(s for s in client.get(f"/api/scenes/{A}/room").json()["spaces"] if s["id"] == "living")
    assert sp["finish"]["floor"]["texture"] == "finishes/floor.jpg" and sp["finish"]["walls"]["size_m"] == 1.0
    bad = {"floor": {"texture": "../../etc/passwd"}}
    assert client.patch(f"/api/scenes/{A}/spaces/living", json={"finish": bad}).status_code == 422
    assert client.patch(f"/api/scenes/{A}/spaces/living", json={"finish": {"walls": {"color": "white"}}}).status_code == 422
    assert client.patch(f"/api/scenes/{A}/spaces/living", json={"finish": fin_},
                        headers={"X-Actor": "agent"}).status_code == 403
    # splitting a space keeps its finish on both halves
    client.post(f"/api/scenes/{A}/spaces/living/split", json={"axis": "y", "at": 2.0, "new_name": "Dining"})
    sp = {s["id"]: s for s in client.get(f"/api/scenes/{A}/room").json()["spaces"]}
    assert sp["dining"]["finish"]["floor"]["color"] == "#8c8884"
