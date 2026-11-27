import numpy as np

from mise.pipeline.shell import textured_shell
from mise.scene.models import Opening, Room, Space


def _room():
    sp = Space(id="room_1", name="Room 1", polygon=[(0, 0), (4, 0), (4, 3), (0, 3)])
    room = Room(spaces=[sp], floor_polygon=sp.polygon, ceiling_height=2.5)
    w = next(w for w in room.walls if abs(w.a[1] - 3) < 1e-6 and abs(w.b[1] - 3) < 1e-6)  # the far wall, y = 3
    room.openings = [Opening(id="door", kind="door", wall=w.id, center=0.5, width=0.8, bottom=0.0, top=2.0)]
    return room, w


def test_the_wall_is_painted_from_the_photo_and_what_hid_it_is_filled():
    room, far = _room()
    W, H, f = 320, 240, 200.0
    K = [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]
    C = np.array([2.0, 0.5, 1.25])  # looking along +y at the far wall, x right, z up
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # world -> camera
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, -R @ C
    img = np.zeros((H, W, 3), np.float32)
    img[:] = (0.8, 0.2, 0.2)  # a red wall
    lab = np.zeros((H, W), np.uint8)  # all "wall"
    # a cupboard in the middle of the view: brown, labelled cupboard, 0.5 m in front of the wall
    u0, u1 = W // 2 - 30, W // 2 + 30
    img[:, u0:u1] = (0.4, 0.25, 0.1)
    lab[:, u0:u1] = 1
    depth = np.full((H, W), 2.5, np.float32)
    depth[:, u0:u1] = 2.0
    cams = [{"name": "f.jpg", "w2c": w2c.tolist(), "K": K, "size": [W, H]}]
    xs = np.linspace(1.75, 2.25, 20)
    things = np.array([(x, 2.5, z) for x in xs for z in np.linspace(0.1, 2.0, 30)])
    out, rep, _ = textured_shell(room, {"f.jpg": img}, cams, {"f.jpg": lab}, ["wall", "cupboard"],
                              {"f.jpg": depth}, cell=0.05, things=things, say=lambda m: None)
    X, rgb = out.xyz, out.rgb()
    on_far = np.abs(X[:, 1] - 3) < 0.02
    seen = on_far & (X[:, 0] > 2.6) & (X[:, 0] < 3.2) & (X[:, 2] > 0.8) & (X[:, 2] < 1.7)
    assert np.allclose(rgb[seen], (0.8, 0.2, 0.2), atol=0.05)
    behind = on_far & (np.abs(X[:, 0] - 2.0) < 0.15) & (X[:, 2] > 0.8) & (X[:, 2] < 1.7)
    assert behind.any() and np.allclose(rgb[behind], (0.8, 0.2, 0.2), atol=0.1)  # wall, not cupboard
    a, b = np.array(far.a), np.array(far.b)
    dx = (a + 0.5 * (b - a) / np.linalg.norm(b - a))[0]  # 0.5 m along the wall from its start
    door = on_far & (np.abs(X[:, 0] - dx) < 0.3) & (X[:, 2] < 1.9)
    assert not door.any()  # the doorway stays open
    assert rep["wall"]["behind_things_pct"] > 0
