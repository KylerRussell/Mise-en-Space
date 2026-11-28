import numpy as np

from mise.pipeline.materials import _rgb, classify, render_shell
from mise.scene.models import Finish, Room, Space, Surface


def _boards(px=400, side=0.6, width=0.15, along="v", seed=0):
    """A floor patch of boards `width` wide running along the patch's v (image columns) or u (rows)."""
    rng = np.random.default_rng(seed)
    n = int(px * side)
    j = np.arange(n)
    board = (j / (width * px)).astype(int)
    tone = 0.55 + 0.06 * rng.standard_normal(board.max() + 1)
    stripe = tone[board]
    stripe[(j % int(width * px)) < 2] *= 0.8  # the seam
    grain = 0.01 * rng.standard_normal((n, n))
    grain = np.cumsum(grain, axis=0) * 0.15  # streaks along the board
    img = stripe[None, :] + grain if along == "v" else stripe[:, None] + grain.T
    return np.repeat(np.clip(img, 0, 1)[..., None], 3, -1).astype(np.float32)


def test_boards_are_wood_and_run_the_right_way():
    a = classify(_boards(along="v"), "floor")
    assert a["kind"] == "wood" and abs(((a["direction_deg"] - 90) + 90) % 180 - 90) < 15
    b = classify(_boards(along="u"), "floor")
    assert b["kind"] == "wood" and abs(((b["direction_deg"] - 0) + 90) % 180 - 90) < 15
    flat = np.full((240, 240, 3), 0.8, np.float32) + np.random.default_rng(1).normal(0, 0.002, (240, 240, 1)).astype(np.float32)
    assert classify(flat, "wall")["kind"] == "paint"


def test_repainting_changes_the_walls_and_keeps_the_light(tmp_path):
    sp = Space(id="room_1", name="Room 1", polygon=[(0, 0), (3, 0), (3, 2.5), (0, 2.5)])
    room = Room(spaces=[sp], floor_polygon=sp.polygon, ceiling_height=2.4)
    sp.finish = Finish(floor=Surface(color="#a08060", kind="wood", direction_deg=0.0, module_m=0.15),
                       walls=Surface(color="#d0d0d0", kind="paint"), ceiling=Surface(color="#f0f0f0", kind="paint"))
    (tmp_path / "finishes").mkdir()
    # the room's light: walls darker toward the floor
    from mise.pipeline.shell import _surfaces
    shading = {}
    for s in _surfaces(room, 0.025):
        h, w = s["xyz"].shape[:2]
        shading[s["id"]] = (np.linspace(0.6, 1.0, h)[:, None] * np.ones((1, w))).astype(np.float16) \
            if s["kind"] == "wall" else np.ones((h, w), np.float16)
    np.savez_compressed(tmp_path / "finishes" / "shading.npz", cell=np.array(0.025), **shading)

    class _S:  # a scene stand-in: render_shell only reads .room
        pass
    scene = _S()
    scene.room = room
    out = render_shell(scene, tmp_path)
    X, rgb = out.xyz, out.rgb()
    wall = (np.abs(X[:, 1]) < 0.01) & (X[:, 2] > 0.1) & (X[:, 2] < 2.3)
    low, high = wall & (X[:, 2] < 0.5), wall & (X[:, 2] > 1.9)
    assert rgb[high].mean() > rgb[low].mean() + 0.15  # the light kept
    sp.finish.walls = Surface(color="#c04040", kind="paint")  # repaint
    rgb2 = render_shell(scene, tmp_path).rgb()
    r, g = rgb2[high].mean(0)[0], rgb2[high].mean(0)[1]
    assert r > 2 * g and abs(rgb2[high].mean(0)[0] - _rgb("#c04040")[0] * 0.97) < 0.08
    floor = np.abs(X[:, 2]) < 0.01
    assert np.allclose(rgb2[floor], rgb[floor])  # the floor untouched


def test_block_joints_are_told_from_plain_paint():
    from mise.pipeline.materials import block_evidence

    rng = np.random.default_rng(1)
    px = 400
    y = np.arange(200)[:, None] / px  # a 0.5 m patch
    plain = 0.75 + 0.02 * rng.standard_normal((200, 200, 1)).repeat(3, -1)
    joints = plain.copy()
    joints[(np.mod(y[:, 0], 0.2032) < 0.008)] -= 0.08  # 8 in courses, faint as painted-over joints are
    assert block_evidence(joints.astype(np.float32)) >= 0.4
    assert block_evidence(plain.astype(np.float32)) < 0.24
