import numpy as np

from mise.pipeline.shell import _surfaces
from mise.pipeline.trim import decorate, find_openings
from mise.scene.models import Finish, Opening, Room, Space, Surface


def _room():
    sp = Space(id="room_1", name="Room 1", polygon=[(0, 0), (4, 0), (4, 3), (0, 3)])
    room = Room(spaces=[sp], floor_polygon=sp.polygon, ceiling_height=2.5)
    sp.finish = Finish(walls=Surface(color="#d0d0d0", kind="paint"), trim=Surface(color="#606060", kind="paint"),
                       baseboard_m=0.1)

    class _S:
        pass
    sc = _S()
    sc.room = room
    return sc


def test_window_votes_become_a_window_and_doors_a_closed_leaf():
    sc = _room()
    cell = 0.05
    walls = [s for s in _surfaces(sc.room, cell) if s["kind"] == "wall"]
    w = walls[0]
    votes = {}
    for s in walls:
        sh = s["xyz"].shape[:2]
        v = {"total": np.full(sh, 10), "window": np.zeros(sh, int), "door": np.zeros(sh, int), "mirror": np.zeros(sh, int)}
        if s is w:
            S, Z = s["S"], s["Z"]
            v["window"][(S > 1.0) & (S < 2.2) & (Z > 1.0) & (Z < 2.1)] = 8
            v["door"][(S > 3.0) & (S < 3.85) & (Z < 2.0)] = 9
        s["seen"] = np.ones(sh, bool)
        s["rgb"] = np.full(sh + (3,), 0.5)
        votes[s["id"]] = v
    found = find_openings(sc, walls, votes, cell)
    win = [o for o in found if o.kind == "window"]
    door = [o for o in found if o.kind == "door"]
    assert len(win) == 1 and abs(win[0].center - 1.6) < 0.1 and abs(win[0].width - 1.2) < 0.15
    assert abs(win[0].bottom - 1.0) < 0.1 and abs(win[0].top - 2.1) < 0.1
    assert len(door) == 1 and door[0].leaf and door[0].bottom == 0.0 and door[0].top >= 2.0
    assert len(sc.room.openings) == 2


def test_baseboards_casings_and_glass():
    sc = _room()
    w0 = sc.room.walls[0]
    sc.room.openings = [Opening(id="win_x", kind="window", wall=w0.id, center=2.0, width=1.0, bottom=1.0, top=2.0)]
    cell = 0.025
    s = next(x for x in _surfaces(sc.room, cell) if x["id"] == w0.id)
    col = np.broadcast_to(np.array([0.8, 0.8, 0.8]), s["xyz"].shape).copy()
    extra = decorate(s, col, np.ones(s["xyz"].shape[:2]), sc.room.spaces[0].finish, cell)
    S, Z = s["S"], s["Z"]
    assert np.allclose(col[Z < 0.1], 0x60 / 255, atol=0.01)  # baseboard
    casing = (S > 1.45) & (S < 1.5) & (Z > 1.2) & (Z < 1.8)
    assert np.allclose(col[casing], 0x60 / 255, atol=0.01)
    assert not s["inside"][(np.abs(S - 2.0) < 0.45) & (Z > 1.1) & (Z < 1.9)].any()  # the hole...
    assert len(extra) == 1 and extra[0]["xyz"].shape[1] > 0  # ...and the glass in it
