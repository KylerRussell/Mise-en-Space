import numpy as np

from mise.pipeline import ply
from mise.pipeline.empty_room import empty_room


def _splat(xyz, rgb):
    s = ply.Splat(xyz=np.asarray(xyz, np.float32))
    s.f_dc = ((np.asarray(rgb, np.float32) - 0.5) / ply.SH_C0).astype(np.float32)
    s.opacity = np.zeros(len(xyz), np.float32)
    s.scale = np.full((len(xyz), 3), np.log(0.02), np.float32)
    s.rot = np.tile(np.array([1, 0, 0, 0], np.float32), (len(xyz), 1))
    return s


def test_things_go_and_the_floor_under_them_is_filled():
    rng = np.random.default_rng(0)
    classes = ["wall", "floor", "bed", "mirror"]
    floor = np.c_[rng.uniform(0, 4, 8000), rng.uniform(0, 3, 8000), np.zeros(8000)]
    under = (floor[:, 0] > 1) & (floor[:, 0] < 2.5) & (floor[:, 1] > 1) & (floor[:, 1] < 2)
    floor = floor[~under]  # the bed hid the floor under it
    bed = np.c_[rng.uniform(1, 2.5, 3000), rng.uniform(1, 2, 3000), rng.uniform(0.05, 0.6, 3000)]
    wall = np.c_[rng.uniform(0, 4, 2000), np.zeros(2000), rng.uniform(0, 2.4, 2000)]
    outside = np.c_[rng.uniform(6, 7, 500), rng.uniform(0, 3, 500), rng.uniform(0, 2, 500)]  # seen in the mirror
    xyz = np.vstack([floor, bed, wall, outside])
    cls = np.r_[np.full(len(floor), 1), np.full(len(bed), 2), np.full(len(wall), 0), np.full(len(outside), 3)]
    s = _splat(xyz, np.full((len(xyz), 3), 0.5))
    out, rep = empty_room(s, cls, classes, [np.array([(0, 0), (4, 0), (4, 3), (0, 3)], float)])
    assert rep["things"] == len(bed) and rep["outside"] == len(outside) and rep["floor_holes"] == 1
    x, y, z = out.xyz.T
    in_hole = (x > 1.1) & (x < 2.4) & (y > 1.1) & (y < 1.9)
    assert not (in_hole & (z > 0.1)).any()  # no bed left
    assert (in_hole & (np.abs(z) < 0.05)).sum() > 0.5 * 8000 / 12 * 1.3 * 0.8  # floor copied in
    assert (x < 5).all()  # nothing from beyond the room
