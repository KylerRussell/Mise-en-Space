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
    assert rep["things"] + rep["floor_under_things"] >= len(bed) and rep["outside"] == len(outside)
    assert rep["floor_holes"] == 1
    x, y, z = out.xyz.T
    in_hole = (x > 1.1) & (x < 2.4) & (y > 1.1) & (y < 1.9)
    assert not (in_hole & (z > 0.1)).any()  # no bed left
    assert (in_hole & (np.abs(z) < 0.05)).sum() > 0.5 * 8000 / 12 * 1.3 * 0.8  # floor copied in
    assert (x < 5).all()  # nothing from beyond the room
    assert ((np.abs(y) < 0.05) & (z > 0.2)).sum() > 0.95 * (wall[:, 2] > 0.2).sum()  # the wall stays


def test_the_wall_behind_a_wardrobe_is_filled_and_pictures_stay():
    rng = np.random.default_rng(2)
    classes = ["wall", "floor", "wardrobe", "painting"]
    floor = np.c_[rng.uniform(0, 4, 8000), rng.uniform(0, 3, 8000), np.zeros(8000)]
    wall = np.c_[rng.uniform(0, 4, 6000), np.full(6000, 0.03), rng.uniform(0, 2.5, 6000)]  # 3 cm off the outline
    behind = (wall[:, 0] > 1) & (wall[:, 0] < 2) & (wall[:, 2] < 1.8)
    wall = wall[~behind]  # hidden by the wardrobe
    ward = np.c_[rng.uniform(1, 2, 3000), rng.uniform(0.1, 0.6, 3000), rng.uniform(0, 1.8, 3000)]
    pic = np.c_[rng.uniform(2.5, 3, 200), np.full(200, 0.04), rng.uniform(1.2, 1.6, 200)]
    xyz = np.vstack([floor, wall, ward, pic])
    cls = np.r_[np.full(len(floor), 1), np.full(len(wall), 0), np.full(len(ward), 2), np.full(len(pic), 3)]
    s = _splat(xyz, np.full((len(xyz), 3), 0.5))
    out, rep = empty_room(s, cls, classes, [np.array([(0, 0), (4, 0), (4, 3), (0, 3)], float)], [2.5])
    x, y, z = out.xyz.T
    assert rep["wall_holes"] >= 1
    assert not ((y > 0.15) & (z > 0.1)).any()  # the wardrobe is gone
    hole = (x > 1.1) & (x < 1.9) & (z > 0.1) & (z < 1.7)
    assert (hole & (np.abs(y - 0.03) < 0.06)).sum() > 0.5 * 6000 / 10 * 0.8 * 1.6  # wall copied in
    assert ((x > 2.5) & (x < 3) & (z > 1.2) & (z < 1.6) & (y < 0.1)).sum() >= 200  # the picture stays


def test_each_room_comes_from_the_stretch_that_stood_in_it():
    from mise.pipeline.empty_room import merge_splats

    rng = np.random.default_rng(1)
    rooms = [np.array([(0, 0), (3, 0), (3, 3), (0, 3)], float), np.array([(3.1, 0), (6, 0), (6, 3), (3.1, 3)], float)]
    pts = np.c_[rng.uniform(-0.5, 6.5, (4000, 2)), rng.uniform(0, 2.4, 4000)]

    def cams_at(xs):  # cameras standing at (x, 1.5), looking along +y: w2c with center c
        out = []
        for x in xs:
            c = np.array([x, 1.5, 1.5])
            R = np.eye(3)
            out.append({"w2c": np.r_[np.c_[R, -R @ c], [[0, 0, 0, 1]]].tolist()})
        return out
    a = (_splat(pts, np.full((len(pts), 3), 0.2)), cams_at([1.0, 1.5, 2.0, 4.0]))  # mostly in the left room
    b = (_splat(pts, np.full((len(pts), 3), 0.8)), cams_at([4.5, 5.0, 5.5, 1.0]))  # mostly in the right one
    out, rep = merge_splats([a, b], rooms)
    assert rep["owner"] == [0, 1] and len(out) == len(pts)
    left = out.xyz[:, 0] < 3.0
    assert np.allclose(out.rgb()[left], 0.2, atol=1e-3) and np.allclose(out.rgb()[~left & (out.xyz[:, 0] > 3.2)], 0.8, atol=1e-3)


def test_room_visits_fold_glances_back_into_the_stay():
    from mise.pipeline.splat_flow import room_visits

    t = np.arange(300) * 0.5
    r = np.r_[np.zeros(100), [-1, -1], np.zeros(20), np.ones(60), [2], np.ones(10), np.full(107, 2)].astype(int)
    v = room_visits(t, r, min_frames=30)
    assert [x[0] for x in v] == [0, 1, 2]
    assert v[0][3] == 120 and v[1][3] == 70 and v[2][1] == t[193]
