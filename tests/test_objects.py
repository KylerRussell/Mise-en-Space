import numpy as np

from mise.pipeline import ply
from mise.pipeline.objects import find_objects, scene_objects


def _splat(xyz):
    s = ply.Splat(xyz=np.asarray(xyz, np.float32))
    s.f_dc = np.zeros((len(xyz), 3), np.float32)
    s.opacity = np.full(len(xyz), 3.0, np.float32)
    s.scale = np.full((len(xyz), 3), np.log(0.02), np.float32)
    s.rot = np.tile(np.array([1, 0, 0, 0], np.float32), (len(xyz), 1))
    return s


def _box(rng, x0, x1, y0, y1, z0, z1, n):
    return np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)]


def test_things_become_objects_room_by_room(tmp_path):
    rng = np.random.default_rng(0)
    classes = ["wall", "floor", "bed", "computer", "desk", "stove", "toilet", "apparel"]
    rooms = [np.array([(0, 0), (4, 0), (4, 4), (0, 4)], float), np.array([(4.1, 0), (8, 0), (8, 4), (4.1, 4)], float)]
    parts = [
        (_box(rng, 2.5, 4.0, 1, 3, 0.0, 0.5, 3000), 2),   # a bed against the shared wall...
        (_box(rng, 4.1, 5.6, 1, 3, 0.0, 0.5, 3000), 2),   # ...and one on the other side: two beds
        (_box(rng, 0.5, 2.0, 3.4, 3.9, 0.25, 0.75, 2000), 4),  # a desk, legs unseen
        (_box(rng, 0.8, 1.6, 3.6, 3.8, 0.76, 1.2, 800), 3),   # a screen on it
        (_box(rng, 6, 7.5, 3.4, 3.9, 0, 0.9, 1500), 5),  # a stove
        (_box(rng, 3.0, 3.5, 1.5, 2.5, 0.5, 0.6, 300), 7),  # clothes on the first bed
        (_box(rng, 0, 4, 0, 4, 0, 0.01, 4000), 1),
    ]
    xyz = np.vstack([p for p, _ in parts])
    cls = np.concatenate([np.full(len(p), c) for p, c in parts])
    s = _splat(xyz)
    found = find_objects(s, cls, classes, rooms)
    kinds = sorted(f.kind for f in found)
    assert kinds == ["bed", "bed", "desk", "electronics", "kitchen"]
    objs = {o.name: o for o in scene_objects(s, found, tmp_path)}
    bed = next(o for o in objs.values() if o.category == "bed" and o.pose.x < 4)
    assert abs(bed.dims.w * bed.dims.d - 1.5 * 2) < 0.4 and bed.support == "floor"
    assert objs["Desk"].support == "floor" and objs["Desk"].pose.z == 0.0 and objs["Desk"].dims.h > 0.7
    assert objs["Screens"].support == objs["Desk"].id
    assert objs["Stove"].fixture and not bed.fixture
    assert (tmp_path / "objects" / f"{bed.id}.ply").exists()
    # the clothes went with the bed they lay on
    f_bed = next(f for f in found if f.kind == "bed" and f.box.center[0] < 4)
    assert len(f_bed.members) >= 3000 + 250
