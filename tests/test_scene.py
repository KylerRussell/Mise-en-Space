import math

import numpy as np
import pytest

from mise.scene import geometry as g
from mise.scene import placement
from mise.scene.checks import check_layout, check_objects
from mise.scene.models import MAIN, Dims, Pose, SceneObject
from mise.scene.store import PermissionError_


def kinds(vs):
    return sorted((v.kind, tuple(v.objects)) for v in vs)


def test_demo_scene_is_clean(scene):
    assert check_layout(scene) == []


def test_footprint_rotation():
    fp = g.footprint(Pose(x=1, y=2, yaw=90), Dims(w=2, d=1, h=1))
    xs, ys = fp[:, 0], fp[:, 1]
    assert np.isclose(xs.max() - xs.min(), 1) and np.isclose(ys.max() - ys.min(), 2)


def test_front_is_plus_y_rotated():
    right, front = g.local_axes(90)
    assert np.allclose(front, [-1, 0]) and np.allclose(right, [0, 1])
    assert math.isclose(g.yaw_facing(np.array([-1.0, 0])), 90)


def test_penetration_depth_numeric():
    a = g.footprint(Pose(x=0, y=0), Dims(w=1, d=1, h=1))
    b = g.footprint(Pose(x=0.88, y=0), Dims(w=1, d=1, h=1))
    assert math.isclose(g.penetration_depth(a, b), 0.12, abs_tol=1e-9)
    c = g.footprint(Pose(x=1.5, y=0), Dims(w=1, d=1, h=1))
    assert math.isclose(g.polygon_distance(a, c), 0.5, abs_tol=1e-9)


def test_collision_message_is_specific(scene):
    desk = next(o for o in scene.objects if o.id == "desk_1")
    desk.pose.x = 3.0  # into the chair
    vs = check_layout(scene)
    col = [v for v in vs if v.kind == "collision"]
    assert col and "overlaps" in col[0].message and " m" in col[0].message
    assert col[0].amount_m > 0


def test_stacked_objects_do_not_collide(scene):
    # lamp sits on the nightstand: same footprint, no collision
    assert not [v for v in check_layout(scene) if v.kind == "collision"]


def test_blocked_door_and_out_of_room(scene):
    ns = next(o for o in scene.objects if o.id == "nightstand_1")
    ns.pose.x, ns.pose.y = 0.7, 0.4  # right in door_1's swing
    wd = next(o for o in scene.objects if o.id == "wardrobe_1")
    wd.pose.y = 0.1  # pokes through wall 1
    k = {v.kind for v in check_layout(scene)}
    assert {"blocked_door", "out_of_room"} <= k


def test_tall_object_in_front_of_window_is_soft(scene):
    wd = next(o for o in scene.objects if o.id == "wardrobe_1")
    wd.pose.x, wd.pose.y, wd.pose.yaw = 2.1, 3.3, 180
    vs = [v for v in check_layout(scene) if v.kind == "blocked_window"]
    assert vs and vs[0].severity == "soft"


def test_walkway_blocked_between_doors(scene):
    # A long sofa across the room cuts door_1 off from door_2.
    scene.objects.append(SceneObject(id="sofa_9", name="Sofa", category="sofa", dims=Dims(w=2.2, d=0.9, h=0.8),
                                     pose=Pose(x=3.1, y=1.35, yaw=0)))
    scene.objects = [o for o in scene.objects if o.id not in ("desk_1", "chair_1")]
    vs = check_layout(scene)
    assert any(v.kind == "walkway" and v.severity == "hard" for v in vs), [v.message for v in vs]


def test_snap_to_wall_and_yaw():
    from mise.demo import demo_scene
    s = demo_scene()
    o = SceneObject(id="x", name="Shelf", dims=Dims(w=0.8, d=0.3, h=1.8), pose=Pose(x=2.0, y=0.25, yaw=7))
    notes = placement.snap(o, s.room, {})
    assert o.pose.yaw == 0 and math.isclose(o.pose.y, 0.15) and notes


def test_place_relative(scene):
    objs = {o.id: o for o in scene.objects}
    chair = objs["chair_1"]
    p = placement.resolve_relation(chair, placement.Relation(relation="against_wall", target="3", along=1.0), scene.room, objs)
    assert math.isclose(p.pose.y, 3.6 - 0.25) and math.isclose(abs(p.pose.yaw), 180)
    p = placement.resolve_relation(objs["lamp_1"], placement.Relation(relation="on_top_of", target="desk_1"), scene.room, objs)
    assert p.support == "desk_1" and math.isclose(p.pose.z, 0.75)
    p = placement.resolve_relation(chair, placement.Relation(relation="in_front_of", target="desk_1", distance=0.1, face_target=True), scene.room, objs)
    desk = objs["desk_1"]
    _, f = g.local_axes(desk.pose.yaw)
    assert np.isclose((np.array([p.pose.x, p.pose.y]) - [desk.pose.x, desk.pose.y]) @ f, 0.3 + 0.1 + 0.25)


# ---- store ---------------------------------------------------------------------

def test_agent_cannot_write_main_or_move_locked(store):
    with pytest.raises(PermissionError_):
        store.move("demo-bedroom", MAIN, "desk_1", x=1, actor="agent")
    v = store.create_variant("demo-bedroom", "ai", actor="agent")
    with pytest.raises(PermissionError_):
        store.move("demo-bedroom", v.id, "bed_1", x=2, actor="agent")
    with pytest.raises(PermissionError_):
        store.promote("demo-bedroom", v.id, actor="agent")


def test_variant_edits_leave_main_alone_and_promote(store):
    sid = "demo-bedroom"
    v = store.create_variant(sid, "ai", actor="agent")
    store.move(sid, v.id, "desk_1", x=3.9, y=2.6, actor="agent")
    store.set_hidden(sid, v.id, "rug_1", True, actor="agent")
    s = store.load(sid)
    assert s.find(MAIN, "desk_1").pose.y == 1.5
    d = store.diff(sid, MAIN, v.id)
    assert {(c.kind, c.object_id) for c in d.changes} == {("move", "desk_1"), ("remove", "rug_1")}
    store.promote(sid, v.id, ["desk_1"])
    assert math.isclose(s.find(MAIN, "desk_1").pose.y, 2.6)
    assert s.find(MAIN, "desk_1").moved_from_scan
    assert any(o.id == "rug_1" for o in s.objects)  # not promoted
    store.undo(sid, MAIN)
    assert s.find(MAIN, "desk_1").pose.y == 1.5


def test_dependents_move_with_support_and_undo(store):
    sid = "demo-bedroom"
    v = store.create_variant(sid, "x")
    store.move(sid, v.id, "nightstand_1", x=1.0, y=0.9, snap=False)
    s = store.load(sid)
    lamp = s.find(v.id, "lamp_1")
    assert math.isclose(lamp.pose.x, 1.0) and math.isclose(lamp.pose.y, 0.9) and math.isclose(lamp.pose.z, 0.55)
    store.undo(sid, v.id)
    lamp = s.find(v.id, "lamp_1")
    assert math.isclose(lamp.pose.x, 0.2) and math.isclose(lamp.pose.y, 1.3)
    store.redo(sid, v.id)
    assert math.isclose(s.find(v.id, "lamp_1").pose.x, 1.0)


def test_move_reports_violations_caused(store):
    v = store.create_variant("demo-bedroom", "x", actor="agent")
    r = store.move("demo-bedroom", v.id, "nightstand_1", x=0.7, y=0.4, snap=False, actor="agent")
    assert any(x["kind"] == "blocked_door" for x in r["violations_caused"])


def test_persistence_roundtrip(store, tmp_path):
    from mise.scene.store import SceneStore
    v = store.create_variant("demo-bedroom", "x")
    store.move("demo-bedroom", v.id, "desk_1", x=3.8, snap=False)
    again = SceneStore(store.root)
    s = again.load("demo-bedroom")
    assert math.isclose(s.find(v.id, "desk_1").pose.x, 3.8)
    assert len(again.ops("demo-bedroom")) == 2
