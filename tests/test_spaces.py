"""Homes with several spaces: open-plan zones, rooms behind walls, doors that link them."""

import math

import pytest

from mise.demo import demo_apartment, demo_scene
from mise.scene import spaces
from mise.scene.checks import check_layout
from mise.scene.models import MAIN, Dims, Pose, Room, SceneObject
from mise.scene.store import PermissionError_, SceneStore

A = "demo-apartment"


@pytest.fixture
def apt():
    return demo_apartment()


@pytest.fixture
def astore(tmp_path):
    st = SceneStore(tmp_path / "scenes")
    st.create(demo_apartment())
    return st


def kinds(scene, oid=None):
    return [v.kind for v in check_layout(scene) if oid is None or oid in v.objects]


def test_open_plan_has_no_wall_between_zones(apt):
    room = apt.room
    assert not [w for w in room.walls if w.space == "living" and w.a[0] == 5 and w.b[0] == 5]
    assert not [w for w in room.walls if w.space == "kitchen" and w.a[0] == 5 and w.b[0] == 5]
    assert math.isclose(room.floor_area(), 20 + 12 + 4 * 3.38 + 2.38 * 2.38, rel_tol=1e-6)
    assert room.space_at(6, 1).id == "kitchen" and room.space_at(2, 6).id == "bedroom"
    assert room.space_at(2, 4.06) is None  # inside the wall between living room and bedroom
    assert check_layout(apt) == []


def test_object_may_span_open_zones_but_not_walls(apt):
    bench = SceneObject(id="bench_1", name="Bench", category="bench", dims=Dims(w=1.2, d=0.4, h=0.45),
                        pose=Pose(x=5.0, y=1.4))  # straddles the living/kitchen zone line
    apt.objects.append(bench)
    assert "out_of_room" not in kinds(apt, "bench_1") and "crosses_wall" not in kinds(apt, "bench_1")
    bench.pose = Pose(x=2.0, y=4.06, yaw=0)  # sits across the bedroom wall
    assert "out_of_room" in kinds(apt, "bench_1")


def test_crossing_a_shared_wall_without_a_gap():
    room = Room(spaces=[
        {"id": "a", "name": "A", "polygon": [(0, 0), (3, 0), (3, 3), (0, 3)]},
        {"id": "b", "name": "B", "polygon": [(3, 0), (6, 0), (6, 3), (3, 3)]},  # walled, not open
    ])
    from mise.scene.models import Scene
    s = Scene(id="x", name="x", room=room, objects=[
        SceneObject(id="t", name="Table", category="table", dims=Dims(w=1.0, d=0.6, h=0.7), pose=Pose(x=3.1, y=1.5))])
    v = [x for x in check_layout(s) if x.kind == "crosses_wall"]
    assert v and math.isclose(v[0].amount_m, 0.4, abs_tol=1e-6)


def test_ceiling_is_per_space(apt):
    apt.objects.append(SceneObject(id="shelf_9", name="Tall shelf", category="bookshelf", dims=Dims(w=0.6, d=0.3, h=2.4),
                                   pose=Pose(x=5.0, y=6.3)))
    assert "out_of_room" in kinds(apt, "shelf_9")  # 2.4 m in a 2.3 m bathroom
    apt.objects[-1].pose = Pose(x=2.0, y=7.3)
    assert "out_of_room" not in kinds(apt, "shelf_9")  # fine in the 2.6 m bedroom


def test_doors_link_rooms_for_walkways(apt):
    # a dresser across the living-room side of the bedroom door cuts the bedroom off
    apt.objects.append(SceneObject(id="dresser_1", name="Dresser", category="dresser", dims=Dims(w=1.2, d=0.5, h=0.8),
                                   pose=Pose(x=3.0, y=3.6)))
    msgs = [v.message for v in check_layout(apt) if v.kind == "walkway"]
    assert any("bedroom_door" in m for m in msgs)


def test_split_and_merge_spaces_roundtrip(apt):
    r, new = spaces.split_space(apt.room, "living", "y", 2.0, "Dining area", "dining")
    r, _ = spaces.rewall(apt.room, r)
    assert new == "dining_area"
    assert {s.id for s in r.spaces} == {"living", "dining_area", "kitchen", "bedroom", "bathroom"}
    assert r.is_open("living", "dining_area") and r.is_open("dining_area", "kitchen") and r.is_open("living", "kitchen")
    assert not [w for w in r.walls if w.a[1] == 2.0 and w.b[1] == 2.0]  # zone line, not a wall
    merged = spaces.merge_spaces(r, "living", "dining_area")
    merged, _ = spaces.rewall(r, merged)
    assert math.isclose(merged.space("living").area(), 20.0)
    assert len(merged.space("living").polygon) == 4
    with pytest.raises(ValueError):
        spaces.merge_spaces(merged, "living", "bedroom")  # separated by a wall gap


def test_store_space_edits_are_undoable_and_user_only(astore):
    st = astore
    ev = st.edit_room(A, lambda r: spaces.update_space(r, "bedroom", name="Main bedroom", kind="bedroom"))
    assert st.load(A).room.space("bedroom").name == "Main bedroom" and ev["violations"] == []
    st.edit_room(A, lambda r: spaces.split_space(r, "living", "x", 2.5, "Reading nook", "living")[0])
    assert len(st.load(A).room.spaces) == 5
    door = next(o for o in st.load(A).room.openings if o.id == "front_door")
    assert door.wall.startswith("living.")  # front door re-attached after the split
    st.undo(A, MAIN)
    assert len(st.load(A).room.spaces) == 4
    st.undo(A, MAIN)
    assert st.load(A).room.space("bedroom").name == "Bedroom"
    st.redo(A, MAIN)
    assert st.load(A).room.space("bedroom").name == "Main bedroom"
    with pytest.raises(PermissionError_):
        st.edit_room(A, lambda r: spaces.delete_space(r, "bathroom"), actor="agent")


def test_wall_supports_follow_rewalling(tmp_path):
    st = SceneStore(tmp_path / "s")
    st.create(demo_scene())
    st.add_object("demo-bedroom", MAIN, SceneObject(
        id="tv_1", name="TV", category="tv", dims=Dims(w=1.2, d=0.08, h=0.7), pose=Pose(x=4.16, y=2.4, z=1.2, yaw=90),
        support="wall:2"), snap=False)
    # turning the single room into explicit spaces renames its walls
    st.edit_room("demo-bedroom", lambda r: spaces.update_space(r, "room", name="Bedroom", kind="bedroom"))
    s = st.load("demo-bedroom")
    assert s.find(MAIN, "tv_1").support == "wall:room.2"
    assert {o.wall for o in s.room.openings} == {"room.1", "room.2", "room.3"}
    st.undo("demo-bedroom", MAIN)
    assert st.load("demo-bedroom").find(MAIN, "tv_1").support == "wall:2"


# ---- API, MCP and solver ------------------------------------------------------------------

@pytest.fixture
def aclient(client):
    client.app.state.store.create(demo_apartment())
    return client


def test_api_spaces(aclient):
    c = aclient
    room = c.get(f"/api/scenes/{A}/room").json()
    sp = {s["id"]: s for s in room["spaces"]}
    assert set(sp) == {"living", "kitchen", "bedroom", "bathroom"}
    assert sp["bathroom"]["ceiling"] == 2.3 and "toilet_1" in sp["bathroom"]["objects"]
    objs = c.get(f"/api/scenes/{A}/variants/main/objects", params={"space": "kitchen"}).json()
    assert {o["id"] for o in objs} == {"counter_1", "island_1"} and objs[0]["space"] == "kitchen"
    r = c.post(f"/api/scenes/{A}/spaces/living/split", json={"axis": "y", "at": 2.0, "new_name": "Dining", "new_kind": "dining"})
    assert r.status_code == 200 and r.json()["space_id"] == "dining"
    assert c.patch(f"/api/scenes/{A}/spaces/dining", json={"name": "Dining nook"}).status_code == 200
    assert c.patch(f"/api/scenes/{A}/spaces/dining", json={"name": "x"}, headers={"X-Actor": "agent"}).status_code == 403
    assert c.post(f"/api/scenes/{A}/spaces/living/merge", json={"other": "bedroom"}).status_code == 422
    assert c.post(f"/api/scenes/{A}/spaces/living/merge", json={"other": "dining"}).status_code == 200
    png = c.get(f"/api/scenes/{A}/variants/main/render", params={"view": "plan", "space": "bedroom"})
    assert png.headers["content-type"] == "image/png"
    assert c.get(f"/api/scenes/{A}/variants/main/check", params={"space": "bedroom"}).json()["hard"] == 0


def test_mcp_space_tools(aclient):
    import asyncio
    import json

    from mise.mcp_server.server import build_server
    mcp = build_server(client=aclient)

    async def run():
        room = json.loads((await mcp.call_tool("get_room", {"scene": A})).content[0].text)
        assert {s["kind"] for s in room["spaces"]} == {"living", "kitchen", "bedroom", "bathroom"}
        r = await mcp.call_tool("list_objects", {"scene": A, "space": "bedroom"})
        objs = [json.loads(c.text) for c in r.content]  # one content block per list item
        assert [o["id"] for o in objs] == ["bed_1"] and objs[0]["space"] == "bedroom"
        img = await mcp.call_tool("render_view", {"scene": A, "view": "plan", "space": "kitchen"})
        assert img.content[0].type == "image"
    asyncio.run(run())


def test_solver_keeps_objects_in_their_space(apt):
    from mise.agent.solver import Constraint, solve_layout
    apt.objects.append(SceneObject(id="armchair_1", name="Armchair", category="armchair",
                                   dims=Dims(w=0.8, d=0.8, h=0.9), pose=Pose(x=1.0, y=3.0)))
    # pull the armchair toward the bed: it must stay in the living room, not go through the wall
    res = solve_layout(apt, MAIN, [Constraint(kind="near", obj="armchair_1", target="bed_1", max=0.2)], seed=1)
    p = res.poses["armchair_1"]
    assert apt.room.space_at(p.x, p.y).id == "living"
    assert not [v for v in res.violations if "armchair_1" in v.objects and v.severity == "hard"]
    # with space= it furnishes the bedroom instead
    res = solve_layout(apt, MAIN, [Constraint(kind="near", obj="armchair_1", target="bed_1", max=0.3)], seed=1,
                       space="bedroom")
    p = res.poses["armchair_1"]
    assert apt.room.space_at(p.x, p.y).id == "bedroom"


def test_doors_from_floor_points(aclient):
    c = aclient
    # jambs measured on the floor next to the kitchen's east wall (x = 8)
    r = c.post(f"/api/scenes/{A}/openings", json={"id": "pantry_door", "a": [7.93, 1.2], "b": [7.95, 1.96],
                                                   "swing": "out", "hinge": "left"})
    assert r.status_code == 200, r.text
    op = next(o for o in c.get(f"/api/scenes/{A}/room").json()["openings"] if o["id"] == "pantry_door")
    assert op["wall"] == "kitchen.2" and abs(op["width"] - 0.76) < 0.01 and abs(op["center"] - 1.58) < 0.01
    assert c.post(f"/api/scenes/{A}/openings", json={"a": [3.0, 3.0], "b": [3.0, 3.8]}).status_code == 422  # no wall there
    assert c.patch(f"/api/scenes/{A}/openings/pantry_door", json={"width": 0.8}).status_code == 200
    assert c.delete(f"/api/scenes/{A}/openings/pantry_door", headers={"X-Actor": "agent"}).status_code == 403
    assert c.delete(f"/api/scenes/{A}/openings/pantry_door").status_code == 200


def test_reach_in_closet_door_needs_no_walkway_inside(apt):
    # a 0.6 m deep linen closet off the kitchen: you reach in, you don't walk in
    r, sid = spaces.add_space(apt.room, "Linen closet", "closet", [(6.62, 4.12), (7.22, 4.12), (7.22, 5.0), (6.62, 5.0)])
    r, _ = spaces.rewall(apt.room, r)
    r.openings.append(spaces.door_from_points(r, (6.7, 4.12), (7.14, 4.12), opening_id="linen_door", swing="none"))
    apt.room = r
    assert r.wall(r.openings[-1].wall).space == sid
    assert "walkway" not in kinds(apt)
    r.space(sid).kind = "bedroom"  # the same size as a room you'd walk into does get flagged
    assert any("linen_door" in v.message for v in check_layout(apt) if v.kind == "walkway")
