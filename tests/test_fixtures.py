"""Built-ins (fixtures): kept in the layout, hidden on the empty-home view, hung from walls or ceilings."""

import pytest

from mise.demo import demo_apartment
from mise.scene import spaces
from mise.scene.checks import check_layout
from mise.scene.models import (
    Dims,
    Pose,
    Room,
    Scene,
    SceneObject,
    Space,
    structural_support,
)
from mise.scene.store import SceneStore


@pytest.fixture
def aclient(client):
    client.app.state.store.create(demo_apartment())
    return client


def test_ceiling_is_a_structural_support():
    assert structural_support("ceiling") and structural_support("floor") and structural_support("wall:a.1")
    assert not structural_support("sofa_1")


def test_ceiling_fixture_keeps_its_height(tmp_path):
    st = SceneStore(tmp_path)
    st.create(demo_apartment())
    sid = "demo-apartment"
    fan = SceneObject(id="fan", name="Ceiling fan", category="light", dims=Dims(w=1.3, d=1.3, h=0.35),
                      pose=Pose(x=6.5, y=2.0, z=2.15), support="ceiling", fixture=True, locked=True)
    island = SceneObject(id="island", name="Island", category="counter", dims=Dims(w=1.8, d=0.9, h=0.91),
                         pose=Pose(x=6.5, y=2.0), fixture=True, locked=True)
    st.add_object(sid, "main", island, snap=False)
    st.add_object(sid, "main", fan, snap=False)
    sc = st.load(sid)
    by = {o.id: o for o in sc.resolve("main")}
    assert by["fan"].pose.z == 2.15 and by["fan"].fixture
    assert not [v for v in check_layout(sc) if "fan" in v.objects]  # above the counter, not in it


def test_door_clearance_stops_at_the_wall():
    # a shallow closet off room A; room C lies behind the closet's back wall
    room = Room(spaces=[Space(id="a", name="A", polygon=[(0, 0), (4, 0), (4, 3), (0, 3)]),
                        Space(id="b", name="Closet", kind="closet", polygon=[(0, 3.1), (1.2, 3.1), (1.2, 3.6), (0, 3.6)]),
                        Space(id="c", name="C", polygon=[(0, 3.7), (4, 3.7), (4, 6), (0, 6)])])
    room.openings.append(spaces.door_from_points(room, (0.2, 3.1), (1.0, 3.1), opening_id="closet_in", swing="none"))
    assert room.wall(room.openings[-1].wall).space == "b"
    desk = SceneObject(id="desk", name="Desk", dims=Dims(w=1.0, d=0.5, h=0.75), pose=Pose(x=0.6, y=3.96))
    sc = Scene(id="t", name="t", room=room, objects=[desk])
    assert "blocked_door" not in [v.kind for v in check_layout(sc)]
    # the same desk inside the closet does block it
    desk.pose.y = 3.35
    desk.dims = Dims(w=1.0, d=0.4, h=0.75)
    assert "blocked_door" in [v.kind for v in check_layout(sc)]


def test_fixture_flag_is_editable(tmp_path):
    st = SceneStore(tmp_path)
    st.create(demo_apartment())
    oid = st.load("demo-apartment").objects[0].id
    st.update_object("demo-apartment", "main", oid, {"fixture": True})
    assert next(o for o in st.load("demo-apartment").objects if o.id == oid).fixture


def test_render_can_hide_fixtures(aclient):
    c = aclient
    sid = "demo-apartment"
    oid = c.app.state.store.load(sid).objects[0].id
    c.app.state.store.update_object(sid, "main", oid, {"fixture": True})
    with_f = c.get(f"/api/scenes/{sid}/variants/main/render", params={"view": "plan"})
    without = c.get(f"/api/scenes/{sid}/variants/main/render", params={"view": "plan", "fixtures": "false"})
    assert with_f.status_code == without.status_code == 200
    assert with_f.content != without.content
