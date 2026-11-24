"""Objects may be tipped unless they really only work one way up (e.g. a cube shelf on its end)."""

import math

import numpy as np
import pytest

from mise.products.importer import ImportRequest, run_import
from mise.products.scale_fit import fit_scale
from mise.scene import geometry as g
from mise.scene.checks import check_layout
from mise.scene.models import MAIN, Dims, Pose, SceneObject, upright_required
from mise.scene.store import PermissionError_

from .conftest import FakeGenerator, fake_listing

S = "demo-bedroom"


def cube_shelf(**pose):
    # 2x4 cube shelf as listed: lying down, 1.47 wide, 0.39 deep, 0.77 high
    return SceneObject(id="shelf_1", name="Cube shelf", category="bookshelf", dims=Dims(w=1.47, d=0.39, h=0.77),
                       pose=Pose(**pose))


def test_policy_defaults_are_sparing():
    assert upright_required("sofa", None) and upright_required("lamp", None)
    assert not upright_required("bookshelf", None) and not upright_required("furniture", None)
    assert not upright_required("sofa", False) and upright_required("bookshelf", True)


def test_extent_follows_tipping():
    s = cube_shelf(roll=90)
    assert s.extent().as_tuple() == (0.77, 0.39, 1.47)
    assert cube_shelf(pitch=90).extent().as_tuple() == (1.47, 0.77, 0.39)
    assert cube_shelf(roll=180).extent().as_tuple() == (1.47, 0.39, 0.77)
    with pytest.raises(ValueError):
        Pose(roll=45)
    assert Pose(roll=-90).roll == 270


def test_tipped_shelf_footprint_and_height_drive_checks(scene):
    shelf = cube_shelf(x=2.9, y=3.405, yaw=180, roll=90)  # on its end against wall 3, right of the window
    scene.objects.append(shelf)
    fp = g.object_footprint(shelf)
    assert np.isclose(np.ptp(fp[:, 0]), 0.77) and np.isclose(np.ptp(fp[:, 1]), 0.39)
    assert g.z_interval(shelf) == (0.0, 1.47)
    assert not [v for v in check_layout(scene) if "shelf_1" in v.objects and v.severity == "hard"]
    # lying down in the same spot it is 1.47 wide and runs into the window zone / desk side
    shelf.pose.roll = 0
    assert np.isclose(np.ptp(g.object_footprint(shelf)[:, 0]), 1.47)


def test_store_tips_and_refuses_upright_only(store):
    st = store
    st.add_object(S, MAIN, cube_shelf(x=3.5, y=3.4, yaw=180), snap=False)
    r = st.move(S, MAIN, "shelf_1", roll=90, snap=True)
    s = st.load(S)
    shelf = s.find(MAIN, "shelf_1")
    assert shelf.pose.roll == 90 and math.isclose(shelf.pose.y, 3.6 - 0.39 / 2)  # still snapped flush to the wall
    # things placed on top sit at the tipped height
    st.add_object(S, MAIN, SceneObject(id="plant_1", name="Plant", category="plant", dims=Dims(w=0.2, d=0.2, h=0.3),
                                       pose=Pose(x=3.5, y=3.4), support="shelf_1"), snap=False)
    assert math.isclose(s.find(MAIN, "plant_1").pose.z, 1.47)
    st.undo(S, MAIN)
    st.undo(S, MAIN)
    assert s.find(MAIN, "shelf_1").pose.roll == 0
    with pytest.raises(ValueError, match="only stands upright"):
        st.move(S, MAIN, "desk_1", roll=90)
    v = st.create_variant(S, "ai", actor="agent")
    with pytest.raises(PermissionError_):
        st.update_object(S, v.id, "desk_1", {"upright_only": False}, actor="agent")
    st.update_object(S, MAIN, "desk_1", {"upright_only": False})  # you decide it can tip
    st.move(S, MAIN, "desk_1", roll=90)
    assert s.find(MAIN, "desk_1").extent().h == 1.2


def test_tip_shows_in_diff(store):
    store.add_object(S, MAIN, cube_shelf(x=3.5, y=3.4, yaw=180), snap=False)
    v = store.create_variant(S, "ai", actor="agent")
    store.move(S, v.id, "shelf_1", roll=90, snap=False, actor="agent")
    assert [c.object_id for c in store.diff(S, MAIN, v.id).changes] == ["shelf_1"]


def test_fit_scale_tips_only_when_allowed():
    listed = (1.47, 0.39, 0.77)  # listed lying down
    standing = (0.385, 0.195, 0.735)  # generated standing on its end, at half scale
    upright = fit_scale(listed, standing)
    assert upright.gate == "reject" and not upright.tipped
    free = fit_scale(listed, standing, allow_tipping=True)
    assert free.gate == "accept" and free.tipped and free.perm == (2, 1, 0)
    assert np.allclose(free.fitted_dims, listed, rtol=0.03)


def test_import_matches_listing_by_tipping_a_free_object(tmp_path):
    listing = fake_listing(url="https://shop.example/shelf", dims="W 147 x D 39 x H 77 cm", price=89.0)
    gen = FakeGenerator([(0.385, 0.195, 0.735)])  # model came out standing on its end
    res = run_import(ImportRequest(url=listing.url, category="bookshelf"), tmp_path / "assets", gen, listing=listing)
    a = res.asset
    assert a.status == "ready" and a.asset_kind == "splat"
    assert np.allclose(a.dims.as_tuple(), (1.47, 0.39, 0.77), rtol=0.10)
    assert any("tipping" in n for n in res.attempts[0]["notes"])
    assert not any("ambiguous" in w for w in a.warnings)


def test_import_keeps_upright_objects_upright(tmp_path):
    listing = fake_listing(url="https://shop.example/shelf", dims="W 147 x D 39 x H 77 cm")
    gen = FakeGenerator([(0.385, 0.195, 0.735)] * 2)
    res = run_import(ImportRequest(url=listing.url, category="bookshelf", upright_only=True), tmp_path / "assets", gen,
                     listing=listing)
    assert res.asset.status == "fallback_box"  # will not silently tip something you said stays upright
