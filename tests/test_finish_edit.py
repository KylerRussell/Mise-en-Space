import asyncio
import json

import pytest

from mise.mcp_server.server import build_server
from mise.scene.store import PermissionError_

S = "demo-bedroom"
AGENT = {"X-Actor": "agent"}


def test_repaint_in_a_variant_then_promote_and_undo(store):
    v = store.create_variant(S, "sage", actor="agent")
    with pytest.raises(PermissionError_):
        store.set_finish(S, "main", "room", "walls", {"color": "#8a9a7b"}, actor="agent")
    out = store.set_finish(S, v.id, "room", "walls", {"color": "#8a9a7b"}, actor="agent")
    assert out["surface"]["color"] == "#8a9a7b"
    sc = store.load(S)
    assert sc.resolve_room(v.id).space("room").finish.walls.color == "#8a9a7b"
    assert sc.room.space("room").finish is None  # main untouched
    d = store.diff(S, "main", v.id)
    fin = [c for c in d.changes if c.kind == "finish"]
    assert len(fin) == 1 and fin[0].object_id == "finish:room:walls" and "#8a9a7b" in fin[0].name
    store.promote(S, v.id)
    assert store.load(S).room.space("room").finish.walls.color == "#8a9a7b"
    store.undo(S, "main")
    f = store.load(S).room.space("room").finish
    assert f is None or f.walls is None


def test_new_floor_material_gets_its_pattern_and_bad_colors_are_refused(store):
    v = store.create_variant(S, "oak", actor="agent")
    out = store.set_finish(S, v.id, "room", "floor", {"kind": "wood", "color": "#a07850", "module_m": 0.18}, actor="agent")
    sf = out["surface"]
    assert sf["kind"] == "wood" and sf["texture"].startswith("finishes/") and sf["module_m"] == 0.18
    assert (store.scene_dir(S) / sf["texture"]).exists()
    with pytest.raises(ValueError):
        store.set_finish(S, v.id, "room", "walls", {"color": "sage"}, actor="agent")
    with pytest.raises(KeyError):
        store.set_finish(S, v.id, "room", "wall:99", {"color": "#ffffff"}, actor="agent")
    store.undo(S, v.id, actor="agent")
    assert "floor" not in store.load(S).variant(v.id).finishes.get("room", {})


def test_repainting_walls_repaints_block_and_accent_walls():
    from mise.scene.models import Finish, Space, Surface, apply_finish

    sp = Space(id="r", name="R", polygon=[(0, 0), (1, 0), (1, 1)], finish=Finish(
        walls=Surface(color="#c0c0c0", kind="paint"),
        wall_overrides={"r.1": Surface(color="#a0a0a0", kind="block", texture="finishes/b.jpg", module_m=0.203)}))
    apply_finish(sp, "walls", Surface(color="#556b2f", kind="paint"))
    b = sp.finish.wall_overrides["r.1"]
    assert b.kind == "block" and b.texture == "finishes/b.jpg" and b.color == "#556b2f"


def test_mcp_finish_tools_and_renders_show_the_paint(client):
    mcp = build_server(client=client)

    async def run():
        vid = json.loads((await mcp.call_tool("create_variant", {"scene": S, "name": "paint"})).content[0].text)["variant_id"]
        before = json.loads((await mcp.call_tool("get_finishes", {"scene": S, "variant": vid})).content[0].text)
        assert before["spaces"][0]["space"] == "room"
        r = json.loads((await mcp.call_tool("set_finish", {"scene": S, "variant": vid, "space": "room", "part": "floor",
                                                           "color": "#203040"})).content[0].text)
        assert r["surface"]["color"] == "#203040"
        after = json.loads((await mcp.call_tool("get_finishes", {"scene": S, "variant": vid})).content[0].text)
        assert after["spaces"][0]["floor"]["color"] == "#203040" and after["spaces"][0]["changed"] == ["floor"]
        d = json.loads((await mcp.call_tool("diff_variants", {"scene": S, "a": "main", "b": vid})).content[0].text)
        assert [c["kind"] for c in d["changes"]] == ["finish"]

    asyncio.run(run())
    import io

    from PIL import Image
    vid = next(v["id"] for v in client.get(f"/api/scenes/{S}/variants").json() if v["id"] != "main")
    main = Image.open(io.BytesIO(client.get(f"/api/scenes/{S}/variants/main/render?view=eye").content)).convert("RGB")
    new = Image.open(io.BytesIO(client.get(f"/api/scenes/{S}/variants/{vid}/render?view=eye").content)).convert("RGB")
    near = lambda im: sum(1 for p in im.getdata() if abs(p[0] - 0x20) + abs(p[1] - 0x30) + abs(p[2] - 0x40) < 30)  # noqa: E731
    assert near(new) > near(main) + 20000  # the floor, in its new color
