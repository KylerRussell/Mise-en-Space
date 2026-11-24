import asyncio
import json
import time

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from mise.agent.shopping import recheck, shopping_list
from mise.agent.solver import Constraint, solve_layout
from mise.eval.hypotheses import BlindStudy, h4
from mise.mcp_server.server import build_server

from .conftest import fake_listing

S = "demo-bedroom"
AGENT = {"X-Actor": "agent"}


def test_backend_basics(client):
    assert client.get("/api/scenes").json()[0]["id"] == S
    room = client.get(f"/api/scenes/{S}/room").json()
    assert room["floor_area_m2"] == 15.12 and len(room["walls"]) == 4 and room["openings"][0]["polygon"]
    r = client.get(f"/api/scenes/{S}/variants/main/render?view=plan")
    assert r.headers["content-type"] == "image/png" and json.loads(r.headers["X-Label-Map"])["1"] == "bed_1"
    assert client.get(f"/api/scenes/{S}/variants/main/check").json()["hard"] == 0


def test_agent_is_sandboxed(client):
    r = client.post(f"/api/scenes/{S}/variants/main/objects/desk_1/move", json={"x": 2}, headers=AGENT)
    assert r.status_code == 403
    vid = client.post(f"/api/scenes/{S}/variants", json={"name": "ai"}, headers=AGENT).json()["variant_id"]
    assert client.post(f"/api/scenes/{S}/variants/{vid}/objects/bed_1/move", json={"x": 2}, headers=AGENT).status_code == 403
    assert client.post(f"/api/scenes/{S}/variants/{vid}/promote", json={}, headers=AGENT).status_code == 403


def test_import_and_place_product(client):
    job = client.post(f"/api/scenes/{S}/imports", json={"url": "https://shop.example/sofa", "category": "sofa"}).json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job["status"] == "done", job
    res = job["result"]
    assert res["anisotropy"] <= 1.1 and res["product"]["price"] == 899.0
    vid = client.post(f"/api/scenes/{S}/variants", json={"name": "ai"}, headers=AGENT).json()["variant_id"]
    add = client.post(f"/api/scenes/{S}/variants/{vid}/objects", headers=AGENT,
                      json={"asset_id": res["asset_id"], "relation": {"relation": "against_wall", "target": "2", "along": 1.6}}).json()
    assert add["object_id"].startswith("sofa")
    shop = client.get(f"/api/scenes/{S}/variants/{vid}/shopping").json()
    assert shop["total"] == 899.0 and shop["rows"][0]["link"] == "https://shop.example/sofa"
    diff = client.get(f"/api/scenes/{S}/diff", params={"a": "main", "b": vid}).json()
    assert diff["cost_delta"] == 899.0
    # the asset file is served
    obj = next(o for o in client.get(f"/api/scenes/{S}/variants/{vid}/objects").json() if o["id"] == add["object_id"])
    assert client.get(f"/files/{S}/{obj['asset']}").status_code == 200
    assert client.get(f"/files/{S}/../../etc/passwd").status_code == 404


def test_solver_resolves_collisions(scene):
    desk = next(o for o in scene.objects if o.id == "desk_1")
    chair = next(o for o in scene.objects if o.id == "chair_1")
    desk.pose.x, desk.pose.y, desk.pose.yaw = 2.0, 1.0, 30
    chair.pose.x, chair.pose.y = 2.0, 1.2
    res = solve_layout(scene, "main", [
        Constraint(kind="against_wall", obj="desk_1", target="2"),
        Constraint(kind="in_front_of", obj="chair_1", target="desk_1", min=0.0, max=0.3),
    ], iterations=800)
    assert res.hard_energy == 0
    assert not [v for v in res.violations if v.severity == "hard"], [v.message for v in res.violations]
    assert all(c.satisfied for c in res.constraints)


def test_mcp_tools_end_to_end(client):
    mcp = build_server(client=client)

    async def run():
        names = {t.name for t in await mcp.list_tools()}
        assert {"get_room", "list_objects", "render_view", "check_layout", "create_variant", "move_object",
                "place_relative", "hide_object", "show_object", "search_products", "import_product",
                "add_object", "diff_variants"} <= names
        v = json.loads((await mcp.call_tool("create_variant", {"scene": S, "name": "AI try"})).content[0].text)
        vid = v["variant_id"]
        with pytest.raises(ToolError, match="only to variants"):
            await mcp.call_tool("move_object", {"scene": S, "variant": "main", "object_id": "desk_1", "x": 1.0})
        r = await mcp.call_tool("move_object", {"scene": S, "variant": vid, "object_id": "nightstand_1", "x": 0.7, "y": 0.4, "snap": False})
        body = json.loads(r.content[0].text)
        assert any(x["kind"] == "blocked_door" for x in body["violations_caused"])
        img = await mcp.call_tool("render_view", {"scene": S, "variant": vid, "view": "plan"})
        assert img.content[0].type == "image" and "label map" in img.content[1].text
        r = await mcp.call_tool("hide_object", {"scene": S, "variant": vid, "object_id": "rug_1"})
        d = json.loads((await mcp.call_tool("diff_variants", {"scene": S, "a": "main", "b": vid})).content[0].text)
        assert {c["kind"] for c in d["changes"]} == {"move", "remove"}
        imp = json.loads((await mcp.call_tool("import_product", {"scene": S, "url": "https://shop.example/sofa", "category": "sofa"})).content[0].text)
        assert imp["status"] == "done"
        add = json.loads((await mcp.call_tool("add_object", {"scene": S, "variant": vid, "asset_id": imp["result"]["asset_id"],
                                                              "relation": "against_wall", "target": "2", "along": 1.6})).content[0].text)
        assert "object_id" in add
        chk = json.loads((await mcp.call_tool("check_layout", {"scene": S, "variant": vid})).content[0].text)
        assert chk["hard"] >= 1  # the nightstand still blocks the door

    asyncio.run(run())


def test_shopping_recheck_and_eval_helpers(client):
    vid = client.post(f"/api/scenes/{S}/variants", json={"name": "x"}).json()["variant_id"]
    client.post(f"/api/scenes/{S}/variants/{vid}/objects", json={"object": {
        "id": "chair_9", "name": "Chair", "category": "chair", "dims": {"w": 0.5, "d": 0.5, "h": 0.9}, "source": "product",
        "product": {"url": "https://shop.example/c", "price": 120, "currency": "USD", "listed_dims": {"w": 0.5, "d": 0.5, "h": 0.9}}},
        "pose": {"x": 2.5, "y": 2.0}})
    sc = client.app.state.store.load(S)
    sl = shopping_list(sc, vid)
    assert sl.total == 120 and "| Chair" in sl.to_markdown().replace("Chair |", "| Chair")
    res = recheck(sl, lambda url: fake_listing(url, dims="W 50 x D 50 x H 90 cm"))
    assert res["rate"] == 1.0 and res["pass"]
    assert h4([1.02] * 17 + [1.3] * 3)["prediction_held"]
    st = BlindStudy.build([("b1", "va", "vb"), ("b2", "vc", "vd")], seed=1)
    for p in st.pairs:
        p.choice = p.solver_side
    assert st.score()["solver_win_rate"] == 1.0
    assert all("solver_side" not in row for row in st.judge_sheet())
