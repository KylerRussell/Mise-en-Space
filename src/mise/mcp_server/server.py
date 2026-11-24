"""MCP server: a thin, stateless layer over the scene backend.

Eyes (labeled renders), hands (pose edits in a sandboxed variant) and a
referee (layout checks). Every call names its scene and variant; all state
lives in the backend. Every request is sent as ``X-Actor: agent`` so the
backend refuses writes to the main layout and moves of locked objects.

Run:  mise mcp                      (stdio, for Claude Desktop / Claude Code)
      mise mcp --http --port 8766   (Streamable HTTP, if it runs elsewhere)
Backend URL: MISE_BACKEND (default http://127.0.0.1:8765).
"""

from __future__ import annotations

import json
import os
import time
from typing import Annotated, Any, Literal

import httpx
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

INSTRUCTIONS = """\
Tools for redesigning a real, to-scale 3D room. Units are meters, yaw is degrees
counter-clockwise about +z, z is up and the floor is z = 0. An object's front faces
its local +y. Objects are numbered in renders; the label map ties numbers to ids.

Orientation: objects can be tipped in 90° steps (roll about the front axis, pitch about
the width axis) unless upright_required is true. `dims` is the object as modeled; `extent`
is its size as placed. Many things work more than one way up: a cube shelf listed lying
down often stands well on its end. When an object may tip, consider both ways and keep the
one that fits the room and the brief better. Never tip an upright_required object.

Rules:
- You can only edit variants. Start with create_variant; the user's own layout ("main")
  is read-only to you. The user promotes your changes from the editor, item by item.
- Locked objects never move.
- After each edit, read the returned violations (numeric, in meters) and fix them.
  Run check_layout before presenting a variant: aim for zero hard violations.
- Use render_view(plan) to see the layout and render_view(eye) to judge it as a person would.
- For new furniture: search the web yourself for products with listed dimensions, then
  import_product(url) and add_object(asset_id). Drop products that fail the size gate.
- For several objects at once, propose relations with solve_layout and let the solver
  resolve collisions and clearances.
"""

Scene = Annotated[str, Field(description="scene id (see list_scenes)")]
Variant = Annotated[str, Field(description="variant id; 'main' is the user's layout (read-only for you)")]


def build_server(client: httpx.Client | None = None, backend: str | None = None) -> MCPServer:
    http = client or httpx.Client(base_url=backend or os.environ.get("MISE_BACKEND", "http://127.0.0.1:8765"),
                                  timeout=120)
    headers = {"X-Actor": "agent"}

    def call(method: str, path: str, **kw) -> Any:
        r = http.request(method, path, headers=headers, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except (json.JSONDecodeError, AttributeError):
                detail = r.text
            raise ToolError(f"{r.status_code}: {detail}")
        if r.headers.get("content-type", "").startswith("image/"):
            return r
        return r.json()

    mcp = MCPServer(name="mise-en-space", instructions=INSTRUCTIONS, version="0.1.0")

    @mcp.tool()
    def list_scenes() -> list[dict]:
        """List scenes in the backend."""
        return call("GET", "/api/scenes")

    @mcp.tool()
    def get_room(scene: Scene, variant: Variant = "main") -> dict:
        """Room size, floor polygon, walls (with inward normals), doors, windows, free floor area and layout rules."""
        return call("GET", f"/api/scenes/{scene}/room", params={"variant": variant})

    @mcp.tool()
    def list_variants(scene: Scene) -> list[dict]:
        """List layouts: 'main' plus named variants."""
        return call("GET", f"/api/scenes/{scene}/variants")

    @mcp.tool()
    def list_objects(scene: Scene, variant: Variant = "main", include_hidden: bool = False) -> list[dict]:
        """Every object in a variant: render label, dims as modeled, extent as placed (after tipping), pose,
        support, lock flag, whether it must stay upright, and product info."""
        objs = call("GET", f"/api/scenes/{scene}/variants/{variant}/objects", params={"include_hidden": include_hidden})
        keep = ("label", "id", "name", "category", "dims", "extent", "pose", "support", "locked", "upright_required",
                "source", "hidden", "warnings", "anisotropy")
        out = []
        for o in objs:
            row = {k: o.get(k) for k in keep}
            if o.get("product"):
                p = o["product"]
                row["product"] = {k: p.get(k) for k in ("title", "retailer", "price", "currency", "url")}
            out.append(row)
        return out

    @mcp.tool(structured_output=False)
    def render_view(scene: Scene, variant: Variant = "main",
                    view: Literal["plan", "eye", "object"] = "plan",
                    object_id: str | None = None,
                    eye: Annotated[list[float] | None, Field(description="[x, y, z] camera position for view='eye'")] = None,
                    target: Annotated[list[float] | None, Field(description="[x, y, z] look-at point for view='eye'")] = None) -> list:
        """Render a plan, eye-level or object close-up view with numbered labels. Returns the image and a label map."""
        params: dict[str, Any] = {"view": view}
        if object_id:
            params["object_id"] = object_id
        if eye:
            params["eye"] = ",".join(map(str, eye))
        if target:
            params["target"] = ",".join(map(str, target))
        r = call("GET", f"/api/scenes/{scene}/variants/{variant}/render", params=params)
        labels = json.loads(r.headers.get("X-Label-Map", "{}"))
        return [Image(data=r.content, format="png"),
                "label map (number -> object id): " + json.dumps(labels)]

    @mcp.tool()
    def check_layout(scene: Scene, variant: Variant) -> dict:
        """Collisions, walkway clearance, blocked doors and windows, objects outside the room. Distances in meters."""
        return call("GET", f"/api/scenes/{scene}/variants/{variant}/check")

    @mcp.tool()
    def create_variant(scene: Scene, name: str, base: Variant = "main", notes: str = "") -> dict:
        """Copy a layout into a new variant you can edit. You never edit 'main' directly."""
        return call("POST", f"/api/scenes/{scene}/variants", json={"name": name, "base": base, "notes": notes})

    @mcp.tool()
    def move_object(scene: Scene, variant: Variant, object_id: str,
                    x: float | None = None, y: float | None = None, yaw: float | None = None,
                    roll: Annotated[float | None, Field(description=(
                        "tip about the object's front axis in 90° steps (0, 90, 180, 270): 90 stands a wide, "
                        "low shelf on its end. Refused for upright_required objects"))] = None,
                    pitch: Annotated[float | None, Field(description="tip about the width axis in 90° steps")] = None,
                    snap: Annotated[bool, Field(description="snap yaw to 15° and backs to nearby walls")] = True) -> dict:
        """Set x, y, yaw (deg) and optionally tip it (roll/pitch). Keeps the object on its support.
        Returns the new pose and violations this move caused."""
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/objects/{object_id}/move",
                    json={"x": x, "y": y, "yaw": yaw, "roll": roll, "pitch": pitch, "snap": snap})

    @mcp.tool()
    def place_relative(scene: Scene, variant: Variant, object_id: str,
                       relation: Literal["against_wall", "facing", "in_front_of", "behind", "left_of", "right_of", "on_top_of", "center"],
                       target: Annotated[str | None, Field(description="wall id for against_wall (e.g. '2'), else an object id")] = None,
                       distance: Annotated[float, Field(description="gap in meters")] = 0.0,
                       along: Annotated[float | None, Field(description="against_wall: meters from the wall's start")] = None,
                       face_target: bool = False,
                       apply: bool = True) -> dict:
        """Place by relation: "against wall 2", "facing bed_1", "0.1 m left of bed_1". Returns the resolved pose (and applies it)."""
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/objects/{object_id}/place",
                    params={"apply": apply},
                    json={"relation": relation, "target": target, "distance": distance, "along": along,
                          "face_target": face_target})

    @mcp.tool()
    def hide_object(scene: Scene, variant: Variant, object_id: str) -> dict:
        """Remove an existing item from a variant (restorable with show_object)."""
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/objects/{object_id}/hide")

    @mcp.tool()
    def show_object(scene: Scene, variant: Variant, object_id: str) -> dict:
        """Restore an item hidden in a variant."""
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/objects/{object_id}/show")

    @mcp.tool()
    def search_products(query: str, max_width: float | None = None, max_depth: float | None = None,
                        max_height: float | None = None, max_price: float | None = None) -> dict:
        """Retailer search filtered by footprint and budget. Only retailer APIs that are configured are searched;
        otherwise use your own web search, then import_product on a product URL."""
        from ..products.search import search
        return search(query, max_width, max_depth, max_height, max_price)

    @mcp.tool()
    def import_product(scene: Scene, url: str,
                       category: Annotated[str, Field(description="e.g. sofa, armchair, desk, lamp, rug, bookshelf")] = "furniture",
                       name: str | None = None,
                       dims: Annotated[str | None, Field(description="dimension string if the page lacks one, e.g. '84\"W x 38\"D x 34\"H'")] = None,
                       image_urls: list[str] | None = None,
                       image_type: Literal["real", "synthetic", "auto"] = "auto",
                       upright_only: Annotated[bool | None, Field(description=(
                           "true if it only works one way up, false if it may stand on any side; "
                           "omit to use the category default (sofas, beds, lamps... are upright-only)"))] = None,
                       wait_seconds: Annotated[int, Field(description="how long to wait for the job (0 = return the job id)")] = 240) -> dict:
        """Turn a product URL into a 3D asset sized to its listed dimensions (async job).
        Returns asset id, dims, anisotropy and warnings. status 'fallback_box' means generation failed the size gate."""
        job = call("POST", f"/api/scenes/{scene}/imports",
                   json={"url": url, "category": category, "name": name, "dims": dims, "image_urls": image_urls,
                         "image_type": image_type, "upright_only": upright_only})
        deadline = time.time() + max(0, wait_seconds)
        while job["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(1.0)
            job = call("GET", f"/api/jobs/{job['id']}")
        return job

    @mcp.tool()
    def get_job(job_id: str) -> dict:
        """Status of an import job."""
        return call("GET", f"/api/jobs/{job_id}")

    @mcp.tool()
    def add_object(scene: Scene, variant: Variant, asset_id: str,
                   x: float | None = None, y: float | None = None, yaw: float = 0.0,
                   relation: Literal["against_wall", "facing", "in_front_of", "behind", "left_of", "right_of", "on_top_of", "center"] | None = None,
                   target: str | None = None, distance: float = 0.0, along: float | None = None) -> dict:
        """Place an imported asset by pose (x, y, yaw) or by relation. Returns the object id and violations caused."""
        body: dict[str, Any] = {"asset_id": asset_id}
        if relation:
            body["relation"] = {"relation": relation, "target": target, "distance": distance, "along": along}
        elif x is not None and y is not None:
            body["pose"] = {"x": x, "y": y, "yaw": yaw}
        else:
            raise ToolError("give x and y, or a relation")
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/objects", json=body)

    @mcp.tool()
    def solve_layout(scene: Scene, variant: Variant,
                     constraints: Annotated[list[dict], Field(description=(
                         "relational constraints, each {kind, obj, target?, min?, max?, at?, weight?}; kinds: against_wall, "
                         "centered_on_wall, near, far_from, distance, facing, in_front_of, beside, parallel, "
                         "away_from_window, at"))],
                     movable: list[str] | None = None, iterations: int = 1500, seed: int = 0) -> dict:
        """Resolve poses for several objects from relations plus physics (no overlaps, inside the room,
        door swings and walkways clear). Applies the result to the variant."""
        res = call("POST", f"/api/scenes/{scene}/variants/{variant}/solve",
                   json={"constraints": constraints, "movable": movable, "iterations": iterations, "seed": seed})
        res["constraints"] = [{"kind": c["constraint"]["kind"], "obj": c["constraint"]["obj"],
                               "target": c["constraint"].get("target"), "cost": c["cost"], "satisfied": c["satisfied"]}
                              for c in res["constraints"]]
        return res

    @mcp.tool()
    def diff_variants(scene: Scene, a: Variant, b: Variant) -> dict:
        """Change list from layout a to layout b: moves, adds, removals and the cost delta."""
        return call("GET", f"/api/scenes/{scene}/diff", params={"a": a, "b": b})

    @mcp.tool()
    def shopping_list(scene: Scene, variant: Variant) -> dict:
        """Items to buy for a variant: item, retailer, price, listed W x D x H, link, date checked, and the total."""
        return call("GET", f"/api/scenes/{scene}/variants/{variant}/shopping")

    @mcp.tool()
    def undo(scene: Scene, variant: Variant) -> dict:
        """Undo your last change in a variant."""
        return call("POST", f"/api/scenes/{scene}/variants/{variant}/undo")

    return mcp


def main(http: bool = False, host: str = "127.0.0.1", port: int = 8766) -> None:
    server = build_server()
    if http:
        server.run("streamable-http", host=host, port=port)
    else:
        server.run("stdio")
