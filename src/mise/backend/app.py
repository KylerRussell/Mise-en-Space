"""Scene backend: owns scene.json, applies ops, runs checks and imports, pushes changes.

REST under /api, WebSocket at /ws/scenes/{scene_id}, static scene assets
under /files/{scene_id}/..., and the built editor at / (if editor/dist exists).

Who is calling is sent in the ``X-Actor`` header (``user`` | ``agent``); the
MCP server sends ``agent``. The store enforces that agents write only to
variants and never move locked objects.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..scene import checks, placement, render
from ..scene import spaces as spaces_mod
from ..scene.models import MAIN, Dims, LibraryAsset, Pose, Scene, SceneObject, upright_required
from ..scene.store import PermissionError_, SceneStore

DATA_DIR = Path(os.environ.get("MISE_DATA", "data/scenes"))
EDITOR_DIST = Path(os.environ.get("MISE_EDITOR_DIST", Path(__file__).resolve().parents[3] / "editor" / "dist"))


# ---- request bodies -----------------------------------------------------------

class MoveBody(BaseModel):
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None
    roll: float | None = None
    pitch: float | None = None
    support: str | None = None
    snap: bool = True


class AddBody(BaseModel):
    asset_id: str | None = None  # from the library
    object: SceneObject | None = None  # or a full object (user uploads, boxes)
    pose: Pose | None = None
    relation: placement.Relation | None = None
    support: str = "floor"
    snap: bool = True


class UploadBody(BaseModel):
    name: str
    category: str = "furniture"
    w: float
    d: float
    h: float
    image_url: str | None = None
    upright_only: bool | None = None


class VariantBody(BaseModel):
    name: str
    base: str = MAIN
    notes: str = ""


class PromoteBody(BaseModel):
    object_ids: list[str] | None = None


class ImportBody(BaseModel):
    url: str
    category: str = "furniture"
    name: str | None = None
    dims: str | None = None
    image_urls: list[str] | None = None
    image_type: Literal["real", "synthetic", "auto"] = "auto"
    mesh: bool = False
    upright_only: bool | None = None


class UpdateBody(BaseModel):
    fields: dict[str, Any]


class SolveBody(BaseModel):
    constraints: list[dict]
    movable: list[str] | None = None
    iterations: int = 1500
    seed: int = 0
    apply: bool = True
    space: str | None = None  # keep movable objects inside this space


class SpaceBody(BaseModel):
    name: str
    kind: str = "other"
    polygon: list[tuple[float, float]]
    ceiling_height: float | None = None
    open_to: list[str] = []


class SpacePatch(BaseModel):
    name: str | None = None
    kind: str | None = None
    ceiling_height: float | None = None
    open_to: list[str] | None = None
    polygon: list[tuple[float, float]] | None = None


class SplitBody(BaseModel):
    axis: Literal["x", "y"]
    at: float
    new_name: str
    new_kind: str = "other"


class MergeBody(BaseModel):
    other: str


class OpeningBody(BaseModel):
    """A door or window: either on a wall (wall + center + width) or from two floor points
    (a, b: e.g. the jambs) that snap onto the nearest parallel wall."""
    id: str | None = None
    kind: Literal["door", "window"] = "door"
    wall: str | None = None
    center: float | None = None
    width: float | None = None
    a: tuple[float, float] | None = None
    b: tuple[float, float] | None = None
    bottom: float | None = None
    top: float | None = None
    swing: Literal["in", "out", "none", "sliding"] | None = None
    hinge: Literal["left", "right"] | None = None


# ---- jobs (Stage 4 imports run in a thread) ----------------------------------------

class Job(BaseModel):
    id: str
    kind: str
    status: Literal["queued", "running", "done", "needs_input", "failed"] = "queued"
    result: dict | None = None
    error: str | None = None
    question: str | None = None


def create_app(data_dir: Path | None = None, generator=None, listing_fetcher=None, image_fetcher=None) -> FastAPI:
    """generator: Stage 4 image-to-3D client (defaults to Idol if FAL_KEY is set).
    listing_fetcher / image_fetcher: injectable for tests (default: fetch over HTTP)."""
    store = SceneStore(data_dir or DATA_DIR)
    loop_holder: dict[str, asyncio.AbstractEventLoop] = {}

    @asynccontextmanager
    async def lifespan(_app):
        loop_holder["loop"] = asyncio.get_running_loop()
        yield

    app = FastAPI(title="Mise-en-Space scene backend", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    jobs: dict[str, Job] = {}
    sockets: dict[str, set[WebSocket]] = {}

    if generator is None and os.environ.get("FAL_KEY"):
        from ..products.idol import IdolClient
        generator = IdolClient()

    # ---- live push ------------------------------------------------------------
    def on_event(scene_id: str, event: dict) -> None:
        loop = loop_holder.get("loop")
        if not loop:
            return
        msg = json.dumps({"scene": scene_id, **event}, default=str)
        for ws in list(sockets.get(scene_id, ())):
            asyncio.run_coroutine_threadsafe(_send(ws, msg), loop)

    async def _send(ws: WebSocket, msg: str) -> None:
        try:
            await ws.send_text(msg)
        except Exception:
            for s in sockets.values():
                s.discard(ws)

    store.subscribe(on_event)

    @app.websocket("/ws/scenes/{scene_id}")
    async def ws_scene(ws: WebSocket, scene_id: str) -> None:
        await ws.accept()
        loop_holder["loop"] = asyncio.get_running_loop()
        sockets.setdefault(scene_id, set()).add(ws)
        try:
            while True:
                await ws.receive_text()  # keepalive pings from the editor
        except WebSocketDisconnect:
            sockets[scene_id].discard(ws)

    # ---- helpers ------------------------------------------------------------------
    def actor_of(x_actor: str | None) -> str:
        a = (x_actor or "user").lower()
        if a not in ("user", "agent", "pipeline"):
            raise HTTPException(400, "X-Actor must be user, agent or pipeline")
        return a

    def scene_or_404(scene_id: str) -> Scene:
        try:
            return store.load(scene_id)
        except KeyError:
            raise HTTPException(404, f"no scene {scene_id!r}")

    def guard(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except PermissionError_ as e:
            raise HTTPException(403, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'\""))
        except ValueError as e:
            raise HTTPException(422, str(e))

    def variant_ok(scene: Scene, variant: str) -> None:
        if variant != MAIN:
            guard(scene.variant, variant)

    # ---- scenes ---------------------------------------------------------------------
    @app.get("/api/scenes")
    def list_scenes():
        return store.list_scenes()

    @app.post("/api/scenes")
    def create_scene(scene: Scene, overwrite: bool = False):
        try:
            return store.create(scene, overwrite=overwrite).model_dump()
        except FileExistsError:
            raise HTTPException(409, f"scene {scene.id!r} exists")

    @app.get("/api/scenes/{scene_id}")
    def get_scene(scene_id: str):
        return scene_or_404(scene_id).model_dump()

    def space_of(room, o) -> str | None:
        sp = room.space_at(o.pose.x, o.pose.y)
        return sp.id if sp else None

    @app.get("/api/scenes/{scene_id}/room")
    def get_room(scene_id: str, variant: str = MAIN):
        from ..scene import geometry as g
        import numpy as np
        s = scene_or_404(scene_id)
        variant_ok(s, variant)
        objs = s.resolve(variant)
        floor_objs = [o for o in objs if o.support == "floor" and o.category not in checks.FLOOR_IGNORED]
        # free area per space, approximated by footprint union on a grid
        step = 0.05
        (lx, ly), (hx, hy) = s.room.bounds()
        xs, ys = np.meshgrid(np.arange(lx + step / 2, hx, step), np.arange(ly + step / 2, hy, step))
        pts = np.c_[xs.ravel(), ys.ravel()]
        occ = np.zeros(len(pts), bool)
        for o in floor_objs:
            occ |= g.points_in_polygon(pts, g.object_footprint(o))
        spaces_out, free_total = [], 0.0
        for sp in s.room.space_list():
            inside = g.points_in_polygon(pts, np.asarray(sp.polygon))
            free = float((inside & ~occ).sum() * step * step)
            free_total += free
            spaces_out.append({**sp.model_dump(), "area_m2": round(sp.area(), 2), "free_area_m2": round(free, 2),
                               "ceiling": s.room.ceiling_for(sp), "explicit": bool(s.room.spaces),
                               "open_with": [o.id for o in s.room.space_list() if o.id != sp.id and s.room.is_open(sp.id, o.id)],
                               "objects": [o.id for o in objs if space_of(s.room, o) == sp.id]})
        return {
            "floor_polygon": s.room.floor_polygon, "floor_polygons": s.room.polygons(),
            "floor_area_m2": round(s.room.floor_area(), 3),
            "free_floor_area_m2": round(free_total, 2), "ceiling_height": s.room.ceiling_height,
            "bounds": {"min": [lx, ly], "max": [hx, hy]},
            "spaces": spaces_out,
            "walls": [{**w.model_dump(), "length": round(w.length, 3),
                       "inward_normal": [round(float(v), 4) for v in g.wall_inward_normal(w, s.room)]} for w in s.room.walls],
            "openings": [{**o.model_dump(), "polygon": g.opening_polygon(o, s.room)} for o in s.room.openings],
            "rules": s.rules.model_dump(),
        }

    # ---- spaces (rooms and zones); only you can edit them -----------------------------
    @app.post("/api/scenes/{scene_id}/spaces")
    def add_space(scene_id: str, body: SpaceBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        made = {}

        def fn(r):
            r, made["id"] = spaces_mod.add_space(r, body.name, body.kind, body.polygon, body.ceiling_height, body.open_to)
            return r
        out = guard(store.edit_room, scene_id, fn, actor_of(x_actor), f"add space {body.name}")
        return {**out, "space_id": made["id"]}

    @app.patch("/api/scenes/{scene_id}/spaces/{space_id}")
    def update_space(scene_id: str, space_id: str, body: SpacePatch, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        fields = body.model_dump(exclude_unset=True)
        return guard(store.edit_room, scene_id, lambda r: spaces_mod.update_space(r, space_id, **fields),
                     actor_of(x_actor), f"update space {space_id}")

    @app.post("/api/scenes/{scene_id}/spaces/{space_id}/split")
    def split_space(scene_id: str, space_id: str, body: SplitBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        made = {}

        def fn(r):
            r, made["id"] = spaces_mod.split_space(r, space_id, body.axis, body.at, body.new_name, body.new_kind)
            return r
        out = guard(store.edit_room, scene_id, fn, actor_of(x_actor), f"split {space_id} at {body.axis}={body.at}")
        return {**out, "space_id": made["id"]}

    @app.post("/api/scenes/{scene_id}/spaces/{space_id}/merge")
    def merge_space(scene_id: str, space_id: str, body: MergeBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        return guard(store.edit_room, scene_id, lambda r: spaces_mod.merge_spaces(r, space_id, body.other),
                     actor_of(x_actor), f"merge {body.other} into {space_id}")

    @app.post("/api/scenes/{scene_id}/openings")
    def add_opening(scene_id: str, body: OpeningBody, x_actor: str | None = Header(None)):
        from ..scene.models import Opening
        s = scene_or_404(scene_id)
        oid = body.id or spaces_mod.slug(body.kind, {o.id for o in s.room.openings})
        extra = body.model_dump(exclude_none=True, include={"bottom", "top", "swing", "hinge"})

        def fn(r):
            if body.a is not None and body.b is not None:
                op = spaces_mod.door_from_points(r, body.a, body.b, opening_id=oid, kind=body.kind, **extra)
            elif body.wall and body.center is not None and body.width:
                op = Opening(id=oid, kind=body.kind, wall=body.wall, center=body.center, width=body.width, **extra)
            else:
                raise ValueError("give a and b (floor points), or wall, center and width")
            return spaces_mod.add_opening(r, op)
        out = guard(store.edit_room, scene_id, fn, actor_of(x_actor), f"add {body.kind} {oid}")
        return {**out, "opening_id": oid}

    @app.patch("/api/scenes/{scene_id}/openings/{opening_id}")
    def update_opening(scene_id: str, opening_id: str, body: OpeningBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        fields = body.model_dump(exclude_none=True, exclude={"id", "a", "b"})
        return guard(store.edit_room, scene_id, lambda r: spaces_mod.update_opening(r, opening_id, **fields),
                     actor_of(x_actor), f"update {opening_id}")

    @app.delete("/api/scenes/{scene_id}/openings/{opening_id}")
    def delete_opening(scene_id: str, opening_id: str, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        return guard(store.edit_room, scene_id, lambda r: spaces_mod.delete_opening(r, opening_id),
                     actor_of(x_actor), f"delete {opening_id}")

    @app.delete("/api/scenes/{scene_id}/spaces/{space_id}")
    def delete_space(scene_id: str, space_id: str, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        return guard(store.edit_room, scene_id, lambda r: spaces_mod.delete_space(r, space_id),
                     actor_of(x_actor), f"delete space {space_id}")

    @app.get("/api/scenes/{scene_id}/variants")
    def list_variants(scene_id: str):
        s = scene_or_404(scene_id)
        return [{"id": MAIN, "name": "Main layout", "created_by": "user"}] + [
            {"id": v.id, "name": v.name, "created_by": v.created_by, "created_at": v.created_at, "notes": v.notes}
            for v in s.variants]

    @app.post("/api/scenes/{scene_id}/variants")
    def create_variant(scene_id: str, body: VariantBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        v = guard(store.create_variant, scene_id, body.name, body.base, actor_of(x_actor), body.notes)
        return {"variant_id": v.id, "name": v.name}

    @app.delete("/api/scenes/{scene_id}/variants/{variant}")
    def delete_variant(scene_id: str, variant: str, x_actor: str | None = Header(None)):
        guard(store.delete_variant, scene_id, variant, actor_of(x_actor))
        return {"deleted": variant}

    @app.get("/api/scenes/{scene_id}/variants/{variant}/objects")
    def list_objects(scene_id: str, variant: str, include_hidden: bool = False, space: str | None = None):
        s = scene_or_404(scene_id)
        variant_ok(s, variant)
        if space:
            guard(s.room.space, space)
        objs = s.resolve(variant, include_hidden=include_hidden)
        hidden = set(s.variant(variant).hidden) if variant != MAIN else set()
        out = [{**o.model_dump(), "label": i + 1, "hidden": o.id in hidden, "extent": o.extent().model_dump(),
                "upright_required": o.upright_required, "space": space_of(s.room, o)} for i, o in enumerate(objs)]
        return [o for o in out if not space or o["space"] == space]

    # ---- edits ---------------------------------------------------------------------------
    @app.post("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}/move")
    def move(scene_id: str, variant: str, object_id: str, body: MoveBody, x_actor: str | None = Header(None)):
        scene_or_404(scene_id)
        return guard(store.move, scene_id, variant, object_id, x=body.x, y=body.y, z=body.z, yaw=body.yaw,
                     support=body.support, snap=body.snap, roll=body.roll, pitch=body.pitch,
                     actor=actor_of(x_actor))

    @app.post("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}/place")
    def place(scene_id: str, variant: str, object_id: str, rel: placement.Relation, apply: bool = True,
              x_actor: str | None = Header(None)):
        s = scene_or_404(scene_id)
        obj = guard(s.find, variant, object_id)
        objs = {o.id: o for o in s.resolve(variant)}
        p = guard(placement.resolve_relation, obj, rel, s.room, objs)
        out = {"pose": p.pose.model_dump(), "support": p.support, "note": p.note}
        if apply:
            out["applied"] = guard(store.move, scene_id, variant, object_id, x=p.pose.x, y=p.pose.y, yaw=p.pose.yaw,
                                   support=p.support, snap=False, actor=actor_of(x_actor))
        return out

    @app.post("/api/scenes/{scene_id}/variants/{variant}/objects")
    def add(scene_id: str, variant: str, body: AddBody, x_actor: str | None = Header(None)):
        s = scene_or_404(scene_id)
        actor = actor_of(x_actor)
        if body.asset_id:
            a = guard(s.library_asset, body.asset_id)
            obj = SceneObject(id=store.new_object_id(s, a.category), name=a.name, category=a.category, asset=a.asset,
                              asset_kind=a.asset_kind, asset_transform=a.asset_transform, dims=a.dims, upright_only=a.upright_only,
                              source=a.source, product=a.product, anisotropy=a.anisotropy, warnings=list(a.warnings),
                              support=body.support)
        elif body.object:
            obj = body.object
        else:
            raise HTTPException(422, "give asset_id or object")
        if body.pose:
            obj.pose = body.pose
        note = None
        if body.relation:
            objs = {o.id: o for o in s.resolve(variant)}
            p = guard(placement.resolve_relation, obj, body.relation, s.room, objs)
            obj.pose, obj.support, note = p.pose, p.support, p.note
        res = guard(store.add_object, scene_id, variant, obj, snap=body.snap and not body.relation, actor=actor)
        if note:
            res["placement"] = note
        return res

    @app.post("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}/hide")
    def hide(scene_id: str, variant: str, object_id: str, x_actor: str | None = Header(None)):
        return guard(store.set_hidden, scene_id, variant, object_id, True, actor_of(x_actor))

    @app.post("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}/show")
    def show(scene_id: str, variant: str, object_id: str, x_actor: str | None = Header(None)):
        return guard(store.set_hidden, scene_id, variant, object_id, False, actor_of(x_actor))

    @app.delete("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}")
    def delete(scene_id: str, variant: str, object_id: str, x_actor: str | None = Header(None)):
        return guard(store.delete_object, scene_id, variant, object_id, actor_of(x_actor))

    @app.patch("/api/scenes/{scene_id}/variants/{variant}/objects/{object_id}")
    def update(scene_id: str, variant: str, object_id: str, body: UpdateBody, x_actor: str | None = Header(None)):
        return guard(store.update_object, scene_id, variant, object_id, body.fields, actor_of(x_actor))

    @app.post("/api/scenes/{scene_id}/variants/{variant}/undo")
    def undo(scene_id: str, variant: str, x_actor: str | None = Header(None)):
        op = guard(store.undo, scene_id, variant, actor_of(x_actor))
        return {"undone": op.model_dump() if op else None}

    @app.post("/api/scenes/{scene_id}/variants/{variant}/redo")
    def redo(scene_id: str, variant: str, x_actor: str | None = Header(None)):
        op = guard(store.redo, scene_id, variant, actor_of(x_actor))
        return {"redone": op.model_dump() if op else None}

    @app.get("/api/scenes/{scene_id}/ops")
    def ops(scene_id: str, variant: str | None = None, limit: int = 100):
        scene_or_404(scene_id)
        return [o.model_dump() for o in store.ops(scene_id, variant)[-limit:]]

    # ---- checks, renders, diffs ---------------------------------------------------------
    @app.get("/api/scenes/{scene_id}/variants/{variant}/check")
    def check(scene_id: str, variant: str, space: str | None = None):
        s = scene_or_404(scene_id)
        variant_ok(s, variant)
        vs = checks.check_layout(s, variant)
        if space:
            guard(s.room.space, space)
            ids = {o.id for o in s.resolve(variant) if space_of(s.room, o) == space}
            vs = [v for v in vs if (set(v.objects) & ids) or v.kind == "walkway"]
        return {"hard": sum(v.severity == "hard" for v in vs), "soft": sum(v.severity == "soft" for v in vs),
                "violations": [v.model_dump() for v in vs]}

    @app.get("/api/scenes/{scene_id}/variants/{variant}/render")
    def render_png(scene_id: str, variant: str, view: Literal["plan", "eye", "object"] = "plan",
                   object_id: str | None = None, eye: str | None = None, target: str | None = None,
                   labels_header: bool = True, space: str | None = None):
        s = scene_or_404(scene_id)
        variant_ok(s, variant)
        objs = s.resolve(variant)
        e = tuple(map(float, eye.split(","))) if eye else None
        t = tuple(map(float, target.split(","))) if target else None
        png, labels = guard(render.render_view, s.room, objs, view, checks.check_objects(s.room, objs, s.rules),
                            object_id, e, t, space)
        headers = {"X-Label-Map": json.dumps(labels)} if labels_header else {}
        return Response(png, media_type="image/png", headers=headers)

    @app.get("/api/scenes/{scene_id}/diff")
    def diff(scene_id: str, a: str = MAIN, b: str = Query(...)):
        scene_or_404(scene_id)
        return guard(store.diff, scene_id, a, b).model_dump()

    @app.post("/api/scenes/{scene_id}/variants/{variant}/promote")
    def promote(scene_id: str, variant: str, body: PromoteBody, x_actor: str | None = Header(None)):
        return guard(store.promote, scene_id, variant, body.object_ids, actor_of(x_actor))

    @app.post("/api/scenes/{scene_id}/variants/{variant}/solve")
    def solve(scene_id: str, variant: str, body: SolveBody, x_actor: str | None = Header(None)):
        from ..agent.solver import Constraint, solve_layout
        s = scene_or_404(scene_id)
        actor = actor_of(x_actor)
        if body.apply and actor == "agent" and variant == MAIN:
            raise HTTPException(403, "the AI writes only to variants; call create_variant first")
        cons = [guard(Constraint.model_validate, c) for c in body.constraints]
        res = guard(solve_layout, s, variant, cons, body.movable, body.iterations, body.seed, space=body.space)
        if body.apply:
            for oid, pose in res.poses.items():
                guard(store.move, scene_id, variant, oid, x=pose.x, y=pose.y, yaw=pose.yaw, snap=False, actor=actor)
        return res.model_dump()

    @app.get("/api/scenes/{scene_id}/variants/{variant}/shopping")
    def shopping(scene_id: str, variant: str):
        from ..agent.shopping import shopping_list
        s = scene_or_404(scene_id)
        variant_ok(s, variant)
        return shopping_list(s, variant).model_dump()

    # ---- library and imports ------------------------------------------------------------
    @app.get("/api/scenes/{scene_id}/library")
    def library(scene_id: str):
        return [a.model_dump() for a in scene_or_404(scene_id).library]

    @app.post("/api/scenes/{scene_id}/library/upload")
    def upload(scene_id: str, body: UploadBody, x_actor: str | None = Header(None)):
        """An uploaded photo plus typed dimensions: placed as a sized box until generated."""
        s = scene_or_404(scene_id)
        a = LibraryAsset(id=f"upl_{uuid.uuid4().hex[:8]}", name=body.name, category=body.category, asset_kind="box",
                         dims=Dims(w=body.w, d=body.d, h=body.h), source="upload", status="fallback_box",
                         upright_only=body.upright_only,
                         warnings=["typed dimensions; no 3D model yet"])
        store.add_library_asset(scene_id, a, actor_of(x_actor))
        return a.model_dump()

    @app.post("/api/scenes/{scene_id}/imports")
    def start_import(scene_id: str, body: ImportBody):
        scene_or_404(scene_id)
        job = Job(id=f"job_{uuid.uuid4().hex[:8]}", kind="import_product")
        jobs[job.id] = job

        def work():
            from ..products.importer import ImportRequest, NeedsInput, run_import
            job.status = "running"
            try:
                listing = listing_fetcher(body.url) if listing_fetcher else None
                req = ImportRequest(url=body.url, category=body.category, name=body.name, dims_override=body.dims,
                                    image_urls=body.image_urls, image_type=body.image_type, mesh=body.mesh,
                                    upright_only=body.upright_only)
                res = run_import(req, store.scene_dir(scene_id) / "assets", generator, listing=listing,
                                 fetch_image=image_fetcher or _fetch_bytes)
                store.add_library_asset(scene_id, res.asset)
                a = res.asset
                job.result = {"asset_id": a.id, "name": a.name, "dims": a.dims.model_dump(), "anisotropy": a.anisotropy,
                              "upright_only": upright_required(a.category, a.upright_only),
                              "status": a.status, "warnings": a.warnings, "attempts": res.attempts,
                              "product": a.product.model_dump() if a.product else None}
                job.status = "done"
            except NeedsInput as e:
                job.status, job.question = "needs_input", e.question
            except Exception as e:  # report, don't crash the server
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"

        threading.Thread(target=work, daemon=True).start()
        return job.model_dump()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        if job_id not in jobs:
            raise HTTPException(404, "no such job")
        return jobs[job_id].model_dump()

    # ---- files and editor -----------------------------------------------------------------
    @app.get("/files/{scene_id}/{path:path}")
    def files(scene_id: str, path: str):
        base = store.scene_dir(scene_id).resolve()
        p = (base / path).resolve()
        if base not in p.parents or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p)

    @app.get("/healthz")
    def health():
        return JSONResponse({"ok": True})

    if EDITOR_DIST.exists():
        app.mount("/", StaticFiles(directory=EDITOR_DIST, html=True), name="editor")
    return app


def _fetch_bytes(url: str) -> bytes:
    import httpx
    r = httpx.get(url, follow_redirects=True, timeout=20)
    r.raise_for_status()
    return r.content
