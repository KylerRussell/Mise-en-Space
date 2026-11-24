"""Scene store: the single source of truth for `scene.json`.

Every change is an :class:`Op` recorded in an append-only log
(`ops.jsonl`). Ops carry the state before and after, so undo/redo is just
applying the inverse, and an AI proposal (a variant) can be accepted change
by change with :meth:`SceneStore.promote`.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import checks, placement, spaces
from .models import MAIN, LibraryAsset, Pose, Room, Scene, SceneObject, Variant, now_iso

Actor = Literal["user", "agent", "pipeline"]


class PermissionError_(Exception):
    """An actor tried to do something it is not allowed to."""


class Op(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    ts: str = Field(default_factory=now_iso)
    actor: Actor = "user"
    variant: str = MAIN
    kind: Literal["move", "add", "hide", "show", "delete", "update", "create_variant", "delete_variant", "library_add", "promote", "room"]
    object_id: str | None = None
    before: dict | None = None
    after: dict | None = None
    note: str = ""
    undone: bool = False


class Change(BaseModel):
    """One entry in a diff between two layouts."""

    kind: Literal["move", "add", "remove", "restore"]
    object_id: str
    name: str
    before: Pose | None = None
    after: Pose | None = None
    distance_m: float | None = None
    rotation_deg: float | None = None
    cost: float | None = None
    currency: str | None = None


class Diff(BaseModel):
    a: str
    b: str
    changes: list[Change]
    cost_delta: float
    currency: str | None


class SceneStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._scenes: dict[str, Scene] = {}
        self._ops: dict[str, list[Op]] = {}
        self._listeners: list[Callable[[str, dict], None]] = []
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- persistence -------------------------------------------------------
    def scene_dir(self, scene_id: str) -> Path:
        return self.root / scene_id

    def list_scenes(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*/scene.json")):
            data = json.loads(p.read_text())
            out.append({"id": data["id"], "name": data["name"], "updated_at": data.get("updated_at")})
        return out

    def load(self, scene_id: str) -> Scene:
        with self._lock:
            if scene_id not in self._scenes:
                path = self.scene_dir(scene_id) / "scene.json"
                if not path.exists():
                    raise KeyError(f"no scene {scene_id!r}")
                self._scenes[scene_id] = Scene.model_validate_json(path.read_text())
                ops_path = self.scene_dir(scene_id) / "ops.jsonl"
                self._ops[scene_id] = (
                    [Op.model_validate_json(l) for l in ops_path.read_text().splitlines() if l.strip()]
                    if ops_path.exists() else []
                )
            return self._scenes[scene_id]

    def create(self, scene: Scene, overwrite: bool = False) -> Scene:
        with self._lock:
            d = self.scene_dir(scene.id)
            if (d / "scene.json").exists() and not overwrite:
                raise FileExistsError(scene.id)
            d.mkdir(parents=True, exist_ok=True)
            (d / "ops.jsonl").write_text("")
            self._scenes[scene.id] = scene
            self._ops[scene.id] = []
            self._save(scene.id)
            return scene

    def _save(self, scene_id: str) -> None:
        scene = self._scenes[scene_id]
        scene.updated_at = now_iso()
        path = self.scene_dir(scene_id) / "scene.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(scene.model_dump_json(indent=2))
        tmp.replace(path)

    def _log(self, scene_id: str, op: Op) -> None:
        self._ops[scene_id].append(op)
        with open(self.scene_dir(scene_id) / "ops.jsonl", "a") as f:
            f.write(op.model_dump_json() + "\n")

    def _rewrite_log(self, scene_id: str) -> None:
        (self.scene_dir(scene_id) / "ops.jsonl").write_text(
            "".join(o.model_dump_json() + "\n" for o in self._ops[scene_id])
        )

    def ops(self, scene_id: str, variant: str | None = None) -> list[Op]:
        self.load(scene_id)
        return [o for o in self._ops[scene_id] if variant is None or o.variant == variant]

    # ---- events ------------------------------------------------------------
    def subscribe(self, fn: Callable[[str, dict], None]) -> Callable[[], None]:
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def _emit(self, scene_id: str, event: dict) -> None:
        for fn in list(self._listeners):
            try:
                fn(scene_id, event)
            except Exception:  # a broken listener must not break edits
                pass

    # ---- permission rules ----------------------------------------------------
    @staticmethod
    def _guard(actor: Actor, variant: str) -> None:
        if actor == "agent" and variant == MAIN:
            raise PermissionError_("the AI writes only to variants; call create_variant first")

    # ---- low-level state access ------------------------------------------------
    def _get_state(self, scene: Scene, variant: str, object_id: str) -> dict | None:
        """Snapshot of an object's per-variant state (None = not present)."""
        if variant == MAIN:
            for o in scene.objects:
                if o.id == object_id:
                    return {"object": o.model_dump()}
            return None
        v = scene.variant(variant)
        base = next((o for o in scene.objects if o.id == object_id), None)
        added = next((o for o in v.added if o.id == object_id), None)
        if base is None and added is None:
            return None
        return {
            "added": added.model_dump() if added else None,
            "pose": v.poses[object_id].model_dump() if object_id in v.poses else None,
            "support": v.supports.get(object_id),
            "hidden": object_id in v.hidden,
        }

    def _set_state(self, scene: Scene, variant: str, object_id: str, state: dict | None) -> None:
        if variant == MAIN:
            idx = next((i for i, o in enumerate(scene.objects) if o.id == object_id), None)
            if state is None:
                if idx is not None:
                    scene.objects.pop(idx)
                return
            obj = SceneObject.model_validate(state["object"])
            if idx is None:
                scene.objects.append(obj)
            else:
                scene.objects[idx] = obj
            return
        v = scene.variant(variant)
        v.added = [o for o in v.added if o.id != object_id]
        v.poses.pop(object_id, None)
        v.supports.pop(object_id, None)
        v.hidden = [h for h in v.hidden if h != object_id]
        if state is None:
            return
        if state.get("added"):
            v.added.append(SceneObject.model_validate(state["added"]))
        if state.get("pose"):
            v.poses[object_id] = Pose.model_validate(state["pose"])
        if state.get("support"):
            v.supports[object_id] = state["support"]
        if state.get("hidden"):
            v.hidden.append(object_id)

    def _commit(self, scene_id: str, op: Op, extra: dict | None = None) -> dict:
        self._log(scene_id, op)
        self._save(scene_id)
        evt = {"type": "op", "op": op.model_dump(), **(extra or {})}
        self._emit(scene_id, evt)
        return evt

    # ---- edits ---------------------------------------------------------------
    def move(self, scene_id: str, variant: str, object_id: str, *, x: float | None = None, y: float | None = None,
             yaw: float | None = None, z: float | None = None, support: str | None = None, snap: bool = True,
             roll: float | None = None, pitch: float | None = None, actor: Actor = "user") -> dict:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            obj = scene.find(variant, object_id)
            if obj.locked and actor == "agent":
                raise PermissionError_(f"{obj.name} ({obj.id}) is locked; the AI may not move it")
            before_checks = checks.check_layout(scene, variant)
            before = self._get_state(scene, variant, object_id)
            if x is not None:
                obj.pose.x = x
            if y is not None:
                obj.pose.y = y
            if yaw is not None:
                obj.pose.yaw = yaw
            if z is not None:
                obj.pose.z = z
            if support is not None:
                obj.support = support
            if roll is not None or pitch is not None:
                new_pose = Pose.model_validate({**obj.pose.model_dump(),
                                                **({"roll": roll} if roll is not None else {}),
                                                **({"pitch": pitch} if pitch is not None else {})})
                if new_pose.tipped and obj.upright_required:
                    hint = "" if actor == "agent" else "; untick 'Upright only' if it can stand another way"
                    raise ValueError(f"{obj.name} ({obj.id}) only stands upright{hint}")
                obj.pose.roll, obj.pose.pitch = new_pose.roll, new_pose.pitch
            objs = {o.id: o for o in scene.resolve(variant)}
            objs[object_id] = obj
            notes = placement.snap(obj, scene.room, objs, scene.rules.snap_yaw_step, scene.rules.snap_wall_distance) if snap \
                else (placement.apply_support(obj, objs) or [])
            if variant == MAIN:
                if obj.source == "scanned":
                    obj.moved_from_scan = True
                self._set_state(scene, MAIN, object_id, {"object": obj.model_dump()})
            else:
                st = before or {}
                st = dict(st)
                st["pose"] = obj.pose.model_dump()
                st["support"] = obj.support if obj.support != self._base_support(scene, object_id) else None
                if st.get("added"):
                    added = SceneObject.model_validate(st["added"])
                    added.pose, added.support = obj.pose, obj.support
                    st["added"], st["pose"], st["support"] = added.model_dump(), None, None
                self._set_state(scene, variant, object_id, st)
            deps = [o.id for o in scene.resolve(variant) if o.support == object_id]
            dep_before = [{"id": d, "state": self._get_state(scene, variant, d)} for d in deps]
            self._move_dependents(scene, variant, object_id, before, actor)
            after_checks = checks.check_layout(scene, variant)
            ids = [object_id, *deps]
            op = Op(actor=actor, variant=variant, kind="move", object_id=object_id,
                    before={"objects": [{"id": object_id, "state": before}, *dep_before]},
                    after={"objects": [{"id": i, "state": self._get_state(scene, variant, i)} for i in ids]},
                    note="; ".join(notes))
            caused = checks.caused_by(before_checks, after_checks, object_id)
            self._commit(scene_id, op)
            return {"op_id": op.id, "pose": obj.pose.model_dump(), "support": obj.support, "notes": notes,
                    "violations_caused": [v.model_dump() for v in caused]}

    def _base_support(self, scene: Scene, object_id: str) -> str | None:
        o = next((o for o in scene.objects if o.id == object_id), None)
        return o.support if o else None

    def _move_dependents(self, scene: Scene, variant: str, object_id: str, before: dict | None, actor: Actor) -> None:

        """Carry objects resting on a moved object along with it (lamp on nightstand)."""
        if before is None:
            return
        objs = {o.id: o for o in scene.resolve(variant)}
        new = objs.get(object_id)
        old_pose = None
        if variant == MAIN:
            old_pose = Pose.model_validate(before["object"]["pose"])
        else:
            if before.get("added"):
                old_pose = Pose.model_validate(before["added"]["pose"])
            elif before.get("pose"):
                old_pose = Pose.model_validate(before["pose"])
            else:
                base = next((o for o in scene.objects if o.id == object_id), None)
                old_pose = base.pose if base else None
        if new is None or old_pose is None:
            return
        dyaw = new.pose.yaw - old_pose.yaw
        c, s = math.cos(math.radians(dyaw)), math.sin(math.radians(dyaw))
        for dep in objs.values():
            if dep.support != object_id:
                continue
            rx, ry = dep.pose.x - old_pose.x, dep.pose.y - old_pose.y
            dep.pose.x = new.pose.x + c * rx - s * ry
            dep.pose.y = new.pose.y + s * rx + c * ry
            dep.pose.yaw += dyaw
            dep.pose.z = new.pose.z + new.extent().h
            st = self._get_state(scene, variant, dep.id)
            if variant == MAIN:
                self._set_state(scene, MAIN, dep.id, {"object": dep.model_dump()})
            elif st and st.get("added"):
                st["added"] = dep.model_dump()
                self._set_state(scene, variant, dep.id, st)
            else:
                st = dict(st or {})
                st["pose"] = dep.pose.model_dump()
                self._set_state(scene, variant, dep.id, st)

    def add_object(self, scene_id: str, variant: str, obj: SceneObject, *, snap: bool = True, actor: Actor = "user") -> dict:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            if obj.id in scene.all_ids():
                obj.id = self.new_object_id(scene, obj.category)
            objs = {o.id: o for o in scene.resolve(variant)}
            notes = placement.snap(obj, scene.room, objs, scene.rules.snap_yaw_step, scene.rules.snap_wall_distance) if snap else []
            placement.apply_support(obj, objs)
            before_checks = checks.check_layout(scene, variant)
            state = {"object": obj.model_dump()} if variant == MAIN else {"added": obj.model_dump()}
            self._set_state(scene, variant, obj.id, state)
            op = Op(actor=actor, variant=variant, kind="add", object_id=obj.id, before=None,
                    after=self._get_state(scene, variant, obj.id), note="; ".join(notes))
            caused = checks.caused_by(before_checks, checks.check_layout(scene, variant), obj.id)
            self._commit(scene_id, op)
            return {"op_id": op.id, "object_id": obj.id, "pose": obj.pose.model_dump(), "notes": notes,
                    "violations_caused": [v.model_dump() for v in caused]}

    def set_hidden(self, scene_id: str, variant: str, object_id: str, hidden: bool, actor: Actor = "user") -> dict:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            if variant == MAIN:
                raise PermissionError_("hide/show works on variants; in the main layout, delete instead")
            obj = scene.find(variant, object_id)
            if obj.locked and actor == "agent" and hidden:
                raise PermissionError_(f"{obj.name} ({obj.id}) is locked; the AI may not hide it")
            before = self._get_state(scene, variant, object_id)
            st = dict(before or {})
            st["hidden"] = hidden
            self._set_state(scene, variant, object_id, st)
            op = Op(actor=actor, variant=variant, kind="hide" if hidden else "show", object_id=object_id,
                    before=before, after=self._get_state(scene, variant, object_id))
            self._commit(scene_id, op)
            return {"op_id": op.id, "object_id": object_id, "hidden": hidden}

    def delete_object(self, scene_id: str, variant: str, object_id: str, actor: Actor = "user") -> dict:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            obj = scene.find(variant, object_id)
            if obj.locked and actor == "agent":
                raise PermissionError_(f"{obj.name} ({obj.id}) is locked")
            before = self._get_state(scene, variant, object_id)
            if variant != MAIN and not (before or {}).get("added"):
                return self.set_hidden(scene_id, variant, object_id, True, actor)
            self._set_state(scene, variant, object_id, None)
            op = Op(actor=actor, variant=variant, kind="delete", object_id=object_id, before=before, after=None)
            self._commit(scene_id, op)
            return {"op_id": op.id, "object_id": object_id, "deleted": True}

    def update_object(self, scene_id: str, variant: str, object_id: str, fields: dict, actor: Actor = "user") -> dict:
        """Change non-pose fields (name, locked, dims...) of an object in main or an added object."""
        allowed = {"name", "locked", "dims", "category", "color", "warnings", "product", "asset", "asset_kind", "upright_only"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update fields {sorted(bad)}")
        with self._lock:
            self._guard(actor, variant)
            if actor == "agent" and {"locked", "upright_only"} & set(fields):
                raise PermissionError_("only you can lock objects or change whether they may be tipped")
            scene = self.load(scene_id)
            before = self._get_state(scene, variant, object_id)
            if before is None:
                raise KeyError(object_id)
            key = "object" if variant == MAIN else "added"
            if not before.get(key):
                raise PermissionError_("only objects added in this variant can be edited here; edit the main layout instead")
            obj = SceneObject.model_validate({**before[key], **fields})
            st = dict(before)
            st[key] = obj.model_dump()
            self._set_state(scene, variant, object_id, st)
            op = Op(actor=actor, variant=variant, kind="update", object_id=object_id, before=before,
                    after=self._get_state(scene, variant, object_id))
            self._commit(scene_id, op)
            return {"op_id": op.id, "object": obj.model_dump()}

    # ---- spaces (rooms) ---------------------------------------------------------
    def _support_refs(self, scene: Scene) -> list[dict]:
        """Every wall support in main and in variants, with where its object stands."""
        refs = []
        for o in scene.objects:
            if o.support.startswith("wall:"):
                refs.append({"where": "main", "id": o.id, "support": o.support, "xy": (o.pose.x, o.pose.y)})
        for v in scene.variants:
            for o in v.added:
                if o.support.startswith("wall:"):
                    refs.append({"where": f"added:{v.id}", "id": o.id, "support": o.support, "xy": (o.pose.x, o.pose.y)})
            for oid, sup in v.supports.items():
                if sup.startswith("wall:"):
                    pose = v.poses.get(oid) or next((o.pose for o in scene.objects if o.id == oid), None)
                    xy = (pose.x, pose.y) if pose else (0.0, 0.0)
                    refs.append({"where": f"supports:{v.id}", "id": oid, "support": sup, "xy": xy})
        return refs

    def _restore_support(self, scene: Scene, item: dict) -> None:
        where, oid, sup = item["where"], item["id"], item["support"]
        if where == "main":
            for o in scene.objects:
                if o.id == oid:
                    o.support = sup
        elif where.startswith("added:"):
            for o in scene.variant(where.split(":", 1)[1]).added:
                if o.id == oid:
                    o.support = sup
        else:
            scene.variant(where.split(":", 1)[1]).supports[oid] = sup

    def edit_room(self, scene_id: str, fn: Callable[[Room], Room], actor: Actor = "user", note: str = "") -> dict:
        """Apply a spaces edit (see scene/spaces.py) as one undoable op. Only you can do this."""
        if actor == "agent":
            raise PermissionError_("only you can change rooms, spaces and walls")
        with self._lock:
            scene = self.load(scene_id)
            old = scene.room
            new, mapping = spaces.rewall(old, fn(old.model_copy(deep=True)))
            new = Room.model_validate(new.model_dump())
            refs = self._support_refs(scene)
            before_sup, after_sup = [], []
            for r in refs:
                wid = r["support"].removeprefix("wall:")
                nid = mapping.get(wid) or spaces.wall_for_point(new, r["xy"])
                target = f"wall:{nid}" if nid else "floor"
                if target != r["support"]:
                    before_sup.append({k: r[k] for k in ("where", "id", "support")})
                    after_sup.append({"where": r["where"], "id": r["id"], "support": target})
            scene.room = new
            for item in after_sup:
                self._restore_support(scene, item)
            op = Op(actor=actor, kind="room", before={"room": old.model_dump(), "supports": before_sup},
                    after={"room": new.model_dump(), "supports": after_sup}, note=note)
            self._commit(scene_id, op, {"room": new.model_dump()})
            return {"op_id": op.id, "room": new.model_dump(), "violations": [v.model_dump() for v in checks.check_layout(scene)]}

    # ---- variants ------------------------------------------------------------
    def create_variant(self, scene_id: str, name: str, base: str = MAIN, actor: Actor = "user", notes: str = "") -> Variant:
        with self._lock:
            scene = self.load(scene_id)
            vid = f"v{uuid.uuid4().hex[:6]}"
            if base == MAIN:
                v = Variant(id=vid, name=name, created_by="agent" if actor == "agent" else "user", notes=notes)
            else:
                src = scene.variant(base)
                v = src.model_copy(deep=True)
                v.id, v.name, v.created_by, v.created_at, v.notes = vid, name, ("agent" if actor == "agent" else "user"), now_iso(), notes
            scene.variants.append(v)
            op = Op(actor=actor, variant=vid, kind="create_variant", after=v.model_dump(), note=f"from {base}")
            self._commit(scene_id, op)
            return v

    def delete_variant(self, scene_id: str, variant: str, actor: Actor = "user") -> None:
        with self._lock:
            scene = self.load(scene_id)
            v = scene.variant(variant)
            if actor == "agent" and v.created_by != "agent":
                raise PermissionError_("the AI may only delete variants it created")
            scene.variants = [x for x in scene.variants if x.id != variant]
            self._commit(scene_id, Op(actor=actor, variant=variant, kind="delete_variant", before=v.model_dump()))

    # ---- library -------------------------------------------------------------
    def add_library_asset(self, scene_id: str, asset: LibraryAsset, actor: Actor = "pipeline") -> LibraryAsset:
        with self._lock:
            scene = self.load(scene_id)
            scene.library = [a for a in scene.library if a.id != asset.id] + [asset]
            self._commit(scene_id, Op(actor=actor, kind="library_add", object_id=asset.id, after=asset.model_dump()))
            return asset

    def place_library_asset(self, scene_id: str, variant: str, asset_id: str, pose: Pose, support: str = "floor",
                            actor: Actor = "user", snap: bool = True) -> dict:
        scene = self.load(scene_id)
        a = scene.library_asset(asset_id)
        obj = SceneObject(
            id=self.new_object_id(scene, a.category), name=a.name, category=a.category, asset=a.asset,
            asset_kind=a.asset_kind, asset_transform=a.asset_transform, pose=pose, dims=a.dims, upright_only=a.upright_only, support=support, source=a.source, product=a.product,
            anisotropy=a.anisotropy, warnings=list(a.warnings),
        )
        return self.add_object(scene_id, variant, obj, snap=snap, actor=actor)

    @staticmethod
    def new_object_id(scene: Scene, category: str = "obj") -> str:
        ids = scene.all_ids()
        prefix = "".join(c for c in category.lower() if c.isalnum())[:10] or "obj"
        k = 1
        while f"{prefix}_{k}" in ids:
            k += 1
        return f"{prefix}_{k}"

    # ---- undo / redo -----------------------------------------------------------
    _UNDOABLE = {"move", "add", "hide", "show", "delete", "update", "promote", "room"}

    def undo(self, scene_id: str, variant: str = MAIN, actor: Actor = "user") -> Op | None:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            for op in reversed(self._ops[scene_id]):
                if op.variant == variant and op.kind in self._UNDOABLE and not op.undone:
                    self._apply_state(scene, op, use="before")
                    op.undone = True
                    self._rewrite_log(scene_id)
                    self._save(scene_id)
                    self._emit(scene_id, {"type": "undo", "op": op.model_dump()})
                    return op
            return None

    def redo(self, scene_id: str, variant: str = MAIN, actor: Actor = "user") -> Op | None:
        with self._lock:
            self._guard(actor, variant)
            scene = self.load(scene_id)
            ops = [o for o in self._ops[scene_id] if o.variant == variant and o.kind in self._UNDOABLE]
            # redo the earliest undone op after the last live one
            target = None
            for op in reversed(ops):
                if op.undone:
                    target = op
                else:
                    break
            if target is None:
                return None
            self._apply_state(scene, target, use="after")
            target.undone = False
            self._rewrite_log(scene_id)
            self._save(scene_id)
            self._emit(scene_id, {"type": "redo", "op": target.model_dump()})
            return target

    def _apply_state(self, scene: Scene, op: Op, use: Literal["before", "after"]) -> None:
        data = op.before if use == "before" else op.after
        if op.kind == "room":  # whole-home edit: room plus any wall supports it re-homed
            scene.room = Room.model_validate(data["room"])
            for item in data.get("supports", []):
                self._restore_support(scene, item)
            return
        if isinstance(data, dict) and "objects" in data:  # multi-object op (move with dependents, promote)
            for item in data["objects"]:
                self._set_state(scene, op.variant, item["id"], item["state"])
            return
        self._set_state(scene, op.variant, op.object_id, data)

    # ---- diff and promote ---------------------------------------------------------
    def diff(self, scene_id: str, a: str, b: str) -> Diff:
        scene = self.load(scene_id)
        A = {o.id: o for o in scene.resolve(a)}
        B = {o.id: o for o in scene.resolve(b)}
        all_known = {o.id: o for o in scene.resolve(b, include_hidden=True)} | {o.id: o for o in scene.resolve(a, include_hidden=True)}
        changes: list[Change] = []
        currency = None
        delta = 0.0
        for oid, ob in B.items():
            oa = A.get(oid)
            if oa is None:
                in_a_hidden = oid in all_known and a != MAIN and oid in {o.id for o in scene.resolve(a, include_hidden=True)}
                price = ob.product.price if ob.product and ob.product.price is not None and not in_a_hidden else None
                if price is not None:
                    delta += price
                    currency = currency or ob.product.currency
                changes.append(Change(kind="restore" if in_a_hidden else "add", object_id=oid, name=ob.name,
                                      after=ob.pose, cost=price, currency=ob.product.currency if ob.product else None))
                continue
            dist = math.hypot(ob.pose.x - oa.pose.x, ob.pose.y - oa.pose.y) + abs(ob.pose.z - oa.pose.z)
            rot = abs((ob.pose.yaw - oa.pose.yaw + 180) % 360 - 180)
            tipped = ob.pose.roll != oa.pose.roll or ob.pose.pitch != oa.pose.pitch
            if dist > 1e-3 or rot > 0.1 or tipped:
                changes.append(Change(kind="move", object_id=oid, name=ob.name, before=oa.pose, after=ob.pose,
                                      distance_m=round(dist, 3), rotation_deg=round(rot, 1)))
        for oid, oa in A.items():
            if oid not in B:
                price = None
                if oa.product and oa.product.price is not None and oa.source == "product":
                    price = -oa.product.price
                    delta += price
                    currency = currency or oa.product.currency
                changes.append(Change(kind="remove", object_id=oid, name=oa.name, before=oa.pose, cost=price))
        return Diff(a=a, b=b, changes=changes, cost_delta=round(delta, 2), currency=currency)

    def promote(self, scene_id: str, variant: str, object_ids: list[str] | None = None, actor: Actor = "user") -> dict:
        """Copy chosen changes from a variant into the main layout (user only)."""
        if actor != "user":
            raise PermissionError_("only you can promote changes into the main layout")
        with self._lock:
            scene = self.load(scene_id)
            d = self.diff(scene_id, MAIN, variant)
            chosen = [c for c in d.changes if object_ids is None or c.object_id in object_ids]
            before, after = [], []
            resolved = {o.id: o for o in scene.resolve(variant, include_hidden=True)}
            for c in chosen:
                prev = self._get_state(scene, MAIN, c.object_id)
                if c.kind in ("move", "add", "restore"):
                    obj = resolved[c.object_id].model_copy(deep=True)
                    if obj.source == "scanned" and c.kind == "move":
                        obj.moved_from_scan = True
                    new = {"object": obj.model_dump()}
                else:
                    new = None
                before.append({"id": c.object_id, "state": prev})
                after.append({"id": c.object_id, "state": new})
                self._set_state(scene, MAIN, c.object_id, new)
            op = Op(actor=actor, variant=MAIN, kind="promote", before={"objects": before}, after={"objects": after},
                    note=f"from {variant}: " + ", ".join(f"{c.kind} {c.object_id}" for c in chosen))
            self._commit(scene_id, op)
            return {"op_id": op.id, "promoted": [c.model_dump() for c in chosen]}
