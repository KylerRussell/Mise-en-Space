"""Layout checks. Violations are numeric and specific so an agent can fix them.

Hard violations: collisions, objects outside the room, blocked doors,
doors cut off from each other by a walkway narrower than ``walkway_min``.
Soft violations: tall objects in front of windows, objects with no
``access_min`` path from a door, unsupported objects.
The editor shows both as warnings; nothing is blocked.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
from pydantic import BaseModel

from . import geometry as g
from .models import MAIN, Opening, Room, Scene, SceneObject

# Categories that need someone to be able to walk up to them.
NEEDS_ACCESS = {"bed", "sofa", "desk", "chair", "armchair", "wardrobe", "dresser", "table", "dining_table", "bookshelf", "cabinet"}
# Things that may sit on/under others or be ignored by the walkway grid.
FLOOR_IGNORED = {"rug", "mat"}


class Violation(BaseModel):
    kind: Literal[
        "collision",
        "out_of_room",
        "crosses_wall",
        "blocked_door",
        "blocked_window",
        "walkway",
        "no_access",
        "unsupported",
    ]
    severity: Literal["hard", "soft"]
    objects: list[str]
    message: str
    amount_m: float | None = None

    def key(self) -> tuple:
        return (self.kind, tuple(sorted(self.objects)))


def _name(o: SceneObject) -> str:
    return f"{o.name} ({o.id})"


def _on_floor(o: SceneObject) -> bool:
    return o.support == "floor" and o.category not in FLOOR_IGNORED


def _supports(a: SceneObject, b: SceneObject, by_id: dict[str, SceneObject]) -> bool:
    """True if a rests (directly or transitively) on b."""
    seen = set()
    cur = a
    while cur.support not in ("floor",) and not cur.support.startswith("wall:"):
        if cur.support in seen:
            return False
        seen.add(cur.support)
        if cur.support == b.id:
            return True
        cur = by_id.get(cur.support)
        if cur is None:
            return False
    return False


def check_objects(room: Room, objects: list[SceneObject], rules, only: set[str] | None = None) -> list[Violation]:
    out: list[Violation] = []
    by_id = {o.id: o for o in objects}
    fps = {o.id: g.object_footprint(o) for o in objects}

    # Collisions: footprint overlap plus vertical overlap.
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if only and a.id not in only and b.id not in only:
                continue
            if a.category in FLOOR_IGNORED or b.category in FLOOR_IGNORED:
                continue
            if _supports(a, b, by_id) or _supports(b, a, by_id):
                continue
            za, zb = g.z_interval(a), g.z_interval(b)
            zover = min(za[1], zb[1]) - max(za[0], zb[0])
            if zover <= rules.collision_tolerance:
                continue
            depth = g.penetration_depth(fps[a.id], fps[b.id])
            if depth > rules.collision_tolerance:
                out.append(Violation(
                    kind="collision", severity="hard", objects=[a.id, b.id], amount_m=round(depth, 3),
                    message=f"{_name(a)} overlaps {_name(b)} by {depth:.2f} m",
                ))

    # Inside the room (or the union of spaces), not through interior walls.
    polys = room.polygons()
    multi = len(polys) > 1
    for o in objects:
        if only and o.id not in only:
            continue
        amt = g.outside_amount_multi(fps[o.id], polys)
        if amt > rules.collision_tolerance:
            out.append(Violation(
                kind="out_of_room", severity="hard", objects=[o.id], amount_m=round(amt, 3),
                message=f"{_name(o)} extends {amt:.2f} m outside the {'floor' if multi else 'room'}",
            ))
        elif multi and not o.support.startswith("wall:"):
            worst = max(((g.wall_crossing(fps[o.id], w), w) for w in room.walls), key=lambda t: t[0])
            if worst[0] > rules.collision_tolerance:
                out.append(Violation(
                    kind="crosses_wall", severity="hard", objects=[o.id], amount_m=round(worst[0], 3),
                    message=f"{_name(o)} cuts {worst[0]:.2f} m through wall {worst[1].id}",
                ))
        sp = room.space_at(o.pose.x, o.pose.y)
        ceiling = room.ceiling_for(sp)
        if o.pose.z + o.extent().h > ceiling + rules.collision_tolerance:
            amt = o.pose.z + o.extent().h - ceiling
            out.append(Violation(
                kind="out_of_room", severity="hard", objects=[o.id], amount_m=round(amt, 3),
                message=f"{_name(o)} is {amt:.2f} m taller than the ceiling allows",
            ))
        if o.support not in ("floor",) and not o.support.startswith("wall:") and o.support not in by_id:
            out.append(Violation(
                kind="unsupported", severity="soft", objects=[o.id],
                message=f"{_name(o)} rests on {o.support!r}, which is not in this layout",
            ))

    # Doors and windows.
    for op in room.openings:
        if op.kind == "door":
            depth = rules.door_clearance
            if op.swing == "in":
                depth = max(depth, op.width)
            zone = g.opening_zone(op, room, depth)
            for o in objects:
                if (only and o.id not in only) or not _on_floor(o):
                    continue
                pen = g.penetration_depth(zone, fps[o.id])
                if pen > rules.collision_tolerance:
                    out.append(Violation(
                        kind="blocked_door", severity="hard", objects=[o.id], amount_m=round(pen, 3),
                        message=f"{_name(o)} intrudes {pen:.2f} m into the swing/clearance zone of door {op.id}",
                    ))
        else:
            zone = g.opening_zone(op, room, rules.window_zone_depth)
            for o in objects:
                if only and o.id not in only:
                    continue
                top = o.pose.z + o.extent().h
                if top < max(rules.tall_object_height, op.bottom + 0.1):
                    continue
                pen = g.penetration_depth(zone, fps[o.id])
                if pen > rules.collision_tolerance:
                    out.append(Violation(
                        kind="blocked_window", severity="soft", objects=[o.id], amount_m=round(top - op.bottom, 3),
                        message=f"{_name(o)} ({top:.2f} m tall) stands in front of window {op.id}, {top - op.bottom:.2f} m above its sill",
                    ))

    out.extend(_walkway_checks(room, objects, rules, fps, only))
    return out


def _walkway_checks(room: Room, objects, rules, fps, only) -> list[Violation]:
    """Clearance on an occupancy grid.

    A cell is free for walking if it is at least walkway_min/2 from every
    floor obstacle and wall. Doors must be connected through such cells.
    Objects that need access must touch a cell that is free at access_min/2
    and connected to a door.
    """
    out: list[Violation] = []
    doors = [op for op in room.openings if op.kind == "door"]
    if not doors:
        return out
    polys = [np.asarray(pl, dtype=float) for pl in room.polygons()]
    step = rules.grid
    (lox, loy), (hix, hiy) = room.bounds()
    xs = np.arange(lox + step / 2, hix, step)
    ys = np.arange(loy + step / 2, hiy, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    inside = np.zeros(len(pts), dtype=bool)
    for pl in polys:
        inside |= g.points_in_polygon(pts, pl)

    # distance to walls (open boundaries between spaces are not walls)
    dist = np.full(len(pts), np.inf)
    for w in room.walls:
        dist = np.minimum(dist, g.points_segment_distance(pts, np.array(w.a, float), np.array(w.b, float)))
    # distance to obstacles (floor objects that are low enough to block walking)
    nearest = np.full(len(pts), -1)
    obj_dist = np.full(len(pts), np.inf)
    floor_objs = [o for o in objects if _on_floor(o)]
    for k, o in enumerate(floor_objs):
        d = g.points_polygon_distance(pts, fps[o.id])
        closer = d < dist
        nearest[closer] = k
        dist = np.minimum(dist, d)
        obj_dist = np.minimum(obj_dist, d)
    dist[~inside] = -1
    shape = gx.shape

    def components(threshold: float) -> np.ndarray:
        free = (dist >= threshold).reshape(shape)
        lab = np.full(shape, -1, dtype=int)
        cur = 0
        for start in zip(*np.nonzero(free)):
            if lab[start] >= 0:
                continue
            q = deque([start])
            lab[start] = cur
            while q:
                i, j = q.popleft()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if 0 <= a < shape[0] and 0 <= b < shape[1] and free[a, b] and lab[a, b] < 0:
                        lab[a, b] = cur
                        q.append((a, b))
            cur += 1
        return lab

    def cell_of(p: np.ndarray) -> tuple[int, int]:
        i = int(np.clip(round((p[0] - xs[0]) / step), 0, shape[0] - 1))
        j = int(np.clip(round((p[1] - ys[0]) / step), 0, shape[1] - 1))
        return i, j

    def door_label(op: Opening, lab: np.ndarray, r: float, side: int = 1, strict: bool = False) -> int:
        """Walkway component next to a door: side 1 is the wall's own space, -1 the far side.

        strict: walk out from the doorway and give up at the first cell that furniture
        crowds (so a dresser parked across a door does not count as a way through).
        """
        w = room.wall(op.wall)
        n = g.wall_inward_normal(w, room) * side
        c = g.wall_point(w, op.center)
        start = step if strict else r  # the far side may be another space behind a thick wall
        for t in np.arange(start, r + (0.6 if side == 1 else 0.9), step):
            if strict:
                k = np.ravel_multi_index(cell_of(c + n * t), shape)
                if inside[k] and obj_dist[k] < r:
                    return -1
            for s in np.linspace(-op.width / 2 + 0.05, op.width / 2 - 0.05, 5):
                p = c + n * t + g.wall_direction(w) * s
                ij = cell_of(p)
                if lab[ij] >= 0:
                    return int(lab[ij])
        return -1

    def merged(lab: np.ndarray, r: float) -> np.ndarray:
        """Join components on the two sides of each door (doors link rooms)."""
        n = int(lab.max()) + 1
        parent = list(range(max(n, 1)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for op in doors:
            a, b = door_label(op, lab, r, 1, strict=True), door_label(op, lab, r, -1, strict=True)
            if a >= 0 and b >= 0:
                parent[find(a)] = find(b)
        out = lab.copy()
        ok = lab >= 0
        out[ok] = [find(int(i)) for i in lab[ok]]
        return out

    def bottleneck(op_cells: list[tuple[int, int]]) -> str:
        cand = (dist > 0) & (dist < rules.walkway_min / 2)
        if not cand.any() or not floor_objs:
            return ""
        ks = nearest[cand]
        ks = ks[ks >= 0]
        if len(ks) == 0:
            return ""
        counts = np.bincount(ks, minlength=len(floor_objs))
        o = floor_objs[int(counts.argmax())]
        return f"; narrowest stretch is around {_name(o)}"

    # Door-to-door walkway.
    lab_walk = merged(components(rules.walkway_min / 2), rules.walkway_min / 2)
    labels = {op.id: door_label(op, lab_walk, rules.walkway_min / 2) for op in doors}
    for op in doors:
        if labels[op.id] < 0:
            out.append(Violation(
                kind="walkway", severity="hard", objects=[], amount_m=rules.walkway_min,
                message=f"no {rules.walkway_min:.2f} m wide approach to door {op.id}{bottleneck([])}",
            ))
    good = [op for op in doors if labels[op.id] >= 0]
    for a, b in zip(good, good[1:]):
        if labels[a.id] != labels[b.id]:
            out.append(Violation(
                kind="walkway", severity="hard", objects=[], amount_m=rules.walkway_min,
                message=f"doors {a.id} and {b.id} are not joined by a {rules.walkway_min:.2f} m walkway{bottleneck([])}",
            ))

    # Access to furniture from a door.
    lab_acc = merged(components(rules.access_min / 2), rules.access_min / 2)
    door_labels = {door_label(op, lab_acc, rules.access_min / 2) for op in doors} - {-1}
    for o in floor_objs:
        if (only and o.id not in only) or o.category not in NEEDS_ACCESS:
            continue
        ring = g.object_footprint(o, pad=rules.access_min / 2 + step)
        near = g.points_polygon_distance(pts, ring) == 0
        lbls = set(lab_acc.ravel()[near & (lab_acc.ravel() >= 0)].tolist())
        if not (lbls & door_labels):
            out.append(Violation(
                kind="no_access", severity="soft", objects=[o.id], amount_m=rules.access_min,
                message=f"{_name(o)} has no {rules.access_min:.2f} m path from a door",
            ))
    return out


def check_layout(scene: Scene, variant_id: str = MAIN, only: set[str] | None = None) -> list[Violation]:
    return check_objects(scene.room, scene.resolve(variant_id), scene.rules, only)


def caused_by(before: list[Violation], after: list[Violation], object_id: str) -> list[Violation]:
    """Violations involving `object_id` that are new or worse after a change."""
    prev = {v.key(): v for v in before}
    out = []
    for v in after:
        involved = object_id in v.objects or (v.kind == "walkway")
        if not involved:
            continue
        p = prev.get(v.key())
        if p is None or (v.amount_m or 0) > (p.amount_m or 0) + 1e-3:
            out.append(v)
    return out
