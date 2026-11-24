"""Layout solver: the model proposes relations, the solver enforces physics.

Following Holodeck (LLM-written relational constraints satisfied by an
optimizer) and LayoutVLM (VLM-proposed poses optimized jointly with physics
objectives). Hard terms: overlaps, staying inside the room, door swing
zones. Soft terms: the agent's constraints. Simulated annealing over
(x, y, yaw) of the movable objects, starting from their current poses.
The full layout check (including walkway connectivity) runs on the result.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from ..scene import checks
from ..scene import geometry as g
from ..scene.models import Pose, Room, Scene, SceneObject

HARD_W = 50.0


class Constraint(BaseModel):
    kind: Literal["against_wall", "near", "far_from", "distance", "facing", "in_front_of", "beside", "parallel",
                  "centered_on_wall", "away_from_window", "at"]
    obj: str
    target: str | None = Field(None, description="object id, or wall id ('wall:2' / '2') for wall constraints")
    min: float | None = None
    max: float | None = None
    at: tuple[float, float] | None = None
    weight: float = 1.0


class ConstraintReport(BaseModel):
    constraint: Constraint
    cost: float
    satisfied: bool


class SolveResult(BaseModel):
    poses: dict[str, Pose]
    energy: float
    hard_energy: float
    constraints: list[ConstraintReport]
    violations: list[checks.Violation]
    iterations: int


def _wall(room: Room, target: str | None, pt: np.ndarray):
    if target:
        return room.wall(target)
    return g.nearest_wall(pt, room)[0]


def constraint_cost(c: Constraint, objs: dict[str, SceneObject], room: Room) -> float:
    o = objs.get(c.obj)
    if o is None:
        return 0.0
    center = np.array([o.pose.x, o.pose.y])
    right, front = g.local_axes(o.pose.yaw)
    fp = g.object_footprint(o)
    t = objs.get(c.target or "") if c.target and not c.target.startswith("wall") else None

    if c.kind in ("against_wall", "centered_on_wall"):
        back = center - front * o.dims.d / 2
        w = _wall(room, c.target, back)
        n = g.wall_inward_normal(w, room)
        dist = g.point_segment_distance(back, np.array(w.a), np.array(w.b))
        ang = math.acos(float(np.clip(front @ n, -1, 1)))
        cost = dist + ang / (math.pi / 4) * 0.5
        if c.kind == "centered_on_wall":
            along = float((center - np.array(w.a)) @ g.wall_direction(w))
            cost += abs(along - w.length / 2)
        return cost
    if c.kind == "at":
        return float(np.linalg.norm(center - np.array(c.at))) if c.at else 0.0
    if c.kind == "away_from_window":
        worst = 0.0
        for op in room.openings:
            if op.kind == "window":
                depth = c.min if c.min is not None else 0.8
                worst = max(worst, g.penetration_depth(g.opening_zone(op, room, depth), fp))
        return worst
    if t is None:
        return 0.0
    tc = np.array([t.pose.x, t.pose.y])
    tfp = g.object_footprint(t)
    gap = g.polygon_distance(fp, tfp)
    if c.kind == "near":
        return max(0.0, gap - (c.max if c.max is not None else 0.5))
    if c.kind == "far_from":
        return max(0.0, (c.min if c.min is not None else 1.0) - gap)
    if c.kind == "distance":
        lo, hi = c.min or 0.0, c.max if c.max is not None else math.inf
        return max(0.0, lo - gap) + max(0.0, gap - hi)
    if c.kind == "facing":
        v = tc - center
        if np.linalg.norm(v) < 1e-6:
            return 0.0
        v /= np.linalg.norm(v)
        return math.acos(float(np.clip(front @ v, -1, 1))) / math.pi
    if c.kind == "parallel":
        d = abs(g.wrap_deg(o.pose.yaw - t.pose.yaw)) % 180
        return min(d, 180 - d) / 90
    if c.kind == "in_front_of":
        t_right, t_front = g.local_axes(t.pose.yaw)
        rel = center - tc
        ahead = float(rel @ t_front) - t.dims.d / 2 - o.dims.d / 2
        lateral = abs(float(rel @ t_right))
        lo, hi = c.min if c.min is not None else 0.3, c.max if c.max is not None else 1.5
        return max(0.0, lo - ahead) + max(0.0, ahead - hi) + 0.5 * max(0.0, lateral - t.dims.w / 2)
    if c.kind == "beside":
        t_right, t_front = g.local_axes(t.pose.yaw)
        rel = center - tc
        offset_front = abs(float(rel @ t_front) - (t.dims.d - o.dims.d) / 2)  # align backs
        side = abs(float(rel @ t_right)) - t.dims.w / 2 - o.dims.w / 2
        hi = c.max if c.max is not None else 0.1
        return max(0.0, side - hi) + max(0.0, -side) + 0.5 * offset_front
    return 0.0


def hard_energy(objs: dict[str, SceneObject], room: Room, rules, movable: set[str]) -> float:
    e = 0.0
    items = [o for o in objs.values() if o.category not in checks.FLOOR_IGNORED]
    fps = {o.id: g.object_footprint(o) for o in items}
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if a.id not in movable and b.id not in movable:
                continue
            if a.support == b.id or b.support == a.id:
                continue
            za, zb = g.z_interval(a), g.z_interval(b)
            if min(za[1], zb[1]) - max(za[0], zb[0]) <= rules.collision_tolerance:
                continue
            d = g.penetration_depth(fps[a.id], fps[b.id])
            if d > 0:
                e += d + 0.05
    for oid in movable:
        o = objs[oid]
        if o.category in checks.FLOOR_IGNORED:
            continue
        out = g.outside_amount(fps[oid], room.floor_polygon)
        if out > 0:
            e += out + 0.05
        if o.support == "floor":
            for op in room.openings:
                if op.kind != "door":
                    continue
                depth = max(rules.door_clearance, op.width if op.swing == "in" else 0)
                d = g.penetration_depth(g.opening_zone(op, room, depth), fps[oid])
                if d > 0:
                    e += d + 0.05
    return e


def solve_layout(scene: Scene, variant: str, constraints: list[Constraint], movable: list[str] | None = None,
                 iterations: int = 1500, seed: int = 0, repair_iterations: int = 120) -> SolveResult:
    rng = np.random.default_rng(seed)
    objs = {o.id: o for o in scene.resolve(variant)}
    for c in constraints:
        if c.obj not in objs:
            raise KeyError(f"constraint refers to unknown object {c.obj!r}")
    if movable is None:
        movable = sorted({c.obj for c in constraints})
    mov = [m for m in movable if m in objs and not objs[m].locked and objs[m].support == "floor"]
    if not mov:
        raise ValueError("no movable floor objects (locked or supported objects stay put)")
    room, rules = scene.room, scene.rules
    poly = np.asarray(room.floor_polygon, float)
    lo, hi = poly.min(0), poly.max(0)
    span = float(np.max(hi - lo))

    def energy() -> tuple[float, float]:
        h = hard_energy(objs, room, rules, set(mov))
        s = sum(c.weight * constraint_cost(c, objs, room) for c in constraints)
        return HARD_W * h + s, h

    def wall_proposal(o: SceneObject) -> tuple[float, float, float]:
        w = room.walls[rng.integers(len(room.walls))]
        n = g.wall_inward_normal(w, room)
        half = o.dims.w / 2
        along = rng.uniform(half, max(half, w.length - half))
        p = g.wall_point(w, along) + n * o.dims.d / 2
        return float(p[0]), float(p[1]), g.yaw_facing(n)

    E, H = energy()
    best = (E, H, {m: objs[m].pose.model_copy() for m in mov})
    T0, T1 = 0.5, 0.002
    for it in range(iterations):
        T = T0 * (T1 / T0) ** (it / max(1, iterations - 1))
        sigma = max(0.02, span * 0.25 * (1 - it / iterations))
        m = mov[rng.integers(len(mov))]
        o = objs[m]
        saved = {k: objs[k].pose.model_copy() for k in mov}
        r = rng.random()
        if r < 0.6:
            o.pose.x += rng.normal(0, sigma)
            o.pose.y += rng.normal(0, sigma)
        elif r < 0.8:
            o.pose.yaw = g.wrap_deg(o.pose.yaw + rng.choice([-90, 90, 180, -15, 15]))
        elif r < 0.9:
            o.pose.x, o.pose.y, o.pose.yaw = wall_proposal(o)
        else:  # swap two movable objects
            m2 = mov[rng.integers(len(mov))]
            if m2 != m:
                o2 = objs[m2]
                o.pose.x, o.pose.y, o2.pose.x, o2.pose.y = o2.pose.x, o2.pose.y, o.pose.x, o.pose.y
        o.pose.x = float(np.clip(o.pose.x, lo[0], hi[0]))
        o.pose.y = float(np.clip(o.pose.y, lo[1], hi[1]))
        E2, H2 = energy()
        if E2 <= E or rng.random() < math.exp(-(E2 - E) / T):
            E, H = E2, H2
            if E < best[0]:
                best = (E, H, {k: objs[k].pose.model_copy() for k in mov})
        else:  # reject: restore (a swap touches two objects)
            for k, p in saved.items():
                objs[k].pose = p
    # restore the best state found
    E, H, poses = best
    for k, p in poses.items():
        objs[k].pose = p

    # Repair: walkway connectivity is too slow for every step, so nudge until it holds.
    def walk_pen() -> float:
        vs = [v for v in checks.check_objects(room, list(objs.values()), rules) if v.kind == "walkway"]
        return 0.3 * len(vs)

    W = walk_pen()
    if W > 0 and repair_iterations:
        Er = E + HARD_W * W
        for _ in range(repair_iterations):
            m = mov[rng.integers(len(mov))]
            saved = objs[m].pose.model_copy()
            objs[m].pose.x += rng.normal(0, 0.08)
            objs[m].pose.y += rng.normal(0, 0.08)
            E2, H2 = energy()
            if H2 > H + 1e-9:  # never trade walkway for a collision
                objs[m].pose = saved
                continue
            W2 = walk_pen()
            if E2 + HARD_W * W2 < Er:
                E, H, W, Er = E2, H2, W2, E2 + HARD_W * W2
                if W == 0:
                    break
            else:
                objs[m].pose = saved
        poses = {k: objs[k].pose.model_copy() for k in mov}
    reports = []
    for c in constraints:
        cost = constraint_cost(c, objs, room)
        reports.append(ConstraintReport(constraint=c, cost=round(cost, 4), satisfied=cost < 0.05))
    vio = checks.check_objects(room, list(objs.values()), rules)
    return SolveResult(poses={k: Pose(x=round(p.x, 4), y=round(p.y, 4), z=p.z, yaw=round(p.yaw, 2)) for k, p in poses.items()},
                       energy=round(E, 4), hard_energy=round(H, 4), constraints=reports, violations=vio, iterations=iterations)
