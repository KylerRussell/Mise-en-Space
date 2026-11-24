"""Editing a home's spaces: add, rename, split an open area in two, merge, delete.

These are pure functions on a :class:`Room`; the store wraps them in undoable ops.
Walls are re-derived after every edit, and doors, windows and wall-mounted objects
are re-attached to whichever new wall now runs through them.
"""

from __future__ import annotations

import math
import re

from .models import Opening, Room, Space, Wall, polygon_area


def as_spaces(room: Room) -> Room:
    """A single-room scene turned into one explicit space (so more can be added)."""
    if room.spaces:
        return room
    r = room.model_copy(deep=True)
    r.spaces = [Space(id="room", name="Room", kind="other", polygon=list(room.floor_polygon))]
    r.floor_polygon = None
    return r


def slug(name: str, taken: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "space"
    sid, k = base, 2
    while sid in taken:
        sid, k = f"{base}_{k}", k + 1
    return sid


# ---- polygon helpers ------------------------------------------------------------

def _clip(poly: list[tuple[float, float]], axis: int, at: float, keep_greater: bool) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip against the half-plane axis >= at (or <= at)."""
    def inside(p):
        return p[axis] >= at if keep_greater else p[axis] <= at

    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        if inside(cur):
            out.append(cur)
        if inside(cur) != inside(nxt):
            t = (at - cur[axis]) / (nxt[axis] - cur[axis])
            out.append((cur[0] + (nxt[0] - cur[0]) * t, cur[1] + (nxt[1] - cur[1]) * t))
    return _clean(out)


def _clean(poly: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    """Drop repeated and collinear vertices."""
    pts = []
    for p in poly:
        p = (round(p[0], 6), round(p[1], 6))
        if not pts or math.dist(p, pts[-1]) > eps:
            pts.append(p)
    if len(pts) > 1 and math.dist(pts[0], pts[-1]) <= eps:
        pts.pop()
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(cross) < 1e-9:
                pts.pop(i)
                changed = True
                break
    return pts


def union_touching(a: list[tuple[float, float]], b: list[tuple[float, float]], tol: float = 1e-4) -> list[tuple[float, float]]:
    """Union of two polygons that share part of their boundary (e.g. the halves of a split).

    Edges are cut at every vertex lying on them; edges the two polygons share (in
    opposite directions) cancel, and the rest chain into the outline.
    """
    A = a if polygon_area(a) > 0 else list(reversed(a))
    B = b if polygon_area(b) > 0 else list(reversed(b))
    verts = [*A, *B]

    def cut(poly):
        edges = []
        for i in range(len(poly)):
            p, q = poly[i], poly[(i + 1) % len(poly)]
            L = math.dist(p, q)
            ts = [0.0, 1.0]
            for v in verts:
                t = ((v[0] - p[0]) * (q[0] - p[0]) + (v[1] - p[1]) * (q[1] - p[1])) / (L * L)
                if 1e-9 < t < 1 - 1e-9:
                    proj = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
                    if math.dist(proj, v) < tol:
                        ts.append(t)
            ts = sorted(set(ts))
            pts = [(round(p[0] + (q[0] - p[0]) * t, 6), round(p[1] + (q[1] - p[1]) * t, 6)) for t in ts]
            edges += list(zip(pts, pts[1:]))
        return edges

    ea, eb = cut(A), cut(B)
    sa, sb = set(ea), set(eb)
    shared = {e for e in ea if (e[1], e[0]) in sb}
    if not shared:
        raise ValueError("these spaces do not share a boundary")
    rest = [e for e in ea if e not in shared] + [e for e in eb if (e[1], e[0]) not in sa]
    nxt = {}
    for p, q in rest:
        nxt.setdefault(p, []).append(q)
    start = rest[0][0]
    loop, cur = [start], start
    for _ in range(len(rest)):
        cands = nxt.get(cur)
        if not cands:
            raise ValueError("could not merge these outlines")
        cur = cands.pop()
        if cur == start:
            break
        loop.append(cur)
    if len(loop) < len(rest):
        raise ValueError("these spaces only touch at a corner or enclose a gap; merge is not supported")
    return _clean(loop)


# ---- edits ------------------------------------------------------------------------

def add_space(room: Room, name: str, kind: str, polygon, ceiling_height: float | None = None,
              open_to: list[str] | None = None, space_id: str | None = None) -> tuple[Room, str]:
    r = as_spaces(room)
    sid = space_id or slug(name, {s.id for s in r.spaces})
    if sid in {s.id for s in r.spaces}:
        raise ValueError(f"space {sid!r} already exists")
    r.spaces.append(Space(id=sid, name=name, kind=kind, polygon=polygon, ceiling_height=ceiling_height,
                          open_to=open_to or []))
    return r, sid


def update_space(room: Room, space_id: str, **fields) -> Room:
    r = as_spaces(room)
    allowed = {"name", "kind", "ceiling_height", "open_to", "polygon"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot update fields {sorted(bad)}")
    idx = next((i for i, s in enumerate(r.spaces) if s.id == space_id), None)
    if idx is None:
        raise KeyError(f"no space {space_id!r}")
    r.spaces[idx] = Space.model_validate({**r.spaces[idx].model_dump(), **{k: v for k, v in fields.items() if v is not None or k == "ceiling_height"}})
    return r


def split_space(room: Room, space_id: str, axis: str, at: float, new_name: str, new_kind: str = "other") -> tuple[Room, str]:
    """Split a space along x = at or y = at. The part on the greater side becomes the new
    space; the two stay open to each other (a zone line, not a wall)."""
    r = as_spaces(room)
    s = r.space(space_id)
    ax = {"x": 0, "y": 1}[axis]
    lo = _clip(s.polygon, ax, at, keep_greater=False)
    hi = _clip(s.polygon, ax, at, keep_greater=True)
    if len(lo) < 3 or len(hi) < 3 or abs(polygon_area(lo)) < 0.05 or abs(polygon_area(hi)) < 0.05:
        raise ValueError(f"{axis} = {at} does not cut {s.name} into two parts")
    new_id = slug(new_name, {x.id for x in r.spaces})
    idx = r.spaces.index(s)
    r.spaces[idx] = s.model_copy(update={"polygon": lo, "open_to": sorted(set(s.open_to) | {new_id})})
    r.spaces.insert(idx + 1, Space(id=new_id, name=new_name, kind=new_kind, polygon=hi,
                                   ceiling_height=s.ceiling_height, open_to=sorted(set(s.open_to) | {space_id})))
    # spaces that were open to the old space stay open to both halves
    for i, o in enumerate(r.spaces):
        if space_id in o.open_to and o.id not in (space_id, new_id):
            r.spaces[i] = o.model_copy(update={"open_to": sorted(set(o.open_to) | {new_id})})
    return r, new_id


def merge_spaces(room: Room, keep_id: str, other_id: str) -> Room:
    r = as_spaces(room)
    a, b = r.space(keep_id), r.space(other_id)
    poly = union_touching(a.polygon, b.polygon)
    open_to = sorted((set(a.open_to) | set(b.open_to)) - {keep_id, other_id})
    r.spaces = [s for s in r.spaces if s.id != other_id]
    r.spaces[[s.id for s in r.spaces].index(keep_id)] = a.model_copy(update={"polygon": poly, "open_to": open_to})
    for i, o in enumerate(r.spaces):
        if other_id in o.open_to:
            ot = set(o.open_to) - {other_id}
            if o.id != keep_id:
                ot.add(keep_id)
            r.spaces[i] = o.model_copy(update={"open_to": sorted(ot)})
    return r


def delete_space(room: Room, space_id: str) -> Room:
    r = as_spaces(room)
    if len(r.spaces) == 1:
        raise ValueError("cannot delete the last space")
    r.space(space_id)
    r.spaces = [s.model_copy(update={"open_to": [x for x in s.open_to if x != space_id]})
                for s in r.spaces if s.id != space_id]
    return r


# ---- re-attaching things to walls -----------------------------------------------------

def _locate(walls: list[Wall], pt: tuple[float, float], direction: tuple[float, float] | None = None,
            tol: float = 0.03) -> tuple[Wall, float] | None:
    best = None
    for w in walls:
        ax, ay = w.a
        dx, dy = w.b[0] - ax, w.b[1] - ay
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx / L, dy / L
        if direction is not None and abs(ux * direction[0] + uy * direction[1]) < 0.99:
            continue
        along = (pt[0] - ax) * ux + (pt[1] - ay) * uy
        off = abs((pt[0] - ax) * uy - (pt[1] - ay) * ux)
        if -tol <= along <= L + tol and off <= tol and (best is None or off < best[2]):
            best = (w, float(min(max(along, 0.0), L)), off)
    return (best[0], best[1]) if best else None


def rewall(old: Room, new: Room) -> tuple[Room, dict[str, str]]:
    """Re-derive walls on `new` and move its openings onto them. Returns (room, old wall id -> new wall id).

    `new` is the edited room, still carrying the walls it was edited against, so openings
    added by the edit are re-attached too."""
    new = new.model_copy(deep=True)
    edited_walls, edited_openings = list(new.walls), list(new.openings)
    new.walls = new.derive_walls()
    mapping: dict[str, str] = {}
    for w in old.walls:
        mid = ((w.a[0] + w.b[0]) / 2, (w.a[1] + w.b[1]) / 2)
        d = (w.b[0] - w.a[0], w.b[1] - w.a[1])
        L = math.hypot(*d) or 1.0
        hit = _locate(new.walls, mid, (d[0] / L, d[1] / L))
        if hit:
            mapping[w.id] = hit[0].id
    openings: list[Opening] = []
    stale = {w.id: w for w in old.walls}
    stale.update({w.id: w for w in (edited_walls or [])})
    for op in edited_openings:  # the edit's openings, placed on the walls they were given against
        w = stale.get(op.wall)
        if w is None:
            continue
        L = w.length or 1.0
        d = ((w.b[0] - w.a[0]) / L, (w.b[1] - w.a[1]) / L)
        c = (w.a[0] + d[0] * op.center, w.a[1] + d[1] * op.center)
        hit = _locate(new.walls, c, d)
        if hit is None:  # its wall became an open boundary: the opening goes away
            continue
        nw, along = hit  # hinge is described from inside the room, so it carries over as is
        openings.append(op.model_copy(update={"wall": nw.id, "center": along}))
    new.openings = openings
    return new, mapping


def wall_for_point(room: Room, pt: tuple[float, float], tol: float = 0.25) -> str | None:
    """Nearest wall id within tol of a point (for re-homing wall-mounted objects)."""
    best = None
    for w in room.walls:
        ax, ay = w.a
        dx, dy = w.b[0] - ax, w.b[1] - ay
        L = math.hypot(dx, dy) or 1.0
        t = max(0.0, min(1.0, ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / (L * L)))
        dist = math.dist(pt, (ax + dx * t, ay + dy * t))
        if dist <= tol and (best is None or dist < best[1]):
            best = (w.id, dist)
    return best[0] if best else None


# ---- doors and windows ----------------------------------------------------------------

def door_from_points(room: Room, a, b, *, opening_id: str, kind: str = "door", max_dist: float = 0.35,
                     **fields) -> Opening:
    """An opening on the wall that runs along floor points a-b (e.g. the two jambs of a door).

    Picks the nearest wall within max_dist that is roughly parallel to a-b; the opening is
    centered on a-b's midpoint and as wide as a-b along that wall.
    """
    a, b = (float(a[0]), float(a[1])), (float(b[0]), float(b[1]))
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    seg = (b[0] - a[0], b[1] - a[1])
    seg_len = math.hypot(*seg)
    if seg_len < 0.2:
        raise ValueError("the two points are too close for a door or window")
    best = None
    for w in room.walls:
        dx, dy = w.b[0] - w.a[0], w.b[1] - w.a[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ux, uy = dx / L, dy / L
        if abs(ux * seg[0] + uy * seg[1]) / seg_len < math.cos(math.radians(25)):
            continue
        along = (mx - w.a[0]) * ux + (my - w.a[1]) * uy
        off = abs((mx - w.a[0]) * uy - (my - w.a[1]) * ux)
        if -0.1 <= along <= L + 0.1 and off <= max_dist and (best is None or off < best[0]):
            width = abs(seg[0] * ux + seg[1] * uy)
            best = (off, w, along, width)
    if best is None:
        raise ValueError(f"no wall runs along ({a[0]:.2f}, {a[1]:.2f})-({b[0]:.2f}, {b[1]:.2f}) within {max_dist} m; "
                         "add or move the wall first")
    _, w, along, width = best
    width = min(width, w.length)
    along = min(max(along, width / 2), w.length - width / 2)
    defaults = {"top": 2.03} if kind == "door" else {"bottom": 0.9, "top": 2.1, "swing": "none"}
    return Opening(id=opening_id, kind=kind, wall=w.id, center=along, width=width, **{**defaults, **fields})


def add_opening(room: Room, op: Opening) -> Room:
    r = room.model_copy(deep=True)
    if any(o.id == op.id for o in r.openings):
        raise ValueError(f"opening {op.id!r} already exists")
    r.wall(op.wall)
    r.openings.append(op)
    return r


def update_opening(room: Room, opening_id: str, **fields) -> Room:
    r = room.model_copy(deep=True)
    idx = next((i for i, o in enumerate(r.openings) if o.id == opening_id), None)
    if idx is None:
        raise KeyError(f"no opening {opening_id!r}")
    op = Opening.model_validate({**r.openings[idx].model_dump(), **fields})
    r.wall(op.wall)
    r.openings[idx] = op
    return r


def delete_opening(room: Room, opening_id: str) -> Room:
    r = room.model_copy(deep=True)
    if not any(o.id == opening_id for o in r.openings):
        raise KeyError(f"no opening {opening_id!r}")
    r.openings = [o for o in r.openings if o.id != opening_id]
    return r
