"""Room outline from a sparse reconstruction (no dense splat needed).

Every camera-to-point sight line crosses open floor, so in plan view the cells a
line passes through are free space. Cells crossed by enough lines form the room;
its outline, snapped to the two wall directions, is the floor polygon. Lines that
run through a doorway into the next room add a thin spur that the opening step
(morphology) removes.

Input is metric and leveled (floor z = 0, walls along x/y), e.g. after
`normalize` + a scale. Needs opencv (the `poses` extra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INSET = 1.5  # cells between the outline of free space and the wall, measured on synthetic rooms (see _finish)


@dataclass
class Outline:
    polygon: list[tuple[float, float]]  # counter-clockwise, meters
    area_m2: float
    ceiling_m: float | None
    free: np.ndarray  # the free-space grid (for debugging plots)
    origin: tuple[float, float]
    cell: float
    kind: str = "room"  # "closet": a pocket split off the room it opens off


# Sight lines go only to points this high: above most furniture, so lines pass over sofas,
# desks and counters to the walls behind them instead of treating furniture as walls.
Z_BAND = (1.0, 2.3)


def free_space(cams: np.ndarray, sights: list[np.ndarray], cell: float = 0.05,
               z_band: tuple[float, float] = Z_BAND) -> tuple[np.ndarray, tuple[float, float]]:
    """cams: Nx3 camera centers. sights[i]: Mx3 points camera i observed. Returns hit counts per cell."""
    allp = np.vstack([cams] + [s for s in sights if len(s)])
    lo = np.percentile(allp[:, :2], 1, axis=0) - 0.5
    hi = np.percentile(allp[:, :2], 99, axis=0) + 0.5
    shape = (int(np.ceil((hi[1] - lo[1]) / cell)), int(np.ceil((hi[0] - lo[0]) / cell)))
    grid = np.zeros(shape, np.int32)
    for c, pts in zip(cams, sights):
        if not len(pts):
            continue
        p = pts[(pts[:, 2] > z_band[0]) & (pts[:, 2] < z_band[1])]
        if not len(p):
            continue
        # stop each line 2 cells short of the point so the wall itself stays closed
        d = p[:, :2] - c[:2]
        L = np.linalg.norm(d, axis=1)
        keep = L > 3 * cell
        d, L = d[keep], L[keep]
        if not len(L):
            continue
        stop = np.maximum(L - 2 * cell, 0) / L
        n = int(np.ceil(L.max() / (cell / 2))) + 1
        t = np.linspace(0, 1, n)[None, :] * stop[:, None]  # (M, n)
        xs = c[0] + d[:, 0:1] * t
        ys = c[1] + d[:, 1:2] * t
        ix = ((xs - lo[0]) / cell).astype(int).ravel()
        iy = ((ys - lo[1]) / cell).astype(int).ravel()
        ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
        cells = np.unique(iy[ok] * shape[1] + ix[ok])  # count each cell once per camera
        np.add.at(grid.ravel(), cells, 1)
    return grid, (float(lo[0]), float(lo[1]))


def _rectilinear(poly: np.ndarray, min_edge: float) -> np.ndarray:
    """Snap a closed polygon to axis-aligned edges and merge short jogs."""
    pts = [tuple(p) for p in poly]
    # alternate horizontal/vertical: each vertex takes x from one neighbor edge, y from the other
    out = []
    n = len(pts)
    for i in range(n):
        a, b = np.array(pts[i]), np.array(pts[(i + 1) % n])
        out.append((a, b, "h" if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else "v"))
    # merge consecutive edges of the same orientation
    merged = []
    for a, b, o in out:
        if merged and merged[-1][2] == o:
            merged[-1] = (merged[-1][0], b, o)
        else:
            merged.append((a, b, o))
    if len(merged) > 1 and merged[0][2] == merged[-1][2]:
        a, _, o = merged[-1]
        merged[0] = (a, merged[0][1], o)
        merged.pop()
    # each edge's coordinate: mean of its constant axis
    lines = [(o, (a[1] + b[1]) / 2 if o == "h" else (a[0] + b[0]) / 2, np.linalg.norm(b - a)) for a, b, o in merged]
    # drop short jogs (an edge shorter than min_edge between two of the other orientation)
    changed = True
    while changed and len(lines) > 4:
        changed = False
        for i, (o, v, L) in enumerate(lines):
            if L < min_edge:
                prev, nxt = lines[i - 1], lines[(i + 1) % len(lines)]
                if prev[0] == nxt[0]:
                    w = prev[2] + nxt[2]
                    joined = (prev[0], (prev[1] * prev[2] + nxt[1] * nxt[2]) / max(w, 1e-9), w)
                    idx = sorted([(i - 1) % len(lines), i, (i + 1) % len(lines)])
                    lines = [x for k, x in enumerate(lines) if k not in idx]
                    lines.insert(min(idx), joined)
                    changed = True
                    break
    verts = []
    for i, (o, v, _) in enumerate(lines):
        o2, v2, _ = lines[(i + 1) % len(lines)]
        if o == o2:
            continue
        verts.append((v2, v) if o == "h" else (v, v2))
    return np.array(verts)


def _offset_ccw(poly: np.ndarray, d: float) -> np.ndarray:
    """Push every edge of a counter-clockwise rectilinear polygon outward by d."""
    n = len(poly)
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        out = np.array([e[1], -e[0]]) / (np.linalg.norm(e) or 1.0)  # right of travel = outside for CCW
        lines.append((a + out * d, e))
    verts = []
    for i in range(n):
        (p1, e1), (p2, e2) = lines[i - 1], lines[i]
        A = np.array([e1, -e2]).T
        if abs(np.linalg.det(A)) < 1e-12:
            verts.append(p2)
            continue
        t = np.linalg.solve(A, p2 - p1)
        verts.append(p1 + e1 * t[0])
    return np.array(verts)


def _finish(poly: np.ndarray, inset: float) -> tuple[np.ndarray, float]:
    """CCW orientation, then undo the inset that stopping sight lines short of walls leaves."""
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 0:
        poly = poly[::-1]
    poly = _offset_ccw(poly, inset)
    x, y = poly[:, 0], poly[:, 1]
    return poly, 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def outline(cams: np.ndarray, sights: list[np.ndarray], points: np.ndarray | None = None, cell: float = 0.05,
            min_hits: int = 3, open_m: float = 0.35, min_edge: float = 0.3, z_band: tuple[float, float] = Z_BAND) -> Outline:
    import cv2

    grid, origin = free_space(cams, sights, cell, z_band)
    free = (grid >= min_hits).astype(np.uint8)
    k = max(1, int(round(open_m / cell)))
    kernel = np.ones((k, k), np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, kernel)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(free)
    if n < 2:
        raise RuntimeError("no free space found; check that points and cameras are leveled and metric")
    # the room is the component holding the most camera centers
    ci = ((cams[:, 0] - origin[0]) / cell).astype(int), ((cams[:, 1] - origin[1]) / cell).astype(int)
    ok = (ci[0] >= 0) & (ci[0] < free.shape[1]) & (ci[1] >= 0) & (ci[1] < free.shape[0])
    votes = np.bincount(lab[ci[1][ok], ci[0][ok]], minlength=n)
    votes[0] = 0
    best = int(votes.argmax()) if votes.max() > 0 else 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    room = (lab == best).astype(np.uint8)
    cs, _ = cv2.findContours(room, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].astype(float)
    poly = (c + 0.5) * cell + np.array(origin)
    poly, area = _finish(_rectilinear(poly, min_edge), INSET * cell)
    ceiling = outline_ceiling(points)
    return Outline(polygon=[(float(a), float(b)) for a, b in poly], area_m2=area, ceiling_m=ceiling,
                   free=grid, origin=origin, cell=cell)


# ---- a whole home: split free space into rooms at doorways ------------------------------

@dataclass
class Doorway:
    center: tuple[float, float]
    width: float
    rooms: tuple[int, int]  # indexes into HomeOutline.rooms
    axis: str  # "x" if the doorway runs along x (a wall parallel to x), else "y"


@dataclass
class HomeOutline:
    rooms: list[Outline]
    doorways: list[Doorway]
    labels: np.ndarray  # room index per free cell (-1 elsewhere)
    origin: tuple[float, float]
    cell: float


def segment_home(cams: np.ndarray, sights: list[np.ndarray], points: np.ndarray | None = None, cell: float = 0.05,
                 min_hits: int = 3, core_m: float = 1.0, min_room_m2: float = 1.5, min_edge: float = 0.3,
                 z_band: tuple[float, float] = Z_BAND, pocket_m2: float = 0.45, pocket_mouth_m: float = 1.2,
                 body_m: float = 1.4) -> HomeOutline:
    """Free space -> rooms. Doorways (narrower than core_m) separate rooms; open-plan areas stay one
    room (split those into zones in software). Returns rectilinear outlines and the doorways found.

    Closets and nooks too small to hold a room core of their own would be grown into the room
    they open off; instead, a part of a room that its body (the room opened by a core_m disk)
    doesn't cover, at least `pocket_m2` big and joined to the body by a mouth no wider than
    `pocket_mouth_m`, becomes its own space with a doorway at the mouth."""
    import cv2
    from collections import deque

    grid, origin = free_space(cams, sights, cell, z_band)
    free = (grid >= min_hits).astype(np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # room cores: what survives eroding by half a doorway-plus
    r = max(1, int(round(core_m / 2 / cell)))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    core = cv2.erode(free, disk)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] * cell * cell >= 0.25]
    labels = -np.ones(free.shape, int)
    q = deque()
    for k, i in enumerate(keep):
        ys, xs = np.nonzero(lab == i)
        labels[ys, xs] = k
        q.extend(zip(ys.tolist(), xs.tolist()))
    # grow cores over free space (breadth first, so each cell goes to the nearest core)
    H, W = free.shape
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and free[yy, xx] and labels[yy, xx] < 0:
                labels[yy, xx] = labels[y, x]
                q.append((yy, xx))
    # closets and nooks: pockets of a room its body doesn't reach, joined by a narrow mouth
    n_main = len(keep)
    small = set()
    # a square (walls run along x and y), so the body can't reach through a mouth narrower than
    # body_m into what lies behind it, and keeps the room's own corners
    rb = max(1, int(round(body_m / 2 / cell)))
    square = np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)
    for k in range(n_main):
        m = (labels == k).astype(np.uint8)
        body = cv2.morphologyEx(m, cv2.MORPH_OPEN, square)
        if body.sum() < 0.5 * m.sum():
            continue  # a narrow room (hallway): no body to hang pockets off
        pocket = (m & (1 - body)).astype(np.uint8)
        pocket = cv2.morphologyEx(pocket, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        npk, plab, pstats, _ = cv2.connectedComponentsWithStats(pocket)
        grown = cv2.dilate(body, np.ones((3, 3), np.uint8))
        for j in range(1, npk):
            if pstats[j, cv2.CC_STAT_AREA] * cell * cell < pocket_m2:
                continue
            pm = plab == j
            mouth = pm & grown.astype(bool)
            ys, xs = np.nonzero(mouth)
            if not len(ys):
                continue
            width = max(np.ptp(xs), np.ptp(ys)) * cell + cell
            if width > pocket_mouth_m:
                continue
            new = len(keep) + len(small)
            labels[pm] = new
            small.add(new)
    rooms, index = [], {}
    ceiling = outline_ceiling(points)
    for k in range(n_main + len(small)):
        m = (labels == k).astype(np.uint8)
        if m.sum() * cell * cell < (pocket_m2 if k in small else min_room_m2):
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].astype(float)
        poly = _rectilinear((c + 0.5) * cell + np.array(origin), min_edge)
        if len(poly) < 4:
            continue
        poly, area = _finish(poly, INSET * cell)
        index[k] = len(rooms)
        rooms.append(Outline(polygon=[(float(a), float(b)) for a, b in poly], area_m2=area, ceiling_m=ceiling,
                             free=grid, origin=origin, cell=cell, kind="closet" if k in small else "room"))
    # doorways: where two rooms' cells touch
    doorways = []
    for axis, (dy, dx) in (("x", (1, 0)), ("y", (0, 1))):
        a, b = labels[: H - dy, : W - dx], labels[dy:, dx:]
        touch = (a >= 0) & (b >= 0) & (a != b)
        pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for y, x in zip(*np.nonzero(touch)):
            ka, kb = int(a[y, x]), int(b[y, x])
            if ka in index and kb in index:
                pairs.setdefault(tuple(sorted((index[ka], index[kb]))), []).append((y, x))
        for (ia, ib), cells in pairs.items():
            cells = np.array(cells)
            along = cells[:, 1] if axis == "x" else cells[:, 0]
            width = (along.max() - along.min() + 1) * cell
            cy, cx = cells.mean(0)
            center = ((cx + 0.5 + dx / 2) * cell + origin[0], (cy + 0.5 + dy / 2) * cell + origin[1])
            doorways.append(Doorway(center=(float(center[0]), float(center[1])), width=float(width), rooms=(ia, ib), axis=axis))
    # one doorway per pair of rooms (the widest contact), and nothing narrower than a door
    best: dict[tuple[int, int], Doorway] = {}
    for d in doorways:
        if d.width >= 0.4 and (d.rooms not in best or d.width > best[d.rooms].width):
            best[d.rooms] = d
    doorways = list(best.values())
    return HomeOutline(rooms=rooms, doorways=doorways, labels=labels, origin=origin, cell=cell)


def outline_ceiling(points: np.ndarray | None) -> float | None:
    """Ceiling height: a dense layer of points near the top if there is one (textured or
    lit ceilings), else the top of the point cloud (plain ceilings yield few points)."""
    if points is None or not len(points):
        return None
    z = points[:, 2]
    hi = z[z > 1.9]
    if len(hi) <= 30:
        return None
    top = float(np.percentile(hi, 99.5))
    h, e = np.histogram(hi, bins=np.arange(1.9, top + 0.06, 0.03))
    peak = float(e[h.argmax()] + 0.015)
    # a real ceiling layer sits at the top; a peak far below it is wall clutter (cabinets, frames)
    return peak if top - peak < 0.1 else top


@dataclass
class EdgeFix:
    room: int
    edge: int  # index of the edge's first vertex
    moved_m: float
    reason: str


def push_furniture_edges(rooms: list[Outline], points: np.ndarray, low: tuple[float, float] = (0.1, 1.4),
                         max_push: float = 1.0, min_push: float = 0.2, margin: float = 0.03,
                         slab: float = 0.05, skip: set[tuple[int, int]] | None = None) -> list[EdgeFix]:
    """Move walls that are really furniture fronts back to the wall behind them (in place).

    Free space stops at whatever the camera couldn't see past, so a long sideboard or bookcase
    reads as a wall. Plain painted walls give almost no points, but furniture does: its front,
    top and the things on it fill a layer of points that is *deep* (tens of cm) and stops at
    the wall. A real wall's points, if any, form a thin layer at the edge. So for each edge:
    if the low points just outside it form a deep layer with nothing beyond, the wall is at
    the back of that layer.
    """
    fixes = []
    for ri, r in enumerate(rooms):
        if r.kind == "closet":
            continue  # shallow, walls close to the camera: nothing stands in front of them
        poly = np.array(r.polygon, float)
        n = len(poly)
        others = [np.array(o.polygon) for k, o in enumerate(rooms) if k != ri]
        moves = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < 0.6 or (skip and (ri, i) in skip):
                continue
            u = e / L
            out = np.array([u[1], -u[0]])  # outside for CCW
            rel = points[:, :2] - a
            s, d = rel @ u, rel @ out
            span = (s > 0.15) & (s < L - 0.15)
            z = points[:, 2]
            lo = span & (z > low[0]) & (z < low[1]) & (d > -0.1) & (d < max_push + 0.1)
            if lo.sum() < 30:
                continue
            bins = np.arange(-0.1, max_push + 0.1 + slab, slab)
            h, _ = np.histogram(d[lo], bins)
            need = max(3, 0.02 * lo.sum())
            k = max(j for j in range(len(h)) if h[j] >= need) if (h >= need).any() else None
            if k is None:
                continue
            back = float(np.percentile(d[lo][d[lo] < bins[k + 1]], 98)) + margin
            # a layer that keeps going (the room continues, or a doorway) is not furniture against a wall
            beyond = span & (d > back + 0.1) & (d < back + 0.8) & (z > 0.05) & (z < 2.3)
            if back < min_push or back > max_push or beyond.sum() > 0.05 * lo.sum():
                continue
            mid = (a + b) / 2 + out * back
            if any(_inside(o, mid) for o in others):
                continue
            moves.append((i, out * back, int(lo.sum()), back))
        for i, v, cnt, back in moves:
            poly[i] = poly[i] + v
            poly[(i + 1) % n] = poly[(i + 1) % n] + v
            fixes.append(EdgeFix(ri, i, round(back, 2), f"{cnt} points of furniture in a {back:.2f} m deep layer"))
        if moves:
            x, y = poly[:, 0], poly[:, 1]
            r.polygon = [(round(float(p), 3), round(float(q), 3)) for p, q in poly]
            r.area_m2 = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return fixes


def snap_edges_to_walls(rooms: list[Outline], cams: np.ndarray, sights: list[np.ndarray], band: tuple[float, float] = (1.3, 2.3),
                        reach: float = 0.6, min_move: float = 0.05, min_pts: int = 40,
                        bin_m: float = 0.02) -> tuple[list[EdgeFix], set[tuple[int, int]]]:
    """Move each wall to where the points seen from inside its room say it is (in place).

    Free space stops short of walls by however far the points scatter, and at furniture the
    camera couldn't see past. Above furniture height, though, the wall itself is what the
    cameras see: for each edge, the densest layer of points between 1.3 and 2.3 m within
    `reach` of it, seen from cameras inside the room (so a neighbor's side of a shared wall
    doesn't count), is the wall. Dense learned depth gives plenty of such points even on
    plain walls; sparse solves often don't, and those edges are left for push_furniture_edges.

    Returns the walls moved and every (room, edge) whose wall layer was found, moved or not
    (leave those out of push_furniture_edges: their wall is known).
    """
    fixes, placed = [], set()
    for ri, r in enumerate(rooms):
        poly = np.array(r.polygon, float)
        n = len(poly)
        inside = [k for k, c in enumerate(cams) if _inside(poly, c[:2]) and len(sights[k])]
        if len(inside) < 3:
            continue
        P = np.vstack([sights[k] for k in inside])
        P = P[(P[:, 2] > band[0]) & (P[:, 2] < band[1])]
        moves = []
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            e = b - a
            L = float(np.linalg.norm(e))
            if L < 0.4:
                continue
            u = e / L
            out = np.array([u[1], -u[0]])
            rel = P[:, :2] - a
            s_, d = rel @ u, rel @ out
            sel = (s_ > 0.15) & (s_ < L - 0.15) & (np.abs(d) < reach)
            if sel.sum() < min_pts:
                continue
            dd = d[sel]
            edges = np.arange(-reach, reach + bin_m, bin_m)
            h, _ = np.histogram(dd, edges)
            h = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
            peak = edges[int(h.argmax())] + bin_m / 2
            near = np.abs(dd - peak) < 0.06
            if near.sum() < max(min_pts // 2, 0.2 * len(dd)):
                continue  # no clear wall layer
            at = float(np.median(dd[near]))
            placed.add((ri, i))
            if abs(at) < min_move:
                continue
            moves.append((i, out * at, int(near.sum()), at))
        for i, v, cnt, at in moves:
            poly[i] = poly[i] + v
            poly[(i + 1) % n] = poly[(i + 1) % n] + v
            fixes.append(EdgeFix(ri, i, round(at, 2), f"wall layer of {cnt} points {at:+.2f} m from the free-space edge"))
        if moves:
            x, y = poly[:, 0], poly[:, 1]
            r.polygon = [(round(float(p_), 3), round(float(q), 3)) for p_, q in poly]
            r.area_m2 = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return fixes, placed


def settle_closets(rooms: list[Outline], reach: float = 0.25) -> int:
    """Put each closet's mouth on the wall line of the room it opens off (in place).

    A closet's free space starts inside the doorway, so its outline overlaps the room's by the
    wall's thickness or so. Each closet edge that lies within `reach` of a parallel edge of a
    room (overlapping along its length) takes that edge's line. Returns the edges moved.
    """
    moved = 0
    for c in rooms:
        if c.kind != "closet":
            continue
        cp = np.array(c.polygon, float)
        n = len(cp)
        for i in range(n):
            a, b = cp[i], cp[(i + 1) % n]
            axis = 0 if abs(a[0] - b[0]) < 1e-6 else 1  # 0: edge at fixed x
            if abs(a[axis] - b[axis]) > 1e-6:
                continue
            lo, hi = sorted([a[1 - axis], b[1 - axis]])
            best = None
            for r in rooms:
                if r is c or r.kind == "closet":
                    continue
                rp = np.array(r.polygon, float)
                for p_, q in zip(rp, np.roll(rp, -1, axis=0)):
                    if abs(p_[axis] - q[axis]) > 1e-6:
                        continue
                    rlo, rhi = sorted([p_[1 - axis], q[1 - axis]])
                    if min(hi, rhi) - max(lo, rlo) < 0.3 or abs(p_[axis] - a[axis]) > reach:
                        continue
                    if best is None or abs(p_[axis] - a[axis]) < abs(best - a[axis]):
                        best = p_[axis]
            if best is not None and abs(best - a[axis]) > 1e-3:
                cp[i][axis] = best
                cp[(i + 1) % n][axis] = best
                moved += 1
        x, y = cp[:, 0], cp[:, 1]
        c.polygon = [(round(float(p_), 3), round(float(q), 3)) for p_, q in cp]
        c.area_m2 = abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return moved


def _inside(poly: np.ndarray, p) -> bool:
    x, y = p
    inside = False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside
