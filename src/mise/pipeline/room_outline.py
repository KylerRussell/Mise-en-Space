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
                 z_band: tuple[float, float] = Z_BAND) -> HomeOutline:
    """Free space -> rooms. Doorways (narrower than core_m) separate rooms; open-plan areas stay one
    room (split those into zones in software). Returns rectilinear outlines and the doorways found."""
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
    rooms, index = [], {}
    ceiling = outline_ceiling(points)
    for k in range(len(keep)):
        m = (labels == k).astype(np.uint8)
        if m.sum() * cell * cell < min_room_m2:
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
                             free=grid, origin=origin, cell=cell))
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
