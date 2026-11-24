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


@dataclass
class Outline:
    polygon: list[tuple[float, float]]  # counter-clockwise, meters
    area_m2: float
    ceiling_m: float | None
    free: np.ndarray  # the free-space grid (for debugging plots)
    origin: tuple[float, float]
    cell: float


def free_space(cams: np.ndarray, sights: list[np.ndarray], cell: float = 0.05,
               z_band: tuple[float, float] = (0.05, 2.2)) -> tuple[np.ndarray, tuple[float, float]]:
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


def outline(cams: np.ndarray, sights: list[np.ndarray], points: np.ndarray | None = None, cell: float = 0.05,
            min_hits: int = 3, open_m: float = 0.35, min_edge: float = 0.3) -> Outline:
    import cv2

    grid, origin = free_space(cams, sights, cell)
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
    poly = _rectilinear(poly, min_edge)
    # counter-clockwise
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area < 0:
        poly = poly[::-1]
        area = -area
    ceiling = None
    if points is not None and len(points):
        z = points[:, 2]
        hi = z[z > 1.9]
        if len(hi) > 30:
            h, e = np.histogram(hi, bins=np.arange(1.9, max(hi.max(), 2.0) + 0.05, 0.03))
            ceiling = float(e[h.argmax()] + 0.015)
    return Outline(polygon=[(float(a), float(b)) for a, b in poly], area_m2=area, ceiling_m=ceiling,
                   free=grid, origin=origin, cell=cell)
