"""2D/3D geometry on the floor plane: oriented footprints, overlap depth, distances."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .models import Dims, Opening, Pose, Room, SceneObject, Wall

Point = tuple[float, float]


def footprint(pose: Pose, dims: Dims, pad: float = 0.0) -> np.ndarray:
    """Corners (4x2, CCW) of an object's oriented footprint on the floor."""
    hw, hd = dims.w / 2 + pad, dims.d / 2 + pad
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
    c, s = math.cos(pose.yaw_rad()), math.sin(pose.yaw_rad())
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([pose.x, pose.y])


def object_footprint(o: SceneObject, pad: float = 0.0) -> np.ndarray:
    return footprint(o.pose, o.extent(), pad)


def z_interval(o: SceneObject) -> tuple[float, float]:
    return (o.pose.z, o.pose.z + o.extent().h)


def local_axes(yaw_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(right, front) unit vectors of an object with this yaw."""
    a = math.radians(yaw_deg)
    right = np.array([math.cos(a), math.sin(a)])
    front = np.array([-math.sin(a), math.cos(a)])
    return right, front


# ---- convex polygon tests -----------------------------------------------

def _axes(poly: np.ndarray) -> list[np.ndarray]:
    out = []
    for i in range(len(poly)):
        e = poly[(i + 1) % len(poly)] - poly[i]
        n = np.array([-e[1], e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            out.append(n / norm)
    return out


def penetration_depth(p: np.ndarray, q: np.ndarray) -> float:
    """Separating-axis overlap of two convex polygons.

    Returns the minimum translation distance needed to separate them
    (> 0 when they overlap, <= 0 when separated; the negative value is a
    lower bound on the gap along the best separating axis).
    """
    best = math.inf
    for ax in _axes(p) + _axes(q):
        pp, qq = p @ ax, q @ ax
        overlap = min(pp.max(), qq.max()) - max(pp.min(), qq.min())
        best = min(best, overlap)
    return best


def point_segment_distance(pt: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(pt - a, ab) / np.dot(ab, ab), 0, 1))
    return float(np.linalg.norm(pt - (a + t * ab)))


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def polygon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Euclidean distance between two convex polygons (0 if they overlap)."""
    if penetration_depth(p, q) > 0:
        return 0.0
    best = math.inf
    for P, Q in ((p, q), (q, p)):
        for v in P:
            for i in range(len(Q)):
                best = min(best, point_segment_distance(v, Q[i], Q[(i + 1) % len(Q)]))
    for i in range(len(p)):
        for j in range(len(q)):
            if _segments_intersect(p[i], p[(i + 1) % len(p)], q[j], q[(j + 1) % len(q)]):
                return 0.0
    return best


def point_in_polygon(pt: Sequence[float], poly: Sequence[Sequence[float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xin > x:
                inside = not inside
    return inside


def points_in_polygon(pts: np.ndarray, poly: Sequence[Sequence[float]]) -> np.ndarray:
    """Vectorised even-odd test for an (N, 2) array."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (xin > x)
    return inside


def points_segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab)) or 1e-12
    t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(pts - proj, axis=1)


def points_polygon_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point to a polygon (0 inside)."""
    d = np.full(len(pts), np.inf)
    for i in range(len(poly)):
        d = np.minimum(d, points_segment_distance(pts, poly[i], poly[(i + 1) % len(poly)]))
    d[points_in_polygon(pts, poly)] = 0.0
    return d


def outside_amount(poly: np.ndarray, room_poly: Sequence[Point]) -> float:
    """How far (m) the polygon's corners stick out of the room polygon."""
    room = np.asarray(room_poly, dtype=float)
    worst = 0.0
    for v in poly:
        if not point_in_polygon(v, room):
            d = min(point_segment_distance(v, room[i], room[(i + 1) % len(room)]) for i in range(len(room)))
            worst = max(worst, d)
    return worst


# ---- walls and openings ---------------------------------------------------

def wall_inward_normal(wall: Wall, room: Room) -> np.ndarray:
    a, b = np.array(wall.a), np.array(wall.b)
    e = (b - a) / (np.linalg.norm(b - a) or 1.0)
    n = np.array([-e[1], e[0]])  # left normal
    mid = (a + b) / 2
    if not point_in_polygon(mid + n * 0.05, room.floor_polygon):
        n = -n
    return n


def wall_direction(wall: Wall) -> np.ndarray:
    a, b = np.array(wall.a), np.array(wall.b)
    return (b - a) / (np.linalg.norm(b - a) or 1.0)


def wall_point(wall: Wall, along: float) -> np.ndarray:
    return np.array(wall.a) + wall_direction(wall) * along


def yaw_facing(direction: np.ndarray) -> float:
    """Yaw (deg) that makes an object's front (+y) point along `direction`."""
    return math.degrees(math.atan2(-direction[0], direction[1]))


def opening_polygon(op: Opening, room: Room) -> list[tuple[float, float, float]]:
    """3D outline of a door/window (4 corners on the wall plane)."""
    w = room.wall(op.wall)
    d = wall_direction(w)
    c = wall_point(w, op.center)
    p0, p1 = c - d * op.width / 2, c + d * op.width / 2
    return [
        (float(p0[0]), float(p0[1]), op.bottom),
        (float(p1[0]), float(p1[1]), op.bottom),
        (float(p1[0]), float(p1[1]), op.top),
        (float(p0[0]), float(p0[1]), op.top),
    ]


def opening_zone(op: Opening, room: Room, depth: float) -> np.ndarray:
    """Rectangle on the floor in front of an opening (inside the room)."""
    w = room.wall(op.wall)
    d = wall_direction(w)
    n = wall_inward_normal(w, room)
    c = wall_point(w, op.center)
    p0, p1 = c - d * op.width / 2, c + d * op.width / 2
    return np.array([p0, p1, p1 + n * depth, p0 + n * depth])


def door_swing_arc(op: Opening, room: Room, segments: int = 12) -> np.ndarray:
    """Polyline of the door leaf's swing (for drawing)."""
    w = room.wall(op.wall)
    n = wall_inward_normal(w, room)
    if op.swing == "out":
        n = -n
    c = wall_point(w, op.center)
    n_in = wall_inward_normal(w, room)
    # Viewer stands inside facing the wall (direction -n_in); their left is (n_in.y, -n_in.x).
    left = np.array([n_in[1], -n_in[0]])
    hinge_dir = left if op.hinge == "left" else -left
    hinge = c + hinge_dir * op.width / 2
    start = -hinge_dir
    pts = []
    for k in range(segments + 1):
        t = (math.pi / 2) * k / segments
        v = start * math.cos(t) + n * math.sin(t)
        pts.append(hinge + v * op.width)
    return np.array([hinge, *pts])


def nearest_wall(pt: np.ndarray, room: Room) -> tuple[Wall, float, float]:
    """(wall, distance, along) of the wall closest to a point."""
    best = None
    for w in room.walls:
        a, b = np.array(w.a), np.array(w.b)
        dist = point_segment_distance(pt, a, b)
        along = float(np.clip(np.dot(pt - a, wall_direction(w)), 0, w.length))
        if best is None or dist < best[1]:
            best = (w, dist, along)
    assert best is not None
    return best


def min_area_rect(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Minimum-area bounding rectangle of 2D points.

    Returns (center, width, depth, yaw_deg) with width >= depth so the long
    side is local x.
    """
    hull = convex_hull(points)
    if len(hull) < 3:
        c = points.mean(axis=0)
        return c, 1e-3, 1e-3, 0.0
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        ang = math.atan2(e[1], e[0])
        c, s = math.cos(-ang), math.sin(-ang)
        rot = np.array([[c, -s], [s, c]])
        r = hull @ rot.T
        lo, hi = r.min(axis=0), r.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, ang, lo, hi)
    _, ang, lo, hi = best
    w, d = hi - lo
    center_r = (lo + hi) / 2
    c, s = math.cos(ang), math.sin(ang)
    center = np.array([[c, -s], [s, c]]) @ center_r
    yaw = math.degrees(ang)
    if d > w:
        w, d = d, w
        yaw += 90.0
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return center, float(w), float(d), yaw


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; returns CCW hull vertices."""
    pts = sorted(set(map(tuple, np.round(points, 9))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
