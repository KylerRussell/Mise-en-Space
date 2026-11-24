"""Stage 4 step 4: find an object's up and front axes and move it to the canonical frame.

Canonical frame: centered on x/y, z up, footprint on z = 0, front faces +y.

Up is the axis normal to the object's base. Candidates are the asset's
coordinate axes and principal axes (both signs). Each candidate is scored by
how much of the object sits in a flat bottom band and whether the center
of mass projects inside that band's footprint (a gravity check: a chair
balanced on its back fails it). Front is the horizontal box axis that best
faces the input camera. When the best two up candidates score within a
small margin the result is marked ambiguous, so the caller can ask a VLM
with two renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..scene.geometry import convex_hull, min_area_rect, points_in_polygon


@dataclass
class Orientation:
    R: np.ndarray  # asset frame -> canonical, 3x3
    t: np.ndarray  # applied after R
    dims: tuple[float, float, float]  # measured (w, d, h) in asset units
    up_score: float
    ambiguous: bool
    candidates: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _basis_from_up(up: np.ndarray) -> np.ndarray:
    up = up / np.linalg.norm(up)
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.stack([x, y, up])  # rows: new x, y, z expressed in asset frame


def _hull_area(pts2: np.ndarray) -> float:
    if len(pts2) < 3:
        return 0.0
    h = convex_hull(pts2)
    if len(h) < 3:
        return 0.0
    x, y = h[:, 0], h[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def score_up(points: np.ndarray, up: np.ndarray, weights: np.ndarray | None = None, band: float = 0.04) -> float:
    B = _basis_from_up(up)
    q = points @ B.T
    h = q[:, 2]
    lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
    rng = max(hi - lo, 1e-9)
    bottom = q[h < lo + band * rng]
    if len(bottom) < 3:
        return 0.0
    base_area = _hull_area(bottom[:, :2])
    full_area = _hull_area(q[:, :2])
    if full_area <= 0:
        return 0.0
    w = np.ones(len(q)) if weights is None else weights
    com = (q[:, :2] * w[:, None]).sum(0) / w.sum()
    hull = convex_hull(bottom[:, :2])
    stable = len(hull) >= 3 and bool(points_in_polygon(com[None], hull)[0])
    base_ratio = min(base_area / full_area, 1.0)
    return float(base_ratio * (1.0 if stable else 0.3))


def orient(points: np.ndarray, weights: np.ndarray | None = None, up_hint: np.ndarray | None = None,
           camera_dir: np.ndarray | None = None, ambiguity_margin: float = 0.15) -> Orientation:
    """Compute the canonical transform for an object's point set (asset units).

    up_hint: a declared up axis (e.g. +Y for Idol Max GLBs); it gets a strong prior.
    camera_dir: direction from the input camera toward the object, in asset frame.
    """
    pts = np.asarray(points, dtype=np.float64)
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    cands: dict[str, np.ndarray] = {}
    for i, name in enumerate("xyz"):
        e = np.zeros(3)
        e[i] = 1
        cands[f"+{name}"], cands[f"-{name}"] = e, -e
    for i in range(3):
        cands[f"+pca{i}"], cands[f"-pca{i}"] = vt[i], -vt[i]
    scored = []
    for name, u in cands.items():
        s = score_up(pts, u, weights)
        if up_hint is not None and np.dot(u, up_hint / np.linalg.norm(up_hint)) > 0.99:
            s += 0.5
        elif not name.startswith(("+pca", "-pca")):
            s += 0.02  # generators output axis-aligned objects; break ties toward coordinate axes
        scored.append((name, s))
    scored.sort(key=lambda t: -t[1])
    # merge near-duplicate directions (pca axes aligned with coordinate axes)
    best_name, best = scored[0]
    up = cands[best_name]
    runner = next(((n, s) for n, s in scored[1:] if np.dot(cands[n], up) < 0.95), (None, 0.0))
    ambiguous = (best - runner[1]) < ambiguity_margin
    notes = []
    if ambiguous:
        notes.append(f"up axis ambiguous: {best_name} ({best:.2f}) vs {runner[0]} ({runner[1]:.2f}); confirm with a VLM")

    B = _basis_from_up(up)
    q = pts @ B.T
    center, w, d, yaw = min_area_rect(q[:, :2])
    a = np.radians(yaw)
    # box axes in the leveled frame
    ax_w = np.array([np.cos(a), np.sin(a)])
    ax_d = np.array([-np.sin(a), np.cos(a)])
    # pick front among ±ax_w, ±ax_d
    front_opts = [(ax_d, False), (-ax_d, False), (ax_w, True), (-ax_w, True)]
    if camera_dir is not None:
        cam = (np.asarray(camera_dir, float) @ B.T)[:2]
        if np.linalg.norm(cam) > 1e-9:
            to_cam = -cam / np.linalg.norm(cam)
            front, swap = max(front_opts, key=lambda o: float(np.dot(o[0], to_cam)))
        else:
            front, swap = front_opts[0]
            notes.append("camera looks straight down; front unknown")
    else:
        front, swap = front_opts[0]
        notes.append("no camera direction; front set to the box's +depth side")
    if swap:
        w, d = d, w
    # yaw that maps `front` to +y
    phi = np.arctan2(front[1], front[0])
    rz = np.pi / 2 - phi
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = Rz @ B
    p2 = pts @ R.T
    lo, hi = p2.min(axis=0), p2.max(axis=0)
    # robust z floor: ignore a few floaters
    zmin = np.percentile(p2[:, 2], 0.2)
    t = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, zmin])
    h = float(np.percentile(p2[:, 2], 99.8) - zmin)
    dims = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), h)
    return Orientation(R=R, t=t, dims=dims, up_score=best, ambiguous=ambiguous, candidates=scored[:4], notes=notes)


def robust_dims(points_canonical: np.ndarray, q: float = 0.5) -> tuple[float, float, float]:
    """Extents after trimming q percent of points on each side (floaters)."""
    lo = np.percentile(points_canonical, q, axis=0)
    hi = np.percentile(points_canonical, 100 - q, axis=0)
    w, d, h = (hi - lo).tolist()
    return (w, d, h)
