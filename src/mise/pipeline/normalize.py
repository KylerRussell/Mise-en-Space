"""Stage 1 step 5: put a reconstructed room in a metric, z-up, wall-aligned frame.

1. scale to meters,
2. RANSAC the floor plane on Gaussian centers and rotate its normal to +z
   with the floor at z = 0,
3. rotate about z so walls align with the x and y axes (Manhattan
   alignment: the yaw that makes the x/y histograms sharpest).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ply


@dataclass
class Plane:
    n: np.ndarray  # unit normal
    d: float  # n . p + d = 0
    inliers: int


def ransac_plane(pts: np.ndarray, iters: int = 400, thresh: float = 0.02, up_hint: np.ndarray | None = None,
                 max_tilt_deg: float = 30.0, rng: np.random.Generator | None = None) -> Plane:
    rng = rng or np.random.default_rng(0)
    best: Plane | None = None
    n_pts = len(pts)
    for _ in range(iters):
        a, b, c = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if up_hint is not None:
            if np.dot(n, up_hint) < 0:
                n = -n
            if np.degrees(np.arccos(np.clip(np.dot(n, up_hint), -1, 1))) > max_tilt_deg:
                continue
        d = -float(np.dot(n, a))
        dist = pts @ n + d
        inl = int((np.abs(dist) < thresh).sum())
        # The floor has (almost) everything above it.
        below = (dist < -3 * thresh).mean()
        if below > 0.05:
            continue
        if best is None or inl > best.inliers:
            best = Plane(n, d, inl)
    if best is None:
        raise RuntimeError("no floor plane found; pass up_hint or check the scan")
    # refine on inliers (least squares)
    dist = pts @ best.n + best.d
    inl = pts[np.abs(dist) < thresh]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c)
    n = vt[-1] if np.dot(vt[-1], best.n) > 0 else -vt[-1]
    return Plane(n, -float(np.dot(n, c)), len(inl))


def manhattan_yaw(xy: np.ndarray, bin_size: float = 0.02, step_deg: float = 0.25) -> float:
    """Yaw (deg) that, applied as a rotation, aligns dominant walls with x/y."""
    best, best_e = 0.0, np.inf
    for deg in np.arange(0, 90, step_deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        r = xy @ np.array([[c, -s], [s, c]]).T
        e = 0.0
        for k in range(2):
            v = r[:, k]
            h, _ = np.histogram(v, bins=max(8, int((v.max() - v.min()) / bin_size)))
            p = h[h > 0] / len(v)
            e += float(-(p * np.log(p)).sum())
        if e < best_e:
            best, best_e = float(deg), e
    return best


@dataclass
class NormalizeResult:
    R: np.ndarray
    t: np.ndarray
    scale: float
    floor: Plane
    yaw_deg: float

    def matrix(self) -> list[list[float]]:
        M = np.eye(4)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M.tolist()


def normalize(s: ply.Splat, scale: float = 1.0, up_hint=(0, 0, 1), sample: int = 200_000,
              wall_height_band: tuple[float, float] = (0.3, 2.0)) -> tuple[ply.Splat, NormalizeResult]:
    rng = np.random.default_rng(0)
    keep = s.alpha() > 0.3
    pts = s.xyz[keep].astype(np.float64) * scale
    if len(pts) > sample:
        pts = pts[rng.choice(len(pts), sample, replace=False)]
    up = np.asarray(up_hint, float)
    up /= np.linalg.norm(up)
    floor = ransac_plane(pts, up_hint=up)
    R1 = ply.rotation_between(floor.n, np.array([0.0, 0, 1]))
    p1 = pts @ R1.T
    z0 = -floor.d  # after rotation, floor points have z = -d
    p1[:, 2] -= z0
    band = (p1[:, 2] > wall_height_band[0]) & (p1[:, 2] < wall_height_band[1])
    yaw = manhattan_yaw(p1[band, :2]) if band.sum() > 100 else 0.0
    R = ply.rot_z(yaw) @ R1
    t = np.array([0, 0, -z0])
    t = ply.rot_z(yaw) @ t
    out = ply.transform(s, R=R, t=t, scale=scale)
    # move the room's min corner to the origin on x/y (keeps coordinates positive)
    xy = out.xyz[out.alpha() > 0.3][:, :2].astype(np.float64)
    lo = np.percentile(xy, 0.5, axis=0)
    out = ply.transform(out, t=np.array([-lo[0], -lo[1], 0.0]))
    t = t + np.array([-lo[0], -lo[1], 0.0])
    return out, NormalizeResult(R=R, t=t, scale=scale, floor=floor, yaw_deg=yaw)


def crop_boxes(s: ply.Splat, boxes: list[tuple[np.ndarray, np.ndarray]]) -> ply.Splat:
    """Remove Gaussians inside axis-aligned boxes (e.g. the printed markers)."""
    keep = np.ones(len(s), bool)
    for lo, hi in boxes:
        keep &= ~np.all((s.xyz >= lo) & (s.xyz <= hi), axis=1)
    return s.subset(keep)
