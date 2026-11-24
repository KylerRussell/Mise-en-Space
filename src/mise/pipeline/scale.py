"""Stage 1: metric scale from printed markers, verified on held-out tape distances.

Inputs are camera poses/intrinsics from COLMAP (or ARKit) and per-frame
marker corner detections (see markers.py). Each marker's four corners are
triangulated across frames; its scale factor is
``known side length / mean triangulated side``. The median factor is the
scene scale and the spread across markers is the error bar. Held-out tape
distances, never used in calibration, give the percent error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

TARGET = 0.01  # 1%
KILL = 0.02  # 2%


@dataclass
class Camera:
    K: np.ndarray  # 3x3 intrinsics
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])


def triangulate(cams: list[Camera], uv: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one point seen in several frames."""
    rows = []
    for c, (u, v) in zip(cams, uv):
        P = c.P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return X[:3] / X[3]


@dataclass
class MarkerScale:
    marker_id: int
    n_views: int
    side_model: float  # mean triangulated side length, model units
    scale: float  # meters per model unit
    side_spread: float  # relative spread of the 4 sides (planarity/detection check)


def marker_scales(cams: dict[str, Camera], detections: dict[str, dict[int, np.ndarray]], side_m: float,
                  min_views: int = 3) -> list[MarkerScale]:
    """detections: frame -> {marker id -> 4x2 corner pixels (OpenCV order)}."""
    by_marker: dict[int, list[tuple[Camera, np.ndarray]]] = {}
    for frame, dets in detections.items():
        if frame not in cams:
            continue
        for mid, corners in dets.items():
            by_marker.setdefault(mid, []).append((cams[frame], np.asarray(corners, float)))
    out = []
    for mid, obs in sorted(by_marker.items()):
        if len(obs) < min_views:
            continue
        cs = [c for c, _ in obs]
        X = np.array([triangulate(cs, [o[k] for _, o in obs]) for k in range(4)])
        sides = np.array([np.linalg.norm(X[(k + 1) % 4] - X[k]) for k in range(4)])
        mean = float(sides.mean())
        out.append(MarkerScale(mid, len(obs), mean, side_m / mean, float(sides.std() / mean)))
    return out


@dataclass
class HeldOut:
    name: str
    a: tuple[float, float, float]  # model-space point (unscaled)
    b: tuple[float, float, float]
    tape_m: float


@dataclass
class ScaleReport:
    scale: float
    source: str
    markers: list[dict] = field(default_factory=list)
    spread_rel: float | None = None
    held_out: list[dict] = field(default_factory=list)
    max_abs_error: float | None = None
    verdict: str = "unverified"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def build_report(markers: list[MarkerScale], held_out: list[HeldOut], scale_override: float | None = None,
                 source: str = "markers") -> ScaleReport:
    if scale_override is not None:
        scale = scale_override
    elif markers:
        scale = float(np.median([m.scale for m in markers]))
    else:
        raise ValueError("need marker scales or a scale_override (e.g. 1.0 for metric ARKit poses)")
    spread = None
    if len(markers) >= 2:
        s = np.array([m.scale for m in markers])
        spread = float((s.max() - s.min()) / np.median(s))
    rep = ScaleReport(scale=scale, source=source, markers=[asdict(m) for m in markers], spread_rel=spread)
    errs = []
    for h in held_out:
        d = float(np.linalg.norm(np.subtract(h.b, h.a))) * scale
        e = (d - h.tape_m) / h.tape_m
        errs.append(abs(e))
        rep.held_out.append({"name": h.name, "model_m": round(d, 4), "tape_m": h.tape_m, "error_pct": round(100 * e, 3)})
    if errs:
        rep.max_abs_error = float(max(errs))
        rep.verdict = "pass" if rep.max_abs_error <= TARGET else ("marginal" if rep.max_abs_error <= KILL else "fail")
    return rep


# ---- COLMAP text model ----------------------------------------------------------

def qvec_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_colmap_text(sparse_dir: str | Path) -> dict[str, Camera]:
    """Read cameras.txt + images.txt (PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, OPENCV: distortion ignored)."""
    d = Path(sparse_dir)
    Ks = {}
    for line in (d / "cameras.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cid, model, params = int(p[0]), p[1], list(map(float, p[4:]))
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            fx, fy, cx, cy = params[:4]
        Ks[cid] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    cams = {}
    lines = [l for l in (d / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    for line in lines[::2]:
        p = line.split()
        q = list(map(float, p[1:5]))
        t = np.array(list(map(float, p[5:8])))
        cams[p[9]] = Camera(K=Ks[int(p[8])], R=qvec_to_R(q), t=t)
    return cams


# ---- reference object (no printed markers) ------------------------------------------

@dataclass
class ReferenceScale:
    """Scale from an object of known size seen in the capture (e.g. a box 1 ft wide).

    Weaker than markers: one object, a footprint fitted to sparse points, so expect a
    few percent of error. Held-out tape distances still decide whether it is good enough.
    """

    known_m: float
    measured_model: float  # the matching side in model units
    scale: float  # meters per model unit
    long_side_model: float
    height_model: float
    n_points: int
    footprint_model: list[list[float]] = field(default_factory=list)


def points_on_reference(rec, frames_dir: str | Path, mask_fn, min_votes: int = 2) -> np.ndarray:
    """3D points (model frame) whose observations fall inside `mask_fn(image_bgr) -> bool HxW` in >= min_votes frames."""
    import cv2

    votes: dict[int, int] = {}
    seen: dict[int, int] = {}
    for img in rec.images.values():
        bgr = cv2.imread(str(Path(frames_dir) / img.name))
        if bgr is None:
            continue
        m = mask_fn(bgr)
        if not m.any():
            continue
        h, w = m.shape
        for p2 in img.points2D:
            if not p2.has_point3D():
                continue
            x, y = int(round(p2.xy[0])), int(round(p2.xy[1]))
            pid = int(p2.point3D_id)
            seen[pid] = seen.get(pid, 0) + 1
            if 0 <= x < w and 0 <= y < h and m[y, x]:
                votes[pid] = votes.get(pid, 0) + 1
    ids = [pid for pid, v in votes.items() if v >= min_votes and v >= 0.5 * seen[pid]]
    return np.array([rec.points3D[pid].xyz for pid in ids]) if ids else np.zeros((0, 3))


def largest_cluster(pts: np.ndarray, radius: float) -> np.ndarray:
    """Keep the biggest connected cluster (points within `radius` of each other)."""
    if len(pts) < 3:
        return pts
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    lab = -np.ones(len(pts), int)
    cur = 0
    for i in range(len(pts)):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = cur
        while stack:
            j = stack.pop()
            for k in np.nonzero((d[j] < radius) & (lab < 0))[0]:
                lab[k] = cur
                stack.append(k)
        cur += 1
    big = np.bincount(lab).argmax()
    return pts[lab == big]


def scale_from_reference(points_zup: np.ndarray, known_m: float, side: str = "short", trim: float = 5.0) -> ReferenceScale:
    """points_zup: the object's points in a leveled frame (floor at z = 0, z up), any units."""
    from ..scene.geometry import convex_hull, min_area_rect

    p = np.asarray(points_zup, float)
    lo, hi = np.percentile(p, trim, axis=0), np.percentile(p, 100 - trim, axis=0)
    p = p[np.all((p >= lo) & (p <= hi), axis=1)]
    center, w, d, yaw = min_area_rect(p[:, :2])
    meas = d if side == "short" else w
    s = known_m / meas
    hull = convex_hull(p[:, :2])
    return ReferenceScale(known_m=known_m, measured_model=float(meas), scale=float(s), long_side_model=float(w),
                          height_model=float(p[:, 2].max()), n_points=len(p), footprint_model=hull.tolist())


# ---- reference box from clicked rim corners ----------------------------------------------

CORNERS = ("FL", "FR", "BR", "BL")  # front-left, front-right, back-right, back-left, going around the rim
_SIGNS = {"FL": (-1, -1), "FR": (1, -1), "BR": (1, 1), "BL": (-1, 1)}


@dataclass
class LevelRect:
    """A horizontal rectangle (e.g. a box's top rim) fitted to clicked corners, in the leveled frame."""

    center: tuple[float, float]
    height: float  # above the floor, model units
    yaw_deg: float
    long_side: float  # FL->FR, model units
    short_side: float  # FR->BR, model units
    rms_px: float
    short_side_rel_sigma: float  # 1-sigma relative uncertainty of short_side
    n_obs: int

    def scale_for(self, known_m: float, side: str = "short") -> float:
        return known_m / (self.short_side if side == "short" else self.long_side)


def _rect_corners(q: np.ndarray) -> dict[str, np.ndarray]:
    cx, cy, h, yaw, a, b = q
    c, s = np.cos(yaw), np.sin(yaw)
    return {k: np.array([cx + c * sx * a / 2 - s * sy * b / 2, cy + s * sx * a / 2 + c * sy * b / 2, h])
            for k, (sx, sy) in _SIGNS.items()}


def fit_level_rectangle(cams: dict[str, Camera], clicks: dict[str, dict[str, tuple[float, float]]],
                        level_R: np.ndarray, level_t: np.ndarray, iters: int = 50, huber_px: float = 3.0) -> LevelRect:
    """Fit a level rectangle to corner clicks seen from posed cameras.

    clicks: frame -> {corner name (FL/FR/BR/BL) -> undistorted pixel}. Each corner should be
    clicked in 2+ frames, and FL->FR must be a long side. level_R, level_t map model
    coordinates to the leveled frame (floor z = 0, z up): p_level = level_R @ p + level_t.
    Assuming the rim is level and square-cornered removes most of the depth error that
    short-baseline triangulation of single corners suffers from.
    """
    obs = [(cams[f], k, np.asarray(uv, float)) for f, cs in clicks.items() if f in cams for k, uv in cs.items()]
    if len(obs) < 6:
        raise ValueError("need at least 6 corner clicks")
    Rinv = level_R.T

    def residuals(q):
        cs = _rect_corners(q)
        r = []
        for cam, k, uv in obs:
            X = Rinv @ (cs[k] - level_t)
            h = cam.P @ np.r_[X, 1.0]
            r.extend(h[:2] / h[2] - uv)
        return np.asarray(r)

    # init: triangulate each corner that has 2+ views, then read the rectangle off them
    pts = {}
    for k in CORNERS:
        o = [(c, uv) for c, kk, uv in obs if kk == k]
        if len(o) >= 2:
            pts[k] = level_R @ triangulate([c for c, _ in o], [uv for _, uv in o]) + level_t
    if not {"FL", "FR"} <= set(pts) or len(pts) < 3:
        raise ValueError("need FL, FR and one more corner each clicked in 2+ frames")
    ctr = np.mean(list(pts.values()), axis=0)
    v = pts["FR"] - pts["FL"]
    a0 = np.linalg.norm(v[:2])
    other = pts.get("BR", pts.get("BL"))
    b0 = np.linalg.norm((other - (pts["FR"] if "BR" in pts else pts["FL"]))[:2])
    q = np.array([ctr[0], ctr[1], ctr[2], np.arctan2(v[1], v[0]), a0, b0])
    # robust Gauss-Newton (IRLS with Huber weights), numeric Jacobian
    for _ in range(iters):
        r = residuals(q)
        J = np.empty((len(r), 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-6 * max(1.0, abs(q[j]))
            J[:, j] = (residuals(q + dq) - r) / dq[j]
        e = np.linalg.norm(r.reshape(-1, 2), axis=1)
        w = np.repeat(np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-9)), 2)
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-9 * np.eye(6), -J.T @ (w * r), rcond=None)[0]
        q = q + step
        if np.linalg.norm(step) < 1e-10 * (1 + np.linalg.norm(q)):
            break
    r = residuals(q)
    e = np.linalg.norm(r.reshape(-1, 2), axis=1)
    sigma = float(np.median(e)) or 1e-3
    cov = np.linalg.pinv(J.T @ J) * sigma ** 2
    if q[4] < 0:
        q[4] = -q[4]
    if q[5] < 0:
        q[5] = -q[5]
    return LevelRect(center=(float(q[0]), float(q[1])), height=float(q[2]), yaw_deg=float(np.degrees(q[3])),
                     long_side=float(q[4]), short_side=float(q[5]), rms_px=float(np.sqrt(np.mean(e ** 2))),
                     short_side_rel_sigma=float(np.sqrt(max(cov[5, 5], 0)) / abs(q[5])), n_obs=len(obs))
