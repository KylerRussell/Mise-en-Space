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
