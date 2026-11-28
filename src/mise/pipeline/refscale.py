"""Metric scale from what any home has in it, several references at once.

No markers, no tape. A metric model's own scale is good to about 4%; the references here each
give another estimate with its own error bar, and the scale used is their inverse-variance mean:

    model    the solve's own metres (learned depth): factor 1, sigma ~4% (first tour: needed
             0.99-1.03 on joined homes, up to 1.08 on single windows)
    doors    the underside of each doorway's head: a standard 80" (2032 mm) interior door, with
             its gaps, hangs in an opening about 2.06 m tall. Each doorway the layout finds is
             measured from the floor beside it to where the wall above it starts. One door is
             +-4%; the median of several is better (first tour, 4 joined homes: -2.0..+1.8%)
    box      an object of known size whose rim corners were clicked in a few frames
             (data/floorplan/box_clicks.json; see fit_level_rectangle); its own error bar from
             the fit, 5-9% for a 1 ft box

The floor plan stays out of this: it is the check (python -m mise.eval.floorplan), not an input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DOOR_OPENING_M = 2.06
MODEL_SIGMA = 0.04
DOOR_SIGMA_ONE = 0.045


def lintel_height(points: np.ndarray, center, axis: str, width: float) -> float | None:
    """Height of the underside of a doorway's head above the floor beside it (model units)."""
    c = np.asarray(center, float)
    al, ac = (0, 1) if axis == "x" else (1, 0)  # along the opening, across the wall
    half = width / 2 - 0.08
    if half < 0.12:
        return None
    near = (np.abs(points[:, al] - c[al]) < half) & (np.abs(points[:, ac] - c[ac]) < 0.12)
    floor = points[near & (np.abs(points[:, 2]) < 0.15), 2]
    f0 = float(np.median(floor)) if len(floor) > 20 else 0.0
    z = points[near & (points[:, 2] > 1.7) & (points[:, 2] < 2.6), 2] - f0
    if len(z) < 80:
        return None
    h, e = np.histogram(z, bins=np.arange(1.7, 2.6, 0.02))
    run = np.convolve(h, np.ones(3), "same")
    top = np.median(run[run > 0][-10:]) if (run > 0).sum() >= 10 else run.max()
    k = np.nonzero(run >= 0.5 * top)[0]  # the lowest height from which the wall above is filled in
    return float(e[k[0]] + 0.01) if len(k) else None


def door_estimate(points: np.ndarray, doorways) -> dict | None:
    hs = [h for d in doorways if (h := lintel_height(points, d.center, d.axis, d.width)) is not None]
    hs = [h for h in hs if 1.75 < h < 2.5]
    if len(hs) < 2:
        return None
    return {"factor": DOOR_OPENING_M / float(np.median(hs)), "sigma": max(0.02, DOOR_SIGMA_ONE / np.sqrt(len(hs))),
            "doors": len(hs), "lintels": [round(h, 3) for h in hs]}


def box_estimate(rec, L, box: dict) -> dict | None:
    """Scale from an object of known size: a level rectangle fitted to its clicked rim corners."""
    from . import scale as sc

    by_name = {im.name: im for im in rec.images.values()}
    cams, clicks = {}, {}
    for n, cs in box["clicks"].items():
        im = by_name.get(n)
        if im is None:
            continue
        cam = rec.cameras[im.camera_id]
        K = cam.calibration_matrix()
        cw = im.cam_from_world()
        cams[n] = sc.Camera(K=K, R=cw.rotation.matrix(), t=np.asarray(cw.translation))
        clicks[n] = {k: tuple((K @ np.r_[cam.cam_from_img(np.array([uv], float))[0], 1.0])[:2]) for k, uv in cs.items()}
    if sum(len(v) for v in clicks.values()) < 6:
        return None
    r = sc.fit_level_rectangle(cams, clicks, L.R, L.t / L.scale)  # model units, leveled
    side = r.short_side if box.get("known_side", "short").startswith("short") else r.long_side
    return {"factor": box["known_m"] / (side * L.scale), "sigma": float(r.short_side_rel_sigma), "frames": len(clicks),
            "rim_rms_px": round(r.rms_px, 1)}


def combine(estimates: dict[str, dict]) -> dict:
    """Inverse-variance mean of the factors (in log space, as scale errors multiply)."""
    use = {k: v for k, v in estimates.items() if v and np.isfinite(v["factor"]) and v["sigma"] > 0}
    w = np.array([1 / v["sigma"] ** 2 for v in use.values()])
    lf = np.array([np.log(v["factor"]) for v in use.values()])
    f = float(np.exp((w * lf).sum() / w.sum()))
    return {"factor": round(f, 4), "sigma_pct": round(100 / np.sqrt(w.sum()), 1),
            "from": {k: {**{kk: vv for kk, vv in v.items()}, "factor": round(v["factor"], 4),
                         "sigma_pct": round(100 * v["sigma"], 1)} for k, v in use.items()}}


def reference_scale(solve: Path, box: dict | None = None) -> dict:
    """The scale (factor on the solve's own units) from the model, the doors and the box."""
    from ..eval.floorplan import load_solve
    from . import room_outline
    from .scan import ScanOptions, _scale_and_level

    recs, meta, _ = load_solve(Path(solve))
    rec = recs[0]
    L, _ = _scale_and_level(rec, ScanOptions(scene_id="scale", scale=1.0))
    home = room_outline.segment_home(L.cams, L.sights, L.points, walk=L.cams[np.argsort(L.image_names)])
    est = {"model": {"factor": 1.0, "sigma": MODEL_SIGMA} if meta.get("metric") else None,
           "doors": door_estimate(L.points, home.doorways),
           "box": box_estimate(rec, L, box) if box else None}
    return combine(est)
