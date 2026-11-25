"""How true to the home is a camera solve? Score it against a known floor plan.

    python -m mise.eval.floorplan PLAN.json --room bedroom mapanything=data/scans/learned/bedroom \
        colmap=data/scans/colmap-bedroom

The plan is a JSON of room polygons in meters (see `load_plan`). Each solve is leveled and
scaled exactly as the scan pipeline would (metric if its meta.json says so, else from the
phone's usual height), then the plan's walls are fitted to the solve's wall points (heights
0.3-2.0 m) in 2D: robust ICP from the four 90-degree turns (leveling squares the walls to the
axes), and the best fit kept. Reported:

- scale: how much the solve must be stretched to match the plan (1.00 = sizes are right);
- off_walls_cm: median distance of wall points from the plan's walls, at the pipeline's scale
  and with the best scale (shape error alone);
- the room's spans (wall to wall) as the solve has them, next to the plan's;
- camera heights and the ceiling (for the scale cues).

Points seen through doorways land on other rooms' walls, which is why the whole plan is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_plan(path: Path) -> dict:
    """{"rooms": {name: {"polygon": [[x, y], ...]}}, "heights": {"ceiling_m": ...}} in meters."""
    return json.loads(Path(path).read_text())


def plan_segments(plan: dict) -> tuple[np.ndarray, list[str]]:
    segs, owner = [], []
    for name, r in plan["rooms"].items():
        poly = np.asarray(r["polygon"], float)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            segs.append([a, b])
            owner.append(name)
    return np.asarray(segs), owner


def nearest_on_segments(P: np.ndarray, segs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each point, the closest point on any segment, its distance and the segment's index."""
    a, b = segs[:, 0], segs[:, 1]
    ab = b - a
    t = np.clip(np.einsum("nmk,mk->nm", P[:, None, :] - a[None], ab) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    C = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(P[:, None, :] - C, axis=2)
    k = d.argmin(1)
    i = np.arange(len(P))
    return C[i, k], d[i, k], k


def _similarity_2d(X: np.ndarray, Y: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimizing |s R x + t - y| (Umeyama), rotation only (no mirror)."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()) if with_scale else 1.0
    return s, R, ym - s * R @ xm


def fit_plan(P: np.ndarray, segs: np.ndarray, R0: np.ndarray, t0: np.ndarray, s0: float = 1.0,
             with_scale: bool = False, iters: int = 60) -> dict:
    """Trimmed ICP of 2D points onto the plan's walls. Starts from x -> s0 R0 x + t0."""
    s, R, t = s0, R0, t0
    for it in range(iters):
        Q = s * P @ R.T + t
        C, d, _ = nearest_on_segments(Q, segs)
        thr = max(0.15, 1.0 * (1 - it / (0.6 * iters)))  # wide at first, then only points on the walls
        keep = d < max(thr, np.percentile(d, 30))
        if keep.sum() < 20:
            break
        ds, dR, dt = _similarity_2d(Q[keep], C[keep], with_scale)
        s, R, t = ds * s, dR @ R, ds * dR @ t + dt
    Q = s * P @ R.T + t
    _, d, k = nearest_on_segments(Q, segs)
    return {"s": s, "R": R, "t": t, "d": d, "seg": k, "trimmed": float(np.sort(d)[: int(0.7 * len(d))].mean())}


def _rot(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s], [s, c]])


def best_fit(P: np.ndarray, cams: np.ndarray, plan: dict, room: str, with_scale: bool = False) -> dict:
    segs, _ = plan_segments(plan)
    target = np.asarray(plan["rooms"][room]["polygon"], float).mean(0)
    best = None
    for yaw in (0, 90, 180, 270):
        R0 = _rot(yaw)
        for dx in (-1.0, 0.0, 1.0):
            for dy in (-1.0, 0.0, 1.0):
                t0 = target + np.array([dx, dy]) - R0 @ cams.mean(0)
                f = fit_plan(P, segs, R0, t0)
                if best is None or f["trimmed"] < best["trimmed"]:
                    best = f
    if with_scale:
        best = fit_plan(P, segs, best["R"], best["t"], best["s"], with_scale=True, iters=40)
    return best


def _peak(v: np.ndarray, bin_m: float = 0.02) -> float:
    """Densest spot of 1D values (the wall face among stray points)."""
    edges = np.arange(v.min(), v.max() + 2 * bin_m, bin_m)
    h, _ = np.histogram(v, edges)
    h = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
    k = int(h.argmax())
    near = v[np.abs(v - (edges[k] + bin_m / 2)) < 2.5 * bin_m]
    return float(np.median(near))


def room_spans(Q: np.ndarray, plan: dict, room: str, band: float = 0.6, min_pts: int = 15) -> dict:
    """Wall-to-wall spans of `room` as the (aligned) points have them, next to the plan's.

    Each of the room's walls is placed at the densest layer of points within `band` of it (and
    along its length), so a solve that is too small or large still finds its own walls; spans
    are between the westmost and eastmost walls, and from the southmost wall to each other
    wall facing north-south.
    """
    poly = np.asarray(plan["rooms"][room]["polygon"], float)
    walls = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        axis = 0 if abs(a[0] - b[0]) < 1e-6 else 1  # 0: wall runs north-south (fixed x)
        along = 1 - axis
        lo, hi = sorted([a[along], b[along]])
        sel = (np.abs(Q[:, axis] - a[axis]) < band) & (Q[:, along] > lo + 0.15) & (Q[:, along] < hi - 0.15)
        walls.append({"axis": axis, "plan": float(a[axis]), "n": int(sel.sum()),
                      "seen": _peak(Q[sel, axis]) if sel.sum() >= min_pts else None})
    out = {}
    xs = sorted((w for w in walls if w["axis"] == 0), key=lambda w: w["plan"])
    if xs[0]["seen"] is not None and xs[-1]["seen"] is not None:
        out["east-west"] = {"plan": round(xs[-1]["plan"] - xs[0]["plan"], 3),
                            "solve": round(xs[-1]["seen"] - xs[0]["seen"], 3)}
    ys = sorted((w for w in walls if w["axis"] == 1), key=lambda w: w["plan"])
    for w in ys[1:]:
        if ys[0]["seen"] is not None and w["seen"] is not None and w["plan"] - ys[0]["plan"] > 0.5:
            out[f"north-south to y={w['plan']:.2f}"] = {"plan": round(w["plan"] - ys[0]["plan"], 3),
                                                        "solve": round(w["seen"] - ys[0]["seen"], 3)}
    return out


def load_solve(path: Path):
    """A learned-model output (meta.json + sparse/0) or a scan work folder (largest piece)."""
    import pycolmap

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    dirs = [d for d in sorted((path / "sparse").iterdir()) if d.is_dir()]
    recs = sorted((pycolmap.Reconstruction(str(d)) for d in dirs), key=lambda r: -r.num_reg_images())
    frames = len(json.loads((path / "frames.json").read_text())) if (path / "frames.json").exists() else None
    return recs, meta, frames


def score(path: Path, plan: dict, room: str) -> dict:
    from ..pipeline import room_outline
    from ..pipeline.scan import ScanOptions, _scale_and_level

    recs, meta, frames = load_solve(path)
    if not recs:
        return {"error": "no solve"}
    metric = bool(meta.get("metric"))
    # as the pipeline does: the largest piece that levels and keeps the cameras at hand height
    rec = L = None
    for r in recs:
        if r.num_reg_images() < 10:
            break
        try:
            Lr, how = _scale_and_level(r, ScanOptions(scene_id="eval", scale=1.0 if metric else None))
        except Exception:  # noqa: BLE001 - no floor found: not a usable piece
            continue
        if Lr.pose_sanity()["ok"]:
            rec, L = r, Lr
            break
    if rec is None:
        return {"frames": frames, "pieces": len(recs), "error": "no piece levels with sane camera heights"}
    P3 = L.points[(L.points[:, 2] > 0.3) & (L.points[:, 2] < 2.0)]
    rng = np.random.default_rng(0)
    P = P3[rng.choice(len(P3), min(len(P3), 6000), replace=False), :2] if len(P3) else P3[:, :2]
    out = {"frames": frames, "solved": rec.num_reg_images(), "pieces": len(recs), "scale_from": how,
           "cam_height_m": L.pose_sanity(), "ceiling_m": room_outline.outline_ceiling(L.points)}
    if len(P) < 50:
        out["error"] = f"only {len(P)} wall points"
        return out
    fixed = best_fit(P, L.cams[:, :2], plan, room)
    free = best_fit(P, L.cams[:, :2], plan, room, with_scale=True)
    Q = (fixed["s"] * P @ fixed["R"].T) + fixed["t"]
    out.update({
        "scale_needed": round(free["s"], 3),
        "off_walls_cm": {"as_scaled": round(100 * float(np.median(fixed["d"])), 1),
                         "best_scale": round(100 * float(np.median(free["d"])), 1)},
        "within_10cm": {"as_scaled": round(float((fixed["d"] < 0.1).mean()), 2),
                        "best_scale": round(float((free["d"] < 0.1).mean()), 2)},
        "spans": room_spans(Q, plan, room),
    })
    if out["ceiling_m"] is not None:
        out["ceiling_m"] = round(out["ceiling_m"], 2)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path)
    ap.add_argument("--room", required=True, help="the plan's room the stretch was filmed in")
    ap.add_argument("solves", nargs="+", help="LABEL=PATH (learned output folder or scan work folder)")
    ap.add_argument("--json", type=Path, help="also write the scores here")
    a = ap.parse_args(argv)
    plan = load_plan(a.plan)
    res = {}
    for item in a.solves:
        label, _, path = item.partition("=")
        res[label] = score(Path(path), plan, a.room)
        print(f"== {label}: {json.dumps(res[label], default=float)}", flush=True)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
