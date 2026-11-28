"""Scanned objects made whole by an image-to-3D model (Idol), from the walkthrough's own frames.

A scanned object's splat only has the sides the walk saw, smeared where it saw them from far or
few. Idol makes a complete object from one image. For each object this picks the frame that shows
it best (large, wholly in frame, labelled as that thing rather than what stands in front), crops
it, and after Idol has run (scripts/colab/idol_run.py) places the result: Idol predicts the
camera it was seen from, the frame's own camera says where that was in the room, so the two give
the object's heading; one uniform scale fits it to the scanned box, and it sits where the scan
put it. The anisotropy of that fit (largest / smallest per-axis ratio) says how well it agrees.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ply
from .objects import KINDS


@dataclass
class View:
    object_id: str
    frame: str
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in the frame's pixels
    score: float
    seen: float  # how much of the box's outline the labels call this kind of thing
    camera_to_object: tuple[float, float, float]  # scene frame


def _kind_of(object_id: str) -> str | None:
    k = object_id.removeprefix("scan_").rsplit("_", 1)[0]
    return k if k in KINDS else None


def _corners(o) -> np.ndarray:
    e = o.extent()
    c, s = math.cos(math.radians(o.pose.yaw)), math.sin(math.radians(o.pose.yaw))
    pts = []
    for a in (-e.w / 2, e.w / 2):
        for b in (-e.d / 2, e.d / 2):
            for z in (0.0, e.h):
                pts.append((o.pose.x + a * c - b * s, o.pose.y + a * s + b * c, o.pose.z + z))
    return np.array(pts)


def best_view(o, cameras: list[dict], labels: dict[str, np.ndarray], classes: list[str],
              margin: float = 0.15, min_frac: float = 0.12) -> View | None:
    """The frame that shows object `o` best, and the square crop around it."""
    import cv2

    kind = _kind_of(o.id)
    if kind is None:
        return None
    ok_cls = np.array([c.strip() in KINDS[kind][0] for c in classes] + [False])
    X = _corners(o)
    center = X.mean(0)
    best = None
    for cam in cameras:
        name = cam["name"]
        if name not in labels:
            continue
        w2c, K = np.array(cam["w2c"], float), np.array(cam["K"], float)
        W, H = cam["size"]
        Pc = X @ w2c[:3, :3].T + w2c[:3, 3]
        if (Pc[:, 2] < 0.3).any():
            continue
        uv = Pc[:, :2] / Pc[:, 2:] * np.diag(K)[:2] + K[:2, 2]
        lo, hi = uv.min(0), uv.max(0)
        if lo[0] < 4 or lo[1] < 4 or hi[0] > W - 4 or hi[1] > H - 4:  # wholly in frame
            continue
        frac = float((hi - lo).max() / min(W, H))
        if frac < min_frac:
            continue
        lab = labels[name]
        lh, lw = lab.shape
        hull = cv2.convexHull((uv * [lw / W, lh / H]).astype(np.int32))
        mask = np.zeros((lh, lw), np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        inside = lab[mask.astype(bool)]
        if inside.size < 30:
            continue
        seen = float(ok_cls[np.minimum(inside, len(classes))].mean())
        C = -w2c[:3, :3].T @ w2c[:3, 3]
        v = center - C
        elev = math.degrees(math.atan2(-v[2], math.hypot(v[0], v[1])))  # camera above the object: positive
        tilt = 1.0 if -5 <= elev <= 45 else 0.6
        score = seen ** 2 * min(frac / 0.45, 1.0) * tilt
        if best is None or score > best.score:
            side = (hi - lo).max() * (1 + 2 * margin)
            cx, cy = (lo + hi) / 2
            side = min(side, W, H)
            x0 = int(np.clip(cx - side / 2, 0, W - side))
            y0 = int(np.clip(cy - side / 2, 0, H - side))
            best = View(o.id, name, (x0, y0, x0 + int(side), y0 + int(side)), score, seen, tuple(v.tolist()))
    return best


def hint_mask(o, cam: dict, labels: dict[str, np.ndarray], classes: list[str], grow_px: int = 3) -> np.ndarray:
    """Where the frame shows this object, at the frame's size (0/255): pixels labelled as its kind
    inside the outline of its box, grown a little. The labels are coarse; Idol's own cut-out is kept
    only where it overlaps this, so a desk behind the chair isn't made too."""
    import cv2

    kind = _kind_of(o.id)
    ok_cls = np.array([c.strip() in KINDS[kind][0] for c in classes] + [False])
    w2c, K = np.array(cam["w2c"], float), np.array(cam["K"], float)
    W, H = cam["size"]
    Pc = _corners(o) @ w2c[:3, :3].T + w2c[:3, 3]
    uv = Pc[:, :2] / Pc[:, 2:] * np.diag(K)[:2] + K[:2, 2]
    lab = labels[cam["name"]]
    lh, lw = lab.shape
    hull = np.zeros((lh, lw), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull((uv * [lw / W, lh / H]).astype(np.int32)), 1)
    m = (ok_cls[np.minimum(lab, len(classes))] & hull.astype(bool)).astype(np.uint8)
    m = cv2.dilate(m, np.ones((2 * grow_px + 1,) * 2, np.uint8))
    return cv2.resize(m * 255, (W, H), interpolation=cv2.INTER_LINEAR)


def write_crops(views: list[View], frames_dir: Path, out_dir: Path, px: int = 1024, masks: dict | None = None) -> Path:
    """The crops as images plus images.json, ready for idol_run.py (real photos); with `masks`
    (object id -> hint_mask), each crop's hint goes beside it as <id>_hint.png."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in views:
        img = cv2.imread(str(Path(frames_dir) / v.frame))
        x0, y0, x1, y1 = v.crop
        interp = cv2.INTER_CUBIC if x1 - x0 < px else cv2.INTER_AREA
        crop = cv2.resize(img[y0:y1, x0:x1], (px, px), interpolation=interp)
        cv2.imwrite(str(out_dir / f"{v.object_id}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        job = {"name": f"{v.object_id}.jpg", "mode": "real"}
        if masks and v.object_id in masks:
            m = masks[v.object_id]
            m = cv2.resize(m, (img.shape[1], img.shape[0]))[y0:y1, x0:x1]
            cv2.imwrite(str(out_dir / f"{v.object_id}_hint.png"), cv2.resize(m, (px, px), interpolation=cv2.INTER_LINEAR))
            job["hint"] = f"{v.object_id}_hint.png"
        jobs.append(job)
    (out_dir / "images.json").write_text(json.dumps(jobs))
    (out_dir / "views.json").write_text(json.dumps([v.__dict__ for v in views], indent=1))
    return out_dir


def place(o, view: View, idol_ply: Path, idol_camera: dict, opacity_min: float = 0.3) -> tuple[ply.Splat, dict]:
    """Idol's object in the scanned object's own frame (footprint centered, base at 0, +x along its
    width), headed as the frame saw it and scaled to the scanned box. Returns it and the fit."""
    from ..products.idol import idol_camera_dir

    s = ply.read_ply(idol_ply)
    to_idol = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # undo save_ply: back to Idol's own z-up frame
    s = ply.transform(s, R=to_idol, drop_sh_rest=True)
    cam_a = to_idol @ np.array(idol_camera_dir(idol_camera))  # camera -> object, in Idol's frame
    v = np.array(view.camera_to_object)
    turn = math.atan2(v[1], v[0]) - math.atan2(cam_a[1], cam_a[0]) - math.radians(o.pose.yaw)
    s = ply.transform(s, R=ply.rot_z(math.degrees(turn)))  # now in the object's frame: +x its width
    a = 1 / (1 + np.exp(-s.opacity)) if s.opacity is not None else np.ones(len(s))
    P = s.xyz[a > opacity_min].astype(np.float64)
    lo, hi = np.percentile(P, 1, 0), np.percentile(P, 99, 0)
    meas = hi - lo
    e = o.extent()
    want = np.array([e.w, e.d, e.h])
    ratios = want / np.maximum(meas, 1e-6)
    scale = float(np.median(ratios))
    s = ply.transform(s, t=-np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]))
    s = ply.transform(s, scale=scale)
    fit = {"scale": round(scale, 4), "anisotropy": round(float(ratios.max() / ratios.min()), 3),
           "measured_m": [round(float(x), 3) for x in meas * scale], "scanned_m": [round(float(x), 3) for x in want],
           "turn_deg": round(math.degrees(turn) % 360, 1)}
    return s, fit


AGREE_MIN = 0.3  # Idol's cut-out must cover this much of the label hint, else the "object" is likely a mislabel


def adopt(scene, scene_dir: Path, run_dir: Path, views: list[View], agreement: dict[str, float | None],
          skip: set[str] = frozenset()) -> dict[str, dict]:
    """Swap each scanned object's splat for Idol's, placed by `place`, where Idol's cut-out agreed
    with the labels. The scanned splat stays as objects/<id>.ply; Idol's is objects/<id>-idol.ply.
    Returns per object: the fit and whether it was adopted (and why not)."""
    objs = {o.id: o for o in scene.objects}
    out = {}
    for v in views:
        o = objs.get(v.object_id)
        ply_path, cam = run_dir / f"{v.object_id}.ply", run_dir / f"{v.object_id}_camera.json"
        if o is None or not ply_path.exists() or not cam.exists():
            continue
        agree = agreement.get(v.object_id)
        if v.object_id in skip:
            out[v.object_id] = {"adopted": False, "why": "skipped"}
            continue
        if agree is not None and agree < AGREE_MIN:
            out[v.object_id] = {"adopted": False, "why": f"cut-out covers {agree:.0%} of the labels: likely not one object"}
            continue
        s, fit = place(o, v, ply_path, json.loads(cam.read_text()))
        rel = f"objects/{o.id}-idol.ply"
        ply.write_ply(scene_dir / rel, s)
        o.asset = rel
        out[v.object_id] = {"adopted": True, **fit}
    return out
