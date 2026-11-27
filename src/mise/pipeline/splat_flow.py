"""A home's splat from its learned scan: cameras refined, one splat per room visit, merged.

One splat over a whole tour is spread too thin to fit even its own frames (training PSNR ~19-23
however long it trains); one per room visit, on bundle-adjusted cameras, with the learned depth
maps as a guide and each camera learning a small correction, is sharp (held out 25-28 dB). Each
visit's window runs a few seconds into the rooms either side, for the views through the
doorways. The windows train side by side on Colab, then each room is taken from the window that
stood in it, and the empty room is made from the merged splat.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import ply
from .learned_flow import _finish, colab_job


def room_visits(times: np.ndarray, rooms: np.ndarray, min_frames: int = 30) -> list[tuple[int, float, float, int]]:
    """Consecutive stretches in one room, as (room, start, end, frames), in time order.

    Frames outside every room (doorways) are skipped, and stretches shorter than `min_frames`
    (a glance back through a door) are folded into what is around them.
    """
    o = np.argsort(times)
    runs: list[list] = []
    for t, r in zip(np.asarray(times)[o], np.asarray(rooms)[o]):
        if r < 0:
            continue
        if runs and runs[-1][0] == r:
            runs[-1][2], runs[-1][3] = t, runs[-1][3] + 1
        else:
            runs.append([int(r), float(t), float(t), 1])
    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if r[3] < min_frames]
        if not short:
            break
        del runs[min(short, key=lambda i: runs[i][3])]
        merged: list[list] = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2], merged[-1][3] = r[2], merged[-1][3] + r[3]
            else:
                merged.append(r)
        runs = merged
    return [(r[0], r[1], r[2], r[3]) for r in runs]


def _in_room(xy: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    import cv2

    out = np.full(len(xy), -1, int)
    for k, p in enumerate(polys):
        pc = np.asarray(p, np.float32)
        for i, q in enumerate(xy):
            if out[i] < 0 and cv2.pointPolygonTest(pc, (float(q[0]), float(q[1])), False) >= 0:
                out[i] = k
    return out


def _camera_times_in_scene(solve: Path, frames_dir: Path, scene_from: dict) -> tuple[np.ndarray, np.ndarray]:
    import pycolmap

    rec = pycolmap.Reconstruction(str(solve / "sparse" / "0"))
    t_of = {k["name"]: k["t"] for k in json.loads((frames_dir / "frames.json").read_text())}
    s, R, t = float(scene_from["scale"]), np.array(scene_from["R"], float), np.array(scene_from["t"], float)
    names = [im.name for im in rec.images.values() if im.name in t_of]
    c = np.array([rec.images[i].projection_center() for i in rec.images if rec.images[i].name in t_of])
    return np.array([t_of[n] for n in names]), (s * c @ R.T + t)


def splat_home(scene_dir: Path, work: Path, solve: Path, frames_dir: Path, scan_result: Path,
               semantic: Path | None = None, refined: Path | None = None, gpu: str = "A100,G4,L4,H100",
               margin: float = 5.0, tag: str = "home", say=print) -> dict:
    from ..scene.models import Scene
    from .empty_room import empty_scene, merge_splats
    from .learned import refine_poses

    run = work / "splat-run"
    run.mkdir(parents=True, exist_ok=True)
    scene_from = next(pc for pc in json.loads(scan_result.read_text())["pieces"] if pc.get("scene_from_solve"))
    scene_from = scene_from["scene_from_solve"]

    if refined is None:
        refined = run / "refined"
        if not (refined / "sparse" / "0" / "images.txt").exists():
            say("bundle-adjusting the cameras (CPU)")
            refine_poses(solve, frames_dir, run / "refine-work", refined, say=say)
        for f in ("depth.npz", "frames.json"):
            if (solve / f).exists() and not (refined / f).exists():
                shutil.copy2(solve / f, refined / f)

    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    polys = [np.array(sp.polygon, float) for sp in scene.room.spaces]
    times, centers = _camera_times_in_scene(refined, frames_dir, scene_from)
    visits = room_visits(times, _in_room(centers[:, :2], polys))
    t_end = float(times.max())
    say("room visits: " + ", ".join(f"{scene.room.spaces[r].name} {a:.0f}-{b:.0f} s ({n})" for r, a, b, n in visits))

    jobs, outs = {}, []
    for i, (r, a, b, n) in enumerate(visits):
        out = run / f"w{i}"
        outs.append(out)
        if (out / "meta.json").exists():
            continue
        steps = int(np.clip(60 * n, 10000, 20000))
        extra = ["--splat", str(refined), "--scene-from", str(scan_result), "--steps", str(steps)]
        jobs[f"w{i}"] = (colab_job(f"{tag}-splat{i}", frames_dir, out, run / f"w{i}.log", "", gpu,
                                   max(0.0, a - margin), min(t_end, b + margin), extra=extra), out, run / f"w{i}.log")
    if jobs:
        say(f"training {len(jobs)} room splats on Colab ({gpu})")
        _finish(jobs, say)
    scores = [json.loads((o / "meta.json").read_text()) for o in outs]

    parts = [(ply.read_ply(o / "splat.ply"), json.loads((o / "cameras.json").read_text())) for o in outs]
    merged, rep = merge_splats(parts, polys)
    md = run / "home"
    md.mkdir(exist_ok=True)
    ply.write_ply(md / "splat.ply", merged)
    (md / "cameras.json").write_text(json.dumps([c for _, cams in parts for c in cams]))
    (scene_dir / "splats").mkdir(exist_ok=True)
    shutil.copyfile(md / "splat.ply", scene_dir / "splats" / "home.ply")
    (scene_dir / "scene.json").write_text(scene.model_dump_json(indent=2))
    say(f"merged: {len(merged)} Gaussians; rooms from windows {rep['owner']}")

    empty = empty_scene(scene_dir, md, "empty.ply", semantic, frames=frames_dir,
                        depth=refined / "depth.npz" if (refined / "depth.npz").exists() else None,
                        depth_scale=float(scene_from["scale"]), say=say)
    from .objects import add_scanned_objects

    objs = add_scanned_objects(scene_dir, md)  # empty_scene labelled the merged splat
    # the room is the empty home; what stood in it comes back as the objects (splats/home.ply, the
    # home as filmed, stays beside it)
    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    scene.room.splat = "splats/empty.ply"
    (scene_dir / "scene.json").write_text(scene.model_dump_json(indent=2))
    say(f"{objs['objects']} objects ({objs['fixtures']} fixtures): " + ", ".join(f"{k} {v}" for k, v in objs["by_kind"].items()))
    report = {"objects": {k: v for k, v in objs.items() if k != "list"}, "windows": [{"room": scene.room.spaces[r].name, "t": [round(a, 1), round(b, 1)], "frames": n,
                           "psnr_held_out": m.get("psnr_held_out"), "psnr_train": m.get("psnr_train"),
                           "gaussians": m.get("gaussians")} for (r, a, b, n), m in zip(visits, scores)],
              "merge": rep, "empty": empty, "splat": "splats/empty.ply", "as_filmed": "splats/home.ply"}
    (run / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report
