"""One walkthrough video to one solve of the whole home, with a learned model on a Colab GPU.

    frames  -> the scan's own sharp-frame picking (3 fps)
    rough   -> one pass of the model over the whole walk: rooms in the right places, small
               rooms bent (Depth Anything 3 takes ~900 frames in one pass on a 96 GB GPU)
    windows -> the room each frame was taken in, from the rough pass; one window per room
               stay, the stretches the rough pass can't sort out together (plan_windows)
    rooms   -> one pass per window, all at once (concurrent Colab sessions); with variants > 1,
               each window is solved again over slightly grown stretches and the solve the others
               agree with most is kept (consensus): now and then a run folds a room
    chain   -> the window solves joined through the frames they share (join_solves)
    scale   -> the model's metres, checked against the doorways and a known box (refscale)

The result is a learned_recon.py-style folder for `mise scan --poses`. Runs go through
scripts/colab/run_learned_recon.py (Colab CLI, sessions named mise-<tag>-...), so this needs a
source checkout and a signed-in `colab` CLI; nothing runs on this machine's GPUs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .learned import camera_centers, consensus, join, plan_windows, variant_bounds

LAUNCHER = Path(__file__).resolve().parents[3] / "scripts" / "colab" / "run_learned_recon.py"


def pick_frames(video: Path, frames_dir: Path, fps: float = 3.0) -> list[dict]:
    """The scan's sharp frames (frames_dir/frames/*.jpg + frames.json); reused when already there."""
    from .scan import ScanEvents, ScanOptions, _frames

    frames_dir = Path(frames_dir)
    if (frames_dir / "frames.json").exists():
        return json.loads((frames_dir / "frames.json").read_text())
    frames_dir.mkdir(parents=True, exist_ok=True)
    return _frames(Path(video), frames_dir, ScanOptions(scene_id="frames", fps=fps), ScanEvents())


def colab_job(name: str, frames_dir: Path, out: Path, log: Path, model: str, gpu: str,
              t0: float | None = None, t1: float | None = None, chunk: int = 900, wait_min: float = 240) -> subprocess.Popen:
    """Start one run_learned_recon.py run (it stops its VM when done or failing)."""
    args = [sys.executable, str(LAUNCHER), name, "--frames", str(frames_dir), "--out", str(out), "--model", model,
            "--gpu", gpu, "--wait", str(wait_min), "--chunk", str(chunk)]
    args += [f"--t0={t0}"] if t0 is not None else []
    args += [f"--t1={t1}"] if t1 is not None else []
    log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(args, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)


def _finish(jobs: dict[str, tuple[subprocess.Popen, Path, Path]], say) -> None:
    while jobs:
        for name, (p, out, log) in list(jobs.items()):
            if p.poll() is None:
                continue
            del jobs[name]
            if p.returncode != 0 or not (out / "meta.json").exists():
                tail = log.read_text()[-1500:] if log.exists() else ""
                raise RuntimeError(f"{name} failed (exit {p.returncode}); log {log}:\n{tail}")
            say(f"{name}: done")
        time.sleep(15)


def frame_rooms(solve: Path) -> tuple[np.ndarray, np.ndarray]:
    """Time and room label of every frame of a solve, from rooms traced on that solve."""
    from ..eval.floorplan import load_solve
    from . import room_outline
    from .scan import ScanOptions, _scale_and_level

    recs, meta, _ = load_solve(Path(solve))
    L, _ = _scale_and_level(recs[0], ScanOptions(scene_id="rooms", scale=1.0 if meta.get("metric") else None))
    home = room_outline.segment_home(L.cams, L.sights, L.points, walk=L.cams[np.argsort(L.image_names)])
    times = {k["name"]: k["t"] for k in json.loads((Path(solve) / "frames.json").read_text())}
    lab, o = home.labels, np.asarray(home.origin)
    ys, xs = np.nonzero(lab >= 0)
    rooms = []
    for x, y in np.floor((L.cams[:, :2] - o) / home.cell).astype(int):
        if 0 <= y < lab.shape[0] and 0 <= x < lab.shape[1] and lab[y, x] >= 0:
            rooms.append(int(lab[y, x]))
        else:  # a camera standing in a doorway or over furniture: the nearest room
            k = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2)) if len(ys) else 0
            rooms.append(int(lab[ys[k], xs[k]]) if len(ys) else 0)
    return np.array([times[n] for n in L.image_names]), np.array(rooms)


def solve_home(video: Path, work: Path, tag: str, model: str = "da3_nested", gpu: str = "G4,H100",
               fps: float = 3.0, variants: int = 3, say=print) -> Path:
    """Video -> work/learned/home (a joined solve for `mise scan --poses`). Steps already done
    (their outputs in `work`) are reused, so a failed run can be started again."""
    work = Path(work)
    frames_dir, learned = work / "picked", work / "learned"
    kept = pick_frames(video, frames_dir, fps)
    say(f"{len(kept)} frames picked")
    rough = learned / "rough"
    if not (rough / "meta.json").exists():
        say(f"rough pass over the whole walk ({model}, Colab {gpu.split(',')[0]})")
        _finish({"rough": (colab_job(f"{tag}-rough", frames_dir, rough, learned / "rough.log", model, gpu, chunk=900),
                           rough, learned / "rough.log")}, say)
    times, rooms = frame_rooms(rough)
    windows = plan_windows(times, rooms)
    say("windows: " + ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in windows))
    (learned / "windows.json").write_text(json.dumps(windows))
    t_end = float(times.max())
    jobs, runs = {}, []
    for i, (a, b) in enumerate(windows):
        runs.append([])
        for v, (va, vb) in enumerate(variant_bounds(a, b, max(1, variants), t_end)):
            key = f"w{i}" if v == 0 else f"w{i}v{v}"
            out = learned / key
            runs[-1].append(out)
            if (out / "meta.json").exists():
                continue
            n = int(np.sum((times >= va) & (times <= vb)))
            g = gpu if n > 400 else gpu + ",A100"  # 40 GB holds about 400 frames of Depth Anything 3
            jobs[key] = (colab_job(f"{tag}-{key}", frames_dir, out, learned / f"{key}.log", model, g, va, vb, chunk=900),
                         out, learned / f"{key}.log")
    if jobs:
        say(f"solving {len(jobs)} runs at once ({len(windows)} windows x {max(1, variants)})")
        _finish(jobs, say)
    outs, picks = [], []
    for i, rs in enumerate(runs):
        if len(rs) > 1:
            best, med = consensus([camera_centers(r) for r in rs])
            say(f"w{i}: kept {rs[best].name} (runs disagree by " + ", ".join(f"{100 * m:.1f}" for m in med) + " cm)")
        else:
            best, med = 0, [0.0]
        outs.append(rs[best])
        picks.append({"window": windows[i], "runs": [r.name for r in rs], "kept": rs[best].name,
                      "disagreement_cm": [round(100 * m, 1) for m in med]})
    (learned / "consensus.json").write_text(json.dumps(picks, indent=1))
    home = learned / "home"
    meta = join(outs, home) if len(outs) > 1 else None
    if meta is None:  # one window: it is the home
        import shutil

        if home.exists():
            shutil.rmtree(home)
        shutil.copytree(outs[0], home)
    else:
        for r in meta["joins"][1:]:
            say(f"joined {Path(r['solve']).name}: {r['shared']} frames shared, {r['median_cm']} cm apart, "
                f"cameras {r['camera_gap_cm']} cm / {r['camera_turn_deg']} deg")
    return home
