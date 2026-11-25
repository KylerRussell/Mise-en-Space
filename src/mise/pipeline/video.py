"""Stage 1 on a CPU: video -> sharp frames -> capture report -> camera poses (pycolmap).

This is the no-GPU path for checking a capture before spending GPU time on it.
Splat training (Splatfacto) still needs CUDA; see external.py.

Needs the `poses` extra: pip install -e '.[poses]'  (pycolmap, opencv-python-headless)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def probe(video: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    s = d["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = s["width"], s["height"]
    if abs(rot) in (90, 270):
        w, h = h, w
    return {"width": w, "height": h, "fps": num / den, "duration_s": float(d["format"]["duration"]), "rotation": rot}


def sharpness(gray: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_sharp_frames(video: Path, out_dir: Path, target_fps: float = 1.5, oversample: int = 3,
                         width: int = 1600, t0: float | None = None, t1: float | None = None) -> list[dict]:
    """Decode at target_fps * oversample, keep the sharpest frame of each group of `oversample`.

    t0 / t1 (seconds) decode only that stretch; times in the result stay video times.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_all"
    tmp.mkdir(exist_ok=True)
    span = (["-ss", f"{t0:.3f}"] if t0 else []) + (["-t", f"{t1 - (t0 or 0):.3f}"] if t1 is not None else [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *span, "-i", str(video), "-vf",
                    f"fps={target_fps * oversample},scale={width}:-2", "-q:v", "2", str(tmp / "f_%05d.jpg")], check=True)
    cands = sorted(tmp.glob("f_*.jpg"))
    kept = []
    for k in range(0, len(cands), oversample):
        group = cands[k:k + oversample]
        scored = []
        for p in group:
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            scored.append((sharpness(cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1])))), p))
        s, best = max(scored)
        name = f"frame_{len(kept):04d}.jpg"
        best.rename(out_dir / name)
        kept.append({"name": name, "t": (t0 or 0) + (k + scored.index((s, best))) / (target_fps * oversample), "sharpness": s,
                     "group_min_sharpness": min(x[0] for x in scored)})
    for p in tmp.glob("*"):
        p.unlink()
    tmp.rmdir()
    return kept


@dataclass
class CaptureReport:
    """How a capture compares with the Stage 1 protocol (landscape, slow, 3 height passes, steady light)."""

    width: int
    height: int
    duration_s: float
    landscape: bool
    frames: int
    blurry_frames: int
    blur_threshold: float
    median_motion_px_per_s: float | None = None
    fast_segments_s: list[tuple[float, float]] = field(default_factory=list)
    brightness_range: tuple[float, float] | None = None
    height_passes: int | None = None
    camera_height_span_m: float | None = None
    notes: list[str] = field(default_factory=list)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))


def capture_report(info: dict, frames_dir: Path, kept: list[dict], blur_rel: float = 0.35) -> CaptureReport:
    """Blur, motion and exposure from the kept frames. Height passes are filled in after poses."""
    import cv2

    sharp = np.array([k["sharpness"] for k in kept])
    thr = float(np.median(sharp) * blur_rel)
    rep = CaptureReport(width=info["width"], height=info["height"], duration_s=info["duration_s"],
                        landscape=info["width"] >= info["height"], frames=len(kept),
                        blurry_frames=int((sharp < thr).sum()), blur_threshold=thr)
    # motion between consecutive kept frames (phase correlation on small grayscale)
    prev, speeds, bright = None, [], []
    for k in kept:
        g = cv2.imread(str(frames_dir / k["name"]), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (480, int(480 * g.shape[0] / g.shape[1]))).astype(np.float32)
        bright.append(float(g.mean()))
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            dt = max(k["t"] - prev_t, 1e-3)
            speeds.append((k["t"], float(np.hypot(dx, dy)) * (info["width"] / 480) / dt))
        prev, prev_t = g, k["t"]
    if speeds:
        v = np.array([s for _, s in speeds])
        rep.median_motion_px_per_s = float(np.median(v))
        fast = v > 4 * np.median(v)
        t = np.array([s[0] for s in speeds])
        seg = None
        for ti, f in zip(t, fast):
            if f and seg is None:
                seg = [ti, ti]
            elif f:
                seg[1] = ti
            elif seg is not None:
                rep.fast_segments_s.append((round(float(seg[0]), 1), round(float(seg[1]), 1)))
                seg = None
        if seg:
            rep.fast_segments_s.append((round(float(seg[0]), 1), round(float(seg[1]), 1)))
    rep.brightness_range = (round(min(bright), 1), round(max(bright), 1))
    if not rep.landscape:
        rep.notes.append("portrait capture: the protocol asks for landscape")
    if rep.blurry_frames > 0.1 * rep.frames:
        rep.notes.append(f"{rep.blurry_frames} of {rep.frames} kept frames are blurry; move more slowly")
    if rep.fast_segments_s:
        rep.notes.append(f"{len(rep.fast_segments_s)} stretches of fast motion (likely pans): {rep.fast_segments_s[:6]}")
    if rep.brightness_range[1] - rep.brightness_range[0] > 80:
        rep.notes.append("exposure varies a lot between frames (lights/blinds or auto-exposure)")
    return rep


def height_passes(centers_z: np.ndarray, bin_m: float = 0.15) -> tuple[int, float]:
    """Count distinct camera-height layers (peaks in a smoothed histogram). Expects metric, z-up centers."""
    z = np.asarray(centers_z)
    span = float(np.percentile(z, 95) - np.percentile(z, 5))
    if span < bin_m:
        return 1, span
    h, edges = np.histogram(z, bins=max(3, int(span / (bin_m / 3)) + 1))
    h = np.convolve(h, [1, 2, 1], mode="same")
    peaks = [i for i in range(len(h)) if h[i] > 0.15 * h.max() and h[i] >= h[max(i - 1, 0)] and h[i] >= h[min(i + 1, len(h) - 1)]]
    merged = []
    for p in peaks:
        if not merged or (edges[p] - edges[merged[-1]]) > bin_m:
            merged.append(p)
    return len(merged), span


def enhance_frames(src: Path, dst: Path, clip: float = 3.0, tiles: int = 8) -> None:
    """Local contrast boost (CLAHE on lightness) so dim, plain walls yield more features."""
    import cv2

    dst.mkdir(parents=True, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    for p in sorted(src.glob("*.jpg")):
        lab = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        cv2.imwrite(str(dst / p.name), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def solve_poses(frames_dir: Path, work_dir: Path, max_image_size: int = 1600, max_features: int = 4096,
                overlap: int = 10, num_threads: int = -1, peak_threshold: float | None = None,
                low_texture: bool = False):
    """SIFT + sequential matching + incremental mapping on CPU. Returns the largest reconstruction.

    low_texture: looser registration for plain walls and dim rooms (more matches tried,
    lower inlier minimums). Check the reprojection error and held-out distances after.
    """
    import pycolmap

    work_dir.mkdir(parents=True, exist_ok=True)
    db = work_dir / "database.db"
    if db.exists():
        db.unlink()
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    ext.sift.max_num_features = max_features
    if peak_threshold is not None:
        ext.sift.peak_threshold = peak_threshold
    ext.num_threads = num_threads
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"  # phone lenses: estimate k1, k2, p1, p2
    pycolmap.extract_features(db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = overlap
    pair.quadratic_overlap = True
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = num_threads
    pycolmap.match_sequential(db, matching_options=match, pairing_options=pair, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = num_threads
    if low_texture:
        opts.min_num_matches = 10
        opts.mapper.init_min_num_inliers = 50
        opts.mapper.abs_pose_min_num_inliers = 15
        opts.mapper.abs_pose_min_inlier_ratio = 0.15
        opts.mapper.max_reg_trials = 6
    recs = pycolmap.incremental_mapping(db, frames_dir, work_dir / "sparse", options=opts)
    if not recs:
        raise RuntimeError("pose solving failed: no reconstruction")
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    out = work_dir / "sparse_best"
    out.mkdir(exist_ok=True)
    best.write_text(str(out))
    return best, recs
