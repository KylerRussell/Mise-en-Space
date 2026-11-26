"""Learned reconstruction on a Colab GPU: frames in, a COLMAP model `mise scan --poses` can use.

Runs a feed-forward model (MapAnything by default, or VGGT) over a stretch of the frames that
`mise` already picked from the walkthrough, then writes the contract in docs/HANDOFF.md:

    OUT/frames.json          [{"name", "t"}] for the frames used (names as in the input)
    OUT/sparse/0/*.txt       COLMAP text model: one PINHOLE camera per frame (at the frame's
                             own size), poses, and points sampled from each frame's depth with
                             that frame as their one observation (so `sights` work)
    OUT/depth.npz            per-frame depth and confidence, downsampled (optional denser points)
    OUT/meta.json            model, whether the scale is metric, chunks and how well they joined

Long stretches go through the model in overlapping chunks. Each chunk joins the one before on
the frames they share: scale from the ratio of the two depth estimates of those frames (works
even when the phone only turns), rotation from their orientations, offset from their centers.

Usage on the VM (see run_learned_recon.sh, which uploads, runs and downloads):
    python learned_recon.py --frames /content/frames --out /content/out --t0 134 --t1 182
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_W, MODEL_H = 518, 294  # model input size; 16:9 frames at 518 wide -> 294 (a multiple of 14); set per model

# Other models, run through MapAnything's own wrappers (one interface, same outputs as MapAnything):
# name -> (pip extra of map-anything, input width, height, image normalization, patch size, metric)
EXTERNAL = {
    "pi3": ("", 518, 294, "identity", 14, False),
    "pi3x": ("pi3", 518, 294, "identity", 14, True),  # "approximate metric scale"
    "da3": ("depth-anything-3", 504, 280, "dinov2", 14, False),
    "da3_nested": ("depth-anything-3", 504, 280, "dinov2", 14, True),
    "must3r": ("must3r", 512, 288, "dust3r", 16, False),
    "vggt_omega": ("vggt-omega", 512, 288, "identity", 16, False),  # gated weights: needs HF access
}
CKPT_DIR = Path("/content/ckpt")
CHECKPOINTS = {
    "must3r": ["https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth",
               "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth",
               "https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_codebook.pkl"],
    "vggt_omega": ["hf://facebook/VGGT-Omega/vggt_omega_1b_512.pt"],
}
MODELS = ["mapanything", "vggt", *EXTERNAL]


# ---- setup ------------------------------------------------------------------------------

def install(model: str) -> None:
    repo = ("https://github.com/facebookresearch/vggt.git" if model == "vggt"
            else "https://github.com/facebookresearch/map-anything.git")
    dst = Path("/content") / Path(repo).stem
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth", "1", repo, str(dst)], check=True)
    if model in EXTERNAL:
        extra = EXTERNAL[model][0]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{dst}[{extra}]" if extra else str(dst)],
                       check=True)
        # the wrappers' Hydra configs want a "machine": point every folder at /content
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        (dst / "configs" / "machine" / "colab.yaml").write_text("\n".join(
            f"{k}: {CKPT_DIR if 'checkpoint' in k else '/content/unused'}" for k in (
                "root_data_dir", "mapanything_dataset_metadata_dir", "root_pretrained_checkpoints_dir",
                "root_experiments_dir", "root_uniception_pretrained_checkpoints_dir",
                "external_benchmark_data_root_data_dir")) + "\n")
        for url in CHECKPOINTS.get(model, []):
            dest = CKPT_DIR / url.rsplit("/", 1)[1]
            if dest.exists():
                continue
            print(f"[recon] fetching {dest.name}", flush=True)
            if url.startswith("hf://"):
                from huggingface_hub import hf_hub_download

                org, name, fname = url[5:].split("/", 2)
                hf_hub_download(f"{org}/{name}", fname, local_dir=str(CKPT_DIR))
            else:
                subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
    elif model == "mapanything":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(dst)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "einops", "safetensors"],
                       check=True)
    # an editable install registers itself in a .pth file, which this already running interpreter
    # won't read: put the clone on the path directly, and let it see the dependencies just installed
    import importlib

    sys.path.insert(0, str(dst))
    importlib.invalidate_caches()


def load_model(model: str, model_id: str | None):
    import torch

    if model in EXTERNAL:
        from mapanything.models import init_model_from_config

        return init_model_from_config(model, device="cuda", machine="colab").eval(), f"map-anything wrapper: {model}"
    if model == "mapanything":
        from mapanything.models import MapAnything

        mid = model_id or "facebook/map-anything"
        return MapAnything.from_pretrained(mid).to("cuda").eval(), mid
    from vggt.models.vggt import VGGT

    mid = model_id or "facebook/VGGT-1B"
    return VGGT.from_pretrained(mid).to("cuda").eval(), mid


# ---- one chunk through the model ----------------------------------------------------------
# Every runner returns, per frame (model input size MODEL_H x MODEL_W):
#   c2w (4x4, OpenCV axes), K (3x3), depth (HxW, z-depth), conf (HxW), valid (HxW bool), rgb (HxWx3 uint8)

def run_mapanything(model, paths: list[Path], K: np.ndarray | None = None) -> list[dict]:
    """K: the camera's intrinsics at the model's input size; without them the model guesses
    a focal length per frame, and a wrong one shrinks or stretches the room across the view."""
    import torch

    if K is None:
        from mapanything.utils.image import load_images

        views = load_images([str(p) for p in paths])
    else:
        from mapanything.utils.image import preprocess_inputs
        from PIL import Image

        views = preprocess_inputs([{"img": np.asarray(Image.open(p).convert("RGB")), "intrinsics": K.astype(np.float32)}
                                   for p in paths])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)
    out = []
    for p in preds:
        depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
        if depth.shape != (MODEL_H, MODEL_W):
            raise RuntimeError(f"model resized the frames to {depth.shape}, expected {(MODEL_H, MODEL_W)}: "
                               "the pixel mapping back to the frames would be wrong")
        out.append({"c2w": p["camera_poses"][0].float().cpu().numpy(), "K": p["intrinsics"][0].float().cpu().numpy(),
                    "depth": depth, "conf": p["conf"][0].float().cpu().numpy(),
                    "valid": p["mask"][0, ..., 0].bool().cpu().numpy() & (depth > 0),
                    "rgb": np.clip(p["img_no_norm"][0].float().cpu().numpy() * (255 if p["img_no_norm"].max() <= 1.5 else 1),
                                   0, 255).astype(np.uint8)})
    return out


def run_external(model, paths: list[Path], K: np.ndarray | None = None, name: str = "") -> list[dict]:
    """Any model in EXTERNAL, through MapAnything's wrapper and output post-processing (images only)."""
    import torch
    from mapanything.utils.image import load_images
    from mapanything.utils.inference import postprocess_model_outputs_for_inference

    _, w, h, norm, patch, _ = EXTERNAL[name]
    views = load_images([str(p) for p in paths], resize_mode="fixed_size", size=(w, h), norm_type=norm,
                        patch_size=patch)
    for v in views:
        for k, val in list(v.items()):
            if torch.is_tensor(val):
                v[k] = val.to("cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw = model(views)
        preds = postprocess_model_outputs_for_inference(raw, views, apply_mask=False)
    out = []
    for p in preds:
        depth = p["depth_z"][0, ..., 0].float().cpu().numpy()
        if depth.shape != (MODEL_H, MODEL_W):
            raise RuntimeError(f"{name} returned depth {depth.shape}, expected {(MODEL_H, MODEL_W)}")
        conf = p["conf"][0].float().cpu().numpy() if "conf" in p else np.ones_like(depth)
        if conf.ndim == 3:
            conf = conf[..., 0]
        img = p["img_no_norm"][0].float().cpu().numpy()
        out.append({"c2w": p["camera_poses"][0].float().cpu().numpy(), "K": p["intrinsics"][0].float().cpu().numpy(),
                    "depth": depth, "conf": conf, "valid": np.isfinite(depth) & (depth > 0),
                    "rgb": np.clip(img * (255 if img.max() <= 1.5 else 1), 0, 255).astype(np.uint8)})
    return out


def run_vggt(model, paths: list[Path], K: np.ndarray | None = None) -> list[dict]:  # VGGT takes no intrinsics
    import torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images([str(p) for p in paths]).to("cuda")
    if tuple(images.shape[-2:]) != (MODEL_H, MODEL_W):
        raise RuntimeError(f"VGGT preprocessed to {tuple(images.shape[-2:])}, expected {(MODEL_H, MODEL_W)}")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            tokens, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, conf = model.depth_head(tokens, images[None], ps_idx)
    ext = extrinsic[0].float().cpu().numpy()
    Ks = intrinsic[0].float().cpu().numpy()
    depth = depth[0, ..., 0].float().cpu().numpy()
    conf = conf[0].float().cpu().numpy()
    rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)
    out = []
    for i in range(len(paths)):
        c2w = np.eye(4)
        c2w[:3, :3] = ext[i, :3, :3].T
        c2w[:3, 3] = -ext[i, :3, :3].T @ ext[i, :3, 3]
        valid = (depth[i] > 0) & (conf[i] > np.percentile(conf[i], 25))
        out.append({"c2w": c2w, "K": Ks[i], "depth": depth[i], "conf": conf[i], "valid": valid, "rgb": rgb[i]})
    return out


# ---- joining chunks -----------------------------------------------------------------------

def mean_rot(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def join(prev: dict[int, dict], cur: dict[int, dict]) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Similarity (s, R, t) taking `cur`'s frame into `prev`'s, from the frames both chunks hold."""
    shared = sorted(set(prev) & set(cur))
    if len(shared) < 3:
        raise RuntimeError(f"chunks share only {len(shared)} frames")
    ratios = []
    for i in shared:
        a, b = prev[i], cur[i]
        ok = a["valid"] & b["valid"]
        if ok.sum() > 100:
            ratios.append(np.median(a["depth"][ok] / b["depth"][ok]))
    s = float(np.median(ratios))
    Rs = np.array([prev[i]["c2w"][:3, :3] @ cur[i]["c2w"][:3, :3].T for i in shared])
    R = mean_rot(Rs)
    ang = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rs, R) - 1) / 2, -1, 1)))
    cp = np.array([prev[i]["c2w"][:3, 3] for i in shared])
    cc = np.array([cur[i]["c2w"][:3, 3] for i in shared])
    t = np.median(cp - s * cc @ R.T, axis=0)
    res = np.linalg.norm(s * cc @ R.T + t - cp, axis=1)
    return s, R, t, {"shared": len(shared), "scale": round(s, 4), "scale_spread": round(float(np.std(ratios) / s), 4),
                     "rot_spread_deg": round(float(np.median(ang)), 2), "center_residual": round(float(np.median(res)), 4)}


def apply(fr: dict, s: float, R: np.ndarray, t: np.ndarray) -> dict:
    c2w = np.eye(4)
    c2w[:3, :3] = R @ fr["c2w"][:3, :3]
    c2w[:3, 3] = s * R @ fr["c2w"][:3, 3] + t
    return {**fr, "c2w": c2w, "depth": fr["depth"] * s}


# ---- COLMAP text output ---------------------------------------------------------------------

def quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = R
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S]
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


class ColmapWriter:
    """Streams cameras/images/points3D text files; each frame adds its camera, pose and points."""

    def __init__(self, out: Path, frame_w: int, frame_h: int):
        out.mkdir(parents=True, exist_ok=True)
        self.cams = (out / "cameras.txt").open("w")
        self.imgs = (out / "images.txt").open("w")
        self.pts = (out / "points3D.txt").open("w")
        self.cams.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        self.imgs.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        self.pts.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        self.sx, self.sy = frame_w / MODEL_W, frame_h / MODEL_H
        self.fw, self.fh = frame_w, frame_h
        self.next_pid = 1
        self.n_points = 0

    def add(self, image_id: int, name: str, fr: dict, n_points: int, rng: np.random.Generator) -> None:
        K = fr["K"]
        fx, fy, cx, cy = K[0, 0] * self.sx, K[1, 1] * self.sy, K[0, 2] * self.sx, K[1, 2] * self.sy
        self.cams.write(f"{image_id} PINHOLE {self.fw} {self.fh} {fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")
        R_wc = fr["c2w"][:3, :3].T
        t_wc = -R_wc @ fr["c2w"][:3, 3]
        q = quat_wxyz(R_wc)
        # points: confident, valid pixels; at most n_points, spread at random over the frame
        valid = fr["valid"] & np.isfinite(fr["depth"]) & (fr["depth"] > 0)
        if valid.sum() > 0:
            valid &= fr["conf"] >= np.percentile(fr["conf"][valid], 40)
        vv, uu = np.nonzero(valid)
        if len(uu) > n_points:
            k = rng.choice(len(uu), n_points, replace=False)
            vv, uu = vv[k], uu[k]
        z = fr["depth"][vv, uu]
        cam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
        world = cam @ fr["c2w"][:3, :3].T + fr["c2w"][:3, 3]
        rgb = fr["rgb"][vv, uu]
        obs = []
        for j in range(len(uu)):
            pid = self.next_pid
            self.next_pid += 1
            X, Y, Z = world[j]
            r, g, b = (int(c) for c in rgb[j])
            self.pts.write(f"{pid} {X:.5f} {Y:.5f} {Z:.5f} {r} {g} {b} 1.0 {image_id} {j}\n")
            obs.append(f"{uu[j] * self.sx:.2f} {vv[j] * self.sy:.2f} {pid}")
        self.n_points += len(uu)
        self.imgs.write(f"{image_id} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                        f"{t_wc[0]:.6f} {t_wc[1]:.6f} {t_wc[2]:.6f} {image_id} {name}\n{' '.join(obs)}\n")

    def close(self) -> None:
        for f in (self.cams, self.imgs, self.pts):
            f.close()


# ---- main -----------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, type=Path, help="folder with frames/ and frames.json (from mise)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", choices=MODELS, default="mapanything")
    ap.add_argument("--model-id", help="Hugging Face id (default: facebook/map-anything or facebook/VGGT-1B)")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--chunk", type=int, default=200, help="frames per pass through the model")
    ap.add_argument("--overlap", type=int, default=24, help="frames shared by consecutive chunks")
    ap.add_argument("--points-per-frame", type=int, default=500)
    ap.add_argument("--limit", type=int, help="only the first N frames of the stretch (smoke test)")
    ap.add_argument("--intrinsics", help="the camera's fx,fy,cx,cy in frame pixels (MapAnything uses them; "
                    "default: FRAMES/intrinsics.json if there is one)")
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args(argv)
    global MODEL_W, MODEL_H
    if a.model in EXTERNAL:
        MODEL_W, MODEL_H = EXTERNAL[a.model][1:3]
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()

    from PIL import Image

    kept = json.loads((a.frames / "frames.json").read_text())
    sel = [k for k in kept if (a.t0 is None or k["t"] >= a.t0) and (a.t1 is None or k["t"] <= a.t1)]
    if a.limit:
        sel = sel[: a.limit]
    if len(sel) < 3:
        raise SystemExit(f"only {len(sel)} frames in the stretch")
    fw, fh = Image.open(a.frames / "frames" / sel[0]["name"]).size
    print(f"[recon] {len(sel)} frames ({sel[0]['t']:.1f}-{sel[-1]['t']:.1f} s), {fw}x{fh}, model {a.model}", flush=True)

    # the model sees the frames at exactly MODEL_W x MODEL_H, so mapping back to the frame is a plain scale
    small = Path("/content/_model_in") if Path("/content").exists() else a.out / "_model_in"
    small.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in sel:
        p = small / k["name"]
        if not p.exists():
            Image.open(a.frames / "frames" / k["name"]).convert("RGB").resize((MODEL_W, MODEL_H), Image.LANCZOS).save(p, quality=95)
        paths.append(p)

    if not a.no_install:
        install(a.model)
    import torch

    K = None
    kfile = a.frames / "intrinsics.json"
    if a.intrinsics or kfile.exists():
        fx, fy, cx, cy = (map(float, a.intrinsics.split(",")) if a.intrinsics else
                          (json.loads(kfile.read_text())[k] for k in ("fx", "fy", "cx", "cy")))
        sx, sy = MODEL_W / fw, MODEL_H / fh
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
        print(f"[recon] known intrinsics: fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f} (frame pixels)", flush=True)
    model, model_id = load_model(a.model, a.model_id)
    gpu = torch.cuda.get_device_name(0)
    if a.model in EXTERNAL:
        def runner(m, ps, k):
            return run_external(m, ps, k, a.model)
        if K is not None:
            print(f"[recon] {a.model} runs on images only here: the intrinsics are not used", flush=True)
    else:
        runner = run_mapanything if a.model == "mapanything" else run_vggt
    print(f"[recon] loaded {model_id} on {gpu} in {time.time() - started:.0f} s", flush=True)

    n = len(sel)
    step = max(a.chunk - a.overlap, 1)
    starts = list(range(0, max(n - a.overlap, 1), step))
    writer = ColmapWriter(a.out / "sparse" / "0", fw, fh)
    rng = np.random.default_rng(0)
    prev: dict[int, dict] = {}
    chunks, depth_small, conf_small, done = [], {}, {}, set()
    for ci, s0 in enumerate(starts):
        s1 = min(s0 + a.chunk, n)
        t = time.time()
        frs = runner(model, paths[s0:s1], K)
        torch.cuda.empty_cache()
        cur = {s0 + i: fr for i, fr in enumerate(frs)}
        info = {"chunk": ci, "frames": [s0, s1], "seconds": round(time.time() - t, 1),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        if prev:
            s, R, tt, rep = join(prev, cur)
            cur = {i: apply(fr, s, R, tt) for i, fr in cur.items()}
            info["join"] = rep
        chunks.append(info)
        print(f"[recon] chunk {ci + 1}/{len(starts)}: {json.dumps(info)}", flush=True)
        for i in range(s0, s1):
            if i in done:  # shared with the chunk before: keep the first estimate
                continue
            fr = cur[i]
            writer.add(i + 1, sel[i]["name"], fr, a.points_per_frame, rng)
            depth_small[i] = fr["depth"][::2, ::2].astype(np.float16)
            conf_small[i] = fr["conf"][::2, ::2].astype(np.float16)
            done.add(i)
        prev = {i: cur[i] for i in range(max(s0, s1 - a.overlap), s1)}
        if s1 >= n:
            break
    writer.close()

    names = [sel[i]["name"] for i in sorted(done)]
    np.savez_compressed(a.out / "depth.npz", names=np.array(names),
                        depth=np.stack([depth_small[i] for i in sorted(done)]),
                        conf=np.stack([conf_small[i] for i in sorted(done)]),
                        scale_xy=np.array([fw / (MODEL_W / 2), fh / (MODEL_H / 2)]))
    (a.out / "frames.json").write_text(json.dumps([{"name": sel[i]["name"], "t": sel[i]["t"]} for i in sorted(done)]))
    meta = {"model": a.model, "model_id": model_id, "metric": a.model == "mapanything" or EXTERNAL.get(a.model, (0,) * 6)[5], "gpu": gpu,
            "intrinsics_given": None if K is None else [round(float(v), 2) for v in (K[0, 0] / sx, K[1, 1] / sy, K[0, 2] / sx, K[1, 2] / sy)],
            "frames": len(done), "t0": sel[0]["t"], "t1": sel[-1]["t"], "points": writer.n_points,
            "chunks": chunks, "seconds": round(time.time() - started, 1), "model_size": [MODEL_W, MODEL_H],
            "frame_size": [fw, fh]}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[recon] done: {json.dumps({k: v for k, v in meta.items() if k != 'chunks'})}", flush=True)


if __name__ == "__main__":
    main()
