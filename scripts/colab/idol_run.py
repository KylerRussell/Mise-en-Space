"""Idol (speridlabs/idol, image -> 3D Gaussian splat) on a Colab GPU, for product images.

    python idol_run.py --images /content/in/images --out /content/out [--num-gaussians 262144]

/content/in/images.json lists {"name": "sofa.jpg", "mode": "real" | "synthetic"}; each image gives
out/<stem>.ply (standard 3DGS), out/<stem>_input.png (what the model saw, background removed) and
out/<stem>_camera.json. out/meta.json has the times and each .ply's header. The gated encoders
(DINOv3, FLUX.2's VAE) need a Hugging Face token: /content/in/hf_token, moved into the HF cache and
removed from the inputs before anything runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def say(m: str) -> None:
    print(f"[idol] {m}", flush=True)


def install() -> None:
    try:
        import idol  # noqa: F401
        return
    except ImportError:
        pass
    if not Path("/content/idol-sdk").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/speridlabs/idol-sdk.git",
                        "/content/idol-sdk"], check=True)
    say("pip install -e idol-sdk")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/idol-sdk"], check=True)
    # an editable install is only importable in a new interpreter: start again (once)
    if os.environ.get("IDOL_RESTARTED"):
        raise SystemExit("idol still not importable after installing idol-sdk")
    os.environ["IDOL_RESTARTED"] = "1"
    os.execv(sys.executable, [sys.executable, "-u", *sys.argv])


def token(path: Path) -> None:
    if path.exists():
        d = Path.home() / ".cache" / "huggingface"
        d.mkdir(parents=True, exist_ok=True)
        (d / "token").write_text(path.read_text().strip())
        path.unlink()
        os.environ["HF_TOKEN"] = (d / "token").read_text()


DINOV3_HUB = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
DINOV3_TIMM = "vit_huge_plus_patch16_dinov3.lvd1689m"  # Hugging Face's timm port of the same weights, not gated


def hub_dinov3_ok() -> bool:
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(DINOV3_HUB, "config.json")
        return True
    except Exception as e:  # GatedRepoError while Meta reviews the request
        say(f"official DINOv3 not available to this account ({type(e).__name__})")
        return False


def timm_dinov3():
    """DINOv3 ViT-H+/16 from timm, shaped like transformers' DINOv3ViTModel for Idol: called with
    pixel_values, returns .last_hidden_state = the final-normed [CLS, 4 registers, patches] tokens."""
    import types

    import torch
    try:
        import timm
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "timm"], check=True)
        import timm

    class Dinov3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = timm.create_model(DINOV3_TIMM, pretrained=True, dynamic_img_size=True)

        def forward(self, pixel_values):
            return types.SimpleNamespace(last_hidden_state=self.m.forward_features(pixel_values))

    return Dinov3().to(torch.bfloat16)


def parity(model, n_px: int = 512) -> None:
    """With both at hand: how close timm's tokens are to the official model's (cosine per token)."""
    import torch
    from transformers import AutoModel

    ref = AutoModel.from_pretrained(DINOV3_HUB, dtype=torch.float32).cuda().eval()
    x = torch.rand(1, 3, n_px, n_px, device="cuda")
    with torch.no_grad():
        a = ref(pixel_values=x).last_hidden_state
        b = model.float().cuda()(pixel_values=x).last_hidden_state
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    say(f"parity timm vs official DINOv3: tokens {tuple(a.shape)} vs {tuple(b.shape)}, "
        f"cosine min {cos.min():.4f} mean {cos.mean():.4f}")


def hinted(pipe, src: Path, hint: Path, out: Path) -> tuple[Path, float]:
    """The photo with an alpha: BiRefNet's, times the (blurred) hint; the hint alone if they barely meet."""
    import numpy as np
    from PIL import Image, ImageFilter

    rgb = Image.open(src).convert("RGB")
    cut = pipe.background_remover.remove_background(rgb)
    a = np.asarray(cut.getchannel(3), np.float32) / 255
    h = np.asarray(Image.open(hint).convert("L").resize(rgb.size).filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
    both = a * h
    agree = float(both.sum() / max(h.sum(), 1e-6))  # how much of the hint BiRefNet also cut out
    if agree < 0.3:
        say(f"{src.name}: the cut-out and the hint barely meet; using the hint")
        both = h
    rgba = rgb.copy()
    rgba.putalpha(Image.fromarray((both * 255).astype(np.uint8)))
    rgba.save(out)
    return out, agree


def ply_header(p: Path) -> dict:
    with open(p, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            line = fh.readline()
            if not line:
                break
            head += line
    lines = head.decode(errors="replace").splitlines()
    props = [ln.split()[-1] for ln in lines if ln.startswith("property")]
    n = next((int(ln.split()[-1]) for ln in lines if ln.startswith("element vertex")), 0)
    return {"vertices": n, "properties": props}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-gaussians", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dinov3", choices=["auto", "hub", "timm"], default="auto",
                    help="the official (gated) DINOv3, timm's ungated port of it, or the official one when allowed")
    ap.add_argument("--parity", action="store_true", help="compare timm's DINOv3 with the official one first")
    a = ap.parse_args()
    token(a.images.parent / "hf_token")
    t0 = time.time()
    install()
    import torch
    from idol import IdolPipeline

    say(f"torch {torch.__version__}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    use_timm = a.dinov3 == "timm" or (a.dinov3 == "auto" and not hub_dinov3_ok())
    if a.parity:
        parity(timm_dinov3())
    enc = {"dinov3": timm_dinov3()} if use_timm else None
    say(f"DINOv3 from {'timm (' + DINOV3_TIMM + ')' if use_timm else DINOV3_HUB}")
    pipe = IdolPipeline.from_pretrained("speridlabs/idol", image_encoders=enc)
    say(f"loaded in {time.time() - t0:.0f} s")
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.images.parent / "images.json").read_text())
    meta = {"setup_s": round(time.time() - t0, 1), "items": [], "dinov3": DINOV3_TIMM if use_timm else DINOV3_HUB}
    for j in jobs:
        src = a.images / j["name"]
        stem = Path(j["name"]).stem
        t = time.time()
        agree = None
        if j.get("hint"):  # keep BiRefNet's cut-out only where the hint says the object is
            src, agree = hinted(pipe, src, a.images / j["hint"], a.out / f"{stem}_cut.png")
        try:
            o = pipe(str(src), mode=j.get("mode", "real"), seed=a.seed, num_gaussians=a.num_gaussians)
        except TypeError:  # older SDKs: no num_gaussians keyword
            o = pipe(str(src), mode=j.get("mode", "real"), seed=a.seed)
        o.gaussian.save_ply(str(a.out / f"{stem}.ply"))
        if getattr(o, "preprocessed_image", None) is not None:
            o.preprocessed_image.save(str(a.out / f"{stem}_input.png"))
        cam = getattr(o, "camera", None)
        if cam is not None:
            try:
                (a.out / f"{stem}_camera.json").write_text(json.dumps(cam if isinstance(cam, dict) else
                                                                      getattr(cam, "__dict__", str(cam)), default=str))
            except Exception as e:  # the camera is a nice-to-have
                say(f"camera not saved: {e}")
        dt = time.time() - t
        h = ply_header(a.out / f"{stem}.ply")
        say(f"{j['name']}: {h['vertices']} Gaussians in {dt:.1f} s")
        meta["items"].append({"name": j["name"], "mode": j.get("mode", "real"), "seconds": round(dt, 1), "ply": h,
                              "hint_agreement": None if agree is None else round(agree, 3)})
    meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    say("done")


if __name__ == "__main__":
    main()
