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
    a = ap.parse_args()
    token(a.images.parent / "hf_token")
    t0 = time.time()
    install()
    import torch
    from idol import IdolPipeline

    say(f"torch {torch.__version__}, {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    pipe = IdolPipeline.from_pretrained("speridlabs/idol")
    say(f"loaded in {time.time() - t0:.0f} s")
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((a.images.parent / "images.json").read_text())
    meta = {"setup_s": round(time.time() - t0, 1), "items": []}
    for j in jobs:
        src = a.images / j["name"]
        stem = Path(j["name"]).stem
        t = time.time()
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
        meta["items"].append({"name": j["name"], "mode": j.get("mode", "real"), "seconds": round(dt, 1), "ply": h})
    meta["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    say("done")


if __name__ == "__main__":
    main()
