"""Label every pixel of mise's picked frames with an ADE20K class, on a Colab GPU.

    python segment_frames.py --frames /content/in --out /content/out

Model: Mask2Former (Swin-L) trained on ADE20K semantic segmentation (facebook/
mask2former-swin-large-ade-semantic; MIT-licensed code and weights). Its 150 classes include
the structure (wall, floor, ceiling, door, windowpane, ...) and the things in front of it
(bed, cabinet, wardrobe, curtain, mirror, person, ...). mise builds the layout from structure
only, and uses the rest as what to take out for the empty room.

Output: out/labels.npz (one uint8 map per frame, keyed by frame name, at --size) and
out/meta.json (model, class names, size, seconds).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MODEL = "facebook/mask2former-swin-large-ade-semantic"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, nargs=2, default=[512, 288], help="label map width height")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to("cuda").eval().half()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    frames = json.loads((a.frames / "frames.json").read_text())
    print(f"[seg] {len(frames)} frames, {MODEL} on {torch.cuda.get_device_name()}", flush=True)
    W, H = a.size
    out = {}
    for b in range(0, len(frames), a.batch):
        ks = frames[b: b + a.batch]
        ims = [Image.open(a.frames / "frames" / k["name"]).convert("RGB") for k in ks]
        x = proc(images=ims, return_tensors="pt")
        with torch.no_grad():
            r = model(pixel_values=x["pixel_values"].to("cuda").half())
        maps = proc.post_process_semantic_segmentation(r, target_sizes=[(H, W)] * len(ims))
        for k, m in zip(ks, maps):
            out[k["name"]] = m.to(torch.uint8).cpu().numpy()
        if (b // a.batch) % 20 == 0:
            print(f"[seg] {b + len(ks)}/{len(frames)} ({time.time() - t0:.0f} s)", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "labels.npz", **out)
    meta = {"model": MODEL, "classes": names, "size": [W, H], "frames": len(out), "seconds": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name()}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[seg] " + json.dumps({k: v for k, v in meta.items() if k != "classes"}), flush=True)


if __name__ == "__main__":
    main()
