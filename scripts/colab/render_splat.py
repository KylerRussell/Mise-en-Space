"""Render splats from given cameras on a Colab GPU, side by side, to look at them.

    python render_splat.py --splats /content/in/a.ply /content/in/b.ply --cameras /content/in/cameras.json \
        --frames /content/in/frames --names frame_0100.jpg frame_0500.jpg --out /content/out

Each output image is the photo (when the frame is there) and then each splat's render from that
frame's camera (cameras.json from splat_train.py, scene frame).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def read_ply(path: Path):
    with open(path, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += fh.readline()
        props = [ln.split()[-1].decode() for ln in head.splitlines() if ln.startswith(b"property")]
        n = int(next(ln for ln in head.splitlines() if ln.startswith(b"element vertex")).split()[-1])
        data = np.frombuffer(fh.read(), dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    take = lambda names: data[:, [col[x] for x in names]]  # noqa: E731
    rest = sorted((p for p in props if p.startswith("f_rest_")), key=lambda p: int(p.split("_")[-1]))
    return {"means": take(["x", "y", "z"]), "dc": take(["f_dc_0", "f_dc_1", "f_dc_2"]),
            "rest": take(rest) if rest else np.zeros((n, 0), np.float32), "opacity": data[:, col["opacity"]],
            "scales": take(["scale_0", "scale_1", "scale_2"]), "quats": take(["rot_0", "rot_1", "rot_2", "rot_3"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splats", type=Path, nargs="+", required=True)
    ap.add_argument("--cameras", type=Path, required=True)
    ap.add_argument("--frames", type=Path)
    ap.add_argument("--names", nargs="*", help="frames to render (default: 8 spread over the cameras)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=640)
    a = ap.parse_args(argv)
    t0 = time.time()
    import splat_train  # same folder on the VM: its gsplat install
    splat_train.install()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import rasterization

    cams = json.loads(a.cameras.read_text())
    by = {c["name"]: c for c in cams}
    names = a.names or [cams[i]["name"] for i in np.linspace(0, len(cams) - 1, 8).astype(int)]
    splats = []
    for p in a.splats:
        g = read_ply(p)
        n = len(g["means"])
        k = g["rest"].shape[1] // 3
        sh = np.concatenate([g["dc"][:, None, :], g["rest"].reshape(n, 3, k).transpose(0, 2, 1)], 1)
        deg = int(round(np.sqrt(sh.shape[1]))) - 1
        t = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)  # noqa: E731
        splats.append((t(g["means"]), F.normalize(t(g["quats"]), dim=-1), torch.exp(t(g["scales"])),
                       torch.sigmoid(t(g["opacity"])), t(sh), deg))
        print(f"[render] {p.name}: {n} Gaussians, SH degree {deg}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        c = by[name]
        W, H = c["size"]
        s = a.width / W
        K = torch.tensor(c["K"], device="cuda", dtype=torch.float32)
        K[:2] *= s
        w, h = int(W * s), int(H * s)
        V = torch.tensor(c["w2c"], device="cuda", dtype=torch.float32)[None]
        tiles = []
        if a.frames and (a.frames / name).exists():
            tiles.append(np.asarray(Image.open(a.frames / name).convert("RGB").resize((w, h))))
        for means, quats, scales, opac, sh, deg in splats:
            with torch.no_grad():
                out, _, _ = rasterization(means, quats, scales, opac, sh, V, K[None], w, h, sh_degree=deg,
                                          near_plane=0.01, far_plane=100.0)
            tiles.append((out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        Image.fromarray(np.concatenate(tiles, 1)).save(a.out / f"{name[:-4]}.jpg", quality=88)
    (a.out / "meta.json").write_text(json.dumps({"frames": names, "splats": [p.name for p in a.splats],
                                                 "seconds": round(time.time() - t0, 1)}))
    print(f"[render] {len(names)} views in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
