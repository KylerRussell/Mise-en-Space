"""Train a 3D Gaussian splat of the home on a Colab GPU, from mise's picked frames and a solve.

Runs on the VM (see run_splat.py, which uploads the inputs and fetches splat.ply):

    python splat_train.py --frames /content/in --solve /content/in/solve --out /content/out

Inputs: frames/ + frames.json (mise's picked frames) and a solve in the layout
`mise scan --poses` takes (sparse/0 as COLMAP text, with points3D seeding the Gaussians).
Training is plain 3DGS with gsplat's rasterizer and its default densification: L1 + D-SSIM on
every frame but a held-out eighth, whose PSNR is reported (meta.json). A learned model's
cameras are good to a few cm and about a degree, which blurs a splat, so each camera also
learns a small rotation and offset (--pose-opt, on by default); before the held-out frames are
scored their cameras get the same alignment against the finished splat (test-time pose
alignment), so the score measures the splat, not the model's pose error. A phone's exposure
and white balance drift from room to room: each frame also learns a color correction (a 3x3
matrix and offset on the rendered color, --color-opt), aligned the same way for held-out frames. The splat stays in the
solve's frame unless --world gives the scene's (scene = scale * R @ solve + t, from the scan's
result.json): then cameras and seeds are moved first, so the splat comes out in the scene's
frame with its colors' view dependence intact.

Output: out/splat.ply (standard 3DGS layout: xyz, f_dc, f_rest, opacity, scale, rot),
out/cameras.json (the refined cameras), out/meta.json (steps, Gaussians, held-out PSNR, seconds)
and, given --labels (segment_frames.py output), out/gauss_labels.npz: each Gaussian's class by
vote of the frames that see it (its center projects onto a pixel whose rendered depth is the
Gaussian's own, so walls don't take the label of the bed in front of them): cls, share, seen.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def install() -> None:
    try:
        import gsplat  # noqa: F401
        return
    except ImportError:
        pass
    import torch

    tv = "".join(torch.__version__.split("+")[0].split(".")[:2])
    cu = (torch.version.cuda or "").replace(".", "")
    for idx in ([f"https://docs.gsplat.studio/whl/pt{tv}cu{cu}"] if cu else []) + [None]:
        cmd = [sys.executable, "-m", "pip", "install", "-q", "gsplat"] + (["--index-url", idx] if idx else [])
        print("[splat] " + " ".join(cmd[3:]), flush=True)
        if subprocess.run(cmd).returncode == 0:
            try:
                import gsplat  # noqa: F401,F811
                return
            except ImportError:
                continue
    raise SystemExit("could not install gsplat")


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_solve(sparse: Path):
    cams = {}
    for line in (sparse / "cameras.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        assert p[1] == "PINHOLE", p[1]
        fx, fy, cx, cy = map(float, p[4:8])
        cams[int(p[0])] = (int(p[2]), int(p[3]), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]))
    images = []
    lines = [ln for ln in (sparse / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    for ln in lines[0::2]:
        p = ln.split()
        if len(p) < 10:
            continue
        R = qvec2rotmat(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = R, t
        W, H, K = cams[int(p[8])]
        images.append({"name": p[9], "w2c": w2c, "K": K, "size": (W, H)})
    xyz, rgb = [], []
    for ln in (sparse / "points3D.txt").read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.split()
        xyz.append(list(map(float, p[1:4])))
        rgb.append(list(map(int, p[4:7])))
    return sorted(images, key=lambda im: im["name"]), np.array(xyz, np.float32), np.array(rgb, np.float32) / 255.0


def so3_exp(w):
    """Rotation matrices from axis-angle vectors (N, 3)."""
    import torch

    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros(len(w), 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def ssim(a, b):
    """D-SSIM term on (1, 3, H, W) images (11x11 Gaussian window)."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=3), F.conv2d(b, w, padding=5, groups=3)
    saa = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    sbb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean()


def write_ply(path: Path, xyz, f_dc, f_rest, opacity, scales, quats) -> None:
    n = len(xyz)
    fields = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + \
             [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] + [f"scale_{i}" for i in range(3)] + \
             [f"rot_{i}" for i in range(4)]
    data = np.concatenate([xyz, np.zeros_like(xyz), f_dc, f_rest, opacity[:, None], scales, quats], axis=1).astype("<f4")
    head = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
           "".join(f"property float {f}\n" for f in fields) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(data.tobytes())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--solve", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--downscale", type=int, default=1, help="train on frames this many times smaller")
    ap.add_argument("--sh", type=int, default=3)
    ap.add_argument("--holdout", type=int, default=8, help="every Nth frame is held out for the PSNR check")
    ap.add_argument("--max-init", type=int, default=600000)
    ap.add_argument("--pose-opt", type=float, default=1e-4, help="learning rate of the camera corrections (0: off)")
    ap.add_argument("--color-opt", type=float, default=0.0, help="learning rate of the per-frame color correction (0: off)")
    ap.add_argument("--refine-stop", type=float, default=0.5, help="share of the steps that densify")
    ap.add_argument("--grow-grad", type=float, default=0.0008, help="densify where the 2D gradient is above this")
    ap.add_argument("--depth-weight", type=float, default=0.1,
                    help="pull rendered depth toward the solve's depth maps (depth.npz) by this much (0: off)")
    ap.add_argument("--labels", type=Path, help="segment_frames.py output: label every Gaussian by vote")
    ap.add_argument("--world", help='JSON {"scale": s, "R": 3x3, "t": 3}: train in the frame scale * R @ x + t')
    a = ap.parse_args(argv)
    t_start = time.time()
    install()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    dev = "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    images, xyz, rgb = read_solve(a.solve / "sparse" / "0")
    images = [im for im in images if (a.frames / "frames" / im["name"]).exists()]  # a stretch: its frames only
    if a.world:
        wd = json.loads(a.world)
        sc, Rw, tw = float(wd["scale"]), np.array(wd["R"], float), np.array(wd["t"], float)
        xyz = (sc * xyz.astype(np.float64) @ Rw.T + tw).astype(np.float32)
        for im in images:  # x_cam = Rc x + tc, x = Rw^T (x' - tw) / sc; cameras scaled with the world
            Rc, tc = im["w2c"][:3, :3].copy(), im["w2c"][:3, 3].copy()  # copies: both are rewritten below
            im["w2c"][:3, :3] = Rc @ Rw.T
            im["w2c"][:3, 3] = sc * tc - Rc @ Rw.T @ tw
    if len(xyz) > a.max_init:
        keep = np.random.default_rng(0).choice(len(xyz), a.max_init, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    print(f"[splat] {len(images)} frames, {len(xyz)} seed points, GPU {torch.cuda.get_device_name()}", flush=True)

    # frames on the GPU as uint8 (877 x 1280 x 720 x 3 is ~2.4 GB)
    ds = a.downscale
    imgs, Ks, w2cs = [], [], []
    for im in images:
        pic = Image.open(a.frames / "frames" / im["name"]).convert("RGB")
        W, H = im["size"]
        if pic.size != (W, H):
            pic = pic.resize((W, H), Image.BICUBIC)
        if ds > 1:
            pic = pic.resize((W // ds, H // ds), Image.BICUBIC)
        imgs.append(torch.from_numpy(np.asarray(pic)).to(dev))
        K = im["K"].copy()
        K[:2] /= ds
        Ks.append(K)
        w2cs.append(im["w2c"])
    # the learned model's depth for each frame (solve units; scaled with the world): keeps
    # Gaussians on the surfaces, where plain walls and tight closets give the colors little hold
    dmaps = None
    if a.depth_weight > 0 and (a.solve / "depth.npz").exists():
        dz = np.load(a.solve / "depth.npz")
        at = {str(nm): k for k, nm in enumerate(dz["names"])}
        sc_d = float(json.loads(a.world)["scale"]) if a.world else 1.0
        dl, cl = [], []
        for im in images:
            k = at.get(im["name"])
            dl.append(dz["depth"][k].astype(np.float32) * sc_d if k is not None else np.zeros(dz["depth"].shape[1:], np.float32))
            cl.append(dz["conf"][k].astype(np.float32) if k is not None else np.zeros(dz["conf"].shape[1:], np.float32))
        dmaps = torch.tensor(np.array(dl), device=dev)
        cmaps = torch.tensor(np.array(cl), device=dev)
        cthr = torch.quantile(cmaps[cmaps > 0][::97], 0.3)  # the least sure 30% of depth left out
        print(f"[splat] depth maps {tuple(dmaps.shape)}, weight {a.depth_weight}", flush=True)
    Ks = torch.tensor(np.array(Ks), dtype=torch.float32, device=dev)
    w2cs = torch.tensor(np.array(w2cs), dtype=torch.float32, device=dev)
    n = len(imgs)
    test = [i for i in range(n) if a.holdout and i % a.holdout == a.holdout // 2]
    train = [i for i in range(n) if i not in set(test)]
    Hh, Ww = imgs[0].shape[:2]

    # scene extent: camera spread (for learning rates and densification thresholds)
    centers = torch.linalg.inv(w2cs)[:, :3, 3]
    extent = float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1

    pts = torch.tensor(xyz, device=dev)
    d = torch.cdist(pts[:: max(1, len(pts) // 20000)], pts[:: max(1, len(pts) // 20000)])
    d.fill_diagonal_(1e9)
    s0 = float(d.topk(4, largest=False).values[:, 1:].mean())  # typical spacing of the seeds
    N = len(pts)
    sh_n = (a.sh + 1) ** 2
    colors = torch.zeros(N, sh_n, 3, device=dev)
    colors[:, 0] = (torch.tensor(rgb, device=dev) - 0.5) / 0.28209479177387814
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.full((N, 3), math.log(s0), device=dev)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0], device=dev).repeat(N, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(colors[:, :1].contiguous()),
        "shN": torch.nn.Parameter(colors[:, 1:].contiguous()),
    })
    lrs = {"means": 1.6e-4 * extent, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / a.steps))
    strategy = DefaultStrategy(verbose=False, refine_stop_iter=int(a.steps * a.refine_stop), reset_every=3000,
                               refine_every=100, absgrad=True, grow_grad2d=a.grow_grad)
    strategy.check_sanity(params, opt)
    state = strategy.initialize_state(scene_scale=extent)

    # per-camera corrections, applied in the camera's own frame: w2c' = [R(dw) | dt] @ w2c
    dw = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
    dt = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
    pose_opt = torch.optim.Adam([dw, dt], lr=a.pose_opt) if a.pose_opt > 0 else None
    cA = torch.nn.Parameter(torch.zeros(n, 3, 3, device=dev))  # color = (I + cA) @ rgb + cb, per frame
    cb = torch.nn.Parameter(torch.zeros(n, 3, device=dev))
    color_opt = torch.optim.Adam([cA, cb], lr=a.color_opt) if a.color_opt > 0 else None

    def colored(rgb, i):
        if color_opt is None:
            return rgb.clamp(0, 1)
        M = torch.eye(3, device=dev) + cA[i]
        return (rgb @ M.T + cb[i]).clamp(0, 1)

    def viewmat(idx):
        idx = torch.as_tensor(idx, device=dev)
        D = torch.eye(4, device=dev).repeat(len(idx), 1, 1)
        D[:, :3, :3] = so3_exp(dw[idx] + 1e-12)
        D[:, :3, 3] = dt[idx]
        return D @ w2cs[idx]

    def render(idx, sh_degree, depth=False):
        out, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), torch.cat([params["sh0"], params["shN"]], 1),
            viewmat(idx), Ks[idx], Ww, Hh, sh_degree=sh_degree, packed=False, absgrad=True,
            near_plane=0.01, far_plane=100.0, render_mode="RGB+ED" if depth else "RGB")
        return out, alpha, info

    rng = np.random.default_rng(0)
    last = time.time()
    for step in range(a.steps):
        i = int(rng.choice(train))
        sh_degree = min(step // 1000, a.sh)
        out, alpha, info = render([i], sh_degree, depth=dmaps is not None)
        gt = imgs[i].float()[None] / 255.0
        pred = colored(out[..., :3], i)
        l1 = (pred - gt).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(pred.permute(0, 3, 1, 2), gt.permute(0, 3, 1, 2)))
        if dmaps is not None:
            rd = F.interpolate(out[..., 3:4].permute(0, 3, 1, 2), size=dmaps.shape[1:], mode="area")[0, 0]
            td, ok = dmaps[i], (cmaps[i] >= cthr) & (dmaps[i] > 0.05)
            if ok.any():
                loss = loss + a.depth_weight * ((rd[ok] - td[ok]).abs() / td[ok]).mean()
        strategy.step_pre_backward(params, opt, state, step, info)
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if pose_opt is not None and step >= 500:  # let the Gaussians settle first
            pose_opt.step()
        if pose_opt is not None:
            pose_opt.zero_grad(set_to_none=True)
        if color_opt is not None:
            color_opt.step()
            color_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opt, state, step, info, packed=False)
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"[splat] step {step}: loss {loss.item():.4f}, {len(params['means'])} Gaussians, "
                  f"{(time.time() - last):.0f} s", flush=True)
            last = time.time()

    if (pose_opt is not None or color_opt is not None) and test:  # held-out frames: align them to the splat
        for k in params:
            params[k].requires_grad_(False)
        groups = ([{"params": [dw, dt], "lr": a.pose_opt * 5}] if pose_opt is not None else []) + \
                 ([{"params": [cA, cb], "lr": a.color_opt}] if color_opt is not None else [])
        topt = torch.optim.Adam(groups)
        for it in range(40 * len(test)):
            i = test[it % len(test)]
            out, _, _ = render([i], a.sh)
            loss = (colored(out[..., :3], i) - imgs[i].float()[None] / 255.0).abs().mean()
            loss.backward()
            topt.step()
            topt.zero_grad(set_to_none=True)
    with torch.no_grad():
        moved = dt.norm(dim=1)
        turned = torch.rad2deg(dw.norm(dim=1))
        psnr_train = []
        for i in train[:: max(1, len(train) // 60)]:
            out, _, _ = render([i], a.sh)
            mse = ((colored(out[..., :3], i) - imgs[i].float()[None] / 255.0) ** 2).mean()
            psnr_train.append(float(-10 * torch.log10(mse)))
        psnr = []
        (a.out / "renders").mkdir(exist_ok=True)
        for k, i in enumerate(test):
            out, _, _ = render([i], a.sh)
            pred = colored(out[..., :3], i)
            mse = ((pred - imgs[i].float()[None] / 255.0) ** 2).mean()
            psnr.append(float(-10 * torch.log10(mse)))
            if k % max(1, len(test) // 8) == 0:  # a few held-out frames: photo | render
                pair = torch.cat([imgs[i].float() / 255.0, pred[0]], 1).clamp(0, 1)
                Image.fromarray((pair.cpu().numpy() * 255).astype(np.uint8)).resize((Ww, Hh // 2)).save(
                    a.out / "renders" / f"{images[i]['name'][:-4]}_{psnr[-1]:.1f}.jpg", quality=85)
        sh = torch.cat([params["sh0"], params["shN"]], 1)
        write_ply(a.out / "splat.ply", params["means"].cpu().numpy(),
                  sh[:, 0].cpu().numpy(),
                  sh[:, 1:].transpose(1, 2).reshape(len(sh), -1).cpu().numpy(),  # 3DGS order: channel-major
                  params["opacities"].cpu().numpy(), params["scales"].cpu().numpy(),
                  F.normalize(params["quats"], dim=-1).cpu().numpy())
        V = torch.cat([viewmat([i]) for i in range(n)]).cpu().numpy()
        (a.out / "cameras.json").write_text(json.dumps([
            {"name": images[i]["name"], "w2c": V[i].round(7).tolist(), "K": Ks[i].cpu().numpy().round(4).tolist(),
             "size": [Ww, Hh], "held_out": i in set(test)} for i in range(n)]))
        label_info = None
        if a.labels and (a.labels / "labels.npz").exists():
            lab = np.load(a.labels / "labels.npz")
            class_names = [c.strip() for c in json.loads((a.labels / "meta.json").read_text())["classes"]]
            ncls = len(class_names)
            means = params["means"]
            votes = torch.zeros(len(means), ncls, device=dev)
            colors_all = torch.cat([params["sh0"], params["shN"]], 1)
            for i in range(n):
                name = images[i]["name"]
                if name not in lab.files:
                    continue
                vm = viewmat([i])
                out, _, _ = rasterization(means, F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
                                          torch.sigmoid(params["opacities"]), colors_all, vm, Ks[[i]], Ww, Hh,
                                          sh_degree=a.sh, packed=False, near_plane=0.01, far_plane=100.0,
                                          render_mode="RGB+ED")
                depth = out[0, ..., 3]
                pc = means @ vm[0, :3, :3].T + vm[0, :3, 3]
                z = pc[:, 2]
                uvw = pc @ Ks[i].T
                u, v = uvw[:, 0] / z.clamp_min(1e-6), uvw[:, 1] / z.clamp_min(1e-6)
                ok = (z > 0.05) & (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
                idx = torch.nonzero(ok).squeeze(1)
                ui, vi = u[idx].long(), v[idx].long()
                vis = (z[idx] - depth[vi, ui]).abs() < 0.04 + 0.02 * z[idx]  # on the visible surface
                idx, ui, vi = idx[vis], ui[vis], vi[vis]
                m = torch.from_numpy(lab[name].astype(np.int64)).to(dev)
                cl = m[(vi * m.shape[0] // Hh).clamp(max=m.shape[0] - 1), (ui * m.shape[1] // Ww).clamp(max=m.shape[1] - 1)]
                votes.index_put_((idx, cl), torch.ones_like(idx, dtype=torch.float32), accumulate=True)
            seen = votes.sum(1)
            cls = votes.argmax(1)
            share = votes.max(1).values / seen.clamp_min(1)
            cls[seen == 0] = -1
            np.savez_compressed(a.out / "gauss_labels.npz", cls=cls.cpu().numpy().astype(np.int16),
                                share=share.cpu().numpy().astype(np.float16), seen=seen.cpu().numpy().astype(np.int32),
                                classes=np.array(class_names))
            label_info = {"labelled": int((seen > 0).sum()), "of": int(len(means))}
    meta = {"steps": a.steps, "gaussians": int(len(params["means"])), "frames": n, "held_out": len(test),
            "labels": label_info,
            "psnr_held_out": round(float(np.mean(psnr)), 2) if psnr else None,
            "psnr_train": round(float(np.mean(psnr_train)), 2),
            "camera_moved_cm_median": round(100 * float(moved[train].median()), 1),
            "camera_turned_deg_median": round(float(turned[train].median()), 2),
            "camera_turned_deg_p90": round(float(turned[train].quantile(0.9)), 2), "size": [Ww, Hh],
            "gpu": torch.cuda.get_device_name(), "seconds": round(time.time() - t_start, 1)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[splat] " + json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
