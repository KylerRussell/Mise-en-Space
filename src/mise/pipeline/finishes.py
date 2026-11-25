"""Surface finishes from the capture: floor and wall textures and colors, at real scale.

A patch is a rectangle on a known plane in the leveled metric frame (a floor square,
a stretch of wall). Every texel is a 3D point; project it through the solved camera
(lens distortion included) and sample the frame. The result is a fronto-parallel,
metric texture: 1 m of floor is ``px`` pixels, whichever way the camera looked.

Needs opencv and a solved reconstruction (the `poses` extra).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Patch:
    origin: np.ndarray  # 3D corner in the leveled frame
    u: np.ndarray  # unit vector along the texture's width
    v: np.ndarray  # unit vector along the texture's height (image rows go along -v)
    size_m: tuple[float, float]

    def grid(self, px_per_m: int) -> np.ndarray:
        w, h = round(self.size_m[0] * px_per_m), round(self.size_m[1] * px_per_m)
        s = (np.arange(w) + 0.5) / px_per_m
        t = self.size_m[1] - (np.arange(h) + 0.5) / px_per_m  # row 0 is the top (far / high) edge
        S, T = np.meshgrid(s, t)
        return self.origin + S[..., None] * self.u + T[..., None] * self.v


def floor_patch(x: float, y: float, size_m: float = 1.0) -> Patch:
    """A floor square centered at (x, y), texture up = +y."""
    h = size_m / 2
    return Patch(np.array([x - h, y - h, 0.0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), (size_m, size_m))


def wall_patch(a, b, along: float, z0: float, width_m: float, height_m: float, inward_normal, off: float = 0.0) -> Patch:
    """A rectangle on the wall a->b starting `along` m from a, from height z0 up; seen from the room."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    n = np.asarray(inward_normal, float)
    # seen from inside, texture right is (-n.y, n.x)... pick u so that u x v points into the room
    u = np.array([d[0], d[1], 0.0])
    if np.cross(u, [0, 0, 1])[:2] @ n > 0:  # u x z must point out of the wall, toward the viewer's back
        u = -u
        start = a + d * (along + width_m)
    else:
        start = a + d * along
    origin = np.array([start[0], start[1], z0]) + np.array([n[0], n[1], 0]) * off
    return Patch(origin, u, np.array([0.0, 0, 1]), (width_m, height_m))


def sample_patch(patch: Patch, image: np.ndarray, project, px_per_m: int = 256) -> tuple[np.ndarray, float]:
    """Resample `image` onto the patch. `project(Nx3 leveled points) -> (Nx2 pixels, Nx1 depth)`.

    Returns (texture, fraction of texels that landed inside the frame in front of the camera).
    """
    import cv2

    P = patch.grid(px_per_m)
    shape = P.shape[:2]
    uv, depth = project(P.reshape(-1, 3))
    uv = np.asarray(uv, np.float32).reshape(*shape, 2)
    H, W = image.shape[:2]
    ok = (depth.reshape(shape) > 0.05) & (uv[..., 0] >= 0) & (uv[..., 0] < W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] < H - 1)
    tex = cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return tex, float(ok.mean())


def colmap_projector(rec, image, leveled):
    """project() for one solved image, taking points in `leveled`'s frame."""
    cam = rec.cameras[image.camera_id]
    M = image.cam_from_world().matrix()

    def project(P: np.ndarray):
        Pm = ((P - leveled.t) @ leveled.R) / leveled.scale
        pc = Pm @ M[:, :3].T + M[:, 3]
        depth = pc[:, 2].copy()
        pc[depth <= 1e-6, 2] = 1e-6
        return np.asarray(cam.img_from_cam(pc)), depth

    return project


def view_quality(patch: Patch, cam_center: np.ndarray, normal) -> float:
    """Higher is better: close and head-on (cosine of the view angle over distance)."""
    c = patch.origin + patch.u * patch.size_m[0] / 2 + patch.v * patch.size_m[1] / 2
    d = np.asarray(cam_center) - c
    dist = float(np.linalg.norm(d))
    return float(abs(d @ np.asarray(normal, float)) / dist) / max(dist, 0.3)


def tileable(tex: np.ndarray) -> np.ndarray:
    """Hide the seams when the texture repeats: crossfade with a half-offset copy of itself."""
    h, w = tex.shape[:2]
    rolled = np.roll(np.roll(tex, h // 2, axis=0), w // 2, axis=1).astype(np.float32)
    y = 1 - np.abs(np.linspace(-1, 1, h))
    x = 1 - np.abs(np.linspace(-1, 1, w))
    wgt = np.minimum.outer(y, x)  # 1 in the middle, 0 at the edges (where the original seams)
    wgt = np.clip(wgt * 2.5, 0, 1)[..., None]
    out = tex.astype(np.float32) * wgt + rolled * (1 - wgt)
    return np.clip(out, 0, 255).astype(np.uint8)


def median_color(tex: np.ndarray, trim: float = 0.15) -> str:
    """Robust color of a patch as #rrggbb (BGR input, as OpenCV loads it): trims dark and bright outliers."""
    px = tex.reshape(-1, 3).astype(float)
    lum = px.mean(1)
    lo, hi = np.quantile(lum, [trim, 1 - trim])
    keep = px[(lum >= lo) & (lum <= hi)]
    b, g, r = np.median(keep, axis=0)
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


# ---- textures for the editor -------------------------------------------------------------

def hex_to_bgr(c: str) -> np.ndarray:
    c = c.lstrip("#")
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], float)


def bgr_to_hex(b: np.ndarray) -> str:
    b = np.clip(np.round(b), 0, 255).astype(int)
    return f"#{b[2]:02x}{b[1]:02x}{b[0]:02x}"


def gain(color: str, k: float) -> str:
    """Brighten a sampled color: phone video exposes white rooms down to mid-grey."""
    return bgr_to_hex(hex_to_bgr(color) * k)


def flat_field(tex: np.ndarray, sigma_px: float) -> np.ndarray:
    """Remove lighting falloff (low frequencies), keep the surface detail around 1.0."""
    import cv2

    f = tex.astype(np.float32) + 1
    low = cv2.GaussianBlur(f, (0, 0), sigma_px)
    return f / low


def tint(detail: np.ndarray, color: str, contrast: float = 1.0) -> np.ndarray:
    """Detail map (around 1.0) times a color, as an 8-bit BGR texture."""
    d = 1 + (detail - 1) * contrast
    return np.clip(d * hex_to_bgr(color), 0, 255).astype(np.uint8)


def plank_texture(detail: np.ndarray, color: str, plank_m: float, planks: int = 6, px_per_m: int = 512,
                  seed: int = 0, seam: float = 0.82) -> tuple[np.ndarray, float]:
    """Tileable floor of `planks` boards (running along v) with staggered end joints.

    detail: a flat-fielded sample of the real floor (grain, tone variation).
    Returns (texture, tile size in meters).
    """
    import cv2

    size = plank_m * planks
    n = round(size * px_per_m)
    d = cv2.resize(detail, (n, n), interpolation=cv2.INTER_AREA)
    tex = tint(d, color).astype(np.float32)
    tex = tileable(tex.astype(np.uint8)).astype(np.float32)
    rng = np.random.default_rng(seed)
    pw = n / planks
    for i in range(planks):
        x0 = round(i * pw)
        tex[:, x0:x0 + 2] *= seam  # long seam
        tex[:, round((i + 0.5) * pw) - 1:round((i + 0.5) * pw) + 1] *= 0.97  # faint grain line
        y = int(rng.uniform(0, n))  # one end joint per board per tile, staggered
        x1 = round((i + 1) * pw)
        tex[y:y + 2, x0:x1] *= seam
        tex[:, x0:x1] *= rng.uniform(0.95, 1.05)  # boards differ a little in tone
    return np.clip(tex, 0, 255).astype(np.uint8), size


def block_texture(color: str, detail: np.ndarray | None = None, block_m: tuple[float, float] = (0.406, 0.203),
                  px_per_m: int = 512, joint_m: float = 0.01, joint: float = 0.88) -> tuple[np.ndarray, float]:
    """Painted concrete block in running bond, two blocks by four courses per tile."""
    import cv2

    w, h = block_m
    size = 2 * w
    n = round(size * px_per_m)
    if detail is not None:
        tex = tint(cv2.resize(detail, (n, n), interpolation=cv2.INTER_AREA), color, 0.6).astype(np.float32)
        tex = tileable(tex.astype(np.uint8)).astype(np.float32)
    else:
        tex = np.ones((n, n, 3), np.float32) * hex_to_bgr(color)
    j = max(2, round(joint_m * px_per_m))
    bw, bh = n // 2, round(h * px_per_m)
    for c in range(int(np.ceil(n / bh))):
        y = c * bh
        tex[y:y + j] *= joint
        off = 0 if c % 2 == 0 else bw // 2
        for x in range(off % bw, n + bw, bw):
            if x < n:
                tex[y:y + bh, x:x + j] *= joint
    return np.clip(tex, 0, 255).astype(np.uint8), size


# ---- automatic finishes for a scanned scene -----------------------------------------------

def _best_sample(rec, leveled, patch: Patch, normal, frames_dir, px_per_m: int, top: int = 3):
    """Sample the patch from the few views that see it best (closest, most head-on, fully in frame)."""
    import cv2

    images = sorted(rec.images.values(), key=lambda im: im.name)
    centers = leveled.apply(np.array([im.projection_center() for im in images]))
    scored = sorted(range(len(images)), key=lambda k: -view_quality(patch, centers[k], normal))
    out = []
    for k in scored[:12]:
        im = images[k]
        img = cv2.imread(str(Path(frames_dir) / im.name))
        if img is None:
            continue
        # the solve may have used a smaller image than the file: project at the camera's size
        cam = rec.cameras[im.camera_id]
        if img.shape[1] != cam.width:
            img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_AREA)
        tex, ok = sample_patch(patch, img, colmap_projector(rec, im, leveled), px_per_m)
        if ok > 0.98:
            out.append(tex)
        if len(out) >= top:
            break
    if not out:
        return None
    return np.median(np.stack(out), axis=0).astype(np.uint8)  # median of views: drops a passer-by or a glare


def auto_finishes(rec, leveled, scene, frames_dir, scene_dir, exposure: float = 1.5, px_per_m: int = 256) -> int:
    """Floor and wall finishes for every space, sampled from the capture. Returns spaces finished.

    The floor patch is taken at the middle of the space, the wall patch on its longest wall
    between knee and head height (above furniture, below upper cabinets). Writes textures to
    scene_dir/finishes/ and sets ``space.finish``.
    """
    import cv2

    from ..scene import geometry as g
    from ..scene.models import Finish, Surface

    out_dir = Path(scene_dir) / "finishes"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for sp in scene.room.spaces:
        poly = np.array(sp.polygon, float)
        floor = walls = None
        # floor: the free point farthest from the walls (a pole of inaccessibility, on a grid)
        lo, hi = poly.min(0), poly.max(0)
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
        pts = np.c_[gx.ravel(), gy.ravel()]
        inside = np.array([g.point_in_polygon(p, poly) for p in pts])
        if inside.any():
            pts = pts[inside]
            d = np.min([g.points_segment_distance(pts, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))], axis=0)
            c = pts[d.argmax()]
            size = float(min(0.8, 2 * d.max() * 0.9))
            if size >= 0.3:
                tex = _best_sample(rec, leveled, floor_patch(c[0], c[1], size), (0, 0, 1), frames_dir, px_per_m)
                if tex is not None:
                    color = gain(median_color(tex), exposure)
                    detail = tileable(tint(flat_field(tex, px_per_m * size / 4), "#808080"))
                    name = f"floor_{sp.id}.jpg"
                    cv2.imwrite(str(out_dir / name), tint(detail.astype(np.float32) / 128.0, color))
                    floor = Surface(color=color, texture=f"finishes/{name}", size_m=round(size, 3))
        ws = [w for w in scene.room.walls if w.space == sp.id]
        if ws:
            w = max(ws, key=lambda w: w.length)
            width = min(0.8, w.length - 0.2)
            n = g.wall_inward_normal(w, scene.room)
            if width >= 0.3:
                patch = wall_patch(w.a, w.b, (w.length - width) / 2, 0.95, width, 0.5, n, off=0.0)
                tex = _best_sample(rec, leveled, patch, (n[0], n[1], 0), frames_dir, px_per_m)
                if tex is not None:
                    walls = Surface(color=gain(median_color(tex), exposure))
        if floor or walls:
            ceiling = Surface(color=gain(walls.color, 1.08)) if walls else None
            sp.finish = Finish(floor=floor, walls=walls, ceiling=ceiling)
            done += 1
    return done
