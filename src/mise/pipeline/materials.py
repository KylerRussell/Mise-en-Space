"""The empty home's surfaces as materials: what each floor, wall and ceiling is, its color, its pattern.

The shell painted from the photos (shell.py) looks like the photos, patches and all. Here each
surface is boiled down to a material the owner can change: paint and its color, wood planks
(their color, width and which way they run), tile, carpet, painted block, a popcorn ceiling.
Colors are what the surface looks like on average in the capture; the light (brighter by the
window, darker in the corners) is kept apart as a shading map, so a repainted wall is still lit
like the room. The empty home is then made from the materials (`render_shell`): change a color
or a floor in the scene and render again, no photos needed.

Each room's walls share one paint unless a wall clearly differs in hue (an accent wall) or in
material (painted block): those become wall overrides. The pattern is read off one close,
square-on, unobstructed view of a clear patch, at about 2.5 mm a pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import finishes as fin
from . import ply
from .shell import CEILING, FLOOR, WALL, _inpaint, _surfaces


def _lab(rgb: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(rgb, np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _hex(rgb: np.ndarray) -> str:
    c = np.clip(np.round(np.asarray(rgb) * 255), 0, 255).astype(int)
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def _clean(s: dict) -> np.ndarray:
    """Texels of a painted surface that are the surface itself: seen, nothing in front, not a shadow."""
    m = s["seen"] & s["inside"] & ~s.get("hidden", np.zeros_like(s["seen"]))
    if m.sum() < 20:
        return m
    lum = s["rgb"].mean(-1)
    return m & (lum > 0.7 * np.median(lum[m]))


def _color(s_list: list[dict]) -> tuple[np.ndarray | None, int]:
    px = np.concatenate([s["rgb"][_clean(s)] for s in s_list]) if s_list else np.zeros((0, 3))
    if len(px) < 50:
        return None, len(px)
    lum = px.mean(1)
    lo, hi = np.quantile(lum, [0.15, 0.85])
    return np.median(px[(lum >= lo) & (lum <= hi)], axis=0), len(px)


# ---- a close look at a clear patch -------------------------------------------------------

def clear_patches(s: dict, cell: float, size_m: float = 0.6, min_m: float = 0.3, k: int = 8) -> list[tuple[np.ndarray, float]]:
    """Up to k clear squares spread over the surface: (center in the grid's cells, side m), largest first."""
    import cv2

    good = _clean(s).astype(np.uint8)
    if good.sum() == 0:
        return []
    d = cv2.distanceTransform(np.pad(good, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
    out = []
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(d), d.shape)
        side = min(size_m, 2 * d[i, j] * cell)
        if side < min_m:
            break
        out.append((np.array([i, j]), float(side)))
        r = int(side / cell)
        d[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] = 0  # the next one elsewhere
    return out


def clear_patch(s: dict, cell: float, size_m: float = 0.6, min_m: float = 0.3) -> tuple[np.ndarray, float] | None:
    c = clear_patches(s, cell, size_m, min_m, k=1)
    return c[0] if c else None


def sample_patch(s: dict, center: np.ndarray, side: float, cell: float, cameras: list[dict], frames_dir: Path,
                 labels: dict[str, np.ndarray], classes: list[str], depth: dict[str, np.ndarray] | None,
                 px_per_m: int = 400, with_score: bool = False):
    """The patch, fronto-parallel at px_per_m, from the single best view that sees all of it clean (RGB 0-1)."""
    import cv2

    allowed = {"wall": WALL, "floor": FLOOR, "ceiling": CEILING}[s["kind"]]
    ok_cls = np.array([c.strip() in allowed for c in classes] + [False])
    n = int(round(side * px_per_m))
    t = (np.arange(n) + 0.5) / px_per_m - side / 2
    A, B = np.meshgrid(t, t[::-1])  # row 0: the far / high edge
    c3 = s["xyz"][int(center[0]), int(center[1])]
    P = c3 + A[..., None] * s["u"] + B[..., None] * s["v"]
    pts = P.reshape(-1, 3)
    corners = np.array([c3 + a * s["u"] + b * s["v"] for a in (-side / 2, side / 2) for b in (-side / 2, side / 2)])
    best = None
    for cam in cameras:
        name = cam["name"]
        if name not in labels:
            continue
        w2c, K = np.array(cam["w2c"], float), np.array(cam["K"], float)
        W, H = cam["size"]
        Pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
        if (Pc[:, 2] < 0.3).any():
            continue
        uv = Pc[:, :2] / Pc[:, 2:] * np.diag(K)[:2] + K[:2, 2]
        if (uv < 8).any() or (uv[:, 0] > W - 8).any() or (uv[:, 1] > H - 8).any():
            continue
        C = -w2c[:3, :3].T @ w2c[:3, 3]
        view = C - c3
        dist = float(np.linalg.norm(view))
        cos = float(view @ s["normal"] / dist)
        if cos < 0.35 or dist > 4.5:
            continue
        score = cos ** 2 / max(dist, 0.5) ** 2
        if best is not None and score <= best[0]:  # (quality below only lowers it)
            continue
        Pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        uv = Pc[:, :2] / Pc[:, 2:] * np.diag(K)[:2] + K[:2, 2]
        lab = labels[name]
        lh, lw = lab.shape
        lv = lab[np.clip((uv[:, 1] * lh / H).astype(int), 0, lh - 1), np.clip((uv[:, 0] * lw / W).astype(int), 0, lw - 1)]
        q_lab = float(ok_cls[np.minimum(lv, len(classes))].mean())
        if q_lab < 0.9:
            continue
        q_dep = 1.0
        if depth is not None and name in depth:
            dm = depth[name]
            dh, dw = dm.shape
            dv = dm[np.clip((uv[:, 1] * dh / H).astype(int), 0, dh - 1), np.clip((uv[:, 0] * dw / W).astype(int), 0, dw - 1)]
            q_dep = float((np.abs(Pc[:, 2] - dv) < np.maximum(0.15, 0.08 * dv)).mean())
            if q_dep < 0.8:
                continue
        score *= q_lab ** 4 * q_dep ** 2  # all of it the surface, and nothing in front
        if best is None or score > best[0]:
            best = (score, name, uv.reshape(n, n, 2).astype(np.float32))
    if best is None:
        return (None, 0.0) if with_score else None
    img = cv2.imread(str(Path(frames_dir) / best[1]))
    if img is None:
        return (None, 0.0) if with_score else None
    W, H = next(c["size"] for c in cameras if c["name"] == best[1])
    uv = best[2] * np.array([img.shape[1] / W, img.shape[0] / H], np.float32)
    out = cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return (out, best[0]) if with_score else out


# ---- what the patch is -------------------------------------------------------------------

def _period(profile: np.ndarray, px_per_m: int, lo_m: float, hi_m: float) -> tuple[float | None, float]:
    """Strongest repeat of a 1-D profile between lo_m and hi_m: (period m, strength 0-1)."""
    sig = max(1.0, lo_m * px_per_m / 2)  # high-pass: slow trends (light, a hanger) out, the repeat kept
    k = np.exp(-0.5 * (np.arange(-int(3 * sig), int(3 * sig) + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.pad(profile, len(k) // 2, mode="reflect")
    p = profile - np.convolve(pad, k, "valid")
    if p.std() < 1e-6:
        return None, 0.0
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / ac[0]
    a, b = int(lo_m * px_per_m), min(int(hi_m * px_per_m), len(ac) - 2)
    if b <= a + 2:
        return None, 0.0
    seg = ac[a:b]
    k = int(np.argmax(seg))
    if k in (0, len(seg) - 1):  # not a peak, a slope
        return None, 0.0
    return (a + k) / px_per_m, float(seg[k])


def classify(patch: np.ndarray, kind: str, px_per_m: int = 400) -> dict:
    """Material and pattern of a fronto-parallel patch (RGB 0-1; rows go down the surface's v)."""
    import cv2

    g = patch.mean(-1).astype(np.float32)
    flat = g / np.maximum(cv2.GaussianBlur(g, (0, 0), px_per_m * 0.08), 1e-3)  # lighting out, ~8 cm and finer kept
    fine = float((flat - cv2.GaussianBlur(flat, (0, 0), 2.0)).std())  # grain, carpet pile, popcorn
    coarse = float(cv2.GaussianBlur(flat, (0, 0), 4.0).std())  # boards, joints, tiles
    gx = cv2.Sobel(cv2.GaussianBlur(flat, (0, 0), 1.5), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(flat, (0, 0), 1.5), cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 6) for v in (gx * gx, gy * gy, gx * gy))
    e = jxx + jyy
    coh = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2).sum() / max(e.sum(), 1e-9))
    th = 0.5 * np.arctan2(2 * (jxy * e).sum(), ((jxx - jyy) * e).sum())  # gradient direction (image x right, y down)
    grain_img = np.degrees(th) + 90  # grain runs across the gradient
    # in the surface's (u, v) frame image y runs along -v
    grain_uv = (-grain_img) % 180
    rows_p, rows_s = _period(flat.mean(1), px_per_m, 0.12, 0.45)  # repeats down the surface (courses, tile rows)
    cols_p, cols_s = _period(flat.mean(0), px_per_m, 0.12, 0.45)
    out = {"fine": round(fine, 4), "coarse": round(coarse, 4), "coherence": round(coh, 3),
           "grain_deg": round(float(grain_uv), 1), "rows": [rows_p, round(rows_s, 2)], "cols": [cols_p, round(cols_s, 2)]}
    if kind == "floor":
        if rows_s > 0.35 and cols_s > 0.35 and abs((rows_p or 0) - (cols_p or 0)) < 0.05:
            out.update(kind="tile", module_m=round((rows_p + cols_p) / 2, 3))
        elif coh > 0.35 and coarse > 0.008:
            # boards: their width is the repeat across the grain
            R = cv2.getRotationMatrix2D((flat.shape[1] / 2, flat.shape[0] / 2), -float(grain_img) + 90, 1.0)
            rot = cv2.warpAffine(flat, R, flat.shape[::-1], borderMode=cv2.BORDER_REFLECT)
            m = flat.shape[0] // 4
            pw, ps = _period(rot[m:-m, m:-m].mean(0), px_per_m, 0.07, 0.3)
            out.update(kind="wood", direction_deg=round(float(grain_uv), 1), module_m=round(pw, 3) if ps > 0.1 else 0.18)
        elif fine > 0.035:
            out.update(kind="carpet")
        else:
            out.update(kind="other")
    elif kind == "wall":
        # Painted block shows as weak repeats that door frames and hangers mimic as well; not told
        # apart reliably from one patch, so walls are paint unless set otherwise (kind "block").
        out.update(kind="paint")
    else:
        out.update(kind="popcorn" if fine > 0.03 else "paint")
    return out


def block_evidence(patch: np.ndarray, px_per_m: int = 400) -> float:
    """Painted concrete block: horizontal joints repeating at one course (8 in, 0.19-0.225 m).
    Joints span the row, so the row profile is the median across it of the vertical gradient.
    Returns how strongly the patch repeats at that course (0: not at all)."""
    import cv2

    g = patch.mean(-1).astype(np.float32)
    flat = g / np.maximum(cv2.GaussianBlur(g, (0, 0), 24), 1e-3)
    gy = np.abs(cv2.Sobel(cv2.GaussianBlur(flat, (0, 0), 1.2), cv2.CV_32F, 0, 1))
    per, strength = _period(np.median(gy, axis=1), px_per_m, 0.15, 0.26)
    return float(strength) if per is not None and 0.19 <= per <= 0.225 else 0.0


BLOCK_BEST, BLOCK_SECOND = 0.40, 0.24  # set on the home's walls in two runs: block 0.41-0.51 / 0.25-0.46,
# drywall at most 0.39 alone or 0.30 / 0.22


def block_hits(s: dict, cell: float, cameras, frames_dir, labels, classes, depth, k: int = 12) -> tuple[bool, list, np.ndarray | None]:
    """Is this wall painted block? Up to k patches spread over it; most views of a wall are too far,
    blurred or half hidden to show joints, so it takes one clear view of the course and a second
    that agrees (one alone can be a door frame or a hanger). Returns (block, strengths, best patch)."""
    found = []
    for c, side in clear_patches(s, cell, size_m=0.5, k=k):
        p = sample_patch(s, c, side, cell, cameras, frames_dir, labels, classes, depth)
        if p is not None:
            found.append((block_evidence(p), p))
    found.sort(key=lambda t: -t[0])
    st = [round(e, 2) for e, _ in found]
    block = len(st) >= 2 and st[0] >= BLOCK_BEST and st[1] >= BLOCK_SECOND
    return block, st[:3], (found[0][1] if found else None)


def _texture(kind: str, patch: np.ndarray | None, color: str, info: dict, out_path: Path) -> tuple[str | None, float]:
    """A tileable texture image for the material (written as out_path), and the size it covers."""
    import cv2

    if patch is None or kind == "paint":
        return None, 1.0
    bgr = cv2.cvtColor((np.clip(patch, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    px = 400
    side = patch.shape[0] / px
    detail = fin.flat_field(cv2.GaussianBlur(bgr, (0, 0), 0.8), px * side / 4)
    if kind == "wood":
        # boards run along v in plank_texture: turn the sample so its grain runs along image y
        g = info.get("grain_deg", 90.0)
        R = cv2.getRotationMatrix2D((detail.shape[1] / 2, detail.shape[0] / 2), 90 - g, 1.0)
        detail = cv2.warpAffine(detail, R, detail.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
        w = info.get("module_m") or 0.19
        # boards about 1.2 m long, as laminate and engineered boards come: the tile is that long
        tex, size = fin.plank_texture(detail, color, w, planks=max(6, int(np.ceil(1.2 / w))), px_per_m=256)
    elif kind == "block":
        c = info.get("module_m") or 0.203
        # painted block: the paint fills the joints, which show as faint lines, not mortar
        tex, size = fin.block_texture(color, detail, block_m=(2 * c, c), px_per_m=256, joint=0.94)
    else:  # carpet, tile, popcorn, other: the sample itself, tiled
        tex, size = fin.tileable(fin.tint(detail, color, 0.8)), side
    cv2.imwrite(str(out_path), tex)
    return out_path.name, round(float(size), 3)


# ---- estimate ------------------------------------------------------------------------------

def estimate(scene, surfaces: list[dict], cell: float, cameras: list[dict], frames_dir: Path,
             labels: dict[str, np.ndarray], classes: list[str], depth: dict[str, np.ndarray] | None,
             scene_dir: Path, say=print) -> dict:
    """Set every space's finish (floor, walls, wall overrides, ceiling) from the painted shell and a
    close look at a patch of each; write the textures and the shading maps. Returns a report."""
    from ..scene.models import Finish, Surface

    scene_dir = Path(scene_dir)
    (scene_dir / "finishes").mkdir(parents=True, exist_ok=True)
    by_id = {s["id"]: s for s in surfaces}
    walls_of = {}
    for w in scene.room.walls:
        walls_of.setdefault(w.space, []).append(w.id)
    shading, report = {}, {}
    # phone video exposes a white room down to mid-grey: one gain for the whole home, from its
    # brightest surfaces (walls and ceilings), puts colors back near what the paint is. The light
    # stays in the shading maps (they absorb the gain), so the render looks as captured.
    lit = np.concatenate([s["rgb"][_clean(s)].mean(-1) for s in surfaces if s["kind"] != "floor"] or [np.ones(1)])
    gain = float(np.clip(0.9 / max(np.quantile(lit, 0.9), 1e-3), 1.0, 2.5)) if len(lit) > 100 else 1.0

    def look(s: dict) -> tuple[dict, np.ndarray | None]:
        best = (None, 0.0)
        for c, side in clear_patches(s, cell):  # the patch with the best view of the few spread over it
            got = sample_patch(s, c, side, cell, cameras, frames_dir, labels, classes, depth, with_score=True)
            if got[0] is not None and got[1] * side > best[1]:
                best = (got[0], got[1] * side)
        if best[0] is None:
            return {"kind": None}, None
        return classify(best[0], s["kind"]), best[0]

    def surface(kind_s: str, ss: list[dict], name: str, default: str) -> tuple[Surface | None, dict]:
        col, n = _color(ss)
        if col is None or n * cell * cell < 0.5:  # too little seen to say: borrowed below
            return None, {"seen_texels": n}
        col = np.clip(col * gain, 0, 1)
        best = max(ss, key=lambda s: int(_clean(s).sum()))
        info, patch = look(best)
        kind = info.get("kind") or default
        if kind == "wood":
            sp_ = next(x for x in scene.room.spaces if x.id == best["space"])
            axis = _wall_axis(np.array(sp_.polygon, float))
            d = info.get("direction_deg", axis)
            for a in (axis, axis + 90, axis + 180, axis - 90):
                if abs(((d - a) + 90) % 180 - 90) < 20:
                    info["direction_deg"] = round(a % 180, 1)
                    break
            info["module_m"] = float(np.clip(info.get("module_m") or 0.19, 0.12, 0.25))
        hexc = _hex(col)
        tex, size = _texture(kind, patch, hexc, info, scene_dir / "finishes" / f"{name}.jpg") \
            if kind != "paint" else (None, 1.0)
        sf = Surface(color=hexc, texture=f"finishes/{tex}" if tex else None, size_m=size, kind=kind,
                     direction_deg=info.get("direction_deg"), module_m=info.get("module_m"))
        return sf, {k: v for k, v in info.items() if k != "kind"} | {"kind": kind, "color": hexc}

    for sp in scene.room.spaces:
        f = Finish()
        rep = {}
        fl = by_id.get(f"{sp.id}.floor")
        if fl is not None:
            f.floor, rep["floor"] = surface("floor", [fl], f"floor_{sp.id}", "other")
        ce = by_id.get(f"{sp.id}.ceiling")
        if ce is not None:
            f.ceiling, rep["ceiling"] = surface("ceiling", [ce], f"ceiling_{sp.id}", "paint")
        ws = [by_id[w] for w in walls_of.get(sp.id, []) if w in by_id]
        if ws:
            f.walls, rep["walls"] = surface("wall", ws, f"walls_{sp.id}", "paint")
            if f.walls is not None:
                base = _lab(_rgb(f.walls.color))[0]
                for s in ws:
                    col, n = _color([s])
                    if col is None or n * cell * cell < 0.8:
                        continue
                    col = np.clip(col * gain, 0, 1)
                    d_ab = float(np.linalg.norm(_lab(col)[0][1:] - base[1:]))
                    is_block, hits, patch = block_hits(s, cell, cameras, frames_dir, labels, classes, depth)
                    kind = "block" if is_block else "paint"
                    if d_ab > 6.0 or kind != f.walls.kind:  # another paint, or painted block
                        hexc = _hex(col)
                        info = {"module_m": 0.203}
                        tex, size = _texture(kind, patch, hexc, info, scene_dir / "finishes" / f"wall_{s['id']}.jpg") \
                            if kind == "block" else (None, 1.0)
                        f.wall_overrides[s["id"]] = Surface(color=hexc, texture=f"finishes/{tex}" if tex else None,
                                                            size_m=size, kind=kind,
                                                            module_m=0.203 if kind == "block" else None)
                        rep.setdefault("wall_overrides", {})[s["id"]] = {"kind": kind, "color": hexc,
                                                                         "d_ab": round(d_ab, 1), "block_hits": hits}
        sp.finish = f
        report[sp.name] = rep
        say(f"{sp.name}: " + ", ".join(f"{k} {v.get('kind')} {v.get('color', '')}" for k, v in rep.items()
                                        if isinstance(v, dict) and "kind" in v))
    # what was seen too little to say borrows: a ceiling from the home's other ceilings, a floor or
    # walls from the nearest room that has them (a closet's floor is its room's)
    cent = {sp.id: np.array(sp.polygon, float).mean(0) for sp in scene.room.spaces}
    for part in ("floor", "walls", "ceiling"):
        have = [sp for sp in scene.room.spaces if sp.finish and getattr(sp.finish, part)]
        for sp in scene.room.spaces:
            if not have or (sp.finish and getattr(sp.finish, part)):
                continue
            if part == "ceiling":
                cols = np.array([_rgb(getattr(x.finish, part).color) for x in have])
                src = getattr(have[0].finish, part).model_copy(update={"color": _hex(np.median(cols, 0))})
            else:
                near = min(have, key=lambda x: float(np.linalg.norm(cent[x.id] - cent[sp.id])))
                src = getattr(near.finish, part).model_copy()
            sp.finish = sp.finish or Finish()
            setattr(sp.finish, part, src)
            report.setdefault(sp.name, {})[part] = {"kind": src.kind, "color": src.color, "borrowed": True}

    # one material, one color: rooms whose floors (walls, ceilings) are the same kind and hue are the
    # same floor (paint) lit differently. They share the color of the best-seen of them; how each
    # room is lit is left to its shading.
    seen_area = {sp.id: {} for sp in scene.room.spaces}
    for x in surfaces:
        part = {"floor": "floor", "ceiling": "ceiling", "wall": "walls"}[x["kind"]]
        seen_area[x["space"]][part] = seen_area[x["space"]].get(part, 0) + int(_clean(x).sum())
    for part in ("floor", "walls", "ceiling"):
        todo = [sp for sp in scene.room.spaces if sp.finish and getattr(sp.finish, part)]
        while todo:
            lead = max(todo, key=lambda sp: seen_area[sp.id].get(part, 0))
            lf = getattr(lead.finish, part)
            la = _lab(_rgb(lf.color))[0]
            same = [sp for sp in todo if getattr(sp.finish, part).kind == lf.kind
                    and np.linalg.norm(_lab(_rgb(getattr(sp.finish, part).color))[0][1:] - la[1:]) < 6.0]
            for sp in same:
                if sp is not lead:
                    setattr(sp.finish, part, lf.model_copy())
                    report.setdefault(sp.name, {}).setdefault(part, {})["same_as"] = lead.name
            todo = [sp for sp in todo if sp not in same]

    # trim, windows and closed doors
    from .trim import estimate_trim, find_openings, vote

    report["trim"] = estimate_trim(scene, surfaces, cell, gain)
    found = find_openings(scene, surfaces, vote(surfaces, cameras, labels, classes, depth, say=say), cell, gain)
    report["openings"] = [(o.id, o.kind, o.wall, round(o.width, 2), round(o.bottom, 2), round(o.top, 2), o.color)
                          for o in found]
    say(f"trim {report['trim']}; {sum(o.kind == 'window' for o in found)} windows, "
        f"{sum(o.kind == 'door' for o in found)} closed doors")

    # the light on each surface, relative to its material's color: kept apart from the color
    for s in surfaces:
        sf = _material_of(scene, s)
        if sf is None or sf.color is None:
            continue
        lit = _inpaint(s, cell)
        import cv2

        low = cv2.GaussianBlur(lit.astype(np.float32), (0, 0), 0.3 / cell)
        shade = low.mean(-1) / max(float(_rgb(sf.color).mean()), 1e-3)
        shading[s["id"]] = np.clip(shade, 0.2, 1.6).astype(np.float16)
    np.savez_compressed(scene_dir / "finishes" / "shading.npz", cell=np.array(cell), **shading)
    report["exposure_gain"] = round(gain, 2)
    (scene_dir / "scene.json").write_text(scene.model_dump_json(indent=2))
    return report


def _wall_axis(poly: np.ndarray) -> float:
    """The room's main wall direction (degrees, 0-90): length-weighted, folded to a quarter turn."""
    e = np.roll(poly, -1, 0) - poly
    L = np.linalg.norm(e, axis=1)
    a = np.arctan2(e[:, 1], e[:, 0]) * 4  # four-fold: walls at right angles agree
    m = np.arctan2((L * np.sin(a)).sum(), (L * np.cos(a)).sum()) / 4
    return float(np.degrees(m) % 90)


def _material_of(scene, s: dict):
    for sp in scene.room.spaces:
        if s["kind"] in ("floor", "ceiling") and s["id"] == f"{sp.id}.{s['kind']}":
            return getattr(sp.finish, s["kind"], None) if sp.finish else None
    if s["kind"] == "wall":
        sp_id = s.get("space")
        sp = next((x for x in scene.room.spaces if x.id == sp_id), None)
        if sp is None or sp.finish is None:
            return None
        return sp.finish.wall_overrides.get(s["id"]) or sp.finish.walls
    return None


# ---- render --------------------------------------------------------------------------------

def _albedo(sf, s: dict, cell: float, scene_dir: Path, rng: np.random.Generator) -> np.ndarray:
    import cv2

    h, w = s["xyz"].shape[:2]
    color = _rgb(sf.color) if sf.color else np.array([0.8, 0.8, 0.8])
    if not sf.texture or not (scene_dir / sf.texture).exists():
        return np.broadcast_to(color, (h, w, 3)) * (1 + rng.normal(0, 0.006, (h, w, 1)))
    tex = cv2.cvtColor(cv2.imread(str(scene_dir / sf.texture)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    n = max(2, int(round(sf.size_m / cell)))
    small = cv2.resize(tex, (n, n), interpolation=cv2.INTER_AREA)
    small = small * (color / np.maximum(small.reshape(-1, 3).mean(0), 1e-3))  # the color is the surface's, now
    o = s["xyz"][0, 0] - (s["u"] + s["v"]) * cell / 2
    rel = s["xyz"] - o
    a, b = rel @ s["u"], rel @ s["v"]
    if s["kind"] == "floor":  # world x, y; planks along direction_deg
        x, y = s["xyz"][..., 0], s["xyz"][..., 1]
        th = np.radians(sf.direction_deg or 0.0)
        along, across = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
        a, b = across, along
    col = np.floor(a / cell).astype(int) % n
    row = (n - 1 - np.floor(b / cell).astype(int)) % n  # texture row 0 is its top / far edge
    return small[row, col]


def ceiling_path(path: Path) -> Path:
    """Where a shell's ceilings go: next to it (splats/empty.ply -> splats/empty-ceiling.ply)."""
    return Path(path).with_name(Path(path).stem + "-ceiling.ply")


def write_shell(scene, scene_dir: Path, rel: str, lighting: bool = True) -> int:
    """The empty home as two splats: walls and floors at `rel`, ceilings beside it (ceiling_path), so
    a view from above (the editor's orbit and plan) can leave the ceilings out. Returns the count."""
    walls = render_shell(scene, scene_dir, lighting=lighting, ceilings=False)
    ceil = render_shell(scene, scene_dir, lighting=lighting, ceilings="only")
    out = Path(scene_dir) / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    ply.write_ply(out, walls)
    ply.write_ply(ceiling_path(out), ceil)
    return len(walls) + len(ceil)


def render_shell(scene, scene_dir: Path, cell: float = 0.025, floor_cell: float = 0.0125,
                 lighting: bool = True, ceilings: bool | str = True) -> ply.Splat:
    """The empty home from the scene's finishes (and the shading maps, unless lighting=False).
    ceilings: True (all of it), False (without them) or "only"."""
    import cv2

    scene_dir = Path(scene_dir)
    sh_path = scene_dir / "finishes" / "shading.npz"
    shading = dict(np.load(sh_path)) if (lighting and sh_path.exists()) else {}
    rng = np.random.default_rng(0)
    parts = []
    from .trim import decorate

    fine = {s["id"]: s for s in _surfaces(scene.room, floor_cell)}
    finish_of = {sp.id: sp.finish for sp in scene.room.spaces}
    for s0 in _surfaces(scene.room, cell):
        if (s0["kind"] == "ceiling") != (ceilings == "only") and ceilings is not True:
            continue
        sf = _material_of(scene, s0)
        if sf is None:
            continue
        # floors and painted block at the fine grid (boards, joints); plain paint at the coarse one
        s, c = (fine[s0["id"]], floor_cell) if (s0["kind"] == "floor" or sf.kind == "block") else (s0, cell)
        col = _albedo(sf, s, c, scene_dir, rng)
        sh = np.ones(s["xyz"].shape[:2], np.float32)
        if s["id"] in shading:
            sh = cv2.resize(shading[s["id"]].astype(np.float32), s["xyz"].shape[1::-1], interpolation=cv2.INTER_LINEAR)
        col = col * sh[..., None]
        extra = decorate(s, col, sh, finish_of.get(s.get("space")), c) if s["kind"] == "wall" else []
        s["final"] = np.clip(col, 0, 1)
        parts.append((s, c))
        for e in extra:
            e["final"] = np.clip(e["final"], 0, 1)
            parts.append((e, c))
    return _gaussians(parts)


def _gaussians(parts: list[tuple[dict, float]]) -> ply.Splat:
    xyz, rgb, rot, scl = [], [], [], []
    for s, c in parts:
        m = s["inside"]
        n = int(m.sum())
        if not n:
            continue
        xyz.append(s["xyz"][m])
        rgb.append(s["final"][m])
        R = np.stack([s["u"], s["v"], s["normal"]], 1)
        if np.linalg.det(R) < 0:
            R[:, 1] *= -1
        rot.append(np.tile(ply.rotmat_to_quat(R), (n, 1)))
        scl.append(np.tile(np.log([0.85 * c, 0.85 * c, 0.002]), (n, 1)))
    out = ply.Splat(xyz=np.concatenate(xyz).astype(np.float32))
    out.f_dc = ((np.concatenate(rgb) - 0.5) / ply.SH_C0).astype(np.float32)
    out.opacity = np.full(len(out), 5.0, np.float32)
    out.scale = np.concatenate(scl).astype(np.float32)
    out.rot = np.concatenate(rot).astype(np.float32)
    return out


def finish_summary(scene) -> dict:
    out = {}
    for sp in scene.room.spaces:
        f = sp.finish
        if f is None:
            continue
        d = {k: (getattr(f, k).kind, getattr(f, k).color) for k in ("floor", "walls", "ceiling") if getattr(f, k)}
        if f.wall_overrides:
            d["wall_overrides"] = {k: (v.kind, v.color) for k, v in f.wall_overrides.items()}
        out[sp.name] = d
    return json.loads(json.dumps(out))


# ---- a surface changed to another material (a new floor, block painted over...) ---------------

def retexture(sf, scene_dir: Path, name: str, seed: int = 0):
    """The surface with a texture made for its material, when it changed to one with a pattern
    (wood planks, painted block, tile, carpet) and has none; flat paint needs none. The texture
    is written under finishes/ and only carries the pattern: the surface's color tints it."""
    import hashlib

    import cv2

    if sf.texture or sf.kind not in ("wood", "block", "tile", "carpet"):
        return sf
    rng = np.random.default_rng(seed)
    color = sf.color or "#b0a898"
    n = 256
    if sf.kind == "wood":
        w = float(np.clip(sf.module_m or 0.19, 0.08, 0.3))
        grain = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=9)
        detail = np.repeat((1 + 0.06 * grain / max(grain.std(), 1e-6))[..., None], 3, -1)
        tex, size = fin.plank_texture(detail, color, w, planks=max(6, int(np.ceil(1.2 / w))), px_per_m=256, seed=seed)
        sf = sf.model_copy(update={"module_m": w, "direction_deg": sf.direction_deg if sf.direction_deg is not None else 0.0})
    elif sf.kind == "block":
        c = sf.module_m or 0.203
        tex, size = fin.block_texture(color, None, block_m=(2 * c, c), px_per_m=256, joint=0.94)
        sf = sf.model_copy(update={"module_m": c})
    elif sf.kind == "tile":
        t = float(np.clip(sf.module_m or 0.3, 0.05, 1.2))
        k = 2 if t >= 0.2 else 4
        size, m = k * t, round(k * t * 256)
        tex = np.ones((m, m, 3), np.float32) * fin.hex_to_bgr(color)
        tex *= 1 + 0.02 * cv2.GaussianBlur(rng.normal(0, 1, (m, m)).astype(np.float32), (0, 0), 3)[..., None]
        g = max(2, round(0.004 * 256))
        for i in range(k):
            tex[round(i * m / k):round(i * m / k) + g] *= 0.8
            tex[:, round(i * m / k):round(i * m / k) + g] *= 0.8
        tex = np.clip(tex, 0, 255).astype(np.uint8)
        sf = sf.model_copy(update={"module_m": t})
    else:  # carpet: pile
        pile = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 1.2)
        detail = np.repeat((1 + 0.08 * pile / max(pile.std(), 1e-6))[..., None], 3, -1)
        tex, size = fin.tileable(fin.tint(detail, color)), 1.0
    (scene_dir / "finishes").mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{name}{sf.kind}{sf.module_m}{seed}".encode()).hexdigest()[:8]
    fn = f"finishes/{name}_{sf.kind}_{key}.jpg"
    cv2.imwrite(str(scene_dir / fn), tex)
    return sf.model_copy(update={"texture": fn, "size_m": round(float(size), 3)})
