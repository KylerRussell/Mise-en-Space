"""Windows, doors and trim on the empty home's walls.

Windows and doors are found by letting the frames vote, through their labels, on each point of
each wall: where most views of a point see a window pane or blind, that is window; where they see
a door (or a mirrored closet door reaching the floor), door. The votes' rectangles become
openings in the scene (windows; doors with a closed leaf, for the entry and closets), colored as
captured. Trim is measured off the walls' bottom rows: the baseboard is the band whose color
differs from the wall above it. Rendering (`decorate`) paints baseboards and casings onto the
wall and adds the door leaves and window glass, recessed a little into the wall.
"""

from __future__ import annotations

import numpy as np

WINDOW = frozenset({"windowpane", "window", "blind", "shade"})
DOOR = frozenset({"door", "screen door"})
MIRROR = frozenset({"mirror"})
GLASS = np.array([0.80, 0.85, 0.90])  # a daylit window, seen from inside


def vote(surfaces: list[dict], cameras: list[dict], labels: dict[str, np.ndarray], classes: list[str],
         depth: dict[str, np.ndarray] | None, max_dist: float = 6.0, say=print) -> dict[str, dict[str, np.ndarray]]:
    """Per wall: how many views saw each point, and how many saw window, door or mirror there."""
    groups = {"window": WINDOW, "door": DOOR, "mirror": MIRROR}
    ids = {g: np.array([c.strip() in v for c in classes] + [False]) for g, v in groups.items()}
    walls = [s for s in surfaces if s["kind"] == "wall"]
    pts = np.concatenate([s["xyz"].reshape(-1, 3) for s in walls])
    nrm = np.concatenate([np.broadcast_to(s["normal"], s["xyz"].shape).reshape(-1, 3) for s in walls])
    n = len(pts)
    tot = np.zeros(n, np.int32)
    cnt = {g: np.zeros(n, np.int32) for g in groups}
    for j, cam in enumerate(cameras):
        name = cam["name"]
        if name not in labels:
            continue
        w2c, K = np.array(cam["w2c"], float), np.array(cam["K"], float)
        W, H = cam["size"]
        Pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        d = Pc[:, 2]
        view = -w2c[:3, :3].T @ w2c[:3, 3] - pts
        dist = np.linalg.norm(view, axis=1)
        cos = (view * nrm).sum(1) / np.maximum(dist, 1e-9)
        idx = np.nonzero((d > 0.2) & (dist < max_dist) & (cos > 0.2))[0]
        if not len(idx):
            continue
        uv = Pc[idx, :2] / d[idx, None] * np.diag(K)[:2] + K[:2, 2]
        inn = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        idx, uv = idx[inn], uv[inn]
        if depth is not None and name in depth:  # not behind another wall (glass may read deeper: one-sided)
            dm = depth[name]
            dh, dw = dm.shape
            dv = dm[np.clip((uv[:, 1] * dh / H).astype(int), 0, dh - 1), np.clip((uv[:, 0] * dw / W).astype(int), 0, dw - 1)]
            ok = d[idx] < dv + np.maximum(0.3, 0.1 * dv)
            idx, uv = idx[ok], uv[ok]
        lab = labels[name]
        lh, lw = lab.shape
        lv = lab[np.clip((uv[:, 1] * lh / H).astype(int), 0, lh - 1), np.clip((uv[:, 0] * lw / W).astype(int), 0, lw - 1)]
        lv = np.minimum(lv, len(classes))
        tot[idx] += 1
        for g in groups:
            cnt[g][idx[ids[g][lv]]] += 1
        if j % 300 == 0:
            say(f"window and door votes from {j + 1}/{len(cameras)} frames")
    out, i0 = {}, 0
    for s in walls:
        m = s["xyz"].shape[0] * s["xyz"].shape[1]
        sh = s["xyz"].shape[:2]
        out[s["id"]] = {"total": tot[i0:i0 + m].reshape(sh), **{g: cnt[g][i0:i0 + m].reshape(sh) for g in groups}}
        i0 += m
    return out


def _rects(mask: np.ndarray, cell: float, min_m2: float) -> list[tuple[float, float, float, float]]:
    """(left, right, bottom, top) in metres of the mask's blobs on a wall grid (rows = height)."""
    import cv2

    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for j in range(1, n):
        x, y, w, h, a = st[j]
        if a * cell * cell < min_m2:
            continue
        out.append((x * cell, (x + w) * cell, y * cell, (y + h) * cell))
    return out


def find_openings(scene, surfaces: list[dict], votes: dict, cell: float, gain: float = 1.0,
                  min_votes: int = 3) -> list:
    """Windows and closed doors on the walls, as scene openings (replacing ones found before)."""
    from ..scene.models import Opening

    found = []
    for s in surfaces:
        if s["kind"] != "wall" or s["id"] not in votes:
            continue
        v = votes[s["id"]]
        enough = v["total"] >= min_votes
        frac = {g: np.where(enough, v[g] / np.maximum(v["total"], 1), 0) for g in ("window", "door", "mirror")}
        H = s["Z"].max() + cell / 2
        L = s["S"].max() + cell / 2
        taken = [(c - o.width / 2, c + o.width / 2) for o, c in s.get("openings", []) if o.kind == "door"]

        def free(l, r):
            return all(min(r, b) - max(l, a) < 0.3 * (r - l) for a, b in taken)

        def color(l, r, bt, tp):
            m = (s["S"] >= l) & (s["S"] <= r) & (s["Z"] >= bt) & (s["Z"] <= tp) & s["seen"]
            if m.sum() < 10:
                return None
            c = np.clip(np.median(s["rgb"][m], 0) * gain, 0, 1)
            return "#" + "".join(f"{int(round(x * 255)):02x}" for x in c)

        for l, r, bt, tp in _rects(frac["window"] > 0.35, cell, 0.12):
            if r - l < 0.35 or tp - bt < 0.35 or not free(l, r):
                continue
            bt, tp = max(bt, 0.25), min(tp, H - 0.08)
            found.append(Opening(id=f"win_{s['id']}_{len(found)}", kind="window", wall=s["id"], center=(l + r) / 2,
                                 width=r - l, bottom=round(bt, 3), top=round(tp, 3), swing="none",
                                 color=color(l, r, bt, tp)))
        doorish = (frac["door"] > 0.4) | (frac["mirror"] > 0.4)
        for l, r, bt, tp in _rects(doorish, cell, 0.6):
            if bt > 0.2 or tp < 1.6 or not (0.5 <= r - l <= 2.0) or not free(l, r):
                continue
            l, r = max(l, 0.0), min(r, L)
            found.append(Opening(id=f"leaf_{s['id']}_{len(found)}", kind="door", wall=s["id"], center=(l + r) / 2,
                                 width=r - l, bottom=0.0, top=round(min(max(tp, 2.0), H - 0.05), 3),
                                 swing="sliding" if (frac["mirror"][:, int(l / cell):int(r / cell) + 1] > 0.4).any()
                                 else "in", leaf=True, color=color(l, r, 0.1, 1.9)))
    scene.room.openings = [o for o in scene.room.openings if not o.id.startswith(("win_", "leaf_"))] + found
    return found


def estimate_trim(scene, surfaces: list[dict], cell: float, gain: float = 1.0) -> dict:
    """Baseboard height and trim color from the walls' bottom rows (one for the home, set on every room)."""
    import cv2

    from ..scene.models import Surface

    rows, mids = [], []
    for s in surfaces:
        if s["kind"] != "wall":
            continue
        ok = s["seen"] & s["inside"] & ~s.get("hidden", np.zeros_like(s["seen"]))
        z = s["Z"][:, 0]
        prof = [np.median(s["rgb"][k][ok[k]], 0) if ok[k].sum() >= 8 else None for k in range(min(len(z), int(0.3 / cell)))]
        mid = s["rgb"][(z > 0.6) & (z < 1.6)][ok[(z > 0.6) & (z < 1.6)]]
        if len(mid) < 50 or any(p is None for p in prof[:4]):
            continue
        rows.append(prof)
        mids.append(np.median(mid, 0))
    if not rows:
        return {}
    lab = lambda c: cv2.cvtColor(np.asarray(c, np.float32).reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]  # noqa: E731
    heights, cols = [], []
    for prof, mid in zip(rows, mids):
        lm = lab(mid)
        k = 0
        while k < len(prof) and prof[k] is not None and np.linalg.norm(lab(prof[k]) - lm) > 5.0:
            k += 1
        if 2 <= k < len(prof):
            heights.append(k * cell)
            cols.append(np.median(np.array(prof[:max(1, k - 1)]), 0))
    if len(heights) < 2:
        return {}
    h = float(np.clip(np.median(heights), 0.06, 0.2))
    c = np.clip(np.median(np.array(cols), 0) * gain, 0, 1)
    hexc = "#" + "".join(f"{int(round(x * 255)):02x}" for x in c)
    for sp in scene.room.spaces:
        if sp.finish is not None:
            sp.finish.trim = Surface(color=hexc, kind="paint")
            sp.finish.baseboard_m = round(h, 3)
    return {"baseboard_m": round(h, 3), "color": hexc, "walls_measured": len(heights)}


def _rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


def decorate(s: dict, col: np.ndarray, shade: np.ndarray, finish, cell: float, casing: float = 0.07) -> list[dict]:
    """Paint baseboard and casings onto a wall's colors (in place); return the door leaves and glass
    as extra surfaces (dicts like shell surfaces, with "final" colors)."""
    S, Z = s["S"], s["Z"]
    trim = _rgb(finish.trim.color) if finish is not None and finish.trim and finish.trim.color else None
    if trim is not None:
        hb = finish.baseboard_m or 0.1
        base = Z < hb
        col[base] = trim * shade[base][:, None]
        top = (Z >= hb) & (Z < hb + cell)
        col[top] *= 0.85  # the baseboard's shadow line
    extra = []
    for o, c in s.get("openings", []):
        l, r = c - o.width / 2, c + o.width / 2
        if trim is not None:  # casing: around the sides and top (and under a window)
            side = ((np.abs(S - l) < casing) & (S < l) | (np.abs(S - r) < casing) & (S > r)) & (Z <= o.top + casing) \
                & (Z >= (o.bottom - casing if o.kind == "window" else 0))
            head = (S > l - casing) & (S < r + casing) & (Z > o.top) & (Z < o.top + casing)
            band = side | head
            if o.kind == "window":
                band |= (S > l - casing) & (S < r + casing) & (Z < o.bottom) & (Z > o.bottom - casing)
            col[band] = trim * shade[band][:, None]
        if o.kind == "window" or o.leaf:
            m = (np.abs(S - c) < o.width / 2) & (Z >= o.bottom) & (Z <= o.top)
            if not m.any():
                continue
            if o.kind == "window":  # glass by day: what showed there was mostly curtain or blind (things)
                seen_c = _rgb(o.color) if o.color else GLASS
                base_c = np.minimum(0.5 * seen_c + 0.5 * GLASS, 0.92)
            else:
                base_c = _rgb(o.color) if o.color else np.array([0.7, 0.7, 0.68])
            fc = base_c * shade[m][:, None]
            if o.kind == "door" and trim is not None:  # a leaf's edges and its panel lines, as on a plain door
                ss, zz = S[m], Z[m]
                edge = (np.abs(ss - l) < 0.02) | (np.abs(ss - r) < 0.02) | (np.abs(zz - o.top) < 0.02)
                fc[edge] *= 0.8
            inset = 0.03 if o.kind == "window" else 0.015
            xyz = s["xyz"][m] - s["normal"] * inset
            extra.append({"kind": "panel", "xyz": xyz[None], "inside": np.ones((1, len(xyz)), bool),
                          "u": s["u"], "v": s["v"], "normal": s["normal"], "final": fc[None]})
    return extra
