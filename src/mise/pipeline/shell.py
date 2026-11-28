"""The empty home as its shell: walls, floors and ceilings of the traced layout, painted from the photos.

A splat of a lived-in home can't simply have its things taken out: plain walls give training little
to hold on to, so the Gaussians of a wall spread 20-30 cm through it and in among what stands
against it, and no cut at the wall's line separates the two. The layout already knows where each
wall, floor and ceiling is (within a few cm of the plan), so here they are made directly: flat
Gaussians on a grid over each surface, colored by the frames that see each point, using only
pixels the frame labels call that surface (wall, floor, ceiling), and only when the model's depth
says nothing stood in front. What no frame saw clearly (behind a wardrobe, under the bed) is
inpainted from around it. Doorways stay open.
"""

from __future__ import annotations

import numpy as np

from . import ply

WALL = frozenset({"wall", "door", "screen door", "column", "windowpane", "window"})
FLOOR = frozenset({"floor"})
CEILING = frozenset({"ceiling"})


def wall_openings(room, w, thickness: float = 0.4) -> list[tuple]:
    """The openings in wall `w`, as (opening, center along w): its own, and those listed on a wall
    of the room on the other side (a doorway goes through both faces of the wall)."""
    a, b = np.array(w.a, float), np.array(w.b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return []
    u = (b - a) / L
    by_id = {x.id: x for x in room.walls}
    out = []
    for o in room.openings:
        ow = by_id.get(o.wall)
        if ow is None:
            continue
        if ow.id == w.id:
            out.append((o, float(o.center)))
            continue
        if o.kind != "door" or ow.space == w.space:
            continue
        oa, ob = np.array(ow.a, float), np.array(ow.b, float)
        ou = (ob - oa) / max(float(np.linalg.norm(ob - oa)), 1e-6)
        if abs(float(ou @ u)) < 0.97:  # not parallel
            continue
        c = oa + ou * o.center  # the opening's middle, on the other face
        off = c - a
        if abs(float(off[0] * u[1] - off[1] * u[0])) > thickness:  # the other face is not across the wall
            continue
        along = float(off @ u)
        if -o.width / 2 < along < L + o.width / 2:
            out.append((o, along))
    return out


def _surfaces(room, cell: float) -> list[dict]:
    """Sample points of every wall, floor and ceiling: dict(kind, xyz (h, w, 3), normal, u, v, inside (h, w))."""
    import cv2

    out = []
    sp_of = {s.id: s for s in room.spaces}
    for w in room.walls:
        a, b = np.array(w.a, float), np.array(w.b, float)
        L = float(np.linalg.norm(b - a))
        if L < cell:
            continue
        u = (b - a) / L
        P = np.array(sp_of[w.space].polygon, float) if w.space in sp_of else None
        if P is not None:
            Q = np.roll(P, -1, 0)
            ccw = (P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]).sum() > 0
        else:
            ccw = True
        n2 = (1 if ccw else -1) * np.array([-u[1], u[0]])  # into the room
        H = float(w.height)
        s = np.arange(cell / 2, L, cell)
        z = np.arange(cell / 2, H, cell)
        S, Z = np.meshgrid(s, z)
        xyz = np.stack([a[0] + S * u[0] + 0.004 * n2[0], a[1] + S * u[1] + 0.004 * n2[1], Z], -1)
        inside = np.ones(S.shape, bool)
        ops = wall_openings(room, w)
        for o, c in ops:
            inside &= ~((np.abs(S - c) < o.width / 2) & (Z >= o.bottom) & (Z <= o.top))
        out.append({"kind": "wall", "xyz": xyz, "normal": np.array([n2[0], n2[1], 0.0]),
                    "u": np.array([u[0], u[1], 0.0]), "v": np.array([0.0, 0.0, 1.0]), "inside": inside, "id": w.id,
                    "space": w.space, "S": S, "Z": Z, "openings": ops})
    for sp in room.spaces:
        P = np.array(sp.polygon, float)
        lo, hi = P.min(0), P.max(0)
        xs, ys = np.arange(lo[0] + cell / 2, hi[0], cell), np.arange(lo[1] + cell / 2, hi[1], cell)
        X, Y = np.meshgrid(xs, ys)
        m = np.zeros(X.shape, np.uint8)
        cv2.fillPoly(m, [np.round((P - lo) / cell - 0.5).astype(np.int32)], 1)
        H = room.ceiling_for(sp) if hasattr(room, "ceiling_for") else room.ceiling_height
        for kind, zz, nz in (("floor", 0.002, 1.0), ("ceiling", H - 0.002, -1.0)):
            out.append({"kind": kind, "xyz": np.stack([X, Y, np.full(X.shape, zz)], -1),
                        "normal": np.array([0.0, 0.0, nz]), "u": np.array([1.0, 0.0, 0.0]),
                        "v": np.array([0.0, nz, 0.0]), "inside": m > 0, "id": f"{sp.id}.{kind}", "space": sp.id})
    return out


def paint(surfaces: list[dict], frames: dict[str, np.ndarray], cameras: list[dict], labels: dict[str, np.ndarray],
          classes: list[str], depth: dict[str, np.ndarray] | None = None, max_dist: float = 5.0,
          min_cos: float = 0.2, erode_px: int = 2, rounds: int = 3, say=print) -> None:
    """Each sample point's color from the frames that see it as its own kind of surface.

    Near, square-on views count most (cos^2 / d^2); a label must hold a few pixels in from its
    edge (labels are coarse at object outlines); and the average is made robust over a few
    rounds, each color weighed by how far it is from the last round's (a stray view of a desk
    or a reflection loses to the many that saw the wall).
    """
    import cv2

    allowed = {"wall": WALL, "floor": FLOOR, "ceiling": CEILING}
    cls_ok = {k: np.array([c.strip() in v for c in classes] + [False]) for k, v in allowed.items()}
    pts = np.concatenate([s["xyz"].reshape(-1, 3) for s in surfaces])
    nrm = np.concatenate([np.broadcast_to(s["normal"], s["xyz"].shape).reshape(-1, 3) for s in surfaces])
    kind = np.concatenate([np.full(s["xyz"].shape[0] * s["xyz"].shape[1], i) for i, s in enumerate(surfaces)])
    kinds = np.array([s["kind"] for s in surfaces])[kind]
    kid = {k: kinds == k for k in allowed}
    I, C, Wt = [], [], []
    ker = np.ones((2 * erode_px + 1,) * 2, np.uint8)
    for j, cam in enumerate(cameras):
        name = cam["name"]
        if name not in frames or name not in labels:
            continue
        img, lab = frames[name], labels[name]
        w2c = np.array(cam["w2c"], float)
        K = np.array(cam["K"], float)
        W, H = cam["size"]
        Pc = pts @ w2c[:3, :3].T + w2c[:3, 3]
        d = Pc[:, 2]
        view = -w2c[:3, :3].T @ w2c[:3, 3] - pts
        dist = np.linalg.norm(view, axis=1)
        cos = (view * nrm).sum(1) / np.maximum(dist, 1e-9)
        idx = np.nonzero((d > 0.2) & (dist < max_dist) & (cos > min_cos))[0]
        if not len(idx):
            continue
        uv = Pc[idx, :2] / d[idx, None] * np.diag(K)[:2] + K[:2, 2]
        inn = (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
        idx, uv = idx[inn], uv[inn]
        if not len(idx):
            continue
        lab = np.where(lab < len(classes), lab, len(classes))
        lh, lw = lab.shape
        li = np.clip((uv[:, 1] * lh / H).astype(int), 0, lh - 1), np.clip((uv[:, 0] * lw / W).astype(int), 0, lw - 1)
        good = np.zeros(len(idx), bool)
        for k, ok in cls_ok.items():
            sel = kid[k][idx]
            if sel.any():
                mask = cv2.erode(ok[lab].astype(np.uint8), ker) > 0
                good[sel] = mask[li[0][sel], li[1][sel]]
        if depth is not None and name in depth:
            dm = depth[name]
            dh, dw = dm.shape
            dv = dm[np.clip((uv[:, 1] * dh / H).astype(int), 0, dh - 1), np.clip((uv[:, 0] * dw / W).astype(int), 0, dw - 1)]
            # what the frame sees there is this surface: nothing in front, and not a wall farther
            # off showing through where the outline stands short of it
            good &= np.abs(d[idx] - dv) < np.maximum(0.12, 0.06 * dv)
        idx, uv = idx[good], uv[good]
        if not len(idx):
            continue
        ih, iw = img.shape[:2]
        I.append(idx.astype(np.int32))
        C.append(img[np.clip((uv[:, 1] * ih / H).astype(int), 0, ih - 1),
                     np.clip((uv[:, 0] * iw / W).astype(int), 0, iw - 1)].astype(np.float32))
        Wt.append((cos[idx] ** 2 / np.maximum(dist[idx], 0.5) ** 2).astype(np.float32))
        if j % 200 == 0:
            say(f"painted from {j + 1}/{len(cameras)} frames")
    n = len(pts)
    if I:
        I, C, Wt = np.concatenate(I), np.concatenate(C), np.concatenate(Wt)
    else:
        I, C, Wt = np.zeros(0, np.int32), np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    w = Wt.copy()
    for _ in range(rounds):
        ws = np.bincount(I, w, n)
        col = np.stack([np.bincount(I, w * C[:, c], n) for c in range(3)], 1) / np.maximum(ws, 1e-12)[:, None]
        e = np.linalg.norm(C - col[I], axis=1)
        w = Wt / (1 + (e / 0.08) ** 2)
    ws = np.bincount(I, Wt, n)
    i0 = 0
    for s in surfaces:
        m = s["xyz"].shape[0] * s["xyz"].shape[1]
        s["rgb"] = col[i0:i0 + m].reshape(s["xyz"].shape)
        s["seen"] = (ws[i0:i0 + m] > 0).reshape(s["xyz"].shape[:2])
        i0 += m


def hide_behind_things(surfaces: list[dict], things: np.ndarray, cell: float, depth: float = 0.8,
                       lift: float = 1.2, min_count: int = 2, grow: int = 4) -> None:
    """Mark as unseen what things stand in front of: wall cells with things within `depth` in front
    of them, floor cells with things up to `lift` above. Their colors in the photos are the
    things' (labels are coarse at outlines, and what stands against a wall passes any depth
    test), so they are filled from around instead."""
    import cv2

    k = np.ones((2 * grow + 1,) * 2, np.uint8)
    for s in surfaces:
        if s["kind"] == "ceiling" or not len(things):
            continue
        o = s["xyz"][0, 0] - (s["u"] + s["v"]) * cell / 2  # the grid's corner
        rel = things - o
        a, b, n = rel @ s["u"], rel @ s["v"], rel @ s["normal"]
        h, w = s["xyz"].shape[:2]
        m = (n > 0.03) & (n < (depth if s["kind"] == "wall" else lift)) & (a >= 0) & (a < w * cell) & (b >= 0) & (b < h * cell)
        cnt = np.zeros((h, w), np.int32)
        np.add.at(cnt, (np.floor(b[m] / cell).astype(int), np.floor(a[m] / cell).astype(int)), 1)
        occ = cv2.morphologyEx((cnt >= min_count).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        occ = cv2.dilate(occ, k) > 0
        s["seen"] = s["seen"] & ~occ
        s["hidden"] = occ & s["inside"]


def _copy_fill(rgb: np.ndarray, seen: np.ndarray, inside: np.ndarray, tries: int = 16) -> np.ndarray:
    """Holes in a floor filled with the nearest same-size patch of seen floor (the boards go on)."""
    import cv2

    out = rgb.copy()
    hole = (~seen & inside).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hole)
    good = seen & inside
    H, W = hole.shape
    for j in range(1, n):
        x, y, w, h = st[j, :4]
        m = lab[y:y + h, x:x + w] == j
        best = None
        for t in range(tries):
            ang = 2 * np.pi * t / tries
            for f in (1.1, 1.6, 2.2):
                dx, dy = int(round(np.cos(ang) * w * f)), int(round(np.sin(ang) * h * f))
                sx, sy = x + dx, y + dy
                if sx < 0 or sy < 0 or sx + w > W or sy + h > H:
                    continue
                frac = good[sy:sy + h, sx:sx + w][m].mean()
                if best is None or frac > best[0]:
                    best = (frac, sx, sy)
                if frac > 0.98:
                    break
        if best is not None and best[0] > 0.9:
            src = out[best[2]:best[2] + h, best[1]:best[1] + w]
            out[y:y + h, x:x + w][m] = src[m]
            seen = seen.copy()
            seen[y:y + h, x:x + w][m & good[best[2]:best[2] + h, best[1]:best[1] + w]] = True
    return out, seen


def _smooth_fill(rgb: np.ndarray, seen: np.ndarray, sigma: float) -> np.ndarray:
    """Unseen cells take a broad, seen-weighted blur of what is around them (wider until all are reached)."""
    import cv2

    out = rgb.copy()
    todo = ~seen
    wgt = seen.astype(np.float32)
    val = (rgb * wgt[..., None]).astype(np.float32)
    sg = sigma
    while todo.any() and sg < 4 * max(rgb.shape[:2]):
        ws = cv2.GaussianBlur(wgt, (0, 0), sg)
        vs = cv2.GaussianBlur(val, (0, 0), sg)
        got = todo & (ws > 1e-3)
        out[got] = vs[got] / ws[got, None]
        todo &= ~got
        sg *= 2
    if todo.any():
        out[todo] = rgb[seen].mean(0) if seen.any() else 0.75
    return out


def _inpaint(s: dict, cell: float = 0.025, dark: float = 0.65) -> np.ndarray:
    rgb, seen, inside = s["rgb"], s["seen"] & s["inside"], s["inside"]
    if seen.sum() == 0:
        return np.full(rgb.shape, 0.75)
    lum = rgb.mean(-1)
    cut = 0.75 if s["kind"] == "wall" else dark
    seen = seen & (lum > cut * np.median(lum[seen]))  # contact shadows, gaps under things: not the surface
    if seen.sum() == 0:
        return np.full(rgb.shape, 0.75)
    if s["kind"] == "ceiling":  # plain; views up through openings and bulkheads mislead: its broad color
        return _broad(rgb, seen, 1.0 / cell)
    if s["kind"] == "floor":  # the boards go on where they can; the rest blurred in
        rgb, seen = _copy_fill(rgb, seen, inside)
    # walls are near one color: a hole takes the wall around it broadly, not its darkest rim
    return _smooth_fill(rgb, seen, sigma=(0.5 if s["kind"] == "wall" else 0.25) / cell)


def _broad(rgb: np.ndarray, seen: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    w = cv2.GaussianBlur(seen.astype(np.float32), (0, 0), sigma)
    v = cv2.GaussianBlur((rgb * seen[..., None]).astype(np.float32), (0, 0), sigma)
    out = v / np.maximum(w, 1e-6)[..., None]
    out[w < 1e-3] = rgb[seen].mean(0)
    return out


def to_splat(surfaces: list[dict], cell: float) -> ply.Splat:
    xyz, rgb, rot, scl = [], [], [], []
    for s in surfaces:
        col = _inpaint(s, cell)
        m = s["inside"]
        n = int(m.sum())
        if not n:
            continue
        xyz.append(s["xyz"][m])
        rgb.append(col[m])
        R = np.stack([s["u"], s["v"], s["normal"]], 1)
        if np.linalg.det(R) < 0:
            R[:, 1] *= -1
        rot.append(np.tile(ply.rotmat_to_quat(R), (n, 1)))
        scl.append(np.tile(np.log([0.85 * cell, 0.85 * cell, 0.002]), (n, 1)))  # discs that overlap: no dot grid up close
    out = ply.Splat(xyz=np.concatenate(xyz).astype(np.float32))
    out.f_dc = ((np.concatenate(rgb) - 0.5) / ply.SH_C0).astype(np.float32)
    out.opacity = np.full(len(out), 5.0, np.float32)
    out.scale = np.concatenate(scl).astype(np.float32)
    out.rot = np.concatenate(rot).astype(np.float32)
    return out


def textured_shell(room, frames: dict[str, np.ndarray], cameras: list[dict], labels: dict[str, np.ndarray],
                   classes: list[str], depth: dict[str, np.ndarray] | None = None, cell: float = 0.025,
                   things: np.ndarray | None = None, say=print) -> tuple[ply.Splat, dict, list[dict]]:
    """`things`: points (scene frame) of what is not the shell (a labelled splat's furniture)."""
    surfaces = _surfaces(room, cell)
    say(f"{len(surfaces)} surfaces, {sum(int(s['inside'].sum()) for s in surfaces)} points")
    paint(surfaces, frames, cameras, labels, classes, depth, say=say)
    if things is not None:
        hide_behind_things(surfaces, np.asarray(things, float), cell)
    out = to_splat(surfaces, cell)
    rep = {}
    for k in ("wall", "floor", "ceiling"):
        ss = [s for s in surfaces if s["kind"] == k]
        tot = sum(int(s["inside"].sum()) for s in ss)
        seen = sum(int((s["seen"] & s["inside"]).sum()) for s in ss)
        hid = sum(int(s.get("hidden", np.zeros(1, bool)).sum()) for s in ss)
        rep[k] = {"m2": round(tot * cell * cell, 1), "seen_pct": round(100 * seen / max(tot, 1), 1),
                  "behind_things_pct": round(100 * hid / max(tot, 1), 1)}
    return out, rep, surfaces


SET_IN = frozenset({"mirror", "windowpane", "window", "glass"})  # in the wall's plane: part of it


def thing_points(s: ply.Splat, cls: np.ndarray, classes: list[str], min_opacity: float = 0.3) -> np.ndarray:
    """Centers of a labelled splat's Gaussians that are things (not walls, floors, ceilings, doors or
    what is set into walls; screens and rugs are things), and opaque enough to hide what is behind."""
    from .structure import STRUCTURE

    shell = (set(STRUCTURE) - {"rug"}) | SET_IN
    keep = np.array([c.strip() in shell for c in classes] + [True])  # never labelled: not counted as a thing
    thing = ~keep[np.where(cls < 0, len(classes), cls)]
    if s.opacity is not None:
        thing &= 1 / (1 + np.exp(-s.opacity)) > min_opacity
    return s.xyz[thing].astype(np.float64)
