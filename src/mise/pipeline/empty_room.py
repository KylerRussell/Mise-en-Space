"""The empty home: a scanned splat without the things in it.

Each Gaussian carries the class the frames that see it voted for (splat_train.py --labels, from
segment_frames.py). The home's shell (walls, floor, ceiling, doors, and what is set into them:
windows, mirrors) stays; beds, clothes, cabinets, bottles... go, and so does anything outside the
traced rooms (what showed through windows and in mirrors). Where things stood on the floor, the
floor is filled by copying the nearest intact floor (decompose.fill_floor_plane, the plane +
texture-copy baseline of the plan). Walls behind furniture are left as they are for now.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import decompose, ply
from .structure import SEE_THROUGH, STRUCTURE

KEEP = STRUCTURE | SEE_THROUGH


def _grid_mask(polys: list[np.ndarray], lo: np.ndarray, shape: tuple[int, int], cell: float, grow: float):
    import cv2

    m = np.zeros(shape, np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round((np.asarray(p) - lo) / cell).astype(np.int32)], 1)
    k = max(1, int(round(grow / cell)))
    return cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) if grow > 0 else m


# set into the home: they stay wherever they are (cabinets only against a wall: the kitchen's)
FIXTURES = frozenset({"bathtub", "counter", "countertop", "sink", "refrigerator", "toilet", "stove", "kitchen island",
                      "oven", "microwave", "dishwasher", "hood", "shower", "radiator", "fireplace"})
CABINETS = frozenset({"cabinet"})


def _peak(v: np.ndarray, lo: float, hi: float, bin_: float = 0.01) -> float | None:
    """Where values in [lo, hi] are densest (smoothed histogram), or None when there are few."""
    v = v[(v >= lo) & (v <= hi)]
    if len(v) < 50:
        return None
    h, e = np.histogram(v, bins=max(1, int(round((hi - lo) / bin_))), range=(lo, hi))
    h = np.convolve(h, np.ones(3) / 3, mode="same")
    if h.max() < 4 * h.mean():  # no surface stands out: spread evenly (things, or nothing)
        return None
    return float(e[np.argmax(h)] + bin_ / 2)


def _holes(cover: np.ndarray, shadow: np.ndarray, region: np.ndarray, cell: float, lo: np.ndarray,
           grow: int = 2, min_m2: float = 0.03) -> tuple[list[np.ndarray], float]:
    """Polygons (in the grid's units) of `region` cells not covered by what was kept but in the
    shadow of what was removed: where something stood in front of the surface."""
    import cv2

    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    cov = cv2.dilate(cover.astype(np.uint8), k)
    sh = cv2.dilate(shadow.astype(np.uint8), np.ones((3, 3), np.uint8))
    hole = ((cov == 0) & (sh > 0) & (region > 0)).astype(np.uint8)
    hole = cv2.morphologyEx(hole, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hole = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hole)
    out, area = [], 0.0
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] * cell * cell < min_m2:
            continue
        cs, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
        out.append((c + 0.5) * cell + lo)
        area += float(stats[j, cv2.CC_STAT_AREA]) * cell * cell
    return out, area


def empty_room(s: ply.Splat, cls: np.ndarray, classes: list[str], rooms: list[np.ndarray],
               ceilings: list[float] | None = None, margin: float = 0.3, cell: float = 0.05, on: float = 0.05,
               reach: float = 1.2) -> tuple[ply.Splat, dict]:
    """The home's shell: what lies on its walls, floor and ceiling (and fixtures) stays, the rest of
    what is inside the rooms goes, and so does everything beyond them; then the floor and walls
    are filled where something stood in front of them.

    The rooms' outlines put each wall within a few cm; the wall's surface is where the Gaussians
    near that line are densest, likewise the floor and ceiling heights. "On" a surface is within
    `on` of it. Labels only decide fixtures (counters, appliances, bath): kept. Holes to fill are
    surface cells nothing kept covers that removed Gaussians within `reach` stood in front of.
    """
    import cv2

    X = s.xyz.astype(np.float64)
    n_cls = len(classes)
    name = np.array(list(classes) + ["unseen"])[np.where(cls < 0, n_cls, cls)]
    ceilings = ceilings or [2.5] * len(rooms)
    fine = 0.02
    allp = np.vstack(rooms)
    lo = allp.min(0) - margin - 1.0
    shape = tuple((np.ceil((allp.max(0) + margin + 1.0 - lo) / fine)).astype(int)[::-1])
    lab = np.full(shape, -1, np.int32)
    for k, poly in enumerate(rooms):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round((np.asarray(poly) - lo) / fine).astype(np.int32)], 1)
        lab[(m > 0) & (lab < 0)] = k
    dist = np.zeros(shape, np.float32)  # inside a room: how far to its outline
    for k in range(len(rooms)):
        m = (lab == k).astype(np.uint8)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5) * fine
        dist[m > 0] = d[m > 0]
    near = cv2.dilate((lab >= 0).astype(np.uint8), np.ones((2 * int(margin / fine) + 1,) * 2, np.uint8))
    ij = np.floor((X[:, :2] - lo) / fine).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    room = np.full(len(X), -1)
    room[ok] = lab[ij[ok, 1], ij[ok, 0]]
    beyond = np.ones(len(X), bool)
    beyond[ok] = near[ij[ok, 1], ij[ok, 0]] == 0
    inside = room >= 0
    keep = ~beyond & ~inside  # wall thickness, doorways

    # floor and ceiling of each room: their surfaces' heights
    flat = np.zeros(len(X), bool)
    floor_at, ceil_at = {}, {}
    for k in range(len(rooms)):
        m = room == k
        f = _peak(X[m, 2], -0.15, 0.15)
        c = _peak(X[m, 2], ceilings[k] - 0.3, ceilings[k] + 0.3)
        floor_at[k], ceil_at[k] = (0.0 if f is None else f), (ceilings[k] if c is None else c)
        flat |= m & ((np.abs(X[:, 2] - floor_at[k]) < on) | (np.abs(X[:, 2] - ceil_at[k]) < on))

    # walls: each outline edge, its surface where Gaussians near the line are densest
    walls, on_wall = [], np.zeros(len(X), bool)
    for k, poly in enumerate(rooms):
        P = np.asarray(poly, float)
        Q = np.roll(P, -1, 0)
        sgn = 1.0 if (P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]).sum() > 0 else -1.0  # counter-clockwise: left is in
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            L = float(np.linalg.norm(b - a))
            if L < 0.2:
                continue
            u = (b - a) / L
            nrm = sgn * np.array([-u[1], u[0]])  # into the room
            along = (X[:, :2] - a) @ u
            off = (X[:, :2] - a) @ nrm
            span = (along > -0.05) & (along < L + 0.05) & (X[:, 2] > floor_at[k] + 0.1) & (X[:, 2] < ceil_at[k] - 0.1)
            w0 = _peak(off[span], -0.25, 0.25)
            if w0 is None:
                continue
            sup = span & (np.abs(off - w0) < on)
            if np.unique(np.floor(along[sup] / 0.1)).size < 0.3 * L / 0.1:  # a wall runs along its edge
                continue
            hit = ((along > -0.05) & (along < L + 0.05) & (X[:, 2] > floor_at[k] - on) & (X[:, 2] < ceil_at[k] + on)
                   & (np.abs(off - w0) < on) & ~beyond)  # the whole wall, skirting to cornice
            on_wall |= hit
            walls.append((k, a, u, nrm, L, w0))

    dk = np.zeros(len(X))
    dk[ok] = dist[ij[ok, 1], ij[ok, 0]]
    fixture = np.isin(name, list(FIXTURES)) | (np.isin(name, list(CABINETS)) & (dk < 0.75))
    keep |= inside & (flat | on_wall | fixture)
    gone = inside & ~keep

    # what stood on the floor: dense cells of removed things low down, padded; the floor there
    # (their feet and contact shadows) goes too and is filled
    fz = np.array([floor_at.get(r, 0.0) for r in room])
    cshape = tuple((np.ceil((allp.max(0) + margin + 1.0 - lo) / cell)).astype(int)[::-1])
    cij = np.floor((X[:, :2] - lo) / cell).astype(int)
    cok = (cij[:, 0] >= 0) & (cij[:, 0] < cshape[1]) & (cij[:, 1] >= 0) & (cij[:, 1] < cshape[0])
    low = gone & cok & (X[:, 2] - fz < 1.0)
    cnt = np.zeros(cshape, np.int32)
    np.add.at(cnt, (cij[low, 1], cij[low, 0]), 1)
    foot = cv2.morphologyEx((cnt >= 4).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    foot = cv2.dilate(cv2.morphologyEx(foot, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8))
    under = np.zeros(len(X), bool)
    under[cok] = foot[cij[cok, 1], cij[cok, 0]] > 0
    under &= keep & inside & flat & (np.abs(X[:, 2] - fz) < on) & ~fixture
    keep &= ~under
    kept = s.subset(keep)
    patches = []

    # floor: cells under removed things that nothing kept covers
    fl = keep & inside & (np.abs(X[:, 2] - fz) < on)
    rem = (gone | under) & (X[:, 2] < reach)
    cover, shadow = np.zeros(cshape, np.uint8), np.zeros(cshape, np.uint8)
    cover[cij[fl & cok, 1], cij[fl & cok, 0]] = 1
    shadow[cij[rem & cok, 1], cij[rem & cok, 0]] = 1
    region = cv2.resize((lab >= 0).astype(np.uint8), (cshape[1], cshape[0]), interpolation=cv2.INTER_NEAREST)
    fholes, farea = _holes(cover, shadow, region, cell, lo)
    if fholes:
        base = s.subset(fl)
        filled = decompose.fill_floor_plane(base, fholes, floor_z=on)
        patches.append(filled.subset(np.arange(len(filled)) >= len(base)))

    # walls: in each wall's frame (along, up, out of the surface) the same, then back
    wn, warea = 0, 0.0
    for k, a, u, nrm, L, w0 in walls:
        R = np.array([[u[0], u[1], 0.0], [0.0, 0.0, 1.0], [nrm[0], nrm[1], 0.0]])
        if np.linalg.det(R) < 0:
            R[2] *= -1
        sgn_n = float(R[2, :2] @ nrm)  # +1 or -1: which way the frame's z points
        t = -R @ np.array([a[0], a[1], 0.0]) - np.array([0.0, 0.0, sgn_n * w0])
        F = X @ R.T + t  # along, height, out of the surface (sign sgn_n)
        band = keep & (np.abs(F[:, 2]) < on) & (F[:, 0] > 0) & (F[:, 0] < L) & on_wall
        if band.sum() < 100:
            continue
        front = gone & (F[:, 0] > 0) & (F[:, 0] < L) & (sgn_n * F[:, 2] > on) & (sgn_n * F[:, 2] < reach) & (room == k)
        if not front.any():
            continue
        flo = np.array([0.0, floor_at[k]])
        wshape = (int(np.ceil((ceil_at[k] - floor_at[k]) / cell)) + 1, int(np.ceil(L / cell)) + 1)
        gij = lambda m: np.clip(np.floor((F[m, :2] - flo) / cell).astype(int), 0, np.array(wshape[::-1]) - 1)  # noqa: E731
        cover, shadow = np.zeros(wshape, np.uint8), np.zeros(wshape, np.uint8)
        g = gij(band)
        cover[g[:, 1], g[:, 0]] = 1
        g = gij(front)
        shadow[g[:, 1], g[:, 0]] = 1
        holes, area = _holes(cover, shadow, np.ones(wshape, np.uint8), cell, flo)
        if not holes:
            continue
        base = ply.transform(s.subset(band), R=R, t=t)
        filled = decompose.fill_floor_plane(base, holes, floor_z=on)
        if len(filled) > len(base):
            add = filled.subset(np.arange(len(filled)) >= len(base))
            patches.append(ply.transform(add, R=R.T, t=-R.T @ t))
            wn, warea = wn + len(holes), warea + area
    out = ply.Splat.concat([kept, *patches]) if patches else kept
    names, counts = np.unique(name[gone], return_counts=True)
    top = np.argsort(-counts)[:10]
    return out, {"gaussians": len(s), "things": int(gone.sum()), "floor_under_things": int(under.sum()),
                 "outside": int(beyond.sum()),
                 "fixtures": int((inside & fixture).sum()), "walls": len(walls),
                 "floor_holes": len(fholes), "floor_hole_m2": round(farea, 2),
                 "wall_holes": wn, "wall_hole_m2": round(warea, 2),
                 "kept": len(kept), "filled_with": len(out) - len(kept),
                 "removed_classes": {str(names[i]): int(counts[i]) for i in top}}


def spread_labels(xyz: np.ndarray, cls: np.ndarray, cell: float = 0.1) -> np.ndarray:
    """Gaussians no frame saw as the front surface take the commonest class of the ones that were
    in their `cell`-sized voxel, else in the 26 around it; the rest stay -1."""
    known = cls >= 0
    if known.all() or not known.any():
        return cls
    v = np.floor(xyz / cell).astype(np.int64)
    out = cls.copy()
    todo = ~known
    for reach in (0, 1):
        if not todo.any():
            break
        offs = [(0, 0, 0)] if reach == 0 else [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        pairs = []  # (voxel key, class) of known Gaussians, shifted into the voxels they vouch for
        for o in offs:
            pairs.append((_key(v[known] + np.array(o)), cls[known]))
        keys = np.concatenate([k for k, _ in pairs])
        vals = np.concatenate([c for _, c in pairs])
        order = np.lexsort((vals, keys))
        keys, vals = keys[order], vals[order]
        # commonest class per voxel
        kv = keys * 1000 + vals
        uk, cnt = np.unique(kv, return_counts=True)
        vk, vc = uk // 1000, uk % 1000
        best = {}
        for k_, c_, n_ in zip(vk, vc, cnt):
            if k_ not in best or n_ > best[k_][1]:
                best[k_] = (c_, n_)
        q = _key(v[todo])
        got = np.array([best.get(k_, (-1, 0))[0] for k_ in q])
        idx = np.nonzero(todo)[0]
        out[idx] = got
        todo = out < 0
    return out


def _key(v: np.ndarray) -> np.ndarray:
    v = v + 100000
    return (v[:, 0] * 200003 + v[:, 1]) * 200003 + v[:, 2]


def empty_scene(scene_dir: Path, splat_dir: Path, name: str = "empty.ply", semantic: Path | None = None,
                frames: Path | None = None, depth: Path | None = None, depth_scale: float = 1.0,
                say=print) -> dict:
    """Write scene_dir/splats/<name>, the empty home.

    With the frames (`frames`: the picked frames folder) and the model's depth maps (`depth`:
    depth.npz, scaled to the scene by `depth_scale`), the empty home is the layout's shell painted
    from the photos (shell.py): walls, floors and ceilings where the plan has them, free of what
    stood against them. Without them, the splat itself with its things cut out (`empty_room`).
    Gaussians are labelled from the frame labels (`semantic`) and the trained cameras when given,
    else from the trainer's gauss_labels.npz.
    """
    import json

    from ..scene.models import Scene
    from . import structure

    scene_dir, splat_dir = Path(scene_dir), Path(splat_dir)
    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    s = ply.read_ply(splat_dir / "splat.ply")
    cams = json.loads((splat_dir / "cameras.json").read_text())
    labels = None
    if semantic is not None:
        labels, classes = structure.load_labels(Path(semantic))
        cls, seen = label_gaussians(s, cams, labels, len(classes), stride=2)
        np.savez_compressed(splat_dir / "gauss_labels.npz", cls=cls.astype(np.int16), seen=seen.astype(np.int32),
                            classes=np.array(classes))
    else:
        z = np.load(splat_dir / "gauss_labels.npz")
        cls, classes = z["cls"].astype(int), [str(c).strip() for c in z["classes"]]
    cls = spread_labels(s.xyz.astype(np.float64), cls)
    (scene_dir / "splats").mkdir(exist_ok=True)
    if frames is not None and depth is not None and labels is not None:
        import cv2

        from .shell import textured_shell, thing_points

        frames = Path(frames)
        fd = frames / "frames" if (frames / "frames").is_dir() else frames
        by = {c["name"]: c for c in cams}  # a frame in two overlapping windows: either camera
        imgs = {}
        for n in by:
            im = cv2.imread(str(fd / n))
            if im is not None:
                imgs[n] = cv2.cvtColor(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        dz = np.load(depth)
        dmaps = {str(n): dz["depth"][i].astype(np.float32) * depth_scale for i, n in enumerate(dz["names"])}
        from .materials import estimate, finish_summary, render_shell

        photo, rep, surfaces = textured_shell(scene.room, imgs, list(by.values()), labels, classes, dmaps,
                                              things=thing_points(s, cls, classes), say=say)
        ply.write_ply(scene_dir / "splats" / "empty_photo.ply", photo)  # as painted from the photos, to compare
        # what each surface is (paint, wood, block...) and its color; the empty home made from that
        estimate(scene, surfaces, 0.025, list(by.values()), fd, labels, classes, dmaps, scene_dir, say=say)
        out = render_shell(scene, scene_dir)
        rep = {"kind": "shell", **rep, "finishes": finish_summary(scene), "gaussians": len(out)}
    else:
        rooms = [np.array(sp.polygon, float) for sp in scene.room.spaces]
        ceilings = [sp.ceiling_height or scene.room.ceiling_height for sp in scene.room.spaces]
        out, rep = empty_room(s, cls, classes, rooms, ceilings)
        rep = {"kind": "cut", **rep}
    ply.write_ply(scene_dir / "splats" / name, out)
    rep["path"] = f"splats/{name}"
    return rep


def label_gaussians(s: ply.Splat, cameras: list[dict], labels: dict[str, np.ndarray], n_classes: int,
                    grid: tuple[int, int] = (160, 90), tol: float = 0.06, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Each Gaussian's class by vote of the frames that see it (CPU).

    Per frame, every center is projected; on a coarse z-buffer only the centers within `tol`
    (plus 2% of the distance) of the nearest at their cell count as seen, and vote for the
    label under them. Returns (class per Gaussian, -1 when never seen; votes for it).
    """
    X = s.xyz.astype(np.float64)
    keep_o = 1 / (1 + np.exp(-s.opacity)) > 0.2 if s.opacity is not None else np.ones(len(X), bool)
    gw, gh = grid
    votes = np.zeros((len(X), n_classes), np.int32)
    for cam in cameras[::stride]:
        m = labels.get(cam["name"])
        if m is None:
            continue
        V, K = np.array(cam["w2c"]), np.array(cam["K"])
        W, H = cam["size"]
        pc = X @ V[:3, :3].T + V[:3, 3]
        z = pc[:, 2]
        ok = (z > 0.1) & keep_o
        u = K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        gx = (u[idx] * gw / W).astype(int)
        gy = (v[idx] * gh / H).astype(int)
        cell = gy * gw + gx
        zbuf = np.full(gw * gh, np.inf)
        np.minimum.at(zbuf, cell, z[idx])
        seen = z[idx] <= zbuf[cell] + tol + 0.02 * zbuf[cell]
        idx, u_, v_ = idx[seen], u[idx][seen], v[idx][seen]
        cl = m[(v_ * m.shape[0] / H).astype(int).clip(0, m.shape[0] - 1), (u_ * m.shape[1] / W).astype(int).clip(0, m.shape[1] - 1)]
        np.add.at(votes, (idx, cl.astype(int)), 1)
    total = votes.sum(1)
    cls = np.where(total > 0, votes.argmax(1), -1)
    return cls, total


def merge_splats(parts: list[tuple[ply.Splat, list[dict]]], rooms: list[np.ndarray], cell: float = 0.05) -> tuple[ply.Splat, dict]:
    """One home splat from splats trained on overlapping stretches (all in the scene's frame).

    A splat trained on a whole tour is spread too thin; one per stretch (a room or two) is
    sharp. Each room is taken from the stretch whose cameras stood in it most; Gaussians
    outside every room go to the room nearest them. `parts` are (splat, its cameras.json).
    """
    import cv2

    allp = np.vstack(rooms)
    lo = allp.min(0) - 2.0
    shape = tuple((np.ceil((allp.max(0) + 2.0 - lo) / cell)).astype(int)[::-1])
    lab = np.full(shape, -1, np.int32)
    for k, p in enumerate(rooms):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round((np.asarray(p) - lo) / cell).astype(np.int32)], 1)
        lab[m > 0] = k
    # every cell takes the nearest room's label
    dist, idx = cv2.distanceTransformWithLabels((lab < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(lab >= 0)
    near = np.zeros(idx.max() + 1, np.int32)
    near[idx[ys, xs]] = lab[ys, xs]
    lab_all = near[idx]

    def room_of(xy):
        ij = np.clip(np.floor((xy - lo) / cell).astype(int), 0, np.array(shape[::-1]) - 1)
        return lab_all[ij[:, 1], ij[:, 0]]

    centers = [np.array([-np.array(cm["w2c"])[:3, :3].T @ np.array(cm["w2c"])[:3, 3] for cm in cams])[:, :2]
               for _, cams in parts]
    owner = []
    for k in range(len(rooms)):
        counts = [int((room_of(c) == k).sum()) for c in centers]
        if max(counts) > 0:
            owner.append(int(np.argmax(counts)))
        else:  # nobody stood in it (a closet looked into): the stretch that came closest
            mid = np.asarray(rooms[k], float).mean(0)
            owner.append(int(np.argmin([np.linalg.norm(c - mid, axis=1).min() for c in centers])))
    out = [s.subset(np.isin(room_of(s.xyz[:, :2].astype(np.float64)), [k for k, o in enumerate(owner) if o == j]))
           for j, (s, _) in enumerate(parts)]
    return ply.Splat.concat(out), {"owner": owner, "gaussians": [len(o) for o in out]}
