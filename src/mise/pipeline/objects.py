"""The home's things as movable objects, from a labelled splat.

The empty home is the shell (shell.py); what stood in it comes back as separate objects the
owner can move, restyle or swap for products. Their splats only need to be recognizable: what
matters is each one's kind, footprint, height and heading. Gaussians are grouped by kind (the
frame labels, a few classes to a kind: a bed with its pillows, a desk with its table top),
each kind split into instances by where it stands (connected footprints), and each instance
boxed (yaw only). Clutter (clothes, bottles, towels, books) and curtains don't become objects.
Kitchen and bath pieces are marked fixtures: part of the home, kept in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import decompose, ply

# kind -> (classes, category, fixture)
KINDS: dict[str, tuple[frozenset, str, bool]] = {
    "kitchen": (frozenset({"countertop", "counter", "stove", "microwave", "refrigerator", "dishwasher", "oven", "hood",
                           "sink", "kitchen island"}), "kitchen", True),
    "bath": (frozenset({"toilet", "bathtub", "shower", "tank"}), "bath", True),
    "bed": (frozenset({"bed", "pillow", "blanket", "quilt"}), "bed", False),
    "sofa": (frozenset({"sofa", "cushion", "armchair"}), "sofa", False),
    "desk": (frozenset({"desk"}), "desk", False),
    "table": (frozenset({"table", "coffee table", "pool table"}), "table", False),
    "chair": (frozenset({"chair", "swivel chair", "stool", "seat", "bench", "ottoman"}), "chair", False),
    "storage": (frozenset({"shelf", "wardrobe", "chest of drawers", "bookcase", "cabinet", "buffet", "case"}),
                "storage", False),
    "box": (frozenset({"box", "basket", "trunk"}), "storage", False),
    "electronics": (frozenset({"computer", "monitor", "crt screen", "television receiver", "screen", "arcade machine"}),
                    "electronics", False),
    "plant": (frozenset({"plant", "flower", "pot", "vase"}), "decor", False),
    "lamp": (frozenset({"lamp"}), "lighting", False),
    "fan": (frozenset({"fan"}), "lighting", True),
    "bin": (frozenset({"ashcan"}), "storage", False),
}
NAMES = {"kitchen": "Kitchen run", "bath": "Bath fixture", "bed": "Bed", "sofa": "Sofa", "desk": "Desk",
         "table": "Table", "chair": "Chair", "storage": "Storage", "box": "Box", "electronics": "Screens",
         "plant": "Plant", "lamp": "Lamp", "fan": "Ceiling fan", "bin": "Bin"}

FIXTURE_NAMES = {"stove": "Stove", "oven": "Oven", "refrigerator": "Fridge", "microwave": "Microwave",
                 "dishwasher": "Dishwasher", "sink": "Sink", "hood": "Range hood", "countertop": "Counter and cabinets",
                 "counter": "Counter and cabinets", "cabinet": "Cabinets", "kitchen island": "Kitchen island",
                 "toilet": "Toilet", "bathtub": "Bathtub", "shower": "Shower", "tank": "Toilet"}


@dataclass
class Found:
    kind: str
    box: decompose.ObjectBox
    members: np.ndarray  # indices into the splat
    n_labelled: int
    main: str = ""  # its commonest class
    room: int = -1


def find_objects(s: ply.Splat, cls: np.ndarray, classes: list[str], rooms: list[np.ndarray], cell: float = 0.05,
                 min_gaussians: int = 150, min_size: float = 0.25, min_fixture: float = 0.4, min_opacity: float = 0.3,
                 kitchen_reach: float = 0.5) -> list[Found]:
    import cv2

    from .structure import STRUCTURE

    names = np.array([c.strip() for c in classes] + ["unseen"])[np.where(cls < 0, len(classes), cls)]
    X = s.xyz.astype(np.float64)
    opaque = (1 / (1 + np.exp(-s.opacity)) > min_opacity) if s.opacity is not None else np.ones(len(X), bool)
    allp = np.vstack(rooms)
    lo = allp.min(0) - 0.5
    shape = tuple((np.ceil((allp.max(0) + 0.5 - lo) / cell)).astype(int)[::-1])
    home = np.zeros(shape, np.uint8)
    room_of = np.full(shape, -1, np.int32)
    for k, p in enumerate(rooms):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round((np.asarray(p) - lo) / cell).astype(np.int32)], 1)
        home |= m
        room_of[(m > 0) & (room_of < 0)] = k
    ij = np.floor((X[:, :2] - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    inside = np.zeros(len(X), bool)
    inside[ok] = home[ij[ok, 1], ij[ok, 0]] > 0
    base = opaque & inside & ~np.isin(names, list(STRUCTURE | {"mirror", "windowpane", "glass", "unseen"}))

    def components(mask: np.ndarray) -> tuple[np.ndarray, int]:
        cnt = np.zeros(shape, np.int32)
        np.add.at(cnt, (ij[mask, 1], ij[mask, 0]), 1)
        occ = cv2.morphologyEx((cnt >= 2).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        lab, n = np.zeros(shape, np.int32), 1
        for k in range(len(rooms)):  # never across a wall
            nk, lk = cv2.connectedComponents(((occ > 0) & (room_of == k)).astype(np.uint8))
            lab[lk > 0] = lk[lk > 0] + n - 1
            n += nk - 1
        out = np.full(len(X), 0, np.int32)
        out[ok] = lab[ij[ok, 1], ij[ok, 0]]
        return out, n

    found: list[Found] = []
    kitchen_cells = np.zeros(shape, np.uint8)
    taken = np.zeros(len(X), bool)
    order = ["kitchen", "bath"] + [k for k in KINDS if k not in ("kitchen", "bath")]
    for kind in order:
        classes_k = KINDS[kind][0]
        m = base & np.isin(names, list(classes_k)) & ~taken
        if kind == "kitchen":  # cabinets beside the counters and appliances are the kitchen's
            m0 = base & np.isin(names, list(classes_k))
            cnt = np.zeros(shape, np.int32)
            np.add.at(cnt, (ij[m0, 1], ij[m0, 0]), 1)
            r = int(round(kitchen_reach / cell))
            kitchen_cells = cv2.dilate((cnt >= 2).astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8))
            cab = base & (names == "cabinet")
            near = np.zeros(len(X), bool)
            near[ok] = kitchen_cells[ij[ok, 1], ij[ok, 0]] > 0
            m |= cab & near
        if not m.any():
            continue
        lab, n = components(m)
        for j in range(1, n):
            mem = np.nonzero(m & (lab == j))[0]
            if len(mem) < min_gaussians:
                continue
            box = decompose.oriented_box(s.subset(np.isin(np.arange(len(X)), mem)))
            if max(box.w, box.d) < (min_fixture if KINDS[kind][2] else min_size) or box.h < 0.05:
                continue
            u, c = np.unique(names[mem], return_counts=True)
            rk = room_of[ij[mem, 1], ij[mem, 0]]
            found.append(Found(kind, box, mem, len(mem), str(u[np.argmax(c)]),
                               int(np.bincount(rk[rk >= 0]).argmax()) if (rk >= 0).any() else -1))
            taken[mem] = True
    # each object takes everything inside its box that isn't the shell: its unlabelled legs and
    # edges, a bed's clothes, a desk's clutter (the room's splat is the empty shell, so what isn't
    # in an object isn't shown)
    shell = np.isin(names, ["wall", "ceiling", "door", "screen door", "windowpane", "window", "mirror", "glass"])
    on_floor = np.isin(names, ["floor", "rug"]) & (X[:, 2] < 0.03)
    faint = (1 / (1 + np.exp(-s.opacity)) < 0.1) if s.opacity is not None else np.zeros(len(X), bool)
    if s.scale is not None:  # big, soft Gaussians are the room's haze, not the thing
        faint |= np.exp(s.scale).max(1) > 0.12
    rest = inside & ~shell & ~on_floor & ~faint & ~taken
    for f in sorted(found, key=lambda f: -f.box.w * f.box.d):
        poly = f.box.footprint(0.03)
        lo_z = 0.015 if f.box.z0 < 0.45 else f.box.z0 - 0.03  # down to the floor for what stands on it
        inb = rest & decompose.points_in_polygon(X[:, :2], poly) & (X[:, 2] >= lo_z) & \
            (X[:, 2] <= f.box.z0 + f.box.h + 0.15)
        f.members = np.union1d(f.members, np.nonzero(inb)[0])
        rest &= ~inb
    return found


def scene_objects(s: ply.Splat, found: list[Found], scene_dir: Path, walls=None) -> list:
    """Write objects/<id>.ply (canonical frame) for each and return SceneObjects."""
    from ..scene.models import Dims, Pose, SceneObject

    (scene_dir / "objects").mkdir(parents=True, exist_ok=True)
    out, count, seen_names = [], {}, {}
    boxes = {}
    for f in sorted(found, key=lambda f: (f.kind, -f.box.w * f.box.d)):
        count[f.kind] = count.get(f.kind, 0) + 1
        oid = f"scan_{f.kind}_{count[f.kind]}"
        boxes[oid] = f.box
        cat, fixture = KINDS[f.kind][1], KINDS[f.kind][2]
        base = NAMES[f.kind]
        if f.kind == "kitchen":
            bath_room = any(g.kind == "bath" and g.room == f.room for g in found)
            base = "Vanity" if bath_room and f.main in ("sink", "countertop", "counter", "cabinet") else \
                FIXTURE_NAMES.get(f.main, "Counter and cabinets")
            cat = "bath" if base == "Vanity" else cat
        elif f.kind == "bath":
            base = FIXTURE_NAMES.get(f.main, base)
        seen_names[base] = seen_names.get(base, 0) + 1
        name = base + (f" {seen_names[base]}" if seen_names[base] > 1 else "")
        out.append(SceneObject(id=oid, name=name, category=cat, asset=f"objects/{oid}.ply", asset_kind="splat",
                               pose=Pose(x=float(f.box.center[0]), y=float(f.box.center[1]), z=max(0.0, f.box.z0),
                                         yaw=float(f.box.yaw)),
                               dims=Dims(w=f.box.w, d=f.box.d, h=f.box.h), source="scanned", fixture=fixture,
                               support="floor"))
    # what stands on something else (a screen on a desk) is supported by it
    for o in out:
        b = boxes[o.id]
        under = [u for u in out if u.id != o.id and boxes[u.id].z0 < b.z0 - 0.1
                 and b.z0 < boxes[u.id].z0 + boxes[u.id].h + 0.15
                 and decompose.points_in_polygon(b.center[None], boxes[u.id].footprint(0.05))[0]]
        if under and b.z0 > 0.2:
            o.support = max(under, key=lambda u: boxes[u.id].w * boxes[u.id].d).id
        elif b.z0 < 0.45:  # on the floor: thin legs and shadowed feet are seldom labelled
            o.support = "floor"
            o.dims.h, o.pose.z = b.z0 + b.h, 0.0
        else:
            o.support = "ceiling" if o.id.startswith("scan_fan") else "wall"
    # the splats, in the objects' own frame: footprint centered, base at the pose's z (the floor,
    # for what stands on it)
    from dataclasses import replace as _replace

    for o, f in zip(out, sorted(found, key=lambda f: (f.kind, -f.box.w * f.box.d))):
        keep = np.isin(np.arange(len(s)), f.members)
        if s.opacity is not None:  # faint or big, soft Gaussians are the room's haze, not the thing
            keep &= 1 / (1 + np.exp(-s.opacity)) >= 0.1
        if s.scale is not None:
            keep &= np.exp(s.scale).max(1) <= 0.12
        # inside its box: the box leaves out the strays (floaters near the cameras), so they go too
        keep &= decompose.points_in_polygon(s.xyz[:, :2], f.box.footprint(0.05))
        keep &= (s.xyz[:, 2] >= f.box.z0 - 0.05) & (s.xyz[:, 2] <= f.box.z0 + f.box.h + 0.15)
        if o.support == "floor":
            keep &= s.xyz[:, 2] > 0.005  # nothing of it below the floor (floaters)
        sub = s.subset(keep)
        b = f.box if o.support != "floor" else _replace(f.box, z0=0.0)
        ply.write_ply(scene_dir / "objects" / f"{o.id}.ply", decompose.to_canonical(sub, b))
    return out


def settle(scene, rounds: int = 25) -> dict:
    """Scanned boxes are rough: nudge them so they sit inside the floor, under the ceiling and
    clear of one another (of two that overlap, the smaller movable one moves; fixtures stay
    unless they stick out of the room). Returns what is left over."""
    from ..scene import geometry as g
    from ..scene.checks import check_layout

    scanned = {o.id: o for o in scene.objects if o.source == "scanned"}
    moved = set()
    vs = []
    for _ in range(rounds):
        vs = [v for v in check_layout(scene) if v.kind in ("collision", "out_of_room", "crosses_wall")
              and all(i in scanned for i in v.objects)
              and not (v.kind != "collision" and scanned[v.objects[0]].fixture and "ceiling" not in v.message)]
        if not vs:
            break
        for v in vs:
            if v.kind == "collision":
                a, b = (scanned[i] for i in v.objects)
                area = lambda o: o.dims.w * o.dims.d  # noqa: E731
                mover, other = sorted((a, b), key=lambda o: (not o.fixture, -area(o)))[::-1] \
                    if a.fixture != b.fixture else sorted((a, b), key=area)
                if mover.fixture and not other.fixture:
                    mover, other = other, mover
                d = np.array([mover.pose.x - other.pose.x, mover.pose.y - other.pose.y])
                d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else np.array([1.0, 0.0])
                step = (v.amount_m or 0.02) + 0.01
            else:
                mover = scanned[v.objects[0]]
                if "ceiling" in v.message:
                    mover.dims.h = max(0.05, mover.dims.h - (v.amount_m or 0) - 0.01)
                    continue
                if mover.fixture:  # built in where it was filmed: the outline is what's short
                    continue
                fp = g.object_footprint(mover)  # its outline, every 1 cm: a thin overhang shows
                pts = np.concatenate([np.linspace(fp[i], fp[(i + 1) % len(fp)],
                                                  max(2, int(np.linalg.norm(fp[(i + 1) % len(fp)] - fp[i]) / 0.01)))
                                      for i in range(len(fp))])
                ins = np.zeros(len(pts), bool)
                for poly in scene.room.polygons():
                    ins |= g.points_in_polygon(pts, poly)
                if not ins.any() or ins.all():
                    continue
                d = pts[ins].mean(0) - pts[~ins].mean(0)
                d = d / max(np.linalg.norm(d), 1e-6)
                step = (v.amount_m or 0.02) + 0.01
            mover.pose.x += float(d[0] * step)
            mover.pose.y += float(d[1] * step)
            moved.add(mover.id)
    return {"moved": len(moved), "left": [v.message for v in vs]}


def add_scanned_objects(scene_dir: Path, splat_dir: Path) -> dict:
    """Find the objects in splat_dir's labelled splat (gauss_labels.npz) and put them in the scene
    (replacing scanned objects from before; products and uploads stay)."""
    from ..scene.models import Scene
    from .empty_room import spread_labels

    scene_dir, splat_dir = Path(scene_dir), Path(splat_dir)
    scene = Scene.model_validate_json((scene_dir / "scene.json").read_text())
    s = ply.read_ply(splat_dir / "splat.ply")
    z = np.load(splat_dir / "gauss_labels.npz")
    cls, classes = z["cls"].astype(int), [str(c).strip() for c in z["classes"]]
    cls = spread_labels(s.xyz.astype(np.float64), cls)
    found = find_objects(s, cls, classes, [np.array(sp.polygon, float) for sp in scene.room.spaces])
    objs = scene_objects(s, found, scene_dir)
    from ..scene import geometry as g

    for o in objs:
        if o.support == "wall" and scene.room.walls:
            o.support = f"wall:{g.nearest_wall(np.array([o.pose.x, o.pose.y]), scene.room)[0].id}"
    scene.objects = [o for o in scene.objects if o.source != "scanned"] + objs
    settled = settle(scene)
    (scene_dir / "scene.json").write_text(scene.model_dump_json(indent=2))
    return {"objects": len(objs), "fixtures": sum(o.fixture for o in objs), "settled": settled,
            "by_kind": {k: sum(1 for f in found if f.kind == k) for k in KINDS if any(f.kind == k for f in found)},
            "list": [(o.id, o.name, round(o.dims.w, 2), round(o.dims.d, 2), round(o.dims.h, 2), o.support) for o in objs]}


def compose(scene, scene_dir: Path, variant: str = "main", fixtures: bool = True):
    """One splat of a layout: the room's splat (the empty home) with every splat object where the
    layout puts it. For renders outside the editor; the editor composes the same way."""
    parts = []
    if scene.room.splat and (scene_dir / scene.room.splat).exists():
        parts.append(ply.read_ply(scene_dir / scene.room.splat))
    for o in scene.resolve(variant):
        if (not fixtures and o.fixture) or o.asset_kind != "splat" or not o.asset:
            continue
        if not (scene_dir / o.asset).exists():
            continue
        s = ply.read_ply(scene_dir / o.asset)
        parts.append(ply.transform(s, R=ply.rot_z(o.pose.yaw), t=np.array([o.pose.x, o.pose.y, o.pose.z])))
    if not parts:
        raise ValueError("nothing to compose: no room splat and no splat objects")
    k = max(0 if p.f_rest is None else p.f_rest.shape[1] for p in parts)  # pad view-dependent colour to one size
    for p in parts:
        n = 0 if p.f_rest is None else p.f_rest.shape[1]
        if n < k:
            p.f_rest = np.concatenate([np.zeros((len(p), n), np.float32) if p.f_rest is None else p.f_rest,
                                       np.zeros((len(p), k - n), np.float32)], 1)
        p.extra = {}
    return ply.Splat.concat(parts)
