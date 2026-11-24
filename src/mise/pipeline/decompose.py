"""Stage 2 steps 4-6: split a labeled room splat, fill the floor, package objects.

Segmentation itself (a per-Gaussian instance id) comes from Gaussian
Grouping or InstaScene, and object completion from SAM 3D; see
external.py for those commands. This module takes their output and
produces `room_empty.ply`, `objects/<id>.ply`, `layout.json` and a
`scene.json` the backend can load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..scene.geometry import min_area_rect, points_in_polygon
from ..scene.models import Dims, Opening, Pose, Room, Scene, SceneObject
from . import ply


@dataclass
class ObjectBox:
    center: np.ndarray  # x, y on floor
    w: float
    d: float
    h: float
    z0: float
    yaw: float

    def footprint(self, pad: float = 0.0) -> np.ndarray:
        a = np.radians(self.yaw)
        c, s = np.cos(a), np.sin(a)
        hw, hd = self.w / 2 + pad, self.d / 2 + pad
        loc = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
        return loc @ np.array([[c, -s], [s, c]]).T + self.center


def oriented_box(s: ply.Splat, alpha_min: float = 0.3, trim: float = 0.5) -> ObjectBox:
    """Yaw-only oriented bounding box of an object's Gaussians (z up)."""
    keep = s.alpha() > alpha_min
    p = s.xyz[keep].astype(np.float64)
    lo = np.percentile(p, trim, axis=0)
    hi = np.percentile(p, 100 - trim, axis=0)
    p = p[np.all((p >= lo) & (p <= hi), axis=1)]
    center, w, d, yaw = min_area_rect(p[:, :2])
    return ObjectBox(center=center, w=w, d=d, h=float(p[:, 2].max() - p[:, 2].min()), z0=float(p[:, 2].min()), yaw=yaw)


def to_canonical(s: ply.Splat, box: ObjectBox) -> ply.Splat:
    """Center on x/y, footprint on z=0, box axes on x/y (front = +y by convention; check in the editor)."""
    moved = ply.transform(s, t=-np.array([box.center[0], box.center[1], box.z0]))
    return ply.transform(moved, R=ply.rot_z(-box.yaw))


def split_instances(s: ply.Splat, labels: np.ndarray, background: set[int] = frozenset({0}),
                    min_gaussians: int = 200) -> tuple[ply.Splat, dict[int, ply.Splat]]:
    objs = {}
    for lab in np.unique(labels):
        if int(lab) in background:
            continue
        m = labels == lab
        if m.sum() >= min_gaussians:
            objs[int(lab)] = s.subset(m)
    bg_mask = np.isin(labels, list(background)) | ~np.isin(labels, list(objs))
    return s.subset(bg_mask), objs


def remove_with_shadow(room: ply.Splat, boxes: list[ObjectBox], dilate: float = 0.15, floor_z: float = 0.03) -> tuple[ply.Splat, np.ndarray]:
    """Delete Gaussians inside each object's box, dilated to catch contact shadows on the floor.

    Returns the cleaned splat and a boolean (N,) mask of floor Gaussians removed
    (the hole to inpaint).
    """
    xy = room.xyz[:, :2].astype(np.float64)
    z = room.xyz[:, 2]
    remove = np.zeros(len(room), bool)
    floor_hole = np.zeros(len(room), bool)
    for b in boxes:
        inside_fp = points_in_polygon(xy, b.footprint(pad=0.02))
        remove |= inside_fp & (z > b.z0 - 0.02) & (z < b.z0 + b.h + 0.02)
        shadow = points_in_polygon(xy, b.footprint(pad=dilate)) & (np.abs(z) < floor_z)
        floor_hole |= shadow
    remove |= floor_hole
    return room.subset(~remove), floor_hole


def fill_floor_plane(room: ply.Splat, holes: list[np.ndarray], floor_z: float = 0.03, offset_tries: int = 24,
                     rng: np.random.Generator | None = None) -> ply.Splat:
    """Plane-fit + texture-copy baseline (H2's control arm).

    For each hole polygon, copy floor Gaussians from the nearest same-size
    patch of intact floor (translated), so the texture continues.
    """
    rng = rng or np.random.default_rng(0)
    floor = room.subset(np.abs(room.xyz[:, 2]) < floor_z)
    fxy = floor.xyz[:, :2].astype(np.float64)
    patches = []
    for poly in holes:
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        size = hi - lo
        best = None
        for _ in range(offset_tries):
            ang = rng.uniform(0, 2 * np.pi)
            dist = np.linalg.norm(size) * rng.uniform(0.6, 1.5)
            off = np.array([np.cos(ang), np.sin(ang)]) * dist
            src = poly + off
            m = points_in_polygon(fxy, src)
            density = m.sum() / max(float(np.prod(size)), 1e-6)
            if best is None or density > best[0]:
                best = (density, off, m)
        if best is None or best[2].sum() == 0:
            continue
        patch = floor.subset(best[2])
        patch = ply.transform(patch, t=np.array([-best[1][0], -best[1][1], 0.0]))
        # keep only what lands inside the hole
        patch = patch.subset(points_in_polygon(patch.xyz[:, :2].astype(np.float64), poly))
        patch.xyz[:, 2] = np.clip(patch.xyz[:, 2], -floor_z, floor_z)
        patches.append(patch)
    return ply.Splat.concat([room, *patches]) if patches else room


def package(out_dir: Path, room_empty: ply.Splat, objects: dict[str, tuple[ply.Splat, str, str]],
            floor_polygon: list[tuple[float, float]], ceiling_height: float,
            openings: list[Opening] | None = None, scene_id: str = "room", name: str = "Scanned room") -> Scene:
    """Write room_empty.ply, objects/<id>.ply, layout.json and scene.json.

    objects: id -> (splat in room frame, display name, category)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ply.write_ply(out_dir / "room_empty.ply", room_empty)
    scene_objs = []
    for oid, (s, nm, cat) in objects.items():
        box = oriented_box(s)
        canon = to_canonical(s, box)
        rel = f"objects/{oid}.ply"
        ply.write_ply(out_dir / rel, canon)
        meta = {"id": oid, "name": nm, "category": cat, "dims": {"w": box.w, "d": box.d, "h": box.h},
                "pose": {"x": float(box.center[0]), "y": float(box.center[1]), "z": box.z0, "yaw": box.yaw}}
        (out_dir / f"objects/{oid}.json").write_text(json.dumps(meta, indent=2))
        scene_objs.append(SceneObject(
            id=oid, name=nm, category=cat, asset=rel, asset_kind="splat",
            pose=Pose(x=float(box.center[0]), y=float(box.center[1]), z=max(0.0, box.z0), yaw=box.yaw),
            dims=Dims(w=box.w, d=box.d, h=box.h), source="scanned",
        ))
    room = Room(splat="room_empty.ply", floor_polygon=floor_polygon, ceiling_height=ceiling_height,
                openings=openings or [])
    layout = {"floor_polygon": floor_polygon, "ceiling_height": ceiling_height,
              "walls": [w.model_dump() for w in room.walls], "openings": [o.model_dump() for o in room.openings]}
    (out_dir / "layout.json").write_text(json.dumps(layout, indent=2))
    scene = Scene(id=scene_id, name=name, room=room, objects=scene_objs)
    (out_dir / "scene.json").write_text(scene.model_dump_json(indent=2))
    return scene


def floor_polygon_from_splat(s: ply.Splat, floor_z: float = 0.03) -> list[tuple[float, float]]:
    """Axis-aligned rectangle around the floor (walls are already x/y aligned). Edit by hand for L-shapes."""
    f = s.xyz[(np.abs(s.xyz[:, 2]) < floor_z) & (s.alpha() > 0.3)][:, :2].astype(np.float64)
    lo, hi = np.percentile(f, 1, axis=0), np.percentile(f, 99, axis=0)
    return [(float(lo[0]), float(lo[1])), (float(hi[0]), float(lo[1])), (float(hi[0]), float(hi[1])), (float(lo[0]), float(hi[1]))]
