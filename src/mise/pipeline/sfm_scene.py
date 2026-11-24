"""A home scene straight from camera poses + sparse points (no splat, no GPU).

reconstruction --(scale)--> leveled, wall-aligned metric frame --> free space -->
rooms split at doorways --> scene.json with one space per room and the doorways
as doors. Name and type the spaces afterwards (editor or API); split open-plan
rooms into zones there too.

Needs pycolmap and opencv (the `poses` extra).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene.models import Opening, Room, Scene, Space
from . import normalize, ply, room_outline


@dataclass
class Leveled:
    R: np.ndarray  # model -> leveled rotation (includes Manhattan yaw)
    t: np.ndarray  # translation after scaling: p_level = scale * R @ p + t
    scale: float
    points: np.ndarray  # Nx3 leveled sparse points
    point_ids: np.ndarray
    cams: np.ndarray  # Mx3 leveled camera centers
    image_names: list[str]
    sights: list[np.ndarray]  # per camera: leveled points it observed
    yaw_deg: float

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(p) @ self.R.T) + self.t

    def pose_sanity(self, lo: float = 0.8, hi: float = 2.2) -> dict:
        """A handheld walkthrough keeps the phone between ~1 and 1.8 m. Heights far outside that,
        or a wide spread, mean a bent reconstruction or a wrong floor: don't build a scene from it."""
        z = self.cams[:, 2]
        bad = float(((z < lo) | (z > hi)).mean())
        q1, med, q3 = np.percentile(z, [25, 50, 75])
        ok = bad < 0.05 and (q3 - q1) < 0.4 and lo < med < hi
        return {"median_m": round(float(med), 2), "iqr_m": round(float(q3 - q1), 2),
                "outside_band": round(bad, 3), "ok": bool(ok)}


def level_reconstruction(rec, scale: float, floor_thresh_m: float = 0.03) -> Leveled:
    """Floor to z = 0 (RANSAC, guided by the cameras' up direction), walls onto x/y, meters."""
    ids = np.array(list(rec.points3D.keys()))
    xyz = np.array([rec.points3D[i].xyz for i in ids]) * scale
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    ups = np.array([-im.cam_from_world().rotation.matrix()[1] for im in imgs])  # camera -y is up in the image
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    floor = normalize.ransac_plane(xyz, thresh=floor_thresh_m, up_hint=up, max_tilt_deg=35, iters=800)
    R1 = ply.rotation_between(floor.n, np.array([0.0, 0, 1]))
    p1 = xyz @ R1.T
    p1[:, 2] += floor.d
    band = (p1[:, 2] > 0.3) & (p1[:, 2] < 2.0)
    yaw = normalize.manhattan_yaw(p1[band, :2], bin_size=0.03) if band.sum() > 100 else 0.0
    R = ply.rot_z(yaw) @ R1
    t = ply.rot_z(yaw) @ np.array([0, 0, floor.d])
    pts = xyz @ R.T + t
    idx = {int(i): k for k, i in enumerate(ids)}
    cams, sights = [], []
    for im in imgs:
        cams.append(scale * (R @ im.projection_center()) + t)
        seen = [idx[int(p.point3D_id)] for p in im.points2D if p.has_point3D() and int(p.point3D_id) in idx]
        sights.append(pts[seen] if seen else np.zeros((0, 3)))
    return Leveled(R=R, t=t, scale=scale, points=pts, point_ids=ids, cams=np.array(cams),
                   image_names=[im.name for im in imgs], sights=sights, yaw_deg=yaw)


def _door_on_wall(room: Room, space_id: str, center, width: float, door_id: str) -> Opening | None:
    best = None
    for w in room.walls:
        if w.space != space_id:
            continue
        a, b = np.array(w.a), np.array(w.b)
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / L
        along = float((np.array(center) - a) @ u)
        off = abs(float((np.array(center) - a) @ np.array([-u[1], u[0]])))
        if -0.1 <= along <= L + 0.1 and (best is None or off < best[0]):
            best = (off, w, min(max(along, width / 2), L - width / 2))
    if best is None or best[0] > 0.5:
        return None
    _, w, along = best
    return Opening(id=door_id, kind="door", wall=w.id, center=along, width=min(width, w.length), top=2.03, swing="none")


def home_scene(home: room_outline.HomeOutline, scene_id: str, name: str, ceiling: float | None = None) -> Scene:
    """Spaces from the room outlines (named Room 1, Room 2...), doorways as doors on both sides."""
    ceil = ceiling or next((r.ceiling_m for r in home.rooms if r.ceiling_m), None) or 2.5
    spaces = [Space(id=f"room_{i + 1}", name=f"Room {i + 1}", kind="other", polygon=r.polygon)
              for i, r in enumerate(home.rooms)]
    room = Room(spaces=spaces, ceiling_height=round(ceil, 2))
    openings = []
    for k, d in enumerate(home.doorways):
        for side in d.rooms:  # one door record per side, so each room's walls show the opening
            op = _door_on_wall(room, f"room_{side + 1}", d.center, d.width, f"door_{k + 1}_{side + 1}")
            if op:
                openings.append(op)
    room.openings = openings
    return Scene(id=scene_id, name=name, room=room)
