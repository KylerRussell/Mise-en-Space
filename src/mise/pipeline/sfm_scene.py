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
    return _leveled(rec, R, t, scale, yaw)


def _leveled(rec, R: np.ndarray, t: np.ndarray, scale: float, yaw: float) -> Leveled:
    ids = np.array(list(rec.points3D.keys()))
    pts = scale * (np.array([rec.points3D[i].xyz for i in ids]).reshape(-1, 3) @ R.T) + t
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    idx = {int(i): k for k, i in enumerate(ids)}
    cams, sights = [], []
    for im in imgs:
        cams.append(scale * (R @ im.projection_center()) + t)
        seen = [idx[int(p.point3D_id)] for p in im.points2D if p.has_point3D() and int(p.point3D_id) in idx]
        sights.append(pts[seen] if seen else np.zeros((0, 3)))
    return Leveled(R=R, t=t, scale=scale, points=pts, point_ids=ids, cams=np.array(cams),
                   image_names=[im.name for im in imgs], sights=sights, yaw_deg=yaw)


def _world_from_cam(im) -> np.ndarray:
    return im.cam_from_world().rotation.matrix().T


def align_by_time(ref_rec, ref: Leveled, ref_times: dict[str, float], rec, times: dict[str, float],
                  max_dt: float = 0.2, max_gap: float = 0.8, outlier_deg: float = 5.0) -> tuple[Leveled, dict]:
    """Put a separately solved piece of the same walkthrough into `ref`'s leveled metric frame.

    The two solves share stretches of video. Frames close in time see the camera in the same
    place facing the same way, so: rotation from the camera orientations (robust mean), then
    scale and offset from the camera centers (the reference path interpolated to each frame's
    time). Needs a shared stretch where the camera moves, ideally turns, for a few seconds.
    """
    ref_ims = sorted(ref_rec.images.values(), key=lambda im: ref_times[im.name])
    rt = np.array([ref_times[im.name] for im in ref_ims])
    name_k = {n: k for k, n in enumerate(ref.image_names)}
    rc = np.array([ref.cams[name_k[im.name]] for im in ref_ims])
    pairs = []
    for im in sorted(rec.images.values(), key=lambda im: times[im.name]):
        tt = times[im.name]
        i = int(np.searchsorted(rt, tt))
        if not 0 < i < len(rt) or rt[i] - rt[i - 1] > max_gap:
            continue
        j = i if rt[i] - tt < tt - rt[i - 1] else i - 1
        if abs(rt[j] - tt) > max_dt:
            continue
        w = (tt - rt[i - 1]) / (rt[i] - rt[i - 1])
        pairs.append((im, ref_ims[j], rc[i - 1] * (1 - w) + rc[i] * w))
    if len(pairs) < 8:
        raise RuntimeError(f"only {len(pairs)} frames shared in time; solve an overlapping stretch")
    Rs = np.array([ref.R @ _world_from_cam(b) @ _world_from_cam(a).T for a, b, _ in pairs])

    def mean_rot(M):
        U, _, Vt = np.linalg.svd(M.sum(0))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    R = mean_rot(Rs)
    for _ in range(3):  # drop frames whose orientation disagrees (mis-registered or wrong time match)
        ang = np.degrees(np.arccos(np.clip((np.einsum("kij,ij->k", Rs, R) - 1) / 2, -1, 1)))
        keep = ang < max(outlier_deg, np.median(ang) * 3)
        R = mean_rot(Rs[keep])
    X = np.array([a.projection_center() for a, _, _ in pairs])[keep] @ R.T
    Y = np.array([c for _, _, c in pairs])[keep]
    for _ in range(3):  # scale + translation, trimmed
        xm, ym = X.mean(0), Y.mean(0)
        s = float(((X - xm) * (Y - ym)).sum() / ((X - xm) ** 2).sum())
        t = ym - s * xm
        res = np.linalg.norm(s * X + t - Y, axis=1)
        good = res <= max(0.1, 2.5 * np.median(res))
        X, Y = X[good], Y[good]
    xm, ym = X.mean(0), Y.mean(0)
    s = float(((X - xm) * (Y - ym)).sum() / ((X - xm) ** 2).sum())
    t = ym - s * xm
    res = np.linalg.norm(s * X + t - Y, axis=1)
    lev = _leveled(rec, R, t, s, ref.yaw_deg)
    report = {"shared": len(pairs), "used": int(len(X)), "rot_spread_deg": round(float(np.median(ang[keep])), 2),
              "center_residual_m": round(float(np.median(res)), 3), "path_m": round(float(np.linalg.norm(
                  np.diff(Y, axis=0), axis=1).sum()), 2), "scale": round(s, 5)}
    return lev, report


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
    spaces, n_room, n_closet = [], 0, 0
    for i, r in enumerate(home.rooms):
        if getattr(r, "kind", "room") == "closet":
            n_closet += 1
            spaces.append(Space(id=f"room_{i + 1}", name=f"Closet {n_closet}", kind="closet", polygon=r.polygon))
        else:
            n_room += 1
            spaces.append(Space(id=f"room_{i + 1}", name=f"Room {n_room}", kind="other", polygon=r.polygon))
    room = Room(spaces=spaces, ceiling_height=round(ceil, 2))
    openings = []
    for k, d in enumerate(home.doorways):
        for side in d.rooms:  # one door record per side, so each room's walls show the opening
            op = _door_on_wall(room, f"room_{side + 1}", d.center, d.width, f"door_{k + 1}_{side + 1}")
            if op:
                openings.append(op)
    room.openings = openings
    return Scene(id=scene_id, name=name, room=room)
