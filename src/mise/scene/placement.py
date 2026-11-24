"""Snapping and relational placement ("against wall 2", "facing object 4")."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel

from . import geometry as g
from .models import Pose, Room, SceneObject


class Relation(BaseModel):
    """A structured placement request.

    relation:
      * ``against_wall``  target = wall id; ``along`` = position along the wall
        (m from its start; default centered); ``distance`` = gap to the wall.
      * ``facing``        target = object id; turns to face it (keeps x, y).
      * ``in_front_of`` / ``behind`` / ``left_of`` / ``right_of``
        target = object id, relative to the target's own frame, ``distance`` = gap.
      * ``on_top_of``     target = object id; centered on it, support set.
      * ``center``        center of the room's bounding box (or ``at`` = [x, y]).
    """

    relation: Literal["against_wall", "facing", "in_front_of", "behind", "left_of", "right_of", "on_top_of", "center"]
    target: str | None = None
    distance: float = 0.0
    along: float | None = None
    at: tuple[float, float] | None = None
    face_target: bool = False


class Placement(BaseModel):
    pose: Pose
    support: str
    note: str = ""


def snap_yaw(yaw: float, step: float = 15.0) -> float:
    return g.wrap_deg(round(yaw / step) * step)


def support_height(support: str, objects: dict[str, SceneObject]) -> float | None:
    if support == "floor":
        return 0.0
    if support.startswith("wall:"):
        return None  # wall-mounted: keep z
    base = objects.get(support)
    if base is None:
        return 0.0
    return base.pose.z + base.extent().h


def apply_support(obj: SceneObject, objects: dict[str, SceneObject]) -> None:
    z = support_height(obj.support, objects)
    if z is not None:
        obj.pose.z = round(z, 4)


def snap_to_wall(obj: SceneObject, room: Room, max_dist: float = 0.15, max_angle: float = 20.0) -> str | None:
    """If the object's back is near a wall and roughly parallel, push it flush.

    Returns the wall id it snapped to, or None.
    """
    center = np.array([obj.pose.x, obj.pose.y])
    right, front = g.local_axes(obj.pose.yaw)
    back_mid = center - front * obj.extent().d / 2
    wall, dist, along = g.nearest_wall(back_mid, room)
    n = g.wall_inward_normal(wall, room)
    ang = math.degrees(math.acos(float(np.clip(np.dot(front, n), -1, 1))))
    if dist > max_dist or ang > max_angle:
        return None
    yaw = g.yaw_facing(n)
    a = np.array(wall.a)
    foot = a + g.wall_direction(wall) * along
    new_center = foot + n * obj.extent().d / 2
    obj.pose.x, obj.pose.y, obj.pose.yaw = float(new_center[0]), float(new_center[1]), round(yaw, 6)
    return wall.id


def snap(obj: SceneObject, room: Room, objects: dict[str, SceneObject], yaw_step: float = 15.0, wall_dist: float = 0.15) -> list[str]:
    notes = []
    obj.pose.yaw = snap_yaw(obj.pose.yaw, yaw_step)
    if obj.support == "floor":
        wid = snap_to_wall(obj, room, wall_dist)
        if wid:
            notes.append(f"snapped back to wall {wid}")
    apply_support(obj, objects)
    return notes


def resolve_relation(obj: SceneObject, rel: Relation, room: Room, objects: dict[str, SceneObject]) -> Placement:
    pose = obj.pose.model_copy()
    support = obj.support if obj.support != "floor" or rel.relation != "on_top_of" else obj.support

    if rel.relation == "against_wall":
        if not rel.target:
            raise ValueError("against_wall needs a wall id as target")
        wall = room.wall(rel.target)
        n = g.wall_inward_normal(wall, room)
        along = wall.length / 2 if rel.along is None else rel.along
        half = obj.extent().w / 2
        along = float(np.clip(along, half, max(half, wall.length - half)))
        p = g.wall_point(wall, along) + n * (rel.distance + obj.extent().d / 2)
        pose.x, pose.y, pose.yaw = float(p[0]), float(p[1]), g.yaw_facing(n)
        support = "floor"
        note = f"back against wall {wall.id}, {along:.2f} m along it"
    elif rel.relation == "center":
        if rel.at:
            pose.x, pose.y = rel.at
        else:
            poly = np.asarray(room.floor_polygon)
            c = (poly.min(axis=0) + poly.max(axis=0)) / 2
            pose.x, pose.y = float(c[0]), float(c[1])
        support = "floor"
        note = "centered"
    else:
        tgt = objects.get(rel.target or "")
        if tgt is None:
            raise KeyError(f"no object {rel.target!r}")
        tc = np.array([tgt.pose.x, tgt.pose.y])
        t_right, t_front = g.local_axes(tgt.pose.yaw)
        if rel.relation == "facing":
            v = tc - np.array([pose.x, pose.y])
            if np.linalg.norm(v) < 1e-6:
                raise ValueError("object is at the target's position")
            pose.yaw = g.yaw_facing(v / np.linalg.norm(v))
            note = f"facing {tgt.id}"
        elif rel.relation == "on_top_of":
            pose.x, pose.y = float(tc[0]), float(tc[1])
            support = tgt.id
            note = f"on top of {tgt.id}"
        else:
            if rel.relation == "in_front_of":
                p = tc + t_front * (tgt.extent().d / 2 + rel.distance + obj.extent().d / 2)
                yaw = g.yaw_facing(-t_front) if rel.face_target else pose.yaw
            elif rel.relation == "behind":
                p = tc - t_front * (tgt.extent().d / 2 + rel.distance + obj.extent().d / 2)
                yaw = g.yaw_facing(t_front) if rel.face_target else pose.yaw
            elif rel.relation == "left_of":
                # target's left as seen by someone facing the target's front
                p = tc + t_right * (tgt.extent().w / 2 + rel.distance + obj.extent().w / 2)
                yaw = tgt.pose.yaw
            else:  # right_of
                p = tc - t_right * (tgt.extent().w / 2 + rel.distance + obj.extent().w / 2)
                yaw = tgt.pose.yaw
            pose.x, pose.y, pose.yaw = float(p[0]), float(p[1]), g.wrap_deg(yaw)
            support = "floor" if tgt.support == "floor" else tgt.support
            note = f"{rel.relation.replace('_', ' ')} {tgt.id}, gap {rel.distance:.2f} m"
    tmp = obj.model_copy(deep=True)
    tmp.pose, tmp.support = pose, support
    apply_support(tmp, objects)
    return Placement(pose=tmp.pose, support=support, note=note)
