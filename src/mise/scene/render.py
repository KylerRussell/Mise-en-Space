"""Labeled renders for the agent (and quick previews): plan, eye-level and close-up.

These are deterministic box renders made with Pillow: each object is drawn
from its oriented bounding box and tagged with a number, the visual marking
LayoutVLM uses for spatial grounding. The label map ties numbers to ids.
For photoreal splat views, capture from the web editor (`?view=eye&capture=1`).
"""

from __future__ import annotations

import io
import math
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as g
from .checks import Violation
from .models import Room, SceneObject

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex(c: str, alpha: int = 255) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha)


def label_map(objects: list[SceneObject]) -> dict[int, str]:
    return {i + 1: o.id for i, o in enumerate(objects)}


def _centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return poly.mean(0)
    return np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * a)


def _dashed(dr, a, b, color, width=2, dash=10):
    a, b = np.array(a, float), np.array(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return
    u = (b - a) / L
    for t in np.arange(0, L, 2 * dash):
        dr.line([tuple(a + u * t), tuple(a + u * min(t + dash, L))], fill=color, width=width)


def render_plan(room: Room, objects: list[SceneObject], violations: list[Violation] | None = None,
                px_per_m: int = 120, margin: float = 0.4, show_dims: bool = True, focus: str | None = None,
                space: str | None = None) -> tuple[bytes, dict[int, str]]:
    spaces = room.space_list()
    multi = len(spaces) > 1
    if space:
        sp = np.asarray(room.space(space).polygon, float)
        lo, hi = sp.min(0) - margin - 0.3, sp.max(0) + margin + 0.3
    else:
        (lx, ly), (hx, hy) = room.bounds()
        lo, hi = np.array([lx, ly]) - margin, np.array([hx, hy]) + margin
    if multi and not space:
        px_per_m = min(px_per_m, int(2400 / max(hi - lo)))  # keep whole-home plans a sane size
    if focus:
        o = next(o for o in objects if o.id == focus)
        fp = g.object_footprint(o)
        lo, hi = fp.min(0) - 1.0, fp.max(0) + 1.0
    W, H = (hi - lo) * px_per_m
    img = Image.new("RGBA", (int(W), int(H)), (250, 249, 246, 255))
    dr = ImageDraw.Draw(img, "RGBA")
    f_small, f_big = _font(max(11, px_per_m // 10)), _font(max(14, px_per_m // 7))

    def P(xy) -> tuple[float, float]:
        x, y = xy
        return ((x - lo[0]) * px_per_m, (hi[1] - y) * px_per_m)  # y up on screen

    # grid every 0.5 m
    for gx in np.arange(math.floor(lo[0] * 2) / 2, hi[0], 0.5):
        dr.line([P((gx, lo[1])), P((gx, hi[1]))], fill=(0, 0, 0, 18))
    for gy in np.arange(math.floor(lo[1] * 2) / 2, hi[1], 0.5):
        dr.line([P((lo[0], gy)), P((hi[0], gy))], fill=(0, 0, 0, 18))
    for sp_ in spaces:
        pl = np.asarray(sp_.polygon, float)
        fill = (255, 255, 255, 255) if not space or sp_.id == space else (243, 242, 238, 255)
        dr.polygon([P(p) for p in pl], fill=fill)
        for i in range(len(pl)):  # space outlines show as faint dashes where there is no wall
            _dashed(dr, P(pl[i]), P(pl[(i + 1) % len(pl)]), (150, 150, 150, 160))
    for w in room.walls:
        dr.line([P(w.a), P(w.b)], fill=(40, 40, 40, 255), width=5)
    if multi:
        f_space = _font(max(16, px_per_m // 5))
        for sp_ in spaces:  # name in the space's top-left corner, clear of most furniture
            pl = np.asarray(sp_.polygon, float)
            corner = np.array([pl[:, 0].min() + 0.25, pl[:, 1].max() - 0.25])
            if not g.point_in_polygon(corner, pl):
                corner = _centroid(pl)
            dr.text(P(corner), f"{sp_.name}  {sp_.area():.1f} m²", fill=(150, 140, 125, 255), font=f_space, anchor="la")
    for w in room.walls:
        a, b = np.array(w.a), np.array(w.b)
        if space and w.space != space:
            continue
        mid = (a + b) / 2 + g.wall_inward_normal(w, room) * 0.18
        dr.text(P(mid), f"W{w.id}", fill=(120, 120, 120, 255), font=f_small, anchor="mm")
        if show_dims and not focus and (space or not multi):
            out = (a + b) / 2 - g.wall_inward_normal(w, room) * 0.2
            dr.text(P(out), f"{w.length:.2f} m", fill=(90, 90, 90, 255), font=f_small, anchor="mm")
    for op in room.openings:
        poly3 = g.opening_polygon(op, room)
        a, b = poly3[0][:2], poly3[1][:2]
        col = (60, 130, 200, 255) if op.kind == "window" else (180, 110, 40, 255)
        dr.line([P(a), P(b)], fill=col, width=7)
        if op.kind == "door" and op.swing in ("in", "out"):
            arc = g.door_swing_arc(op, room)
            dr.line([P(arc[0]), P(arc[-1])], fill=col, width=2)
            dr.line([P(p) for p in arc[1:]], fill=(*col[:3], 140), width=2)
        mid = (np.array(a) + np.array(b)) / 2 + g.wall_inward_normal(room.wall(op.wall), room) * 0.12
        dr.text(P(mid), op.id, fill=col, font=f_small, anchor="mm")

    bad = {oid for v in (violations or []) if v.severity == "hard" for oid in v.objects}
    labels = label_map(objects)
    order = sorted(range(len(objects)), key=lambda i: objects[i].pose.z + (0 if objects[i].category != "rug" else -10))
    for i in order:
        o = objects[i]
        n = i + 1
        fp = g.object_footprint(o)
        base = o.color or PALETTE[i % len(PALETTE)]
        fill = _hex(base, 90 if o.category == "rug" else 170)
        outline = (220, 30, 30, 255) if o.id in bad else (30, 30, 30, 255)
        dr.polygon([P(p) for p in fp], fill=fill, outline=outline, width=3 if o.id in bad else 2)
        # front marker
        right, front = g.local_axes(o.pose.yaw)
        c = np.array([o.pose.x, o.pose.y])
        f0 = c + front * o.extent().d / 2
        dr.line([P(f0 - right * o.extent().w * 0.3), P(f0 + right * o.extent().w * 0.3)], fill=(0, 0, 0, 200), width=4)
        r = max(10, px_per_m // 8)
        cx, cy = P(c)
        if o.support != "floor" and not o.support.startswith("wall:"):
            cx, cy = cx + 1.6 * r, cy - 1.6 * r  # stacked: keep the label clear of its support's
            dr.line([P(c), (cx, cy)], fill=(20, 20, 20, 200), width=2)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20, 230))
        dr.text((cx, cy), str(n), fill=(255, 255, 255, 255), font=f_big, anchor="mm")
        if show_dims:
            dr.text((cx, cy + r + 2), f"{o.extent().w:.2f}x{o.extent().d:.2f}", fill=(40, 40, 40, 220), font=f_small, anchor="ma")
    # scale bar
    x0, y0 = 12, img.height - 18
    dr.line([(x0, y0), (x0 + px_per_m, y0)], fill=(0, 0, 0, 255), width=3)
    dr.text((x0 + px_per_m + 6, y0), "1 m", fill=(0, 0, 0, 255), font=f_small, anchor="lm")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue(), labels


def _box_faces(o: SceneObject) -> list[np.ndarray]:
    fp = g.object_footprint(o)
    z0, z1 = o.pose.z, o.pose.z + o.extent().h
    bot = [np.array([*p, z0]) for p in fp]
    top = [np.array([*p, z1]) for p in fp]
    faces = [np.array(top)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append(np.array([bot[i], bot[j], top[j], top[i]]))
    return faces


def render_perspective(room: Room, objects: list[SceneObject], eye: tuple[float, float, float],
                       target: tuple[float, float, float], fov_deg: float = 70, size: tuple[int, int] = (960, 640),
                       highlight: str | None = None) -> tuple[bytes, dict[int, str]]:
    W, H = size
    eye_v, tgt = np.array(eye, float), np.array(target, float)
    fwd = tgt - eye_v
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = (W / 2) / math.tan(math.radians(fov_deg) / 2)

    def proj(p: np.ndarray):
        v = p - eye_v
        z = float(v @ fwd)
        if z < 0.05:
            return None, z
        return (W / 2 + f * float(v @ right) / z, H / 2 - f * float(v @ up) / z), z

    img = Image.new("RGBA", (W, H), (236, 233, 226, 255))
    dr = ImageDraw.Draw(img, "RGBA")
    polys = []  # (depth, pts, fill, outline)
    # walls (as quads); clip crudely by skipping faces with points behind the camera
    for w in room.walls:
        a, b = np.array([*w.a, 0.0]), np.array([*w.b, 0.0])
        h = w.height or room.ceiling_height
        quad = [a, b, b + [0, 0, h], a + [0, 0, h]]
        pr = [proj(q) for q in quad]
        if all(p[0] is not None for p in pr):
            polys.append((max(p[1] for p in pr) + 100, [p[0] for p in pr], (245, 243, 238, 255), (180, 175, 165, 255)))
    for pl in room.polygons():  # floors
        fl = [proj(np.array([*p, 0.0])) for p in pl]
        if all(p[0] is not None for p in fl):
            polys.append((1e9, [p[0] for p in fl], (214, 200, 178, 255), (150, 140, 120, 255)))
    for op in room.openings:
        quad = [np.array(q) for q in g.opening_polygon(op, room)]
        quad = [q + np.array([*g.wall_inward_normal(room.wall(op.wall), room), 0]) * 0.01 for q in quad]
        pr = [proj(q) for q in quad]
        if all(p[0] is not None for p in pr):
            col = (170, 205, 235, 255) if op.kind == "window" else (160, 120, 80, 255)
            polys.append((max(p[1] for p in pr) + 50, [p[0] for p in pr], col, (90, 90, 90, 255)))
    labels = label_map(objects)
    centers = []
    for i, o in enumerate(objects):
        base = _hex(o.color or PALETTE[i % len(PALETTE)])
        for k, face in enumerate(_box_faces(o)):
            pr = [proj(q) for q in face]
            if any(p[0] is None for p in pr):
                continue
            shade = 1.0 if k == 0 else 0.8 - 0.05 * (k % 2)
            col = tuple(int(c * shade) for c in base[:3]) + (255,)
            outline = (220, 30, 30, 255) if o.id == highlight else (40, 40, 40, 255)
            polys.append((float(np.mean([p[1] for p in pr])), [p[0] for p in pr], col, outline))
        c, z = proj(np.array([o.pose.x, o.pose.y, o.pose.z + o.extent().h + 0.05]))
        if c:
            centers.append((i + 1, c))
    for _, pts, fill, outline in sorted(polys, key=lambda t: -t[0]):
        dr.polygon(pts, fill=fill, outline=outline)
    fnt = _font(18)
    for n, (cx, cy) in centers:
        dr.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], fill=(20, 20, 20, 230))
        dr.text((cx, cy), str(n), fill=(255, 255, 255, 255), font=fnt, anchor="mm")
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue(), labels


def render_view(room: Room, objects: list[SceneObject], view: Literal["plan", "eye", "object"] = "plan",
                violations: list[Violation] | None = None, object_id: str | None = None,
                eye: tuple[float, float, float] | None = None, target: tuple[float, float, float] | None = None,
                space: str | None = None) -> tuple[bytes, dict[int, str]]:
    if view == "plan":
        return render_plan(room, objects, violations, space=space)
    if view == "object":
        if not object_id:
            raise ValueError("view='object' needs object_id")
        o = next(o for o in objects if o.id == object_id)
        _, front = g.local_axes(o.pose.yaw)
        right, _ = g.local_axes(o.pose.yaw)
        dist = max(o.extent().w, o.extent().d, o.extent().h) * 1.6 + 0.8
        e2 = np.array([o.pose.x, o.pose.y]) + (front * 0.85 + right * 0.5) * dist
        eye = (float(e2[0]), float(e2[1]), o.pose.z + o.extent().h + 0.6)
        target = (o.pose.x, o.pose.y, o.pose.z + o.extent().h / 2)
        return render_perspective(room, objects, eye, target, fov_deg=55, highlight=object_id)
    sp = room.space(space) if space else room.space_list()[0]
    poly = np.asarray(sp.polygon, float)
    if eye is None:
        # default: stand in a door of the space looking at its center
        doors = [op for op in room.openings if op.kind == "door" and (room.wall(op.wall).space in (None, sp.id))]
        center = _centroid(poly)
        if doors:
            w = room.wall(doors[0].wall)
            p = g.wall_point(w, doors[0].center) + g.wall_inward_normal(w, room) * 0.3
        else:
            p = poly[0] + (center - poly[0]) * 0.1
        eye = (float(p[0]), float(p[1]), 1.6)
        target = target or (float(center[0]), float(center[1]), 0.9)
    target = target or (eye[0] + 1, eye[1], eye[2])
    return render_perspective(room, objects, eye, target)
