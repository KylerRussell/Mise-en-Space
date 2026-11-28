"""Which points are the home itself (walls, floor, ceiling, doors, windows) and which are things.

Every pixel of every picked frame gets an ADE20K class (scripts/colab/segment_frames.py,
Mask2Former). A learned solve's points each come from one pixel of one frame, so each point
takes its pixel's class. The layout is then traced from the structure alone: hanging clothes,
beds, shelves and the scene in a mirror never reach it. The other classes are the objects to
take out for the empty room (Stage 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# the home's shell; fixtures (counters, cabinets, bathtub) stand in front of its walls, so they're out
STRUCTURE = {"wall", "floor", "ceiling", "door", "screen door", "column", "rug", "stairs", "stairway",
             "step", "railing", "bannister"}
# surfaces the depth looks through rather than at: their points lie beyond the wall (a mirror's
# reflection, the view out of a window), so they neither carve free space nor place walls
SEE_THROUGH = {"mirror", "windowpane", "glass", "screen", "crt screen", "television receiver", "monitor"}


def load_labels(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    """labels.npz (frame name -> class map) and the class names from its meta.json."""
    path = Path(path)
    z = np.load(path / "labels.npz")
    names = json.loads((path / "meta.json").read_text())["classes"]
    return {k: z[k] for k in z.files}, [n.strip() for n in names]


def fill_see_through(labels: dict[str, np.ndarray], classes: list[str], min_px: int = 150) -> dict[str, np.ndarray]:
    """Each mirror or window in a frame, filled in: what shows in a mirror is labelled as what it
    reflects (walls, doors), so only its frame and patches of glass come out as "mirror". The
    convex hull of each such patch, taken as the mirror, marks all of it see-through."""
    import cv2

    ids = [i for i, n in enumerate(classes) if n in SEE_THROUGH]
    mark = ids[0] if ids else 0
    out = {}
    for name, m in labels.items():
        st = np.isin(m, ids).astype(np.uint8)
        if not st.any():
            out[name] = m
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(st)
        fill = np.zeros_like(st)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_px:
                continue
            pts = np.c_[np.nonzero(lab == k)[::-1]].astype(np.int32)
            cv2.fillConvexPoly(fill, cv2.convexHull(pts), 1)
        m2 = m.copy()
        m2[(fill > 0) & ~np.isin(m, ids)] = mark
        out[name] = m2
    return out


def point_classes(rec, labels: dict[str, np.ndarray]) -> dict[int, int]:
    """Class of each 3D point: the label at the pixel of its first observation (-1 when unknown)."""
    out = {}
    for pid, p in rec.points3D.items():
        el = p.track.elements
        if not len(el):
            out[pid] = -1
            continue
        im = rec.images[el[0].image_id]
        m = labels.get(im.name)
        if m is None:
            out[pid] = -1
            continue
        cam = rec.cameras[im.camera_id]
        u, v = im.points2D[el[0].point2D_idx].xy
        x = int(np.clip(u * m.shape[1] / cam.width, 0, m.shape[1] - 1))
        y = int(np.clip(v * m.shape[0] / cam.height, 0, m.shape[0] - 1))
        out[pid] = int(m[y, x])
    return out


def structure_only(rec, labels: dict[str, np.ndarray], classes: list[str], keep: set[str] | None = STRUCTURE,
                   drop: set[str] = frozenset()):
    """A copy of the solve with only the points on `keep` classes (all classes when None), less those
    on `drop` (unlabelled points stay)."""
    import pycolmap

    ok = {i for i, n in enumerate(classes) if (keep is None or n in keep) and n not in drop}
    cls = point_classes(rec, labels)
    out = pycolmap.Reconstruction(rec)
    drop = [pid for pid, c in cls.items() if c >= 0 and c not in ok]
    for pid in drop:
        out.delete_point3D(pid)
    return out, {"points": len(cls), "dropped": len(drop),
                 "by_class": _count(cls, classes)}


def _count(cls: dict[int, int], classes: list[str], top: int = 12) -> dict[str, int]:
    v, c = np.unique(np.fromiter(cls.values(), int), return_counts=True)
    o = np.argsort(-c)[:top]
    return {(classes[v[i]] if v[i] >= 0 else "unlabelled"): int(c[i]) for i in o}
