"""Scene file data model (`scene.json`). All units are meters, angles in degrees.

Conventions
-----------
* World frame: z up, floor at z = 0, walls roughly aligned with x and y.
* Object canonical frame: centered on x/y, z up, footprint on z = 0,
  width along local x, depth along local y, **front faces local +y**.
* `pose.yaw` rotates the canonical frame counter-clockwise about +z.
* `support` is ``"floor"``, ``"wall:<wall id>"`` or another object's id.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAIN = "main"  # id of the user's own layout


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # degrees, CCW about +z

    def yaw_rad(self) -> float:
        return math.radians(self.yaw)


class Dims(BaseModel):
    """Oriented bounding box extents in the object's canonical frame."""

    w: float = Field(gt=0, description="width, local x")
    d: float = Field(gt=0, description="depth, local y")
    h: float = Field(gt=0, description="height, local z")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.w, self.d, self.h)


class Product(BaseModel):
    retailer: str | None = None
    url: str
    title: str | None = None
    price: float | None = None
    currency: str | None = None
    listed_dims: Dims | None = None
    raw_dims: str | None = Field(None, description="dimension string exactly as listed, for audit")
    image_url: str | None = None
    fetched_at: str | None = None
    in_stock: bool | None = None


class Opening(BaseModel):
    """A door or window on a wall, described along the wall.

    ``center`` is the distance from the wall's start point to the opening's
    center. Doors swing ``in`` (into the room) or ``out``; ``hinge`` is seen
    from inside the room looking at the wall.
    """

    id: str
    kind: Literal["door", "window"]
    wall: str
    center: float
    width: float
    bottom: float = 0.0
    top: float = 2.0
    swing: Literal["in", "out", "none", "sliding"] = "in"
    hinge: Literal["left", "right"] = "left"


class Wall(BaseModel):
    id: str
    a: tuple[float, float]
    b: tuple[float, float]
    height: float | None = None

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)


class Room(BaseModel):
    splat: str | None = Field(None, description="empty-room splat path, relative to the scene dir")
    floor_polygon: list[tuple[float, float]] = Field(min_length=3)
    ceiling_height: float = 2.5
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    planes: list[dict] = Field(default_factory=list, description="raw fitted planes from Stage 2 (n, d, label)")

    @model_validator(mode="after")
    def _derive_walls(self) -> "Room":
        if not self.walls:
            pts = self.floor_polygon
            self.walls = [
                Wall(id=str(i + 1), a=pts[i], b=pts[(i + 1) % len(pts)], height=self.ceiling_height)
                for i in range(len(pts))
            ]
        return self

    def wall(self, wall_id: str) -> Wall:
        wid = wall_id.removeprefix("wall:")
        for w in self.walls:
            if w.id == wid:
                return w
        raise KeyError(f"no wall {wall_id!r}")

    def floor_area(self) -> float:
        pts = self.floor_polygon
        s = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2

    def is_ccw(self) -> bool:
        pts = self.floor_polygon
        s = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
        return s > 0


class SceneObject(BaseModel):
    id: str
    name: str
    category: str = "furniture"
    asset: str | None = Field(None, description="splat/mesh path in canonical frame, relative to the scene dir")
    asset_kind: Literal["splat", "mesh", "box"] = "box"
    asset_transform: list[list[float]] | None = Field(
        None, description="row-major 4x4 taking the asset file's frame to the canonical frame (meshes); None = file is canonical")
    pose: Pose = Field(default_factory=Pose)
    dims: Dims
    support: str = "floor"
    source: Literal["scanned", "product", "upload"] = "scanned"
    product: Product | None = None
    locked: bool = False
    anisotropy: float | None = None
    warnings: list[str] = Field(default_factory=list)
    color: str | None = Field(None, description="display color for box assets")
    moved_from_scan: bool = False


class LibraryAsset(BaseModel):
    """An imported asset that is not placed yet (Stage 4 output)."""

    id: str
    name: str
    category: str = "furniture"
    asset: str | None = None
    asset_kind: Literal["splat", "mesh", "box"] = "box"
    asset_transform: list[list[float]] | None = None
    dims: Dims
    source: Literal["product", "upload", "scanned"] = "product"
    status: Literal["ready", "warning", "fallback_box"] = "ready"
    product: Product | None = None
    anisotropy: float | None = None
    warnings: list[str] = Field(default_factory=list)


class Variant(BaseModel):
    """A named layout expressed as changes on top of the main layout."""

    id: str
    name: str
    created_by: Literal["user", "agent"] = "user"
    created_at: str = Field(default_factory=now_iso)
    notes: str = ""
    poses: dict[str, Pose] = Field(default_factory=dict)
    supports: dict[str, str] = Field(default_factory=dict)
    added: list[SceneObject] = Field(default_factory=list)
    hidden: list[str] = Field(default_factory=list)


class LayoutRules(BaseModel):
    walkway_min: float = 0.9
    access_min: float = 0.6
    door_clearance: float = 0.9
    window_zone_depth: float = 0.6
    tall_object_height: float = 1.2
    collision_tolerance: float = 0.01
    snap_yaw_step: float = 15.0
    snap_wall_distance: float = 0.15
    grid: float = 0.05


class Scene(BaseModel):
    version: int = 1
    id: str
    name: str
    units: Literal["m"] = "m"
    room: Room
    objects: list[SceneObject] = Field(default_factory=list)
    library: list[LibraryAsset] = Field(default_factory=list)
    variants: list[Variant] = Field(default_factory=list)
    rules: LayoutRules = Field(default_factory=LayoutRules)
    updated_at: str = Field(default_factory=now_iso)

    # ---- lookups -------------------------------------------------------
    def variant(self, variant_id: str) -> Variant:
        for v in self.variants:
            if v.id == variant_id:
                return v
        raise KeyError(f"no variant {variant_id!r}")

    def library_asset(self, asset_id: str) -> LibraryAsset:
        for a in self.library:
            if a.id == asset_id:
                return a
        raise KeyError(f"no library asset {asset_id!r}")

    def resolve(self, variant_id: str = MAIN, include_hidden: bool = False) -> list[SceneObject]:
        """Effective object list for a variant (main = the base objects)."""
        if variant_id == MAIN:
            return [o.model_copy(deep=True) for o in self.objects]
        v = self.variant(variant_id)
        out: list[SceneObject] = []
        for o in [*self.objects, *v.added]:
            if o.id in v.hidden and not include_hidden:
                continue
            o = o.model_copy(deep=True)
            if o.id in v.poses:
                o.pose = v.poses[o.id].model_copy()
            if o.id in v.supports:
                o.support = v.supports[o.id]
            out.append(o)
        return out

    def find(self, variant_id: str, object_id: str) -> SceneObject:
        for o in self.resolve(variant_id, include_hidden=True):
            if o.id == object_id:
                return o
        raise KeyError(f"no object {object_id!r} in variant {variant_id!r}")

    def all_ids(self) -> set[str]:
        ids = {o.id for o in self.objects}
        for v in self.variants:
            ids |= {o.id for o in v.added}
        return ids
