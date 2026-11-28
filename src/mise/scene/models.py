"""Scene file data model (`scene.json`). All units are meters, angles in degrees.

Conventions
-----------
* World frame: z up, floor at z = 0, walls roughly aligned with x and y.
* Object canonical frame: centered on x/y, z up, footprint on z = 0,
  width along local x, depth along local y, **front faces local +y**.
* `pose.yaw` rotates the canonical frame counter-clockwise about +z.
* `support` is ``"floor"``, ``"ceiling"``, ``"wall:<wall id>"`` or another object's id.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAIN = "main"  # id of the user's own layout

# Categories that only work one way up. Everything else (shelves, cube storage, boxes,
# unknown categories) may be tipped onto a side or end. Keep this list short: it only
# holds things that are unusable or unsafe any other way, and every object can override
# it with `upright_only`.
UPRIGHT_CATEGORIES = frozenset({
    "sofa", "armchair", "chair", "stool", "bench", "bed", "crib", "desk", "table", "dining_table",
    "coffee_table", "side_table", "nightstand", "dresser", "wardrobe", "lamp", "floor_lamp",
    "plant", "rug", "mat", "tv", "tv_stand", "toilet", "mirror", "refrigerator",
})


def upright_required(category: str, upright_only: bool | None) -> bool:
    if upright_only is not None:
        return upright_only
    return category.lower() in UPRIGHT_CATEGORIES


def tipped_extent(dims: "Dims", roll: float, pitch: float) -> "Dims":
    """Axis-aligned size (in the object's yaw frame) after roll/pitch in 90° steps."""
    w, d, h = dims.w, dims.d, dims.h
    if round(roll / 90) % 2:  # about local y: width and height trade places
        w, h = h, w
    if round(pitch / 90) % 2:  # about local x: depth and height trade places
        d, h = h, d
    return Dims(w=w, d=d, h=h)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pose(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # degrees, CCW about +z
    # Tipping, in 90° steps, applied before yaw. roll turns the object about its own
    # front axis (local y): roll 90 stands a wide, low shelf on its end. pitch turns it
    # about its width axis (local x). Both are 0 for an object standing as listed.
    roll: float = 0.0
    pitch: float = 0.0

    @field_validator("roll", "pitch")
    @classmethod
    def _right_angles(cls, v: float) -> float:
        q = round(v / 90.0)
        if abs(v - 90.0 * q) > 1e-6:
            raise ValueError("roll and pitch must be multiples of 90°")
        return float((q % 4) * 90)

    def yaw_rad(self) -> float:
        return math.radians(self.yaw)

    @property
    def tipped(self) -> bool:
        return bool(self.roll or self.pitch)


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
    leaf: bool = Field(False, description="a closed door stands in it (an entry or closet door), not an open doorway")
    color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$", description="the door's or glass's look, #rrggbb")


class Wall(BaseModel):
    id: str
    a: tuple[float, float]
    b: tuple[float, float]
    height: float | None = None
    space: str | None = Field(None, description="the space this wall bounds (None for a single-room scene)")

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)


SPACE_KINDS = (
    "living", "kitchen", "dining", "bedroom", "bathroom", "office", "hallway", "entry", "closet",
    "laundry", "storage", "balcony", "other",
)


def polygon_area(pts) -> float:
    """Signed area (positive when counter-clockwise)."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2


HEX_COLOR = r"^#[0-9a-fA-F]{6}$"


MATERIALS = ("paint", "wood", "tile", "carpet", "block", "popcorn", "stone", "other")


class Surface(BaseModel):
    """How a floor, wall or ceiling looks: a color, optionally a repeating texture image, and what
    it is made of (so it can be changed: repaint the walls, lay a different floor)."""

    color: str | None = Field(None, pattern=HEX_COLOR, description="average color, #rrggbb")
    texture: str | None = Field(None, description="image under the scene's folder, e.g. finishes/floor.jpg")
    size_m: float = Field(1.0, gt=0, description="the texture image covers size_m x size_m of surface")
    kind: Literal["paint", "wood", "tile", "carpet", "block", "popcorn", "stone", "other"] | None = Field(
        None, description="material: paint (flat color), wood (planks), tile, carpet, block (painted masonry), ...")
    direction_deg: float | None = Field(
        None, description="planks/rows run this way: degrees CCW from +x on floors, from horizontal on walls")
    module_m: float | None = Field(None, gt=0, description="plank width, tile size or block course height")

    @field_validator("texture")
    @classmethod
    def _rel(cls, v: str | None) -> str | None:
        if v is not None and (v.startswith("/") or ".." in Path(v).parts):
            raise ValueError("texture must be a path inside the scene folder")
        return v


class Finish(BaseModel):
    floor: Surface | None = None
    walls: Surface | None = None
    ceiling: Surface | None = None
    wall_overrides: dict[str, Surface] = Field(
        default_factory=dict, description="walls (by wall id) that differ from `walls`: an accent wall, block")
    trim: Surface | None = Field(None, description="baseboards and door and window casings")
    baseboard_m: float | None = Field(None, gt=0, description="baseboard height")


FINISH_PARTS = ("floor", "walls", "ceiling", "trim")


def finish_of(sp: "Space", part: str) -> Surface | None:
    """The surface a space has for a part: floor, walls, ceiling, trim, or wall:<id> (one wall)."""
    f = sp.finish
    if f is None:
        return None
    if part.startswith("wall:"):
        return f.wall_overrides.get(part[5:]) or f.walls
    if part not in FINISH_PARTS:
        raise ValueError(f"part is one of {', '.join(FINISH_PARTS)} or wall:<wall id>, not {part!r}")
    return getattr(f, part)


def apply_finish(sp: "Space", part: str, surface: Surface | None) -> None:
    """Set a part's surface. Repainting `walls` repaints every wall of the space, painted block
    and accent walls included (each keeps its own material and pattern); `wall:<id>` sets one."""
    if sp.finish is None:
        sp.finish = Finish()
    f = sp.finish
    if part.startswith("wall:"):
        if surface is None:
            f.wall_overrides.pop(part[5:], None)
        else:
            f.wall_overrides[part[5:]] = surface
        return
    if part not in FINISH_PARTS:
        raise ValueError(f"part is one of {', '.join(FINISH_PARTS)} or wall:<wall id>, not {part!r}")
    if part == "walls" and surface is not None and surface.color:
        for wid, w in list(f.wall_overrides.items()):  # painted block, an accent wall: the new paint too
            f.wall_overrides[wid] = w.model_copy(update={"color": surface.color})
    setattr(f, part, surface)


class Space(BaseModel):
    """A named part of the home: a room behind walls, or a zone of an open-plan area.

    Every edge of ``polygon`` is a wall unless it runs along a space listed in
    ``open_to`` (in either space's list): living room and kitchen sharing one open
    floor are two spaces open to each other, with no wall between them.
    """

    id: str
    name: str
    kind: str = "other"
    polygon: list[tuple[float, float]] = Field(min_length=3)
    ceiling_height: float | None = None
    open_to: list[str] = Field(default_factory=list)
    finish: Finish | None = Field(None, description="floor, wall and ceiling look (e.g. sampled from the capture)")

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        v = v.strip().lower().replace(" ", "_")
        return v if v in SPACE_KINDS else "other"

    @model_validator(mode="after")
    def _ccw(self) -> "Space":
        if polygon_area(self.polygon) < 0:
            self.polygon = list(reversed(self.polygon))
        return self

    def area(self) -> float:
        return abs(polygon_area(self.polygon))


def _collinear_overlaps(a, b, others, tol: float = 0.03) -> list[tuple[float, float]]:
    """Intervals (as fractions of a->b) where segment a-b runs along any segment in `others`."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return []
    ux, uy = dx / L, dy / L
    out = []
    for c, d in others:
        # both ends of c-d within tol of the line through a-b, and roughly parallel
        dc = abs((c[0] - ax) * uy - (c[1] - ay) * ux)
        dd = abs((d[0] - ax) * uy - (d[1] - ay) * ux)
        if dc > tol or dd > tol:
            continue
        t1 = ((c[0] - ax) * ux + (c[1] - ay) * uy) / L
        t2 = ((d[0] - ax) * ux + (d[1] - ay) * uy) / L
        lo, hi = max(0.0, min(t1, t2)), min(1.0, max(t1, t2))
        if hi - lo > 1e-6:
            out.append((lo, hi))
    return out


def _subtract(intervals: list[tuple[float, float]], min_len: float) -> list[tuple[float, float]]:
    """[0, 1] minus the union of intervals, keeping pieces longer than min_len."""
    cur, out = 0.0, []
    for lo, hi in sorted(intervals):
        if lo > cur:
            out.append((cur, lo))
        cur = max(cur, hi)
    if cur < 1.0:
        out.append((cur, 1.0))
    return [(lo, hi) for lo, hi in out if hi - lo > min_len]


class Room(BaseModel):
    """The scanned floor. A single room uses ``floor_polygon``; a home with several
    rooms lists them as ``spaces`` (``floor_polygon`` may then be left out)."""

    splat: str | None = Field(None, description="empty-room splat path, relative to the scene dir")
    floor_polygon: list[tuple[float, float]] | None = None
    ceiling_height: float = 2.5
    spaces: list[Space] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    planes: list[dict] = Field(default_factory=list, description="raw fitted planes from Stage 2 (n, d, label)")

    @model_validator(mode="after")
    def _derive_walls(self) -> "Room":
        if not self.spaces:
            if not self.floor_polygon or len(self.floor_polygon) < 3:
                raise ValueError("a room needs floor_polygon or at least one space")
        else:
            ids = [s.id for s in self.spaces]
            if len(set(ids)) != len(ids):
                raise ValueError("space ids must be unique")
        if not self.walls:
            self.walls = self.derive_walls()
        return self

    # ---- spaces ----------------------------------------------------------
    def space_list(self) -> list[Space]:
        """The spaces, or one implicit space covering a single-room floor."""
        if self.spaces:
            return self.spaces
        return [Space(id="room", name="Room", polygon=self.floor_polygon)]

    def own_spaces(self) -> list[Space]:
        """The spaces, made explicit (a single-room floor's implicit space becomes a real one), so a
        change to one (its finish) is kept."""
        if not self.spaces:
            self.spaces = self.space_list()
        return self.spaces

    def polygons(self) -> list[list[tuple[float, float]]]:
        return [s.polygon for s in self.space_list()]

    def space(self, space_id: str) -> Space:
        for s in self.space_list():
            if s.id == space_id:
                return s
        raise KeyError(f"no space {space_id!r}")

    def space_at(self, x: float, y: float) -> Space | None:
        for s in self.space_list():
            if _point_in(x, y, s.polygon):
                return s
        return None

    def ceiling_for(self, space: Space | None) -> float:
        return (space.ceiling_height if space and space.ceiling_height else None) or self.ceiling_height

    def is_open(self, a: str, b: str) -> bool:
        A = next((s for s in self.space_list() if s.id == a), None)
        B = next((s for s in self.space_list() if s.id == b), None)
        return bool(A and B and (b in A.open_to or a in B.open_to))

    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        xs = [p[0] for poly in self.polygons() for p in poly]
        ys = [p[1] for poly in self.polygons() for p in poly]
        return (min(xs), min(ys)), (max(xs), max(ys))

    def derive_walls(self) -> list[Wall]:
        if not self.spaces:
            pts = self.floor_polygon
            return [Wall(id=str(i + 1), a=pts[i], b=pts[(i + 1) % len(pts)], height=self.ceiling_height)
                    for i in range(len(pts))]
        walls = []
        for s in self.spaces:
            open_edges = [(o.polygon[j], o.polygon[(j + 1) % len(o.polygon)])
                          for o in self.spaces if o.id != s.id and self.is_open(s.id, o.id)
                          for j in range(len(o.polygon))]
            k = 0
            n = len(s.polygon)
            for i in range(n):
                a, b = s.polygon[i], s.polygon[(i + 1) % n]
                L = math.dist(a, b)
                for lo, hi in _subtract(_collinear_overlaps(a, b, open_edges), 0.05 / max(L, 1e-9)):
                    k += 1
                    pa = (a[0] + (b[0] - a[0]) * lo, a[1] + (b[1] - a[1]) * lo)
                    pb = (a[0] + (b[0] - a[0]) * hi, a[1] + (b[1] - a[1]) * hi)
                    walls.append(Wall(id=f"{s.id}.{k}", a=pa, b=pb, height=self.ceiling_for(s), space=s.id))
        return walls

    def wall(self, wall_id: str) -> Wall:
        wid = wall_id.removeprefix("wall:")
        for w in self.walls:
            if w.id == wid:
                return w
        raise KeyError(f"no wall {wall_id!r}")

    def floor_area(self) -> float:
        return sum(s.area() for s in self.space_list())

    def is_ccw(self) -> bool:
        return polygon_area(self.space_list()[0].polygon) > 0


def _point_in(x: float, y: float, poly) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
            inside = not inside
    return inside


def structural_support(support: str) -> bool:
    """Resting on the building itself (floor, ceiling or a wall), not on another object."""
    return support in ("floor", "ceiling") or support.startswith("wall:")


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
    upright_only: bool | None = Field(
        None, description="True: may only stand as modeled. False: may be tipped. None: category default")
    fixture: bool = Field(
        False, description="built in (cabinets, counters, appliances, plumbing, light fixtures): part of the home, "
                           "not furniture. Kept in layout checks; hidden on the blank-slate view; never on a shopping list")

    @property
    def upright_required(self) -> bool:
        return upright_required(self.category, self.upright_only)

    def extent(self) -> Dims:
        """Size as placed (after tipping): width along the yaw frame's x, depth along y, height up."""
        if not self.pose.tipped:
            return self.dims
        return tipped_extent(self.dims, self.pose.roll, self.pose.pitch)


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
    upright_only: bool | None = None
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
    finishes: dict[str, dict[str, Surface]] = Field(
        default_factory=dict, description="finish changes: space id -> part (floor, walls, ceiling, trim, "
        "wall:<wall id>) -> the surface it has in this layout")
    room_splat: str | None = Field(None, description="this layout's empty-home splat (its finishes), if rebuilt")


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

    def resolve_room(self, variant_id: str = MAIN) -> Room:
        """The room as a layout has it: the main room with the variant's finish changes."""
        if variant_id == MAIN:
            return self.room
        v = self.variant(variant_id)
        if not v.finishes:
            return self.room
        room = self.room.model_copy(deep=True)
        for sp_id, parts in v.finishes.items():
            sp = next((x for x in room.own_spaces() if x.id == sp_id), None)
            if sp is None:
                continue
            for part in sorted(parts, key=lambda k: k.startswith("wall:")):  # whole walls first, then single walls
                apply_finish(sp, part, parts[part])
        return room

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
