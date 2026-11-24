"""Stage 4 end to end: product URL -> sized, oriented, gated library asset."""

from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from ..pipeline import ply
from ..scene.models import Dims, LibraryAsset, Product
from . import color, dimensions, orient, scale_fit
from .idol import IdolResult
from .listing import Listing, fetch_listing
from .mesh import read_glb_positions


class Generator(Protocol):
    def generate(self, image_url: str, out_dir: Path, image_type: Literal["real", "synthetic"] = "real",
                 mesh: bool = False) -> IdolResult: ...


class NeedsInput(Exception):
    """The import cannot continue without the user (e.g. implausible dimensions)."""

    def __init__(self, question: str, listing: Listing | None = None):
        super().__init__(question)
        self.question = question
        self.listing = listing


@dataclass
class ImportRequest:
    url: str
    category: str = "furniture"
    name: str | None = None
    dims_override: str | None = None  # typed dims, e.g. '84"W x 38"D x 34"H'
    image_urls: list[str] | None = None  # override listing images (or an uploaded photo)
    image_type: Literal["real", "synthetic", "auto"] = "auto"
    mesh: bool = False  # Idol Max (GLB) instead of base Idol (splat)
    camera_dir: tuple[float, float, float] | None = None  # input camera viewing direction, asset frame
    max_attempts: int = 2
    accept: float = scale_fit.ACCEPT
    warn: float = scale_fit.WARN


@dataclass
class ImportResult:
    asset: LibraryAsset
    attempts: list[dict] = field(default_factory=list)
    listing: Listing | None = None


def guess_image_type(image_bytes: bytes) -> Literal["real", "synthetic"]:
    """Heuristic: product renders and cut-outs sit on a flat white/uniform border."""
    try:
        from PIL import Image
    except ImportError:
        return "real"
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im.thumbnail((128, 128))
    a = np.asarray(im, dtype=float)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    uniform = (border.std(axis=0) < 6).all()
    return "synthetic" if uniform and border.mean() > 225 else "real"


def _points_from_result(res: IdolResult) -> tuple[np.ndarray, np.ndarray | None]:
    if res.kind == "mesh":
        return read_glb_positions(res.path), None
    s = ply.read_ply(res.path)
    a = s.alpha()
    keep = a > 0.1
    return s.xyz[keep].astype(np.float64), a[keep]


def run_import(req: ImportRequest, out_dir: Path, generator: Generator | None,
               listing: Listing | None = None, fetch_image=None) -> ImportResult:
    """Run Stage 4. `out_dir` is the scene's asset directory; files go in out_dir/<asset id>/."""
    listing = listing or fetch_listing(req.url, snapshot_dir=out_dir / "_snapshots")
    raw = req.dims_override or listing.raw_dims
    if not raw:
        raise NeedsInput("The listing has no dimensions I can find. What are its width, depth and height?", listing)
    try:
        parsed = dimensions.parse_dimensions(raw, req.category)
    except dimensions.DimensionParseError as e:
        raise NeedsInput(f"I could not read the dimensions {raw!r} ({e}). What are its W x D x H?", listing) from e
    if not parsed.plausible:
        raise NeedsInput(f"The listed dimensions look wrong for a {req.category}: {'; '.join(parsed.notes)}. "
                         "Please confirm W x D x H.", listing)
    listed = parsed.as_tuple()
    asset_id = f"prod_{uuid.uuid4().hex[:8]}"
    adir = out_dir / asset_id
    adir.mkdir(parents=True, exist_ok=True)
    product = Product(retailer=listing.retailer, url=listing.url, title=listing.title, price=listing.price,
                      currency=listing.currency, listed_dims=Dims(w=listed[0], d=listed[1], h=listed[2]),
                      raw_dims=parsed.raw, image_url=None, fetched_at=listing.fetched_at, in_stock=listing.in_stock)
    name = req.name or (listing.title or "Product")[:60]
    warnings = list(parsed.notes)
    attempts: list[dict] = []
    images = req.image_urls or listing.images

    for img in images[: max(1, req.max_attempts)]:
        if generator is None:
            attempts.append({"image": img, "error": "no generator configured (set FAL_KEY)"})
            break
        image_type = req.image_type
        if image_type == "auto":
            try:
                image_type = guess_image_type(fetch_image(img)) if fetch_image else "real"
            except Exception:
                image_type = "real"
        try:
            res = generator.generate(img, adir / f"gen{len(attempts)}", image_type=image_type, mesh=req.mesh)
            pts, w = _points_from_result(res)
            o = orient.orient(pts, w, up_hint=np.array(res.up_axis) if res.up_axis else None,
                              camera_dir=np.array(req.camera_dir) if req.camera_dir else None)
            fit = scale_fit.fit_scale(listed, o.dims, req.accept, req.warn)
        except Exception as e:  # keep going with the next image
            attempts.append({"image": img, "image_type": image_type, "error": str(e)})
            continue
        attempt = {"image": img, "image_type": image_type, "anisotropy": round(fit.anisotropy, 4), "gate": fit.gate,
                   "scale": fit.scale, "notes": o.notes + fit.notes, "orientation_ambiguous": o.ambiguous}
        attempts.append(attempt)
        if fit.gate == "reject":
            continue
        # Build the canonical, metric asset.
        R = o.R
        if fit.swapped_wd:
            R = ply.rot_z(90) @ R
        t_after = None
        if res.kind == "splat":
            s = ply.read_ply(res.path)
            s = ply.transform(s, R=R, scale=fit.scale, drop_sh_rest=True)
            c = s.xyz.astype(np.float64)
            lo, hi = np.percentile(c, 0.2, axis=0), np.percentile(c, 99.8, axis=0)
            shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            s = ply.transform(s, t=shift)
            path = adir / "asset.ply"
            ply.write_ply(path, s)
            kind = "splat"
            dims_m = orient.robust_dims(s.xyz.astype(np.float64), 0.2)
        else:
            p2 = (pts @ R.T) * fit.scale
            lo, hi = p2.min(axis=0), p2.max(axis=0)
            shift = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
            M = np.eye(4)
            M[:3, :3] = R * fit.scale
            M[:3, 3] = shift
            t_after = M.tolist()
            path = res.path.rename(adir / "asset.glb")
            kind = "mesh"
            dims_m = tuple((hi - lo).tolist())
        if fit.gate == "warn":
            warnings.append(f"size mismatch: anisotropy {fit.anisotropy:.2f} (accepted with warning)")
        if o.ambiguous:
            warnings += o.notes
        if res.kind == "splat":
            warnings.append("Idol splats carry base color only; may look flatter than the room")
        product.image_url = img
        asset = LibraryAsset(
            id=asset_id, name=name, category=req.category, asset=str(path.relative_to(out_dir.parent)),
            asset_kind=kind, asset_transform=t_after, dims=Dims(w=dims_m[0], d=dims_m[1], h=dims_m[2]),
            source="product", product=product, anisotropy=round(fit.anisotropy, 4), warnings=warnings,
            status="warning" if fit.gate == "warn" else "ready",
        )
        return ImportResult(asset=asset, attempts=attempts, listing=listing)

    # Fallback: a labeled box at the listed size.
    warnings.append("3D generation failed the size gate or was unavailable; showing a labeled box at the listed size")
    asset = LibraryAsset(id=asset_id, name=name, category=req.category, asset=None, asset_kind="box",
                         dims=Dims(w=listed[0], d=listed[1], h=listed[2]), source="product", product=product,
                         warnings=warnings, status="fallback_box")
    return ImportResult(asset=asset, attempts=attempts, listing=listing)


__all__ = ["ImportRequest", "ImportResult", "NeedsInput", "run_import", "guess_image_type", "color"]
