"""Parse retailer dimension strings into width/depth/height in meters.

Retailers order and label W x D x H differently, so labels win when present.
Unlabeled triples fall back to W x D x H and are flagged as assumed. Every
parse is checked against a plausible range for the category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

UNIT_M = {
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001, "millimetre": 0.001, "millimetres": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01, "centimetre": 0.01, "centimetres": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254, '"': 0.0254, "”": 0.0254, "''": 0.0254,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "'": 0.3048, "’": 0.3048,
}

LABELS = {
    "w": "w", "width": "w", "wide": "w", "l": "w", "length": "w", "long": "w",
    "d": "d", "depth": "d", "deep": "d",
    "h": "h", "height": "h", "high": "h", "tall": "h",
}

# (min, max) meters for (w, d, h) by category; None = unchecked.
PLAUSIBLE: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    "sofa": ((1.0, 4.0), (0.6, 2.2), (0.5, 1.2)),
    "armchair": ((0.5, 1.3), (0.5, 1.2), (0.6, 1.3)),
    "chair": ((0.35, 0.9), (0.35, 0.9), (0.6, 1.3)),
    "bed": ((0.8, 2.4), (1.8, 2.5), (0.2, 1.6)),
    "nightstand": ((0.3, 0.8), (0.25, 0.6), (0.35, 0.8)),
    "desk": ((0.6, 2.2), (0.4, 1.0), (0.6, 1.3)),
    "table": ((0.4, 3.0), (0.4, 1.5), (0.3, 1.1)),
    "dining_table": ((0.7, 3.0), (0.7, 1.4), (0.65, 0.85)),
    "coffee_table": ((0.4, 1.6), (0.4, 1.2), (0.25, 0.6)),
    "wardrobe": ((0.4, 3.0), (0.35, 0.75), (1.4, 2.6)),
    "dresser": ((0.5, 2.0), (0.3, 0.65), (0.6, 1.5)),
    "bookshelf": ((0.3, 2.5), (0.2, 0.5), (0.6, 2.6)),
    "lamp": ((0.1, 0.8), (0.1, 0.8), (0.2, 2.1)),
    "floor_lamp": ((0.15, 0.8), (0.15, 0.8), (1.0, 2.2)),
    "rug": ((0.5, 5.0), (0.5, 4.0), (0.002, 0.05)),
    "plant": ((0.1, 1.5), (0.1, 1.5), (0.1, 2.5)),
}


@dataclass
class ParsedDims:
    w: float
    d: float
    h: float
    raw: str
    assumed_order: bool = False
    notes: list[str] = field(default_factory=list)
    plausible: bool = True

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.w, self.d, self.h)


class DimensionParseError(ValueError):
    pass


_NUM = r"(\d+(?:[.,]\d+)?(?:\s*(?:-|\s)\s*\d+/\d+)?|\d+/\d+)"
_UNIT = r"(mm|cm|m(?![a-z])|in(?:ch(?:es)?)?\.?|\"|”|''|ft|feet|foot|'|’|millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?)"
_LABEL = r"(?:width|depth|height|length|wide|deep|high|tall|long|[wdhl])"


def _num(s: str) -> float:
    s = s.strip().replace(",", ".")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:-|\s)\s*(\d+)/(\d+)", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return float(s)


def _unit_factor(u: str | None) -> float | None:
    if not u:
        return None
    u = u.strip().lower().rstrip(".")
    return UNIT_M.get(u)


def _feet_inches(text: str) -> str:
    """Rewrite 7' 2" / 7 ft 2 in into inches so the main parser sees one unit."""
    def repl(m: re.Match) -> str:
        ft, inch = float(m.group(1)), float(m.group(2) or 0)
        return f"{ft * 12 + inch:g} in"
    return re.sub(r"(\d+(?:\.\d+)?)\s*(?:'|’|ft\.?|feet)\s*(\d+(?:\.\d+)?)\s*(?:\"|”|in\.?|inches)", repl, text)


def parse_dimensions(text: str, category: str | None = None) -> ParsedDims:
    raw = text
    t = _feet_inches(text.replace("×", "x").replace("X", "x").replace("*", "x"))
    t = re.sub(r"\s+", " ", t)
    notes: list[str] = []

    # Collect (value, unit, label) tokens. Labels may come before or after the number.
    tok = re.compile(
        rf"(?:\b(?P<pre>{_LABEL})\s*[:=]?\s*)?(?P<num>{_NUM[1:-1]})\s*(?P<unit>{_UNIT[1:-1]})?\.?\s*(?:(?P<post>{_LABEL})\b\.?)?",
        re.IGNORECASE,
    )
    matches = list(tok.finditer(t))
    raw_labels = {(m.group("pre") or m.group("post") or "").lower() for m in matches}
    # "L x W x H": length is the long side and W then means depth.
    length_and_width = bool(raw_labels & {"l", "length", "long"}) and bool(raw_labels & {"w", "width", "wide"})
    items: list[tuple[float, str | None, str | None]] = []
    for m in matches:
        label = m.group("pre") or m.group("post")
        if label is not None:
            key = label.lower()
            label = "d" if (length_and_width and key in ("w", "width", "wide")) else LABELS.get(key)
        items.append((_num(m.group("num")), m.group("unit"), label))
    items = [it for it in items if it[0] > 0]
    if len(items) < 2:
        raise DimensionParseError(f"could not find dimensions in {raw!r}")

    # Units: carry the last stated unit backwards ("84 x 38 x 34 in").
    units = [it[1] for it in items]
    default_unit = next((u for u in reversed(units) if u), None)
    if default_unit is None:
        # Guess from magnitude: furniture in mm is >= 100 typically; cm 10..400; in 5..150
        mx = max(v for v, _, _ in items)
        default_unit = "mm" if mx > 400 else "cm"
        notes.append(f"no unit stated; assumed {default_unit}")
    vals: list[tuple[float, str | None]] = []
    for v, u, lbl in items:
        f = _unit_factor(u or default_unit)
        if f is None:
            f = _unit_factor(default_unit) or 0.01
        vals.append((v * f, lbl))

    labeled = {lbl: v for v, lbl in vals if lbl}
    assumed = False
    if {"w", "d", "h"} <= labeled.keys():
        w, d, h = labeled["w"], labeled["d"], labeled["h"]
    else:
        unl = [v for v, lbl in vals if not lbl]
        slots = [k for k in ("w", "d", "h") if k not in labeled]
        if len(unl) < len(slots):
            if len(vals) == 2 and not labeled:  # rugs etc.: W x L
                w, d = sorted((vals[0][0], vals[1][0]), reverse=True)
                h = None
                notes.append("only two dimensions; height unknown")
            else:
                raise DimensionParseError(f"missing {', '.join(slots)} in {raw!r}")
        else:
            fill = dict(zip(slots, unl))
            merged = {**fill, **labeled}
            w, d, h = merged["w"], merged["d"], merged.get("h")
            if slots:
                assumed = True
                notes.append(f"unlabeled values assumed to be {' x '.join(s.upper() for s in slots)}")
    if h is None:
        defaults = {"rug": 0.01}
        if category in defaults:
            h = defaults[category]
        else:
            raise DimensionParseError(f"no height in {raw!r}")

    out = ParsedDims(w=round(w, 4), d=round(d, 4), h=round(h, 4), raw=raw, assumed_order=assumed, notes=notes)
    check_plausible(out, category)
    return out


def check_plausible(p: ParsedDims, category: str | None) -> None:
    rng = PLAUSIBLE.get((category or "").lower())
    if not rng:
        return
    for name, v, (lo, hi) in zip(("width", "depth", "height"), p.as_tuple(), rng):
        if not lo <= v <= hi:
            p.plausible = False
            p.notes.append(f"{name} {v:.2f} m is outside the plausible {lo:.2f}-{hi:.2f} m for a {category}")
    # Swapped W/D is common: if swapping fixes it, suggest that.
    if not p.plausible:
        (wl, wh), (dl, dh), _ = rng
        if wl <= p.d <= wh and dl <= p.w <= dh:
            p.notes.append("width and depth look swapped")
