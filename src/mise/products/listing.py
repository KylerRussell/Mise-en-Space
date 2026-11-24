"""Stage 4 step 1: pull name, price, raw dimension string and images from a product page.

Order of evidence: schema.org Product JSON-LD, then OpenGraph/meta tags,
then dimension-looking text in the page body. The raw HTML is snapshotted so
every number can be audited later.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..scene.models import now_iso

USER_AGENT = "mise-en-space/0.1 (personal room planner)"

DIM_HINT = re.compile(
    r"(?:(?:overall\s+)?(?:dimensions?|size|measurements?|assembled)\s*[:\-]?\s*)"
    r"([^\n<]{0,30}?\d[^\n<]{4,90})",
    re.IGNORECASE,
)
TRIPLE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:\"|”|in\.?|cm|mm|m)?\s*[a-zA-Z]{0,6}\.?\s*[x×*]\s*"
    r"\d+(?:[.,]\d+)?\s*(?:\"|”|in\.?|cm|mm|m)?\s*[a-zA-Z]{0,6}\.?\s*[x×*]\s*"
    r"\d+(?:[.,]\d+)?\s*(?:\"|”|in\.?|cm|mm|m)?(?:\s*[a-zA-Z]{1,6}\.?)?",
)


@dataclass
class Listing:
    url: str
    retailer: str
    title: str | None = None
    price: float | None = None
    currency: str | None = None
    raw_dims: str | None = None
    dim_candidates: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    in_stock: bool | None = None
    fetched_at: str = field(default_factory=now_iso)
    snapshot: str | None = None


def retailer_of(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _product_nodes(data) -> list[dict]:
    out = []
    for node in _as_list(data):
        if not isinstance(node, dict):
            continue
        if "@graph" in node:
            out += _product_nodes(node["@graph"])
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if "Product" in types or "ProductGroup" in types:
            out.append(node)
    return out


_QV_UNITS = {"CMT": "cm", "MMT": "mm", "MTR": "m", "INH": "in", "FOT": "ft", "cm": "cm", "mm": "mm", "m": "m", "in": "in", "inch": "in"}


def _qv(v) -> str | None:
    """schema.org QuantitativeValue -> '84 in'."""
    if v is None:
        return None
    if isinstance(v, (int, float, str)):
        return str(v)
    if isinstance(v, dict):
        val = v.get("value")
        unit = _QV_UNITS.get(str(v.get("unitCode") or v.get("unitText") or ""), v.get("unitText") or "")
        return f"{val} {unit}".strip() if val is not None else None
    return None


def parse_listing(html: str, url: str) -> Listing:
    soup = BeautifulSoup(html, "html.parser")
    lst = Listing(url=url, retailer=retailer_of(url))

    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or tag.get_text() or "null")
        except json.JSONDecodeError:
            continue
        for p in _product_nodes(data):
            lst.title = lst.title or p.get("name")
            imgs = [i if isinstance(i, str) else (i or {}).get("url") for i in _as_list(p.get("image"))]
            lst.images += [urljoin(url, i) for i in imgs if i]
            for off in _as_list(p.get("offers")):
                if not isinstance(off, dict):
                    continue
                price = off.get("price") or off.get("lowPrice")
                if price is not None and lst.price is None:
                    try:
                        lst.price = float(str(price).replace(",", ""))
                    except ValueError:
                        pass
                    lst.currency = off.get("priceCurrency")
                avail = str(off.get("availability", ""))
                if avail:
                    lst.in_stock = "InStock" in avail or "LimitedAvailability" in avail
            w, d, h = (_qv(p.get(k)) for k in ("width", "depth", "height"))
            if w and d and h:
                lst.raw_dims = f"W {w} x D {d} x H {h}"
                lst.dim_candidates.append(lst.raw_dims)
            for prop in _as_list(p.get("additionalProperty")):
                if isinstance(prop, dict) and re.search(r"dimension|size", str(prop.get("name", "")), re.I):
                    val = str(prop.get("value", ""))
                    if re.search(r"\d", val):
                        lst.dim_candidates.append(val)

    def meta(*names):
        for n in names:
            m = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if m and m.get("content"):
                return m["content"]
        return None

    lst.title = lst.title or meta("og:title") or (soup.title.string.strip() if soup.title and soup.title.string else None)
    if lst.price is None:
        p = meta("product:price:amount", "og:price:amount")
        if p:
            try:
                lst.price = float(p.replace(",", ""))
            except ValueError:
                pass
            lst.currency = meta("product:price:currency", "og:price:currency")
    og_img = meta("og:image")
    if og_img:
        lst.images.append(urljoin(url, og_img))

    text = soup.get_text("\n")
    for m in DIM_HINT.finditer(text):
        cand = m.group(1).strip()
        if TRIPLE.search(cand) or re.search(r"(width|depth|height)", cand, re.I):
            lst.dim_candidates.append(cand)
    for m in TRIPLE.finditer(text):
        lst.dim_candidates.append(m.group(0).strip())
    seen = set()
    lst.dim_candidates = [c for c in lst.dim_candidates if not (c in seen or seen.add(c))]
    lst.raw_dims = lst.raw_dims or (lst.dim_candidates[0] if lst.dim_candidates else None)
    seen = set()
    lst.images = [i for i in lst.images if not (i in seen or seen.add(i))][:4]
    return lst


def fetch_listing(url: str, snapshot_dir: Path | None = None, timeout: float = 20.0) -> Listing:
    import httpx

    r = httpx.get(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}, follow_redirects=True, timeout=timeout)
    r.raise_for_status()
    lst = parse_listing(r.text, str(r.url))
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(url.encode()).hexdigest()[:12] + ".html"
        (snapshot_dir / name).write_text(r.text)
        lst.snapshot = str(snapshot_dir / name)
    return lst
