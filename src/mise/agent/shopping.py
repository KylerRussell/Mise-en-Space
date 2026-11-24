"""Shopping list for a variant, and the H6 re-check of every link."""

from __future__ import annotations

import csv
import io
from typing import Callable

from pydantic import BaseModel

from ..scene.models import Scene, now_iso


class ShoppingRow(BaseModel):
    object_id: str
    item: str
    retailer: str | None
    price: float | None
    currency: str | None
    listed_w_d_h_m: str | None
    link: str
    date_checked: str | None
    status: str | None = None


class ShoppingList(BaseModel):
    variant: str
    rows: list[ShoppingRow]
    total: float
    currency: str | None
    mixed_currency: bool = False

    def to_markdown(self) -> str:
        lines = ["| Item | Retailer | Price | Listed W x D x H | Link | Checked |", "| --- | --- | --- | --- | --- | --- |"]
        for r in self.rows:
            price = f"{r.price:,.2f} {r.currency or ''}".strip() if r.price is not None else "?"
            lines.append(f"| {r.item} | {r.retailer or ''} | {price} | {r.listed_w_d_h_m or '?'} | [link]({r.link}) | {r.date_checked or ''} |")
        tot = f"{self.total:,.2f} {self.currency or ''}".strip()
        lines.append(f"| **Total** | | **{tot}**{' (mixed currencies)' if self.mixed_currency else ''} | | | |")
        return "\n".join(lines)

    def to_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["item", "retailer", "price", "currency", "listed_w_d_h_m", "link", "date_checked"])
        for r in self.rows:
            w.writerow([r.item, r.retailer, r.price, r.currency, r.listed_w_d_h_m, r.link, r.date_checked])
        w.writerow(["TOTAL", "", self.total, self.currency, "", "", ""])
        return buf.getvalue()


def shopping_list(scene: Scene, variant: str) -> ShoppingList:
    """Every product-sourced object in the variant that is not in the main layout."""
    main_ids = {o.id for o in scene.objects}
    rows, total, currencies = [], 0.0, set()
    for o in scene.resolve(variant):
        if o.source != "product" or not o.product or o.id in main_ids:
            continue
        p = o.product
        dims = f"{p.listed_dims.w:.2f} x {p.listed_dims.d:.2f} x {p.listed_dims.h:.2f}" if p.listed_dims else None
        rows.append(ShoppingRow(object_id=o.id, item=p.title or o.name, retailer=p.retailer, price=p.price,
                                currency=p.currency, listed_w_d_h_m=dims, link=p.url, date_checked=p.fetched_at))
        if p.price is not None:
            total += p.price
            if p.currency:
                currencies.add(p.currency)
    return ShoppingList(variant=variant, rows=rows, total=round(total, 2),
                        currency=next(iter(currencies)) if len(currencies) == 1 else None,
                        mixed_currency=len(currencies) > 1)


def recheck(sl: ShoppingList, fetch: Callable[[str], object], dim_tolerance: float = 0.02) -> dict:
    """H6: re-fetch each link; count live links whose dimensions still match.

    `fetch(url)` returns a Listing (see products.listing.fetch_listing).
    """
    from ..products.dimensions import DimensionParseError, parse_dimensions

    live = matched = 0
    for r in sl.rows:
        try:
            lst = fetch(r.link)
        except Exception as e:
            r.status = f"dead: {type(e).__name__}"
            continue
        live += 1
        r.date_checked = now_iso()
        ok = False
        if r.listed_w_d_h_m and getattr(lst, "raw_dims", None):
            try:
                pd = parse_dimensions(lst.raw_dims)
                old = [float(x) for x in r.listed_w_d_h_m.split(" x ")]
                ok = all(abs(a - b) / b <= dim_tolerance for a, b in zip(pd.as_tuple(), old))
            except (DimensionParseError, ValueError):
                ok = False
        matched += ok
        r.status = "ok" if ok else "dimensions changed or missing"
        if getattr(lst, "price", None) is not None:
            r.price = lst.price
    n = len(sl.rows) or 1
    return {"n": len(sl.rows), "live": live, "live_and_matching": matched, "rate": matched / n,
            "pass": matched / n >= 0.9, "switch": matched / n < 0.8}
