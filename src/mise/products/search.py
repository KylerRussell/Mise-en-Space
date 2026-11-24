"""`search_products`: retailer search where an official API exists.

The agent does open-web research with its own search and fetch tools; this
only wraps retailer APIs you configure. Amazon PA-API 5 is deprecated in
favor of the Creators API, so no Amazon adapter ships here yet. Register an
adapter with :func:`register`; each returns dicts with at least
title, url, price, currency and dims (w, d, h in meters).
"""

from __future__ import annotations

from typing import Callable

Adapter = Callable[[str], list[dict]]
_ADAPTERS: dict[str, Adapter] = {}


def register(name: str, fn: Adapter) -> None:
    _ADAPTERS[name] = fn


def search(query: str, max_w: float | None = None, max_d: float | None = None, max_h: float | None = None,
           max_price: float | None = None) -> dict:
    if not _ADAPTERS:
        return {"products": [], "note": "no retailer APIs configured; use web search, then import_product(url)"}
    out, errors, dropped_no_dims = [], {}, 0
    for name, fn in _ADAPTERS.items():
        try:
            items = fn(query)
        except Exception as e:
            errors[name] = str(e)
            continue
        for it in items:
            d = it.get("dims")
            if not d:
                dropped_no_dims += 1
                continue
            # allow either orientation of the footprint
            fits = True
            if max_w is not None and max_d is not None:
                fits = (d["w"] <= max_w and d["d"] <= max_d) or (d["d"] <= max_w and d["w"] <= max_d)
            elif max_w is not None:
                fits = d["w"] <= max_w
            if max_h is not None:
                fits &= d["h"] <= max_h
            if max_price is not None and it.get("price") is not None:
                fits &= it["price"] <= max_price
            if fits:
                out.append({**it, "retailer_api": name})
    return {"products": out, "dropped_without_dimensions": dropped_no_dims, "errors": errors}
