"""Stage 4: fit one uniform scale to listed dimensions and gate on anisotropy.

Per axis, ratio = listed / measured. The median ratio becomes the uniform
scale; anisotropy = largest ratio / smallest ratio. We never stretch each
axis to match: that would hide a bad generation instead of surfacing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ACCEPT = 1.10
WARN = 1.25

Gate = Literal["accept", "warn", "reject"]


@dataclass
class ScaleFit:
    scale: float
    ratios: tuple[float, float, float]
    anisotropy: float
    gate: Gate
    fitted_dims: tuple[float, float, float]
    per_axis_error: tuple[float, float, float]
    swapped_wd: bool = False
    notes: list[str] = field(default_factory=list)


def gate_for(anisotropy: float, accept: float = ACCEPT, warn: float = WARN) -> Gate:
    if anisotropy <= accept:
        return "accept"
    if anisotropy <= warn:
        return "warn"
    return "reject"


def _fit(listed: np.ndarray, measured: np.ndarray) -> tuple[float, np.ndarray, float]:
    ratios = listed / measured
    s = float(np.median(ratios))
    return s, ratios, float(ratios.max() / ratios.min())


def fit_scale(listed: tuple[float, float, float], measured: tuple[float, float, float],
              accept: float = ACCEPT, warn: float = WARN, allow_wd_swap: bool = True) -> ScaleFit:
    """listed/measured are (w, d, h). Measured is in the generator's units."""
    L = np.asarray(listed, dtype=float)
    M = np.asarray(measured, dtype=float)
    if (L <= 0).any() or (M <= 0).any():
        raise ValueError("dimensions must be positive")
    s, ratios, aniso = _fit(L, M)
    notes: list[str] = []
    swapped = False
    if allow_wd_swap:
        M2 = M[[1, 0, 2]]
        s2, r2, a2 = _fit(L, M2)
        # A 90° error in "front" swaps width and depth. Prefer the swap only if clearly better.
        if a2 < aniso / 1.05:
            notes.append(f"width and depth match better rotated 90° (anisotropy {a2:.3f} vs {aniso:.3f}); front axis was probably wrong")
            s, ratios, aniso, M, swapped = s2, r2, a2, M2, True
    fitted = M * s
    err = (fitted - L) / L
    g = gate_for(aniso, accept, warn)
    worst = int(np.argmax(np.abs(err)))
    if g != "accept":
        notes.append(f"{'wdh'[worst]} is off by {err[worst] * 100:+.1f}% after uniform scaling")
    return ScaleFit(scale=s, ratios=tuple(map(float, ratios)), anisotropy=aniso, gate=g,
                    fitted_dims=tuple(map(float, fitted)), per_axis_error=tuple(map(float, err)),
                    swapped_wd=swapped, notes=notes)
