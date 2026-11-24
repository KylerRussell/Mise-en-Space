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
    # Which measured axis became (w, d, h). (0, 1, 2) = as generated; (1, 0, 2) = turned
    # 90° about the vertical; anything moving index 2 = the object was tipped to match.
    perm: tuple[int, int, int] = (0, 1, 2)

    @property
    def tipped(self) -> bool:
        return self.perm[2] != 2


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
              accept: float = ACCEPT, warn: float = WARN, allow_wd_swap: bool = True,
              allow_tipping: bool = False) -> ScaleFit:
    """listed/measured are (w, d, h). Measured is in the generator's units.

    allow_wd_swap: also try the object turned 90° about the vertical (a wrong "front").
    allow_tipping: also try every way of standing it on a side or end. Use this for
    objects that are not upright-only: the listing may show a cube shelf lying down while
    the generator made it standing up, and either is a valid way to own it.
    """
    L = np.asarray(listed, dtype=float)
    M0 = np.asarray(measured, dtype=float)
    if (L <= 0).any() or (M0 <= 0).any():
        raise ValueError("dimensions must be positive")
    perms = [(0, 1, 2)]
    if allow_wd_swap or allow_tipping:
        perms.append((1, 0, 2))
    if allow_tipping:
        perms += [(0, 2, 1), (2, 1, 0), (1, 2, 0), (2, 0, 1)]
    fits = [(p, *_fit(L, M0[list(p)])) for p in perms]
    best = fits[0]
    for f in fits[1:]:
        # Only leave the generated arrangement when another one is clearly better.
        if f[3] < best[3] / 1.05:
            best = f
    perm, s, ratios, aniso = best
    M = M0[list(perm)]
    notes: list[str] = []
    if perm == (1, 0, 2):
        notes.append(f"width and depth match better turned 90° (anisotropy {aniso:.3f} vs {fits[0][3]:.3f}); front axis was probably wrong")
    elif perm[2] != 2:
        notes.append(f"matched the listing by tipping the model onto its {'side' if perm[2] == 0 else 'back'} "
                     f"(anisotropy {aniso:.3f} vs {fits[0][3]:.3f} as generated)")
    swapped = perm == (1, 0, 2)
    fitted = M * s
    err = (fitted - L) / L
    g = gate_for(aniso, accept, warn)
    worst = int(np.argmax(np.abs(err)))
    if g != "accept":
        notes.append(f"{'wdh'[worst]} is off by {err[worst] * 100:+.1f}% after uniform scaling")
    return ScaleFit(scale=s, ratios=tuple(map(float, ratios)), anisotropy=aniso, gate=g,
                    fitted_dims=tuple(map(float, fitted)), per_axis_error=tuple(map(float, err)),
                    swapped_wd=swapped, notes=notes, perm=perm)
