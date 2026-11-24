"""Stage 4 step 7 (speculative): match an object's white balance and brightness to the room.

Samples the room splat's floor near the placement, and shifts the object's
base colors so its mean chromaticity and luminance move toward the floor
sample's. ``strength`` < 1 keeps some of the object's own look. This does
not fix light direction; that is a known limit (see Risks in docs/PLAN.md).
"""

from __future__ import annotations

import numpy as np

from ..pipeline.ply import Splat

LUMA = np.array([0.2126, 0.7152, 0.0722])


def floor_sample(room: Splat, x: float, y: float, radius: float = 0.8, max_z: float = 0.05) -> np.ndarray:
    d = np.hypot(room.xyz[:, 0] - x, room.xyz[:, 1] - y)
    m = (d < radius) & (np.abs(room.xyz[:, 2]) < max_z) & (room.alpha() > 0.3)
    if m.sum() < 20:
        m = (np.abs(room.xyz[:, 2]) < max_z) & (room.alpha() > 0.3)
    return room.rgb()[m]


def match_color(obj: Splat, reference_rgb: np.ndarray, reference_white: np.ndarray | None = None,
                strength: float = 0.5) -> Splat:
    """Gray-world style gain toward the room's light color and level.

    reference_white: the expected albedo of the floor sample (if known, e.g.
    from a product photo of the same floor); without it, only the cast
    (chromaticity) and a damped brightness shift are applied.
    """
    rgb = obj.rgb()
    a = obj.alpha()
    w = a / a.sum()
    obj_mean = (rgb * w[:, None]).sum(0)
    ref_mean = reference_rgb.mean(0)
    if reference_white is not None:
        illum = ref_mean / np.clip(reference_white, 1e-3, None)
    else:
        illum = ref_mean / max(float(ref_mean @ LUMA), 1e-3)  # chromaticity only
        illum = illum * (float(ref_mean @ LUMA) / max(float(obj_mean @ LUMA), 1e-3)) ** 0.5
    gain = 1 + strength * (illum - 1)
    out = Splat(obj.xyz, obj.f_dc, obj.f_rest, obj.opacity, obj.scale, obj.rot, dict(obj.extra))
    out.set_rgb(rgb * gain)
    return out
