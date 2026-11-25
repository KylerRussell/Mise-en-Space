"""Scoring a solve against a floor plan: the fit finds the turn, offset and scale."""

import numpy as np

from mise.eval.floorplan import best_fit, plan_segments, room_spans

PLAN = {"rooms": {
    "main": {"polygon": [[0, 0], [7, 0], [7, 4.2], [4.6, 4.2], [4.6, 2.2], [3.8, 2.2], [3.8, 3.3], [0, 3.3]]},
    "bedroom": {"polygon": [[0, 3.4], [3.8, 3.4], [3.8, 6.6], [0, 6.6]]},
}}


def _wall_points(rng, n=3000):
    segs, owner = plan_segments(PLAN)
    k = rng.integers(0, len(segs), n)
    t = rng.uniform(0, 1, n)[:, None]
    return segs[k, 0] + t * (segs[k, 1] - segs[k, 0]) + rng.normal(0, 0.02, (n, 2))


def test_fit_recovers_scale_turn_and_offset():
    rng = np.random.default_rng(0)
    P = _wall_points(rng)
    bedroom = P[(P[:, 1] > 3.35)]  # a stretch filmed in the bedroom sees mostly its walls
    seen = np.vstack([bedroom, P[rng.choice(len(P), 300)]])
    th = np.radians(90)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    model = 0.9 * seen @ R.T + np.array([3.0, -2.0])  # the solve: 10% small, turned, shifted
    cams = 0.9 * np.array([[1.9, 5.3], [1.7, 4.6]]) @ R.T + np.array([3.0, -2.0])
    free = best_fit(model, cams, PLAN, "bedroom", with_scale=True)
    assert abs(free["s"] - 1 / 0.9) < 0.01 and np.median(free["d"]) < 0.03
    fixed = best_fit(model, cams, PLAN, "bedroom")
    spans = room_spans(fixed["s"] * model @ fixed["R"].T + fixed["t"], PLAN, "bedroom")
    assert abs(spans["east-west"]["solve"] - 0.9 * 3.8) < 0.08
