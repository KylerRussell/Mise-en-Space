"""Capture checks: telling turning on the spot from walking, and flagging bad footage."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from mise.pipeline import capture_qc as qc  # noqa: E402


def _views(R, t, seed=0):
    rng = np.random.default_rng(seed)
    K = qc.default_K(640, 480)
    P = np.c_[rng.uniform(-3, 3, 600), rng.uniform(-1.5, 1.5, 600), rng.uniform(2, 6, 600)]
    a = (P / P[:, 2:]) @ K.T
    pc = P @ R.T + t
    b = (pc / pc[:, 2:]) @ K.T
    ok = (b[:, 0] > 0) & (b[:, 0] < 640) & (b[:, 1] > 0) & (b[:, 1] < 480)
    return a[ok, :2], b[ok, :2] + rng.normal(0, 0.5, (ok.sum(), 2)), K


def _yaw(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_turning_on_the_spot_has_no_parallax():
    a, b, K = _views(_yaw(12), np.zeros(3))
    rot, par, _, rot_only = qc.measure_motion(a, b, K)
    assert rot_only and abs(rot - 12) < 0.5 and par < 0.3


def test_walking_sideways_has_parallax():
    a, b, K = _views(_yaw(3), np.array([-0.25, 0, 0]))
    rot, par, _, rot_only = qc.measure_motion(a, b, K)
    assert not rot_only and abs(rot - 3) < 1.0 and par > 1.5


def test_flags_and_issue_runs():
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (360, 640)).astype(np.uint8), (0, 0), 1.0)
    a = qc.Analyzer(width=640)
    for i in range(12):  # 4 s of good, detailed frames
        a.add(i / 3, cv2.cvtColor(np.roll(texture, 3 * i, axis=1), cv2.COLOR_GRAY2BGR))
    for i in range(12, 24):  # then 4 s of a dark, featureless view
        a.add(i / 3, np.full((360, 640, 3), 12, np.uint8))
    tl = a.timeline()
    assert all(x["status"] != "dark" for x in tl[:12])
    assert all(x["status"] in ("dark", "lost", "low_texture", "blurry") for x in tl[13:])
    kinds = {i.kind for i in a.issues()}
    assert {"dark", "low_texture"} <= kinds
    dark = next(i for i in a.issues() if i.kind == "dark")
    assert dark.t0 == pytest.approx(4.0) and dark.t1 == pytest.approx(23 / 3, abs=0.01)
