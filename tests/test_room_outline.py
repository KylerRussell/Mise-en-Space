import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mise.pipeline import room_outline  # noqa: E402

L_ROOM = np.array([(0, 0), (5, 0), (5, 2.5), (2.5, 2.5), (2.5, 4), (0, 4)], float)


def _cross(a, b, c, d):
    def o(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return o(a, b, c) * o(a, b, d) < 0 and o(c, d, a) * o(c, d, b) < 0


def _visible(c, p, poly, eps=0.02):
    q = p[:2] + (c[:2] - p[:2]) * eps / max(np.linalg.norm(c[:2] - p[:2]), 1e-9)  # nudge off the wall
    return not any(_cross(c[:2], q, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def test_outline_recovers_l_shaped_room():
    rng = np.random.default_rng(0)
    pts = []
    for i in range(len(L_ROOM)):
        a, b = L_ROOM[i], L_ROOM[(i + 1) % len(L_ROOM)]
        t = rng.uniform(0, 1, int(60 * np.linalg.norm(b - a)))
        xy = a + (b - a) * t[:, None]
        pts.append(np.c_[xy, rng.uniform(0.1, 2.4, len(t))])
    pts.append(np.c_[rng.uniform(0, 5, 200), rng.uniform(0, 2.5, 200), np.full(200, 2.5)])  # ceiling
    pts = np.vstack(pts)
    cams = np.array([(1, 1, 1.5), (2, 1.2, 1.4), (3.5, 1.2, 1.5), (4.3, 1.8, 1.5), (1.2, 3.2, 1.5), (1.5, 2.2, 1.6),
                     (3.0, 0.8, 1.5), (0.8, 2.0, 1.5)])
    sights = [np.array([p for p in pts if _visible(c, p, L_ROOM)]) for c in cams]
    out = room_outline.outline(cams, sights, points=pts)
    assert abs(out.area_m2 - 16.25) / 16.25 < 0.08
    assert len(out.polygon) == 6
    xs = sorted({round(x, 1) for x, _ in out.polygon})
    assert np.allclose(xs, [0, 2.5, 5], atol=0.15)
    assert out.ceiling_m is not None and abs(out.ceiling_m - 2.5) < 0.05
