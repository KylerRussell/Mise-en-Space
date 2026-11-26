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
    assert abs(out.area_m2 - 16.25) / 16.25 < 0.04
    assert len(out.polygon) == 6
    xs = sorted({round(x, 1) for x, _ in out.polygon})
    assert np.allclose(xs, [0, 2.5, 5], atol=0.15)
    assert out.ceiling_m is not None and abs(out.ceiling_m - 2.5) < 0.05


def test_segment_home_splits_rooms_at_doorway():
    # two rooms side by side, joined by a 0.8 m doorway in a 0.12 m wall at x = 3..3.12
    rng = np.random.default_rng(0)
    walls = [((0, 0), (3, 0)), ((3, 0), (3, 1.0)), ((3, 1.8), (3, 3)), ((3, 3), (0, 3)), ((0, 3), (0, 0)),
             ((3.12, 0), (6, 0)), ((6, 0), (6, 3)), ((6, 3), (3.12, 3)), ((3.12, 3), (3.12, 1.8)), ((3.12, 1.0), (3.12, 0)),
             ((3, 1.0), (3.12, 1.0)), ((3, 1.8), (3.12, 1.8))]
    pts = []
    for a, b in walls:
        a, b = np.array(a, float), np.array(b, float)
        t = rng.uniform(0, 1, max(4, int(80 * np.linalg.norm(b - a))))
        xy = a + (b - a) * t[:, None]
        pts.append(np.c_[xy, rng.uniform(0.1, 2.4, len(t))])
    pts = np.vstack(pts)
    segs = [(np.array(a, float), np.array(b, float)) for a, b in walls]

    def visible(c, p):
        q = p[:2] + (c[:2] - p[:2]) * 0.02 / max(np.linalg.norm(c[:2] - p[:2]), 1e-9)
        return not any(_cross(c[:2], q, a, b) for a, b in segs)

    cams = np.array([(1, 1, 1.5), (2, 2, 1.5), (1.5, 1.4, 1.5), (2.6, 1.4, 1.5), (3.5, 1.4, 1.5), (4.5, 1, 1.5),
                     (5, 2.2, 1.5), (4, 2.4, 1.5)])
    sights = [np.array([p for p in pts if visible(c, p)]) for c in cams]
    home = room_outline.segment_home(cams, sights)
    assert len(home.rooms) == 2
    areas = sorted(r.area_m2 for r in home.rooms)
    assert np.allclose(areas, [8.64, 9.0], atol=0.35)
    assert len(home.doorways) == 1
    d = home.doorways[0]
    assert abs(d.center[1] - 1.4) < 0.15 and abs(d.width - 0.8) < 0.2


def test_home_scene_from_outline():
    from mise.pipeline import sfm_scene
    from mise.scene.checks import check_layout

    home = room_outline.HomeOutline(
        rooms=[room_outline.Outline(polygon=[(0, 0), (3, 0), (3, 3), (0, 3)], area_m2=9, ceiling_m=2.45, free=None,
                                    origin=(0, 0), cell=0.05),
               room_outline.Outline(polygon=[(3.12, 0), (6, 0), (6, 3), (3.12, 3)], area_m2=8.64, ceiling_m=None,
                                    free=None, origin=(0, 0), cell=0.05)],
        doorways=[room_outline.Doorway(center=(3.06, 1.4), width=0.8, rooms=(0, 1), axis="y")],
        labels=np.zeros((1, 1)), origin=(0, 0), cell=0.05)
    sc = sfm_scene.home_scene(home, "h", "Home")
    assert [s.id for s in sc.room.spaces] == ["room_1", "room_2"] and sc.room.ceiling_height == 2.45
    assert {o.wall.split(".")[0] for o in sc.room.openings} == {"room_1", "room_2"}
    assert all(abs(o.width - 0.8) < 1e-9 for o in sc.room.openings)
    assert check_layout(sc) == []  # the two rooms are joined through the doorway


def test_fuzzy_walls_furniture_and_a_closet():
    """Dense, noisy learned depth: walls are a fuzzy layer, a sideboard stands against one wall, and
    a reach-in closet opens off the room. The room keeps its true walls; the closet is its own space."""
    rng = np.random.default_rng(3)
    # room 0..4 x 0..3.6; closet 4.12..4.72 x 1.0..2.4, opening 1.2..2.2 in the wall x = 4..4.12
    walls = [((0, 0), (4, 0)), ((4, 0), (4, 1.2)), ((4, 2.2), (4, 3.6)), ((4, 3.6), (0, 3.6)), ((0, 3.6), (0, 0)),
             ((4, 1.2), (4.12, 1.2)), ((4, 2.2), (4.12, 2.2)), ((4.12, 1.2), (4.12, 1.0)), ((4.12, 1.0), (4.72, 1.0)),
             ((4.72, 1.0), (4.72, 2.4)), ((4.72, 2.4), (4.12, 2.4)), ((4.12, 2.4), (4.12, 2.2))]
    segs = [(np.array(a, float), np.array(b, float)) for a, b in walls]
    pts = []
    for a, b in segs:
        t = rng.uniform(0, 1, max(4, int(300 * np.linalg.norm(b - a))))
        e = (b - a) / np.linalg.norm(b - a)
        xy = a + (b - a) * t[:, None] + np.array([e[1], -e[0]]) * rng.normal(0, 0.08, len(t))[:, None]
        pts.append(np.c_[xy, rng.uniform(0.1, 2.4, len(t))])
    # a 0.45 m deep sideboard along the south wall, 0.9 m tall: its front and top are points too
    side = np.c_[rng.uniform(0.5, 3.0, 1500), rng.uniform(0.0, 0.45, 1500), rng.uniform(0.1, 0.9, 1500)]
    pts = np.vstack(pts + [side])
    cams = np.array([(x, y, 1.45) for x in (0.8, 1.6, 2.4, 3.2) for y in (0.9, 1.8, 2.7)] + [(3.6, 1.7, 1.45), (3.8, 1.5, 1.4)])

    def visible(c, p):
        q = p[:2] + (c[:2] - p[:2]) * 0.12 / max(np.linalg.norm(c[:2] - p[:2]), 1e-9)
        if 0.5 < p[0] < 3.0 and p[1] < 0.45 and p[2] > 0.9 and c[1] > 0.45:
            return True  # sideboard top and front
        blocked = any(_cross(c[:2], q, a, b) for a, b in segs)
        behind_side = c[2] > 0 and p[2] < 0.9 and p[1] < 0.45 and 0.5 < p[0] < 3.0 and not (0.40 < p[1] < 0.45)
        return not blocked and not behind_side

    sights = [np.array([p for p in pts if visible(c, p)]) for c in cams]
    home = room_outline.segment_home(cams, sights, pts)
    _, placed = room_outline.snap_edges_to_walls(home.rooms, cams, sights)
    room_outline.push_furniture_edges(home.rooms, pts, skip=placed)
    room_outline.settle_closets(home.rooms)
    areas = sorted(r.area_m2 for r in home.rooms)
    assert len(home.rooms) == 2, areas
    assert abs(areas[1] - 14.4) / 14.4 < 0.06  # the room, walls on the wall layer, not at the sideboard
    assert 0.6 < areas[0] < 1.05  # the closet: 0.84 m2, plus the doorway's depth through the wall
    closet = min(home.rooms, key=lambda r: r.area_m2)
    assert closet.kind == "closet"
    room_poly, closet_poly = np.array(max(home.rooms, key=lambda r: r.area_m2).polygon), np.array(closet.polygon)
    assert closet_poly[:, 0].min() >= room_poly[:, 0].max() - 0.01  # no overlap: the mouth is on the room's wall
    room = max(home.rooms, key=lambda r: r.area_m2)
    ys = [y for _, y in room.polygon]
    assert min(ys) < 0.08  # south wall at the wall, behind the sideboard
    assert any(set(d.rooms) == {0, 1} for d in home.doorways)
