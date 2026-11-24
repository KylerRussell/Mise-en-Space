import numpy as np

from mise.pipeline import decompose, normalize, ply, scale


def _gauss(xyz):
    n = len(xyz)
    return ply.Splat(xyz=xyz.astype(np.float32), f_dc=np.zeros((n, 3), np.float32), opacity=np.full(n, 4.0, np.float32),
                     scale=np.full((n, 3), np.log(0.01), np.float32), rot=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32))


def test_ply_roundtrip_and_transform(tmp_path):
    rng = np.random.default_rng(0)
    s = _gauss(rng.normal(size=(100, 3)))
    s.f_rest = rng.normal(size=(100, 45)).astype(np.float32)
    ply.write_ply(tmp_path / "a.ply", s)
    r = ply.read_ply(tmp_path / "a.ply")
    assert np.allclose(r.xyz, s.xyz) and r.f_rest.shape == (100, 45)
    t = ply.transform(r, R=ply.rot_z(90), t=np.array([1, 2, 3]), scale=2.0)
    assert np.allclose(t.xyz[0], 2 * ply.rot_z(90) @ r.xyz[0] + [1, 2, 3], atol=1e-5)
    assert np.allclose(t.scale, r.scale + np.log(2.0))
    assert np.allclose(np.linalg.norm(t.rot, axis=1), 1, atol=1e-5)


def _room_points(rng, tilt_deg=8.0, yaw_deg=23.0, scale_units=0.5):
    """A 4 x 3 m room (floor + 4 walls), then tilted, rotated and scaled into 'reconstruction units'."""
    n = 6000
    floor = np.c_[rng.uniform(0, 4, n), rng.uniform(0, 3, n), np.zeros(n)]
    walls = []
    for x in (0, 4):
        walls.append(np.c_[np.full(n // 4, x), rng.uniform(0, 3, n // 4), rng.uniform(0, 2.5, n // 4)])
    for y in (0, 3):
        walls.append(np.c_[rng.uniform(0, 4, n // 4), np.full(n // 4, y), rng.uniform(0, 2.5, n // 4)])
    pts = np.vstack([floor, *walls])
    a = np.radians(tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    pts = (pts @ ply.rot_z(yaw_deg).T) @ Rx.T
    return pts / scale_units + np.array([3.0, -1.0, 7.0])


def test_normalize_recovers_metric_z_up_axis_aligned():
    rng = np.random.default_rng(0)
    s = _gauss(_room_points(rng))
    out, res = normalize.normalize(s, scale=0.5, up_hint=(0, 0, 1))
    p = out.xyz.astype(np.float64)
    assert abs(np.percentile(p[:, 2], 1)) < 0.03  # floor at z = 0
    ext = np.sort(np.percentile(p[:, :2], 99, axis=0) - np.percentile(p[:, :2], 1, axis=0))
    assert np.allclose(ext, [2.94, 3.92], atol=0.08)  # walls on axes -> tight extents


def test_scale_from_markers_and_held_out():
    rng = np.random.default_rng(0)
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    true_scale = 2.0  # meters per model unit
    side = 0.15
    corners_m = np.array([[0, 0, 0], [side, 0, 0], [side, side, 0], [0, side, 0]]) + [1.0, 1.0, 0]
    corners_model = corners_m / true_scale
    cams, dets = {}, {}
    for i in range(5):
        R = ply.rot_z(10 * i) @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        C = np.array([0.5 + 0.05 * i, 0.5, 1.5])  # camera center (model units) above the marker
        t = -R @ C
        cams[f"f{i}"] = scale.Camera(K=K, R=R, t=t)
        uv = []
        for X in corners_model:
            x = K @ (R @ X + t)
            uv.append(x[:2] / x[2] + rng.normal(0, 0.1, 2))
        dets[f"f{i}"] = {7: np.array(uv)}
    ms = scale.marker_scales(cams, dets, side)
    assert len(ms) == 1 and abs(ms[0].scale - true_scale) / true_scale < 0.01
    held = [scale.HeldOut("room length", (0, 0, 0), (4.2 / true_scale, 0, 0), 4.2)]
    rep = scale.build_report(ms, held)
    assert rep.verdict == "pass" and abs(rep.held_out[0]["error_pct"]) < 1


def test_decompose_package(tmp_path):
    rng = np.random.default_rng(0)
    room = np.c_[rng.uniform(0, 4, 8000), rng.uniform(0, 3, 8000), rng.uniform(-0.01, 0.01, 8000)]
    # a 1.2 x 0.6 x 0.75 'desk' rotated 30°, centered at (2, 1.5)
    loc = np.c_[rng.uniform(-0.6, 0.6, 3000), rng.uniform(-0.3, 0.3, 3000), rng.uniform(0, 0.75, 3000)]
    desk = loc @ ply.rot_z(30).T + [2, 1.5, 0]
    s = _gauss(np.vstack([room, desk]))
    labels = np.r_[np.zeros(len(room), int), np.ones(len(desk), int)]
    bg, objs = decompose.split_instances(s, labels)
    box = decompose.oriented_box(objs[1])
    assert np.allclose([box.w, box.d, box.h], [1.2, 0.6, 0.75], atol=0.03)
    cleaned, hole = decompose.remove_with_shadow(bg, [box])
    assert hole.sum() > 0
    filled = decompose.fill_floor_plane(cleaned, [box.footprint(0.15)])
    assert len(filled) > len(cleaned)
    sc = decompose.package(tmp_path / "out", filled, {"desk_1": (objs[1], "Desk", "desk")},
                           decompose.floor_polygon_from_splat(filled), 2.5)
    assert (tmp_path / "out/objects/desk_1.ply").exists() and (tmp_path / "out/layout.json").exists()
    o = sc.objects[0]
    assert np.isclose(o.pose.x, 2, atol=0.03) and np.isclose(o.dims.w, 1.2, atol=0.03)
    canon = ply.read_ply(tmp_path / "out/objects/desk_1.ply").xyz
    assert np.allclose(np.ptp(canon[:, :2], axis=0), [1.2, 0.6], atol=0.03)


def test_scale_from_reference_box():
    # a 1 ft x 1.5 ft x 1 ft box, rotated 25°, in model units where 1 unit = 0.4 m, with noise and an outlier
    rng = np.random.default_rng(3)
    w, d, h = 0.4572 / 0.4, 0.3048 / 0.4, 0.3048 / 0.4
    top = np.c_[rng.uniform(-w / 2, w / 2, 300), rng.uniform(-d / 2, d / 2, 300), np.full(300, h)]
    side = np.c_[rng.uniform(-w / 2, w / 2, 150), np.full(150, -d / 2), rng.uniform(0, h, 150)]
    pts = np.vstack([top, side]) @ ply.rot_z(25).T + [2.0, 1.0, 0] + rng.normal(0, 0.003, (450, 3))
    pts = np.vstack([pts, [[9, 9, 0.5]]])
    pts = scale.largest_cluster(pts, radius=0.3)
    ref = scale.scale_from_reference(pts, known_m=0.3048, trim=1.0)
    assert abs(ref.scale - 0.4) / 0.4 < 0.03


def test_fit_level_rectangle_from_clicked_corners():
    # a 0.476 x 0.305 rim 0.28 above the floor, in model units of 0.5 m, seen from three nearby cameras
    rng = np.random.default_rng(1)
    k = 0.5
    level_R, level_t = ply.rot_z(15), np.array([0.3, -0.2, 0.1])  # model -> leveled
    rect_m = dict(cx=1.0, cy=2.0, h=0.28, yaw=np.radians(30), a=0.476, b=0.305)
    q = np.array([rect_m["cx"], rect_m["cy"], rect_m["h"], rect_m["yaw"], rect_m["a"], rect_m["b"]]) / [k, k, k, 1, k, k]
    corners_level = scale._rect_corners(q)
    K = np.array([[1300, 0, 800], [0, 1300, 450], [0, 0, 1.0]])
    cams, clicks = {}, {}
    for i, C_m in enumerate([(0.2, 1.2, 1.5), (0.5, 1.0, 1.4), (1.6, 1.1, 1.5)]):
        C = level_R.T @ (np.array(C_m) / k - level_t)  # camera center in model frame
        target = level_R.T @ (np.array([rect_m["cx"], rect_m["cy"], 0.0]) / k - level_t)
        f = target - C
        f /= np.linalg.norm(f)
        up_model = level_R.T @ np.array([0, 0, 1.0])
        x = np.cross(f, up_model)
        x /= np.linalg.norm(x)
        y = np.cross(f, x)
        R = np.vstack([x, y, f])
        cam = scale.Camera(K=K, R=R, t=-R @ C)
        cams[f"f{i}"] = cam
        clicks[f"f{i}"] = {}
        for name, Xl in corners_level.items():
            h = cam.P @ np.r_[level_R.T @ (Xl - level_t), 1]
            clicks[f"f{i}"][name] = h[:2] / h[2] + rng.normal(0, 1.0, 2)
    del clicks["f2"]["BL"]
    clicks["f0"]["FR"] = clicks["f0"]["FR"] + [15, 0]  # one bad click
    rect = scale.fit_level_rectangle(cams, clicks, level_R, level_t)
    s = rect.scale_for(0.3048)
    assert abs(s - k) / k < 0.03
    assert abs(rect.long_side * s - 0.476) < 0.02 and abs(rect.height * s - 0.28) < 0.02


def test_align_by_time_recovers_rotation_scale_offset():
    """Two solves of overlapping video: the second lands in the first's metric frame."""
    from types import SimpleNamespace as NS

    from mise.pipeline import sfm_scene

    def rz(d):
        a = np.radians(d)
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    def rec_from(centers, rots, names):
        ims = {}
        for k, (c, Rwc, n) in enumerate(zip(centers, rots, names)):
            Rcw = Rwc.T
            ims[k] = NS(name=n, projection_center=lambda c=c: c, points2D=[],
                        cam_from_world=lambda Rcw=Rcw: NS(rotation=NS(matrix=lambda: Rcw)))
        return NS(images=ims, points3D={})

    T = np.linspace(0, 20, 61)
    path = np.c_[np.cos(T / 4) * 2, np.sin(T / 3) * 1.5, 1.4 + 0.05 * np.sin(T)]  # metric, leveled
    looks = [rz(10 * t) for t in T]
    ref_names = [f"a{k:03d}" for k in range(len(T))]
    ref_rec = rec_from(path, looks, ref_names)
    ref = sfm_scene.Leveled(R=np.eye(3), t=np.zeros(3), scale=1.0, points=np.zeros((0, 3)), point_ids=np.zeros(0),
                            cams=path, image_names=ref_names, sights=[np.zeros((0, 3))] * len(T), yaw_deg=0.0)
    # the second solve: other frames (offset in time), its own arbitrary frame and units
    Rm, sm, tm = rz(70) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), 2.5, np.array([3.0, -1.0, 0.5])
    T2 = T[10:50] + 0.05
    path2 = np.c_[np.cos(T2 / 4) * 2, np.sin(T2 / 3) * 1.5, 1.4 + 0.05 * np.sin(T2)]
    model = (path2 * sm) @ Rm.T + tm
    rots = [Rm @ rz(10 * t) for t in T2]
    names2 = [f"b{k:03d}" for k in range(len(T2))]
    rec = rec_from(model, rots, names2)
    times = {**dict(zip(ref_names, T)), **dict(zip(names2, T2))}
    lev, rep = sfm_scene.align_by_time(ref_rec, ref, times, rec, times)
    assert rep["used"] >= 30
    assert np.allclose(lev.R, Rm.T, atol=0.02) and abs(lev.scale - 1 / sm) < 1e-3  # 0.05 s apart while turning 10°/s
    assert np.abs(lev.cams - path2).max() < 0.03
