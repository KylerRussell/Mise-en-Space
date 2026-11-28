"""Video in, base layout out: the whole scan pipeline, one call, with live progress.

    frames -> footage checks -> features + matching -> solve (pieces) -> place pieces
    (scale from camera height, level, align) -> layout (rooms, doorways, furniture
    fronts pushed back to the wall) -> built-in candidates -> finishes -> scene

Every step reports to a ScanEvents log (``events.jsonl`` in the work folder, plus an
optional callback), so a monitor can show, while it runs, which stretches of the video
mapped, which didn't and why. The result is a draft: rooms are named "Room 1, 2...",
and anything the pipeline had to guess is listed as an issue to check in the editor.

Needs pycolmap, opencv and ffmpeg (the `poses` extra).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

STAGES = [
    ("frames", "Pick sharp frames"),
    ("check", "Check the footage"),
    ("features", "Find and match features"),
    ("solve", "Solve camera positions"),
    ("place", "Scale, level and join pieces"),
    ("layout", "Trace rooms and doorways"),
    ("builtins", "Find built-ins"),
    ("finishes", "Sample floors and walls"),
    ("scene", "Build the scene"),
]

EYE_HEIGHT = 1.43  # median phone height on a handheld walkthrough (measured on our first scan)


@dataclass
class ScanOptions:
    scene_id: str
    name: str = "Scanned home"
    fps: float = 3.0  # kept frames per second of video
    width: int = 1280
    cam_height: float = EYE_HEIGHT  # scale prior when there are no markers
    scale: float | None = None  # known model->meters scale (markers, LiDAR); overrides cam_height
    max_features: int = 4096
    overlap: int = 12  # sequential matching window (frames)
    revisit_frames: int = 110  # frames thinned to this many are all matched with each other (loop closure)
    min_piece: int = 10  # smaller solved pieces are ignored
    t0: float | None = None  # only this stretch of the video (seconds)
    t1: float | None = None
    resume: bool = False  # reuse the frames and solve already in the work folder
    semantic: Path | None = None  # per-frame class labels (segment_frames.py): walls from structure only


class Cancelled(Exception):
    pass


class ScanEvents:
    """Append-only progress log: one JSON object per line, also handed to `on_event`."""

    def __init__(self, path: Path | None = None, on_event: Callable[[dict], None] | None = None):
        self.path = path
        self.on_event = on_event
        self.events: list[dict] = []
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def emit(self, kind: str, **data) -> dict:
        ev = {"seq": len(self.events), "time": round(time.time(), 3), "kind": kind, **data}
        with self._lock:
            self.events.append(ev)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(ev, default=_jsonable) + "\n")
        if self.on_event:
            self.on_event(ev)
        if self.cancel.is_set():
            raise Cancelled()
        return ev

    def stage(self, name: str, status: str, **data) -> None:
        self.emit("stage", stage=name, status=status, **data)

    def issue(self, kind: str, message: str, severity: str = "warn", t0: float | None = None,
              t1: float | None = None, where: list | None = None, frame: str | None = None) -> None:
        self.emit("issue", issue=kind, message=message, severity=severity, t0=t0, t1=t1, where=where, frame=frame)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


@dataclass
class ScanResult:
    scene_id: str
    frames: int
    registered: int
    pieces: list[dict]
    rooms: int
    area_m2: float
    issues: int
    seconds: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---- the stages ------------------------------------------------------------------------

def _frames(video: Path, work: Path, opts: ScanOptions, ev: ScanEvents) -> list[dict]:
    from .video import extract_sharp_frames, probe

    info = probe(video)
    ev.emit("video", duration_s=info["duration_s"], width=info["width"], height=info["height"])
    fdir = work / "frames"
    kept = extract_sharp_frames(video, fdir, target_fps=opts.fps, oversample=2, width=opts.width, t0=opts.t0, t1=opts.t1)
    _thumbs(work, kept)
    (work / "frames.json").write_text(json.dumps(kept))
    ev.emit("frames", frames=[{"name": k["name"], "t": round(k["t"], 2)} for k in kept])
    return kept


def _thumbs(work: Path, kept: list[dict]) -> None:
    import cv2

    thumbs = work / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for k in kept:
        img = cv2.imread(str(work / "frames" / k["name"]))
        cv2.imwrite(str(thumbs / k["name"]), cv2.resize(img, (240, round(240 * img.shape[0] / img.shape[1]))),
                    [cv2.IMWRITE_JPEG_QUALITY, 70])


def import_poses(poses: Path, work: Path, frames_src: Path | None = None) -> dict:
    """Put a camera solve made elsewhere into the work folder, so a `resume` run builds the home from it.

    `poses` is what scripts/colab/learned_recon.py writes: frames.json ([{"name", "t"}]), a COLMAP
    model in sparse/0 (text or binary) and meta.json (model, whether its scale is metric). The
    frames it names are copied from `frames_src` (default: poses/frames). Earlier frames and
    solves in the work folder are replaced. Returns meta.json (empty if there is none).
    """
    import shutil

    import pycolmap

    poses, work = Path(poses), Path(work)
    meta = json.loads((poses / "meta.json").read_text()) if (poses / "meta.json").exists() else {}
    kept = json.loads((poses / "frames.json").read_text())
    src = Path(frames_src) if frames_src else poses / "frames"
    if kept and not (src / kept[0]["name"]).exists() and (src / "frames").is_dir():
        src = src / "frames"  # the pick folder (frames/ + frames.json), as run_learned_recon.py takes it
    missing = [k["name"] for k in kept if not (src / k["name"]).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames named in {poses / 'frames.json'} are not in {src} "
                                f"(first: {missing[0]}); pass the folder they were picked into")
    model = poses / "sparse" / "0"
    if not model.exists():
        raise FileNotFoundError(f"no COLMAP model at {model}")
    rec = pycolmap.Reconstruction(str(model))
    unknown = {im.name for im in rec.images.values()} - {k["name"] for k in kept}
    if unknown:
        raise ValueError(f"the model has {len(unknown)} images not in frames.json (first: {sorted(unknown)[0]})")
    work.mkdir(parents=True, exist_ok=True)
    for old in ("frames", "thumbs", "frames_eq", "sparse"):
        if (work / old).exists():
            shutil.rmtree(work / old)
    (work / "database.db").unlink(missing_ok=True)
    (work / "frames").mkdir()
    for k in kept:
        shutil.copy2(src / k["name"], work / "frames" / k["name"])
    _thumbs(work, kept)
    (work / "frames.json").write_text(json.dumps(kept))
    out = work / "sparse" / "0"
    out.mkdir(parents=True)
    rec.write_binary(str(out))
    (work / "poses_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _check(work: Path, kept: list[dict], ev: ScanEvents) -> list:
    import cv2

    from . import capture_qc

    a = capture_qc.Analyzer()
    for k in kept:
        fq = a.add(k["t"], cv2.imread(str(work / "frames" / k["name"])))
        status = next((f for f in ["lost", "turning", "close_up", "blurry", "low_texture", "dark"] if f in fq.flags), "ok")
        ev.emit("qc", name=k["name"], t=round(k["t"], 2), status=status,
                parallax=None if np.isnan(fq.parallax_deg) else round(fq.parallax_deg, 2))
    by_t = {round(k["t"], 2): k["name"] for k in kept}
    issues = a.issues()
    for i in issues:
        ev.issue(i.kind, i.advice, i.severity, i.t0, i.t1, frame=by_t.get(round(i.frame_t, 2)) or _nearest(kept, i.frame_t))
    return issues


def _nearest(kept: list[dict], t: float) -> str:
    return min(kept, key=lambda k: abs(k["t"] - t))["name"]


def _features(work: Path, kept: list[dict], opts: ScanOptions, ev: ScanEvents) -> Path:
    """SIFT on contrast-boosted frames, then two kinds of matching:

    - neighbors: each frame with the next `overlap` (and a few further, quadratically);
    - revisits: a thinned set of frames against each other across the whole video, so a
      wall seen at 10 s and again at 80 s ties the two stretches together (loop closure).
    """
    import pycolmap

    from .video import enhance_frames

    db = work / "database.db"
    if db.exists():
        db.unlink()
    eq = work / "frames_eq"
    enhance_frames(work / "frames", eq)  # plain, dim walls: local contrast gives SIFT something to hold on to
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = opts.width
    ext.sift.max_num_features = opts.max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(db, eq, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
                              extraction_options=ext, device=pycolmap.Device.cpu)
    ev.emit("progress", stage="features", message="features found; matching neighbors")
    pair = pycolmap.SequentialPairingOptions()
    pair.overlap = opts.overlap
    pair.quadratic_overlap = True
    pycolmap.match_sequential(db, matching_options=pycolmap.FeatureMatchingOptions(), pairing_options=pair,
                              device=pycolmap.Device.cpu)
    names = [k["name"] for k in kept]
    stride = max(1, int(np.ceil(len(names) / opts.revisit_frames)))
    sub = list(range(0, len(names), stride))
    pairs = [(names[a], names[b]) for i, a in enumerate(sub) for b in sub[i + 1:] if b - a > opts.overlap]
    if pairs:
        ev.emit("progress", stage="features", message=f"matching {len(pairs)} revisit pairs")
        (work / "revisit_pairs.txt").write_text("\n".join(f"{a} {b}" for a, b in pairs))
        po = pycolmap.ImportedPairingOptions()
        po.match_list_path = work / "revisit_pairs.txt"
        pycolmap.match_image_pairs(db, matching_options=pycolmap.FeatureMatchingOptions(), pairing_options=po,
                                   device=pycolmap.Device.cpu)
    return db


def _solve(work: Path, db: Path, n_frames: int, ev: ScanEvents) -> list:
    import pycolmap

    o = pycolmap.IncrementalPipelineOptions()
    o.num_threads = -1
    # plain walls and dim rooms: accept weaker evidence, then check the result (pose sanity)
    o.min_num_matches = 10
    o.mapper.init_min_num_inliers = 50
    o.mapper.abs_pose_min_num_inliers = 15
    o.mapper.abs_pose_min_inlier_ratio = 0.15
    o.mapper.max_reg_trials = 6
    o.multiple_models = True
    o.min_model_size = 6
    count = {"n": 0}
    token = pycolmap.CancellationToken() if hasattr(pycolmap, "CancellationToken") else None

    def next_image():
        count["n"] += 1
        if count["n"] % 5 == 0:
            try:
                ev.emit("progress", stage="solve", registered=count["n"], of=n_frames)
            except Cancelled:
                if token is not None:
                    token.cancel()

    out = work / "sparse"
    out.mkdir(exist_ok=True)
    kw = {"next_image_callback": next_image}
    if token is not None:
        kw["cancellation_token"] = token
    recs = pycolmap.incremental_mapping(db, work / "frames_eq", out, options=o, **kw)
    if ev.cancel.is_set():
        raise Cancelled()
    return sorted(recs.values(), key=lambda r: -r.num_reg_images())


def _scale_and_level(rec, opts: ScanOptions):
    from . import sfm_scene

    if opts.scale:
        return sfm_scene.level_reconstruction(rec, opts.scale), "given"
    # level once at model scale, read the phone's height above the floor, then scale so it matches
    # the usual eye height (and level again: the floor threshold is in meters)
    spread = np.ptp(np.array([im.projection_center() for im in rec.images.values()]), axis=0).max()
    s0 = 3.0 / max(spread, 1e-6)  # a rough guess: walkthroughs span a few meters
    L0 = sfm_scene.level_reconstruction(rec, s0)
    h = float(np.median(L0.cams[:, 2]))
    if h <= 0.05:
        raise RuntimeError("could not find the floor under the cameras")
    s = s0 * opts.cam_height / h
    return sfm_scene.level_reconstruction(rec, s), "camera height"


def _place(recs: list, times: dict[str, float], opts: ScanOptions, ev: ScanEvents):
    """The largest sane piece is the home's frame; other pieces join it if they share time with it."""
    from . import sfm_scene

    pieces = []
    for k, r in enumerate(recs):
        names = sorted(im.name for im in r.images.values())
        ts = [times[n] for n in names if n in times]
        pieces.append({"id": k, "frames": len(names), "t0": round(min(ts), 2) if ts else None,
                       "t1": round(max(ts), 2) if ts else None, "names": names, "status": "pending"})
    ref = L = None
    for p in pieces:
        if p["frames"] < opts.min_piece:
            p["status"] = "too small"
            continue
        if ref is None:
            try:
                Lp, how = _scale_and_level(recs[p["id"]], opts)
            except Exception as e:  # noqa: BLE001
                p["status"] = f"could not level: {e}"
                continue
            sanity = Lp.pose_sanity()
            p["sanity"] = sanity
            if not sanity["ok"]:
                p["status"] = "bent"  # cameras below the floor or spread far in height: a collapsed solve
                continue
            ref, L = p, Lp
            p["status"] = "reference"
            p["scale_from"] = how
            p["scale"] = round(Lp.scale, 5)
            # scene = scale * R @ solve + t: what a splat trained on this solve's poses needs
            p["scene_from_solve"] = {"scale": float(Lp.scale), "R": np.asarray(Lp.R).tolist(), "t": np.asarray(Lp.t).tolist()}
            continue
        try:
            Lp, rep = sfm_scene.align_by_time(recs[ref["id"]], L, times, recs[p["id"]], times)
            p["align"] = rep
            ok = rep["rot_spread_deg"] < 5 and rep["center_residual_m"] < 0.25 and Lp.pose_sanity()["ok"]
            p["status"] = "joined" if ok else "could not join"
            if ok:
                p["leveled"] = Lp
        except RuntimeError as e:
            p["status"] = "could not join"
            p["reason"] = str(e)
    for p in pieces:
        ev.emit("piece", **{k: v for k, v in p.items() if k not in ("names", "leveled")},
                sample=p["names"][:: max(1, len(p["names"]) // 40)])
    return pieces, ref, L


def _merge(L, pieces):
    """The reference piece plus joined pieces, as one set of cameras, sights and points."""
    cams, sights, points, names = [L.cams], list(L.sights), [L.points], list(L.image_names)
    for p in pieces:
        Lp = p.get("leveled")
        if Lp is None:
            continue
        cams.append(Lp.cams)
        sights += Lp.sights
        points.append(Lp.points)
        names += Lp.image_names
    return np.vstack(cams), sights, np.vstack(points), names


def counter_layer(points: np.ndarray, lo: float = 0.76, hi: float = 1.02, step: float = 0.02) -> float | None:
    """Height of the densest horizontal layer of points where counters sit (standard 0.90-0.92 m).

    The search window is wide because the scale from camera height can be ~10% off.
    """
    z = points[:, 2]
    zz = z[(z > lo - 0.1) & (z < hi + 0.1)]
    if len(zz) < 100:
        return None
    centers = np.arange(lo, hi + 1e-9, step)
    counts = np.array([((zz > c - 0.03) & (zz < c + 0.03)).sum() for c in centers])
    # a layer, not a slope: the peak must stand out from the band around it
    k = int(counts.argmax())
    if counts[k] < 1.4 * np.median(counts):
        return None
    sel = zz[(zz > centers[k] - 0.03) & (zz < centers[k] + 0.03)]
    return float(np.median(sel))


def detect_counters(points: np.ndarray, rooms: list, z: tuple[float, float] | None = None, cell: float = 0.05,
                    min_area: float = 0.35, min_density: int = 3) -> list[dict]:
    """Counter tops: dense, flat layers of points at counter height (sinks, islands, runs of cabinets).

    z: the height band; by default around the densest layer near counter height. Tables and
    desks sit lower (~0.75 m); a sideboard can still pass, so every hit is a candidate to
    confirm. Returns boxes {x0, y0, x1, y1, top}.
    """
    import cv2

    if z is None:
        top = counter_layer(points)
        if top is None:
            return []
        z = (top - 0.05, top + 0.05)
    band = points[(points[:, 2] > z[0]) & (points[:, 2] < z[1])]
    if len(band) < 50:
        return []
    lo = band[:, :2].min(0) - cell
    ij = ((band[:, :2] - lo) / cell).astype(int)
    H, W = ij[:, 1].max() + 2, ij[:, 0].max() + 2
    grid = np.zeros((H, W), np.int32)
    np.add.at(grid, (ij[:, 1], ij[:, 0]), 1)
    occ = (grid >= min_density).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(occ)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area * cell * cell < min_area:
            continue
        fill = area / float(w * h)
        if fill < 0.55 or min(w, h) * cell < 0.35:  # counters are solid rectangles at least ~40 cm deep
            continue
        tops = band[:, 2][(ij[:, 0] >= x) & (ij[:, 0] < x + w) & (ij[:, 1] >= y) & (ij[:, 1] < y + h)]
        out.append({"x0": float(lo[0] + x * cell), "y0": float(lo[1] + y * cell), "x1": float(lo[0] + (x + w) * cell),
                    "y1": float(lo[1] + (y + h) * cell), "top": round(float(np.median(tops)), 3)})
    return out


def _builtins(points: np.ndarray, scene, ev: ScanEvents) -> list:
    from ..scene.models import Dims, Pose, SceneObject

    objs = []
    for k, b in enumerate(detect_counters(points, scene.room.spaces)):
        w, d = b["x1"] - b["x0"], b["y1"] - b["y0"]
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        if scene.room.space_at(cx, cy) is None:
            continue
        # long side as width; front faces the room's larger open side (refine in the editor)
        yaw = 0.0 if w >= d else 90.0
        dims = Dims(w=round(max(w, d), 2), d=round(min(w, d), 2), h=b["top"])
        objs.append(SceneObject(
            id=f"counter_{k + 1}", name="Counter / cabinets (detected)", category="counter", dims=dims,
            pose=Pose(x=round(cx, 3), y=round(cy, 3), yaw=yaw), fixture=True, locked=True, color="#c9c7c1",
            warnings=[("Detected from a flat layer of points at counter height: check it is built in, "
                       "and add appliances and upper cabinets by hand.")]))
        ev.emit("builtin", id=objs[-1].id, box=[b["x0"], b["y0"], b["x1"], b["y1"]], top=b["top"])
    return objs


def _layout_payload(cams, points, home, fixes) -> dict:
    rng = np.random.default_rng(0)
    pts = points[(points[:, 2] > 0.1) & (points[:, 2] < 2.3)]
    if len(pts) > 4000:
        pts = pts[rng.choice(len(pts), 4000, replace=False)]
    return {"cams": np.round(cams[:, :2], 3).tolist(), "points": np.round(pts[:, :2], 3).tolist(),
            "rooms": [{"polygon": [[round(v, 3) for v in p] for p in r.polygon], "area_m2": round(r.area_m2, 2)}
                      for r in home.rooms],
            "doorways": [{"center": list(d.center), "width": round(d.width, 2)} for d in home.doorways],
            "suspect": fixes}


# ---- the pipeline ------------------------------------------------------------------------

def run_scan(video: Path, work: Path, opts: ScanOptions, store=None, ev: ScanEvents | None = None) -> ScanResult:
    """Run every stage; write the scene into `store` (a SceneStore) if given, else to work/scene."""
    from . import finishes, room_outline, sfm_scene

    started = time.time()
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ev = ev or ScanEvents(work / "events.jsonl")
    ev.emit("start", scene_id=opts.scene_id, options=asdict(opts))
    current = None
    try:
        current = "frames"
        ev.stage("frames", "running")
        resume = opts.resume and (work / "frames.json").exists() and any((work / "sparse").glob("*/images.bin"))
        if resume:  # keep the frames and the solve from the last run; redo everything after
            kept = json.loads((work / "frames.json").read_text())
            ev.emit("frames", frames=[{"name": k["name"], "t": round(k["t"], 2)} for k in kept])
        else:
            kept = _frames(Path(video), work, opts, ev)
        if len(kept) < 20:
            raise RuntimeError(f"only {len(kept)} frames: film at least a minute, walking slowly")
        times = {k["name"]: k["t"] for k in kept}
        ev.stage("frames", "done", count=len(kept))

        current = "check"
        ev.stage("check", "running")
        qc_issues = _check(work, kept, ev)
        ev.stage("check", "done", issues=len(qc_issues))

        current = "features"
        if resume:
            import pycolmap

            ev.stage("features", "skipped", reason="kept from the last run")
            current = "solve"
            ev.stage("solve", "running")
            recs = sorted((pycolmap.Reconstruction(d) for d in sorted((work / "sparse").iterdir())
                           if (d / "images.bin").exists()), key=lambda r: -r.num_reg_images())
        else:
            ev.stage("features", "running")
            db = _features(work, kept, opts, ev)
            ev.stage("features", "done")
            current = "solve"
            ev.stage("solve", "running")
            recs = _solve(work, db, len(kept), ev)
        reg = {im.name: k for k, r in enumerate(recs) for im in r.images.values()}
        ev.emit("registration", frames={n: reg.get(n) for n in times})
        ev.stage("solve", "done", pieces=len(recs), registered=len(reg))
        if not recs:
            raise RuntimeError("no part of the video could be solved: see the footage checks")

        current = "place"
        ev.stage("place", "running")
        pieces, ref, L = _place(recs, times, opts, ev)
        if ref is None:
            raise RuntimeError("every solved piece looked bent (cameras under the floor): see the footage checks")
        placed = {n for p in pieces if p["status"] in ("reference", "joined") for n in p["names"]}
        # stretches of video that ended up nowhere: what to film again
        run = []
        for k in kept + [None]:
            if k is not None and k["name"] not in placed:
                run.append(k)
                continue
            if len(run) >= 3:
                t0, t1 = run[0]["t"], run[-1]["t"]
                where = "solved on its own but could not be tied to the rest" if any(r["name"] in reg for r in run) \
                    else "could not be solved"
                ev.issue("unplaced", f"{t0:.0f}-{t1:.0f} s {where}. Film this part again, starting from somewhere "
                         "already mapped and walking (not turning) through the doorway.", "fail", round(t0, 2), round(t1, 2),
                         frame=run[len(run) // 2]["name"])
            run = []
        ev.stage("place", "done", placed=len(placed), of=len(kept), scale_from=ref.get("scale_from"))

        current = "layout"
        ev.stage("layout", "running")
        cams, sights, points, names = _merge(L, pieces)
        wall_points, wall_cams, wall_sights = points, cams, sights
        joined = [p for p in pieces if p.get("leveled") is not None]
        if opts.semantic and not joined:
            # what each point is (wall, bed, mirror...): free space is carved by every sight line but
            # those through mirrors and windows, and walls are placed from the home's shell only
            from . import sfm_scene, structure

            labels, classes = structure.load_labels(Path(opts.semantic))
            labels = structure.fill_see_through(labels, classes)
            rr = recs[ref["id"]]
            carve, rc = structure.structure_only(rr, labels, classes, keep=None, drop=structure.SEE_THROUGH)
            shell, rs = structure.structure_only(rr, labels, classes)
            Lc = sfm_scene._leveled(carve, L.R, L.t, L.scale, L.yaw_deg)
            Ls = sfm_scene._leveled(shell, L.R, L.t, L.scale, L.yaw_deg)
            cams, sights, names = Lc.cams, Lc.sights, Lc.image_names
            wall_points, wall_cams, wall_sights = Ls.points, Ls.cams, Ls.sights
            ev.emit("progress", stage="layout", message=f"{rc['dropped']} points seen through mirrors and windows set "
                    f"aside; walls from {len(Ls.points)} of {rs['points']} points (wall, floor, ceiling, doors)")
        elif opts.semantic:
            ev.emit("progress", stage="layout", message="frame labels not used: the home is in several pieces")
        home = room_outline.segment_home(cams, sights, wall_points, walk=cams[np.argsort(names)])  # names sort in time
        # walls where the points above furniture height put them; the furniture test only where those are too few
        snapped, placed = room_outline.snap_edges_to_walls(home.rooms, wall_cams, wall_sights)
        if wall_sights is not sights:  # where the shell is hidden (a closet full of clothes): every point, outermost layer
            more, placed2 = room_outline.snap_edges_to_walls(home.rooms, cams, sights, skip=placed)
            snapped, placed = snapped + more, placed | placed2
        ev.emit("progress", stage="layout", message=f"{len(placed)} walls found on their points, {len(snapped)} moved")
        fixes = room_outline.push_furniture_edges(home.rooms, points, skip=placed)
        room_outline.settle_closets(home.rooms)
        reg = room_outline.regularize_walls(home.rooms)
        ev.emit("progress", stage="layout", message=f"{reg['shared_walls']} shared walls set {100 * reg['thickness_m']:.0f} cm "
                f"thick, {reg['wall_lines']} wall lines straightened")
        fix_info = []
        for f in fixes:
            poly = home.rooms[f.room].polygon
            a, b = poly[f.edge], poly[(f.edge + 1) % len(poly)]
            fix_info.append({"room": f.room, "a": a, "b": b, "moved_m": f.moved_m, "reason": f.reason})
            ev.issue("furniture_wall", f"A wall here was traced along furniture; moved {f.moved_m:.2f} m back to "
                     "where the furniture's points end. Check it in the editor.", "warn", where=[a, b])
        ev.emit("layout", **_layout_payload(cams, points, home, fix_info))
        ev.stage("layout", "done", rooms=len(home.rooms), doorways=len(home.doorways))
        if not home.rooms:
            raise RuntimeError("no room outline: too few points (plain walls?). See the footage checks")
        scene = sfm_scene.home_scene(home, opts.scene_id, opts.name)
        if ref.get("scale_from") == "camera height":
            ceiling, counter = room_outline.outline_ceiling(points), counter_layer(points)
            cues = [f"ceiling {ceiling:.2f} m (usually 2.4-2.7)" if ceiling else None,
                    f"counter tops {counter:.2f} m (usually 0.90-0.92)" if counter else None]
            ev.issue("scale", "Sizes come from your phone's usual height, so they can be off by up to ~10%. Measured: "
                     + ("; ".join(c for c in cues if c) or "nothing to cross-check")
                     + ". For exact sizes, scan with printed markers or give one real measurement (--scale).", "warn")

        current = "builtins"
        ev.stage("builtins", "running")
        scene.objects = _builtins(points, scene, ev)
        ev.stage("builtins", "done", count=len(scene.objects))

        current = "finishes"
        ev.stage("finishes", "running")
        scene_dir = (store.scene_dir(opts.scene_id) if store is not None else work / "scene")
        ref_rec = recs[ref["id"]]
        try:
            done = finishes.auto_finishes(ref_rec, L, scene, work / "frames", scene_dir)
            ev.stage("finishes", "done", spaces=done)
        except Exception as e:  # noqa: BLE001 - finishes are cosmetic; never lose the layout over them
            ev.stage("finishes", "skipped", reason=f"{type(e).__name__}: {e}")

        current = "scene"
        ev.stage("scene", "running")
        if store is not None:
            store.create(scene, overwrite=True)
        else:
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / "scene.json").write_text(scene.model_dump_json(indent=1))
        n_issues = sum(1 for e in ev.events if e["kind"] == "issue")
        res = ScanResult(scene_id=opts.scene_id, frames=len(kept), registered=len(reg),
                         pieces=[{k: v for k, v in p.items() if k not in ("names", "leveled")} for p in pieces],
                         rooms=len(home.rooms), area_m2=round(sum(r.area_m2 for r in home.rooms), 1), issues=n_issues,
                         seconds=round(time.time() - started, 1))
        ev.stage("scene", "done")
        ev.emit("done", result=res.to_dict())
        (work / "result.json").write_text(json.dumps(res.to_dict(), indent=1, default=_jsonable))
        return res
    except Cancelled:
        ev.cancel.clear()
        ev.emit("cancelled", stage=current)
        raise
    except Exception as e:
        ev.cancel.clear()
        ev.emit("error", stage=current, message=f"{type(e).__name__}: {e}")
        raise
