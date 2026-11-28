"""`mise` command line.

  mise demo                         create the demo bedroom and demo apartment scenes
  mise serve [--port 8765]          run the scene backend (+ built editor)
  mise mcp [--http]                 run the MCP server
  mise check SCENE [--variant V]    print layout violations
  mise render SCENE out.png [--view plan|eye]
  mise scale-report ...             Stage 1: marker scale + held-out errors
  mise normalize in.ply out.ply     Stage 1: metric, z up, floor at 0, walls on axes
  mise package ...                  Stage 2: split labeled splat into room + objects + scene.json
  mise import-product SCENE URL     Stage 4: product page -> sized library asset
  mise stage-commands CAPTURE_DIR   print the Nerfstudio commands for a capture
  mise scan VIDEO --scene ID        walkthrough video -> base layout (rooms, doors, built-ins, finishes)
            [--poses DIR]           ... with cameras solved elsewhere (scripts/colab/run_learned_recon.py)
  mise learned-scan VIDEO --scene ID  video -> scene with a learned model on Colab (rough pass, room windows, chain)
  mise join-poses OUT DIR DIR...    join learned solves of overlapping stretches into one (for --poses)
  mise calibrate FRAMES [--t0 --t1] the camera's intrinsics from a CPU solve of one stretch (intrinsics.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _store(args):
    from .scene.store import SceneStore
    return SceneStore(Path(args.data))


def cmd_demo(args):
    from .demo import demo_apartment, demo_scene
    st = _store(args)
    for scene in (demo_scene(args.id), demo_apartment()):
        try:
            s = st.create(scene, overwrite=args.overwrite)
            print(f"created scene {s.id!r} in {st.scene_dir(s.id)}")
        except FileExistsError:
            print(f"scene {scene.id!r} exists (use --overwrite to replace it)")


def cmd_serve(args):
    import uvicorn
    os.environ["MISE_DATA"] = args.data
    from .backend.app import create_app
    uvicorn.run(create_app(Path(args.data)), host=args.host, port=args.port)


def cmd_mcp(args):
    from .mcp_server.server import main
    main(http=args.http, host=args.host, port=args.port)


def cmd_check(args):
    from .scene.checks import check_layout
    s = _store(args).load(args.scene)
    vs = check_layout(s, args.variant)
    for v in vs:
        print(f"[{v.severity}] {v.kind}: {v.message}")
    print(f"{sum(v.severity == 'hard' for v in vs)} hard, {sum(v.severity == 'soft' for v in vs)} soft")
    sys.exit(1 if any(v.severity == "hard" for v in vs) else 0)


def cmd_render(args):
    from .scene.checks import check_layout
    from .scene.render import render_view
    s = _store(args).load(args.scene)
    png, labels = render_view(s.room, s.resolve(args.variant), args.view, check_layout(s, args.variant), args.object)
    Path(args.out).write_bytes(png)
    print(json.dumps(labels))


def cmd_scale_report(args):
    """Inputs: COLMAP text model, detections JSON (from `mise detect-markers`) and held-out JSON:
    [{"name": "room length", "a": [x,y,z], "b": [x,y,z], "tape_m": 4.212}, ...]"""
    import numpy as np

    from .pipeline import scale
    held = [scale.HeldOut(**h) for h in json.loads(Path(args.held_out).read_text())] if args.held_out else []
    markers = []
    if args.colmap and args.detections:
        cams = scale.read_colmap_text(args.colmap)
        det = {f: {int(k): np.array(v) for k, v in d.items()} for f, d in json.loads(Path(args.detections).read_text()).items()}
        markers = scale.marker_scales(cams, det, args.side_mm / 1000)
    rep = scale.build_report(markers, held, scale_override=args.scale, source="markers" if markers else "override")
    rep.write(args.out)
    print(json.dumps({"scale": rep.scale, "spread_rel": rep.spread_rel, "verdict": rep.verdict, "held_out": rep.held_out}, indent=2))


def cmd_detect_markers(args):
    from .pipeline.markers import detect_markers
    imgs = sorted(Path(args.images).glob("*.[jp][pn]g"))
    det = detect_markers(imgs, args.dictionary)
    Path(args.out).write_text(json.dumps({f: {k: v.tolist() for k, v in d.items()} for f, d in det.items()}))
    print(f"markers found in {len(det)} of {len(imgs)} frames")


def cmd_normalize(args):
    from .pipeline import normalize, ply
    s = ply.read_ply(args.input)
    scale = args.scale
    if args.scale_report:
        scale = json.loads(Path(args.scale_report).read_text())["scale"]
    out, res = normalize.normalize(s, scale=scale, up_hint=tuple(args.up))
    ply.write_ply(args.output, out)
    print(json.dumps({"scale": res.scale, "floor_inliers": res.floor.inliers, "yaw_deg": res.yaw_deg, "matrix": res.matrix()}, indent=2))


def cmd_package(args):
    import numpy as np

    from .pipeline import decompose, ply
    from .scene.models import Opening
    s = ply.read_ply(args.splat)
    labels = np.load(args.labels)
    names = json.loads(Path(args.names).read_text()) if args.names else {}
    room, objs = decompose.split_instances(s, labels, background=set(args.background))
    boxes = {k: decompose.oriented_box(v) for k, v in objs.items()}
    room, _ = decompose.remove_with_shadow(room, list(boxes.values()))
    room = decompose.fill_floor_plane(room, [b.footprint(0.15) for b in boxes.values()])
    floor = decompose.floor_polygon_from_splat(room)
    openings = [Opening(**o) for o in json.loads(Path(args.openings).read_text())] if args.openings else []
    named = {}
    for lab, sp in objs.items():
        meta = names.get(str(lab), {})
        oid = meta.get("id", f"obj_{lab}")
        named[oid] = (sp, meta.get("name", f"Object {lab}"), meta.get("category", "furniture"))
    scene = decompose.package(Path(args.out), room, named, floor, args.ceiling, openings, args.id, args.name)
    print(f"packaged {len(scene.objects)} objects into {args.out}; copy the folder to {args.data}/{args.id} to serve it")


def cmd_import(args):
    from .products.importer import ImportRequest, NeedsInput, run_import
    st = _store(args)
    st.load(args.scene)
    gen = None
    if os.environ.get("FAL_KEY"):
        from .products.idol import IdolClient
        gen = IdolClient()
    req = ImportRequest(url=args.url, category=args.category, dims_override=args.dims, mesh=args.mesh)
    try:
        res = run_import(req, st.scene_dir(args.scene) / "assets", gen)
    except NeedsInput as e:
        print("needs input:", e.question)
        sys.exit(2)
    st.add_library_asset(args.scene, res.asset, actor="user")
    print(res.asset.model_dump_json(indent=2))


def cmd_stage_commands(args):
    from .pipeline import external
    cap = Path(args.capture)
    external.run(external.ns_process(cap, args.source))
    external.run(external.ns_train_splatfacto(cap))
    external.run(external.ns_export(cap / "outputs" / "<run>" / "config.yml", cap))
    print(external.NOTES)


def cmd_scan(args):
    from .pipeline.scan import STAGES, ScanEvents, ScanOptions, run_scan

    work = Path(args.work or Path(args.data).parent / "scans" / args.scene)
    labels = dict(STAGES)

    def show(ev):
        k = ev["kind"]
        if k == "stage":
            extra = {x: v for x, v in ev.items() if x not in ("seq", "time", "kind", "stage", "status")}
            print(f"[{ev['status']:>7}] {labels.get(ev['stage'], ev['stage'])} {json.dumps(extra) if extra else ''}", flush=True)
        elif k == "progress" and "registered" in ev:
            print(f"          solved {ev['registered']} of {ev['of']} frames", flush=True)
        elif k == "issue":
            span = f" {ev['t0']:.0f}-{ev['t1']:.0f} s" if ev.get("t0") is not None else ""
            print(f"  ! {ev['severity']} {ev['issue']}{span}: {ev['message']}", flush=True)
        elif k == "error":
            print(f"  x {ev['message']}", flush=True)

    work.mkdir(parents=True, exist_ok=True)
    # so the scan monitor (mise serve, /scan.html) can follow this run
    (work / "job.json").write_text(json.dumps({"id": f"cli_{work.name}"[:64], "scene_id": args.scene, "name": args.name,
                                               "created": time.time()}))
    opts = ScanOptions(scene_id=args.scene, name=args.name, fps=args.fps, scale=args.scale, t0=args.t0, t1=args.t1,
                       resume=args.resume, semantic=Path(args.semantic) if getattr(args, "semantic", None) else None)
    if args.poses:  # a solve made elsewhere (scripts/colab/learned_recon.py): import it, then carry on as --resume
        from .pipeline.scan import import_poses

        meta = import_poses(Path(args.poses), work, Path(args.frames) if args.frames else None)
        opts.resume = True
        if opts.scale is None and meta.get("metric"):
            opts.scale = 1.0
        print(f"imported {meta.get('frames', '?')} frames solved by {meta.get('model_id', 'an outside model')}"
              f" ({'metric, --scale 1.0' if opts.scale == 1.0 else 'scale from camera height' if opts.scale is None else f'--scale {opts.scale}'})",
              flush=True)
    res = run_scan(Path(args.video), work, opts, store=_store(args), ev=ScanEvents(work / "events.jsonl", show))
    print(json.dumps(res.to_dict(), indent=1, default=str))


def cmd_learned_scan(args):
    from .pipeline.learned_flow import solve_home

    work = Path(args.work or Path(args.data).parent / "scans" / args.scene)
    say = lambda m: print(f"[learned] {m}", flush=True)  # noqa: E731
    home = solve_home(Path(args.video), work / "learned-run", tag=args.scene, model=args.model, gpu=args.gpu, fps=args.fps,
                      variants=args.variants, say=say)
    if args.scale is None:  # the model's metres, checked against the doorways and the box
        from .pipeline.refscale import reference_scale

        box = json.loads(Path(args.box).read_text()) if args.box else None
        rep = reference_scale(home, box)
        (work / "learned-run" / "scale.json").write_text(json.dumps(rep, indent=1, default=float))
        say(f"scale x{rep['factor']} +-{rep['sigma_pct']}% from " + ", ".join(
            f"{k} x{v['factor']} +-{v['sigma_pct']}%" for k, v in rep["from"].items()))
        args.scale = rep["factor"]
    args.poses, args.frames, args.work = str(home), str(work / "learned-run" / "picked"), str(work)
    sem = work / "learned-run" / "learned" / "semantic"
    args.semantic = str(sem) if (sem / "labels.npz").exists() else None
    args.resume, args.t0, args.t1 = False, None, None
    cmd_scan(args)
    if args.splat:  # Stage 2: the splat, the empty home (materials, trim, windows, doors) and the objects
        from .pipeline.splat_flow import splat_home

        rep = splat_home(Path(args.data) / args.scene, work, home, Path(args.frames), work / "result.json",
                         semantic=sem if (sem / "labels.npz").exists() else None, gpu=args.splat_gpu, tag=args.scene,
                         say=lambda m: print(f"[splat] {m}", flush=True))
        say("splat: " + ", ".join(f"{w['room']} {w['psnr_held_out']} dB" for w in rep["windows"])
            + f"; {rep['objects']['objects']} objects")


def cmd_splat_attach(args):
    """Copy a splat (already in the scene's frame) into the scene and make it the room's splat."""
    import shutil

    from .scene.models import Scene

    d = Path(args.data) / args.scene
    scene = Scene.model_validate_json((d / "scene.json").read_text())
    rel = Path("splats") / (args.name or Path(args.splat).name)
    (d / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.splat, d / rel)
    scene.room.splat = rel.as_posix()
    (d / "scene.json").write_text(scene.model_dump_json(indent=2))
    print(f"{args.scene}: room splat {rel} ({(d / rel).stat().st_size / 1e6:.0f} MB)")


def cmd_empty_room(args):
    from .pipeline.empty_room import empty_scene
    from .scene.models import Scene

    d = Path(args.data) / args.scene
    scale = 1.0
    if args.scan_result:
        ref = next(pc for pc in json.loads(Path(args.scan_result).read_text())["pieces"] if pc.get("scene_from_solve"))
        scale = float(ref["scene_from_solve"]["scale"])
    rep = empty_scene(d, Path(args.splat_dir), args.name, Path(args.semantic) if args.semantic else None,
                      frames=Path(args.frames) if args.frames else None, depth=Path(args.depth) if args.depth else None,
                      depth_scale=scale)
    if args.use:
        scene = Scene.model_validate_json((d / "scene.json").read_text())
        scene.room.splat = rep["path"]
        (d / "scene.json").write_text(scene.model_dump_json(indent=2))
    print(json.dumps(rep, indent=1))


def cmd_refine_poses(args):
    """Bundle-adjust a learned solve's cameras (CPU) for splat training; depth maps carried along."""
    import shutil

    from .pipeline.learned import refine_poses

    solve, out = Path(args.solve), Path(args.out)
    rep = refine_poses(solve, Path(args.frames), Path(args.work or str(out) + "-work"), out)
    for f in ("depth.npz", "frames.json"):  # splat_train's depth loss; the launcher's frame list
        if (solve / f).exists() and not (out / f).exists():
            shutil.copy2(solve / f, out / f)
    print(json.dumps(rep, indent=1))


def cmd_splat_home(args):
    """The scene's splat and its empty room from a learned scan (paths as learned-scan lays them out)."""
    from .pipeline.splat_flow import splat_home

    work = Path(args.work or Path(args.data).parent / "scans" / args.scene)
    lr = work / "learned-run"
    sem = Path(args.semantic) if args.semantic else lr / "learned" / "semantic"
    rep = splat_home(Path(args.data) / args.scene, work, Path(args.solve or lr / "learned" / "home"),
                     Path(args.frames or lr / "picked"), Path(args.scan_result or work / "result.json"),
                     semantic=sem if (sem / "labels.npz").exists() else None,
                     refined=Path(args.refined) if args.refined else None, gpu=args.gpu, tag=args.scene,
                     say=lambda m: print(f"[splat] {m}", flush=True))
    print(json.dumps(rep, indent=1, default=float))


def cmd_shell_render(args):
    """The empty home again from the scene's finishes (after a repaint or a new floor)."""
    from .pipeline.materials import finish_summary, write_shell
    from .scene.models import Scene

    d = Path(args.data) / args.scene
    scene = Scene.model_validate_json((d / "scene.json").read_text())
    n = write_shell(scene, d, f"splats/{args.name}", lighting=not args.flat)
    print(json.dumps({"path": f"splats/{args.name}", "gaussians": n, "finishes": finish_summary(scene)}, indent=1))


def cmd_scene_splat(args):
    """One splat of a layout (the empty home and the objects where they stand), for renders."""
    from .pipeline import ply
    from .pipeline.objects import compose
    from .scene.models import Scene

    d = Path(args.data) / args.scene
    scene = Scene.model_validate_json((d / "scene.json").read_text())
    out = compose(scene, d, args.variant, fixtures=not args.no_fixtures)
    path = Path(args.out) if args.out else d / "splats" / f"layout-{args.variant}.ply"
    ply.write_ply(path, out)
    print(json.dumps({"path": str(path), "gaussians": len(out)}))


def cmd_scan_objects(args):
    """The things in a labelled splat as movable objects in the scene (fixtures marked)."""
    from .pipeline.objects import add_scanned_objects

    rep = add_scanned_objects(Path(args.data) / args.scene, Path(args.splat_dir))
    print(json.dumps(rep, indent=1, default=float))


def cmd_splat_merge(args):
    """Splats of overlapping stretches (all trained in the scene's frame) as one: each room from
    the stretch that stood in it most."""
    import numpy as np

    from .pipeline import ply
    from .pipeline.empty_room import merge_splats
    from .scene.models import Scene

    d = Path(args.data) / args.scene
    scene = Scene.model_validate_json((d / "scene.json").read_text())
    parts = [(ply.read_ply(Path(p) / "splat.ply"), json.loads((Path(p) / "cameras.json").read_text())) for p in args.parts]
    out, rep = merge_splats(parts, [np.array(sp.polygon, float) for sp in scene.room.spaces])
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    ply.write_ply(o / "splat.ply", out)
    (o / "cameras.json").write_text(json.dumps([c for _, cams in parts for c in cams]))
    print(json.dumps({**rep, "parts": args.parts, "out": str(o)}, indent=1))


def cmd_join_poses(args):
    from .pipeline.learned import join

    meta = join([Path(d) for d in args.solves], Path(args.out), square=args.square,
                onto=Path(args.onto) if args.onto else None)
    for r in meta["joins"]:
        print(json.dumps(r), flush=True)
    print(f"{meta['frames']} frames, {meta['points']} points -> {args.out}")


def cmd_calibrate(args):
    from .pipeline.learned import calibrate

    out = calibrate(Path(args.frames), Path(args.work or Path(args.frames) / "calibration"), args.t0, args.t1)
    print(json.dumps(out, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mise", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.environ.get("MISE_DATA", "data/scenes"), help="scene store directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("demo"); a.add_argument("--id", default="demo-bedroom"); a.add_argument("--overwrite", action="store_true"); a.set_defaults(fn=cmd_demo)
    a = sub.add_parser("serve"); a.add_argument("--host", default="127.0.0.1"); a.add_argument("--port", type=int, default=8765); a.set_defaults(fn=cmd_serve)
    a = sub.add_parser("mcp"); a.add_argument("--http", action="store_true"); a.add_argument("--host", default="127.0.0.1"); a.add_argument("--port", type=int, default=8766); a.set_defaults(fn=cmd_mcp)
    a = sub.add_parser("check"); a.add_argument("scene"); a.add_argument("--variant", default="main"); a.set_defaults(fn=cmd_check)
    a = sub.add_parser("render"); a.add_argument("scene"); a.add_argument("out"); a.add_argument("--variant", default="main")
    a.add_argument("--view", default="plan", choices=["plan", "eye", "object"]); a.add_argument("--object"); a.set_defaults(fn=cmd_render)

    a = sub.add_parser("detect-markers"); a.add_argument("images"); a.add_argument("out"); a.add_argument("--dictionary", default="DICT_4X4_50"); a.set_defaults(fn=cmd_detect_markers)
    a = sub.add_parser("scale-report"); a.add_argument("--colmap"); a.add_argument("--detections"); a.add_argument("--side-mm", type=float, default=150.0)
    a.add_argument("--held-out"); a.add_argument("--scale", type=float, help="skip markers (e.g. 1.0 for metric LiDAR/ARKit poses)")
    a.add_argument("--out", default="scale_report.json"); a.set_defaults(fn=cmd_scale_report)
    a = sub.add_parser("normalize"); a.add_argument("input"); a.add_argument("output"); a.add_argument("--scale", type=float, default=1.0)
    a.add_argument("--scale-report"); a.add_argument("--up", type=float, nargs=3, default=[0, 0, 1], help="rough up direction in the input frame"); a.set_defaults(fn=cmd_normalize)
    a = sub.add_parser("splat-attach", help="make a trained splat (in the scene's frame) the scene's room splat")
    a.add_argument("scene"); a.add_argument("splat"); a.add_argument("--name", help="file name in the scene's splats/")
    a.set_defaults(fn=cmd_splat_attach)
    a = sub.add_parser("splat-home", help="the scene's splat (room by room on Colab) and its empty room, from a learned scan")
    a.add_argument("scene")
    a.add_argument("--work", help="the scan's folder (default ../scans/SCENE, as learned-scan makes it)")
    a.add_argument("--solve"); a.add_argument("--frames"); a.add_argument("--scan-result")
    a.add_argument("--semantic"); a.add_argument("--refined", help="cameras already bundle-adjusted (refine-poses)")
    a.add_argument("--gpu", default="A100,G4,L4,H100")
    a.set_defaults(fn=cmd_splat_home)
    a = sub.add_parser("refine-poses", help="bundle-adjust a learned solve's cameras for splat training (CPU)")
    a.add_argument("solve"); a.add_argument("frames", help="the picked frames (with intrinsics.json)")
    a.add_argument("out"); a.add_argument("--work", help="scratch folder (default OUT-work)")
    a.set_defaults(fn=cmd_refine_poses)
    a = sub.add_parser("shell-render", help="the empty home from the scene's finishes (after changing a color or floor)")
    a.add_argument("scene"); a.add_argument("--name", default="empty.ply")
    a.add_argument("--flat", action="store_true", help="without the room's light (shading maps)")
    a.set_defaults(fn=cmd_shell_render)
    a = sub.add_parser("scene-splat", help="one splat of a layout: the empty home with its objects placed (for renders)")
    a.add_argument("scene"); a.add_argument("--variant", default="main"); a.add_argument("--out")
    a.add_argument("--no-fixtures", action="store_true", help="leave the built-ins out")
    a.set_defaults(fn=cmd_scene_splat)
    a = sub.add_parser("scan-objects", help="the things in a labelled splat as movable objects (after empty-room labels it)")
    a.add_argument("scene"); a.add_argument("splat_dir", help="splat.ply + gauss_labels.npz")
    a.set_defaults(fn=cmd_scan_objects)
    a = sub.add_parser("splat-merge", help="one splat from splats of overlapping stretches, room by room")
    a.add_argument("scene"); a.add_argument("out"); a.add_argument("parts", nargs="+", help="splat_train.py outputs")
    a.set_defaults(fn=cmd_splat_merge)
    a = sub.add_parser("empty-room", help="the scene's home without its things, from a labelled splat")
    a.add_argument("scene"); a.add_argument("splat_dir", help="splat_train.py output: splat.ply + gauss_labels.npz")
    a.add_argument("--name", default="empty.ply"); a.add_argument("--use", action="store_true", help="make it the room splat")
    a.add_argument("--semantic", help="frame labels (segment_frames.py): label the Gaussians here, with the trained cameras")
    a.add_argument("--frames", help="the picked frames: paint the layout's shell from them (with --depth)")
    a.add_argument("--depth", help="the solve's depth.npz (the model's depth maps)")
    a.add_argument("--scan-result", help="the scan's result.json: the depth maps' scale to the scene")
    a.set_defaults(fn=cmd_empty_room)
    a = sub.add_parser("package"); a.add_argument("splat"); a.add_argument("labels", help="labels.npy, one int per Gaussian")
    a.add_argument("out"); a.add_argument("--names", help="JSON {label: {id, name, category}}"); a.add_argument("--openings", help="JSON list of openings")
    a.add_argument("--background", type=int, nargs="*", default=[0]); a.add_argument("--ceiling", type=float, default=2.5)
    a.add_argument("--id", default="room"); a.add_argument("--name", default="Scanned room"); a.set_defaults(fn=cmd_package)
    a = sub.add_parser("import-product"); a.add_argument("scene"); a.add_argument("url"); a.add_argument("--category", default="furniture")
    a.add_argument("--dims"); a.add_argument("--mesh", action="store_true"); a.set_defaults(fn=cmd_import)
    a = sub.add_parser("stage-commands"); a.add_argument("capture"); a.add_argument("--source", default="video", choices=["video", "images", "record3d", "polycam"]); a.set_defaults(fn=cmd_stage_commands)

    a = sub.add_parser("scan"); a.add_argument("video"); a.add_argument("--scene", required=True); a.add_argument("--name", default="Scanned home")
    a.add_argument("--fps", type=float, default=3.0); a.add_argument("--scale", type=float, help="known scale (markers/LiDAR); default: from camera height")
    a.add_argument("--t0", type=float); a.add_argument("--t1", type=float); a.add_argument("--work", help="work folder (default: next to the scene store)")
    a.add_argument("--resume", action="store_true", help="reuse the frames and camera solve from the last run in the work folder")
    a.add_argument("--poses", help="use this camera solve instead of solving: a folder from scripts/colab/learned_recon.py "
                   "(frames.json, sparse/0, meta.json)")
    a.add_argument("--frames", help="with --poses: folder holding the frames it names (default: POSES/frames)")
    a.add_argument("--semantic", help="per-frame class labels (labels.npz + meta.json from segment_frames.py): walls "
                   "from wall/floor/ceiling/door points only, nothing carved through mirrors and windows")
    a.set_defaults(fn=cmd_scan)

    a = sub.add_parser("learned-scan", help="video -> scene, cameras solved by a learned model on Colab, room by room")
    a.add_argument("video"); a.add_argument("--scene", required=True); a.add_argument("--name", default="Scanned home")
    a.add_argument("--fps", type=float, default=3.0); a.add_argument("--scale", type=float, help="known scale; default: the model's, checked against doorways and --box")
    a.add_argument("--work", help="work folder (default: next to the scene store)")
    a.add_argument("--model", default="da3_nested", help="learned_recon.py model (default: Depth Anything 3, metric)")
    a.add_argument("--gpu", default="G4,H100", help="Colab GPUs to try, in order")
    a.add_argument("--variants", type=int, default=3, help="solves per window; the one the others agree with is kept")
    a.add_argument("--box", help="clicked rim corners of an object of known size (JSON: known_m, known_side, clicks)")
    a.add_argument("--splat", action="store_true",
                   help="then Stage 2 (splat-home): the home's splat, its empty shell and its objects")
    a.add_argument("--splat-gpu", default="A100,G4,L4,H100", help="Colab GPUs for the room splats")
    a.set_defaults(fn=cmd_learned_scan)

    a = sub.add_parser("join-poses", help="join learned solves of overlapping stretches of one video")
    a.add_argument("out"); a.add_argument("solves", nargs="+")
    a.add_argument("--square", action="store_true", help="also line up floors and wall directions with the first solve")
    a.add_argument("--onto", help="a solve of the whole walk: place each solve on it (rooms keep their own shape)")
    a.set_defaults(fn=cmd_join_poses)

    a = sub.add_parser("calibrate", help="camera intrinsics from a CPU solve of one stretch of picked frames")
    a.add_argument("frames", help="folder with frames/ and frames.json (picked by mise)")
    a.add_argument("--t0", type=float, default=0.0); a.add_argument("--t1", type=float, default=60.0)
    a.add_argument("--work", help="scratch folder (default: FRAMES/calibration)")
    a.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
