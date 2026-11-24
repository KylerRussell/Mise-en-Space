"""`mise` command line.

  mise demo                         create the demo bedroom scene
  mise serve [--port 8765]          run the scene backend (+ built editor)
  mise mcp [--http]                 run the MCP server
  mise check SCENE [--variant V]    print layout violations
  mise render SCENE out.png [--view plan|eye]
  mise scale-report ...             Stage 1: marker scale + held-out errors
  mise normalize in.ply out.ply     Stage 1: metric, z up, floor at 0, walls on axes
  mise package ...                  Stage 2: split labeled splat into room + objects + scene.json
  mise import-product SCENE URL     Stage 4: product page -> sized library asset
  mise stage-commands CAPTURE_DIR   print the Nerfstudio commands for a capture
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _store(args):
    from .scene.store import SceneStore
    return SceneStore(Path(args.data))


def cmd_demo(args):
    from .demo import demo_scene
    st = _store(args)
    s = st.create(demo_scene(args.id), overwrite=args.overwrite)
    print(f"created scene {s.id!r} in {st.scene_dir(s.id)}")


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
    a = sub.add_parser("package"); a.add_argument("splat"); a.add_argument("labels", help="labels.npy, one int per Gaussian")
    a.add_argument("out"); a.add_argument("--names", help="JSON {label: {id, name, category}}"); a.add_argument("--openings", help="JSON list of openings")
    a.add_argument("--background", type=int, nargs="*", default=[0]); a.add_argument("--ceiling", type=float, default=2.5)
    a.add_argument("--id", default="room"); a.add_argument("--name", default="Scanned room"); a.set_defaults(fn=cmd_package)
    a = sub.add_parser("import-product"); a.add_argument("scene"); a.add_argument("url"); a.add_argument("--category", default="furniture")
    a.add_argument("--dims"); a.add_argument("--mesh", action="store_true"); a.set_defaults(fn=cmd_import)
    a = sub.add_parser("stage-commands"); a.add_argument("capture"); a.add_argument("--source", default="video", choices=["video", "images", "record3d", "polycam"]); a.set_defaults(fn=cmd_stage_commands)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
