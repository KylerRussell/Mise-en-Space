# Mise-en-Space

A phone scan becomes a to-scale 3D model of a room: an empty base room plus separate
objects that you, or an AI agent, can move, remove and add to. Every added object is
sized from its listed product dimensions and linked to where it can be bought.

The full plan, with goals, stages, hypotheses and risks, is in [docs/PLAN.md](docs/PLAN.md).

```
Phone capture ─▶ Reconstruct + scale ─▶ Decompose ─┐
                                                    ▼
             Web editor ◀──────────────▶ Scene backend ◀──▶ MCP server ◀──▶ Product import ─▶ Idol
                 ▲                      (scene.json, ops,        ▲
                 │                       variants, WebSocket)    │
                You ─────────────────────────────────────▶ AI design agent (Claude + web)
```

## Quick start

```bash
pip install -e '.[dev]'
```

```bash
(cd editor && npm install && npm run build)
```

```bash
mise demo
```

```bash
mise serve
```

Open http://127.0.0.1:8765. The demo bedroom uses labeled boxes in place of scanned splats,
so everything works before you have a scan.

To let Claude edit the room, keep `mise serve` running and open this repo in Claude Code:
`.mcp.json` registers the MCP server (`mise mcp`, stdio) and `/redesign-room` runs the
Stage 6 workflow. For Claude Desktop, add the same `mcpServers` entry to its config.
Use `mise mcp --http --port 8766` to serve Streamable HTTP instead.

For editor development with hot reload, run `npm run dev` in `editor/` (it proxies
`/api`, `/files` and `/ws` to the backend on port 8765).

## What is in the box

| Stage | Code | Status |
| --- | --- | --- |
| 1 Capture and metric scale | `pipeline/scale.py` (triangulate markers, per-marker scale, held-out report), `pipeline/markers.py` (ArUco detection, printable sheets), `pipeline/normalize.py` (RANSAC floor, z up, Manhattan wall alignment) | Implemented and tested on synthetic data; needs a real capture |
| 2 Reconstruction and decomposition | `pipeline/external.py` (Nerfstudio commands), `pipeline/decompose.py` (split labeled splat, OBBs, canonical frames, shadow-dilated removal, plane + texture-copy floor fill, `scene.json` packaging) | Packaging and the H2 baseline arm are implemented. Gaussian Grouping, SAM 3D and Inpaint360GS run as external tools |
| 3 Scene model and editor | `scene/` (models, geometry, checks, snapping, op log with undo/redo, variants, diff/promote), `backend/app.py` (FastAPI + WebSocket), `editor/` (three.js + Spark) | Working: orbit / walk / plan, drag, rotate, snap, live warnings, add, delete, history, per-change accept |
| 4 Adding objects at true scale | `products/` (listing extraction, dimension parser, Idol client, orientation, uniform scale fit + anisotropy gate, box fallback, color match) | Working end to end with a stand-in generator; the Idol endpoint details still need checking (see below) |
| 5 MCP server | `mcp_server/server.py`: every tool in the plan's table, plus `solve_layout`, `shopping_list`, `list_variants`, `get_job`, `undo` | Working over stdio and in-process |
| 6 AI design agent | `.claude/skills/redesign-room/SKILL.md`, `agent/solver.py` (relations + physics, simulated annealing + walkway repair), `agent/shopping.py` | Working; layout quality is what H5 will measure |
| Evaluation | `eval/hypotheses.py`: H1 to H6 scoring, Chamfer, masked LPIPS, blind A/B study builder | Helpers ready; experiments not run |

## Scene conventions

Meters, z up, floor at z = 0. Each object's canonical frame is centered on x/y with its
footprint on z = 0 and **its front facing local +y**; `pose.yaw` rotates it counter-clockwise.
`support` is `floor`, `wall:<id>` or another object's id, and dependents move with their support.

Objects can also be **tipped** in 90° steps: `pose.roll` turns an object about its own front
axis (roll 90 stands a wide, low cube shelf on its end) and `pose.pitch` about its width axis.
`dims` is the object as modeled; `extent()` is its size as placed, and every check, snap and
render uses the extent. Only categories that truly work one way up default to
`upright_only` (sofas, beds, chairs, tables, desks, lamps, wardrobes...; see
`UPRIGHT_CATEGORIES` in `scene/models.py`). Shelves, cube storage and unknown categories
may tip, and you can override any object with the editor's "Upright only" box. The AI can
tip objects in its variants but can't change that setting.

**Homes with several rooms.** A scene can hold `spaces`: named parts of the home
(living, kitchen, bedroom, bathroom, hallway...), each with its own outline and optional
ceiling height. Every edge of a space is a wall unless it borders a space it is open to,
so an open-plan living room and kitchen are two zones of one floor with no wall between.
Walls are derived from the spaces, doors join rooms for the walkway check, and checks
flag furniture that cuts through an interior wall. In the editor's Spaces panel you can
rename and retype a space, split an open area with a zone line (click the plan), merge
zones back and delete spaces; each edit is undoable. The AI can read spaces and scope
`list_objects`, `render_view`, `check_layout` and `solve_layout` to one of them, but only
you can change them. `mise demo` also creates a small demo apartment.

**What the empty home looks like.** Each space can carry a `finish`: a color and optional
repeating texture (with its real size in meters) for its floor, walls and ceiling. The
editor draws solid walls with door and window openings cut through, open door leaves,
baseboards, and ceilings when you walk around; in orbit view the walls nearest you fade.
`mise.pipeline.finishes` samples finishes from the capture: it resamples a metric patch
of floor or wall straight from a posed frame (lens distortion included), removes the
lighting falloff, and makes it tile. Set finishes with `PATCH /api/scenes/{id}/spaces/{space}`
(`{"finish": {...}}`); texture paths are relative to the scene's folder.

The AI (header `X-Actor: agent`) may write only to variants and never moves locked objects.
You promote its changes into `main` from the editor, one change at a time, and every
change is an undoable operation in `ops.jsonl`.

## Scan to scene (once you have a capture)

```bash
mise stage-commands path/to/capture --source record3d
```

```bash
mise scale-report --colmap sparse/0 --detections det.json --side-mm 150 --held-out tape.json
```

```bash
mise normalize exports/splat.ply room_raw.ply --scale-report scale_report.json
```

```bash
mise package room_raw.ply labels.npy out/ --names names.json --openings openings.json --id bedroom
```

With LiDAR poses (Polycam or Record3D), pass `--scale 1.0` and use the markers only as a
cross-check. `labels.npy` holds one instance id per Gaussian from Gaussian Grouping or
InstaScene. Doors and windows are entered by hand in v1 (`openings.json`). Copy `out/` to
`data/scenes/bedroom` to serve it.

## Product import (Stage 4)

```bash
export FAL_KEY=...
```

```bash
mise import-product demo-bedroom https://example.com/some-sofa --category sofa
```

The importer reads the page (schema.org JSON-LD first, then meta tags, then text) and
parses the dimension string. It rejects implausible sizes and asks instead. It then
generates 3D, finds up and front, fits **one** uniform scale (the median of the per-axis
ratios) and gates on anisotropy: up to 1.10 accept, up to 1.25 accept with a warning
badge, otherwise retry with another image and finally fall back to a labeled box at the
listed size. It never stretches each axis to match.

It only guesses up and down for upright-only objects. For anything that may tip, it keeps
the generator's axes and tries every 90° arrangement against the listing instead. A shelf
listed lying down but generated standing up is matched by tipping it, not rejected.
Pass `upright_only` to `import_product` to override the category default.

**Idol API details are not verified.** The docs hosts were unreachable from the build
environment, so `products/idol.py` uses fal's generic queue API, and the model ids and
the image field name are settings (`IDOL_MODEL`, `IDOL_MAX_MODEL`, `IDOL_IMAGE_FIELD`).
The response is searched for the first `.ply`/`.glb` URL. Check these against
[the Idol Max docs](https://fal.ai/models/speridlabs/idol/max/llms.txt) before first use.
Also confirm Idol's output license before any non-personal use.

## Tests

```bash
python -m pytest -q
```

## Open questions (from the plan)

- Which phone? A LiDAR iPhone gives metric poses for free in Stage 1.
- Which room goes first?
- Default budget for an AI redesign request? The agent asks until one is set.
