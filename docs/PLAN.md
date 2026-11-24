# 3D Room Redesign — Project Plan

Nov 24, 2026 · @Kyler

## Overview and goals

The system turns a phone scan into a to-scale 3D model of a room: an empty base room plus separate objects that you or an AI agent can move, remove and add to. Every added object is sized from its listed product dimensions and linked to where it can be bought.

| Goal | Done when |
| --- | --- |
| Separate room and objects | One scan yields an empty-room splat plus one splat per object, each with an oriented bounding box |
| True scale | Held-out tape measurements match the model within 1% (2% is the kill line) |
| Manual editing | In the browser you can orbit, walk, drag, rotate, delete and add objects, with snapping and collision warnings |
| Add real products | A product URL becomes a placed 3D object whose size matches the listing within 10% per axis |
| AI suggestions | A text request returns 2–3 layout variants plus a shopping list with prices and purchase links |

Out of scope for v1: relighting, multi-room scenes, phone AR and automated checkout.

## System architecture

Everything flows through one scene backend. The scan pipeline fills it once per room, and both you (in the editor) and the AI agent (through the MCP server) edit it.

&#91;embedded content: system architecture · 9 components\]

Product import fetches a listing, calls Idol and returns a sized asset through the MCP server, and the editor shows every AI change live.

## Stage 1 — Capture and metric scale

Scale comes from printed markers of known size, checked against tape measurements the calibration never sees. Target: within 1% on held-out distances.

**Why markers instead of a ruler.** A ruler works, but finding its ends is manual and its ticks are a few pixels wide in video frames. Printed ArUco or ChArUco markers (OpenCV) of known side length are detected automatically in every frame. Triangulating their corners with the solved camera poses gives one scale factor per marker; the spread across markers is your error bar. Keep the ruler in shot as a manual fallback.

1. Print 4 markers at a known side length (150 mm works) and check the printed size with calipers, since printers rescale. Put 2 on the floor and 2 on different walls so they are not coplanar.
2. Tape-measure 3 held-out distances: room length, room width and one doorway. These are never used for calibration.
3. Capture with lights on and blinds fixed: phone in landscape, slow walk, 3 passes at different heights. A capture study found landscape, slow movement and 3+ height layers worked best ([Journal of Imaging 2025](https://www.preprints.org/manuscript/202412.1959/v1)).
4. Solve poses. With a LiDAR iPhone, Polycam or Record3D captures feed Nerfstudio directly without COLMAP, and their ARKit poses are already metric ([Nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html)); the markers become a cross-check. Otherwise run COLMAP and let the markers set scale.
5. Normalize: apply the scale, fit the floor plane (RANSAC on Gaussian centers) to make z up with the floor at z = 0, and align walls to the x and y axes. Nerfstudio rescales poses during training, so confirm the exported splat is back in the original metric frame.
6. Verify: measure the 3 held-out distances in the model and report percent error.
7. Crop the markers out of the room splat.

Optional: a second clean-plate capture in the same session (furniture pushed to one half, then the other) gives ground truth for evaluating Stage 2.

Deliverables: `room_raw.ply` in meters with z up, and `scale_report.json` holding per-marker scale factors, their spread and the held-out errors.

## Stage 2 — Reconstruction and decomposition

One furnished scan becomes an empty room plus one complete splat per object. Splitting the scan is close to solved; filling in what the camera never saw is the main risk.

1. **Reconstruct.** Train the room splat with Nerfstudio's Splatfacto on one 3090. The repo you found wraps exactly this pipeline ([3d\_gaussian\_splatting\_for\_model\_city](https://github.com/rakeshsuthar6322/3d_gaussian_splatting_for_model_city)).
2. **Segment.** Give every Gaussian an instance label. Start with Gaussian Grouping ([arXiv 2312.00732](https://arxiv.org/abs/2312.00732)), whose code is released. Try InstaScene ([arXiv 2507.08416](https://arxiv.org/abs/2507.08416)), built to pick arbitrary objects out of cluttered splat scenes. Neither handles transparent or highly reflective objects, so budget manual cleanup for mirrors and glass.
3. **Complete objects.** Run each object's best unoccluded frame through SAM 3D Objects ([arXiv 2511.16624](https://arxiv.org/abs/2511.16624)), which is trained for occlusion and clutter. Align the result to the observed Gaussians and keep observed Gaussians wherever they exist. InstaScene's in-situ generation is the alternative.
4. **Complete the room.** Remove objects with masks dilated to cover their baked contact shadows, then inpaint. Candidates: Inpaint360GS ([arXiv 2511.06457](https://arxiv.org/abs/2511.06457)) for multi-object 360° scenes, and a plane-fit plus texture-copy baseline for floors and walls. A 2026 benchmark found re-initializing from inpainted views beats fine-tuning the existing splat ([arXiv 2605.30987](https://arxiv.org/abs/2605.30987)).
5. **Extract structure.** Fit floor, wall and ceiling planes, and mark doors and windows as polygons (by hand in v1). These drive snapping and layout rules later.
6. **Package.** Save each object as its own splat in a canonical frame (centered, z up, footprint on z = 0) with its oriented bounding box and its pose in the room.

All-in-one alternative: DP-Recon ([arXiv 2503.14830](https://arxiv.org/abs/2503.14830)) reconstructs objects and background together with diffusion guidance, but it uses a neural SDF rather than splats and is slow.

Deliverables: `room_empty.ply`, `objects/<id>.ply` with metadata, and `layout.json` holding planes, doors and windows.

## Stage 3 — Scene model and interactive editor

One scene file is the single source of truth. The browser editor and the MCP server both read and change it, so your edits and the AI's edits land in the same place.

**Scene file (`scene.json`), units in meters:**

| Field | Holds |
| --- | --- |
| `room` | Empty-room splat path, floor polygon, wall planes, door and window polygons, ceiling height |
| `objects[].asset` | Splat or mesh path in its canonical frame |
| `objects[].pose` | x, y, z and yaw; z comes from what the object rests on |
| `objects[].dims` | Width, depth and height of the oriented bounding box |
| `objects[].support` | Floor, a wall, or another object's id (a lamp on a nightstand) |
| `objects[].source` | Scanned, product or user upload |
| `objects[].product` | Retailer, URL, price, currency, listed dimensions, date fetched |
| `objects[].locked` | Whether the AI may move it |
| `variants[]` | Named layouts: poses plus added and hidden objects |

**Editor (web app).** Use three.js with Spark, World Labs' MIT-licensed splat renderer, which sorts several splat objects together correctly and mixes splats with meshes ([Spark](https://github.com/sparkjsdev/spark)). PlayCanvas, the engine behind SuperSplat, is the fallback.

- Views: orbit, first-person walk at eye height, and a top-down plan with labels and dimensions.
- Select: raycast against each object's bounding box, not its Gaussians, which is cheaper and more predictable.
- Move: drag on the floor plane, rotate around z, snap yaw to 15° steps and backs to walls, keep objects on their support.
- Check: live footprint overlap (oriented boxes on the floor plus height), walkway clearance and blocked doors, shown as warnings rather than hard blocks.
- Add: from your object library, a product URL (Stage 4), or an uploaded photo plus typed dimensions.
- History: every change is an operation in a log, giving undo and redo and letting you accept or reject an AI proposal change by change.

**Sync.** A small backend (FastAPI or Node) owns `scene.json` and pushes changes to the editor over a WebSocket. The MCP server calls the same backend, so the editor shows AI moves live.

## Stage 4 — Adding objects at true scale

A product becomes a placed object by fitting one uniform scale factor to its listed dimensions. The mismatch on the other axes then flags bad generations instead of hiding them.

Idol needs this step: base Idol returns a centered Gaussian-splat `.ply` with no stated up axis or units ([Idol API docs](https://www.pixazo.ai/models/idol.md)). Its splats carry only base color (no higher-order spherical harmonics), so they will look flatter than the room under shiny materials.

1. **Pull the listing.** From the product page, extract name, price, the raw dimension string and 1–4 images. Store the raw string and URL so every number can be audited.
2. **Parse dimensions.** Convert to meters and map to width, depth and height; retailers order and label W × D × H differently. Reject parses outside a plausible range for the category (a sofa under 0.5 m or over 1.2 m tall) and ask.
3. **Generate 3D.** Send the cleanest image to Idol, ideally a three-quarter view on a plain background. Set `image_type` to `real` for photos or `synthetic` for renders, since many product shots are renders. Idol Max is the mesh option: $0.20 per generation, with `model_glb` upright in +Y-up coordinates ([Idol Max API](https://fal.ai/models/speridlabs/idol/max/llms.txt)).
4. **Orient.** Set up as the axis normal to the object's base (PCA on Gaussian centers plus a gravity check), and front as the side facing the input camera. When that is ambiguous, show a VLM two renders and ask.
5. **Fit scale.** For each axis, ratio = listed ÷ measured. Use the median ratio as one uniform scale. Anisotropy = largest ratio ÷ smallest ratio.
6. **Gate.** Anisotropy up to 1.10: accept. From 1.10 to 1.25: accept with a warning badge in the editor. Above 1.25: retry with another image, or fall back to a labeled box of the right size.
7. **Match color (speculative).** Shift each new object's white balance and brightness toward the room's, sampled from the floor near its placement. This does not fix light direction.

Don't stretch each axis to hit the listing exactly: that hides a bad generation and distorts the shape, while the gate surfaces it.

## Stage 5 — MCP server

The MCP server is a thin, stateless layer over the scene backend. It gives the AI eyes (labeled renders), hands (pose edits in a sandboxed variant) and a referee (layout checks).

| Tool | Does | Returns |
| --- | --- | --- |
| `get_room` | Room size, floor polygon, walls, doors, windows, free floor area | JSON |
| `list_objects` | Every object in a variant with dims, pose, support and lock flag | JSON |
| `render_view` | Plan, eye-level or object close-up render with numbered labels | Image plus label map |
| `check_layout` | Collisions, walkway clearance, blocked doors and windows, out-of-room | Violations with distances in m |
| `create_variant` | Copies a layout so the AI never edits yours directly | Variant id |
| `move_object` | Sets x, y and yaw; snaps to support | New pose plus violations caused |
| `place_relative` | "Against wall 2", "facing object 4", "0.1 m left of bed" | Resolved pose |
| `hide_object` / `show_object` | Removes or restores an existing item within a variant | Status |
| `search_products` | Retailer search filtered by max footprint and budget | Products with dims, price, URL |
| `import_product` | Runs Stage 4 on a product URL as an async job | Asset id, dims, anisotropy, warnings |
| `add_object` | Places an imported asset by pose or relation | Object id plus violations |
| `diff_variants` | Change list between two layouts | Moves, adds, removals, cost delta |

Design rules:

- **Every call names its scene and variant.** MCP's 2026-07-28 revision removed protocol-level sessions from Streamable HTTP, so all state lives in the backend ([MCP transports](https://modelcontextprotocol.io/specification/latest/basic/transports/streamable-http)).
- **Renders are tool results.** MCP tool results can carry images as well as text ([MCP tools](https://modelcontextprotocol.io/specification/latest/server/tools)). Numbered labels let the model refer to objects by number, the visual marking LayoutVLM uses for spatial grounding.
- **Errors are numeric and specific** ("overlaps bed by 0.12 m") so the agent can fix them itself.
- **The AI writes only to variants.** You promote changes into your main layout from the editor, item by item.
- **Web research stays outside the server.** The agent uses its own web search and fetch; `search_products` only wraps retailer APIs where they exist.

Build it with the official Python SDK (`mcp` on PyPI), which speaks stdio and Streamable HTTP ([MCP Python SDK](https://pypi.org/pypi/mcp)). Use stdio with Claude Desktop or Claude Code on the same machine; use Streamable HTTP if the server runs elsewhere.

## Stage 6 — AI design agent

The agent turns a text request into 2–3 layout variants and a shopping list. It uses web research for style and products, and the MCP tools for everything spatial. Layout quality comes from splitting the work: the model proposes poses and relations, and a solver enforces physics.

1. **Brief.** Turn the request into a structured brief: style words, budget, must-keep items, what the room is for, hard constraints. Show it back in one line.
2. **Style card.** Search the web for the style and write a card: palette, materials, signature furniture types, density and things to avoid, with source links. You confirm or edit it, a cheap checkpoint before any shopping.
3. **Gap analysis.** Compare the current room (`list_objects` plus `render_view`) with the card: what to keep, move, hide and add.
4. **Source products.** For each item to add, compute the largest footprint that fits, then search retailers within budget. Drop listings without dimensions. Prefer official APIs; Amazon's PA-API 5 is deprecated in favor of the Creators API ([Amazon notice](https://affiliate-program.amazon.com/creatorsapi/docs/en-us/paapiv5-deprecation)).
5. **Import.** Run `import_product` on each shortlisted item (Stage 4) and drop any that fail the anisotropy gate.
6. **Lay out.** Propose poses plus spatial relations, then let the solver resolve collisions and clearances. This follows Holodeck ([arXiv 2312.09067](https://arxiv.org/abs/2312.09067)), where an LLM writes relational constraints that an optimizer satisfies. It also follows LayoutVLM ([arXiv 2412.02193](https://arxiv.org/abs/2412.02193)), which jointly optimizes VLM-proposed poses with physics objectives.
7. **Critique.** Render plan and eye-level views, check them against the brief and the card, and revise. Up to 3 rounds per variant.
8. **Present.** Each variant gets renders, a change list and a shopping list. Re-check price and stock just before showing it.

Shopping list columns: item, retailer, price, listed W × D × H, link and date checked, plus a variant total.

Default layout rules, all editable: main walkways at least 0.9 m wide, door swings clear, nothing tall in front of windows, and locked items never move.

Run it as Claude in Claude Desktop or Claude Code with the MCP server attached; it already has web search and fetch.

## Hypotheses, evaluation and kill criteria

Each stage has a pre-registered prediction, a paired test and a switch rule. Thresholds are proposed values: fix them before each test runs, then score the prediction against the result.

| ID | Prediction | Paired test | Kill or switch if |
| --- | --- | --- | --- |
| H1 Scale | Marker calibration within 1% on 3 held-out tape distances | Same room solved with COLMAP + markers and with LiDAR poses; both checked against the tape | Error over 2%: add markers, use LiDAR poses, check for pose drift |
| H2 Empty room | Learned inpainting beats plane-fit plus texture on masked LPIPS | Furnished scan vs clean-plate scan of the same room in one session; score inside the dilated removal mask on held-out views | No win: ship the plane baseline |
| H3 Object backs | SAM 3D completions within 5% of the object's diagonal (Chamfer) | 3 objects completed from the room scan vs the same objects scanned alone, all the way around | Worse: scan the pieces you will move on their own |
| H4 Product scale | At least 80% of 20 products pass the 1.10 anisotropy gate | Includes 3 products you own, measured with a tape | Under 60%: switch to Idol Max meshes or require confirmed dimensions |
| H5 Layout solver | Agent + solver has zero hard violations and wins at least 70% of blind comparisons | Same brief and products with and without the solver, shown side by side in random order | Not preferred: keep the solver only as a checker |
| H6 Shopping list | At least 90% of links stay live with matching dimensions 7 days later | Re-fetch every listed product a week after generation | Under 80%: restrict to API-backed retailers |

Method notes:

- **You are designer and judge.** Blind H5: hide which variant used the solver and randomize left and right.
- **Winner's curse on prompts.** Write 15 briefs and hold out 5 that you never tune prompts against; report H5 on those 5.
- **n = 1 room is not a result.** Run H2 and H3 on at least two rooms (a bedroom and a living room) before trusting them.

## Risks and open questions

The biggest risk is visual, not technical: added objects carry their product-photo lighting and can look pasted in.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Lighting mismatch | Idol objects look pasted into the baked room lighting | Color matching (speculative); fall back to meshes in a lit engine such as Blender |
| Moving scanned furniture | Shading stays baked from its original spot | Accept for v1; flag strongly moved items in the editor |
| Mirrors, glass, screens | Segmentation and reconstruction fail | Manual cleanup, or replace with a sized proxy |
| Retailer access | Scraping can break terms of service, and pages change | Official APIs first; snapshot pages; re-check before showing |
| Idol license | Output license not stated on the pages read | Confirm terms before any non-personal use |
| Privacy | Scans show the inside of your home | Keep scans local; send only product images to Idol |
| Price and stock drift | Shopping lists go stale | Re-check at presentation; show the check date |

Open questions:

- [ ] Which phone will you scan with? A LiDAR iPhone gives metric poses for free in Stage 1.
- [ ] Which room goes first?
- [ ] Default budget for an AI redesign request?
- [ ] Project name (none chosen yet).

## References

| Source | Kind | Used in |
| --- | --- | --- |
| [Impact of capture methods on 3DGS](https://www.preprints.org/manuscript/202412.1959/v1) (Journal of Imaging 2025) | Paper | Stage 1 capture protocol |
| [Nerfstudio custom data](https://docs.nerf.studio/quickstart/custom_dataset.html) | Docs | Stage 1 poses, Stage 2 training |
| [3d\_gaussian\_splatting\_for\_model\_city](https://github.com/rakeshsuthar6322/3d_gaussian_splatting_for_model_city) | Repo | Stage 2 reference pipeline |
| [Gaussian Grouping, arXiv 2312.00732](https://arxiv.org/abs/2312.00732) | Paper + code | Stage 2 segmentation and removal |
| [InstaScene, arXiv 2507.08416](https://arxiv.org/abs/2507.08416) | Paper | Stage 2 decomposition and completion |
| [SAM 3D, arXiv 2511.16624](https://arxiv.org/abs/2511.16624) | Paper + model | Stage 2 object completion |
| [Inpaint360GS, arXiv 2511.06457](https://arxiv.org/abs/2511.06457) | Paper | Stage 2 room inpainting |
| [3DGS inpainting benchmark, arXiv 2605.30987](https://arxiv.org/abs/2605.30987) | Paper | Stage 2 re-init vs fine-tune |
| [DP-Recon, arXiv 2503.14830](https://arxiv.org/abs/2503.14830) | Paper | Stage 2 all-in-one alternative |
| [Spark](https://github.com/sparkjsdev/spark) | Library | Stage 3 editor renderer |
| [SuperSplat scene management](https://developer.playcanvas.com/user-manual/supersplat/editor/scene-management/) | Tool | Stage 3 fallback, manual cleanup |
| [Idol API](https://www.pixazo.ai/models/idol.md) | API docs | Stage 4 splat output |
| [Idol Max API](https://fal.ai/models/speridlabs/idol/max/llms.txt) | API docs | Stage 4 mesh output, pricing |
| [MCP tools spec](https://modelcontextprotocol.io/specification/latest/server/tools) | Spec | Stage 5 image results |
| [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/latest/basic/transports/streamable-http) | Spec | Stage 5 stateless design |
| [MCP Python SDK](https://pypi.org/pypi/mcp) | Library | Stage 5 server |
| [Holodeck, arXiv 2312.09067](https://arxiv.org/abs/2312.09067) | Paper | Stage 6 constraints + optimizer |
| [LayoutVLM, arXiv 2412.02193](https://arxiv.org/abs/2412.02193) | Paper | Stage 6 layout, visual marking |
| [Amazon PA-API 5 deprecation](https://affiliate-program.amazon.com/creatorsapi/docs/en-us/paapiv5-deprecation) | Notice | Stage 6 product sourcing |
