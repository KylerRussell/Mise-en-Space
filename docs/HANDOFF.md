# Handoff: where the project stands

Read this first in a new session. The code is on branch `claude/dazzling-cerf-s0haef`, and
`README.md` explains how to use it. This file holds what the code doesn't: decisions made with
the owner, what the first real scan taught us, data that lives outside the repo, and the next
experiment.

## Working rules (from the owner)

- **Scans stay local.** Videos, frames, solves and textures sampled from the video are never
  committed or sent to outside services. Only product images go to Idol (Stage 4).
- The first capture is `Room_Tour.mp4` (Android phone, about 290 s, one walkthrough of a
  one-bedroom apartment). It is in the owner's Google Drive. Ask for it; don't commit it.
- The apartment scene built so far was handed to the owner as `my-apartment-scene.zip`. To
  restore it, unzip it into the scene store (`data/scenes/my-apartment/`, holding `scene.json`,
  `ops.jsonl` and `finishes/`). Don't commit it either.
- Don't open pull requests unless asked. Commit and push to the branch.
- **Never use the GPUs on the owner's machine** (two RTX 3090s, busy with other work). GPU
  work goes to Colab. CPU work there runs pinned off the GPUs' NUMA node:
  `taskset -c $(seq 0 63 | awk '$1%4!=1' | paste -sd,)` (node 1 = CPUs 1, 5, 9, ... feeds the GPUs).
- **Colab is driven from the machine by the `colab` CLI** (already signed in; `colab skill`
  prints its guide). It is a paid account: all GPU types, several sessions at once. Another
  project's agent uses the same account: only touch `mise-*` sessions. Sending the picked
  frames of a stretch to the owner's own Colab VM is fine (it was the plan); nothing else
  leaves the machine.
- **The listing floor plan is the ground truth** for sizes: `data/floorplan/` (plan image and
  `truth.json`). Local only, like the video: the repo is public. Report errors as % or cm.
- On the owner's machine the video is `~/Downloads/Room_Tour.mp4` and the tour's picked frames
  (3 fps, 1280 px, 877 frames) are in `data/scans/tour-frames/`. The project venv is `.venv`.
- **The product is one coached walkthrough.** One video (or one live walk in the web app, which
  says where to linger because the reconstruction is unsure there). **Never markers** on walls.
- **Scale reference: the blue bin.** Its horizontal short side (the face with the logo) is
  exactly 1 ft. The plan is only the accuracy check. Rim clicks: `data/floorplan/box_clicks.json`
  (local), scored with `mise.eval.floorplan --box`.
- Any mix of tools is fine; benchmark models against the plan rather than settling on one.

## What exists (all tested)

| Area | Code |
| --- | --- |
| Camera solve on CPU (pycolmap), leveling, scale | `pipeline/video.py`, `pipeline/sfm_scene.py` (`level_reconstruction`, `align_by_time`, `pose_sanity`) |
| Rooms, doorways, walls traced along furniture fixed | `pipeline/room_outline.py` (`segment_home`, `push_furniture_edges`) |
| Floor and wall finishes sampled from frames | `pipeline/finishes.py` (`auto_finishes`) |
| Footage checks (blur, plain view, lost track, turning on the spot) | `pipeline/capture_qc.py` |
| One-command scan with progress events | `pipeline/scan.py`, `mise scan VIDEO --scene ID [--t0 --t1 --resume --scale]` |
| Scan jobs API, live events, live coach | `backend/scans.py` (`/api/scans`, `/ws/scans/{id}`, `/api/coach/{session}`) |
| Scan monitor page | `editor/scan.html`, `editor/src/scan.js` |
| Learned camera solve on Colab (MapAnything / VGGT), pose import | `scripts/colab/`, `pipeline/scan.py` (`import_poses`), `mise scan --poses` |
| Scoring a solve against a floor plan | `eval/floorplan.py` (`python -m mise.eval.floorplan`) |
| Spaces, doors, built-ins (fixtures), empty-home toggle | `scene/models.py`, `scene/checks.py`, `editor/` (`?fixtures=0`) |

Run `python -m pytest -q` (94 tests). Build the editor with `cd editor && npm run build`.
`mise serve` serves the editor at `/` and the monitor at `/scan.html`.

## What the first real scan taught us

- **COLMAP needs sideways movement.** The main room solved well: 3 fps frames,
  contrast-boosted (CLAHE), sequential plus "revisit" matching. The bedroom (about 134-182 s)
  and the walk-in closet (about 182-232 s) collapsed: the phone mostly turned on the spot, in
  front of plain painted-block walls. The bathroom (about 236-292 s) solved only in fragments.
  Growing or joining pieces didn't rescue them.
- **Scale.** The true scale came from a blue bin whose short side is exactly 1 ft (main-room
  model `colmap4/incr/0`, scale 0.6154). Without a reference, scale comes from the median phone
  height (1.43 m over this tour). On the 0-45 s test that was 7% off: the true median there
  was 1.52 m. The ceiling is about 2.47 m and the counters are 0.91 m.
- **Bedroom, closets and bathroom are from the owner's description, not the solve**, and the
  description's sizes were off: the plan's bedroom is larger and squarer than "about
  2.7 x 4.3 m" (see `data/floorplan/truth.json`). Rebuild those rooms from the learned solve
  or the plan. The owner's
  words: the bedroom runs from the main window to the door; the walk-in closet runs from that
  door to the main wall and has a doorway, not a door; the closet is as long as the bedroom
  door; the connecting closet sits between bathroom and bedroom; the bathroom's back wall lines
  up with the main room's south wall. The wall left of the front door (seen from inside,
  looking out) is flat.
- The kitchen's south wall was moved from y = -2.69 to -3.02 because counter and peninsula
  points reach there. The bathroom's back wall moved with it.
- The bathroom fixtures (tub, toilet, vanity, mirror, light) were placed by eye and carry a
  "check the position" warning. The kitchen fixtures and the ceiling fan were measured from
  solved frames.
- The automatic scan of 0-45 s: 122 of 135 frames solved, 109 in one piece, in 4.5 minutes on
  CPU. It split the open main room into two rooms with a doorway that doesn't exist. That is a
  known weakness of `segment_home`.
- The footage checks flag the bad stretches, but also some good ones. Treat them as coaching.
  The timeline's "mapped" row (did the frame actually solve?) is the reliable signal.

## Learned reconstruction (done 2026-11-25)

**What runs.** `scripts/colab/run_learned_recon.py NAME --t0 --t1 [--model mapanything|vggt]`
uploads the stretch's picked frames to a Colab GPU, runs `scripts/colab/learned_recon.py`
(MapAnything by default: metric scale), downloads a COLMAP model with points sampled from each
frame's depth, and stops the VM. `mise scan VIDEO --scene ID --poses data/scans/learned/NAME
--frames data/scans/tour-frames` builds the scene from it. `python -m mise.eval.floorplan
data/floorplan/truth.json --room bedroom|main LABEL=PATH ...` scores any solve against the plan.
Outputs are in `data/scans/learned/` (bedroom, main, tour, vggt-bedroom, vggt-main) and the
COLMAP baselines in `data/scans/colmap-{bedroom,main}`.

**Results against the floor plan** ("scale" = stretch needed to match the plan; "walls" =
median distance of wall points from the plan's walls at the best scale):

| Stretch | Solver | Frames placed | Scale | Walls | Notes |
| --- | --- | --- | --- | --- | --- |
| Bedroom 134-182 s | COLMAP (CPU, 4 min) | 37 / 144 | (x0.54) | - | 4 pieces; no room outline: the scan fails |
| | MapAnything (L4) | 144 / 144 | x1.065 | 7.0 cm | spans -6% and -4%; phone 1.41 m; ceiling 2.49 m |
| | VGGT (L4, camera-height scale) | 144 / 144 | x1.059 | 8.5 cm | ceiling 2.37 m |
| Main room 0-130 s | COLMAP (CPU, 14.6 min) | 274 / 390 | x0.955 | 11.8 cm | 6 pieces; split in two at a doorway that isn't there; 80% of the area |
| | MapAnything (G4, 3 chunks) | 390 / 390 | x1.024 | 15.3 cm | east-west -0.8%, living room -0.9%; ceiling 2.62 m (+6%) |
| | VGGT (L4, 6 chunks) | 390 / 390 | x0.946 | 17.2 cm | one weak chunk join (5.8 deg) |
| Whole tour | MapAnything, 5 chunks (G4) | 877 / 877 | - | - | 5.3 min in all, but the closet and bathroom chunks joined badly (16.6 deg, 35 cm): bent, fails the height check |
| | MapAnything, 2 chunks joined in the bedroom (G4, 186 s on the GPU) | 877 / 877 | x1.062 (main room) | 16 cm (main), 26 cm (bedroom) | one piece, phone heights sane in every stretch (1.35-1.50 m), but the join is off by 3.3 deg and 30 cm: the bedroom sits wrong against the main room (east-west reads 5.29 m) |
| | MapAnything, one pass (G4, 142 s, 76.5 GB peak) | 877 / 877 | - | - | bent: phone height reads 1.58 m in the main room, 1.39 m in the bedroom, and spreads 0.5-0.6 m in the closets and bathroom; fails the height check |

- **MapAnything is the pick.** It solves every frame, including the turning-on-the-spot
  stretches COLMAP loses, and its metric scale lands within 1-6% of the plan with no reference.
  Heights agree (phone 1.41-1.42 m against the 1.43 m prior; ceiling 2.49 m in the bedroom).
  VGGT is close in the bedroom but has no metric scale, and its chunks join less well.
- **Walls are fuzzier than COLMAP's** (7-15 cm against 12 cm, but COLMAP keeps only the walls
  it can solve). In the main room the wall band also holds the couch, counters and peninsula,
  and the kitchen's north wall reads 0.34 m short, about the depth of the upper cabinets.
- **Building the scene from the learned poses works** (`--poses`): the main room comes out as
  one L-shaped room with 8 corners, like the plan, area +3.5%, and no false doorway. The bedroom
  room is too big (+25%): `segment_home` swallows the closet nook and the walk-in passage
  that open off it, and `push_furniture_edges` moved four walls out 0.30-0.48 m. Counters were
  not found in the learned points.
- **Colab:** GPUs are often refused (503 "Service Unavailable", on every type at once at
  times); the launcher cycles through a list and waits (`--wait`). Two GPU sessions at once
  worked (`mise-main` on G4 with `mise-bedroom` on L4, 13.5 units/h together). The whole day's
  runs used about 1 compute unit. A MapAnything pass takes about 80 s to load, then ~25 s per
  200 frames on a G4 (48 GB peak) or ~45 s per 144 frames on an L4 (18 GB).
- Listing plans can be off by a few % themselves: treat differences under ~3% as noise.

## Whole home from one video (done 2026-11-26)

`mise learned-scan VIDEO --scene ID` runs the lot: frames, a rough Depth Anything 3 pass over
the whole walk, one window per room stay (`learned.plan_windows` on rooms traced in the rough
pass), all windows at once on Colab, chained (`join-poses`), then the scan.

Models on the first tour (`scripts/colab/learned_recon.py --model ...`, MapAnything's wrappers),
walls off the plan at best scale:

| model | bedroom alone | main room alone | whole walk in one pass |
|---|---|---|---|
| da3_nested (Depth Anything 3, metric) | 4.0 cm | 10.1 cm | rooms placed right, small rooms bent |
| pi3 | 3.7 cm | 16 cm | bedroom 24% long |
| pi3x (metric-ish) | 4.8 cm | | |
| must3r | 5.2 cm | | |
| mapanything (+ calibrated K) | 20-22 cm | 11-15 cm | bends |

VGGT-Omega's weights are gated (HF access request needed). pi3x and MUSt3R weights are
non-commercial.

Depth Anything 3 rooms chained through ~10 s overlaps (joins 2.5-6 cm): every frame placed,
walls 8.4 cm off the plan, metric scale within 1.3% of the plan. Traced rooms (IoU with the plan):
bedroom 0.97, main 0.91, bathroom 0.82, walk-in 0.76, bedroom closet 0.60, entry closet missed.
Spans within ~1-4% except the main room's 4.88 m (+5.5%). The fully automatic run: main 0.91,
bedroom 0.84, walk-in 0.83, bathroom 0.68. A single window across closets + bathroom
(182-293 s) works as well as two (bathroom 0.89).

What didn't work: joining room windows onto the rough pass (`--onto`: it is bent in the small
rooms and drags them), fixed 70 s windows (a window straddling main room and bedroom turned
8 deg; the chain bent), MapAnything room windows (14 cm joins).

The bin: Depth Anything 3's poses jitter around it (rim reprojection 55 px; COLMAP 5.6 px), so
its scale from the bin is +-9%; COLMAP's bin scale carried over through the cameras says 8-12%
bigger than the plan. The model's own metric scale was the better scale on this tour. To make
the bin precise, the coached walk should circle it for a few seconds.

## Repeatable solves and scale without the plan (done 2026-11-26)

Two runs of Depth Anything 3 over nearly the same stretch don't give the same geometry (scale
wanders ~4%, now and then a room folds). So `learned-scan` solves each window three times over
stretches grown a few seconds at one end or the other (`learned.variant_bounds`) and keeps the
medoid: the solve whose cameras the others agree with best after a similarity fit
(`learned.consensus`; `learned-run/learned/consensus.json`). Giving the model the calibrated
intrinsics (now done for da3/da3_nested/pi3x) changed nothing: its focal lengths were already
within 1-5%.

Scale (`pipeline/refscale.py`, `learned-run/scale.json`): inverse-variance mean of the model's
metres (+-4%), the doorway heads (2.06 m openings, +-2% from the median of 3-5 doors) and the
box (+-5-9%). On four joined homes it landed within -1.1..+0.7% of the plan (the model alone:
-1.3..+3.2%). Ceiling height is no use (2.37-2.76 m across solves of the same rooms).

The automatic run with both (data/scans/home-v3, layout rebuilt as home-v3b): joins 2.9 and
2.5 cm, scale x1.034 (the plan asks x1.032; the scene needs x0.999), walls 5.5 cm off the plan.
Spans of the main room +1.2/+1.8/+1.4%, bedroom -0.7/+2.1%. IoU: main 0.96, bedroom 0.96,
bathroom 0.91, walk-in 0.90, bedroom closet 0.34, entry closet missed. The main room's 4.88 m
span still reads -6% (+5.5% in another run): that corner is the least stable part of the solve.
Layout fixes that got there: a closet's wall is the outermost solid layer of points (hanging
clothes in front are denser), and a space no camera stood in and no doorway opens onto is
dropped (the bathroom mirror's reflection). The scorer now scores a scan folder at the scale
the scan used (it had used the camera-height guess, so the scene's IoUs were off).

The straightening pass (`room_outline.regularize_walls`: shared walls one thickness apart,
wall lines shared) moves IoUs by <=0.02; squaring rooms onto one grid (`--square`) gains nothing.

## Layout from the home's shell, not its contents (done 2026-11-27)

Clothes, beds, shelves and mirrors were behind most of the layout errors. Every picked frame is
now labelled per pixel (`scripts/colab/segment_frames.py`: Mask2Former Swin-L, ADE20K, MIT
licence; 877 frames in a minute on an L4; `run_learned_recon.py NAME --segment`), and each
point of a learned solve takes its pixel's class (`pipeline/structure.py`). With
`mise scan --semantic DIR` (learned-scan runs the labelling alongside the windows):
- free space is carved by every sight line except those through mirrors, windows and screens
  (`SEE_THROUGH`; each mirror's labelled patches are filled to their hull, as what shows in a
  mirror is labelled as what it reflects);
- walls come from wall/floor/ceiling/door points only (`STRUCTURE`), falling back to every point
  (outermost layer) where the shell is hidden, as in a closet full of clothes.
A space no camera stood in that shows almost no floor (under 1% of the walked rooms' floor
points per m2) is dropped: the bathroom mirror's reflection over the vanity showed none, a
closet peeked into shows some. Carving only from structure points was worse: a sight line to a
shirt still proves the space in front of it is empty.

First tour, automatic solve with labels (data/scans/home-v3c): main 0.96, bedroom 0.96,
bathroom 0.91, walk-in 0.89, bedroom closet 0.33 (mirrored doors; on solve dc4 the mirror's
reflection happened to carve the closet and scored 0.63), entry closet missed; 5 rooms,
4 doorways, no phantom; scene scale x0.999 of the plan, walls 5.5 cm off.

## Stage 2: splats and the empty home (2026-11-27)

One command after the learned scan: `mise --data DATA splat-home SCENE` (paths as `learned-scan`
lays them out; `--solve/--frames/--scan-result/--semantic/--refined` override). It:

1. Bundle-adjusts the learned cameras on SIFT (CPU, `mise refine-poses`; `learned.refine_poses`).
2. Splits the tour into room visits (cameras placed in the scene's rooms; glances back through a
   door folded in) and trains one splat per visit on Colab, side by side, each window running
   5 s into the rooms either side (`splat_flow.room_visits`).
3. Merges them: each room from the window whose cameras stood in it, a room nobody stood in
   (a closet looked into) from the window that came closest (`empty_room.merge_splats`).
4. Writes `splats/home.ply` (the home as filmed) and `splats/empty.ply` (the empty home, with
   materials, trim, windows and doors), adds the objects, and makes the empty home the scene's
   room splat (the editor shows the shell with the objects in it; nothing is shown twice).

`mise learned-scan VIDEO --scene S --box CLICKS --splat` runs it at the end of Stage 1: video to
the finished scene in one command (Stage 1's Colab solves are reused when the work folder has
them). `mise scene-splat S [--variant V] [--no-fixtures]` writes one splat of a layout (shell +
objects at their poses, `objects.compose`) for renders outside the editor.

What it took to get sharp splats (held-out PSNR; renders saved beside the photos):

| | whole tour | room windows |
|---|---|---|
| model cameras, 30k steps | 17-22 | 20-24 |
| one splat, 90k steps, 1.1M Gaussians | 17.7 (train 19.4) | |
| bundle-adjusted cameras + depth loss + camera corrections | | **25.8-27.7** |

A whole-tour splat can't even fit its own frames however long it trains; a 40 s stretch scores
30. The depth loss (rendered depth pulled toward the model's depth maps, confident pixels, weight
0.1) is what stops floaters in the plain small rooms (closets/bath 20.2 -> 23.8, with camera
corrections 25.4). Per-frame color correction cost 1-3 dB and is off. Weak spots: what is only
seen through an opening or from a doorway is smeared, and the main room's 435 frames is about as
many as one window handles well.

The empty home is the layout's shell painted from the photos (`shell.py`), not the splat with
things cut out: a splat's plain walls are 20-30 cm thick and tangled with what stands against
them, so no cut at the wall separates the two (tried: `empty_room.empty_room`, kept as the
fallback). The shell is flat Gaussians every 2.5 cm on each wall, floor and ceiling of the
traced rooms; each point takes the robust mean color of the frames that see it as that kind of
surface (labels eroded at outlines, the model's depth agreeing within ~6%, near square-on views
counting most). Where the labelled splat has things in front of a wall or above a floor, the
point counts as unseen and is filled: floors with the nearest same-size patch of seen floor,
then a broad blur; walls with a broad blur of the wall around; ceilings one smooth color.
Doorways stay open. On the home: walls 59% seen, 28% behind things; floors 54% / 40%.

The shell is then boiled down to materials (`materials.py`), so it can be changed: each room's
floor, walls and ceiling get a `Surface` in `space.finish` with a `kind` (paint, wood, tile,
carpet, block, popcorn...), a color, and for wood the boards' direction and width (`direction_deg`,
`module_m`); a wall that differs goes in `finish.wall_overrides`. The color is the surface's
robust median over what was seen clean, times one exposure gain for the whole home (phone video
exposes a white room to mid-grey; the brightest walls and ceilings set it, 1.67 here). The pattern
is read off a close, square-on view of a clear patch (2.5 mm/px): structure-tensor grain
direction and repeats; board direction snapped to the room's walls, width kept to 12-25 cm,
boards 1.2 m long. Rooms whose floors (walls, ceilings) are the same kind and hue share one color
(the best-seen room's); a surface seen too little borrows (a ceiling from the others, a closet's
floor from the nearest room). The light is kept apart as shading maps (`finishes/shading.npz`,
the photo-painted shell over its material's color, blurred 0.3 m), so `splats/empty.ply` =
material x shading looks as captured, and `mise shell-render SCENE` rebuilds it from the scene
in about a second after a repaint or a new floor (`--flat` without the light). The photo-painted
version stays as `splats/empty_photo.ply`. On the home: one laminate floor (wood, boards along
y, #b6aea3), one wall paint (#c2bdb7; the walk-in's walls warmer), one ceiling. Painted block is read
off up to twelve patches per wall: horizontal joints repeating at one 8 in course (0.19-0.225 m;
the row profile is the median across the row of the vertical gradient, so an object doesn't
count). Most views of a wall are too far, blurred or half hidden to show joints, and averaging
them drowns the few that do, so a wall is block when its clearest patch repeats at >= 0.40 and a
second agrees at >= 0.24 (`BLOCK_BEST`, `BLOCK_SECOND`, set on this home's walls in two runs:
block 0.41-0.51 / 0.25-0.46, drywall at most 0.39 alone or 0.30 / 0.22). Finds the three
exterior block walls (main room behind the desk, walk-in, bedroom) in both runs, drawn at 1.25
cm so the joints show, meeting the drywall at the corners as in the photos. The desk wall is the
marginal one (second patch 0.25 in the second run): it is mostly behind the desk, screens and
curtains.

Trim, windows and doors (`trim.py`): the frames' labels vote on every wall point; where most
views see window pane or blind it is window, door (or a mirrored closet door reaching the floor)
a closed door (`Opening.leaf`), each colored as captured (`Opening.color`) and added to the
scene (`win_*`, `leaf_*`; doorways between rooms are now cut through both faces of the wall).
Baseboard height and trim color come off the walls' bottom rows (10 cm, grey here; `finish.trim`,
`finish.baseboard_m`). The render paints baseboards and casings (7 cm, sides and head; windows
all round) onto the walls and sets door leaves and glass (a daylight tint, not the curtains)
slightly into the wall. Found: the window behind the desk and three closed doors including the
entry. Missed: the bedroom window (its curtains were drawn; curtains are things, not window).
Not modelled: the pipe along the block wall, crown moulding, the bulkhead.

The things come back as movable objects (`objects.py`, `mise scan-objects`, run by `splat-home`):
the owner's call (2026-11-27) is that furniture is removed and re-added as separate movable 3D
objects, so its reconstruction only has to be recognizable; what matters is each one's kind,
footprint, height and heading. Labelled Gaussians are grouped by kind (a bed with its pillows,
a desk, screens, storage, chairs...), split into instances by connected footprint within each
room (never across a wall), boxed (yaw only), and written as `objects/<id>.ply` in their own
frame. What lies on or in them (clothes on a bed) goes with them; what starts below 45 cm stands
on the floor (thin legs are seldom labelled); a screen on a desk is supported by the desk.
Kitchen and bath pieces are fixtures, named by their main class (a sink or counter in a room
with a toilet is a vanity); cabinets beside the kitchen join it. Clutter (clothes, bottles,
towels, books) and curtains don't become objects. On the home: 27 objects, 7 of them fixtures;
the bed, desk with its screens, sofa, chairs, kitchen run and island, toilet and vanity all
where they are in the photos. Smaller mislabels remain (built-in closet shelving as "Cabinets").
Each object's splat keeps what is inside its box but isn't shell (unlabelled legs and edges),
without faint or large soft Gaussians (the room's haze); floor-standing objects are cut at the
floor. The boxes are rough, so `objects.settle` nudges movable ones apart and back inside the
floor (the smaller of two moves; built-ins stay where they were filmed, so a fixture past a short
outline stays a warning), and an object on a wall names the wall (`wall:<id>`).

Known limits of the shell: fixtures go with the furniture (kitchen cabinets, counters); the main
room's bulkhead isn't in the layout; patches of floor differ in shade where copied; where the
outline stands short of the wall (the main room's 4.88 m span) that wall is mostly fill.

The Colab launcher sends inputs in 40 MB pieces (one region refused ~90 MB in one request), each
retried, and can render splats from given cameras (`--render A.ply B.ply --cameras cameras.json`,
`render_splat.py`).

## Next steps

1. Stage 1 loose ends: the main room's 4.88 m span (-6%: the outline stops at the desk), the
   closets peeked into from outside (bedroom closet 0.34, entry closet missed), doorway widths
   0.7 m for ~0.8 m. The shell inherits these.
2. Fixtures show in the empty home only as scanned objects (`fixture: true`); the shell itself
   has no kitchen. The bulkhead belongs in the layout.
3. Objects: swap scanned objects for catalog products of the same kind and size; check
   headings (front faces away from the wall it stands against). Splat sharpness of furniture
   doesn't matter (it is re-added as objects); the shell's does.
4. Coaching: circle the box and look up at each doorway head for a second (both feed the scale).

## To resolve later: the empty home (noted 2026-11-27)

- The bedroom window is missed: its curtains were drawn all through the walk, and curtains are
  things, not window evidence. Add it by hand, or coach "open the curtains for a second".
- Not modelled: the pipe along the main room's block wall, crown moulding, the bulkhead.
- A change of material part-way along one wall (drywall to block) isn't modelled; materials are
  per wall. Every transition in this home falls at a corner, so it hasn't mattered yet.
- Door detection can take an open leaf standing against a wall for a closed door (the bedroom
  has a borderline one beside a doorway, `leaf_room_4.6_*`).
- Block walls are found by joint spacing on close patches; brick (7.6 cm courses) isn't looked
  for. Materials are otherwise named by hand-set rules; an image model could name them.
- Under furniture the floor is filled from the open floor around it, so it can be a shade off.
- Window glass is a daylight tint, not what is outside.

## Smaller open items

- Windows in `my-apartment` (the older scene): the main room's north wall and the bedroom.
- Lights in the bedroom and bathroom; shelves and rods in the closets.
- Detecting appliances (only counters are detected automatically).
- `segment_home` over-splits open-plan rooms at narrow spots.
- The live coach needs https to use a phone camera. Not yet tried on a real phone.
