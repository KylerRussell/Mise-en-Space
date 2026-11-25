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
| | MapAnything, one pass | not yet | | | needs a GPU with 80 GB or more (G4, H100): the 40 GB A100 ran out of memory. Colab had none free for over an hour on 2026-11-25 |

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

## Next steps

1. Whole tour as one piece: `python scripts/colab/run_learned_recon.py tour1 --chunk 900 --gpu G4,H100`
   (one pass, no joins). On a 40 GB GPU, `tour2 --chunk 480 --overlap 80` puts the one join
   in the bedroom (133-160 s), where MapAnything is solid, not in the closets. Score each room
   with `mise.eval.floorplan`, then build the home with `mise scan --poses`.
2. `segment_home` on learned points: keep closets (nooks behind narrow openings) out of the
   room, and re-tune `push_furniture_edges` and `counter_layer` for dense, noisier points.
   Score the rooms it outputs against `truth.json`, not just the solve.
3. Tighter walls: keep only higher-confidence pixels (`--points-per-frame`, the percentile in
   `ColmapWriter.add`), or MapAnything's multi-view confidence (`use_multiview_confidence`).
4. Scale: MapAnything reads 2-6% small across the floor while its heights are about right.
   Cross-check with the blue bin (0.305 m) in the main room before trusting it to 1%.
5. Rebuild the bedroom, closets and bathroom of `my-apartment` from the learned solve and the
   plan instead of the description.

## Smaller open items

- Windows aren't in the apartment scene yet: the main room's north wall and the bedroom.
- Lights in the bedroom and bathroom; shelves and rods in the closets.
- Detecting appliances (only counters are detected automatically).
- `segment_home` over-splits open-plan rooms at narrow spots.
- The live coach needs https to use a phone camera. Not yet tried on a real phone.
