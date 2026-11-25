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
| Spaces, doors, built-ins (fixtures), empty-home toggle | `scene/models.py`, `scene/checks.py`, `editor/` (`?fixtures=0`) |

Run `python -m pytest -q` (90 tests). Build the editor with `cd editor && npm run build`.
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
- **Bedroom, closets and bathroom are from the owner's description, not the solve.** The owner's
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

## Next experiment: a learned reconstruction model on Colab

**Goal.** Replace or back up the COLMAP solve with a feed-forward or learned-SLAM model:
VGGT, MapAnything, MASt3R-SLAM, or whatever is current and best for indoor video. Such models
estimate depth from each frame, so turning on the spot and plain walls should hurt much less,
and some estimate metric scale. Check what has been released before choosing.

**Where it runs.** The owner has Colab (GPU). The cloud container here has no GPU. Write a
notebook (for example `notebooks/learned_reconstruction.ipynb`) that the owner opens from
GitHub in Colab. It should read the video from their Drive and write small outputs back to
Drive. The video never goes anywhere else.

**Output contract** (so the existing pipeline can use it unchanged):

1. A COLMAP-format model (`cameras`, `images`, `points3D`; text or binary) for the frames, named
   `frame_0000.jpg`... Each frame's video time goes in `frames.json` (`[{"name", "t"}]`).
2. If the model gives metric scale, say so (the scan then uses `--scale 1.0`).
3. Optionally, per-frame depth maps for denser points. Downsample them before saving.

Then add an option to `mise scan` that imports poses instead of solving: frames + model in
the work folder, then the same steps as `--resume`.

**How to judge it:**

- **Bedroom stretch (134-182 s), where COLMAP collapsed.** Do the cameras stay at 1-1.8 m
  height (`pose_sanity`)? Does a bedroom outline come out of `segment_home`? The owner's
  description says the bedroom is about 2.7 x 4.3 m.
- **Main room (0-130 s).** Align it to `colmap4/incr/0` with `align_by_time` (the owner can
  re-run the old solve, or compare against the scene's main-room outline). Then check the
  scale: the blue bin's short side is 0.305 m.
- **Whole tour.** Does it come out as one piece? How long does it take on a Colab GPU?

## Smaller open items

- Windows aren't in the apartment scene yet: the main room's north wall and the bedroom.
- Lights in the bedroom and bathroom; shelves and rods in the closets.
- Detecting appliances (only counters are detected automatically).
- `segment_home` over-splits open-plan rooms at narrow spots.
- The live coach needs https to use a phone camera. Not yet tried on a real phone.
