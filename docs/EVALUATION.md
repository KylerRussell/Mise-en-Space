# Running the hypotheses

Thresholds live in `src/mise/eval/hypotheses.py` (`THRESHOLDS`). Fix them in a commit
**before** each test runs, then score the prediction against the result.

| ID | How to run it | Helper |
| --- | --- | --- |
| H1 Scale | Same room, solved twice: COLMAP + markers, and LiDAR poses. `mise scale-report` for each, with the same 3 tape distances (length, width, one doorway) that calibration never sees. | `h1(scale_report)` |
| H2 Empty room | Furnished scan plus a clean-plate scan in one session. Render held-out views of (a) learned inpainting and (b) `decompose.fill_floor_plane`, then score masked LPIPS inside the dilated removal mask against the clean plate. | `masked_lpips`, `h2` |
| H3 Object backs | 3 objects completed from the room scan vs the same objects scanned alone all the way around. Align them first (ICP), then compute Chamfer as a fraction of the diagonal. | `chamfer`, `h3` |
| H4 Product scale | 20 products, including 3 you own and have measured with a tape. Record each import's anisotropy. | `h4` |
| H5 Layout solver | 15 briefs; hold out 5 that you never tune prompts against. For each, one variant with `solve_layout` and one with poses only. `BlindStudy.build` randomizes sides; judge from `judge_sheet()` only; score on the held-out 5. | `BlindStudy` |
| H6 Shopping list | 7 days after generation, `shopping.recheck(list, fetch_listing)`. | `h6` |

n = 1 room is not a result: run H2 and H3 on a bedroom and a living room.
