"""Command lines for the GPU tools the pipeline wraps (run them on the 3090 box).

These print or run the commands; they don't reimplement the tools.
Install each tool per its own README. Paths are relative to a capture dir:

  capture/
    video.mp4 | record3d/ | polycam/
    nerfstudio/          <- ns-process-data output
    outputs/             <- ns-train output
    exports/splat.ply    <- ns-export output (room_raw before normalization)
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def ns_process(capture: Path, source: str = "video") -> list[str]:
    """source: video | images | record3d | polycam (LiDAR captures skip COLMAP, poses are metric)."""
    data = {"video": capture / "video.mp4", "images": capture / "images", "record3d": capture / "record3d",
            "polycam": capture / "polycam"}[source]
    return ["ns-process-data", source, "--data", str(data), "--output-dir", str(capture / "nerfstudio")]


def ns_train_splatfacto(capture: Path) -> list[str]:
    return ["ns-train", "splatfacto", "--data", str(capture / "nerfstudio"), "--output-dir", str(capture / "outputs")]


def ns_export(config_yml: Path, capture: Path) -> list[str]:
    return ["ns-export", "gaussian-splat", "--load-config", str(config_yml), "--output-dir", str(capture / "exports")]


def run(cmd: list[str], dry: bool = True) -> int:
    print("$", shlex.join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


NOTES = """
Stage 2 tools (not wrapped; follow each README, then feed outputs to `mise package`):
  * Gaussian Grouping (arXiv 2312.00732): per-Gaussian identity encodings -> instance labels.
    Export one integer label per Gaussian, aligned with the splat's vertex order, as labels.npy.
  * InstaScene (arXiv 2507.08416): alternative decomposition + in-situ generation.
  * SAM 3D Objects (arXiv 2511.16624): complete each object from its best unoccluded frame,
    then align to the observed Gaussians (ICP) and keep observed Gaussians where they exist.
  * Inpaint360GS (arXiv 2511.06457): learned room inpainting. Compare against
    `fill_floor_plane` (the plane baseline) for H2 on held-out views.
Nerfstudio rescales poses during training: after export, check the splat is back in the
metric frame (the scale report's held-out errors will show it if not).
"""
