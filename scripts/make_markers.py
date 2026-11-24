"""Print 4 ArUco markers at 150 mm (check the printed size with calipers; printers rescale).

    pip install -e '.[markers]'
    python scripts/make_markers.py out_dir
"""
import sys
from pathlib import Path

from mise.pipeline.markers import make_marker_sheet

out = Path(sys.argv[1] if len(sys.argv) > 1 else "markers")
out.mkdir(parents=True, exist_ok=True)
for i in range(4):
    make_marker_sheet(out / f"marker_{i}.png", i, side_mm=150)
print(f"wrote 4 markers to {out}/ - print at 100% scale, 300 dpi")
