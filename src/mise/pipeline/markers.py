"""ArUco marker detection (needs `opencv-contrib-python`, the `markers` extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detect_markers(image_paths: list[Path], dictionary: str = "DICT_4X4_50") -> dict[str, dict[int, np.ndarray]]:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("install the markers extra: pip install -e '.[markers]'") from e
    aruco = cv2.aruco
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), aruco.DetectorParameters())
    out: dict[str, dict[int, np.ndarray]] = {}
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        frame = {}
        for c, i in zip(corners, ids.ravel()):
            c = c.reshape(4, 1, 2).astype(np.float32)
            cv2.cornerSubPix(img, c, (5, 5), (-1, -1), crit)
            frame[int(i)] = c.reshape(4, 2)
        out[Path(p).name] = frame
    return out


def make_marker_sheet(out_png: Path, marker_id: int, side_mm: float = 150, dpi: int = 300,
                      dictionary: str = "DICT_4X4_50") -> None:
    """Render one marker at its true printed size (check the print with calipers)."""
    import cv2

    aruco = cv2.aruco
    px = int(round(side_mm / 25.4 * dpi))
    img = aruco.generateImageMarker(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), marker_id, px)
    margin = int(0.15 * px)
    img = cv2.copyMakeBorder(img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(img, f"id {marker_id}  side {side_mm:g} mm", (margin, margin // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.imwrite(str(out_png), img)
