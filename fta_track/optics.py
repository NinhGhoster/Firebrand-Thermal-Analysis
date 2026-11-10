"""Real-world size from a thermal recording's own metadata.

FLIR software (Research Studio) measures length and area from the camera's field of
view and the object distance stored in the file: the footprint of one pixel on a plane
at distance D. This module does the same, for both apps (desktop and HPC), without
any GUI or SDK import.

    pixel size  s = 2 D tan(HFOV / 2) / W        (m per pixel, at distance D)
    area        A = n_px s^2
    length, width = the blob's longest and shortest extent in pixels, times s
    equivalent diameter  d = 2 sqrt(A / pi)

The FOV comes, in order of preference, from:
  1. the per-pixel field of view the file carries (FLIR SDK SourceInfo.ihfov, when
     fov_valid): s = D * IFOV;
  2. the lens recorded in the file (e.g. FOL21 = the 45 degree lens);
  3. the angle in the camera model name (FLIR writes "FLIR T1040 45__" for 45°).
The T1040 files in this project carry the lens and the distance but not the IFOV
(fov_valid is False), so route 2 applies.

Limits, to state with every size: the scale is exact only for objects at distance D (a
firebrand 0.5 m nearer to or farther from a 2 m plane is ~25 % off); motion blur and the
optics' blur make small hot objects look bigger, and an ember smaller than a pixel is
counted as a whole pixel.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Tuple

import numpy as np

# Horizontal field of view of FLIR T-series (T1020/T1030/T1040) lenses, degrees.
LENS_HFOV_DEG = {
    "FOL21": 45.0,   # 21.2 mm, the 45° lens (this project's)
    "FOL36": 28.0,   # 36 mm, 28°
    "FOL83": 12.0,   # 83.4 mm, 12°
}


def hfov_from_metadata(lens: Optional[str] = None, camera_model: Optional[str] = None) -> Tuple[Optional[float], str]:
    """(horizontal FOV in degrees, where it came from) from the lens or the camera model name."""
    if lens:
        key = str(lens).strip().upper()
        if key in LENS_HFOV_DEG:
            return LENS_HFOV_DEG[key], f"lens {key}"
    if camera_model:
        # FLIR replaces the degree sign with '__' or '_' in the model string
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|__|_|deg)", str(camera_model))
        if m:
            return float(m.group(1)), f"camera model '{camera_model}'"
    return None, "unknown"


def pixel_size_m(distance_m: Optional[float], width_px: int, hfov_deg: Optional[float] = None,
                 ifov_rad: Optional[float] = None) -> Optional[float]:
    """Metres covered by one pixel on a plane at distance_m; None if it cannot be known."""
    if distance_m is None or not (distance_m > 0) or width_px <= 0:
        return None
    if ifov_rad is not None and ifov_rad > 0:
        return float(distance_m) * float(ifov_rad)
    if hfov_deg is None or not (0 < hfov_deg < 180):
        return None
    return 2.0 * float(distance_m) * math.tan(math.radians(hfov_deg) / 2.0) / float(width_px)


def blob_extent_px(mask: np.ndarray) -> Tuple[float, float]:
    """(length, width) of a binary blob in pixels: the sides of its minimum-area rectangle,
    counting the pixels themselves (a single pixel is 1 x 1)."""
    import cv2

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0, 0.0
    if xs.size == 1:
        return 1.0, 1.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(max(w, h)) + 1.0, float(min(w, h)) + 1.0


def sizes_mm(area_px: float, length_px: float, width_px: float, s_m: Optional[float]) -> dict:
    """Real sizes of one blob; empty values when the scale is unknown."""
    if not s_m:
        return {"area_mm2": None, "length_mm": None, "width_mm": None, "equiv_diameter_mm": None}
    s_mm = s_m * 1000.0
    area = float(area_px) * s_mm * s_mm
    return {"area_mm2": area, "length_mm": float(length_px) * s_mm, "width_mm": float(width_px) * s_mm,
            "equiv_diameter_mm": 2.0 * math.sqrt(area / math.pi)}
