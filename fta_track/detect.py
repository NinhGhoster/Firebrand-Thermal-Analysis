"""Per-frame blob detection in degrees Celsius.

Ported from `detect_firebrands()` in FirebrandThermalAnalysis.py (absolute
threshold, 8-connected components, statistics on the component mask), with:
  - the persistent-hot mask removed before labelling,
  - a temperature-weighted centroid (sub-pixel) for tracking,
  - a local background from a ring around the blob, and the integrated excess
    temperature sum(T - background), which stays meaningful for embers smaller
    than a pixel whose peak reads low.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def detect_blobs(frame: np.ndarray, hot: np.ndarray, persistent: np.ndarray,
                 min_area: int, max_area: int, ring_px: int) -> List[Dict]:
    cand = (hot & ~persistent).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    H, W = frame.shape
    out: List[Dict] = []
    for lid in range(1, n):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x0 = int(stats[lid, cv2.CC_STAT_LEFT])
        y0 = int(stats[lid, cv2.CC_STAT_TOP])
        w = int(stats[lid, cv2.CC_STAT_WIDTH])
        h = int(stats[lid, cv2.CC_STAT_HEIGHT])

        # expanded box: blob plus a ring of ring_px for the local background
        ex0, ey0 = max(0, x0 - ring_px), max(0, y0 - ring_px)
        ex1, ey1 = min(W, x0 + w + ring_px), min(H, y0 + h + ring_px)
        sub = frame[ey0:ey1, ex0:ex1]
        lab = labels[ey0:ey1, ex0:ex1]
        blob = lab == lid
        temps = sub[blob]
        if temps.size == 0:
            continue
        ring = ~hot[ey0:ey1, ex0:ex1] & ~blob
        bg = float(np.median(sub[ring])) if ring.any() else float("nan")

        ys, xs = np.nonzero(blob)
        wts = temps - bg if np.isfinite(bg) else np.ones_like(temps)
        wts = np.clip(wts, 0.0, None)
        if float(wts.sum()) > 0.0:
            cx = float(np.dot(wts, xs) / wts.sum()) + ex0
            cy = float(np.dot(wts, ys) / wts.sum()) + ey0
        else:
            cx = float(xs.mean()) + ex0
            cy = float(ys.mean()) + ey0

        out.append({
            "cx": cx, "cy": cy,
            "bbox": (x0, y0, w, h),
            "area": area,
            "max_c": float(temps.max()),
            "mean_c": float(temps.mean()),
            "median_c": float(np.median(temps)),
            "bg_c": bg,
            "excess": float(np.sum(temps - bg)) if np.isfinite(bg) else float("nan"),
        })
    return out
