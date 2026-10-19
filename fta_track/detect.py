"""Per-frame blob detection in degrees Celsius.

Ported from `detect_firebrands()` in FirebrandThermalAnalysis.py (absolute
threshold, 8-connected components, statistics on the component mask), with:
  - the persistent-hot mask removed before labelling,
  - a temperature-weighted centroid (sub-pixel) for tracking,
  - a local background from a ring around the blob, and the integrated excess
    temperature sum(T - background), which stays meaningful for embers smaller
    than a pixel whose peak reads low,
  - a temporal background: the median, over frames sampled across a few
    seconds, of what the blob's own pixels read. A flicker on a rail that sits
    near 290 C has a temporal background near 290; an ember crossing dark air
    has one near the air temperature,
  - the edge gradient: the temperature gradient (degC per pixel) across the
    blob's boundary. A solid firebrand is optically thick with a hard edge, so
    temperature drops by hundreds of degrees within a pixel or two; a parcel of
    hot gas is optically thin and diffuse, and fades into the plume around it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def detect_blobs(frame: np.ndarray, hot: np.ndarray, persistent: np.ndarray,
                 min_area: int, max_area: int, ring_px: int,
                 samples: Optional[Sequence[np.ndarray]] = None) -> List[Dict]:
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

        # the temperature step across the boundary, in degC per pixel: the blob's
        # boundary pixels against the pixels just outside them. (A gradient filter
        # would read zero on a 1-px ember, whose neighbours on both sides are cold.)
        b8 = blob.astype(np.uint8)
        edge = blob & ~cv2.erode(b8, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        outer = cv2.dilate(b8, _CROSS).astype(bool) & ~blob
        edge_grad = float(np.median(sub[edge]) - np.median(sub[outer])) if outer.any() else float("nan")

        bg_t = float("nan")
        if samples:
            own = labels[y0:y0 + h, x0:x0 + w] == lid
            stack = np.stack([sm[y0:y0 + h, x0:x0 + w] for sm in samples])
            bg_t = float(np.median(np.median(stack, axis=0)[own]))

        ys, xs = np.nonzero(blob)
        wts = temps - bg if np.isfinite(bg) else np.ones_like(temps)
        wts = np.clip(wts, 0.0, None)
        if float(wts.sum()) > 0.0:
            cx = float(np.dot(wts, xs) / wts.sum()) + ex0
            cy = float(np.dot(wts, ys) / wts.sum()) + ey0
        else:
            cx = float(xs.mean()) + ex0
            cy = float(ys.mean()) + ey0

        # Rounded to the precision detections.csv.gz stores, so that re-tracking from
        # the saved file (FTA_CLI.py retrack) sees exactly what this run saw.
        out.append({
            "cx": round(cx, 2), "cy": round(cy, 2),
            "bbox": (x0, y0, w, h),
            "area": area,
            "max_c": round(float(temps.max()), 2),
            "mean_c": round(float(temps.mean()), 2),
            "median_c": round(float(np.median(temps)), 2),
            "bg_c": round(bg, 2) if np.isfinite(bg) else bg,
            "bg_t_c": round(bg_t, 2) if np.isfinite(bg_t) else bg_t,
            "edge_grad": round(edge_grad, 2) if np.isfinite(edge_grad) else edge_grad,
            "excess": round(float(np.sum(temps - bg)), 1) if np.isfinite(bg) else float("nan"),
        })
    return out
