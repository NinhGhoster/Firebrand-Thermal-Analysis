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

from .optics import blob_extent_px


_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def detect_weak_blobs(frame: np.ndarray, hot: np.ndarray, low_c: float, max_area: int, near_px: int,
                      ring_px: int, min_edge: float, samples: Optional[Sequence[np.ndarray]] = None) -> List[Dict]:
    """Small hard-edged spots reading low_c <= T < the hot threshold, away from any hot pixel.

    A component that comes within near_px of a hot pixel is dropped whole: it is the halo of a
    hot blob, not an object of its own. Each kept blob carries weak=True.
    """
    weak = (frame >= low_c) & ~hot
    if not weak.any():
        return []
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    if near_px > 0 and hot.any():
        k = 2 * near_px + 1
        near = cv2.dilate(hot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))).astype(bool)
        bad = np.unique(lab[near & weak])
        weak = weak & ~np.isin(lab, bad)
    out = [d for d in detect_blobs(frame, weak, np.zeros_like(weak), 1, max_area, ring_px, samples)
           if np.isfinite(d["edge_grad"]) and d["edge_grad"] >= min_edge]
    for d in out:
        d["weak"] = True
    return out


def patch_at(frame: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    """size x size temperatures centred on the pixel nearest (cx, cy); edges replicated."""
    c = (float(int(round(cx))), float(int(round(cy))))
    return cv2.getRectSubPix(np.ascontiguousarray(frame, dtype=np.float32), (size, size), c).astype(np.float16)


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

        length_px, width_px = blob_extent_px(blob)
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
            "length_px": round(length_px, 2),
            "width_px": round(width_px, 2),
            "max_c": round(float(temps.max()), 2),
            "mean_c": round(float(temps.mean()), 2),
            "median_c": round(float(np.median(temps)), 2),
            "bg_c": round(bg, 2) if np.isfinite(bg) else bg,
            "bg_t_c": round(bg_t, 2) if np.isfinite(bg_t) else bg_t,
            "edge_grad": round(edge_grad, 2) if np.isfinite(edge_grad) else edge_grad,
            "excess": round(float(np.sum(temps - bg)), 1) if np.isfinite(bg) else float("nan"),
        })
    return out
