"""Per-track kinematics and temperature history, and the firebrand decision.

A confirmed track is counted as a firebrand when it
  1. has at least `min_track_dets` detections,
  2. starts within `start_tol_px` of the source region (every pixel that was
     ever persistently hot: the burning sample, burner and flame),
  3. ends at least `min_exit_m` from that region (it left the sample),
  4. moves a net `min_disp_m`, and
  5. if it has >= 4 motion vectors, has straightness (net / path length) of at
     least `min_straightness`. Fewer vectors are not scored: a straightness from
     two or three steps says nothing about the motion.
The first failing test is recorded as `reject_reason`.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .params import Params
from .units import lsq_slope, to_metres, velocity_series, vertical_acceleration


def track_features(points: List[tuple], fps: float, s: float, height_px: int,
                   dist_to_source: np.ndarray, p: Params) -> Dict:
    frames = np.array([f for f, _ in points], dtype=float)
    t = frames / fps
    xs = np.array([d["cx"] for _, d in points])
    ys = np.array([d["cy"] for _, d in points])
    X, Y = to_metres(xs, ys, s, height_px)
    vx, vy = velocity_series(t, X, Y, p.lsq_window)
    speed = np.hypot(vx, vy)
    tmax = np.array([d["max_c"] for _, d in points])
    tmean = np.array([d["mean_c"] for _, d in points])
    area = np.array([d["area"] for _, d in points], dtype=float)

    n = len(points)
    steps = np.hypot(np.diff(X), np.diff(Y))
    path = float(steps.sum())
    disp = float(math.hypot(X[-1] - X[0], Y[-1] - Y[0]))
    straight = disp / path if (n - 1) >= 4 and path > 0 else None

    k = min(p.lsq_window, n)
    lvx, lvy = lsq_slope(t[:k], X[:k]), lsq_slope(t[:k], Y[:k])

    def _dist(i: int) -> float:
        h, w = dist_to_source.shape
        xi = int(min(max(round(xs[i]), 0), w - 1))
        yi = int(min(max(round(ys[i]), 0), h - 1))
        return float(dist_to_source[yi, xi])

    cool = lsq_slope(t, tmax) if n >= 4 and t[-1] > t[0] else float("nan")
    return {
        "n_det": n,
        "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
        "start_time_s": float(t[0]), "end_time_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "launch_x_px": float(xs[0]), "launch_y_px": float(ys[0]),
        "launch_x_m": float(X[0]), "launch_y_m": float(Y[0]),
        "end_x_m": float(X[-1]), "end_y_m": float(Y[-1]),
        "displacement_m": disp, "path_length_m": path,
        "straightness": straight,
        "launch_speed_mps": float(math.hypot(lvx, lvy)) if np.isfinite(lvx) else float("nan"),
        "launch_dir_deg": float(math.degrees(math.atan2(lvy, lvx))) if np.isfinite(lvx) else float("nan"),
        "mean_speed_mps": float(np.nanmean(speed)) if np.isfinite(speed).any() else float("nan"),
        "max_speed_mps": float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan"),
        "vert_accel_mps2": vertical_acceleration(t, Y),
        "peak_c": float(tmax.max()), "mean_c": float(tmean.mean()),
        "cooling_c_per_s": cool,
        "mean_area_px": float(area.mean()),
        "start_dist_source_px": _dist(0), "end_dist_source_px": _dist(n - 1),
        # per-point series, written to track_points.csv.gz
        "_series": {"frame": frames.astype(int), "t": t, "x_px": xs, "y_px": ys, "X": X, "Y": Y,
                    "vx": vx, "vy": vy, "speed": speed, "max_c": tmax, "mean_c": tmean, "area": area},
    }


def classify(f: Dict, s: float, p: Params, have_source: bool) -> str:
    """Return '' for a firebrand, otherwise the first failing test."""
    if f["n_det"] < p.min_track_dets:
        return "too_short"
    if not have_source:
        return "no_source_region"
    if f["start_dist_source_px"] > p.start_tol_px:
        return "not_from_source"
    if f["end_dist_source_px"] * s < p.min_exit_m:
        return "never_left_source"
    if f["displacement_m"] < p.min_disp_m:
        return "too_little_displacement"
    if f["straightness"] is not None and f["straightness"] < p.min_straightness:
        return "not_straight"
    return ""
