"""The flame zone, per-track kinematics and temperature history, and the firebrand decision.

Inside the flame zone (the burning sample, the burner and the flame: see `flame_zone`)
temperature cannot tell a firebrand from flame. Pixels there flicker above the hot
threshold for much of a burn, and a tracker fed those blobs chains them into paths that
look like flights. So only detections outside the zone are tracked (pipeline.py), and a
track is a firebrand when it

  1. has at least `smooth_points` detections spanning at least `min_track_span_s`,
  2. starts at least `settle_s` after ignition (the burner's lighting flash throws off
     hot blobs before any fuel can have charred),
  3. moves smoothly: each position is predicted from a local quadratic through its
     neighbours, and the median miss is at most `smooth_floor_px` or, for a fast track,
     `smooth_step_ratio` of its median step (a solid body follows a smooth path; a
     centroid hopping between flickers misses by about its hop length),
  4. has enough detections (`min_evidence_s`, converted per frame rate) that are clearly
     hotter than what those pixels usually read (contrast >= `min_contrast_c`) and have a
     solid object's hard edge (edge step >= `min_edge_grad`, not a gas parcel fading
     into the plume),
  5. moves a net `min_disp_m`, and
  6. if `max_crowding` is set, is not in a swarm of other detections.

The first failing test is recorded as `reject_reason`, and every failing test in `fails`,
so a near-miss (one failing test) can be told from junk.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from .params import Params
from .units import (launch_velocity, lsq_slope, smoothness_px, to_metres, turning_angle_deg,
                    velocity_series, vertical_acceleration)

REASONS = ("too_short", "ignition_flash", "rough", "low_contrast", "too_little_displacement", "crowded")


def zone_threshold(n_post: int, fps: float, p: Params) -> float:
    """Hot fraction at and above which a pixel is flame, sample or burner.

    zone_frac of the post-ignition frames, but never fewer hot frames than zone_min_s
    worth: in a 300-frame recording 1 % is three frames, so three firebrands crossing the
    same pixels would turn their own path into flame.
    """
    if n_post <= 0:
        return p.zone_frac
    return min(max(p.zone_frac, p.zone_min_s * fps / n_post), 1.0)


def flame_zone(hot_frac: np.ndarray, n_post: int, fps: float, p: Params) -> np.ndarray:
    """True where the flame, the burning sample or the burner is, widened and hole-filled."""
    z = (hot_frac >= zone_threshold(n_post, fps, p)).astype(np.uint8)
    if p.zone_dilate_px > 0:
        k = 2 * p.zone_dilate_px + 1
        z = cv2.dilate(z, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ndi.binary_fill_holes(z.astype(bool))


def zone_distance(zone: np.ndarray) -> np.ndarray:
    """Distance in pixels from each pixel to the nearest zone pixel (0 inside the zone)."""
    if not zone.any():
        return np.full(zone.shape, np.inf, dtype=np.float32)
    return ndi.distance_transform_edt(~zone).astype(np.float32)


def _at(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    H, W = img.shape
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    return img[yi, xi]


def track_features(points: List[tuple], fps: float, s: float, ctx: Dict, p: Params, dv: Dict) -> Dict:
    """Kinematics, temperature history and test inputs of one track.

    ctx: "zone_dist" (zone_distance of the recording's flame zone), "hot_frac",
    "ignition_frame", and "crowd" (per-detection crowding, aligned with points).
    """
    zone_dist = ctx["zone_dist"]
    H = zone_dist.shape[0]
    frames = np.array([f for f, _ in points], dtype=float)
    t = frames / fps
    xs = np.array([d["cx"] for _, d in points])
    ys = np.array([d["cy"] for _, d in points])
    X, Y = to_metres(xs, ys, s, H)
    vx, vy = velocity_series(t, X, Y, dv["lsq_points"])
    speed = np.hypot(vx, vy)
    tmax = np.array([d["max_c"] for _, d in points])
    tmean = np.array([d["mean_c"] for _, d in points])
    area = np.array([d["area"] for _, d in points], dtype=float)
    egrad = np.array([d.get("edge_grad", np.nan) for _, d in points], dtype=float)
    bg_t = np.array([d.get("bg_t_c", np.nan) for _, d in points], dtype=float)
    bg_r = np.array([d["bg_c"] for _, d in points], dtype=float)
    contrast = tmax - np.where(np.isfinite(bg_t), bg_t, bg_r)
    zd = _at(zone_dist, xs, ys).astype(float)
    crowd = np.asarray(ctx.get("crowd") or [np.nan] * len(points), dtype=float)
    solid = ~(np.nan_to_num(egrad, nan=np.inf) < p.min_edge_grad)   # missing (older outputs) counts as solid
    evidence = (np.nan_to_num(contrast, nan=-1.0) >= p.min_contrast_c) & solid

    n = len(points)
    # typical frame-to-frame size change: a solid firebrand stays near 0, gas puffs do not
    area_jump = float(np.median(np.abs(np.diff(np.log(area + 2.0))))) if n >= 2 else float("nan")
    steps = np.hypot(np.diff(X), np.diff(Y))
    path = float(steps.sum())
    disp = float(math.hypot(X[-1] - X[0], Y[-1] - Y[0]))
    turn = turning_angle_deg(t, X, Y, p.straight_dt_s, p.straight_min_s)
    ign = ctx.get("ignition_frame")

    evx, evy, et, emethod, espan = launch_velocity(t, X, Y, p.exit_fit_s)
    cool = lsq_slope(t, tmax) if n >= 4 and t[-1] > t[0] else float("nan")
    finite = np.isfinite(speed)
    return {
        "n_det": n,
        "n_evidence": int(evidence.sum()),
        "max_contrast_c": float(np.nanmax(contrast)) if np.isfinite(contrast).any() else float("nan"),
        "smooth_px": smoothness_px(frames, xs, ys, p.smooth_points),
        "median_step_px": float(np.median(np.hypot(np.diff(xs), np.diff(ys)))) if n >= 2 else float("nan"),
        "crowding": float(np.nanmedian(crowd)) if np.isfinite(crowd).any() else float("nan"),
        "start_after_ignition_s": float((frames[0] - ign) / fps) if ign is not None else float("nan"),
        "exit_zone_dist_px": float(zd[0]), "end_zone_dist_px": float(zd[-1]),
        "exit_hot_frac": float(_at(ctx["hot_frac"], xs[:1], ys[:1])[0]) if ctx.get("hot_frac") is not None
        else float("nan"),
        "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
        "start_time_s": float(t[0]), "end_time_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "exit_x_px": float(xs[0]), "exit_y_px": float(ys[0]),
        "exit_x_m": float(X[0]), "exit_y_m": float(Y[0]),
        "end_x_m": float(X[-1]), "end_y_m": float(Y[-1]),
        "displacement_m": disp, "path_length_m": path,
        "turn_deg": turn,
        "exit_speed_mps": float(math.hypot(evx, evy)) if np.isfinite(evx) else float("nan"),
        "exit_dir_deg": float(math.degrees(math.atan2(evy, evx))) if np.isfinite(evx) else float("nan"),
        "exit_t_s": et, "exit_fit": emethod, "exit_span_s": espan,
        "mean_speed_mps": float(np.nanmean(speed)) if finite.any() else float("nan"),
        "max_speed_mps": float(np.nanmax(speed)) if finite.any() else float("nan"),
        "vert_accel_mps2": vertical_acceleration(t, Y),
        "peak_c": float(tmax.max()), "mean_c": float(tmean.mean()),
        "cooling_c_per_s": cool,
        "mean_area_px": float(area.mean()),
        "area_jump": area_jump,
        "edge_grad": float(np.nanmedian(egrad)) if np.isfinite(egrad).any() else float("nan"),
        # per-point series, written to track_points.csv.gz
        "_series": {"frame": frames.astype(int), "t": t, "x_px": xs, "y_px": ys, "X": X, "Y": Y,
                    "vx": vx, "vy": vy, "speed": speed, "max_c": tmax, "mean_c": tmean, "area": area,
                    "contrast": contrast, "edge_grad": egrad, "zone_dist": zd, "crowd": crowd},
    }


def classify(f: Dict, p: Params, dv: Dict) -> Tuple[str, str]:
    """('', '') for a firebrand; otherwise (first failing test, every failing test joined by ';')."""
    fails = []
    if f["n_det"] < dv["min_dets"] or f["duration_s"] < p.min_track_span_s - 1e-9:
        fails.append("too_short")
    if math.isfinite(f["start_after_ignition_s"]) and f["start_after_ignition_s"] < p.settle_s - 1e-9:
        fails.append("ignition_flash")
    limit = max(p.smooth_floor_px, p.smooth_step_ratio * f["median_step_px"]) \
        if math.isfinite(f["median_step_px"]) else p.smooth_floor_px
    if not (f["smooth_px"] <= limit):                      # NaN (too few points) fails too
        fails.append("rough")
    if f["n_evidence"] < dv["min_evidence_dets"]:
        fails.append("low_contrast")
    if f["displacement_m"] < p.min_disp_m:
        fails.append("too_little_displacement")
    if p.max_crowding is not None and math.isfinite(f["crowding"]) and f["crowding"] > p.max_crowding:
        fails.append("crowded")
    return (fails[0] if fails else ""), ";".join(fails)
