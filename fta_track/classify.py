"""Per-track kinematics and temperature history, and the firebrand decision.

The rig (burner tray, frame, side panels), the burning sample and the flame sit
at or near the hot threshold for much of a burn; pixels hovering there flicker
into blobs that can be tracked for a few frames but never go anywhere. A flying
firebrand is different in two ways that temperature measures directly: it is
seen in *cold space* (where the long-term hot fraction is below `env_frac`), and
there it is clearly hotter than the air around it.

A confirmed track is counted as a firebrand when it
  1. has at least 3 detections spanning at least `min_track_span_s` (measured
     from the detection times, so a coast in between counts as time),
  2. has enough detections (`min_cold_s`, converted per frame rate) in cold space whose contrast
     (blob peak minus what those pixels usually read: the temporal background,
     or the surrounding ring for older outputs without it) is >= `min_contrast_c`
     and whose edge gradient is >= `min_edge_grad` (a solid object's hard edge,
     not a parcel of hot gas fading into the plume),
  3. moves a net `min_disp_m`, and
  4. once it spans `straight_min_s`, does not move erratically: the median
     turning angle between successive `straight_dt_s` steps is at most
     `max_turn_deg`. A lofted firebrand flies an arc and turns little per step;
     a centroid hopping between flicker blobs turns ~90 degrees per step. (A
     whole-track net/path would reject the arcs; a net/path over 3-step windows
     passes most random wandering.)
The first failing test is recorded as `reject_reason`.
"""

from __future__ import annotations

import math
from typing import Dict, List

import cv2
import numpy as np

from .params import Params
from .units import (launch_velocity, lsq_slope, to_metres, turning_angle_deg, velocity_series,
                    vertical_acceleration)


def env_threshold(n_post: int, fps: float, p: Params) -> float:
    """Hot fraction at and above which a pixel is rig or flame, not cold space.

    env_frac of the post-ignition frames, but never fewer hot frames than
    env_min_s worth: in a 316-frame recording 5 % is only 16 frames, so a few
    firebrands crossing the same pixels would otherwise turn their own flight
    path into "rig" and reject themselves.
    """
    if n_post <= 0:
        return p.env_frac
    # capped at persist_frac: above it, the flickering flame zone would start to count
    # as cold space, and past 1.0 nothing would be rig at all in a very short recording
    return min(max(p.env_frac, p.env_min_s * fps / n_post), p.persist_frac)


def cold_space(hot_frac: np.ndarray, n_post: int, fps: float, p: Params) -> np.ndarray:
    """True where a firebrand can be told apart from the rig and the flame."""
    env = (hot_frac >= env_threshold(n_post, fps, p)).astype(np.uint8)
    if p.env_dilate_px > 0:
        k = 2 * p.env_dilate_px + 1
        env = cv2.dilate(env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return ~env.astype(bool)


def track_features(points: List[tuple], fps: float, s: float, cold: np.ndarray, p: Params, dv: Dict) -> Dict:
    H, W = cold.shape
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
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    in_cold = cold[yi, xi]
    solid = ~(np.nan_to_num(egrad, nan=np.inf) < p.min_edge_grad)   # missing (older outputs) counts as solid
    evidence = in_cold & (np.nan_to_num(contrast, nan=-1.0) >= p.min_contrast_c) & solid

    n = len(points)
    # typical frame-to-frame size change: a solid firebrand stays near 0, gas puffs do not
    area_jump = float(np.median(np.abs(np.diff(np.log(area + 2.0))))) if n >= 2 else float("nan")
    steps = np.hypot(np.diff(X), np.diff(Y))
    path = float(steps.sum())
    disp = float(math.hypot(X[-1] - X[0], Y[-1] - Y[0]))
    turn = turning_angle_deg(t, X, Y, p.straight_dt_s, p.straight_min_s)

    lvx, lvy, lt, lmethod, lspan = launch_velocity(t, X, Y, p.launch_fit_s)
    cool = lsq_slope(t, tmax) if n >= 4 and t[-1] > t[0] else float("nan")
    finite = np.isfinite(speed)
    return {
        "n_det": n,
        "n_cold": int(in_cold.sum()),
        "n_evidence": int(evidence.sum()),
        "max_contrast_c": float(np.nanmax(contrast)) if np.isfinite(contrast).any() else float("nan"),
        "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
        "start_time_s": float(t[0]), "end_time_s": float(t[-1]),
        "duration_s": float(t[-1] - t[0]),
        "launch_x_px": float(xs[0]), "launch_y_px": float(ys[0]),
        "launch_x_m": float(X[0]), "launch_y_m": float(Y[0]),
        "end_x_m": float(X[-1]), "end_y_m": float(Y[-1]),
        "displacement_m": disp, "path_length_m": path,
        "turn_deg": turn,
        "launch_speed_mps": float(math.hypot(lvx, lvy)) if np.isfinite(lvx) else float("nan"),
        "launch_dir_deg": float(math.degrees(math.atan2(lvy, lvx))) if np.isfinite(lvx) else float("nan"),
        "launch_t_s": lt, "launch_fit": lmethod, "launch_span_s": lspan,
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
                    "contrast": contrast, "edge_grad": egrad, "cold": in_cold},
    }


def classify(f: Dict, p: Params, dv: Dict) -> str:
    """Return '' for a firebrand, otherwise the first failing test."""
    if f["n_det"] < 3 or f["duration_s"] < p.min_track_span_s - 1e-9:
        return "too_short"
    if f["n_cold"] < dv["min_cold_dets"]:
        return "not_in_cold_space"
    if f["n_evidence"] < dv["min_cold_dets"]:
        return "low_contrast"
    if f["displacement_m"] < p.min_disp_m:
        return "too_little_displacement"
    if f["turn_deg"] is not None and f["turn_deg"] > p.max_turn_deg:
        return "erratic"
    return ""
