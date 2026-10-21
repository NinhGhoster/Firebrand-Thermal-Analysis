"""Pixels and frames to metres and seconds, and the kinematic formulas.

    time            t   = n / f
    metres / pixel  s   = 2 D tan(theta / 2) / W
    position        X   = s x,   Y = s (H - 1 - y)      (Y up, origin bottom-left)
    velocity        v   = least-squares slope of position against time
    speed           |v| = sqrt(vx^2 + vy^2),  direction = atan2(vy, vx)

Velocities are the projected, in-image-plane component: one camera cannot see
motion towards or away from it, and the scale is exact only at distance D.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def metres_per_pixel(distance_m: float, hfov_deg: float, width_px: int) -> float:
    return 2.0 * distance_m * math.tan(math.radians(hfov_deg) / 2.0) / float(width_px)


def to_metres(x_px, y_px, s: float, height_px: int):
    x = np.asarray(x_px, dtype=float)
    y = np.asarray(y_px, dtype=float)
    return s * x, s * (height_px - 1 - y)


def lsq_slope(t: np.ndarray, v: np.ndarray) -> float:
    """Slope of the least-squares straight line through (t, v)."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size < 2:
        return float("nan")
    tm = t - t.mean()
    den = float(np.dot(tm, tm))
    if den <= 0.0:
        return float("nan")
    return float(np.dot(tm, v - v.mean()) / den)


def velocity_series(t: np.ndarray, X: np.ndarray, Y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity at each point from a least-squares line over `window` neighbouring points."""
    n = len(t)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    if n < 2:
        return vx, vy
    w = max(2, min(window, n))
    half = w // 2
    for k in range(n):
        a = max(0, min(k - half, n - w))
        b = a + w
        vx[k] = lsq_slope(t[a:b], X[a:b])
        vy[k] = lsq_slope(t[a:b], Y[a:b])
    return vx, vy


def launch_velocity(t: np.ndarray, X: np.ndarray, Y: np.ndarray, span_s: float):
    """Velocity at the first detection, and how it was estimated. Applied to a
    firebrand's first detection outside the flame zone, this is its exit velocity.

    Fitted over every point within span_s of the first, so the fit covers the same
    duration at every frame rate (a fixed number of points would cover 0.4 s at 10 fps
    and 0.13 s at 30 fps, and a decelerating firebrand would read slower at 10 fps).
    With >= 4 such points: the derivative at t[0] of a quadratic fit (the speed *at*
    launch). With 2-3: the least-squares slope, which refers to the points' mean time.
    Returns (vx, vy, t_ref, method, span_used_s).
    """
    t = np.asarray(t, dtype=float)
    m = int(np.searchsorted(t, t[0] + span_s + 1e-9))
    m = max(m, min(len(t), 2))
    tt = t[:m] - t[0]
    span = float(tt[-1]) if m else 0.0
    if m >= 4:
        cx = np.polyfit(tt, np.asarray(X[:m], dtype=float), 2)
        cy = np.polyfit(tt, np.asarray(Y[:m], dtype=float), 2)
        return float(cx[1]), float(cy[1]), float(t[0]), "quadratic", span
    return (lsq_slope(t[:m], X[:m]), lsq_slope(t[:m], Y[:m]),
            float(np.mean(t[:m])) if m else float("nan"), "linear", span)


def turning_angle_deg(t: np.ndarray, X: np.ndarray, Y: np.ndarray, dt: float, min_span: float):
    """Median absolute turning angle (degrees) between successive steps of dt seconds.

    Positions are interpolated onto a time grid of step dt, so the statistic covers
    the same durations at every frame rate. None if the track spans less than
    min_span or gives fewer than three steps.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or t[-1] - t[0] < min_span - 1e-9:
        return None
    grid = np.arange(t[0], t[-1] + 1e-9, dt)
    if len(grid) < 4:
        return None
    gx = np.interp(grid, t, np.asarray(X, dtype=float))
    gy = np.interp(grid, t, np.asarray(Y, dtype=float))
    vx, vy = np.diff(gx), np.diff(gy)
    moving = np.hypot(vx, vy) > 1e-9
    ang = np.arctan2(vy, vx)
    pairs = moving[:-1] & moving[1:]
    if not pairs.any():
        return None
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)[pairs]
    return float(np.degrees(np.median(turn)))


def vertical_acceleration(t: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """Second derivative of a quadratic fit to height against time (needs >= 5 points)."""
    if len(t) < 5:
        return None
    tt = np.asarray(t, dtype=float) - float(np.mean(t))
    coeffs = np.polyfit(tt, np.asarray(Y, dtype=float), 2)
    return float(2.0 * coeffs[0])


def smoothness_px(frames, x_px, y_px, points: int = 5) -> float:
    """How far a track strays from a smooth path, in pixels (leave-one-out).

    Each detection is predicted from a quadratic in time fitted, per axis, through the
    other detections among the `points` nearest it (centred where the track allows),
    and the result is the median distance between prediction and detection. A solid
    body under gravity and drag follows a smooth path, so its detections are predicted
    to within centroid jitter (the pilot's real firebrands: 1-6 px). A centroid hopping
    between flame flickers is not: its next position is as far from any prediction as
    its hops are long (the pilot's old flicker chains: medians of 18-31 px). Leaving the
    point out keeps the statistic sharp when a track has only five detections, as at
    10 fps, where an in-sample fit would absorb much of a random hop. NaN if the track
    has fewer than `points` detections.
    """
    f = np.asarray(frames, dtype=float)
    x = np.asarray(x_px, dtype=float)
    y = np.asarray(y_px, dtype=float)
    n = len(f)
    k = max(4, int(points))
    if n < k:
        return float("nan")
    res = np.empty(n)
    for i in range(n):
        lo = max(0, min(i - k // 2, n - k))
        idx = [j for j in range(lo, lo + k) if j != i]
        t = f[idx] - f[idx].mean()
        ti = f[i] - f[idx].mean()
        px = np.polyfit(t, x[idx], 2)
        py = np.polyfit(t, y[idx], 2)
        res[i] = math.hypot(np.polyval(px, ti) - x[i], np.polyval(py, ti) - y[i])
    return float(np.median(res))
