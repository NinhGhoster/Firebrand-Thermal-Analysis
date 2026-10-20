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


def launch_velocity(t: np.ndarray, X: np.ndarray, Y: np.ndarray, window: int):
    """Velocity at the first detection, and the time it refers to.

    With >= 5 points, the derivative at t[0] of a quadratic fit over the first
    max(5, window + 1) points: the speed *at* launch, not the mean over the first
    few frames (which lags by about window/2 frames and so depends on frame rate
    for a firebrand that is slowing down or speeding up). With fewer points, the
    least-squares slope, which refers to the mean time of those points.
    Returns (vx, vy, t_ref, method).
    """
    n = len(t)
    m = min(n, max(5, window + 1))
    if m >= 5:
        tt = np.asarray(t[:m], dtype=float) - float(t[0])
        cx = np.polyfit(tt, np.asarray(X[:m], dtype=float), 2)
        cy = np.polyfit(tt, np.asarray(Y[:m], dtype=float), 2)
        return float(cx[1]), float(cy[1]), float(t[0]), "quadratic"
    return (lsq_slope(t[:m], X[:m]), lsq_slope(t[:m], Y[:m]),
            float(np.mean(t[:m])) if m else float("nan"), "linear")


def vertical_acceleration(t: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """Second derivative of a quadratic fit to height against time (needs >= 5 points)."""
    if len(t) < 5:
        return None
    tt = np.asarray(t, dtype=float) - float(np.mean(t))
    coeffs = np.polyfit(tt, np.asarray(Y, dtype=float), 2)
    return float(2.0 * coeffs[0])
