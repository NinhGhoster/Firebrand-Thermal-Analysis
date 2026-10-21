"""Multi-object tracking: constant-velocity Kalman filter + Hungarian assignment.

Each frame, every live track predicts where its firebrand should be; detections
are matched to predictions by minimising a normalised cost over all pairs
(scipy.optimize.linear_sum_assignment).

Search region. A track with one detection has no velocity yet, so it searches
a disc whose radius is the farthest a firebrand at `v_max` can move in one
frame. An established track searches inside its own predicted uncertainty: a
detection is accepted when the Mahalanobis distance of the innovation,
d^2 = nu' S^-1 nu with S = H P H' + R, is within the chi-square gate (or within
`gate_floor_px`, so centroid jitter on irregular blobs cannot break a track).
P grows on every predict, so the region widens by itself while a track coasts
through missed frames and narrows again once it is re-acquired. Whatever P says, a
link is also refused when it would mean moving faster than `v_max`: farther from the
track's last detection than `gate_new_px` per elapsed frame plus `gate_floor_px`.
Without that cap a coasting track's region grew to ~0.7 m and, among the 5-20 flame
flickers of a real frame, always found something to take.

Motion model: continuous white-noise acceleration (CWNA). Per axis, the process
noise over one frame is Q = q [[1/3, 1/2], [1/2, 1]] with q the acceleration
power spectral density in px^2/frame^3 (converted by the caller from m^2/s^3).
Unlike a fixed acceleration per frame, its uncertainty grows the same way per
second at any frame rate: a coast of 0.18 s opens the same physical search
region at 10, 15 and 30 fps.

Assignment is a two-stage cascade. Established tracks (>= 2 detections) are
matched first, on the Mahalanobis cost; only the detections they leave are
offered to one-detection tracks, on the distance cost. A one-frame flicker blob
can therefore never take a detection from an established firebrand.

Deliberately avoids the defects found in firebrand-vision's tracker:
  - a confirmed track that misses frames keeps coasting and can be re-associated;
  - tracks are confirmed only after `min_hits` detections, and only confirmed
    tracks are reported;
  - any detection, however small, can start a track;
  - a link is refused when the two areas differ by more than `area_ratio_max`,
    so a gas puff and a speck are never chained together, or when the two peak
    temperatures' excesses over the threshold differ by more than `excess_ratio_max`,
    so a hot ember is never continued by a flicker hovering at the threshold.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
# CWNA process noise for dt = 1 frame, per unit spectral density (state x, y, vx, vy)
_QC = np.array([[1 / 3, 0, 1 / 2, 0], [0, 1 / 3, 0, 1 / 2], [1 / 2, 0, 1, 0], [0, 1 / 2, 0, 1]], dtype=float)
_BIG = 1e9


class _Track:
    def __init__(self, frame: int, det: Dict, q: float, r: float, v0_px: float):
        self.x = np.array([det["cx"], det["cy"], 0.0, 0.0])
        self.P = np.diag([r * r, r * r, v0_px * v0_px, v0_px * v0_px])
        self.Q = q * _QC                      # q: acceleration spectral density, px^2/frame^3
        self.R = (r * r) * np.eye(2)
        self.points: List[tuple] = [(frame, det)]
        self.area = float(det["area"])
        self.peak = float(det.get("max_c", np.nan))
        self.hits = 1
        self.misses = 0
        self.public_id: Optional[int] = None

    def predict(self) -> None:
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self.Q

    def innovation(self, z: np.ndarray):
        nu = z - _H @ self.x
        S = _H @ self.P @ _H.T + self.R
        return nu, S

    def update(self, frame: int, det: Dict) -> None:
        z = np.array([det["cx"], det["cy"]])
        nu, S = self.innovation(z)
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.P = (np.eye(4) - K @ _H) @ self.P
        self.points.append((frame, det))
        self.area = float(det["area"])
        self.peak = float(det.get("max_c", np.nan))
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, gate_new_px: float, q_px: float, r_px: float, chi2_gate: float,
                 gate_floor_px: float, max_misses: int, min_hits: int,
                 area_ratio_max: float = float("inf"), excess_ratio_max: float = float("inf"),
                 excess_base_c: float = 290.0):
        self.gate_new = float(gate_new_px)
        self.q, self.r = float(q_px), float(r_px)
        self.chi2 = float(chi2_gate)
        self.floor = float(gate_floor_px)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.area_ratio_max = float(area_ratio_max)
        self.excess_ratio_max = float(excess_ratio_max)
        self.excess_base = float(excess_base_c)
        self.live: List[_Track] = []
        self.finished: List[_Track] = []
        self._next_id = 1

    def _cost_row(self, t: _Track, obs: np.ndarray, da: np.ndarray, dpeak: np.ndarray, frame: int) -> np.ndarray:
        """Cost of linking track t to each detection; _BIG outside its gate."""
        ratio = np.maximum(t.area + 2.0, da) / np.minimum(t.area + 2.0, da)
        size_ok = ratio <= self.area_ratio_max
        if np.isfinite(t.peak) and np.isfinite(self.excess_ratio_max):
            et = max(t.peak - self.excess_base, 1.0)
            ed = np.maximum(dpeak - self.excess_base, 1.0)
            with np.errstate(invalid="ignore"):
                warm_ok = ~(np.maximum(et, ed) / np.minimum(et, ed) > self.excess_ratio_max)
            size_ok = size_ok & warm_ok
        if t.hits < 2:
            dist = np.linalg.norm(obs - t.x[:2], axis=1)
            ok = size_ok & (dist <= self.gate_new)
            c = dist
        else:
            nu = obs - (_H @ t.x)
            S = _H @ t.P @ _H.T + t.R
            d2 = np.einsum("ij,jk,ik->i", nu, np.linalg.inv(S), nu)
            dist = np.linalg.norm(nu, axis=1)
            last_frame, last = t.points[-1]
            reach = self.gate_new * (frame - last_frame) + self.floor
            jump = np.linalg.norm(obs - np.array([last["cx"], last["cy"]]), axis=1)
            ok = size_ok & ((d2 <= self.chi2) | (dist <= self.floor)) & (jump <= reach)
            c = d2
        return np.where(ok, c, _BIG)

    def _assign(self, frame: int, track_idx: List[int], dets: List[Dict], free: List[int],
                matched_t: set, matched_d: set) -> None:
        if not track_idx or not free:
            return
        obs = np.array([[dets[i]["cx"], dets[i]["cy"]] for i in free])
        da = np.array([dets[i]["area"] for i in free], dtype=float) + 2.0
        dpeak = np.array([dets[i].get("max_c", np.nan) for i in free], dtype=float)
        cost = np.vstack([self._cost_row(self.live[ti], obs, da, dpeak, frame) for ti in track_idx])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] >= _BIG:
                continue
            ti, di = track_idx[r], free[c]
            self.live[ti].update(frame, dets[di])
            matched_t.add(ti)
            matched_d.add(di)

    def step(self, frame: int, dets: List[Dict]) -> None:
        for t in self.live:
            t.predict()

        matched_t, matched_d = set(), set()
        if self.live and dets:
            established = [i for i, t in enumerate(self.live) if t.hits >= 2]
            newborn = [i for i, t in enumerate(self.live) if t.hits < 2]
            self._assign(frame, established, dets, list(range(len(dets))), matched_t, matched_d)
            left = [i for i in range(len(dets)) if i not in matched_d]
            self._assign(frame, newborn, dets, left, matched_t, matched_d)

        survivors: List[_Track] = []
        for ti, t in enumerate(self.live):
            if ti not in matched_t:
                t.misses += 1
                tentative = t.hits < self.min_hits
                if tentative or t.misses > self.max_misses:
                    if not tentative:
                        self.finished.append(t)
                    continue
            if t.public_id is None and t.hits >= self.min_hits:
                t.public_id = self._next_id
                self._next_id += 1
            survivors.append(t)

        for di, d in enumerate(dets):
            if di not in matched_d:
                survivors.append(_Track(frame, d, self.q, self.r, self.gate_new))
        self.live = survivors

    def flush(self) -> List[_Track]:
        """End of video: finish every confirmed track; return all confirmed tracks in ID order."""
        for t in self.live:
            if t.public_id is not None:
                self.finished.append(t)
        self.live = []
        return sorted(self.finished, key=lambda t: t.public_id)
