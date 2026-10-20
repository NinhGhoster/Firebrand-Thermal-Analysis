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
through missed frames and narrows again once it is re-acquired.

The process noise is an acceleration in pixels per frame squared, converted
by the caller from m/s^2, so the same physical trajectory is tracked the same
way at 10 fps and at 31.25 fps.

Deliberately avoids the defects found in firebrand-vision's tracker:
  - a confirmed track that misses frames keeps coasting and can be re-associated;
  - tracks are confirmed only after `min_hits` detections, and only confirmed
    tracks are reported;
  - any detection, however small, can start a track;
  - a link is refused when the two areas differ by more than `area_ratio_max`,
    so a gas puff and a speck are never chained together.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
_G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], dtype=float)
_BIG = 1e9


class _Track:
    def __init__(self, frame: int, det: Dict, q: float, r: float, v0_px: float):
        self.x = np.array([det["cx"], det["cy"], 0.0, 0.0])
        self.P = np.diag([r * r, r * r, v0_px * v0_px, v0_px * v0_px])
        self.Q = (q * q) * (_G @ _G.T)
        self.R = (r * r) * np.eye(2)
        self.points: List[tuple] = [(frame, det)]
        self.area = float(det["area"])
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
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, gate_new_px: float, q_px: float, r_px: float, chi2_gate: float,
                 gate_floor_px: float, max_misses: int, min_hits: int,
                 area_ratio_max: float = float("inf")):
        self.gate_new = float(gate_new_px)
        self.q, self.r = float(q_px), float(r_px)
        self.chi2 = float(chi2_gate)
        self.floor = float(gate_floor_px)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.area_ratio_max = float(area_ratio_max)
        self.live: List[_Track] = []
        self.finished: List[_Track] = []
        self._next_id = 1

    def _costs(self, dets: List[Dict]) -> np.ndarray:
        """Normalised cost (<= 1 inside the gate, _BIG outside) for every track/detection pair."""
        obs = np.array([[d["cx"], d["cy"]] for d in dets])
        da = np.array([d["area"] for d in dets], dtype=float) + 2.0
        cost = np.full((len(self.live), len(dets)), _BIG)
        for ti, t in enumerate(self.live):
            ratio = np.maximum(t.area + 2.0, da) / np.minimum(t.area + 2.0, da)
            size_ok = ratio <= self.area_ratio_max
            if t.hits < 2:
                dist = np.linalg.norm(obs - t.x[:2], axis=1)
                ok = size_ok & (dist <= self.gate_new)
                c = (dist / self.gate_new) ** 2
            else:
                nu = obs - (_H @ t.x)
                S = _H @ t.P @ _H.T + t.R
                d2 = np.einsum("ij,jk,ik->i", nu, np.linalg.inv(S), nu)
                dist = np.linalg.norm(nu, axis=1)
                ok = size_ok & ((d2 <= self.chi2) | (dist <= self.floor))
                c = np.minimum(d2 / self.chi2, 1.0)
            cost[ti] = np.where(ok, c, _BIG)
        return cost

    def step(self, frame: int, dets: List[Dict]) -> None:
        for t in self.live:
            t.predict()

        matched_t, matched_d = set(), set()
        if self.live and dets:
            cost = self._costs(dets)
            rows, cols = linear_sum_assignment(cost)
            for ti, di in zip(rows, cols):
                if cost[ti, di] >= _BIG:
                    continue
                self.live[ti].update(frame, dets[di])
                matched_t.add(ti)
                matched_d.add(di)

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
