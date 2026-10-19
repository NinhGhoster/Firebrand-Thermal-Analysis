"""Multi-object tracking: constant-velocity Kalman filter + Hungarian assignment.

Each frame, every live track predicts where its firebrand should be; detections
are matched to predictions by globally minimising total distance
(scipy.optimize.linear_sum_assignment), within a per-track search radius.

Deliberately avoids the defects found in firebrand-vision's tracker:
  - a confirmed track that misses frames keeps coasting on its prediction and
    can be re-associated (it is not frozen out after one miss);
  - tracks are confirmed only after `min_hits` detections, and only confirmed
    tracks are reported;
  - any detection, however small, can start a track.
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
        self.hits = 1
        self.misses = 0
        self.public_id: Optional[int] = None

    def predict(self) -> None:
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + self.Q

    def update(self, frame: int, det: Dict) -> None:
        z = np.array([det["cx"], det["cy"]])
        S = _H @ self.P @ _H.T + self.R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(4) - K @ _H) @ self.P
        self.points.append((frame, det))
        self.hits += 1
        self.misses = 0

    @property
    def speed_px(self) -> float:
        return float(np.hypot(self.x[2], self.x[3]))


class Tracker:
    def __init__(self, gate_new_px: float, gate_min_px: float, gate_speed_frac: float,
                 q_px: float, r_px: float, max_misses: int, min_hits: int):
        self.gate_new = float(gate_new_px)
        self.gate_min = float(gate_min_px)
        self.gate_frac = float(gate_speed_frac)
        self.q, self.r = float(q_px), float(r_px)
        self.max_misses = int(max_misses)
        self.min_hits = int(min_hits)
        self.live: List[_Track] = []
        self.finished: List[_Track] = []
        self._next_id = 1

    def _gate(self, t: _Track) -> float:
        if t.hits < 2:
            return self.gate_new
        return self.gate_min + self.gate_frac * t.speed_px

    def step(self, frame: int, dets: List[Dict]) -> None:
        for t in self.live:
            t.predict()

        matched_t, matched_d = set(), set()
        if self.live and dets:
            pred = np.array([t.x[:2] for t in self.live])
            obs = np.array([[d["cx"], d["cy"]] for d in dets])
            dist = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
            gates = np.array([self._gate(t) for t in self.live])
            cost = np.where(dist <= gates[:, None], dist, _BIG)
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
