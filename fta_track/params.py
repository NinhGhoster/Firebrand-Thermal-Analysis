"""All tunable parameters in one place, recorded verbatim in every summary.json."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Params:
    # --- detection -------------------------------------------------------
    # The T1040's calibrated floor. Level 1.1 keeps full precision only at and
    # above 299 degC, so 300 is also the lowest threshold with usable resolution.
    hot_thresh_c: float = 300.0
    # Ignition is the first frame with at least this many pixels >= hot_thresh_c.
    # Nothing before it can be a firebrand.
    ignition_min_px: int = 20
    min_area_px: int = 1          # a 1-px blob may start a track; tracking filters noise
    max_area_px: int = 600        # ~16 cm^2 at 1.62 mm/px
    ring_px: int = 3              # width of the local-background ring around a blob

    # --- persistent-hot mask (burning sample, burner, flame) --------------
    # A pixel is "persistent" if it is >= hot_thresh_c in at least persist_frac of
    # the frames in a window of persist_window_s centred on the frame. A flying
    # ember covers any one pixel for a frame or two (~1 % of a 3 s window).
    persist_window_s: float = 3.0
    persist_frac: float = 0.15
    persist_dilate_px: int = 3

    # --- tracking ----------------------------------------------------------
    v_max_mps: float = 8.0        # sets the search radius for a track with no velocity yet
    gate_min_px: float = 15.0     # search radius around the prediction for a moving track
    gate_speed_frac: float = 0.5  # ... plus this fraction of the predicted speed (px/frame)
    kalman_q_px: float = 3.0      # process noise: unmodelled acceleration, px/frame^2
    kalman_r_px: float = 1.0      # measurement noise, px
    max_miss_s: float = 0.2       # a confirmed track coasts on its prediction this long
    min_hits: int = 3             # detections needed before a track is confirmed

    # --- firebrand classification -----------------------------------------
    min_track_dets: int = 4
    start_tol_px: float = 15.0    # must start within this distance of the source region
    min_exit_m: float = 0.05      # ... and end at least this far from it
    min_disp_m: float = 0.05      # net displacement
    min_straightness: float = 0.5 # net / path length, applied only with >= 4 motion vectors

    # --- kinematics and units ---------------------------------------------
    lsq_window: int = 5           # positions per least-squares velocity fit
    lens_hfov_deg: float = 45.0   # FLIR T1040, FOL21 lens
    m_per_px: Optional[float] = None  # override the lens-derived scale

    def window_frames(self, fps: float) -> int:
        """Odd window length in frames, so the frame being processed sits at its centre."""
        n = max(3, int(round(self.persist_window_s * fps)))
        return n if n % 2 else n + 1

    def max_miss_frames(self, fps: float) -> int:
        return max(1, int(math.ceil(self.max_miss_s * fps)))

    def to_dict(self) -> dict:
        return asdict(self)
