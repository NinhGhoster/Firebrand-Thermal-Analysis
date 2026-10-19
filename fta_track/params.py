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
    max_area_px: int = 2000       # ~52 cm^2 at 1.62 mm/px: large bark pieces must not be lost
    ring_px: int = 3              # width of the local-background ring around a blob
    # Temporal background: what the blob's own pixels usually read, as the median
    # of this many frames sampled across the persistence window. Unlike the ring,
    # it is not dragged down by cold air beside a hot rail or panel edge.
    bg_samples: int = 15

    # --- persistent-hot mask (burning sample, burner, flame) --------------
    # A pixel is "persistent" if it is >= hot_thresh_c in at least persist_frac of
    # the frames in a window of persist_window_s centred on the frame. A flying
    # ember covers any one pixel for a frame or two (~1 % of a 3 s window).
    persist_window_s: float = 3.0
    persist_frac: float = 0.15
    persist_dilate_px: int = 3

    # --- tracking ----------------------------------------------------------
    v_max_mps: float = 4.0        # sets the search radius for a track with no velocity yet
    gate_min_px: float = 15.0     # search radius around the prediction for a moving track
    gate_speed_frac: float = 0.5  # ... plus this fraction of the predicted speed (px/frame)
    kalman_q_px: float = 3.0      # process noise: unmodelled acceleration, px/frame^2
    kalman_r_px: float = 1.0      # measurement noise, px
    max_miss_s: float = 0.2       # a confirmed track coasts on its prediction this long
    min_hits: int = 3             # detections needed before a track is confirmed
    # A solid firebrand does not change size a hundredfold between frames; gas
    # puffs and specks do. Links whose areas differ by more than this are refused
    # (ratio of area + 2 px, so 1-4 px specks are not over-penalised).
    track_area_ratio_max: float = 4.0

    # --- firebrand classification -----------------------------------------
    # The rig (burner tray, frame, panels) and the flame sit at or near the hot
    # threshold for much of a burn, and pixels hovering there flicker into blobs.
    # "Cold space" is everywhere hot in less than env_frac of post-ignition frames;
    # a firebrand must be seen there, clearly hotter than its surroundings.
    min_track_dets: int = 4
    env_frac: float = 0.05        # long-term hot fraction that marks rig and flame
    env_dilate_px: int = 5
    min_contrast_c: float = 80.0  # blob peak minus its temporal background (ring if unavailable)
    min_cold_dets: int = 2        # detections in cold space at >= min_contrast_c
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
