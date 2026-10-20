"""All tunable parameters in one place, recorded verbatim in every summary.json.

Every threshold that concerns time is a duration in seconds, converted to frames
per recording by `derived()`, because the corpus mixes frame rates (10, 14.93,
15.15, 15.38, 29.41, 30.3 and 31.25 fps) within the same treatment cells. A
threshold counted in frames would let the frame rate decide what is counted.
The durations below were chosen so that each conversion rounds the same way
across every rate in the corpus (the ~15 fps group never splits), and they
reproduce the frame counts the pipeline was tuned with at ~30 fps.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

_EPS = 1e-9


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
    v_max_mps: float = 4.0        # search radius for a track that has no velocity yet
    # Unmodelled acceleration (gravity, drag, plume turbulence) as the power spectral
    # density of a continuous white-noise acceleration, m^2/s^3. 13 makes the 1-sigma
    # position spread after a 0.18 s coast equal the drift of an unmodelled g:
    # q = 3 a^2 T / 4 with a = 9.81 m/s^2, T = 0.18 s. Converted per recording to
    # px^2/frame^3 as q / (s^2 fps^3), so the search region grows the same per second
    # at every frame rate.
    accel_psd_m2s3: float = 13.0
    kalman_r_px: float = 2.0      # centroid measurement noise, px
    gate_chi2: float = 13.8       # Mahalanobis gate: chi-square, 2 dof, 99.9 %
    gate_floor_px: float = 8.0    # always accept a detection this close to the prediction
    max_miss_s: float = 0.18      # a confirmed track coasts on its prediction this long
    confirm_span_s: float = 0.05  # detections must span this long before a track is confirmed
    # A solid firebrand does not change size a hundredfold between frames; gas
    # puffs and specks do. Links whose areas differ by more than this are refused
    # (ratio of area + 2 px, so 1-4 px specks are not over-penalised).
    track_area_ratio_max: float = 4.0

    # --- firebrand classification -----------------------------------------
    # The rig (burner tray, frame, panels) and the flame sit at or near the hot
    # threshold for much of a burn, and pixels hovering there flicker into blobs.
    # "Cold space" is everywhere hot in fewer than env_frac of post-ignition frames
    # (and, in short recordings, fewer than env_min_s worth of frames); a firebrand
    # must be seen there, clearly hotter than its surroundings, with a hard edge.
    min_track_span_s: float = 0.08    # a firebrand's detections must span at least this long
    env_frac: float = 0.05            # long-term hot fraction that marks rig and flame
    env_min_s: float = 3.0            # ... but never fewer hot frames than this many seconds
    env_dilate_px: int = 5
    min_contrast_c: float = 80.0      # blob peak minus its temporal background (ring if unavailable)
    min_edge_grad: float = 30.0       # degC per pixel across the blob edge: solid, not gas
    min_cold_s: float = 0.066         # evidence detections must cover this long
    min_disp_m: float = 0.05          # net displacement
    # Erratic motion: the median turning angle between successive displacement steps
    # of straight_dt_s. Random wandering (a centroid hopping between flicker blobs or
    # along a flame front) turns ~90 degrees per step; a firebrand's arc turns little.
    # Measured on a time grid, so it means the same at every frame rate; scored only
    # once the track spans straight_min_s (at least three steps).
    straight_dt_s: float = 0.1
    straight_min_s: float = 0.3
    max_turn_deg: float = 60.0

    # --- kinematics and units ---------------------------------------------
    lsq_window_s: float = 0.165       # duration of each least-squares velocity fit
    launch_fit_s: float = 0.3         # launch velocity: quadratic over the first this many seconds
    lens_hfov_deg: float = 45.0       # FLIR T1040, FOL21 lens
    m_per_px: Optional[float] = None  # override the lens-derived scale

    def window_frames(self, fps: float) -> int:
        """Odd window length in frames, so the frame being processed sits at its centre."""
        n = max(3, int(round(self.persist_window_s * fps)))
        return n if n % 2 else n + 1

    def max_miss_frames(self, fps: float) -> int:
        return max(1, int(math.ceil(self.max_miss_s * fps - _EPS)))

    def derived(self, fps: float, m_per_px: float) -> dict:
        """The per-recording frame counts and pixel values the thresholds turn into."""
        return {
            "window_frames": self.window_frames(fps),
            "max_miss_frames": self.max_miss_frames(fps),
            "min_hits": max(2, 1 + int(math.ceil(self.confirm_span_s * fps - _EPS))),
            "min_cold_dets": max(1, int(round(self.min_cold_s * fps))),
            "lsq_points": max(3, int(round(self.lsq_window_s * fps))),
            "gate_new_px": self.v_max_mps / (m_per_px * fps),
            "new_track_speed_limit_mps": self.v_max_mps,
            "q_px": self.accel_psd_m2s3 / (m_per_px * m_per_px * fps ** 3),
        }

    def to_dict(self) -> dict:
        return asdict(self)
