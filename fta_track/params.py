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
    # The fastest a firebrand moves in the image plane. It sets the search radius of a
    # track that has no velocity yet, and caps every link: a detection farther from a
    # track's last position than v_max * (time since then) + gate_floor_px is refused,
    # whatever the track's uncertainty says.
    v_max_mps: float = 4.0
    # Unmodelled acceleration (gravity, drag, plume turbulence) as the power spectral
    # density of a continuous white-noise acceleration, m^2/s^3, converted per recording
    # to px^2/frame^3 as q / (s^2 fps^3), so the search region grows the same per second
    # at every frame rate. With 2, after the longest coast (max_miss_s) the 1-sigma spread
    # is ~0.024 m and an unmodelled g drifts ~0.044 m, about 1.8 sigma: inside the
    # 3.7-sigma gate. The earlier 13, with a 0.18 s coast, opened a ~0.7 m search radius,
    # and with 5-20 flame flickers in every frame a coasting track always found one.
    accel_psd_m2s3: float = 2.0
    kalman_r_px: float = 2.0      # centroid measurement noise, px
    gate_chi2: float = 13.8       # Mahalanobis gate: chi-square, 2 dof, 99.9 %
    gate_floor_px: float = 8.0    # always accept a detection this close to the prediction
    max_miss_s: float = 0.095     # a confirmed track coasts this long: 3 frames at ~30 fps, 2 at ~15, 1 at 10
    confirm_span_s: float = 0.05  # detections must span this long before a track is confirmed
    # A solid firebrand does not change size a hundredfold between frames; gas
    # puffs and specks do. Links whose areas differ by more than this are refused
    # (ratio of area + 2 px, so 1-4 px specks are not over-penalised).
    track_area_ratio_max: float = 4.0
    # A solid ember cools smoothly; flicker hovers at the threshold. Links whose
    # temperature excesses (peak minus hot_thresh_c, plus excess_floor_c so a blob at the
    # threshold has 10) differ by more than this ratio are refused: a 600 C ember is never
    # continued by a 305 C flicker (ratio 21), while the pilot's real embers changed by at
    # most 3.4 between frames.
    track_excess_ratio_max: float = 5.0
    excess_floor_c: float = 10.0

    # --- flame zone ----------------------------------------------------------
    # Where the burning sample, the burner and the flame are: pixels >= hot_thresh_c in at
    # least zone_frac of the post-ignition frames (and never fewer than zone_min_s worth of
    # frames, so in a short recording a few firebrands crossing the same pixels cannot turn
    # their own path into flame), widened by zone_dilate_px, with enclosed holes filled.
    # Inside it a firebrand cannot be told from flame by temperature: in the pilot it held
    # 83-99 % of all detections, and tracks there were flame flicker chained together. So
    # detections inside it are not tracked, and every count, trajectory, velocity and
    # temperature history starts where a firebrand leaves the zone.
    zone_frac: float = 0.01
    zone_min_s: float = 1.0
    zone_dilate_px: int = 5

    # --- firebrand classification -----------------------------------------
    # The burner's lighting flash throws off hot blobs before any fuel can have charred.
    settle_s: float = 3.0             # nothing that starts sooner than this after ignition counts
    min_track_span_s: float = 0.15    # a firebrand's detections must span at least this long
    # Smooth motion: each position is predicted from a quadratic through the other
    # detections among the smooth_points nearest it in time (units.smoothness_px); the
    # track's median miss must be at most max(smooth_floor_px, smooth_step_ratio x its
    # median step). The pilot's real firebrands missed by 1-6 px, the old flicker chains
    # by 18-31 px. The step term keeps a fast firebrand's larger centroid scatter from
    # failing it, and means the same at every frame rate: a random hop misses by about its
    # own length, a flight by a small fraction of it. smooth_points is a count, not a
    # duration: a quadratic needs five points to be tested, so at 10 fps a firebrand must
    # be seen for 0.4 s, and a light flake decelerating hard over those 0.4 s can fail.
    smooth_points: int = 5
    smooth_floor_px: float = 8.0
    smooth_step_ratio: float = 0.25
    min_contrast_c: float = 80.0      # blob peak minus its temporal background (ring if unavailable)
    min_edge_grad: float = 30.0       # degC per pixel across the blob edge: solid, not gas
    min_evidence_s: float = 0.066     # detections passing both tests must cover this long
    min_disp_m: float = 0.05          # net displacement
    # Crowding: other detections within crowd_radius_px of each of the track's detections,
    # per frame over +/- crowd_window_s (per frame, so it does not grow with the frame
    # rate); the track's median. Flicker comes in swarms; a firebrand in clear air has few
    # neighbours. Reported always; a test only when max_crowding is set.
    crowd_radius_px: float = 30.0
    crowd_window_s: float = 0.2
    max_crowding: Optional[float] = None
    # Turning angle between successive straight_dt_s steps, reported as turn_deg once a
    # track spans straight_min_s (no longer a test: smooth_max_px replaces it).
    straight_dt_s: float = 0.1
    straight_min_s: float = 0.3
    # Zone re-entries: a firebrand that dips back into the zone's ragged edge and leaves
    # again is two tracks but one firebrand. A counted track that starts within
    # stitch_zone_px of the zone, within stitch_gap_s of a counted track that ended within
    # stitch_zone_px of it, and within that track's reach (v_max * gap + gate_floor_px),
    # takes its firebrand_id.
    stitch_gap_s: float = 0.5
    stitch_zone_px: float = 15.0

    # --- kinematics and units ---------------------------------------------
    lsq_window_s: float = 0.165       # duration of each least-squares velocity fit
    exit_fit_s: float = 0.3           # exit velocity: quadratic over the first this many seconds outside the zone
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
            "min_evidence_dets": max(1, int(round(self.min_evidence_s * fps))),
            "min_dets": max(3, int(self.smooth_points)),
            "crowd_frames": max(1, int(round(self.crowd_window_s * fps))),
            "lsq_points": max(3, int(round(self.lsq_window_s * fps))),
            "gate_new_px": self.v_max_mps / (m_per_px * fps),
            "new_track_speed_limit_mps": self.v_max_mps,
            "q_px": self.accel_psd_m2s3 / (m_per_px * m_per_px * fps ** 3),
        }

    def to_dict(self) -> dict:
        return asdict(self)
