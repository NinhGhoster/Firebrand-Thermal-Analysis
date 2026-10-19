"""Synthetic tests for fta_track: scenes with known embers, speeds and cooling rates."""

import math

import numpy as np
import pytest

from fta_track.params import Params
from fta_track.pipeline import run_frames
from fta_track.units import lsq_slope, metres_per_pixel, velocity_series

H, W, FPS, S = 240, 320, 30.0, 0.002  # frame size, frame rate, metres per pixel


def _params(**kw):
    p = Params(m_per_px=S)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _paint(frame, x, y, temp, size=2):
    xi, yi = int(round(x)), int(round(y))
    frame[yi:yi + size, xi:xi + size] = temp


def _scene(n, sample_from=None, embers=(), seed=0):
    """Yield frames: 20 C background, an optional burning sample, and moving embers.

    Each ember is (start_frame, x0, y0, vx, vy, T0, dT_per_frame, size), in px/frame.
    """
    rng = np.random.default_rng(seed)
    for i in range(n):
        f = np.full((H, W), 20.0, dtype=np.float32) + rng.normal(0, 0.2, (H, W)).astype(np.float32)
        if sample_from is not None and i >= sample_from:
            f[180:230, 130:190] = 600.0 + rng.normal(0, 30, (50, 60))
        for (s0, x0, y0, vx, vy, t0, dt, size) in embers:
            k = i - s0
            if k >= 0:
                x, y, T = x0 + vx * k, y0 + vy * k, t0 + dt * k
                if 0 <= x < W - size and 0 <= y < H - size and T >= 300:
                    _paint(f, x, y, T, size)
        yield i, f


def _run(frames, **kw):
    return run_frames(frames, (H, W), FPS, 2.0, _params(**kw), log=lambda m: None)


class TestUnits:
    def test_scale_formula(self):
        # 2 * 2.0 m * tan(22.5 deg) / 1024 px
        assert metres_per_pixel(2.0, 45.0, 1024) == pytest.approx(0.0016180, rel=1e-4)

    def test_lsq_slope_exact_on_a_line(self):
        t = np.arange(10) / 30.0
        assert lsq_slope(t, 3.0 + 1.5 * t) == pytest.approx(1.5)

    def test_velocity_series_constant_motion(self):
        t = np.arange(12) / 30.0
        vx, vy = velocity_series(t, 0.6 * t, -0.2 * t, 5)
        assert np.allclose(vx, 0.6) and np.allclose(vy, -0.2)


class TestDetection:
    def test_cold_video_has_no_ignition_and_no_detections(self):
        res = _run(_scene(60))
        assert res["ignition_frame"] is None
        assert res["detections"] == [] and res["tracks"] == []

    def test_burning_sample_alone_is_never_a_track(self):
        res = _run(_scene(150, sample_from=10))
        assert res["ignition_frame"] == 10
        assert not any(t["is_firebrand"] for t in res["tracks"])

    def test_nothing_detected_before_ignition(self):
        # a lone hot pixel before the sample lights is below ignition_min_px
        res = _run(_scene(150, sample_from=40, embers=[(5, 20, 20, 0, 0, 900, 0, 1)]))
        assert all(d["frame"] >= 40 for d in res["detections"])


class TestFirebrand:
    # launched from the top edge of the sample: 3 px/frame right, 5 px/frame up, cooling 3 C/frame
    EMBER = (40, 160, 175, 3.0, -5.0, 900.0, -3.0, 2)

    def test_one_ember_is_one_firebrand(self):
        res = _run(_scene(150, sample_from=10, embers=[self.EMBER]))
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len(fb) == 1, [(t["track_id"], t["reject_reason"], t["n_det"]) for t in res["tracks"]]

    def test_velocity_recovered_from_formulas(self):
        res = _run(_scene(150, sample_from=10, embers=[self.EMBER]))
        t = [t for t in res["tracks"] if t["is_firebrand"]][0]
        # vx = s f dx = 0.002*30*3 = 0.18 m/s; vy = -s f dy = 0.30 m/s (up); |v| = 0.35 m/s
        assert t["mean_speed_mps"] == pytest.approx(math.hypot(0.18, 0.30), rel=0.05)
        assert t["launch_dir_deg"] == pytest.approx(math.degrees(math.atan2(0.30, 0.18)), abs=5)
        s = t["_series"]
        assert np.nanmedian(s["vx"]) == pytest.approx(0.18, rel=0.08)
        assert np.nanmedian(s["vy"]) == pytest.approx(0.30, rel=0.08)

    def test_cooling_rate_recovered(self):
        res = _run(_scene(150, sample_from=10, embers=[self.EMBER]))
        t = [t for t in res["tracks"] if t["is_firebrand"]][0]
        assert t["cooling_c_per_s"] == pytest.approx(-3.0 * FPS, rel=0.02)  # -90 C/s
        assert t["peak_c"] <= 900.0

    def test_single_pixel_ember_can_start_a_track(self):
        tiny = (40, 160, 175, 3.0, -5.0, 900.0, -3.0, 1)
        res = _run(_scene(150, sample_from=10, embers=[tiny]))
        assert sum(t["is_firebrand"] for t in res["tracks"]) == 1

    def test_ember_that_stays_on_the_sample_is_rejected(self):
        # creeps along the sample's top edge: never reaches cold space
        stay = (40, 140, 176, 0.5, 0.0, 700.0, 0.0, 2)
        res = _run(_scene(80, sample_from=10, embers=[stay]))
        assert not any(t["is_firebrand"] for t in res["tracks"])

    def test_threshold_flicker_on_a_warm_plate_is_rejected(self):
        # A plate at 295 C never reaches the hot threshold, so it is cold space by
        # the hot map; a 310 C blob sliding across it is only 15 C above its
        # surroundings. This is the rig-edge flicker seen in the real recordings.
        def frames():
            for i, f in _scene(150, sample_from=10, embers=[(40, 20, 40, 3.0, 0.0, 310.0, 0.0, 2)]):
                plate = f[30:60, :]          # full width: the blob never leaves the plate
                plate[plate < 300] = 295.0
                yield i, f
        res = _run(frames())
        movers = [t for t in res["tracks"] if t["displacement_m"] > 0.05]
        assert movers and all(t["reject_reason"] == "low_contrast" for t in movers)


class TestTracker:
    def test_crossing_embers_keep_their_ids(self):
        # paths cross at x ~ 160, passing 3 px apart vertically
        a = (5, 40, 100, 6.0, 0.0, 800.0, 0.0, 2)
        b = (5, 280, 103, -6.0, 0.0, 800.0, 0.0, 2)
        res = _run(_scene(45, embers=[a, b]), ignition_min_px=1)
        long_tracks = [t for t in res["tracks"] if t["n_det"] >= 20]
        assert len(long_tracks) == 2
        for t in long_tracks:
            vx = t["_series"]["vx"]
            assert (np.all(vx[np.isfinite(vx)] > 0) or np.all(vx[np.isfinite(vx)] < 0)), \
                "a track changed direction: IDs swapped at the crossing"

    def test_short_gap_is_bridged(self):
        # an ember invisible for 3 frames mid-flight stays one track
        def frames():
            for i, f in _scene(60, embers=[(5, 20, 120, 5.0, 0.0, 800.0, 0.0, 2)]):
                if 25 <= i < 28:
                    f[:, :] = 20.0
                yield i, f
        res = _run(frames(), ignition_min_px=1)
        assert len([t for t in res["tracks"] if t["n_det"] >= 10]) == 1
