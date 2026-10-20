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


    def test_flicker_on_a_rail_edge_is_rejected(self):
        # A rail at 298 C with cold air beside it. A 310 C flicker running along its
        # edge looks 290 C hot against the ring (half of it is cold air) but only
        # 12 C above what those pixels usually read.
        def frames():
            for i, f in _scene(150, sample_from=10, embers=[(40, 20, 100, 3.0, 0.0, 310.0, 0.0, 2)]):
                f[100:104, :] = np.maximum(f[100:104, :], 298.0)
                yield i, f
        res = _run(frames())
        movers = [t for t in res["tracks"] if t["displacement_m"] > 0.05]
        assert movers and all(t["reject_reason"] == "low_contrast" for t in movers)
        assert all(t["max_contrast_c"] < 40 for t in movers)


    def test_diffuse_gas_parcel_is_rejected_but_a_solid_ember_is_not(self):
        # Two objects rise through cold air from the sample, same size and speed.
        # The gas parcel is a smooth hump peaking at 340 C that fades into a warm
        # plume; the ember is a hard-edged 600 C block in cold air.
        yy, xx = np.mgrid[0:H, 0:W]
        def frames():
            for i, f in _scene(150, sample_from=10):
                k = i - 40
                if k >= 0:
                    for cx, kind in ((100, "gas"), (220, "ember")):
                        cy = 175 - 5 * k
                        if cy < 20:
                            continue
                        if kind == "gas":
                            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
                            hump = 260.0 + 80.0 * np.exp(-r2 / (2 * 6.0 ** 2))
                            f[:] = np.maximum(f, np.where(r2 < 30 ** 2, hump, f))
                        else:
                            f[cy - 4:cy + 4, cx - 4:cx + 4] = 600.0
                yield i, f
        res = _run(frames())
        ember = [t for t in res["tracks"] if abs(t["launch_x_px"] - 220) < 15 and t["n_det"] >= 10]
        gas = [t for t in res["tracks"] if abs(t["launch_x_px"] - 100) < 15 and t["n_det"] >= 10]
        assert ember and all(t["is_firebrand"] for t in ember)
        assert gas and not any(t["is_firebrand"] for t in gas)
        assert max(t["edge_grad"] for t in gas) < 30 < min(t["edge_grad"] for t in ember)


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

    def test_a_speck_and_a_puff_are_not_linked(self):
        # a 2x2 ember moves right; at frame 25 it vanishes and a 25x25 hot patch
        # appears exactly where it was heading
        def frames():
            for i, f in _scene(45, embers=[(5, 20, 120, 5.0, 0.0, 800.0, 0.0, 2)]):
                if i >= 25:
                    f[:, :] = np.where(f > 300, 20.0, f)
                    x = int(20 + 5.0 * (i - 5))
                    if i < 30:
                        f[108:133, x - 12:x + 13] = 450.0
                yield i, f
        res = _run(frames(), ignition_min_px=1)
        for t in res["tracks"]:
            areas = t["_series"]["area"]
            assert areas.max() / areas.min() < 10, "a speck and a puff were chained into one track"

    def test_short_gap_is_bridged(self):
        # an ember invisible for 3 frames mid-flight stays one track
        def frames():
            for i, f in _scene(60, embers=[(5, 20, 120, 5.0, 0.0, 800.0, 0.0, 2)]):
                if 25 <= i < 28:
                    f[:, :] = 20.0
                yield i, f
        res = _run(frames(), ignition_min_px=1)
        assert len([t for t in res["tracks"] if t["n_det"] >= 10]) == 1


# --- frame-rate independence: real frame size and scale, real physics -----------------
HR, WR, SR, G = 768, 1024, 0.0016180, 9.81


def _ballistic(fps, v=(0.5, 3.0), life_s=0.6, drop=(), seed=3):
    """A sample burning at the bottom, and one 3x3 ember launched from beside it at
    v = (vx, vy_up) m/s that decelerates under gravity. `drop` lists frames where the
    ember is not visible."""
    rng = np.random.default_rng(seed)
    n = int(round((0.5 + life_s + 0.3) * fps))
    t_launch = 0.5
    for i in range(n):
        f = np.full((HR, WR), 20.0, dtype=np.float32) + rng.normal(0, 0.2, (HR, WR)).astype(np.float32)
        f[600:700, 400:600] = 600.0 + rng.normal(0, 30, (100, 200))
        t = i / fps - t_launch
        if 0 <= t <= life_s and i not in drop:
            x = 360 + v[0] * t / SR
            y = 595 - (v[1] * t - 0.5 * G * t * t) / SR
            if 0 <= x < WR - 3 and 0 <= y < HR - 3:
                f[int(round(y)):int(round(y)) + 3, int(round(x)):int(round(x)) + 3] = 800.0
        yield i, f


def _run_real(frames, fps):
    from fta_track.params import Params
    return run_frames(frames, (HR, WR), fps, 2.0, Params(m_per_px=SR), log=lambda m: None)


class TestFrameRateIndependence:
    @pytest.mark.parametrize("fps", [10.0, 15.38, 30.3])
    def test_a_ballistic_ember_is_one_firebrand_at_every_frame_rate(self, fps):
        res = _run_real(_ballistic(fps), fps)
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len(fb) == 1, [(t["n_det"], t["reject_reason"]) for t in res["tracks"]]
        expected = int(0.6 * fps) + 1
        assert fb[0]["n_det"] >= expected - 1, "the ember's track was cut short"

    @pytest.mark.parametrize("fps", [15.38, 30.3])
    def test_a_two_frame_dropout_at_the_apex_does_not_split_the_track(self, fps):
        apex = int(round((0.5 + 3.0 / G) * fps))
        res = _run_real(_ballistic(fps, drop=(apex, apex + 1)), fps)
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len(fb) == 1, [(t["n_det"], t["start_frame"], t["reject_reason"]) for t in res["tracks"]]

    @pytest.mark.parametrize("fps", [10.0, 15.38, 30.3])
    def test_launch_speed_is_the_speed_at_launch(self, fps):
        res = _run_real(_ballistic(fps), fps)
        fb = [t for t in res["tracks"] if t["is_firebrand"]][0]
        true_speed = math.hypot(0.5, 3.0)
        # the track starts at the first frame the ember is detected, which can be up
        # to one frame after the launch, by when gravity has taken off up to g/fps
        slowest = math.hypot(0.5, 3.0 - G / fps)
        assert slowest * 0.93 <= fb["launch_speed_mps"] <= true_speed * 1.07
        assert fb["launch_fit"] == "quadratic"


class TestBoundedMemory:
    def test_background_samples_stay_bounded_before_ignition(self):
        from fta_track.params import Params
        from fta_track.pipeline import detect_stream
        p = Params(m_per_px=S)
        res = detect_stream(_scene(400), (H, W), FPS, p, log=lambda m: None)
        win = p.window_frames(FPS)
        assert res["ignition_frame"] is None
        assert res["max_samples_held"] <= win // (win // p.bg_samples) + 3


class TestReviewRendering:
    def test_a_one_pixel_ember_survives_downsampling(self):
        from fta_track.review import _shrink
        crop = np.full((800, 800), 30.0, dtype=np.float32)
        crop[401, 399] = 650.0
        scale = 192 / 800
        tile = _shrink(crop, 192, 192, scale)
        assert tile.max() == 650.0


class TestColdSpace:
    def test_short_recordings_use_the_absolute_floor(self):
        from fta_track.classify import env_threshold
        from fta_track.params import Params
        p = Params()
        assert env_threshold(300, 30.0, p) == pytest.approx(0.3)      # 3 s of 300 frames
        assert env_threshold(18000, 30.0, p) == pytest.approx(0.05)   # 5 % wins in long records
