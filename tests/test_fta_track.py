"""Synthetic tests for fta_track: scenes with known embers, speeds and cooling rates."""

import math

import numpy as np
import pytest

from fta_track.params import Params
from fta_track.pipeline import run_frames
from fta_track.units import lsq_slope, metres_per_pixel, velocity_series

H, W, FPS, S = 240, 320, 30.0, 0.002  # frame size, frame rate, metres per pixel


def _params(**kw):
    # The synthetic scenes are a few seconds long: no ignition flash to wait out, unless a
    # test asks for one.
    p = Params(m_per_px=S, settle_s=0.0)
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
        assert t["exit_dir_deg"] == pytest.approx(math.degrees(math.atan2(0.30, 0.18)), abs=5)
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
        # creeps along the sample's top edge: never leaves the flame zone
        stay = (40, 140, 176, 0.5, 0.0, 700.0, 0.0, 2)
        res = _run(_scene(80, sample_from=10, embers=[stay]))
        assert not any(t["is_firebrand"] for t in res["tracks"])

    def test_threshold_flicker_on_a_warm_plate_is_rejected(self):
        # A plate at 295 C never reaches the hot threshold, so it is outside the flame
        # zone; a 310 C blob sliding across it is only 15 C above its
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
        ember = [t for t in res["tracks"] if abs(t["exit_x_px"] - 220) < 15 and t["n_det"] >= 10]
        gas = [t for t in res["tracks"] if abs(t["exit_x_px"] - 100) < 15 and t["n_det"] >= 10]
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


def _run_real(frames, fps, **kw):
    from fta_track.params import Params
    return run_frames(frames, (HR, WR), fps, 2.0, Params(m_per_px=SR, settle_s=0.0, **kw), log=lambda m: None)


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
    def test_exit_speed_is_the_speed_at_launch(self, fps):
        res = _run_real(_ballistic(fps), fps)
        fb = [t for t in res["tracks"] if t["is_firebrand"]][0]
        true_speed = math.hypot(0.5, 3.0)
        # the track starts at the first frame the ember is detected, which can be up
        # to one frame after the launch, by when gravity has taken off up to g/fps
        slowest = math.hypot(0.5, 3.0 - G / fps)
        assert slowest * 0.93 <= fb["exit_speed_mps"] <= true_speed * 1.07
        assert fb["exit_fit"] == "quadratic"


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


class TestZoneThreshold:
    def test_short_recordings_use_the_absolute_floor(self):
        from fta_track.classify import zone_threshold
        from fta_track.params import Params
        p = Params()
        assert zone_threshold(1000, 30.0, p) == pytest.approx(0.03)    # 1 s of 1000 frames
        assert zone_threshold(300, 30.0, p) == pytest.approx(0.10)
        assert zone_threshold(18000, 30.0, p) == pytest.approx(0.01)   # 1 % wins in long records
        assert zone_threshold(20, 30.0, p) == pytest.approx(1.0)       # never above every frame


def _flight(fps, x0, y0, v, life_s, drop=(), shift=None, extra=None, seed=5):
    """One ember through open air (no sample in its path), plus a burning sample at the
    bottom for ignition. v = (vx, vy_up) m/s under gravity. shift: frame -> (dx, dy) px
    added to the ember's position; extra: frame -> list of (x, y) 3x3 blobs at 700 C."""
    rng = np.random.default_rng(seed)
    t_launch = 0.5
    n = int(round((t_launch + life_s + 0.3) * fps))
    pos = {}
    for i in range(n):
        f = np.full((HR, WR), 20.0, dtype=np.float32) + rng.normal(0, 0.2, (HR, WR)).astype(np.float32)
        f[680:760, 400:600] = 600.0 + rng.normal(0, 30, (80, 200))
        t = i / fps - t_launch
        if 0 <= t <= life_s:
            x = x0 + v[0] * t / SR
            y = y0 - (v[1] * t - 0.5 * G * t * t) / SR
            pos[i] = (x, y)
            if i not in drop:
                dx, dy = (shift or {}).get(i, (0, 0))
                xi, yi = int(round(x + dx)), int(round(y + dy))
                if 0 <= xi < WR - 3 and 0 <= yi < 660:
                    f[yi:yi + 3, xi:xi + 3] = 800.0
        for (bx, by) in (extra or {}).get(i, []):
            f[int(by):int(by) + 3, int(bx):int(bx) + 3] = 700.0
        yield i, f


def _ember_frames(fps, x0, y0, v, life_s):
    t_launch = 0.5
    return [i for i in range(int(round((t_launch + life_s + 0.3) * fps)))
            if 0 <= i / fps - t_launch <= life_s]


class TestTrackerRateInvariance:
    @pytest.mark.parametrize("v", [(0.5, 3.0), (1.5, 2.0), (2.5, 1.0), (1.0, 0.5)])
    @pytest.mark.parametrize("gap_at", [0.12, 0.25])
    def test_a_coast_inside_max_miss_is_bridged_at_30_fps(self, v, gap_at):
        # 3 missed frames (0.099 s), the longest coast at ~30 fps, at several points of an arc
        fps = 30.3
        frames = _ember_frames(fps, 150, 500, v, 0.6)
        g0 = frames[0] + int(round(gap_at * fps))
        drop = tuple(range(g0, g0 + 3))
        res = _run_real(_flight(fps, 150, 500, v, 0.6, drop=drop), fps)
        long = [t for t in res["tracks"] if t["n_det"] >= 4]
        assert len(long) == 1, [(t["n_det"], t["start_frame"], t["end_frame"]) for t in res["tracks"]]

    @pytest.mark.parametrize("fps", [10.0, 30.3])
    def test_a_flicker_blob_cannot_take_an_established_track_detection(self, fps):
        # The ember flies right at 2 m/s. In frame k a one-frame blob appears 19 px beyond
        # where the ember will be in frame k+1; in frame k+1 the ember's centroid is 6 px
        # off its path (inside its own gate), 13 px from the blob's newborn track.
        v, life = (2.0, 2.0), 0.3 if fps > 20 else 0.6     # stays in view at both rates
        frames = _ember_frames(fps, 100, 200, v, life)
        k = frames[len(frames) // 2]
        t1 = (k + 1) / fps - 0.5
        x1 = 100 + v[0] * t1 / SR
        y1 = 200 - (v[1] * t1 - 0.5 * G * t1 * t1) / SR
        res = _run_real(_flight(fps, 100, 200, v, life, shift={k + 1: (6, 0)}, extra={k: [(x1 + 19, y1)]}), fps)
        longest = max(res["tracks"], key=lambda t: t["n_det"])
        assert longest["n_det"] == len(frames), [(t["n_det"], t["start_frame"]) for t in res["tracks"]]

    def test_a_fast_ember_starts_a_track_at_10_fps(self):
        # 3.5 m/s is 216 px/frame at 10 fps: beyond the old 200 px cap on the first link
        res = _run_real(_flight(10.0, 60, 300, (3.5, 0.8), 0.4), 10.0)
        assert any(t["n_det"] >= 4 for t in res["tracks"]), [(t["n_det"],) for t in res["tracks"]]


class TestMotionStatistics:
    @pytest.mark.parametrize("fps", [10.0, 15.38, 30.3])
    def test_random_wandering_is_erratic_and_arcs_are_not(self, fps):
        from fta_track.units import turning_angle_deg
        rng = np.random.default_rng(11)
        t = np.arange(int(1.0 * fps)) / fps
        passed = 0
        for _ in range(200):          # random-direction hops of 5-25 px, as a flicker-hopping centroid
            steps = rng.uniform(5, 25, len(t) - 1) * SR
            ang = rng.uniform(-np.pi, np.pi, len(t) - 1)
            X = np.concatenate([[0], np.cumsum(steps * np.cos(ang))])
            Y = np.concatenate([[0], np.cumsum(steps * np.sin(ang))])
            turn = turning_angle_deg(t, X, Y, 0.1, 0.3)
            passed += turn is not None and turn <= 60.0
        # measured ~15-30 % for 1 s tracks at every rate (the old 3-step net/path passed 73-75 %)
        assert passed / 200 < 0.35, f"{passed}/200 random walks look like firebrands at {fps} fps"
        # a near-vertical toss at 1.5 m/s with 1 px jitter
        X = 0.05 * t + rng.normal(0, SR, len(t))
        Y = 1.5 * t - 0.5 * G * t * t + rng.normal(0, SR, len(t))
        assert turning_angle_deg(t, X, Y, 0.1, 0.3) <= 60.0

    def test_exit_speed_under_drag_does_not_depend_on_frame_rate(self):
        # quadratic drag, launch 3 m/s at 45 degrees, terminal speed 1 m/s
        from fta_track.units import launch_velocity
        def track(fps, T=0.6):
            k = G / 1.0 ** 2
            dt = 1e-4
            x = y = 0.0
            vx = vy = 3.0 / math.sqrt(2)
            ts, xs, ys = [], [], []
            for i in range(int(T / dt) + 1):
                if i % int(round(1 / (fps * dt))) == 0:
                    ts.append(i * dt); xs.append(x); ys.append(y)
                v = math.hypot(vx, vy)
                vx, vy = vx - k * v * vx * dt, vy - (G + k * v * vy) * dt
                x, y = x + vx * dt, y + vy * dt
            return np.array(ts), np.array(xs), np.array(ys)
        speeds = {}
        for fps in (10.0, 30.3):
            t, X, Y = track(fps)
            vx, vy, *_ = launch_velocity(t, X, Y, 0.3)
            speeds[fps] = math.hypot(vx, vy)
        assert abs(speeds[10.0] - speeds[30.3]) / speeds[30.3] < 0.10, speeds

    @pytest.mark.parametrize("fps", [10.0, 15.38, 30.3])
    def test_smoothness_tells_flicker_hops_from_flights(self, fps):
        from fta_track.units import smoothness_px
        p = Params()

        def rough(f, x, y):
            step = float(np.median(np.hypot(np.diff(x), np.diff(y))))
            return smoothness_px(f, x, y) > max(p.smooth_floor_px, p.smooth_step_ratio * step)

        rng = np.random.default_rng(7)
        # flights of 0.6 s, or 1 s at 10 fps: over 0.4 s with five points a quadratic
        # cannot follow a light flake's hardest braking (the documented 10 fps limit)
        n = int(round((1.0 if fps < 12 else 0.6) * fps))
        f = np.arange(n, dtype=float)
        hops = 0
        for _ in range(200):          # a centroid hopping 5-25 px between flicker blobs
            steps = rng.uniform(5, 25, n - 1)
            ang = rng.uniform(-np.pi, np.pi, n - 1)
            hops += rough(f, np.concatenate([[0], np.cumsum(steps * np.cos(ang))]),
                          np.concatenate([[0], np.cumsum(steps * np.sin(ang))]))
        assert hops / 200 > 0.85, f"only {hops}/200 flicker hops look rough at {fps} fps"
        t = f / fps
        flights = 0
        for _ in range(100):          # free flight with 1-3 px centroid jitter
            jit = rng.uniform(1, 3)
            flights += rough(f, 0.5 * t / SR + rng.normal(0, jit, n),
                             -(3.0 * t - 0.5 * G * t * t) / SR + rng.normal(0, jit, n))
            X, Y = _drag_path(fps, n, 3.0, 1.0)      # light flake: launch 3 m/s, terminal 1 m/s
            flights += rough(f, X / SR + rng.normal(0, jit, n), -Y / SR + rng.normal(0, jit, n))
            flights += rough(f, 0.25 * t / SR + rng.normal(0, jit, n), -0.1 * t / SR + rng.normal(0, jit, n))
        assert flights / 300 < 0.05, f"{flights}/300 real flights look rough at {fps} fps"


def _drag_path(fps, n, v0, vt):
    """Positions (m, y up) of a body launched at v0 m/s at 45 degrees under gravity and
    quadratic drag with terminal speed vt, sampled at fps."""
    k = G / vt ** 2
    dt = 1e-4
    x = y = 0.0
    vx = vy = v0 / math.sqrt(2)
    xs, ys = [], []
    every = 1.0 / fps
    t = next_t = 0.0
    while len(xs) < n:
        if t >= next_t - 1e-12:
            xs.append(x)
            ys.append(y)
            next_t += every
        v = math.hypot(vx, vy)
        vx, vy = vx - k * v * vx * dt, vy - (G + k * v * vy) * dt
        x, y = x + vx * dt, y + vy * dt
        t += dt
    return np.array(xs), np.array(ys)


class TestFlameZone:
    """Detections where the flame is are never tracked; firebrands are measured from its edge."""

    def _flicker(self, n=200, ember=None, seed=4):
        # a sample from frame 10 and, just above it, a band where the flame flickers:
        # every frame six 3x3 blobs at 320 C in random places
        rng = np.random.default_rng(seed)
        for i, f in _scene(n, sample_from=10, embers=[ember] if ember else (), seed=seed):
            if i >= 10:
                for _ in range(6):
                    by, bx = rng.integers(150, 176), rng.integers(130, 188)
                    f[by:by + 3, bx:bx + 3] = np.maximum(f[by:by + 3, bx:bx + 3], 320.0)
            yield i, f

    def test_flicker_inside_the_zone_never_becomes_a_track(self):
        res = _run(self._flicker(), zone_min_s=0.1)
        band = [d for d in res["detections"] if 150 <= d["cy"] <= 178 and 130 <= d["cx"] <= 190]
        assert len(band) > 300, "the scene must actually flicker"
        assert all(d["in_zone"] for d in band)
        for t in res["tracks"]:
            s = t["_series"]
            inside = (s["y_px"] >= 150) & (s["y_px"] <= 178) & (s["x_px"] >= 130) & (s["x_px"] <= 190)
            assert not inside.any(), "a track was built from flicker inside the flame zone"
        assert not any(t["is_firebrand"] for t in res["tracks"])

    def test_a_firebrand_is_measured_from_where_it_leaves_the_zone(self):
        # launched from the sample's top edge, it rises through the flickering band
        ember = (60, 160, 175, 1.0, -5.0, 900.0, -2.0, 2)
        res = _run(self._flicker(ember=ember), zone_min_s=0.1)
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len(fb) == 1, [(t["n_det"], t["reject_reason"], t["exit_y_px"]) for t in res["tracks"]]
        assert fb[0]["exit_y_px"] < 150 and fb[0]["exit_zone_dist_px"] <= 6.0

    def test_the_zone_is_reported(self):
        res = _run(self._flicker(), zone_min_s=0.1)
        assert res["zone"].shape == (H, W) and res["zone"][160, 160] and not res["zone"][20, 20]
        assert 0 < res["derived"]["zone_share"] < 0.5


class TestTrackerGates:
    def _tracker(self, **kw):
        from fta_track.track import Tracker
        args = dict(gate_new_px=10.0, q_px=1e4, r_px=2.0, chi2_gate=13.8, gate_floor_px=8.0,
                    max_misses=3, min_hits=2, area_ratio_max=4.0, excess_ratio_max=5.0, excess_base_c=290.0)
        args.update(kw)
        return Tracker(**args)

    @staticmethod
    def _det(x, y, c=800.0):
        return {"cx": float(x), "cy": float(y), "area": 4, "max_c": c}

    def test_a_coasting_track_cannot_jump_farther_than_v_max_allows(self):
        # huge process noise: the statistical gate alone would accept anything
        tr = self._tracker()
        for k in range(5):
            tr.step(k, [self._det(100 + 2 * k, 100)])
        tr.step(5, [])                                   # one miss
        tr.step(6, [self._det(170, 100)])                # 62 px on: beyond 10 px/frame x 2 + 8
        tr.step(7, [self._det(119, 100)])                # back on course, within reach
        first = tr.flush()[0]
        assert [round(d["cx"]) for _, d in first.points] == [100, 102, 104, 106, 108, 119]

    def test_a_hot_ember_is_not_continued_by_threshold_flicker(self):
        tr = self._tracker(gate_new_px=40.0, q_px=10.0)
        for k in range(4):
            tr.step(k, [self._det(100 + 3 * k, 100, 750.0)])
        tr.step(4, [self._det(112, 100, 305.0)])        # where the ember should be, but 305 C
        first = tr.flush()[0]
        assert all(d["max_c"] > 700 for _, d in first.points)


class TestIgnitionFlash:
    def test_an_ember_in_the_first_seconds_after_ignition_is_not_counted(self):
        ember = (40, 160, 175, 3.0, -5.0, 900.0, -3.0, 2)        # 1 s after the sample lights
        res = _run(_scene(150, sample_from=10, embers=[ember]), settle_s=3.0)
        movers = [t for t in res["tracks"] if t["n_det"] >= 5]
        assert movers and all(t["reject_reason"] == "ignition_flash" for t in movers)


class TestZoneReentry:
    def test_a_firebrand_that_crosses_a_hot_patch_is_counted_once(self):
        # flies right at 5 px/frame through a permanently hot block: tracked before and
        # after it, and both tracks get one firebrand_id
        def frames():
            for i, f in _scene(80, embers=[(2, 20, 100, 5.0, 0.0, 800.0, 0.0, 2)]):
                f[90:112, 140:162] = 650.0
                yield i, f
        res = _run(frames())
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len(fb) == 2, [(t["n_det"], t["reject_reason"], t["exit_x_px"]) for t in res["tracks"]]
        assert fb[0]["firebrand_id"] == fb[1]["firebrand_id"]

    def test_two_firebrands_far_apart_are_two(self):
        def frames():
            for i, f in _scene(80, embers=[(2, 20, 60, 5.0, 0.0, 800.0, 0.0, 2),
                                          (2, 20, 200, 5.0, 0.0, 800.0, 0.0, 2)]):
                yield i, f
        res = _run(frames(), ignition_min_px=1)
        fb = [t for t in res["tracks"] if t["is_firebrand"]]
        assert len({t["firebrand_id"] for t in fb}) == 2
