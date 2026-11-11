"""Weak readings (below the calibrated 300 C, position only) and per-detection patches."""

import gzip
import csv
import json

import numpy as np
import pytest

import FTA_CLI
from fta_track.detect import detect_weak_blobs
from fta_track.params import Params
from fta_track.pipeline import run_frames
from tests.test_fta_track import G, HR, SR, WR
from tests.test_fta_track_e2e import recording  # noqa: F401  (fixture)


def _blank(h=120, w=160):
    return np.full((h, w), 20.0, dtype=np.float32)


class TestWeakDetection:
    def test_a_hard_edged_spot_below_the_threshold_is_a_weak_detection(self):
        f = _blank()
        f[50:53, 60:63] = 270.0
        d = detect_weak_blobs(f, f >= 300, 250.0, 50, 2, 3, 30.0)
        assert len(d) == 1 and d[0]["weak"] is True
        assert d[0]["cx"] == pytest.approx(61.0) and d[0]["max_c"] == 270.0

    def test_the_halo_of_a_hot_blob_is_not_a_detection(self):
        f = _blank()
        f[45:58, 55:68] = 270.0           # warm halo ...
        f[49:54, 59:64] = 800.0           # ... around a hot ember
        assert detect_weak_blobs(f, f >= 300, 250.0, 50, 2, 3, 30.0) == []

    def test_diffuse_warm_gas_is_not_a_detection(self):
        f = _blank()
        yy, xx = np.mgrid[0:120, 0:160]
        # a soft-edged parcel: 255-280 C fading over several pixels into 200 C plume
        f[:] = 200.0
        f += 80.0 * np.exp(-((xx - 80) ** 2 + (yy - 60) ** 2) / (2 * 2.0 ** 2))
        assert detect_weak_blobs(f, f >= 300, 250.0, 50, 2, 3, 30.0) == []

    def test_nothing_is_found_when_off_or_out_of_range(self):
        f = _blank()
        f[50:53, 60:63] = 240.0           # below weak_low_c
        assert detect_weak_blobs(f, f >= 300, 250.0, 50, 2, 3, 30.0) == []
        f[10:30, 10:30] = 270.0           # too big for a weak blob
        assert detect_weak_blobs(f, f >= 300, 250.0, 50, 2, 3, 30.0) == []


def _cooling_flight(fps, hot_s, life_s, weak_temp=270.0, v=(1.5, 2.0), x0=150, y0=500, seed=5):
    """An ember that reads 800 C for hot_s, then weak_temp (below 300 C) until life_s."""
    rng = np.random.default_rng(seed)
    t_launch = 0.5
    n = int(round((t_launch + life_s + 0.3) * fps))
    for i in range(n):
        f = np.full((HR, WR), 20.0, dtype=np.float32) + rng.normal(0, 0.2, (HR, WR)).astype(np.float32)
        f[680:760, 400:600] = 600.0 + rng.normal(0, 30, (80, 200))
        t = i / fps - t_launch
        if 0 <= t <= life_s:
            x = x0 + v[0] * t / SR
            y = y0 - (v[1] * t - 0.5 * G * t * t) / SR
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < WR - 3 and 0 <= yi < 660:
                f[yi:yi + 3, xi:xi + 3] = 800.0 if t <= hot_s else weak_temp
        yield i, f


def _run(frames, fps, **kw):
    return run_frames(frames, (HR, WR), fps, 2.0, Params(m_per_px=SR, settle_s=0.0, **kw), log=lambda m: None)


class TestWeakTracking:
    @pytest.mark.parametrize("fps", [15.38, 30.3])
    def test_a_cooling_ember_is_followed_below_the_threshold(self, fps):
        # launched at 3 m/s up, it stays in view to 0.65 s: 0.3 s of flight below 300 C
        off = _run(_cooling_flight(fps, 0.35, 0.65, v=(1.5, 3.0)), fps)
        on = _run(_cooling_flight(fps, 0.35, 0.65, v=(1.5, 3.0)), fps, weak_low_c=250.0)
        a = [t for t in off["tracks"] if t["is_firebrand"]]
        b = [t for t in on["tracks"] if t["is_firebrand"]]
        assert len(a) == 1 and len(b) == 1, [(t["n_det"], t["reject_reason"]) for t in on["tracks"]]
        assert b[0]["start_frame"] == a[0]["start_frame"]
        assert b[0]["end_frame"] > a[0]["end_frame"] + 0.25 * fps     # followed on through the weak part
        assert b[0]["n_weak"] >= 0.25 * fps and a[0]["n_weak"] == 0
        # temperatures come from calibrated readings only
        assert b[0]["peak_c"] == a[0]["peak_c"] == 800.0
        assert b[0]["mean_c"] == pytest.approx(800.0)

    def test_a_short_hot_sighting_counts_once_its_weak_flight_is_seen(self):
        # 4 hot frames at 15 fps are too short to count; followed on below 300 C, it counts
        fps = 15.38
        off = _run(_cooling_flight(fps, 0.25, 0.6), fps)
        on = _run(_cooling_flight(fps, 0.25, 0.6), fps, weak_low_c=250.0)
        assert not any(t["is_firebrand"] for t in off["tracks"])
        assert sum(t["is_firebrand"] for t in on["tracks"]) == 1

    def test_weak_readings_alone_never_make_a_track(self):
        fps = 30.3
        res = _run(_cooling_flight(fps, -1.0, 0.6), fps, weak_low_c=250.0)   # never hot
        assert [t for t in res["tracks"] if t["n_weak"]] == []

    def test_weak_readings_do_not_count_as_evidence(self):
        # 2 hot frames then weak: too little calibrated evidence to count, however long it flies
        fps = 30.3
        res = _run(_cooling_flight(fps, 1.5 / fps, 0.6), fps, weak_low_c=250.0)
        assert not any(t["is_firebrand"] for t in res["tracks"])


class TestPatches:
    def test_each_detection_has_a_patch_centred_on_it(self):
        fps = 30.3
        res = _run(_cooling_flight(fps, 0.25, 0.6), fps, weak_low_c=250.0, patch_px=15)
        P = res["patches"]
        assert P.shape == (len(res["detections"]), 15, 15) and P.dtype == np.float16
        for d, p in zip(res["detections"], P):
            assert float(p.max()) == pytest.approx(d["max_c"], abs=1.0)
            assert float(p[7, 7]) >= 250.0      # the centre pixel is the object

    def test_no_patches_unless_asked(self):
        fps = 30.3
        assert _run(_cooling_flight(fps, 0.25, 0.6), fps)["patches"] is None


def test_weak_and_patch_outputs_survive_a_retrack(recording, tmp_path):  # noqa: F811
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002",
                  "--set", "weak_low_c=250", "--set", "patch_px=15"])
    z = np.load(out / "patches.npz")
    with gzip.open(out / "detections.csv.gz", "rt") as fh:
        rows = list(csv.DictReader(fh))
    assert z["patches"].shape == (len(rows), 15, 15)
    assert [int(r["frame"]) for r in rows] == z["frame"].tolist()
    assert "weak" in rows[0] and "n_weak" in open(out / "tracks.csv").readline()
    before = open(out / "tracks.csv").read()
    assert FTA_CLI.main(["retrack", str(out)]) == 0
    assert open(out / "tracks.csv").read() == before
    with gzip.open(out / "detections.csv.gz", "rt") as fh:
        assert [int(r["frame"]) for r in csv.DictReader(fh)] == z["frame"].tolist()   # patches stay aligned
    assert json.load(open(out / "summary.json"))["params"]["weak_low_c"] == 250.0


def test_old_outputs_retrack_with_weak_readings_off(recording, tmp_path):  # noqa: F811
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    s = json.load(open(out / "summary.json"))
    for k in ("weak_low_c", "weak_max_area_px", "weak_near_hot_px", "patch_px"):
        s["params"].pop(k)                  # as written before these parameters existed
    json.dump(s, open(out / "summary.json", "w"))
    before = open(out / "tracks.csv").read()
    assert FTA_CLI.main(["retrack", str(out)]) == 0                     # plain: the run's values
    assert open(out / "tracks.csv").read() == before
    assert FTA_CLI.main(["retrack", str(out), "--from-defaults"]) == 0
    assert open(out / "tracks.csv").read() == before
