"""End to end: synthetic NetCDF -> FTA_CLI run -> outputs -> review batch -> score."""

import csv
import gzip
import json
import os
import re

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

import FTA_CLI  # noqa: E402
from fta_track.review import make_batch, score  # noqa: E402
from tests.test_fta_track import H, W, _scene  # noqa: E402


@pytest.fixture
def recording(tmp_path):
    """A 150-frame recording: a sample lights at frame 10, one ember launches at frame 40."""
    path = tmp_path / "rec.nc"
    ds = nc.Dataset(path, "w")
    ds.createDimension("frame", 150)
    ds.createDimension("y", H)
    ds.createDimension("x", W)
    v = ds.createVariable("temperature", "f4", ("frame", "y", "x"), chunksizes=(10, H, W), zlib=True)
    for i, f in _scene(150, sample_from=10, embers=[(40, 160, 175, 3.0, -5.0, 900.0, -3.0, 2)]):
        v[i] = f
    ds.setncatts({"frame_rate": 30.0, "distance": 2.0, "threshold_degC": 299.0,
                  "emissivity_original": 0.94})
    ds.close()
    return str(path)


def test_run_writes_every_output(recording, tmp_path):
    out = tmp_path / "res"
    assert FTA_CLI.main(["run", recording, "--out", str(out), "--label", "branchlet/X/150kW/Twigs/1",
                         "--set", "m_per_px=0.002"]) == 0
    s = json.load(open(out / "summary.json"))
    assert s["firebrand_count"] == 1
    assert s["ignition_frame"] == 10
    assert s["fps"] == 30.0 and s["m_per_px"] == 0.002
    with open(out / "tracks.csv") as fh:
        rows = list(csv.DictReader(fh))
    fb = [r for r in rows if r["is_firebrand"] == "1"]
    assert len(fb) == 1
    with gzip.open(out / "track_points.csv.gz", "rt") as fh:
        pts = [r for r in csv.DictReader(fh) if r["track_id"] == fb[0]["track_id"]]
    assert len(pts) == int(fb[0]["n_det"])                      # one row per detection
    assert all(float(p["max_c"]) >= 300 for p in pts)          # temperature history present
    coco = json.load(open(out / "labels_coco.json"))
    assert len(coco["annotations"]) == int(fb[0]["n_det"])     # labels for the firebrand only


def test_aggregate_keeps_the_branchlet_class(recording, tmp_path):
    for lab in ("branchlet/E/100kW/Twigs/C1/Rec-1", "branchlet/E/100kW/Twigs/C2/Rec-2"):
        FTA_CLI.main(["run", recording, "--out", str(tmp_path / "results" / lab), "--label", lab,
                      "--set", "m_per_px=0.002", "--stop", "120"])
    table = tmp_path / "counts.csv"
    FTA_CLI.main(["aggregate", str(tmp_path / "results"), "--out", str(table)])
    rows = list(csv.DictReader(open(table)))
    assert sorted(r["class"] for r in rows) == ["C1", "C2"]
    assert all(abs(float(r["duration_s"]) - 120 / 30.0) < 1e-9 for r in rows)   # the span actually run


def test_aggregate_parses_factors(recording, tmp_path):
    out = tmp_path / "results" / "branchlet" / "X" / "150kW" / "Twigs" / "1"
    FTA_CLI.main(["run", recording, "--out", str(out), "--label", "branchlet/X/150kW/Twigs/1",
                  "--set", "m_per_px=0.002"])
    table = tmp_path / "counts.csv"
    assert FTA_CLI.main(["aggregate", str(tmp_path / "results"), "--out", str(table)]) == 0
    row = list(csv.DictReader(open(table)))[0]
    assert (row["fuel_type"], row["species"], row["hrr"], row["structure"]) == ("branchlet", "X", "150kW", "Twigs")
    assert row["firebrand_count"] == "1"


def test_review_page_is_blind_and_scores(recording, tmp_path):
    res = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(res), "--set", "m_per_px=0.002"])
    batch = tmp_path / "batch"
    n = make_batch(str(res), recording, str(batch), "branchlet/X/150kW/Twigs/1")
    assert n >= 1
    html = open(batch / "index.html").read()
    for leak in ("pipeline_firebrand", "reject_reason", "is_firebrand", "features", "n_evidence", "point_data"):
        assert leak not in html
    man = json.load(open(batch / "manifest.json"))
    for c in man["clips"]:
        assert os.path.exists(batch / "clips" / (c["id"] + ".png"))
        assert any(m is not None for m in c["marks"])
        assert len(c["point_data"]) == c["features"]["n_det"]
    for c in man["clips"]:                       # ids are opaque random tokens, not label__track
        assert re.fullmatch(r"[0-9a-f]{12}", c["id"])
    for part in "branchlet/X/150kW/Twigs/1".split("/"):
        assert f'"{part}"' not in html and f"__{part}" not in html
    V = {c["id"]: ("y" if c["pipeline_firebrand"] else "n") for c in man["clips"]}
    vpath = tmp_path / "verdicts.json"
    json.dump({"nonce": man["nonce"], "verdicts": V}, open(vpath, "w"))
    out = score(str(batch), str(vpath))
    p = out["raw_sample"]["precision_of_counted_firebrands"]
    assert p["yes"] == sum(c["pipeline_firebrand"] for c in man["clips"]) and p["no"] == 0
    assert out["answered"] == len(man["clips"]) and out["unanswered"] == 0


def test_score_refuses_verdicts_from_another_batch(recording, tmp_path):
    res = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(res), "--set", "m_per_px=0.002"])
    batch = tmp_path / "batch"
    make_batch(str(res), recording, str(batch), "branchlet/X/150kW/Twigs/1")
    man = json.load(open(batch / "manifest.json"))
    vpath = tmp_path / "verdicts.json"
    json.dump({"nonce": "not-this-batch", "verdicts": {c["id"]: "y" for c in man["clips"]}}, open(vpath, "w"))
    with pytest.raises(ValueError):
        score(str(batch), str(vpath))
    # foreign ids and bad values are reported, not counted
    json.dump({"nonce": man["nonce"], "verdicts": {"zz": "y", man["clips"][0]["id"]: "Y"}}, open(vpath, "w"))
    out = score(str(batch), str(vpath))
    assert out["answered"] == 0 and out["unknown_ids"] == ["zz"] and len(out["invalid_values"]) == 1


def test_weighted_estimates_use_stratum_sizes(tmp_path):
    # two strata in one recording: 100 counted tracks (sample 2: both real) and
    # 1000 rejected too_short tracks (sample 2: one real)
    batch = tmp_path / "b"
    os.makedirs(batch / "clips")
    clips = []
    for i, (stratum, pop, fbflag, reason) in enumerate([("counted", 100, True, ""), ("counted", 100, True, ""),
                                                        ("rejected:too_short", 1000, False, "too_short"),
                                                        ("rejected:too_short", 1000, False, "too_short")]):
        clips.append({"id": f"c{i}", "label": "L", "track_id": i, "pipeline_firebrand": fbflag,
                      "reject_reason": reason, "stratum": stratum, "stratum_population": pop,
                      "stratum_sampled": 2, "fps": 30.0, "frames": [0, 1], "marks": [None, None],
                      "n_tiles": 2, "tile": 192, "cols": 12, "crop_px": [0, 0, 10, 10]})
    json.dump({"nonce": "n1", "clips": clips}, open(batch / "manifest.json", "w"))
    vpath = tmp_path / "v.json"
    json.dump({"nonce": "n1", "verdicts": {"c0": "y", "c1": "y", "c2": "y", "c3": "n"}}, open(vpath, "w"))
    w = score(str(batch), str(vpath))["weighted"]
    assert w["estimated_real_among_counted"] == 100.0
    assert w["estimated_real_among_rejected"] == 500.0
    assert w["estimated_recall_among_confirmed_tracks"] == round(100 / 600, 3)


def test_retrack_reproduces_the_run(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    before = open(out / "tracks.csv").read()
    # a plain retrack keeps the run's own parameters, overrides included
    assert FTA_CLI.main(["retrack", str(out)]) == 0
    assert open(out / "tracks.csv").read() == before
    s = json.load(open(out / "summary.json"))
    assert s["retracked"] is True and s["firebrand_count"] == 1


def test_retrack_applies_new_thresholds(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    FTA_CLI.main(["retrack", str(out), "--set", "min_contrast_c=5000"])
    s = json.load(open(out / "summary.json"))
    assert s["firebrand_count"] == 0
    assert s["params"]["m_per_px"] == 0.002          # the run's override survived the retrack


def test_retrack_keeps_a_run_override(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002", "--set", "min_contrast_c=5000"])
    FTA_CLI.main(["retrack", str(out)])
    assert json.load(open(out / "summary.json"))["firebrand_count"] == 0


def test_set_rejects_a_missing_value_before_reading_any_data(tmp_path):
    with pytest.raises(SystemExit):
        FTA_CLI.main(["run", str(tmp_path / "does-not-exist.nc"), "--out", str(tmp_path / "o"),
                      "--set", "min_disp_m="])


def test_coco_labels_name_their_images(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--label", "branchlet/X/150kW/Twigs/1",
                  "--set", "m_per_px=0.002"])
    coco = json.load(open(out / "labels_coco.json"))
    assert coco["images"] and all(im["file_name"].startswith("branchlet__X__150kW__Twigs__1/frame_")
                                  for im in coco["images"])
    assert coco["info"]["fps"] == 30.0


def test_retrack_refuses_detection_parameters(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    with pytest.raises(SystemExit):
        FTA_CLI.main(["retrack", str(out), "--set", "hot_thresh_c=350"])


def test_review_merge_combines_batches(recording, tmp_path):
    res = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(res), "--set", "m_per_px=0.002"])
    a, b = tmp_path / "a", tmp_path / "b"
    na = make_batch(str(res), recording, str(a), "branchlet/X/150kW/Twigs/1")
    nb = make_batch(str(res), recording, str(b), "branchlet/Y/150kW/Twigs/2")
    assert FTA_CLI.main(["review-merge", str(tmp_path / "all"), str(a), str(b)]) == 0
    man = json.load(open(tmp_path / "all" / "manifest.json"))
    assert len(man["clips"]) == na + nb
    assert all(os.path.exists(tmp_path / "all" / "clips" / (c["id"] + ".png")) for c in man["clips"])


def test_retrack_from_defaults_resets_tracking_but_keeps_detection(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002",
                  "--set", "min_contrast_c=5000", "--set", "persist_frac=0.2"])
    FTA_CLI.main(["retrack", str(out), "--from-defaults"])
    s = json.load(open(out / "summary.json"))
    assert s["params"]["min_contrast_c"] == 80.0          # back to the default
    assert s["params"]["persist_frac"] == 0.2             # detection parameter kept from the run
    assert s["params"]["m_per_px"] == 0.002               # scale override kept
    assert s["firebrand_count"] == 1
