"""End to end: synthetic NetCDF -> FTA_CLI run -> outputs -> review batch -> score."""

import csv
import gzip
import json
import os

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
    V = {c["id"]: ("y" if c["pipeline_firebrand"] else "n") for c in man["clips"]}
    vpath = tmp_path / "verdicts.json"
    json.dump({"batch": "batch", "verdicts": V}, open(vpath, "w"))
    out = score(str(batch), str(vpath))
    p = out["precision_of_counted_firebrands"]
    assert p["yes"] == sum(c["pipeline_firebrand"] for c in man["clips"]) and p["no"] == 0


def test_retrack_reproduces_the_run(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    before = open(out / "tracks.csv").read()
    assert FTA_CLI.main(["retrack", str(out), "--set", "m_per_px=0.002"]) == 0
    assert open(out / "tracks.csv").read() == before
    s = json.load(open(out / "summary.json"))
    assert s["retracked"] is True and s["firebrand_count"] == 1


def test_retrack_applies_new_thresholds(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    FTA_CLI.main(["retrack", str(out), "--set", "m_per_px=0.002", "--set", "min_contrast_c=5000"])
    assert json.load(open(out / "summary.json"))["firebrand_count"] == 0


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
