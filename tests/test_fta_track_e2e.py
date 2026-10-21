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


N_FRAMES = 220


@pytest.fixture
def recording(tmp_path):
    """A 220-frame recording: a sample lights at frame 10, one ember launches at frame 110,
    past the 3 s the burner's lighting flash is given."""
    path = tmp_path / "rec.nc"
    ds = nc.Dataset(path, "w")
    ds.createDimension("frame", N_FRAMES)
    ds.createDimension("y", H)
    ds.createDimension("x", W)
    v = ds.createVariable("temperature", "f4", ("frame", "y", "x"), chunksizes=(10, H, W), zlib=True)
    for i, f in _scene(N_FRAMES, sample_from=10, embers=[(110, 160, 175, 3.0, -5.0, 900.0, -3.0, 2)]):
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
    html = open(batch / "public" / "index.html").read()
    for leak in ("pipeline_firebrand", "reject_reason", "is_firebrand", "features", "n_evidence", "point_data",
                 "fails", "smooth_px", "stratum"):
        assert leak not in html
    assert '"others"' in html                                  # the neutral dots travel with every clip
    assert "sheets" not in os.listdir(batch / "public")
    assert all(os.path.exists(batch / "sheets" / (c["id"] + ".jpg")) for c in json.load(open(batch / "manifest.json"))["clips"])
    man = json.load(open(batch / "manifest.json"))
    for c in man["clips"]:
        assert os.path.exists(batch / "public" / "clips" / (c["id"] + ".png"))
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
    os.makedirs(batch / "public" / "clips")
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
    assert all(os.path.exists(tmp_path / "all" / "public" / "clips" / (c["id"] + ".png")) for c in man["clips"])


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


def test_aggregate_joins_continuation_segments(recording, tmp_path):
    for lab in ("fibrous_bark/E_radiata/High/150kW/S16a", "fibrous_bark/E_radiata/High/150kW/S16b",
                "fibrous_bark/E_radiata/High/150kW/S4"):
        FTA_CLI.main(["run", recording, "--out", str(tmp_path / "results" / lab), "--label", lab,
                      "--set", "m_per_px=0.002", "--stop", "99999"])
    table = tmp_path / "counts.csv"
    FTA_CLI.main(["aggregate", str(tmp_path / "results"), "--out", str(table)])
    rows = {r["recording"]: r for r in csv.DictReader(open(table))}
    a, b = rows["fibrous_bark/E_radiata/High/150kW/S16a"], rows["fibrous_bark/E_radiata/High/150kW/S16b"]
    assert a["experiment"] == b["experiment"] == "fibrous_bark/E_radiata/High/150kW/S16"
    assert (a["section"], a["segment"], b["segment"]) == ("S16", "a", "b")
    assert rows["fibrous_bark/E_radiata/High/150kW/S4"]["segment"] == ""
    assert float(a["duration_s"]) == pytest.approx(N_FRAMES / 30.0)   # frames read, not the --stop asked for


def test_plain_retrack_of_an_old_summary_is_refused(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    s = json.load(open(out / "summary.json"))
    s.pop("params_hash")
    s["params"]["min_hits"] = 3                      # a field this version no longer has
    json.dump(s, open(out / "summary.json", "w"))
    with pytest.raises(SystemExit):
        FTA_CLI.main(["retrack", str(out)])
    assert FTA_CLI.main(["retrack", str(out), "--from-defaults"]) == 0


def test_an_interrupted_retrack_leaves_the_detections_intact(recording, tmp_path, monkeypatch):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])
    before = {n: open(out / n, "rb").read() for n in ("detections.csv.gz", "summary.json", "tracks.csv")}
    import fta_track.export as ex
    def boom(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(ex, "_write_summary", boom)
    with pytest.raises(KeyboardInterrupt):
        ex.retrack(str(out), {"min_contrast_c": 90.0})
    for n, data in before.items():
        assert open(out / n, "rb").read() == data, f"{n} changed by an interrupted retrack"
    assert not [n for n in os.listdir(out) if ".tmp-" in n], "temporary files left behind"


def _two_label_batch(recording, tmp_path):
    res = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(res), "--set", "m_per_px=0.002"])
    a, b = tmp_path / "a", tmp_path / "b"
    make_batch(str(res), recording, str(a), "branchlet/X/150kW/Twigs/1")
    make_batch(str(res), recording, str(b), "branchlet/Y/150kW/Twigs/2")
    return res, a, b


def test_the_public_folder_holds_no_answer_key_and_no_telling_timestamps(recording, tmp_path):
    from fta_track.review import merge_batches
    res, a, b = _two_label_batch(recording, tmp_path)
    out = tmp_path / "all"
    merge_batches(str(out), [str(a), str(b)])
    pub = out / "public"
    names = set(os.listdir(pub))
    assert names == {"index.html", "clips"}
    assert os.path.exists(out / "manifest.json")
    mtimes = {os.stat(pub / "clips" / n).st_mtime for n in os.listdir(pub / "clips")}
    assert len(mtimes) == 1, "clip timestamps vary, so file order could reveal the strata"


def test_re_merging_keeps_the_nonce_and_refuses_duplicates(recording, tmp_path):
    from fta_track.review import merge_batches
    res, a, b = _two_label_batch(recording, tmp_path)
    out = tmp_path / "all"
    merge_batches(str(out), [str(a), str(b)])
    n1 = json.load(open(out / "manifest.json"))["nonce"]
    merge_batches(str(out), [str(a), str(b)])
    assert json.load(open(out / "manifest.json"))["nonce"] == n1
    c = tmp_path / "c"
    make_batch(str(res), recording, str(c), "branchlet/X/150kW/Twigs/1")      # same recording again
    with pytest.raises(ValueError):
        merge_batches(str(tmp_path / "dup"), [str(a), str(c)])


def test_merge_refuses_old_style_clip_ids(recording, tmp_path):
    from fta_track.review import merge_batches
    res, a, b = _two_label_batch(recording, tmp_path)
    man = json.load(open(a / "manifest.json"))
    man["clips"][0]["id"] = "branchlet__X__150kW__Twigs__1__3"
    json.dump(man, open(a / "manifest.json", "w"))
    with pytest.raises(ValueError):
        merge_batches(str(tmp_path / "all"), [str(a), str(b)])


def test_a_failed_re_render_leaves_the_batch_intact(recording, tmp_path):
    res, a, b = _two_label_batch(recording, tmp_path)
    before = json.load(open(a / "manifest.json"))
    with pytest.raises(Exception):
        make_batch(str(res), str(tmp_path / "missing.nc"), str(a), "branchlet/X/150kW/Twigs/1")
    after = json.load(open(a / "manifest.json"))
    assert after["clips"] == before["clips"]
    assert all(os.path.exists(a / "public" / "clips" / (c["id"] + ".png")) for c in after["clips"])


def test_partial_coverage_is_left_out_of_the_weighted_recall(tmp_path):
    # two recordings; B's counted stratum has only an 'unsure' verdict
    batch = tmp_path / "b"
    os.makedirs(batch / "public" / "clips")
    spec = [("A", "counted", 20, True, "", "y"), ("A", "rejected:too_short", 50, False, "too_short", "n"),
            ("B", "counted", 200, True, "", "u"), ("B", "rejected:too_short", 1000, False, "too_short", "y")]
    clips, V = [], {}
    for i, (lab, st, pop, fbf, reason, verdict) in enumerate(spec):
        cid = f"{i:012x}"
        clips.append({"id": cid, "label": lab, "track_id": i, "pipeline_firebrand": fbf, "reject_reason": reason,
                      "stratum": st, "stratum_population": pop, "stratum_sampled": 1, "fps": 30.0,
                      "frames": [0, 1], "marks": [None, None], "n_tiles": 2, "tile": 192, "cols": 12,
                      "crop_px": [0, 0, 10, 10]})
        V[cid] = verdict
    json.dump({"nonce": "n", "clips": clips}, open(batch / "manifest.json", "w"))
    vpath = tmp_path / "v.json"
    json.dump({"nonce": "n", "verdicts": V}, open(vpath, "w"))
    w = score(str(batch), str(vpath))["weighted"]
    assert w["recordings_included"] == ["A"] and "B" in w["recordings_excluded"]
    assert w["estimated_recall_among_confirmed_tracks"] == 1.0      # A alone: 20 real counted, 0 real rejected


def test_partly_and_not_flying_verdicts_are_scored(recording, tmp_path):
    res = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(res), "--set", "m_per_px=0.002"])
    batch = tmp_path / "batch"
    make_batch(str(res), recording, str(batch), "branchlet/X/150kW/Twigs/1", n_firebrand=None)
    man = json.load(open(batch / "manifest.json"))
    counted = [c for c in man["clips"] if c["pipeline_firebrand"]]
    assert counted
    V = {c["id"]: ("p" if c in counted else "g") for c in man["clips"]}
    vpath = tmp_path / "verdicts.json"
    json.dump({"nonce": man["nonce"], "verdicts": V}, open(vpath, "w"))
    out = score(str(batch), str(vpath))
    p = out["raw_sample"]["precision_of_counted_firebrands"]
    assert out["answered"] == len(man["clips"]) and out["invalid_values"] == []
    assert (p["yes"], p["partly"], p["share_yes"], p["share_yes_or_partly"]) == (0, len(counted), 0.0, 1.0)


def test_retrack_keeps_every_detection_and_marks_the_zone(recording, tmp_path):
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002"])

    def rows():
        with gzip.open(out / "detections.csv.gz", "rt") as fh:
            return list(csv.DictReader(fh))
    before = rows()
    assert any(r["in_zone"] == "1" for r in before) and any(r["in_zone"] == "0" for r in before)
    FTA_CLI.main(["retrack", str(out), "--set", "zone_frac=0.5"])
    after = rows()
    assert len(after) == len(before)
    key = lambda r: (r["frame"], r["x_px"], r["y_px"], r["max_c"])
    assert sorted(map(key, after)) == sorted(map(key, before))
    assert os.path.exists(out / "zone.png")
    s = json.load(open(out / "summary.json"))
    assert s["derived"]["zone_threshold"] == 0.5 and "count_note" in s


def test_infall_finds_an_ember_falling_through_the_flame(tmp_path):
    """A flickering flame band with an ember dropping through it: the ember is a candidate."""
    from fta_track.infall import write_infall_candidates
    rng = np.random.default_rng(2)
    path = tmp_path / "fall.nc"
    ds = nc.Dataset(path, "w")
    ds.createDimension("frame", N_FRAMES)
    ds.createDimension("y", H)
    ds.createDimension("x", W)
    v = ds.createVariable("temperature", "f4", ("frame", "y", "x"), chunksizes=(10, H, W), zlib=True)
    for i, f in _scene(N_FRAMES, sample_from=10, embers=[(120, 60, 100, 0.0, 3.0, 800.0, 0.0, 2)]):
        if i >= 10:                                   # flame flickering over a wide band, rows 90-178
            for _ in range(40):
                by, bx = rng.integers(90, 176), rng.integers(20, 300)
                f[by:by + 3, bx:bx + 3] = np.maximum(f[by:by + 3, bx:bx + 3], 320.0)
        v[i] = f
    ds.setncatts({"frame_rate": 30.0, "distance": 2.0, "threshold_degC": 299.0, "emissivity_original": 0.94})
    ds.close()
    res = tmp_path / "res"
    FTA_CLI.main(["run", str(path), "--out", str(res), "--set", "m_per_px=0.002", "--set", "zone_min_s=0.1"])
    n = write_infall_candidates(str(res), str(tmp_path / "infall"), {"m_per_px": 0.002, "zone_min_s": 0.1})
    rows = list(csv.DictReader(open(tmp_path / "infall" / "tracks.csv")))
    assert n == len(rows) >= 1
    assert any(abs(float(r["exit_x_px"]) - 61) < 4 for r in rows), [(r["exit_x_px"], r["exit_y_px"]) for r in rows]
    batch = tmp_path / "batch"
    assert make_batch(str(tmp_path / "infall"), str(path), str(batch), "branchlet/X/150kW/Twigs/1/infall",
                      n_firebrand=None, n_rejected=0) == n
