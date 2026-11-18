"""The learned classifier: shared features, training, and its use in tracking and retrack."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

import FTA_CLI  # noqa: E402
from fta_track import learned  # noqa: E402
from tests.test_fta_track_e2e import recording  # noqa: E402,F401
from tests.test_weak_and_patches import _cooling_flight, _run  # noqa: E402


def _constant_model(path, prob, cut):
    """A saved classifier whose probability is `prob` for every track."""
    from sklearn.dummy import DummyClassifier
    n = 100
    y = np.r_[np.ones(int(round(prob * n))), np.zeros(n - int(round(prob * n)))].astype(int)
    m = DummyClassifier(strategy="prior").fit(np.zeros((n, len(learned.FEATURES))), y)
    learned.Classifier(m, cut, np.zeros(len(learned.FEATURES)), {"test": True}).save(str(path))
    learned._cache.clear()
    return str(path)


class TestFeatures:
    def test_a_compact_blob_ends_sharply_and_keeps_its_shape(self):
        p = np.full((15, 15), 30.0, dtype=np.float32)
        p[6:9, 6:9] = 800.0
        f = learned.patch_features([p, p, p])
        prof1, prof2, prof3, ring_std, hot_share, ring_bg, ncc = f
        assert prof1 > 0.5 and prof2 < 0.05 and prof3 == pytest.approx(0.0, abs=1e-6)
        assert ring_std == pytest.approx(0.0, abs=1e-6) and ring_bg == pytest.approx(30.0)
        assert hot_share == pytest.approx(9 / 225) and ncc == pytest.approx(1.0)

    def test_the_vector_has_every_feature_in_order(self):
        f = {k: 1.0 for k in learned.BASE}
        p = np.full((15, 15), 30.0, dtype=np.float32)
        v = learned.feature_vector(f, [400.0, 600.0], [p, p])
        assert v.shape == (len(learned.FEATURES),)
        assert v[learned.FEATURES.index("median_peak_c")] == 500.0

    def test_training_reports_a_leave_one_recording_out_score(self):
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, len(learned.FEATURES)))
        y = (X[:, 0] > 0.5).astype(int)
        groups = np.repeat(np.arange(6), n // 6)
        clf, rep = learned.train(X, y, groups, 0.9)
        assert rep["precision"] >= 0.85 and rep["recall"] > 0.5
        assert 0 < clf.cut <= 1 and len(rep["per_recording"]) == 6


class TestInTracking:
    def test_the_classifier_decides_tracks_that_pass_the_basic_tests(self, tmp_path):
        fps = 30.3
        frames = lambda: _cooling_flight(fps, 0.35, 0.65, v=(1.5, 3.0))
        yes = _run(frames(), fps, weak_low_c=250.0, patch_px=15, classifier=_constant_model(tmp_path / "y.pkl", 0.3, 0.2))
        no = _run(frames(), fps, weak_low_c=250.0, patch_px=15, classifier=_constant_model(tmp_path / "n.pkl", 0.3, 0.5))
        a = [t for t in yes["tracks"] if t["fb_prob"] is not None]
        b = [t for t in no["tracks"] if t["fb_prob"] is not None]
        assert a and all(t["is_firebrand"] and t["fb_prob"] == pytest.approx(0.3) for t in a)
        assert b and not any(t["is_firebrand"] for t in b)
        assert all(t["reject_reason"] == "classifier" and t["fails"] == "classifier" for t in b)
        # a track failing a basic test is never handed to the classifier
        for t in yes["tracks"]:
            if any(x in learned.BASIC_TESTS for x in t["fails"].split(";") if x):
                assert t["fb_prob"] is None and not t["is_firebrand"]

    def test_it_refuses_to_run_without_patches(self, tmp_path):
        fps = 30.3
        with pytest.raises(ValueError, match="patch"):
            _run(_cooling_flight(fps, 0.35, 0.65, v=(1.5, 3.0)), fps,
                 classifier=_constant_model(tmp_path / "m.pkl", 0.3, 0.2))


def test_retrack_reproduces_a_classifier_run(recording, tmp_path):  # noqa: F811
    model = _constant_model(tmp_path / "m.pkl", 0.3, 0.2)
    out = tmp_path / "res"
    FTA_CLI.main(["run", recording, "--out", str(out), "--set", "m_per_px=0.002", "--set", "patch_px=15",
                  "--set", f"classifier={model}"])
    before = open(out / "tracks.csv").read()
    rows = list(csv.DictReader(open(out / "tracks.csv")))
    assert any(r["fb_prob"] for r in rows)
    assert FTA_CLI.main(["retrack", str(out)]) == 0
    assert open(out / "tracks.csv").read() == before
    assert json.load(open(out / "summary.json"))["params"]["classifier"] == model
