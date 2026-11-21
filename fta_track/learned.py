"""Learned firebrand classifier: track measurements plus what each blob looks like.

The hand rule (peak >= 450 C, crowding <= 0.4, speed >= 0.4 m/s) finds about 4 in 10 of the
firebrands that blind labellers see among the tracks. A gradient-boosted classifier on the same
track measurements plus the appearance of each blob (a 15 x 15 temperature patch around every
detection) found about 3 in 4 at the same precision on a 20-recording pilot, scored leaving one
recording out. It decides only for tracks that already pass the basic tests (long enough, smooth,
solid evidence, real displacement, after the ignition flash); those tests still reject the rest.

Appearance, per detection, from its patch (median over a track's calibrated detections):
  prof1..3   mean temperature at 1, 2 and 3 px from the centre, as a share of the centre's
             excess over the patch's outer ring: how sharply the blob ends
  ring_std   spread of the outer ring (>= 6 px from the centre): flame surroundings are noisy,
             clear air is flat
  hot_share  share of the patch >= 300 C
  ring_bg    median of the outer ring
  shape_ncc  median correlation of consecutive patches: a solid object keeps its shape

Every feature is computed here, by the same code for training and for use, so the model always
sees what it was trained on.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, Optional, Sequence

import numpy as np

BASE = ["peak_c", "crowding", "mean_speed_mps", "smooth_px", "max_contrast_c", "edge_grad", "mean_area_px",
        "exit_zone_dist_px", "displacement_m", "duration_s", "n_det", "vert_accel_mps2", "median_step_px",
        "exit_hot_frac", "area_jump", "median_peak_c", "n_weak"]
PATCH = ["prof1", "prof2", "prof3", "ring_std", "hot_share", "ring_bg", "shape_ncc"]
FEATURES = BASE + PATCH
# Tests the classifier does not override: a track failing any of them is never a firebrand.
# A model records the set it was trained with (meta["basic_tests"]) and is always used with it;
# this is the default for training. firebrand_v1 kept "rough" among them; firebrand_v2 leaves
# the path's smoothness to the classifier, which found more firebrands at the same precision.
BASIC_TESTS = ("too_short", "ignition_flash", "low_contrast", "too_little_displacement")
V1_BASIC_TESTS = ("too_short", "ignition_flash", "rough", "low_contrast", "too_little_displacement")
MODELS = os.path.join(os.path.dirname(__file__), "models")
BUILTIN = os.path.join(MODELS, "firebrand_v2.pkl")

_size = 15
_yy, _xx = np.mgrid[0:_size, 0:_size]
_RAD = np.hypot(_xx - _size // 2, _yy - _size // 2)


def patch_features(seq: Sequence[np.ndarray]) -> list:
    """The PATCH features of one track from the patches of its calibrated detections, in time order."""
    seq = [np.asarray(p, dtype=np.float32) for p in seq if p is not None]
    if not seq or seq[0].shape != (_size, _size):
        return [float("nan")] * len(PATCH)
    fs = []
    for p in seq:
        bg = float(np.median(p[_RAD >= 6]))
        pk = float(p[_RAD <= 1.5].max())
        span = max(pk - bg, 1.0)
        prof = [float((p[(_RAD > r - 0.5) & (_RAD <= r + 0.5)] - bg).mean() / span) for r in (1, 2, 3)]
        fs.append(prof + [float(p[_RAD >= 6].std()), float((p >= 300).mean()), bg])
    ncc = []
    for a, b in zip(seq, seq[1:]):
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum()))
        if den > 0:
            ncc.append(float((a * b).sum()) / den)
    return [float(v) for v in np.median(np.array(fs), axis=0)] + [float(np.median(ncc)) if ncc else float("nan")]


def feature_vector(f: Dict, hot_max_c: Sequence[float], patches: Sequence[np.ndarray]) -> np.ndarray:
    """FEATURES of one track: f its track features (track_features or a tracks.csv row), hot_max_c
    the peak temperatures of its calibrated detections, patches their patches."""
    def num(k):
        v = f.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v
    hm = np.asarray([float(x) for x in hot_max_c], dtype=float)
    vals = {k: num(k) for k in BASE}
    vals["median_peak_c"] = float(np.median(hm)) if hm.size else float("nan")
    if not math.isfinite(vals["n_weak"]):
        vals["n_weak"] = 0.0
    return np.array([vals[k] for k in BASE] + patch_features(patches), dtype=float)


class Classifier:
    """A trained model, its probability cut-off and its training record."""

    def __init__(self, model, cut: float, medians: np.ndarray, meta: Dict):
        self.model, self.cut, self.medians, self.meta = model, float(cut), np.asarray(medians, float), meta
        self.basic_tests = tuple(meta.get("basic_tests") or V1_BASIC_TESTS)   # v1 predates the field

    def prob(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        X = np.where(np.isnan(X), self.medians, X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str, X_ref: Optional[np.ndarray] = None) -> None:
        """<name>.pkl, plus <name>.json: everything but the fitted trees, and the probabilities of the
        first training rows. With the training table (<name>_training.csv, same row order) the
        json lets load() rebuild the model where the pickle will not load (another numpy)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "cut": self.cut, "medians": self.medians.tolist(),
                         "meta": self.meta, "features": FEATURES}, fh)
        ref = [] if X_ref is None else [float(v) for v in self.prob(np.asarray(X_ref, float)[:_N_REF])]
        with open(os.path.splitext(path)[0] + ".json", "w") as fh:
            json.dump({"cut": self.cut, "medians": self.medians.tolist(), "meta": self.meta,
                       "features": FEATURES, "reference_probs": ref}, fh, indent=1, default=str)


_cache: Dict[str, Classifier] = {}
_N_REF = 25


def _rebuild(path: str) -> Classifier:
    """Refit a model from its json and training table, and check it against the stored probabilities."""
    import csv

    base = os.path.splitext(path)[0]
    with open(base + ".json") as fh:
        d = json.load(fh)
    with open(base + "_training.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    X = np.array([[float(r[k]) if r[k] not in ("", "nan") else np.nan for k in FEATURES] for r in rows])
    y = np.array([int(r["firebrand"]) for r in rows])
    med = np.array(d["medians"], float)
    clf = Classifier(make_model().fit(np.where(np.isnan(X), med, X), y), d["cut"], med, d["meta"])
    ref = np.array(d.get("reference_probs") or [], float)
    if ref.size:
        got = clf.prob(X[:ref.size])
        if float(np.max(np.abs(got - ref))) > 0.02:
            raise ValueError(f"{path}: rebuilt model differs from the trained one "
                             f"(max probability difference {np.max(np.abs(got - ref)):.3f}); retrain it")
    return clf


def load(name: str) -> Classifier:
    """'builtin' (the newest shipped model), a shipped model's name (e.g. 'firebrand_v1'), or a path."""
    shipped = os.path.join(MODELS, f"{name}.pkl")
    path = BUILTIN if name == "builtin" else (shipped if os.path.exists(shipped) else name)
    if path not in _cache:
        try:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
        except Exception:                         # e.g. pickled under another numpy
            if not os.path.exists(os.path.splitext(path)[0] + ".json"):
                raise
            _cache[path] = _rebuild(path)
            return _cache[path]
        if d.get("features") != FEATURES:
            raise ValueError(f"{path} was trained on other features; retrain it (FTA_CLI.py train-classifier)")
        _cache[path] = Classifier(d["model"], d["cut"], np.array(d["medians"], float), d["meta"])
    return _cache[path]


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=10, random_state=0)


def _cutoff(p: np.ndarray, y: np.ndarray, target: float) -> float:
    best = None
    for t in np.unique(p)[::-1]:
        k = p >= t
        if y[k].mean() >= target and (best is None or y[k].sum() > best[1]):
            best = (float(t), int(y[k].sum()))
    return best[0] if best else 1.01


def _oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold probabilities, one fold per group (recording)."""
    p = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = groups == g
        if y[~te].sum() == 0:
            continue
        med = np.nan_to_num(np.nanmedian(X[~te], axis=0))
        a, b = np.where(np.isnan(X[~te]), med, X[~te]), np.where(np.isnan(X[te]), med, X[te])
        p[te] = make_model().fit(a, y[~te]).predict_proba(b)[:, 1]
    return p


def train(X: np.ndarray, y: np.ndarray, groups: np.ndarray, target: float = 0.9,
          meta: Optional[Dict] = None, basic_tests: Sequence[str] = BASIC_TESTS) -> tuple:
    """Fit on everything; return (Classifier, report).

    The report scores the procedure leaving one recording out: for each recording, model and
    cut-off are chosen on the others only (cut-off: most firebrands kept with inner leave-one-
    recording-out precision >= target), then applied once to it. The final cut-off is chosen the
    same way on the pooled out-of-fold probabilities of all recordings.
    """
    import sklearn

    X, y, groups = np.asarray(X, float), np.asarray(y, int), np.asarray(groups)
    keep = np.zeros(len(y), bool)
    per = {}
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        t = _cutoff(_oof(X[tr], y[tr], groups[tr]), y[tr], target)
        med = np.nan_to_num(np.nanmedian(X[tr], axis=0))
        a, b = np.where(np.isnan(X[tr]), med, X[tr]), np.where(np.isnan(X[te]), med, X[te])
        keep[te] = make_model().fit(a, y[tr]).predict_proba(b)[:, 1] >= t
        per[str(g)] = {"kept": int(keep[te].sum()), "firebrands_kept": int(y[te & keep].sum()),
                       "firebrands": int(y[te].sum())}
    report = {"target_precision": target, "candidates": int(len(y)), "firebrands": int(y.sum()),
              "kept": int(keep.sum()), "firebrands_kept": int(y[keep].sum()),
              "recall": round(float(y[keep].sum() / max(1, y.sum())), 3),
              "precision": round(float(y[keep].mean()) if keep.any() else float("nan"), 3),
              "per_recording": per}
    cut = _cutoff(_oof(X, y, groups), y, target)
    med = np.nan_to_num(np.nanmedian(X, axis=0))
    model = make_model().fit(np.where(np.isnan(X), med, X), y)
    m = dict(meta or {}, sklearn_version=sklearn.__version__, leave_one_recording_out=report,
             basic_tests=list(basic_tests))
    return Classifier(model, cut, med, m), report
