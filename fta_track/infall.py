"""Candidate firebrands falling through the flame zone, for labelling only.

The counted firebrands are those seen outside the flame zone. With the camera framing of
these burns the zone reaches nearly to the bottom of the picture, so a firebrand that
drops straight down stays inside it until it leaves the frame, and is never counted.
Inside the zone temperature alone cannot tell it from flame, but motion might: flame and
hot gas rise, and a falling solid accelerates downwards. This module tracks the detections
*inside* the zone with the same tracker, keeps the smooth tracks that fall, and writes
them as a results directory that `review.make_batch` can render, so that labels can
decide whether "falls like a solid" separates firebrand from flame. Nothing here changes
the counts.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import shutil
from typing import Dict, List, Optional

import numpy as np

from .classify import flame_zone, track_features, zone_distance
from .export import _Staged, _write_tables, resolve_retrack_params
from .track import Tracker
from .units import metres_per_pixel


def _load_detections(results_dir: str) -> Dict[int, List[Dict]]:
    nan = float("nan")
    by: Dict[int, List[Dict]] = {}
    with gzip.open(os.path.join(results_dir, "detections.csv.gz"), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            c = int(r["frame"])
            by.setdefault(c, []).append({
                "frame": c, "cx": float(r["x_px"]), "cy": float(r["y_px"]),
                "bbox": (int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])),
                "area": int(r["area_px"]), "max_c": float(r["max_c"]), "mean_c": float(r["mean_c"]),
                "median_c": float(r["median_c"]),
                "bg_c": float(r["bg_c"]) if r["bg_c"] else nan,
                "bg_t_c": float(r["bg_t_c"]) if r.get("bg_t_c") else nan,
                "edge_grad": float(r["edge_grad"]) if r.get("edge_grad") else nan,
                "excess": float(r["excess_c_px"]) if r["excess_c_px"] else nan,
            })
    return by


def write_infall_candidates(results_dir: str, out_dir: str, overrides: Optional[Dict] = None,
                            min_drop_m: float = 0.05, limit: Optional[int] = None) -> int:
    """Track inside the flame zone; write the smooth, falling tracks to out_dir.

    A candidate has at least smooth_points detections, starts settle_s after ignition,
    passes the smoothness test, and ends at least min_drop_m lower than it started.
    out_dir gets tracks.csv, track_points.csv.gz, an empty detections.csv.gz, the
    recording's hot_fraction.npz and a summary.json marked "infall_candidates". Every
    candidate is written with is_firebrand = 1 so make_batch renders all of them as one
    stratum; the flag means "candidate", not "counted".
    """
    p, summary = resolve_retrack_params(results_dir, overrides or {}, from_defaults=True)
    fps = float(summary["fps"])
    s = p.m_per_px if p.m_per_px else metres_per_pixel(summary["distance_m"], p.lens_hfov_deg, summary["width"])
    ign = summary["ignition_frame"]
    os.makedirs(out_dir, exist_ok=True)
    hot = np.load(os.path.join(results_dir, "hot_fraction.npz"))["hot_frac"].astype(np.float32)
    n_post = int(summary.get("post_ignition_frames") or 0)
    rows: List[Dict] = []
    if ign is not None:
        zone = flame_zone(hot, n_post, fps, p)
        H, W = zone.shape
        inside: Dict[int, List[Dict]] = {}
        for c, dets in _load_detections(results_dir).items():
            for d in dets:
                if zone[min(H - 1, max(0, int(round(d["cy"])))), min(W - 1, max(0, int(round(d["cx"]))))]:
                    inside.setdefault(c, []).append(d)
        dv = p.derived(fps, s)
        tr = Tracker(dv["gate_new_px"], dv["q_px"], p.kalman_r_px, p.gate_chi2, p.gate_floor_px,
                     dv["max_miss_frames"], dv["min_hits"], p.track_area_ratio_max,
                     p.track_excess_ratio_max, p.hot_thresh_c - p.excess_floor_c)
        for c in range(ign, summary["last_frame"] + 1):
            tr.step(c, inside.get(c, []))
        zd = zone_distance(~zone)        # here: how deep inside the flame zone each pixel is
        for t in tr.flush():
            if len(t.points) < max(5, p.smooth_points):
                continue
            ctx = {"zone_dist": zd, "hot_frac": hot, "ignition_frame": ign, "crowd": None}
            f = track_features(t.points, fps, s, ctx, p, dv)
            limit_px = max(p.smooth_floor_px, p.smooth_step_ratio * f["median_step_px"])
            drop = f["exit_y_m"] - f["end_y_m"]          # metres fallen (Y is up)
            if (f["start_after_ignition_s"] < p.settle_s or not (f["smooth_px"] <= limit_px)
                    or drop < min_drop_m or f["duration_s"] < p.min_track_span_s):
                continue
            f["track_id"] = t.public_id
            f["firebrand_id"] = None
            f["is_firebrand"] = True
            f["reject_reason"] = ""
            f["fails"] = ""
            rows.append(f)
    rows.sort(key=lambda r: r["start_frame"])
    if limit is not None and len(rows) > limit:
        idx = np.unique(np.linspace(0, len(rows) - 1, limit).round().astype(int))
        rows = [rows[i] for i in idx]
    res = {"tracks": rows, "detections": [], "derived": {}}
    st = _Staged(out_dir)
    try:
        _write_tables(out_dir, res, fps, summary["width"], summary["height"], summary.get("label", ""), st)
        st.commit()
    except BaseException:
        st.abort()
        raise
    shutil.copyfile(os.path.join(results_dir, "hot_fraction.npz"), os.path.join(out_dir, "hot_fraction.npz"))
    out_summary = {**summary, "params": p.to_dict(), "infall_candidates": True,
                   "firebrand_count": None, "candidates": len(rows), "min_drop_m": min_drop_m,
                   "note": "smooth falling tracks inside the flame zone, for labelling; not counted"}
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out_summary, fh, indent=2, default=str)
    return len(rows)

