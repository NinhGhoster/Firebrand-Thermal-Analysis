"""Write one recording's outputs, and rebuild tracks from them.

    summary.json          firebrand count, ignition, scale, flame zone, parameters, code version
    tracks.csv            one row per confirmed track (firebrand or rejected, with reason)
    track_points.csv.gz   trajectory and temperature history: one row per detection of a track
    detections.csv.gz     every detection after ignition, tracked or not, and whether in the flame zone
    hot_fraction.npz      share of post-ignition frames each pixel was >= the hot threshold
    hot_fraction.png      the same, as a picture (yellow = hot in 50 %+ of frames)
    zone.png              the hot fraction with the flame zone outlined in red
    labels_coco.json      boxes for firebrand tracks only, for training a detector

A firebrand can be two tracks (it dipped back into the flame zone and left again):
firebrand_count counts firebrand_ids, firebrand_tracks counts tracks.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
from dataclasses import fields
from typing import Dict, List, Optional

import cv2
import numpy as np

TRACK_COLS = [
    "track_id", "firebrand_id", "is_firebrand", "reject_reason", "fails", "n_det", "n_evidence",
    "max_contrast_c", "smooth_px", "median_step_px", "crowding", "start_after_ignition_s",
    "exit_zone_dist_px", "end_zone_dist_px", "exit_hot_frac",
    "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s",
    "exit_x_px", "exit_y_px", "exit_x_m", "exit_y_m", "end_x_m", "end_y_m",
    "displacement_m", "path_length_m", "turn_deg",
    "exit_speed_mps", "exit_dir_deg", "mean_speed_mps", "max_speed_mps", "vert_accel_mps2",
    "peak_c", "mean_c", "cooling_c_per_s", "mean_area_px", "area_jump", "edge_grad",
    "exit_t_s", "exit_fit", "exit_span_s",
]
POINT_COLS = ["track_id", "firebrand_id", "is_firebrand", "frame", "time_s", "x_px", "y_px", "x_m", "y_m",
              "vx_mps", "vy_mps", "speed_mps", "max_c", "mean_c", "contrast_c", "edge_grad",
              "zone_dist_px", "crowding", "area_px"]
DET_COLS = ["frame", "time_s", "track_id", "x_px", "y_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "area_px", "max_c", "mean_c", "median_c", "bg_c", "bg_t_c", "edge_grad", "excess_c_px", "in_zone"]
# Parameters that set the pixel scale: always taken from the run, even with --from-defaults.
SCALE_PARAMS = ("m_per_px", "lens_hfov_deg")
# Parameters used by the detection pass; changing any of them needs a new `run`.
DETECTION_PARAMS = ("hot_thresh_c", "ignition_min_px", "min_area_px", "max_area_px", "ring_px",
                    "bg_samples", "persist_window_s", "persist_frac", "persist_dilate_px")


def _fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return ""
        return f"{v:.{nd}f}"
    return str(v)


def code_version() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--", "."],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ("-dirty" if dirty else "") if sha else "unknown"
    except Exception:
        return "unknown"


class _Staged:
    """Write each output to a temporary name, then move them all into place at once.

    A retrack reads detections.csv.gz and rewrites it; writing it in place would leave
    a truncated file, the only copy of the detection pass, if the job were interrupted.
    Every file is complete before any is replaced, and summary.json is replaced last.
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.pending: List[tuple] = []

    def path(self, name: str) -> str:
        root, ext = os.path.splitext(name)
        if ext == ".gz":
            root, ext2 = os.path.splitext(root)
            ext = ext2 + ext
        tmp = os.path.join(self.out_dir, f"{root}.tmp-{os.getpid()}{ext}")
        self.pending.append((tmp, os.path.join(self.out_dir, name)))
        return tmp

    def commit(self) -> None:
        last = [x for x in self.pending if os.path.basename(x[1]) == "summary.json"]
        for tmp, final in [x for x in self.pending if x not in last] + last:
            os.replace(tmp, final)
        self.pending = []

    def abort(self) -> None:
        for tmp, _ in self.pending:
            if os.path.exists(tmp):
                os.remove(tmp)
        self.pending = []


def _write_tables(out_dir: str, res: Dict, fps: float, width: int, height: int, label: str,
                  st: Optional[_Staged] = None) -> None:
    tracks = res["tracks"]
    st = st or _Staged(out_dir)
    with open(st.path("tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACK_COLS)
        for t in tracks:
            w.writerow([_fmt(t[c]) for c in TRACK_COLS])

    with gzip.open(st.path("track_points.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(POINT_COLS)
        for t in tracks:
            s = t["_series"]
            for k in range(t["n_det"]):
                w.writerow([t["track_id"], _fmt(t.get("firebrand_id")), _fmt(t["is_firebrand"]), int(s["frame"][k]),
                            _fmt(s["t"][k]),
                            _fmt(s["x_px"][k], 2), _fmt(s["y_px"][k], 2), _fmt(s["X"][k]), _fmt(s["Y"][k]),
                            _fmt(s["vx"][k]), _fmt(s["vy"][k]), _fmt(s["speed"][k]),
                            _fmt(s["max_c"][k], 2), _fmt(s["mean_c"][k], 2), _fmt(s["contrast"][k], 2),
                            _fmt(s["edge_grad"][k], 2), _fmt(s["zone_dist"][k], 1), _fmt(s["crowd"][k], 0),
                            int(s["area"][k])])

    with gzip.open(st.path("detections.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DET_COLS)
        for d in res["detections"]:
            x, y, bw, bh = d["bbox"]
            w.writerow([d["frame"], _fmt(d["frame"] / fps), d.get("track_id") or "",
                        _fmt(d["cx"], 2), _fmt(d["cy"], 2), x, y, bw, bh, d["area"],
                        _fmt(d["max_c"], 2), _fmt(d["mean_c"], 2), _fmt(d["median_c"], 2),
                        _fmt(d["bg_c"], 2), _fmt(d.get("bg_t_c"), 2), _fmt(d.get("edge_grad"), 2),
                        _fmt(d["excess"], 1), _fmt(bool(d.get("in_zone", False)))])

    fb_ids = {t["track_id"] for t in tracks if t["is_firebrand"]}
    safe = (label or "recording").replace("/", "__")
    images, anns, seen = [], [], {}
    for d in res["detections"]:
        if d.get("track_id") not in fb_ids:
            continue
        fr = d["frame"]
        if fr not in seen:
            seen[fr] = len(images) + 1
            images.append({"id": seen[fr], "file_name": f"{safe}/frame_{fr:06d}.png", "frame": fr,
                           "width": width, "height": height})
        x, y, bw, bh = d["bbox"]
        anns.append({"id": len(anns) + 1, "image_id": seen[fr], "category_id": 1,
                     "bbox": [x, y, bw, bh], "area": d["area"], "iscrowd": 0, "track_id": d["track_id"],
                     "firebrand_id": d.get("firebrand_id")})
    with open(st.path("labels_coco.json"), "w") as fh:
        json.dump({"info": {"label": label, "fps": fps, "width": width, "height": height,
                            "frame_index": "0-based frame of the Level 1.1 NetCDF",
                            "file_name": "<label with / as __>/frame_<6-digit frame>.png, for extracted frames",
                            "image_ids": "1-based and per file: remap when merging recordings",
                            "note": "thermal frame boxes [x, y, w, h] of firebrand tracks only, outside the "
                                    "flame zone (inside it a firebrand cannot be told from flame)"},
                   "images": images, "annotations": anns, "categories": [{"id": 1, "name": "firebrand"}]}, fh)


def _write_zone_png(path: str, hot_frac: np.ndarray, zone: Optional[np.ndarray]) -> None:
    """The hot fraction (yellow = hot in 50 %+ of frames) with the flame zone outlined in red."""
    pic = cv2.applyColorMap((np.clip(hot_frac.astype(np.float32) / 0.5, 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_VIRIDIS)
    if zone is not None and zone.any():
        cs, _ = cv2.findContours(zone.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(pic, cs, -1, (0, 0, 255), 2)
    cv2.imwrite(path, pic)


def params_hash(d: Dict, keys=None) -> str:
    sel = {k: d[k] for k in sorted(d) if keys is None or k in keys}
    return hashlib.sha1(json.dumps(sel, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _track_summary(res: Dict, p) -> Dict:
    tracks = res["tracks"]
    reasons: Dict[str, int] = {}
    for t in tracks:
        if t["reject_reason"]:
            reasons[t["reject_reason"]] = reasons.get(t["reject_reason"], 0) + 1
    pd = p.to_dict()
    return {
        "firebrand_count": len({t["firebrand_id"] for t in tracks if t["is_firebrand"]}),
        "firebrand_tracks": sum(1 for t in tracks if t["is_firebrand"]),
        "count_note": "firebrands seen leaving the flame zone (summary derived.zone_threshold, zone.png); "
                      "firebrands that stay inside it while in view, such as most that fall straight "
                      "down through the flame, are not counted",
        "confirmed_tracks": len(tracks),
        "rejected_by_reason": reasons,
        "detections": len(res["detections"]),
        "derived": res["derived"],
        "params": pd,
        "params_hash": params_hash(pd),
        "detection_params_hash": params_hash(pd, DETECTION_PARAMS),
        "tracking_code_version": code_version(),
    }


def write_outputs(out_dir: str, res: Dict, rec, p, label: str = "",
                  start: int = 0, stop: Optional[int] = None) -> None:
    st = _Staged(out_dir)
    try:
        _write_tables(out_dir, res, rec.fps, rec.width, rec.height, label or rec.path, st)
        hf = res["hot_frac"].astype(np.float16)
        np.savez_compressed(st.path("hot_fraction.npz"), hot_frac=hf)
        pic = cv2.applyColorMap((np.clip(hf.astype(np.float32) / 0.5, 0, 1) * 255).astype(np.uint8),
                                cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(st.path("hot_fraction.png"), pic)
        _write_zone_png(st.path("zone.png"), hf, res.get("zone"))
        _write_summary(st, _run_summary(res, rec, p, label, start, stop))
        st.commit()
    except BaseException:
        st.abort()
        raise


def _write_summary(st: _Staged, summary: Dict) -> None:
    with open(st.path("summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def _run_summary(res: Dict, rec, p, label: str, start: int, stop: Optional[int]) -> Dict:

    ign = res["ignition_frame"]
    fps = rec.fps
    stop_eff = rec.n_frames if stop is None else min(stop, rec.n_frames)
    summary = {
        "label": label,
        "source_file": rec.path,
        "n_frames": rec.n_frames,
        "frames_processed": res["frames_processed"],
        "frame_range": [start, stop_eff],
        "first_frame": res["first_frame"], "last_frame": res["last_frame"],
        "post_ignition_frames": res["n_post"],
        "fps": fps, "width": rec.width, "height": rec.height,
        "distance_m": rec.distance_m, "emissivity": rec.emissivity,
        "m_per_px": res["m_per_px"],
        "ignition_frame": ign,
        "ignition_time_s": (ign / fps) if ign is not None else None,
        "ignition_definition": f"first frame with >= {p.ignition_min_px} px >= {p.hot_thresh_c:g} C",
        "persist_window_frames": res["window_frames"],
        "max_background_samples_held": res.get("max_samples_held"),
        "velocity_note": "projected in-image-plane component; scale exact only at distance_m",
        "temperature_note": "apparent temperature at the recorded emissivity; sub-pixel embers read low",
        **_track_summary(res, p),
        "detection_code_version": code_version(),
        "runtime_s": round(res["seconds"], 1),
        "nc_attrs": getattr(rec, "attrs", {}),
    }
    summary["short_post_ignition"] = bool(ign is not None and res["n_post"] < p.zone_min_s * fps / p.zone_frac)
    return summary


def resolve_retrack_params(out_dir: str, overrides: Dict, from_defaults: bool = False):
    """The parameters a re-track uses.

    By default the run's own parameters plus explicit overrides: starting from the
    defaults would silently undo any override the run was made with. With
    from_defaults, tracking and classification start from the current defaults
    instead (for re-tracking after the code's defaults change). Either way the
    detection-stage parameters are the run's, because they shaped
    detections.csv.gz; an override that changes one raises ValueError.
    """
    from .params import Params

    with open(os.path.join(out_dir, "summary.json")) as fh:
        summary = json.load(fh)
    known = {f.name for f in fields(Params)}
    raw = summary.get("params", {})
    recorded = {k: v for k, v in raw.items() if k in known}
    dropped = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if not from_defaults and (dropped or missing or "params_hash" not in summary):
        raise ValueError(
            f"{out_dir}: made by an older version of the code (parameters no longer used: "
            f"{', '.join(dropped) or 'none'}; parameters it does not record: {', '.join(missing) or 'none'}). "
            f"A plain retrack would mix the run's values with today's defaults. Use --from-defaults, "
            f"restating any override the run was made with under its current name.")
    for k in DETECTION_PARAMS:
        if k not in recorded:
            raise ValueError(f"{out_dir}: summary.json does not record {k}; re-run `run`")
    changed = [k for k in DETECTION_PARAMS if k in overrides and overrides[k] != recorded[k]]
    if changed:
        raise ValueError(f"{out_dir}: {', '.join(changed)} shape the detections themselves; "
                         f"re-run `run` to change them")
    if from_defaults:
        base = {**Params().to_dict(), **{k: recorded[k] for k in DETECTION_PARAMS}}
        for k in SCALE_PARAMS:                              # a scale override is not a tracking default
            if k in recorded:
                base[k] = recorded[k]
    else:
        base = recorded
    return Params(**{**base, **overrides}), summary


def retrack(out_dir: str, overrides: Optional[Dict] = None, from_defaults: bool = False) -> Dict:
    """Re-run tracking and classification from a recording's saved outputs.

    Reads detections.csv.gz and hot_fraction.npz; never touches the NetCDF. With no
    overrides it reproduces the run exactly.
    """
    from .pipeline import track_classify
    from .units import metres_per_pixel

    p, summary = resolve_retrack_params(out_dir, overrides or {}, from_defaults)
    dets_by_frame: Dict[int, List[Dict]] = {}
    nan = float("nan")
    with gzip.open(os.path.join(out_dir, "detections.csv.gz"), "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            c = int(r["frame"])
            dets_by_frame.setdefault(c, []).append({
                "frame": c, "cx": float(r["x_px"]), "cy": float(r["y_px"]),
                "bbox": (int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])),
                "area": int(r["area_px"]), "max_c": float(r["max_c"]), "mean_c": float(r["mean_c"]),
                "median_c": float(r["median_c"]),
                "bg_c": float(r["bg_c"]) if r["bg_c"] else nan,
                "bg_t_c": float(r["bg_t_c"]) if r.get("bg_t_c") else nan,
                "edge_grad": float(r["edge_grad"]) if r.get("edge_grad") else nan,
                "excess": float(r["excess_c_px"]) if r["excess_c_px"] else nan,
            })
    hot_frac = np.load(os.path.join(out_dir, "hot_fraction.npz"))["hot_frac"].astype(np.float32)
    fps = float(summary["fps"])
    s = p.m_per_px if p.m_per_px else metres_per_pixel(summary["distance_m"], p.lens_hfov_deg, summary["width"])
    ign = summary["ignition_frame"]
    n_post = summary.get("post_ignition_frames")
    if n_post is None and ign is not None:
        n_post = int(summary["last_frame"]) - int(ign) + 1    # outputs from before this was recorded
    n_post = int(n_post or 0)
    if ign is None:
        tracks, dv, zone = [], p.derived(fps, s), None
        dv["zone_share"] = None
    else:
        tracks, dv, zone = track_classify(dets_by_frame, (ign, summary["last_frame"]), fps, s, hot_frac, n_post, p)
    res = {"tracks": tracks, "detections": [d for c in sorted(dets_by_frame) for d in dets_by_frame[c]],
           "derived": dv}
    for stale in ("gate_new_px", "cold_space_share"):       # superseded by derived.*
        summary.pop(stale, None)
    summary.update(_track_summary(res, p))
    summary["m_per_px"] = s
    summary["post_ignition_frames"] = n_post
    summary["short_post_ignition"] = bool(ign is not None and n_post < p.zone_min_s * fps / p.zone_frac)
    summary["retracked"] = True
    st = _Staged(out_dir)
    try:
        _write_tables(out_dir, res, fps, summary["width"], summary["height"],
                      summary.get("label") or summary["source_file"], st)
        _write_zone_png(st.path("zone.png"), hot_frac, zone)
        _write_summary(st, summary)
        st.commit()
    except BaseException:
        st.abort()
        raise
    return res
