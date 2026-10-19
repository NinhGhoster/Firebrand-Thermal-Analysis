"""Write one recording's outputs, and rebuild tracks from them.

    summary.json          firebrand count, ignition, scale, parameters, code version
    tracks.csv            one row per confirmed track (firebrand or rejected, with reason)
    track_points.csv.gz   trajectory and temperature history: one row per detection of a track
    detections.csv.gz     every detection after ignition, tracked or not
    hot_fraction.npz      share of post-ignition frames each pixel was >= the hot threshold
    hot_fraction.png      the same, as a picture (yellow = hot in 50 %+ of frames)
    labels_coco.json      boxes for firebrand tracks only, for training a detector
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np

TRACK_COLS = [
    "track_id", "is_firebrand", "reject_reason", "n_det", "n_cold", "n_evidence", "max_contrast_c",
    "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s",
    "launch_x_px", "launch_y_px", "launch_x_m", "launch_y_m", "end_x_m", "end_y_m",
    "displacement_m", "path_length_m", "straightness",
    "launch_speed_mps", "launch_dir_deg", "mean_speed_mps", "max_speed_mps", "vert_accel_mps2",
    "peak_c", "mean_c", "cooling_c_per_s", "mean_area_px",
]
POINT_COLS = ["track_id", "is_firebrand", "frame", "time_s", "x_px", "y_px", "x_m", "y_m",
              "vx_mps", "vy_mps", "speed_mps", "max_c", "mean_c", "contrast_c", "cold", "area_px"]
DET_COLS = ["frame", "time_s", "track_id", "x_px", "y_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "area_px", "max_c", "mean_c", "median_c", "bg_c", "excess_c_px"]
# Parameters used by the detection pass; changing any of them needs a new `run`.
DETECTION_PARAMS = ("hot_thresh_c", "ignition_min_px", "min_area_px", "max_area_px", "ring_px",
                    "persist_window_s", "persist_frac", "persist_dilate_px")


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


def _write_tables(out_dir: str, res: Dict, fps: float, width: int, height: int, label: str) -> None:
    tracks = res["tracks"]
    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACK_COLS)
        for t in tracks:
            w.writerow([_fmt(t[c]) for c in TRACK_COLS])

    with gzip.open(os.path.join(out_dir, "track_points.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(POINT_COLS)
        for t in tracks:
            s = t["_series"]
            for k in range(t["n_det"]):
                w.writerow([t["track_id"], _fmt(t["is_firebrand"]), int(s["frame"][k]), _fmt(s["t"][k]),
                            _fmt(s["x_px"][k], 2), _fmt(s["y_px"][k], 2), _fmt(s["X"][k]), _fmt(s["Y"][k]),
                            _fmt(s["vx"][k]), _fmt(s["vy"][k]), _fmt(s["speed"][k]),
                            _fmt(s["max_c"][k], 2), _fmt(s["mean_c"][k], 2), _fmt(s["contrast"][k], 2),
                            _fmt(bool(s["cold"][k])), int(s["area"][k])])

    with gzip.open(os.path.join(out_dir, "detections.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DET_COLS)
        for d in res["detections"]:
            x, y, bw, bh = d["bbox"]
            w.writerow([d["frame"], _fmt(d["frame"] / fps), d.get("track_id") or "",
                        _fmt(d["cx"], 2), _fmt(d["cy"], 2), x, y, bw, bh, d["area"],
                        _fmt(d["max_c"], 2), _fmt(d["mean_c"], 2), _fmt(d["median_c"], 2),
                        _fmt(d["bg_c"], 2), _fmt(d["excess"], 1)])

    fb_ids = {t["track_id"] for t in tracks if t["is_firebrand"]}
    images, anns, seen = [], [], {}
    for d in res["detections"]:
        if d.get("track_id") not in fb_ids:
            continue
        fr = d["frame"]
        if fr not in seen:
            seen[fr] = len(images) + 1
            images.append({"id": seen[fr], "frame": fr, "width": width, "height": height})
        x, y, bw, bh = d["bbox"]
        anns.append({"id": len(anns) + 1, "image_id": seen[fr], "category_id": 1,
                     "bbox": [x, y, bw, bh], "area": d["area"], "iscrowd": 0, "track_id": d["track_id"]})
    with open(os.path.join(out_dir, "labels_coco.json"), "w") as fh:
        json.dump({"info": {"source": label, "note": "thermal frame indices; firebrand tracks only"},
                   "images": images, "annotations": anns, "categories": [{"id": 1, "name": "firebrand"}]}, fh)


def _track_summary(res: Dict, p) -> Dict:
    tracks = res["tracks"]
    reasons: Dict[str, int] = {}
    for t in tracks:
        if t["reject_reason"]:
            reasons[t["reject_reason"]] = reasons.get(t["reject_reason"], 0) + 1
    return {
        "firebrand_count": sum(1 for t in tracks if t["is_firebrand"]),
        "confirmed_tracks": len(tracks),
        "rejected_by_reason": reasons,
        "detections": len(res["detections"]),
        "gate_new_px": res["gate_new_px"],
        "cold_space_share": res.get("cold_share"),
        "params": p.to_dict(),
        "tracking_code_version": code_version(),
    }


def write_outputs(out_dir: str, res: Dict, rec, p, label: str = "",
                  start: int = 0, stop: Optional[int] = None) -> None:
    from .classify import cold_space
    res["cold_share"] = round(float(cold_space(res["hot_frac"], p).mean()), 4)
    _write_tables(out_dir, res, rec.fps, rec.width, rec.height, label or rec.path)

    hf = res["hot_frac"].astype(np.float16)
    np.savez_compressed(os.path.join(out_dir, "hot_fraction.npz"), hot_frac=hf)
    pic = cv2.applyColorMap((np.clip(hf.astype(np.float32) / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(out_dir, "hot_fraction.png"), pic)

    ign = res["ignition_frame"]
    fps = rec.fps
    summary = {
        "label": label,
        "source_file": rec.path,
        "n_frames": rec.n_frames,
        "frames_processed": res["frames_processed"],
        "frame_range": [start, stop if stop is not None else rec.n_frames],
        "first_frame": res["first_frame"], "last_frame": res["last_frame"],
        "fps": fps, "width": rec.width, "height": rec.height,
        "distance_m": rec.distance_m, "emissivity": rec.emissivity,
        "m_per_px": res["m_per_px"],
        "ignition_frame": ign,
        "ignition_time_s": (ign / fps) if ign is not None else None,
        "ignition_definition": f"first frame with >= {p.ignition_min_px} px >= {p.hot_thresh_c:g} C",
        "persist_window_frames": res["window_frames"],
        "velocity_note": "projected in-image-plane component; scale exact only at distance_m",
        "temperature_note": "apparent temperature at the recorded emissivity; sub-pixel embers read low",
        **_track_summary(res, p),
        "detection_code_version": code_version(),
        "runtime_s": round(res["seconds"], 1),
        "nc_attrs": getattr(rec, "attrs", {}),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def retrack(out_dir: str, p) -> Dict:
    """Re-run tracking and classification from a recording's saved outputs.

    Reads detections.csv.gz and hot_fraction.npz; never touches the NetCDF.
    Detection-stage parameters cannot change here: they shaped detections.csv.gz.
    """
    from .pipeline import track_classify
    from .units import metres_per_pixel

    with open(os.path.join(out_dir, "summary.json")) as fh:
        summary = json.load(fh)
    old = summary["params"]
    changed = [k for k in DETECTION_PARAMS if k in old and old[k] != getattr(p, k)]
    if changed:
        raise ValueError(f"{', '.join(changed)} shape the detections themselves; re-run `run` to change them")

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
                "excess": float(r["excess_c_px"]) if r["excess_c_px"] else nan,
            })
    hot_frac = np.load(os.path.join(out_dir, "hot_fraction.npz"))["hot_frac"].astype(np.float32)
    fps = float(summary["fps"])
    s = p.m_per_px if p.m_per_px else metres_per_pixel(summary["distance_m"], p.lens_hfov_deg, summary["width"])
    ign = summary["ignition_frame"]
    if ign is None:
        tracks, gate_new = [], min(200.0, p.v_max_mps / (s * fps))
    else:
        tracks, gate_new = track_classify(dets_by_frame, (ign, summary["last_frame"]), fps, s, hot_frac, p)
    from .classify import cold_space
    res = {"tracks": tracks, "detections": [d for c in sorted(dets_by_frame) for d in dets_by_frame[c]],
           "gate_new_px": gate_new, "cold_share": round(float(cold_space(hot_frac, p).mean()), 4)}
    _write_tables(out_dir, res, fps, summary["width"], summary["height"], summary.get("label") or summary["source_file"])
    summary.update(_track_summary(res, p))
    summary["m_per_px"] = s
    summary["retracked"] = True
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return res
