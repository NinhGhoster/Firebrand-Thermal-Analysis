"""Write one recording's outputs.

    summary.json          firebrand count, ignition, scale, parameters, code version
    tracks.csv            one row per confirmed track (firebrand or rejected, with reason)
    track_points.csv.gz   trajectory and temperature history: one row per detection of a track
    detections.csv.gz     every detection after ignition, tracked or not
    labels_coco.json      boxes for firebrand tracks only, for training a detector
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import subprocess
from typing import Dict, Optional

TRACK_COLS = [
    "track_id", "is_firebrand", "reject_reason", "n_det",
    "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_s",
    "launch_x_px", "launch_y_px", "launch_x_m", "launch_y_m", "end_x_m", "end_y_m",
    "displacement_m", "path_length_m", "straightness",
    "launch_speed_mps", "launch_dir_deg", "mean_speed_mps", "max_speed_mps", "vert_accel_mps2",
    "peak_c", "mean_c", "cooling_c_per_s", "mean_area_px",
    "start_dist_source_px", "end_dist_source_px",
]
POINT_COLS = ["track_id", "is_firebrand", "frame", "time_s", "x_px", "y_px", "x_m", "y_m",
              "vx_mps", "vy_mps", "speed_mps", "max_c", "mean_c", "area_px"]
DET_COLS = ["frame", "time_s", "track_id", "x_px", "y_px", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "area_px", "max_c", "mean_c", "median_c", "bg_c", "excess_c_px"]


def _fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
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


def write_outputs(out_dir: str, res: Dict, rec, p, label: str = "",
                  start: int = 0, stop: Optional[int] = None) -> None:
    fps = rec.fps
    tracks = res["tracks"]
    fb = [t for t in tracks if t["is_firebrand"]]

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
                w.writerow([t["track_id"], _fmt(t["is_firebrand"]), int(s["frame"][k]), _fmt(float(s["t"][k])),
                            _fmt(float(s["x_px"][k]), 2), _fmt(float(s["y_px"][k]), 2),
                            _fmt(float(s["X"][k])), _fmt(float(s["Y"][k])),
                            _fmt(float(s["vx"][k])), _fmt(float(s["vy"][k])), _fmt(float(s["speed"][k])),
                            _fmt(float(s["max_c"][k]), 2), _fmt(float(s["mean_c"][k]), 2), int(s["area"][k])])

    with gzip.open(os.path.join(out_dir, "detections.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DET_COLS)
        for d in res["detections"]:
            x, y, bw, bh = d["bbox"]
            w.writerow([d["frame"], _fmt(d["frame"] / fps), d.get("track_id", ""),
                        _fmt(d["cx"], 2), _fmt(d["cy"], 2), x, y, bw, bh, d["area"],
                        _fmt(d["max_c"], 2), _fmt(d["mean_c"], 2), _fmt(d["median_c"], 2),
                        _fmt(d["bg_c"], 2), _fmt(d["excess"], 1)])

    fb_ids = {t["track_id"] for t in fb}
    images, anns, seen = [], [], {}
    for d in res["detections"]:
        if d.get("track_id") not in fb_ids:
            continue
        fr = d["frame"]
        if fr not in seen:
            seen[fr] = len(images) + 1
            images.append({"id": seen[fr], "frame": fr, "width": rec.width, "height": rec.height})
        x, y, bw, bh = d["bbox"]
        anns.append({"id": len(anns) + 1, "image_id": seen[fr], "category_id": 1,
                     "bbox": [x, y, bw, bh], "area": d["area"], "iscrowd": 0,
                     "track_id": d["track_id"]})
    with open(os.path.join(out_dir, "labels_coco.json"), "w") as fh:
        json.dump({"info": {"source": label or rec.path, "note": "thermal frame indices; firebrand tracks only"},
                   "images": images, "annotations": anns,
                   "categories": [{"id": 1, "name": "firebrand"}]}, fh)

    reasons: Dict[str, int] = {}
    for t in tracks:
        if t["reject_reason"]:
            reasons[t["reject_reason"]] = reasons.get(t["reject_reason"], 0) + 1
    ign = res["ignition_frame"]
    summary = {
        "label": label,
        "source_file": rec.path,
        "n_frames": rec.n_frames,
        "frames_processed": res["frames_processed"],
        "frame_range": [start, stop if stop is not None else rec.n_frames],
        "fps": fps,
        "width": rec.width, "height": rec.height,
        "distance_m": rec.distance_m,
        "emissivity": rec.emissivity,
        "m_per_px": res["m_per_px"],
        "ignition_frame": ign,
        "ignition_time_s": (ign / fps) if ign is not None else None,
        "ignition_definition": f"first frame with >= {p.ignition_min_px} px >= {p.hot_thresh_c:g} C",
        "firebrand_count": len(fb),
        "confirmed_tracks": len(tracks),
        "rejected_by_reason": reasons,
        "detections": len(res["detections"]),
        "source_region_px": res["source_px"],
        "persist_window_frames": res["window_frames"],
        "gate_new_px": res["gate_new_px"],
        "velocity_note": "projected in-image-plane component; scale exact only at distance_m",
        "temperature_note": "apparent temperature at the recorded emissivity; sub-pixel embers read low",
        "params": p.to_dict(),
        "code_version": code_version(),
        "runtime_s": round(res["seconds"], 1),
        "nc_attrs": rec.attrs,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
