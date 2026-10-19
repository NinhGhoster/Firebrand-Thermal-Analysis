"""Detection pass and tracking pass.

`detect_stream` reads a recording once, in order, and returns every blob after
ignition plus the long-term hot fraction of each pixel. It is the only stage
that touches the NetCDF, and the only slow one. Memory stays bounded whatever
the recording length: only the frames inside the centred persistence window
are held.

`track_classify` links detections into tracks and decides which are firebrands.
It needs only the detections and the hot fraction, so it can be re-run from a
recording's saved outputs in seconds (`FTA_CLI.py retrack`) when a tracking or
classification parameter changes.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .background import PersistenceWindow
from .classify import classify, cold_space, track_features
from .detect import detect_blobs
from .params import Params
from .track import Tracker
from .units import metres_per_pixel

Log = Callable[[str], None]


def _stderr(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def detect_stream(frames: Iterable[Tuple[int, np.ndarray]], shape: Tuple[int, int], fps: float,
                  p: Params, log: Log = _stderr) -> Dict:
    win = p.window_frames(fps)
    half = win // 2
    pw = PersistenceWindow(shape, win, p.persist_frac, p.persist_dilate_px)
    pending: deque = deque()
    dets_by_frame: Dict[int, List[Dict]] = {}
    hot_total = np.zeros(shape, dtype=np.uint32)
    state = {"ignition": None, "n_post": 0, "n_dets": 0}
    first = last = None
    n_seen = 0
    t0 = time.time()

    def process(c: int, frame: np.ndarray, hot: np.ndarray) -> None:
        mask = pw.mask_for(c)
        ign = state["ignition"]
        if ign is None or c < ign:
            return
        np.add(hot_total, hot, out=hot_total, casting="unsafe")
        state["n_post"] += 1
        dets = detect_blobs(frame, hot, mask, p.min_area_px, p.max_area_px, p.ring_px)
        if dets:
            for d in dets:
                d["frame"] = c
            dets_by_frame[c] = dets
            state["n_dets"] += len(dets)

    for idx, frame in frames:
        if first is None:
            first = idx
        last = idx
        hot = frame >= p.hot_thresh_c
        if state["ignition"] is None and int(hot.sum()) >= p.ignition_min_px:
            state["ignition"] = idx
            log(f"  ignition (first frame with >= {p.ignition_min_px} px >= {p.hot_thresh_c:g} C): "
                f"frame {idx}, t = {idx / fps:.2f} s")
        pw.push(idx, hot)
        pending.append((idx, frame, hot))
        while pending and pending[0][0] + half <= idx:
            process(*pending.popleft())
        n_seen += 1
        if n_seen % 2000 == 0:
            log(f"  frame {idx}: {state['n_dets']} detections, {n_seen / max(1e-9, time.time() - t0):.0f} frames/s")
    while pending:
        process(*pending.popleft())

    # stored as float16 in hot_fraction.npz; use the same precision here so a
    # re-track from the saved file reproduces this run exactly
    hot_frac = (hot_total / max(1, state["n_post"])).astype(np.float16).astype(np.float32)
    return {"dets_by_frame": dets_by_frame, "ignition_frame": state["ignition"], "hot_frac": hot_frac,
            "first_frame": first, "last_frame": last, "frames_processed": n_seen,
            "window_frames": win, "seconds": time.time() - t0}


def track_classify(dets_by_frame: Dict[int, List[Dict]], span: Tuple[int, int], fps: float, s: float,
                   hot_frac: np.ndarray, p: Params) -> Tuple[List[Dict], float]:
    """Track every frame in `span` (inclusive, empty frames too: misses count) and classify."""
    gate_new = min(200.0, p.v_max_mps / (s * fps))
    tracker = Tracker(gate_new, p.gate_min_px, p.gate_speed_frac, p.kalman_q_px, p.kalman_r_px,
                      p.max_miss_frames(fps), p.min_hits)
    for dets in dets_by_frame.values():
        for d in dets:
            d.pop("track_id", None)
    for c in range(span[0], span[1] + 1):
        tracker.step(c, dets_by_frame.get(c, []))
    cold = cold_space(hot_frac, p)
    rows = []
    for t in tracker.flush():
        f = track_features(t.points, fps, s, cold, p)
        reason = classify(f, p)
        f["track_id"] = t.public_id
        f["is_firebrand"] = reason == ""
        f["reject_reason"] = reason
        for _, d in t.points:
            d["track_id"] = t.public_id
        rows.append(f)
    return rows, gate_new


def run_frames(frames: Iterable[Tuple[int, np.ndarray]], shape: Tuple[int, int], fps: float,
               distance_m: float, p: Params, log: Log = _stderr) -> Dict:
    s = p.m_per_px if p.m_per_px else metres_per_pixel(distance_m, p.lens_hfov_deg, shape[1])
    det = detect_stream(frames, shape, fps, p, log)
    t0 = time.time()
    if det["ignition_frame"] is None:
        tracks, gate_new = [], min(200.0, p.v_max_mps / (s * fps))
    else:
        tracks, gate_new = track_classify(det["dets_by_frame"], (det["ignition_frame"], det["last_frame"]),
                                          fps, s, det["hot_frac"], p)
    all_dets = [d for c in sorted(det["dets_by_frame"]) for d in det["dets_by_frame"][c]]
    return {
        "detections": all_dets,
        "tracks": tracks,
        "ignition_frame": det["ignition_frame"],
        "first_frame": det["first_frame"], "last_frame": det["last_frame"],
        "hot_frac": det["hot_frac"],
        "m_per_px": s,
        "window_frames": det["window_frames"],
        "gate_new_px": gate_new,
        "frames_processed": det["frames_processed"],
        "seconds": det["seconds"] + (time.time() - t0),
    }


def render_qa(frames: Iterable[Tuple[int, np.ndarray]], res: Dict, fps: float, path: str,
              vmax_c: float = 800.0) -> None:
    """Half-resolution overlay video: firebrand tracks green and labelled, other tracks
    orange, untracked blobs grey. A second pass over the recording, after tracking."""
    fb = {t["track_id"] for t in res["tracks"] if t["is_firebrand"]}
    by_frame: Dict[int, List[Dict]] = {}
    for d in res["detections"]:
        by_frame.setdefault(d["frame"], []).append(d)
    vw = None
    for c, frame in frames:
        h, w = frame.shape
        size = (w // 2, h // 2)
        if vw is None:
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        g = (np.clip(frame, 0.0, vmax_c) / vmax_c * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(cv2.resize(g, size, interpolation=cv2.INTER_AREA), cv2.COLORMAP_INFERNO)
        for d in by_frame.get(c, []):
            x, y, bw, bh = d["bbox"]
            tid = d.get("track_id")
            col = (0, 255, 0) if tid in fb else ((0, 165, 255) if tid else (150, 150, 150))
            p0, p1 = (x // 2 - 3, y // 2 - 3), ((x + bw) // 2 + 3, (y + bh) // 2 + 3)
            cv2.rectangle(img, p0, p1, col, 1)
            if tid in fb:
                cv2.putText(img, str(tid), (p1[0] + 2, p0[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.putText(img, f"frame {c}  t={c / fps:7.2f}s", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        vw.write(img)
    if vw is not None:
        vw.release()


def run_recording(path: str, out_dir: str, p: Optional[Params] = None, label: str = "",
                  qa: bool = False, start: int = 0, stop: Optional[int] = None) -> Dict:
    import os

    from .export import write_outputs
    from .reader import open_recording

    p = p or Params()
    rec = open_recording(path)
    os.makedirs(out_dir, exist_ok=True)
    _stderr(f"{label or path}: {rec.n_frames} frames, {rec.width}x{rec.height}, {rec.fps:g} fps, "
            f"D = {rec.distance_m:g} m")
    try:
        res = run_frames(rec.iter_frames(start, stop), (rec.height, rec.width), rec.fps, rec.distance_m, p)
        if qa:
            _stderr("  rendering qa.mp4")
            render_qa(rec.iter_frames(start, stop), res, rec.fps, os.path.join(out_dir, "qa.mp4"))
    finally:
        rec.close()
    write_outputs(out_dir, res, rec, p, label=label, start=start, stop=stop)
    return res
