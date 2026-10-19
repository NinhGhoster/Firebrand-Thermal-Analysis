"""Stream a recording through mask -> detect -> track -> classify.

Memory stays bounded whatever the recording length: only the frames inside the
centred persistence window are held, so a 28,000-frame recording needs the same
memory as a 300-frame one.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from .background import PersistenceWindow
from .classify import classify, track_features
from .detect import detect_blobs
from .params import Params
from .track import Tracker
from .units import metres_per_pixel


class _QAWriter:
    """Half-resolution overlay video: every detection boxed, confirmed tracks labelled."""

    def __init__(self, path: str, shape: Tuple[int, int], fps: float, vmax_c: float = 800.0):
        h, w = shape
        self.size = (w // 2, h // 2)
        self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)
        self.vmax = vmax_c

    def write(self, frame: np.ndarray, dets, idmap: Dict[int, int], c: int, fps: float) -> None:
        g = (np.clip(frame, 0.0, self.vmax) / self.vmax * 255.0).astype(np.uint8)
        img = cv2.applyColorMap(cv2.resize(g, self.size, interpolation=cv2.INTER_AREA), cv2.COLORMAP_INFERNO)
        for d in dets:
            x, y, w, h = d["bbox"]
            pid = idmap.get(id(d))
            col = (0, 255, 0) if pid else (0, 255, 255)
            p0 = (x // 2 - 3, y // 2 - 3)
            p1 = ((x + w) // 2 + 3, (y + h) // 2 + 3)
            cv2.rectangle(img, p0, p1, col, 1)
            if pid:
                cv2.putText(img, str(pid), (p1[0] + 2, p0[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
        cv2.putText(img, f"frame {c}  t={c / fps:7.2f}s", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.vw.write(img)

    def close(self) -> None:
        self.vw.release()


def run_frames(frames: Iterable[Tuple[int, np.ndarray]], shape: Tuple[int, int], fps: float,
               distance_m: float, p: Params, qa_path: Optional[str] = None,
               log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict:
    H, W = shape
    s = p.m_per_px if p.m_per_px else metres_per_pixel(distance_m, p.lens_hfov_deg, W)
    win = p.window_frames(fps)
    half = win // 2
    gate_new = min(200.0, p.v_max_mps / (s * fps))
    pw = PersistenceWindow(shape, win, p.persist_frac, p.persist_dilate_px)
    tracker = Tracker(gate_new, p.gate_min_px, p.gate_speed_frac, p.kalman_q_px, p.kalman_r_px,
                      p.max_miss_frames(fps), p.min_hits)
    qa = _QAWriter(qa_path, shape, fps) if qa_path else None

    pending: deque = deque()
    all_dets = []
    ignition: Optional[int] = None
    n_seen = 0
    t0 = time.time()

    def process(c: int, frame: np.ndarray, hot: np.ndarray) -> None:
        mask = pw.mask_for(c)
        if ignition is None or c < ignition:
            dets = []
        else:
            dets = detect_blobs(frame, hot, mask, p.min_area_px, p.max_area_px, p.ring_px)
            for d in dets:
                d["frame"] = c
            all_dets.extend(dets)
        tracker.step(c, dets)
        if qa is not None:
            idmap = {id(d): t.public_id for t in tracker.live if t.public_id
                     for f, d in t.points[-1:] if f == c}
            qa.write(frame, dets, idmap, c, fps)

    for idx, frame in frames:
        hot = frame >= p.hot_thresh_c
        if ignition is None and int(hot.sum()) >= p.ignition_min_px:
            ignition = idx
            log(f"  ignition (first frame with >= {p.ignition_min_px} px >= {p.hot_thresh_c:g} C): "
                f"frame {idx}, t = {idx / fps:.2f} s")
        pw.push(idx, hot)
        pending.append((idx, frame, hot))
        while pending and pending[0][0] + half <= idx:
            process(*pending.popleft())
        n_seen += 1
        if n_seen % 2000 == 0:
            rate = n_seen / max(1e-9, time.time() - t0)
            log(f"  frame {idx}: {len(all_dets)} detections, {rate:.0f} frames/s")
    while pending:
        process(*pending.popleft())
    if qa is not None:
        qa.close()

    tracks = tracker.flush()
    src = pw.source
    have_source = bool(src.any())
    if have_source:
        dist = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 5)
    else:
        dist = np.full(shape, np.inf, dtype=np.float32)

    track_rows = []
    for t in tracks:
        f = track_features(t.points, fps, s, H, dist, p)
        reason = classify(f, s, p, have_source)
        f["track_id"] = t.public_id
        f["is_firebrand"] = reason == ""
        f["reject_reason"] = reason
        for _, d in t.points:
            d["track_id"] = t.public_id
        track_rows.append(f)

    return {
        "detections": all_dets,
        "tracks": track_rows,
        "ignition_frame": ignition,
        "m_per_px": s,
        "window_frames": win,
        "gate_new_px": gate_new,
        "source_px": int(src.sum()),
        "frames_processed": n_seen,
        "seconds": time.time() - t0,
    }


def run_recording(path: str, out_dir: str, p: Optional[Params] = None, label: str = "",
                  qa: bool = False, start: int = 0, stop: Optional[int] = None) -> Dict:
    from .export import write_outputs
    from .reader import open_recording
    import os

    p = p or Params()
    rec = open_recording(path)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{label or path}: {rec.n_frames} frames, {rec.width}x{rec.height}, {rec.fps:g} fps, "
          f"D = {rec.distance_m:g} m", file=sys.stderr, flush=True)
    try:
        res = run_frames(rec.iter_frames(start, stop), (rec.height, rec.width), rec.fps,
                         rec.distance_m, p, qa_path=os.path.join(out_dir, "qa.mp4") if qa else None)
    finally:
        rec.close()
    write_outputs(out_dir, res, rec, p, label=label, start=start, stop=stop)
    return res
