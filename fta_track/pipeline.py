"""Detection pass and tracking pass.

`detect_stream` reads a recording once, in order, and returns every blob after
ignition plus the long-term hot fraction of each pixel. It is the only stage
that touches the NetCDF, and the only slow one. Memory stays bounded whatever
the recording length: only the frames inside the centred persistence window
are held.

`track_classify` works out the flame zone from the hot fraction, links the
detections outside it into tracks, decides which are firebrands, and gives a
firebrand that left the zone twice one firebrand_id. It needs only the detections
and the hot fraction, so it can be re-run from a recording's saved outputs in
seconds (`FTA_CLI.py retrack`) when a tracking or classification parameter changes.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .background import PersistenceWindow
from .classify import classify, flame_zone, track_features, zone_distance, zone_threshold
from .detect import detect_blobs, detect_weak_blobs, patch_at
from .params import Params
from .track import Tracker
from .units import metres_per_pixel, recording_hfov_deg

Log = Callable[[str], None]


def _stderr(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def detect_stream(frames: Iterable[Tuple[int, np.ndarray]], shape: Tuple[int, int], fps: float,
                  p: Params, log: Log = _stderr) -> Dict:
    win = p.window_frames(fps)
    half = win // 2
    pw = PersistenceWindow(shape, win, p.persist_frac, p.persist_dilate_px)
    pending: deque = deque()
    samples: deque = deque()        # (idx, frame) every `step` frames, for the temporal background
    step = max(1, win // max(1, p.bg_samples))
    dets_by_frame: Dict[int, List[Dict]] = {}
    hot_total = np.zeros(shape, dtype=np.uint32)
    state = {"ignition": None, "n_post": 0, "n_dets": 0, "max_samples": 0}
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
        while samples and samples[0][0] < c - half:
            samples.popleft()
        smp = [f for _, f in samples]
        dets = detect_blobs(frame, hot, mask, p.min_area_px, p.max_area_px, p.ring_px, smp)
        if p.weak_low_c is not None:
            dets += detect_weak_blobs(frame, hot, p.weak_low_c, p.weak_max_area_px, p.weak_near_hot_px,
                                      p.ring_px, p.min_edge_grad, smp)
        if p.patch_px:
            for d in dets:
                d["patch"] = patch_at(frame, d["cx"], d["cy"], p.patch_px)
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
        if idx % step == 0:
            samples.append((idx, np.array(frame, copy=True)))  # a copy: a view would pin its whole chunk
        # Trim here, not only in process(): before ignition process() returns early,
        # and the samples would otherwise pile up for the whole pre-ignition period.
        # Every frame still to be processed is >= idx - half and needs samples from
        # >= its own index - half, so anything older than idx - 2*half is never used.
        while samples and samples[0][0] < idx - 2 * half:
            samples.popleft()
        state["max_samples"] = max(state["max_samples"], len(samples))
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
            "n_post": state["n_post"], "first_frame": first, "last_frame": last, "frames_processed": n_seen,
            "window_frames": win, "max_samples_held": state["max_samples"], "seconds": time.time() - t0}


def _crowding(by_frame: Dict[int, List[Dict]], frame: int, det: Dict, own: set, w: int, r: float) -> float:
    """Other detections within r px of one detection of a track, per frame over +/- w frames.

    Per frame, so the number means the same at every frame rate.
    """
    n = 0
    r2 = r * r
    for c in range(frame - w, frame + w + 1):
        for o in by_frame.get(c, ()):
            if id(o) in own:
                continue
            dx, dy = o["cx"] - det["cx"], o["cy"] - det["cy"]
            if dx * dx + dy * dy <= r2:
                n += 1
    return n / (2 * w + 1)


def _end_state(series: Dict) -> Tuple[float, float, float, float]:
    """Last position (m, Y up) and velocity (m/s) of a track; velocity from the last finite fit."""
    X, Y, vx, vy = series["X"], series["Y"], series["vx"], series["vy"]
    ok = np.isfinite(vx) & np.isfinite(vy)
    if ok.any():
        k = int(np.nonzero(ok)[0][-1])
        return float(X[-1]), float(Y[-1]), float(vx[k]), float(vy[k])
    return float(X[-1]), float(Y[-1]), 0.0, 0.0


def _stitch(rows: List[Dict], fps: float, s: float, p: Params) -> None:
    """Give every firebrand track a firebrand_id; a track that continues another inherits its id.

    One firebrand becomes two tracks when it is lost for longer than the tracker coasts:
    it dips back into the flame zone, or drops below the detection threshold in clear air.
    A counted track b continues a counted track a when b starts at most stitch_gap_s after a
    ended, where a was heading: within stitch_tol_m plus half of g times the gap squared of
    a's end position carried forward by its end velocity and gravity. The tolerance is set by
    physics, not by v_max, so two different firebrands leaving the zone at the same time in
    different places are never merged. Each track hands its id on at most once, to the
    closest match.
    """
    g = 9.81
    fb = sorted((r for r in rows if r["is_firebrand"]), key=lambda r: (r["start_frame"], r["track_id"]))
    ends = {r["track_id"]: _end_state(r["_series"]) for r in fb}
    taken = set()
    next_id = 1
    for i, b in enumerate(fb):
        best = None
        bX, bY = float(b["_series"]["X"][0]), float(b["_series"]["Y"][0])
        for a in fb[:i]:
            if a["track_id"] in taken or a["end_frame"] >= b["start_frame"]:
                continue
            dt = (b["start_frame"] - a["end_frame"]) / fps
            if dt > p.stitch_gap_s + 1e-9:
                continue
            aX, aY, avx, avy = ends[a["track_id"]]
            px, py = aX + avx * dt, aY + avy * dt - 0.5 * g * dt * dt
            miss = math.hypot(bX - px, bY - py)
            if miss > p.stitch_tol_m + 0.5 * g * dt * dt:
                continue
            if best is None or miss < best[0]:
                best = (miss, a)
        if best is None:
            b["firebrand_id"] = next_id
            next_id += 1
        else:
            taken.add(best[1]["track_id"])
            b["firebrand_id"] = best[1]["firebrand_id"]
    for r in rows:
        r.setdefault("firebrand_id", None)


def track_classify(dets_by_frame: Dict[int, List[Dict]], span: Tuple[int, int], fps: float, s: float,
                   hot_frac: np.ndarray, n_post: int, p: Params) -> Tuple[List[Dict], Dict, np.ndarray]:
    """Track every frame in `span` (inclusive, empty frames too: misses count) and classify.

    span[0] is the ignition frame. Detections inside the flame zone are marked
    in_zone and never tracked. Returns the track rows, the per-recording values the
    thresholds turned into, and the zone itself.
    """
    dv = p.derived(fps, s)
    zone = flame_zone(hot_frac, n_post, fps, p)
    zdist = zone_distance(zone)
    H, W = zone.shape
    outside: Dict[int, List[Dict]] = {}
    for c, dets in dets_by_frame.items():
        for d in dets:
            d.pop("track_id", None)
            d.pop("firebrand_id", None)
            xi = min(W - 1, max(0, int(round(d["cx"]))))
            yi = min(H - 1, max(0, int(round(d["cy"]))))
            d["in_zone"] = bool(zone[yi, xi])
            if not d["in_zone"]:
                outside.setdefault(c, []).append(d)
    # crowding counts hot detections only, so it means the same with or without weak readings
    outside_hot = {c: [d for d in ds if not d.get("weak")] for c, ds in outside.items()}
    tracker = Tracker(dv["gate_new_px"], dv["q_px"], p.kalman_r_px, p.gate_chi2, p.gate_floor_px,
                      dv["max_miss_frames"], dv["min_hits"], p.track_area_ratio_max,
                      p.track_excess_ratio_max, p.hot_thresh_c - p.excess_floor_c)
    for c in range(span[0], span[1] + 1):
        tracker.step(c, outside.get(c, []))
    rows = []
    for t in tracker.flush():
        own = {id(d) for _, d in t.points}
        crowd = [_crowding(outside_hot, f, d, own, dv["crowd_frames"], p.crowd_radius_px) for f, d in t.points]
        ctx = {"zone_dist": zdist, "hot_frac": hot_frac, "ignition_frame": span[0], "crowd": crowd}
        f = track_features(t.points, fps, s, ctx, p, dv)
        reason, fails = classify(f, p, dv)
        f["track_id"] = t.public_id
        f["is_firebrand"] = reason == ""
        f["reject_reason"] = reason
        f["fails"] = fails
        for _, d in t.points:
            d["track_id"] = t.public_id
        rows.append(f)
    _stitch(rows, fps, s, p)
    fid = {r["track_id"]: r["firebrand_id"] for r in rows if r["firebrand_id"] is not None}
    for dets in dets_by_frame.values():
        for d in dets:
            if d.get("track_id") in fid:
                d["firebrand_id"] = fid[d["track_id"]]
    dv["zone_threshold"] = round(zone_threshold(n_post, fps, p), 6)
    dv["zone_share"] = round(float(zone.mean()), 4)
    dv["detections_in_zone"] = int(sum(1 for dets in dets_by_frame.values() for d in dets if d["in_zone"]))
    return rows, dv, zone


def run_frames(frames: Iterable[Tuple[int, np.ndarray]], shape: Tuple[int, int], fps: float,
               distance_m: float, p: Params, log: Log = _stderr, hfov_deg: Optional[float] = None) -> Dict:
    s = p.m_per_px if p.m_per_px else metres_per_pixel(distance_m, hfov_deg or p.lens_hfov_deg, shape[1])
    det = detect_stream(frames, shape, fps, p, log)
    t0 = time.time()
    if det["ignition_frame"] is None:
        tracks, dv, zone = [], p.derived(fps, s), np.zeros(shape, dtype=bool)
        dv["zone_share"] = None
    else:
        tracks, dv, zone = track_classify(det["dets_by_frame"], (det["ignition_frame"], det["last_frame"]),
                                          fps, s, det["hot_frac"], det["n_post"], p)
    all_dets = [d for c in sorted(det["dets_by_frame"]) for d in det["dets_by_frame"][c]]
    patches = None
    if p.patch_px and all_dets:
        patches = np.stack([d.pop("patch") for d in all_dets])
    return {
        "detections": all_dets,
        "patches": patches,
        "tracks": tracks,
        "ignition_frame": det["ignition_frame"],
        "first_frame": det["first_frame"], "last_frame": det["last_frame"],
        "hot_frac": det["hot_frac"],
        "zone": zone,
        "n_post": det["n_post"],
        "m_per_px": s,
        "window_frames": det["window_frames"],
        "derived": dv,
        "max_samples_held": det["max_samples_held"],
        "frames_processed": det["frames_processed"],
        "seconds": det["seconds"] + (time.time() - t0),
    }


def render_qa(frames: Iterable[Tuple[int, np.ndarray]], res: Dict, fps: float, path: str,
              vmax_c: float = 800.0) -> None:
    """Half-resolution overlay video: firebrand tracks green and labelled, other tracks
    orange, untracked blobs grey, the flame zone's outline yellow. A second pass over the
    recording, after tracking."""
    fb = {t["track_id"] for t in res["tracks"] if t["is_firebrand"]}
    outline = None
    if res.get("zone") is not None and res["zone"].any():
        z = cv2.resize(res["zone"].astype(np.uint8), (res["zone"].shape[1] // 2, res["zone"].shape[0] // 2),
                       interpolation=cv2.INTER_NEAREST)
        outline, _ = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if outline is not None:
            cv2.drawContours(img, outline, -1, (0, 255, 255), 1)
        for d in by_frame.get(c, []):
            x, y, bw, bh = d["bbox"]
            tid = d.get("track_id")
            col = (0, 255, 0) if tid in fb else ((0, 165, 255) if tid else (150, 150, 150))
            p0, p1 = (x // 2 - 3, y // 2 - 3), ((x + bw) // 2 + 3, (y + bh) // 2 + 3)
            cv2.rectangle(img, p0, p1, col, 1)
            if tid in fb:
                cv2.putText(img, str(d.get("firebrand_id") or tid), (p1[0] + 2, p0[1] + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
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
        res = run_frames(rec.iter_frames(start, stop), (rec.height, rec.width), rec.fps, rec.distance_m, p,
                         hfov_deg=recording_hfov_deg(rec.attrs, p.lens_hfov_deg))
        if qa:
            _stderr("  rendering qa.mp4")
            render_qa(rec.iter_frames(start, stop), res, rec.fps, os.path.join(out_dir, "qa.mp4"))
    finally:
        rec.close()
    write_outputs(out_dir, res, rec, p, label=label, start=start, stop=stop)
    return res
