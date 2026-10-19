# -*- coding: utf-8 -*-
"""
nc_player.py - lightweight NetCDF (.nc) thermal video player with optional
Roboflow "flying tiny white particles" detection overlay.

A minimal, dependency-light viewer for the heavily compressed NetCDF files
produced by the companion SEQ-CSQ-compressor tool (variable
``temperature[frame, y, x]``).  Detections come from the serverless Roboflow
workflow "Find flying tiny white particles" (see roboflow_particles.py) and
are cached per frame, with stable nearest-centroid track IDs.

Usage:
    uv run python nc_player.py [path-to.nc]

Requires ROBOFLOW_API_KEY (export it) for the detection buttons; the player
itself works without it.

Keyboard shortcuts:
    Space      play / pause
    Left / ,   previous frame
    Right / .  next frame
    Home       jump to first frame
    End        jump to last frame
    [ / ]      jump to previous / next detected frame
    H          jump to the hottest frame (scanned lazily)
    D          detect particles on the current frame (cached)
    B          detect particles on the next 10 frames (background thread)
    C          toggle detection backend (Roboflow / CV)
    A          scan the whole video in the background (A again to stop)

The timeline bar below the toolbar shows green ticks for frames that have
detections; click or drag anywhere on it to scrub.

Detection rendering: the raw min-max view is dominated by the hot plume, so
the model misses tiny firebrands.  The detection input is instead rendered
as white ember blobs on black (top-``DET_PERCENTILE`` percentile threshold,
connected blobs of ``DET_MIN_AREA``-``DET_MAX_AREA`` px, upscaled
``DET_UPSCALE``x) - matching the model's "tiny white particles" training
distribution.  Boxes returned by the model are mapped back to native pixels.

Two detection backends:
    CV         local blob extraction (free, offline, instant full scans)
    Roboflow   the serverless RF-DETR workflow (needs ROBOFLOW_API_KEY)
"""
import argparse
import hashlib
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import netCDF4 as nc
import numpy as np
from PIL import Image, ImageTk

from roboflow_particles import (
    RoboflowConfigError,
    iter_prediction_lists,
    run_workflow,
)

PLAYBACK_FPS = 30
PLAYBACK_MS = int(1000 / PLAYBACK_FPS)
BATCH_SIZE = 10
MAX_TRACK_DIST = 80.0  # px - centroids farther apart start a new track
MIN_VISIBLE_BOX = 24  # px - smallest on-screen box indicator
TRACK_COLORS = [
    (0, 255, 0),      # green
    (0, 200, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 128, 255),    # amber
    (255, 0, 128),    # pink
    (200, 200, 0),    # yellow
    (0, 255, 200),    # teal
]
BG_CANVAS = "#101014"
TIMELINE_PAD = 8
TIMELINE_TICK = "#3ddc84"
TIMELINE_POS = "#ff9f43"
DET_PERCENTILE = 99.9  # top-% threshold for candidate ember blobs
DET_MIN_AREA = 4  # px - smaller blobs are sensor noise
DET_MAX_AREA = 60  # px - larger blobs are plume, not embers
DET_UPSCALE = 2  # detector sees tiny particles better upscaled
MODEL_WORKERS = 8  # parallel Roboflow calls during a full scan
CACHE_SAVE_EVERY = 500  # persist scan progress this often
DETECT_MODES = ("Roboflow", "CV")


# ---------------------------------------------------------------------------
# Pure helpers (testable without a GUI)
# ---------------------------------------------------------------------------

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Min-max normalize a float temperature frame to uint8 (0-255)."""
    norm = np.empty_like(frame, dtype=np.uint8)
    cv2.normalize(frame, norm, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return norm


def render_rgb(frame: np.ndarray, cmap: str = "Grayscale") -> np.ndarray:
    """Return an RGB visualization of a temperature frame."""
    norm = normalize_frame(frame)
    if cmap == "Inferno":
        bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    else:
        bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def detection_bbox(det: dict) -> tuple:
    """Convert Roboflow (center-based) x/y/width/height to (left, top, w, h)."""
    w = float(det.get("width", 0))
    h = float(det.get("height", 0))
    cx = float(det.get("x", 0))
    cy = float(det.get("y", 0))
    return int(cx - w / 2), int(cy - h / 2), int(w), int(h)


def find_ember_blobs(arr: np.ndarray, percentile: float = DET_PERCENTILE,
                     min_area: int = DET_MIN_AREA, max_area: int = DET_MAX_AREA) -> list:
    """Find small hot blobs (candidate firebrands) in a temperature frame.

    Pixels at/above the ``percentile`` threshold are connected into blobs;
    those whose area falls within ``[min_area, max_area]`` px are returned as
    center-based box dicts (same schema as Roboflow detections) with the
    blob area and peak temperature attached.
    """
    thr = float(np.percentile(arr, percentile))
    mask = (arr >= thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            blobs.append({
                "x": x + w / 2.0, "y": y + h / 2.0,
                "width": float(w), "height": float(h),
                "area": area, "peak": float(arr[labels == i].max()),
            })
    return blobs


def render_detection_input(arr: np.ndarray, percentile: float = DET_PERCENTILE,
                           min_area: int = DET_MIN_AREA, max_area: int = DET_MAX_AREA,
                           upscale: int = DET_UPSCALE) -> tuple:
    """Render candidate ember blobs as white-on-black for the detection model.

    The min-max render the model previously saw was dominated by the hot
    plume, hiding tiny firebrands.  Isolating the hottest tail as small white
    blobs on a black background matches the model's training distribution
    ("tiny white particles").  Returns ``(rgb image, scale)`` - model box
    coordinates must be divided by ``scale`` to map back to native pixels.
    """
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for b in find_ember_blobs(arr, percentile, min_area, max_area):
        x, y, w, h = detection_bbox(b)
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + w), arr.shape[1]), min(int(y + h), arr.shape[0])
        rgb[y1:y2, x1:x2] = (255, 255, 255)
    if upscale != 1:
        rgb = cv2.resize(rgb, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_NEAREST)
    return rgb, float(upscale)


def scale_detections(dets: list, scale: float) -> list:
    """Map model boxes from upscaled input space back to native pixels."""
    if scale == 1.0:
        return dets
    return [
        {**d, "x": float(d["x"]) / scale, "y": float(d["y"]) / scale,
         "width": float(d["width"]) / scale, "height": float(d["height"]) / scale}
        for d in dets
    ]


def assign_tracks(
    dets: list,
    prev_tracks: list,
    next_id: int,
    max_dist: float = MAX_TRACK_DIST,
) -> list:
    """Greedy nearest-centroid track assignment.

    Args:
        dets: detection dicts (any order).
        prev_tracks: list of ``{"track_id", "cx", "cy"}`` from the previous
            frame (may be empty).
        next_id: next free track id.

    Returns:
        ``dets`` (copies) with a ``track_id`` key added, sorted by confidence
        descending.
    """
    tracked = []
    used = set()
    for det in sorted(dets, key=lambda d: -float(d.get("confidence", 0.0))):
        x, y, w, h = detection_bbox(det)
        cx, cy = x + w / 2.0, y + h / 2.0
        best_id, best_d = None, max_dist
        for t in prev_tracks:
            if t["track_id"] not in used:
                d = float(np.hypot(t["cx"] - cx, t["cy"] - cy))
                if d < best_d:
                    best_d, best_id = d, t["track_id"]
        if best_id is None:
            best_id = next_id
            next_id += 1
        used.add(best_id)
        tracked.append({**det, "track_id": best_id})
    return tracked


def frame_from_x(x: float, width: int, num_frames: int, pad: int = TIMELINE_PAD) -> int:
    """Map a timeline pixel x to a frame index (clamped)."""
    if num_frames <= 1:
        return 0
    inner = max(width - 2 * pad - 1, 1)
    frac = (x - pad) / inner
    return max(0, min(int(round(frac * (num_frames - 1))), num_frames - 1))


def x_from_frame(f: int, width: int, num_frames: int, pad: int = TIMELINE_PAD) -> int:
    """Map a frame index to a timeline pixel x."""
    if num_frames <= 1:
        return pad
    inner = max(width - 2 * pad - 1, 1)
    return pad + int(round(f * inner / (num_frames - 1)))


def hottest_stride(num_frames: int, max_scans: int = 1000) -> int:
    """Frame stride for a throttled hottest-frame scan."""
    return max(1, num_frames // max_scans)


def track_color(det: dict) -> tuple:
    """BGR color for a detection's track id."""
    return TRACK_COLORS[det.get("track_id", 0) % len(TRACK_COLORS)]


def draw_detection_overlay(vis: np.ndarray, det: dict) -> np.ndarray:
    """Draw box + crosshair + dark label chip for one detection, in place."""
    x, y, w, h = detection_bbox(det)
    cx, cy = x + w / 2.0, y + h / 2.0
    color = track_color(det)
    # padded box so tiny particles stay visible
    pad = max(0, (MIN_VISIBLE_BOX - max(w, h)) // 2)
    bx1, by1 = int(cx - w / 2 - pad), int(cy - h / 2 - pad)
    bx2, by2 = int(cx + w / 2 + pad), int(cy + h / 2 + pad)
    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
    # centroid crosshair ring
    r = max(4, int(max(w, h) / 2) + 3)
    cv2.circle(vis, (int(cx), int(cy)), r, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(vis, (int(cx) - r - 4, int(cy)), (int(cx) + r + 4, int(cy)), (255, 255, 255), 1)
    cv2.line(vis, (int(cx), int(cy) - r - 4), (int(cx), int(cy) + r + 4), (255, 255, 255), 1)
    # dark label chip: #ID conf
    label = f"#{det.get('track_id', 0)} {float(det.get('confidence', 0)):.2f}"
    (tw, th), _base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx = bx1
    ly = by1 - th - 12 if by1 - th - 12 > 6 else by2 + 6
    cv2.rectangle(vis, (lx - 4, ly - 4), (lx + tw + 5, ly + th + 4), (15, 15, 20), -1)
    cv2.rectangle(vis, (lx - 4, ly - 4), (lx + tw + 5, ly + th + 4), color, 1)
    cv2.putText(vis, label, (lx, ly + th - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_hud(vis: np.ndarray, dets: list) -> np.ndarray:
    """Draw the top-right 'DETECTED N objects' confirmation badge, in place."""
    if not dets:
        return vis
    confs = [float(d.get("confidence", 0)) for d in dets]
    text = f"  DETECTED {len(dets)} objects  (conf {min(confs):.2f}-{max(confs):.2f})  "
    (tw, th), _base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    h, w = vis.shape[:2]
    bx1, by1 = w - tw - 22, 10
    cv2.rectangle(vis, (bx1, by1), (w - 10, by1 + th + 8), (15, 15, 20), -1)
    cv2.rectangle(vis, (bx1, by1), (w - 10, by1 + th + 8), (0, 255, 0), 1)
    cv2.circle(vis, (bx1 + 9, by1 + th // 2 + 3), 4, (0, 255, 0), -1)
    cv2.putText(vis, text, (bx1 + 3, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Timeline bar widget
# ---------------------------------------------------------------------------

class TimelineBar(tk.Canvas):
    """Full-range clickable/draggable timeline with detection tick marks."""

    HEIGHT = 46

    def __init__(self, master, num_frames, on_seek, on_hover):
        super().__init__(master, height=self.HEIGHT, bg="#141418",
                         highlightthickness=0, cursor="hand2")
        self.num_frames = max(int(num_frames), 1)
        self.on_seek = on_seek
        self.on_hover = on_hover
        self._position = 0
        self._tick_cols = set()
        self._press_x = None
        self._last_seek = 0.0
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda _e: self._hover(None))

    # ----- public API -------------------------------------------------

    def set_position(self, idx: int):
        idx = max(0, min(int(idx), self.num_frames - 1))
        if idx != self._position:
            self._position = idx
            self._redraw()

    def add_detected_frames(self, frames):
        width = max(self.winfo_width(), 2)
        for f in frames:
            self._tick_cols.add(x_from_frame(int(f), width, self.num_frames))
        self._redraw()

    def detected_frames(self) -> list:
        """Frames currently marked (inverse of the tick columns, approximate)."""
        width = max(self.winfo_width(), 2)
        return sorted(frame_from_x(col, width, self.num_frames) for col in self._tick_cols)

    # ----- interactions ----------------------------------------------

    def _frame_for(self, event) -> int:
        return frame_from_x(event.x, max(self.winfo_width(), 2), self.num_frames)

    def _press(self, event):
        self._press_x = event.x
        self.on_seek(self._frame_for(event))

    def _drag(self, event):
        self.on_seek(self._frame_for(event))

    def _release(self, event):
        self._press_x = None
        self.on_seek(self._frame_for(event))

    def _motion(self, event):
        self._hover(self._frame_for(event))

    def _hover(self, frame):
        if self.on_hover:
            self.on_hover(frame)

    # ----- drawing ----------------------------------------------------

    def _redraw(self):
        self.delete("all")
        width, height = max(self.winfo_width(), 2), self.HEIGHT
        y_track = height - 14
        # track groove
        self.create_rectangle(TIMELINE_PAD, y_track, width - TIMELINE_PAD, y_track + 4,
                              fill="#2a2a31", outline="")
        # detection ticks
        for col in sorted(self._tick_cols):
            self.create_line(col, y_track + 7, col, y_track + 14, fill=TIMELINE_TICK, width=2)
        # current position cursor
        px = x_from_frame(self._position, width, self.num_frames)
        self.create_line(px, 8, px, y_track + 14, fill=TIMELINE_POS, width=2)
        self.create_polygon(px - 5, 8, px + 5, 8, px, 15, fill=TIMELINE_POS, outline="")
        self.create_text(px, height - 4, text=str(self._position + 1), fill="#cfcfd6",
                         font=("Helvetica", 9))
        # end labels
        self.create_text(TIMELINE_PAD + 2, 8, text="1", anchor=tk.W, fill="#6a6a73",
                         font=("Helvetica", 8))
        self.create_text(width - TIMELINE_PAD - 2, 8, text=str(self.num_frames), anchor=tk.E,
                         fill="#6a6a73", font=("Helvetica", 8))


# ---------------------------------------------------------------------------
# Player application
# ---------------------------------------------------------------------------

class NCPlayer(tk.Tk):
    def __init__(self, path: str):
        super().__init__()
        self.title(f"NC Player - {os.path.basename(path)}")
        self.geometry("1120x820")
        self.minsize(720, 560)

        self.ds = nc.Dataset(path, "r")
        self.temp = self.ds.variables["temperature"]
        self.num_frames, self.height, self.width = self.temp.shape
        self._read_lock = threading.Lock()
        self._cmap_names = ("Grayscale", "Inferno")

        self.frame_idx = 0
        self.playing = False
        self.detections = {}       # frame -> list of detection dicts (with track_id)
        self.tracks = {}           # frame -> list of {track_id, cx, cy}
        self._last_tracks = []
        self._next_id = 1
        self._hottest = None
        self._hottest_scanning = False
        self.var_cmap = tk.StringVar(value=self._cmap_names[0])
        self.var_mode = tk.StringVar(value=DETECT_MODES[0])
        self._msg_q = queue.Queue()
        self._photo = None
        self._det_lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._hash_cache = {}       # render-byte hash -> detections (dedup)
        self._scanning = False
        self._scan_stop = False
        self._scan_ticks = set()
        self._scan_start = 0.0
        self._last_saved = 0

        self._build_ui()
        self._update_detect_buttons()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._load_cache()
        self.after(50, self._poll_queue)
        self.after(PLAYBACK_MS, self._tick)

    # ----- UI ---------------------------------------------------------

    def _build_ui(self):
        top = tk.Frame(self, bg="#1b1b21")
        top.pack(fill=tk.X)

        self.btn_open = tk.Button(top, text="Open...", command=self._open_file)
        self.btn_open.pack(side=tk.LEFT, padx=(8, 4), pady=6)
        self.btn_play = tk.Button(top, text="Play", width=6, command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4, pady=6)
        self.btn_prev = tk.Button(top, text="<", width=3, command=lambda: self.seek(-1))
        self.btn_prev.pack(side=tk.LEFT, padx=2, pady=6)
        self.btn_next = tk.Button(top, text=">", width=3, command=lambda: self.seek(1))
        self.btn_next.pack(side=tk.LEFT, padx=2, pady=6)
        self.btn_detect = tk.Button(
            top, text="Detect frame (D)", command=self.detect_current,
            state=tk.NORMAL if os.environ.get("ROBOFLOW_API_KEY") else tk.DISABLED,
        )
        self.btn_detect.pack(side=tk.LEFT, padx=8, pady=6)
        self.btn_batch = tk.Button(
            top, text=f"Detect batch {BATCH_SIZE} (B)", command=self.detect_batch,
            state=tk.NORMAL if os.environ.get("ROBOFLOW_API_KEY") else tk.DISABLED,
        )
        self.btn_batch.pack(side=tk.LEFT, padx=4, pady=6)
        self.btn_mode = tk.Button(
            top, text=f"Mode: {DETECT_MODES[0]}", command=self.toggle_mode,
        )
        self.btn_mode.pack(side=tk.LEFT, padx=4, pady=6)
        self.btn_scan = tk.Button(
            top, text="Scan all (A)", command=self.scan_all,
            state=tk.NORMAL if os.environ.get("ROBOFLOW_API_KEY") else tk.DISABLED,
        )
        self.btn_scan.pack(side=tk.LEFT, padx=(4, 8), pady=6)

        cmap_lbl = tk.Label(top, text="Render:", bg="#1b1b21", fg="#cfcfd6")
        cmap_lbl.pack(side=tk.RIGHT, padx=(0, 4), pady=6)
        self.cmb_cmap = ttk.Combobox(
            top, values=list(self._cmap_names), textvariable=self.var_cmap,
            state="readonly", width=10,
        )
        self.cmb_cmap.bind("<<ComboboxSelected>>", lambda _e: self._render_current())
        self.cmb_cmap.pack(side=tk.RIGHT, padx=(0, 8), pady=6)

        self.lbl_frame = tk.Label(top, text="0 / 0", bg="#1b1b21", fg="#cfcfd6")
        self.lbl_frame.pack(side=tk.RIGHT, padx=8)

        self.timeline = TimelineBar(self, self.num_frames, on_seek=self._timeline_seek,
                                    on_hover=self._timeline_hover)
        self.timeline.pack(fill=tk.X)

        self.canvas = tk.Canvas(self, bg=BG_CANVAS, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._render_current())
        self.canvas.bind("<Button-4>", lambda _e: self.seek(-1))
        self.canvas.bind("<Button-5>", lambda _e: self.seek(1))

        self.lbl_status = tk.Label(
            self, text="Status: ready", anchor=tk.W, bg="#141418", fg="#9a9aa3", padx=8,
        )
        self.lbl_status.pack(fill=tk.X)

        self.bind("<space>", lambda _e: self.toggle_play())
        self.bind("<Left>", lambda _e: self.seek(-1))
        self.bind("<Right>", lambda _e: self.seek(1))
        self.bind("<comma>", lambda _e: self.seek(-1))
        self.bind("<period>", lambda _e: self.seek(1))
        self.bind("<Home>", lambda _e: self.seek_to(0))
        self.bind("<End>", lambda _e: self.seek_to(self.num_frames - 1))
        self.bind("<bracketleft>", lambda _e: self._jump_detected(-1))
        self.bind("<bracketright>", lambda _e: self._jump_detected(1))
        self.bind("h", lambda _e: self._go_hottest())
        self.bind("H", lambda _e: self._go_hottest())
        self.bind("d", lambda _e: self.detect_current())
        self.bind("D", lambda _e: self.detect_current())
        self.bind("b", lambda _e: self.detect_batch())
        self.bind("B", lambda _e: self.detect_batch())
        self.bind("c", lambda _e: self.toggle_mode())
        self.bind("C", lambda _e: self.toggle_mode())
        self.bind("a", lambda _e: self.scan_all())
        self.bind("A", lambda _e: self.scan_all())

    # ----- file / frame access ---------------------------------------

    def _open_file(self):
        path = filedialog.askopenfilename(filetypes=[("NetCDF", "*.nc"), ("All files", "*.*")])
        if path:
            try:
                self.ds.close()
            except Exception:
                pass
            self.__init__(path)

    def _read_frame(self, idx: int) -> np.ndarray:
        with self._read_lock:
            return np.array(self.temp[idx, :, :])

    def seek(self, delta: int):
        self.seek_to(self.frame_idx + delta)

    def seek_to(self, idx: int):
        idx = max(0, min(idx, self.num_frames - 1))
        if idx != self.frame_idx:
            self.frame_idx = idx
            self.timeline.set_position(idx)
            self._render_current()
        self._update_status()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")

    def _tick(self):
        if self.playing:
            nxt = self.frame_idx + 1
            if nxt >= self.num_frames:
                nxt = 0
            self.frame_idx = nxt
            self.timeline.set_position(nxt)
            self._render_current()
            self._update_status()
        self.after(PLAYBACK_MS, self._tick)

    # ----- timeline interactions -------------------------------------

    def _timeline_seek(self, idx: int):
        if idx != self.frame_idx:
            self.seek_to(idx)

    def _timeline_hover(self, frame):
        if frame is None:
            self._update_status()
            return
        dets = self.detections.get(frame)
        n = len(dets) if dets else 0
        self.lbl_status.configure(
            text=f"Status: frame {frame + 1}/{self.num_frames}"
                 f"  |  cached objects: {n}")

    def _jump_detected(self, direction: int):
        marked = sorted(f for f, dets in self.detections.items() if dets)
        if not marked:
            self.lbl_status.configure(
                text="Status: no detected frames yet - press D or B first")
            return
        cur = self.frame_idx
        if direction > 0:
            nxt = next((f for f in marked if f > cur), None)
            if nxt is None:
                nxt = marked[0]
        else:
            nxt = next((f for f in reversed(marked) if f < cur), None)
            if nxt is None:
                nxt = marked[-1]
        self.seek_to(nxt)

    # ----- rendering --------------------------------------------------

    def _render_current(self):
        """Re-render the current frame (fit to canvas) with overlays."""
        if not self.winfo_exists():
            return
        frame = self._read_frame(self.frame_idx)
        vis = render_rgb(frame, self.var_cmap.get())
        dets = self.detections.get(self.frame_idx)
        if dets:
            for det in dets:
                draw_detection_overlay(vis, det)
            draw_hud(vis, dets)
        tmax = float(np.max(frame))
        hud_line = f"frame {self.frame_idx + 1}/{self.num_frames}  max {tmax:.1f} C"
        cv2.putText(vis, hud_line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, hud_line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        img = Image.fromarray(vis)
        cw, ch = max(self.canvas.winfo_width(), 64), max(self.canvas.winfo_height(), 64)
        scale = min(cw / self.width, ch / self.height)
        if scale < 1.0:
            img = img.resize((int(self.width * scale), int(self.height * scale)),
                             Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _update_status(self):
        dets = self.detections.get(self.frame_idx)
        n = len(dets) if dets else 0
        self.lbl_frame.configure(text=f"{self.frame_idx + 1} / {self.num_frames}")
        if self._scanning:
            return
        if not self._detection_enabled():
            self.lbl_status.configure(
                text="Status: set ROBOFLOW_API_KEY or switch to CV mode (C) to detect")
            return
        self.lbl_status.configure(
            text=f"Status: frame {self.frame_idx + 1}/{self.num_frames}"
                 f" ({self.var_mode.get()}) | tracked frames: {len(self.tracks)}"
                 f" | detections: {n}")

    # ----- hottest frame ---------------------------------------------

    def _go_hottest(self):
        if self._hottest is not None:
            self.seek_to(self._hottest)
            return
        if self._hottest_scanning:
            return
        self._hottest_scanning = True
        self.lbl_status.configure(text="Status: scanning for hottest frame ...")

        def worker():
            try:
                stride = hottest_stride(self.num_frames)
                best, best_v = 0, float("-inf")
                for i in range(0, self.num_frames, stride):
                    with self._read_lock:
                        arr = np.array(self.temp[i, ::4, ::4])
                    v = float(np.max(arr))
                    if v > best_v:
                        best_v, best = v, i
                self._msg_q.put(("hottest", best, best_v))
            except Exception as exc:  # surface scan errors in the status bar
                self._msg_q.put(("error", f"hotspot scan failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    # ----- detection backends ----------------------------------------

    def _detection_enabled(self) -> bool:
        """CV mode is offline; Roboflow mode needs the API key."""
        return self.var_mode.get() == "CV" or bool(os.environ.get("ROBOFLOW_API_KEY"))

    def _update_detect_buttons(self):
        state = tk.NORMAL if self._detection_enabled() else tk.DISABLED
        self.btn_detect.configure(state=state)
        self.btn_batch.configure(state=state)
        self.btn_scan.configure(state=state)

    def toggle_mode(self):
        if self._scanning:
            return
        nxt = DETECT_MODES[1] if self.var_mode.get() == DETECT_MODES[0] else DETECT_MODES[0]
        self.var_mode.set(nxt)
        self.btn_mode.configure(text=f"Mode: {nxt}")
        self._update_detect_buttons()
        self._update_status()

    def _model_detect(self, data: bytes) -> list:
        """POST one rendered frame; dedups identical frames; returns dets."""
        h = hashlib.sha1(data).hexdigest()
        with self._det_lock:
            if h in self._hash_cache:
                return [dict(d) for d in self._hash_cache[h]]
        result = run_workflow(data, images_dir="/tmp", timeout=90)
        dets = []
        for _, _, preds in iter_prediction_lists(result.entries):
            dets.extend(preds)
        with self._det_lock:
            if len(self._hash_cache) >= 2000:
                self._hash_cache.pop(next(iter(self._hash_cache)))
            self._hash_cache[h] = [dict(d) for d in dets]
        return dets

    def _model_detect_frame(self, arr: np.ndarray) -> list:
        """Render one frame for the model and map boxes to native pixels."""
        vis, scale = render_detection_input(arr)
        data = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))[1].tobytes()
        return scale_detections(self._model_detect(data), scale)

    def _cv_detect_frame(self, arr: np.ndarray) -> list:
        """Offline blob extraction - the same logic the model input uses."""
        thr = float(np.percentile(arr, DET_PERCENTILE))
        rng = max(float(arr.max()) - thr, 1e-6)
        dets = []
        for b in find_ember_blobs(arr):
            conf = float(np.clip((b["peak"] - thr) / rng, 0.0, 1.0))
            dets.append({
                "x": b["x"], "y": b["y"], "width": b["width"],
                "height": b["height"], "confidence": conf, "class": "ember",
            })
        return dets

    def _detect_frames(self, frames: list):
        """Detect frames with the active backend on a background thread."""
        if self.var_mode.get() == "CV":
            detect = self._cv_detect_frame
        else:
            detect = self._model_detect_frame

        def worker():
            for f in frames:
                try:
                    self._msg_q.put(("detections", f, detect(self._read_frame(f))))
                except RoboflowConfigError as exc:
                    self._msg_q.put(("error", str(exc)))
                except Exception as exc:  # surface typed + unexpected errors
                    self._msg_q.put(("error", f"frame {f}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def detect_current(self):
        if not self._detection_enabled():
            self.lbl_status.configure(
                text="Status: set ROBOFLOW_API_KEY or switch to CV mode (C) to detect")
            return
        f = self.frame_idx
        if f in self.detections:
            self._update_status()
            self.lbl_status.configure(
                text=f"Status: frame {f + 1} already detected "
                     f"({len(self.detections[f])} objects) - cached")
            return
        self.lbl_status.configure(
            text=f"Status: detecting frame {f + 1} ({self.var_mode.get()}) ...")
        self._detect_frames([f])

    def detect_batch(self):
        if not self._detection_enabled():
            self.lbl_status.configure(
                text="Status: set ROBOFLOW_API_KEY or switch to CV mode (C) to detect")
            return
        frames = []
        for f in range(self.frame_idx + 1, min(self.frame_idx + 1 + BATCH_SIZE, self.num_frames)):
            if f not in self.detections:
                frames.append(f)
        if not frames:
            self.lbl_status.configure(text="Status: next frames already detected")
            return
        self.lbl_status.configure(
            text=f"Status: detecting batch ({len(frames)} frames, "
                 f"{self.var_mode.get()}) in background ...")
        self._detect_frames(frames)

    # ----- full-video scan -------------------------------------------

    def scan_all(self):
        """Start (or stop, if running) a full-video scan in the background."""
        if self._scanning:
            self._scan_stop = True
            self.lbl_status.configure(text="Status: stopping scan ...")
            return
        if not self._detection_enabled():
            self.lbl_status.configure(
                text="Status: set ROBOFLOW_API_KEY or switch to CV mode (C) to scan")
            return
        mode = self.var_mode.get()
        frames = [f for f in range(self.num_frames) if f not in self.detections]
        if not frames:
            self.lbl_status.configure(text="Status: all frames already detected")
            return
        if mode == "Roboflow" and not messagebox.askyesno(
                "Full scan", f"Scan {len(frames)} frames with the Roboflow "
                f"workflow?\nThis takes a while and uses API credits.\n"
                f"Interrupted scans resume (results are cached)."):
            return
        self._scanning = True
        self._scan_stop = False
        self._scan_ticks.clear()
        self._scan_start = time.time()
        self._last_saved = 0
        self.btn_scan.configure(text="Stop scan (A)")
        self.lbl_status.configure(
            text=f"Status: scanning 0/{len(frames)} ({mode}) ...")
        if mode == "CV":
            threading.Thread(target=self._scan_cv, args=(frames,), daemon=True).start()
        else:
            threading.Thread(target=self._scan_model, args=(frames,), daemon=True).start()

    def _scan_progress(self, done: int, total: int, found: int):
        if done and done % CACHE_SAVE_EVERY == 0 and done != self._last_saved:
            self._last_saved = done
            threading.Thread(target=self._save_cache, daemon=True).start()
        if not self._scanning:
            return
        elapsed = max(time.time() - self._scan_start, 1e-6)
        rate = done / elapsed
        eta_min = int((total - done) / rate / 60) if rate else 0
        self.lbl_status.configure(
            text=f"Status: scanning {done}/{total} ({self.var_mode.get()}, "
                 f"~{eta_min} min left) | frames with objects: {found} | "
                 f"stop: A")

    def _scan_model(self, frames: list):
        """Parallel Roboflow scan over ``frames`` (skips already-detected)."""
        q = queue.Queue()
        for f in frames:
            q.put(f)
        total = len(frames)
        done, found = 0, 0

        def one_worker():
            nonlocal done, found
            while not self._scan_stop:
                try:
                    f = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    dets = self._model_detect_frame(self._read_frame(f))
                    self._msg_q.put(("detections", f, dets))
                    if dets:
                        found += 1
                except Exception as exc:
                    self._msg_q.put(("error", f"frame {f}: {exc}"))
                finally:
                    done += 1
                    self._msg_q.put(("scan_progress", done, total, found))

        threads = [threading.Thread(target=one_worker, daemon=True)
                   for _ in range(MODEL_WORKERS)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._msg_q.put(("scan_complete", done, found))

    def _scan_cv(self, frames: list):
        """Single-threaded offline scan (free, no network)."""
        total = len(frames)
        found, done = 0, 0
        for f in frames:
            if self._scan_stop:
                break
            try:
                dets = self._cv_detect_frame(self._read_frame(f))
                self._msg_q.put(("detections", f, dets))
                if dets:
                    found += 1
            except Exception as exc:
                self._msg_q.put(("error", f"frame {f}: {exc}"))
            done += 1
            self._msg_q.put(("scan_progress", done, total, found))
        self._msg_q.put(("scan_complete", done, found))

    # ----- tracking + persistence ------------------------------------

    def rebuild_tracks(self):
        """Deterministically reassign track ids over all cached frames."""
        self.tracks = {}
        prev, nxt = [], 1
        for f in sorted(self.detections):
            assigned = assign_tracks(self.detections[f], prev, nxt)
            self.detections[f] = assigned
            tracked = []
            for det in assigned:
                x, y, w, h = detection_bbox(det)
                tracked.append({"track_id": det["track_id"],
                                "cx": x + w / 2.0, "cy": y + h / 2.0})
            nxt = max(nxt, (max((t["track_id"] for t in tracked), default=0) + 1))
            if tracked:
                self.tracks[f] = tracked
            prev = tracked
        self._next_id = nxt
        self._last_tracks = prev
        if self.detections:
            self.timeline.add_detected_frames(
                [f for f, dets in self.detections.items() if dets])

    def _cache_path(self) -> str:
        return os.path.splitext(self.ds.filepath())[0] + ".detections.json"

    def _save_cache(self):
        if not self.detections:
            return
        data = {"v": 1, "mode": self.var_mode.get(),
                "detections": {str(f): dets
                               for f, dets in sorted(self.detections.items())}}
        try:
            with self._save_lock:
                with open(self._cache_path(), "w") as fh:
                    json.dump(data, fh)
        except Exception as exc:  # cache is best-effort
            print(f"cache save failed: {exc}")

    def _load_cache(self):
        path = self._cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            for key, dets in data.get("detections", {}).items():
                self.detections[int(key)] = dets
        except Exception as exc:  # stale/corrupt cache - start fresh
            print(f"cache load failed: {exc}")
            return
        self.rebuild_tracks()
        print(f"loaded {len(self.detections)} cached frames from {path}")

    def _on_close(self):
        if self._scanning:
            self._scan_stop = True
        self._save_cache()
        try:
            self.ds.close()
        except Exception:
            pass
        self.destroy()

    def _on_detections(self, frame: int, dets: list):
        assigned = assign_tracks(dets, self._last_tracks, self._next_id)
        self.detections[frame] = assigned
        tracked = []
        for det in assigned:
            x, y, w, h = detection_bbox(det)
            tracked.append({"track_id": det["track_id"], "cx": x + w / 2.0, "cy": y + h / 2.0})
        self._next_id = max(self._next_id, (max((t["track_id"] for t in tracked), default=0) + 1))
        if tracked:
            self.tracks[frame] = tracked
        self._last_tracks = tracked
        if assigned:
            if self._scanning:
                self._scan_ticks.add(frame)  # batch the timeline redraws
            else:
                self.timeline.add_detected_frames([frame])
        self._update_status()
        if frame == self.frame_idx:
            self._render_current()

    def _poll_queue(self):
        try:
            while True:
                kind, *payload = self._msg_q.get_nowait()
                if kind == "detections":
                    self._on_detections(payload[0], payload[1])
                elif kind == "scan_progress":
                    self._scan_progress(payload[0], payload[1], payload[2])
                elif kind == "scan_complete":
                    self._scanning = False
                    if self._scan_ticks:
                        self.timeline.add_detected_frames(sorted(self._scan_ticks))
                        self._scan_ticks.clear()
                    self.rebuild_tracks()
                    self._save_cache()
                    self.btn_scan.configure(text="Scan all (A)")
                    self.lbl_status.configure(
                        text=f"Status: scan done ({payload[0]} frames processed, "
                             f"{payload[1]} with objects) - [ ] hops between them")
                elif kind == "hottest":
                    self._hottest, hottest_max = payload[0], payload[1]
                    self._hottest_scanning = False
                    self.seek_to(self._hottest)
                    self.lbl_status.configure(
                        text=f"Status: hottest frame {self._hottest + 1}"
                             f" (max {hottest_max:.1f} C)")
                elif kind == "error":
                    self.lbl_status.configure(text=f"Status: {payload[0]}")
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)


def main():
    parser = argparse.ArgumentParser(description="Lightweight NetCDF thermal player")
    parser.add_argument("path", nargs="?", default="S9.nc", help="path to .nc file")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        raise SystemExit(1)
    if not os.environ.get("ROBOFLOW_API_KEY"):
        print("Note: ROBOFLOW_API_KEY not set - Roboflow detection buttons disabled.")
    app = NCPlayer(args.path)
    app.mainloop()


if __name__ == "__main__":
    main()