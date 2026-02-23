"""
Tkinter dashboard GUI for FLIR SDK tracking-based hotspot/ember detection.
- Preserves SDK.py tracking/detection logic (OrderedDict tracker, ROI, overlays)
- Left sidebar: Play, pause, frame slider, prev/next, ROI numeric/manual, threshold, export frame
- Canvas: shows current frame with overlays
"""
import base64
import concurrent.futures
import csv
import json
import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from collections import OrderedDict
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None
try:
    import cv2
except Exception:
    print("OpenCV required: pip install opencv-python-headless"); sys.exit(1)
try:
    import fnv
    import fnv.file
except Exception:
    print("FLIR SDK required"); sys.exit(1)

# ------- Tracking Parameters/Globals -------
DESIRED_TEMP_UNIT = fnv.TempType.CELSIUS
TEMP_THRESHOLD_CELSIUS = 300.0
MIN_OBJECT_AREA_PIXELS = 3
MAX_OBJECT_AREA_PIXELS = 150
CENTROID_TRACKING_MAX_DIST = 40
TRACK_MEMORY_FRAMES = 15
SUPPORTED_EXTENSIONS = (".seq", ".csq", ".jpg", ".ats", ".sfmov", ".img")
TARGET_FPS = 30
APP_VERSION = "v0.0.2"
GITHUB_OWNER = "NinhGhoster"
GITHUB_REPO = "Firebrand-Thermal-Analysis"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases"
GITHUB_API_LATEST_URL = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest"
)


def clamp_roi(
    roi: Optional[Tuple[int, int, int, int]],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Clamp ROI to frame bounds; returns None if ROI is not usable."""
    if roi is None:
        return None
    try:
        rx, ry, rw, rh = map(int, roi)
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    rx = max(0, min(rx, width - 1))
    ry = max(0, min(ry, height - 1))
    rw = max(1, min(rw, width - rx))
    rh = max(1, min(rh, height - ry))
    if rw <= 0 or rh <= 0:
        return None
    return rx, ry, rw, rh


def detect_firebrands(
    frame: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
    temp_thresh: float,
) -> List[Dict[str, Any]]:
    """Detect firebrand blobs above a temperature threshold inside an optional ROI."""
    det_offset = (0, 0)
    display = frame
    if roi is not None:
        rx, ry, rw, rh = roi
        display = frame[ry : ry + rh, rx : rx + rw]
        det_offset = (rx, ry)

    if display.size == 0:
        return []

    binary_mask = (display > temp_thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )
    detections: List[Dict[str, Any]] = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if not (MIN_OBJECT_AREA_PIXELS <= area <= MAX_OBJECT_AREA_PIXELS):
            continue

        x = stats[lid, cv2.CC_STAT_LEFT] + det_offset[0]
        y = stats[lid, cv2.CC_STAT_TOP] + det_offset[1]
        w = stats[lid, cv2.CC_STAT_WIDTH]
        h = stats[lid, cv2.CC_STAT_HEIGHT]
        centroid = (centroids[lid][0] + det_offset[0], centroids[lid][1] + det_offset[1])

        sub = display[
            stats[lid, cv2.CC_STAT_TOP] : stats[lid, cv2.CC_STAT_TOP] + h,
            stats[lid, cv2.CC_STAT_LEFT] : stats[lid, cv2.CC_STAT_LEFT] + w,
        ]
        mask = (
            labels[
                stats[lid, cv2.CC_STAT_TOP] : stats[lid, cv2.CC_STAT_TOP] + h,
                stats[lid, cv2.CC_STAT_LEFT] : stats[lid, cv2.CC_STAT_LEFT] + w,
            ]
            == lid
        )
        temps = sub[mask]
        if temps.size == 0:
            continue

        detections.append(
            {
                "centroid": centroid,
                "bbox": (x, y, w, h),
                "max_temp": float(np.max(temps)),
                "min_temp": float(np.min(temps)),
                "avg_temp": float(np.mean(temps)),
                "median_temp": float(np.median(temps)),
                "area": int(area),
            }
        )
    return detections


def assign_tracks(
    detections: List[Dict[str, Any]],
    tracked_objects: "OrderedDict[int, Dict[str, Any]]",
    next_id: int,
    frame_idx: int,
) -> Tuple["OrderedDict[int, Dict[str, Any]]", List[Dict[str, Any]], int]:
    """Assign detections to existing tracks by nearest centroid, or create new tracks."""
    new_tracked: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
    for det in detections:
        cx, cy = det["centroid"]
        best_id = None
        best_dist = None
        for tid, obj in tracked_objects.items():
            dist = np.hypot(cx - obj["centroid"][0], cy - obj["centroid"][1])
            if dist <= CENTROID_TRACKING_MAX_DIST and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_id = tid

        if best_id is None:
            best_id = next_id
            next_id += 1

        det["track_id"] = best_id
        new_tracked[best_id] = {"centroid": det["centroid"], "last_seen": frame_idx}

    for tid, obj in tracked_objects.items():
        if tid not in new_tracked and frame_idx - obj.get("last_seen", frame_idx) <= TRACK_MEMORY_FRAMES:
            new_tracked[tid] = obj
    return new_tracked, detections, next_id


def export_seq_to_csv_worker(seq_path: str, settings: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    """Parallel worker: export a single SEQ to CSV next to the source file.

    Returns (seq_path, out_csv_path, error_message). error_message is None on success.
    """
    try:
        if hasattr(cv2, "setNumThreads"):
            cv2.setNumThreads(1)

        try:
            start_f = max(1, int(settings.get("export_start", 1)))
        except (TypeError, ValueError):
            start_f = 1

        end_raw = str(settings.get("export_end", "max")).strip().lower()
        temp_thresh = float(settings.get("thresh", TEMP_THRESHOLD_CELSIUS))

        im = fnv.file.ImagerFile(seq_path)
        unit_is_temp = im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
        if unit_is_temp:
            im.unit = fnv.Unit.TEMPERATURE_FACTORY
            im.temp_type = DESIRED_TEMP_UNIT
        else:
            im.unit = fnv.Unit.COUNTS

        try:
            obj_params = im.object_parameters
            obj_params.emissivity = float(settings.get("emissivity", 0.9))
            im.object_parameters = obj_params
        except Exception:
            pass

        num_frames = im.num_frames
        width = im.width
        height = im.height
        roi = clamp_roi(settings.get("roi"), width, height)

        try:
            end_f = num_frames if end_raw in ("max", "", "none") else int(end_raw)
        except (TypeError, ValueError):
            end_f = num_frames

        if end_f <= 0 or end_f > num_frames:
            end_f = num_frames

        start_use = max(1, min(start_f, num_frames))
        if start_use > end_f:
            start_use = end_f

        tracked: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        next_id = 1
        rows = []
        for idx in range(start_use - 1, end_f):
            im.get_frame(idx)
            frame = np.array(im.final, copy=False).reshape((height, width))
            detections = detect_firebrands(frame, roi, temp_thresh)
            tracked, detections, next_id = assign_tracks(detections, tracked, next_id, idx)
            for det in sorted(detections, key=lambda d: d.get("track_id", -1)):
                x, y, w, h = det["bbox"]
                rows.append(
                    (
                        idx + 1,
                        det.get("track_id", -1),
                        det["max_temp"],
                        det.get("min_temp", det["max_temp"]),
                        det.get("avg_temp", det["max_temp"]),
                        det.get("median_temp", det["max_temp"]),
                        det.get("area", 0),
                        x,
                        y,
                        w,
                        h,
                    )
                )

        out_path = str(Path(seq_path).with_suffix(".csv"))
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame",
                    "firebrand_id",
                    "max_temperature",
                    "min_temperature",
                    "avg_temperature",
                    "median_temperature",
                    "area_pixels",
                    "bbox_x",
                    "bbox_y",
                    "bbox_w",
                    "bbox_h",
                ]
            )
            writer.writerows(rows)

        return seq_path, out_path, None
    except Exception as ex:
        return seq_path, "", f"{ex}\n{traceback.format_exc()}"


def _parse_version(tag: str) -> Optional[Tuple[int, ...]]:
    tag = tag.strip()
    if tag.lower().startswith("v"):
        tag = tag[1:]
    if not tag:
        return None
    parts = tag.split(".")
    nums: List[int] = []
    for part in parts:
        if not part.isdigit():
            return None
        nums.append(int(part))
    return tuple(nums) if nums else None


def _is_newer_version(current: str, latest: str) -> bool:
    cur_parsed = _parse_version(current)
    latest_parsed = _parse_version(latest)
    if cur_parsed and latest_parsed:
        max_len = max(len(cur_parsed), len(latest_parsed))
        cur_parsed += (0,) * (max_len - len(cur_parsed))
        latest_parsed += (0,) * (max_len - len(latest_parsed))
        return latest_parsed > cur_parsed
    return latest != current

class SKDDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Firebrand Thermal Analysis")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.im = None
        self.seq_path = ""
        self.num_frames = 0
        self.width = 0
        self.height = 0
        self.current_idx = 0
        self.playing = False
        self.tracked_objects = OrderedDict()
        self.next_track_id = 1
        self.last_tracked_frame = -1
        self.temp_threshold = TEMP_THRESHOLD_CELSIUS
        self._applied_emissivity = None
        self.unit_is_temp = False
        self.unit_label = "C"
        self.roi_rect: Optional[Tuple[int,int,int,int]] = None
        self.mouse_down = None
        self._in_slider_update = False
        self._resize_job = None
        self._canvas_size: Tuple[int, int] = (0, 0)
        self._last_frame = None
        self._base_status = "Status: ready"
        self._export_in_progress = False
        self._export_thread: Optional[threading.Thread] = None
        self._last_frame_time: float = 0.0
        self._prefetch_queue: queue.Queue = queue.Queue(maxsize=2)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop = threading.Event()
        self.var_export_start = tk.IntVar(value=1)
        self.var_export_end = tk.StringVar(value="max")
        self.batch_paths: List[str] = []
        self.batch_index: int = 0
        self.file_settings = {}
        self._build_ui()
        self._bind_events()
        self._reset_tracking()
    def _build_ui(self):
        # ------- Dark Mode Theme -------
        BG = "#1e1e1e"
        FG = "#e0e0e0"
        BG_LIGHT = "#2d2d2d"
        BG_ENTRY = "#383838"
        ACCENT = "#4a9eff"
        ACCENT_HOVER = "#6cb3ff"
        MUTED = "#888888"
        BORDER = "#444444"
        STATUS_BG = "#161616"

        self.configure(bg=BG)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=BG, foreground=FG, fieldbackground=BG_ENTRY,
                         bordercolor=BORDER, troughcolor=BG_LIGHT, arrowcolor=FG)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=FG, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT)
        style.configure("TButton", background=BG_LIGHT, foreground=FG,
                         bordercolor=BORDER, padding=(8, 4))
        style.map("TButton",
                  background=[("active", ACCENT), ("pressed", ACCENT_HOVER)],
                  foreground=[("active", "#ffffff"), ("pressed", "#ffffff")])
        style.configure("TEntry", fieldbackground=BG_ENTRY, foreground=FG,
                         insertcolor=FG, bordercolor=BORDER)
        style.map("TEntry", fieldbackground=[("focus", "#404040")])
        style.configure("TNotebook", background=BG, bordercolor=BORDER)
        style.configure("TNotebook.Tab", background=BG_LIGHT, foreground=FG,
                         padding=(10, 4))
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        style.configure("TScrollbar", background=BG_LIGHT, troughcolor=BG,
                         bordercolor=BG, arrowcolor=FG)
        style.configure("Horizontal.TScale", background=BG, troughcolor=BG_LIGHT,
                         bordercolor=BORDER)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Status.TLabel", background=STATUS_BG, foreground="#ff8c42",
                         padding=(6, 2))
        style.configure("ROI.TNotebook", background=BG, borderwidth=0)
        style.configure("ROI.TNotebook.Tab", background=BG_LIGHT, foreground=FG,
                         padding=(10, 2))
        style.map("ROI.TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "#ffffff")])
        try:
            style.layout("ROI.TNotebook", [("Notebook.client", {"sticky": "nswe"})])
        except Exception:
            pass

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.Frame(main_frame, width=340)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 6), pady=10)
        sidebar.pack_propagate(False)

        content = ttk.Frame(main_frame)
        content.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=(6, 10), pady=10)

        sidebar_bg = BG
        self.sidebar_canvas = tk.Canvas(sidebar, highlightthickness=0, bd=0, background=sidebar_bg)
        self.sidebar_scroll = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=self.sidebar_canvas.yview)
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scroll.set)
        self.sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sidebar_inner = ttk.Frame(self.sidebar_canvas)
        self._sidebar_window_id = self.sidebar_canvas.create_window(
            (0, 0), window=self.sidebar_inner, anchor="nw"
        )

        def _on_sidebar_inner_configure(_event):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

        def _on_sidebar_canvas_configure(event):
            self.sidebar_canvas.itemconfigure(self._sidebar_window_id, width=event.width)

        self.sidebar_inner.bind("<Configure>", _on_sidebar_inner_configure)
        self.sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        def _is_descendant(widget: Optional[tk.Misc], ancestor: tk.Misc) -> bool:
            while widget is not None:
                if widget == ancestor:
                    return True
                if widget == self:
                    return False
                widget = getattr(widget, "master", None)
            return False

        def _should_scroll_sidebar(event) -> bool:
            widget = self.winfo_containing(event.x_root, event.y_root)
            return _is_descendant(widget, self.sidebar_canvas)

        def _on_mousewheel(event):
            if not _should_scroll_sidebar(event):
                return
            if sys.platform == "darwin":
                delta = event.delta
            else:
                delta = int(event.delta / 120)
            if delta:
                self.sidebar_canvas.yview_scroll(-delta, "units")

        def _on_mousewheel_linux(event):
            if not _should_scroll_sidebar(event):
                return
            if event.num == 4:
                self.sidebar_canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.sidebar_canvas.yview_scroll(1, "units")

        self.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind_all("<Button-4>", _on_mousewheel_linux)
        self.bind_all("<Button-5>", _on_mousewheel_linux)

        # Data source
        self.file_frame = ttk.LabelFrame(self.sidebar_inner, text="Data source (none)")
        self.file_frame.pack(fill=tk.X, pady=6)
        file_row = ttk.Frame(self.file_frame)
        file_row.pack(fill=tk.X, pady=2)
        self.open_menu = tk.Menu(self, tearoff=0)
        self.open_menu.add_command(label="Open file(s)...", command=self.on_open)
        self.open_menu.add_command(label="Open folder...", command=self.on_open_folder)
        self.btn_open = ttk.Button(file_row, text="Open", command=self._show_open_menu)
        self.btn_open.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_prev_file = ttk.Button(file_row, text="<<", command=self.on_prev_file)
        self.btn_prev_file.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_next_file = ttk.Button(file_row, text=">>", command=self.on_next_file)
        self.btn_next_file.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # Playback
        playback_frame = ttk.LabelFrame(self.sidebar_inner, text="Playback")
        playback_frame.pack(fill=tk.X, pady=6)
        playback = ttk.Frame(playback_frame)
        playback.pack(fill=tk.X, pady=2)
        self.btn_play = ttk.Button(playback, text="Play", command=self.on_play_pause)
        self.btn_play.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.btn_prev = ttk.Button(playback, text="<", command=self.on_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.btn_next = ttk.Button(playback, text=">", command=self.on_next)
        self.btn_next.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        # Export settings
        cfg = ttk.LabelFrame(self.sidebar_inner, text="Export settings")
        cfg.pack(fill=tk.X, pady=6)
        cfg_grid = ttk.Frame(cfg)
        cfg_grid.pack(fill=tk.X, pady=2)
        cfg_grid.columnconfigure(1, weight=1)
        ttk.Label(cfg_grid, text="Detection threshold (C):").grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
        self.var_thresh = tk.DoubleVar(value=self.temp_threshold)
        ttk.Entry(cfg_grid, width=10, textvariable=self.var_thresh).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(cfg_grid, text="Metadata emissivity:").grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
        self.lbl_meta_emiss = ttk.Label(cfg_grid, text="-")
        self.lbl_meta_emiss.grid(row=1, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(cfg_grid, text="Emissivity override:").grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
        self.var_emissivity = tk.DoubleVar(value=0.9)
        self.entry_emissivity = ttk.Entry(cfg_grid, width=10, textvariable=self.var_emissivity)
        self.entry_emissivity.grid(row=2, column=1, sticky=tk.W, padx=4, pady=2)

        range_frame = ttk.LabelFrame(cfg, text="Export range (frames)")
        range_frame.pack(fill=tk.X, padx=6, pady=6)
        rf_row = ttk.Frame(range_frame)
        rf_row.pack(fill=tk.X, pady=2)
        ttk.Label(rf_row, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(rf_row, width=7, textvariable=self.var_export_start).pack(side=tk.LEFT, padx=(2, 8))

        end_note = ttk.Label(range_frame, text='End (type "max" for full length):', style="Muted.TLabel")
        end_note.pack(anchor=tk.W, pady=(0, 2))
        end_row = ttk.Frame(range_frame)
        end_row.pack(fill=tk.X, pady=2)
        ttk.Label(end_row, text="End:").pack(side=tk.LEFT)
        ttk.Entry(end_row, width=8, textvariable=self.var_export_end).pack(side=tk.LEFT, padx=(2, 4))
        rf_btns = ttk.Frame(range_frame)
        rf_btns.pack(fill=tk.X, pady=2)
        self.btn_set_start = ttk.Button(rf_btns, text="Set start", command=self.set_export_start)
        self.btn_set_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_set_end = ttk.Button(rf_btns, text="Set end", command=self.set_export_end)
        self.btn_set_end.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        roi_group = ttk.LabelFrame(cfg, text="Region of Interest")
        roi_group.pack(fill=tk.X, padx=6, pady=6)

        roi_tabs = ttk.Notebook(roi_group, style="ROI.TNotebook")
        roi_tabs.pack(fill=tk.X, expand=True)

        # Manual ROI tab
        manual_tab = ttk.Frame(roi_tabs)
        roi_tabs.add(manual_tab, text="Manual")
        roi_fields = ttk.Frame(manual_tab)
        roi_fields.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(roi_fields, text="X").pack(side=tk.LEFT)
        self.var_roi_x = tk.IntVar(value=0)
        self.entry_roi_x = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_x)
        self.entry_roi_x.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="Y").pack(side=tk.LEFT)
        self.var_roi_y = tk.IntVar(value=0)
        self.entry_roi_y = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_y)
        self.entry_roi_y.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="W").pack(side=tk.LEFT)
        self.var_roi_w = tk.IntVar(value=0)
        self.entry_roi_w = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_w)
        self.entry_roi_w.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="H").pack(side=tk.LEFT)
        self.var_roi_h = tk.IntVar(value=0)
        self.entry_roi_h = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_h)
        self.entry_roi_h.pack(side=tk.LEFT)
        btn_row = ttk.Frame(manual_tab)
        btn_row.pack(fill=tk.X, pady=2)
        self.btn_roi_update = ttk.Button(btn_row, text="Apply ROI", command=self.update_roi_from_fields)
        self.btn_roi_update.pack(side=tk.LEFT, padx=2, pady=1)
        self.btn_roi_clear = ttk.Button(btn_row, text="Reset ROI", command=self.clear_roi)
        self.btn_roi_clear.pack(side=tk.LEFT, padx=2, pady=1)

        # Auto ROI tab
        auto_tab = ttk.Frame(roi_tabs)
        roi_tabs.add(auto_tab, text="Auto")
        ttk.Label(auto_tab, text="Detect ROI above fuel bed using first frame.", style="Muted.TLabel").pack(anchor=tk.W, padx=4, pady=(4, 2))
        auto_row = ttk.Frame(auto_tab)
        auto_row.pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(auto_row, text="Margin (px):").pack(side=tk.LEFT)
        self.var_auto_margin = tk.IntVar(value=180)
        ttk.Entry(auto_row, width=6, textvariable=self.var_auto_margin).pack(side=tk.LEFT, padx=(4, 8))
        self.btn_auto_roi = ttk.Button(auto_tab, text="Auto-detect ROI", command=self.detect_auto_roi)
        self.btn_auto_roi.pack(fill=tk.X, padx=4, pady=4)
        self.lbl_auto_result = ttk.Label(auto_tab, text="Auto ROI: (not set)", style="Muted.TLabel")
        self.lbl_auto_result.pack(anchor=tk.W, padx=4, pady=(0, 4))

        apply_row = ttk.Frame(cfg)
        apply_row.pack(fill=tk.X, pady=(0, 4))
        self.btn_apply_current = ttk.Button(apply_row, text="Apply to current file", command=self.apply_current_settings)
        self.btn_apply_current.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_apply_all = ttk.Button(apply_row, text="Apply all", command=self.apply_current_settings_all)
        self.btn_apply_all.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self._update_apply_labels()

        # Export actions
        export_frame = ttk.LabelFrame(self.sidebar_inner, text="Export")
        export_frame.pack(fill=tk.X, pady=6)
        self.btn_export_menu = ttk.Button(export_frame, text="Export...", command=self.show_export_menu)
        self.btn_export_menu.pack(fill=tk.X, pady=2)

        footer = ttk.Frame(self.sidebar_inner)
        footer.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(
            footer,
            text=f"Developed from FLIR SDK by H. Nguyen ({APP_VERSION})",
            style="Muted.TLabel",
        ).pack(anchor=tk.W)
        ttk.Button(footer, text="Check for updates", command=self.on_check_updates).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

        # Image/canvas
        canvas_frame = ttk.Frame(content)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        # Status bar (kept at very bottom)
        self.status = ttk.Label(self, text="Status: ready", anchor=tk.W, style="Status.TLabel")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        slider_bar = ttk.Frame(self)
        slider_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.slider = ttk.Scale(slider_bar, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X, padx=2, pady=4)
    def _bind_events(self):
        # Global bindings so space toggles play/pause even when focus is on inputs
        self.bind_all("<Key-space>", lambda e: self.on_play_pause())
        self.bind("s", lambda e: self.on_stop())
        self.bind("<Key-Right>", lambda e: self.on_next())
        self.bind("<Key-Left>", lambda e: self.on_prev())
        self.bind("<Key-period>", lambda e: self.on_next())
        self.bind("<Key-comma>", lambda e: self.on_prev())
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
    def on_check_updates(self):
        try:
            req = urllib.request.Request(
                GITHUB_API_LATEST_URL,
                headers={"User-Agent": "Firebrand-Thermal-Analysis"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as ex:
            messagebox.showerror("Updates", f"Could not check for updates.\n{ex}")
            return

        latest = data.get("tag_name") or ""
        release_url = data.get("html_url") or GITHUB_RELEASES_URL
        if not latest:
            messagebox.showinfo(
                "Updates",
                f"Current version: {APP_VERSION}\nNo release info found.",
            )
            return

        if _is_newer_version(APP_VERSION, latest):
            open_now = messagebox.askyesno(
                "Update available",
                f"New version {latest} is available (current {APP_VERSION}).\n"
                "Open the release page?",
            )
            if open_now:
                webbrowser.open(release_url)
        else:
            open_now = messagebox.askyesno(
                "Up to date",
                f"You are up to date ({APP_VERSION}).\nOpen the release page?",
            )
            if open_now:
                webbrowser.open(release_url)
    def show_export_menu(self):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Export CSV (current)", command=self.export_video_csv)
        menu.add_command(label="Export CSV (all files)", command=self.export_video_csv_all)
        menu.add_command(label="Save frame image (JPG)", command=self.export_frame)
        try:
            x = self.btn_export_menu.winfo_rootx()
            y = self.btn_export_menu.winfo_rooty() + self.btn_export_menu.winfo_height()
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()
    def _reset_tracking(self):
        self.tracked_objects = OrderedDict()
        self.next_track_id = 1
        self.last_tracked_frame = -1
    def set_export_start(self):
        if self.im is None:
            return
        frame_num = self.current_idx + 1
        self.var_export_start.set(frame_num)
        self._update_range_button_labels()
    def set_export_end(self):
        if self.im is None:
            return
        frame_num = self.current_idx + 1
        self.var_export_end.set(frame_num)
        self._update_range_button_labels()
    def _update_file_label(self):
        total = len(self.batch_paths)
        name = os.path.basename(self.seq_path) if self.seq_path else "none"
        idx_str = f" {self.batch_index+1}/{total}" if total else ""
        if hasattr(self, "file_frame"):
            self.file_frame.configure(text=f"Data source ({name}{idx_str})")
        self._update_apply_labels()
    def _update_apply_labels(self):
        if not hasattr(self, "btn_apply_current") or not hasattr(self, "btn_apply_all"):
            return
        file_name = os.path.basename(self.seq_path) if self.seq_path else "current file"
        self.btn_apply_current.configure(text=f"Apply to {file_name}")
        self.btn_apply_all.configure(text="Apply all")
    def _clamp_roi_to_frame(self):
        if self.width <= 0 or self.height <= 0:
            return
        try:
            x = max(0, int(self.var_roi_x.get()))
            y = max(0, int(self.var_roi_y.get()))
            w = int(self.var_roi_w.get())
            h = int(self.var_roi_h.get())
        except Exception:
            x, y, w, h = 0, 0, self.width, self.height
        if w <= 0:
            w = self.width
        if h <= 0:
            h = self.height
        w = min(self.width - x, w)
        h = min(self.height - y, h)
        w = max(1, w); h = max(1, h)
        self.roi_rect = (x, y, w, h)
        self.update_roi_fields_from_rect()
    def _update_metadata_label(self, emiss: Optional[float]):
        if hasattr(self, "lbl_meta_emiss"):
            text = "-" if emiss is None else f"{emiss:.3f}"
            self.lbl_meta_emiss.configure(text=text)
    def _update_range_button_labels(self):
        if not hasattr(self, "btn_set_start") or not hasattr(self, "btn_set_end"):
            return
        frame_num = self.current_idx + 1 if self.im is not None else 0
        if frame_num > 0:
            self.btn_set_start.configure(text=f"Start = {frame_num}")
            self.btn_set_end.configure(text=f"End = {frame_num}")
        else:
            self.btn_set_start.configure(text="Set start")
            self.btn_set_end.configure(text="Set end")
    def _current_settings_snapshot(self) -> dict:
        self.update_roi_from_fields()
        try:
            start_f = int(self.var_export_start.get())
        except Exception:
            start_f = 1
        end_f = str(self.var_export_end.get()).strip() or "max"
        try:
            thresh = float(self.var_thresh.get())
        except Exception:
            thresh = self.temp_threshold
        try:
            emiss = float(self.var_emissivity.get())
        except Exception:
            emiss = 0.9
        return {
            "thresh": thresh,
            "emissivity": emiss,
            "export_start": start_f,
            "export_end": end_f,
            "roi": self.roi_rect,
        }
    def _auto_roi_from_frame(self, frame: np.ndarray, margin: int = 20) -> Tuple[int, int, int, int]:
        h, w = frame.shape
        global_mean = float(frame.mean())
        global_std = float(frame.std())
        if global_std <= 0:
            return (0, 0, w, max(1, int(0.4 * h)))
        thresh = global_mean + 0.5 * global_std
        bottom = frame[h // 2 :, :]
        row_mean = bottom.mean(axis=1)
        above = row_mean > thresh
        best_start = None
        best_len = 0
        cur_start = None
        cur_len = 0
        for idx, val in enumerate(above):
            if val:
                if cur_start is None:
                    cur_start = idx
                    cur_len = 1
                else:
                    cur_len += 1
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = None
                cur_len = 0
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
        if best_start is not None:
            fuel_top = (h // 2) + best_start
        else:
            fuel_top = int(0.6 * h)
        roi_h = max(1, fuel_top - max(0, margin))
        roi_h = min(h, roi_h)
        return (0, 0, w, roi_h)
    def detect_auto_roi(self):
        if self._export_in_progress:
            return
        if self.im is None:
            messagebox.showerror("Auto ROI", "Open a file first.")
            return
        try:
            self.im.get_frame(0)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            margin = max(0, int(self.var_auto_margin.get()))
            roi = self._auto_roi_from_frame(frame, margin=margin)
            self.roi_rect = roi
            self.update_roi_fields_from_rect()
            self.status.configure(text=f"Status: auto ROI set to {roi}")
            if hasattr(self, "lbl_auto_result"):
                self.lbl_auto_result.configure(text=f"Auto ROI: {roi}")
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Auto ROI", f"Auto ROI failed: {ex}")
    def _apply_settings(self, settings: dict):
        self.var_thresh.set(settings.get("thresh", self.temp_threshold))
        self.var_emissivity.set(settings.get("emissivity", 0.9))
        self.var_export_start.set(settings.get("export_start", 1))
        self.var_export_end.set(settings.get("export_end", "max"))
        self.roi_rect = settings.get("roi")
        self.update_roi_fields_from_rect()
    def apply_current_settings(self):
        if not self.seq_path:
            return
        self.file_settings[self.seq_path] = self._current_settings_snapshot()
        self.apply_configuration(update_status=False)
        self.status.configure(text=f"{self._base_status} | settings applied to current file")
    def apply_current_settings_all(self):
        settings = self._current_settings_snapshot()
        targets = self.batch_paths or ([self.seq_path] if self.seq_path else [])
        for path in targets:
            self.file_settings[path] = dict(settings)
        self.apply_configuration(update_status=False)
        self.status.configure(text=f"{self._base_status} | settings applied to all files")
    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets = [
            getattr(self, "btn_open", None),
            getattr(self, "btn_prev_file", None),
            getattr(self, "btn_next_file", None),
            getattr(self, "btn_play", None),
            getattr(self, "btn_prev", None),
            getattr(self, "btn_next", None),
            getattr(self, "btn_set_start", None),
            getattr(self, "btn_set_end", None),
            getattr(self, "btn_roi_update", None),
            getattr(self, "btn_roi_clear", None),
            getattr(self, "btn_auto_roi", None),
            getattr(self, "btn_apply_current", None),
            getattr(self, "btn_apply_all", None),
            getattr(self, "btn_export_menu", None),
            getattr(self, "entry_emissivity", None),
            getattr(self, "entry_roi_x", None),
            getattr(self, "entry_roi_y", None),
            getattr(self, "entry_roi_w", None),
            getattr(self, "entry_roi_h", None),
        ]
        for w in widgets:
            if w is None:
                continue
            try:
                w.configure(state=state)
            except Exception:
                pass
        try:
            self.slider.configure(state=state)
        except Exception:
            pass
    def _set_export_busy(self, busy: bool):
        self._export_in_progress = busy
        if busy:
            self.playing = False
            try:
                self.btn_play.configure(text="Play")
            except Exception:
                pass
        self._set_controls_enabled(not busy)
    def _default_parallel_workers(self, task_count: int) -> int:
        cpu = os.cpu_count() or 1
        return max(1, min(task_count, max(1, cpu - 1)))
    def _show_open_menu(self):
        try:
            x = self.btn_open.winfo_rootx()
            y = self.btn_open.winfo_rooty() + self.btn_open.winfo_height()
            self.open_menu.tk_popup(x, y)
        finally:
            try:
                self.open_menu.grab_release()
            except Exception:
                pass
    def on_open(self, path_override: Optional[str] = None):
        if path_override:
            self._load_seq(path_override, reset_settings=False)
            return
        paths = filedialog.askopenfilenames(
            title="Select radiometric file(s)",
            filetypes=[("Radiometric Files", [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]), ("All Files", "*.*")],
        )
        seq_paths = [p for p in paths if str(p).lower().endswith(SUPPORTED_EXTENSIONS)] if paths else []
        if not seq_paths:
            return
        self._set_batch_paths(seq_paths)
    def on_open_folder(self, folder_override: Optional[str] = None):
        folder = folder_override or filedialog.askdirectory(title="Select folder containing radiometric files")
        if not folder:
            return
        seq_paths = []
        for root, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith("."):
                    continue
                if name.lower().endswith(SUPPORTED_EXTENSIONS):
                    seq_paths.append(os.path.join(root, name))
        if not seq_paths:
            messagebox.showinfo(
                "Open files",
                "No radiometric files found in the selected folder (including subfolders).",
            )
            return
        self._set_batch_paths(seq_paths)
    def _set_batch_paths(self, seq_paths: List[str]):
        seq_paths = sorted(dict.fromkeys(seq_paths))
        self.batch_paths = seq_paths
        self.batch_index = 0
        self._load_seq(seq_paths[0], reset_settings=True)
    def _load_seq(self, path: str, reset_settings: bool):
        try:
            im = fnv.file.ImagerFile(path)
            self.unit_is_temp = im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
            if self.unit_is_temp:
                im.unit = fnv.Unit.TEMPERATURE_FACTORY
                im.temp_type = DESIRED_TEMP_UNIT
                self.unit_label = "C"
            else:
                im.unit = fnv.Unit.COUNTS
                self.unit_label = "counts"
            self.im = im
            self.seq_path = path
            if path in self.batch_paths:
                self.batch_index = self.batch_paths.index(path)
            self.num_frames = im.num_frames
            self.width = im.width
            self.height = im.height
            self.current_idx = 0
            self.slider.configure(from_=0, to=max(0, self.num_frames-1))
            self._reset_tracking()
            self._applied_emissivity = None
            self.status.configure(text=f"Status: opened {os.path.basename(path)} | {self.width}x{self.height} | {self.num_frames} frames")
            meta_emiss = None
            try:
                meta_emiss = float(im.object_parameters.emissivity)
            except Exception:
                meta_emiss = None
            self._update_metadata_label(meta_emiss)
            if path in self.file_settings:
                self._apply_settings(self.file_settings[path])
            else:
                if reset_settings:
                    self.var_export_start.set(1)
                    self.var_export_end.set("max")
                if meta_emiss is not None:
                    self.var_emissivity.set(meta_emiss)
                self.roi_rect = None
                self.update_roi_fields_from_rect()
            self._clamp_roi_to_frame()
            self.apply_configuration(update_status=False)
            self._update_file_label()
            if reset_settings:
                self.apply_current_settings_all()
            self._update_range_button_labels()
            self.after(10, self._render_current)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Open error", f"Failed to open file.\n{e}")
    def on_play_pause(self):
        if self._export_in_progress:
            return
        if self.im is None: return
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._last_frame_time = time.monotonic()
            self._start_prefetch()
            self.after(1, self._play_loop)
        else:
            self._stop_prefetch()
    def on_stop(self):
        if self.im is None: return
        self.playing = False
        self._stop_prefetch()
        self.btn_play.configure(text="Play")
        self.current_idx = 0
        self._reset_tracking()
        self._render_current()
    def on_prev(self):
        if self._export_in_progress:
            return
        if self.im is None: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = max(0, self.current_idx - 1)
        self._render_current()
    def on_next(self):
        if self._export_in_progress:
            return
        if self.im is None: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = min(self.num_frames - 1, self.current_idx + 1)
        self._render_current()
    def on_prev_file(self):
        if self._export_in_progress:
            return
        if not self.batch_paths:
            return
        self.batch_index = (self.batch_index - 1) % len(self.batch_paths)
        self._load_seq(self.batch_paths[self.batch_index], reset_settings=False)
    def on_next_file(self):
        if self._export_in_progress:
            return
        if not self.batch_paths:
            return
        self.batch_index = (self.batch_index + 1) % len(self.batch_paths)
        self._load_seq(self.batch_paths[self.batch_index], reset_settings=False)
    def on_slider(self, val):
        if self._export_in_progress:
            return
        if self.im is None: return
        if self._in_slider_update:
            return
        try: idx = int(float(val))
        except Exception: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = max(0, min(self.num_frames-1, idx))
        self._render_current()
    def update_roi_from_fields(self):
        if self._export_in_progress:
            return
        try:
            x = max(0, int(self.var_roi_x.get()))
            y = max(0, int(self.var_roi_y.get()))
            w = max(1, int(self.var_roi_w.get()))
            h = max(1, int(self.var_roi_h.get()))
            if self.width > 0 and self.height > 0:
                w = min(self.width-x, w)
                h = min(self.height-y, h)
            self.roi_rect = (x, y, w, h)
            self._render_current()
        except Exception:
            pass
    def apply_configuration(self, update_status: bool = True):
        if self.im is None:
            return
        try:
            emiss = float(self.var_emissivity.get())
            emiss = max(0.01, min(1.0, emiss))
            obj_params = self.im.object_parameters
            obj_params.emissivity = emiss
            self.im.object_parameters = obj_params
            self._applied_emissivity = emiss
            if update_status:
                self.status.configure(text=f"{self._base_status} | Emissivity: {emiss:.3f}")
        except Exception as ex:
            if update_status:
                messagebox.showerror("Configuration", f"Failed to apply configuration: {ex}")
    def _ensure_emissivity(self):
        if self.im is None:
            return
        try:
            emiss = float(self.var_emissivity.get())
            emiss = max(0.01, min(1.0, emiss))
        except Exception:
            return
        if self._applied_emissivity is None or abs(self._applied_emissivity - emiss) > 1e-4:
            try:
                obj_params = self.im.object_parameters
                obj_params.emissivity = emiss
                self.im.object_parameters = obj_params
                self._applied_emissivity = emiss
            except Exception:
                pass
    def _bbox_iou(self, box_a: Tuple[int,int,int,int], box_b: Tuple[int,int,int,int]) -> float:
        ax0, ay0, aw, ah = box_a
        bx0, by0, bw, bh = box_b
        ax1 = ax0 + aw; ay1 = ay0 + ah
        bx1 = bx0 + bw; by1 = by0 + bh
        inter_x0 = max(ax0, bx0); inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1); inter_y1 = min(ay1, by1)
        inter_w = max(0, inter_x1 - inter_x0)
        inter_h = max(0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = aw * ah
        area_b = bw * bh
        return inter_area / float(area_a + area_b - inter_area)
    def _fit_display_size(self) -> Tuple[int, int]:
        if self.width == 0 or self.height == 0:
            return 0, 0
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        scale = min(cw / float(self.width), ch / float(self.height))
        scale = max(scale, 0.01)
        disp_w = max(1, int(self.width * scale))
        disp_h = max(1, int(self.height * scale))
        return disp_w, disp_h
    def on_canvas_resize(self, event):
        self._canvas_size = (event.width, event.height)
        if self.im is None:
            return
        if self._resize_job:
            try:
                self.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.after(50, self._render_current)
    def clear_roi(self):
        self.roi_rect = None
        self.var_roi_x.set(0)
        self.var_roi_y.set(0)
        self.var_roi_w.set(self.width)
        self.var_roi_h.set(self.height)
        self._render_current()
    def on_mouse_down(self, e):
        if self._export_in_progress:
            return
        if self.im is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        self.mouse_down = (xi, yi)
        self.roi_rect = (xi, yi, 1, 1)
        self.update_roi_fields_from_rect()
        self._render_current()
    def on_mouse_drag(self, e):
        if self._export_in_progress:
            return
        if self.im is None or self.mouse_down is None: return
        x0, y0 = self.mouse_down
        x1, y1 = self._canvas_to_image(e.x, e.y)
        rx = max(0, min(x0, x1)); ry = max(0, min(y0, y1))
        rw = max(1, min(self.width - rx, abs(x1 - x0)))
        rh = max(1, min(self.height - ry, abs(y1 - y0)))
        self.roi_rect = (rx, ry, rw, rh)
        self.update_roi_fields_from_rect()
        self._render_current()
    def on_mouse_up(self, e):
        if self._export_in_progress:
            return
        self.mouse_down = None
        self.update_roi_fields_from_rect()
    def on_mouse_move(self, e):
        if self._export_in_progress:
            return
        if self.im is None or self._last_frame is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        if xi < 0 or yi < 0 or xi >= self.width or yi >= self.height:
            return
        try:
            temp_val = float(self._last_frame[yi, xi])
        except Exception:
            return
        unit_label = "C"
        try:
            if hasattr(self.im, 'unit') and self.im.unit != fnv.Unit.TEMPERATURE_FACTORY:
                unit_label = "counts"
            else:
                if hasattr(self.im, 'temp_type'):
                    if self.im.temp_type == fnv.TempType.KELVIN: unit_label = "K"
                    elif self.im.temp_type == fnv.TempType.FAHRENHEIT: unit_label = "F"
        except Exception:
            pass
        self.status.configure(text=f"{self._base_status} | Cursor: ({xi},{yi}) {temp_val:.1f}{unit_label}")
    def update_roi_fields_from_rect(self):
        if self.roi_rect:
            x, y, w, h = self.roi_rect
            self.var_roi_x.set(x); self.var_roi_y.set(y); self.var_roi_w.set(w); self.var_roi_h.set(h)
        else:
            self.var_roi_x.set(0); self.var_roi_y.set(0); self.var_roi_w.set(self.width); self.var_roi_h.set(self.height)
    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int,int]:
        bbox = self.canvas.bbox("img");
        if not bbox: return 0, 0
        x0, y0, x1, y1 = bbox
        draw_w = max(1, x1 - x0); draw_h = max(1, y1 - y0)
        if self.width == 0 or self.height == 0: return 0, 0
        xi = int((cx - x0) * self.width / draw_w)
        yi = int((cy - y0) * self.height / draw_h)
        xi = max(0, min(self.width-1, xi)); yi = max(0, min(self.height-1, yi))
        return xi, yi
    def _detect_firebrands(self, frame: np.ndarray, roi: Optional[Tuple[int,int,int,int]], temp_thresh: float):
        roi = clamp_roi(roi, self.width, self.height)
        return detect_firebrands(frame, roi, temp_thresh)
    def _assign_tracks(self, detections: List[dict], tracked_objects: OrderedDict, next_id: int, frame_idx: int):
        return assign_tracks(detections, tracked_objects, next_id, frame_idx)
    def _get_tracked_detections(self, frame_idx: int, frame: np.ndarray,
                                roi: Optional[Tuple[int,int,int,int]], temp_thresh: float,
                                update_tracker: bool = True) -> List[dict]:
        detections = self._detect_firebrands(frame, roi, temp_thresh)
        prev_tracker = OrderedDict(self.tracked_objects) if self.tracked_objects is not None else OrderedDict()
        tracker = OrderedDict(prev_tracker)
        next_id = self.next_track_id
        last_frame = self.last_tracked_frame
        if frame_idx != (last_frame + 1):
            tracker = OrderedDict()
            next_id = 1
        tracker, detections, next_id = self._assign_tracks(detections, tracker, next_id, frame_idx)
        for det in detections:
            det['new_track'] = det['track_id'] not in prev_tracker
        if update_tracker:
            self.tracked_objects = tracker
            self.next_track_id = next_id
            self.last_tracked_frame = frame_idx
        return detections
    def _stop_prefetch(self):
        """Stop the background prefetch thread if running."""
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=0.5)
        self._prefetch_thread = None
        # Drain the queue
        while not self._prefetch_queue.empty():
            try:
                self._prefetch_queue.get_nowait()
            except queue.Empty:
                break
    def _start_prefetch(self):
        """Start a background thread that pre-decodes frames ahead of playback."""
        self._stop_prefetch()
        self._prefetch_stop.clear()

        def _worker():
            idx = self.current_idx + 1
            while not self._prefetch_stop.is_set() and idx < self.num_frames:
                try:
                    self.im.get_frame(idx)
                    frame = np.array(self.im.final, copy=True).reshape(
                        (self.height, self.width)
                    )
                    self._prefetch_queue.put((idx, frame), timeout=0.5)
                    idx += 1
                except queue.Full:
                    if self._prefetch_stop.is_set():
                        break
                except Exception:
                    break

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()
    def _play_loop(self):
        if not self.playing or self.im is None:
            return
        now = time.monotonic()
        frame_interval = 1.0 / max(1, TARGET_FPS)
        elapsed = now - self._last_frame_time
        if elapsed < frame_interval:
            delay_ms = max(1, int((frame_interval - elapsed) * 1000))
            self.after(delay_ms, self._play_loop)
            return
        self._last_frame_time = now
        self.current_idx += 1
        if self.current_idx >= self.num_frames:
            self.current_idx = self.num_frames - 1
            self.playing = False
            self._stop_prefetch()
            self.btn_play.configure(text="Play")
            return
        self._render_current()
        self.after(1, self._play_loop)
    def _render_current(self):
        if self.im is None: return
        try:
            self._ensure_emissivity()
            self.im.get_frame(self.current_idx)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            self._last_frame = frame
            roi = self.roi_rect
            temp_thresh = float(self.var_thresh.get())
            detections = self._get_tracked_detections(self.current_idx, frame, roi, temp_thresh, update_tracker=True)
            vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            vis = np.clip(vis, 0, 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if roi:
                rx, ry, rw, rh = roi
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0,255,0), 1)
            for d in detections:
                x,y,w,h = d['bbox']; cx,cy = map(int, d['centroid'])
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.circle(vis, (cx, cy), 3, (255,255,255), -1)
                prefix = "Detect" if d.get('new_track') else "Tracking"
                cv2.putText(vis, f"{prefix}: {d['max_temp']:.1f}{self.unit_label}", (x, max(0, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            disp_w, disp_h = self._fit_display_size()
            vis_for_canvas = vis
            if disp_w > 0 and disp_h > 0 and (disp_w != vis.shape[1] or disp_h != vis.shape[0]):
                interp = cv2.INTER_AREA if disp_w < vis.shape[1] or disp_h < vis.shape[0] else cv2.INTER_LINEAR
                vis_for_canvas = cv2.resize(vis, (disp_w, disp_h), interpolation=interp)
            img_rgb = cv2.cvtColor(vis_for_canvas, cv2.COLOR_BGR2RGB)
            drawn = False
            if Image is not None and ImageTk is not None:
                try:
                    im_pil = Image.fromarray(img_rgb)
                    self._tk_img = ImageTk.PhotoImage(im_pil)
                    drawn = True
                except Exception:
                    traceback.print_exc()
            if not drawn:
                try:
                    ok, buf = cv2.imencode(".png", vis_for_canvas)
                    if ok:
                        data = base64.b64encode(buf.tobytes())
                        self._tk_img = tk.PhotoImage(data=data)
                        drawn = True
                except Exception:
                    traceback.print_exc()
            if drawn:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self._tk_img, anchor=tk.NW, tags=("img",))
            try:
                self._in_slider_update = True
                self.slider.set(float(self.current_idx))
            finally:
                self._in_slider_update = False
            roistr = f"ROI: {self.roi_rect if self.roi_rect else 'Full'}"
            self._base_status = f"Status: frame {self.current_idx+1}/{self.num_frames} | {roistr} | thresh: {temp_thresh:.1f}{self.unit_label}"
            if not self._export_in_progress:
                self.status.configure(text=self._base_status)
            self._update_range_button_labels()
        except Exception:
            traceback.print_exc()
    def export_frame(self):
        try:
            if not self.seq_path:
                messagebox.showerror("Export", "Open a file first.")
                return
            self._ensure_emissivity()
            self.im.get_frame(self.current_idx)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            roi = self.roi_rect
            temp_thresh = float(self.var_thresh.get())
            detections = self._get_tracked_detections(self.current_idx, frame, roi, temp_thresh, update_tracker=False)
            vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            vis = np.clip(vis, 0, 255).astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if roi:
                rx, ry, rw, rh = roi
                cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), (0,255,0), 1)
            for d in detections:
                x,y,w,h = d['bbox']; cx,cy = map(int, d['centroid'])
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.circle(vis, (cx, cy), 3, (255,255,255), -1)
                prefix = "Detect" if d.get('new_track') else "Tracking"
                cv2.putText(vis, f"{prefix}: {d['max_temp']:.1f}C", (x, max(0, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            # Save
            base = os.path.splitext(os.path.basename(self.seq_path))[0]
            out_dir = os.path.dirname(self.seq_path)
            out_path = os.path.join(out_dir, f"{base}_frame_{self.current_idx+1:05d}.jpg")
            ok = cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY),95])
            messagebox.showinfo("Export", f"Exported frame to: {out_path}" if ok else "Export failed!")
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Export", f"Export failed: {ex}")
    def export_video_csv(self):
        if not self.seq_path:
            messagebox.showerror("Export", "Open a file first.")
            return
        self._export_csv_for_paths([self.seq_path])
    def export_video_csv_all(self):
        paths = self.batch_paths if self.batch_paths else ([self.seq_path] if self.seq_path else [])
        if not paths:
            messagebox.showerror("Export", "Open a file first.")
            return
        if self._export_in_progress:
            messagebox.showinfo("Export", "Export is already running.")
            return
        if len(paths) <= 1:
            self._export_csv_for_paths(paths)
            return
        self._export_csv_for_paths_parallel(paths)
    def _export_csv_for_paths_parallel(self, paths: List[str]):
        try:
            self.update_roi_from_fields()
        except Exception:
            pass

        default_settings = self._current_settings_snapshot()
        tasks: List[Tuple[str, Dict[str, Any]]] = []
        for seq_path in paths:
            settings = dict(self.file_settings.get(seq_path, default_settings))
            tasks.append((seq_path, settings))

        max_workers = self._default_parallel_workers(len(tasks))
        base_status = self._base_status
        total = len(tasks)

        self._set_export_busy(True)
        self.status.configure(
            text=f"Status: exporting CSV (parallel, {max_workers} workers) 0/{total}"
        )
        self.update_idletasks()

        def _run_exports():
            completed = 0
            errors: List[Tuple[str, str]] = []
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(export_seq_to_csv_worker, seq_path, settings): seq_path
                        for (seq_path, settings) in tasks
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        seq_path = future_map[future]
                        try:
                            _, out_path, err = future.result()
                        except Exception as ex:
                            out_path = ""
                            err = f"{ex}\n{traceback.format_exc()}"

                        if err:
                            errors.append((seq_path, err))

                        completed += 1
                        name = os.path.basename(seq_path)
                        status_text = (
                            f"Status: exporting CSV (parallel) {completed}/{total} done | last: {name}"
                        )
                        self.after(0, lambda t=status_text: self.status.configure(text=t))
            finally:
                def _finish():
                    self._set_export_busy(False)
                    if errors:
                        msg_lines = ["Export finished with errors:"]
                        for seq_path, err in errors[:5]:
                            first_line = (err.splitlines() or ["error"])[0]
                            msg_lines.append(f"- {os.path.basename(seq_path)}: {first_line}")
                        if len(errors) > 5:
                            msg_lines.append(f"... and {len(errors) - 5} more")
                        messagebox.showerror("Export", "\n".join(msg_lines))
                        self.status.configure(text=f"{base_status} | export completed with errors")
                    else:
                        messagebox.showinfo("Export", "CSV export complete.")
                        self.status.configure(text=f"{base_status} | CSV export complete")

                self.after(0, _finish)

        self._export_thread = threading.Thread(target=_run_exports, daemon=True)
        self._export_thread.start()
    def _export_csv_for_paths(self, paths: List[str]):
        try:
            self._ensure_emissivity()
            self.update_roi_from_fields()
            original_width, original_height = self.width, self.height
            default_settings = self._current_settings_snapshot()
            for p_idx, seq_path in enumerate(paths):
                settings = self.file_settings.get(seq_path, default_settings)
                try:
                    start_f = max(1, int(settings.get("export_start", 1)))
                except Exception:
                    start_f = 1
                end_raw = str(settings.get("export_end", "max")).strip().lower()
                temp_thresh = float(settings.get("thresh", self.temp_threshold))
                roi = settings.get("roi")
                try:
                    im = fnv.file.ImagerFile(seq_path)
                except Exception as ex:
                    traceback.print_exc()
                    messagebox.showerror("Export", f"Failed to open {seq_path}: {ex}")
                    continue
                unit_is_temp = im.has_unit(fnv.Unit.TEMPERATURE_FACTORY)
                if unit_is_temp:
                    im.unit = fnv.Unit.TEMPERATURE_FACTORY
                    im.temp_type = DESIRED_TEMP_UNIT
                else:
                    im.unit = fnv.Unit.COUNTS
                try:
                    obj_params = im.object_parameters
                    obj_params.emissivity = float(settings.get("emissivity", 0.9))
                    im.object_parameters = obj_params
                except Exception:
                    pass
                num_frames = im.num_frames
                self.width = im.width
                self.height = im.height
                try:
                    end_f = num_frames if end_raw in ("max", "", "none") else int(end_raw)
                except Exception:
                    end_f = num_frames
                if end_f <= 0 or end_f > num_frames:
                    end_f = num_frames
                start_use = max(1, min(start_f, num_frames))
                if start_use > end_f:
                    start_use = end_f
                tracked = OrderedDict()
                next_id = 1
                rows = []
                for idx in range(start_use - 1, end_f):
                    im.get_frame(idx)
                    frame = np.array(im.final, copy=False).reshape((self.height, self.width))
                    detections = self._detect_firebrands(frame, roi, temp_thresh)
                    tracked, detections, next_id = self._assign_tracks(detections, tracked, next_id, idx)
                    for det in sorted(detections, key=lambda d: d.get('track_id', -1)):
                        x, y, w, h = det['bbox']
                        rows.append((
                            idx + 1,
                            det.get('track_id', -1),
                            det['max_temp'],
                            det.get('min_temp', det['max_temp']),
                            det.get('avg_temp', det['max_temp']),
                            det.get('median_temp', det['max_temp']),
                            det.get('area', 0),
                            x, y, w, h
                        ))
                    self.status.configure(text=f"Status: exporting CSV [{p_idx+1}/{len(paths)}]: {os.path.basename(seq_path)} frame {idx+1}/{num_frames} (range {start_use}-{end_f})")
                    self.update_idletasks()
                out_path = str(Path(seq_path).with_suffix(".csv"))
                with open(out_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "frame",
                        "firebrand_id",
                        "max_temperature",
                        "min_temperature",
                        "avg_temperature",
                        "median_temperature",
                        "area_pixels",
                        "bbox_x",
                        "bbox_y",
                        "bbox_w",
                        "bbox_h",
                    ])
                    writer.writerows(rows)
                self.status.configure(text=f"Status: CSV saved: {out_path}")
            self.width = original_width
            self.height = original_height
            messagebox.showinfo("Export", "CSV export complete.")
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Export", f"Export failed: {ex}")

def main():
    multiprocessing.freeze_support()
    app = SKDDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
