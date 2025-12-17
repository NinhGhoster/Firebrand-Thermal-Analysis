"""
Tkinter dashboard GUI for FLIR SDK tracking-based hotspot/ember detection.
- Preserves SDK.py tracking/detection logic (OrderedDict tracker, ROI, overlays)
- Left sidebar: Play, pause, frame slider, prev/next, ROI numeric/manual, threshold, export frame
- Canvas: shows current frame with overlays
"""
import base64
import os
import sys
import traceback
from typing import Optional, Tuple, List
import csv
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
    print("OpenCV required: pip install opencv-python"); sys.exit(1)
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

class SKDDashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ninh Firebrand Detection and Tracking Dashboard")
        self.geometry("1100x700")
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
        self._base_status = "Ready"
        self.var_export_start = tk.IntVar(value=1)
        self.var_export_end = tk.StringVar(value="max")
        self.batch_paths: List[str] = []
        self.batch_index: int = 0
        self.file_settings = {}
        self._build_ui()
        self._bind_events()
        self._reset_tracking()
    def _build_ui(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        dashboard = ttk.Frame(main_frame)
        dashboard.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        # Files
        file_frame = ttk.LabelFrame(dashboard, text="Files")
        file_frame.pack(fill=tk.X, pady=4)
        self.btn_open = ttk.Button(file_frame, text="Open SEQ", command=self.on_open)
        self.btn_open.pack(fill=tk.X, pady=2)
        self.lbl_file = ttk.Label(file_frame, text="File: (none)", wraplength=240)
        self.lbl_file.pack(anchor=tk.W, pady=(2,2))
        file_nav = ttk.Frame(file_frame); file_nav.pack(fill=tk.X, pady=2)
        ttk.Button(file_nav, text="Prev File", command=self.on_prev_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(file_nav, text="Next File", command=self.on_next_file).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        # Playback
        playback_frame = ttk.LabelFrame(dashboard, text="Playback")
        playback_frame.pack(fill=tk.X, pady=4)
        playback = ttk.Frame(playback_frame); playback.pack(fill=tk.X, pady=2)
        self.btn_play = ttk.Button(playback, text="Play", command=self.on_play_pause)
        self.btn_play.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.btn_stop = ttk.Button(playback, text="Stop", command=self.on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        nav_frame = ttk.Frame(playback_frame); nav_frame.pack(fill=tk.X)
        self.btn_prev = ttk.Button(nav_frame, text="Prev", command=self.on_prev, width=6)
        self.btn_prev.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        self.btn_next = ttk.Button(nav_frame, text="Next", command=self.on_next, width=6)
        self.btn_next.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        # Configuration
        cfg = ttk.LabelFrame(dashboard, text="Configuration for Exporting")
        cfg.pack(fill=tk.X, pady=4)
        cfg_top = ttk.Frame(cfg); cfg_top.pack(fill=tk.X, pady=2)
        ttk.Label(cfg_top, text="Threshold (°C):").pack(side=tk.LEFT)
        self.var_thresh = tk.DoubleVar(value=self.temp_threshold)
        ttk.Entry(cfg_top, width=8, textvariable=self.var_thresh).pack(side=tk.LEFT, padx=4)
        emiss_frame = ttk.Frame(cfg); emiss_frame.pack(fill=tk.X, pady=2)
        self.lbl_meta_emiss = ttk.Label(emiss_frame, text="Metadata emissivity: -")
        self.lbl_meta_emiss.pack(side=tk.LEFT, padx=(0,6))
        ttk.Label(emiss_frame, text="Emissivity:").pack(side=tk.LEFT)
        self.var_emissivity = tk.DoubleVar(value=0.9)
        self.entry_emissivity = ttk.Entry(emiss_frame, width=8, textvariable=self.var_emissivity)
        self.entry_emissivity.pack(side=tk.LEFT, padx=4)
        range_frame = ttk.LabelFrame(cfg, text="Export Range (frames)")
        range_frame.pack(fill=tk.X, pady=3)
        rf_row = ttk.Frame(range_frame); rf_row.pack(fill=tk.X, pady=1)
        ttk.Label(rf_row, text="Start:").pack(side=tk.LEFT)
        ttk.Entry(rf_row, width=6, textvariable=self.var_export_start).pack(side=tk.LEFT, padx=2)
        ttk.Label(rf_row, text="End:").pack(side=tk.LEFT)
        ttk.Entry(rf_row, width=8, textvariable=self.var_export_end).pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text="Type max if you don't want to set a value").pack(anchor=tk.W, pady=(0,2))
        rf_btns = ttk.Frame(range_frame); rf_btns.pack(fill=tk.X, pady=2)
        self.btn_set_start = ttk.Button(rf_btns, text="Set Start from Current Frame", command=self.set_export_start)
        self.btn_set_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.btn_set_end = ttk.Button(rf_btns, text="Set End from Current Frame", command=self.set_export_end)
        self.btn_set_end.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        roi_group = ttk.LabelFrame(cfg, text="ROI")
        roi_group.pack(fill=tk.X, pady=3)
        roi_fields = ttk.Frame(roi_group); roi_fields.pack(fill=tk.X)
        ttk.Label(roi_fields, text="X").pack(side=tk.LEFT); self.var_roi_x = tk.IntVar(value=0)
        self.entry_roi_x = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_x); self.entry_roi_x.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="Y").pack(side=tk.LEFT); self.var_roi_y = tk.IntVar(value=0)
        self.entry_roi_y = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_y); self.entry_roi_y.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="W").pack(side=tk.LEFT); self.var_roi_w = tk.IntVar(value=0)
        self.entry_roi_w = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_w); self.entry_roi_w.pack(side=tk.LEFT)
        ttk.Label(roi_fields, text="H").pack(side=tk.LEFT); self.var_roi_h = tk.IntVar(value=0)
        self.entry_roi_h = ttk.Entry(roi_fields, width=5, textvariable=self.var_roi_h); self.entry_roi_h.pack(side=tk.LEFT)
        btn_row = ttk.Frame(roi_group); btn_row.pack(fill=tk.X, pady=2)
        self.btn_roi_update = ttk.Button(btn_row, text="Update ROI", command=self.update_roi_from_fields); self.btn_roi_update.pack(side=tk.LEFT, padx=2, pady=1)
        self.btn_roi_clear = ttk.Button(btn_row, text="Clear ROI", command=self.clear_roi); self.btn_roi_clear.pack(side=tk.LEFT, padx=2, pady=1)
        apply_row = ttk.Frame(cfg); apply_row.pack(fill=tk.X, pady=3)
        ttk.Button(apply_row, text="Apply", command=self.apply_current_settings).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(apply_row, text="Apply All", command=self.apply_current_settings_all).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        # Export
        export_frame = ttk.LabelFrame(dashboard, text="Export")
        export_frame.pack(fill=tk.X, pady=4)
        self.btn_export = ttk.Button(export_frame, text="Export Current Frame to JPG", command=self.export_frame)
        self.btn_export.pack(fill=tk.X, pady=2)
        self.btn_export_csv = ttk.Button(export_frame, text="Export Current File to CSV", command=self.export_video_csv)
        self.btn_export_csv.pack(fill=tk.X, pady=2)
        self.btn_export_csv_all = ttk.Button(export_frame, text="Export All Files to CSV", command=self.export_video_csv_all)
        self.btn_export_csv_all.pack(fill=tk.X, pady=2)
        # Image/canvas
        self.canvas = tk.Canvas(image_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        slider_bar = ttk.Frame(self)
        slider_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.slider = ttk.Scale(slider_bar, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X, padx=6, pady=4)
        # Status bar (kept at very bottom)
        self.status = ttk.Label(self, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        footer = ttk.Frame(dashboard)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(8,0))
        ttk.Label(footer, text="Developed from FLIR SDK by H. Nguyen").pack(anchor=tk.W)
        ttk.Label(footer, text="Version 0.0.1").pack(anchor=tk.W)
        ttk.Button(footer, text="Check for Updates", command=self.on_check_updates).pack(anchor=tk.W, fill=tk.X, pady=(2,0))
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
        messagebox.showinfo("Updates", "Version 0.0.1\nNo update source configured.")
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
        name = os.path.basename(self.seq_path) if self.seq_path else "(none)"
        total = len(self.batch_paths)
        idx_str = f" ({self.batch_index+1}/{total})" if total else ""
        if hasattr(self, "lbl_file"):
            self.lbl_file.configure(text=f"File: {name}{idx_str}")
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
            self.lbl_meta_emiss.configure(text=f"Metadata emissivity: {text}")
    def _update_range_button_labels(self):
        if not hasattr(self, "btn_set_start") or not hasattr(self, "btn_set_end"):
            return
        frame_num = self.current_idx + 1 if self.im is not None else 0
        if frame_num > 0:
            self.btn_set_start.configure(text=f"Set Start from {frame_num}")
            self.btn_set_end.configure(text=f"Set End from {frame_num}")
        else:
            self.btn_set_start.configure(text="Set Start from Current Frame")
            self.btn_set_end.configure(text="Set End from Current Frame")
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
        self.status.configure(text=f"{self._base_status} | Settings applied to current file")
    def apply_current_settings_all(self):
        settings = self._current_settings_snapshot()
        targets = self.batch_paths or ([self.seq_path] if self.seq_path else [])
        for path in targets:
            self.file_settings[path] = dict(settings)
        self.apply_configuration(update_status=False)
        self.status.configure(text=f"{self._base_status} | Settings applied to all files")
    def on_open(self, path_override: Optional[str] = None):
        if path_override:
            self._load_seq(path_override, reset_settings=False)
            return
        paths = filedialog.askopenfilenames(title="Select SEQ file(s)", filetypes=[("SEQ Files","*.seq"), ("All Files","*.*")])
        seq_paths = list(paths) if paths else []
        if not seq_paths:
            folder = filedialog.askdirectory(title="Select folder containing SEQ files")
            if folder:
                seq_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".seq")]
        if not seq_paths:
            return
        seq_paths = sorted(seq_paths)
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
            self.status.configure(text=f"Opened: {os.path.basename(path)} | {self.width}x{self.height} | {self.num_frames} frames")
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
        if self.im is None: return
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.after(1, self._play_loop)
    def on_stop(self):
        if self.im is None: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = 0
        self._reset_tracking()
        self._render_current()
    def on_prev(self):
        if self.im is None: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = max(0, self.current_idx - 1)
        self._render_current()
    def on_next(self):
        if self.im is None: return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = min(self.num_frames - 1, self.current_idx + 1)
        self._render_current()
    def on_prev_file(self):
        if not self.batch_paths:
            return
        self.batch_index = (self.batch_index - 1) % len(self.batch_paths)
        self._load_seq(self.batch_paths[self.batch_index], reset_settings=False)
    def on_next_file(self):
        if not self.batch_paths:
            return
        self.batch_index = (self.batch_index + 1) % len(self.batch_paths)
        self._load_seq(self.batch_paths[self.batch_index], reset_settings=False)
    def on_slider(self, val):
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
        if self.im is None: return
        xi, yi = self._canvas_to_image(e.x, e.y)
        self.mouse_down = (xi, yi)
        self.roi_rect = (xi, yi, 1, 1)
        self.update_roi_fields_from_rect()
        self._render_current()
    def on_mouse_drag(self, e):
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
        self.mouse_down = None
        self.update_roi_fields_from_rect()
    def on_mouse_move(self, e):
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
        det_offset = (0, 0)
        display = frame
        if roi is not None:
            rx, ry, rw, rh = roi
            display = frame[ry:ry+rh, rx:rx+rw]
            det_offset = (rx, ry)
        binary_mask = (display > temp_thresh).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        detections = []
        for lid in range(1, num_labels):
            area = stats[lid, cv2.CC_STAT_AREA]
            if MIN_OBJECT_AREA_PIXELS <= area <= MAX_OBJECT_AREA_PIXELS:
                x = stats[lid, cv2.CC_STAT_LEFT] + det_offset[0]
                y = stats[lid, cv2.CC_STAT_TOP] + det_offset[1]
                w = stats[lid, cv2.CC_STAT_WIDTH]
                h = stats[lid, cv2.CC_STAT_HEIGHT]
                centroid = (centroids[lid][0] + det_offset[0], centroids[lid][1] + det_offset[1])
                sub = display[stats[lid, cv2.CC_STAT_TOP]:stats[lid, cv2.CC_STAT_TOP]+h,
                              stats[lid, cv2.CC_STAT_LEFT]:stats[lid, cv2.CC_STAT_LEFT]+w]
                mask = (labels[stats[lid, cv2.CC_STAT_TOP]:stats[lid, cv2.CC_STAT_TOP]+h,
                               stats[lid, cv2.CC_STAT_LEFT]:stats[lid, cv2.CC_STAT_LEFT]+w] == lid)
                temps = sub[mask]
                if temps.size == 0:
                    continue
                max_temp = float(np.max(temps))
                min_temp = float(np.min(temps))
                avg_temp = float(np.mean(temps))
                median_temp = float(np.median(temps))
                detections.append({
                    'centroid': centroid,
                    'bbox': (x, y, w, h),
                    'max_temp': max_temp,
                    'min_temp': min_temp,
                    'avg_temp': avg_temp,
                    'median_temp': median_temp,
                    'area': int(area)
                })
        return detections
    def _assign_tracks(self, detections: List[dict], tracked_objects: OrderedDict, next_id: int, frame_idx: int):
        new_tracked = OrderedDict()
        for det in detections:
            cx, cy = det['centroid']
            best_id = None
            best_dist = None
            for tid, obj in tracked_objects.items():
                dist = np.hypot(cx - obj['centroid'][0], cy - obj['centroid'][1])
                if dist <= CENTROID_TRACKING_MAX_DIST and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = next_id
                next_id += 1
            det['track_id'] = best_id
            new_tracked[best_id] = {'centroid': det['centroid'], 'last_seen': frame_idx}
        for tid, obj in tracked_objects.items():
            if tid not in new_tracked and frame_idx - obj.get('last_seen', frame_idx) <= TRACK_MEMORY_FRAMES:
                new_tracked[tid] = obj
        return new_tracked, detections, next_id
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
    def _play_loop(self):
        if not self.playing or self.im is None: return
        self.current_idx += 1
        if self.current_idx >= self.num_frames:
            self.current_idx = self.num_frames - 1
            self.playing = False
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
            self._base_status = f"Frame {self.current_idx+1}/{self.num_frames} | {roistr} | Thresh: {temp_thresh:.1f}{self.unit_label}"
            self.status.configure(text=self._base_status)
            self._update_range_button_labels()
        except Exception:
            traceback.print_exc()
    def export_frame(self):
        try:
            if not self.seq_path:
                messagebox.showerror("Export", "Open a SEQ file first.")
                return
            self._ensure_emissivity()
            self.im.get_frame(self.current_idx)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            roi = self.roi_rect
            temp_thresh = float(self.var_thresh.get())
            detections = self._get_tracked_detections(self.current_idx, frame, roi, temp_thresh, update_tracker=False)
            vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
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
            messagebox.showerror("Export", "Open a SEQ file first.")
            return
        self._export_csv_for_paths([self.seq_path])
    def export_video_csv_all(self):
        paths = self.batch_paths if self.batch_paths else ([self.seq_path] if self.seq_path else [])
        if not paths:
            messagebox.showerror("Export", "Open a SEQ file first.")
            return
        self._export_csv_for_paths(paths)
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
                    self.status.configure(text=f"Exporting CSV [{p_idx+1}/{len(paths)}]: {os.path.basename(seq_path)} frame {idx+1}/{num_frames} (range {start_use}-{end_f})")
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
                self.status.configure(text=f"CSV saved: {out_path}")
            self.width = original_width
            self.height = original_height
            messagebox.showinfo("Export", "CSV export complete.")
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Export", f"Export failed: {ex}")
def main():
    app = SKDDashboard()
    app.mainloop()
if __name__ == "__main__":
    main()
