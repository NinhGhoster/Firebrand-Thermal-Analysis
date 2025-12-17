"""
GUI viewer similar in spirit to FLIR 'fnvfileexample':
- Open SEQ file, play/pause, frame slider
- AOI rectangle drawing
- Hotspot detection overlay (>300°C), threshold control

Builds on hotspot_detector.py functions for detection
"""

import os
import sys
import traceback
from typing import Optional, Tuple, List

import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception as e:
    print("ERROR: tkinter is required for the GUI.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import cv2
except Exception:
    print("ERROR: OpenCV is required. pip install opencv-python")
    sys.exit(1)

try:
    import fnv
    import fnv.file
except Exception:
    fnv = None

import hotspot_detector as hd


class HotspotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hotspot Detector - SEQ Viewer")
        self.geometry("1000x700")

        # State
        self.im = None  # fnv.file.ImagerFile
        self.seq_path: str = ""
        self.num_frames: int = 0
        self.width: int = 0
        self.height: int = 0
        self.current_idx: int = 0
        self.playing: bool = False
        self.aoi_rect: Optional[Tuple[int,int,int,int]] = None
        self.mouse_down: Optional[Tuple[int,int]] = None
        self.temp_threshold: float = 300.0

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        # Main layout: left dashboard, right image
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        dashboard = ttk.Frame(main_frame)
        dashboard.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # --- Dashboard controls panel ---
        self.btn_open = ttk.Button(dashboard, text="Open SEQ", command=self.on_open)
        self.btn_open.pack(fill=tk.X, pady=2)
        self.btn_play = ttk.Button(dashboard, text="Play", command=self.on_play_pause)
        self.btn_play.pack(fill=tk.X, pady=2)
        nav_frame = ttk.Frame(dashboard)
        nav_frame.pack(fill=tk.X)
        self.btn_prev = ttk.Button(nav_frame, text="Prev", command=self.on_prev, width=5)
        self.btn_prev.pack(side=tk.LEFT, padx=1, pady=2)
        self.btn_next = ttk.Button(nav_frame, text="Next", command=self.on_next, width=5)
        self.btn_next.pack(side=tk.LEFT, padx=1, pady=2)
        ttk.Label(dashboard, text=" ").pack()  # Spacer
        self.slider = ttk.Scale(dashboard, from_=0, to=0, orient=tk.HORIZONTAL, command=self.on_slider)
        self.slider.pack(fill=tk.X, pady=4)
        ttk.Label(dashboard, text=" ").pack()  # Spacer
        thresh_frame = ttk.Frame(dashboard)
        thresh_frame.pack(fill=tk.X)
        ttk.Label(thresh_frame, text="Threshold (°C):").pack(side=tk.LEFT)
        self.var_thresh = tk.DoubleVar(value=self.temp_threshold)
        self.entry_thresh = ttk.Entry(thresh_frame, width=8, textvariable=self.var_thresh)
        self.entry_thresh.pack(side=tk.RIGHT)
        ttk.Label(dashboard, text=" ").pack()  # Spacer
        dets_cb = ttk.Checkbutton(dashboard, text="Show detections", variable=self.var_show_dets)
        dets_cb.pack(anchor=tk.W)
        draw_cb = ttk.Checkbutton(dashboard, text="Draw AOI", variable=self.var_draw_aoi)
        draw_cb.pack(anchor=tk.W)
        ttk.Label(dashboard, text=" ").pack()  # Spacer
        # AOI/ROI
        roi_group = ttk.LabelFrame(dashboard, text="Adjust AOI")
        roi_group.pack(fill=tk.X, pady=4)
        aoi_fields = ttk.Frame(roi_group)
        aoi_fields.pack(fill=tk.X)
        ttk.Label(aoi_fields, text="X").pack(side=tk.LEFT)
        self.var_aoi_x = tk.IntVar(value=0)
        self.entry_aoi_x = ttk.Entry(aoi_fields, width=5, textvariable=self.var_aoi_x)
        self.entry_aoi_x.pack(side=tk.LEFT)
        ttk.Label(aoi_fields, text="Y").pack(side=tk.LEFT)
        self.var_aoi_y = tk.IntVar(value=0)
        self.entry_aoi_y = ttk.Entry(aoi_fields, width=5, textvariable=self.var_aoi_y)
        self.entry_aoi_y.pack(side=tk.LEFT)
        ttk.Label(aoi_fields, text="W").pack(side=tk.LEFT)
        self.var_aoi_w = tk.IntVar(value=0)
        self.entry_aoi_w = ttk.Entry(aoi_fields, width=5, textvariable=self.var_aoi_w)
        self.entry_aoi_w.pack(side=tk.LEFT)
        ttk.Label(aoi_fields, text="H").pack(side=tk.LEFT)
        self.var_aoi_h = tk.IntVar(value=0)
        self.entry_aoi_h = ttk.Entry(aoi_fields, width=5, textvariable=self.var_aoi_h)
        self.entry_aoi_h.pack(side=tk.LEFT)
        btn_row = ttk.Frame(roi_group)
        btn_row.pack(fill=tk.X)
        self.btn_aoi_update = ttk.Button(btn_row, text="Update AOI", command=self.update_aoi_from_fields)
        self.btn_aoi_update.pack(side=tk.LEFT, padx=(2,2), pady=1)
        self.btn_aoi_clear = ttk.Button(btn_row, text="Clear AOI", command=self.clear_aoi)
        self.btn_aoi_clear.pack(side=tk.LEFT, padx=(2,2), pady=1)
        ttk.Label(dashboard, text=" ").pack()  # Spacer
        self.btn_export = ttk.Button(dashboard, text="Export Current Frame", command=self.export_frame)
        self.btn_export.pack(fill=tk.X, pady=3)

        # --- Image Canvas ---
        self.canvas = tk.Canvas(image_frame, bg="#222222")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # --- Status bar ---
        self.status = ttk.Label(self, text="Ready", anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        self.bind("<Key-space>", lambda e: self.on_play_pause())
        self.bind("<Key-Right>", lambda e: self.on_next())
        self.bind("<Key-Left>", lambda e: self.on_prev())
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_open(self):
        if fnv is None:
            messagebox.showerror("Error", "FLIR SDK Python module 'fnv' is not available.")
            return
        path = filedialog.askopenfilename(title="Select SEQ", filetypes=(("SEQ Files","*.seq"),("All Files","*.*")))
        if not path:
            return
        self._open_seq(path)

    def _open_seq(self, path: str):
        try:
            hd._try_add_fnv_dll_dirs()
            im = fnv.file.ImagerFile(path)
            if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
                im.unit = fnv.Unit.TEMPERATURE_FACTORY
                im.temp_type = fnv.TempType.CELSIUS
            else:
                im.unit = fnv.Unit.COUNTS
            self.im = im
            self.seq_path = path
            self.num_frames = im.num_frames
            self.width = im.width
            self.height = im.height
            self.current_idx = 0
            self.slider.configure(from_=0, to=max(0, self.num_frames-1))
            self.playing = False
            self.btn_play.configure(text="Play")
            # AOI to full frame
            self.aoi_rect = None
            self.update_aoi_fields_from_rect()
            self.status.configure(text=f"Opened: {os.path.basename(path)}  |  {self.width}x{self.height}  |  {self.num_frames} frames")
            self.after(10, self._render_current)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Open error", f"Failed to open file.\n{e}")

    def on_play_pause(self):
        if self.im is None:
            return
        self.playing = not self.playing
        self.btn_play.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.after(1, self._play_loop)

    def on_prev(self):
        if self.im is None:
            return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = max(0, self.current_idx - 1)
        self._render_current()

    def on_next(self):
        if self.im is None:
            return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = min(self.num_frames - 1, self.current_idx + 1)
        self._render_current()

    def on_slider(self, val):
        if self.im is None:
            return
        try:
            idx = int(float(val))
        except Exception:
            return
        self.playing = False
        self.btn_play.configure(text="Play")
        self.current_idx = max(0, min(self.num_frames-1, idx))
        self._render_current()

    def update_aoi_from_fields(self):
        try:
            x = max(0, int(self.var_aoi_x.get()))
            y = max(0, int(self.var_aoi_y.get()))
            w = max(1, int(self.var_aoi_w.get()))
            h = max(1, int(self.var_aoi_h.get()))
            if self.width > 0 and self.height > 0:
                w = min(self.width-x, w)
                h = min(self.height-y, h)
            self.aoi_rect = (x, y, w, h)
            self._render_current()
        except Exception:
            pass

    def clear_aoi(self):
        self.aoi_rect = None
        self.var_aoi_x.set(0)
        self.var_aoi_y.set(0)
        self.var_aoi_w.set(self.width)
        self.var_aoi_h.set(self.height)
        self._render_current()

    def on_mouse_down(self, e):
        if not self.var_draw_aoi.get() or self.im is None:
            return
        xi, yi = self._canvas_to_image(e.x, e.y)
        self.mouse_down = (xi, yi)
        self.aoi_rect = (xi, yi, 1, 1)
        self.update_aoi_fields_from_rect()
        self._render_current()

    def on_mouse_drag(self, e):
        if not self.var_draw_aoi.get() or self.im is None or self.mouse_down is None:
            return
        x0, y0 = self.mouse_down
        x1, y1 = self._canvas_to_image(e.x, e.y)
        rx = max(0, min(x0, x1)); ry = max(0, min(y0, y1))
        rw = max(1, min(self.width - rx, abs(x1 - x0)))
        rh = max(1, min(self.height - ry, abs(y1 - y0)))
        self.aoi_rect = (rx, ry, rw, rh)
        self.update_aoi_fields_from_rect()
        self._render_current()

    def on_mouse_up(self, e):
        if not self.var_draw_aoi.get():
            return
        self.mouse_down = None
        self.update_aoi_fields_from_rect()

    def update_aoi_fields_from_rect(self):
        if self.aoi_rect:
            x, y, w, h = self.aoi_rect
            self.var_aoi_x.set(x)
            self.var_aoi_y.set(y)
            self.var_aoi_w.set(w)
            self.var_aoi_h.set(h)
        else:
            self.var_aoi_x.set(0)
            self.var_aoi_y.set(0)
            self.var_aoi_w.set(self.width)
            self.var_aoi_h.set(self.height)

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int,int]:
        # Map canvas coordinates to image coordinates based on current draw size
        bbox = self.canvas.bbox("img")
        if not bbox:
            return 0, 0
        x0, y0, x1, y1 = bbox
        draw_w = max(1, x1 - x0)
        draw_h = max(1, y1 - y0)
        if self.width == 0 or self.height == 0:
            return 0, 0
        xi = int((cx - x0) * self.width / draw_w)
        yi = int((cy - y0) * self.height / draw_h)
        xi = max(0, min(self.width-1, xi))
        yi = max(0, min(self.height-1, yi))
        return xi, yi

    def _play_loop(self):
        if not self.playing or self.im is None:
            return
        self.current_idx += 1
        if self.current_idx >= self.num_frames:
            self.current_idx = self.num_frames - 1
            self.playing = False
            self.btn_play.configure(text="Play")
            return
        self._render_current()
        self.after(1, self._play_loop)

    def _render_current(self):
        if self.im is None:
            return
        try:
            self.im.get_frame(self.current_idx)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            # Celsius
            if self.im.unit == fnv.Unit.TEMPERATURE_FACTORY and self.im.temp_type == fnv.TempType.KELVIN:
                frame_c = frame - 273.15
            else:
                frame_c = frame
            # Detection only in AOI if set
            detections: List[dict]
            draw_rect = self.aoi_rect
            if draw_rect is not None:
                rx, ry, rw, rh = draw_rect
                sub = frame_c[ry:ry+rh, rx:rx+rw]
                # Override threshold from UI
                self.temp_threshold = float(self.var_thresh.get())
                old_th = hd.TEMP_THRESHOLD_C
                hd.TEMP_THRESHOLD_C = self.temp_threshold
                detections = hd.detect_hotspots_from_temperature(sub, self.current_idx, f"frame_{self.current_idx}", (rx, ry))
                hd.TEMP_THRESHOLD_C = old_th
            else:
                self.temp_threshold = float(self.var_thresh.get())
                old_th = hd.TEMP_THRESHOLD_C
                hd.TEMP_THRESHOLD_C = self.temp_threshold
                detections = hd.detect_hotspots_from_temperature(frame_c, self.current_idx, f"frame_{self.current_idx}", (0, 0))
                hd.TEMP_THRESHOLD_C = old_th

            # Build visualization image
            vis = cv2.normalize(frame_c, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if self.var_show_dets.get():
                for d in detections:
                    x, y, w, h = d["bbox"]
                    cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.putText(vis, f"{d['max_temp_c']:.1f}C", (x, max(0, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            if draw_rect is not None:
                x, y, w, h = draw_rect
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)

            # Render to canvas with nearest-neighbor
            img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            if Image is not None:
                im_pil = Image.fromarray(img_rgb)
                # Fit to canvas size while preserving aspect, using NEAREST
                c_w = max(1, self.canvas.winfo_width())
                c_h = max(1, self.canvas.winfo_height())
                scale = min(c_w / self.width, c_h / self.height)
                new_w = max(1, int(self.width * scale))
                new_h = max(1, int(self.height * scale))
                im_pil = im_pil.resize((new_w, new_h), resample=Image.NEAREST)
                self._tk_img = ImageTk.PhotoImage(im_pil)
                self.canvas.delete("all")
                self.canvas.create_image((c_w - new_w)//2, (c_h - new_h)//2, image=self._tk_img, anchor=tk.NW, tags=("img",))
            else:
                # Fallback: no PIL, draw nothing
                pass

            # Update UI elements
            self.slider.set(self.current_idx)
            self.status.configure(text=f"Frame {self.current_idx+1}/{self.num_frames}  |  AOI: {self.aoi_rect if self.aoi_rect else 'Full'}  |  Thresh: {self.temp_threshold:.1f}C")
        except Exception:
            traceback.print_exc()

    def export_frame(self):
        # Save the currently displayed frame (with overlays)
        try:
            if not hasattr(self, '_tk_img') or self._tk_img is None:
                messagebox.showinfo("Export", "No image to export.")
                return
            # Ask for export path
            out_path = filedialog.asksaveasfilename(title="Export Frame", defaultextension=".png",
                                                    filetypes=[("PNG","*.png"),("JPEG","*.jpg")])
            if not out_path:
                return
            # Use current frame data, with overlays (as shown)
            frame = np.array(self.im.final, copy=False).reshape((self.height, self.width))
            if self.im.unit == fnv.Unit.TEMPERATURE_FACTORY and self.im.temp_type == fnv.TempType.KELVIN:
                frame_c = frame - 273.15
            else:
                frame_c = frame
            draw_rect = self.aoi_rect
            if draw_rect is not None:
                rx, ry, rw, rh = draw_rect
                sub = frame_c[ry:ry+rh, rx:rx+rw]
                self.temp_threshold = float(self.var_thresh.get())
                old_th = hd.TEMP_THRESHOLD_C
                hd.TEMP_THRESHOLD_C = self.temp_threshold
                detections = hd.detect_hotspots_from_temperature(sub, self.current_idx, f"frame_{self.current_idx}", (rx, ry))
                hd.TEMP_THRESHOLD_C = old_th
            else:
                self.temp_threshold = float(self.var_thresh.get())
                old_th = hd.TEMP_THRESHOLD_C
                hd.TEMP_THRESHOLD_C = self.temp_threshold
                detections = hd.detect_hotspots_from_temperature(frame_c, self.current_idx, f"frame_{self.current_idx}", (0, 0))
                hd.TEMP_THRESHOLD_C = old_th
            vis = cv2.normalize(frame_c, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            if self.var_show_dets.get():
                for d in detections:
                    x, y, w, h = d["bbox"]
                    cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
                    cv2.putText(vis, f"{d['max_temp_c']:.1f}C", (x, max(0, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            if draw_rect is not None:
                x, y, w, h = draw_rect
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Write file
            ext = os.path.splitext(out_path)[1].lower()
            if ext == ".jpg" or ext == ".jpeg":
                ok = cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY),95])
            else:
                ok = cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            messagebox.showinfo("Export", f"Exported frame to: {out_path}" if ok else "Export failed!")
        except Exception as ex:
            traceback.print_exc()
            messagebox.showerror("Export", f"Export failed: {ex}")


def main():
    app = HotspotGUI()
    app.mainloop()


if __name__ == "__main__":
    main()


