"""
Classic hotspot detector for FLIR radiometric SEQ files.

Features
- Reads radiometric `.seq` via FLIR Science File SDK (`fnv`)
- Detects hotspots > 300°C using thresholding and size filtering
- Optional ROI, morphological cleanup, and simple motion gating
- Realtime visualization
- CSV logging of per-frame detections (bbox, centroid, max temp)
- AI hook: optional classifier function to accept/reject detections

Requirements
- FLIR Science File SDK Python bindings installed: imports `fnv`, `fnv.file`, `fnv.reduce`
- OpenCV: `pip install opencv-python`
- numpy, matplotlib

Usage
- Run directly and select a `.seq` file from dialog, or set `INPUT_PATH` below
"""

import os
import sys
import csv
import traceback
from typing import Callable, List, Optional, Tuple

import numpy as np

# Optional plotting
try:
    from matplotlib import pyplot as plt
    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Install with: pip install opencv-python")
    sys.exit(1)

# Try to import FLIR SDK for .seq
_fnv_available = True
try:
    import fnv
    import fnv.file
    import fnv.reduce
except Exception:
    _fnv_available = False

try:
    from tkinter import filedialog
    import tkinter
    _tk_available = True
except Exception:
    _tk_available = False


# ---------------- Configuration ----------------
# If empty, a file dialog will open when running the script.
INPUT_PATH = ""

# Output options
WRITE_CSV = True
CSV_PATH = "hotspot_detections.csv"
WRITE_VIDEO = False
VIDEO_PATH = "hotspot_visualization.mp4"  # not used when WRITE_VIDEO is False
VIDEO_FPS = 30

# Visualization
SHOW_PLOT = False and _matplotlib_available
SHOW_OPENCV = True  # Realtime window via OpenCV
KEYBOARD_CONTROL = True  # Space=pause/resume, q/ESC=quit
SHARP_UPSCALE_FACTOR = 2  # 1 = no upscale; uses nearest-neighbor for crisp view
AOI_DRAW_ENABLED = True  # Allow drawing an AOI (Area Of Interest) with the mouse

# Detection parameters
TEMP_UNIT = "C"  # 'C' or 'K' (for .seq only); image sequences are treated as counts and thresholding is not temperature-calibrated
TEMP_THRESHOLD_C = 300.0
MIN_AREA_PX = 3
MAX_AREA_PX = 1500

# ROI (applied before detection)
USE_ROI = False
ROI_X = 0
ROI_Y = 0
ROI_WIDTH = 640
ROI_HEIGHT = 480

# Morphological cleanup (disabled by default to avoid smoothing)
MORPH_OPEN_KSIZE = 0   # 0 to disable
MORPH_CLOSE_KSIZE = 0  # 0 to disable

# Simple motion gating (optional): ignore detections that do not change compared to previous frame
USE_MOTION_GATE = False
MOTION_DIFF_THRESHOLD = 2.0  # degrees C; only for calibrated data

# AI Hook: set to a callable that takes (patch: np.ndarray[H,W], metadata: dict) -> bool
AI_CLASSIFIER: Optional[Callable[[np.ndarray, dict], bool]] = None
AI_PATCH_SIZE = 32  # pixels, square crop centered on detection centroid for inference/logging

# ------------------------------------------------


def _try_add_fnv_dll_dirs() -> None:
    """Attempt to add fnv/_lib and typical SDK bin directories to DLL search path on Windows."""
    try:
        if os.name != 'nt':
            return
        dirs_to_add: List[str] = []
        # 1) Package-local _lib next to fnv
        try:
            import fnv as _fnv_mod
            pkg_dir = os.path.dirname(_fnv_mod.__file__)
            dll_dir = os.path.join(pkg_dir, '_lib')
            if os.path.isdir(dll_dir):
                dirs_to_add.append(dll_dir)
        except Exception:
            pass
        # 2) Common install locations (if user installed SDK separately)
        candidates = [
            r"C:\\Program Files\\FLIR Systems\\sdks\\file\\bin\\Release",
            r"C:\\Program Files\\FLIR Systems\\sdks\\file\\bin",
            r"C:\\Program Files (x86)\\FLIR Systems\\sdks\\file\\bin\\Release",
        ]
        for c in candidates:
            if os.path.isdir(c):
                dirs_to_add.append(c)
        # Add to search path
        for d in dirs_to_add:
            try:
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(d)
            except Exception:
                pass
            if d not in os.environ.get('PATH', ''):
                os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass


def open_file_dialog() -> Optional[str]:
    if not _tk_available:
        return None
    root = tkinter.Tk(); root.withdraw(); root.call('wm', 'attributes', '.', '-topmost', True)
    path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select FLIR SEQ file",
        filetypes=(("SEQ Files", "*.seq"),("All Files", "*.*")))
    root.destroy()
    return path if path else None


def is_seq_file(path: str) -> bool:
    return path.lower().endswith(".seq")


def list_image_sequence(path_or_dir: str) -> List[str]:
    # Image sequence input not supported in this configuration.
    return []


def ensure_roi_within(width: int, height: int) -> Tuple[int, int, int, int, bool]:
    if not USE_ROI:
        return 0, 0, width, height, False
    x = max(0, ROI_X); y = max(0, ROI_Y)
    w = min(ROI_WIDTH, width - x)
    h = min(ROI_HEIGHT, height - y)
    ok = (w > 0 and h > 0)
    return x, y, w, h, ok


def apply_morphology(mask: np.ndarray) -> np.ndarray:
    out = mask
    if MORPH_OPEN_KSIZE and MORPH_OPEN_KSIZE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if MORPH_CLOSE_KSIZE and MORPH_CLOSE_KSIZE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def detect_hotspots_from_temperature(temp_image_c: np.ndarray, frame_index: int, frame_id: str,
    roi_offset: Tuple[int, int]) -> List[dict]:
    # Threshold
    binary = (temp_image_c > TEMP_THRESHOLD_C).astype(np.uint8) * 255
    binary = apply_morphology(binary)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    detections: List[dict] = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue
        x = stats[lid, cv2.CC_STAT_LEFT] + roi_offset[0]
        y = stats[lid, cv2.CC_STAT_TOP] + roi_offset[1]
        w = stats[lid, cv2.CC_STAT_WIDTH]
        h = stats[lid, cv2.CC_STAT_HEIGHT]
        cx = centroids[lid][0] + roi_offset[0]
        cy = centroids[lid][1] + roi_offset[1]

        # Max temperature within component (compute on ROI slice then offset)
        mask_roi = (labels == lid)
        # Extract the ROI portion that aligns to mask
        # For efficiency compute max via masking applied to temp_image subset
        # Build ROI absolute bounds
        x0 = int(cx - roi_offset[0])  # not exact bbox; use mask directly
        # Safer: slice using stats ROI
        rx = stats[lid, cv2.CC_STAT_LEFT]
        ry = stats[lid, cv2.CC_STAT_TOP]
        rw = stats[lid, cv2.CC_STAT_WIDTH]
        rh = stats[lid, cv2.CC_STAT_HEIGHT]
        temp_roi = temp_image_c[ry:ry+rh, rx:rx+rw]
        mask_roi_crop = mask_roi[ry:ry+rh, rx:rx+rw]
        if temp_roi.size == 0:
            continue
        max_temp = float(np.max(temp_roi[mask_roi_crop])) if np.any(mask_roi_crop) else float(np.max(temp_roi))

        detections.append({
            "frame_index": frame_index,
            "frame_id": frame_id,
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "max_temp_c": max_temp,
        })
    return detections


def crop_patch(image: np.ndarray, center_xy: Tuple[float, float], size: int) -> np.ndarray:
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    half = size // 2
    h, w = image.shape[:2]
    x0 = max(0, cx - half)
    x1 = min(w, cx + half)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half)
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] == 0 or patch.shape[1] == 0:
        return np.zeros((size, size), dtype=image.dtype)
    # Pad if needed
    if patch.shape[0] != size or patch.shape[1] != size:
        patch = cv2.copyMakeBorder(patch, 0, max(0, size - patch.shape[0]), 0, max(0, size - patch.shape[1]), cv2.BORDER_CONSTANT, value=0)
        patch = patch[:size, :size]
    return patch


def visualize_frame(temp_image_c: np.ndarray, detections: List[dict], roi_rect: Optional[Tuple[int,int,int,int]],
    title: str, video_writer) -> int:
    # Normalize to 8-bit for display
    vis = cv2.normalize(temp_image_c, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    # ROI
    if roi_rect is not None:
        x, y, w, h = roi_rect
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # Detections
    for d in detections:
        x, y, w, h = d["bbox"]
        cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.circle(vis, (cx, cy), 3, (255, 255, 255), -1)
        cv2.putText(vis, f"{d['max_temp_c']:.1f}C", (x, max(0, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    if SHOW_PLOT:
        plt.clf()
        # Use inferno for thermal-like look; keep sharp with nearest interpolation
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), interpolation='nearest')
        plt.title(title)
        plt.axis('off')
        plt.pause(0.01)

    if video_writer is not None:
        # Ensure size matches writer
        video_writer.write(vis)

    key = -1
    if SHOW_OPENCV:
        try:
            if SHARP_UPSCALE_FACTOR and SHARP_UPSCALE_FACTOR > 1:
                vis_show = cv2.resize(vis, (vis.shape[1]*SHARP_UPSCALE_FACTOR, vis.shape[0]*SHARP_UPSCALE_FACTOR), interpolation=cv2.INTER_NEAREST)
            else:
                vis_show = vis
            cv2.imshow("Hotspot Detection", vis_show)
            key = cv2.waitKey(1) & 0xFF
        except Exception:
            key = -1
    return key


def write_csv_header(csv_path: str) -> None:
    with open(csv_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "frame_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "centroid_x", "centroid_y", "max_temp_c"]) 


def append_csv_rows(csv_path: str, detections: List[dict]) -> None:
    if not detections:
        return
    with open(csv_path, mode='a', newline='') as f:
        w = csv.writer(f)
        for d in detections:
            x, y, w_box, h_box = d["bbox"]
            cx, cy = d["centroid"]
            w.writerow([d["frame_index"], d["frame_id"], x, y, w_box, h_box, f"{cx:.2f}", f"{cy:.2f}", f"{d['max_temp_c']:.2f}"])


def process_seq_file(seq_path: str) -> None:
    if not _fnv_available:
        print("ERROR: FLIR SDK Python module 'fnv' not available; cannot read .seq.")
        return
    # Normalize path and validate
    seq_path = os.path.normpath(seq_path)
    exists = os.path.exists(seq_path)
    if not exists:
        print(f"ERROR: File not found: {seq_path}")
        return
    print(f"Opening SEQ: {seq_path}")
    im = None
    try:
        # Ensure DLL search path is set (Windows)
        _try_add_fnv_dll_dirs()
        im = fnv.file.ImagerFile(seq_path)
        if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
            im.unit = fnv.Unit.TEMPERATURE_FACTORY
            im.temp_type = fnv.TempType.CELSIUS if TEMP_UNIT.upper() == "C" else fnv.TempType.KELVIN
        else:
            print("Warning: File lacks temperature calibration; using counts.")
            im.unit = fnv.Unit.COUNTS

        num_frames = im.num_frames
        height = im.height
        width = im.width
        print(f"Frames: {num_frames}, Size: {width}x{height}")

        rx, ry, rw, rh, roi_ok = ensure_roi_within(width, height)
        roi_rect = (rx, ry, rw, rh) if USE_ROI and roi_ok else None
        use_roi = roi_rect is not None

        writer = None
        if WRITE_VIDEO:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))

        if WRITE_CSV:
            write_csv_header(CSV_PATH)

        # Ensure OpenCV window exists before loop
        if SHOW_OPENCV:
            try:
                cv2.namedWindow("Hotspot Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Hotspot Detection", max(320, width // 2), max(240, height // 2))
            except Exception:
                pass

        # AOI interactive drawing state (mapped to source resolution; not the upscaled view)
        aoi_state = {"rect": roi_rect, "drawing": False, "start": None}
        upscale = SHARP_UPSCALE_FACTOR if (SHOW_OPENCV and SHARP_UPSCALE_FACTOR and SHARP_UPSCALE_FACTOR > 1) else 1
        if SHOW_OPENCV and AOI_DRAW_ENABLED:
            def _mouse_cb(event, x, y, flags, param):
                try:
                    # Map view coords back to source image coords
                    xi = int(x / upscale)
                    yi = int(y / upscale)
                    xi = max(0, min(width-1, xi))
                    yi = max(0, min(height-1, yi))
                    if event == cv2.EVENT_LBUTTONDOWN:
                        aoi_state["drawing"] = True
                        aoi_state["start"] = (xi, yi)
                        aoi_state["rect"] = (xi, yi, 1, 1)
                    elif event == cv2.EVENT_MOUSEMOVE and aoi_state["drawing"]:
                        x0, y0 = aoi_state["start"]
                        x1, y1 = xi, yi
                        rx = min(x0, x1); ry = min(y0, y1)
                        rw = max(1, abs(x1 - x0)); rh = max(1, abs(y1 - y0))
                        aoi_state["rect"] = (rx, ry, rw, rh)
                    elif event == cv2.EVENT_LBUTTONUP and aoi_state["drawing"]:
                        aoi_state["drawing"] = False
                        x0, y0 = aoi_state["start"]
                        x1, y1 = xi, yi
                        rx = min(x0, x1); ry = min(y0, y1)
                        rw = max(1, abs(x1 - x0)); rh = max(1, abs(y1 - y0))
                        aoi_state["rect"] = (rx, ry, rw, rh)
                except Exception:
                    pass
            try:
                cv2.setMouseCallback("Hotspot Detection", _mouse_cb)
                print("AOI: Drag with left mouse to set area. Press 'c' to clear.")
            except Exception:
                pass

        prev_temp = None
        # Timeline/controls state
        paused = False
        i = 0

        # Trackbar setup for timeline (seek)
        seek_to = {"val": None}  # mutable holder for callback
        if SHOW_OPENCV:
            def _on_seek(val):
                try:
                    seek_to["val"] = int(val)
                except Exception:
                    seek_to["val"] = None
            try:
                cv2.createTrackbar("Frame", "Hotspot Detection", 0, max(0, num_frames-1), _on_seek)
            except Exception:
                pass

        while i < num_frames:
            # Apply seek from trackbar
            if seek_to["val"] is not None:
                i = max(0, min(num_frames-1, seek_to["val"]))
                seek_to["val"] = None
                paused = True

            im.get_frame(i)
            temp = np.array(im.final, copy=False).reshape((height, width))
            # Convert to Celsius if needed
            if im.unit == fnv.Unit.TEMPERATURE_FACTORY and im.temp_type == fnv.TempType.KELVIN:
                temp_c = temp - 273.15
            elif im.unit == fnv.Unit.TEMPERATURE_FACTORY and im.temp_type == fnv.TempType.CELSIUS:
                temp_c = temp
            else:
                # Counts: not calibrated. For consistency, treat as Celsius-like values (not recommended)
                temp_c = temp

            # Determine active AOI/ROI dynamically (AOI overrides static ROI if present)
            active_rect = aoi_state["rect"] if (aoi_state.get("rect") is not None) else roi_rect
            if active_rect is not None:
                rx, ry, rw, rh = active_rect
                temp_roi = temp_c[ry:ry+rh, rx:rx+rw]
                detections = detect_hotspots_from_temperature(temp_roi, i, f"frame_{i}", (rx, ry))
            else:
                detections = detect_hotspots_from_temperature(temp_c, i, f"frame_{i}", (0, 0))

            # Motion gate
            if USE_MOTION_GATE and prev_temp is not None:
                diff = np.abs(temp_c - prev_temp)
                gated = []
                for d in detections:
                    x, y, w_box, h_box = d["bbox"]
                    patch = diff[max(0,y):y+h_box, max(0,x):x+w_box]
                    if patch.size == 0 or np.mean(patch) < MOTION_DIFF_THRESHOLD:
                        continue
                    gated.append(d)
                detections = gated
            prev_temp = temp_c

            # AI hook
            if AI_CLASSIFIER is not None:
                filtered = []
                for d in detections:
                    patch = crop_patch(temp_c, d["centroid"], AI_PATCH_SIZE)
                    meta = {"frame_index": d["frame_index"], "max_temp_c": d["max_temp_c"], "bbox": d["bbox"]}
                    try:
                        if AI_CLASSIFIER(patch, meta):
                            filtered.append(d)
                    except Exception:
                        # Fail open to classic behavior
                        filtered.append(d)
                detections = filtered

            if WRITE_CSV:
                append_csv_rows(CSV_PATH, detections)

            title = f"Hotspot Detection - Frame {i+1}/{num_frames}"
            # Draw current AOI (if any) instead of static ROI
            draw_rect = aoi_state["rect"] if (aoi_state.get("rect") is not None) else roi_rect
            key = visualize_frame(temp_c, detections, draw_rect, title, writer)

            # Keep trackbar position synced
            if SHOW_OPENCV:
                try:
                    cv2.setTrackbarPos("Frame", "Hotspot Detection", int(i))
                except Exception:
                    pass

            if KEYBOARD_CONTROL and SHOW_OPENCV:
                if key in (27, ord('q')):
                    break
                if key == 32:  # space toggles pause
                    paused = not paused
                # Arrow keys (left/right) or 'b'/'f' for stepping
                if key in (81, ord('b')):  # left arrow or 'b' -> back one frame
                    i = max(0, i-1)
                    paused = True
                    continue
                if key in (83, ord('f')):  # right arrow or 'f' -> forward one frame
                    i = min(num_frames-1, i+1)
                    paused = True
                    continue
                # Clear AOI
                if key in (ord('c'),):
                    aoi_state["rect"] = None
                    paused = True
                    continue

                # Pause loop: show current frame and poll keys/trackbar
                while paused:
                    key2 = visualize_frame(temp_c, detections, roi_rect, title + " (Paused)", writer if WRITE_VIDEO else None)
                    if SHOW_OPENCV:
                        try:
                            cv2.setTrackbarPos("Frame", "Hotspot Detection", int(i))
                        except Exception:
                            pass
                    if key2 in (27, ord('q')):
                        paused = False
                        i = num_frames  # force exit outer loop
                        break
                    if key2 == 32:
                        paused = False
                        break
                    if key2 in (81, ord('b')):
                        i = max(0, i-1)
                        break
                    if key2 in (83, ord('f')):
                        i = min(num_frames-1, i+1)
                        break
                    if key2 in (ord('c'),):
                        aoi_state["rect"] = None
                        break
                    # Apply pending seek while paused
                    if seek_to["val"] is not None:
                        i = max(0, min(num_frames-1, seek_to["val"]))
                        seek_to["val"] = None
                        break
            # Auto-advance if not paused or stepped
            if not paused:
                i += 1
        if writer is not None:
            writer.release()
        if SHOW_OPENCV:
            # Keep the last frame open until user closes
            try:
                print("Press 'q' or ESC to close the window...")
                while True:
                    key = cv2.waitKey(50) & 0xFF
                    if key in (27, ord('q')):
                        break
                cv2.destroyAllWindows()
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
        print("Done.")
    except OSError as ose:
        print("\nOSError while opening SEQ with FLIR SDK.")
        print(f"Details: {ose}")
        print("Hints: \n- Ensure FLIR Science File SDK is installed and matches Python bitness (x64).\n- Ensure required DLLs are discoverable. This script attempts to add 'fnv/_lib' and SDK bin folders to PATH.\n- Verify the file is not locked and is a valid radiometric SEQ.")
        traceback.print_exc()
    except Exception:
        traceback.print_exc()
    finally:
        im = None


def process_image_sequence(path_or_dir: str) -> None:
    files = list_image_sequence(path_or_dir)
    if not files:
        print("No images found for sequence.")
        return
    # Read first to get size
    first = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        print("Failed to read first image.")
        return
    height, width = first.shape[:2]
    print(f"Image sequence: {len(files)} frames, Size: {width}x{height}")

    rx, ry, rw, rh, roi_ok = ensure_roi_within(width, height)
    roi_rect = (rx, ry, rw, rh) if USE_ROI and roi_ok else None

    writer = None
    if WRITE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(VIDEO_PATH, fourcc, VIDEO_FPS, (width, height))

    if WRITE_CSV:
        write_csv_header(CSV_PATH)

    for i, fpath in enumerate(files):
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # If 3-channel, convert to grayscale; these are counts, not calibrated temps
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # Treat counts as pseudo-temp; threshold still applied as numeric cutoff
        temp_like = gray.astype(np.float32)

        if roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            sub = temp_like[ry:ry+rh, rx:rx+rw]
            detections = detect_hotspots_from_temperature(sub, i, os.path.basename(fpath), (rx, ry))
        else:
            detections = detect_hotspots_from_temperature(temp_like, i, os.path.basename(fpath), (0, 0))

        if AI_CLASSIFIER is not None:
            filtered = []
            for d in detections:
                patch = crop_patch(temp_like, d["centroid"], AI_PATCH_SIZE)
                meta = {"frame_index": d["frame_index"], "bbox": d["bbox"], "source": fpath}
                try:
                    if AI_CLASSIFIER(patch, meta):
                        filtered.append(d)
                except Exception:
                    filtered.append(d)
            detections = filtered

        if WRITE_CSV:
            append_csv_rows(CSV_PATH, detections)

        title = f"Hotspot Detection - Frame {i+1}/{len(files)}"
        visualize_frame(temp_like, detections, roi_rect, title, writer)

    if writer is not None:
        writer.release()
    print("Done.")


def main():
    path = INPUT_PATH.strip()
    if not path:
        sel = open_file_dialog()
        if not sel:
            print("No input selected.")
            return
        path = sel

    if is_seq_file(path):
        process_seq_file(path)
    else:
        print("ERROR: Only .seq files are supported. Please select a SEQ file.")


if __name__ == "__main__":
    if SHOW_PLOT and _matplotlib_available:
        plt.figure("Hotspot Detection")
    main()
    if SHOW_PLOT and _matplotlib_available:
        try:
            plt.show()
        except Exception:
            pass


