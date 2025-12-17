# -*- coding: utf-8 -*-
"""
Reads temperature data from FLIR radiometric files (SEQ, etc.)
using the 'fnv' module from the FLIR Science File SDK.

Detects, tracks, and reports max temperature of small, hot objects (e.g., embers)
within a defined Region of Interest (ROI).
Includes detailed debug prints for troubleshooting.
"""

import numpy as np
import os
import sys
import traceback
from collections import OrderedDict # To help manage tracked objects

# --- Import required libraries ---
try:
    import cv2 # OpenCV for image processing
    print("Successfully imported 'cv2' module.")
except ImportError:
    print("ERROR: OpenCV library not found.")
    print("Please install it using: pip install opencv-python")
    sys.exit(1)

try:
    import fnv
    import fnv.file
    import fnv.reduce
    print("Successfully imported 'fnv' module.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Could not import the 'fnv' module.")
    print("Ensure the FLIR Science File SDK's Python interface was installed correctly.")
    sys.exit(1)
# --- End Import ---

# --- Optional imports for plotting and file dialog ---
try:
    from matplotlib import pyplot as plt
    _matplotlib_available = True
except ImportError:
    _matplotlib_available = False
    print("Warning: matplotlib not found. Visualization disabled.")

try:
    from tkinter import filedialog
    import tkinter
    _tkinter_available = True
except ImportError:
    _tkinter_available = False
    print("Warning: tkinter not found. File dialog disabled.")
# --- End Optional Imports ---


# --- Configuration ---
# Option 1: Use a fixed file path (only if USE_FILE_DIALOG is False)
# SEQ_FILE_PATH = r"C:\Users\Dustwun\Downloads\Rec-0019.seq"

# Option 2: Use a file dialog to browse for the file
USE_FILE_DIALOG = True
if not _tkinter_available:
    print("Tkinter not available, disabling file dialog.")
    USE_FILE_DIALOG = False
    if 'SEQ_FILE_PATH' not in locals() and 'SEQ_FILE_PATH' not in globals():
         print("ERROR: USE_FILE_DIALOG is False, but SEQ_FILE_PATH is not defined.")
         sys.exit(1)

# --- Region of Interest (ROI) ---
# Define the area where embers are expected to fall and be tracked.
# Processing will only occur within this rectangle.
# Determine these coordinates from your video (e.g., using an image viewer).
USE_ROI = True # Set to False to process the full frame
ROI_X = 200      # Top-left corner X coordinate
ROI_Y = 300      # Top-left corner Y coordinate
ROI_WIDTH = 600  # Width of the ROI rectangle
ROI_HEIGHT = 400 # Height of the ROI rectangle
# --- End ROI ---

# Detection Parameters (TUNING REQUIRED!)
DESIRED_TEMP_UNIT = fnv.TempType.CELSIUS # Or KELVIN, FAHRENHEIT
TEMP_THRESHOLD_CELSIUS = 300.0 # <-- Increased threshold for hot embers
MIN_OBJECT_AREA_PIXELS = 3   # Minimum number of pixels (adjust for ember size)
MAX_OBJECT_AREA_PIXELS = 150 # Maximum number of pixels (adjust for ember size)

# Tracking Parameters (TUNING REQUIRED!)
CENTROID_TRACKING_MAX_DIST = 40 # Max pixel distance (might need increase for falling)
TRACK_MEMORY_FRAMES = 15 # How long to remember a track (increase slightly?)

# Reporting/Visualization
SHOW_VISUALIZATION = True # Draw boxes, IDs, temps on frames
if not _matplotlib_available:
    SHOW_VISUALIZATION = False
# --- End Configuration ---


# --- Helper Functions ---
def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x, y)."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_file_path_with_dialog():
    """Opens a Tkinter dialog to select a radiometric file."""
    if not _tkinter_available: return None
    root = tkinter.Tk(); root.withdraw(); root.call('wm', 'attributes', '.', '-topmost', True)
    path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select FLIR SEQ File", filetypes=(("SEQ Files", "*.seq"),("All Files", "*.*")))
    root.destroy()
    return path if path else None
# --- End Helper Functions ---


# --- Main Processing Function ---
def detect_track_hotspots(file_path):
    print(f"DEBUG: Entered detect_track_hotspots with path: {file_path}")
    im = None
    tracked_objects = OrderedDict()
    next_track_id = 0
    # These four are global/mutable so ROI can be changed live
    global ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT, USE_ROI

    # Define ROI bounds based on config
    roi_slice_y = slice(ROI_Y, ROI_Y + ROI_HEIGHT)
    roi_slice_x = slice(ROI_X, ROI_X + ROI_WIDTH)
    roi_offset = (ROI_X, ROI_Y)

    def _select_roi_on_frame(vis_img):
        print("ROI ADJUSTMENT: Drag a rectangle with the mouse, press ENTER or SPACE to accept, or C to reset to full frame.")
        roi = cv2.selectROI("Detection & Tracking", vis_img, fromCenter=False, showCrosshair=True)
        # selectROI returns (x, y, w, h)
        if roi is not None and len(roi) == 4 and roi[2] > 0 and roi[3] > 0:
            print(f"New ROI set: {roi}")
            return roi
        else:
            print("ROI selection cancelled/reset.")
            return None

    # Helper for processing one frame (extract tracking logic)
    def process_and_show_frame(im, i, prev_tracked_objects, prev_next_track_id):
        im.get_frame(i)
        full_height, full_width = im.height, im.width
        full_temp_data = np.array(im.final, copy=False).reshape((full_height, full_width))
        # Use global ROI params
        if USE_ROI:
            temp_data_roi = full_temp_data[ROI_Y:ROI_Y+ROI_HEIGHT, ROI_X:ROI_X+ROI_WIDTH]
            current_offset = (ROI_X, ROI_Y)
        else:
            temp_data_roi = full_temp_data
            current_offset = (0, 0)
        # Detection logic
        binary_mask = (temp_data_roi > TEMP_THRESHOLD_CELSIUS).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        current_frame_detections = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if MIN_OBJECT_AREA_PIXELS <= area <= MAX_OBJECT_AREA_PIXELS:
                x_roi = stats[label_id, cv2.CC_STAT_LEFT]
                y_roi = stats[label_id, cv2.CC_STAT_TOP]
                w = stats[label_id, cv2.CC_STAT_WIDTH]
                h = stats[label_id, cv2.CC_STAT_HEIGHT]
                centroid_roi = centroids[label_id]
                x_full = x_roi + current_offset[0]
                y_full = y_roi + current_offset[1]
                centroid_full = (centroid_roi[0] + current_offset[0], centroid_roi[1] + current_offset[1])
                bbox_full = (x_full, y_full, w, h)
                blob_mask_roi = (labels == label_id)
                max_temp_object = np.max(temp_data_roi[blob_mask_roi])
                current_frame_detections.append({
                    'centroid': centroid_full,
                    'bbox': bbox_full,
                    'max_temp': max_temp_object,
                    'matched_track_id': None
                })
        # Re-do tracking (match to previous)
        tracked_objects = OrderedDict()
        current_frame_tracked_output = {}
        unmatched_track_ids = list(prev_tracked_objects.keys())
        assigned = {}
        for det_idx, det in enumerate(current_frame_detections):
            best_match_id = -1
            min_dist = CENTROID_TRACKING_MAX_DIST
            for track_id in unmatched_track_ids:
                dist = calculate_distance(det['centroid'], prev_tracked_objects[track_id]['centroid'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = track_id
            if best_match_id != -1:
                tracked_objects[best_match_id] = dict(prev_tracked_objects[best_match_id])
                tracked_objects[best_match_id]['centroid'] = det['centroid']
                tracked_objects[best_match_id]['last_frame'] = i
                tracked_objects[best_match_id]['max_temp_history'] = tracked_objects[best_match_id].get('max_temp_history', []) + [det['max_temp']]
                current_frame_detections[det_idx]['matched_track_id'] = best_match_id
                current_frame_tracked_output[best_match_id] = det['max_temp']
                assigned[best_match_id] = True
        # Add new tracks
        next_track_id = prev_next_track_id
        for det_idx, det in enumerate(current_frame_detections):
            if det['matched_track_id'] is None:
                tracked_objects[next_track_id] = {'centroid': det['centroid'], 'last_frame': i,'max_temp_history':[det['max_temp']]}
                current_frame_tracked_output[next_track_id] = det['max_temp']
                next_track_id += 1
        # Remove old tracks
        stale_ids = [track_id for track_id, data in tracked_objects.items() if (i - data['last_frame']) > TRACK_MEMORY_FRAMES]
        for tid in stale_ids:
            del tracked_objects[tid]
        # Visualization
        if SHOW_VISUALIZATION:
            vis_img = cv2.normalize(full_temp_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
            if USE_ROI:
                cv2.rectangle(vis_img, (ROI_X, ROI_Y), (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT), (0, 255, 0), 1)
            for track_id, max_temp in current_frame_tracked_output.items():
                if track_id in tracked_objects:
                    center_x, center_y = map(int, tracked_objects[track_id]['centroid'])
                    cv2.circle(vis_img, (center_x, center_y), 5, (0, 0, 255), -1)
                    label_text = f"ID:{track_id} T:{max_temp:.1f}"
                    cv2.putText(vis_img, label_text, (center_x + 5, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Detection & Tracking", vis_img)
        return tracked_objects, next_track_id, current_frame_tracked_output

    try:
        print("DEBUG: Attempting to open file with fnv.file.ImagerFile...")
        im = fnv.file.ImagerFile(file_path)
        print(f"DEBUG: File object created successfully.")

        # --- Initialization Phase ---
        try:
            print("DEBUG: Setting units...")
            unit_str = "°C" # Default based on config
            if im.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
                im.unit = fnv.Unit.TEMPERATURE_FACTORY
                im.temp_type = DESIRED_TEMP_UNIT
                if DESIRED_TEMP_UNIT == fnv.TempType.KELVIN: unit_str = "K"
                elif DESIRED_TEMP_UNIT == fnv.TempType.FAHRENHEIT: unit_str = "°F"
                print(f"DEBUG: Units set to Temperature ({unit_str})")
            else:
                im.unit = fnv.Unit.COUNTS; unit_str = "counts"
                print(f"DEBUG: Units set to Counts")

            print("DEBUG: Accessing metadata...")
            num_frames = im.num_frames
            full_height = im.height
            full_width = im.width
            print(f"DEBUG: Metadata accessed: Frames={num_frames}, Full Dim={full_height}x{full_width}")

            # Validate ROI dimensions
            if USE_ROI:
                 if ROI_X < 0 or ROI_Y < 0 or \
                    (ROI_X + ROI_WIDTH) > full_width or \
                    (ROI_Y + ROI_HEIGHT) > full_height:
                     print(f"ERROR: ROI [{ROI_X},{ROI_Y},{ROI_WIDTH},{ROI_HEIGHT}] is outside frame dimensions [{full_width}x{full_height}]. Disabling ROI.")
                     use_roi_processing = False
                 else:
                     use_roi_processing = True
                     print(f"DEBUG: Using ROI: X={ROI_X}, Y={ROI_Y}, W={ROI_WIDTH}, H={ROI_HEIGHT}")
            else:
                 use_roi_processing = False
                 print("DEBUG: Not using ROI (processing full frame).")

            print("-" * 30)

            if num_frames <= 0: print("DEBUG: No frames found."); return

            if SHOW_VISUALIZATION:
                print("DEBUG: Preparing visualization window...")
                plt.figure("Detection & Tracking")
                print("DEBUG: plt.figure called.")

            print(f"DEBUG: Starting frame loop for {num_frames} frames...")

        except Exception as init_err:
            print("\n!!! DEBUG: Error during initialization phase !!!"); traceback.print_exc()
            if im is not None: im = None # Basic cleanup
            return
        # --- End Initialization Phase ---


        # --- Frame Loop ---
        i = 0
        tracked_objects = OrderedDict()
        next_track_id = 0
        while i < num_frames:
            tracked_objects, next_track_id, current_frame_tracked_output = process_and_show_frame(im, i, tracked_objects, next_track_id)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r'):
                # Re-run just ROI selection, then redraw current frame
                full_temp_data = np.array(im.final, copy=False).reshape((im.height, im.width))
                vis_img = cv2.normalize(full_temp_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
                roi = _select_roi_on_frame(vis_img)
                if roi:
                    ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = map(int, roi)
                    USE_ROI = True
                else:
                    print("ROI NOT changed.")
                i -= 1  # so on next iteration we re-analyze current frame with new ROI
                continue
            if key == ord('c'):
                print("ROI cleared, using full frame.")
                ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT = 0, 0, full_width, full_height
                USE_ROI = False
                i -= 1
                continue
            if key == ord(' '):
                print("Paused. Space: resume, ←/→: move frame, q: quit.")
                while True:
                    key2 = cv2.waitKey(0) & 0xFF
                    if key2 == ord(' '):
                        print("Resumed.")
                        break
                    if key2 in (ord('q'), 27):
                        print("Quit from pause.")
                        i = num_frames
                        break
                    if key2 == 83:  # right arrow →
                        if i < num_frames - 1:
                            i += 1
                            tracked_objects, next_track_id, current_frame_tracked_output = process_and_show_frame(im, i, tracked_objects, next_track_id)
                    if key2 == 81:  # left arrow ←
                        if i > 0:
                            i -= 1
                            tracked_objects, next_track_id, current_frame_tracked_output = process_and_show_frame(im, i, tracked_objects, next_track_id)
            i += 1
        print("DEBUG: Frame loop finished.")

    except Exception as e:
        print(f"\n!!! DEBUG: An error occurred OUTSIDE the main frame loop !!!")
        traceback.print_exc()
    finally:
        # Cleanup
        if im is not None:
            print("DEBUG: Releasing ImagerFile resources...")
            im = None
            print("DEBUG: ImagerFile resources released (set to None).")

# --- Run ---
if __name__ == "__main__":
    print("DEBUG: Script execution started.")
    file_to_process = ""
    if USE_FILE_DIALOG:
        file_to_process = get_file_path_with_dialog()
    else:
        try: file_to_process = SEQ_FILE_PATH
        except NameError: print("ERROR: USE_FILE_DIALOG is False, but SEQ_FILE_PATH is not defined."); sys.exit(1)

    if file_to_process and isinstance(file_to_process, str) and os.path.exists(file_to_process):
        detect_track_hotspots(file_to_process)
    elif not file_to_process: print("DEBUG: No file path provided or selected.")
    else: print(f"ERROR: File path invalid or file does not exist: '{file_to_process}'")

    if SHOW_VISUALIZATION and _matplotlib_available:
        print("DEBUG: Processing finished. Calling plt.show() to keep plot open (if plot window exists).")
        try: plt.show()
        except Exception as plot_err: print(f"DEBUG: Error during final plt.show(): {plot_err}")

    print("DEBUG: Script execution finished.")