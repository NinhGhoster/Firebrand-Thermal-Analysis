# Firebrand Thermal Analysis
Firebrand Thermal Analysis dashboard for FLIR SEQ files.

## Highlights
- Open one SEQ, multiple SEQs, or a folder of SEQ files in a single flow.
- Batch CSV export (parallel across files) using shared configuration, ROI, and export range.
- 1-based start/end trim with `max` to use each file's full length.
- Per-detection stats: max/min/avg/median temperature, area, and bbox.
- Export current frame to JPG with ROI and detection overlays.
- Keyboard shortcuts for playback and frame stepping.

## Requirements
- FLIR Science File SDK installed (see `SDK/` for wheels).
- Python 3.12 (conda environment recommended).
- OpenCV via `opencv-python-headless` (included in the conda env).
- macOS builds are Apple Silicon (arm64) only.

## Quick Start
```bash
conda env create -f environment.yml
conda activate firebrand-thermal

# Install the FLIR SDK Python wheel for your OS
# macOS:
pip install "SDK/FileSDK-2024.7.1-cp312-cp312-macosx_10_14_universal2.whl"
# Windows:
# pip install "SDK/FileSDK-2024.7.1-cp312-cp312-win_amd64.whl"
# Linux:
# pip install "SDK/FileSDK-2024.7.1-cp312-cp312-linux_x86_64.whl"

python FirebrandThermalAnalysis.py
```

## Update the Conda Environment
```bash
conda env update -f environment.yml --prune
```

## Using the Dashboard
### Open SEQ files
- **Open** opens a small menu where you can load SEQ file(s) or a folder.
- When a folder is selected, all `.seq` files are discovered recursively (including subfolders) and loaded in sorted order.
- Use **<< / >>** to switch the current view.

### Playback
- **Play/Pause** toggles play/pause, **< / >** step frames, and the frame slider scrubs.
- Keyboard: `Space` toggles play/pause, `Left`/`Right` or `,`/`.` steps frames.

### Export settings
- **Detection Threshold**: temperature threshold (C) for firebrand detection.
- **Emissivity**: the metadata value is shown for the current file; default input is 0.9.
- **Export Range**: start/end are 1-based frame numbers. End accepts `max`.
- **Start = N / End = N** uses the current frame number (shows Set start/end when no file is loaded).
- **Apply to `<file>`** saves settings for the current SEQ; **Apply all** applies to all loaded files.

### Region of Interest (ROI)
- Manual tab: drag on the canvas or edit ROI fields numerically.
- Auto tab: auto-detect ROI above the fuel bed from the first frame (margin adjustable).
- ROI updates apply to the current file or all files via Apply to `<file>` / Apply all.

### Export actions
- Single **Export...** button opens a menu:
  - **Export CSV (current)**: saves `basename.csv` next to the SEQ.
  - **Export CSV (all files)**: exports all loaded SEQs in parallel (one process per SEQ).
  - **Save frame image (JPG)**: saves `basename_frame_00001.jpg` with overlays next to the SEQ.

## How it works
1) Load SEQ  
   - First file sets the batch; `<< / >>` moves through loaded files.  
   - Metadata emissivity is read and shown; override is optional.  
   - Unit defaults to temperature; falls back to counts if unavailable.

2) ROI  
   - Manual tab: numeric ROI fields and canvas drag.  
   - Auto tab: one-click auto ROI from the first frame. It finds the bright fuel bed in the lower half (mean > global_mean + 0.5*std) and sets ROI from the top of the image down to just above the fuel bed (minus a user margin). If detection fails, it falls back to the top 40% band. Margin (px) is adjustable.

3) Detection  
   - Temperature threshold (C) applied to ROI; binary mask -> connected components (8-connectivity).  
   - Filters by area (`MIN_OBJECT_AREA_PIXELS` to `MAX_OBJECT_AREA_PIXELS`).  
   - Per detection stats: max/min/avg/median temperature and area + bbox.

4) Tracking  
   - Nearest-centroid matching with distance cap (`CENTROID_TRACKING_MAX_DIST`) and short memory (`TRACK_MEMORY_FRAMES`).  
   - Track IDs reset per export range; IDs increment as new objects appear.

5) Export  
   - CSV: 1-based frame numbers; start/end range; `max` uses full length. Saves `basename.csv` next to the SEQ.  
   - JPG: saves `basename_frame_00001.jpg` with ROI/detections overlay next to the SEQ.  
   - Status bar always prefixes `Status:` and updates during export.

6) Controls & shortcuts  
   - |< rewind, > play/pause, < / >> step, space/arrow/comma/period keys for stepping, slider for scrubbing.  
   - Apply to `<file>` saves settings per file; Apply all propagates to all loaded files.

## CSV Schema
Each row is one detected firebrand in a frame.

| Column | Description |
| --- | --- |
| `frame` | 1-based frame index |
| `firebrand_id` | Track ID (assigned per export) |
| `max_temperature` | Max temperature in the detection |
| `min_temperature` | Min temperature in the detection |
| `avg_temperature` | Mean temperature in the detection |
| `median_temperature` | Median temperature in the detection |
| `area_pixels` | Connected-component area in pixels |
| `bbox_x` | Bounding box left |
| `bbox_y` | Bounding box top |
| `bbox_w` | Bounding box width |
| `bbox_h` | Bounding box height |

## Detection + Tracking Overview
- Detection: temperature threshold + connected components (8-connectivity).
- Tracking: nearest-centroid matching with a distance cap and short-term memory.
- Track IDs start at 1 for each export range and increment with new detections.

## Build Executables
Builds must be done on the target OS with FLIR SDK installed.
PyInstaller is included in `environment.yml`; install it manually if missing.

### GitHub Actions (CI)
The workflow at `.github/workflows/build.yml` builds macOS, Windows, and Linux
distributions on demand or when you push a tag like `v0.1.0`.
Artifacts are uploaded to the workflow run and published on releases as
`FirebrandThermalAnalysis.dmg`, `FirebrandThermalAnalysis.exe`, and
`FirebrandThermalAnalysis.AppImage`.

### macOS
```bash
./build/build_macos.sh
```
Optional env vars:
- `FLIR_SDK_WHEEL` (preferred) or `FLIR_SDK_PYTHON_DIR` + `FLIR_SDK_SHADOW_DIR`
- `FLIR_SDK_LIB_DIR` and `FLIR_SDK_BIN_DIR` to bundle SDK dylibs

### Windows
```powershell
.\build\build_windows.ps1
```
Build output: `dist/FirebrandThermalAnalysis.exe` (single file).
Optional env vars:
- `FLIR_SDK_WHEEL` (preferred) or `FLIR_SDK_PYTHON_DIR` + `FLIR_SDK_SHADOW_DIR`
- `FLIR_SDK_LIB_DIR` and `FLIR_SDK_BIN_DIR` to bundle SDK DLLs

### Linux
```bash
./build/build_linux.sh
```
Optional env vars:
- `FLIR_SDK_WHEEL` (preferred) or `FLIR_SDK_PYTHON_DIR` + `FLIR_SDK_SHADOW_DIR`
- `FLIR_SDK_LIB_DIR` and `FLIR_SDK_BIN_DIR` to bundle SDK shared libs

## Package Installers
### macOS (DMG)
```bash
./build/package_macos_dmg.sh
```

### Windows (Installer)
Install Inno Setup, then run:
```powershell
.\build\package_windows.ps1
```
This packages the single-file EXE into an optional installer.

### Linux (AppImage)
Install `appimagetool`, then run:
```bash
./build/package_linux_appimage.sh
```
The script generates a default 256x256 icon for AppImage metadata.

## Troubleshooting
- **"FLIR SDK required"**: ensure the SDK wheel is installed and the build
  scripts bundle the `fnv` package.
- **OpenCV warning about metadata depth**: the SDK encoder falls back to 8-bit;
  it is expected and does not affect temperature calculations.
- **Counts vs C**: if the file has no temperature unit, values are in counts.

## Repository Layout
- `FirebrandThermalAnalysis.py`: main dashboard UI and export logic.
- `SDK.py`: legacy tracking + detection implementation.
- `SDK/`: FLIR SDK installers and wheels.
- `build/`: platform build scripts and packaging helpers.
- `dist/`: packaged outputs (generated/ignored).
- `tutorial/`: SDK usage examples.

## Credits
Developed from FLIR SDK by H. Nguyen.
