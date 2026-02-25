# Firebrand Thermal Analysis
Firebrand Thermal Analysis dashboard for FLIR radiometric files (SEQ, CSQ, JPG, ATS, SFMOV, IMG).

## Highlights
- Open one file, multiple files, or a folder of radiometric files in a single flow.
- Batch CSV export (parallel across files) using shared configuration, ROI, and export range.
- 1-based start/end trim with `max` to use each file's full length.
- Per-detection stats: max/min/avg/median temperature, area, and bbox.
- Export current frame to JPG with ROI and detection overlays.
- Thermal colormaps (Inferno, Jet, Hot, Magma, Plasma, Bone, Turbo, Grayscale).
- Interactive zoom & pan with mouse hover temperature readout.
- Color bar showing temperature-to-color gradient.
- Comprehensive keyboard shortcuts for playback, zoom, colormaps, and fullscreen.

## Compressed File Support
Firebrand Thermal Analysis natively supports **NetCDF4 (`.nc`)** files heavily compressed by the companion [SEQ-CSQ-compressor](https://github.com/NinhGhoster/SEQ-CSQ-compressor) tool. This dramatically speeds up analysis workflows by letting you process 37GB thermal videos as ~10GB heavily-compressed random-access files without needing to wait for decompression.

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



## Using the Dashboard
### Open files
- **Open** opens a small menu where you can load radiometric file(s) or a folder.
- When a folder is selected, all supported files (`.seq`, `.csq`, `.jpg`, `.ats`, `.sfmov`, `.img`) are discovered recursively (including subfolders) and loaded in sorted order.
- Use **<< / >>** to switch the current view.

### Playback
- **Play/Pause** toggles play/pause, **< / >** step frames, and the frame slider scrubs.
- Keyboard: `Space` toggles play/pause, `Left`/`Right` or `,`/`.` steps frames.

### Export settings
- **Detection Threshold**: temperature threshold (C) for firebrand detection.
- **Emissivity**: the metadata value is shown for the current file; default input is 0.9.
- **Export Range**: start/end are 1-based frame numbers. End accepts `max`.
- **Start = N / End = N** uses the current frame number (shows Set start/end when no file is loaded).
- **Apply to `<file>`** saves settings for the current file; **Apply all** applies to all loaded files.

### Region of Interest (ROI)
- Manual tab: drag on the canvas or edit ROI fields numerically.
- Auto tab: auto-detect ROI above the fuel bed from the first frame (margin adjustable).
- ROI updates apply to the current file or all files via Apply to `<file>` / Apply all.

### Export actions
- Single **Export...** button opens a menu:
  - **Export CSV (current)**: saves `basename.csv` next to the source file.
  - **Export CSV (all files)**: exports all loaded files in parallel (one process per file).
  - **Save frame image (JPG)**: saves `basename_frame_00001.jpg` with overlays next to the source file.



### Visualisation & Interface
- **Modern Interface**: Deep dark mode UI powered by CustomTkinter featuring a sleek Bento Grid layout.
- **Colormaps**: select from the Visualisation dropdown (Inferno, Jet, Hot, Magma, Plasma, Bone, Turbo, Grayscale), or press `1`–`8` to quick-select.
- **Zoom**: scroll wheel zooms 0.5×–10× centred on cursor; `+`/`-` keys zoom in/out; `0` or double-click resets.
- **Pan**: middle-click drag pans the view.
- **Temperature readout**: hover over the canvas to see the temperature at the cursor in the status bar.
- **Color bar**: gradient strip to the right of the canvas showing the current colormap with min / max temperature labels.
- **Fullscreen**: press `F` to toggle fullscreen, `Escape` to exit.

### Keyboard Shortcuts
| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `S` | Stop |
| `←` / `,` | Previous frame |
| `→` / `.` | Next frame |
| `Home` | Jump to first frame |
| `End` | Jump to last frame |
| `+` / `-` | Zoom in / out |
| `0` | Reset zoom |
| `1`–`8` | Quick-select colormap |
| `R` | Reset ROI |
| `F` | Toggle fullscreen |
| `Escape` | Exit fullscreen |
| `Double-click` | Reset zoom |
| `Scroll wheel` | Zoom on cursor |
| `Middle-drag` | Pan view |

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



## Build & Package
Builds must be done on the target OS with FLIR SDK installed. PyInstaller is included in `environment.yml`.

```bash
# macOS
./build/build_macos.sh && ./build/package_macos_dmg.sh

# Windows
.\build\build_windows.ps1 && .\build\package_windows.ps1

# Linux
./build/build_linux.sh && ./build/package_linux_appimage.sh
```

CI via `.github/workflows/build.yml` builds all platforms on tag push (e.g. `v0.1.0`).

Optional env vars for all platforms: `FLIR_SDK_WHEEL`, `FLIR_SDK_LIB_DIR`, `FLIR_SDK_BIN_DIR`.

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
- `tests/`: unit tests (`pytest`).
- `tutorial/`: SDK usage examples.

## About the Project
Firebrand Thermal Analysis was developed to provide researchers with a high-performance, GUI-driven tool for extracting and tracking thermal data from FLIR radiometric video files. It is built to optimize the workflow of analyzing massive datasets, automating the detection of firebrands (embers) and fuel bed hotspots in combustion experiments.

**Project Team:**
- H. Nguyen
- J. Filippi
- T. Penman
- M. Peace
- A. Filkov

### Companion Tools
If you are dealing with extremely large radiometric videos (e.g., 30GB+ SEQ files), we highly recommend using our companion **[SEQ-CSQ-compressor](https://github.com/NinhGhoster/SEQ-CSQ-compressor)** tool. It uses NetCDF4 and zlib deflation to permanently reduce file sizes by up to 70%—while retaining strict physical temperature accuracy (0.01 °C) and all embedded camera metadata. Firebrand Thermal Analysis natively reads these `.nc` files for instant, random-access playback without requiring any manual decompression.
