# Firebrand Thermal Analysis
Firebrand Thermal Analysis dashboard for FLIR radiometric files (SEQ, CSQ, JPG, ATS, SFMOV, IMG, NC). Developed to provide researchers with a high-performance, GUI-driven tool for extracting and tracking thermal data from FLIR radiometric video files, automating the detection of firebrands (embers) and fuel bed hotspots in combustion experiments.

## Highlights
- Open single files, multiple files, or entire folders of radiometric files.
- Batch CSV export (parallel across files) with shared configuration, ROI, and export range.
- Per-detection stats: max/min/avg/median temperature, area, and bounding box.
- Thermal colormaps (Inferno, Jet, Hot, Magma, Plasma, Bone, Turbo, Grayscale).
- Interactive zoom & pan with live temperature readout under cursor.
- Native support for compressed **NetCDF4 (`.nc`)** files from the companion [SEQ-CSQ-compressor](https://github.com/NinhGhoster/SEQ-CSQ-compressor).

## Requirements
- Python 3.12 (conda recommended)
- FLIR Science File SDK (see `SDK/` for platform wheels)
- OpenCV (`opencv-python-headless`, included in conda env)

## Quick Start
```bash
conda env create -f environment.yml
conda activate firebrand-thermal

# Install FLIR SDK wheel for your platform
pip install "SDK/FileSDK-2024.7.1-cp312-cp312-macosx_10_14_universal2.whl"

python FirebrandThermalAnalysis.py
```

## Usage

| Feature | Details |
|---|---|
| **Open files** | Load file(s) or a folder; supported formats are discovered recursively. Use `<< / >>` to switch between files. |
| **Playback** | Play/Pause, frame stepping (`< / >`), slider scrubbing. `Space` toggles playback. |
| **Detection** | Temperature threshold + connected components (8-connectivity), filtered by area. |
| **Tracking** | Nearest-centroid matching with distance cap and short-term memory. |
| **ROI** | Draw on canvas, enter numerically, or auto-detect fuel bed region. |
| **Export CSV** | Per-file or batch parallel export. 1-based frame range with `max` support. |
| **Export JPG** | Save current frame with ROI and detection overlays. |
| **Emissivity** | Reads metadata value; user can override per-file or globally. |

### Keyboard Shortcuts
| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `S` | Stop |
| `← / ,` | Previous frame |
| `→ / .` | Next frame |
| `Home / End` | First / last frame |
| `+ / -` | Zoom in / out |
| `0` | Reset zoom |
| `1`–`8` | Quick-select colormap |
| `R` | Reset ROI |
| `F` | Toggle fullscreen |

## CSV Schema
| Column | Description |
|---|---|
| `frame` | 1-based frame index |
| `firebrand_id` | Track ID (per export) |
| `max/min/avg/median_temperature` | Detection temperature stats |
| `area_pixels` | Connected-component area |
| `bbox_x/y/w/h` | Bounding box |

## Build & Package
Builds require the target OS with FLIR SDK installed. PyInstaller is included in the conda env.

```bash
# macOS
./build/build_macos.sh && ./build/package_macos_dmg.sh

# Windows
.\build\build_windows.ps1 && .\build\package_windows.ps1

# Linux
./build/build_linux.sh && ./build/package_linux_appimage.sh
```

CI via `.github/workflows/build.yml` builds all platforms on tag push (e.g. `v0.1.0`).

## Troubleshooting
- **"FLIR SDK required"**: ensure the SDK wheel is installed.
- **Counts vs °C**: files without temperature units display raw counts.

## Repository Layout
| Path | Description |
|---|---|
| `FirebrandThermalAnalysis.py` | Main dashboard UI and export logic |
| `SDK/` | FLIR SDK installers and wheels |
| `build/` | Platform build and packaging scripts |
| `tests/` | Unit tests (`pytest`) |

**Project Team:** H. Nguyen, J. Filippi, T. Penman, M. Peace, A. Filkov

### Companion Tools
For extremely large radiometric videos (30GB+), use the companion **[SEQ-CSQ-compressor](https://github.com/NinhGhoster/SEQ-CSQ-compressor)** to reduce file sizes by up to 70% while preserving 0.01 °C precision and all camera metadata. This app natively reads the resulting `.nc` files for instant random-access playback.
