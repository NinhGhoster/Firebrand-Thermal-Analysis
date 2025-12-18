# FirebrandThermalAnalysis
Thermal firebrand detection and tracking dashboard for FLIR SDK SEQ files.

## Features
- Open one SEQ, multiple SEQs, or a folder of SEQ files in a single flow.
- Batch CSV export using shared configuration, ROI, and export range.
- Trim exports with start/end frames; use `max` for full length.
- Per-detection stats: max/min/avg/median temperature, area, and bbox.
- Export current frame to JPG with ROI and detection overlays.

## Requirements
- FLIR SDK (install from `SDK installation/`)
- Python 3.12 (conda or system Python)

## Setup
```bash
conda env create -f environment.yml
# or
pip install -r requirements.txt
```

## Run
```bash
python SDK_dashboard.py
```

## Usage
1. Click **Open SEQ** and select one or more SEQ files (or a folder).
2. Configure threshold, emissivity, export range, and ROI.
3. Use **Apply** for the current file or **Apply All** for all loaded files.
4. Export current frame to JPG or export CSV for the current file/all files.

## Export Notes
- Start/end are 1-based frame numbers.
- End accepts `max` to use each file’s full length.
- CSV files are saved next to the source SEQ with the same base name.

## Credits
Developed from FLIR SDK by H. Nguyen.
