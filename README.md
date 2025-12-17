# FirebrandThermalAnalysis
Thermal firebrand detection and tracking dashboard for FLIR SDK SEQ files.

## Features
- Open a single SEQ, multiple SEQs, or a folder of SEQ files.
- Batch CSV export with shared configuration, ROI, and export range.
- Trim exports with start/end frames; use `max` for full length.
- Per-frame detection stats: max/min/avg/median temperature, area, and bbox.

## Quick Start
1. Install FLIR SDK and Python dependencies.
2. Run `python SDK_dashboard.py`.
3. Open SEQ files, adjust configuration, then export CSV.

## Export Notes
- Start/end are 1-based frame numbers.
- End accepts `max` to use each file’s full length.
