# Thermal Radiometric Compression Tool — Project Spec

## Problem
FLIR SEQ/CSQ radiometric video files are ~30 GB for 15 minutes of recording (640×512 @ 30fps, 16-bit per pixel). Uploading, sharing, and archiving these files is impractical. A lossless compression/conversion tool is needed.

## Goal
Convert proprietary FLIR SEQ/CSQ files into **compressed NetCDF4** format while retaining **all radiometric temperature data** losslessly. The converted files should be:
- 50–70% smaller than the originals
- Self-describing (embedded metadata)
- Readable by standard scientific tools (Python/xarray, Matlab, R, Julia)
- Publishable to data repositories (Zenodo, Figshare, Dryad)

## Format: NetCDF4

### Why NetCDF4
- Scientific standard for atmospheric/remote sensing data
- HDF5-backed with built-in chunked compression (`zlib` + `shuffle`)
- Self-describing: metadata, units, calibration embedded in the file
- Random access: read any frame without decompressing the whole file
- Universal tool support: `xarray`, `netCDF4-python`, `h5py`, Matlab, R, Julia

### Data Schema
```
Dimensions:
  frame  = UNLIMITED (appendable)
  y      = <height>       (e.g. 512)
  x      = <width>        (e.g. 640)

Variables:
  temperature(frame, y, x) — float32 or scaled int16
    units:       "degC" (or "K")
    long_name:   "Radiometric temperature"
    chunksizes:  (10, height, width)
    zlib:        True
    complevel:   6
    shuffle:     True

Global Attributes:
  source_file:     original filename
  camera_model:    e.g. "FLIR A655sc"
  emissivity:      float
  frame_rate:      float (fps)
  date:            ISO 8601
  experiment:      free-text description
  software:        "thermal-compress v1.0"
  compression:     "netcdf4-zlib-shuffle"
```

### Compression Optimisation
- **`shuffle=True`**: rearranges bytes before compression — big win for 16-bit data
- **Scaled int16**: store as `int16` with `scale_factor=0.01` and `add_offset=0` for 0.01°C precision — halves storage vs float32, compresses much better
- **Chunk size**: `(10, H, W)` balances sequential playback (temporal locality) with single-frame random access
- **`complevel=6`**: good balance of ratio vs speed; increase to 9 for archival

## CLI Interface

```bash
# Convert SEQ/CSQ → NetCDF4
thermal-compress encode input.seq -o output.nc \
  --emissivity 0.95 \
  --experiment "Stringybark, 50kW/m, Rep 3" \
  --complevel 6

# Batch convert a folder  
thermal-compress encode /path/to/seq_folder/ -o /path/to/output/ --batch

# Verify lossless round-trip
thermal-compress verify input.seq output.nc

# Inspect metadata
thermal-compress info output.nc

# Extract frames back to raw numpy (optional)
thermal-compress decode output.nc -o frames/ --format npy
```

## Architecture

```
thermal-compress/
├── thermal_compress/
│   ├── __init__.py
│   ├── cli.py              # Click/argparse CLI entry point
│   ├── encoder.py           # SEQ/CSQ → NetCDF4 conversion
│   ├── decoder.py           # NetCDF4 → raw frames / back to SEQ
│   ├── verify.py            # Bitwise verification
│   ├── metadata.py          # FLIR SDK metadata extraction
│   └── compression.py       # Compression config (zlib, shuffle, chunking)
├── tests/
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_roundtrip.py
├── environment.yml
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## Dependencies
```yaml
# environment.yml
name: thermal-compress
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - numpy
  - netcdf4
  - xarray
  - h5py
  - click
  - tqdm
  - pytest
  - pip:
    - # FLIR SDK wheel (per OS)
```

## Implementation Notes

### Reading SEQ/CSQ via FLIR SDK
```python
import fnv
import numpy as np

im = fnv.file.ImagerFile(seq_path)
im.unit = fnv.Unit.TEMPERATURE_FACTORY
im.temp_type = fnv.TempType.CELSIUS
width, height = im.width, im.height
num_frames = im.num_frames

for idx in range(num_frames):
    im.get_frame(idx)
    frame = np.array(im.final, copy=True).reshape((height, width))
    # → write to NetCDF variable
```

### Writing NetCDF4
```python
import netCDF4 as nc

ds = nc.Dataset("output.nc", "w", format="NETCDF4")
ds.createDimension("frame", None)
ds.createDimension("y", height)
ds.createDimension("x", width)

temp = ds.createVariable(
    "temperature", "f4", ("frame", "y", "x"),
    chunksizes=(10, height, width),
    zlib=True, complevel=6, shuffle=True,
)
temp.units = "degC"

for idx in range(num_frames):
    im.get_frame(idx)
    frame = np.array(im.final, copy=True).reshape((height, width))
    temp[idx, :, :] = frame

ds.close()
```

### Verification
```python
# Read back and compare frame-by-frame
ds = nc.Dataset("output.nc", "r")
for idx in range(num_frames):
    im.get_frame(idx)
    original = np.array(im.final, copy=True).reshape((height, width))
    restored = ds.variables["temperature"][idx, :, :]
    assert np.allclose(original, restored, atol=1e-6), f"Frame {idx} mismatch!"
```

## Expected Compression Ratios
| Storage Method | Size (15 min) | Reduction |
|---|---|---|
| Raw SEQ/CSQ | ~30 GB | — |
| NetCDF4 float32 + zlib + shuffle | ~10–12 GB | ~60–65% |
| NetCDF4 scaled int16 + zlib + shuffle | ~5–8 GB | ~73–83% |

## Context
- This tool is a companion to [Firebrand Thermal Analysis](https://github.com/NinhGhoster/Firebrand-Thermal-Analysis)
- SEQ/CSQ files are produced by FLIR cameras during bushfire firebrand experiments
- The FLIR Science File SDK (`fnv` Python package) is required for reading proprietary files
- Compressed files should be usable by the dashboard app (add NetCDF reader as an input source)
