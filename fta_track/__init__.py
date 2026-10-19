"""Headless firebrand detection and tracking from radiometric thermal video.

Temperature, not pixel brightness, is the detection signal: a firebrand is a
transient object at or above the camera's calibrated floor (300 degC) that
leaves the burning sample. The package reads Level 1.1 NetCDF recordings
(float32 degC) and writes per-detection, per-track and per-recording outputs.

Entry point for HPC use: ``FTA_CLI.py run <file.nc> --out <dir>``.
"""

__all__ = ["Params", "run_recording"]

from .params import Params
from .pipeline import run_recording
