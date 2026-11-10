"""Real-world size from the file's metadata, in the desktop app and the shared optics module."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fta_track.optics import blob_extent_px, hfov_from_metadata, pixel_size_m, sizes_mm  # noqa: E402


def test_lens_and_camera_name_give_the_field_of_view():
    assert hfov_from_metadata("FOL21", "FLIR T1040 45__") == (45.0, "lens FOL21")
    assert hfov_from_metadata(None, "FLIR T1040 45__")[0] == 45.0      # the degree sign is written '__'
    assert hfov_from_metadata("FOL36", None)[0] == 28.0
    assert hfov_from_metadata(None, None) == (None, "unknown")


def test_pixel_size_matches_the_t1040_at_2_m():
    # 2 x 2 m x tan(22.5 deg) / 1024 px = 1.618 mm per pixel
    assert pixel_size_m(2.0, 1024, 45.0) * 1000 == pytest.approx(1.618, abs=1e-3)
    assert pixel_size_m(1.0, 1024, 45.0) * 1000 == pytest.approx(0.809, abs=1e-3)
    assert pixel_size_m(2.0, 1024, ifov_rad=0.8e-3) == pytest.approx(1.6e-3)   # the file's IFOV wins
    assert pixel_size_m(None, 1024, 45.0) is None and pixel_size_m(2.0, 1024, None) is None


def test_blob_size_in_millimetres():
    mask = np.zeros((20, 20), bool)
    mask[5:8, 4:14] = True                      # a 10 x 3 px strip
    length, width = blob_extent_px(mask)
    assert (length, width) == (10.0, 3.0)
    s = 0.002                                   # 2 mm per pixel
    z = sizes_mm(mask.sum(), length, width, s)
    assert z["area_mm2"] == pytest.approx(30 * 4.0)
    assert (z["length_mm"], z["width_mm"]) == (20.0, 6.0)
    assert z["equiv_diameter_mm"] == pytest.approx(2 * math.sqrt(120 / math.pi))
    assert sizes_mm(30, 10, 3, None)["area_mm2"] is None


def test_a_diagonal_strip_is_measured_along_its_length():
    mask = np.zeros((40, 40), bool)
    for i in range(20):
        mask[10 + i, 10 + i] = True
    length, _ = blob_extent_px(mask)
    assert length == pytest.approx(19 * math.sqrt(2) + 1, rel=0.05)


fta = pytest.importorskip("FirebrandThermalAnalysis")


def test_detections_and_csv_rows_carry_real_size():
    frame = np.full((50, 60), 20.0, np.float32)
    frame[10:13, 20:30] = 600.0
    det = fta.detect_firebrands(frame, None, 300.0)[0]
    assert (det["length_px"], det["width_px"]) == (10.0, 3.0)
    row = dict(zip(fta.CSV_HEADER, fta.detection_csv_row(7, det, 0.0016)))
    assert row["frame"] == 7 and row["area_pixels"] == 30
    assert row["pixel_size_mm"] == pytest.approx(1.6)
    assert row["length_mm"] == pytest.approx(16.0) and row["area_mm2"] == pytest.approx(76.8)
    blank = dict(zip(fta.CSV_HEADER, fta.detection_csv_row(7, det, None)))
    assert blank["area_mm2"] == "" and blank["area_pixels"] == 30


def test_netcdf_reader_reads_distance_and_lens(tmp_path):
    nc = pytest.importorskip("netCDF4")
    path = tmp_path / "r.nc"
    ds = nc.Dataset(path, "w")
    ds.createDimension("frame", 2); ds.createDimension("y", 8); ds.createDimension("x", 1024)
    ds.createVariable("temperature", "f4", ("frame", "y", "x"))[:] = 20.0
    ds.setncatts({"distance": 2.0, "lens": "FOL21", "camera_model": "FLIR T1040 45__"})
    ds.close()
    r = fta.create_video_reader(str(path))
    assert (r.distance_m, r.hfov_deg, r.scale_source) == (2.0, 45.0, "lens FOL21")
    assert fta.reader_pixel_size_m(r) * 1000 == pytest.approx(1.618, abs=1e-3)
    assert fta.reader_pixel_size_m(r, 1.0) * 1000 == pytest.approx(0.809, abs=1e-3)   # distance override
