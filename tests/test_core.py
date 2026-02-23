# -*- coding: utf-8 -*-
"""
Unit tests for core detection, tracking, ROI, and version logic.

These tests exercise the pure functions in FirebrandThermalAnalysis.py
without requiring the FLIR SDK (fnv) or a GUI. Numpy is the only
runtime dependency.
"""
import sys
import os
from collections import OrderedDict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Insert project root so we can import the module directly.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from FirebrandThermalAnalysis import (
    clamp_roi,
    detect_firebrands,
    assign_tracks,
    _parse_version,
    _is_newer_version,
    SUPPORTED_EXTENSIONS,
    TARGET_FPS,
)


# ===================================================================
# clamp_roi
# ===================================================================

class TestClampRoi:
    """Tests for the ``clamp_roi`` helper."""

    def test_none_returns_none(self):
        assert clamp_roi(None, 640, 480) is None

    def test_valid_roi_unchanged(self):
        roi = (10, 20, 100, 80)
        result = clamp_roi(roi, 640, 480)
        assert result == roi

    def test_roi_exceeds_width(self):
        """ROI width should be clamped to frame boundary."""
        result = clamp_roi((600, 0, 100, 50), 640, 480)
        assert result is not None
        x, y, w, h = result
        assert x + w <= 640

    def test_roi_exceeds_height(self):
        """ROI height should be clamped to frame boundary."""
        result = clamp_roi((0, 450, 100, 100), 640, 480)
        assert result is not None
        x, y, w, h = result
        assert y + h <= 480

    def test_zero_size_clamped_to_minimum(self):
        """A zero-area ROI should be clamped to minimum 1px size."""
        result = clamp_roi((0, 0, 0, 0), 640, 480)
        assert result is not None
        x, y, w, h = result
        assert w >= 1
        assert h >= 1

    def test_negative_origin_clamped(self):
        """Negative ROI origin should be clamped to 0."""
        result = clamp_roi((-10, -20, 100, 80), 640, 480)
        if result is not None:
            x, y, w, h = result
            assert x >= 0
            assert y >= 0


# ===================================================================
# detect_firebrands
# ===================================================================

class TestDetectFirebrands:
    """Tests for firebrand detection on synthetic frames."""

    def _make_frame(self, width: int = 100, height: int = 100,
                    bg_temp: float = 20.0) -> np.ndarray:
        """Create a uniform-temperature frame."""
        return np.full((height, width), bg_temp, dtype=np.float32)

    def test_empty_frame_no_detections(self):
        """A cold frame should yield zero detections."""
        frame = self._make_frame()
        dets = detect_firebrands(frame, None, temp_thresh=100.0)
        assert dets == []

    def test_single_hotspot(self):
        """A single hot blob should produce exactly one detection."""
        frame = self._make_frame()
        # Paint a 5x5 hot blob
        frame[40:45, 50:55] = 500.0
        dets = detect_firebrands(frame, None, temp_thresh=100.0)
        assert len(dets) == 1
        det = dets[0]
        assert det["max_temp"] == pytest.approx(500.0)
        assert det["area"] == 25

    def test_roi_excludes_outside(self):
        """Detections outside the ROI should be ignored."""
        frame = self._make_frame()
        frame[10:15, 10:15] = 500.0  # hot blob at (10,10)
        roi = (50, 50, 50, 50)  # only look at bottom-right
        dets = detect_firebrands(frame, roi, temp_thresh=100.0)
        assert dets == []

    def test_roi_includes_inside(self):
        """Detections inside the ROI should be found."""
        frame = self._make_frame()
        frame[60:65, 60:65] = 500.0
        roi = (50, 50, 50, 50)
        dets = detect_firebrands(frame, roi, temp_thresh=100.0)
        assert len(dets) == 1

    def test_area_filter(self):
        """Objects too small should be filtered out."""
        frame = self._make_frame()
        frame[50, 50] = 500.0  # single pixel
        dets = detect_firebrands(frame, None, temp_thresh=100.0)
        # MIN_OBJECT_AREA_PIXELS is 3 by default, so 1px should be filtered
        assert dets == []


# ===================================================================
# assign_tracks
# ===================================================================

class TestAssignTracks:
    """Tests for nearest-centroid tracking."""

    def _det(self, cx: float, cy: float, tid: int = -1) -> dict:
        """Create a minimal detection dict."""
        return {
            "centroid": (cx, cy),
            "max_temp": 400.0,
            "min_temp": 400.0,
            "avg_temp": 400.0,
            "median_temp": 400.0,
            "area": 10,
            "bbox": (int(cx) - 2, int(cy) - 2, 5, 5),
        }

    def test_new_track_assigned(self):
        """A detection with no prior tracks should get ID 1."""
        tracked = OrderedDict()
        dets = [self._det(50, 50)]
        tracked, dets, next_id = assign_tracks(dets, tracked, 1, 0)
        assert dets[0]["track_id"] == 1
        assert next_id == 2

    def test_continue_track(self):
        """A nearby detection should inherit the existing track ID."""
        tracked = OrderedDict()
        dets1 = [self._det(50, 50)]
        tracked, dets1, next_id = assign_tracks(dets1, tracked, 1, 0)

        dets2 = [self._det(52, 52)]  # nearby
        tracked, dets2, next_id = assign_tracks(dets2, tracked, next_id, 1)
        assert dets2[0]["track_id"] == 1

    def test_far_detection_new_track(self):
        """A far-away detection should get a new track ID."""
        tracked = OrderedDict()
        dets1 = [self._det(50, 50)]
        tracked, dets1, next_id = assign_tracks(dets1, tracked, 1, 0)

        dets2 = [self._det(200, 200)]  # far away
        tracked, dets2, next_id = assign_tracks(dets2, tracked, next_id, 1)
        assert dets2[0]["track_id"] == 2

    def test_multiple_detections(self):
        """Multiple detections should each get unique track IDs."""
        tracked = OrderedDict()
        dets = [self._det(10, 10), self._det(80, 80)]
        tracked, dets, next_id = assign_tracks(dets, tracked, 1, 0)
        ids = [d["track_id"] for d in dets]
        assert len(set(ids)) == 2


# ===================================================================
# _parse_version / _is_newer_version
# ===================================================================

class TestVersionParsing:
    """Tests for version string parsing and comparison."""

    def test_parse_simple(self):
        assert _parse_version("v1.2.3") == (1, 2, 3)

    def test_parse_no_prefix(self):
        assert _parse_version("1.2.3") == (1, 2, 3)

    def test_parse_two_part(self):
        result = _parse_version("v1.2")
        assert result[0] == 1
        assert result[1] == 2

    def test_newer_true(self):
        assert _is_newer_version("v0.0.1", "v0.0.2") is True

    def test_newer_false_same(self):
        assert _is_newer_version("v1.0.0", "v1.0.0") is False

    def test_newer_false_older(self):
        assert _is_newer_version("v2.0.0", "v1.0.0") is False


# ===================================================================
# Constants sanity checks
# ===================================================================

class TestConstants:
    """Sanity checks for module-level constants."""

    def test_supported_extensions(self):
        assert ".seq" in SUPPORTED_EXTENSIONS
        assert ".csq" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".ats" in SUPPORTED_EXTENSIONS
        assert ".sfmov" in SUPPORTED_EXTENSIONS
        assert ".img" in SUPPORTED_EXTENSIONS

    def test_target_fps_positive(self):
        assert TARGET_FPS > 0
