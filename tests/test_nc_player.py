# -*- coding: utf-8 -*-
"""
Offline tests for nc_player.py pure helpers (no GUI, no network).
"""
import os
import sys

import numpy as np
import pytest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import nc_player as np_  # noqa: E402


class TestNormalizeRender:
    """Min-max normalization and RGB rendering."""

    def test_normalize_uint8_range(self):
        frame = np.zeros((32, 32), dtype=np.float32)
        frame[10:14, 10:14] = 500.0
        norm = np_.normalize_frame(frame)
        assert norm.dtype == np.uint8
        assert norm.min() == 0 and norm.max() == 255

    def test_render_shapes(self):
        frame = np.zeros((32, 32), dtype=np.float32)
        for cmap in ("Grayscale", "Inferno"):
            vis = np_.render_rgb(frame, cmap)
            assert vis.shape == (32, 32, 3)
            assert vis.dtype == np.uint8


class TestDetectionBbox:
    """Center-based Roboflow boxes to top-left corner boxes."""

    def test_center_to_corner(self):
        assert np_.detection_bbox(
            {"x": 40.0, "y": 118.0, "width": 12.0, "height": 12.0}
        ) == (34, 112, 12, 12)


class TestEmberBlobs:
    """find_ember_blobs: hot-tail connected components with size filtering."""

    def _frame(self):
        # flat 300-deg background; blobs all at 700 so the p99.9 threshold
        # lands on the tied plateau and every blob is in the tail
        arr = np.full((60, 60), 300.0, dtype=np.float32)
        arr[10:15, 10:15] = 700.0   # 5x5 ember
        arr[40:44, 40:44] = 700.0   # 4x4 ember
        arr[0, 0] = 700.0           # 1x1 noise - below min_area
        arr[20:50, 5:35] = 700.0    # big plume - above max_area
        return arr

    def test_finds_ember_sized_blobs_only(self):
        blobs = np_.find_ember_blobs(self._frame())
        sizes = sorted(b["area"] for b in blobs)
        assert sizes == [16, 25]  # noise and plume filtered out

    def test_center_and_peak_fields(self):
        blobs = np_.find_ember_blobs(self._frame())
        b = next(x for x in blobs if x["area"] == 25)
        assert (b["x"], b["y"]) == (12.5, 12.5)
        assert b["peak"] == 700.0

    def test_empty_when_no_hot_pixels(self):
        assert np_.find_ember_blobs(np.full((30, 30), 300.0, np.float32)) == []


class TestRenderDetectionInput:
    """Blob-isolated white-on-black render for the detection model."""

    def _frame(self):
        # same construction as TestEmberBlobs: 2 embers + noise + plume
        arr = np.full((60, 60), 300.0, dtype=np.float32)
        arr[10:15, 10:15] = 700.0   # 5x5 ember
        arr[40:44, 40:44] = 700.0   # 4x4 ember
        arr[0, 0] = 700.0           # 1x1 noise
        arr[20:50, 5:35] = 700.0    # plume
        return arr

    def test_shape_and_scale(self):
        rgb, scale = np_.render_detection_input(self._frame())
        assert scale == float(np_.DET_UPSCALE)
        assert rgb.shape == (120, 120, 3)
        assert rgb.dtype == np.uint8

    def test_white_embers_on_black(self):
        rgb, scale = np_.render_detection_input(self._frame())
        white = int(np.sum(rgb[..., 0] == 255))
        assert white == (25 + 16) * int(scale) * int(scale)
        assert np.all(rgb[0, 0] == 0)  # background stays black

    def test_plume_and_noise_never_rendered(self):
        # white pixels come only from the two embers, never the plume
        rgb, _ = np_.render_detection_input(self._frame(), upscale=1)
        assert int(np.sum(rgb[..., 0] == 255)) == 41

    def test_no_upscale(self):
        rgb, scale = np_.render_detection_input(self._frame(), upscale=1)
        assert scale == 1.0
        assert rgb.shape == (60, 60, 3)


class TestScaleDetections:
    """Model boxes from upscaled input space back to native pixels."""

    def test_halved_at_2x(self):
        dets = [{"x": 80.0, "y": 40.0, "width": 24.0, "height": 20.0}]
        out = np_.scale_detections(dets, 2.0)
        assert out[0] == {"x": 40.0, "y": 20.0, "width": 12.0, "height": 10.0}

    def test_identity_at_1x(self):
        dets = [{"x": 80.0, "y": 40.0, "width": 24.0, "height": 20.0}]
        assert np_.scale_detections(dets, 1.0) is dets

    def test_other_keys_preserved(self):
        dets = [{"x": 80.0, "y": 40.0, "width": 24.0, "height": 20.0,
                 "confidence": 0.7, "class": "flying tiny white particle"}]
        out = np_.scale_detections(dets, 2.0)
        assert out[0]["confidence"] == 0.7
        assert out[0]["class"] == "flying tiny white particle"


class TestAssignTracks:
    """Nearest-centroid track assignment."""

    def _det(self, x, y, conf=0.7):
        return {"x": x, "y": y, "width": 12.0, "height": 12.0, "confidence": conf}

    def _centroids(self, tracked):
        # assign_tracks returns dets with center coords in x / y
        return [{"track_id": t["track_id"], "cx": t["x"], "cy": t["y"]} for t in tracked]

    def test_new_tracks(self):
        t = np_.assign_tracks([self._det(40, 40), self._det(200, 200)], [], 1)
        assert sorted(x["track_id"] for x in t) == [1, 2]

    def test_nearby_reuses_id(self):
        first = np_.assign_tracks([self._det(40, 40)], [], 1)
        prev = self._centroids(first)
        second = np_.assign_tracks([self._det(44, 42, 0.9)], prev, 2)
        assert second[0]["track_id"] == 1

    def test_far_gets_new_id(self):
        first = np_.assign_tracks([self._det(40, 40)], [], 1)
        prev = self._centroids(first)
        second = np_.assign_tracks([self._det(900, 700, 0.9)], prev, 2)
        assert second[0]["track_id"] == 2

    def test_higher_confidence_matched_first(self):
        first = np_.assign_tracks([self._det(40, 40, 0.5)], [], 1)
        prev = self._centroids(first)
        second = np_.assign_tracks([self._det(40, 40, 0.9), self._det(41, 40, 0.6)], prev, 2)
        assert second[0]["track_id"] == 1  # best conf wins the existing id

    def test_returns_copies_with_track_id(self):
        det = self._det(40, 40)
        out = np_.assign_tracks([det], [], 1)
        assert det.get("track_id") is None  # input untouched


class TestTimelineMapping:
    """Pixel <-> frame mapping for the TimelineBar."""

    W, N = 800, 1000

    def test_roundtrip_random(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            f = int(rng.integers(0, self.N))
            x = np_.x_from_frame(f, self.W, self.N)
            # pixel columns are coarser than frames (782 px vs 1000 frames)
            assert abs(np_.frame_from_x(x, self.W, self.N) - f) <= 1

    def test_monotonic_no_reversal(self):
        prev = -1
        for f in range(0, self.N, 7):
            x = np_.x_from_frame(f, self.W, self.N)
            assert x >= prev
            prev = x

    def test_endpoints(self):
        assert np_.frame_from_x(0, self.W, self.N) == 0
        assert np_.frame_from_x(self.W - 1, self.W, self.N) == self.N - 1
        assert np_.x_from_frame(0, self.W, self.N) == np_.TIMELINE_PAD
        assert np_.x_from_frame(self.N - 1, self.W, self.N) == self.W - 1 - np_.TIMELINE_PAD

    def test_fractional_proportional(self):
        # left half of the timeline maps to the first half of frames
        assert np_.frame_from_x(0.25 * self.W, self.W, self.N) < self.N // 2

    def test_single_frame(self):
        assert np_.frame_from_x(500, self.W, 1) == 0
        assert np_.x_from_frame(0, self.W, 1) == np_.TIMELINE_PAD

    def test_clamped_out_of_range(self):
        assert np_.frame_from_x(-100, self.W, self.N) == 0
        assert np_.frame_from_x(self.W + 100, self.W, self.N) == self.N - 1


class TestHottestStride:
    def test_throttled(self):
        assert np_.hottest_stride(16976) == 16
        assert np_.hottest_stride(1000) == 1

    def test_min_one(self):
        assert np_.hottest_stride(5) == 1
        assert np_.hottest_stride(1) == 1


class TestTrackColor:
    def test_cycles_through_palette(self):
        a = np_.track_color({"track_id": 0})
        b = np_.track_color({"track_id": len(np_.TRACK_COLORS)})
        assert a == b
        assert len(a) == 3

    def test_default_when_missing(self):
        assert np_.track_color({}) == np_.TRACK_COLORS[0]


class TestOverlays:
    """Detection overlay + HUD drawing are in-place and shape-preserving."""

    def _canvas(self, w=640, h=480):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_overlay_keeps_shape(self):
        vis = self._canvas()
        before = vis.copy()
        det = {"x": 40.0, "y": 118.0, "width": 12.0, "height": 12.0,
               "confidence": 0.77, "track_id": 1}
        out = np_.draw_detection_overlay(vis, det)
        assert out.shape == vis.shape
        assert not np.array_equal(out, before)  # something was drawn

    def test_small_box_padded_to_min_visible(self):
        vis = self._canvas()
        det = {"x": 40.0, "y": 118.0, "width": 2.0, "height": 2.0, "track_id": 1}
        np_.draw_detection_overlay(vis, det)
        # a padded box must have painted pixels beyond the 2x2 box
        assert not np.array_equal(vis, self._canvas())

    def test_hud_empty_unchanged(self):
        vis = self._canvas()
        assert np_.draw_hud(vis, []) is vis

    def test_hud_badge_drawn(self):
        vis = self._canvas()
        dets = [{"confidence": 0.5, "track_id": 1}, {"confidence": 0.9, "track_id": 2}]
        np_.draw_hud(vis, dets)
        assert not np.array_equal(vis, self._canvas())