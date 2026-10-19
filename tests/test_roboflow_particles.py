# -*- coding: utf-8 -*-
"""
Tests for the Roboflow workflow integration (roboflow_particles).

Covers input coercion, base64 decode-to-disk, polygon-point stripping, and the
generic prediction-list walker.  The live smoke test calls the real serverless
workflow and is skipped unless ``ROBOFLOW_API_KEY`` is set in the environment.
"""
import base64
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Insert project root so we can import the module directly.
# ---------------------------------------------------------------------------
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import roboflow_particles as rp  # noqa: E402

SAMPLE_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "particles_smoke.jpg")


# ===================================================================
# Input coercion
# ===================================================================

class TestCoerceInput:
    """Input normalisation into Roboflow {type, value} refs."""

    def test_file_path_becomes_base64(self):
        result = rp._coerce_input(SAMPLE_IMAGE)
        assert result["type"] == "base64"
        blob = base64.b64decode(result["value"])
        assert blob[:3] == b"\xff\xd8\xff"  # JPEG magic

    def test_https_url_passthrough(self):
        url = "https://example.com/sparks.jpg"
        assert rp._coerce_input(url) == {"type": "url", "value": url}

    def test_http_url_rejected(self):
        with pytest.raises(rp.RoboflowConfigError):
            rp._coerce_input("http://example.com/sparks.jpg")

    def test_bytes_base64_encoded(self):
        raw = b"\xff\xd8\xff fakejpeg"
        result = rp._coerce_input(raw)
        assert result["type"] == "base64"
        assert base64.b64decode(result["value"]) == raw

    def test_prebuilt_base64_dict_passthrough(self):
        pair = {"type": "base64", "value": "AAAA"}
        assert rp._coerce_input(pair) == pair

    def test_invalid_type_rejected(self):
        with pytest.raises(rp.RoboflowConfigError):
            rp._coerce_input({"type": "path", "value": "/tmp/x.jpg"})


# ===================================================================
# Base64 image materialisation + points stripping
# ===================================================================

class TestMaterialise:
    """Decode workflow images to disk and drop raw polygon points."""

    def test_base64_image_written_to_disk(self, tmp_path):
        png = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
        payload = {"viz": {"type": "base64", "value": base64.b64encode(png).decode()}}
        images = []
        out = rp._materialise(payload, str(tmp_path), "outputs[0]", images)
        assert isinstance(out["viz"], str)
        assert os.path.exists(out["viz"])
        assert images and images[0].output_key.startswith("outputs[0]")
        with open(out["viz"], "rb") as fh:
            assert fh.read() == png

    def test_polygon_points_stripped(self, tmp_path):
        entry = {"raw": {"class": "particle", "points": [[1.0, 2.0], [3.0, 4.0]]}}
        out = rp._materialise(entry, str(tmp_path), "outputs[0]", [])
        assert "points" not in out["raw"]

    def test_non_coordinate_points_kept(self, tmp_path):
        payload = {"raw": {"points": "not-a-polygon"}}
        out = rp._materialise(payload, str(tmp_path), "outputs[0]", [])
        assert out["raw"]["points"] == "not-a-polygon"


# ===================================================================
# Response walker (generic over workflow output names)
# ===================================================================

class TestIterPredictionLists:
    """Match the grounded live response shape generically."""

    def test_grounded_empty_shape(self):
        grounded = [{"predictions": {"image": {"width": None, "height": None},
                                     "predictions": []}}]
        found = list(rp.iter_prediction_lists(grounded))
        assert len(found) == 1
        idx, key, list_ = found[0]
        assert idx == 0 and key == "predictions" and list_ == []

    def test_nested_nonempty_predictions(self):
        entry = [{"predictions": {"predictions": [{"class_name": "particle"}]}}]
        found = list(rp.iter_prediction_lists(entry))
        assert len(found) == 1
        assert found[0][2] == [{"class_name": "particle"}]
        assert rp.detection_count(entry) == 1

    def test_multiple_entries_and_outputs(self):
        entries = [
            {"a": {"predictions": [{"id": 1}]}},
            {"b": {"predictions": [{"id": 2}, {"id": 3}]}},
        ]
        found = list(rp.iter_prediction_lists(entries))
        assert [len(l) for _, _, l in found] == [1, 2]
        assert rp.detection_count(entries) == 3

    def test_unrecognised_entry_no_crash(self):
        assert rp.detection_count([{"weird": [1, 2, 3]}]) == 0


# ===================================================================
# Live smoke test
# ===================================================================

class TestLiveWorkflow:
    """Calls the real serverless endpoint (needs ROBOFLOW_API_KEY)."""

    def test_live_returns_expected_output_shape(self, tmp_path):
        if not os.environ.get("ROBOFLOW_API_KEY"):
            pytest.skip("ROBOFLOW_API_KEY not set; skipping live workflow test")

        result = rp.run_workflow(SAMPLE_IMAGE, images_dir=str(tmp_path), timeout=90)

        # one entry per input image; each is a non-empty dict keyed by the
        # workflow's own output names
        assert isinstance(result.entries, list) and result.entries
        entry = result.entry(0)
        assert isinstance(entry, dict) and entry

        # the grounded workflow output key ('predictions') must be present and
        # its detection list iterable
        lists = list(rp.iter_prediction_lists(result.entries))
        assert lists, "expected at least one prediction list in the live output"
        for idx, key, list_ in lists:
            assert isinstance(list_, list)
        assert rp.detection_count(result.entries) >= 0

        # decoded image outputs, if any, must live on disk, not in memory
        for img in result.images:
            assert os.path.exists(img.path)