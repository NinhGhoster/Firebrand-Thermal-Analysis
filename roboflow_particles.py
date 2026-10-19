# -*- coding: utf-8 -*-
"""
Roboflow workflow integration - "Find flying tiny white particles".

Runs the serverless Roboflow workflow that detects flying tiny white
particles in visual (non-radiometric) imagery using an RF-DETR 2xLarge
model.  All HTTP is done with the standard library only (urllib), following
the same conventions as the rest of this repository.

Run endpoint
------------
POST https://serverless.roboflow.com/{workspace_name}/workflows/{workflow_id}

The workflow lives under the ``firebrand`` workspace:

    WORKSPACE_NAME = "firebrand"
    WORKFLOW_ID    = "find-flying-tiny-white-particles-vfind-flying-tiny-white-particles-2-rfdetr-2xlarge-t1-logic"

Authentication
--------------
The Roboflow API key is read from the ``ROBOFLOW_API_KEY`` environment
variable (never hardcoded).  Get it at https://app.roboflow.com/settings/api.

Response shape (grounded against the live endpoint)
----------------------------------------------------
The API returns JSON like::

    {"outputs": [ <one entry per input image> ], "profiler_trace": []}

Each entry is a dict keyed by the workflow's own output names.  For this
workflow a typical entry looks like::

    {"predictions": {"image": {"width": null, "height": null},
                     "predictions": []}}

Image-shaped outputs come back as ``{"type": "base64", "value": "..."}``
blobs; they are decoded and written to disk (never logged, never held in
memory), and raw polygon ``points`` arrays are stripped to keep payloads small.
"""
import base64
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://serverless.roboflow.com"
WORKSPACE_NAME = "firebrand"
WORKFLOW_ID = (
    "find-flying-tiny-white-particles-vfind-flying-tiny-white-particles-2"
    "-rfdetr-2xlarge-t1-logic"
)
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_IMAGES_DIR = "roboflow_images"
_ENV_API_KEY = "ROBOFLOW_API_KEY"

# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class RoboflowWorkflowError(Exception):
    """Base class for Roboflow workflow integration failures."""


class RoboflowConfigError(RoboflowWorkflowError):
    """Bad input, wrong parameters, or missing/invalid configuration."""


class RoboflowAuthError(RoboflowWorkflowError):
    """The API key is missing, revoked, or rejected (HTTP 401/403)."""


class RoboflowTimeoutError(RoboflowWorkflowError):
    """The request did not succeed within the configured retry/timeout budget."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class WorkflowImage:
    """A decoded workflow image output written to disk."""

    output_key: str  # dotted path of the output key it came from
    path: str


@dataclasses.dataclass
class WorkflowRunResult:
    """Parsed result of a Roboflow workflow run.

    ``entries`` mirrors the workflow's own output structure: one dict per
    input image, keyed by the workflow's declared output names.  Image-shaped
    outputs are already materialised to disk and polygon ``points`` stripped.
    """

    entries: List[Dict[str, Any]]
    images: List[WorkflowImage]
    predict_url: str

    def entry(self, index: int = 0) -> Dict[str, Any]:
        """Return the raw output entry for the ``index``-th input image."""
        try:
            return self.entries[index]
        except IndexError:
            raise IndexError(
                f"no entry for input image index {index} of {len(self.entries)}"
            )


def iter_prediction_lists(entries: List[Dict[str, Any]]
                          ) -> Iterator[Tuple[int, str, List[Dict[str, Any]]]]:
    """Yield ``(entry_index, output_key, prediction_list)`` tuples.

    Deliberately generic: finds every output whose value is a dict containing a
    ``predictions`` list, matching the shape the live endpoint returns, without
    hardcoding a single output name.
    """
    for idx, entry in enumerate(entries):
        yield from _iter_prediction_lists(entry, key="output", index=idx)


def detection_count(entries: List[Dict[str, Any]]) -> int:
    """Total number of detected objects across all entries."""
    return sum(len(preds) for _, _, preds in iter_prediction_lists(entries))


def _iter_prediction_lists(node: Any, key: str, index: int):
    if isinstance(node, dict):
        preds = node.get("predictions")
        if isinstance(preds, list):
            yield (index, key, preds)
        for subkey, subval in node.items():
            yield from _iter_prediction_lists(subval, subkey, index)
    elif isinstance(node, list):
        for subval in node:
            yield from _iter_prediction_lists(subval, key, index)


# ---------------------------------------------------------------------------
# Low-level request helper
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: Dict[str, Any], timeout: int, max_retries: int) -> Dict[str, Any]:
    """POST JSON and decode the response, retrying transient failures."""
    body = json.dumps(payload).encode("utf-8")
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        if attempt > 0:  # exponential backoff between attempts
            time.sleep(2 ** attempt)
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Firebrand-Thermal-Analysis",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                decoded = resp.read().decode("utf-8")
                return json.loads(decoded)
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status in (401, 403):
                raise RoboflowAuthError(
                    f"Roboflow rejected the API key (HTTP {status}). "
                    "Check ROBOFLOW_API_KEY at https://app.roboflow.com/settings/api"
                ) from exc
            if status in (429, 500, 502, 503, 504):  # transient - retry
                last_exc = exc
                continue
            raise RoboflowWorkflowError(
                f"Roboflow workflow request failed (HTTP {status}). "
                f"Body: {_safe_body(exc)}"
            ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            last_exc = exc  # transport / timeout - retry
            continue
        except json.JSONDecodeError as exc:
            raise RoboflowWorkflowError("Roboflow returned non-JSON response") from exc
    raise RoboflowTimeoutError(
        f"Roboflow workflow request did not succeed after {max_retries} attempt(s): {last_exc}"
    )


def _safe_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read(1024).decode("utf-8", "replace").strip()[:500]
    except Exception:  # pragma: no cover - defensive
        return "<unreadable body>"


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

ImageInput = Union[str, bytes, bytearray, Dict[str, str]]


def _coerce_input(image: ImageInput) -> Dict[str, str]:
    """Normalise an image reference into a Roboflow ``{"type", "value"}`` input.

    Accepts a file path, an ``https://`` URL, raw bytes (base64-encoded), or a
    pre-built input dict.  Plain ``http://`` URLs are rejected by Roboflow and
    raise ``RoboflowConfigError``.
    """
    if isinstance(image, dict):
        img_type = image.get("type")
        value = image.get("value")
        if img_type == "base64" and isinstance(value, str) and value:
            return image
        if img_type == "url" and isinstance(value, str) and value:
            if not value.startswith("https://"):
                raise RoboflowConfigError(
                    "Roboflow URL inputs must be https:// - plain http:// is rejected"
                )
            return image
        raise RoboflowConfigError(
            "image dict must have type 'url' or 'base64' and a non-empty 'value'"
        )
    if isinstance(image, (bytes, bytearray)):
        return {"type": "base64", "value": base64.b64encode(bytes(image)).decode("ascii")}
    if isinstance(image, str):
        if image.startswith("https://"):
            return {"type": "url", "value": image}
        if image.startswith("http://"):
            raise RoboflowConfigError(
                "Roboflow URL inputs must be https:// - plain http:// rejected: " + image
            )
        try:
            with open(os.path.expanduser(image), "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise RoboflowConfigError(f"could not read image file '{image}': {exc}") from exc
        return {"type": "base64", "value": base64.b64encode(data).decode("ascii")}
    raise RoboflowConfigError(
        "image must be a file path, an https:// URL, bytes, or a {'type','value'} dict"
    )


# ---------------------------------------------------------------------------
# Base64 image materialisation
# ---------------------------------------------------------------------------


def _image_extension(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".bin"


def _has_polygon_points(value: Any) -> bool:
    """True when ``value`` looks like a polygon/coordinate list."""
    if not isinstance(value, list) or not value:
        return False
    return all(
        isinstance(pt, list)
        and len(pt) >= 2
        and all(isinstance(c, (int, float)) for c in pt)
        for pt in value
    )


def _materialise(node: Any, out_dir: str, key_path: str, images: List[WorkflowImage]) -> Any:
    """Recursively decode base64 image blobs to disk and drop polygon points."""
    if isinstance(node, dict):
        result: Dict[str, Any] = {}
        for key, value in node.items():
            child_path = f"{key_path}.{key}" if key_path else key
            if key == "points" and _has_polygon_points(value):
                continue
            if isinstance(value, dict) and value.get("type") == "base64" and isinstance(value.get("value"), str):
                blob = base64.b64decode(value["value"])
                os.makedirs(out_dir, exist_ok=True)
                safe = "".join(ch if ch.isalnum() else "_" for ch in key) or "image"
                path = _unique_path(os.path.join(out_dir, f"{safe}{_image_extension(blob)}"))
                with open(path, "wb") as fh:
                    fh.write(blob)
                del blob
                result[key] = path
                images.append(WorkflowImage(output_key=child_path, path=path))
                continue
            result[key] = _materialise(value, out_dir, child_path, images)
        return result
    if isinstance(node, list):
        return [
            _materialise(item, out_dir, f"{key_path}[{i}]", images) for i, item in enumerate(node)
        ]
    return node


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for n in range(2, 1000):
        candidate = f"{stem}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get(_ENV_API_KEY, "").strip()
    if not key:
        raise RoboflowConfigError(
            "ROBOFLOW_API_KEY is not set. Export it first, e.g.:\n"
            "    export ROBOFLOW_API_KEY=<your key from app.roboflow.com/settings/api>"
        )
    return key


def run_workflow(
    image: ImageInput,
    *,
    api_url: str = DEFAULT_API_URL,
    parameters: Optional[Dict[str, Any]] = None,
    images_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    api_key: Optional[str] = None,
) -> WorkflowRunResult:
    """Run the "Find flying tiny white particles" workflow on one image.

    Args:
        image: file path, ``https://`` URL, raw bytes, or a ``{"type","value"}``
            dict (see ``_coerce_roboflow_input``).
        parameters: workflow parameters, matching the workflow's declared
            names/types exactly (e.g. ``{"confidence": 0.5}``).
        images_dir: directory for decoded image outputs (default:
            ``roboflow_images`` under CWD).
        timeout: per-request timeout in seconds (default 60).
        max_retries: total attempts with exponential backoff (default 3).
        api_key: override; defaults to ``$ROBOFLOW_API_KEY``.

    Returns:
        A ``WorkflowRunResult``.  Raises a ``RoboflowWorkflowError`` subclass
        on failure.
    """
    if timeout is None:
        timeout = int(os.environ.get("ROBOFLOW_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
    if max_retries is None:
        max_retries = int(os.environ.get("ROBOFLOW_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    if images_dir is None:
        images_dir = os.environ.get("ROBOFLOW_IMAGES_DIR", DEFAULT_IMAGES_DIR)
    key = api_key if api_key else _api_key()

    payload: Dict[str, Any] = {
        "api_key": key,
        "inputs": {"image": _coerce_input(image)},
    }
    if parameters:
        payload["parameters"] = parameters

    url = f"{api_url.rstrip('/')}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}"
    response = _post_json(url, payload, timeout=timeout, max_retries=max_retries)

    raw_entries = response.get("outputs")
    if not isinstance(raw_entries, list):
        raise RoboflowWorkflowError("unexpected Roboflow response: missing 'outputs' list")

    images: List[WorkflowImage] = []
    materialised_entries: List[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        materialised_entries.append(_materialise(entry, images_dir, f"outputs[{idx}]", images))
    return WorkflowRunResult(entries=materialised_entries, images=images, predict_url=url)


def find_flying_particles(
    image: ImageInput,
    *,
    images_dir: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Tuple[WorkflowRunResult, List[Any]]:
    """Convenience wrapper returning ``(result, flattened_detections)``.

    ``flattened_detections`` is the concatenation of every ``predictions``
    list found across the response entries (raw detections for each image,
    in order).  If the workflow ever emits other output names, they remain
    available in ``result.entries`` untouched.
    """
    result = run_workflow(
        image,
        images_dir=images_dir,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )
    flattened: List[Dict[str, Any]] = []
    for _, _, preds in iter_prediction_lists(result.entries):
        flattened.extend(preds)
    return result, flattened