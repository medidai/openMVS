#!/usr/bin/env python3
"""Deterministic request contract for depth-map drill-down captures."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import yaml


SCHEMA_NAME = "openmvs.dmap.drilldown_request"
SCHEMA_VERSION = 2
INDEX_SCHEMA_NAME = "openmvs.dmap.drilldown_index"
DEFAULT_MAX_TRACE_PIXELS = 4096


def parse_pixel(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"invalid pixel {value!r}; expected X,Y")
    try:
        x, y = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid pixel {value!r}; X and Y must be integers") from exc
    if x < 0 or y < 0:
        raise ValueError(f"invalid pixel {value!r}; coordinates must be non-negative")
    return x, y


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(f"invalid ROI {value!r}; expected X,Y,W,H")
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"invalid ROI {value!r}; all values must be integers") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"invalid ROI {value!r}; X/Y must be non-negative and W/H must be positive"
        )
    return x, y, width, height


def normalize_pixels(values: Iterable[str]) -> list[dict[str, int]]:
    return [
        {"x": x, "y": y}
        for x, y in sorted({parse_pixel(value) for value in values}, key=lambda point: (point[1], point[0]))
    ]


def expand_trace_pixels(request: dict[str, Any]) -> list[dict[str, int]]:
    target = request.get("target") or {}
    pixels = target.get("pixels") or []
    if pixels:
        return [{"x": int(pixel["x"]), "y": int(pixel["y"])} for pixel in pixels]
    roi = target.get("roi")
    if not roi:
        return []
    x = int(roi["x"])
    y = int(roi["y"])
    width = int(roi["width"])
    height = int(roi["height"])
    return [
        {"x": px, "y": py}
        for py in range(y, y + height)
        for px in range(x, x + width)
    ]


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def request_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_runs(config: dict[str, Any], variants: Iterable[str]) -> list[dict[str, Any]]:
    run_specs = [run for run in config.get("runs") or [] if isinstance(run, dict)]
    baselines = [run for run in run_specs if str(run.get("role", "variant")) == "baseline"]
    if len(baselines) != 1:
        raise ValueError("drill-down requires exactly one run with role=baseline")
    baseline = baselines[0]
    by_label = {str(run.get("label", "")): run for run in run_specs}
    requested = list(dict.fromkeys(str(value) for value in variants))
    if requested:
        unknown = sorted(set(requested) - set(by_label))
        if unknown:
            raise ValueError(f"unknown drill-down variant(s): {', '.join(unknown)}")
        if str(baseline["label"]) in requested:
            raise ValueError("--variant selects variants only; the baseline is included automatically")
        selected_variants = [by_label[label] for label in requested]
    else:
        selected_variants = [run for run in run_specs if run is not baseline]
    if not selected_variants:
        raise ValueError("drill-down requires at least one variant run")
    selected_variants.sort(key=lambda run: str(run["label"]))
    selected = [baseline, *selected_variants]
    return [
        {
            "label": str(run["label"]),
            "role": "baseline" if run is baseline else "variant",
            "densify_args": [str(value) for value in run.get("densify_args") or []],
            "ini_overrides": {
                str(key): str(value)
                for key, value in (run.get("ini_overrides") or {}).items()
            },
        }
        for run in selected
    ]


def build_request(
    config: dict[str, Any],
    config_path: Path,
    scene_id: str,
    image_id: int,
    pixel_values: Iterable[str] = (),
    roi_value: str | None = None,
    variants: Iterable[str] = (),
    source_revision: str | None = None,
    source_dirty: bool | None = None,
) -> dict[str, Any]:
    if not scene_id.strip():
        raise ValueError("scene ID must not be empty")
    if image_id < 0:
        raise ValueError("frame/image ID must be non-negative")
    pixels = normalize_pixels(pixel_values)
    if pixels and roi_value:
        raise ValueError("select pixels or an ROI, not both")
    roi = None
    if roi_value:
        x, y, width, height = parse_roi(roi_value)
        roi = {"x": x, "y": y, "width": width, "height": height}
    trace_count = len(pixels) if pixels else (roi["width"] * roi["height"] if roi else 0)
    max_trace_pixels = int(
        (config.get("instrumentation") or {}).get(
            "max_trace_pixels_per_request", DEFAULT_MAX_TRACE_PIXELS
        )
    )
    if max_trace_pixels <= 0:
        raise ValueError("instrumentation.max_trace_pixels_per_request must be positive")
    if trace_count > max_trace_pixels:
        raise ValueError(
            f"trace request selects {trace_count} pixels, exceeding the configured limit "
            f"of {max_trace_pixels}"
        )
    instrumentation = config.get("instrumentation") or {}
    extents_by_scene = instrumentation.get("expected_extent_by_scene")
    if extents_by_scene is None:
        extents_by_scene = {}
    if not isinstance(extents_by_scene, dict):
        raise ValueError("instrumentation.expected_extent_by_scene must be a mapping")
    scene_extent = extents_by_scene.get(scene_id)
    if scene_extent is None:
        scene_extent = {}
    if not isinstance(scene_extent, dict):
        raise ValueError(
            f"instrumentation.expected_extent_by_scene[{scene_id!r}] must be a mapping"
        )
    expected_width = int(
        scene_extent.get("width", instrumentation.get("expected_width", 0)) or 0
    )
    expected_height = int(
        scene_extent.get("height", instrumentation.get("expected_height", 0)) or 0
    )
    if expected_width > 0 and expected_height > 0:
        outside = [
            pixel for pixel in pixels
            if pixel["x"] >= expected_width or pixel["y"] >= expected_height
        ]
        if outside:
            raise ValueError(
                f"trace pixel ({outside[0]['x']},{outside[0]['y']}) is outside the configured "
                f"{expected_width}x{expected_height} depth-map extent"
            )
        if roi and (
            roi["x"] + roi["width"] > expected_width
            or roi["y"] + roi["height"] > expected_height
        ):
            raise ValueError(
                f"trace ROI extends outside the configured {expected_width}x{expected_height} "
                "depth-map extent"
            )
    profile = "trace" if trace_count else "deep"
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "experiment_id": str(config.get("experiment_id", "")),
            "config_name": config_path.name,
            "config_sha256": file_digest(config_path),
            "source_revision": source_revision,
            "source_dirty": source_dirty,
        },
        "capture_profile": profile,
        "target": {
            "scene_id": scene_id,
            "image_id": image_id,
            "pixels": pixels,
            "roi": roi,
            "trace_pixel_count": trace_count,
        },
        "runs": select_runs(config, variants),
        "capture": {
            "instrumentation_level": "maps",
            "write_maps": True,
            "patch_match_cuda_instances": 1,
            "process_specialization": "Process<true>",
            "compact_exact_trace_available": False,
            "storage_policy": "full-frame exact maps plus selected trace rows",
        },
    }
    payload["request_sha256"] = request_digest(payload)
    return payload


def validate_request(request: dict[str, Any]) -> None:
    if request.get("schema_name") != SCHEMA_NAME or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported drill-down request schema")
    expected = str(request.get("request_sha256", ""))
    payload = dict(request)
    payload.pop("request_sha256", None)
    actual = request_digest(payload)
    if expected != actual:
        raise ValueError(f"drill-down request digest mismatch: expected {expected}, computed {actual}")
    profile = request.get("capture_profile")
    if profile not in {"deep", "trace"}:
        raise ValueError(f"unsupported drill-down capture profile: {profile!r}")
    pixels = expand_trace_pixels(request)
    if profile == "deep" and pixels:
        raise ValueError("deep request must not contain trace pixels")
    if profile == "trace" and not pixels:
        raise ValueError("trace request must contain pixels or an ROI")
    capture = request.get("capture") or {}
    if (
        capture.get("instrumentation_level") != "maps"
        or capture.get("write_maps") is not True
        or capture.get("process_specialization") != "Process<true>"
        or capture.get("compact_exact_trace_available") is not False
    ):
        raise ValueError(
            "drill-down capture must use full-frame Process<true> maps; "
            "compact exact trace is unavailable"
        )


def write_immutable_request(drilldown_root: Path, request: dict[str, Any]) -> Path:
    validate_request(request)
    request_dir = drilldown_root / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    path = request_dir / f"{request['request_sha256']}.yaml"
    serialized = yaml.safe_dump(request, sort_keys=False)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing != request:
            raise RuntimeError(f"immutable drill-down request is corrupt or conflicting: {path}")
        return path
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=request_dir, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_request(path: Path) -> dict[str, Any]:
    request = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(request, dict):
        raise ValueError(f"drill-down request is not an object: {path}")
    validate_request(request)
    if path.stem != request["request_sha256"]:
        raise ValueError(f"drill-down request filename does not match its digest: {path}")
    return request
