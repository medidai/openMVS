#!/usr/bin/env python3
"""Deterministic request contract for depth-map drill-down captures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Iterable

import yaml


SCHEMA_NAME = "openmvs.dmap.drilldown_request"
SCHEMA_VERSION = 2
INDEX_SCHEMA_NAME = "openmvs.dmap.drilldown_index"
DEFAULT_MAX_TRACE_PIXELS = 4096
MAX_TRACE_REPORT_ROWS = 4096
OBSERVER_APP_NAME = "DensifyPointCloudDMapObserve"
# Above this value, converting a signed 32-bit coordinate to float32 can round
# to 2^31 and make C++'s subsequent float-to-int conversion undefined.
MAX_SAFE_TRACE_COORDINATE = (1 << 31) - 65


@dataclass(frozen=True)
class TracePyramidSlot:
    """One compact CUDA trace slot and the request pixels represented by it."""

    request_indices: tuple[int, ...]
    requested_coordinates: tuple[tuple[int, int], ...]
    coordinate: tuple[int, int]


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
    if x > MAX_SAFE_TRACE_COORDINATE or y > MAX_SAFE_TRACE_COORDINATE:
        raise ValueError(
            f"invalid pixel {value!r}; coordinates exceed the safe C++ float32-to-int limit"
        )
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
    if (
        x > MAX_SAFE_TRACE_COORDINATE
        or y > MAX_SAFE_TRACE_COORDINATE
        or width - 1 > MAX_SAFE_TRACE_COORDINATE - x
        or height - 1 > MAX_SAFE_TRACE_COORDINATE - y
    ):
        raise ValueError(
            f"invalid ROI {value!r}; coordinates exceed the safe C++ float32-to-int limit"
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
        if not isinstance(pixels, list) or len(pixels) > DEFAULT_MAX_TRACE_PIXELS:
            raise ValueError(
                f"trace request pixels must be a list with at most {DEFAULT_MAX_TRACE_PIXELS} entries"
            )
        result: list[dict[str, int]] = []
        for pixel in pixels:
            if not isinstance(pixel, dict):
                raise ValueError("trace request pixels must contain mappings")
            x, y = parse_pixel(f"{pixel.get('x')},{pixel.get('y')}")
            result.append({"x": x, "y": y})
        return result
    roi = target.get("roi")
    if not roi:
        return []
    if not isinstance(roi, dict):
        raise ValueError("trace request ROI must be a mapping")
    x, y, width, height = parse_roi(
        f"{roi.get('x')},{roi.get('y')},{roi.get('width')},{roi.get('height')}"
    )
    if width * height > DEFAULT_MAX_TRACE_PIXELS:
        raise ValueError(
            f"trace request ROI contains more than {DEFAULT_MAX_TRACE_PIXELS} pixels"
        )
    return [
        {"x": px, "y": py}
        for py in range(y, y + height)
        for px in range(x, x + width)
    ]


def _scaled_trace_coordinate(coordinate: int, pyramid_level: int) -> int:
    """Match ROUND2INT((float)coordinate * float32(2^-level))."""

    if coordinate > MAX_SAFE_TRACE_COORDINATE:
        raise ValueError(
            "trace coordinate exceeds the largest value with a safe C++ "
            "float32-to-int conversion"
        )

    def float32(value: float) -> float:
        return struct.unpack("<f", struct.pack("<f", value))[0]

    # SelectInstrumentTracePixels casts the integer to float, multiplies by a
    # float scale, then Round2Int(float) adds 0.5f before floor. Preserve every
    # float32 rounding point, including the non-obvious behavior above 2^24.
    coordinate_float = float32(float(coordinate))
    scale = float32(math.ldexp(1.0, -pyramid_level))
    scaled = float32(coordinate_float * scale)
    return math.floor(float32(scaled + float32(0.5)))


def trace_pyramid_layout(
    requested_coordinates: Iterable[tuple[int, int]],
    pyramid_level: int,
    *,
    width: int | None = None,
    height: int | None = None,
) -> list[TracePyramidSlot]:
    """Reconstruct PatchMatch's per-level trace slots in stable request order."""

    if (
        isinstance(pyramid_level, bool)
        or not isinstance(pyramid_level, int)
        or not 0 <= pyramid_level <= 30
    ):
        raise ValueError("pyramid_level must be an integer in [0,30]")
    if (width is None) != (height is None):
        raise ValueError("trace extent requires both width and height")
    if width is not None and (
        isinstance(width, bool) or isinstance(height, bool)
        or not isinstance(width, int) or not isinstance(height, int)
        or width <= 0 or height <= 0
    ):
        raise ValueError("trace extent width and height must be positive integers")

    slot_by_coordinate: dict[tuple[int, int], int] = {}
    request_indices: list[list[int]] = []
    aliases: list[list[tuple[int, int]]] = []
    selected_coordinates: list[tuple[int, int]] = []
    for request_index, requested in enumerate(requested_coordinates):
        if (
            len(requested) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in requested)
        ):
            raise ValueError("trace coordinates must contain two non-negative integers")
        coordinate = (
            _scaled_trace_coordinate(requested[0], pyramid_level),
            _scaled_trace_coordinate(requested[1], pyramid_level),
        )
        if width is not None and not (
            0 <= coordinate[0] < width and 0 <= coordinate[1] < height
        ):
            continue
        slot_index = slot_by_coordinate.get(coordinate)
        if slot_index is not None:
            request_indices[slot_index].append(request_index)
            aliases[slot_index].append(requested)
            continue
        slot_by_coordinate[coordinate] = len(selected_coordinates)
        request_indices.append([request_index])
        aliases.append([requested])
        selected_coordinates.append(coordinate)
    return [
        TracePyramidSlot(tuple(indices), tuple(coordinates), coordinate)
        for indices, coordinates, coordinate in zip(
            request_indices, aliases, selected_coordinates, strict=True
        )
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


def validate_request_runs(
    config: dict[str, Any], request: dict[str, Any]
) -> list[dict[str, Any]]:
    """Bind immutable request run specs to the current experiment config."""

    requested_runs = request.get("runs")
    if not isinstance(requested_runs, list) or not requested_runs:
        raise ValueError("drill-down request runs must be a non-empty list")
    variant_labels: list[str] = []
    for index, run in enumerate(requested_runs):
        if not isinstance(run, dict):
            raise ValueError(f"drill-down request runs[{index}] must be a mapping")
        role = run.get("role")
        if role == "variant":
            variant_labels.append(str(run.get("label", "")))
        elif role != "baseline":
            raise ValueError(
                f"drill-down request run {run.get('label', index)!r} has invalid role {role!r}"
            )
    expected = select_runs(config, variant_labels)
    if requested_runs != expected:
        raise ValueError(
            "drill-down request run specs do not match the selected runs in the current config"
        )
    return expected


def _validated_argument_overrides(raw: Any, owner: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"argument_overrides for {owner} must be a mapping")
    result: dict[str, str] = {}
    for raw_option, raw_value in raw.items():
        option = str(raw_option)
        value = str(raw_value)
        if re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9-]*", option) is None:
            raise ValueError(f"invalid argument override for {owner}: {option!r}")
        if any(character in value for character in "\r\n"):
            raise ValueError(f"invalid argument override value for {owner}: {option!r}")
        result[option] = value
    return result


def _selected_scene_config(
    config: dict[str, Any], scene_id: str
) -> dict[str, Any] | None:
    scenes = config.get("scenes") or []
    if not isinstance(scenes, list):
        raise ValueError("scenes must be a list")
    if any(not isinstance(scene, dict) for scene in scenes):
        raise ValueError("scenes entries must be mappings")
    matches = [
        scene for scene in scenes
        if str(scene.get("scan_id", "")) == scene_id
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate scene scan_id {scene_id!r}")
    if not matches:
        return None
    return matches[0]


def selected_scene_argument_overrides(
    config: dict[str, Any], scene_id: str
) -> dict[str, str]:
    """Return configured overrides for a scene, if it has an explicit row."""

    scene = _selected_scene_config(config, scene_id)
    if scene is None:
        return {}
    return _validated_argument_overrides(
        scene.get("argument_overrides"), f"scene {scene_id!r}"
    )


def validate_no_implicit_program_options_file(
    working_folder: Path,
    app_name: str = OBSERVER_APP_NAME,
) -> Path:
    """Reject the implicit APPNAME.cfg input that bypasses trace admission."""

    if re.fullmatch(r"[A-Za-z0-9_.-]+", app_name) is None:
        raise ValueError(f"invalid observer application name: {app_name!r}")
    candidate = working_folder.expanduser() / f"{app_name}.cfg"
    if candidate.exists() or candidate.is_symlink():
        raise ValueError(
            "trace topology admission refuses the implicit program-options file "
            f"loaded by {app_name}: {candidate}"
        )
    return candidate


def _replace_argument_value(
    arguments: list[str], name: str, value: str
) -> list[str]:
    result: list[str] = []
    index = 0
    replaced = False
    while index < len(arguments):
        item = arguments[index]
        if item == name:
            result.extend([name, value])
            index += 2
            replaced = True
            continue
        if item.startswith(name + "="):
            result.extend([name, value])
            index += 1
            replaced = True
            continue
        result.append(item)
        index += 1
    if not replaced:
        result.extend([name, value])
    return result


def _reject_external_config_file(arguments: Iterable[Any], owner: str) -> None:
    for raw in arguments:
        argument = str(raw)
        if (
            argument in {"--config-file", "-c"}
            or argument.startswith("--config-file=")
            or argument.startswith("-c")
        ):
            raise ValueError(
                f"{owner} uses --config-file; trace topology admission cannot "
                "safely derive options from an external program-options file"
            )


def effective_trace_arguments(
    config: dict[str, Any],
    run: dict[str, Any],
    argument_overrides: dict[str, Any] | None = None,
) -> list[str]:
    """Resolve arguments that determine trace rows for one run and scene."""

    default_arguments = [str(value) for value in config.get("default_densify_args") or []]
    run_arguments = [str(value) for value in run.get("densify_args") or []]
    _reject_external_config_file(default_arguments, "default_densify_args")
    _reject_external_config_file(
        run_arguments, f"run {run.get('label', '<unknown>')!r} densify_args"
    )
    overrides = _validated_argument_overrides(
        argument_overrides,
        "selected scene",
    )
    if "--config-file" in overrides:
        raise ValueError(
            "selected scene argument_overrides uses --config-file; trace topology "
            "admission cannot safely derive options from an external program-options file"
        )
    arguments = [*default_arguments, *run_arguments]
    for option, value in overrides.items():
        arguments = _replace_argument_value(arguments, option, value)
    return arguments


def _nonnegative_argument(
    arguments: Iterable[Any], name: str, default: int
) -> int:
    values = [str(value) for value in arguments]
    result = default
    index = 0
    while index < len(values):
        value = values[index]
        if value == name:
            if index + 1 >= len(values):
                raise ValueError(f"{name} requires an integer value")
            raw = values[index + 1]
            index += 2
        elif value.startswith(name + "="):
            raw = value.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        try:
            result = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} requires a non-negative integer") from exc
        if result < 0:
            raise ValueError(f"{name} requires a non-negative integer")
    return result


def trace_row_upper_bound(
    config: dict[str, Any],
    runs: Iterable[dict[str, Any]],
    trace_pixels: int,
    *,
    argument_overrides: dict[str, Any] | None = None,
) -> int:
    """Conservative report-row bound across requested runs and DMAP stages."""

    if trace_pixels < 0:
        raise ValueError("trace_pixels must be non-negative")
    rows = 0
    for run in runs:
        arguments = effective_trace_arguments(config, run, argument_overrides)
        iterations = _nonnegative_argument(arguments, "--iters", 4)
        geometric_iterations = _nonnegative_argument(
            arguments, "--geometric-iters", 2
        )
        sub_resolution_levels = _nonnegative_argument(
            arguments, "--sub-resolution-levels", 2
        )
        # The photometric stage traverses all pyramid levels using the configured
        # iteration count. Every geometric-consistency stage runs at level 0 and
        # PatchMatch::Init(true) forces exactly one iteration (two logical states),
        # independently of --iters.
        photometric_states = (sub_resolution_levels + 1) * (iterations + 1)
        geometric_states = geometric_iterations * 2
        rows += trace_pixels * (photometric_states + geometric_states)
    return rows


def configured_trace_limits(config: dict[str, Any]) -> tuple[int, int]:
    instrumentation = config.get("instrumentation") or {}
    if not isinstance(instrumentation, dict):
        raise ValueError("instrumentation must be a mapping")

    def positive_integer(name: str, default: int) -> int:
        raw = instrumentation.get(name, default)
        if isinstance(raw, bool):
            raise ValueError(f"instrumentation.{name} must be an integer")
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"instrumentation.{name} must be an integer") from exc
        if value <= 0:
            raise ValueError(f"instrumentation.{name} must be positive")
        return value

    max_pixels = positive_integer(
        "max_trace_pixels_per_request", DEFAULT_MAX_TRACE_PIXELS
    )
    if max_pixels > DEFAULT_MAX_TRACE_PIXELS:
        raise ValueError(
            "instrumentation.max_trace_pixels_per_request must be in "
            f"[1,{DEFAULT_MAX_TRACE_PIXELS}]"
        )
    max_rows = positive_integer(
        "max_trace_rows_per_request", MAX_TRACE_REPORT_ROWS
    )
    if max_rows > MAX_TRACE_REPORT_ROWS:
        raise ValueError(
            "instrumentation.max_trace_rows_per_request must be in "
            f"[1,{MAX_TRACE_REPORT_ROWS}]"
        )
    return max_pixels, max_rows


def validate_trace_row_admission(
    config: dict[str, Any],
    request: dict[str, Any],
    *,
    argument_overrides: dict[str, Any] | None = None,
) -> int:
    """Recheck immutable-request row admission before launching CUDA work."""

    trace_count = len(expand_trace_pixels(request))
    max_pixels, max_rows = configured_trace_limits(config)
    if trace_count > max_pixels:
        raise ValueError(
            f"trace request selects {trace_count} pixels, exceeding the configured limit "
            f"of {max_pixels}"
        )
    selected_runs = validate_request_runs(config, request)
    if argument_overrides is None:
        argument_overrides = selected_scene_argument_overrides(
            config, str((request.get("target") or {}).get("scene_id", ""))
        )
    upper_bound = trace_row_upper_bound(
        config,
        selected_runs,
        trace_count,
        argument_overrides=argument_overrides,
    )
    if upper_bound > max_rows:
        raise ValueError(
            f"trace request can produce up to {upper_bound} rows, exceeding the "
            f"configured report limit of {max_rows}; reduce pixels, runs, PatchMatch "
            "iterations, sub-resolution levels, or geometric iterations"
        )
    capture = request.get("capture") or {}
    declared_bound = capture.get("trace_row_upper_bound")
    declared_limit = capture.get("trace_row_limit")
    if (declared_bound is None) != (declared_limit is None):
        raise ValueError("trace request contains an incomplete row-admission declaration")
    if declared_bound is not None and (
        isinstance(declared_bound, bool)
        or isinstance(declared_limit, bool)
        or declared_bound != upper_bound
        or declared_limit != max_rows
    ):
        raise ValueError("trace request row-admission declaration does not match its config")
    return upper_bound


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
    argument_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not scene_id.strip():
        raise ValueError("scene ID must not be empty")
    if image_id < 0:
        raise ValueError("frame/image ID must be non-negative")
    instrumentation = config.get("instrumentation") or {}
    max_trace_pixels, max_trace_rows = configured_trace_limits(config)
    selected_runs = select_runs(config, variants)
    if argument_overrides is None:
        argument_overrides = selected_scene_argument_overrides(config, scene_id)
    else:
        argument_overrides = _validated_argument_overrides(
            argument_overrides, f"resolved scene {scene_id!r}"
        )
    scene_config = _selected_scene_config(config, scene_id)
    if scene_config is not None and scene_config.get("mvs_file"):
        configured_mvs = Path(str(scene_config["mvs_file"])).expanduser()
        if not configured_mvs.is_absolute():
            configured_mvs = config_path.expanduser().resolve().parent / configured_mvs
        validate_no_implicit_program_options_file(configured_mvs.resolve().parent)
    pixels = normalize_pixels(pixel_values)
    if pixels and roi_value:
        raise ValueError("select pixels or an ROI, not both")
    roi = None
    if roi_value:
        x, y, width, height = parse_roi(roi_value)
        roi = {"x": x, "y": y, "width": width, "height": height}
    trace_count = len(pixels) if pixels else (roi["width"] * roi["height"] if roi else 0)
    if trace_count > max_trace_pixels:
        raise ValueError(
            f"trace request selects {trace_count} pixels, exceeding the configured limit "
            f"of {max_trace_pixels}"
        )
    trace_rows = trace_row_upper_bound(
        config,
        selected_runs,
        trace_count,
        argument_overrides=argument_overrides,
    )
    if trace_rows > max_trace_rows:
        raise ValueError(
            f"trace request can produce up to {trace_rows} rows, exceeding the "
            f"configured report limit of {max_trace_rows}; reduce pixels, runs, "
            "PatchMatch iterations, sub-resolution levels, or geometric iterations"
        )
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
        "runs": selected_runs,
        "capture": {
            "instrumentation_level": "maps",
            "write_maps": True,
            "patch_match_cuda_instances": 1,
            "process_specialization": "Process<true>",
            "compact_exact_trace_available": False,
            "storage_policy": "full-frame exact maps plus selected trace rows",
            "trace_row_upper_bound": trace_rows,
            "trace_row_limit": max_trace_rows,
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
    target = request.get("target") or {}
    if target.get("pixels") and target.get("roi"):
        raise ValueError("drill-down request cannot contain both pixels and an ROI")
    pixels = expand_trace_pixels(request)
    declared_trace_count = target.get("trace_pixel_count")
    if (
        isinstance(declared_trace_count, bool)
        or not isinstance(declared_trace_count, int)
        or declared_trace_count != len(pixels)
    ):
        raise ValueError("drill-down trace_pixel_count does not match the target")
    if profile == "deep" and pixels:
        raise ValueError("deep request must not contain trace pixels")
    if profile == "trace" and not pixels:
        raise ValueError("trace request must contain pixels or an ROI")
    capture = request.get("capture") or {}
    if (
        capture.get("instrumentation_level") != "maps"
        or capture.get("write_maps") is not True
        or capture.get("patch_match_cuda_instances") != 1
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
