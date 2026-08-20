"""Versioned validation for the fixed CUDA reference-patch layout contract."""

from __future__ import annotations

import math
from typing import Any


SCHEMA_NAME = "openmvs.dmap.reference_patch_layout"
SCHEMA_VERSION = 1
MAX_SAMPLES = 4096

TEXTURE_CONTRACT = {
    "texture_address_mode_configured": "wrap",
    "texture_address_mode_effective": "clamp",
    "texture_address_mode_effective_basis": (
        "cuda_runtime_unnormalized_wrap_is_clamped"
    ),
    "texture_coordinates_normalized": False,
    "texture_filter_mode": "linear",
}

CAPTURE_FIELDS = (
    "sample_locations_captured_by_kernel",
    "sample_values_captured_by_kernel",
    "source_view_footprints_captured_by_kernel",
)


def strict_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def normalize(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate schema v1 and return a bounded canonical representation."""

    if not isinstance(value, dict):
        return None, "reference_patch_layout is missing or is not an object"
    if (
        value.get("schema_name") != SCHEMA_NAME
        or strict_integer(value.get("schema_version")) != SCHEMA_VERSION
    ):
        return None, "reference_patch_layout has an unsupported schema identity"
    if value.get("kind") != "fixed_cartesian_grid":
        return None, "reference_patch_layout kind is not fixed_cartesian_grid"
    if value.get("coordinate_domain") != "reference_pyramid_pixels":
        return None, "reference_patch_layout coordinate domain is unsupported"
    if value.get("sample_position") != "integer_offset_from_pixel_center":
        return None, "reference_patch_layout sample-position convention is unsupported"
    if any(value.get(field) != expected for field, expected in TEXTURE_CONTRACT.items()):
        return None, "reference_patch_layout texture sampling contract is unsupported"

    half_window = strict_integer(value.get("half_window_pixels"))
    step = strict_integer(value.get("step_pixels"))
    sample_count = strict_integer(value.get("sample_count"))
    try:
        texel_center = float(value.get("texel_center_offset"))
    except (TypeError, ValueError):
        texel_center = math.nan
    if half_window is None or half_window < 0:
        return None, "reference_patch_layout half_window_pixels is invalid"
    if step is None or step <= 0:
        return None, "reference_patch_layout step_pixels is invalid"
    if not math.isfinite(texel_center) or texel_center != 0.5:
        return None, "reference_patch_layout texel_center_offset must be 0.5"
    if sample_count is None or not 1 <= sample_count <= MAX_SAMPLES:
        return None, "reference_patch_layout sample_count is invalid"

    axis_count = 2 * half_window // step + 1
    if axis_count * axis_count != sample_count:
        return None, "reference_patch_layout dimensions do not match sample_count"
    axis = list(range(-half_window, half_window + 1, step))
    expected_offsets = [[x, y] for y in axis for x in axis]
    raw_offsets = value.get("sample_offsets_pixels")
    if not isinstance(raw_offsets, list):
        return None, "reference_patch_layout sample_offsets_pixels is not a list"
    if len(raw_offsets) != sample_count:
        return None, "reference_patch_layout sample_count does not match its offsets"

    offsets: list[list[int]] = []
    seen: set[tuple[int, int]] = set()
    for index, raw_offset in enumerate(raw_offsets):
        if not isinstance(raw_offset, list) or len(raw_offset) != 2:
            return None, f"reference_patch_layout offset {index} is not an [x,y] pair"
        x = strict_integer(raw_offset[0])
        y = strict_integer(raw_offset[1])
        if x is None or y is None:
            return None, f"reference_patch_layout offset {index} is not integral"
        if (x, y) in seen:
            return None, f"reference_patch_layout offset {index} is duplicated"
        seen.add((x, y))
        offsets.append([x, y])
    if offsets != expected_offsets:
        return None, "reference_patch_layout offsets do not match half-window and step"

    if any(not isinstance(value.get(field), bool) for field in CAPTURE_FIELDS):
        return None, "reference_patch_layout capture-availability flags are invalid"
    if any(value[field] for field in CAPTURE_FIELDS):
        return None, "reference_patch_layout schema v1 does not support captured patch samples"
    provenance = value.get("layout_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        return None, "reference_patch_layout layout_provenance is missing"

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "kind": "fixed_cartesian_grid",
        "coordinate_domain": "reference_pyramid_pixels",
        "sample_position": "integer_offset_from_pixel_center",
        "texel_center_offset": texel_center,
        **TEXTURE_CONTRACT,
        "half_window_pixels": half_window,
        "step_pixels": step,
        "sample_count": sample_count,
        "sample_offsets_pixels": offsets,
        "layout_provenance": provenance,
        **{field: value[field] for field in CAPTURE_FIELDS},
    }, None
