#!/usr/bin/env python3
"""Validate one schema-v2, schema-v3, or schema-v4 depth-map instrumentation frame."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any

import numpy as np
from PIL import Image
import tyro


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from report_dmap_annotation_fit import load_dmap  # noqa: E402
from dmap_instrumentation_report import read_pfm  # noqa: E402
from dmap_observability import reference_patch_layout  # noqa: E402


REQUIRED_FINAL_SIGNALS = {
    "depth_final_before_filter",
    "normal_final_before_filter",
    "cost_final_before_filter",
    "cost_photometric",
    "cost_photo_prior",
    "cost_geometric",
    "cost_total_components",
    "confidence_gap",
    "reference_variance",
    "view_entropy",
    "selected_view_count",
    "accepted_update_count",
}
REQUIRED_PASS_SIGNALS = {
    "depth_delta",
    "depth_relative_delta",
    "normal_angle_delta",
    "view_churn",
}

V3_SCHEMA_NAME = "openmvs.dmap.map_manifest"
V3_REQUIRED_LOGICAL_STATE_SIGNALS = {
    "cost_stored",
    "confidence_stored",
    "cost_photo_raw_equal_selected_rescore_proxy",
    "cost_photo_prior_equal_selected_rescore_proxy",
    "cost_geometric_equal_selected_rescore_proxy",
    "cost_total_equal_selected_rescore_proxy",
    "cost_stored_minus_rescore",
    "depth_prior_disagreement_equal_selected_rescore_proxy",
    "depth_prior_weight_equal_selected_rescore_proxy",
    "gap_local_neighbor_equal_selected_rescore_proxy",
    "reference_variance_equal_selected_rescore_proxy",
}
V3_PROXY_LOGICAL_STATE_SIGNALS = {
    signal for signal in V3_REQUIRED_LOGICAL_STATE_SIGNALS if signal.endswith("_proxy")
}
V3_MEASUREMENT_QUALITY = {
    "cost_stored": "exact",
    "confidence_stored": "derived_exact",
    "cost_stored_minus_rescore": "derived_exact",
    **{signal: "proxy" for signal in V3_PROXY_LOGICAL_STATE_SIGNALS},
}
V3_EXACT_MEASUREMENT_BASES = {
    "cost_stored": "production_cost_snapshot",
    "confidence_stored": "max(1-cost,0)",
}
V3_PROXY_MEASUREMENT_BASIS = "equal_selected_view_binary_post_pass_rescore"
V3_REQUIRED_LOGICAL_EVENT_SIGNALS = REQUIRED_PASS_SIGNALS
LOGICAL_COST_IMPROVEMENT_SIGNAL = "cost_improvement_exact"
LOGICAL_COST_IMPROVEMENT_QUALITY = "derived_exact"
LOGICAL_COST_IMPROVEMENT_BASIS = (
    "sum_of_disjoint_checkerboard_production_cost_reductions"
)
V3_MEASUREMENT_MODEL = (
    "production PatchMatch state snapshots plus explicitly classified post-pass diagnostics"
)
V4_MEASUREMENT_MODEL = (
    "production hot-kernel exact observability plus retained state snapshots and "
    "explicitly classified post-pass proxies"
)
V4_REQUIRED_EXACT_STATE_SIGNALS = {
    "cost_photo_raw_production_exact",
    "cost_photo_prior_production_exact",
    "cost_geometric_production_exact",
    "cost_total_production_exact",
    "depth_prior_disagreement_production_exact",
    "depth_prior_weight_production_exact",
    "gap_winner_runner_up_exact",
    "reference_variance_production_exact",
}
V4_REQUIRED_EXACT_EVENT_SIGNALS = {
    "candidate_stored_cost_before_exact",
    "candidate_incumbent_cost_exact",
    "candidate_winner_cost_exact",
    "candidate_runner_up_cost_exact",
    "candidate_tested_mask_exact",
    "candidate_finite_mask_exact",
    "candidate_accepted_mask_exact",
    "candidate_counts_exact",
    "candidate_identity_exact",
    "selected_view_counts_exact",
    "selected_views_before_mask_exact",
    "selected_views_after_mask_exact",
}
V4_REQUIRED_EXACT_VIEW_SIGNALS = {
    "view_cost_components_exact",
    "view_selection_metrics_exact",
    "view_weighted_contribution_exact",
    "view_selection_state_exact",
    "view_agreement_state_exact",
}
V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS = {
    "low_texture_update_eligible_exact",
    "low_texture_update_ambiguity_exact",
    "low_texture_update_required_gain_exact",
    "low_texture_update_best_proposed_gain_exact",
    "low_texture_update_rejected_mask_exact",
    "low_texture_update_would_have_won_source_exact",
    "low_texture_update_rejected_count_exact",
}
V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS = {
    "candidate_raw_best_cost_exact",
    "candidate_raw_runner_up_cost_exact",
    "gap_raw_best_runner_up_exact",
    "candidate_retained_minus_raw_best_exact",
    "candidate_raw_suppression_identity_exact",
}
V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS = (
    "low_texture_gate_eligible",
    "low_texture_propagation_accepted",
    "low_texture_propagation_rejected",
    "low_texture_refinement_accepted",
    "low_texture_refinement_rejected",
    "low_texture_required_gain_sum",
    "low_texture_best_proposed_gain_sum",
)
V4_LOW_TEXTURE_PROPAGATION_ACCEPTED_MASK = 0x01FE
V4_LOW_TEXTURE_REFINEMENT_ACCEPTED_MASK = 0x1E00

APD_SCHEMA_NAME = "openmvs.dmap.apd_pixel_mechanics"
APD_SCHEMA_VERSION = 3
APD_SUMMARY_SCHEMA_NAME = "openmvs.dmap.apd_observability"
APD_SUMMARY_SCHEMA_VERSION = 3
APD_SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
APD_RECORD_CONTRACTS = {
    1: {"state": 56, "update": 64, "trace": 996, "candidate_slots": 13},
    2: {"state": 56, "update": 88, "trace": 1508, "candidate_slots": 21},
    3: {"state": 64, "update": 120, "trace": 1548, "candidate_slots": 22},
}
APD_STATE_FLOAT_SIGNALS_V2 = {
    "apd_average_baseline",
    "apd_current_disparity",
    "apd_global_minimum_offset",
    "apd_global_minimum_cost",
    "apd_profile_separation",
    "apd_nearest_reliable_distance",
    "apd_ransac_threshold",
    "apd_ransac_center_residual",
    "apd_ransac_mean_inlier_residual",
}
APD_STATE_BYTE_SIGNALS_V2 = {
    "apd_reliability_class",
    "apd_profile_reason",
    "apd_profile_eta",
    "apd_profile_finite_count",
    "apd_profile_local_minimum_count",
    "apd_profile_plateau_start",
    "apd_profile_plateau_end",
    "apd_sector_candidate_count",
    "apd_ransac_inlier_count",
    "apd_ransac_outlier_count",
    "apd_anchor_count",
    "apd_anchor_reason",
    "apd_ransac_valid",
    "apd_deformable_eligible",
}
APD_STATE_FLOAT_SIGNALS = APD_STATE_FLOAT_SIGNALS_V2 | {
    "apd_fitted_plane_depth",
}
APD_STATE_BYTE_SIGNALS = APD_STATE_BYTE_SIGNALS_V2 | {
    "apd_fitted_plane_valid",
}
APD_UPDATE_FLOAT_SIGNALS_V1 = {
    "apd_working_winner_cost",
    "apd_native_persistent_cost",
    "apd_runner_up_working_cost",
    "apd_winner_runner_up_gap",
    "apd_center_cost",
    "apd_anchor_mean_cost",
    "apd_deformable_photometric_cost",
    "apd_geometric_cost",
    "apd_native_minus_working_cost",
    "apd_native_stored_cost_before",
    "apd_incumbent_working_cost",
    "apd_candidate_tested_mask",
    "apd_candidate_finite_mask",
    "apd_candidate_accepted_mask",
}
APD_UPDATE_BYTE_SIGNALS_V1 = {
    "apd_update_source",
    "apd_winner_slot",
    "apd_runner_up_slot",
    "apd_candidate_tested_count",
    "apd_candidate_finite_count",
    "apd_candidate_accepted_count",
    "apd_selected_view_count",
    "apd_deformable_active",
}
APD_UPDATE_FLOAT_SIGNALS_V2 = APD_UPDATE_FLOAT_SIGNALS_V1 | {
    "apd_best_anchor_working_cost",
    "apd_accepted_anchor_native_cost",
    "apd_accepted_anchor_index",
}
APD_UPDATE_BYTE_SIGNALS_V2 = APD_UPDATE_BYTE_SIGNALS_V1 | {
    "apd_view_selection_mode",
    "apd_anchor_evidence_count",
    "apd_anchor_proposal_count",
    "apd_anchor_finite_count",
    "apd_anchor_accepted_slot",
    "apd_immutable_anchor_state",
    "apd_selected_view_weight_sum",
}
APD_UPDATE_FLOAT_SIGNALS = APD_UPDATE_FLOAT_SIGNALS_V2 | {
    "apd_fitted_plane_working_cost",
    "apd_fitted_plane_native_cost",
    "apd_final_refinement_incumbent_cost",
    "apd_final_refinement_best_cost",
    "apd_final_refinement_improvement",
    "apd_final_refinement_depth",
}
APD_UPDATE_BYTE_SIGNALS = APD_UPDATE_BYTE_SIGNALS_V2 | {
    "apd_update_stage",
    "apd_fitted_plane_available",
    "apd_fitted_plane_tested",
    "apd_fitted_plane_accepted",
    "apd_final_refinement_offset",
    "apd_final_refinement_tested_count",
    "apd_final_refinement_finite_count",
    "apd_final_refinement_accepted",
}
APD_UPDATE_RGBA_SIGNALS = {
    "apd_working_selected_views_mask",
}
APD_REQUIRED_SIGNALS_V1 = (
    APD_STATE_FLOAT_SIGNALS_V2
    | APD_STATE_BYTE_SIGNALS_V2
    | APD_UPDATE_FLOAT_SIGNALS_V1
    | APD_UPDATE_BYTE_SIGNALS_V1
)
APD_REQUIRED_SIGNALS_V2 = (
    APD_STATE_FLOAT_SIGNALS_V2
    | APD_STATE_BYTE_SIGNALS_V2
    | APD_UPDATE_FLOAT_SIGNALS_V2
    | APD_UPDATE_BYTE_SIGNALS_V2
    | APD_UPDATE_RGBA_SIGNALS
)
APD_REQUIRED_SIGNALS = (
    APD_STATE_FLOAT_SIGNALS
    | APD_STATE_BYTE_SIGNALS
    | APD_UPDATE_FLOAT_SIGNALS
    | APD_UPDATE_BYTE_SIGNALS
    | APD_UPDATE_RGBA_SIGNALS
)
APD_MULTISCALE_INPUT_SIGNALS = {
    "apd_transferred_reliability",
    "apd_transferred_anchor_count",
    "apd_transferred_deformable_eligible",
}
APD_MULTISCALE_OUTPUT_SIGNALS = {
    "apd_output_reliability",
    "apd_output_anchor_count",
    "apd_output_deformable_eligible",
}
APD_MULTISCALE_SIGNALS = APD_MULTISCALE_INPUT_SIGNALS | APD_MULTISCALE_OUTPUT_SIGNALS
APD_PROFILE_REASON_NAMES = (
    "unknown_invalid_input",
    "unknown_nonfinite_cost",
    "unreliable_no_local_minimum",
    "unreliable_global_minimum_outside_eta",
    "unreliable_global_minimum_cost_too_high",
    "unreliable_single_minimum_cost_not_strictly_below_t2",
    "unreliable_multi_minimum_separation_not_above_t3",
    "reliable_single_minimum",
    "reliable_separated_minima",
)
APD_ANCHOR_REASON_NAMES = (
    "unknown",
    "pixel_not_unreliable",
    "invalid_center_depth",
    "insufficient_sector_candidates",
    "no_valid_ransac_model",
    "insufficient_model_inliers",
    "ready",
)
APD_UPDATE_SOURCE_NAMES = (
    "none",
    "init",
    "propagate",
    "refine_depth",
    "refine_normal",
    "refine_random_normal",
    "refine_surface_normal",
    "filtered",
    "changed_unknown",
    "apd_anchor_propagate",
    "apd_fitted_plane",
    "apd_final_refinement",
)
APD_VIEW_SELECTION_MODE_NAMES = (
    "native",
    "anchor_evidence",
    "previous_weights_fallback",
    "selected_mask_fallback",
    "first_view_fallback",
)
APD_COMPONENT_CLOSURE_TOLERANCE = 2.0e-6
APD_SUMMARY_MEAN_TOLERANCE = 5.0e-4

# Component maps are written independently as float32 values. Reconstructing a
# total from those files can differ by a few ULPs even when the CUDA-side
# arithmetic is correct. Keep output-parity and direct-definition checks on the
# caller's (zero by default) tolerance; only additive component closure uses
# this explicit storage-arithmetic allowance.
FLOAT32_COMPONENT_CLOSURE_TOLERANCE = 1.0e-6
FLOAT32_EXP_ULP_TOLERANCE = 8
NEGATIVE_VARIANCE_PRIOR_WEIGHT_WARNING = (
    "negative_reference_variance_prior_weight_overshoot"
)

OPTIONAL_MAP_MANIFESTS = (
    (
        "postprocess_filters.json",
        "openmvs.dmap.postprocess_filters",
        frozenset({1, 2}),
        "postprocess_filters",
    ),
    (
        "confidence_adjustment.json",
        "openmvs.dmap.confidence_adjustment",
        frozenset({1}),
        "confidence_adjustment",
    ),
)


@dataclass
class Arguments:
    """Validate map schema and optional production-output parity."""

    frame_dir: Path
    instrumented_dmap: Path | None = None
    reference_dmap: Path | None = None
    output: Path | None = None
    expect_geometric_zero: bool = False
    tolerance: float = 0.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_manifest_map(frame_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    path, path_error = owned_regular_artifact_path(frame_dir, entry.get("path"))
    if path_error or path is None:
        raise ValueError(path_error or "map artifact path is unavailable")
    if path.suffix.lower() == ".pfm":
        return read_pfm(path, np)
    if path.suffix.lower() == ".png":
        return np.asarray(Image.open(path))
    raise ValueError(f"unsupported map format: {path}")


def max_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        return float("inf")
    finite_first = np.isfinite(first)
    finite_second = np.isfinite(second)
    if not np.array_equal(finite_first, finite_second):
        return float("inf")
    if not finite_first.any():
        return 0.0
    return float(np.max(np.abs(first[finite_first] - second[finite_first])))


def strict_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def machine_readable_result(result: dict[str, Any]) -> dict[str, Any]:
    """Replace non-finite diagnostics with JSON null and retain their locations."""

    replaced_paths: list[str] = []

    def normalize(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item, f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                normalize(item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, (float, np.floating)):
            number = float(value)
            if not np.isfinite(number):
                replaced_paths.append(path)
                return None
            return number
        return value

    normalized = normalize(result, "$")
    if replaced_paths:
        normalized["machine_readable_diagnostics"] = {
            "nonfinite_values_represented_as_null": replaced_paths,
            "meaning": (
                "The validator used a non-finite internal sentinel because the "
                "reported comparison was unavailable or invalid. The associated "
                "failed check retains the validation outcome."
            ),
        }
    return normalized


def csv_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def csv_integer(value: Any) -> int | None:
    number = csv_finite_number(value)
    return int(number) if number is not None and number.is_integer() else None


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


SUMMARY_UNAVAILABLE_EXACT_SIGNALS = {
    "candidate_family_tested_finite_accepted",
    "exact_propagation_vs_refinement_acceptance",
    "exact_same_pass_runner_up_gap",
    "exact_per_view_reliability_and_contributions",
}

EXACT_TRACE_REQUIRED_CANDIDATE_FIELDS = {
    "available", "candidate_tested_mask", "candidate_finite_mask",
    "candidate_production_valid_mask", "candidate_accepted_mask", "tested_count",
    "finite_count", "production_valid_count", "accepted_count",
    "selected_views_before_mask", "selected_views_after_mask",
    "selected_count_before", "selected_count_after", "stored_cost_before",
    "incumbent_cost", "winner_cost", "runner_up_cost", "winner_runner_up_gap",
    "raw_best_cost", "raw_runner_up_cost", "raw_best_runner_up_gap",
    "retained_minus_raw_best", "source", "source_code", "winner_slot",
    "winner_slot_name", "runner_up_slot", "runner_up_slot_name", "raw_best_slot",
    "raw_best_slot_name", "raw_runner_up_slot", "raw_runner_up_slot_name",
    "raw_best_suppressed", "raw_best_suppressed_source",
    "raw_best_suppressed_source_code", "low_texture_update_hysteresis",
}
EXACT_TRACE_REQUIRED_VIEW_FIELDS = {
    "available", "source_view_index", "source_image_id", "source_image_name",
    "photometric_cost", "geometric_cost", "total_cost", "weighted_contribution",
    "selection_prior", "sampling_score", "sampling_probability", "metadata",
    "weight", "rank", "neighbor_agreement_count", "neighbor_bad_count",
    "decision", "decision_code", "selected", "production_valid_cost",
    "contribution_available", "selection_prior_available", "sampling_score_available",
    "probability_available", "rank_available", "decision_available",
}
LEGACY_TRACE_COMPONENT_VIEW_LIMIT = 4
LEGACY_TRACE_COMPONENT_FIELDS = (
    "view_costs",
    "view_photometric_costs",
    "view_geometric_costs",
)
PATCHMATCH_SELECTED_VIEW_BIN_COUNT = 33


def observer_sidecars_complete(container: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    sidecars = container.get("observer_sidecars")
    sidecars = sidecars if isinstance(sidecars, dict) else {}
    errors = sidecars.get("write_errors")
    count = strict_int(sidecars.get("write_error_count"))
    valid = (
        sidecars.get("complete") is True
        and count == 0
        and isinstance(errors, list)
        and not errors
    )
    return valid, sidecars


def validate_frame_census(
    frame_dir: Path,
    summary: dict[str, Any],
    expected_iterations: list[int],
) -> tuple[bool, dict[str, Any]]:
    """Validate one frame's complete logical-state Process census."""

    instrumentation_root = (
        frame_dir.parent.parent if frame_dir.parent.name == "depthmaps" else frame_dir.parent
    )
    counters_path, path_error = owned_regular_artifact_path(
        instrumentation_root, "instrumentation/counters.csv"
    )
    errors: list[str] = []
    rows: list[dict[str, str]] = []
    if path_error or counters_path is None:
        errors.append(path_error or "instrumentation/counters.csv is unavailable")
    else:
        try:
            with counters_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception as exc:
            errors.append(f"cannot read instrumentation/counters.csv: {exc}")

    image_id = strict_int(summary.get("image_id"))
    scale_level = strict_int(summary.get("scale_level"))
    width = strict_int(summary.get("width"))
    height = strict_int(summary.get("height"))
    area = width * height if width is not None and height is not None else None
    selected_bin_fields = [
        f"selected_views_{index}" for index in range(PATCHMATCH_SELECTED_VIEW_BIN_COUNT)
    ]
    candidate_prefixes = (
        "candidate_tested_", "candidate_finite_", "candidate_accepted_",
    )
    candidate_fields = [
        field
        for prefix in candidate_prefixes
        for field in (rows[0].keys() if rows else [])
        if field.startswith(prefix)
    ]
    required_fields = {
        "image_id", "scale_number", "width", "height", "iteration", "pass_index",
        "processed", "valid_depth", "invalid_depth", "accepted", "component_samples",
        *selected_bin_fields,
    }
    if rows:
        missing_fields = sorted(required_fields - rows[0].keys())
        if missing_fields:
            errors.append(f"counters.csv is missing fields: {missing_fields}")
        missing_candidate_groups = [
            prefix for prefix in candidate_prefixes
            if not any(field.startswith(prefix) for field in rows[0].keys())
        ]
        if missing_candidate_groups:
            errors.append(
                f"counters.csv is missing candidate counter groups: {missing_candidate_groups}"
            )

    matching_rows: list[tuple[int, dict[str, str]]] = []
    for line_index, row in enumerate(rows, start=2):
        if (
            csv_integer(row.get("image_id")) == image_id
            and csv_integer(row.get("scale_number")) == scale_level
        ):
            matching_rows.append((line_index, row))

    observed_iterations: list[int] = []
    for line_index, row in matching_rows:
        iteration = csv_integer(row.get("iteration"))
        observed_iterations.append(iteration if iteration is not None else sys.maxsize)
        prefix = f"counters.csv line {line_index}"
        integer_fields = {
            field: csv_integer(row.get(field))
            for field in (
                "width", "height", "pass_index", "processed", "valid_depth",
                "invalid_depth", "accepted", "component_samples", *selected_bin_fields,
            )
        }
        if any(value is None or value < 0 for value in integer_fields.values()):
            errors.append(f"{prefix}: census counts must be non-negative integers")
            continue
        processed = integer_fields["processed"]
        valid = integer_fields["valid_depth"]
        invalid = integer_fields["invalid_depth"]
        component_samples = integer_fields["component_samples"]
        selected_bins = [integer_fields[field] for field in selected_bin_fields]
        if area is None or processed != area:
            errors.append(f"{prefix}: processed={processed} does not match frame area={area}")
        if integer_fields["width"] != width or integer_fields["height"] != height:
            errors.append(f"{prefix}: counter dimensions do not match the frame summary")
        if iteration is None or integer_fields["pass_index"] != iteration + 1:
            errors.append(f"{prefix}: pass_index does not match the logical iteration")
        if valid + invalid != processed:
            errors.append(f"{prefix}: valid_depth + invalid_depth does not equal processed")
        if integer_fields["accepted"] > processed:
            errors.append(f"{prefix}: accepted exceeds processed")
        if sum(selected_bins) != processed:
            errors.append(f"{prefix}: selected-view bins do not sum to processed")
        if component_samples != processed - selected_bins[0]:
            errors.append(
                f"{prefix}: component_samples does not match pixels with selected views"
            )

        candidate_values = [csv_integer(row.get(field)) for field in candidate_fields]
        if any(value is None or value < 0 for value in candidate_values):
            errors.append(f"{prefix}: candidate counters must be non-negative integers")
        elif (
            summary.get("candidate_accounting_mode") == "unavailable_post_pass_snapshot"
            and any(candidate_values)
        ):
            errors.append(
                f"{prefix}: exact candidate counters are nonzero while accounting is unavailable"
            )

    if observed_iterations != expected_iterations:
        errors.append(
            "counters.csv logical iterations do not match the configured topology: "
            f"expected {expected_iterations}, observed {observed_iterations}"
        )
    if matching_rows and expected_iterations:
        terminal = matching_rows[-1][1]
        terminal_valid = csv_integer(terminal.get("valid_depth"))
        summary_valid = strict_int(summary.get("num_valid_before_filter"))
        if terminal_valid != summary_valid:
            errors.append(
                "terminal census valid_depth does not match summary num_valid_before_filter: "
                f"counter={terminal_valid}, summary={summary_valid}"
            )
    return not errors, {
        "path": str(counters_path) if counters_path is not None else None,
        "expected_iterations": expected_iterations,
        "observed_iterations": observed_iterations,
        "matching_rows": len(matching_rows),
        "candidate_accounting_mode": summary.get("candidate_accounting_mode"),
        "errors": errors,
    }


def validate_exact_targeted_trace(
    frame_dir: Path,
    summary: dict[str, Any],
    resource_plan: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Validate the fine-level compact exact records declared by a frame summary."""

    instrumentation_root = (
        frame_dir.parent.parent if frame_dir.parent.name == "depthmaps" else frame_dir.parent
    )
    trace_path = instrumentation_root / "instrumentation" / "traces.jsonl"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    if not trace_path.is_file():
        errors.append("targeted trace JSONL is missing")
    else:
        for line_number, text in enumerate(
            trace_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not text.strip():
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: record is not an object")
                continue
            if (
                strict_int(value.get("image_id")) == strict_int(summary.get("image_id"))
                and strict_int(value.get("scale_number")) == 0
            ):
                records.append(value)

    expected_candidates = strict_int(resource_plan.get("exact_candidate_record_count"))
    expected_views = strict_int(resource_plan.get("exact_view_record_count"))
    trace_pixels = strict_int(resource_plan.get("num_trace_pixels"))
    pixel_stride = strict_int(resource_plan.get("exact_record_pixel_stride"))
    if (
        expected_candidates is None or expected_candidates <= 0
        or expected_views is None or expected_views <= 0
        or trace_pixels is None or trace_pixels <= 0
        or pixel_stride != trace_pixels
        or expected_candidates % trace_pixels != 0
    ):
        errors.append("invalid compact exact record counts or stride in resource plan")
        logical_states = None
    else:
        logical_states = expected_candidates // trace_pixels

    observed_pairs: set[tuple[int, int]] = set()
    observed_view_records = 0
    width = strict_int(summary.get("width"))
    height = strict_int(summary.get("height"))
    for record_index, record in enumerate(records):
        prefix = f"record {record_index}"
        trace_index = strict_int(record.get("trace_index"))
        logical_state = strict_int(record.get("logical_state_index"))
        num_views = strict_int(record.get("num_views"))
        x = strict_int(record.get("x"))
        y = strict_int(record.get("y"))
        if (
            record.get("schema_name") != "openmvs.dmap.targeted_trace"
            or strict_int(record.get("schema_version")) != 2
            or record.get("process_specialization") != "Process<true>"
            or record.get("measurement_quality") != "exact"
            or record.get("measurement_basis") != "production_hot_kernel_targeted_trace"
            or record.get("exact_hot_kernel_record") is not True
        ):
            errors.append(f"{prefix}: invalid exact targeted-trace schema or basis")
        if (
            trace_index is None or trace_index < 0 or trace_index >= (trace_pixels or 0)
            or logical_state is None or logical_state < 0
            or logical_states is None or logical_state >= logical_states
        ):
            errors.append(f"{prefix}: trace/state index is outside the declared compact layout")
        else:
            pair = (trace_index, logical_state)
            if pair in observed_pairs:
                errors.append(f"{prefix}: duplicate trace/state record {pair}")
            observed_pairs.add(pair)
        if (
            width is None or height is None or x is None or y is None
            or x < 0 or x >= width or y < 0 or y >= height
        ):
            errors.append(f"{prefix}: fine-level pixel coordinate is out of bounds")

        candidate = record.get("exact_candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        missing_candidate = sorted(EXACT_TRACE_REQUIRED_CANDIDATE_FIELDS - candidate.keys())
        if candidate.get("available") is not True or missing_candidate:
            errors.append(f"{prefix}: incomplete exact_candidate fields {missing_candidate}")
        for count_name, mask_name in (
            ("tested_count", "candidate_tested_mask"),
            ("finite_count", "candidate_finite_mask"),
            ("production_valid_count", "candidate_production_valid_mask"),
            ("accepted_count", "candidate_accepted_mask"),
        ):
            count = strict_int(candidate.get(count_name))
            mask = strict_int(candidate.get(mask_name))
            if count is None or mask is None or mask < 0 or count != mask.bit_count():
                errors.append(f"{prefix}: {count_name} does not match {mask_name}")

        exact_views = record.get("exact_views")
        exact_views = exact_views if isinstance(exact_views, list) else []
        observed_view_records += len(exact_views)
        observability = record.get("exact_observability")
        observability = observability if isinstance(observability, dict) else {}
        if (
            num_views is None or num_views <= 0
            or record.get("exact_views_available") is not True
            or len(exact_views) != num_views
            or observability.get("available") is not True
            or observability.get("record_layout") != "selected_pixels_compact"
            or observability.get("candidate_record_available") is not True
            or observability.get("view_records_available") is not True
            or strict_int(observability.get("view_record_count")) != num_views
        ):
            errors.append(f"{prefix}: exact view count/availability contract is inconsistent")
        component_count = strict_int(record.get("view_component_count"))
        expected_component_count = (
            min(num_views, LEGACY_TRACE_COMPONENT_VIEW_LIMIT)
            if num_views is not None and num_views > 0 else None
        )
        component_arrays = [record.get(field) for field in LEGACY_TRACE_COMPONENT_FIELDS]
        truncation_reason = record.get("view_component_unavailable_reason")
        if (
            component_count != expected_component_count
            or component_count is None
            or any(
                not isinstance(values, list) or len(values) != component_count
                for values in component_arrays
            )
            or (
                num_views is not None
                and component_count < num_views
                and (
                    not isinstance(truncation_reason, str)
                    or not truncation_reason.strip()
                )
            )
        ):
            errors.append(f"{prefix}: legacy view-component contract is inconsistent")
        source_indices: list[int | None] = []
        for view_index, view in enumerate(exact_views):
            if not isinstance(view, dict):
                errors.append(f"{prefix}: view {view_index} is not an object")
                continue
            missing_view = sorted(EXACT_TRACE_REQUIRED_VIEW_FIELDS - view.keys())
            if view.get("available") is not True or missing_view:
                errors.append(f"{prefix}: incomplete exact view {view_index} fields {missing_view}")
            source_indices.append(strict_int(view.get("source_view_index")))
        if num_views is not None and source_indices != list(range(num_views)):
            errors.append(f"{prefix}: source views are padded, duplicated, or out of runtime order")

    if expected_candidates is not None and len(records) != expected_candidates:
        errors.append(
            f"expected {expected_candidates} fine-level candidate records, found {len(records)}"
        )
    if expected_views is not None and observed_view_records != expected_views:
        errors.append(
            f"expected {expected_views} fine-level view records, found {observed_view_records}"
        )
    if logical_states is not None:
        expected_pairs = {
            (trace_index, logical_state)
            for trace_index in range(trace_pixels or 0)
            for logical_state in range(logical_states)
        }
        if observed_pairs != expected_pairs:
            errors.append("compact trace/state coverage is incomplete")

    return not errors, {
        "path": str(trace_path),
        "record_layout": "selected_pixels_compact",
        "expected_candidate_records": expected_candidates,
        "observed_candidate_records": len(records),
        "expected_view_records": expected_views,
        "observed_view_records": observed_view_records,
        "errors": errors,
    }


def validate_summary_only(arguments: Arguments, frame_dir: Path) -> dict[str, Any]:
    """Validate an atomically completed summary capture without requiring maps."""

    summary = load_json(frame_dir / "summary.json")
    marker = load_json(frame_dir / "summary_complete.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    schema_version = strict_int(summary.get("schema_version"))
    width = strict_int(summary.get("width"))
    height = strict_int(summary.get("height"))
    image_id = strict_int(summary.get("image_id"))
    total_pixels = width * height if width is not None and height is not None else None
    check("summary_schema", summary.get("schema_name") == "openmvs.dmap.frame_summary" and schema_version == 4, {
        "schema_name": summary.get("schema_name"),
        "schema_version": summary.get("schema_version"),
    })
    patch_layout_valid, patch_layout_detail = (
        validate_reference_patch_layout_contract(frame_dir, summary)
    )
    check(
        "reference_patch_layout_contract",
        patch_layout_valid,
        patch_layout_detail,
    )
    check("summary_dimensions", width is not None and width > 0 and height is not None and height > 0, {
        "width": summary.get("width"), "height": summary.get("height"),
    })

    sidecars_valid, sidecars = observer_sidecars_complete(summary)
    check("summary_observer_sidecars_complete", sidecars_valid, sidecars)

    marker_summary = marker.get("summary") if isinstance(marker.get("summary"), dict) else {}
    marker_valid = (
        marker.get("schema_name") == "openmvs.dmap.summary_complete"
        and marker.get("schema_version") == 1
        and marker.get("capture_kind") == "summary_only"
        and marker.get("summary_complete") is True
        and marker.get("maps_complete") is False
        and marker.get("observer_sidecars_complete") is True
        and marker_summary.get("path") == "summary.json"
        and marker_summary.get("schema_version") == schema_version
        and marker.get("image_id") == image_id
        and marker.get("image_name") == summary.get("image_name")
        and marker.get("estimation_stage") == summary.get("estimation_stage")
        and marker.get("geometric_iteration") == summary.get("geometric_iteration")
    )
    check("summary_completion_marker", marker_valid, marker)
    expected_completion = {
        "schema_name": "openmvs.dmap.summary_complete",
        "schema_version": 1,
        "path": "summary_complete.json",
        "maps_complete": False,
        "eligible": True,
    }
    check(
        "summary_completion_reference",
        summary.get("completion_marker") == expected_completion,
        {"actual": summary.get("completion_marker"), "expected": expected_completion},
    )

    resource_plan = summary.get("resource_plan") if isinstance(summary.get("resource_plan"), dict) else {}
    maps_requested = resource_plan.get("maps_requested")
    exact_reason = str(resource_plan.get("exact_unavailable_reason") or "")
    exact_trace_available = resource_plan.get("exact_trace_available") is True
    exact_trace_requested = resource_plan.get("exact_trace_requested") is True
    exact_trace_reason = str(resource_plan.get("exact_trace_unavailable_reason") or "")
    exact_trace_contract = not exact_trace_available or (
        resource_plan.get("exact_trace_requested") is True
        and resource_plan.get("exact_trace_compatible") is True
        and resource_plan.get("exact_trace_record_layout") == "selected_pixels_compact"
        and strict_int(resource_plan.get("exact_record_pixel_stride"))
        == strict_int(resource_plan.get("num_trace_pixels"))
        and (strict_int(resource_plan.get("exact_candidate_record_count")) or 0) > 0
        and (strict_int(resource_plan.get("exact_view_record_count")) or 0) > 0
    )
    resource_valid = (
        resource_plan.get("summary_available") is True
        and resource_plan.get("maps_available") is False
        and resource_plan.get("exact_available") is False
        and marker.get("maps_requested") == maps_requested
        and nonempty_text(exact_reason)
        and exact_trace_contract
    )
    check("summary_resource_plan", resource_valid, resource_plan)
    unavailable = {
        str(value) for value in summary.get("unavailable_signals") or []
    }
    apd_observability = (
        summary.get("apd_observability")
        if isinstance(summary.get("apd_observability"), dict) else {}
    )
    apd_aggregate_available = (
        apd_observability.get("enabled") is True
        and resource_plan.get("apd_requested") is True
        and resource_plan.get("apd_summary_available") is True
        and resource_plan.get("apd_maps_available") is False
    )
    exact_trace_detail: dict[str, Any] = {}
    if exact_trace_available:
        exact_modes_valid = (
            summary.get("candidate_accounting_mode")
            == "exact_production_hot_kernel_counters_and_targeted_pixels"
            and summary.get("confidence_gap_mode")
            == "exact_process_pixel_winner_runner_up_at_targeted_pixels"
            and not (SUMMARY_UNAVAILABLE_EXACT_SIGNALS & unavailable)
        )
        exact_trace_valid, exact_trace_detail = validate_exact_targeted_trace(
            frame_dir, summary, resource_plan
        )
        check("summary_exact_targeted_trace", exact_modes_valid and exact_trace_valid, {
            "candidate_accounting_mode": summary.get("candidate_accounting_mode"),
            "confidence_gap_mode": summary.get("confidence_gap_mode"),
            "unavailable_signals": sorted(unavailable),
            "trace": exact_trace_detail,
        })
    else:
        generic_exact_unavailable = (
            summary.get("candidate_accounting_mode") == "unavailable_post_pass_snapshot"
            and summary.get("confidence_gap_mode") == "post_pass_current_plus_eight_neighbors"
            and SUMMARY_UNAVAILABLE_EXACT_SIGNALS <= unavailable
        )
        apd_exact_aggregate = (
            apd_aggregate_available
            and summary.get("candidate_accounting_mode")
            == "exact_apd_working_objective_aggregate"
            and summary.get("confidence_gap_mode")
            == "exact_apd_working_winner_runner_up_aggregate"
            and "apd_exact_full_frame_pixel_mechanics" in unavailable
        )
        check("summary_exact_maps_unavailable", (
            generic_exact_unavailable or apd_exact_aggregate
        ), {
            "candidate_accounting_mode": summary.get("candidate_accounting_mode"),
            "confidence_gap_mode": summary.get("confidence_gap_mode"),
            "apd_aggregate_available": apd_aggregate_available,
            "unavailable_signals": sorted(unavailable),
        })

    unexpected_markers = [
        path.name for path in (frame_dir / "map_manifest.json", frame_dir / "capture_complete.json")
        if path.exists()
    ]
    unexpected_maps = sorted(
        path.relative_to(frame_dir).as_posix()
        for path in frame_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pfm", ".png"}
    )
    check("summary_has_no_map_capture", not unexpected_markers and not unexpected_maps, {
        "unexpected_markers": unexpected_markers,
        "unexpected_maps": unexpected_maps,
    })

    iteration_path = frame_dir / "iteration.csv"
    iteration_rows: list[dict[str, str]] = []
    iteration_errors: list[str] = []
    if iteration_path.is_file():
        try:
            with iteration_path.open(encoding="utf-8", newline="") as handle:
                iteration_rows = list(csv.DictReader(handle))
        except Exception as exc:
            iteration_errors.append(str(exc))
    logical_iterations: list[int] = []
    iteration_topology_errors: list[str] = []
    estimation_iterations = strict_int(
        (summary.get("cuda_patchmatch_parameters") or {}).get("estimation_iterations")
        if isinstance(summary.get("cuda_patchmatch_parameters"), dict)
        else None
    )
    expected_iterations = (
        list(range(-1, estimation_iterations))
        if estimation_iterations is not None and estimation_iterations >= 0 else []
    )
    if estimation_iterations is None or estimation_iterations < 0:
        iteration_errors.append(
            "summary cuda_patchmatch_parameters.estimation_iterations is invalid"
        )
    for index, row in enumerate(iteration_rows):
        try:
            raw_scale = row.get("scale_level")
            summary_scale = strict_int(summary.get("scale_level"))
            if summary_scale is not None and raw_scale not in (None, ""):
                if int(float(raw_scale)) != summary_scale:
                    continue
            value = float(row.get("iteration", ""))
            iteration = int(value)
            if value != iteration:
                raise ValueError("not an integer")
            logical_iterations.append(iteration)
            if int(float(row.get("image_id", "-1"))) != image_id:
                iteration_errors.append(f"row {index}: image_id mismatch")
            num_pixels = csv_integer(row.get("num_pixels"))
            valid_ratio = csv_finite_number(row.get("valid_ratio"))
            changed_ratio = csv_finite_number(row.get("changed_ratio"))
            pass_index = csv_integer(row.get("pass_index"))
            phase = row.get("phase")
            if num_pixels != total_pixels:
                iteration_topology_errors.append(
                    f"row {index}: num_pixels={num_pixels} does not match {total_pixels}"
                )
            if valid_ratio is None or not 0.0 <= valid_ratio <= 1.0:
                iteration_topology_errors.append(f"row {index}: invalid valid_ratio")
            if changed_ratio is None or not 0.0 <= changed_ratio <= 1.0:
                iteration_topology_errors.append(f"row {index}: invalid changed_ratio")
            if pass_index != iteration + 1:
                iteration_topology_errors.append(f"row {index}: invalid pass_index")
            expected_phase = "initialization" if iteration == -1 else "iteration"
            if phase != expected_phase:
                iteration_topology_errors.append(f"row {index}: invalid phase")
        except (TypeError, ValueError) as exc:
            iteration_errors.append(f"row {index}: invalid logical iteration ({exc})")
    unique_iterations = sorted(set(logical_iterations))
    check(
        "summary_logical_iterations",
        bool(iteration_rows)
        and not iteration_errors
        and not iteration_topology_errors
        and len(logical_iterations) == len(unique_iterations)
        and logical_iterations == expected_iterations
        and unique_iterations == expected_iterations,
        {
            "path": str(iteration_path),
            "logical_iterations": unique_iterations,
            "expected_iterations": expected_iterations,
            "errors": iteration_errors,
            "topology_errors": iteration_topology_errors,
        },
    )
    census_valid, census_detail = validate_frame_census(
        frame_dir, summary, expected_iterations
    )
    check("summary_process_census", census_valid, census_detail)

    filtering_path = frame_dir / "filtering.json"
    view_support_path = frame_dir / "view_support.csv"
    filtering = load_json(filtering_path) if filtering_path.is_file() else {}
    view_support_rows: list[dict[str, str]] = []
    if view_support_path.is_file():
        try:
            with view_support_path.open(encoding="utf-8", newline="") as handle:
                view_support_rows = list(csv.DictReader(handle))
        except Exception:
            view_support_rows = []
    count_keys = (
        "num_pixels_total", "num_valid_before_filter", "num_invalid_before_filter",
        "num_valid_after_filter", "num_rejected_by_filter",
    )
    filter_matches = bool(filtering) and all(
        filtering.get(key) == summary.get(key) for key in count_keys
    )
    support_pixels: list[int] = []
    try:
        support_pixels = [int(row["pixels"]) for row in view_support_rows]
    except (KeyError, TypeError, ValueError):
        support_pixels = []
    check("summary_artifacts_complete", filter_matches and bool(support_pixels) and sum(support_pixels) == total_pixels, {
        "filtering_path": str(filtering_path),
        "filter_counts_match": filter_matches,
        "view_support_path": str(view_support_path),
        "view_support_pixels": sum(support_pixels) if support_pixels else None,
        "expected_pixels": total_pixels,
    })

    summary_total = strict_int(summary.get("num_pixels_total"))
    valid_before = strict_int(summary.get("num_valid_before_filter"))
    invalid_before = strict_int(summary.get("num_invalid_before_filter"))
    valid_after = strict_int(summary.get("num_valid_after_filter"))
    rejected = strict_int(summary.get("num_rejected_by_filter"))
    count_values = (summary_total, valid_before, invalid_before, valid_after, rejected)
    counts_valid = (
        total_pixels is not None
        and all(value is not None and value >= 0 for value in count_values)
        and summary_total == total_pixels
        and valid_before + invalid_before == total_pixels
        and valid_after + rejected == valid_before
    )
    check("summary_pixel_counts", counts_valid, {
        "total": summary_total, "valid_before": valid_before,
        "invalid_before": invalid_before, "valid_after": valid_after, "rejected": rejected,
    })

    dmap_checked = False
    dmap_shapes: dict[str, Any] = {}
    instrumented: dict[str, Any] | None = None
    if arguments.instrumented_dmap is not None:
        instrumented = load_dmap(arguments.instrumented_dmap.expanduser().resolve())
        expected_shapes = {
            "depth_map": (height, width),
            "normal_map": (height, width, 3),
            "confidence_map": (height, width),
        }
        dmap_shapes = {
            key: list(value.shape) if isinstance(value, np.ndarray) else None
            for key, value in instrumented.items()
            if key in expected_shapes
        }
        dmap_checked = all(
            isinstance(instrumented.get(key), np.ndarray)
            and instrumented[key].shape == shape
            for key, shape in expected_shapes.items()
        )
        check("summary_instrumented_dmap", dmap_checked, {
            "path": str(arguments.instrumented_dmap), "shapes": dmap_shapes,
        })

    parity: dict[str, float] = {}
    if arguments.reference_dmap is not None:
        if instrumented is None:
            raise ValueError("--reference-dmap requires --instrumented-dmap")
        reference = load_dmap(arguments.reference_dmap.expanduser().resolve())
        for key in ("depth_map", "normal_map", "confidence_map"):
            if key in instrumented and key in reference:
                parity[key] = max_abs_difference(instrumented[key], reference[key])
        check(
            "production_output_parity",
            len(parity) == 3 and all(value <= arguments.tolerance for value in parity.values()),
            parity,
        )

    maps_reason = (
        "summary_profile_maps_not_requested"
        if maps_requested is False
        else str(resource_plan.get("decision") or "summary_capture_maps_unavailable")
    )
    result = {
        "schema_version": schema_version,
        "capture_kind": "summary_only",
        "valid": all(item["passed"] for item in checks),
        "warnings": [],
        "frame_dir": str(frame_dir),
        "checks": checks,
        "manifest_map_count": 0,
        "maps_available": False,
        "maps_unavailable_reason": maps_reason,
        "exact_maps_available": False,
        "exact_maps_unavailable_reason": exact_reason or maps_reason,
        "exact_targeted_trace_available": exact_trace_available,
        "exact_targeted_trace_unavailable_reason": (
            "" if exact_trace_available else exact_trace_reason
        ),
        "instrumented_dmap_checked": dmap_checked,
        "dmap_consistency_max_abs": {},
        "dmap_terminal_state": {},
        "parity_max_abs": parity,
        "logical_state_validation": {
            "available": False, "unavailable_reason": maps_reason,
        },
        "logical_iterations": unique_iterations,
        "exact_validation": ({
            "available": True,
            "scope": "selected_pixels_compact",
            **exact_trace_detail,
        } if exact_trace_available else {
            "available": False,
            "unavailable_reason": (
                exact_trace_reason if exact_trace_requested and exact_trace_reason
                else exact_reason or maps_reason
            ),
        }),
        "candidate_counts": summary.get("candidate_acceptance") or [],
        "valid_ratio_after_filter": summary.get("valid_ratio_after_filter"),
        "ignore_mask": None,
    }
    result = machine_readable_result(result)
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def validate_prefilter_only(arguments: Arguments, frame_dir: Path) -> dict[str, Any]:
    """Validate the bounded Process<false> prefilter snapshot profile."""

    manifest = load_json(frame_dir / "prefilter_manifest.json")
    summary = load_json(frame_dir / "summary.json")
    marker = load_json(frame_dir / "prefilter_capture_complete.json")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    width = strict_int(manifest.get("width"))
    height = strict_int(manifest.get("height"))
    schema_valid = (
        manifest.get("schema_name") == "openmvs.dmap.prefilter_manifest"
        and manifest.get("schema_version") == 1
        and manifest.get("complete") is True
        and manifest.get("process_specialization") == "Process<false>"
        and summary.get("schema_name") == "openmvs.dmap.frame_summary"
        and summary.get("schema_version") == 4
    )
    check("prefilter_schema", schema_valid, {
        "manifest_schema": manifest.get("schema_name"),
        "manifest_version": manifest.get("schema_version"),
        "summary_schema": summary.get("schema_name"),
        "summary_version": summary.get("schema_version"),
        "process_specialization": manifest.get("process_specialization"),
    })
    patch_layout_valid, patch_layout_detail = (
        validate_reference_patch_layout_contract(frame_dir, summary)
    )
    check(
        "reference_patch_layout_contract",
        patch_layout_valid,
        patch_layout_detail,
    )
    check(
        "prefilter_dimensions",
        width is not None
        and width > 0
        and height is not None
        and height > 0
        and summary.get("width") == width
        and summary.get("height") == height,
        {"manifest": [width, height], "summary": [summary.get("width"), summary.get("height")]},
    )
    entries = manifest.get("maps")
    entries = entries if isinstance(entries, list) else []
    entry = entries[0] if len(entries) == 1 and isinstance(entries[0], dict) else {}
    relative_path = safe_relative_path(entry.get("path"))
    entry_valid = (
        len(entries) == 1
        and entry.get("signal") == "depth_final_before_filter"
        and relative_path == "maps/depth_final_before_filter.pfm"
        and entry.get("dtype") == "float32"
        and entry.get("measurement_quality", entry.get("quality")) == "exact"
        and entry.get("measurement_basis", entry.get("basis"))
        == "production_pre_filter_snapshot"
    )
    check("prefilter_map_contract", entry_valid, entry)
    depth_path = frame_dir / str(relative_path or "")
    path_valid = False
    depth: np.ndarray | None = None
    error = ""
    if relative_path is not None:
        try:
            resolved_depth, path_error = owned_regular_artifact_path(
                frame_dir, relative_path
            )
            path_valid = (
                path_error is None
                and resolved_depth is not None
                and strict_int(entry.get("bytes")) == resolved_depth.stat().st_size
            )
            if path_valid:
                depth_path = resolved_depth
                depth = read_pfm(depth_path, np)
            elif path_error:
                error = path_error
        except Exception as exc:
            error = str(exc)
    expected_shape = (height, width) if width is not None and height is not None else None
    map_valid = (
        path_valid
        and depth is not None
        and depth.shape == expected_shape
        and bool(np.isfinite(depth).all())
    )
    check("prefilter_map_file", map_valid, {
        "path": str(depth_path),
        "declared_bytes": entry.get("bytes"),
        "actual_bytes": depth_path.stat().st_size if depth_path.is_file() else None,
        "shape": list(depth.shape) if depth is not None else None,
        "error": error,
    })
    actual_maps = sorted(
        path.relative_to(frame_dir).as_posix()
        for path in frame_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pfm", ".png"}
    )
    check(
        "prefilter_indexes_all_maps",
        actual_maps == ["maps/depth_final_before_filter.pfm"],
        actual_maps,
    )

    sidecars_valid, sidecars = observer_sidecars_complete(summary)
    check("prefilter_observer_sidecars_complete", sidecars_valid, sidecars)
    marker_manifest = marker.get("manifest") if isinstance(marker.get("manifest"), dict) else {}
    marker_summary = marker.get("summary") if isinstance(marker.get("summary"), dict) else {}
    marker_valid = (
        marker.get("schema_name") == "openmvs.dmap.prefilter_capture_complete"
        and marker.get("schema_version") == 1
        and marker.get("capture_kind") == "prefilter"
        and marker.get("eligible") is True
        and marker.get("prefilter_complete") is True
        and marker.get("maps_complete") is True
        and marker.get("observer_sidecars_complete") is True
        and marker_manifest.get("path") == "prefilter_manifest.json"
        and marker_manifest.get("schema_version") == 1
        and strict_int(marker_manifest.get("bytes"))
        == (frame_dir / "prefilter_manifest.json").stat().st_size
        and marker_summary.get("path") == "summary.json"
        and marker_summary.get("schema_version") == 4
        and strict_int(marker_summary.get("bytes"))
        == (frame_dir / "summary.json").stat().st_size
        and marker.get("image_id") == summary.get("image_id")
        and marker.get("image_name") == summary.get("image_name")
        and marker.get("estimation_stage") == summary.get("estimation_stage")
        and marker.get("geometric_iteration") == summary.get("geometric_iteration")
    )
    check("prefilter_completion_marker", marker_valid, marker)
    expected_completion = {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "path": "prefilter_capture_complete.json",
        "maps_complete": True,
        "prefilter_complete": True,
        "eligible": True,
    }
    check(
        "prefilter_summary_completion_reference",
        summary.get("completion_marker") == expected_completion,
        {"actual": summary.get("completion_marker"), "expected": expected_completion},
    )
    resource = summary.get("resource_plan")
    resource = resource if isinstance(resource, dict) else {}
    resource_valid = (
        resource.get("summary_available") is True
        and resource.get("prefilter_requested") is True
        and resource.get("prefilter_available") is True
        and resource.get("maps_available") is False
        and resource.get("exact_available") is False
    )
    check("prefilter_resource_plan", resource_valid, resource)
    estimation_iterations = strict_int(
        (summary.get("cuda_patchmatch_parameters") or {}).get("estimation_iterations")
        if isinstance(summary.get("cuda_patchmatch_parameters"), dict)
        else None
    )
    expected_iterations = (
        list(range(-1, estimation_iterations))
        if estimation_iterations is not None and estimation_iterations >= 0 else []
    )
    census_valid, census_detail = validate_frame_census(
        frame_dir, summary, expected_iterations
    )
    if estimation_iterations is None or estimation_iterations < 0:
        census_valid = False
        census_detail["errors"].append(
            "summary cuda_patchmatch_parameters.estimation_iterations is invalid"
        )
    check("prefilter_process_census", census_valid, census_detail)
    unexpected = [
        path.name
        for path in (frame_dir / "map_manifest.json", frame_dir / "capture_complete.json")
        if path.exists()
    ]
    check("prefilter_excludes_deep_map_markers", not unexpected, unexpected)

    result = {
        "schema_name": "openmvs.dmap.prefilter_validation",
        "schema_version": 1,
        "frame_dir": str(frame_dir),
        "manifest": str(frame_dir / "prefilter_manifest.json"),
        "summary": str(frame_dir / "summary.json"),
        "valid": all(row["passed"] for row in checks),
        "prefilter_capture_available": map_valid and marker_valid and resource_valid,
        "exact_maps_available": False,
        "process_specialization": manifest.get("process_specialization"),
        "checks": checks,
    }
    result = machine_readable_result(result)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def bounded_domain_stats(values: list[np.ndarray], lower: float, upper: float) -> dict[str, int]:
    total = sum(int(value.size) for value in values)
    valid = sum(
        int((np.isfinite(value) & (value >= lower) & (value <= upper)).sum())
        for value in values
    )
    return {"pixels": total, "valid_domain_pixels": valid, "invalid_domain_pixels": total - valid}


def low_texture_decay_metadata(frame_dir: Path) -> dict[str, Any]:
    """Load the stage-local parameter needed to reproduce production prior weights."""

    path = frame_dir.parent.parent / "run_metadata.json"
    result: dict[str, Any] = {
        "available": False,
        "path": str(path),
        "low_texture_decay_scale": None,
        "unavailable_reason": "run_metadata.json is missing",
    }
    if not path.is_file():
        return result
    try:
        metadata = load_json(path)
    except Exception as exc:
        result["unavailable_reason"] = f"run_metadata.json could not be read: {exc}"
        return result
    if metadata.get("schema_name") != "openmvs.dmap.run" or strict_int(
        metadata.get("schema_version")
    ) != 4:
        result["unavailable_reason"] = "run_metadata.json is not an openmvs.dmap.run schema-v4 artifact"
        return result
    parameters = metadata.get("cuda_patchmatch_parameters")
    decay_scale = finite_number(
        parameters.get("low_texture_decay_scale") if isinstance(parameters, dict) else None
    )
    if decay_scale is None or decay_scale <= 0.0:
        result["unavailable_reason"] = "low_texture_decay_scale is missing, non-finite, or non-positive"
        return result
    result.update(
        available=True,
        low_texture_decay_scale=decay_scale,
        unavailable_reason=None,
    )
    return result


def exact_view_samples_metadata(frame_dir: Path) -> dict[str, Any]:
    """Load the captured Monte Carlo sample count used by exact view weights."""

    path = frame_dir.parent.parent / "run_metadata.json"
    result: dict[str, Any] = {
        "available": False,
        "path": str(path),
        "view_samples": None,
        "unavailable_reason": "run_metadata.json is missing",
    }
    if not path.is_file():
        return result
    try:
        metadata = load_json(path)
    except Exception as exc:
        result["unavailable_reason"] = f"run_metadata.json could not be read: {exc}"
        return result
    if metadata.get("schema_name") != "openmvs.dmap.run" or strict_int(
        metadata.get("schema_version")
    ) != 4:
        result["unavailable_reason"] = (
            "run_metadata.json is not an openmvs.dmap.run schema-v4 artifact"
        )
        return result
    parameters = metadata.get("cuda_patchmatch_parameters")
    raw_view_samples = (
        parameters.get("view_samples") if isinstance(parameters, dict) else None
    )
    view_samples = strict_int(raw_view_samples)
    if view_samples is None:
        result["unavailable_reason"] = "view_samples is missing or is not an integer"
        return result
    if not 1 <= view_samples <= 63:
        result["unavailable_reason"] = "view_samples is outside the supported range [1,63]"
        return result
    result.update(
        available=True,
        view_samples=view_samples,
        unavailable_reason=None,
    )
    return result


def low_texture_update_hysteresis_metadata(frame_dir: Path) -> dict[str, Any]:
    """Load configured controls and derive whether hysteresis actually executed."""

    path = frame_dir.parent.parent / "run_metadata.json"
    summary_path = frame_dir / "summary.json"
    result: dict[str, Any] = {
        "available": False,
        "enabled": False,
        "configured": False,
        "execution_available": False,
        "low_resolution_prior_available": None,
        "estimation_iterations": None,
        "path": str(path),
        "summary_path": str(summary_path),
        "low_texture_update_min_gain": None,
        "low_texture_update_gate": None,
        "configuration_source": None,
        "execution_availability_basis": None,
        "execution_unavailable_reason": None,
        "unavailable_reason": (
            "low-texture update hysteresis parameters are unavailable in both "
            "summary.json and run_metadata.json"
        ),
    }

    summary_parameters: dict[str, Any] | None = None
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
            raw_summary_parameters = summary.get("cuda_patchmatch_parameters")
            if isinstance(raw_summary_parameters, dict):
                summary_parameters = raw_summary_parameters
        except Exception:
            # The main validator reports malformed summary.json separately. Keep this
            # helper compatible with captures whose run metadata is still readable.
            pass

    run_parameters: dict[str, Any] | None = None
    if path.is_file():
        try:
            metadata = load_json(path)
        except Exception as exc:
            result["unavailable_reason"] = f"run_metadata.json could not be read: {exc}"
        else:
            if metadata.get("schema_name") == "openmvs.dmap.run" and strict_int(
                metadata.get("schema_version")
            ) == 4:
                raw_run_parameters = metadata.get("cuda_patchmatch_parameters")
                if isinstance(raw_run_parameters, dict):
                    run_parameters = raw_run_parameters
            elif summary_parameters is None:
                result["unavailable_reason"] = (
                    "run_metadata.json is not an openmvs.dmap.run schema-v4 artifact"
                )

    parameters: dict[str, Any] = {}
    parameter_sources: set[str] = set()
    for key in ("low_texture_update_min_gain", "low_texture_update_gate"):
        if summary_parameters is not None and key in summary_parameters:
            parameters[key] = summary_parameters[key]
            parameter_sources.add("summary.json")
        elif run_parameters is not None and key in run_parameters:
            parameters[key] = run_parameters[key]
            parameter_sources.add("run_metadata.json")

    min_gain = finite_number(parameters.get("low_texture_update_min_gain"))
    gate = strict_int(parameters.get("low_texture_update_gate"))
    if min_gain is None or min_gain < 0.0 or gate is None or gate not in {0, 1, 2, 3}:
        result["unavailable_reason"] = (
            "low_texture_update_min_gain or low_texture_update_gate is missing or invalid"
        )
        return result

    prior_available: bool | None = None
    if (
        summary_parameters is not None
        and "low_resolution_prior_available" in summary_parameters
    ):
        raw_prior_available = summary_parameters["low_resolution_prior_available"]
        if not isinstance(raw_prior_available, bool):
            result["unavailable_reason"] = (
                "summary cuda_patchmatch_parameters.low_resolution_prior_available "
                "must be a boolean"
            )
            return result
        prior_available = raw_prior_available

    estimation_iterations: int | None = None
    if summary_parameters is not None and "estimation_iterations" in summary_parameters:
        estimation_iterations = strict_int(summary_parameters["estimation_iterations"])
        if estimation_iterations is None or not 0 <= estimation_iterations <= 1024:
            result["unavailable_reason"] = (
                "summary cuda_patchmatch_parameters.estimation_iterations must be "
                "an integer in [0,1024]"
            )
            return result

    configured = min_gain > 0.0 and gate != 0
    execution_available = (
        configured
        and prior_available is not False
        and estimation_iterations != 0
    )
    execution_unavailable_reason = None
    if not configured:
        execution_unavailable_reason = (
            "low-texture update hysteresis is disabled by configuration"
        )
    elif prior_available is False:
        execution_unavailable_reason = (
            "configured mechanism did not execute because this level/stage has no "
            "coarse-resolution prior"
        )
    elif estimation_iterations == 0:
        execution_unavailable_reason = (
            "configured mechanism did not execute because estimation_iterations is zero"
        )
    result.update(
        available=True,
        # ``enabled`` is retained for schema-v4 validator compatibility. It now
        # means that the configured mechanism could execute for this frame.
        enabled=execution_available,
        configured=configured,
        execution_available=execution_available,
        low_resolution_prior_available=prior_available,
        estimation_iterations=estimation_iterations,
        low_texture_update_min_gain=min_gain,
        low_texture_update_gate=gate,
        configuration_source="+".join(sorted(parameter_sources)),
        execution_availability_basis=(
            "per_frame_summary_low_resolution_prior"
            if prior_available is not None
            else "legacy_configured_fallback"
        ),
        execution_unavailable_reason=execution_unavailable_reason,
        unavailable_reason=None,
    )
    return result


def depth_prior_weight_domain_stats(
    records: list[tuple[int, np.ndarray, np.ndarray | None]],
    decay_scale: float | None,
    decay_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify prior weights, retaining the known production overshoot as evidence."""

    total = 0
    valid = 0
    explained = 0
    raw_invalid = 0
    fatal_invalid = 0
    maximum_weight: float | None = None
    maximum_explained_weight: float | None = None
    minimum_explained_variance: float | None = None
    maximum_formula_abs_error = 0.0
    maximum_formula_ulp_error = 0
    per_iteration: dict[str, dict[str, Any]] = {}
    valid_decay = decay_scale is not None and np.isfinite(decay_scale) and decay_scale > 0.0
    decay = np.float32(decay_scale) if valid_decay else np.float32(np.nan)

    for iteration, weights_value, variance_value in records:
        weights = np.asarray(weights_value, dtype=np.float32)
        variances = (
            np.asarray(variance_value, dtype=np.float32)
            if variance_value is not None and variance_value.shape == weights.shape
            else None
        )
        finite_weights = np.isfinite(weights)
        in_domain = finite_weights & (weights >= 0.0) & (weights <= 1.0)
        explained_mask = np.zeros(weights.shape, dtype=bool)
        candidate_expected = np.empty(0, dtype=np.float32)
        candidate_weights = np.empty(0, dtype=np.float32)
        candidate_ulp_errors = np.empty(0, dtype=np.int64)
        if variances is not None and valid_decay:
            candidate = finite_weights & (weights > 1.0) & np.isfinite(variances) & (variances < 0.0)
            if np.any(candidate):
                smooth_sigma_depth = np.float32(-1.0) / decay
                exponent = variances[candidate] * smooth_sigma_depth
                with np.errstate(over="ignore", invalid="ignore"):
                    expected = np.exp(exponent.astype(np.float64)).astype(np.float32)
                candidate_expected = expected
                candidate_weights = weights[candidate]
                actual_bits = candidate_weights.view(np.uint32).astype(np.int64)
                expected_bits = candidate_expected.view(np.uint32).astype(np.int64)
                candidate_ulp_errors = np.abs(actual_bits - expected_bits)
                matches = np.isfinite(expected) & (
                    candidate_ulp_errors <= FLOAT32_EXP_ULP_TOLERANCE
                )
                explained_mask[candidate] = matches

        raw_invalid_mask = ~in_domain
        fatal_invalid_mask = raw_invalid_mask & ~explained_mask
        iteration_total = int(weights.size)
        iteration_valid = int(np.count_nonzero(in_domain))
        iteration_explained = int(np.count_nonzero(explained_mask))
        iteration_raw_invalid = int(np.count_nonzero(raw_invalid_mask))
        iteration_fatal_invalid = int(np.count_nonzero(fatal_invalid_mask))
        total += iteration_total
        valid += iteration_valid
        explained += iteration_explained
        raw_invalid += iteration_raw_invalid
        fatal_invalid += iteration_fatal_invalid

        finite_values = weights[finite_weights]
        if finite_values.size:
            current_max = float(np.max(finite_values))
            maximum_weight = current_max if maximum_weight is None else max(maximum_weight, current_max)
        if iteration_explained:
            explained_weights = weights[explained_mask]
            explained_variances = variances[explained_mask] if variances is not None else np.empty(0)
            current_weight = float(np.max(explained_weights))
            current_variance = float(np.min(explained_variances))
            maximum_explained_weight = (
                current_weight
                if maximum_explained_weight is None
                else max(maximum_explained_weight, current_weight)
            )
            minimum_explained_variance = (
                current_variance
                if minimum_explained_variance is None
                else min(minimum_explained_variance, current_variance)
            )
            matched = candidate_ulp_errors <= FLOAT32_EXP_ULP_TOLERANCE
            if np.any(matched):
                maximum_formula_abs_error = max(
                    maximum_formula_abs_error,
                    float(np.max(np.abs(candidate_weights[matched] - candidate_expected[matched]))),
                )
                maximum_formula_ulp_error = max(
                    maximum_formula_ulp_error,
                    int(np.max(candidate_ulp_errors[matched])),
                )
        per_iteration[str(iteration)] = {
            "pixels": iteration_total,
            "valid_domain_pixels": iteration_valid,
            "known_production_warning_pixels": iteration_explained,
            "invalid_domain_pixels": iteration_raw_invalid,
            "fatal_invalid_domain_pixels": iteration_fatal_invalid,
        }

    warning = None
    if explained:
        warning = {
            "code": NEGATIVE_VARIANCE_PRIOR_WEIGHT_WARNING,
            "pixels": explained,
            "maximum_weight": maximum_explained_weight,
            "maximum_overshoot": (
                maximum_explained_weight - 1.0
                if maximum_explained_weight is not None else None
            ),
            "minimum_reference_variance": minimum_explained_variance,
            "low_texture_decay_scale": decay_scale,
            "parameter_source": decay_metadata,
            "formula": "exp(reference_variance * (-1 / low_texture_decay_scale))",
            "maximum_formula_abs_error": maximum_formula_abs_error,
            "maximum_formula_ulp_error": maximum_formula_ulp_error,
            "classification": "known_production_numeric_domain_violation",
        }
    return {
        "pixels": total,
        "valid_domain_pixels": valid,
        "known_production_warning_pixels": explained,
        "invalid_domain_pixels": raw_invalid,
        "fatal_invalid_domain_pixels": fatal_invalid,
        "maximum_weight": maximum_weight,
        "low_texture_decay_scale": decay_scale,
        "parameter_source": decay_metadata,
        "float32_exp_ulp_tolerance": FLOAT32_EXP_ULP_TOLERANCE,
        "maximum_formula_abs_error": maximum_formula_abs_error,
        "maximum_formula_ulp_error": maximum_formula_ulp_error,
        "per_iteration": per_iteration,
        "warning": warning,
    }


def gap_domain_stats(values: list[np.ndarray]) -> dict[str, int]:
    total = sum(int(value.size) for value in values)
    unavailable = sum(int((np.isfinite(value) & (value == -1.0)).sum()) for value in values)
    nonnegative = sum(int((np.isfinite(value) & (value >= 0.0)).sum()) for value in values)
    return {
        "pixels": total,
        "unavailable_pixels": unavailable,
        "finite_nonnegative_pixels": nonnegative,
        "invalid_domain_pixels": total - unavailable - nonnegative,
    }


def safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def owned_regular_artifact_path(
    root: Path, value: Any, *, require_exists: bool = True
) -> tuple[Path | None, str | None]:
    """Resolve one manifest artifact without following links outside its owner."""

    relative = safe_relative_path(value)
    if relative is None:
        return None, f"unsafe relative artifact path: {value!r}"
    owner = root.resolve()
    candidate = owner / relative
    current = owner
    for part in Path(relative).parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if require_exists:
                return None, f"artifact is missing: {relative}"
            return candidate, None
        except OSError as exc:
            return None, f"cannot inspect artifact {relative}: {exc}"
        if stat.S_ISLNK(metadata.st_mode):
            return None, f"artifact path contains a symlink: {relative}"
    try:
        candidate.resolve(strict=require_exists).relative_to(owner)
    except (OSError, ValueError):
        return None, f"artifact escapes its owning directory: {relative}"
    if require_exists:
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            return None, f"cannot inspect artifact {relative}: {exc}"
        if not stat.S_ISREG(metadata.st_mode):
            return None, f"artifact is not a regular file: {relative}"
    return candidate, None


def validate_reference_patch_layout_contract(
    frame_dir: Path,
    summary: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Fail closed when a producer claims the patch-layout contract."""

    instrumentation_root = (
        frame_dir.parent.parent
        if frame_dir.parent.name == "depthmaps" else frame_dir.parent
    )
    metadata_candidate = instrumentation_root / "run_metadata.json"
    detail: dict[str, Any] = {
        "path": str(metadata_candidate),
        "claimed": False,
        "status": "legacy_unclaimed",
        "errors": [],
    }
    if not metadata_candidate.exists() and not metadata_candidate.is_symlink():
        detail["unavailable_reason"] = "run_metadata.json is missing"
        return True, detail
    metadata_path, path_error = owned_regular_artifact_path(
        instrumentation_root, "run_metadata.json"
    )
    if path_error or metadata_path is None:
        detail["status"] = "invalid_metadata"
        detail["errors"].append(path_error or "run_metadata.json is unavailable")
        return False, detail
    try:
        metadata = load_json(metadata_path)
    except Exception as exc:
        detail["status"] = "invalid_metadata"
        detail["errors"].append(f"run_metadata.json could not be read: {exc}")
        return False, detail

    instrumentation = metadata.get("instrumentation")
    capabilities = (
        instrumentation.get("capabilities")
        if isinstance(instrumentation, dict) else None
    )
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    capability_name = "reference_patch_layout_contract"
    if capability_name not in capabilities:
        detail["unavailable_reason"] = "layout capability is not declared"
        return True, detail
    claimed = capabilities.get(capability_name)
    if not isinstance(claimed, bool):
        detail["status"] = "invalid_capability"
        detail["errors"].append(
            "reference_patch_layout_contract capability must be boolean"
        )
        return False, detail
    detail["claimed"] = claimed
    if not claimed:
        detail["unavailable_reason"] = "layout capability is explicitly disabled"
        return True, detail

    detail["status"] = "claimed"
    if (
        metadata.get("schema_name") != "openmvs.dmap.run"
        or strict_int(metadata.get("schema_version")) != 4
    ):
        detail["errors"].append(
            "claimed layout requires openmvs.dmap.run schema v4"
        )
    parameters = metadata.get("cuda_patchmatch_parameters")
    run_layout, run_reason = reference_patch_layout.normalize(
        parameters.get("reference_patch_layout")
        if isinstance(parameters, dict) else None
    )
    summary_parameters = summary.get("cuda_patchmatch_parameters")
    summary_layout, summary_reason = reference_patch_layout.normalize(
        summary_parameters.get("reference_patch_layout")
        if isinstance(summary_parameters, dict) else None
    )
    if run_layout is None:
        detail["errors"].append(f"run metadata: {run_reason}")
    if summary_layout is None:
        detail["errors"].append(f"frame summary: {summary_reason}")
    if run_layout is not None and summary_layout is not None and run_layout != summary_layout:
        detail["errors"].append("run metadata and frame summary layouts differ")

    for name in (
        "reference_patch_sample_locations",
        "reference_patch_sample_values",
        "source_view_patch_footprints",
    ):
        if capabilities.get(name) is not False:
            detail["errors"].append(f"schema-v1 capability {name} must be false")
    if run_layout is not None:
        detail["layout"] = {
            "schema_name": run_layout["schema_name"],
            "schema_version": run_layout["schema_version"],
            "sample_count": run_layout["sample_count"],
            "half_window_pixels": run_layout["half_window_pixels"],
            "step_pixels": run_layout["step_pixels"],
            "texture_address_mode_configured": run_layout[
                "texture_address_mode_configured"
            ],
            "texture_address_mode_effective": run_layout[
                "texture_address_mode_effective"
            ],
        }
    valid = not detail["errors"]
    detail["status"] = "valid" if valid else "invalid_claimed_contract"
    return valid, detail


def optional_manifest_maps(
    *,
    frame_dir: Path,
    artifact: dict[str, Any],
    artifact_name: str,
    expected_schema_name: str,
    expected_schema_versions: frozenset[int],
    owned_directory: str,
    width: int,
    height: int,
    core_declared_paths: set[str],
    other_optional_paths: set[str],
    check: Any,
) -> tuple[list[tuple[dict[str, Any], Path, np.ndarray]], set[str]]:
    """Validate and load maps owned by one optional-stage manifest."""
    prefix = Path(artifact_name).stem
    schema_valid = (
        artifact.get("schema_name") == expected_schema_name
        and strict_int(artifact.get("schema_version")) in expected_schema_versions
    )
    check(f"{prefix}_schema", schema_valid, {
        "schema_name": artifact.get("schema_name"),
        "schema_version": artifact.get("schema_version"),
        "expected_schema_name": expected_schema_name,
        "expected_schema_versions": sorted(expected_schema_versions),
    })

    entries_value = artifact.get("maps")
    entries = entries_value if isinstance(entries_value, list) else []
    entry_errors = [
        f"entry {index}: expected JSON object"
        for index, entry in enumerate(entries)
        if not isinstance(entry, dict)
    ]
    entries = [entry for entry in entries if isinstance(entry, dict)]
    declared_paths = [safe_relative_path(entry.get("path")) for entry in entries]
    unsafe_paths = [
        str(entry.get("path", ""))
        for entry, path in zip(entries, declared_paths)
        if path is None or not Path(path).parts or Path(path).parts[0] != owned_directory
    ]
    valid_declared_paths = [path for path in declared_paths if path is not None]
    duplicate_paths = sorted({
        path for path in valid_declared_paths if valid_declared_paths.count(path) > 1
    })
    ownership_conflicts = sorted(
        set(valid_declared_paths) & (core_declared_paths | other_optional_paths)
    )
    check(f"{prefix}_paths_safe_unique", (
        not entry_errors and not unsafe_paths and not duplicate_paths and not ownership_conflicts
    ), {
        "entry_errors": entry_errors,
        "unsafe_paths": unsafe_paths,
        "duplicate_paths": duplicate_paths,
        "ownership_conflicts": ownership_conflicts,
    })

    owned_root = frame_dir / owned_directory
    actual_paths: set[str] = set()
    unsafe_actual_paths: list[str] = []
    if owned_root.is_dir():
        for path in owned_root.rglob("*"):
            if path.suffix.lower() not in {".pfm", ".png"}:
                continue
            relative = path.relative_to(frame_dir).as_posix()
            _resolved, path_error = owned_regular_artifact_path(frame_dir, relative)
            if path_error:
                if path.exists() or path.is_symlink():
                    unsafe_actual_paths.append(f"{relative}: {path_error}")
                continue
            actual_paths.add(relative)
    declared_path_set = set(valid_declared_paths)
    unindexed_paths = sorted(actual_paths - declared_path_set)
    stale_paths = sorted(declared_path_set - actual_paths)
    check(
        f"{prefix}_indexes_all_map_files",
        not unindexed_paths and not stale_paths and not unsafe_actual_paths,
        {
        "unindexed_paths": unindexed_paths,
        "stale_paths": stale_paths,
        "unsafe_paths": unsafe_actual_paths,
        },
    )

    byte_errors: list[str] = []
    shape_errors: list[str] = []
    loaded: list[tuple[dict[str, Any], Path, np.ndarray]] = []
    for entry, relative_path in zip(entries, declared_paths):
        if relative_path is None:
            continue
        path, path_error = owned_regular_artifact_path(frame_dir, relative_path)
        if path_error or path is None:
            shape_errors.append(f"{relative_path}: {path_error}")
            continue
        file_size = path.stat().st_size
        legacy_bytes = strict_int(entry.get("bytes"))
        file_bytes = strict_int(entry.get("file_bytes"))
        declared_bytes = strict_int(entry.get("declared_bytes"))
        if legacy_bytes is not None:
            if legacy_bytes <= 0 or legacy_bytes != file_size:
                byte_errors.append(
                    f"{relative_path}: bytes={legacy_bytes} does not match file size {file_size}"
                )
        else:
            if declared_bytes is None or declared_bytes <= 0:
                byte_errors.append(f"{relative_path}: declared_bytes must be a positive integer")
            if file_bytes is None or file_bytes <= 0 or file_bytes != file_size:
                byte_errors.append(
                    f"{relative_path}: file_bytes={entry.get('file_bytes')} does not match file size {file_size}"
                )
            if entry.get("file_size_available") is not True:
                byte_errors.append(f"{relative_path}: file_size_available must be true")
        try:
            data = read_manifest_map(frame_dir, entry)
        except Exception as exc:
            shape_errors.append(f"{relative_path}: {exc}")
            continue
        dtype = entry.get("dtype")
        expected_shape = (
            (height, width, 3) if dtype in {"float32x3", "uint8x3"}
            else (height, width, 4) if dtype == "uint8x4"
            else (height, width)
        )
        if data.shape != expected_shape:
            shape_errors.append(f"{relative_path}: expected {expected_shape}, got {data.shape}")
        loaded.append((entry, path, data))
    check(f"{prefix}_file_bytes", not byte_errors, byte_errors)
    check(f"{prefix}_map_shapes", not shape_errors, shape_errors)

    write_errors = artifact.get("write_errors")
    maps_requested = artifact.get("maps_requested")
    maps_enabled = artifact.get("maps_enabled")
    completion_valid = (
        artifact.get("complete") is True
        and isinstance(write_errors, list)
        and not write_errors
        and isinstance(entries_value, list)
        and not stale_paths
        and not unindexed_paths
        and not byte_errors
        and not shape_errors
        and not (maps_enabled is False and entries)
        and not (maps_requested is False and maps_enabled is True)
    )
    check(f"{prefix}_complete", completion_valid, {
        "complete": artifact.get("complete"),
        "maps_requested": maps_requested,
        "maps_enabled": maps_enabled,
        "map_count": len(entries),
        "write_errors": write_errors,
    })
    return loaded, actual_paths


def find_stage_map(
    records: list[tuple[dict[str, Any], Path, np.ndarray]],
    *,
    stage_name: str,
    signal_suffix: str,
) -> tuple[dict[str, Any], Path, np.ndarray] | None:
    candidates = [
        record for record in records
        if str(record[0].get("algorithm_stage", "")) == stage_name
        and str(record[0].get("signal", "")).endswith(signal_suffix)
    ]
    return candidates[-1] if candidates else None


def postprocess_terminal_channel(
    *,
    channel: str,
    artifact: dict[str, Any] | None,
    records: list[tuple[dict[str, Any], Path, np.ndarray]],
    fallback: np.ndarray | None,
    fallback_source: str,
) -> dict[str, Any]:
    state = {
        "array": fallback,
        "source": fallback_source if fallback is not None else None,
        "available": fallback is not None,
        "unavailable_reason": None if fallback is not None else f"core terminal {channel} map unavailable",
    }
    if artifact is None:
        return state
    stages = [stage for stage in artifact.get("stages") or [] if isinstance(stage, dict)]
    stages.sort(key=lambda stage: strict_int(stage.get("stage_index")) or 0)
    for stage in stages:
        if not bool(stage.get("executed")):
            continue
        stage_name = str(stage.get("name", ""))
        terminal = find_stage_map(
            records, stage_name=stage_name, signal_suffix=f"_{channel}_after"
        )
        if terminal is not None:
            state = {
                "array": terminal[2],
                "source": terminal[1].relative_to(terminal[1].parents[1]).as_posix(),
                "available": True,
                "unavailable_reason": None,
            }
            continue
        metrics = stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {}
        changed = strict_int(metrics.get(f"{channel}_changed_pixels"))
        delta_records = [
            record for record in records
            if str(record[0].get("algorithm_stage", "")) == stage_name
            and str(record[0].get("signal", "")).endswith(f"_{channel}_delta")
        ]
        delta_changed = any(np.count_nonzero(record[2]) for record in delta_records)
        legacy_normal_unavailable = (
            channel == "normal"
            and strict_int(artifact.get("schema_version")) == 1
            and terminal is None
        )
        # Schema v1 cannot prove terminal-normal identity; schema v2 declares
        # normal state explicitly.
        known_unchanged = changed == 0
        if legacy_normal_unavailable or changed not in (None, 0) or delta_changed or not known_unchanged:
            state = {
                "array": None,
                "source": None,
                "available": False,
                "unavailable_reason": (
                    f"postprocess stage {stage_name} changed or could not prove identity for "
                    f"{channel}, but did not provide a terminal {channel}_after map"
                ),
            }
    return state


def confidence_adjustment_terminal(
    *,
    artifact: dict[str, Any] | None,
    records: list[tuple[dict[str, Any], Path, np.ndarray]],
    fallback_state: dict[str, Any],
) -> dict[str, Any]:
    if artifact is None:
        return fallback_state
    final_method = next(
        (
            method for method in artifact.get("methods") or []
            if isinstance(method, dict) and method.get("name") == "final_combined_confidence"
        ),
        None,
    )
    if not isinstance(final_method, dict) or not bool(final_method.get("enabled")):
        return fallback_state
    if not bool(final_method.get("executed")) or not bool(final_method.get("output_available")):
        return {
            "array": None,
            "source": None,
            "available": False,
            "unavailable_reason": "enabled final confidence adjustment output is unavailable",
        }
    final_record = next(
        (
            record for record in records
            if str(record[0].get("signal", "")) == "confidence_final"
        ),
        None,
    )
    if final_record is not None:
        return {
            "array": final_record[2],
            "source": final_record[1].relative_to(final_record[1].parents[1]).as_posix(),
            "available": True,
            "unavailable_reason": None,
        }
    metrics = final_method.get("metrics") if isinstance(final_method.get("metrics"), dict) else {}
    if strict_int(metrics.get("changed_pixels")) == 0:
        return fallback_state
    return {
        "array": None,
        "source": None,
        "available": False,
        "unavailable_reason": (
            "final confidence adjustment changed confidence but did not provide "
            "confidence_final"
        ),
    }


def validate_v3_logical_states(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    duplicate_keys: list[str],
    width: int,
    height: int,
    tolerance: float,
    low_texture_decay_scale: float | None,
    decay_metadata: dict[str, Any] | None,
    expect_geometric_zero: bool,
    check: Any,
) -> dict[str, Any]:
    component_tolerance = max(tolerance, FLOAT32_COMPONENT_CLOSURE_TOLERANCE)
    num_iterations = strict_int(manifest.get("num_iterations"))
    num_logical_states = strict_int(manifest.get("num_logical_states"))
    contract_valid = (
        manifest.get("schema_name") == V3_SCHEMA_NAME
        and num_iterations is not None
        and num_iterations >= 0
        and num_logical_states == num_iterations + 1
        and manifest.get("map_granularity") == "logical_iteration"
    )
    check("v3_logical_state_contract", contract_valid, {
        "schema_name": manifest.get("schema_name"),
        "num_iterations": manifest.get("num_iterations"),
        "num_logical_states": manifest.get("num_logical_states"),
        "map_granularity": manifest.get("map_granularity"),
    })

    expected_iterations = (
        set(range(-1, num_iterations))
        if num_iterations is not None and num_iterations >= 0
        else set()
    )
    present_by_signal = {
        signal: {iteration for key_signal, iteration in logical_maps if key_signal == signal}
        for signal in V3_REQUIRED_LOGICAL_STATE_SIGNALS
    }
    missing = {
        signal: sorted(expected_iterations - present)
        for signal, present in present_by_signal.items()
        if expected_iterations - present
    }
    unexpected = {
        signal: sorted(present - expected_iterations)
        for signal, present in present_by_signal.items()
        if present - expected_iterations
    }
    check("v3_logical_state_coverage", contract_valid and not missing and not unexpected, {
        "expected_logical_iterations": sorted(expected_iterations),
        "missing": missing,
        "unexpected": unexpected,
    })
    logical_duplicates = [
        value for value in duplicate_keys
        if any(value.startswith(f"{signal}:logical:") for signal in V3_REQUIRED_LOGICAL_STATE_SIGNALS)
    ]
    check("v3_logical_state_unique", not logical_duplicates, logical_duplicates)

    required_entries = [
        entry for entry in entries
        if str(entry.get("signal", "")) in V3_REQUIRED_LOGICAL_STATE_SIGNALS
    ]
    metadata_errors: list[str] = []
    phase_errors: list[str] = []
    layout_errors: list[str] = []
    byte_errors: list[str] = []
    required_paths: dict[str, list[str]] = {}
    for entry in required_entries:
        signal = str(entry.get("signal", ""))
        iteration = strict_int(entry.get("logical_iteration"))
        label = f"{signal}:{entry.get('logical_iteration', 'missing')}"
        if iteration is None:
            metadata_errors.append(f"{label} logical_iteration must be an integer")
        else:
            expected_stage = "initialization" if iteration == -1 else "iteration"
            if entry.get("stage") != expected_stage:
                metadata_errors.append(f"{label} stage must be {expected_stage!r}")
        if entry.get("role") != "logical_state":
            metadata_errors.append(f"{label} role must be 'logical_state'")
        quality = entry.get("measurement_quality")
        expected_quality = V3_MEASUREMENT_QUALITY[signal]
        if quality != expected_quality:
            metadata_errors.append(
                f"{label} measurement_quality must be {expected_quality!r}, got {quality!r}"
            )
        basis = entry.get("measurement_basis")
        expected_basis = V3_EXACT_MEASUREMENT_BASES.get(signal)
        if expected_basis is not None and basis != expected_basis:
            metadata_errors.append(
                f"{label} measurement_basis must be {expected_basis!r}, got {basis!r}"
            )
        elif signal in V3_PROXY_LOGICAL_STATE_SIGNALS and basis != V3_PROXY_MEASUREMENT_BASIS:
            metadata_errors.append(
                f"{label} measurement_basis must be {V3_PROXY_MEASUREMENT_BASIS!r}, got {basis!r}"
            )
        elif expected_basis is None and signal not in V3_PROXY_LOGICAL_STATE_SIGNALS and not nonempty_text(basis):
            metadata_errors.append(f"{label} measurement_basis must be non-empty")
        if signal in V3_PROXY_LOGICAL_STATE_SIGNALS:
            if not nonempty_text(entry.get("proxy_target")):
                metadata_errors.append(f"{label} proxy_target must be non-empty")
            if not nonempty_text(entry.get("limitations")):
                metadata_errors.append(f"{label} limitations must be non-empty")
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        phase_fields = sorted(key for key in entry if "phase" in str(key).lower())
        if phase_fields:
            phase_errors.append(f"{label} contains phase metadata: {', '.join(phase_fields)}")

        relative_path = str(entry.get("path", ""))
        required_paths.setdefault(relative_path, []).append(label)
        path = Path(relative_path)
        if entry.get("dtype") != "float32":
            layout_errors.append(f"{label} dtype must be 'float32'")
        if path.suffix.lower() != ".pfm":
            layout_errors.append(f"{label} path must reference a PFM map")
        absolute_path = (
            logical_maps.get((signal, iteration), (entry, Path(), np.empty(0)))[1]
            if iteration is not None else Path()
        )
        byte_count = strict_int(entry.get("bytes"))
        if byte_count is None or byte_count <= 0:
            byte_errors.append(f"{label} bytes must be a positive integer")
        elif absolute_path.is_file() and byte_count != absolute_path.stat().st_size:
            byte_errors.append(
                f"{label} bytes={byte_count} does not match file size {absolute_path.stat().st_size}"
            )
        uncompressed_bytes = strict_int(entry.get("uncompressed_bytes"))
        expected_uncompressed_bytes = width * height * 4
        if uncompressed_bytes != expected_uncompressed_bytes:
            byte_errors.append(
                f"{label} uncompressed_bytes must be {expected_uncompressed_bytes}, got {uncompressed_bytes!r}"
            )

    duplicate_paths = {
        path: labels for path, labels in required_paths.items()
        if not path or len(labels) != 1
    }
    if duplicate_paths:
        layout_errors.extend(
            f"logical-state path {path!r} is used by {labels}" for path, labels in duplicate_paths.items()
        )
    for (signal, iteration), (entry, _path, data) in logical_maps.items():
        if signal not in V3_REQUIRED_LOGICAL_STATE_SIGNALS:
            continue
        label = f"{signal}:{iteration}"
        if data.shape != (height, width):
            layout_errors.append(f"{label} expected {(height, width)}, got {data.shape}")
        if data.dtype.kind != "f" or data.dtype.itemsize != 4:
            layout_errors.append(f"{label} expected float32 data, got {data.dtype}")
    check("v3_logical_state_metadata", not metadata_errors, metadata_errors)
    check("v3_logical_state_phase_free", not phase_errors, phase_errors)
    check("v3_logical_state_layout", not layout_errors, layout_errors)
    check("v3_logical_state_bytes", not byte_errors, byte_errors)

    def signal_values(signal: str) -> list[np.ndarray]:
        return [
            logical_maps[(signal, iteration)][2]
            for iteration in sorted(expected_iterations)
            if (signal, iteration) in logical_maps
        ]

    confidence_stats = bounded_domain_stats(signal_values("confidence_stored"), 0.0, 1.0)
    weight_records = []
    for iteration in sorted(expected_iterations):
        weight_record = logical_maps.get(
            ("depth_prior_weight_equal_selected_rescore_proxy", iteration)
        )
        variance_record = logical_maps.get(
            ("reference_variance_equal_selected_rescore_proxy", iteration)
        )
        if weight_record is not None:
            weight_records.append(
                (
                    iteration,
                    weight_record[2],
                    variance_record[2] if variance_record is not None else None,
                )
            )
    weight_stats = depth_prior_weight_domain_stats(
        weight_records, low_texture_decay_scale, decay_metadata
    )
    local_gap_stats = gap_domain_stats(signal_values("gap_local_neighbor_equal_selected_rescore_proxy"))
    check(
        "v3_confidence_domain",
        confidence_stats["pixels"] > 0 and confidence_stats["invalid_domain_pixels"] == 0,
        confidence_stats,
    )
    check(
        "v3_depth_prior_weight_domain",
        weight_stats["pixels"] > 0 and weight_stats["fatal_invalid_domain_pixels"] == 0,
        weight_stats,
    )
    check(
        "v3_gap_domain",
        local_gap_stats["pixels"] > 0 and local_gap_stats["invalid_domain_pixels"] == 0,
        local_gap_stats,
    )

    confidence_definition: dict[str, float] = {}
    component_closure: dict[str, float] = {}
    residual_definition: dict[str, float] = {}
    stored_vs_proxy: dict[str, float] = {}
    geometric_max_abs = 0.0
    for iteration in sorted(expected_iterations):
        stage = str(iteration)
        cost = logical_maps.get(("cost_stored", iteration))
        confidence = logical_maps.get(("confidence_stored", iteration))
        photo_prior = logical_maps.get(("cost_photo_prior_equal_selected_rescore_proxy", iteration))
        geometric = logical_maps.get(("cost_geometric_equal_selected_rescore_proxy", iteration))
        total = logical_maps.get(("cost_total_equal_selected_rescore_proxy", iteration))
        residual = logical_maps.get(("cost_stored_minus_rescore", iteration))
        if cost is not None and confidence is not None:
            confidence_definition[stage] = max_abs_difference(
                confidence[2], np.maximum(1.0 - cost[2], 0.0)
            )
        if photo_prior is not None and geometric is not None and total is not None:
            component_closure[stage] = max_abs_difference(total[2], photo_prior[2] + geometric[2])
            finite_geometric = geometric[2][np.isfinite(geometric[2])]
            if finite_geometric.size:
                geometric_max_abs = max(geometric_max_abs, float(np.max(np.abs(finite_geometric))))
        if cost is not None and total is not None:
            stored_vs_proxy[stage] = max_abs_difference(cost[2], total[2])
            if residual is not None:
                residual_definition[stage] = max_abs_difference(residual[2], cost[2] - total[2])
    check(
        "v3_confidence_definition",
        set(confidence_definition) == {str(value) for value in expected_iterations}
        and all(value <= tolerance for value in confidence_definition.values()),
        confidence_definition,
    )
    check(
        "v3_proxy_component_closure",
        set(component_closure) == {str(value) for value in expected_iterations}
        and all(value <= component_tolerance for value in component_closure.values()),
        {"tolerance": component_tolerance, "per_iteration": component_closure},
    )
    check(
        "v3_stored_minus_rescore_definition",
        set(residual_definition) == {str(value) for value in expected_iterations}
        and all(value <= tolerance for value in residual_definition.values()),
        residual_definition,
    )
    if expect_geometric_zero:
        check("v3_geometric_disabled_zero", geometric_max_abs <= tolerance, geometric_max_abs)

    return {
        "num_iterations": num_iterations,
        "num_logical_states": num_logical_states,
        "expected_logical_iterations": sorted(expected_iterations),
        "confidence_domain": confidence_stats,
        "depth_prior_weight_domain": weight_stats,
        "gap_domain": local_gap_stats,
        "confidence_definition_max_abs": confidence_definition,
        "component_closure_max_abs": component_closure,
        "stored_minus_rescore_definition_max_abs": residual_definition,
        "stored_vs_proxy_max_abs_diagnostic_only": stored_vs_proxy,
        "geometric_max_abs": geometric_max_abs,
    }


def validate_v3_logical_events(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    duplicate_keys: list[str],
    width: int,
    height: int,
    tolerance: float,
    check: Any,
) -> dict[str, Any]:
    num_iterations = strict_int(manifest.get("num_iterations"))
    expected_iterations = (
        set(range(-1, num_iterations))
        if num_iterations is not None and num_iterations >= 0
        else set()
    )
    present_by_signal = {
        signal: {iteration for key_signal, iteration in logical_maps if key_signal == signal}
        for signal in V3_REQUIRED_LOGICAL_EVENT_SIGNALS
    }
    missing = {
        signal: sorted(expected_iterations - present)
        for signal, present in present_by_signal.items()
        if expected_iterations - present
    }
    unexpected = {
        signal: sorted(present - expected_iterations)
        for signal, present in present_by_signal.items()
        if present - expected_iterations
    }
    check(
        "v3_logical_event_coverage",
        num_iterations is not None and num_iterations >= 0 and not missing and not unexpected,
        {
            "expected_logical_iterations": sorted(expected_iterations),
            "missing": missing,
            "unexpected": unexpected,
        },
    )
    logical_duplicates = [
        value for value in duplicate_keys
        if any(value.startswith(f"{signal}:logical:") for signal in V3_REQUIRED_LOGICAL_EVENT_SIGNALS)
    ]
    check("v3_logical_event_unique", not logical_duplicates, logical_duplicates)

    required_entries = [
        entry for entry in entries
        if str(entry.get("signal", "")) in V3_REQUIRED_LOGICAL_EVENT_SIGNALS
    ]
    metadata_errors: list[str] = []
    phase_errors: list[str] = []
    layout_errors: list[str] = []
    paths: dict[str, list[str]] = {}
    for entry in required_entries:
        signal = str(entry.get("signal", ""))
        iteration = strict_int(entry.get("logical_iteration"))
        label = f"{signal}:{entry.get('logical_iteration', 'missing')}"
        if iteration is None:
            metadata_errors.append(f"{label} logical_iteration must be an integer")
        else:
            expected_stage = "initialization" if iteration == -1 else "iteration"
            if entry.get("stage") != expected_stage:
                metadata_errors.append(f"{label} stage must be {expected_stage!r}")
        if entry.get("role") != "logical_event":
            metadata_errors.append(f"{label} role must be 'logical_event'")
        if entry.get("measurement_quality") != "exact":
            metadata_errors.append(f"{label} measurement_quality must be 'exact'")
        if entry.get("measurement_basis") != "post_pass_production_state_difference":
            metadata_errors.append(
                f"{label} measurement_basis must be 'post_pass_production_state_difference'"
            )
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        phase_fields = sorted(key for key in entry if "phase" in str(key).lower())
        if phase_fields:
            phase_errors.append(f"{label} contains phase metadata: {', '.join(phase_fields)}")

        relative_path = str(entry.get("path", ""))
        paths.setdefault(relative_path, []).append(label)
        expected_dtype = "uint8" if signal == "view_churn" else "float32"
        expected_suffix = ".png" if signal == "view_churn" else ".pfm"
        if entry.get("dtype") != expected_dtype:
            layout_errors.append(f"{label} dtype must be {expected_dtype!r}")
        if Path(relative_path).suffix.lower() != expected_suffix:
            layout_errors.append(f"{label} path must end in {expected_suffix!r}")
    for path, labels in paths.items():
        if not path or len(labels) != 1:
            layout_errors.append(f"logical-event path {path!r} is used by {labels}")
    for (signal, iteration), (_entry, _path, data) in logical_maps.items():
        if signal not in V3_REQUIRED_LOGICAL_EVENT_SIGNALS:
            continue
        label = f"{signal}:{iteration}"
        if data.shape != (height, width):
            layout_errors.append(f"{label} expected {(height, width)}, got {data.shape}")
        if signal == "view_churn":
            if data.dtype.kind != "u" or data.dtype.itemsize != 1:
                layout_errors.append(f"{label} expected uint8 data, got {data.dtype}")
        elif data.dtype.kind != "f" or data.dtype.itemsize != 4:
            layout_errors.append(f"{label} expected float32 data, got {data.dtype}")
    check("v3_logical_event_metadata", not metadata_errors, metadata_errors)
    check("v3_logical_event_phase_free", not phase_errors, phase_errors)
    check("v3_logical_event_layout", not layout_errors, layout_errors)

    initialization_zero: dict[str, dict[str, float | int]] = {}
    for signal in ("depth_delta", "depth_relative_delta", "normal_angle_delta"):
        record = logical_maps.get((signal, -1))
        if record is None:
            continue
        values = record[2]
        finite = np.isfinite(values)
        initialization_zero[signal] = {
            "nonfinite_pixels": int((~finite).sum()),
            "nonzero_pixels": int(np.count_nonzero(values[finite])) if finite.any() else 0,
            "max_abs": float(np.max(np.abs(values[finite]))) if finite.any() else float("inf"),
        }
    initialization_valid = (
        set(initialization_zero) == {"depth_delta", "depth_relative_delta", "normal_angle_delta"}
        and all(
            row["nonfinite_pixels"] == 0 and row["max_abs"] <= tolerance
            for row in initialization_zero.values()
        )
    )
    check("v3_logical_event_initialization_zero", initialization_valid, initialization_zero)

    return {
        "expected_logical_iterations": sorted(expected_iterations),
        "initialization_zero": initialization_zero,
    }


def validate_logical_cost_improvement(
    *,
    frame_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    duplicate_keys: list[str],
    width: int,
    height: int,
    tolerance: float,
    check: Any,
) -> dict[str, Any]:
    """Validate the optional complete-iteration production-cost reduction maps."""

    signal = LOGICAL_COST_IMPROVEMENT_SIGNAL
    declared = [entry for entry in entries if entry.get("signal") == signal]
    if not declared:
        detail = {
            "available": False,
            "unavailable_reason": "capture predates or did not export the optional logical cost-improvement signal",
        }
        check("logical_cost_improvement_availability", True, detail)
        return detail

    check("logical_cost_improvement_availability", True, {"available": True})
    num_iterations = strict_int(manifest.get("num_iterations"))
    expected_iterations = (
        set(range(-1, num_iterations))
        if num_iterations is not None and num_iterations >= 0
        else set()
    )
    present_iterations = {
        iteration
        for key_signal, iteration in logical_maps
        if key_signal == signal
    }
    missing = sorted(expected_iterations - present_iterations)
    unexpected = sorted(present_iterations - expected_iterations)
    topology_valid = (
        num_iterations is not None
        and num_iterations >= 0
        and not missing
        and not unexpected
    )
    check("logical_cost_improvement_coverage", topology_valid, {
        "expected_logical_iterations": sorted(expected_iterations),
        "present_logical_iterations": sorted(present_iterations),
        "missing": missing,
        "unexpected": unexpected,
    })
    logical_duplicates = [
        value for value in duplicate_keys
        if value.startswith(f"{signal}:logical:")
    ]
    check(
        "logical_cost_improvement_unique",
        not logical_duplicates and len(declared) == len(present_iterations),
        logical_duplicates,
    )

    metadata_errors: list[str] = []
    layout_errors: list[str] = []
    paths: dict[str, list[str]] = {}
    for entry in declared:
        iteration = strict_int(entry.get("logical_iteration"))
        label = f"{signal}:{entry.get('logical_iteration', 'missing')}"
        expected_stage = (
            "initialization" if iteration == -1
            else "iteration" if iteration is not None and iteration >= 0
            else None
        )
        if expected_stage is None:
            metadata_errors.append(f"{label} logical_iteration is invalid")
        elif entry.get("stage") != expected_stage:
            metadata_errors.append(f"{label} stage must be {expected_stage!r}")
        if entry.get("role") != "logical_event":
            metadata_errors.append(f"{label} role must be 'logical_event'")
        if entry.get("measurement_quality") != LOGICAL_COST_IMPROVEMENT_QUALITY:
            metadata_errors.append(
                f"{label} measurement_quality must be {LOGICAL_COST_IMPROVEMENT_QUALITY!r}"
            )
        if entry.get("measurement_basis") != LOGICAL_COST_IMPROVEMENT_BASIS:
            metadata_errors.append(
                f"{label} measurement_basis must be {LOGICAL_COST_IMPROVEMENT_BASIS!r}"
            )
        expected_aggregation = (
            "defined_zero" if iteration == -1
            else "sum_of_disjoint_checkerboard_passes"
        )
        if entry.get("aggregation") != expected_aggregation:
            metadata_errors.append(
                f"{label} aggregation must be {expected_aggregation!r}"
            )
        if entry.get("checkerboard_identity_exposed") is not False:
            metadata_errors.append(
                f"{label} checkerboard_identity_exposed must be false"
            )
        if entry.get("valid_min") != 0.0:
            metadata_errors.append(f"{label} valid_min must be 0.0")
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        if not nonempty_text(entry.get("limitations")):
            metadata_errors.append(f"{label} limitations must be non-empty")
        forbidden_fields = sorted(
            key for key in entry
            if "phase" in str(key).lower()
            or key in {"pass_index", "raw_pass_indices"}
        )
        if forbidden_fields:
            metadata_errors.append(
                f"{label} exposes checkerboard/pass identity fields: {forbidden_fields}"
            )

        relative_path = str(entry.get("path", ""))
        paths.setdefault(relative_path, []).append(label)
        if entry.get("dtype") != "float32":
            layout_errors.append(f"{label} dtype must be 'float32'")
        if Path(relative_path).suffix.lower() != ".pfm":
            layout_errors.append(f"{label} path must reference a PFM map")
        if strict_int(entry.get("uncompressed_bytes")) != width * height * 4:
            layout_errors.append(
                f"{label} uncompressed_bytes must be {width * height * 4}"
            )
    for path, labels in paths.items():
        if not path or len(labels) != 1:
            layout_errors.append(f"logical cost-improvement path {path!r} is used by {labels}")

    domain: dict[str, dict[str, float | int]] = {}
    for iteration in sorted(expected_iterations):
        record = logical_maps.get((signal, iteration))
        if record is None:
            continue
        values = record[2]
        label = f"{signal}:{iteration}"
        if values.shape != (height, width):
            layout_errors.append(f"{label} expected {(height, width)}, got {values.shape}")
        if values.dtype.kind != "f" or values.dtype.itemsize != 4:
            layout_errors.append(f"{label} expected float32 data, got {values.dtype}")
        finite = np.isfinite(values)
        finite_values = values[finite]
        domain[str(iteration)] = {
            "nonfinite_pixels": int((~finite).sum()),
            "negative_pixels": int(np.count_nonzero(finite_values < 0.0)),
            "min": float(np.min(finite_values)) if finite_values.size else float("inf"),
            "max": float(np.max(finite_values)) if finite_values.size else float("-inf"),
        }
    check("logical_cost_improvement_metadata", not metadata_errors, metadata_errors)
    check("logical_cost_improvement_layout", not layout_errors, layout_errors)
    domain_valid = (
        set(domain) == {str(value) for value in expected_iterations}
        and all(
            row["nonfinite_pixels"] == 0 and row["negative_pixels"] == 0
            for row in domain.values()
        )
    )
    check("logical_cost_improvement_domain", domain_valid, domain)

    initialization = logical_maps.get((signal, -1))
    initialization_max_abs = (
        max_abs_difference(initialization[2], np.zeros((height, width), dtype=np.float32))
        if initialization is not None else float("inf")
    )
    check(
        "logical_cost_improvement_initialization_zero",
        initialization is not None and initialization_max_abs <= tolerance,
        {"max_abs": initialization_max_abs, "tolerance": tolerance},
    )

    image_id = strict_int(summary.get("image_id"))
    scale_level = strict_int(summary.get("scale_level"))
    capture_root = frame_dir.parent.parent
    improvement_relative = Path("instrumentation/improvements")
    improvement_dir = capture_root / improvement_relative
    _probe, improvement_parent_error = owned_regular_artifact_path(
        capture_root,
        (improvement_relative / ".dmap_validator_probe").as_posix(),
        require_exists=False,
    )
    legacy_paths: list[Path] = []
    if (
        improvement_parent_error is None
        and image_id is not None
        and image_id >= 0
        and scale_level is not None
        and scale_level >= 0
    ):
        prefix = f"depth{image_id:04d}_scale{scale_level:02d}"
        legacy_paths = sorted(improvement_dir.glob(f"{prefix}_pass*_improvement.pfm"))

    legacy_detail: dict[str, Any] = {
        "available": bool(legacy_paths),
        "basis": "legacy pass 0 is initialization and is intentionally excluded; each logical iteration sums its two disjoint checkerboard pass maps",
    }
    legacy_valid = improvement_parent_error is None
    if improvement_parent_error is not None:
        legacy_detail.update({
            "unavailable_reason": "legacy pass-map parent failed capture ownership validation",
            "errors": [improvement_parent_error],
        })
    elif legacy_paths:
        pass_maps: dict[int, np.ndarray] = {}
        errors: list[str] = []
        prefix = f"depth{image_id:04d}_scale{scale_level:02d}"
        pattern = re.compile(rf"^{re.escape(prefix)}_pass(\d+)_improvement\.pfm$")
        for path in legacy_paths:
            match = pattern.match(path.name)
            owned_path, path_error = owned_regular_artifact_path(
                capture_root, path.relative_to(capture_root).as_posix()
            )
            if path_error is not None or owned_path is None:
                errors.append(
                    f"unsafe or non-regular legacy pass map {path.name}: {path_error}"
                )
                continue
            if match is None:
                errors.append(f"unexpected legacy pass-map name: {path.name}")
                continue
            pass_index = int(match.group(1))
            if pass_index in pass_maps:
                errors.append(f"duplicate legacy pass index: {pass_index}")
                continue
            try:
                values = read_pfm(owned_path, np)
            except Exception as exc:
                errors.append(f"unable to read {path.name}: {exc}")
                continue
            if values.shape != (height, width):
                errors.append(f"{path.name}: expected {(height, width)}, got {values.shape}")
            if values.dtype.kind != "f" or values.dtype.itemsize != 4:
                errors.append(f"{path.name}: expected float32 data, got {values.dtype}")
            if not np.isfinite(values).all():
                errors.append(f"{path.name}: contains non-finite values")
            pass_maps[pass_index] = values

        expected_num_passes = (
            1 + 2 * num_iterations
            if num_iterations is not None and num_iterations >= 0 else None
        )
        manifest_num_passes = strict_int(manifest.get("num_passes"))
        expected_passes = (
            set(range(expected_num_passes)) if expected_num_passes is not None else set()
        )
        if manifest_num_passes != expected_num_passes:
            errors.append(
                f"manifest num_passes={manifest_num_passes!r}, expected {expected_num_passes!r}"
            )
        if set(pass_maps) != expected_passes:
            errors.append(
                f"legacy pass topology is {sorted(pass_maps)}, expected {sorted(expected_passes)}"
            )
        residuals: dict[str, float] = {}
        for iteration in range(num_iterations or 0):
            first_pass = 1 + 2 * iteration
            second_pass = first_pass + 1
            logical = logical_maps.get((signal, iteration))
            if logical is None or first_pass not in pass_maps or second_pass not in pass_maps:
                continue
            residuals[str(iteration)] = max_abs_difference(
                logical[2], pass_maps[first_pass] + pass_maps[second_pass]
            )
        if set(residuals) != {str(value) for value in range(num_iterations or 0)}:
            errors.append("legacy pass closure is unavailable for one or more logical iterations")
        if any(value > tolerance for value in residuals.values()):
            errors.append("logical map differs from the summed legacy pass maps")
        legacy_detail.update({
            "errors": errors,
            "expected_pass_indices": sorted(expected_passes),
            "present_pass_indices": sorted(pass_maps),
            "per_iteration_max_abs": residuals,
            "tolerance": tolerance,
        })
        legacy_valid = not errors
    else:
        legacy_detail["unavailable_reason"] = (
            "no identity-matched legacy pass improvement maps were retained"
        )
    check("logical_cost_improvement_legacy_pass_closure", legacy_valid, legacy_detail)

    return {
        "available": True,
        "expected_logical_iterations": sorted(expected_iterations),
        "domain": domain,
        "initialization_max_abs": initialization_max_abs,
        "legacy_pass_closure": legacy_detail,
    }


def decode_uint32_rgba(values: np.ndarray) -> np.ndarray:
    channels = np.asarray(values, dtype=np.uint32)
    return (
        channels[..., 0]
        | (channels[..., 1] << 8)
        | (channels[..., 2] << 16)
        | (channels[..., 3] << 24)
    )


def uint32_popcount(values: np.ndarray, bits: int = 32) -> np.ndarray:
    unsigned = np.asarray(values, dtype=np.uint32)
    counts = np.zeros(unsigned.shape, dtype=np.uint8)
    for bit in range(bits):
        counts += ((unsigned >> bit) & 1).astype(np.uint8)
    return counts


def validate_v4_low_texture_update_hysteresis(
    *,
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    num_iterations: int | None,
    metadata: dict[str, Any],
    tolerance: float,
    check: Any,
) -> dict[str, Any]:
    """Validate optional enabled hysteresis maps without breaking older captures."""

    present = {
        (signal, iteration)
        for signal, iteration in logical_maps
        if signal in V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS
    }
    raw_order_present = {
        (signal, iteration)
        for signal, iteration in logical_maps
        if signal in V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
    }
    raw_order_version = strict_int(metadata.get("raw_order_statistics_version"))
    metadata_available = metadata.get("available") is True
    enabled = metadata_available and metadata.get("enabled") is True
    check(
        "v4_low_texture_update_metadata",
        metadata_available or not (present or raw_order_present),
        metadata,
    )
    if not metadata_available:
        return {"available": False, "enabled": False, "metadata": metadata}
    if not enabled:
        check(
            "v4_low_texture_update_disabled_absence",
            not (present or raw_order_present),
            sorted(present | raw_order_present),
        )
        return {"available": True, "enabled": False, "metadata": metadata}

    expected_iterations = set(range(num_iterations or 0))
    expected = {
        (signal, iteration)
        for signal in V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS
        for iteration in expected_iterations
    }
    check(
        "v4_low_texture_update_coverage",
        present == expected,
        {
            "missing": [list(value) for value in sorted(expected - present)],
            "unexpected": [list(value) for value in sorted(present - expected)],
        },
    )
    raw_order_expected = {
        (signal, iteration)
        for signal in V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
        for iteration in expected_iterations
    }
    raw_order_required = raw_order_version is not None and raw_order_version >= 1
    check(
        "v4_low_texture_update_raw_order_coverage",
        (
            raw_order_present == raw_order_expected
            if raw_order_required or raw_order_present
            else True
        ),
        {
            "required": raw_order_required,
            "contract_version": raw_order_version,
            "legacy_schema4_compatibility": not raw_order_required and not raw_order_present,
            "missing": [list(value) for value in sorted(raw_order_expected - raw_order_present)],
            "unexpected": [list(value) for value in sorted(raw_order_present - raw_order_expected)],
        },
    )
    metadata_errors: list[str] = []
    for entry in entries:
        signal = str(entry.get("signal", ""))
        if signal not in (
            V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS
            | V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
        ):
            continue
        label = f"{signal}:{entry.get('logical_iteration')}"
        if entry.get("role") != "logical_event":
            metadata_errors.append(f"{label} role must be 'logical_event'")
        if entry.get("measurement_quality") != "exact":
            metadata_errors.append(f"{label} measurement_quality must be 'exact'")
        if not nonempty_text(entry.get("measurement_basis")):
            metadata_errors.append(f"{label} measurement_basis must be non-empty")
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        if strict_int(entry.get("logical_iteration")) == -1:
            metadata_errors.append(f"{label} must not be emitted for initialization")
    check("v4_low_texture_update_map_metadata", not metadata_errors, metadata_errors)

    domain_errors: list[str] = []
    per_iteration: dict[str, Any] = {}
    min_gain = float(metadata["low_texture_update_min_gain"])
    gate = int(metadata["low_texture_update_gate"])
    for iteration in sorted(expected_iterations):
        values = {
            signal: logical_maps[(signal, iteration)][2]
            for signal in V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS
            if (signal, iteration) in logical_maps
        }
        if len(values) != len(V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS):
            continue
        eligible = values["low_texture_update_eligible_exact"]
        ambiguity = values["low_texture_update_ambiguity_exact"]
        required = values["low_texture_update_required_gain_exact"]
        best = values["low_texture_update_best_proposed_gain_exact"]
        rejected_mask = values["low_texture_update_rejected_mask_exact"]
        source = values["low_texture_update_would_have_won_source_exact"]
        rejected_count = values["low_texture_update_rejected_count_exact"]
        finite = all(bool(np.all(np.isfinite(value))) for value in values.values())
        integer_domain = (
            np.all(eligible == np.rint(eligible))
            and np.all(rejected_mask == np.rint(rejected_mask))
            and np.all(source == np.rint(source))
            and np.all(rejected_count == np.rint(rejected_count))
        )
        domains_valid = (
            finite and integer_domain
            and bool(np.all((eligible >= 0) & (eligible <= 1)))
            and bool(np.all((ambiguity >= 0) & (ambiguity <= 1)))
            and bool(np.all(required >= 0)) and bool(np.all(best >= 0))
            and bool(np.all((rejected_mask >= 0) & (rejected_mask <= 3)))
            and bool(np.all((source >= 0) & (source <= 8)))
            and bool(np.all((rejected_count >= 0) & (rejected_count <= 5)))
        )
        ineligible = eligible == 0
        zero_ineligible = all(bool(np.all(value[ineligible] == 0)) for value in (
            ambiguity, required, best, rejected_mask, source, rejected_count,
        ))
        rejection_consistent = bool(np.all((rejected_count > 0) == (rejected_mask > 0)))
        rejection_consistent &= bool(np.all((rejected_count > 0) == (source > 0)))
        rejected_mask_uint = np.rint(rejected_mask).astype(np.uint8)
        rejected_count_int = np.rint(rejected_count).astype(np.int16)
        propagation_rejected = ((rejected_mask_uint & 1) != 0).astype(np.int16)
        refinement_rejected = rejected_count_int - propagation_rejected
        rejection_consistent &= bool(np.all(refinement_rejected >= 0))
        rejection_consistent &= bool(np.all(
            ((rejected_mask_uint & 2) != 0) == (refinement_rejected > 0)
        ))
        gate_consistent = (
            (gate & 1 or not np.any(rejected_mask_uint & 1))
            and (gate & 2 or not np.any(rejected_mask_uint & 2))
        )
        eligible_mask = eligible == 1
        formula_error = (
            max_abs_difference(required[eligible_mask], min_gain * ambiguity[eligible_mask])
            if np.any(eligible_mask) else 0.0
        )
        per_iteration[str(iteration)] = {
            "eligible_pixels": int(np.count_nonzero(eligible_mask)),
            "rejected_pixels": int(np.count_nonzero(rejected_count)),
            "required_gain_formula_max_abs_error": formula_error,
        }
        if not (
            domains_valid and zero_ineligible and rejection_consistent
            and gate_consistent and formula_error <= max(tolerance, 2.0e-6)
        ):
            domain_errors.append(f"iteration {iteration}: hysteresis map integrity failure")
    check(
        "v4_low_texture_update_domains",
        not domain_errors and len(per_iteration) == len(expected_iterations),
        {"errors": domain_errors, "per_iteration": per_iteration},
    )
    raw_order_errors: list[str] = []
    raw_order_per_iteration: dict[str, Any] = {}
    if raw_order_required or raw_order_present:
        arithmetic_tolerance = max(float(tolerance), 2.0e-6)
        for iteration in sorted(expected_iterations):
            def logical(signal: str) -> np.ndarray | None:
                record = logical_maps.get((signal, iteration))
                return record[2] if record is not None else None

            raw_best = logical("candidate_raw_best_cost_exact")
            raw_runner = logical("candidate_raw_runner_up_cost_exact")
            raw_gap = logical("gap_raw_best_runner_up_exact")
            retained_delta = logical("candidate_retained_minus_raw_best_exact")
            suppression_identity = logical("candidate_raw_suppression_identity_exact")
            retained_winner = logical("candidate_winner_cost_exact")
            retained_runner = logical("candidate_runner_up_cost_exact")
            retained_gap = logical("gap_winner_runner_up_exact")
            retained_identity = logical("candidate_identity_exact")
            rejected_count = logical("low_texture_update_rejected_count_exact")
            required_values = (
                raw_best, raw_runner, raw_gap, retained_delta,
                suppression_identity, retained_winner, retained_runner,
                retained_gap, retained_identity, rejected_count,
            )
            if any(value is None for value in required_values):
                raw_order_errors.append(
                    f"iteration {iteration}: raw-order closure inputs are incomplete"
                )
                continue
            assert raw_best is not None and raw_runner is not None
            assert raw_gap is not None and retained_delta is not None
            assert suppression_identity is not None and retained_winner is not None
            assert retained_runner is not None and retained_gap is not None
            assert retained_identity is not None and rejected_count is not None
            scalar_values = (
                raw_best, raw_runner, raw_gap, retained_delta,
                retained_winner, retained_runner, retained_gap, rejected_count,
            )
            shape_valid = (
                all(value.shape == raw_best.shape for value in scalar_values)
                and suppression_identity.shape == (*raw_best.shape, 4)
                and retained_identity.shape == (*raw_best.shape, 3)
            )
            if not shape_valid:
                raw_order_errors.append(
                    f"iteration {iteration}: raw-order closure map shapes differ"
                )
                continue

            raw_best_available = np.isfinite(raw_best) & (raw_best != -1.0)
            raw_runner_unavailable = raw_runner == -1.0
            raw_runner_valid = raw_runner_unavailable | (
                np.isfinite(raw_runner) & raw_best_available & (raw_runner >= raw_best)
            )
            expected_raw_gap = np.where(
                raw_runner_unavailable, -1.0, np.maximum(raw_runner - raw_best, 0.0)
            )
            expected_retained_delta = np.where(
                raw_best_available,
                np.maximum(retained_winner - raw_best, 0.0),
                -1.0,
            )
            identity = np.rint(suppression_identity).astype(np.int16)
            raw_best_slot = identity[..., 0]
            raw_runner_slot = identity[..., 1]
            retained_winner_slot = identity[..., 2]
            suppression_source = identity[..., 3]
            identity_integer = bool(np.all(suppression_identity == identity))
            raw_best_slot_valid = bool(np.all((raw_best_slot >= 0) & (raw_best_slot <= 12)))
            raw_runner_slot_valid = bool(np.all(
                ((raw_runner_slot >= 0) & (raw_runner_slot <= 12))
                | (raw_runner_slot == 255)
            ))
            retained_slot_valid = bool(np.all(
                (retained_winner_slot >= 0) & (retained_winner_slot <= 12)
            ))
            source_domain_valid = bool(np.all(
                (suppression_source >= 0) & (suppression_source <= 8)
            ))
            suppressed = suppression_source != 0
            expected_suppression = retained_delta > 0.0
            source_for_slot = np.zeros(raw_best_slot.shape, dtype=np.int16)
            source_for_slot[(raw_best_slot >= 1) & (raw_best_slot <= 8)] = 2
            source_for_slot[raw_best_slot == 9] = 3
            source_for_slot[raw_best_slot == 10] = 4
            source_for_slot[raw_best_slot == 11] = 5
            source_for_slot[raw_best_slot == 12] = 6
            identity_closes = (
                bool(np.all(retained_winner_slot == retained_identity[..., 0]))
                and bool(np.all(
                    raw_runner_unavailable == (raw_runner_slot == 255)
                ))
                and bool(np.all(
                    (~suppressed) | (suppression_source == source_for_slot)
                ))
                and bool(np.all((~suppressed) | (raw_best_slot != retained_winner_slot)))
                and bool(np.all((~suppressed) | (rejected_count > 0)))
            )
            retained_contract_closes = (
                bool(np.all(retained_runner[suppressed] == -1.0))
                and bool(np.all(retained_gap[suppressed] == -1.0))
                and bool(np.all(retained_identity[..., 1][suppressed] == 255))
                and max_abs_difference(
                    retained_runner[~suppressed], raw_runner[~suppressed]
                ) <= arithmetic_tolerance
                and max_abs_difference(
                    retained_gap[~suppressed], raw_gap[~suppressed]
                ) <= arithmetic_tolerance
                and bool(np.all(
                    retained_identity[..., 1][~suppressed]
                    == raw_runner_slot[~suppressed]
                ))
            )
            closure_errors = {
                "raw_gap": max_abs_difference(raw_gap, expected_raw_gap),
                "retained_minus_raw_best": max_abs_difference(
                    retained_delta, expected_retained_delta
                ),
            }
            raw_order_per_iteration[str(iteration)] = {
                "suppressed_raw_best_pixels": int(np.count_nonzero(suppressed)),
                "raw_gap_available_pixels": int(np.count_nonzero(~raw_runner_unavailable)),
                "closure_max_abs_error": closure_errors,
            }
            valid = (
                bool(np.all(raw_best_available))
                and bool(np.all(raw_runner_valid))
                and bool(np.all(np.isfinite(raw_gap)))
                and bool(np.all(np.isfinite(retained_delta)))
                and identity_integer and raw_best_slot_valid
                and raw_runner_slot_valid and retained_slot_valid
                and source_domain_valid
                and bool(np.all(suppressed == expected_suppression))
                and identity_closes and retained_contract_closes
                and all(value <= arithmetic_tolerance for value in closure_errors.values())
            )
            if not valid:
                raw_order_errors.append(
                    f"iteration {iteration}: raw-order/retained-winner closure failure"
                )
    check(
        "v4_low_texture_update_raw_order_closure",
        (
            not raw_order_errors
            and (
                len(raw_order_per_iteration) == len(expected_iterations)
                if raw_order_required or raw_order_present else True
            )
        ),
        {
            "required": raw_order_required,
            "contract_version": raw_order_version,
            "errors": raw_order_errors,
            "per_iteration": raw_order_per_iteration,
        },
    )
    return {
        "available": True,
        "enabled": True,
        "metadata": metadata,
        "per_iteration": per_iteration,
        "raw_order_statistics": {
            "required": raw_order_required,
            "contract_version": raw_order_version,
            "per_iteration": raw_order_per_iteration,
        },
    }


def validate_v4_exact_iteration_table(
    *,
    rows: list[dict[str, Any]] | None,
    columns: set[str],
    schema_version: int | None,
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    expected_iterations: set[int],
    pyramid_level: int | None,
    hysteresis_metadata: dict[str, Any],
    atomic_rows: list[dict[str, Any]] | None,
    tolerance: float,
    check: Any,
) -> dict[str, Any]:
    """Close schema-v3 exact-iteration hysteresis rows against exported maps."""

    required_columns = {
        "logical_iteration", "pyramid_level", *V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS,
    }
    schema_v3 = schema_version == 3
    schema_errors: list[str] = []
    if schema_version is not None and schema_version >= 3:
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            schema_errors.append(f"missing schema-v3 columns: {missing_columns}")
    enabled = (
        hysteresis_metadata.get("available") is True
        and hysteresis_metadata.get("enabled") is True
    )
    if enabled and not schema_v3:
        schema_errors.append(
            f"enabled hysteresis requires exact_iteration schema 3, got {schema_version!r}"
        )
    check("v4_exact_iteration_schema", not schema_errors, {
        "schema_version": schema_version,
        "required_columns": sorted(required_columns),
        "errors": schema_errors,
    })
    if not schema_v3 or rows is None:
        return {
            "available": False,
            "schema_version": schema_version,
            "enabled": enabled,
            "reason": "exact_iteration schema 3 unavailable",
        }

    row_by_iteration: dict[int, dict[str, Any]] = {}
    row_errors: list[str] = []
    for row in rows:
        iteration = csv_integer(row.get("logical_iteration"))
        if iteration is None:
            row_errors.append("logical_iteration must be an integer")
            continue
        if iteration in row_by_iteration:
            row_errors.append(f"duplicate logical_iteration {iteration}")
            continue
        row_by_iteration[iteration] = row
        row_level = csv_integer(row.get("pyramid_level"))
        if pyramid_level is None or row_level != pyramid_level:
            row_errors.append(
                f"iteration {iteration}: pyramid_level={row_level!r}, manifest={pyramid_level!r}"
            )
    missing_rows = sorted(expected_iterations - set(row_by_iteration))
    unexpected_rows = sorted(set(row_by_iteration) - expected_iterations)
    if missing_rows:
        row_errors.append(f"missing logical iterations: {missing_rows}")
    if unexpected_rows:
        row_errors.append(f"unexpected logical iterations: {unexpected_rows}")
    initialization = row_by_iteration.get(-1)
    if initialization is not None:
        nonempty = [
            field for field in V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS
            if initialization.get(field) not in (None, "")
        ]
        if nonempty:
            row_errors.append(
                f"initialization hysteresis fields must be unavailable/empty: {nonempty}"
            )
    if not enabled:
        for iteration, row in row_by_iteration.items():
            nonempty = [
                field for field in V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS
                if row.get(field) not in (None, "")
            ]
            if nonempty:
                row_errors.append(
                    f"iteration {iteration}: disabled hysteresis fields must be empty: {nonempty}"
                )
    check("v4_exact_iteration_rows", not row_errors, {
        "pyramid_level": pyramid_level,
        "expected_iterations": sorted(expected_iterations),
        "errors": row_errors,
    })

    if not enabled:
        return {
            "available": True,
            "schema_version": schema_version,
            "enabled": False,
            "pyramid_level": pyramid_level,
        }

    gate = int(hysteresis_metadata["low_texture_update_gate"])
    closure_errors: list[str] = []
    per_iteration: dict[str, Any] = {}
    exact_values_by_iteration: dict[int, dict[str, float]] = {}
    count_fields = V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS[:5]
    sum_fields = V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS[5:]
    for iteration in sorted(expected_iterations - {-1}):
        def logical(signal: str) -> np.ndarray | None:
            record = logical_maps.get((signal, iteration))
            return record[2] if record is not None else None

        eligible_values = logical("low_texture_update_eligible_exact")
        accepted_values = logical("candidate_accepted_mask_exact")
        rejected_count_values = logical("low_texture_update_rejected_count_exact")
        rejected_mask_values = logical("low_texture_update_rejected_mask_exact")
        required_values = logical("low_texture_update_required_gain_exact")
        best_values = logical("low_texture_update_best_proposed_gain_exact")
        if any(value is None for value in (
            eligible_values, accepted_values, rejected_count_values,
            rejected_mask_values, required_values, best_values,
        )):
            closure_errors.append(f"iteration {iteration}: required closure maps unavailable")
            continue
        eligible = np.rint(eligible_values).astype(np.uint8) == 1
        accepted = np.rint(accepted_values).astype(np.uint32)
        rejected_count = np.rint(rejected_count_values).astype(np.int64)
        rejected_mask = np.rint(rejected_mask_values).astype(np.uint8)
        propagation_rejected = ((rejected_mask & 1) != 0).astype(np.int64)
        refinement_rejected = rejected_count - propagation_rejected
        if np.any(refinement_rejected[eligible] < 0):
            closure_errors.append(f"iteration {iteration}: negative refinement rejection count")
            continue
        expected = {
            "low_texture_gate_eligible": float(np.count_nonzero(eligible)),
            "low_texture_propagation_accepted": float(np.sum(
                uint32_popcount(accepted & V4_LOW_TEXTURE_PROPAGATION_ACCEPTED_MASK, 13)[eligible],
                dtype=np.uint64,
            )) if gate & 1 else 0.0,
            "low_texture_propagation_rejected": float(np.sum(
                propagation_rejected[eligible], dtype=np.int64,
            )) if gate & 1 else 0.0,
            "low_texture_refinement_accepted": float(np.sum(
                uint32_popcount(accepted & V4_LOW_TEXTURE_REFINEMENT_ACCEPTED_MASK, 13)[eligible],
                dtype=np.uint64,
            )) if gate & 2 else 0.0,
            "low_texture_refinement_rejected": float(np.sum(
                refinement_rejected[eligible], dtype=np.int64,
            )) if gate & 2 else 0.0,
            "low_texture_required_gain_sum": float(np.sum(
                required_values[eligible], dtype=np.float64,
            )),
            "low_texture_best_proposed_gain_sum": float(np.sum(
                best_values[eligible], dtype=np.float64,
            )),
        }
        exact_values_by_iteration[iteration] = expected
        row = row_by_iteration.get(iteration, {})
        actual = {field: csv_finite_number(row.get(field)) for field in expected}
        differences: dict[str, float | None] = {}
        for field in count_fields:
            differences[field] = (
                None if actual[field] is None else abs(actual[field] - expected[field])
            )
            if actual[field] is None or differences[field] != 0.0:
                closure_errors.append(
                    f"iteration {iteration}: {field} table={actual[field]!r}, map={expected[field]!r}"
                )
        for field in sum_fields:
            difference = (
                None if actual[field] is None else abs(actual[field] - expected[field])
            )
            differences[field] = difference
            sum_tolerance = max(float(tolerance), 2.0e-6, abs(expected[field]) * 1.0e-10)
            if difference is None or difference > sum_tolerance:
                closure_errors.append(
                    f"iteration {iteration}: {field} table={actual[field]!r}, "
                    f"map={expected[field]!r}, tolerance={sum_tolerance}"
                )
        per_iteration[str(iteration)] = {
            "expected_from_maps": expected,
            "table": actual,
            "absolute_difference": differences,
        }
    check("v4_exact_iteration_low_texture_map_closure", not closure_errors, {
        "errors": closure_errors,
        "per_iteration": per_iteration,
    })

    atomic_errors: list[str] = []
    atomic_detail: dict[str, Any] = {"available": atomic_rows is not None, "per_iteration": {}}
    if atomic_rows is not None:
        atomic_by_iteration: dict[int, dict[str, Any]] = {}
        for row in atomic_rows:
            if csv_integer(row.get("scale_level")) != pyramid_level:
                continue
            iteration = csv_integer(row.get("iteration"))
            if iteration is not None:
                atomic_by_iteration[iteration] = row
        for iteration, expected in exact_values_by_iteration.items():
            atomic = atomic_by_iteration.get(iteration)
            if atomic is None:
                atomic_errors.append(f"iteration {iteration}: iteration.csv row unavailable")
                continue
            differences: dict[str, float | None] = {}
            for field in count_fields:
                actual = csv_finite_number(atomic.get(field))
                difference = None if actual is None else abs(actual - expected[field])
                differences[field] = difference
                if difference != 0.0:
                    atomic_errors.append(
                        f"iteration {iteration}: atomic {field}={actual!r}, exact={expected[field]!r}"
                    )
            for field in sum_fields:
                actual = csv_finite_number(atomic.get(field))
                difference = None if actual is None else abs(actual - expected[field])
                differences[field] = difference
                atomic_tolerance = max(1.0e-4, abs(expected[field]) * 0.02)
                if difference is None or difference > atomic_tolerance:
                    atomic_errors.append(
                        f"iteration {iteration}: atomic {field}={actual!r}, "
                        f"exact={expected[field]!r}, tolerance={atomic_tolerance}"
                    )
            atomic_detail["per_iteration"][str(iteration)] = {
                "absolute_difference": differences,
                "float_atomic_sum_tolerance": "max(1e-4, 2% of exact record sum)",
            }
    atomic_detail["errors"] = atomic_errors
    check("v4_exact_iteration_low_texture_atomic_closure", not atomic_errors, atomic_detail)
    return {
        "available": True,
        "schema_version": schema_version,
        "enabled": True,
        "pyramid_level": pyramid_level,
        "map_closure": per_iteration,
        "atomic_closure": atomic_detail,
    }


def validate_v4_candidate_order_statistics_contract(
    capture: dict[str, Any],
    hysteresis_metadata: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Validate legacy and execution-aware candidate-order declarations."""

    contract = capture.get("candidate_order_statistics")
    if contract is None:
        return True, {"status": "legacy_schema4_not_declared"}
    if not isinstance(contract, dict):
        return False, {"contract": contract, "errors": ["contract must be an object"]}

    errors: list[str] = []
    if strict_int(contract.get("schema_version")) != 1:
        errors.append("schema_version must be 1")
    if contract.get("scope") != "enabled_low_texture_hysteresis_iterative_states":
        errors.append("scope is invalid")
    available = contract.get("available")
    if not isinstance(available, bool):
        errors.append("available must be a boolean")
    if contract.get("retained_gap_unavailable_when_raw_best_suppressed") is not True:
        errors.append("retained-gap suppression contract must be true")
    if set(contract.get("raw_signals") or ()) != V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS:
        errors.append("raw_signals do not match the schema-v4 contract")

    metadata_available = hysteresis_metadata.get("available") is True
    expected_configured = hysteresis_metadata.get("configured") is True
    expected_execution = hysteresis_metadata.get("execution_available") is True
    exact_available = capture.get("available") is True

    configured = contract.get("configured")
    if "configured" in contract:
        if not isinstance(configured, bool):
            errors.append("configured must be a boolean")
        elif metadata_available and configured != expected_configured:
            errors.append(
                "configured does not match the effective per-frame parameters"
            )

    execution_available = contract.get("execution_available")
    if "execution_available" in contract:
        if not isinstance(execution_available, bool):
            errors.append("execution_available must be a boolean")
        elif metadata_available and execution_available != expected_execution:
            errors.append(
                "execution_available does not match low_resolution_prior_available"
            )

    # Older schema-v4 captures declared only ``available``. New captures expose
    # configured and execution availability separately, but both forms must close
    # against the effective per-frame execution decision when metadata is readable.
    if isinstance(available, bool) and metadata_available:
        expected_available = exact_available and expected_execution
        if available != expected_available:
            errors.append(
                "available must equal exact_capture.available && execution_available"
            )

    if "unavailable_reason" in contract:
        reason = contract.get("unavailable_reason")
        if not isinstance(reason, str):
            errors.append("unavailable_reason must be a string")
        elif available is True and reason:
            errors.append("unavailable_reason must be empty when available is true")
        elif available is False and not reason.strip():
            errors.append("unavailable_reason must be non-empty when available is false")
        elif (
            available is False
            and metadata_available
            and expected_execution is False
            and isinstance(execution_available, bool)
        ):
            expected_reason = hysteresis_metadata.get("execution_unavailable_reason")
            if isinstance(expected_reason, str) and reason != expected_reason:
                errors.append(
                    "unavailable_reason does not match the per-frame execution reason"
                )

    return not errors, {"contract": contract, "errors": errors}


def validate_v4_exact(
    *,
    frame_dir: Path,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    logical_view_maps: dict[
        tuple[str, int, int], tuple[dict[str, Any], Path, np.ndarray]
    ],
    width: int,
    height: int,
    tolerance: float,
    low_texture_decay_scale: float | None,
    decay_metadata: dict[str, Any] | None,
    view_samples_metadata: dict[str, Any],
    hysteresis_metadata: dict[str, Any],
    check: Any,
) -> dict[str, Any]:
    capture = manifest.get("exact_capture") or {}
    candidate_order_contract = capture.get("candidate_order_statistics")
    candidate_order_version = (
        strict_int(candidate_order_contract.get("schema_version"))
        if isinstance(candidate_order_contract, dict) else None
    )
    candidate_order_available = (
        candidate_order_contract.get("available")
        if isinstance(candidate_order_contract, dict) else None
    )
    contract_valid, contract_detail = (
        validate_v4_candidate_order_statistics_contract(
            capture, hysteresis_metadata
        )
    )
    check(
        "v4_exact_candidate_order_statistics_contract",
        contract_valid,
        contract_detail,
    )
    hysteresis_metadata = dict(hysteresis_metadata)
    hysteresis_metadata["raw_order_statistics_version"] = (
        candidate_order_version if candidate_order_available is True else None
    )
    requested = capture.get("requested") is True
    available = capture.get("available") is True
    num_iterations = strict_int(manifest.get("num_iterations"))
    num_views = strict_int(capture.get("num_views"))
    expected_iterations = (
        set(range(-1, num_iterations))
        if num_iterations is not None and num_iterations >= 0 else set()
    )
    explicit_unavailability = (
        not available and nonempty_text(capture.get("unavailable_reason"))
    )
    check(
        "v4_exact_availability",
        available or explicit_unavailability,
        {
            "requested": requested,
            "available": available,
            "unavailable_reason": capture.get("unavailable_reason"),
        },
    )
    if not available:
        return {
            "requested": requested,
            "available": False,
            "unavailable_reason": capture.get("unavailable_reason"),
            "view_samples_metadata": view_samples_metadata,
        }

    view_samples_available = view_samples_metadata.get("available") is True
    view_samples = (
        strict_int(view_samples_metadata.get("view_samples"))
        if view_samples_available else None
    )
    check(
        "v4_exact_view_samples_metadata",
        view_samples_available and view_samples is not None and 1 <= view_samples <= 63,
        view_samples_metadata,
    )

    state_missing: dict[str, list[int]] = {}
    for signal in V4_REQUIRED_EXACT_STATE_SIGNALS | V4_REQUIRED_EXACT_EVENT_SIGNALS:
        present = {iteration for key_signal, iteration in logical_maps if key_signal == signal}
        if expected_iterations - present:
            state_missing[signal] = sorted(expected_iterations - present)
    check("v4_exact_state_coverage", not state_missing, state_missing)
    hysteresis_validation = validate_v4_low_texture_update_hysteresis(
        entries=entries,
        logical_maps=logical_maps,
        num_iterations=num_iterations,
        metadata=hysteresis_metadata,
        tolerance=tolerance,
        check=check,
    )

    expected_view_keys = {
        (signal, iteration, view)
        for signal in V4_REQUIRED_EXACT_VIEW_SIGNALS
        for iteration in expected_iterations
        for view in range(num_views or 0)
    }
    present_view_keys = {
        key for key in logical_view_maps if key[0] in V4_REQUIRED_EXACT_VIEW_SIGNALS
    }
    check(
        "v4_exact_view_coverage",
        num_views is not None and num_views > 0 and expected_view_keys == present_view_keys,
        {
            "num_views": num_views,
            "missing": [list(key) for key in sorted(expected_view_keys - present_view_keys)],
            "unexpected": [list(key) for key in sorted(present_view_keys - expected_view_keys)],
        },
    )

    required_table_schemas = {
        "openmvs.dmap.exact_iteration",
        "openmvs.dmap.exact_view_summary",
        "openmvs.dmap.exact_observability",
    }
    table_errors: list[str] = []
    present_table_schemas: set[str] = set()
    exact_iteration_rows: list[dict[str, Any]] | None = None
    exact_iteration_columns: set[str] = set()
    exact_iteration_schema_version: int | None = None
    for table in manifest.get("tables") or []:
        if not isinstance(table, dict):
            table_errors.append("table entry must be an object")
            continue
        schema_name = str(table.get("schema_name", ""))
        present_table_schemas.add(schema_name)
        relative = Path(str(table.get("path", "")))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            table_errors.append(f"{schema_name}: unsafe table path {relative}")
            continue
        path = frame_dir / relative
        byte_count = strict_int(table.get("bytes"))
        if not path.is_file():
            table_errors.append(f"{schema_name}: missing table {relative}")
        elif byte_count != path.stat().st_size:
            table_errors.append(
                f"{schema_name}: bytes={byte_count!r}, actual={path.stat().st_size}"
            )
        if schema_name == "openmvs.dmap.exact_iteration":
            if exact_iteration_schema_version is not None:
                table_errors.append("openmvs.dmap.exact_iteration: duplicate table declaration")
                continue
            exact_iteration_schema_version = strict_int(table.get("schema_version"))
            if path.is_file():
                try:
                    with path.open(encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        exact_iteration_rows = list(reader)
                        exact_iteration_columns = set(reader.fieldnames or [])
                except (OSError, csv.Error) as exc:
                    table_errors.append(
                        f"openmvs.dmap.exact_iteration: unable to parse CSV: {exc}"
                    )
    missing_table_schemas = sorted(required_table_schemas - present_table_schemas)
    if missing_table_schemas:
        table_errors.append(f"missing table schemas: {missing_table_schemas}")
    check("v4_exact_tables", not table_errors, table_errors)

    pyramid_level = strict_int(manifest.get("pyramid_level"))
    pyramid_required = (
        exact_iteration_schema_version is not None
        and exact_iteration_schema_version >= 3
    )
    check(
        "v4_exact_pyramid_identity",
        not pyramid_required or pyramid_level == 0,
        {
            "required": pyramid_required,
            "manifest_pyramid_level": pyramid_level,
            "expected": 0 if pyramid_required else None,
            "basis": "schema-v4 exact artifacts are published only from scaleNumber == 0",
        },
    )
    atomic_rows: list[dict[str, Any]] | None = None
    iteration_path = frame_dir / "iteration.csv"
    if iteration_path.is_file():
        try:
            with iteration_path.open(encoding="utf-8", newline="") as handle:
                atomic_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            atomic_rows = None
    exact_iteration_validation = validate_v4_exact_iteration_table(
        rows=exact_iteration_rows,
        columns=exact_iteration_columns,
        schema_version=exact_iteration_schema_version,
        logical_maps=logical_maps,
        expected_iterations=expected_iterations,
        pyramid_level=pyramid_level,
        hysteresis_metadata=hysteresis_metadata,
        atomic_rows=atomic_rows,
        tolerance=tolerance,
        check=check,
    )

    metadata_errors: list[str] = []
    for entry in entries:
        signal = str(entry.get("signal", ""))
        if signal not in (
            V4_REQUIRED_EXACT_STATE_SIGNALS
            | V4_REQUIRED_EXACT_EVENT_SIGNALS
            | V4_REQUIRED_EXACT_VIEW_SIGNALS
        ):
            continue
        label = f"{signal}:{entry.get('logical_iteration')}:{entry.get('source_view_index', '-') }"
        role = entry.get("role")
        expected_role = (
            "logical_view_state" if signal in V4_REQUIRED_EXACT_VIEW_SIGNALS
            else "logical_state" if signal in V4_REQUIRED_EXACT_STATE_SIGNALS
            else "logical_event"
        )
        if role != expected_role:
            metadata_errors.append(f"{label} role must be {expected_role!r}")
        if entry.get("measurement_quality") != "exact":
            metadata_errors.append(f"{label} measurement_quality must be 'exact'")
        if not nonempty_text(entry.get("measurement_basis")):
            metadata_errors.append(f"{label} measurement_basis must be non-empty")
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        phase_fields = [key for key in entry if "phase" in str(key).lower()]
        if phase_fields:
            metadata_errors.append(f"{label} contains phase fields {phase_fields}")
        if signal in V4_REQUIRED_EXACT_VIEW_SIGNALS:
            if strict_int(entry.get("source_view_index")) is None:
                metadata_errors.append(f"{label} source_view_index must be an integer")
            if strict_int(entry.get("source_image_id")) is None:
                metadata_errors.append(f"{label} source_image_id must be an integer")
    check("v4_exact_metadata", not metadata_errors, metadata_errors)

    arithmetic_tolerance = max(float(tolerance), 2.0e-6)
    closure: dict[str, dict[str, float]] = {}
    domain_errors: list[str] = []
    candidate_stats: dict[str, Any] = {}
    view_stats: dict[str, Any] = {}
    weight_records: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    for iteration in sorted(expected_iterations):
        def logical(signal: str) -> np.ndarray | None:
            record = logical_maps.get((signal, iteration))
            return record[2] if record is not None else None

        stored = logical("cost_stored")
        photo = logical("cost_photo_prior_production_exact")
        geometric = logical("cost_geometric_production_exact")
        total = logical("cost_total_production_exact")
        gap = logical("gap_winner_runner_up_exact")
        winner = logical("candidate_winner_cost_exact")
        runner = logical("candidate_runner_up_cost_exact")
        counts = logical("candidate_counts_exact")
        identity = logical("candidate_identity_exact")
        tested_mask = logical("candidate_tested_mask_exact")
        finite_mask = logical("candidate_finite_mask_exact")
        accepted_mask = logical("candidate_accepted_mask_exact")
        before_mask_rgba = logical("selected_views_before_mask_exact")
        after_mask_rgba = logical("selected_views_after_mask_exact")
        selected_counts = logical("selected_view_counts_exact")
        prior_weight = logical("depth_prior_weight_production_exact")
        reference_variance = logical("reference_variance_production_exact")
        if prior_weight is not None:
            weight_records.append((iteration, prior_weight, reference_variance))
        if all(value is not None for value in (stored, photo, geometric, total, winner)):
            closure[str(iteration)] = {
                "component": max_abs_difference(total, photo + geometric),
                "stored_total": max_abs_difference(stored, total),
                "stored_winner": max_abs_difference(stored, winner),
            }
        if gap is not None and runner is not None and winner is not None:
            unavailable = runner == -1.0
            expected_gap = np.where(unavailable, -1.0, np.maximum(runner - winner, 0.0))
            closure.setdefault(str(iteration), {})["gap"] = max_abs_difference(gap, expected_gap)
            valid_gap = np.isfinite(gap) & ((gap == -1.0) | (gap >= 0.0))
            if not bool(np.all(valid_gap)):
                domain_errors.append(f"iteration {iteration}: invalid exact gap domain")
        if all(value is not None for value in (counts, tested_mask, finite_mask, accepted_mask)):
            tested_uint = np.rint(tested_mask).astype(np.uint32)
            finite_uint = np.rint(finite_mask).astype(np.uint32)
            accepted_uint = np.rint(accepted_mask).astype(np.uint32)
            count_mismatch = {
                "tested": int(np.count_nonzero(counts[..., 0] != uint32_popcount(tested_uint, 13))),
                "finite": int(np.count_nonzero(counts[..., 1] != uint32_popcount(finite_uint, 13))),
                "accepted": int(np.count_nonzero(counts[..., 2] != uint32_popcount(accepted_uint, 13))),
            }
            subset_errors = int(np.count_nonzero(finite_uint & ~tested_uint)) + int(
                np.count_nonzero(accepted_uint & ~tested_uint)
            )
            missing_records = int(np.count_nonzero(counts[..., 0] == 0))
            candidate_stats[str(iteration)] = {
                "count_mismatch_pixels": count_mismatch,
                "mask_subset_errors": subset_errors,
                "missing_record_pixels": missing_records,
            }
            if any(count_mismatch.values()) or subset_errors or missing_records:
                domain_errors.append(f"iteration {iteration}: candidate mask/count integrity failure")
        if identity is not None:
            winner_slots = identity[..., 0]
            runner_slots = identity[..., 1]
            if np.any(winner_slots > 12) or np.any((runner_slots > 12) & (runner_slots != 255)):
                domain_errors.append(f"iteration {iteration}: invalid candidate slot")
        if all(value is not None for value in (before_mask_rgba, after_mask_rgba, selected_counts)):
            before_mask = decode_uint32_rgba(before_mask_rgba)
            after_mask = decode_uint32_rgba(after_mask_rgba)
            selected_mismatch = (
                int(np.count_nonzero(selected_counts[..., 0] != uint32_popcount(before_mask)))
                + int(np.count_nonzero(selected_counts[..., 1] != uint32_popcount(after_mask)))
                + int(
                    np.count_nonzero(
                        selected_counts[..., 2] != uint32_popcount(before_mask ^ after_mask)
                    )
                )
            )
            if selected_mismatch:
                domain_errors.append(f"iteration {iteration}: selected-view mask/count mismatch")

        if num_views is not None and num_views > 0 and total is not None:
            contributions: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            for view in range(num_views):
                contribution_record = logical_view_maps.get(
                    ("view_weighted_contribution_exact", iteration, view)
                )
                selection_record = logical_view_maps.get(
                    ("view_selection_state_exact", iteration, view)
                )
                metrics_record = logical_view_maps.get(
                    ("view_selection_metrics_exact", iteration, view)
                )
                if contribution_record is not None:
                    contributions.append(contribution_record[2])
                if selection_record is not None:
                    state = selection_record[2]
                    weights.append(state[..., 0].astype(np.float32))
                    if (
                        np.any(state[..., 0] > 63)
                        or np.any(state[..., 1] > 63)
                        or np.any(state[..., 2] > 6)
                    ):
                        domain_errors.append(
                            f"iteration {iteration}: invalid view weight/rank/decision"
                        )
                if metrics_record is not None:
                    probabilities.append(metrics_record[2][..., 2])
            if len(contributions) == num_views:
                contribution_sum = np.sum(contributions, axis=0)
                closure.setdefault(str(iteration), {})["view_contributions"] = max_abs_difference(
                    contribution_sum, total
                )
            if len(weights) == num_views:
                weight_sum = np.sum(weights, axis=0)
                expected_weight = (
                    float(view_samples)
                    if iteration >= 0 and view_samples is not None else None
                )
                invalid_weight_mask = weight_sum < 1.0
                if expected_weight is not None:
                    invalid_weight_mask |= weight_sum != expected_weight
                invalid_weights = int(np.count_nonzero(invalid_weight_mask))
                view_stats.setdefault(str(iteration), {}).update({
                    "expected_weight_sum": expected_weight,
                    "weight_sum_validation": (
                        "checked" if expected_weight is not None or iteration < 0
                        else "unavailable"
                    ),
                    "invalid_weight_pixels": invalid_weights,
                })
                if invalid_weights:
                    domain_errors.append(f"iteration {iteration}: invalid view weight sum")
            if iteration >= 0 and len(probabilities) == num_views:
                probability_sum = np.sum(probabilities, axis=0)
                finite = np.isfinite(probability_sum)
                invalid_probability = int(np.count_nonzero(~finite)) + int(
                    np.count_nonzero(np.abs(probability_sum[finite] - 1.0) > 2.0e-5)
                )
                view_stats.setdefault(str(iteration), {})[
                    "invalid_probability_pixels"
                ] = invalid_probability
                if invalid_probability:
                    domain_errors.append(f"iteration {iteration}: invalid probability normalization")

    closure_failures = {
        iteration: values
        for iteration, values in closure.items()
        if any(value > arithmetic_tolerance for value in values.values())
    }
    check(
        "v4_exact_arithmetic_closure",
        len(closure) == len(expected_iterations) and not closure_failures,
        {"tolerance": arithmetic_tolerance, "per_iteration": closure},
    )
    weight_stats = depth_prior_weight_domain_stats(
        weight_records, low_texture_decay_scale, decay_metadata
    )
    check(
        "v4_exact_depth_prior_weight_domain",
        weight_stats["pixels"] > 0
        and weight_stats["fatal_invalid_domain_pixels"] == 0,
        weight_stats,
    )
    check("v4_exact_domains", not domain_errors, {
        "errors": domain_errors,
        "candidate": candidate_stats,
        "views": view_stats,
    })
    return {
        "requested": requested,
        "available": available,
        "num_views": num_views,
        "view_samples_metadata": view_samples_metadata,
        "arithmetic_tolerance": arithmetic_tolerance,
        "closure": closure,
        "candidate": candidate_stats,
        "views": view_stats,
        "depth_prior_weight_domain": weight_stats,
        "low_texture_update_hysteresis": hysteresis_validation,
        "exact_iteration_table": exact_iteration_validation,
    }


def validate_apd_maps(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    entries: list[dict[str, Any]],
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]],
    width: int,
    height: int,
    tolerance: float,
    expect_geometric_zero: bool,
    check: Any,
    multiscale_maps: dict[tuple[str, int | None], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Validate the versioned APD per-pixel mechanics contract."""

    capture_value = manifest.get("apd_capture")
    capture = capture_value if isinstance(capture_value, dict) else {}
    capture_version = strict_int(capture.get("schema_version"))
    observability_value = summary.get("apd_observability")
    observability = (
        observability_value if isinstance(observability_value, dict) else {}
    )
    observability_version = strict_int(observability.get("schema_version"))
    version_supported = capture_version in APD_SUPPORTED_SCHEMA_VERSIONS
    version_matches = observability_version == capture_version
    contract = APD_RECORD_CONTRACTS.get(
        capture_version, APD_RECORD_CONTRACTS[APD_SCHEMA_VERSION]
    )
    required_signals = (
        APD_REQUIRED_SIGNALS_V1
        if capture_version == 1 else
        APD_REQUIRED_SIGNALS_V2
        if capture_version == 2 else
        APD_REQUIRED_SIGNALS
    )
    state_float_signals = (
        APD_STATE_FLOAT_SIGNALS_V2
        if capture_version in {1, 2} else APD_STATE_FLOAT_SIGNALS
    )
    state_byte_signals = (
        APD_STATE_BYTE_SIGNALS_V2
        if capture_version in {1, 2} else APD_STATE_BYTE_SIGNALS
    )
    update_float_signals = (
        APD_UPDATE_FLOAT_SIGNALS_V1
        if capture_version == 1 else
        APD_UPDATE_FLOAT_SIGNALS_V2
        if capture_version == 2 else
        APD_UPDATE_FLOAT_SIGNALS
    )
    update_byte_signals = (
        APD_UPDATE_BYTE_SIGNALS_V1
        if capture_version == 1 else
        APD_UPDATE_BYTE_SIGNALS_V2
        if capture_version == 2 else
        APD_UPDATE_BYTE_SIGNALS
    )
    update_rgba_signals = set() if capture_version == 1 else APD_UPDATE_RGBA_SIGNALS
    candidate_slots = int(contract["candidate_slots"])
    source_names = (
        APD_UPDATE_SOURCE_NAMES[:9]
        if capture_version == 1 else
        APD_UPDATE_SOURCE_NAMES[:10]
        if capture_version == 2 else
        APD_UPDATE_SOURCE_NAMES
    )
    apd_entries = [
        entry for entry in entries
        if (
            isinstance(entry, dict)
            and str(entry.get("signal", "")).startswith("apd_")
            and str(entry.get("signal", "")) not in APD_MULTISCALE_SIGNALS
        )
    ]
    if not capture and not apd_entries:
        return {
            "requested": False,
            "available": False,
            "unavailable_reason": "APD mechanics capture was not requested",
        }

    if capture.get("requested") is False and not apd_entries:
        num_iterations = strict_int(manifest.get("num_iterations"))
        disabled_capture_expected = {
            "schema_name": APD_SCHEMA_NAME,
            "requested": False,
            "maps_available": False,
            "summary_available": False,
            "num_iterations": num_iterations,
            "state_record_bytes": contract["state"],
            "update_record_bytes": contract["update"],
            "working_score_persisted": False,
            "persistent_winner_conventionally_rescored": True,
        }
        disabled_summary_expected = {
            "schema_name": APD_SUMMARY_SCHEMA_NAME,
            "requested": False,
            "enabled": False,
            "maps_available": False,
            "summary_available": False,
            "mode": 0,
            "mode_name": "disabled",
            "implementation_label": "disabled",
            "implemented_through": "none",
            "exact_author_code_equivalence_claimed": False,
            "target_label": "paper_mechanics_complete_openmvs",
            "state_record_bytes": contract["state"],
            "update_record_bytes": contract["update"],
            "trace_record_bytes": contract["trace"],
            "targeted_trace_available": False,
            "iterations": [],
            "required_mechanics_not_yet_implemented": [],
        }
        capture_errors = [
            f"{field}: expected {expected!r}, got {capture.get(field)!r}"
            for field, expected in disabled_capture_expected.items()
            if capture.get(field) != expected
        ]
        summary_errors = [
            f"{field}: expected {expected!r}, got {observability.get(field)!r}"
            for field, expected in disabled_summary_expected.items()
            if observability.get(field) != expected
        ]
        if not version_supported:
            capture_errors.append(
                f"schema_version {capture_version!r} is not supported; "
                f"expected one of {list(APD_SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if not version_matches:
            summary_errors.append(
                "APD capture and observability schema versions must match"
            )
        if capture_version == 2:
            if capture.get("anchor_state") != "immutable_per_logical_iteration":
                capture_errors.append("schema v2 requires immutable anchor_state")
            if capture.get("anchor_candidate_slots") != [13, 20]:
                capture_errors.append("schema v2 requires anchor candidate slots [13,20]")
        elif capture_version == 3:
            if capture.get("anchor_state") != (
                "immutable_for_non_reliable_stage_after_reliable_first_updates"
            ):
                capture_errors.append("schema v3 requires reliable-first anchor_state")
            if capture.get("anchor_candidate_slots") != [13, 20]:
                capture_errors.append("schema v3 requires anchor candidate slots [13,20]")
            if capture.get("fitted_plane_candidate_slot") != 21:
                capture_errors.append("schema v3 requires fitted-plane candidate slot 21")
            if capture.get("final_refinement_candidate_accounting") != (
                "separate_native_domain_fields_not_working_candidate_masks"
            ):
                capture_errors.append(
                    "schema v3 requires separate native final-refinement accounting"
                )
        pins = observability.get("pins")
        required_pins = {
            "paper",
            "official_repository_commit",
            "colleague_donor_commit",
        }
        if not isinstance(pins, dict) or any(
            not nonempty_text(pins.get(field)) for field in required_pins
        ):
            summary_errors.append(
                "paper, official repository, and donor pins are required"
            )
        check("apd_disabled_capture_contract", not capture_errors, capture_errors)
        check("apd_disabled_summary_contract", not summary_errors, summary_errors)
        return {
            "requested": False,
            "available": False,
            "unavailable_reason": "APD mode is disabled for this capture",
            "implementation_label": observability.get("implementation_label"),
        }

    stage_active = (
        capture.get("stage_active") is True
        if "stage_active" in capture
        else bool(capture.get("summary_available") or apd_entries)
    )
    if capture.get("requested") is True and not stage_active and not apd_entries:
        inactive_errors: list[str] = []
        for field, expected in {
            "schema_name": APD_SCHEMA_NAME,
            "state_record_bytes": contract["state"],
            "update_record_bytes": contract["update"],
        }.items():
            if capture.get(field) != expected:
                inactive_errors.append(
                    f"capture.{field}: expected {expected!r}, got {capture.get(field)!r}"
                )
        if not version_supported:
            inactive_errors.append(
                f"capture.schema_version {capture_version!r} is unsupported"
            )
        if not version_matches:
            inactive_errors.append("capture and observability schema versions differ")
        if observability.get("schema_name") != APD_SUMMARY_SCHEMA_NAME:
            inactive_errors.append("observability schema_name is invalid")
        for field, expected in {
            "maps_available": False,
            "summary_available": False,
        }.items():
            if capture.get(field) != expected:
                inactive_errors.append(
                    f"capture.{field}: expected {expected!r}, got {capture.get(field)!r}"
                )
            if observability.get(field) != expected:
                inactive_errors.append(
                    f"observability.{field}: expected {expected!r}, got {observability.get(field)!r}"
                )
        if "stage_active" in capture and capture.get("stage_active") is not False:
            inactive_errors.append("capture.stage_active must be false")
        if (
            ("stage_active" in capture or "stage_active" in observability)
            and observability.get("stage_active") is not False
        ):
            inactive_errors.append("observability.stage_active must be false")
        if observability.get("requested") is not True:
            inactive_errors.append("observability.requested must be true")
        if observability.get("enabled") is not True:
            inactive_errors.append("observability.enabled must be true")
        pins = observability.get("pins")
        if not isinstance(pins, dict) or any(
            not nonempty_text(pins.get(field))
            for field in {
                "paper", "official_repository_commit", "colleague_donor_commit"
            }
        ):
            inactive_errors.append("observability paper and source pins are required")
        check("apd_inactive_stage_contract", not inactive_errors, inactive_errors)
        return {
            "requested": True,
            "stage_active": False,
            "available": False,
            "unavailable_reason": (
                "this stage intentionally executes the native OpenMVS schedule"
            ),
            "implementation_label": observability.get("implementation_label"),
        }

    requested = capture.get("requested") is True
    available = capture.get("maps_available") is True
    num_iterations = strict_int(manifest.get("num_iterations"))
    expected_iterations = (
        set(range(num_iterations))
        if num_iterations is not None and num_iterations >= 0 else set()
    )
    contract_errors: list[str] = []
    expected_contract = {
        "schema_name": APD_SCHEMA_NAME,
        "state_record_bytes": contract["state"],
        "update_record_bytes": contract["update"],
        "working_score_persisted": False,
        "persistent_winner_conventionally_rescored": True,
        "summary_available": True,
    }
    for field, expected in expected_contract.items():
        if capture.get(field) != expected:
            contract_errors.append(
                f"{field}: expected {expected!r}, got {capture.get(field)!r}"
            )
    if not version_supported:
        contract_errors.append(
            f"schema_version {capture_version!r} is not supported; "
            f"expected one of {list(APD_SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if not version_matches:
        contract_errors.append(
            "APD capture and observability schema versions must match"
        )
    if capture_version == 2:
        if capture.get("anchor_state") != "immutable_per_logical_iteration":
            contract_errors.append("schema v2 requires immutable anchor_state")
        if capture.get("anchor_candidate_slots") != [13, 20]:
            contract_errors.append("schema v2 requires anchor candidate slots [13,20]")
    elif capture_version == 3:
        if capture.get("anchor_state") != (
            "immutable_for_non_reliable_stage_after_reliable_first_updates"
        ):
            contract_errors.append("schema v3 requires reliable-first anchor_state")
        if capture.get("anchor_candidate_slots") != [13, 20]:
            contract_errors.append("schema v3 requires anchor candidate slots [13,20]")
        if capture.get("fitted_plane_candidate_slot") != 21:
            contract_errors.append("schema v3 requires fitted-plane candidate slot 21")
        if capture.get("final_refinement_candidate_accounting") != (
            "separate_native_domain_fields_not_working_candidate_masks"
        ):
            contract_errors.append(
                "schema v3 requires separate native final-refinement accounting"
            )
    if not requested:
        contract_errors.append("requested must be true when APD maps are present")
    if "stage_active" in capture and capture.get("stage_active") is not True:
        contract_errors.append("stage_active must be true when APD iteration maps are present")
    if not available:
        contract_errors.append("maps_available must be true when APD maps are present")
    if capture.get("num_iterations") != num_iterations:
        contract_errors.append(
            "apd_capture.num_iterations must match map_manifest.num_iterations"
        )
    check("apd_capture_contract", not contract_errors, {
        "capture": capture,
        "errors": contract_errors,
    })

    summary_errors: list[str] = []
    expected_summary_contract = {
        "schema_name": APD_SUMMARY_SCHEMA_NAME,
        "requested": True,
        "enabled": True,
        "maps_available": True,
        "summary_available": True,
        "exact_author_code_equivalence_claimed": False,
        "target_label": "paper_mechanics_complete_openmvs",
        "state_record_bytes": contract["state"],
        "update_record_bytes": contract["update"],
        "trace_record_bytes": contract["trace"],
    }
    for field, expected in expected_summary_contract.items():
        if observability.get(field) != expected:
            summary_errors.append(
                f"{field}: expected {expected!r}, got {observability.get(field)!r}"
            )
    if "stage_active" in observability and observability.get("stage_active") is not True:
        summary_errors.append(
            "stage_active must be true when APD iteration mechanics are available"
        )
    implementation_label = observability.get("implementation_label")
    if implementation_label not in {
        "paper_partial",
        "paper_mechanics_complete_openmvs",
    }:
        summary_errors.append(
            f"unsupported implementation_label {implementation_label!r}"
        )
    missing_mechanics = observability.get("required_mechanics_not_yet_implemented")
    if not isinstance(missing_mechanics, list):
        summary_errors.append(
            "required_mechanics_not_yet_implemented must be a list"
        )
    elif implementation_label == "paper_partial" and not missing_mechanics:
        summary_errors.append("paper_partial requires at least one missing mechanism")
    elif implementation_label == "paper_mechanics_complete_openmvs" and missing_mechanics:
        summary_errors.append(
            "paper_mechanics_complete_openmvs cannot list missing mechanisms"
        )
    profile = observability.get("profile")
    if not isinstance(profile, dict) or profile.get("separation_convention") != (
        "sqrt(sum_squared_minimum_cost_differences)/(number_of_minima-1)"
    ):
        summary_errors.append("profile separation convention is missing or incorrect")
    pins = observability.get("pins")
    required_pins = {"paper", "official_repository_commit", "colleague_donor_commit"}
    if not isinstance(pins, dict) or any(
        not nonempty_text(pins.get(field)) for field in required_pins
    ):
        summary_errors.append("paper, official repository, and donor pins are required")
    if capture_version == 3:
        implemented_through = observability.get("implemented_through")
        if implemented_through not in {"APD14-C3", "APD14-C4"}:
            summary_errors.append(
                "schema v3 must be implemented through APD14-C3 or APD14-C4"
            )
        schedule = observability.get("schedule")
        if not isinstance(schedule, dict) or schedule.get("ordered_stages") != [
            "reliable_black",
            "reliable_red",
            "anchor_and_fitted_plane_generation",
            "non_reliable_black",
            "non_reliable_red",
        ]:
            summary_errors.append("schema v3 reliable-first schedule is missing or invalid")
        fitted_plane = observability.get("fitted_plane")
        if not isinstance(fitted_plane, dict) or fitted_plane.get(
            "candidate_position"
        ) != "before random depth and normal refinement in each non-reliable pixel update":
            summary_errors.append("schema v3 fitted-plane contract is missing or invalid")
        final_refinement = observability.get("final_refinement")
        minimum_improvement = finite_number(
            final_refinement.get("minimum_strict_improvement")
        ) if isinstance(final_refinement, dict) else None
        if not isinstance(final_refinement, dict) or (
            final_refinement.get("disparity_radius") != 5
            or minimum_improvement is None
            or abs(minimum_improvement - 0.1) > 1.0e-6
        ):
            summary_errors.append("schema v3 final-refinement contract is missing or invalid")
        if implemented_through == "APD14-C4":
            multiscale = observability.get("multiscale")
            if not isinstance(multiscale, dict) or (
                multiscale.get("state_schema_version") != 1
                or multiscale.get("coarsest_schedule") != "conventional_native"
                or multiscale.get("iteration_zero_consumes")
                != "transferred_reliability"
                or multiscale.get("checkerboard_exposure") != "timings_only"
            ):
                summary_errors.append(
                    "APD14-C4 multiscale contract is missing or invalid"
                )
            compatibility = observability.get("compatibility_behavior")
            if not isinstance(compatibility, dict) or not nonempty_text(
                compatibility.get("coarsest_reliability_view_weights")
            ):
                summary_errors.append(
                    "APD14-C4 coarsest compatibility behavior is missing"
                )
    check("apd_summary_contract", not summary_errors, {
        "implementation_label": implementation_label,
        "errors": summary_errors,
    })

    expected_keys = {
        (signal, iteration)
        for signal in required_signals
        for iteration in expected_iterations
    }
    present_keys = {
        key for key in logical_maps if key[0] in required_signals
    }
    unexpected_signals = sorted({
        str(entry.get("signal", "")) for entry in apd_entries
    } - required_signals)
    check("apd_map_coverage", expected_keys == present_keys and not unexpected_signals, {
        "expected_signals_per_iteration": len(required_signals),
        "expected_iterations": sorted(expected_iterations),
        "missing": [list(key) for key in sorted(expected_keys - present_keys)],
        "unexpected": [list(key) for key in sorted(present_keys - expected_keys)],
        "unexpected_signals": unexpected_signals,
    })

    metadata_errors: list[str] = []
    for entry in apd_entries:
        signal = str(entry.get("signal", ""))
        iteration = strict_int(entry.get("logical_iteration"))
        if signal not in required_signals or iteration is None:
            continue
        state_signal = signal in state_float_signals | state_byte_signals
        float_signal = signal in state_float_signals | update_float_signals
        rgba_signal = signal in update_rgba_signals
        expected_role = (
            "apd_logical_iteration_state"
            if state_signal else "apd_logical_iteration_update"
        )
        expected_basis = (
            "apd_same_stream_mechanics_record"
            if state_signal else "apd_process_pixel_candidate_record"
        )
        expected_dtype = (
            "float32" if float_signal else "uint8x4" if rgba_signal else "uint8"
        )
        extension = ".pfm" if float_signal else ".png"
        expected_path = (
            f"apd_states/iteration{iteration + 1:02d}/"
            f"{signal.removeprefix('apd_')}{extension}"
        )
        label = f"{signal}:{iteration}"
        expected_fields = {
            "apd_schema_name": APD_SCHEMA_NAME,
            "apd_schema_version": capture_version,
            "measurement_quality": "exact",
            "measurement_basis": expected_basis,
            "role": expected_role,
            "dtype": expected_dtype,
            "stage": "iteration",
            "stage_index": iteration + 1,
            "path": expected_path,
        }
        for field, expected in expected_fields.items():
            if entry.get(field) != expected:
                metadata_errors.append(
                    f"{label} {field}: expected {expected!r}, got {entry.get(field)!r}"
                )
        if not nonempty_text(entry.get("semantics")):
            metadata_errors.append(f"{label} semantics must be non-empty")
        phase_fields = [key for key in entry if "phase" in str(key).lower()]
        if phase_fields:
            metadata_errors.append(f"{label} exposes phase fields {phase_fields}")
    check("apd_map_metadata", not metadata_errors, metadata_errors)

    if not available:
        return {
            "requested": requested,
            "available": False,
            "unavailable_reason": capture.get("unavailable_reason"),
        }

    def logical(signal: str, iteration: int) -> np.ndarray | None:
        record = logical_maps.get((signal, iteration))
        return record[2] if record is not None else None

    def logical_uint32_rgba(signal: str, iteration: int) -> np.ndarray | None:
        values = logical(signal, iteration)
        if values is None or values.shape != (height, width, 4):
            return None
        channels = values.astype(np.uint32)
        return (
            channels[:, :, 0]
            | np.left_shift(channels[:, :, 1], np.uint32(8))
            | np.left_shift(channels[:, :, 2], np.uint32(16))
            | np.left_shift(channels[:, :, 3], np.uint32(24))
        )

    domain_errors: list[str] = []
    relationship_errors: list[str] = []
    closure: dict[str, dict[str, float]] = {}
    candidate_stats: dict[str, Any] = {}
    dispatch_stats: dict[str, Any] = {}
    summary_stats: dict[str, Any] = {}
    summary_rows_value = observability.get("iterations")
    summary_rows = summary_rows_value if isinstance(summary_rows_value, list) else []
    rows_by_iteration = {
        iteration: row
        for row in summary_rows
        if isinstance(row, dict)
        and (iteration := strict_int(row.get("logical_iteration"))) is not None
    }
    if set(rows_by_iteration) != expected_iterations or len(summary_rows) != len(rows_by_iteration):
        summary_errors.append("APD iteration summaries must be unique and complete")

    categorical_ranges = {
        "apd_reliability_class": (0, 2),
        "apd_profile_reason": (0, 8),
        "apd_profile_finite_count": (0, 61),
        "apd_profile_local_minimum_count": (0, 61),
        "apd_profile_plateau_start": (0, 60),
        "apd_profile_plateau_end": (0, 60),
        "apd_sector_candidate_count": (0, 32),
        "apd_ransac_inlier_count": (0, 32),
        "apd_ransac_outlier_count": (0, 32),
        "apd_anchor_count": (0, 8),
        "apd_anchor_reason": (0, 6),
        "apd_ransac_valid": (0, 1),
        "apd_deformable_eligible": (0, 1),
        "apd_update_source": (0, len(source_names) - 1),
        "apd_candidate_tested_count": (0, candidate_slots),
        "apd_candidate_finite_count": (0, candidate_slots),
        "apd_candidate_accepted_count": (0, candidate_slots),
        "apd_selected_view_count": (0, 32),
        "apd_deformable_active": (0, 1),
    }
    if capture_version is not None and capture_version >= 2:
        categorical_ranges.update({
            "apd_view_selection_mode": (0, 4),
            "apd_anchor_evidence_count": (0, 8),
            "apd_anchor_proposal_count": (0, 8),
            "apd_anchor_finite_count": (0, 8),
            "apd_immutable_anchor_state": (0, 1),
            "apd_selected_view_weight_sum": (0, 32),
        })
    if capture_version == 3:
        categorical_ranges.update({
            "apd_fitted_plane_valid": (0, 1),
            "apd_update_stage": (1, 2),
            "apd_fitted_plane_available": (0, 1),
            "apd_fitted_plane_tested": (0, 1),
            "apd_fitted_plane_accepted": (0, 1),
            "apd_final_refinement_offset": (0, 10),
            "apd_final_refinement_tested_count": (0, 11),
            "apd_final_refinement_finite_count": (0, 11),
            "apd_final_refinement_accepted": (0, 1),
        })
    sentinel_nonnegative_signals = {
        "apd_global_minimum_cost",
        "apd_profile_separation",
        "apd_nearest_reliable_distance",
        "apd_ransac_threshold",
        "apd_ransac_center_residual",
        "apd_ransac_mean_inlier_residual",
        "apd_runner_up_working_cost",
        "apd_winner_runner_up_gap",
        "apd_center_cost",
        "apd_anchor_mean_cost",
        "apd_deformable_photometric_cost",
        "apd_geometric_cost",
        "apd_best_anchor_working_cost",
        "apd_accepted_anchor_native_cost",
        "apd_accepted_anchor_index",
        "apd_fitted_plane_depth",
        "apd_fitted_plane_working_cost",
        "apd_fitted_plane_native_cost",
        "apd_final_refinement_incumbent_cost",
        "apd_final_refinement_best_cost",
        "apd_final_refinement_improvement",
        "apd_final_refinement_depth",
    }
    nonnegative_signals = {
        "apd_average_baseline",
        "apd_current_disparity",
        "apd_working_winner_cost",
        "apd_native_persistent_cost",
        "apd_native_stored_cost_before",
        "apd_incumbent_working_cost",
    }
    mask_signals = {
        "apd_candidate_tested_mask",
        "apd_candidate_finite_mask",
        "apd_candidate_accepted_mask",
    }

    for iteration in sorted(expected_iterations):
        for signal in state_float_signals | update_float_signals:
            values = logical(signal, iteration)
            if values is None:
                continue
            if not bool(np.all(np.isfinite(values))):
                domain_errors.append(f"iteration {iteration}: {signal} is non-finite")
                continue
            if signal in sentinel_nonnegative_signals:
                invalid = (values < 0.0) & (values != -1.0)
                if np.any(invalid):
                    domain_errors.append(
                        f"iteration {iteration}: {signal} violates -1/nonnegative domain"
                    )
            elif signal in nonnegative_signals and np.any(values < 0.0):
                domain_errors.append(
                    f"iteration {iteration}: {signal} must be nonnegative"
                )
            elif signal in mask_signals:
                invalid = (
                    (values < 0.0)
                    | (values > float((1 << candidate_slots) - 1))
                    | (values != np.rint(values))
                )
                if np.any(invalid):
                    domain_errors.append(
                        f"iteration {iteration}: {signal} is not an exact "
                        f"{candidate_slots}-bit mask"
                    )

        offset = logical("apd_global_minimum_offset", iteration)
        if offset is not None and np.any(
            (offset < -30.0) | (offset > 30.0) | (offset != np.rint(offset))
        ):
            domain_errors.append(
                f"iteration {iteration}: global minimum offset is not an integer in [-30,30]"
            )
        eta = logical("apd_profile_eta", iteration)
        if eta is not None and np.any(~np.isin(eta, (2, 4, 6))):
            domain_errors.append(f"iteration {iteration}: profile eta is not 2, 4, or 6")
        winner_slot = logical("apd_winner_slot", iteration)
        runner_slot = logical("apd_runner_up_slot", iteration)
        if winner_slot is not None and np.any(winner_slot >= candidate_slots):
            domain_errors.append(f"iteration {iteration}: invalid winner slot")
        if runner_slot is not None and np.any(
            (runner_slot >= candidate_slots) & (runner_slot != 255)
        ):
            domain_errors.append(f"iteration {iteration}: invalid runner-up slot")
        if capture_version is not None and capture_version >= 2:
            accepted_slot = logical("apd_anchor_accepted_slot", iteration)
            if accepted_slot is not None and np.any(
                (accepted_slot > 7) & (accepted_slot != 255)
            ):
                domain_errors.append(
                    f"iteration {iteration}: invalid accepted anchor slot"
                )
        threshold = logical("apd_ransac_threshold", iteration)
        if threshold is not None:
            valid_threshold = threshold >= 0.0
            if np.any(
                valid_threshold
                & ((threshold < 0.005 - 1.0e-7) | (threshold > 0.01 + 1.0e-7))
            ):
                domain_errors.append(f"iteration {iteration}: invalid RANSAC threshold")
        for signal, (minimum, maximum) in categorical_ranges.items():
            values = logical(signal, iteration)
            if values is not None and np.any((values < minimum) | (values > maximum)):
                domain_errors.append(
                    f"iteration {iteration}: {signal} outside [{minimum},{maximum}]"
                )

        reliability = logical("apd_reliability_class", iteration)
        profile_reason = logical("apd_profile_reason", iteration)
        plateau_start = logical("apd_profile_plateau_start", iteration)
        plateau_end = logical("apd_profile_plateau_end", iteration)
        anchor_reason = logical("apd_anchor_reason", iteration)
        ransac_valid = logical("apd_ransac_valid", iteration)
        eligible = logical("apd_deformable_eligible", iteration)
        sector_count = logical("apd_sector_candidate_count", iteration)
        inlier_count = logical("apd_ransac_inlier_count", iteration)
        outlier_count = logical("apd_ransac_outlier_count", iteration)
        anchor_count = logical("apd_anchor_count", iteration)
        active = logical("apd_deformable_active", iteration)
        if reliability is not None and profile_reason is not None:
            expected_reliability = np.where(
                np.isin(profile_reason, (7, 8)),
                2,
                np.where(np.isin(profile_reason, (2, 3, 4, 5, 6)), 1, 0),
            )
            if np.any(reliability != expected_reliability):
                relationship_errors.append(
                    f"iteration {iteration}: reliability/reason mismatch"
                )
        if plateau_start is not None and plateau_end is not None and np.any(
            plateau_start > plateau_end
        ):
            relationship_errors.append(
                f"iteration {iteration}: profile plateau start exceeds end"
            )
        anchor_arrays = (
            anchor_reason,
            ransac_valid,
            eligible,
            sector_count,
            inlier_count,
            outlier_count,
            anchor_count,
        )
        if all(value is not None for value in anchor_arrays):
            ready = anchor_reason == 6
            valid_model = ransac_valid == 1
            if np.any(ready != valid_model) or np.any(valid_model != (eligible == 1)):
                relationship_errors.append(
                    f"iteration {iteration}: ready/RANSAC/eligibility mismatch"
                )
            if np.any(valid_model & (inlier_count + outlier_count != sector_count)):
                relationship_errors.append(
                    f"iteration {iteration}: RANSAC inlier/outlier decomposition mismatch"
                )
            if np.any(valid_model & (anchor_count != np.minimum(inlier_count, 8))):
                relationship_errors.append(
                    f"iteration {iteration}: retained anchor count mismatch"
                )
            if np.any(
                ~valid_model
                & ((inlier_count != 0) | (outlier_count != 0) | (anchor_count != 0))
            ):
                relationship_errors.append(
                    f"iteration {iteration}: invalid RANSAC model retained counts"
                )
        if active is not None and eligible is not None and np.any(active != eligible):
            relationship_errors.append(
                f"iteration {iteration}: deformable active/eligible mismatch"
            )
        if capture_version is not None and capture_version >= 2:
            working_view_mask = logical_uint32_rgba(
                "apd_working_selected_views_mask", iteration
            )
            selected_view_count = logical("apd_selected_view_count", iteration)
            selected_view_weight_sum = logical(
                "apd_selected_view_weight_sum", iteration
            )
            view_selection_mode = logical("apd_view_selection_mode", iteration)
            anchor_evidence_count = logical("apd_anchor_evidence_count", iteration)
            anchor_proposal_count = logical("apd_anchor_proposal_count", iteration)
            anchor_finite_count = logical("apd_anchor_finite_count", iteration)
            anchor_accepted_slot = logical("apd_anchor_accepted_slot", iteration)
            immutable_anchor_state = logical("apd_immutable_anchor_state", iteration)
            best_anchor_cost = logical("apd_best_anchor_working_cost", iteration)
            accepted_anchor_cost = logical(
                "apd_accepted_anchor_native_cost", iteration
            )
            accepted_anchor_index = logical("apd_accepted_anchor_index", iteration)
            source = logical("apd_update_source", iteration)
            c2_arrays = (
                working_view_mask,
                selected_view_count,
                selected_view_weight_sum,
                view_selection_mode,
                anchor_evidence_count,
                anchor_proposal_count,
                anchor_finite_count,
                anchor_accepted_slot,
                immutable_anchor_state,
                best_anchor_cost,
                accepted_anchor_cost,
                accepted_anchor_index,
                source,
                active,
                anchor_count,
            )
            if all(value is not None for value in c2_arrays):
                active_mask = active == 1
                inactive_mask = ~active_mask
                mask_count = uint32_popcount(working_view_mask, 32)
                if np.any(mask_count != selected_view_count):
                    relationship_errors.append(
                        f"iteration {iteration}: working-view mask/count mismatch"
                    )
                if np.any(selected_view_weight_sum < selected_view_count):
                    relationship_errors.append(
                        f"iteration {iteration}: view-weight sum is below selected count"
                    )
                if np.any(active_mask & (view_selection_mode == 0)) or np.any(
                    inactive_mask & (view_selection_mode != 0)
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: APD activity/view-selection mode mismatch"
                    )
                if np.any(immutable_anchor_state != active):
                    relationship_errors.append(
                        f"iteration {iteration}: immutable anchor state/activity mismatch"
                    )
                if np.any(anchor_evidence_count != anchor_proposal_count):
                    relationship_errors.append(
                        f"iteration {iteration}: anchor evidence/proposal mismatch"
                    )
                if np.any(anchor_evidence_count > anchor_count) or np.any(
                    anchor_finite_count > anchor_proposal_count
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: anchor C2 count ordering mismatch"
                    )
                accepted = anchor_accepted_slot != 255
                if np.any(accepted != (accepted_anchor_index >= 0.0)) or np.any(
                    accepted != (accepted_anchor_cost >= 0.0)
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: accepted-anchor availability mismatch"
                    )
                if np.any(
                    accepted
                    & (accepted_anchor_index >= float(width * height))
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: accepted anchor index is outside the frame"
                    )
                if np.any((anchor_finite_count > 0) != (best_anchor_cost >= 0.0)):
                    relationship_errors.append(
                        f"iteration {iteration}: best-anchor cost availability mismatch"
                    )
                if np.any((source == 9) & ~accepted):
                    relationship_errors.append(
                        f"iteration {iteration}: final anchor winner lacks accepted anchor"
                    )

        if capture_version == 3:
            fitted_plane_depth = logical("apd_fitted_plane_depth", iteration)
            fitted_plane_valid = logical("apd_fitted_plane_valid", iteration)
            update_stage = logical("apd_update_stage", iteration)
            fitted_available = logical("apd_fitted_plane_available", iteration)
            fitted_tested = logical("apd_fitted_plane_tested", iteration)
            fitted_accepted = logical("apd_fitted_plane_accepted", iteration)
            fitted_working = logical("apd_fitted_plane_working_cost", iteration)
            fitted_native = logical("apd_fitted_plane_native_cost", iteration)
            final_incumbent = logical(
                "apd_final_refinement_incumbent_cost", iteration
            )
            final_best = logical("apd_final_refinement_best_cost", iteration)
            final_improvement = logical(
                "apd_final_refinement_improvement", iteration
            )
            final_depth = logical("apd_final_refinement_depth", iteration)
            final_offset = logical("apd_final_refinement_offset", iteration)
            final_tested = logical(
                "apd_final_refinement_tested_count", iteration
            )
            final_finite = logical(
                "apd_final_refinement_finite_count", iteration
            )
            final_accepted = logical("apd_final_refinement_accepted", iteration)
            c3_arrays = (
                reliability,
                eligible,
                source,
                fitted_plane_depth,
                fitted_plane_valid,
                update_stage,
                fitted_available,
                fitted_tested,
                fitted_accepted,
                fitted_working,
                fitted_native,
                final_incumbent,
                final_best,
                final_improvement,
                final_depth,
                final_offset,
                final_tested,
                final_finite,
                final_accepted,
            )
            if all(value is not None for value in c3_arrays):
                dispatch_reliability = reliability
                dispatch_basis = "apd_reliability_class"
                multiscale_value = manifest.get("apd_multiscale")
                multiscale = (
                    multiscale_value if isinstance(multiscale_value, dict) else {}
                )
                transfer_value = multiscale.get("transfer")
                transfer = transfer_value if isinstance(transfer_value, dict) else {}
                if iteration == 0 and transfer.get("available") is True:
                    transferred = (
                        multiscale_maps.get(("apd_transferred_reliability", None))
                        if multiscale_maps is not None else None
                    )
                    if transferred is None or transferred.shape != reliability.shape:
                        relationship_errors.append(
                            "iteration 0: transferred reliability dispatch map is unavailable"
                        )
                        dispatch_reliability = None
                    else:
                        dispatch_reliability = transferred
                        dispatch_basis = "apd_transferred_reliability"
                mismatch_pixels = None
                if dispatch_reliability is not None:
                    expected_stage = np.where(dispatch_reliability == 2, 1, 2)
                    mismatch_pixels = int(np.count_nonzero(
                        update_stage != expected_stage
                    ))
                dispatch_stats[str(iteration)] = {
                    "basis": dispatch_basis,
                    "mismatch_pixels": mismatch_pixels,
                    "transferred_iteration_zero": (
                        iteration == 0 and transfer.get("available") is True
                    ),
                }
                if mismatch_pixels:
                    relationship_errors.append(
                        f"iteration {iteration}: {dispatch_basis}/update-stage mismatch"
                    )
                fitted_valid_mask = fitted_plane_valid == 1
                fitted_available_mask = fitted_available == 1
                fitted_tested_mask = fitted_tested == 1
                fitted_accepted_mask = fitted_accepted == 1
                if np.any(fitted_valid_mask & (fitted_plane_depth <= 0.0)) or np.any(
                    ~fitted_valid_mask & (fitted_plane_depth != -1.0)
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted-plane state availability mismatch"
                    )
                if np.any(fitted_valid_mask & (eligible != 1)):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted plane exists outside APD eligibility"
                    )
                if np.any(fitted_available_mask & ~fitted_valid_mask) or np.any(
                    fitted_tested_mask != fitted_available_mask
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted-plane dispatch mismatch"
                    )
                fitted_finite_mask = fitted_working >= 0.0
                if np.any(fitted_finite_mask & ~fitted_tested_mask) or np.any(
                    fitted_accepted_mask & ~fitted_finite_mask
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted-plane score availability mismatch"
                    )
                if np.any((source == 10) & ~fitted_accepted_mask):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted-plane winner lacks acceptance"
                    )
                if np.any((fitted_native >= 0.0) & ~fitted_tested_mask):
                    relationship_errors.append(
                        f"iteration {iteration}: fitted-plane native rescore without test"
                    )

                is_final_iteration = iteration == max(expected_iterations, default=-1)
                if np.any(final_finite > final_tested):
                    relationship_errors.append(
                        f"iteration {iteration}: final-refinement finite/tested ordering failed"
                    )
                accepted_final_mask = final_accepted == 1
                final_available = final_incumbent >= 0.0
                if np.any(accepted_final_mask & (
                    (final_best < 0.0)
                    | (final_depth <= 0.0)
                    | (final_improvement <= 0.1)
                    | (final_finite == 0)
                )):
                    relationship_errors.append(
                        f"iteration {iteration}: accepted final refinement is invalid"
                    )
                if np.any(final_available & (
                    np.abs((final_incumbent - final_best) - final_improvement)
                    > APD_COMPONENT_CLOSURE_TOLERANCE
                )):
                    relationship_errors.append(
                        f"iteration {iteration}: final-refinement improvement closure failed"
                    )
                if not is_final_iteration and (
                    np.any(final_tested != 0)
                    or np.any(final_finite != 0)
                    or np.any(final_accepted != 0)
                    or np.any(final_incumbent != -1.0)
                    or np.any(final_best != -1.0)
                    or np.any(final_improvement != -1.0)
                    or np.any(final_depth != -1.0)
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: final refinement appeared before the terminal iteration"
                    )

        tested_mask = logical("apd_candidate_tested_mask", iteration)
        finite_mask = logical("apd_candidate_finite_mask", iteration)
        accepted_mask = logical("apd_candidate_accepted_mask", iteration)
        tested_count = logical("apd_candidate_tested_count", iteration)
        finite_count = logical("apd_candidate_finite_count", iteration)
        accepted_count = logical("apd_candidate_accepted_count", iteration)
        if all(value is not None for value in (
            tested_mask,
            finite_mask,
            accepted_mask,
            tested_count,
            finite_count,
            accepted_count,
        )):
            tested_uint = np.rint(tested_mask).astype(np.uint32)
            finite_uint = np.rint(finite_mask).astype(np.uint32)
            accepted_uint = np.rint(accepted_mask).astype(np.uint32)
            mismatches = {
                "tested": int(np.count_nonzero(
                    tested_count != uint32_popcount(tested_uint, candidate_slots)
                )),
                "finite": int(np.count_nonzero(
                    finite_count != uint32_popcount(finite_uint, candidate_slots)
                )),
                "accepted": int(np.count_nonzero(
                    accepted_count != uint32_popcount(accepted_uint, candidate_slots)
                )),
            }
            finite_outside_tested = int(np.count_nonzero(finite_uint & ~tested_uint))
            accepted_outside_tested = int(
                np.count_nonzero(accepted_uint & ~tested_uint)
            )
            accepted_non_usable = int(
                np.sum(uint32_popcount(
                    accepted_uint & ~finite_uint, candidate_slots
                ))
            )
            candidate_stats[str(iteration)] = {
                "count_mismatch_pixels": mismatches,
                "finite_outside_tested_pixels": finite_outside_tested,
                "accepted_outside_tested_pixels": accepted_outside_tested,
                "accepted_non_usable_events": accepted_non_usable,
                "finite_definition": "finite and strictly below fBadCost",
            }
            if any(mismatches.values()) or finite_outside_tested or accepted_outside_tested:
                relationship_errors.append(
                    f"iteration {iteration}: candidate mask/count integrity failure"
                )
            if capture_version is not None and capture_version >= 2:
                anchor_tested = uint32_popcount(
                    np.right_shift(tested_uint, np.uint32(13)) & np.uint32(0xFF),
                    8,
                )
                anchor_finite = uint32_popcount(
                    np.right_shift(finite_uint, np.uint32(13)) & np.uint32(0xFF),
                    8,
                )
                if anchor_proposal_count is not None and np.any(
                    anchor_tested != anchor_proposal_count
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: anchor proposal/candidate-mask mismatch"
                    )
                if anchor_finite_count is not None and np.any(
                    anchor_finite != anchor_finite_count
                ):
                    relationship_errors.append(
                        f"iteration {iteration}: finite anchor/candidate-mask mismatch"
                    )
            if winner_slot is not None:
                winner_bits = np.left_shift(
                    np.uint32(1), winner_slot.astype(np.uint32)
                )
                if np.any((tested_uint & winner_bits) == 0):
                    relationship_errors.append(
                        f"iteration {iteration}: winner slot was not tested"
                    )
            if runner_slot is not None:
                runner_available = runner_slot != 255
                runner_bits = np.left_shift(
                    np.uint32(1), np.minimum(runner_slot, 31).astype(np.uint32)
                )
                if np.any(runner_available & ((finite_uint & runner_bits) == 0)):
                    relationship_errors.append(
                        f"iteration {iteration}: runner-up slot is not usable"
                    )

        working = logical("apd_working_winner_cost", iteration)
        native = logical("apd_native_persistent_cost", iteration)
        runner = logical("apd_runner_up_working_cost", iteration)
        gap = logical("apd_winner_runner_up_gap", iteration)
        center = logical("apd_center_cost", iteration)
        anchor = logical("apd_anchor_mean_cost", iteration)
        deformable = logical("apd_deformable_photometric_cost", iteration)
        geometric = logical("apd_geometric_cost", iteration)
        native_minus_working = logical("apd_native_minus_working_cost", iteration)
        iteration_closure: dict[str, float] = {}
        if all(value is not None for value in (native, working, native_minus_working)):
            iteration_closure["native_minus_working"] = max_abs_difference(
                native_minus_working, native - working
            )
        if all(value is not None for value in (runner, gap, working)):
            unavailable = runner == -1.0
            expected_gap = np.where(
                unavailable, -1.0, np.maximum(runner - working, 0.0)
            )
            iteration_closure["winner_runner_up_gap"] = max_abs_difference(
                gap, expected_gap
            )
        if all(value is not None for value in (
            active,
            center,
            anchor,
            deformable,
            geometric,
            working,
        )):
            active_mask = active == 1
            inactive_component_errors = sum(
                int(np.count_nonzero(values[~active_mask] != -1.0))
                for values in (center, anchor, deformable, geometric)
            )
            active_component_errors = sum(
                int(np.count_nonzero(values[active_mask] < 0.0))
                for values in (center, anchor, deformable, geometric)
            )
            if inactive_component_errors or active_component_errors:
                relationship_errors.append(
                    f"iteration {iteration}: APD component availability mismatch"
                )
            if np.any(active_mask):
                expected_deformable = 0.25 * center + 0.75 * anchor
                iteration_closure["deformable_components"] = max_abs_difference(
                    deformable[active_mask], expected_deformable[active_mask]
                )
                iteration_closure["working_components"] = max_abs_difference(
                    working[active_mask],
                    (deformable + geometric)[active_mask],
                )
                if expect_geometric_zero:
                    iteration_closure["geometric_disabled"] = float(
                        np.max(np.abs(geometric[active_mask]))
                    )
        closure[str(iteration)] = iteration_closure

        row = rows_by_iteration.get(iteration)
        if isinstance(row, dict) and all(value is not None for value in (
            reliability,
            profile_reason,
            anchor_reason,
            ransac_valid,
            eligible,
            anchor_count,
            active,
            working,
            native,
            center,
            anchor,
            gap,
        )):
            total_pixels = width * height
            reliability_counts = {
                "unknown": int(np.count_nonzero(reliability == 0)),
                "unreliable": int(np.count_nonzero(reliability == 1)),
                "reliable": int(np.count_nonzero(reliability == 2)),
            }
            profile_counts = {
                name: int(np.count_nonzero(profile_reason == index))
                for index, name in enumerate(APD_PROFILE_REASON_NAMES)
            }
            anchor_reason_counts = {
                name: int(np.count_nonzero(anchor_reason == index))
                for index, name in enumerate(APD_ANCHOR_REASON_NAMES)
            }
            source = logical("apd_update_source", iteration)
            source_counts = {
                name: int(np.count_nonzero(source == index))
                for index, name in enumerate(source_names)
            } if source is not None else {}
            anchor_histogram = {
                str(index): int(np.count_nonzero(anchor_count == index))
                for index in range(9)
            }
            active_mask = active == 1
            available_gap = active_mask & (gap >= 0.0)
            global_minimum = logical("apd_global_minimum_cost", iteration)
            separation = logical("apd_profile_separation", iteration)
            separation_available = separation >= 0.0 if separation is not None else None
            expected_counts = {
                "classified_pixels": total_pixels,
                "reliability": reliability_counts,
                "profile_reason_counts": profile_counts,
                "anchor_count_histogram": anchor_histogram,
                "anchor_reason_counts": anchor_reason_counts,
                "ransac_valid_pixels": int(np.count_nonzero(ransac_valid == 1)),
                "deformable_eligible_pixels": int(np.count_nonzero(eligible == 1)),
                "deformable_updates": int(np.count_nonzero(active_mask)),
                "source_counts": source_counts,
                "working_gap_samples": int(np.count_nonzero(available_gap)),
            }
            actual_counts = {
                "classified_pixels": row.get("classified_pixels"),
                "reliability": {
                    key: (row.get("reliability") or {}).get(key)
                    for key in reliability_counts
                },
                "profile_reason_counts": row.get("profile_reason_counts"),
                "anchor_count_histogram": (
                    (row.get("anchor_model") or {}).get("anchor_count_histogram")
                ),
                "anchor_reason_counts": (
                    (row.get("anchor_model") or {}).get("reason_counts")
                ),
                "ransac_valid_pixels": (
                    (row.get("anchor_model") or {}).get("ransac_valid_pixels")
                ),
                "deformable_eligible_pixels": (
                    (row.get("anchor_model") or {}).get("deformable_eligible_pixels")
                ),
                "deformable_updates": (
                    (row.get("updates") or {}).get("deformable_updates")
                ),
                "source_counts": (row.get("updates") or {}).get("source_counts"),
                "working_gap_samples": (
                    (row.get("updates") or {}).get("working_gap_samples")
                ),
            }
            if capture_version is not None and capture_version >= 2:
                view_mode_counts = {
                    name: int(np.count_nonzero(active_mask & (view_selection_mode == index)))
                    for index, name in enumerate(APD_VIEW_SELECTION_MODE_NAMES)
                }
                expected_counts.update({
                    "immutable_anchor_state_updates": int(
                        np.sum(immutable_anchor_state, dtype=np.uint64)
                    ),
                    "anchor_view_selection_attempted": int(np.count_nonzero(active_mask)),
                    "anchor_view_selection_used": int(
                        np.count_nonzero(active_mask & (view_selection_mode == 1))
                    ),
                    "anchor_view_selection_mode_counts": view_mode_counts,
                    "anchor_proposals_tested": int(
                        np.sum(anchor_proposal_count, dtype=np.uint64)
                    ),
                    "anchor_proposals_finite": int(
                        np.sum(anchor_finite_count, dtype=np.uint64)
                    ),
                    "anchor_proposals_accepted": int(
                        np.count_nonzero(anchor_accepted_slot != 255)
                    ),
                    "anchor_propagation_final_winners": int(
                        np.count_nonzero(source == 9)
                    ),
                    "best_anchor_working_cost_samples": int(
                        np.count_nonzero(active_mask & (best_anchor_cost >= 0.0))
                    ),
                    "accepted_anchor_native_cost_samples": int(
                        np.count_nonzero(active_mask & (accepted_anchor_cost >= 0.0))
                    ),
                })
                actual_counts.update({
                    "immutable_anchor_state_updates": (
                        (row.get("updates") or {}).get("immutable_anchor_state_updates")
                    ),
                    "anchor_view_selection_attempted": (
                        (row.get("anchor_view_selection") or {}).get("attempted_pixels")
                    ),
                    "anchor_view_selection_used": (
                        (row.get("anchor_view_selection") or {}).get("anchor_evidence_pixels")
                    ),
                    "anchor_view_selection_mode_counts": (
                        (row.get("anchor_view_selection") or {}).get("mode_counts")
                    ),
                    "anchor_proposals_tested": (
                        (row.get("anchor_propagation") or {}).get("tested_candidates")
                    ),
                    "anchor_proposals_finite": (
                        (row.get("anchor_propagation") or {}).get("finite_candidates")
                    ),
                    "anchor_proposals_accepted": (
                        (row.get("anchor_propagation") or {}).get("accepted_events")
                    ),
                    "anchor_propagation_final_winners": (
                        (row.get("anchor_propagation") or {}).get("final_winner_pixels")
                    ),
                    "best_anchor_working_cost_samples": (
                        (row.get("anchor_propagation") or {}).get("best_working_cost_samples")
                    ),
                    "accepted_anchor_native_cost_samples": (
                        (row.get("anchor_propagation") or {}).get("accepted_native_cost_samples")
                    ),
                })
            if capture_version == 3:
                expected_counts.update({
                    "stage_counts": {
                        "all_compatibility": int(np.count_nonzero(update_stage == 0)),
                        "reliable_first": int(np.count_nonzero(update_stage == 1)),
                        "non_reliable_second": int(np.count_nonzero(update_stage == 2)),
                    },
                    "fitted_plane_available": int(
                        np.count_nonzero(fitted_available == 1)
                    ),
                    "fitted_plane_tested": int(
                        np.count_nonzero(fitted_tested == 1)
                    ),
                    "fitted_plane_finite": int(
                        np.count_nonzero(fitted_working >= 0.0)
                    ),
                    "fitted_plane_accepted": int(
                        np.count_nonzero(fitted_accepted == 1)
                    ),
                    "fitted_plane_final_winners": int(
                        np.count_nonzero(source == 10)
                    ),
                    "final_refinement_pixels": int(
                        np.count_nonzero(final_incumbent >= 0.0)
                    ),
                    "final_refinement_candidates_tested": int(
                        np.sum(final_tested, dtype=np.uint64)
                    ),
                    "final_refinement_candidates_finite": int(
                        np.sum(final_finite, dtype=np.uint64)
                    ),
                    "final_refinement_accepted": int(
                        np.count_nonzero(final_accepted == 1)
                    ),
                })
                actual_counts.update({
                    "stage_counts": (row.get("updates") or {}).get("stage_counts"),
                    "fitted_plane_available": (
                        (row.get("fitted_plane") or {}).get("available_pixels")
                    ),
                    "fitted_plane_tested": (
                        (row.get("fitted_plane") or {}).get("tested_pixels")
                    ),
                    "fitted_plane_finite": (
                        (row.get("fitted_plane") or {}).get("finite_pixels")
                    ),
                    "fitted_plane_accepted": (
                        (row.get("fitted_plane") or {}).get("accepted_events")
                    ),
                    "fitted_plane_final_winners": (
                        (row.get("fitted_plane") or {}).get("final_winner_pixels")
                    ),
                    "final_refinement_pixels": (
                        (row.get("final_native_refinement") or {}).get("eligible_pixels")
                    ),
                    "final_refinement_candidates_tested": (
                        (row.get("final_native_refinement") or {}).get("tested_candidates")
                    ),
                    "final_refinement_candidates_finite": (
                        (row.get("final_native_refinement") or {}).get("finite_candidates")
                    ),
                    "final_refinement_accepted": (
                        (row.get("final_native_refinement") or {}).get("accepted_pixels")
                    ),
                })
            count_match = expected_counts == actual_counts
            if not count_match:
                summary_errors.append(
                    f"iteration {iteration}: APD summary count closure failed"
                )

            expected_means = {
                "anchor_count_mean": float(np.mean(anchor_count, dtype=np.float64)),
                "global_minimum_cost_mean": float(
                    np.mean(global_minimum, dtype=np.float64)
                ) if global_minimum is not None else None,
                "profile_separation_mean": float(
                    np.mean(separation[separation_available], dtype=np.float64)
                ) if separation_available is not None and np.any(separation_available) else None,
                "center_cost_mean": float(np.mean(center[active_mask], dtype=np.float64)),
                "anchor_mean_cost_mean": float(np.mean(anchor[active_mask], dtype=np.float64)),
                "working_cost_mean": float(np.mean(working[active_mask], dtype=np.float64)),
                "native_persistent_cost_mean": float(
                    np.mean(native[active_mask], dtype=np.float64)
                ),
                "working_gap_mean": float(
                    np.mean(gap[available_gap], dtype=np.float64)
                ) if np.any(available_gap) else None,
            }
            actual_means = {
                "anchor_count_mean": (
                    (row.get("anchor_model") or {}).get("anchor_count_mean")
                ),
                "global_minimum_cost_mean": (
                    (row.get("global_minimum_cost") or {}).get("mean")
                ),
                "profile_separation_mean": (
                    (row.get("profile_separation") or {}).get("mean")
                ),
                "center_cost_mean": (row.get("updates") or {}).get("center_cost_mean"),
                "anchor_mean_cost_mean": (
                    (row.get("updates") or {}).get("anchor_mean_cost_mean")
                ),
                "working_cost_mean": (
                    (row.get("updates") or {}).get("working_cost_mean")
                ),
                "native_persistent_cost_mean": (
                    (row.get("updates") or {}).get("native_persistent_cost_mean")
                ),
                "working_gap_mean": (
                    (row.get("updates") or {}).get("working_gap_mean")
                ),
            }
            if capture_version is not None and capture_version >= 2:
                best_available = active_mask & (best_anchor_cost >= 0.0)
                accepted_available = active_mask & (accepted_anchor_cost >= 0.0)
                expected_means.update({
                    "best_anchor_working_cost_mean": float(
                        np.mean(best_anchor_cost[best_available], dtype=np.float64)
                    ) if np.any(best_available) else None,
                    "accepted_anchor_native_cost_mean": float(
                        np.mean(accepted_anchor_cost[accepted_available], dtype=np.float64)
                    ) if np.any(accepted_available) else None,
                })
                actual_means.update({
                    "best_anchor_working_cost_mean": (
                        (row.get("anchor_propagation") or {}).get("best_working_cost_mean")
                    ),
                    "accepted_anchor_native_cost_mean": (
                        (row.get("anchor_propagation") or {}).get("accepted_native_cost_mean")
                    ),
                })
            mean_errors = {
                name: abs(float(actual_means[name]) - expected)
                for name, expected in expected_means.items()
                if expected is not None and finite_number(actual_means.get(name)) is not None
            }
            missing_means = sorted(
                name for name, expected in expected_means.items()
                if expected is not None and finite_number(actual_means.get(name)) is None
            )
            excessive_mean_errors = {
                name: error for name, error in mean_errors.items()
                if error > APD_SUMMARY_MEAN_TOLERANCE
            }
            if missing_means or excessive_mean_errors:
                summary_errors.append(
                    f"iteration {iteration}: APD summary mean closure failed"
                )
            summary_stats[str(iteration)] = {
                "counts_match": count_match,
                "expected_counts": expected_counts,
                "actual_counts": actual_counts,
                "mean_tolerance": APD_SUMMARY_MEAN_TOLERANCE,
                "mean_absolute_errors": mean_errors,
                "missing_means": missing_means,
            }

    closure_failures = {
        iteration: values
        for iteration, values in closure.items()
        if any(
            value > max(tolerance, APD_COMPONENT_CLOSURE_TOLERANCE)
            for value in values.values()
        )
    }
    check("apd_map_domains", not domain_errors, domain_errors)
    check("apd_map_relationships", not relationship_errors, {
        "errors": relationship_errors,
        "candidate": candidate_stats,
        "dispatch": dispatch_stats,
    })
    check("apd_cost_closure", not closure_failures, {
        "tolerance": max(tolerance, APD_COMPONENT_CLOSURE_TOLERANCE),
        "per_iteration": closure,
    })
    check("apd_summary_map_closure", not summary_errors, {
        "errors": summary_errors,
        "per_iteration": summary_stats,
    })
    return {
        "requested": requested,
        "available": available,
        "schema_name": capture.get("schema_name"),
        "schema_version": capture.get("schema_version"),
        "implementation_label": implementation_label,
        "iterations": sorted(expected_iterations),
        "closure": closure,
        "candidate": candidate_stats,
        "summary": summary_stats,
    }


def validate_apd_multiscale_maps(
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    entries: list[dict[str, Any]],
    maps: dict[tuple[str, int | None], np.ndarray],
    width: int,
    height: int,
    check: Any,
) -> dict[str, Any]:
    """Validate the terminal maps and metadata for one APD stage transfer."""

    capture = manifest.get("apd_capture")
    capture = capture if isinstance(capture, dict) else {}
    requested = capture.get("requested") is True
    selected_entries = {
        str(entry.get("signal")): entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("signal")) in APD_MULTISCALE_SIGNALS
    }
    if not requested and not selected_entries:
        return {
            "requested": False,
            "available": False,
            "unavailable_reason": "APD mode is disabled for this capture",
        }

    stage_value = summary.get("apd_multiscale")
    stage = stage_value if isinstance(stage_value, dict) else {}
    manifest_stage_value = manifest.get("apd_multiscale")
    manifest_stage = (
        manifest_stage_value if isinstance(manifest_stage_value, dict) else {}
    )
    if (
        requested
        and not selected_entries
        and stage.get("schema_name") != "openmvs.dmap.apd_multiscale_stage"
        and manifest_stage.get("schema_name") != "openmvs.dmap.apd_multiscale_stage"
    ):
        return {
            "requested": True,
            "available": False,
            "unavailable_reason": "pre-C4 capture does not declare APD multiscale state",
        }
    clock = stage.get("clock") if isinstance(stage.get("clock"), dict) else {}
    transfer = (
        stage.get("transfer") if isinstance(stage.get("transfer"), dict) else {}
    )
    schedule = (
        stage.get("schedule") if isinstance(stage.get("schedule"), dict) else {}
    )
    input_state = (
        stage.get("input_state")
        if isinstance(stage.get("input_state"), dict) else {}
    )
    output_state = (
        stage.get("output_state")
        if isinstance(stage.get("output_state"), dict) else {}
    )
    transferred = transfer.get("available") is True
    conventional = schedule.get("policy") == "conventional_native"
    contract_errors: list[str] = []
    if stage.get("schema_name") != "openmvs.dmap.apd_multiscale_stage":
        contract_errors.append("unsupported or missing APD multiscale stage schema")
    if strict_int(stage.get("schema_version")) != 1:
        contract_errors.append("APD multiscale stage schema_version must be 1")
    if strict_int(stage.get("state_schema_version")) != 1:
        contract_errors.append("APD multiscale state_schema_version must be 1")
    if manifest_stage != stage:
        contract_errors.append("summary and map-manifest APD multiscale metadata differ")
    if clock.get("status") != "valid":
        contract_errors.append("APD multiscale stage clock is invalid")
    if strict_int(clock.get("level_index")) is None:
        contract_errors.append("APD multiscale level_index is missing or invalid")
    if strict_int(clock.get("stage_index")) is None:
        contract_errors.append("APD multiscale stage_index is missing or invalid")
    if transferred != (transfer.get("status") == "valid"):
        contract_errors.append("transfer availability and status disagree")
    if transferred == conventional:
        contract_errors.append("transfer availability and schedule policy disagree")
    if input_state.get("available") is not transferred:
        contract_errors.append("input-state availability and transfer status disagree")
    if output_state.get("available") is not True:
        contract_errors.append("output APD multiscale state is unavailable")
    check("apd_multiscale_contract", not contract_errors, contract_errors)

    expected_signals = set(APD_MULTISCALE_OUTPUT_SIGNALS)
    if transferred:
        expected_signals.update(APD_MULTISCALE_INPUT_SIGNALS)
    present_signals = set(selected_entries)
    metadata_errors: list[str] = []
    for signal, entry in selected_entries.items():
        if entry.get("dtype") != "uint8":
            metadata_errors.append(f"{signal}: dtype must be uint8")
        if entry.get("logical_iteration") is not None:
            metadata_errors.append(f"{signal}: logical_iteration must be absent")
        if entry.get("lossless") is not True:
            metadata_errors.append(f"{signal}: lossless must be true")
        if strict_int(entry.get("state_schema_version")) != 1:
            metadata_errors.append(f"{signal}: state_schema_version must be 1")
        is_input = signal in APD_MULTISCALE_INPUT_SIGNALS
        expected_role = (
            "multiscale_input"
            if signal == "apd_transferred_reliability" else
            "multiscale_input_provenance"
            if is_input else
            "multiscale_output"
            if signal == "apd_output_reliability" else
            "multiscale_output_provenance"
        )
        if entry.get("role") != expected_role:
            metadata_errors.append(
                f"{signal}: expected role {expected_role!r}, got {entry.get('role')!r}"
            )
        expected_quality = "exact" if is_input or not conventional else "derived_exact"
        if entry.get("measurement_quality") != expected_quality:
            metadata_errors.append(
                f"{signal}: expected measurement_quality {expected_quality!r}"
            )
        expected_basis = (
            "runtime_host_transfer_state"
            if is_input else
            "native_selected_view_mask_uniform_compatibility_classifier"
            if conventional else
            "apd_runtime_profile_and_anchor_state"
        )
        if entry.get("measurement_basis") != expected_basis:
            metadata_errors.append(
                f"{signal}: expected measurement_basis {expected_basis!r}"
            )
    check("apd_multiscale_map_coverage", present_signals == expected_signals, {
        "expected": sorted(expected_signals),
        "present": sorted(present_signals),
        "missing": sorted(expected_signals - present_signals),
        "unexpected": sorted(present_signals - expected_signals),
    })
    check("apd_multiscale_map_metadata", not metadata_errors, metadata_errors)

    domain_errors: list[str] = []
    map_stats: dict[str, dict[str, Any]] = {}
    for signal in sorted(present_signals):
        data = maps.get((signal, None))
        if data is None:
            domain_errors.append(f"{signal}: decoded map is unavailable")
            continue
        values = np.asarray(data)
        if signal.endswith("reliability"):
            invalid = int(np.count_nonzero((values < 0) | (values > 2)))
            counts = np.bincount(values.reshape(-1).astype(np.uint8), minlength=3)
            map_stats[signal] = {
                "num_pixels": int(values.size),
                "unknown": int(counts[0]),
                "unreliable": int(counts[1]),
                "reliable": int(counts[2]),
                "reliable_ratio": float(counts[2] / values.size),
            }
        else:
            maximum = int(np.max(values)) if values.size else 0
            limit = 1 if signal.endswith("deformable_eligible") else 8
            invalid = int(np.count_nonzero((values < 0) | (values > limit)))
            nonzero = int(np.count_nonzero(values))
            map_stats[signal] = {
                "num_pixels": int(values.size),
                "nonzero": nonzero,
                "nonzero_ratio": float(nonzero / values.size),
                "maximum": maximum,
            }
        if invalid:
            domain_errors.append(f"{signal}: {invalid} values are outside its domain")
    check("apd_multiscale_map_domains", not domain_errors, domain_errors)

    stats_errors: list[str] = []
    state_specs = (
        ("input", input_state, "apd_transferred"),
        ("output", output_state, "apd_output"),
    )
    for state_name, state_metadata, signal_prefix in state_specs:
        if state_name == "input" and not transferred:
            continue
        for suffix, metadata_key in (
            ("reliability", "reliability"),
            ("anchor_count", "anchor_count_provenance"),
            ("deformable_eligible", "deformable_eligible_provenance"),
        ):
            signal = f"{signal_prefix}_{suffix}"
            observed = map_stats.get(signal)
            expected = state_metadata.get(metadata_key)
            if not isinstance(observed, dict) or not isinstance(expected, dict):
                stats_errors.append(f"{signal}: map or metadata statistics unavailable")
                continue
            for key, observed_value in observed.items():
                expected_value = expected.get(key)
                if isinstance(observed_value, float):
                    if finite_number(expected_value) is None or abs(
                        observed_value - float(expected_value)
                    ) > 1.0e-12:
                        stats_errors.append(f"{signal}: {key} statistics differ")
                elif expected_value != observed_value:
                    stats_errors.append(f"{signal}: {key} statistics differ")
            if observed.get("num_pixels") != width * height:
                stats_errors.append(f"{signal}: statistics extent differs from frame")
    check("apd_multiscale_map_statistics", not stats_errors, stats_errors)
    return {
        "requested": requested,
        "available": (
            not contract_errors
            and present_signals == expected_signals
            and not metadata_errors
            and not domain_errors
            and not stats_errors
        ),
        "transferred": transferred,
        "schedule_policy": schedule.get("policy"),
        "level_index": clock.get("level_index"),
        "stage_index": clock.get("stage_index"),
        "signals": sorted(present_signals),
        "statistics": map_stats,
    }


def validate(arguments: Arguments) -> dict[str, Any]:
    frame_dir = arguments.frame_dir.expanduser().resolve()
    if (frame_dir / "prefilter_manifest.json").is_file():
        return validate_prefilter_only(arguments, frame_dir)
    if not (frame_dir / "map_manifest.json").is_file() and (
        frame_dir / "summary_complete.json"
    ).is_file():
        return validate_summary_only(arguments, frame_dir)
    manifest = load_json(frame_dir / "map_manifest.json")
    summary = load_json(frame_dir / "summary.json")
    decay_metadata = low_texture_decay_metadata(frame_dir)
    view_samples_metadata = exact_view_samples_metadata(frame_dir)
    hysteresis_metadata = low_texture_update_hysteresis_metadata(frame_dir)
    low_texture_decay_scale = (
        decay_metadata["low_texture_decay_scale"]
        if decay_metadata.get("available") is True else None
    )
    width = int(manifest.get("width", 0))
    height = int(manifest.get("height", 0))
    num_passes = int(manifest.get("num_passes", 0))
    entries = manifest.get("maps") or []
    schema_version = strict_int(manifest.get("schema_version"))
    summary_schema_version = strict_int(summary.get("schema_version"))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("schema_version", schema_version in {2, 3, 4} and summary_schema_version == schema_version, {
        "manifest": manifest.get("schema_version"), "summary": summary.get("schema_version"),
        "supported": [2, 3, 4],
    })
    check("dimensions", width > 0 and height > 0 and summary.get("width") == width and summary.get("height") == height, {
        "width": width, "height": height
    })
    patch_layout_valid, patch_layout_detail = (
        validate_reference_patch_layout_contract(frame_dir, summary)
    )
    check(
        "reference_patch_layout_contract",
        patch_layout_valid,
        patch_layout_detail,
    )
    if schema_version == 4:
        summary_sidecars_valid, summary_sidecars = observer_sidecars_complete(summary)
        manifest_sidecars_valid, manifest_sidecars = observer_sidecars_complete(manifest)
        sidecars_valid = (
            summary_sidecars_valid
            and manifest_sidecars_valid
            and summary_sidecars == manifest_sidecars
        )
        check("v4_observer_sidecars_complete", sidecars_valid, {
            "summary": summary_sidecars,
            "manifest": manifest_sidecars,
        })

        marker_path = frame_dir / "capture_complete.json"
        marker = load_json(marker_path) if marker_path.is_file() else {}
        marker_manifest = marker.get("map_manifest") if isinstance(marker.get("map_manifest"), dict) else {}
        marker_summary = marker.get("summary") if isinstance(marker.get("summary"), dict) else {}
        marker_valid = (
            marker.get("schema_name") == "openmvs.dmap.capture_complete"
            and marker.get("schema_version") == 1
            and marker.get("capture_kind") == "maps"
            and marker.get("maps_complete") is True
            and marker.get("observer_sidecars_complete") is True
            and marker_manifest.get("path") == "map_manifest.json"
            and marker_manifest.get("schema_version") == schema_version
            and marker_manifest.get("complete") is True
            and marker_summary.get("path") == "summary.json"
            and marker_summary.get("schema_version") == summary_schema_version
            and marker.get("image_id") == summary.get("image_id")
            and marker.get("image_name") == summary.get("image_name")
            and marker.get("estimation_stage") == summary.get("estimation_stage")
            and marker.get("geometric_iteration") == summary.get("geometric_iteration")
        )
        check("v4_capture_completion_marker", marker_valid, marker)
        expected_completion = {
            "schema_name": "openmvs.dmap.capture_complete",
            "schema_version": 1,
            "path": "capture_complete.json",
            "maps_complete": True,
            "eligible": True,
        }
        check(
            "v4_summary_completion_reference",
            summary.get("completion_marker") == expected_completion,
            {"actual": summary.get("completion_marker"), "expected": expected_completion},
        )
    expected_measurement_model = {
        3: V3_MEASUREMENT_MODEL,
        4: V4_MEASUREMENT_MODEL,
    }.get(schema_version, "production PatchMatch plus post-pass snapshot diagnostics")
    check(
        "measurement_model",
        manifest.get("measurement_model") == expected_measurement_model,
        {"actual": manifest.get("measurement_model"), "expected": expected_measurement_model},
    )
    exact_available = bool((manifest.get("exact_capture") or {}).get("available"))
    apd_capture = manifest.get("apd_capture") or {}
    apd_available = bool(
        schema_version == 4
        and isinstance(apd_capture, dict)
        and apd_capture.get("requested") is True
        and apd_capture.get("maps_available") is True
    )
    expected_candidate_mode = (
        "exact_apd_working_objective_full_frame" if apd_available
        else "exact_production_hot_kernel_full_frame"
        if schema_version == 4 and exact_available
        else "unavailable_post_pass_snapshot"
    )
    expected_gap_mode = (
        "exact_apd_working_winner_runner_up_full_frame" if apd_available
        else "exact_process_pixel_winner_runner_up_full_frame"
        if schema_version == 4 and exact_available
        else "post_pass_current_plus_eight_neighbors"
    )
    check("candidate_accounting_mode", summary.get("candidate_accounting_mode") == expected_candidate_mode, summary.get("candidate_accounting_mode"))
    check("confidence_gap_mode", summary.get("confidence_gap_mode") == expected_gap_mode, summary.get("confidence_gap_mode"))

    core_declared_paths = {
        path for entry in entries
        if (path := safe_relative_path(entry.get("path"))) is not None
    }
    optional_artifacts: dict[str, dict[str, Any]] = {}
    optional_records: dict[str, list[tuple[dict[str, Any], Path, np.ndarray]]] = {}
    optional_owned_paths: set[str] = set()
    optional_declared_by_name: dict[str, set[str]] = {}
    for artifact_name, _schema_name, _schema_versions, _owned_directory in OPTIONAL_MAP_MANIFESTS:
        artifact_path = frame_dir / artifact_name
        if not artifact_path.is_file():
            continue
        try:
            artifact = load_json(artifact_path)
        except Exception as exc:
            check(f"{Path(artifact_name).stem}_json", False, str(exc))
            artifact = {}
        optional_artifacts[artifact_name] = artifact
        optional_declared_by_name[artifact_name] = {
            path for entry in artifact.get("maps") or []
            if isinstance(entry, dict)
            and (path := safe_relative_path(entry.get("path"))) is not None
        }
    for artifact_name, schema_name, schema_versions, owned_directory in OPTIONAL_MAP_MANIFESTS:
        artifact = optional_artifacts.get(artifact_name)
        if artifact is None:
            continue
        other_optional_paths = set().union(*(
            paths for name, paths in optional_declared_by_name.items()
            if name != artifact_name
        )) if len(optional_declared_by_name) > 1 else set()
        records, owned_paths = optional_manifest_maps(
            frame_dir=frame_dir,
            artifact=artifact,
            artifact_name=artifact_name,
            expected_schema_name=schema_name,
            expected_schema_versions=schema_versions,
            owned_directory=owned_directory,
            width=width,
            height=height,
            core_declared_paths=core_declared_paths,
            other_optional_paths=other_optional_paths,
            check=check,
        )
        optional_records[artifact_name] = records
        optional_owned_paths.update(owned_paths)

    if schema_version in {3, 4}:
        declared_paths = [str(entry.get("path", "")) for entry in entries]
        unsafe_paths = [
            path for path in declared_paths
            if not path or Path(path).is_absolute() or ".." in Path(path).parts
        ]
        duplicate_paths = sorted({path for path in declared_paths if declared_paths.count(path) > 1})
        actual_paths = []
        unsafe_owned_paths = []
        for path in frame_dir.rglob("*"):
            if path.suffix.lower() not in {".pfm", ".png"}:
                continue
            relative = path.relative_to(frame_dir).as_posix()
            if relative in optional_owned_paths:
                continue
            _resolved, path_error = owned_regular_artifact_path(frame_dir, relative)
            if path_error:
                if path.exists() or path.is_symlink():
                    unsafe_owned_paths.append(f"{relative}: {path_error}")
                continue
            actual_paths.append(relative)
        actual_paths.sort()
        unindexed_paths = sorted(set(actual_paths) - set(declared_paths))
        stale_paths = sorted(set(declared_paths) - set(actual_paths))
        byte_errors = []
        phase_entries = []
        for entry in entries:
            relative_path = str(entry.get("path", ""))
            path, path_error = owned_regular_artifact_path(frame_dir, relative_path)
            byte_count = strict_int(entry.get("bytes"))
            if byte_count is None or byte_count <= 0:
                byte_errors.append(f"{relative_path}: bytes must be a positive integer")
            elif path_error or path is None:
                byte_errors.append(f"{relative_path}: {path_error}")
            elif byte_count != path.stat().st_size:
                byte_errors.append(
                    f"{relative_path}: bytes={byte_count} does not match file size {path.stat().st_size}"
                )
            phase_fields = sorted(key for key in entry if "phase" in str(key).lower())
            if phase_fields:
                phase_entries.append(f"{relative_path}: {', '.join(phase_fields)}")
        expected_count = strict_int(manifest.get("expected_map_count"))
        written_count = strict_int(manifest.get("written_map_count"))
        write_errors = manifest.get("write_errors")
        completion_valid = (
            manifest.get("complete") is True
            and expected_count == len(entries)
            and written_count == len(entries)
            and isinstance(write_errors, list)
            and not write_errors
        )
        check("v3_manifest_complete", completion_valid, {
            "complete": manifest.get("complete"),
            "expected_map_count": expected_count,
            "written_map_count": written_count,
            "actual_entries": len(entries),
            "write_errors": write_errors,
        })
        check("v3_manifest_paths_safe_unique", not unsafe_paths and not duplicate_paths and not unsafe_owned_paths, {
            "unsafe_paths": unsafe_paths,
            "unsafe_owned_paths": unsafe_owned_paths,
            "duplicate_paths": duplicate_paths,
        })
        check("v3_manifest_indexes_all_map_files", not unindexed_paths and not stale_paths, {
            "unindexed_paths": unindexed_paths,
            "stale_paths": stale_paths,
        })
        check("v3_manifest_file_bytes", not byte_errors, byte_errors)
        check("v3_manifest_phase_free", not phase_entries, phase_entries)

    maps: dict[tuple[str, int | None], np.ndarray] = {}
    logical_maps: dict[tuple[str, int], tuple[dict[str, Any], Path, np.ndarray]] = {}
    logical_view_maps: dict[
        tuple[str, int, int], tuple[dict[str, Any], Path, np.ndarray]
    ] = {}
    missing_paths: list[str] = []
    shape_errors: list[str] = []
    duplicate_keys: list[str] = []
    map_read_errors: list[str] = []
    for entry in entries:
        signal = str(entry.get("signal", ""))
        logical_iteration = (
            strict_int(entry.get("logical_iteration"))
            if schema_version in {3, 4} and "logical_iteration" in entry
            else None
        )
        pass_index = int(entry["pass_index"]) if "pass_index" in entry else None
        source_view_index = strict_int(entry.get("source_view_index"))
        if schema_version in {3, 4} and "logical_iteration" in entry:
            if logical_iteration is None:
                duplicate_key = None
            elif source_view_index is not None:
                logical_key = (signal, logical_iteration, source_view_index)
                duplicate_key = logical_key in logical_view_maps
            else:
                logical_key = (signal, logical_iteration)
                duplicate_key = logical_key in logical_maps
        else:
            key = (signal, pass_index)
            duplicate_key = key in maps
        if duplicate_key:
            duplicate_keys.append(
                f"{signal}:logical:{logical_iteration}:view:{source_view_index}"
                if source_view_index is not None else f"{signal}:logical:{logical_iteration}"
                if logical_iteration is not None else f"{signal}:{pass_index}"
            )
            continue
        path, path_error = owned_regular_artifact_path(frame_dir, entry.get("path"))
        if path_error or path is None:
            missing_paths.append(str(entry.get("path", "")))
            if path_error:
                map_read_errors.append(f"{signal}: {path_error}")
            continue
        try:
            data = read_manifest_map(frame_dir, entry)
        except Exception as exc:
            if schema_version not in {3, 4}:
                raise
            map_read_errors.append(f"{signal}: {exc}")
            continue
        if schema_version in {3, 4} and "logical_iteration" in entry:
            if logical_iteration is not None:
                if source_view_index is not None:
                    logical_view_maps[(signal, logical_iteration, source_view_index)] = (
                        entry, path, data
                    )
                else:
                    logical_maps[(signal, logical_iteration)] = (entry, path, data)
        else:
            maps[(signal, pass_index)] = data
        dtype = entry.get("dtype")
        expected_shape = (
            (height, width, 3) if dtype in {"float32x3", "uint8x3"}
            else (height, width, 4) if dtype == "uint8x4"
            else (height, width)
        )
        if data.shape != expected_shape:
            map_index = f"logical:{logical_iteration}" if logical_iteration is not None else str(pass_index)
            shape_errors.append(f"{signal}:{map_index} expected {expected_shape}, got {data.shape}")
    check("map_paths", not missing_paths, missing_paths)
    check("map_shapes", not shape_errors and not map_read_errors, {
        "shape_errors": shape_errors,
        "read_errors": map_read_errors,
    } if schema_version in {3, 4} else shape_errors)
    check("map_keys_unique", not duplicate_keys, duplicate_keys)

    v3_logical_state: dict[str, Any] | None = None
    v3_logical_event: dict[str, Any] | None = None
    v4_exact: dict[str, Any] | None = None
    apd_validation: dict[str, Any] | None = None
    apd_multiscale_validation: dict[str, Any] | None = None
    if schema_version in {3, 4}:
        v3_logical_state = validate_v3_logical_states(
            manifest=manifest,
            entries=entries,
            logical_maps=logical_maps,
            duplicate_keys=duplicate_keys,
            width=width,
            height=height,
            tolerance=arguments.tolerance,
            low_texture_decay_scale=low_texture_decay_scale,
            decay_metadata=decay_metadata,
            expect_geometric_zero=arguments.expect_geometric_zero,
            check=check,
        )
        v3_logical_event = validate_v3_logical_events(
            manifest=manifest,
            entries=entries,
            logical_maps=logical_maps,
            duplicate_keys=duplicate_keys,
            width=width,
            height=height,
            tolerance=arguments.tolerance,
            check=check,
        )
        v3_logical_event[LOGICAL_COST_IMPROVEMENT_SIGNAL] = (
            validate_logical_cost_improvement(
                frame_dir=frame_dir,
                summary=summary,
                manifest=manifest,
                entries=entries,
                logical_maps=logical_maps,
                duplicate_keys=duplicate_keys,
                width=width,
                height=height,
                tolerance=arguments.tolerance,
                check=check,
            )
        )
    if schema_version == 4:
        v4_exact = validate_v4_exact(
            frame_dir=frame_dir,
            manifest=manifest,
            entries=entries,
            logical_maps=logical_maps,
            logical_view_maps=logical_view_maps,
            width=width,
            height=height,
            tolerance=arguments.tolerance,
            low_texture_decay_scale=low_texture_decay_scale,
            decay_metadata=decay_metadata,
            view_samples_metadata=view_samples_metadata,
            hysteresis_metadata=hysteresis_metadata,
            check=check,
        )
        apd_validation = validate_apd_maps(
            manifest=manifest,
            summary=summary,
            entries=entries,
            logical_maps=logical_maps,
            width=width,
            height=height,
            tolerance=arguments.tolerance,
            expect_geometric_zero=arguments.expect_geometric_zero,
            check=check,
            multiscale_maps=maps,
        )
        apd_multiscale_validation = validate_apd_multiscale_maps(
            manifest=manifest,
            summary=summary,
            entries=entries,
            maps=maps,
            width=width,
            height=height,
            check=check,
        )

    final_signals = {signal for signal, pass_index in maps if pass_index is None}
    check("required_final_signals", REQUIRED_FINAL_SIGNALS <= final_signals, sorted(REQUIRED_FINAL_SIGNALS - final_signals))
    if schema_version == 2:
        pass_counts = {
            signal: sum(
                1 for key_signal, pass_index in maps
                if key_signal == signal and pass_index is not None
            )
            for signal in REQUIRED_PASS_SIGNALS
        }
        check("raw_map_capture_complete", all(count == num_passes for count in pass_counts.values()), pass_counts)

    photo_prior = maps.get(("cost_photo_prior", None))
    geometric = maps.get(("cost_geometric", None))
    total = maps.get(("cost_total_components", None))
    component_residual = float("inf")
    geometric_max_abs = float("inf")
    if photo_prior is not None and geometric is not None and total is not None:
        component_residual = max_abs_difference(total, photo_prior + geometric)
        geometric_max_abs = float(np.max(np.abs(geometric[np.isfinite(geometric)]))) if np.isfinite(geometric).any() else float("inf")
    component_tolerance = max(
        arguments.tolerance, FLOAT32_COMPONENT_CLOSURE_TOLERANCE
    )
    check(
        "component_reconstruction",
        component_residual <= component_tolerance,
        {"max_abs": component_residual, "tolerance": component_tolerance},
    )
    if arguments.expect_geometric_zero:
        check("geometric_disabled_zero", geometric_max_abs <= arguments.tolerance, geometric_max_abs)

    gap = maps.get(("confidence_gap", None))
    gap_stats = {"unavailable_pixels": 0, "finite_nonnegative_pixels": 0, "invalid_domain_pixels": width * height}
    if gap is not None:
        finite = np.isfinite(gap)
        unavailable = finite & (gap == -1.0)
        nonnegative = finite & (gap >= 0.0)
        invalid = ~(unavailable | nonnegative)
        gap_stats = {
            "unavailable_pixels": int(unavailable.sum()),
            "finite_nonnegative_pixels": int(nonnegative.sum()),
            "invalid_domain_pixels": int(invalid.sum()),
        }
    check("confidence_gap_domain", gap_stats["invalid_domain_pixels"] == 0, gap_stats)

    iteration_stats: dict[str, Any] = {}
    if schema_version in {3, 4}:
        num_iterations = strict_int(manifest.get("num_iterations"))
        logical_iterations = range(-1, num_iterations) if num_iterations is not None else ()
        for logical_iteration in logical_iterations:
            depth_record = logical_maps.get(("depth_delta", logical_iteration))
            normal_record = logical_maps.get(("normal_angle_delta", logical_iteration))
            if depth_record is None or normal_record is None:
                continue
            depth_delta = depth_record[2]
            normal_delta = normal_record[2]
            label = (
                "initialization" if logical_iteration == -1
                else f"iteration {logical_iteration + 1}"
            )
            iteration_stats[label] = {
                "nonzero_depth_pixels": int(np.count_nonzero(depth_delta)),
                "nonzero_normal_pixels": int(np.count_nonzero(normal_delta)),
                "max_abs_depth_delta": float(np.max(np.abs(depth_delta))),
                "max_normal_delta_deg": float(np.max(normal_delta)),
            }
    else:
        num_iteration_stages = 1 + max(0, (num_passes - 1) // 2)
        for stage_index in range(num_iteration_stages):
            raw_indices = [0] if stage_index == 0 else [stage_index * 2 - 1, stage_index * 2]
            depth_parts = [maps.get(("depth_delta", index)) for index in raw_indices]
            normal_parts = [maps.get(("normal_angle_delta", index)) for index in raw_indices]
            if any(part is None for part in depth_parts) or any(part is None for part in normal_parts):
                continue
            depth_delta = sum(depth_parts[1:], depth_parts[0].copy())
            normal_delta = sum(normal_parts[1:], normal_parts[0].copy())
            label = "initialization" if stage_index == 0 else f"iteration {stage_index - 1}"
            iteration_stats[label] = {
                "nonzero_depth_pixels": int(np.count_nonzero(depth_delta)),
                "nonzero_normal_pixels": int(np.count_nonzero(normal_delta)),
                "max_abs_depth_delta": float(np.max(np.abs(depth_delta))),
                "max_normal_delta_deg": float(np.max(normal_delta)),
            }
    initialization = iteration_stats.get("initialization", {})
    check(
        "initialization_delta_zero",
        initialization.get("nonzero_depth_pixels") == 0 and initialization.get("nonzero_normal_pixels") == 0,
        initialization,
    )

    parity: dict[str, float] = {}
    dmap_consistency: dict[str, float] = {}
    dmap_consistency_limits: dict[str, float] = {}
    dmap_terminal_state: dict[str, dict[str, Any]] = {}
    instrumented: dict[str, Any] | None = None
    if arguments.instrumented_dmap is not None:
        instrumented = load_dmap(arguments.instrumented_dmap.expanduser().resolve())
        core_depth = maps.get(("depth_final_after_filter", None))
        core_normal = maps.get(("normal_final", None))
        core_cost = maps.get(("cost_final", None))
        core_confidence = (
            np.maximum(1.0 - core_cost, 0.0) if core_cost is not None else None
        )
        postprocess_artifact = optional_artifacts.get("postprocess_filters.json")
        postprocess_records = optional_records.get("postprocess_filters.json", [])
        depth_state = postprocess_terminal_channel(
            channel="depth",
            artifact=postprocess_artifact,
            records=postprocess_records,
            fallback=core_depth,
            fallback_source="maps/depth_final_after_filter.pfm",
        )
        normal_state = postprocess_terminal_channel(
            channel="normal",
            artifact=postprocess_artifact,
            records=postprocess_records,
            fallback=core_normal,
            fallback_source="maps/normal_final.pfm",
        )
        confidence_state = postprocess_terminal_channel(
            channel="confidence",
            artifact=postprocess_artifact,
            records=postprocess_records,
            fallback=core_confidence,
            fallback_source="derived:max(1-maps/cost_final.pfm,0)",
        )
        confidence_state = confidence_adjustment_terminal(
            artifact=optional_artifacts.get("confidence_adjustment.json"),
            records=optional_records.get("confidence_adjustment.json", []),
            fallback_state=confidence_state,
        )
        terminal_states = {
            "depth_map": depth_state,
            "normal_map": normal_state,
            "confidence_map": confidence_state,
        }
        expected_channels = [key for key in terminal_states if key in instrumented]
        availability_valid = bool(expected_channels)
        for key in expected_channels:
            state = terminal_states[key]
            available = bool(state.get("available")) and isinstance(state.get("array"), np.ndarray)
            availability_valid = availability_valid and available
            dmap_terminal_state[key] = {
                "available": available,
                "source": state.get("source"),
                "unavailable_reason": state.get("unavailable_reason"),
            }
            if available:
                consistency_key = "confidence_from_cost" if key == "confidence_map" else key
                dmap_consistency[consistency_key] = max_abs_difference(
                    state["array"], instrumented[key]
                )
                consistency_limit = arguments.tolerance
                if instrumented.get("format") == "D2":
                    if key == "depth_map":
                        finite = np.asarray(state["array"], dtype=np.float64)
                        finite = finite[np.isfinite(finite)]
                        maximum_depth = float(np.max(np.abs(finite))) if finite.size else 0.0
                        consistency_limit = max(
                            consistency_limit,
                            maximum_depth / 1024.0 + arguments.tolerance,
                        )
                    elif key == "normal_map":
                        consistency_limit = max(
                            consistency_limit, 2.0 / 32767.0 + arguments.tolerance
                        )
                    elif key == "confidence_map":
                        confidence_scale = float(instrumented.get("confidence_scale") or 1.0)
                        consistency_limit = max(
                            consistency_limit,
                            confidence_scale / (2.0 * 255.0)
                            + max(
                                arguments.tolerance,
                                4.0 * np.finfo(np.float32).eps
                                * max(1.0, abs(confidence_scale)),
                            ),
                        )
                dmap_consistency_limits[consistency_key] = consistency_limit
        check(
            "instrumented_dmap_terminal_availability",
            availability_valid,
            dmap_terminal_state,
        )
        check(
            "instrumented_dmap_consistency",
            availability_valid
            and len(dmap_consistency) == len(expected_channels)
            and all(
                value <= dmap_consistency_limits.get(key, arguments.tolerance)
                for key, value in dmap_consistency.items()
            ),
            {"max_abs": dmap_consistency, "limits": dmap_consistency_limits},
        )
    if arguments.reference_dmap is not None:
        if instrumented is None:
            raise ValueError("--reference-dmap requires --instrumented-dmap")
        reference = load_dmap(arguments.reference_dmap.expanduser().resolve())
        for key in ("depth_map", "normal_map", "confidence_map"):
            if key in instrumented and key in reference:
                parity[key] = max_abs_difference(instrumented[key], reference[key])
        check("production_output_parity", len(parity) == 3 and all(value <= arguments.tolerance for value in parity.values()), parity)

    total_pixels = width * height
    summary_total = strict_int(summary.get("num_pixels_total"))
    valid_before = strict_int(summary.get("num_valid_before_filter"))
    invalid_before = strict_int(summary.get("num_invalid_before_filter"))
    valid_after = strict_int(summary.get("num_valid_after_filter"))
    rejected = strict_int(summary.get("num_rejected_by_filter"))
    summary_count_values = (
        summary_total, valid_before, invalid_before, valid_after, rejected
    )
    summary_count_domains = (
        all(value is not None and value >= 0 for value in summary_count_values)
        and summary_total == total_pixels
        and valid_after <= valid_before <= summary_total
        and invalid_before <= summary_total
        and rejected <= valid_before
    )
    summary_counts_consistent = (
        summary_count_domains
        and valid_before + invalid_before == total_pixels
        and valid_after + rejected == valid_before
    )
    check("summary_pixel_counts", summary_counts_consistent, {
        "total": summary_total,
        "valid_before": valid_before,
        "invalid_before": invalid_before,
        "valid_after": valid_after,
        "rejected": rejected,
    })
    check("summary_count_domains", summary_count_domains, {
        "expected_total": total_pixels,
        "total": summary_total,
        "valid_before": valid_before,
        "invalid_before": invalid_before,
        "valid_after": valid_after,
        "rejected": rejected,
    })

    if "supporting_view_histogram" in summary:
        support_value = summary.get("supporting_view_histogram")
        support_histogram = (
            [strict_int(value) for value in support_value]
            if isinstance(support_value, list)
            else []
        )
        support_histogram_valid = (
            summary_count_domains
            and len(support_histogram) == 5
            and all(value is not None and value >= 0 for value in support_histogram)
            and sum(support_histogram) == summary_total
            and support_histogram[0] == summary_total - valid_after
            and sum(support_histogram[1:]) == valid_after
        )
        check("support_histogram_final_validity", support_histogram_valid, {
            "histogram": support_value,
            "expected_zero_support_pixels": (
                summary_total - valid_after
                if summary_total is not None and valid_after is not None
                else None
            ),
            "expected_positive_support_pixels": valid_after,
        })

    filtering_path = frame_dir / "filtering.json"
    ignore_mask_validation: dict[str, Any] | None = None
    if filtering_path.is_file():
        filtering = load_json(filtering_path)
        filter_total = strict_int(filtering.get("num_pixels_total"))
        filter_valid_before = strict_int(filtering.get("num_valid_before_filter"))
        filter_invalid_before = strict_int(filtering.get("num_invalid_before_filter"))
        valid_after_keep = strict_int(filtering.get("num_valid_after_keep_cost_filter"))
        rejected_keep = strict_int(filtering.get("num_rejected_by_keep_cost_filter"))
        filter_valid_after = strict_int(filtering.get("num_valid_after_filter"))
        filter_rejected = strict_int(filtering.get("num_rejected_by_filter"))

        reasons_value = filtering.get("rejection_reasons")
        reasons = reasons_value if isinstance(reasons_value, dict) else {}
        required_reasons = {
            "low_score", "insufficient_view_support", "geometric_inconsistency",
            "normal_inconsistency", "depth_range", "occlusion", "masked",
            "small_component", "unknown",
        }
        masked_reason = reasons.get("masked")

        has_ignore_mask_contract = "ignore_mask" in filtering
        ignore_mask = filtering.get("ignore_mask")
        rejected_mask_value = filtering.get("num_rejected_by_ignore_mask")
        rejected_mask = strict_int(rejected_mask_value)
        mask_status = "legacy_unspecified"
        mask_count_available = rejected_mask is not None
        mask_metadata_valid = not has_ignore_mask_contract
        if has_ignore_mask_contract and isinstance(ignore_mask, dict):
            requested = ignore_mask.get("requested")
            label = strict_int(ignore_mask.get("label"))
            load_attempted = ignore_mask.get("load_attempted")
            loaded = ignore_mask.get("loaded")
            mask_status = ignore_mask.get("status")
            source_path = ignore_mask.get("source_path")
            count_available_value = ignore_mask.get("rejection_count_available")
            unavailable_reason = ignore_mask.get("unavailable_reason")
            label_valid = label is not None and 0 <= label <= 255
            if requested is False:
                mask_metadata_valid = (
                    ignore_mask.get("label") is None
                    and load_attempted is False
                    and loaded is None
                    and mask_status == "not_requested"
                    and source_path is None
                    and count_available_value is True
                    and not nonempty_text(unavailable_reason)
                    and rejected_mask == 0
                    and strict_int(masked_reason) == 0
                )
                mask_count_available = rejected_mask is not None
            elif requested is True and loaded is True:
                mask_metadata_valid = (
                    label_valid
                    and load_attempted is True
                    and mask_status == "loaded"
                    and nonempty_text(source_path)
                    and count_available_value is True
                    and not nonempty_text(unavailable_reason)
                    and rejected_mask is not None
                    and rejected_mask >= 0
                    and strict_int(masked_reason) is not None
                )
                mask_count_available = rejected_mask is not None
            elif requested is True and loaded is False:
                mask_metadata_valid = (
                    label_valid
                    and load_attempted is True
                    and mask_status == "unavailable"
                    and nonempty_text(source_path)
                    and count_available_value is False
                    and nonempty_text(unavailable_reason)
                    and rejected_mask_value is None
                    and masked_reason is None
                )
                mask_count_available = False
            else:
                mask_metadata_valid = False
                mask_count_available = False
        elif has_ignore_mask_contract:
            mask_metadata_valid = False
            mask_count_available = False

        summary_mask_matches = (
            summary.get("ignore_mask") == ignore_mask
            if has_ignore_mask_contract or "ignore_mask" in summary
            else True
        )
        if has_ignore_mask_contract:
            mask_metadata_valid = mask_metadata_valid and summary_mask_matches
        elif rejected_mask is None or rejected_mask < 0 or strict_int(masked_reason) is None:
            mask_metadata_valid = False
        ignore_mask_validation = {
            "contract": "explicit" if has_ignore_mask_contract else "legacy_unspecified",
            "status": mask_status,
            "count_available": mask_count_available,
            "rejected_count": rejected_mask_value,
            "masked_reason_count": masked_reason,
            "summary_matches": summary_mask_matches,
        }
        check("ignore_mask_availability", mask_metadata_valid, ignore_mask_validation)

        reason_counts_valid = (
            isinstance(reasons_value, dict)
            and required_reasons.issubset(reasons)
            and all(
                (name == "masked" and not mask_count_available and value is None)
                or (strict_int(value) is not None and strict_int(value) >= 0)
                for name, value in reasons.items()
            )
        )
        reason_total = (
            sum(value for value in (strict_int(item) for item in reasons.values()) if value is not None)
            if reason_counts_valid
            else None
        )
        core_filter_counts = (
            filter_total, filter_valid_before, filter_invalid_before,
            valid_after_keep, rejected_keep, filter_valid_after, filter_rejected,
        )
        filter_count_domains = (
            all(value is not None and value >= 0 for value in core_filter_counts)
            and filter_total == total_pixels
            and filter_valid_after <= valid_after_keep <= filter_valid_before <= filter_total
            and filter_invalid_before <= filter_total
            and rejected_keep <= filter_valid_before
            and filter_rejected <= filter_valid_before
            and (
                rejected_mask is not None
                and 0 <= rejected_mask <= valid_after_keep
                if mask_count_available
                else mask_status == "unavailable" and rejected_mask_value is None
            )
        )
        check("filter_count_domains", filter_count_domains, {
            "total": filter_total,
            "valid_before": filter_valid_before,
            "invalid_before": filter_invalid_before,
            "valid_after_keep_cost": valid_after_keep,
            "valid_after_all_filters": filter_valid_after,
            "rejected_keep_cost": rejected_keep,
            "rejected_ignore_mask": rejected_mask_value,
            "rejected_all_filters": filter_rejected,
        })

        keep_ratio = finite_number(filtering.get("valid_ratio_after_keep_cost_filter"))
        summary_keep_ratio = finite_number(summary.get("valid_ratio_after_keep_cost_filter"))
        ratio_required = schema_version == 4 or keep_ratio is not None or summary_keep_ratio is not None
        expected_keep_ratio = (
            valid_after_keep / filter_total
            if valid_after_keep is not None and filter_total is not None and filter_total > 0
            else None
        )
        ratio_tolerance = max(arguments.tolerance, 1e-6)
        keep_ratio_valid = (
            not ratio_required
            or (
                keep_ratio is not None
                and summary_keep_ratio is not None
                and expected_keep_ratio is not None
                and abs(keep_ratio - expected_keep_ratio) <= ratio_tolerance
                and abs(summary_keep_ratio - expected_keep_ratio) <= ratio_tolerance
            )
        )
        check("filter_keep_ratio_closure", keep_ratio_valid, {
            "filtering": filtering.get("valid_ratio_after_keep_cost_filter"),
            "summary": summary.get("valid_ratio_after_keep_cost_filter"),
            "expected": expected_keep_ratio,
            "tolerance": ratio_tolerance,
        })

        summary_filter_matches = all(
            summary.get(key) == filtering.get(key)
            for key in (
                "num_pixels_total",
                "num_valid_before_filter",
                "num_invalid_before_filter",
                "num_valid_after_keep_cost_filter",
                "num_valid_after_filter",
                "num_rejected_by_keep_cost_filter",
                "num_rejected_by_ignore_mask",
                "num_rejected_by_filter",
            )
            if key in summary or key in filtering
        )
        mask_decomposition_valid = (
            filter_valid_after + rejected_mask == valid_after_keep
            and rejected_keep + rejected_mask == filter_rejected
            and strict_int(masked_reason) == rejected_mask
            if mask_count_available
            else (
                filter_valid_after == valid_after_keep
                and rejected_keep == filter_rejected
                and masked_reason is None
                and rejected_mask_value is None
            )
        ) if all(value is not None for value in (
            filter_valid_after, valid_after_keep, rejected_keep, filter_rejected
        )) else False
        decomposition_valid = (
            filter_count_domains
            and mask_metadata_valid
            and reason_counts_valid
            and valid_after_keep + rejected_keep == filter_valid_before
            and filter_valid_before + filter_invalid_before == filter_total
            and filter_valid_after + filter_rejected == filter_valid_before
            and mask_decomposition_valid
            and reason_total == filter_rejected
            and summary_filter_matches
        )
        check("filter_rejection_decomposition", decomposition_valid, {
            "valid_before": filter_valid_before,
            "valid_after_keep_cost": valid_after_keep,
            "valid_after_all_filters": filter_valid_after,
            "rejected_keep_cost": rejected_keep,
            "rejected_ignore_mask": rejected_mask_value,
            "rejection_reason_total": reason_total,
            "masked_reason_count": masked_reason,
            "summary_matches": summary_filter_matches,
            "ignore_mask_status": mask_status,
        })

    warnings: list[dict[str, Any]] = []
    warning_sources = (
        (
            v3_logical_state,
            "depth_prior_weight_equal_selected_rescore_proxy",
            "reference_variance_equal_selected_rescore_proxy",
        ),
        (
            v4_exact,
            "depth_prior_weight_production_exact",
            "reference_variance_production_exact",
        ),
    )
    for validation, signal, reference_signal in warning_sources:
        if not isinstance(validation, dict):
            continue
        domain = validation.get("depth_prior_weight_domain")
        warning = domain.get("warning") if isinstance(domain, dict) else None
        if isinstance(warning, dict):
            warnings.append({
                **warning,
                "signal": signal,
                "reference_signal": reference_signal,
            })

    result = {
        "schema_version": schema_version,
        "capture_kind": "maps",
        "valid": all(item["passed"] for item in checks),
        "warnings": warnings,
        "frame_dir": str(frame_dir),
        "checks": checks,
        "manifest_map_count": len(entries),
        "maps_available": True,
        "maps_unavailable_reason": "",
        "exact_maps_available": exact_available,
        "exact_maps_unavailable_reason": (
            "" if exact_available else str((manifest.get("exact_capture") or {}).get("unavailable_reason") or "exact maps unavailable")
        ),
        "instrumented_dmap_checked": arguments.instrumented_dmap is not None,
        "component_total_max_abs_residual": component_residual,
        "geometric_max_abs": geometric_max_abs,
        "confidence_gap": gap_stats,
        "iterations": iteration_stats,
        "dmap_consistency_max_abs": dmap_consistency,
        "dmap_consistency_limits": dmap_consistency_limits,
        "dmap_terminal_state": dmap_terminal_state,
        "parity_max_abs": parity,
        "candidate_counts": summary.get("candidate_acceptance") or [],
        "valid_ratio_after_filter": summary.get("valid_ratio_after_filter"),
        "ignore_mask": ignore_mask_validation,
    }
    if v3_logical_state is not None:
        result["logical_state_validation"] = v3_logical_state
    if v3_logical_event is not None:
        result["logical_event_validation"] = v3_logical_event
    if v4_exact is not None:
        result["exact_validation"] = v4_exact
    if apd_validation is not None:
        result["apd_validation"] = apd_validation
    if apd_multiscale_validation is not None:
        result["apd_multiscale_validation"] = apd_multiscale_validation
    result = machine_readable_result(result)
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> int:
    arguments = tyro.cli(Arguments)
    result = machine_readable_result(validate(arguments))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
