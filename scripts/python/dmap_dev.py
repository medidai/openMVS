#!/usr/bin/env python3
"""Run, analyze, report, and validate OpenMVS depth-map development experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
import fnmatch
from functools import lru_cache
import hashlib
import html
from html.parser import HTMLParser
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Annotated, Any, Iterable, Literal, Union
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import tyro
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import report_dmap_annotation_fit as annotation_fit  # noqa: E402
import dmap_instrumentation_report as instrumentation_report  # noqa: E402
import dmap_drilldown  # noqa: E402
import dmap_report_model  # noqa: E402
from dmap_observability import (  # noqa: E402
    array_store_workflow,
    component_registry,
    config_materialization,
    integrity,
    region_metrics,
)
import validate_dmap_instrumentation as instrumentation_validator  # noqa: E402


SCHEMA_VERSION = 2
EXPERIMENT_LOCK_SCHEMA_NAME = "openmvs.dmap.experiment_lock"
EXPERIMENT_LOCK_SCHEMA_VERSION = 4
EXPERIMENT_PHASE_SCHEMA_NAME = "openmvs.dmap.experiment_phase"
EXPERIMENT_PHASE_SCHEMA_VERSION = 1
EXPERIMENT_PHASE_LOCK_SCHEMA_NAME = "openmvs.dmap.experiment_phase_lock"
EXPERIMENT_PHASE_LOCK_SCHEMA_VERSION = 1
DEFAULT_DENSIFY_BIN = REPO_ROOT / "build-dmap-production/bin/DensifyPointCloud"
DEFAULT_DENSIFY_OBSERVE_BIN = (
    REPO_ROOT / "build-dmap-observer/bin/DensifyPointCloudDMapObserve"
)
RUNTIME_BOUNDARY_SCHEMA_NAME = "openmvs.dmap.runtime_boundary"
RUNTIME_BOUNDARY_SCHEMA_VERSION = 1
RUNTIME_BOUNDARY_RECEIPT_SCHEMA_NAME = "openmvs.dmap.runtime_boundary_receipt"
RUNTIME_BOUNDARY_RECEIPT_SCHEMA_VERSION = 1
CAPTURE_INTENT_SCHEMA_NAME = "openmvs.dmap.capture_intent"
CAPTURE_INTENT_SCHEMA_VERSION = 1
PRIMARY_RANSAC_THRESHOLD_M = 0.02
RANSAC_THRESHOLDS_M = (0.005, 0.01, 0.02, 0.05)
MIN_SCENES_FOR_GATE = 5
REQUIRED_LOGICAL_STATE_SIGNALS = tuple(instrumentation_report.LOGICAL_COST_SIGNAL_PRESENTATION)
SCHEMA4_EXACT_STATE_SIGNALS = (
    "cost_photo_raw_production_exact",
    "cost_photo_prior_production_exact",
    "cost_geometric_production_exact",
    "cost_total_production_exact",
    "depth_prior_disagreement_production_exact",
    "depth_prior_weight_production_exact",
    "gap_winner_runner_up_exact",
    "reference_variance_production_exact",
)
SCHEMA4_EXACT_EVENT_SIGNALS = (
    "candidate_stored_cost_before_exact", "candidate_incumbent_cost_exact",
    "candidate_winner_cost_exact", "candidate_runner_up_cost_exact",
    "candidate_tested_mask_exact", "candidate_finite_mask_exact", "candidate_accepted_mask_exact",
    "candidate_counts_exact", "candidate_identity_exact", "selected_view_counts_exact",
    "selected_views_before_mask_exact", "selected_views_after_mask_exact",
)
SCHEMA4_EXACT_VIEW_SIGNALS = (
    "view_cost_components_exact", "view_selection_metrics_exact",
    "view_weighted_contribution_exact", "view_selection_state_exact", "view_agreement_state_exact",
)
MECHANISM_LOGICAL_STATE_SIGNALS = (
    "view_probability_mass",
    "view_probability_positive_count",
    "view_probability_health_status",
    "view_probability_unassigned_draw_count",
    "view_probability_legacy_last_view_collapse",
)
OPTIONAL_ADAPTIVE_PATCH_LOGICAL_STATE_SIGNALS = (
    "adaptive_patch_valid_sample_fraction",
    "adaptive_patch_fallback_status",
)
LOW_TEXTURE_UPDATE_BASE_EXACT_EVENT_SIGNALS = (
    "low_texture_update_eligible_exact",
    "low_texture_update_ambiguity_exact",
    "low_texture_update_required_gain_exact",
    "low_texture_update_best_proposed_gain_exact",
    "low_texture_update_rejected_mask_exact",
    "low_texture_update_would_have_won_source_exact",
    "low_texture_update_rejected_count_exact",
)
LOW_TEXTURE_UPDATE_RAW_ORDER_EXACT_EVENT_SIGNALS = (
    "candidate_raw_best_cost_exact",
    "candidate_raw_runner_up_cost_exact",
    "gap_raw_best_runner_up_exact",
    "candidate_retained_minus_raw_best_exact",
    "candidate_raw_suppression_identity_exact",
)
LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS = (
    *LOW_TEXTURE_UPDATE_BASE_EXACT_EVENT_SIGNALS,
    *LOW_TEXTURE_UPDATE_RAW_ORDER_EXACT_EVENT_SIGNALS,
)
OPTIONAL_MECHANISM_INITIALIZATION_SIGNALS = (
    "adaptive_patch_support_mode",
    "adaptive_patch_activation",
    "jbu_transfer_depth",
    "jbu_nearest_depth",
    "jbu_transfer_depth_delta",
    "jbu_fallback_status",
)
OPTIONAL_MECHANISM_FINAL_STATE_SIGNALS = (
    "hierarchy_entry_cost",
    "hierarchy_proposed_cost",
    "hierarchy_improvement_margin",
    "hierarchy_update_status",
)
VIEW_PROBABILITY_HEALTH_COUNTERS = (
    "processed",
    "finite_positive_events",
    "zero_mass_events",
    "nonfinite_component_events",
    "negative_component_events",
    "nonfinite_sum_events",
    "degenerate_events",
    "unassigned_draws",
    "legacy_last_view_collapse_events",
    "positive_view_count_sum",
)
REQUIRED_LOGICAL_EVENT_SIGNALS = tuple(sorted(instrumentation_report.LOGICAL_EVENT_SIGNAL_IDS))
REPORT_DERIVED_LOGICAL_EVENT_SIGNALS = ("cost_improvement_exact",)
SUMMARY_PROFILE_FINAL_SIGNALS = tuple(sorted(
    (
        set(component_registry.BUILTIN_DESCRIPTORS)
        | {
            Path(name).stem
            for name in (
                *instrumentation_report.PNG_MAP_NAMES,
                *instrumentation_report.PFM_MAP_NAMES,
            )
        }
        | {
            "confidence_final", "cost_final", "cost_final_preview",
            "depth_final_before_filter", "normal_final_before_filter",
            "cost_final_before_filter", "cost_photometric", "cost_photo_prior",
            "cost_geometric", "cost_total_components", "cost_depth_prior",
            "depth_prior_weight", "confidence_gap", "reference_variance",
            "view_entropy", "low_depth_prior", "valid_before_filter",
            "valid_after_filter", "candidate_source", "num_supporting_views",
            "last_changed_iter", "accepted_update_count",
            *{f"view_{kind}_{view}" for kind in (
                "weight", "cost", "photometric_cost", "geometric_cost",
            ) for view in range(4)},
        }
    )
    - {"reference_rgb", "depth"}
    - set(REQUIRED_LOGICAL_STATE_SIGNALS)
    - set(REQUIRED_LOGICAL_EVENT_SIGNALS)
    - set(REPORT_DERIVED_LOGICAL_EVENT_SIGNALS)
    - set(SCHEMA4_EXACT_STATE_SIGNALS)
    - set(SCHEMA4_EXACT_EVENT_SIGNALS)
    - set(SCHEMA4_EXACT_VIEW_SIGNALS)
    - set(MECHANISM_LOGICAL_STATE_SIGNALS)
    - set(OPTIONAL_ADAPTIVE_PATCH_LOGICAL_STATE_SIGNALS)
    - set(LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS)
    - set(OPTIONAL_MECHANISM_INITIALIZATION_SIGNALS)
    - set(OPTIONAL_MECHANISM_FINAL_STATE_SIGNALS)
))
MAP_CATALOG_COLUMNS = (
    "run", "label", "configured_run", "capture_profile", "run_role", "repeat", "scene_id", "estimation_stage", "geometric_iteration", "frame", "image_id", "image_name", "safe_image_name",
    "manifest_schema_name", "manifest_schema_version", "map_granularity", "signal", "role", "logical_iteration", "pyramid_level",
    "stage", "algorithm_stage", "pass_index", "dtype", "measurement_quality", "measurement_basis", "proxy_target", "limitations",
    "semantics", "stage_index", "source_view_index", "source_image_id", "source_image_name",
    "contribution_basis", "channels_json", "encoding", "unavailable_value", "gap_scope",
    "path", "relative_path", "component_paths_json", "bytes", "exists", "available", "manifest_path",
)
SIGNAL_AVAILABILITY_COLUMNS = (*MAP_CATALOG_COLUMNS, "required", "availability_reason")

MECHANISM_ORDER = (
    "comparison",
    "cost",
    "view",
    "update",
    "filtering",
    "runtime",
    "overview",
)
MECHANISM_LABELS = {
    "comparison": "Cross-run Effects",
    "cost": "Cost Function and Convergence",
    "view": "View Selection and Support",
    "update": "Update Dynamics",
    "filtering": "Filtering and Completeness",
    "runtime": "Runtime and Scalability",
    "overview": "Final State Overview",
}
MECHANISM_DESCRIPTIONS = {
    "comparison": "Shared-scale final maps, spatial deltas, transition maps, CDFs, and paired metrics show where behavior changed.",
    "cost": "Cost evolution, component maps, closure residuals, distributions, ambiguity, and improvement maps show how the objective behaves.",
    "view": "Support, entropy, per-view weights/contributions, transition matrices, and churn show whether source-view selection is stable and useful.",
    "update": "Changed-pixel rates and depth/normal iteration maps show when and where hypotheses continue moving.",
    "filtering": "Validity, rejection counts/reasons, before/after maps, and completeness deltas show what survives estimation and filtering.",
    "runtime": "Production endpoint wall time is the decision authority; observer kernel timing diagnoses where computational cost is spent.",
    "overview": "Reference imagery and final depth, normal, cost, support, update, and filtering state provide frame-level context.",
}


@dataclass(frozen=True)
class MetricSpec:
    level: Literal["frame", "annotation", "performance"]
    direction: Literal["higher", "lower"]
    tolerance: float
    gate: bool
    unit: str


METRICS: dict[str, MetricSpec] = {
    "valid_ratio_after_filter": MetricSpec("frame", "higher", 0.005, True, "fraction"),
    "endpoint_valid_depth_coverage": MetricSpec("frame", "higher", 0.005, True, "fraction"),
    "rejected_by_filter_ratio": MetricSpec("frame", "lower", 0.005, False, "fraction"),
    "final_cost_median": MetricSpec("frame", "lower", 0.005, False, "cost"),
    "final_cost_p90": MetricSpec("frame", "lower", 0.005, False, "cost"),
    "mean_support": MetricSpec("frame", "higher", 0.05, False, "views"),
    "effective_inlier_coverage": MetricSpec("annotation", "higher", 0.01, True, "fraction"),
    "spatial_coverage_fraction": MetricSpec("annotation", "higher", 0.005, False, "fraction"),
    "inlier_threshold_auc": MetricSpec("annotation", "higher", 0.02, True, "auc"),
    "all_residual_p95_m": MetricSpec("annotation", "lower", 0.002, True, "m"),
    "endpoint_elapsed_seconds": MetricSpec(
        "performance", "lower", 0.05, True, "relative"
    ),
}

ACCURACY_PRIMARY_METRICS: dict[str, tuple[Literal["higher", "lower"], float, str]] = {
    "all_residual_p95_m": ("lower", 0.002, "m"),
    "inlier_threshold_auc": ("higher", 0.02, "fraction"),
    "inlier_fraction_5mm": ("higher", 0.02, "fraction"),
}
ACCURACY_CLASS_ORDER = {
    "improved": 0,
    "equivalent": 1,
    "mixed": 2,
    "regressed": 3,
    "inconclusive": 4,
}

MODEL_SWITCH_THRESHOLDS = {
    "line_direction_delta_deg": 5.0,
    "line_extent_delta_m": 0.025,
    "line_extent_relative_delta": 0.25,
    "plane_normal_delta_deg": 5.0,
    "plane_position_delta_m": 0.010,
}

RESOURCE_PLAN_VALIDATION_COLUMNS = (
    "run", "role", "repeat", "scene_id", "frame", "estimation_stage",
    "geometric_iteration", "pyramid_level", "image_id", "plan_kind", "component", "required",
    "available", "valid", "decision", "reason", "validation_errors_json",
    "validation_warnings_json", "effective_device_bytes", "effective_host_bytes",
    "effective_storage_bytes", "current_pyramid_storage_bytes",
    "frame_storage_committed_before_bytes", "full_resolution_priority_reserve_bytes",
    "storage_requested_plus_priority_reserve_bytes",
    "storage_frame_priority_reservation_bytes",
    "storage_frame_priority_reservation_consumed", "source_json",
)

SCHEMA4_EMPTY_TABLE_COLUMNS = {
    "exact_observability": (
        "run", "role", "repeat", "scene_id", "frame", "estimation_stage",
        "geometric_iteration", "image_id", "schema_name", "schema_version",
        "pyramid_level", "num_logical_states", "num_views", "states_json", "source_json",
    ),
    "exact_iterations": (
        "run", "role", "repeat", "scene_id", "frame", "estimation_stage",
        "geometric_iteration", "image_id", "pyramid_level", "logical_iteration",
        "low_texture_gate_eligible", "low_texture_propagation_accepted",
        "low_texture_propagation_rejected", "low_texture_refinement_accepted",
        "low_texture_refinement_rejected", "low_texture_required_gain_sum",
        "low_texture_best_proposed_gain_sum",
        "source_csv",
    ),
    "exact_views": (
        "run", "role", "repeat", "scene_id", "frame", "estimation_stage",
        "geometric_iteration", "image_id", "pyramid_level", "logical_iteration",
        "source_view_index", "source_csv",
    ),
}

EXACT_COST_EVOLUTION_COLUMNS = (
    "run", "run_role", "repeat", "scene_id", "frame", "image_id",
    "estimation_stage", "geometric_iteration", "signal", "logical_iteration",
    "stage", "pyramid_level", "measurement_quality", "measurement_basis",
    "pixels", "available_pixels", "available_ratio", "mean", "median", "p90",
    "min", "max", "source_map",
)

TEXTURE_REGION_METRIC_COLUMNS = (
    "run", "repeat", "scene_id", "image_id", "estimation_stage",
    "geometric_iteration", "pyramid_level", "logical_iteration", "region_type",
    "region", "region_pixels", "region_fraction", "texture_signal",
    "texture_threshold_low_mid", "texture_threshold_mid_high", "texture_mean",
    "signal", "mechanism", "quantity", "valid_pixels", "mean", "median", "p90",
)


@dataclass
class PrepareCommand:
    """Resolve a suite, validate inputs, and estimate storage."""

    config: Path
    allow_over_budget: bool = False


@dataclass
class RunCommand:
    """Execute missing full-map and timing runs, then build the report."""

    config: Path
    dry_run: bool = False
    skip_report: bool = False
    allow_over_budget: bool = False
    profile: list[str] = field(default_factory=list)


@dataclass
class ReportCommand:
    """Analyze existing runs and generate Markdown/HTML reports."""

    config: Path
    output_dir: Path | None = None
    skip_diagnostics: bool = False
    allow_process_specialization_divergence_for_diagnostics: bool = False
    evidence_context: Path | None = None
    published_output_dir: Path | None = None


@dataclass
class ValidateCommand:
    """Validate report structure, links, schemas, and artifacts."""

    report: Path


@dataclass
class TraceRerunCommand:
    """Generate or execute the deterministic targeted trace rerun."""

    config: Path
    execute: bool = False
    allow_over_budget: bool = False


@dataclass
class DrilldownCommand:
    """Create or execute a paired deep frame/pixel investigation request."""

    config: Path
    scene: str
    frame: int
    pixel: list[str] = field(default_factory=list)
    roi: str | None = None
    variant: list[str] = field(default_factory=list)
    execute: bool = False
    refresh_report: bool = False
    report_dir: Path | None = None
    allow_over_budget: bool = False


@dataclass
class ArrayStoreCommand:
    """Convert or validate immutable frame maps in canonical Zarr v3 stores."""

    config: Path
    run: list[str] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    frame: list[str] = field(default_factory=list)
    chunk_size: int = array_store_workflow.DEFAULT_CHUNK_SIZE
    shard_size: int = array_store_workflow.DEFAULT_SHARD_SIZE
    zstd_level: int = array_store_workflow.DEFAULT_ZSTD_LEVEL
    allow_incomplete: bool = False
    max_uncompressed_bytes_per_store: int | None = None
    verify_data: bool = True


Command = Union[
    Annotated[PrepareCommand, tyro.conf.subcommand(name="prepare")],
    Annotated[RunCommand, tyro.conf.subcommand(name="run")],
    Annotated[ReportCommand, tyro.conf.subcommand(name="report")],
    Annotated[ValidateCommand, tyro.conf.subcommand(name="validate")],
    Annotated[TraceRerunCommand, tyro.conf.subcommand(name="trace-rerun")],
    Annotated[DrilldownCommand, tyro.conf.subcommand(name="drilldown")],
    Annotated[ArrayStoreCommand, tyro.conf.subcommand(name="array-store")],
]


@dataclass(frozen=True)
class RunScene:
    label: str
    role: str
    repeat: int
    scene_id: str
    instrumentation_dir: Path
    depth_map_dir: Path | None
    timing_dir: Path | None
    estimation_stage: str = "photometric"
    geometric_iteration: int | None = None
    cross_capture_parity_compatible: bool = True
    cross_capture_parity_reason: str = ""
    diagnostic_only: bool = False
    diagnostic_only_reason: str = ""
    allow_process_specialization_divergence_for_diagnostics: bool = False
    configured_label: str | None = None
    capture_profile: str = "summary"


def manifest_run_scene_rows(run_scenes: Iterable[RunScene]) -> list[dict[str, Any]]:
    return [
        {
            "label": row.label,
            "configured_label": row.configured_label or row.label,
            "role": row.role,
            "repeat": row.repeat,
            "scene_id": row.scene_id,
            "estimation_stage": row.estimation_stage,
            "geometric_iteration": row.geometric_iteration,
            "instrumentation_dir": str(row.instrumentation_dir),
            "depth_map_dir": str(row.depth_map_dir) if row.depth_map_dir else None,
            "diagnostic_only": row.diagnostic_only,
            "diagnostic_only_reason": row.diagnostic_only_reason,
            "capture_profile": row.capture_profile,
        }
        for row in run_scenes
    ]


def expand_instrumentation_stages(run_scene: RunScene) -> list[RunScene]:
    stages = [run_scene]
    nested_root = run_scene.instrumentation_dir / "geometric_iterations"
    indexed_directories: list[tuple[int, Path]] = []
    for iteration_dir in nested_root.glob("iteration*"):
        if not iteration_dir.is_dir():
            continue
        match = re.fullmatch(r"iteration(\d+)", iteration_dir.name)
        if match:
            indexed_directories.append((int(match.group(1)), iteration_dir))
    for iteration, iteration_dir in sorted(indexed_directories):
        nested_timing = (
            run_scene.timing_dir / "geometric_iterations" / iteration_dir.name
            if run_scene.timing_dir is not None else None
        )
        stages.append(replace(
            run_scene,
            instrumentation_dir=iteration_dir,
            timing_dir=nested_timing,
            estimation_stage="geometric_consistency",
            geometric_iteration=iteration,
        ))
    return stages


def instrumentation_stage_roots(root: Path) -> list[tuple[str, int | None, Path]]:
    """List the photometric root followed by complete geometric stages."""

    stages: list[tuple[str, int | None, Path]] = [("photometric", None, root)]
    nested_root = root / "geometric_iterations"
    indexed_directories: list[tuple[int, Path]] = []
    for iteration_dir in nested_root.glob("iteration*"):
        if not iteration_dir.is_dir():
            continue
        match = re.fullmatch(r"iteration(\d+)", iteration_dir.name)
        if match:
            indexed_directories.append((int(match.group(1)), iteration_dir))
    for iteration, iteration_dir in sorted(indexed_directories):
        stages.append(("geometric_consistency", iteration, iteration_dir))
    return stages


def instrumentation_depth_map_dir(root: Path | None) -> Path | None:
    """Resolve a mode's depth-map directory from a nested stage root."""

    if root is None:
        return None
    for candidate in (root, *root.parents):
        if candidate.name == "dmap_instrumentation":
            return candidate.parent / "depth_maps"
    return None


def sibling_capture_depth_map_dir(root: Path, mode: str) -> Path | None:
    """Resolve a sibling endpoint/maps/timing depth-map directory."""

    for candidate in (root, *root.parents):
        if candidate.name == "dmap_instrumentation":
            return candidate.parent.parent / mode / "depth_maps"
    return None


def stable_json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dmap_set_identity(directory: Path) -> dict[str, Any]:
    rows = [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": dmap_drilldown.file_digest(path),
        }
        for path in sorted(directory.glob("depth*.dmap"))
    ]
    return {"count": len(rows), "sha256": stable_json_digest(rows), "files": rows}


def validated_dmap_set_identity(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("files"), list):
        raise ValueError(f"{description} is missing a DMAP file inventory")
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in value["files"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{description} contains a malformed DMAP record")
        name = raw.get("name")
        digest = raw.get("sha256")
        size = raw.get("bytes")
        if not isinstance(name, str) or re.fullmatch(r"depth\d+\.dmap", name) is None:
            raise ValueError(f"{description} contains an invalid DMAP name: {name!r}")
        if name in names:
            raise ValueError(f"{description} contains duplicate DMAP name {name}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{description} contains an invalid digest for {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{description} contains an invalid size for {name}")
        names.add(name)
        rows.append({"name": name, "bytes": size, "sha256": digest})
    rows.sort(key=lambda row: row["name"])
    expected = {"count": len(rows), "sha256": stable_json_digest(rows), "files": rows}
    if value.get("count") != expected["count"] or value.get("sha256") != expected["sha256"]:
        raise ValueError(f"{description} aggregate identity is inconsistent")
    return expected


def complete_dmap_set_identity(directory: Path) -> tuple[dict[str, Any], str]:
    """Return a physical or verified pre-compaction DMAP identity."""

    physical = dmap_set_identity(directory)
    capture_dir = directory.parent
    manifest_path = capture_dir / "retention_manifest.json"
    completion_path = capture_dir / "compacted_completion.json"
    if not manifest_path.exists() and not completion_path.exists():
        return physical, "physical_files"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise ValueError(f"incomplete retention evidence beside {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable retention evidence beside {directory}: {exc}") from exc
    if (
        manifest.get("schema_name") != "openmvs.dmap.retention_manifest"
        or not manifest.get("complete")
        or completion.get("schema_name") != "openmvs.dmap.compacted_completion"
        or not completion.get("complete")
    ):
        raise ValueError(f"invalid retention evidence beside {directory}")
    if completion.get("retention_manifest") != manifest_path.name:
        raise ValueError(f"retention completion references another manifest beside {directory}")
    if completion.get("retention_manifest_sha256") != dmap_drilldown.file_digest(manifest_path):
        raise ValueError(f"retention manifest digest mismatch beside {directory}")

    completion_schema = completion.get("schema_version")
    manifest_schema = manifest.get("schema_version")
    if completion_schema == 1:
        # Legacy compaction was lossless. Its physical identity remains the
        # parity authority; it cannot prove any deleted file.
        return physical, "physical_files"
    if completion_schema != 2 or manifest_schema != 2:
        raise ValueError(
            f"unsupported retention schema beside {directory}: "
            f"manifest={manifest_schema!r}, completion={completion_schema!r}"
        )

    retained = validated_dmap_set_identity(
        manifest.get("retained_dmap_set"), f"retained DMAP identity beside {directory}"
    )
    if retained != physical:
        raise ValueError(f"retained DMAP files do not match their manifest beside {directory}")
    complete = validated_dmap_set_identity(
        manifest.get("complete_pre_compaction_dmap_set"),
        f"pre-compaction DMAP identity beside {directory}",
    )
    if (
        completion.get("complete_pre_compaction_dmap_count") != complete["count"]
        or completion.get("complete_pre_compaction_dmap_set_sha256") != complete["sha256"]
        or completion.get("retained_dmap_count") != retained["count"]
        or completion.get("retained_dmap_set_sha256") != retained["sha256"]
    ):
        raise ValueError(f"retention completion identity mismatch beside {directory}")
    if bool(manifest.get("full_dmap_set_retained")):
        if complete != retained:
            raise ValueError(f"lossless retention identity mismatch beside {directory}")
        return physical, "physical_files"
    if not bool(manifest.get("lossy")):
        raise ValueError(f"incomplete DMAP set is not declared lossy beside {directory}")
    reconstructed_rows = list(retained["files"])
    reconstructed_names = {row["name"] for row in reconstructed_rows}
    removed = manifest.get("removed")
    if not isinstance(removed, list):
        raise ValueError(f"lossy retention evidence has no removal ledger beside {directory}")
    for raw in removed:
        if not isinstance(raw, dict) or not bool(raw.get("lossy")):
            continue
        if raw.get("reason") != "not_instrumented_image_id":
            raise ValueError(f"lossy retention evidence has an unknown reason beside {directory}")
        raw_path = raw.get("path")
        name = Path(str(raw_path)).name
        if raw_path != f"depth_maps/{name}":
            raise ValueError(f"lossy retention evidence has an unsafe path beside {directory}")
        record = validated_dmap_set_identity({
            "count": 1,
            "sha256": stable_json_digest([{
                "name": name, "bytes": raw.get("bytes"), "sha256": raw.get("sha256"),
            }]),
            "files": [{
                "name": name, "bytes": raw.get("bytes"), "sha256": raw.get("sha256"),
            }],
        }, f"removed DMAP identity beside {directory}")["files"][0]
        if name in reconstructed_names:
            raise ValueError(f"lossy retention evidence duplicates {name} beside {directory}")
        reconstructed_names.add(name)
        reconstructed_rows.append(record)
    reconstructed_rows.sort(key=lambda row: row["name"])
    reconstructed = {
        "count": len(reconstructed_rows),
        "sha256": stable_json_digest(reconstructed_rows),
        "files": reconstructed_rows,
    }
    if reconstructed != complete:
        raise ValueError(f"lossy retention removal ledger cannot reconstruct the full set beside {directory}")
    return complete, "verified_pre_compaction_manifest"


def compare_dmap_directories(first: Path, second: Path) -> dict[str, Any]:
    """Compare complete DMAP output sets using file or retained pre-compaction digests."""

    first_identity, first_basis = complete_dmap_set_identity(first)
    second_identity, second_basis = complete_dmap_set_identity(second)
    first_rows = {row["name"]: row for row in first_identity["files"]}
    second_rows = {row["name"]: row for row in second_identity["files"]}
    shared_names = sorted(first_rows.keys() & second_rows.keys())
    mismatched = [
        name for name in shared_names
        if first_rows[name]["sha256"] != second_rows[name]["sha256"]
    ]
    missing_from_first = sorted(second_rows.keys() - first_rows.keys())
    missing_from_second = sorted(first_rows.keys() - second_rows.keys())
    return {
        "first_count": len(first_rows),
        "second_count": len(second_rows),
        "first_basis": first_basis,
        "second_basis": second_basis,
        "shared_count": len(shared_names),
        "mismatched": mismatched,
        "missing_from_first": missing_from_first,
        "missing_from_second": missing_from_second,
        "bit_exact": bool(shared_names) and not (
            mismatched or missing_from_first or missing_from_second
        ),
    }


def select_terminal_run_scenes(run_scenes: list[RunScene]) -> list[RunScene]:
    """Choose the terminal estimation stage for endpoint quality evaluation."""

    grouped: dict[tuple[str, str, int, str], list[RunScene]] = {}
    for run_scene in run_scenes:
        key = (run_scene.label, run_scene.role, run_scene.repeat, run_scene.scene_id)
        grouped.setdefault(key, []).append(run_scene)
    selected: list[RunScene] = []
    for key in sorted(grouped):
        stages = grouped[key]
        geometric = [stage for stage in stages if stage.estimation_stage == "geometric_consistency"]
        if geometric:
            selected.append(max(geometric, key=lambda stage: int(stage.geometric_iteration or 0)))
        else:
            photometric = [stage for stage in stages if stage.estimation_stage == "photometric"]
            selected.append(photometric[0] if photometric else stages[-1])
    return selected


def select_terminal_frames(all_frames: pd.DataFrame) -> pd.DataFrame:
    """Return one terminal photometric/geometric row per run, scene, and frame."""

    if all_frames.empty or "estimation_stage" not in all_frames.columns:
        return all_frames
    ranked = all_frames.copy()
    ranked["_terminal_stage_rank"] = (
        ranked["estimation_stage"].astype(str) == "geometric_consistency"
    ).astype(int)
    ranked["_terminal_geometric_iteration"] = pd.to_numeric(
        ranked.get("geometric_iteration"), errors="coerce"
    ).fillna(-1)
    keys = ["run", "repeat", "scene_id", "image_id"]
    selected = ranked.sort_values(
        [*keys, "_terminal_stage_rank", "_terminal_geometric_iteration"],
        kind="stable",
    ).drop_duplicates(keys, keep="last")
    return selected.drop(columns=["_terminal_stage_rank", "_terminal_geometric_iteration"])


def configured_run_labels(config: dict[str, Any]) -> set[str]:
    """Return labels eligible for quality, annotation, and accuracy metrics."""

    labels = [
        validated_output_component(run.get("label"), "run label")
        for run in config.get("runs") or []
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("configured run labels must be unique")
    return set(labels)


def configured_metric_rows(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Exclude diagnostic capture cohorts from end-quality evaluation."""

    if data.empty or "run" not in data.columns:
        return data
    return data[data["run"].astype(str).isin(configured_run_labels(config))].copy()


def production_quality_metric_rows(
    data: pd.DataFrame,
    config: dict[str, Any],
    instrumentation_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only configured cohorts proven bit-exact to production endpoints."""

    selected = configured_metric_rows(data, config)
    if selected.empty:
        return selected
    if instrumentation_validation.empty:
        return selected.iloc[0:0].copy()
    validation = instrumentation_validation.copy()
    if "terminal_stage" in validation:
        validation = validation[validation["terminal_stage"].fillna(False).astype(bool)]
    if "diagnostic_only" in validation:
        validation = validation[~validation["diagnostic_only"].fillna(False).astype(bool)]
    if "quality_comparison_eligible" not in validation:
        return selected.iloc[0:0].copy()
    validation = validation[
        validation["quality_comparison_eligible"].fillna(False).astype(bool)
    ]
    shared = [
        column for column in ("run", "repeat", "scene_id", "image_id")
        if column in selected.columns and column in validation.columns
    ]
    if not {"run", "repeat", "scene_id"}.issubset(shared):
        return selected.iloc[0:0].copy()
    eligible = validation[shared].drop_duplicates()
    return selected.merge(eligible, on=shared, how="inner", validate="many_to_one")


PROCESS_SPECIALIZATION_PARITY_FAILURES = frozenset({
    "production_output_parity",
    "production_endpoint_dmap_set_bit_exact",
    "production_endpoint_parity",
})

REPORT_POLICY_SCHEMA_NAME = "openmvs.dmap.report_policy"
REPORT_POLICY_SCHEMA_VERSION = 3
CAPTURE_EVIDENCE_POLICY_SCHEMA_NAME = "openmvs.dmap.capture_evidence_policy"
CAPTURE_EVIDENCE_POLICY_SCHEMA_VERSION = 1


def allow_process_specialization_divergence_for_diagnostics(
    config: dict[str, Any],
) -> bool:
    """Return the explicit recovery-only Process specialization policy."""

    instrumentation = config.get("instrumentation") or {}
    value = instrumentation.get(
        "allow_process_specialization_divergence_for_diagnostics", False
    )
    if not isinstance(value, bool):
        raise ValueError(
            "instrumentation.allow_process_specialization_divergence_for_diagnostics "
            "must be a boolean"
        )
    return value


def process_specialization_validation_disposition(
    *,
    validator_valid: bool,
    failed_checks: list[str],
    diagnostic_only: bool,
    allow_divergence: bool,
) -> dict[str, Any]:
    """Classify validation without converting a parity failure into a pass."""

    failures = set(failed_checks)
    frame_valid = bool(validator_valid) and not failures
    diagnostic_divergence = (
        not frame_valid
        and bool(failures)
        and diagnostic_only
        and allow_divergence
        and failures <= PROCESS_SPECIALIZATION_PARITY_FAILURES
    )
    return {
        "valid": frame_valid,
        "report_generation_allowed": frame_valid or diagnostic_divergence,
        "production_qualification_status": (
            "passed"
            if frame_valid
            else "failed_allowed_diagnostic_only"
            if diagnostic_divergence
            else "failed"
        ),
        "process_specialization_divergence": diagnostic_divergence,
    }


def build_report_policy(
    config: dict[str, Any],
    root: Path,
    *,
    skip_diagnostics: bool,
    capture_evidence: dict[str, Any],
    activation_source: str = "effective_config",
    evidence_context_path: Path | None = None,
    published_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Bind report semantics to immutable experiment inputs and explicit policy."""

    phase = config.get("experiment_phase")
    phase_id = phase.get("phase_id") if isinstance(phase, dict) else None
    resolved_experiment = root / "00_resolved_experiment.yaml"
    if isinstance(phase_id, str):
        resolved_experiment = (
            experiment_phase_evidence_dir(root, phase_id)
            / "01_resolved_phase.yaml"
        )
    policy = {
        "schema_name": REPORT_POLICY_SCHEMA_NAME,
        "schema_version": REPORT_POLICY_SCHEMA_VERSION,
        "source_config": file_identity(Path(str(config["_config_path"]))),
        "resolved_experiment": file_identity(resolved_experiment),
        "experiment_lock": file_identity(root / "00_experiment_lock.json"),
        "environment_manifest": file_identity(
            root / ENVIRONMENT_MANIFEST_FILE
        ),
        "capture_evidence": capture_evidence,
        "skip_diagnostics": bool(skip_diagnostics),
        "allow_process_specialization_divergence_for_diagnostics": (
            allow_process_specialization_divergence_for_diagnostics(config)
        ),
        "activation_source": str(activation_source),
        "integrity_contract": {
            "capture_artifact_closure_schema_version": (
                integrity.CAPTURE_CLOSURE_SCHEMA_VERSION
            ),
            "report_tree_closure_schema_version": (
                integrity.REPORT_CLOSURE_SCHEMA_VERSION
            ),
        },
    }
    if isinstance(phase_id, str):
        phase_root = experiment_phase_evidence_dir(root, phase_id)
        policy["experiment_phase"] = {
            "phase_id": phase_id,
            "phase_lock": file_identity(phase_root / "00_phase_lock.json"),
            "parent_experiment_lock": file_identity(root / "00_experiment_lock.json"),
        }
    if evidence_context_path is not None:
        policy["evidence_context"] = file_identity(evidence_context_path)
    if published_output_dir is not None:
        policy["published_output_dir"] = str(
            published_output_dir.expanduser().absolute()
        )
    return policy


def build_capture_evidence_policy(
    root: Path,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Bind report reuse to admitted capture controls and artifact closures."""

    if coverage.get("schema_name") != "openmvs.dmap.capture_profile_coverage":
        raise ValueError("capture evidence policy requires capture-profile coverage")

    artifacts: dict[str, dict[str, Any]] = {}

    def bind(path: Path, kind: str) -> None:
        candidate = _require_non_symlink_path(
            path,
            description=f"capture evidence {kind}",
            final_kind="regular_file",
        )
        identity = integrity.regular_file_identity(candidate)
        experiment_root = _lexical_absolute_path(root)
        try:
            locator = candidate.relative_to(experiment_root).as_posix()
            scope = "experiment"
        except ValueError:
            locator = "sha256:" + hashlib.sha256(
                str(candidate).encode("utf-8")
            ).hexdigest()
            scope = "external"
        key = f"{scope}:{locator}"
        artifacts[key] = {
            "kind": kind,
            "scope": scope,
            "locator": locator,
            "bytes": identity["bytes"],
            "mode": identity["mode"],
            "sha256": identity["sha256"],
        }

    for intent in load_capture_intents(root):
        digest = str(intent["intent_sha256"])
        bind(root / "capture_intents" / f"{digest}.json", "capture_intent")

    complete_without_closure: list[str] = []
    bound_control_labels = {
        "artifact closure",
        "immutable request",
        "invalid request",
        "captured request",
        "executions",
    }
    for unit in coverage.get("units") or []:
        if not isinstance(unit, dict):
            continue
        links = [
            link
            for link in unit.get("evidence_links") or []
            if isinstance(link, dict)
        ]
        closure_links = [
            link for link in links if link.get("label") == "artifact closure"
        ]
        if unit.get("status") == "complete" and not closure_links:
            complete_without_closure.append(
                "/".join(str(unit.get(key, "")) for key in (
                    "configured_run", "repeat", "scene_id", "capture_profile"
                ))
            )
        for link in links:
            label = str(link.get("label", ""))
            path = link.get("path")
            if label in bound_control_labels and isinstance(path, str) and path:
                bind(Path(path), label.replace(" ", "_"))

    rows = [artifacts[path] for path in sorted(artifacts)]
    return {
        "schema_name": CAPTURE_EVIDENCE_POLICY_SCHEMA_NAME,
        "schema_version": CAPTURE_EVIDENCE_POLICY_SCHEMA_VERSION,
        "coverage_sha256": stable_json_digest(coverage),
        "artifact_count": len(rows),
        "artifacts_sha256": stable_json_digest(rows),
        "artifacts": rows,
        "reuse_eligible": not complete_without_closure,
        "complete_units_without_verified_closure": complete_without_closure,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def load_report_evidence_context(path: Path | None) -> dict[str, Any] | None:
    """Load and validate one immutable external-authority presentation sidecar."""

    if path is None:
        return None
    source = path.expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"report evidence context must be a regular non-symlink file: {source}"
        )
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"report evidence context is unreadable: {source}: {exc}") from exc
    validation = dmap_report_model.validate_evidence_context(value)
    if not validation["valid"]:
        raise ValueError(
            "invalid report evidence context: " + "; ".join(validation["errors"])
        )
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


SAFE_OUTPUT_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
ORCHESTRATION_OWNED_DENSIFY_ARGUMENTS = frozenset({
    "--working-folder",
    "--input-file",
    "--output-file",
    "--config-file",
    "--dmap-instrumentation-config",
    "--dmap-instrumentation-dir",
    "--dmap-instrumentation-level",
    "--dmap-instrumentation-sample-rate",
    "--dmap-instrumentation-write-maps",
})


def validated_output_component(value: Any, description: str) -> str:
    """Return one portable path component or fail before output construction."""

    if not isinstance(value, str) or SAFE_OUTPUT_COMPONENT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{description} must be one nonempty portable path component "
            "matching [A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    if value in {".", ".."}:
        raise ValueError(f"{description} cannot be {value!r}")
    return value


def contained_output_path(root: Path, *parts: str, description: str) -> Path:
    """Resolve a derived output path and reject symlink or component escape."""

    resolved_root = root.expanduser().resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"{description} escapes experiment root {resolved_root}: {candidate}"
        ) from exc
    return candidate


def _lexical_absolute_path(path: Path) -> Path:
    """Normalize dot components without resolving symbolic links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _require_non_symlink_path(
    path: Path, *, description: str, final_kind: Literal["directory", "regular_file"]
) -> Path:
    """Validate every existing component of one absolute lexical path."""

    candidate = _lexical_absolute_path(path)
    current = Path(candidate.anchor)
    components = candidate.parts[1:] if candidate.anchor else candidate.parts
    for index, part in enumerate(components):
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"{description} is missing or unreadable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{description} path contains a symlink: {current}")
        final = index == len(components) - 1
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{description} parent is not a directory: {current}")
        if final:
            expected = (
                stat.S_ISDIR(metadata.st_mode)
                if final_kind == "directory"
                else stat.S_ISREG(metadata.st_mode)
            )
            if not expected:
                raise ValueError(f"{description} is not a {final_kind.replace('_', ' ')}: {current}")
    return candidate


def _reject_symlink_components(path: Path, *, description: str) -> Path:
    """Reject symlinks in every component that exists, including a dangling leaf."""

    candidate = _lexical_absolute_path(path)
    current = Path(candidate.anchor)
    components = candidate.parts[1:] if candidate.anchor else candidate.parts
    missing_parent = False
    for part in components:
        current = current / part
        if missing_parent:
            continue
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing_parent = True
            continue
        except OSError as exc:
            raise ValueError(f"{description} is unreadable: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{description} path contains a symlink: {current}")
        if current != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{description} parent is not a directory: {current}")
    return candidate


def owned_regular_artifact_path(
    owning_root: Path, path: Path, *, description: str
) -> Path:
    """Return a lexical child file after rejecting symlinks in either path."""

    root = _lexical_absolute_path(owning_root)
    candidate = _lexical_absolute_path(path)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} is outside its owning root: {candidate}") from exc
    _require_non_symlink_path(root, description=f"{description} owning root", final_kind="directory")
    return _require_non_symlink_path(
        candidate, description=description, final_kind="regular_file"
    )


def validated_argument_overrides(raw: Any, owner: str) -> dict[str, str]:
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
        if option in ORCHESTRATION_OWNED_DENSIFY_ARGUMENTS:
            raise ValueError(
                f"{option} is orchestration-owned and cannot be overridden by {owner}"
            )
        if any(character in value for character in "\r\n"):
            raise ValueError(f"invalid argument override value for {owner}: {option!r}")
        result[option] = value
    return result


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Validate identities and command ownership before any output is created."""

    validated_output_component(
        config.get("experiment_id", "dmap_development"), "experiment_id"
    )
    instrumentation_sample_rate(config)
    runs = config.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("experiment config must define at least one run")
    labels: set[str] = set()
    baseline_count = 0
    validate_supported_densify_args(config.get("default_densify_args") or [])
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        label = validated_output_component(run.get("label"), f"runs[{index}].label")
        if label in labels:
            raise ValueError(f"duplicate run label {label!r}")
        labels.add(label)
        role = run.get("role", "variant")
        if role not in {"baseline", "variant"}:
            raise ValueError(f"run {label!r} has invalid role {role!r}")
        baseline_count += role == "baseline"
        repeats = run.get("repeats", 3 if role == "baseline" else 1)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
            raise ValueError(f"run {label!r} repeats must be a positive integer")
        validate_supported_densify_args(run.get("densify_args") or [])
    if baseline_count != 1:
        raise ValueError(
            f"experiment config must define exactly one baseline run; found {baseline_count}"
        )

    scene_ids: set[str] = set()
    scenes = config.get("scenes") or []
    if not isinstance(scenes, list):
        raise ValueError("scenes must be a list")
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise ValueError(f"scenes[{index}] must be a mapping")
        scan_id = validated_output_component(
            scene.get("scan_id"), f"scenes[{index}].scan_id"
        )
        if scan_id in scene_ids:
            raise ValueError(f"duplicate scene scan_id {scan_id!r}")
        scene_ids.add(scan_id)
        if "name" in scene:
            validated_output_component(scene["name"], f"scene {scan_id!r} name")
        validated_argument_overrides(
            scene.get("argument_overrides"), f"scene {scan_id!r}"
        )

    suite = config.get("suite") or {}
    if not isinstance(suite, dict):
        raise ValueError("suite must be a mapping")
    suite_ids = suite.get("scan_ids") or []
    if not isinstance(suite_ids, list):
        raise ValueError("suite.scan_ids must be a list")
    validated_suite_ids = [
        validated_output_component(value, "suite.scan_ids entry")
        for value in suite_ids
    ]
    if len(validated_suite_ids) != len(set(validated_suite_ids)):
        raise ValueError("suite.scan_ids contains duplicates")
    smoke_scan_id = suite.get("smoke_scan_id")
    if smoke_scan_id not in (None, ""):
        validated_output_component(smoke_scan_id, "suite.smoke_scan_id")

    sweep = config.get("sweep") or {}
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping")
    for index, stage in enumerate(sweep.get("stages") or []):
        if not isinstance(stage, dict):
            raise ValueError(f"sweep.stages[{index}] must be a mapping")
        validated_argument_overrides(
            stage.get("argument_overrides"), f"sweep stage {stage.get('name', index)!r}"
        )


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"experiment config must contain a mapping: {path}")
    version = int(value.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION}, got {version}")
    value["_config_path"] = str(path)
    validate_experiment_config(value)
    return value


def config_path(config: dict[str, Any], key: str, default: Path | None = None) -> Path:
    raw = config.get(key, default)
    if raw in (None, ""):
        raise ValueError(f"experiment config must define {key!r}")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_path"]).parent / path
    return path.resolve()


def ensure_external_output_path(path: Path, label: str) -> Path:
    """Resolve an output path and reject any lexical or resolved repo child."""

    source_root = REPO_ROOT.resolve()
    expanded = path.expanduser()
    lexical = expanded if expanded.is_absolute() else Path.cwd() / expanded
    lexical = lexical.absolute()
    resolved = lexical.resolve()
    for candidate in (lexical, resolved):
        try:
            candidate.relative_to(source_root)
        except ValueError:
            continue
        raise ValueError(
            f"{label} must be outside the OpenMVS source tree: {candidate}"
        )
    return resolved


def densify_binary(config: dict[str, Any], *, instrumented: bool) -> Path:
    """Resolve the production endpoint or the separate observability executable."""
    production = config_path(config, "densify_bin", DEFAULT_DENSIFY_BIN)
    if not instrumented:
        return production
    if config.get("densify_observe_bin"):
        return config_path(
            config,
            "densify_observe_bin",
            production.with_name("DensifyPointCloudDMapObserve"),
        )
    if not config.get("densify_bin"):
        return DEFAULT_DENSIFY_OBSERVE_BIN.resolve()
    if production.name == "DensifyPointCloudDMapObserve":
        return production
    return production.with_name("DensifyPointCloudDMapObserve")


def experiment_root(config: dict[str, Any]) -> Path:
    output_root = ensure_external_output_path(
        config_path(config, "output_root"), "output_root"
    )
    experiment_id = validated_output_component(
        config.get("experiment_id", "dmap_development"), "experiment_id"
    )
    return ensure_external_output_path(
        output_root / experiment_id, "experiment output"
    )


def annotation_counts(row: dict[str, Any]) -> tuple[int, int]:
    annotations = row.get("annotations") or {}
    planes = sum(len(item.get("chunks") or []) for item in annotations.get("controlPlanes") or [])
    edges = sum(len(item.get("chunks") or []) for item in annotations.get("controlEdges") or [])
    return planes, edges


def resolve_suite(config: dict[str, Any]) -> list[str]:
    suite = config.get("suite") or {}
    name = str(suite.get("name", "smoke"))
    explicit = [
        validated_output_component(value, "suite.scan_ids entry")
        for value in suite.get("scan_ids") or []
    ]
    if explicit:
        if len(explicit) != len(set(explicit)):
            raise ValueError("suite.scan_ids contains duplicates")
        return explicit
    configured = [
        validated_output_component(row.get("scan_id"), "scene scan_id")
        for row in config.get("scenes") or []
        if row.get("scan_id") not in (None, "")
    ]
    if name == "smoke":
        smoke_scan = (
            validated_output_component(suite.get("smoke_scan_id"), "suite.smoke_scan_id")
            if suite.get("smoke_scan_id") not in (None, "") else ""
        )
        if smoke_scan:
            return [smoke_scan]
        if configured:
            return [configured[0]]
        raise ValueError(
            "smoke suite requires suite.scan_ids, suite.smoke_scan_id, or a configured scene"
        )

    dataset_root = config_path(config, "dataset_root")
    review_rows = read_jsonl(
        annotation_fit.annotation_db_path(dataset_root) / "scan_reviewers.jsonl"
    )
    counts = []
    for row in review_rows:
        scan_id = str(row.get("scan_id", ""))
        if scan_id:
            validated_output_component(scan_id, "annotation scan_id")
        planes, edges = annotation_counts(row)
        if scan_id and (planes or edges):
            counts.append({"scan_id": scan_id, "planes": planes, "edges": edges})
    if name == "full":
        return sorted(item["scan_id"] for item in counts)
    if name != "development":
        raise ValueError(f"unsupported suite name: {name}")

    def ranked(predicate, score_key: str) -> list[str]:
        selected = [item for item in counts if predicate(item)]
        selected.sort(key=lambda item: (-int(item[score_key]), item["scan_id"]))
        return [item["scan_id"] for item in selected[:3]]

    mixed = [item for item in counts if item["planes"] > 0 and item["edges"] > 0]
    mixed.sort(key=lambda item: (-(item["planes"] + item["edges"]), item["scan_id"]))
    smoke_scan = (
        validated_output_component(suite.get("smoke_scan_id"), "suite.smoke_scan_id")
        if suite.get("smoke_scan_id") not in (None, "") else ""
    )
    scan_ids = [
        *([smoke_scan] if smoke_scan else configured[:1]),
        *[item["scan_id"] for item in mixed[:3]],
    ]
    scan_ids += ranked(lambda item: item["planes"] > 0 and item["edges"] == 0, "planes")
    scan_ids += ranked(lambda item: item["edges"] > 0 and item["planes"] == 0, "edges")
    return list(dict.fromkeys(scan_ids))[:10]


def find_scene_input(cache_root: Path, scan_id: str) -> tuple[Path | None, Path | None]:
    scan_root = cache_root / scan_id
    candidates = sorted(
        path for path in scan_root.glob("**/*.mvs")
        if "runs" not in path.parts and "dense" not in path.stem
    )
    return (candidates[0].parent, candidates[0]) if candidates else (None, None)


def resolve_scenes(config: dict[str, Any], suite_ids: list[str]) -> list[dict[str, Any]]:
    configured: dict[str, dict[str, Any]] = {}
    for index, configured_row in enumerate(config.get("scenes") or []):
        if not isinstance(configured_row, dict):
            raise ValueError(f"scenes[{index}] must be a mapping")
        configured_id = validated_output_component(
            configured_row.get("scan_id"), f"scenes[{index}].scan_id"
        )
        if configured_id in configured:
            raise ValueError(f"duplicate scene scan_id {configured_id!r}")
        configured[configured_id] = dict(configured_row)
    cache_root = config_path(config, "cache_root") if config.get("cache_root") else None
    scenes = []
    seen_suite_ids: set[str] = set()
    for raw_scan_id in suite_ids:
        scan_id = validated_output_component(raw_scan_id, "resolved suite scan_id")
        if scan_id in seen_suite_ids:
            raise ValueError(f"resolved suite contains duplicate scene {scan_id!r}")
        seen_suite_ids.add(scan_id)
        row = configured.get(scan_id, {"scan_id": scan_id, "name": scan_id[:8]})
        if not row.get("working_folder") or not row.get("mvs_file"):
            if cache_root is None:
                raise ValueError(
                    f"scene {scan_id!r} must define working_folder and mvs_file "
                    "when cache_root is not configured"
                )
            working, mvs_file = find_scene_input(cache_root, scan_id)
            if working and mvs_file:
                row.setdefault("working_folder", str(working))
                row.setdefault("mvs_file", str(mvs_file))
        if not row.get("working_folder") or not row.get("mvs_file"):
            raise FileNotFoundError(
                f"scene {scan_id!r} has no resolvable working_folder and mvs_file"
            )
        source_work = Path(str(row["working_folder"])).expanduser()
        source_mvs = Path(str(row["mvs_file"])).expanduser()
        config_parent = Path(
            str(config.get("_config_path", Path.cwd() / "experiment.yaml"))
        ).expanduser().resolve().parent
        if not source_work.is_absolute():
            source_work = config_parent / source_work
        if not source_mvs.is_absolute():
            source_mvs = config_parent / source_mvs
        source_work = source_work.resolve()
        source_mvs = source_mvs.resolve()
        if not source_work.is_dir():
            raise FileNotFoundError(
                f"scene {scan_id!r} working_folder is not a directory: {source_work}"
            )
        if not source_mvs.is_file():
            raise FileNotFoundError(
                f"scene {scan_id!r} mvs_file is not a regular file: {source_mvs}"
            )
        try:
            source_mvs.relative_to(source_work)
        except ValueError as exc:
            raise ValueError(
                f"scene {scan_id!r} mvs_file must be below working_folder"
            ) from exc
        row["working_folder"] = str(source_work)
        row["mvs_file"] = str(source_mvs)
        annotation_sidecar_raw = row.get("annotation_sidecar")
        if annotation_sidecar_raw:
            annotation_sidecar = Path(str(annotation_sidecar_raw)).expanduser()
            if not annotation_sidecar.is_absolute():
                annotation_sidecar = (
                    Path(str(config["_config_path"])).parent / annotation_sidecar
                )
            annotation_sidecar = annotation_sidecar.resolve()
            annotation_fit.load_annotation_sidecar(annotation_sidecar, scan_id)
            row["annotation_sidecar"] = str(annotation_sidecar)
        row.setdefault("name", scan_id[:8])
        row["name"] = validated_output_component(
            row["name"], f"scene {scan_id!r} name"
        )
        scenes.append(row)
    return scenes


def parse_argument(args: Iterable[str], name: str, default: int) -> int:
    values = [str(value) for value in args]
    raw_value = argument_value(values, name, str(default))
    try:
        return int(raw_value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


def has_argument(args: Iterable[str], name: str) -> bool:
    return any(
        value == name or value.startswith(f"{name}=")
        for value in (str(item) for item in args)
    )


def replace_argument_value(arguments: list[str], name: str, value: str) -> list[str]:
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
        if item.startswith(f"{name}="):
            result.extend([name, value])
            index += 1
            replaced = True
            continue
        result.append(item)
        index += 1
    if not replaced:
        result.extend([name, value])
    return result


def argument_value(arguments: list[str], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return default


def without_value_arguments(args: Iterable[str], names: set[str]) -> list[str]:
    """Remove known ``--option value`` or ``--option=value`` pairs."""
    values = [str(value) for value in args]
    filtered: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in names:
            index += 2
            continue
        if any(value.startswith(f"{name}=") for name in names):
            index += 1
            continue
        filtered.append(value)
        index += 1
    return filtered


REMOVED_INSTRUMENTATION_ARGUMENTS = {
    "--dmap-instrumentation-exact",
    "--dmap-instrumentation-view-probability-health",
    "--pm-instrument-config",
    "--pm-instrument-output",
    "--pm-instrument-level",
}


def validate_supported_densify_args(args: Iterable[str]) -> None:
    """Validate option/value shape and reject controls owned by orchestration."""

    values = [str(item) for item in args]
    index = 0
    while index < len(values):
        value = values[index]
        if not value.startswith("--"):
            raise ValueError(
                f"positional DensifyPointCloud input is not allowed in configured arguments: {value!r}"
            )
        name, separator, inline_value = value.partition("=")
        if re.fullmatch(r"--[A-Za-z0-9][A-Za-z0-9-]*", name) is None:
            raise ValueError(f"invalid DensifyPointCloud option {name!r}")
        if name in REMOVED_INSTRUMENTATION_ARGUMENTS:
            raise ValueError(
                f"{name} was removed from the public observability contract; "
                "use --dmap-instrumentation-dir and a capture profile"
            )
        if name in ORCHESTRATION_OWNED_DENSIFY_ARGUMENTS:
            raise ValueError(
                f"{name} is orchestration-owned and cannot be set in configured arguments"
            )
        if separator:
            if not inline_value or any(character in inline_value for character in "\r\n"):
                raise ValueError(f"invalid value for DensifyPointCloud option {name}")
            index += 1
            continue
        if index + 1 >= len(values) or values[index + 1].startswith("--"):
            raise ValueError(f"DensifyPointCloud option {name} requires one value")
        if any(character in values[index + 1] for character in "\r\n"):
            raise ValueError(f"invalid value for DensifyPointCloud option {name}")
        index += 2


CAPTURE_PROFILE_MODES = {
    "endpoint": "endpoint",
    "summary": "timing",
    "prefilter": "prefilter",
    "deep": "maps",
}

EXACT_BASE_STATE_BYTES_PER_PIXEL = 77
EXACT_VIEW_BYTES_PER_PIXEL = 34
EXACT_HYSTERESIS_ITERATION_BYTES_PER_PIXEL = 48
EXACT_BASE_STATE_ARTIFACTS = 20
EXACT_VIEW_ARTIFACTS = 5
EXACT_HYSTERESIS_ITERATION_ARTIFACTS = 12
EXACT_FILE_OVERHEAD_BYTES = 4096
EXACT_TABLE_OVERHEAD_BYTES = 64 * 1024
LEGACY_FIXED_BYTES_PER_PIXEL = 159
LEGACY_PASS_BYTES_PER_PIXEL = 6
LEGACY_LOGICAL_STATE_BYTES_PER_PIXEL = 61


def low_texture_update_hysteresis_enabled(
    config: dict[str, Any], run: dict[str, Any], scenes: list[dict[str, Any]]
) -> bool:
    """Resolve the default-off mechanism conservatively across selected scenes."""
    enabled = False
    for scene in scenes or [{}]:
        overrides = {
            str(key).casefold(): str(value)
            for key, value in config_materialization.merge_ini_overrides(
                config, run, scene
            ).items()
        }
        raw_gain = overrides.get(
            "patchmatch cuda low texture update min gain".casefold(), "0"
        )
        raw_gate = overrides.get(
            "patchmatch cuda low texture update gate".casefold(), "0"
        )
        try:
            minimum_gain = float(raw_gain)
            gate = int(raw_gate, 10)
        except ValueError as exc:
            raise ValueError(
                "low-texture update hysteresis overrides must be numeric"
            ) from exc
        if not math.isfinite(minimum_gain) or minimum_gain < 0:
            raise ValueError(
                "PatchMatch CUDA Low Texture Update Min Gain must be finite and non-negative"
            )
        if gate not in {0, 1, 2, 3}:
            raise ValueError(
                "PatchMatch CUDA Low Texture Update Gate must be an integer in [0,3]"
            )
        if (minimum_gain > 0) != (gate != 0):
            raise ValueError(
                "low-texture update minimum gain and gate must be enabled or disabled together"
            )
        enabled = enabled or minimum_gain > 0
    return enabled


def exact_storage_accounting(
    logical_states: int,
    number_views: int,
    hysteresis_enabled: bool,
    maps_capability: bool,
) -> tuple[int, int, int]:
    """Mirror CUDA's uncompressed exact payload and file-overhead planner."""
    if not maps_capability:
        return 0, 0, 0
    iterative_states = max(0, logical_states - 1)
    bytes_per_pixel = logical_states * (
        EXACT_BASE_STATE_BYTES_PER_PIXEL
        + number_views * EXACT_VIEW_BYTES_PER_PIXEL
    )
    artifact_count = logical_states * (
        EXACT_BASE_STATE_ARTIFACTS
        + number_views * EXACT_VIEW_ARTIFACTS
    )
    if hysteresis_enabled:
        bytes_per_pixel += (
            iterative_states * EXACT_HYSTERESIS_ITERATION_BYTES_PER_PIXEL
        )
        artifact_count += (
            iterative_states * EXACT_HYSTERESIS_ITERATION_ARTIFACTS
        )
    fixed_bytes = (
        artifact_count * EXACT_FILE_OVERHEAD_BYTES + EXACT_TABLE_OVERHEAD_BYTES
        if logical_states > 0
        else 0
    )
    return bytes_per_pixel, artifact_count, fixed_bytes
OBSERVER_VALUE_ARGUMENTS = {
    "--dmap-instrumentation-config",
    "--dmap-instrumentation-dir",
    "--dmap-instrumentation-image-list",
    "--dmap-instrumentation-level",
    "--dmap-instrumentation-sample-rate",
    "--dmap-instrumentation-sample-seed",
    "--dmap-instrumentation-write-maps",
    "--dmap-instrumentation-max-device-mb",
    "--dmap-instrumentation-max-host-mb",
    "--dmap-instrumentation-max-frame-storage-mb",
    "--dmap-instrumentation-budget-policy",
}


def capture_profiles(config: dict[str, Any], requested: Iterable[str] = ()) -> list[str]:
    raw = [str(value) for value in requested]
    if not raw:
        raw = [str(value) for value in config.get("capture_profiles") or ("deep", "summary")]
    profiles: list[str] = []
    for profile in raw:
        if profile not in CAPTURE_PROFILE_MODES:
            raise ValueError(
                f"unsupported capture profile {profile!r}; expected endpoint, summary, prefilter, or deep"
            )
        if profile not in profiles:
            profiles.append(profile)
    if not profiles:
        raise ValueError("at least one capture profile is required")
    return profiles


def instrumentation_sample_rate(config: dict[str, Any]) -> float:
    """Return the workflow-owned deterministic observer sampling rate."""

    instrumentation = config.get("instrumentation") or {}
    if not isinstance(instrumentation, dict):
        raise ValueError("instrumentation must be a mapping")
    raw_value = instrumentation.get("sample_rate", 1.0)
    if isinstance(raw_value, bool):
        raise ValueError("instrumentation.sample_rate must be a number in (0,1]")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "instrumentation.sample_rate must be a number in (0,1]"
        ) from exc
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("instrumentation.sample_rate must be a number in (0,1]")
    return value


def estimate_storage(
    config: dict[str, Any],
    scenes: list[dict[str, Any]],
    requested_profiles: Iterable[str] = (),
) -> dict[str, Any]:
    instrumentation = config.get("instrumentation") or {}
    sample_rate = instrumentation_sample_rate(config)
    if "exact_observability" in instrumentation:
        raise ValueError(
            "instrumentation.exact_observability was removed: deep/maps captures "
            "always use the exact Process<true> observer"
        )
    width = int(instrumentation.get("expected_width", 708))
    height = int(instrumentation.get("expected_height", 944))
    frames_default = int(instrumentation.get("expected_frames_per_scene", 100))
    if width <= 0 or height <= 0:
        raise ValueError("instrumentation expected_width and expected_height must be positive")
    if frames_default <= 0:
        raise ValueError("instrumentation expected_frames_per_scene must be positive")
    run_specs = config.get("runs") or []
    area = width * height
    default_args = config.get("default_densify_args") or []
    validate_supported_densify_args(default_args)
    configured_profiles = capture_profiles(config)
    requested_profile_values = tuple(str(value) for value in requested_profiles)
    effective_profiles = list(configured_profiles)
    for profile in (
        capture_profiles(config, requested_profile_values)
        if requested_profile_values else ()
    ):
        if profile not in effective_profiles:
            effective_profiles.append(profile)
    maps_capability = (
        bool(instrumentation.get("maps_capability", True))
        and "deep" in effective_profiles
    )
    prefilter_capability = "prefilter" in effective_profiles
    # A deep/maps capture always uses the exact Process<true> observer. There
    # is no independent exact-observability runtime toggle.
    exact_capability = maps_capability
    run_estimates: list[dict[str, Any]] = []
    for run in run_specs:
        args = [*default_args, *(run.get("densify_args") or [])]
        validate_supported_densify_args(args)
        iterations = parse_argument(args, "--iters", 3)
        geometric_iterations = parse_argument(args, "--geometric-iters", 0)
        estimation_stages = 1 + geometric_iterations
        num_views = parse_argument(args, "--number-views", 5)
        if iterations < 0 or geometric_iterations < 0 or num_views <= 0:
            raise ValueError(
                "--iters and --geometric-iters must be non-negative and "
                "--number-views must be positive"
            )
        passes = 1 + 2 * iterations
        logical_states = 1 + iterations
        legacy_bytes_per_pixel = (
            LEGACY_FIXED_BYTES_PER_PIXEL
            + LEGACY_PASS_BYTES_PER_PIXEL * passes
            + LEGACY_LOGICAL_STATE_BYTES_PER_PIXEL * logical_states
            if maps_capability else 0
        )
        hysteresis_enabled = low_texture_update_hysteresis_enabled(
            config, run, scenes
        )
        (
            exact_bytes_per_pixel,
            exact_artifact_count,
            exact_fixed_bytes,
        ) = exact_storage_accounting(
            logical_states, num_views, hysteresis_enabled, exact_capability
        )
        # Match the conservative optional-filter planner: two sequential
        # postprocess stages at 54 B/pixel each (including terminal normals)
        # plus the 32 B/pixel confidence bundle and one 4 KiB allowance per map.
        filter_postprocess_stages = max(0, int(instrumentation.get("filter_postprocess_stages", 2)))
        filter_confidence_enabled = bool(instrumentation.get("filter_confidence_adjustment", True))
        filter_map_count = filter_postprocess_stages * 11 + (11 if filter_confidence_enabled else 0)
        filter_bytes_per_pixel = (
            filter_postprocess_stages * 54 + (32 if filter_confidence_enabled else 0)
        ) if maps_capability else 0
        filter_fixed_bytes = filter_map_count * 4096 if maps_capability else 0
        filter_bytes_per_frame = area * filter_bytes_per_pixel + filter_fixed_bytes
        deep_bytes_per_stage = (
            area * (legacy_bytes_per_pixel + exact_bytes_per_pixel)
            + exact_fixed_bytes
            + filter_bytes_per_frame
        )
        # The bounded prefilter profile retains a single float32 depth snapshot
        # plus a deliberately loose allowance for the manifest and completion
        # marker. Device/host admission is enforced independently by CUDA.
        prefilter_bytes_per_stage = (
            area * 4 + 64 * 1024 if prefilter_capability else 0
        )
        bytes_per_stage = deep_bytes_per_stage + prefilter_bytes_per_stage
        bytes_per_frame = bytes_per_stage * estimation_stages
        run_estimates.append({
            "label": str(run.get("label", "run")),
            "role": str(run.get("role", "variant")),
            "repeats": int(run.get("repeats", 3 if run.get("role") == "baseline" else 1)),
            "iterations": iterations,
            "geometric_iterations": geometric_iterations,
            "estimation_stages": estimation_stages,
            "passes": passes,
            "logical_states": logical_states,
            "number_views": num_views,
            "legacy_bytes_per_pixel": legacy_bytes_per_pixel,
            "exact_bytes_per_pixel": exact_bytes_per_pixel,
            "exact_artifact_count": exact_artifact_count,
            "exact_fixed_bytes": exact_fixed_bytes,
            "low_texture_update_hysteresis": hysteresis_enabled,
            "filter_bytes_per_pixel": filter_bytes_per_pixel,
            "filter_fixed_bytes": filter_fixed_bytes,
            "filter_bytes_per_frame_raw": filter_bytes_per_frame,
            "deep_bytes_per_stage_raw": deep_bytes_per_stage,
            "prefilter_bytes_per_stage_raw": prefilter_bytes_per_stage,
            "bytes_per_stage_raw": bytes_per_stage,
            "bytes_per_frame_raw": bytes_per_frame,
        })
    if not run_estimates:
        default_states = 4
        default_views = 5
        default_hysteresis = low_texture_update_hysteresis_enabled(
            config, {}, scenes
        )
        (
            default_exact_bytes_per_pixel,
            default_exact_artifact_count,
            default_exact_fixed_bytes,
        ) = exact_storage_accounting(
            default_states, default_views, default_hysteresis, exact_capability
        )
        default_filter_fixed_bytes = 33 * 4096 if maps_capability else 0
        default_filter_bytes_per_frame = (
            area * (140 if maps_capability else 0) + default_filter_fixed_bytes
        )
        default_deep_bytes_per_stage = (
            area * (
                (445 if maps_capability else 0)
                + default_exact_bytes_per_pixel
            )
            + default_exact_fixed_bytes
            + default_filter_bytes_per_frame
        )
        default_prefilter_bytes_per_stage = (
            area * 4 + 64 * 1024 if prefilter_capability else 0
        )
        default_bytes_per_stage = (
            default_deep_bytes_per_stage + default_prefilter_bytes_per_stage
        )
        run_estimates.append({
            "label": "default", "role": "baseline", "repeats": 1, "iterations": 3,
            "geometric_iterations": 0, "estimation_stages": 1,
            "passes": 7, "logical_states": default_states, "number_views": default_views,
            "legacy_bytes_per_pixel": 445 if maps_capability else 0,
            "exact_bytes_per_pixel": default_exact_bytes_per_pixel,
            "exact_artifact_count": default_exact_artifact_count,
            "exact_fixed_bytes": default_exact_fixed_bytes,
            "low_texture_update_hysteresis": default_hysteresis,
            "filter_bytes_per_pixel": 140 if maps_capability else 0,
            "filter_fixed_bytes": default_filter_fixed_bytes,
            "filter_bytes_per_frame_raw": default_filter_bytes_per_frame,
            "deep_bytes_per_stage_raw": default_deep_bytes_per_stage,
            "prefilter_bytes_per_stage_raw": default_prefilter_bytes_per_stage,
            "bytes_per_stage_raw": default_bytes_per_stage,
            "bytes_per_frame_raw": default_bytes_per_stage,
        })
    bytes_per_frame_raw = max(int(row["bytes_per_frame_raw"]) for row in run_estimates)
    scene_estimates = []
    total_frames = 0
    for scene in scenes:
        frames = int(scene.get("expected_frames", frames_default))
        total_frames += frames
        scene_estimates.append({
            "scan_id": scene["scan_id"],
            "frames": frames,
            "estimated_bytes_per_run": {
                str(row["label"]): frames * int(row["bytes_per_frame_raw"])
                for row in run_estimates
            },
        })
    full_runs = (
        sum(int(row["repeats"]) for row in run_estimates)
        if maps_capability
        else 0
    )
    prefilter_runs = (
        sum(int(row["repeats"]) for row in run_estimates)
        if prefilter_capability
        else 0
    )
    timing_runs = len(run_specs)
    estimated_bytes = total_frames * sum(
        int(row["bytes_per_frame_raw"]) * int(row["repeats"]) for row in run_estimates
    )
    root = experiment_root(config)
    usage_root = root.parent
    while not usage_root.exists():
        parent = usage_root.parent
        if parent == usage_root:
            raise RuntimeError(
                f"cannot find an existing filesystem for storage estimate: {root}"
            )
        usage_root = parent
    if not usage_root.is_dir():
        usage_root = usage_root.parent
    usage = shutil.disk_usage(usage_root)
    budget_gb = float(instrumentation.get("max_artifact_gb", 0.0))
    budget_bytes = int(budget_gb * 1024**3) if budget_gb > 0 else usage.free
    return {
        "schema_version": SCHEMA_VERSION,
        "width": width,
        "height": height,
        "passes": max(int(row["passes"]) for row in run_estimates),
        "logical_states": max(int(row["logical_states"]) for row in run_estimates),
        "number_views": max(int(row["number_views"]) for row in run_estimates),
        "maps_capability": maps_capability,
        "prefilter_capability": prefilter_capability,
        "exact_observability": exact_capability,
        "configured_capture_profiles": configured_profiles,
        "effective_capture_profiles": effective_profiles,
        "sample_rate": sample_rate,
        "sampling_storage_assumption": (
            "conservative full expected-frame count; deterministic sampling may retain fewer frames"
        ),
        "storage_model": "sum of requested capture profiles per photometric/geometric stage: deep=[PatchMatch legacy (159+6P+61L) + exact L*(77+34V)+H*(L-1)*48 + optional filters (54S+32C) bytes/pixel, plus exact/filter file overhead]; prefilter=[one float32 depth map + 64 KiB manifest allowance]",
        "bytes_per_frame_raw": bytes_per_frame_raw,
        "bytes_per_frame_estimated": bytes_per_frame_raw,
        "total_frames": total_frames,
        "full_map_runs": full_runs,
        "prefilter_runs": prefilter_runs,
        "timing_runs": timing_runs,
        "estimated_bytes": estimated_bytes,
        "estimated_gib": estimated_bytes / 1024**3,
        "available_bytes": usage.free,
        "available_gib": usage.free / 1024**3,
        "budget_bytes": budget_bytes,
        "within_budget": estimated_bytes <= min(usage.free, budget_bytes),
        "runs": run_estimates,
        "scenes": scene_estimates,
    }


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "size": resolved.stat().st_size if resolved.is_file() else None,
        "sha256": dmap_drilldown.file_digest(resolved) if resolved.is_file() else None,
    }


ENVIRONMENT_MANIFEST_SCHEMA_NAME = "openmvs.dmap.environment_manifest"
ENVIRONMENT_MANIFEST_SCHEMA_VERSION = 1
ENVIRONMENT_MANIFEST_FILE = "00_environment_manifest.json"
ENVIRONMENT_PROBE_TIMEOUT_SECONDS = 5.0
ENVIRONMENT_VARIABLE_ALLOWLIST = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_SCHEDULE",
    "OPENBLAS_NUM_THREADS",
)
RELEVANT_CMAKE_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_CUDA_ARCHITECTURES",
    "CMAKE_CUDA_COMPILER",
    "CMAKE_CUDA_COMPILER_VERSION",
    "CMAKE_CXX_COMPILER",
    "CMAKE_CXX_COMPILER_VERSION",
    "CMAKE_GENERATOR",
    "CMAKE_TOOLCHAIN_FILE",
    "OpenMVS_DMAP_INSTRUMENTATION",
    "OpenMVS_HEADLESS_DEBUG",
    "OpenMVS_USE_CUDA",
    "VCPKG_TARGET_TRIPLET",
)


def bounded_environment_probe(command: list[str]) -> dict[str, Any]:
    """Run a local, read-only version probe with an explicit time/output bound."""

    executable = shutil.which(command[0])
    if executable is None:
        return {
            "status": "unavailable",
            "reason": f"{command[0]} was not found on PATH",
            "command": command,
        }
    resolved_command = [executable, *command[1:]]
    try:
        completed = subprocess.run(
            resolved_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "unavailable",
            "reason": (
                f"probe exceeded {ENVIRONMENT_PROBE_TIMEOUT_SECONDS:g} seconds"
            ),
            "command": command,
            "executable": file_identity(Path(executable)),
        }
    stdout = completed.stdout[:16_384].strip()
    stderr = completed.stderr[:16_384].strip()
    return {
        "status": "available" if completed.returncode == 0 else "unavailable",
        "reason": "" if completed.returncode == 0 else f"exit code {completed.returncode}",
        "command": command,
        "executable": file_identity(Path(executable)),
        "return_code": int(completed.returncode),
        "stdout": stdout,
        "stderr": stderr,
        "output_truncated": (
            len(completed.stdout) > 16_384 or len(completed.stderr) > 16_384
        ),
    }


def parse_cmake_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        name_with_type, value = line.split("=", 1)
        name = name_with_type.split(":", 1)[0]
        if name in RELEVANT_CMAKE_CACHE_KEYS:
            values[name] = value
    return values


def cmake_cache_for_executable(executable: Path) -> Path | None:
    resolved = executable.expanduser().resolve()
    for parent in resolved.parents:
        candidate = parent / "CMakeCache.txt"
        if candidate.is_file():
            return candidate
        if parent == REPO_ROOT or parent.parent == parent:
            break
    return None


def build_environment_manifest(
    executable_boundary: dict[str, Any],
) -> dict[str, Any]:
    """Capture a deterministic, local-only execution environment identity."""

    distributions = sorted(
        (
            {
                "name": str(distribution.metadata.get("Name") or "unknown"),
                "version": str(distribution.version),
            }
            for distribution in importlib.metadata.distributions()
        ),
        key=lambda row: (row["name"].lower(), row["version"]),
    )
    try:
        os_release: dict[str, Any] = dict(platform.freedesktop_os_release())
        os_release_status = "available"
        os_release_reason = ""
    except (OSError, RuntimeError):
        os_release = {}
        os_release_status = "unavailable"
        os_release_reason = "platform.freedesktop_os_release could not read OS metadata"

    requirements = [
        file_identity(SCRIPT_DIR / "requirements-depth-benchmark.txt"),
        file_identity(SCRIPT_DIR / "requirements-dmap-array-store.txt"),
    ]
    project_inputs = [
        file_identity(REPO_ROOT / "CMakeLists.txt"),
        file_identity(REPO_ROOT / "vcpkg.json"),
    ]
    builds: list[dict[str, Any]] = []
    compiler_paths: set[Path] = set()
    cuda_compiler_paths: set[Path] = set()
    for role in ("production", "observer"):
        identity = (executable_boundary.get(role) or {}).get("identity") or {}
        executable_path = Path(str(identity.get("path") or "")).expanduser()
        cache_path = cmake_cache_for_executable(executable_path)
        cache_values = parse_cmake_cache(cache_path) if cache_path is not None else {}
        build_record: dict[str, Any] = {
            "role": role,
            "executable": identity,
            "runtime_boundary": (executable_boundary.get(role) or {}).get(
                "runtime_boundary"
            ),
            "cmake_cache": (
                file_identity(cache_path) if cache_path is not None else {
                    "path": None,
                    "exists": False,
                    "size": None,
                    "sha256": None,
                }
            ),
            "cmake_values": cache_values,
        }
        toolchain = cache_values.get("CMAKE_TOOLCHAIN_FILE")
        build_record["toolchain_file"] = (
            file_identity(Path(toolchain)) if toolchain else None
        )
        cxx_compiler = cache_values.get("CMAKE_CXX_COMPILER")
        if cxx_compiler:
            compiler_paths.add(Path(cxx_compiler).expanduser().resolve())
        cuda_compiler = cache_values.get("CMAKE_CUDA_COMPILER")
        if cuda_compiler:
            cuda_compiler_paths.add(Path(cuda_compiler).expanduser().resolve())
        builds.append(build_record)

    compiler_probes = [
        bounded_environment_probe([str(path), "--version"])
        for path in sorted(compiler_paths)
    ]
    cuda_compiler_probes = [
        bounded_environment_probe([str(path), "--version"])
        for path in sorted(cuda_compiler_paths)
    ]
    if not cuda_compiler_probes:
        cuda_compiler_probes.append(bounded_environment_probe(["nvcc", "--version"]))

    gpu_probe = bounded_environment_probe([
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    gpu_rows: list[dict[str, str]] = []
    if gpu_probe.get("status") == "available":
        for row in csv.reader(str(gpu_probe.get("stdout") or "").splitlines()):
            if len(row) != 3:
                gpu_probe["status"] = "unavailable"
                gpu_probe["reason"] = "nvidia-smi returned an unexpected GPU record"
                gpu_rows = []
                break
            gpu_rows.append({
                "name": row[0].strip(),
                "driver_version": row[1].strip(),
                "compute_capability": row[2].strip(),
            })
        gpu_rows.sort(
            key=lambda row: (
                row["name"], row["driver_version"], row["compute_capability"]
            )
        )

    manifest: dict[str, Any] = {
        "schema_name": ENVIRONMENT_MANIFEST_SCHEMA_NAME,
        "schema_version": ENVIRONMENT_MANIFEST_SCHEMA_VERSION,
        "network_access_used": False,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:5]),
            "executable": file_identity(Path(sys.executable)),
            "resolved_distributions": distributions,
            "requirements": requirements,
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "os_release_status": os_release_status,
            "os_release_reason": os_release_reason,
            "os_release": os_release,
        },
        "environment_variables": {
            name: os.environ.get(name) for name in ENVIRONMENT_VARIABLE_ALLOWLIST
        },
        "project_inputs": project_inputs,
        "builds": builds,
        "tools": {
            "cmake": bounded_environment_probe(["cmake", "--version"]),
            "ninja": bounded_environment_probe(["ninja", "--version"]),
            "cxx_compilers": compiler_probes,
            "cuda_compilers": cuda_compiler_probes,
        },
        "gpu": {
            "probe": gpu_probe,
            "devices": gpu_rows,
            "status": gpu_probe.get("status", "unavailable"),
            "reason": gpu_probe.get("reason", ""),
        },
    }
    return manifest


def validate_environment_manifest(value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("environment manifest must be a JSON object")
    if (
        value.get("schema_name") != ENVIRONMENT_MANIFEST_SCHEMA_NAME
        or value.get("schema_version") != ENVIRONMENT_MANIFEST_SCHEMA_VERSION
    ):
        raise RuntimeError("environment manifest schema is unsupported")


STAGED_INPUT_EXCLUDE_PATTERNS = (
    "depth*.dmap",
    "*.log",
    "*_dense.mvs",
    "depth_maps",
    "dmap_instrumentation",
)
DEFAULT_INPUT_SNAPSHOT_MAX_FILES = 250_000
DEFAULT_INPUT_SNAPSHOT_MAX_BYTES = 1 << 40


def staged_input_snapshot(
    source_work: Path,
    *,
    max_files: int = DEFAULT_INPUT_SNAPSHOT_MAX_FILES,
    max_bytes: int = DEFAULT_INPUT_SNAPSHOT_MAX_BYTES,
) -> dict[str, Any]:
    """Hash the exact bounded file set copied into each capture workspace."""

    source_work = source_work.expanduser().resolve()
    if not source_work.is_dir():
        raise FileNotFoundError(f"scene working folder is unavailable: {source_work}")
    if max_files <= 0 or max_bytes <= 0:
        raise ValueError("input snapshot limits must be positive")
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(
        source_work, followlinks=False
    ):
        directory_path = Path(directory)
        retained_directories = []
        for name in sorted(directory_names):
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in STAGED_INPUT_EXCLUDE_PATTERNS):
                continue
            child = directory_path / name
            if child.is_symlink():
                raise RuntimeError(
                    f"scene input snapshot rejects symlinked directory: {child}"
                )
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in STAGED_INPUT_EXCLUDE_PATTERNS):
                continue
            path = directory_path / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"scene input snapshot requires regular non-symlink files: {path}"
                )
            total_bytes += int(metadata.st_size)
            if len(rows) + 1 > max_files or total_bytes > max_bytes:
                raise RuntimeError(
                    "scene input snapshot exceeds its configured bound: "
                    f"files={len(rows) + 1}/{max_files}, bytes={total_bytes}/{max_bytes}"
                )
            digest = dmap_drilldown.file_digest(path)
            current = path.lstat()
            if (
                current.st_size != metadata.st_size
                or current.st_mtime_ns != metadata.st_mtime_ns
            ):
                raise RuntimeError(f"scene input changed while it was hashed: {path}")
            rows.append({
                "path": path.relative_to(source_work).as_posix(),
                "bytes": int(metadata.st_size),
                "sha256": digest,
            })
    rows.sort(key=lambda row: row["path"])
    return {
        "schema_name": "openmvs.dmap.staged_input_snapshot",
        "schema_version": 1,
        "excluded_patterns": list(STAGED_INPUT_EXCLUDE_PATTERNS),
        "max_files": int(max_files),
        "max_bytes": int(max_bytes),
        "file_count": len(rows),
        "total_bytes": total_bytes,
        "snapshot_sha256": stable_json_digest(rows),
        "files": rows,
    }


def configured_scene_input_snapshot(
    config: dict[str, Any], scene: dict[str, Any]
) -> dict[str, Any]:
    policy = config.get("input_snapshot") or {}
    if not isinstance(policy, dict):
        raise ValueError("input_snapshot must be a mapping")
    max_files = int(policy.get("max_files", DEFAULT_INPUT_SNAPSHOT_MAX_FILES))
    max_bytes = int(policy.get("max_bytes", DEFAULT_INPUT_SNAPSHOT_MAX_BYTES))
    source_work = Path(str(scene["working_folder"])).expanduser().resolve()
    source_mvs = Path(str(scene["mvs_file"])).expanduser().resolve()
    snapshot = staged_input_snapshot(
        source_work,
        max_files=max_files,
        max_bytes=max_bytes,
    )
    relative_mvs = scene_mvs_relative_path(source_work, source_mvs)
    try:
        source_mvs.relative_to(source_work)
    except ValueError:
        metadata = source_mvs.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"external scene MVS must be a regular non-symlink file: {source_mvs}"
            )
        existing = next(
            (row for row in snapshot["files"] if row["path"] == relative_mvs.as_posix()),
            None,
        )
        row = {
            "path": relative_mvs.as_posix(),
            "bytes": int(metadata.st_size),
            "sha256": dmap_drilldown.file_digest(source_mvs),
        }
        if existing is not None and existing != row:
            raise RuntimeError(
                f"external scene MVS conflicts with staged input path {relative_mvs}"
            )
        if existing is None:
            snapshot["files"].append(row)
            snapshot["files"].sort(key=lambda item: item["path"])
            snapshot["file_count"] += 1
            snapshot["total_bytes"] += int(metadata.st_size)
            if (
                snapshot["file_count"] > max_files
                or snapshot["total_bytes"] > max_bytes
            ):
                raise RuntimeError("external scene MVS exceeds the input snapshot bound")
            snapshot["snapshot_sha256"] = stable_json_digest(snapshot["files"])
    return snapshot


def locked_scene_record(root: Path, scene_id: str) -> dict[str, Any]:
    lock = read_json(root / "00_experiment_lock.json")
    record = next(
        (
            row for row in lock.get("scenes") or []
            if isinstance(row, dict) and row.get("scan_id") == scene_id
        ),
        None,
    )
    if not isinstance(record, dict):
        raise RuntimeError(f"experiment lock has no scene record for {scene_id!r}")
    return record


def validate_scene_input_snapshot(
    config: dict[str, Any], root: Path, scene: dict[str, Any]
) -> None:
    """Fail before a profile stages inputs whose locked bytes have changed."""

    scene_id = str(scene.get("scan_id", ""))
    expected = locked_scene_record(root, scene_id).get("staged_input_snapshot")
    if not isinstance(expected, dict):
        raise RuntimeError(
            f"experiment lock has no staged input snapshot for scene {scene_id!r}"
        )
    actual = configured_scene_input_snapshot(config, scene)
    if actual != expected:
        raise RuntimeError(
            f"staged input snapshot changed for scene {scene_id!r}; "
            "use a new experiment_id"
        )


def frozen_scene_work_dir(root: Path, scene_id: str) -> Path:
    return _lexical_absolute_path(root).joinpath(
        "frozen_inputs",
        validated_output_component(scene_id, "frozen input scene_id"),
        "work",
    )


def scene_mvs_relative_path(
    source_work: Path, source_mvs: Path
) -> Path:
    try:
        return source_mvs.resolve().relative_to(source_work.resolve())
    except ValueError:
        return Path(source_mvs.name)


def _snapshot_limits(snapshot: dict[str, Any]) -> tuple[int, int]:
    return int(snapshot["max_files"]), int(snapshot["max_bytes"])


def validate_staged_scene_input(
    staged_work: Path,
    expected: dict[str, Any],
    expected_mvs: dict[str, Any],
    relative_mvs: Path,
    *,
    description: str,
) -> None:
    max_files, max_bytes = _snapshot_limits(expected)
    actual = staged_input_snapshot(
        staged_work, max_files=max_files, max_bytes=max_bytes
    )
    if actual != expected:
        raise RuntimeError(f"{description} does not match the locked input snapshot")
    local_mvs = staged_work / relative_mvs
    local_identity = file_identity(local_mvs)
    if (
        local_identity.get("exists") is not True
        or local_identity.get("size") != expected_mvs.get("size")
        or local_identity.get("sha256") != expected_mvs.get("sha256")
    ):
        raise RuntimeError(f"{description} scene MVS identity does not match the lock")


def prepare_frozen_scene_input(
    config: dict[str, Any], root: Path, scene: dict[str, Any], scene_record: dict[str, Any]
) -> Path:
    """Copy one independent, read-only scene snapshot owned by the experiment."""

    scene_id = str(scene["scan_id"])
    destination = _lexical_absolute_path(root).joinpath(
        "frozen_inputs",
        validated_output_component(scene_id, "frozen input scene_id"),
        "work",
    )
    _reject_symlink_components(
        destination, description=f"frozen scene {scene_id!r} destination"
    )
    expected = scene_record["staged_input_snapshot"]
    expected_mvs = scene_record["mvs_file"]
    source_work = Path(str(scene["working_folder"])).expanduser().resolve()
    source_mvs = Path(str(scene["mvs_file"])).expanduser().resolve()
    relative_mvs = scene_mvs_relative_path(source_work, source_mvs)
    if destination.exists():
        validate_staged_scene_input(
            destination,
            expected,
            expected_mvs,
            relative_mvs,
            description=f"frozen scene {scene_id!r}",
        )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".work.tmp-{os.getpid()}-{time.time_ns()}"
    _reject_symlink_components(
        temporary, description=f"frozen scene {scene_id!r} temporary directory"
    )
    try:
        shutil.copytree(
            source_work,
            temporary,
            copy_function=shutil.copy2,
            ignore=shutil.ignore_patterns(*STAGED_INPUT_EXCLUDE_PATTERNS),
        )
        try:
            source_mvs.relative_to(source_work)
        except ValueError:
            local_mvs = temporary / relative_mvs
            if local_mvs.exists():
                raise RuntimeError(
                    f"external scene MVS conflicts with frozen path {relative_mvs}"
                )
            local_mvs.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mvs, local_mvs)
        validate_staged_scene_input(
            temporary,
            expected,
            expected_mvs,
            relative_mvs,
            description=f"new frozen scene {scene_id!r}",
        )
        for path in temporary.rglob("*"):
            if path.is_file():
                path.chmod(path.stat().st_mode & ~(
                    stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
                ))
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    write_json(destination.parent / "snapshot.json", expected)
    return destination


EXECUTABLE_BOUNDARY_SCHEMA_NAME = "openmvs.dmap.executable_boundary"
EXECUTABLE_BOUNDARY_SCHEMA_VERSION = 1
REQUIRED_OBSERVER_CLI_FLAGS = frozenset({
    "--dmap-instrumentation-dir",
    "--dmap-instrumentation-level",
    "--dmap-instrumentation-write-maps",
})


def runtime_boundary_identity(executable: Path) -> dict[str, Any]:
    """Bind the executable and build-local OpenMVS shared libraries.

    This intentionally covers the mutable build products that a concurrent
    relink replaces. System, CUDA, and package-manager libraries remain part of
    the separately recorded environment rather than this per-capture boundary.
    """

    loader_overrides = {
        name: os.environ.get(name, "")
        for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT")
        if os.environ.get(name, "")
    }
    if loader_overrides:
        raise RuntimeError(
            "LD_LIBRARY_PATH, LD_PRELOAD, and LD_AUDIT must be empty for attested "
            "captures because loader overrides can bypass or interpose the locked runtime: "
            f"{sorted(loader_overrides)}"
        )
    path = executable.expanduser().resolve(strict=True)
    executable_identity = integrity.regular_file_identity(path)
    if not executable_identity["mode"] & 0o111:
        raise PermissionError(f"DensifyPointCloud binary is not executable: {path}")
    libraries: list[dict[str, Any]] = []
    with os.scandir(path.parent) as iterator:
        entries = sorted(iterator, key=lambda entry: entry.name)
    for entry in entries:
        if not fnmatch.fnmatchcase(entry.name, "lib*.so*"):
            continue
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(
                "runtime-boundary sibling libraries must be regular files; "
                f"symlinked installed runtimes are not supported: {entry.path}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                f"runtime-boundary sibling is not a regular file: {entry.path}"
            )
        libraries.append(integrity.regular_file_identity(Path(entry.path)))
    value: dict[str, Any] = {
        "schema_name": RUNTIME_BOUNDARY_SCHEMA_NAME,
        "schema_version": RUNTIME_BOUNDARY_SCHEMA_VERSION,
        "executable": executable_identity,
        "shared_libraries": libraries,
        "loader_override_policy": {
            "LD_LIBRARY_PATH": "empty_required",
            "LD_PRELOAD": "empty_required",
            "LD_AUDIT": "empty_required",
        },
        "external_dynamic_libraries": "environment_record_only",
    }
    value["boundary_sha256"] = stable_json_digest(value)
    return value


def validate_runtime_boundary_shape(value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("runtime boundary must be a JSON object")
    unsigned = dict(value)
    declared_digest = unsigned.pop("boundary_sha256", None)
    if not (
        value.get("schema_name") == RUNTIME_BOUNDARY_SCHEMA_NAME
        and value.get("schema_version") == RUNTIME_BOUNDARY_SCHEMA_VERSION
        and declared_digest == stable_json_digest(unsigned)
        and isinstance(value.get("executable"), dict)
        and isinstance(value.get("shared_libraries"), list)
    ):
        raise RuntimeError("runtime boundary schema or digest is invalid")


def probe_densify_cli(path: Path) -> dict[str, Any]:
    """Probe one binary's help surface without leaving logs in the source tree."""

    with tempfile.TemporaryDirectory(prefix="openmvs-dmap-cli-probe-") as directory:
        probe_root = Path(directory)
        try:
            result = subprocess.run(
                [str(path), "--help"],
                cwd=probe_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"cannot probe DensifyPointCloud CLI at {path}: {exc}") from exc
        payload = bytearray(result.stdout or b"")
        for candidate in sorted(probe_root.rglob("*")):
            if not candidate.is_file() or candidate.stat().st_size > 4 * 1024 * 1024:
                continue
            payload.extend(b"\n")
            payload.extend(candidate.read_bytes())
    text = bytes(payload).decode("utf-8", errors="replace")
    cli_flags = sorted(set(re.findall(r"(?<![A-Za-z0-9-])--[a-z0-9][a-z0-9-]*", text)))
    observer_flags = [
        flag for flag in cli_flags if flag.startswith("--dmap-instrumentation-")
    ]
    return {
        "command": [str(path), "--help"],
        "return_code": result.returncode,
        "option_count": len(cli_flags),
        "option_surface_sha256": stable_json_digest(cli_flags),
        "observer_flags": observer_flags,
    }


def attest_densify_executable_boundary(config: dict[str, Any]) -> dict[str, Any]:
    """Require distinct executable production/observer binaries and CLI surfaces."""

    production = densify_binary(config, instrumented=False).expanduser().resolve()
    observer = densify_binary(config, instrumented=True).expanduser().resolve()
    for label, path in (("production", production), ("observer", observer)):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} DensifyPointCloud binary is not a regular file: {path}"
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"{label} DensifyPointCloud binary is not executable: {path}"
            )
    if production == observer or os.path.samefile(production, observer):
        raise ValueError("production and observer DensifyPointCloud binaries must be distinct")
    production_runtime = runtime_boundary_identity(production)
    observer_runtime = runtime_boundary_identity(observer)
    production_identity = file_identity(production)
    observer_identity = file_identity(observer)
    if production_identity["sha256"] == observer_identity["sha256"]:
        raise ValueError(
            "production and observer DensifyPointCloud binaries have identical content"
        )
    production_probe = probe_densify_cli(production)
    observer_probe = probe_densify_cli(observer)
    if runtime_boundary_identity(production) != production_runtime:
        raise RuntimeError("production runtime boundary changed during CLI attestation")
    if runtime_boundary_identity(observer) != observer_runtime:
        raise RuntimeError("observer runtime boundary changed during CLI attestation")
    production_flags = set(production_probe["observer_flags"])
    observer_flags = set(observer_probe["observer_flags"])
    if production_flags:
        raise RuntimeError(
            "production DensifyPointCloud unexpectedly exposes observer CLI flags: "
            f"{sorted(production_flags)}"
        )
    missing_observer_flags = REQUIRED_OBSERVER_CLI_FLAGS - observer_flags
    if missing_observer_flags:
        raise RuntimeError(
            "observer DensifyPointCloud is missing required instrumentation CLI flags: "
            f"{sorted(missing_observer_flags)}"
        )
    return {
        "schema_name": EXECUTABLE_BOUNDARY_SCHEMA_NAME,
        "schema_version": EXECUTABLE_BOUNDARY_SCHEMA_VERSION,
        "valid": True,
        "production": {
            "identity": production_identity,
            "runtime_boundary": production_runtime,
            "cli_probe": production_probe,
            "instrumentation_compiled": False,
        },
        "observer": {
            "identity": observer_identity,
            "runtime_boundary": observer_runtime,
            "cli_probe": observer_probe,
            "instrumentation_compiled": True,
        },
        "required_observer_flags": sorted(REQUIRED_OBSERVER_CLI_FLAGS),
    }


def rebase_report_output_paths(
    value: Any,
    generated_output_dir: Path,
    published_output_dir: Path,
) -> Any:
    """Rebase report-owned absolute paths before a staged report is attested."""

    generated = generated_output_dir.expanduser().resolve()
    published = published_output_dir.expanduser().absolute()

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        if not isinstance(item, str) or not Path(item).is_absolute():
            return item
        try:
            relative = Path(item).relative_to(generated)
        except ValueError:
            return item
        return str(published / relative)

    return visit(value)


def rebase_report_dataframe_paths(
    dataframe: pd.DataFrame,
    generated_output_dir: Path,
    published_output_dir: Path,
) -> pd.DataFrame:
    """Rebase report-owned path values without changing the working dataframe."""

    if dataframe.empty:
        return dataframe
    rebased = dataframe.copy()
    for column in rebased.columns:
        if not (
            pd.api.types.is_object_dtype(rebased[column])
            or pd.api.types.is_string_dtype(rebased[column])
        ):
            continue
        rebased[column] = rebased[column].map(
            lambda value: rebase_report_output_paths(
                value, generated_output_dir, published_output_dir
            )
        )
    return rebased

def build_experiment_lock(
    config: dict[str, Any],
    scenes: list[dict[str, Any]],
    executable_boundary: dict[str, Any] | None = None,
    environment_manifest_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one experiment ID to its immutable code/config/input identities."""

    source_config = Path(str(config["_config_path"]))
    boundary = executable_boundary or attest_densify_executable_boundary(config)
    lock: dict[str, Any] = {
        "schema_name": EXPERIMENT_LOCK_SCHEMA_NAME,
        "schema_version": EXPERIMENT_LOCK_SCHEMA_VERSION,
        "experiment_id": str(config.get("experiment_id", "dmap_development")),
        "source_config": file_identity(source_config),
        "git_commit": git_hash(),
        "executables": {
            "production": boundary["production"]["identity"],
            "observer": boundary["observer"]["identity"],
        },
        "runtime_boundaries": {
            "production": boundary["production"]["runtime_boundary"],
            "observer": boundary["observer"]["runtime_boundary"],
        },
        "executable_boundary": boundary,
        "environment_manifest": environment_manifest_identity,
        "scenes": [],
    }
    annotation_root = (
        config_path(config, "dataset_root") if config.get("dataset_root") else None
    )
    annotation_manifest = (
        annotation_root / "annotations" / "import_manifest.json"
        if annotation_root is not None else None
    )
    lock["annotation_snapshot"] = (
        file_identity(annotation_manifest) if annotation_manifest is not None else None
    )
    raw_snapshot_manifest = config.get("input_snapshot_manifest")
    lock["input_snapshot"] = (
        file_identity(config_path(config, "input_snapshot_manifest", Path(str(raw_snapshot_manifest))))
        if raw_snapshot_manifest else None
    )
    for scene in scenes:
        source_work = as_path(scene.get("working_folder"))
        source_mvs = as_path(scene.get("mvs_file"))
        scene_record: dict[str, Any] = {
            "scan_id": str(scene.get("scan_id", "")),
            "mvs_file": file_identity(source_mvs) if source_mvs is not None else None,
        }
        annotation_sidecar_raw = scene.get("annotation_sidecar")
        if annotation_sidecar_raw:
            annotation_sidecar = Path(str(annotation_sidecar_raw)).expanduser()
            if not annotation_sidecar.is_absolute():
                annotation_sidecar = source_config.parent / annotation_sidecar
            scene_record["annotation_sidecar"] = file_identity(
                annotation_sidecar.resolve()
            )
        if source_work is not None:
            scene_record["densify_config"] = file_identity(
                source_mvs.parent / "Densify.ini"
                if source_mvs is not None else source_work / "Densify.ini"
            )
            scene_record["staged_input_snapshot"] = configured_scene_input_snapshot(
                config, scene
            )
        lock["scenes"].append(scene_record)
    lock["scenes"].sort(key=lambda row: row["scan_id"])
    return lock


def write_immutable_json(path: Path, value: dict[str, Any], description: str) -> None:
    if path.is_file():
        existing = read_json(path)
        if existing != value:
            raise RuntimeError(
                f"{description} changed for existing experiment at {path}; "
                "use a new experiment_id"
            )
        return
    write_json(path, value)


def write_immutable_text(path: Path, value: str, description: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(
                f"{description} changed for existing experiment at {path}; "
                "use a new phase_id"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def persist_effective_storage_plan(
    root: Path,
    estimate: dict[str, Any],
) -> Path:
    """Persist the deterministic profile plan separately from live free-space data."""

    stable_plan = {
        "schema_name": "openmvs.dmap.effective_storage_plan",
        "schema_version": 1,
        "effective_capture_profiles": estimate["effective_capture_profiles"],
        "configured_capture_profiles": estimate["configured_capture_profiles"],
        "estimated_bytes": estimate["estimated_bytes"],
        "budget_bytes": estimate["budget_bytes"],
        "storage_model": estimate["storage_model"],
        "runs": estimate["runs"],
        "scenes": estimate["scenes"],
    }
    plan_id = stable_json_digest(stable_plan)
    stable_plan["plan_sha256"] = plan_id
    path = contained_output_path(
        root,
        "storage_plans",
        f"{plan_id}.json",
        description="effective storage plan",
    )
    write_immutable_json(path, stable_plan, "effective storage plan")
    return path


def build_capture_intent(
    config: dict[str, Any],
    root: Path,
    scenes: list[dict[str, Any]],
    profiles: Iterable[str],
    *,
    activation_source: str,
) -> dict[str, Any]:
    """Describe the complete capture matrix before the first subprocess starts."""

    profile_values = list(dict.fromkeys(str(value) for value in profiles))
    units: list[dict[str, Any]] = []
    for run in config.get("runs") or []:
        if run.get("existing"):
            continue
        label = validated_output_component(run.get("label"), "capture-intent run label")
        role = str(run.get("role", "variant"))
        repeats = int(run.get("repeats", 3 if role == "baseline" else 1))
        for repeat in range(repeats):
            for scene in scenes:
                scene_id = validated_output_component(
                    scene.get("scan_id"), "capture-intent scene_id"
                )
                for profile in profile_values:
                    units.append({
                        "configured_run": label,
                        "repeat": repeat,
                        "scene_id": scene_id,
                        "capture_profile": profile,
                    })
    lock_identity = file_identity(root / "00_experiment_lock.json")
    if lock_identity.get("exists") is not True:
        raise RuntimeError("capture intent requires a prepared experiment lock")
    intent: dict[str, Any] = {
        "schema_name": CAPTURE_INTENT_SCHEMA_NAME,
        "schema_version": CAPTURE_INTENT_SCHEMA_VERSION,
        "experiment_id": str(config.get("experiment_id", "dmap_development")),
        "activation_source": activation_source,
        "requested_profiles": profile_values,
        "experiment_lock": lock_identity,
        "units": units,
    }
    intent["intent_sha256"] = stable_json_digest(intent)
    return intent


def persist_capture_intent(root: Path, intent: dict[str, Any]) -> Path:
    digest = str(intent.get("intent_sha256", ""))
    unsigned = dict(intent)
    unsigned.pop("intent_sha256", None)
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest != stable_json_digest(unsigned):
        raise RuntimeError("capture intent digest is invalid")
    path = contained_output_path(
        root,
        "capture_intents",
        f"{digest}.json",
        description="immutable capture intent",
    )
    write_immutable_json(path, intent, "capture intent")
    return path


def load_capture_intents(root: Path) -> list[dict[str, Any]]:
    directory = root / "capture_intents"
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError("capture_intents must be a regular experiment directory")
    current_lock = file_identity(root / "00_experiment_lock.json")
    intents: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"capture intent is not a regular file: {path}")
        value = read_json(path)
        unsigned = dict(value)
        declared_digest = unsigned.pop("intent_sha256", None)
        if not (
            value.get("schema_name") == CAPTURE_INTENT_SCHEMA_NAME
            and value.get("schema_version") == CAPTURE_INTENT_SCHEMA_VERSION
            and isinstance(value.get("requested_profiles"), list)
            and isinstance(value.get("units"), list)
            and declared_digest == stable_json_digest(unsigned)
            and path.stem == declared_digest
            and value.get("experiment_lock") == current_lock
        ):
            raise RuntimeError(f"capture intent is malformed or stale: {path}")
        unknown = set(value["requested_profiles"]) - set(CAPTURE_PROFILE_ORDER)
        if unknown:
            raise RuntimeError(f"capture intent has unsupported profiles: {sorted(unknown)}")
        intents.append(value)
    return intents


def enforce_storage_admission(
    estimate: dict[str, Any],
    *,
    allow_over_budget: bool,
) -> None:
    if estimate["within_budget"] or allow_over_budget:
        return
    permitted_gib = min(
        estimate["available_bytes"], estimate["budget_bytes"]
    ) / 1024**3
    raise RuntimeError(
        f"estimated artifacts require {estimate['estimated_gib']:.1f} GiB, "
        f"available/budget permits {permitted_gib:.1f} GiB; "
        "raise instrumentation.max_artifact_gb or pass --allow-over-budget"
    )


def experiment_artifact_bytes(root: Path) -> int:
    """Count existing experiment artifacts once per physical regular file."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for relative in ("runs", "trace_reruns", "drilldowns/captures", "reports"):
        artifact_root = root / relative
        if not artifact_root.exists() or artifact_root.is_symlink():
            continue
        for directory, directory_names, file_names in os.walk(
            artifact_root, followlinks=False
        ):
            directory_path = Path(directory)
            directory_names[:] = [
                name for name in directory_names
                if not (directory_path / name).is_symlink()
            ]
            for name in file_names:
                path = directory_path / name
                try:
                    metadata = path.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    continue
                identity = (int(metadata.st_dev), int(metadata.st_ino))
                if identity in seen:
                    continue
                seen.add(identity)
                total += int(metadata.st_size)
    return total


def admit_on_demand_storage(
    config: dict[str, Any],
    root: Path,
    *,
    request_id: str,
    kind: str,
    run_scene_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    allow_over_budget: bool,
) -> dict[str, Any]:
    """Fail closed before an on-demand full-frame exact capture is launched."""

    estimates: list[dict[str, Any]] = []
    estimated_bytes = 0
    for run, scene in run_scene_pairs:
        estimate_config = dict(config)
        estimate_config["capture_profiles"] = ["deep"]
        estimate_config["runs"] = [{**run, "repeats": 1}]
        estimate_scene = {**scene, "expected_frames": 1}
        estimate = estimate_storage(
            estimate_config, [estimate_scene], requested_profiles=["deep"]
        )
        estimated_bytes += int(estimate["estimated_bytes"])
        estimates.append({
            "run": str(run.get("label", "")),
            "scene_id": str(scene.get("scan_id", "")),
            "estimated_bytes": int(estimate["estimated_bytes"]),
            "runs": estimate["runs"],
        })

    instrumentation = config.get("instrumentation") or {}
    budget_gb = float(instrumentation.get("max_artifact_gb", 0.0))
    budget_bytes = int(budget_gb * 1024**3) if budget_gb > 0 else None
    used_bytes = experiment_artifact_bytes(root)
    usage_root = root
    while not usage_root.exists():
        usage_root = usage_root.parent
    free_bytes = int(shutil.disk_usage(usage_root).free)
    within_free_space = estimated_bytes <= free_bytes
    within_experiment_budget = (
        budget_bytes is None or used_bytes + estimated_bytes <= budget_bytes
    )
    plan = {
        "schema_name": "openmvs.dmap.on_demand_storage_plan",
        "schema_version": 1,
        "request_sha256": request_id,
        "capture_kind": kind,
        "capture_profile": "trace" if kind == "drilldown_trace" else "deep",
        "estimated_bytes": estimated_bytes,
        "estimates": estimates,
        "budget_bytes": budget_bytes,
    }
    plan["plan_sha256"] = stable_json_digest(plan)
    admission_root = contained_output_path(
        root,
        "storage_admissions",
        validated_output_component(kind, "storage admission kind"),
        validated_output_component(request_id, "storage admission request id"),
        description="on-demand storage admission directory",
    )
    write_immutable_json(
        admission_root / "plan.json", plan, "on-demand storage plan"
    )
    receipt = {
        "schema_name": "openmvs.dmap.on_demand_storage_admission",
        "schema_version": 1,
        "request_sha256": request_id,
        "plan_sha256": plan["plan_sha256"],
        "capture_kind": kind,
        "estimated_bytes": estimated_bytes,
        "existing_artifact_bytes": used_bytes,
        "available_bytes": free_bytes,
        "budget_bytes": budget_bytes,
        "within_free_space": within_free_space,
        "within_experiment_budget": within_experiment_budget,
        "allow_over_budget": bool(allow_over_budget),
        "admitted": bool(
            allow_over_budget or (within_free_space and within_experiment_budget)
        ),
    }
    attempt_id = stable_json_digest(receipt)
    write_immutable_json(
        admission_root / f"receipt_{attempt_id}.json",
        receipt,
        "on-demand storage admission receipt",
    )
    if not receipt["admitted"]:
        raise RuntimeError(
            "on-demand trace storage admission failed: "
            f"request needs {estimated_bytes / 1024**3:.2f} GiB, "
            f"filesystem has {free_bytes / 1024**3:.2f} GiB free, and "
            f"experiment budget has "
            f"{('no explicit limit' if budget_bytes is None else f'{max(0, budget_bytes - used_bytes) / 1024**3:.2f} GiB remaining')}; "
            "pass --allow-over-budget to acknowledge the override"
        )
    return {"plan": plan, "receipt": receipt}


def experiment_phase_evidence_dir(root: Path, phase_id: str) -> Path:
    validated_phase_id = validated_output_component(phase_id, "experiment phase_id")
    return contained_output_path(
        root,
        "configs",
        "phase_evidence",
        validated_phase_id,
        description="experiment phase evidence directory",
    )


def validated_file_identity(
    expected: Any,
    *,
    description: str,
    required_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise RuntimeError(f"experiment phase has no {description} identity")
    raw_path = expected.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError(f"experiment phase has an invalid {description} path")
    path = Path(raw_path).expanduser().resolve()
    if required_path is not None and path != required_path.expanduser().resolve():
        raise RuntimeError(
            f"experiment phase {description} path is not canonical: {path}"
        )
    actual = file_identity(path)
    if actual != expected or actual.get("exists") is not True:
        raise RuntimeError(
            f"experiment phase {description} changed or is unavailable: {path}"
        )
    return actual


def prepare_experiment_phase(
    config: dict[str, Any],
    root: Path,
    allow_over_budget: bool,
    requested_profiles: Iterable[str] = (),
) -> tuple[dict[str, Any], Path]:
    """Prepare a generated phase without replacing the parent experiment lock."""

    phase = config.get("experiment_phase")
    if not isinstance(phase, dict):
        raise RuntimeError("experiment_phase must be a mapping")
    if (
        phase.get("schema_name") != EXPERIMENT_PHASE_SCHEMA_NAME
        or phase.get("schema_version") != EXPERIMENT_PHASE_SCHEMA_VERSION
    ):
        raise RuntimeError("experiment phase schema is unsupported")
    unsigned_phase = dict(phase)
    declared_lineage_digest = unsigned_phase.pop("lineage_sha256", None)
    if declared_lineage_digest != stable_json_digest(unsigned_phase):
        raise RuntimeError("experiment phase lineage digest is invalid")
    phase_id = phase.get("phase_id")
    if not isinstance(phase_id, str) or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", phase_id) is None:
        raise RuntimeError("experiment phase_id is invalid")
    experiment_id = str(config.get("experiment_id", "dmap_development"))
    if phase.get("parent_experiment_id") != experiment_id:
        raise RuntimeError("experiment phase parent_experiment_id does not match")

    config_path_value = Path(str(config["_config_path"])).expanduser().resolve()
    configs_root = (root / "configs").resolve()
    try:
        config_path_value.relative_to(configs_root)
    except ValueError as exc:
        raise RuntimeError(
            f"experiment phase config must be stored below {configs_root}"
        ) from exc

    parent = phase.get("parent")
    if not isinstance(parent, dict):
        raise RuntimeError("experiment phase parent lineage is missing")
    parent_config_identity = validated_file_identity(
        parent.get("source_config"), description="parent source config"
    )
    parent_lock_path = root / "00_experiment_lock.json"
    parent_resolved_path = root / "00_resolved_experiment.yaml"
    parent_suite_path = root / "00_suite.json"
    parent_lock_identity = validated_file_identity(
        parent.get("experiment_lock"),
        description="parent experiment lock",
        required_path=parent_lock_path,
    )
    parent_resolved_identity = validated_file_identity(
        parent.get("resolved_experiment"),
        description="parent resolved experiment",
        required_path=parent_resolved_path,
    )
    parent_suite_identity = validated_file_identity(
        parent.get("suite"),
        description="parent suite",
        required_path=parent_suite_path,
    )
    selection_identity = validated_file_identity(
        phase.get("selection_manifest"), description="selection manifest"
    )

    parent_config = load_config(Path(str(parent_config_identity["path"])))
    if str(parent_config.get("experiment_id", "")) != experiment_id:
        raise RuntimeError("parent source config experiment_id changed")
    if experiment_root(parent_config) != root:
        raise RuntimeError("parent source config resolves to another experiment root")
    parent_suite_ids = resolve_suite(parent_config)
    parent_scenes = resolve_scenes(parent_config, parent_suite_ids)
    stored_parent_lock = read_json(parent_lock_path)
    parent_environment_identity = validated_file_identity(
        stored_parent_lock.get("environment_manifest"),
        description="parent environment manifest",
        required_path=root / ENVIRONMENT_MANIFEST_FILE,
    )
    validate_environment_manifest(
        read_json(Path(str(parent_environment_identity["path"])))
    )
    current_parent_lock = build_experiment_lock(
        parent_config,
        parent_scenes,
        environment_manifest_identity=parent_environment_identity,
    )
    if stored_parent_lock != current_parent_lock:
        raise RuntimeError(
            "parent config, executable, or frozen input identity changed for "
            f"experiment phase {phase_id}"
        )
    if stored_parent_lock.get("source_config") != parent_config_identity:
        raise RuntimeError("parent experiment lock does not bind the declared source config")
    expected_parent_resolved = dict(parent_config)
    expected_parent_resolved.pop("_config_path", None)
    expected_parent_resolved["resolved_scan_ids"] = parent_suite_ids
    expected_parent_resolved["resolved_scenes"] = parent_scenes
    expected_parent_resolved["source_config"] = str(
        Path(str(parent_config["_config_path"])).resolve()
    )
    if parent_resolved_path.read_text(encoding="utf-8") != yaml.safe_dump(
        expected_parent_resolved, sort_keys=False
    ):
        raise RuntimeError("parent resolved experiment does not match its source config")
    expected_parent_suite = {
        "name": (parent_config.get("suite") or {}).get("name", "smoke"),
        "scan_ids": parent_suite_ids,
    }
    if read_json(parent_suite_path) != expected_parent_suite:
        raise RuntimeError("parent suite does not match its source config")

    resolution = (config.get("manual_confirmation") or {}).get("resolution") or {}
    selection_path = Path(str(resolution.get("selection_manifest") or "")).expanduser().resolve()
    if selection_path != Path(str(selection_identity["path"])).resolve():
        raise RuntimeError("experiment phase selection path does not match confirmation resolution")
    if resolution.get("selection_manifest_sha256") != selection_identity.get("sha256"):
        raise RuntimeError(
            "experiment phase selection digest does not match confirmation resolution"
        )
    selection = read_json(selection_path)
    if (
        selection.get("valid") is not True
        or not resolution.get("selected_run")
        or selection.get("selected_run") != resolution.get("selected_run")
    ):
        raise RuntimeError("experiment phase selection does not match confirmation resolution")

    suite_ids = resolve_suite(config)
    scenes = resolve_scenes(config, suite_ids)
    stored_scene_records = {
        str(row["scan_id"]): row
        for row in stored_parent_lock.get("scenes") or []
        if isinstance(row, dict) and row.get("scan_id")
    }
    for scene in scenes:
        scene_id = str(scene["scan_id"])
        scene_record = stored_scene_records.get(scene_id)
        if scene_record is None:
            raise RuntimeError(
                f"parent experiment lock has no frozen input for scene {scene_id!r}"
            )
        if configured_scene_input_snapshot(config, scene) != scene_record.get(
            "staged_input_snapshot"
        ):
            raise RuntimeError(
                f"generated phase scene input changed for {scene_id!r}"
            )
        prepare_frozen_scene_input(config, root, scene, scene_record)
    evidence_dir = experiment_phase_evidence_dir(root, phase_id)
    phase_lock = {
        "schema_name": EXPERIMENT_PHASE_LOCK_SCHEMA_NAME,
        "schema_version": EXPERIMENT_PHASE_LOCK_SCHEMA_VERSION,
        "phase_id": phase_id,
        "experiment_id": experiment_id,
        "phase_config": file_identity(config_path_value),
        "parent": {
            "source_config": parent_config_identity,
            "experiment_lock": parent_lock_identity,
            "resolved_experiment": parent_resolved_identity,
            "suite": parent_suite_identity,
        },
        "selection_manifest": selection_identity,
    }
    resolved = dict(config)
    resolved.pop("_config_path", None)
    resolved["resolved_scan_ids"] = suite_ids
    resolved["resolved_scenes"] = scenes
    resolved["source_config"] = str(config_path_value)
    resolved["parent_experiment_root"] = str(root)
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    suite = {
        "name": (config.get("suite") or {}).get("name", "smoke"),
        "scan_ids": suite_ids,
    }
    estimate = estimate_storage(config, scenes, requested_profiles)
    enforce_storage_admission(estimate, allow_over_budget=allow_over_budget)
    write_immutable_json(
        evidence_dir / "00_phase_lock.json",
        phase_lock,
        "generated experiment phase identity",
    )
    write_immutable_text(
        evidence_dir / "01_resolved_phase.yaml",
        resolved_text,
        "resolved experiment phase",
    )
    write_immutable_json(
        evidence_dir / "02_phase_suite.json",
        suite,
        "resolved experiment phase suite",
    )
    write_json(evidence_dir / "03_storage_estimate.json", estimate)
    persist_effective_storage_plan(root, estimate)
    return config, root


def prepare_experiment(
    config_path_value: Path,
    allow_over_budget: bool,
    requested_profiles: Iterable[str] = (),
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path_value)
    root = experiment_root(config)
    if "experiment_phase" in config:
        return prepare_experiment_phase(
            config, root, allow_over_budget, requested_profiles
        )
    suite_ids = resolve_suite(config)
    scenes = resolve_scenes(config, suite_ids)
    executable_boundary = attest_densify_executable_boundary(config)
    environment_manifest = build_environment_manifest(executable_boundary)
    validate_environment_manifest(environment_manifest)
    root.mkdir(parents=True, exist_ok=True)
    environment_manifest_path = root / ENVIRONMENT_MANIFEST_FILE
    write_immutable_json(
        environment_manifest_path,
        environment_manifest,
        "resolved execution environment",
    )
    experiment_lock = build_experiment_lock(
        config,
        scenes,
        executable_boundary,
        file_identity(environment_manifest_path),
    )
    resolved = dict(config)
    resolved.pop("_config_path", None)
    resolved["resolved_scan_ids"] = suite_ids
    resolved["resolved_scenes"] = scenes
    resolved["source_config"] = str(Path(config["_config_path"]).resolve())
    resolved_text = yaml.safe_dump(resolved, sort_keys=False)
    resolved_path = root / "00_resolved_experiment.yaml"
    if resolved_path.is_file() and resolved_path.read_text(encoding="utf-8") != resolved_text:
        raise RuntimeError(
            f"resolved experiment changed at {resolved_path}; use a new experiment_id"
        )
    suite = {
        "name": (config.get("suite") or {}).get("name", "smoke"),
        "scan_ids": suite_ids,
    }
    estimate = estimate_storage(config, scenes, requested_profiles)
    enforce_storage_admission(estimate, allow_over_budget=allow_over_budget)
    lock_scenes = {
        str(row["scan_id"]): row for row in experiment_lock.get("scenes") or []
    }
    for scene in scenes:
        prepare_frozen_scene_input(
            config, root, scene, lock_scenes[str(scene["scan_id"])]
        )
    write_immutable_json(
        root / "00_experiment_lock.json",
        experiment_lock,
        "config, executable, or frozen input identity",
    )
    if not resolved_path.is_file():
        resolved_path.write_text(resolved_text, encoding="utf-8")
    write_immutable_json(root / "00_suite.json", suite, "resolved suite")
    write_json(root / "00_storage_estimate.json", estimate)
    persist_effective_storage_plan(root, estimate)
    return config, root


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def prepare_working_folder(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(
        source,
        destination,
        copy_function=hardlink_or_copy,
        ignore=shutil.ignore_patterns(*STAGED_INPUT_EXCLUDE_PATTERNS),
    )
    # Frozen benchmark directories are intentionally read-only. The staged tree
    # must admit new DMAPs without changing the immutable source permissions.
    for directory in [destination, *(path for path in destination.rglob("*") if path.is_dir())]:
        directory.chmod(directory.stat().st_mode | stat.S_IWUSR)


def staged_mvs_input_path(source_work: Path, source_mvs: Path, work_dir: Path) -> Path:
    """Derive the relocatable staged MVS path without writing the workspace."""

    try:
        relative_mvs = source_mvs.resolve().relative_to(source_work.resolve())
    except ValueError:
        relative_mvs = Path(source_mvs.name)
    return work_dir / relative_mvs


def stage_mvs_input(source_work: Path, source_mvs: Path, work_dir: Path) -> Path:
    """Return the staged MVS path without flattening relocatable scene layouts."""

    local_mvs = staged_mvs_input_path(source_work, source_mvs, work_dir)
    if not local_mvs.is_file():
        local_mvs.parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_copy(str(source_mvs), str(local_mvs))
    return local_mvs


def dmap_working_folder(local_mvs: Path) -> Path:
    """Use the scene directory so archived ../sfm image paths remain valid."""

    return local_mvs.parent


def prepare_locked_profile_workspace(
    config: dict[str, Any], root: Path, scene: dict[str, Any], work_dir: Path
) -> Path:
    """Stage one profile from the experiment-owned frozen input and revalidate it."""

    scene_id = validated_output_component(scene.get("scan_id"), "scene scan_id")
    source_work = Path(str(scene["working_folder"])).expanduser().resolve()
    source_mvs = Path(str(scene["mvs_file"])).expanduser().resolve()
    relative_mvs = scene_mvs_relative_path(source_work, source_mvs)
    record = locked_scene_record(root, scene_id)
    expected = record.get("staged_input_snapshot")
    if not isinstance(expected, dict):
        raise RuntimeError(
            f"experiment lock has no staged input snapshot for scene {scene_id!r}"
        )
    validate_scene_input_snapshot(config, root, scene)
    frozen_work = frozen_scene_work_dir(root, scene_id)
    validate_staged_scene_input(
        frozen_work,
        expected,
        record["mvs_file"],
        relative_mvs,
        description=f"frozen scene {scene_id!r}",
    )
    prepare_working_folder(frozen_work, work_dir)
    validate_staged_scene_input(
        work_dir,
        expected,
        record["mvs_file"],
        relative_mvs,
        description=f"profile workspace {scene_id!r}",
    )
    return work_dir / relative_mvs


def git_hash() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def execute_command(command: list[str], cwd: Path, output_dir: Path, dry_run: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n", encoding="utf-8"
    )
    started_ns = time.monotonic_ns()
    return_code = 0
    if not dry_run:
        with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / "stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, text=True)
            return_code = process.returncode
    result = {
        "command": command,
        "cwd": str(cwd),
        "return_code": return_code,
        "elapsed_seconds": (time.monotonic_ns() - started_ns) / 1_000_000_000.0,
        "elapsed_clock": "time.monotonic_ns",
        "dry_run": dry_run,
        "git_commit": git_hash(),
    }
    write_json(output_dir / "repro.json", result)
    return result


def locked_runtime_boundary(root: Path, role: str) -> dict[str, Any]:
    """Return the immutable runtime boundary for one executable role."""

    if role not in {"production", "observer"}:
        raise ValueError(f"unsupported runtime-boundary role: {role!r}")
    lock = read_json(root / "00_experiment_lock.json")
    if (
        lock.get("schema_name") != EXPERIMENT_LOCK_SCHEMA_NAME
        or int(lock.get("schema_version", 0)) < EXPERIMENT_LOCK_SCHEMA_VERSION
    ):
        raise RuntimeError(
            "runtime-boundary receipts require an experiment lock at schema version "
            f"{EXPERIMENT_LOCK_SCHEMA_VERSION}"
        )
    boundary = (lock.get("runtime_boundaries") or {}).get(role)
    validate_runtime_boundary_shape(boundary)
    return boundary


def _runtime_boundary_receipt(
    *,
    role: str,
    executable: Path,
    expected: dict[str, Any],
    pre_run: dict[str, Any],
    post_run: dict[str, Any] | None,
    post_run_error: str | None = None,
) -> dict[str, Any]:
    valid = post_run is not None and expected == pre_run == post_run
    if expected != pre_run:
        reason = "runtime boundary differed from the experiment lock before launch"
    elif post_run_error:
        reason = f"runtime boundary could not be read after launch: {post_run_error}"
    elif post_run != expected:
        reason = "runtime boundary changed while the subprocess was running"
    else:
        reason = "locked executable and build-local shared libraries matched before and after launch"
    receipt: dict[str, Any] = {
        "schema_name": RUNTIME_BOUNDARY_RECEIPT_SCHEMA_NAME,
        "schema_version": RUNTIME_BOUNDARY_RECEIPT_SCHEMA_VERSION,
        "role": role,
        "command_executable": str(executable),
        "expected_boundary_sha256": expected["boundary_sha256"],
        "pre_run": pre_run,
        "post_run": post_run,
        "valid": valid,
        "reason": reason,
    }
    receipt["receipt_sha256"] = stable_json_digest(receipt)
    return receipt


def execute_densify_command(
    command: list[str],
    cwd: Path,
    output_dir: Path,
    dry_run: bool,
    *,
    experiment_root: Path,
    runtime_role: Literal["production", "observer"],
) -> dict[str, Any]:
    """Run DensifyPointCloud with a capture-time runtime-boundary receipt.

    The lock is checked immediately before launch and again immediately after
    return. This closes the ordinary concurrent rebuild/relink window while
    retaining the original executable layout and loader behavior.
    """

    if dry_run:
        return execute_command(command, cwd, output_dir, True)
    if not command:
        raise ValueError("DensifyPointCloud command is empty")
    expected = locked_runtime_boundary(experiment_root, runtime_role)
    executable = Path(command[0]).expanduser().resolve(strict=True)
    locked_executable = Path(str(expected["executable"]["path"])).resolve(strict=True)
    if executable != locked_executable:
        raise RuntimeError(
            f"{runtime_role} command executable differs from the experiment lock: "
            f"{executable} != {locked_executable}"
        )
    pre_run = runtime_boundary_identity(executable)
    if pre_run != expected:
        raise RuntimeError(
            f"{runtime_role} runtime boundary changed before capture launch; "
            "use a new experiment_id after rebuilding"
        )

    execution_error: BaseException | None = None
    result: dict[str, Any]
    try:
        result = execute_command(command, cwd, output_dir, False)
    except BaseException as exc:
        execution_error = exc
        result = {
            "command": command,
            "cwd": str(cwd),
            "return_code": None,
            "elapsed_seconds": None,
            "elapsed_clock": "time.monotonic_ns",
            "dry_run": False,
            "git_commit": git_hash(),
            "execution_error": f"{type(exc).__name__}: {exc}",
        }

    post_run: dict[str, Any] | None = None
    post_run_error: str | None = None
    try:
        post_run = runtime_boundary_identity(executable)
    except Exception as exc:
        post_run_error = f"{type(exc).__name__}: {exc}"
    receipt = _runtime_boundary_receipt(
        role=runtime_role,
        executable=executable,
        expected=expected,
        pre_run=pre_run,
        post_run=post_run,
        post_run_error=post_run_error,
    )
    result["runtime_boundary_receipt"] = receipt
    write_json(output_dir / "repro.json", result)
    if not receipt["valid"]:
        raise RuntimeError(receipt["reason"])
    if execution_error is not None:
        raise execution_error
    return result


def _experiment_lock_for_capture(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    for directory in (run_dir, *run_dir.parents):
        lock_path = directory / "00_experiment_lock.json"
        if lock_path.is_file() and not lock_path.is_symlink():
            return directory, read_json(lock_path)
    return None


def validate_runtime_boundary_receipt(
    run_dir: Path, repro: dict[str, Any], mode: str
) -> tuple[bool, str]:
    """Require receipts for captures governed by a schema-v4 experiment lock."""

    locked = _experiment_lock_for_capture(run_dir)
    if locked is None:
        return True, "legacy/imported capture has no schema-v4 experiment lock"
    _root, lock = locked
    if lock.get("schema_name") != EXPERIMENT_LOCK_SCHEMA_NAME:
        return False, "capture ancestor has an unsupported experiment lock"
    lock_version = int(lock.get("schema_version", 0))
    if lock_version < EXPERIMENT_LOCK_SCHEMA_VERSION:
        return True, "legacy capture predates runtime-boundary receipts"
    role = "production" if mode == "endpoint" else "observer"
    expected = (lock.get("runtime_boundaries") or {}).get(role)
    try:
        validate_runtime_boundary_shape(expected)
    except RuntimeError as exc:
        return False, f"locked {role} runtime boundary is invalid: {exc}"
    receipt = repro.get("runtime_boundary_receipt")
    if not isinstance(receipt, dict):
        return False, "schema-v4 capture has no runtime-boundary receipt"
    unsigned = dict(receipt)
    declared_digest = unsigned.pop("receipt_sha256", None)
    if not (
        receipt.get("schema_name") == RUNTIME_BOUNDARY_RECEIPT_SCHEMA_NAME
        and receipt.get("schema_version") == RUNTIME_BOUNDARY_RECEIPT_SCHEMA_VERSION
        and declared_digest == stable_json_digest(unsigned)
        and receipt.get("role") == role
        and receipt.get("expected_boundary_sha256") == expected.get("boundary_sha256")
        and receipt.get("pre_run") == expected
        and receipt.get("post_run") == expected
        and receipt.get("valid") is True
    ):
        return False, "runtime-boundary receipt is malformed or does not match the experiment lock"
    return True, f"validated {role} runtime boundary before and after launch"


def validate_prefilter_frame(frame_dir: Path) -> tuple[bool, str]:
    """Validate the bounded Process<false> prefilter snapshot contract."""

    frame_dir = _lexical_absolute_path(frame_dir)
    try:
        _require_non_symlink_path(
            frame_dir, description="prefilter frame directory", final_kind="directory"
        )
    except ValueError as exc:
        return False, str(exc)
    summary_path = frame_dir / "summary.json"
    manifest_path = frame_dir / "prefilter_manifest.json"
    completion_path = frame_dir / "prefilter_capture_complete.json"
    try:
        owned_regular_artifact_path(
            frame_dir, summary_path, description="prefilter frame summary"
        )
        for artifact, description in (
            (manifest_path, "prefilter manifest"),
            (completion_path, "prefilter completion marker"),
        ):
            if artifact.exists() or artifact.is_symlink():
                owned_regular_artifact_path(
                    frame_dir, artifact, description=description
                )
    except ValueError as exc:
        return False, str(exc)
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    if summary.get("schema_name") != "openmvs.dmap.frame_summary":
        return False, "prefilter frame summary is missing or malformed"
    summary_completion_path = frame_dir / "summary_complete.json"
    if summary_completion_path.exists() or summary_completion_path.is_symlink():
        try:
            owned_regular_artifact_path(
                frame_dir,
                summary_completion_path,
                description="prefilter summary completion marker",
            )
        except ValueError as exc:
            return False, str(exc)
    summary_completion = read_json(summary_completion_path)
    if (
        not manifest_path.is_file()
        and not completion_path.is_file()
        and summary_completion.get("schema_name") == "openmvs.dmap.summary_complete"
        and summary_completion.get("summary_complete") is True
    ):
        return False, (
            "requested prefilter capture emitted summary-only evidence; "
            "prefilter manifest and completion marker are absent"
        )
    if (
        manifest.get("schema_name") != "openmvs.dmap.prefilter_manifest"
        or manifest.get("schema_version") != 1
        or manifest.get("complete") is not True
    ):
        return False, "prefilter manifest is missing, incomplete, or unsupported"
    if (
        completion.get("schema_name")
        != "openmvs.dmap.prefilter_capture_complete"
        or completion.get("schema_version") != 1
        or completion.get("capture_kind") != "prefilter"
        or completion.get("eligible") is not True
        or completion.get("prefilter_complete") is not True
        or completion.get("maps_complete") is not True
        or completion.get("observer_sidecars_complete") is not True
    ):
        return False, "prefilter completion marker is missing or ineligible"
    expected_completion = {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "path": "prefilter_capture_complete.json",
        "maps_complete": True,
        "prefilter_complete": True,
        "eligible": True,
    }
    if summary.get("completion_marker") != expected_completion:
        return False, "prefilter summary completion reference is inconsistent"
    if manifest.get("width") != summary.get("width") or manifest.get(
        "height"
    ) != summary.get("height"):
        return False, "prefilter manifest geometry differs from frame summary"
    if manifest.get("process_specialization") != "Process<false>":
        return False, "prefilter capture did not retain Process<false>"
    maps = manifest.get("maps")
    if not isinstance(maps, list) or len(maps) != 1:
        return False, "prefilter manifest must contain exactly one map"
    record = maps[0]
    if (
        not isinstance(record, dict)
        or record.get("signal") != "depth_final_before_filter"
        or record.get("path") != "maps/depth_final_before_filter.pfm"
        or record.get("measurement_quality", record.get("quality")) != "exact"
        or record.get("measurement_basis", record.get("basis"))
        != "production_pre_filter_snapshot"
    ):
        return False, "prefilter depth-map record is malformed"
    try:
        depth_path = owned_regular_artifact_path(
            frame_dir,
            frame_dir / str(record["path"]),
            description="prefilter depth map",
        )
    except ValueError as exc:
        return False, str(exc)
    if record.get("bytes") != depth_path.stat().st_size:
        return False, "prefilter depth map size differs from its manifest"
    try:
        depth = read_pfm(depth_path)
    except Exception as exc:
        return False, f"prefilter depth map cannot be decoded: {exc}"
    expected_shape = (int(summary["height"]), int(summary["width"]))
    if depth.shape[:2] != expected_shape or depth.ndim != 2:
        return False, "prefilter depth map shape differs from its manifest"
    if not np.isfinite(depth).all():
        return False, "prefilter depth map contains non-finite values"
    manifest_binding = completion.get("manifest") or {}
    summary_binding = completion.get("summary") or {}
    if manifest_binding.get("path") != manifest_path.name:
        return False, "prefilter completion marker references another manifest"
    if summary_binding.get("path") != summary_path.name:
        return False, "prefilter completion marker references another summary"
    if (
        manifest_binding.get("schema_version") != 1
        or manifest_binding.get("bytes") != manifest_path.stat().st_size
        or summary_binding.get("schema_version") != 4
        or summary_binding.get("bytes") != summary_path.stat().st_size
    ):
        return False, "prefilter completion marker artifact bindings are inconsistent"
    for binding, path, label in (
        (manifest_binding, manifest_path, "manifest"),
        (summary_binding, summary_path, "summary"),
    ):
        digest = binding.get("sha256")
        if digest is not None and digest != dmap_drilldown.file_digest(path):
            return False, f"prefilter {label} digest binding is invalid"
    return True, "validated bounded prefilter depth snapshot"


def validate_coarse_compatibility_update_source_map(
    stage_root: Path,
    plan: dict[str, Any],
    image_id: int,
    pyramid_level: int,
) -> tuple[bool, str]:
    """Validate the compatibility PNG promised by one coarse resource plan."""

    path = (
        stage_root
        / "instrumentation"
        / "maps"
        / f"depth{image_id:04d}_scale{pyramid_level:02d}_update_source.png"
    )
    try:
        path = owned_regular_artifact_path(
            stage_root,
            path,
            description="coarse compatibility update-source map",
        )
    except ValueError as exc:
        return False, str(exc)
    try:
        with Image.open(path) as image:
            image_format = image.format
            image_mode = image.mode
            image_size = image.size
            image.load()
            values = np.asarray(image)
    except (OSError, ValueError) as exc:
        return False, f"coarse compatibility update-source map cannot be decoded: {exc}"
    expected_width = _nonnegative_integer(plan.get("width"))
    expected_height = _nonnegative_integer(plan.get("height"))
    if expected_width in (None, 0) or expected_height in (None, 0):
        return False, "coarse resource plan has invalid compatibility-map geometry"
    if image_format != "PNG" or image_mode != "L" or values.dtype != np.uint8:
        return False, "coarse compatibility update-source map is not an 8-bit grayscale PNG"
    if image_size != (expected_width, expected_height):
        return False, (
            "coarse compatibility update-source map geometry differs from its "
            "resource plan"
        )
    return True, "validated coarse compatibility update-source map"


@dataclass(frozen=True)
class CaptureStageTopology:
    photometric_iterations: int
    geometric_iterations: int
    sub_resolution_levels: int
    fusion_mode: int

    @property
    def geometric_stage_indices(self) -> tuple[int, ...]:
        if self.fusion_mode < 0:
            return ()
        return tuple(range(self.geometric_iterations))


def capture_stage_topology(command: Any) -> CaptureStageTopology:
    """Resolve the CUDA PatchMatch topology from one recorded launch command."""

    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
    ):
        raise ValueError("repro.json does not contain a resolved DensifyPointCloud command")
    topology = CaptureStageTopology(
        photometric_iterations=parse_argument(command, "--iters", 4),
        geometric_iterations=parse_argument(command, "--geometric-iters", 2),
        sub_resolution_levels=parse_argument(command, "--sub-resolution-levels", 2),
        fusion_mode=parse_argument(command, "--fusion-mode", 0),
    )
    if topology.photometric_iterations < 0:
        raise ValueError("resolved --iters must be non-negative")
    if topology.geometric_iterations < 0:
        raise ValueError("resolved --geometric-iters must be non-negative")
    if topology.sub_resolution_levels < 0:
        raise ValueError("resolved --sub-resolution-levels must be non-negative")
    return topology


def validate_instrumentation_capture_topology(
    instrumentation_dir: Path,
    command: Any,
    *,
    require_timings: bool = False,
) -> tuple[bool, str]:
    """Fail closed when a capture omits a command-declared stage or level."""

    try:
        topology = capture_stage_topology(command)
    except ValueError as exc:
        return False, str(exc)

    geometric_root = instrumentation_dir / "geometric_iterations"
    observed_geometric: dict[int, Path] = {}
    if geometric_root.exists() or geometric_root.is_symlink():
        if geometric_root.is_symlink() or not geometric_root.is_dir():
            return False, "geometric stage root is not a regular directory"
        for candidate in geometric_root.iterdir():
            if candidate.is_symlink() or not candidate.is_dir():
                return False, f"unexpected geometric stage artifact: {candidate.name}"
            match = re.fullmatch(r"iteration(\d+)", candidate.name)
            if match is None:
                return False, f"unexpected geometric stage directory: {candidate.name}"
            stage_index = int(match.group(1))
            if stage_index in observed_geometric:
                return False, f"duplicate geometric stage index: {stage_index}"
            observed_geometric[stage_index] = candidate
    expected_geometric = set(topology.geometric_stage_indices)
    if set(observed_geometric) != expected_geometric:
        return False, (
            "geometric stage topology does not match the resolved command: "
            f"expected {sorted(expected_geometric)}, observed "
            f"{sorted(observed_geometric)}"
        )

    stage_specs: list[tuple[str, int | None, Path, set[int], int]] = [(
        "photometric",
        None,
        instrumentation_dir,
        set(range(topology.sub_resolution_levels + 1)),
        topology.photometric_iterations + 1,
    )]
    stage_specs.extend(
        (
            "geometric_consistency",
            stage_index,
            observed_geometric[stage_index],
            {0},
            2,
        )
        for stage_index in topology.geometric_stage_indices
    )

    reference_image_ids: set[int] | None = None
    for estimation_stage, geometric_iteration, stage_root, expected_levels, expected_states in stage_specs:
        stage_label = (
            "photometric"
            if geometric_iteration is None
            else f"geometric iteration {geometric_iteration}"
        )
        metadata = read_json(stage_root / "run_metadata.json")
        scene_summary = read_json(stage_root / "scene_summary.json")
        for document_name, expected_schema, document in (
            ("run_metadata.json", "openmvs.dmap.run", metadata),
            (
                "scene_summary.json",
                "openmvs.dmap.scene_summary",
                scene_summary,
            ),
        ):
            if document.get("schema_name") != expected_schema:
                return False, f"{stage_label} {document_name} is missing or malformed"
            if document.get("estimation_stage") != estimation_stage:
                return False, f"{stage_label} {document_name} has the wrong stage identity"
            recorded_iteration = document.get("geometric_iteration")
            if geometric_iteration is None:
                if recorded_iteration is not None:
                    return False, f"{stage_label} {document_name} has the wrong stage identity"
            elif recorded_iteration != geometric_iteration:
                return False, f"{stage_label} {document_name} has the wrong stage identity"

        frame_paths = sorted((stage_root / "depthmaps").glob("*/summary.json"))
        image_ids: list[int] = []
        for summary_path in frame_paths:
            summary = read_json(summary_path)
            image_id = summary.get("image_id")
            if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
                return False, f"{stage_label} contains a malformed frame summary"
            image_ids.append(image_id)
        if not image_ids:
            return False, f"{stage_label} stage has no instrumented frames"
        if len(set(image_ids)) != len(image_ids):
            return False, f"{stage_label} contains duplicate frame image IDs"
        stage_image_ids = set(image_ids)
        if reference_image_ids is None:
            reference_image_ids = stage_image_ids
        elif stage_image_ids != reference_image_ids:
            return False, (
                f"{stage_label} frame set differs from the photometric frame set"
            )

        plans_path = stage_root / "resource_plans.jsonl"
        try:
            plans = read_jsonl(plans_path)
        except (OSError, ValueError) as exc:
            return False, f"{stage_label} resource plans are unreadable: {exc}"
        observed_plan_keys: list[tuple[int, int]] = []
        for plan in plans:
            image_id = plan.get("image_id")
            level = plan.get("pyramid_level")
            logical_states = plan.get("num_logical_states")
            if (
                plan.get("schema_name") != "openmvs.dmap.resource_plan"
                or isinstance(image_id, bool)
                or not isinstance(image_id, int)
                or isinstance(level, bool)
                or not isinstance(level, int)
                or isinstance(logical_states, bool)
                or not isinstance(logical_states, int)
                or logical_states != expected_states
            ):
                return False, f"{stage_label} contains a malformed resource plan"
            observed_plan_keys.append((image_id, level))
        expected_plan_keys = {
            (image_id, level)
            for image_id in stage_image_ids
            for level in expected_levels
        }
        if len(observed_plan_keys) != len(set(observed_plan_keys)):
            return False, f"{stage_label} contains duplicate image/level resource plans"
        if set(observed_plan_keys) != expected_plan_keys:
            return False, (
                f"{stage_label} pyramid topology does not match the resolved command: "
                f"expected {sorted(expected_plan_keys)}, observed "
                f"{sorted(observed_plan_keys)}"
            )

        if require_timings:
            timings_path = instrumentation_csv(stage_root, "timings.csv")
            if timings_path is None:
                return False, f"{stage_label} stage has no timing rows"
            try:
                with timings_path.open(encoding="utf-8") as handle:
                    timing_rows = list(csv.DictReader(handle))
                timing_passes: dict[tuple[int, int], list[int]] = {}
                for row in timing_rows:
                    key = (int(row["image_id"]), int(row["scale_number"]))
                    timing_passes.setdefault(key, []).append(int(row["pass_index"]))
            except (KeyError, OSError, TypeError, ValueError) as exc:
                return False, f"{stage_label} timing rows are malformed: {exc}"
            if set(timing_passes) != expected_plan_keys:
                return False, (
                    f"{stage_label} timing pyramid topology does not match the resolved "
                    f"command: expected {sorted(expected_plan_keys)}, observed "
                    f"{sorted(timing_passes)}"
                )
            expected_passes = set(range(1 + 2 * (expected_states - 1)))
            for key, pass_indices in timing_passes.items():
                if (
                    len(pass_indices) != len(expected_passes)
                    or set(pass_indices) != expected_passes
                ):
                    return False, (
                        f"{stage_label} timing pass topology is incomplete for "
                        f"image/level {key}: expected {sorted(expected_passes)}, "
                        f"observed {sorted(pass_indices)}"
                    )

    return True, (
        f"validated {len(stage_specs)} command-declared stage(s), "
        f"photometric pyramid levels 0..{topology.sub_resolution_levels}"
    )


def validate_completed_run_mode(run_dir: Path, mode: str) -> tuple[bool, str]:
    def close_validation(reason: str) -> tuple[bool, str]:
        closure = integrity.validate_capture_artifact_closure(run_dir)
        if not closure.valid:
            return False, f"capture artifact closure failed: {closure.reason}"
        return True, f"{reason}; artifact closure {closure.status}: {closure.reason}"

    repro_path = run_dir / "repro.json"
    command_path = run_dir / "command.sh"
    instrumentation_dir = run_dir / "dmap_instrumentation"
    if not repro_path.is_file() or not command_path.is_file():
        return False, "missing command.sh or repro.json"
    repro = read_json(repro_path)
    return_code = repro.get("return_code")
    if bool(repro.get("dry_run")) or return_code is None or int(return_code) != 0:
        return False, "run did not complete successfully"
    runtime_valid, runtime_reason = validate_runtime_boundary_receipt(
        run_dir, repro, mode
    )
    if not runtime_valid:
        return False, runtime_reason
    if mode == "endpoint":
        forbidden = run_dir / "dmap_instrumentation"
        depth_maps = sorted((run_dir / "depth_maps").glob("depth*.dmap"))
        if forbidden.exists():
            return False, "production endpoint unexpectedly contains instrumentation output"
        if not depth_maps:
            return False, "production endpoint has no DMAP outputs"
        metadata = read_json(run_dir / "endpoint_metadata.json")
        if metadata.get("schema_name") != "openmvs.dmap.endpoint_capture":
            return False, "endpoint_metadata.json is missing or malformed"
        return close_validation(
            f"validated production endpoint with {len(depth_maps)} DMAPs"
        )
    run_metadata = read_json(instrumentation_dir / "run_metadata.json")
    scene_summary = read_json(instrumentation_dir / "scene_summary.json")
    if run_metadata.get("schema_name") != "openmvs.dmap.run":
        return False, "run_metadata.json is missing or malformed"
    if scene_summary.get("schema_name") != "openmvs.dmap.scene_summary":
        return False, "scene_summary.json is missing or malformed"
    topology_valid, topology_reason = validate_instrumentation_capture_topology(
        instrumentation_dir,
        repro.get("command"),
        require_timings=mode == "timing",
    )
    if not topology_valid:
        return False, topology_reason
    top_level_depthmap_dirs = sorted(
        path for path in (instrumentation_dir / "depthmaps").glob("*") if path.is_dir()
    )
    if not top_level_depthmap_dirs:
        return False, "no instrumented depth-map frames"
    if mode == "timing":
        depth_maps = sorted((run_dir / "depth_maps").glob("depth*.dmap"))
        if not depth_maps:
            return False, "timing capture has no DMAP outputs"
        return close_validation(
            f"validated timing capture with {len(depth_maps)} DMAPs; {topology_reason}"
        )
    if mode == "prefilter":
        stage_roots = instrumentation_stage_roots(instrumentation_dir)
        depth_map_dir = run_dir / "depth_maps"
        validated_frames = 0
        for stage_index, (estimation_stage, _geometric_iteration, stage_root) in enumerate(
            stage_roots
        ):
            terminal_stage = stage_index == len(stage_roots) - 1
            plans_by_image: dict[int, list[dict[str, Any]]] = {}
            plans_path = stage_root / "resource_plans.jsonl"
            for plan in read_jsonl(plans_path):
                plans_by_image.setdefault(int(plan.get("image_id", -1)), []).append(plan)
            frame_dirs = sorted(
                path
                for path in (stage_root / "depthmaps").glob("*")
                if path.is_dir()
            )
            if not frame_dirs:
                return False, f"{estimation_stage} stage has no prefilter frames"
            for frame_dir in frame_dirs:
                valid, reason = validate_prefilter_frame(frame_dir)
                if not valid:
                    return False, f"frame {frame_dir.name} failed validation: {reason}"
                summary = read_json(frame_dir / "summary.json")
                image_id = int(summary.get("image_id", -1))
                if terminal_stage and not (
                    depth_map_dir / f"depth{image_id:04d}.dmap"
                ).is_file():
                    return False, f"terminal prefilter frame {frame_dir.name} has no DMAP"
                fine_plans = [
                    plan
                    for plan in plans_by_image.get(image_id, [])
                    if plan.get("pyramid_level") == 0
                ]
                if len(fine_plans) != 1:
                    return False, f"frame {frame_dir.name} has no unique fine resource plan"
                plan = fine_plans[0]
                if (
                    plan.get("prefilter_requested") is not True
                    or plan.get("prefilter_available") is not True
                    or plan.get("maps_available") is not False
                    or plan.get("exact_available") is not False
                ):
                    return False, f"frame {frame_dir.name} has an invalid prefilter resource tier"
                context = {"image_id": image_id, "pyramid_level": 0}
                row = cuda_resource_plan_row(
                    context, plan, plans_path, required=True, duplicate_count=1
                )
                if not row["valid"]:
                    return False, f"frame {frame_dir.name} has an invalid CUDA resource plan"
                validated_frames += 1
        return close_validation(
            f"validated bounded prefilter capture across {len(stage_roots)} stage(s) "
            f"and {validated_frames} frame(s)"
        )
    depth_map_dir = run_dir / "depth_maps"
    stage_roots = instrumentation_stage_roots(instrumentation_dir)
    validated_frames = 0
    for stage_index, (estimation_stage, geometric_iteration, stage_root) in enumerate(stage_roots):
        terminal_stage = stage_index == len(stage_roots) - 1
        cuda_plans: dict[int, list[dict[str, Any]]] = {}
        plans_path = stage_root / "resource_plans.jsonl"
        for plan in read_jsonl(plans_path):
            raw_image_id = plan.get("image_id")
            image_id = int(raw_image_id) if raw_image_id is not None else -1
            cuda_plans.setdefault(image_id, []).append(plan)
        depthmap_dirs = sorted(path for path in (stage_root / "depthmaps").glob("*") if path.is_dir())
        if not depthmap_dirs:
            return False, f"{estimation_stage} stage has no instrumented frames"
        for frame_dir in depthmap_dirs:
            summary = read_json(frame_dir / "summary.json")
            image_id = int(summary.get("image_id", -1))
            dmap = depth_map_dir / f"depth{image_id:04d}.dmap"
            if terminal_stage and not dmap.is_file():
                return False, f"terminal frame {frame_dir.name} has no saved DMAP output"
            result = instrumentation_validator.validate(
                instrumentation_validator.Arguments(
                    frame_dir=frame_dir,
                    instrumented_dmap=dmap if terminal_stage and dmap.is_file() else None,
                    reference_dmap=None,
                )
            )
            if not result.get("valid"):
                failed = ",".join(
                    str(row.get("name")) for row in result.get("checks") or [] if not row.get("passed")
                )
                return False, f"frame {frame_dir.name} failed validation: {failed}"
            context = {
                "image_id": image_id, "frame": frame_dir.name, "run": "", "role": "",
                "repeat": 0, "scene_id": "", "estimation_stage": estimation_stage,
                "geometric_iteration": geometric_iteration,
            }
            manifest = read_json(frame_dir / "map_manifest.json")
            if (
                manifest.get("schema_version") != 4
                or summary.get("schema_version") != 4
            ):
                return False, (
                    f"frame {frame_dir.name} deep capture requires schema-v4 "
                    "exact Process<true> evidence"
                )
            exact_capture = manifest.get("exact_capture")
            if (
                not isinstance(exact_capture, dict)
                or exact_capture.get("requested") is not True
                or exact_capture.get("available") is not True
            ):
                return False, (
                    f"frame {frame_dir.name} deep capture does not contain "
                    "available exact Process<true> maps"
                )
            cuda_required = True
            matching = cuda_plans.get(image_id, [])
            if not matching:
                cuda_row = cuda_resource_plan_row(
                    context, {}, plans_path, required=cuda_required,
                )
                if cuda_row["required"] and not cuda_row["valid"]:
                    return False, f"frame {frame_dir.name} has no CUDA resource plan"
            else:
                levels = [
                    plan.get("pyramid_level", 0)
                    for plan in matching
                ]
                for plan, level in zip(matching, levels):
                    level_context = dict(context)
                    level_context["pyramid_level"] = level
                    cuda_row = cuda_resource_plan_row(
                        level_context, plan, plans_path, required=cuda_required,
                        duplicate_count=levels.count(level),
                    )
                    if cuda_row["required"] and not cuda_row["valid"]:
                        return False, (
                            f"frame {frame_dir.name} has an invalid CUDA resource "
                            f"plan at pyramid level {level}"
                        )
                    schema_version = int(plan.get("schema_version", 0) or 0)
                    fine_level = schema_version < 3 or level == 0
                    if fine_level:
                        tier_valid = (
                            schema_version >= 2
                            and plan.get("maps_requested") is True
                            and plan.get("maps_available") is True
                            and plan.get("exact_requested") is True
                            and plan.get("exact_available") is True
                        )
                    else:
                        compatibility = plan.get("compatibility_map_contract")
                        explicit_cost_omission = (
                            schema_version < 4
                            or (
                                isinstance(compatibility, dict)
                                and compatibility.get("update_source_map_expected") is True
                                and compatibility.get("cost_map_expected") is False
                                and bool(compatibility.get("cost_map_unavailable_reason"))
                            )
                        )
                        tier_valid = (
                            plan.get("summary_available") is True
                            and plan.get("compatibility_maps_requested") is True
                            and plan.get("maps_requested") is False
                            and plan.get("maps_available") is False
                            and plan.get("exact_requested") is False
                            and plan.get("exact_available") is False
                            and explicit_cost_omission
                        )
                    if not tier_valid:
                        tier = "fine exact-map" if fine_level else "coarse compatibility"
                        return False, (
                            f"frame {frame_dir.name} deep capture has a degraded "
                            f"{tier} resource tier at pyramid level {level}"
                        )
                    if not fine_level:
                        compatibility_valid, compatibility_reason = (
                            validate_coarse_compatibility_update_source_map(
                                stage_root,
                                plan,
                                image_id,
                                int(level),
                            )
                        )
                        if not compatibility_valid:
                            return False, (
                                f"frame {frame_dir.name} has invalid coarse compatibility "
                                f"evidence at pyramid level {level}: "
                                f"{compatibility_reason}"
                            )
                fine_plans = [
                    plan for plan in matching
                    if int(plan.get("schema_version", 0) or 0) < 3
                    or plan.get("pyramid_level") == 0
                ]
                if cuda_required and not fine_plans:
                    return False, f"frame {frame_dir.name} has no fine-scale CUDA resource plan"
            for filter_row in filter_resource_plan_rows(frame_dir, context):
                if filter_row["required"] and not filter_row["valid"]:
                    return False, f"frame {frame_dir.name} has an invalid filter resource plan"
            validated_frames += 1
    return close_validation(
        f"validated map capture across {len(stage_roots)} stage(s) and "
        f"{validated_frames} frame(s)"
    )


def existing_run_mode_action(run_dir: Path, mode: str) -> Literal["run", "skip"]:
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return "run"
    complete, reason = validate_completed_run_mode(run_dir, mode)
    if complete:
        return "skip"
    raise RuntimeError(
        f"refusing to overwrite incomplete or invalid existing {mode} run at {run_dir}: {reason}; "
        "use a new experiment_id or remove the run directory explicitly"
    )


def run_experiment(
    config: dict[str, Any],
    root: Path,
    dry_run: bool,
    requested_profiles: Iterable[str] = (),
    allow_over_budget: bool = False,
) -> None:
    requested_profile_values = tuple(str(value) for value in requested_profiles)
    profiles = capture_profiles(config, requested_profile_values)
    default_args = [str(value) for value in config.get("default_densify_args") or []]
    scenes = resolve_scenes(config, resolve_suite(config))
    effective_estimate = estimate_storage(
        config, scenes, requested_profile_values
    )
    enforce_storage_admission(
        effective_estimate, allow_over_budget=allow_over_budget
    )
    if not dry_run:
        persist_capture_intent(
            root,
            build_capture_intent(
                config,
                root,
                scenes,
                profiles,
                activation_source=(
                    "cli_override" if requested_profile_values else "experiment_config"
                ),
            ),
        )
    persist_effective_storage_plan(root, effective_estimate)
    for run in config.get("runs") or []:
        if run.get("existing"):
            continue
        role = str(run.get("role", "variant"))
        repeats = int(run.get("repeats", 3 if role == "baseline" else 1))
        for repeat_index in range(repeats):
            repeat_label = f"repeat_{repeat_index:02d}"
            for scene in scenes:
                if not scene.get("working_folder") or not scene.get("mvs_file"):
                    raise FileNotFoundError(f"missing cached OpenMVS input for scene {scene['scan_id']}")
                source_work = Path(scene["working_folder"]).expanduser().resolve()
                source_mvs = Path(scene["mvs_file"]).expanduser().resolve()
                scene_id = validated_output_component(
                    scene.get("scan_id"), "scene scan_id"
                )
                scene_record = locked_scene_record(root, scene_id)
                expected_snapshot = scene_record.get("staged_input_snapshot")
                if not isinstance(expected_snapshot, dict):
                    raise RuntimeError(
                        f"experiment lock has no staged input snapshot for scene {scene_id!r}"
                    )
                relative_mvs = scene_mvs_relative_path(source_work, source_mvs)
                frozen_work = frozen_scene_work_dir(root, scene_id)
                for profile in profiles:
                    mode = CAPTURE_PROFILE_MODES[profile]
                    run_label = validated_output_component(run.get("label"), "run label")
                    run_dir = contained_output_path(
                        root,
                        "runs",
                        run_label,
                        repeat_label,
                        scene_id,
                        mode,
                        description="capture run directory",
                    )
                    if existing_run_mode_action(run_dir, mode) == "skip":
                        continue
                    validate_scene_input_snapshot(config, root, scene)
                    validate_staged_scene_input(
                        frozen_work,
                        expected_snapshot,
                        scene_record["mvs_file"],
                        relative_mvs,
                        description=f"frozen scene {scene_id!r}",
                    )
                    work_dir = run_dir / "work"
                    if dry_run:
                        local_mvs = work_dir / relative_mvs
                    else:
                        prepare_working_folder(frozen_work, work_dir)
                        local_mvs = work_dir / relative_mvs
                        validate_staged_scene_input(
                            work_dir,
                            expected_snapshot,
                            scene_record["mvs_file"],
                            relative_mvs,
                            description=(
                                f"profile workspace {run_label}/{scene_id}/{profile}"
                            ),
                        )
                    command_work_dir = dmap_working_folder(local_mvs)
                    scene_name = validated_output_component(
                        scene.get("name", scene_id[:8]), "scene name"
                    )
                    output_mvs = run_dir / f"{scene_name}_dense.mvs"
                    densify_args = [*default_args, *(str(value) for value in run.get("densify_args") or [])]
                    validate_supported_densify_args(densify_args)
                    endpoint = profile == "endpoint"
                    run_binary = densify_binary(config, instrumented=not endpoint)
                    command = [
                        str(run_binary),
                        "--working-folder", str(command_work_dir),
                        "--input-file", str(local_mvs),
                        "--output-file", str(output_mvs),
                    ]
                    if endpoint:
                        densify_args = without_value_arguments(densify_args, OBSERVER_VALUE_ARGUMENTS)
                    else:
                        instrument_dir = run_dir / "dmap_instrumentation"
                        sample_rate = instrumentation_sample_rate(config)
                        level = (
                            "maps"
                            if profile == "deep"
                            else "prefilter"
                            if profile == "prefilter"
                            else "summary"
                        )
                        command.extend([
                            "--dmap-instrumentation-dir", str(instrument_dir),
                            "--dmap-instrumentation-level", level,
                            "--dmap-instrumentation-sample-rate", format(sample_rate, ".9g"),
                            "--dmap-instrumentation-write-maps", "1" if profile == "deep" else "0",
                        ])
                    if not has_argument(densify_args, "--fusion-mode"):
                        command.extend(["--fusion-mode", "1"])
                    command.extend(densify_args)
                    execution_dir = run_dir
                    if dry_run:
                        execution_dir = contained_output_path(
                            root,
                            "plans",
                            "dry_run",
                            run_label,
                            repeat_label,
                            scene_id,
                            profile,
                            description="dry-run capture plan directory",
                        )
                    result = execute_densify_command(
                        command,
                        REPO_ROOT,
                        execution_dir,
                        dry_run,
                        experiment_root=root,
                        runtime_role="production" if endpoint else "observer",
                    )
                    if result["return_code"] != 0:
                        raise RuntimeError(f"DensifyPointCloud failed: {run_dir}")
                    # Planning and capture must both stay bound to the immutable
                    # source snapshot. A dry run still executes local orchestration
                    # hooks and can span enough time for the source to change.
                    validate_scene_input_snapshot(config, root, scene)
                    if not dry_run:
                        # Profile workspaces can share read-only hardlinks with the
                        # frozen input. Revalidate both ends before accepting any
                        # generated artifact so a subprocess cannot corrupt the lock.
                        validate_staged_scene_input(
                            frozen_work,
                            expected_snapshot,
                            scene_record["mvs_file"],
                            relative_mvs,
                            description=f"post-run frozen scene {scene_id!r}",
                        )
                        validate_staged_scene_input(
                            work_dir,
                            expected_snapshot,
                            scene_record["mvs_file"],
                            relative_mvs,
                            description=(
                                f"post-run profile workspace "
                                f"{run_label}/{scene_id}/{profile}"
                            ),
                        )
                        depth_dir = run_dir / "depth_maps"
                        depth_dir.mkdir(exist_ok=True)
                        for dmap in sorted(command_work_dir.glob("depth*.dmap")):
                            hardlink_or_copy(str(dmap), str(depth_dir / dmap.name))
                        if endpoint:
                            write_json(run_dir / "endpoint_metadata.json", {
                                "schema_name": "openmvs.dmap.endpoint_capture",
                                "schema_version": 1,
                                "capture_profile": "endpoint",
                                "instrumentation_compiled": False,
                                "instrumentation_output_present": False,
                                "runtime_authority": "production_endpoint_wall_clock",
                                "runtime_measurement": "repro.json:elapsed_seconds",
                                "executable": str(run_binary.resolve()),
                                "executable_sha256": dmap_drilldown.file_digest(run_binary),
                                "dmap_count": len(list(depth_dir.glob("depth*.dmap"))),
                            })
                        integrity.write_capture_artifact_closure(run_dir, profile)
                        complete, reason = validate_completed_run_mode(run_dir, mode)
                        if not complete:
                            raise RuntimeError(
                                f"{profile} capture validation failed at {run_dir}: {reason}"
                            )


def as_path(value: Any, base: Path | None = None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


CAPTURE_PATH_VALUE_ARGUMENTS = {
    "--working-folder",
    "--input-file",
    "--output-file",
}


def capture_command_argv(capture_dir: Path) -> list[str] | None:
    """Read the single production command emitted by ``execute_command``."""

    command_path = capture_dir / "command.sh"
    if not command_path.is_file():
        return None
    for line in reversed(command_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("set "):
            continue
        try:
            argv = shlex.split(stripped)
        except ValueError:
            return None
        return argv or None
    return None


def capture_command_value(argv: list[str], name: str) -> str | None:
    for index, token in enumerate(argv[1:], start=1):
        if token == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                return None
            return argv[index + 1]
        if token.startswith(f"{name}="):
            return token.split("=", 1)[1]
    return None


def capture_dense_config_identity(argv: list[str], capture_dir: Path) -> str:
    configured = capture_command_value(argv, "--dense-config-file")
    if configured is None:
        return "not_specified"
    path = Path(configured).expanduser()
    if not path.is_absolute():
        working_folder = capture_command_value(argv, "--working-folder")
        path = (Path(working_folder) if working_folder else capture_dir) / path
    path = path.resolve()
    if not path.is_file():
        return f"missing:{Path(configured).name}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_capture_command_signature(
    capture_dir: Path,
) -> tuple[tuple[str, str | None], ...] | None:
    """Return production-relevant arguments independent of capture mechanics.

    Executable and staged input/output paths vary by capture. Instrumentation
    arguments deliberately vary between summary and deep captures. Everything
    else remains in the signature, including resolution and PatchMatch
    overrides that change the produced depth maps.
    """

    argv = capture_command_argv(capture_dir)
    if not argv:
        return None
    ignored = CAPTURE_PATH_VALUE_ARGUMENTS | OBSERVER_VALUE_ARGUMENTS
    normalized: list[tuple[str, str | None]] = [
        ("--dense-config-file-sha256", capture_dense_config_identity(argv, capture_dir))
    ]
    index = 1  # executable identity is validated separately by the experiment lock
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and "=" in token:
            name, value = token.split("=", 1)
            if name not in ignored:
                if name == "--dense-config-file":
                    value = Path(value).name
                normalized.append((name, value))
            index += 1
            continue
        if token.startswith("--"):
            value = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else None
            if token not in ignored:
                if token == "--dense-config-file" and value is not None:
                    value = Path(value).name
                normalized.append((token, value))
            index += 2 if value is not None or token in ignored else 1
            continue
        # No positional production inputs are currently emitted; retain any
        # future ones so an unknown semantic difference cannot be hidden.
        normalized.append(("<positional>", token))
        index += 1
    return tuple(sorted(normalized))


def capture_signature_compatibility(maps_dir: Path, timing_dir: Path) -> tuple[bool, str]:
    """Compare map and summary production arguments conservatively."""

    maps_signature = normalized_capture_command_signature(maps_dir)
    timing_signature = normalized_capture_command_signature(timing_dir)
    if maps_signature is None or timing_signature is None:
        missing = []
        if maps_signature is None:
            missing.append("deep maps command provenance")
        if timing_signature is None:
            missing.append("light-observer command provenance")
        return False, (
            "capture signature compatibility is not applicable because "
            + " and ".join(missing)
            + " is unavailable"
        )
    if maps_signature == timing_signature:
        return True, ""
    maps_values: dict[str, list[str | None]] = {}
    timing_values: dict[str, list[str | None]] = {}
    for name, value in maps_signature:
        maps_values.setdefault(name, []).append(value)
    for name, value in timing_signature:
        timing_values.setdefault(name, []).append(value)
    differences = []
    for name in sorted(maps_values.keys() | timing_values.keys()):
        maps_option = maps_values.get(name, [])
        timing_option = timing_values.get(name, [])
        if maps_option != timing_option:
            differences.append(
                f"{name} maps={maps_option!r} timing={timing_option!r}"
            )
    detail = "; ".join(differences)
    return False, (
        "deep maps and full summary captures have incompatible production "
        f"command signatures: {detail}; cross-capture maps/summary and "
        "production-endpoint parity are not applicable"
    )


def discover_run_scenes(config: dict[str, Any], root: Path) -> list[RunScene]:
    discovered: list[RunScene] = []
    occupied_labels = configured_run_labels(config)
    diagnostic_labels: dict[str, str] = {}
    # Validate the legacy policy value when present. Deep captures are now
    # unconditionally isolated as diagnostic evidence regardless of this flag.
    allow_process_specialization_divergence_for_diagnostics(config)

    def diagnostic_label(label: str, profile: str) -> str:
        key = f"{label}:{profile}"
        existing_label = diagnostic_labels.get(key)
        if existing_label is not None:
            return existing_label
        suffix_label = f"{label} [{profile}]"
        suffix = 2
        while suffix_label in occupied_labels:
            suffix_label = f"{label} [{profile} {suffix}]"
            suffix += 1
        occupied_labels.add(suffix_label)
        diagnostic_labels[key] = suffix_label
        return suffix_label
    for run in config.get("runs") or []:
        label = validated_output_component(run.get("label"), "run label")
        role = str(run.get("role", "variant"))
        existing = run.get("existing") or {}
        if existing:
            for scene_id, row in existing.items():
                if not isinstance(row, dict):
                    continue
                validated_output_component(scene_id, f"existing scene for run {label!r}")
                instrumentation_dir = as_path(row.get("instrumentation_dir"))
                if instrumentation_dir is None:
                    continue
                capture_profile = str(row.get("capture_profile") or "summary")
                diagnostic_only = capture_profile in {"deep", "trace"}
                discovered.append(
                    RunScene(
                        label=(diagnostic_label(label, capture_profile) if diagnostic_only else label),
                        role=role,
                        repeat=int(row.get("repeat", 0)),
                        scene_id=str(scene_id),
                        instrumentation_dir=instrumentation_dir,
                        depth_map_dir=as_path(row.get("depth_map_dir")),
                        timing_dir=as_path(row.get("timing_dir")),
                        configured_label=label,
                        diagnostic_only=diagnostic_only,
                        diagnostic_only_reason=(
                            f"{capture_profile} Process<true> evidence is diagnostic-only"
                            if diagnostic_only else ""
                        ),
                        allow_process_specialization_divergence_for_diagnostics=diagnostic_only,
                        capture_profile=capture_profile,
                    )
                )
            continue
        run_root = contained_output_path(
            root, "runs", label, description=f"run directory for {label!r}"
        )
        for repeat_dir in sorted(run_root.glob("repeat_*")):
            repeat = int(repeat_dir.name.rsplit("_", 1)[-1])
            for scene_dir in sorted(path for path in repeat_dir.iterdir() if path.is_dir()):
                map_dir = scene_dir / "maps"
                prefilter_dir = scene_dir / "prefilter"
                timing_dir = scene_dir / "timing"
                map_instrumentation = map_dir / "dmap_instrumentation"
                prefilter_instrumentation = prefilter_dir / "dmap_instrumentation"
                timing_instrumentation = timing_dir / "dmap_instrumentation"
                maps_available = (map_instrumentation / "run_metadata.json").is_file()
                prefilter_available = (
                    prefilter_instrumentation / "run_metadata.json"
                ).is_file()
                timing_available = (timing_instrumentation / "run_metadata.json").is_file()
                if not maps_available and not prefilter_available and not timing_available:
                    continue
                quality_dir = timing_dir if timing_available else prefilter_dir if prefilter_available else None
                quality_instrumentation = (
                    timing_instrumentation if timing_available
                    else prefilter_instrumentation if prefilter_available
                    else None
                )
                quality_profile = "summary" if timing_available else "prefilter"
                compatible = True
                incompatibility_reason = ""
                if maps_available and quality_dir is not None:
                    compatible, incompatibility_reason = capture_signature_compatibility(
                        map_dir, quality_dir
                    )
                if quality_instrumentation is not None:
                    discovered.append(RunScene(
                        label=label,
                        role=role,
                        repeat=repeat,
                        scene_id=scene_dir.name,
                        instrumentation_dir=quality_instrumentation,
                        depth_map_dir=quality_dir / "depth_maps",
                        timing_dir=(timing_instrumentation if timing_available else quality_instrumentation),
                        configured_label=label,
                        capture_profile=quality_profile,
                    ))
                if maps_available:
                    diagnostic_reason = (
                        "deep Process<true> evidence is diagnostic-only and excluded "
                        "from production quality aggregates"
                    )
                    discovered.append(RunScene(
                        label=diagnostic_label(label, "deep"),
                        role=role,
                        repeat=repeat,
                        scene_id=scene_dir.name,
                        instrumentation_dir=map_instrumentation,
                        depth_map_dir=map_dir / "depth_maps",
                        timing_dir=(
                            timing_instrumentation
                            if timing_available and compatible else map_instrumentation
                        ),
                        cross_capture_parity_compatible=(quality_dir is not None and compatible),
                        cross_capture_parity_reason=(
                            "" if quality_dir is not None and compatible
                            else incompatibility_reason or "no Process<false> quality capture is available"
                        ),
                        diagnostic_only=True,
                        diagnostic_only_reason=diagnostic_reason,
                        allow_process_specialization_divergence_for_diagnostics=True,
                        configured_label=label,
                        capture_profile="deep",
                    ))
    return [stage for run_scene in discovered for stage in expand_instrumentation_stages(run_scene)]


def discover_auxiliary_profile_scenes(
    config: dict[str, Any], root: Path
) -> list[RunScene]:
    """Discover profile evidence that must not enter quality aggregates."""

    auxiliary: list[RunScene] = []
    occupied = configured_run_labels(config)
    for run in config.get("runs") or []:
        label = validated_output_component(run.get("label"), "run label")
        if run.get("existing"):
            continue
        role = str(run.get("role", "variant"))
        run_root = contained_output_path(
            root, "runs", label, description=f"run directory for {label!r}"
        )
        for repeat_dir in sorted(run_root.glob("repeat_*")):
            repeat = int(repeat_dir.name.rsplit("_", 1)[-1])
            for scene_dir in sorted(path for path in repeat_dir.iterdir() if path.is_dir()):
                summary_available = (
                    scene_dir / "timing" / "dmap_instrumentation" / "run_metadata.json"
                ).is_file()
                prefilter_dir = scene_dir / "prefilter"
                prefilter_instrumentation = prefilter_dir / "dmap_instrumentation"
                if not summary_available or not (
                    prefilter_instrumentation / "run_metadata.json"
                ).is_file():
                    continue
                evidence_label = f"{label} [prefilter]"
                suffix = 2
                while evidence_label in occupied:
                    evidence_label = f"{label} [prefilter {suffix}]"
                    suffix += 1
                occupied.add(evidence_label)
                auxiliary.append(RunScene(
                    label=evidence_label,
                    role=role,
                    repeat=repeat,
                    scene_id=scene_dir.name,
                    instrumentation_dir=prefilter_instrumentation,
                    depth_map_dir=prefilter_dir / "depth_maps",
                    timing_dir=prefilter_instrumentation,
                    diagnostic_only=True,
                    diagnostic_only_reason=(
                        "prefilter evidence is auxiliary because summary is the quality authority"
                    ),
                    configured_label=label,
                    capture_profile="prefilter",
                ))
    return [stage for row in auxiliary for stage in expand_instrumentation_stages(row)]


CAPTURE_PROFILE_ORDER = ("endpoint", "summary", "prefilter", "deep", "trace")


def build_capture_profile_coverage(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Describe requested and observed capture evidence without substitution."""

    capture_intents = load_capture_intents(root)
    requested_profiles = list(capture_profiles(config))
    for intent in capture_intents:
        for profile in intent["requested_profiles"]:
            if profile not in requested_profiles:
                requested_profiles.append(profile)
    units: list[dict[str, Any]] = []
    profile_directory = {
        "endpoint": ("endpoint", "endpoint"),
        "summary": ("timing", "timing"),
        "prefilter": ("prefilter", "prefilter"),
        "deep": ("maps", "maps"),
    }

    def capture_unit(
        label: str,
        repeat: int,
        scene_id: str,
        profile: str,
        capture_dir: Path | None,
        *,
        imported: bool = False,
    ) -> dict[str, Any]:
        requested = profile in requested_profiles
        if profile == "trace":
            return {
                "configured_run": label, "repeat": repeat, "scene_id": scene_id,
                "capture_profile": profile,
                "status": "unavailable" if requested else "not_requested",
                "reason": (
                    "trace is an immutable on-demand drilldown request"
                    if requested else "capture profile was not requested"
                ),
                "evidence_links": [], "frames": [],
            }
        assert capture_dir is not None
        exists = capture_dir.is_dir()
        _directory_name, validation_mode = profile_directory[profile]
        valid, reason = (
            validate_completed_run_mode(capture_dir, validation_mode)
            if exists else (False, "capture directory is unavailable")
        )
        closure_path = capture_dir / integrity.CAPTURE_CLOSURE_FILE
        if valid:
            closure_status = (
                "verified" if closure_path.is_file() and not closure_path.is_symlink()
                else "legacy-unverified"
            )
            closure_reason = reason.rsplit("; artifact closure ", 1)[-1]
        elif exists:
            closure_validation = integrity.validate_capture_artifact_closure(
                capture_dir
            )
            closure_status = closure_validation.status
            closure_reason = closure_validation.reason
        else:
            closure_status = "unavailable"
            closure_reason = "capture directory is unavailable"
        status = (
            "complete" if valid else "failed" if exists
            else "unavailable" if requested or imported else "not_requested"
        )
        evidence_links: list[dict[str, Any]] = []
        for name, path in (
            ("command", capture_dir / "command.sh"),
            ("reproduction metadata", capture_dir / "repro.json"),
            ("run metadata", capture_dir / "dmap_instrumentation" / "run_metadata.json"),
        ):
            if path.is_file() and not path.is_symlink():
                evidence_links.append({"label": name, "path": str(path)})
        if closure_path.is_file() and not closure_path.is_symlink():
            evidence_links.append({
                "label": "artifact closure", "path": str(closure_path),
            })
        frames: list[dict[str, Any]] = []
        instrumentation = capture_dir / "dmap_instrumentation"
        for summary_path in sorted(instrumentation.glob("depthmaps/*/summary.json")):
            summary = read_json(summary_path)
            frame_dir = summary_path.parent
            frame_links = [{"label": "summary", "path": str(summary_path)}]
            for name in (
                "map_manifest.json", "capture_complete.json",
                "prefilter_manifest.json", "prefilter_capture_complete.json",
            ):
                artifact = frame_dir / name
                if artifact.is_file() and not artifact.is_symlink():
                    frame_links.append({"label": name, "path": str(artifact)})
            if profile == "prefilter":
                prefilter_map = frame_dir / "maps" / "depth_final_before_filter.pfm"
                if prefilter_map.is_file() and not prefilter_map.is_symlink():
                    frame_links.append({
                        "label": "depth_final_before_filter", "path": str(prefilter_map),
                    })
            frames.append({
                "image_id": int(summary.get("image_id", -1)),
                "status": "complete" if valid else "failed",
                "evidence_links": frame_links,
            })
        if profile == "endpoint":
            frames = [
                {
                    "image_id": int(match.group(1)),
                    "status": "complete" if valid else "failed",
                    "evidence_links": [{"label": "terminal DMAP", "path": str(path)}],
                }
                for path in sorted((capture_dir / "depth_maps").glob("depth*.dmap"))
                if (match := re.fullmatch(r"depth(\d+)\.dmap", path.name))
                and path.is_file() and not path.is_symlink()
            ]
        return {
            "configured_run": label, "repeat": repeat, "scene_id": scene_id,
            "capture_profile": profile, "status": status,
            "reason": (
                reason if status != "complete"
                else closure_reason if closure_status == "legacy-unverified"
                else ""
            ),
            "artifact_closure": {
                "status": closure_status,
                "required": integrity.capture_closure_required(capture_dir),
                "reason": closure_reason,
                "path": str(closure_path) if closure_path.is_file() else None,
            },
            "evidence_links": evidence_links, "frames": frames,
        }

    try:
        configured_scene_ids = [
            validated_output_component(scene_id, "coverage scene_id")
            for scene_id in resolve_suite(config)
        ]
    except ValueError:
        # Keep direct report-model/unit-test ingestion compatible with legacy
        # minimal configs while still deriving a closed matrix when a suite is
        # declared by production configs.
        configured_scene_ids = sorted({
            scene_dir.name
            for repeat_dir in (root / "runs").glob("*/repeat_*")
            if repeat_dir.is_dir()
            for scene_dir in repeat_dir.iterdir() if scene_dir.is_dir()
        })
    for run in config.get("runs") or []:
        label = validated_output_component(run.get("label"), "run label")
        existing = run.get("existing") or {}
        if existing:
            for scene_id, row in sorted(existing.items()):
                if not isinstance(row, dict):
                    continue
                scene_id = validated_output_component(
                    scene_id, f"existing scene for run {label!r}"
                )
                repeat = int(row.get("repeat", 0))
                observed_profile = str(row.get("capture_profile") or "summary")
                instrumentation = as_path(row.get("instrumentation_dir"))
                capture_dir = as_path(row.get("capture_dir"))
                if capture_dir is None and instrumentation is not None:
                    capture_dir = (
                        instrumentation.parent
                        if instrumentation.name == "dmap_instrumentation"
                        else instrumentation
                    )
                for profile in CAPTURE_PROFILE_ORDER:
                    if profile != observed_profile and profile not in requested_profiles:
                        continue
                    units.append(capture_unit(
                        label, repeat, scene_id, profile,
                        capture_dir if profile == observed_profile else root / ".unavailable",
                        imported=profile == observed_profile,
                    ))
            continue
        run_root = contained_output_path(
            root, "runs", label, description=f"run directory for {label!r}"
        )
        role = str(run.get("role", "variant"))
        repeats = int(run.get("repeats", 3 if role == "baseline" else 1))
        expected_identities = {
            (repeat, scene_id)
            for repeat in range(repeats)
            for scene_id in configured_scene_ids
        }
        observed_identities = {
            (int(repeat_dir.name.rsplit("_", 1)[-1]), scene_dir.name)
            for repeat_dir in run_root.glob("repeat_*")
            if re.fullmatch(r"repeat_\d+", repeat_dir.name) and repeat_dir.is_dir()
            for scene_dir in repeat_dir.iterdir() if scene_dir.is_dir()
        }
        for repeat, scene_id in sorted(expected_identities | observed_identities):
            scene_dir = run_root / f"repeat_{repeat:02d}" / scene_id
            for profile in CAPTURE_PROFILE_ORDER:
                if profile not in requested_profiles and profile == "trace":
                    continue
                capture_dir = (
                    scene_dir / profile_directory[profile][0]
                    if profile != "trace" else None
                )
                units.append(capture_unit(
                    label, repeat, scene_id, profile, capture_dir
                ))
    trace_units = {
        (row["configured_run"], row["repeat"], row["scene_id"]): row
        for row in units if row["capture_profile"] == "trace"
    }
    observed_capture_ids: set[str] = set()

    def trace_unit(
        label: str, repeat: int, scene_id: str
    ) -> dict[str, Any]:
        identity = (label, repeat, scene_id)
        unit = trace_units.get(identity)
        if unit is None:
            unit = {
                "configured_run": label,
                "repeat": repeat,
                "scene_id": scene_id,
                "capture_profile": "trace",
                "status": "unavailable",
                "reason": "trace request has not completed",
                "evidence_links": [],
                "frames": [],
                "request_records": [],
            }
            units.append(unit)
            trace_units[identity] = unit
        unit.setdefault("request_records", [])
        return unit

    def add_trace_record(
        unit: dict[str, Any],
        *,
        request_id: str | None,
        status: str,
        reason: str,
        links: list[dict[str, Any]],
        image_id: int | None = None,
    ) -> None:
        unit["request_records"].append({
            "request_sha256": request_id,
            "status": status,
            "reason": reason,
            "evidence_links": links,
        })
        existing_links = {
            (link["label"], link["path"]) for link in unit["evidence_links"]
        }
        unit["evidence_links"].extend(
            link for link in links
            if (link["label"], link["path"]) not in existing_links
        )
        statuses = {
            str(record["status"]) for record in unit["request_records"]
        }
        if "failed" in statuses:
            unit["status"] = "failed"
        elif statuses == {"complete"}:
            unit["status"] = "complete"
        else:
            unit["status"] = "unavailable"
        unit["reason"] = "; ".join(
            dict.fromkeys(
                str(record.get("reason") or "")
                for record in unit["request_records"]
                if record.get("status") != "complete" and record.get("reason")
            )
        )
        if image_id is not None:
            frame = next(
                (item for item in unit["frames"] if item["image_id"] == image_id),
                None,
            )
            if frame is None:
                frame = {
                    "image_id": image_id,
                    "status": status,
                    "evidence_links": [],
                }
                unit["frames"].append(frame)
            elif frame["status"] != "failed":
                frame["status"] = status
            frame_existing = {
                (link["label"], link["path"])
                for link in frame["evidence_links"]
            }
            frame["evidence_links"].extend(
                link for link in links
                if (link["label"], link["path"]) not in frame_existing
            )

    for request_path in sorted((root / "drilldowns" / "requests").glob("*.yaml")):
        try:
            request = dmap_drilldown.load_request(request_path)
        except Exception as exc:
            unit = trace_unit(
                f"invalid_request_{request_path.stem}", 0, "unknown"
            )
            add_trace_record(
                unit,
                request_id=None,
                status="failed",
                reason=f"malformed immutable request: {exc}",
                links=[{"label": "invalid request", "path": str(request_path)}],
            )
            continue
        if request.get("capture_profile") != "trace":
            continue
        request_id = str(request["request_sha256"])
        observed_capture_ids.add(request_id)
        capture_dir = root / "drilldowns" / "captures" / request_id
        capture_request = capture_dir / "request.yaml"
        executions_path = capture_dir / "executions.json"
        capture_exists = capture_dir.exists() or capture_dir.is_symlink()
        capture_request_bound = (
            not capture_request.is_symlink()
            and capture_request.is_file()
            and capture_request.read_bytes() == request_path.read_bytes()
        )
        executions_available = (
            not executions_path.is_symlink() and executions_path.is_file()
        )
        executions_value = read_json(executions_path) if executions_available else {}
        executions = executions_value.get("executions") or []
        executions_contract_valid = not (
            executions_value.get("schema_name") != "openmvs.dmap.drilldown_executions"
            or executions_value.get("schema_version") != 1
            or executions_value.get("request_sha256") != request_id
            or not all(isinstance(item, dict) for item in executions)
        )
        scene_id = str(request["target"]["scene_id"])
        image_id = int(request["target"]["image_id"])
        pixels = dmap_drilldown.expand_trace_pixels(request)
        for requested_run in request.get("runs") or []:
            label = str(requested_run.get("label"))
            repeat = 0
            unit = trace_unit(label, repeat, scene_id)
            links = [{"label": "immutable request", "path": str(request_path)}]
            if capture_request.is_file() and not capture_request.is_symlink():
                links.append({"label": "captured request", "path": str(capture_request)})
            if executions_available:
                links.append({"label": "executions", "path": str(executions_path)})
            if not capture_exists:
                add_trace_record(
                    unit, request_id=request_id, status="unavailable",
                    reason="trace request has no capture yet", links=links,
                    image_id=image_id,
                )
                continue
            if not capture_request_bound:
                add_trace_record(
                    unit, request_id=request_id, status="failed",
                    reason="captured request is missing or does not match the immutable request",
                    links=links, image_id=image_id,
                )
                continue
            if not executions_available or not executions_contract_valid:
                add_trace_record(
                    unit,
                    request_id=request_id,
                    status="failed" if executions_available else "unavailable",
                    reason=(
                        "execution receipt is malformed or bound to another request"
                        if executions_available else "execution receipt is unavailable"
                    ),
                    links=links,
                    image_id=image_id,
                )
                continue
            execution = next(
                (item for item in executions if item.get("run") == label), None
            )
            if (
                execution is None
                or execution.get("scene_id") != scene_id
                or execution.get("capture_profile") != "trace"
                or execution.get("return_code") != 0
            ):
                add_trace_record(
                    unit, request_id=request_id, status="failed",
                    reason="requested run has a missing, mismatched, or failed execution",
                    links=links, image_id=image_id,
                )
                continue
            run_dir = capture_dir / "runs" / label / scene_id
            valid, validation_reason = drilldown_run_complete(
                run_dir, "trace", image_id, pixels
            )
            if not valid:
                add_trace_record(
                    unit, request_id=request_id, status="failed",
                    reason=f"trace capture validation failed: {validation_reason}",
                    links=links, image_id=image_id,
                )
                continue
            traces_path = (
                run_dir / "dmap_instrumentation" / "instrumentation" / "traces.jsonl"
            )
            if traces_path.is_file() and not traces_path.is_symlink():
                links.append({"label": "traces", "path": str(traces_path)})
            closure_path = run_dir / integrity.CAPTURE_CLOSURE_FILE
            if closure_path.is_file() and not closure_path.is_symlink():
                links.append({
                    "label": "artifact closure", "path": str(closure_path),
                })
            add_trace_record(
                unit, request_id=request_id, status="complete", reason="",
                links=links, image_id=image_id,
            )
    for capture_dir in sorted((root / "drilldowns" / "captures").glob("*")):
        if capture_dir.name in observed_capture_ids:
            continue
        unit = trace_unit(f"orphan_capture_{capture_dir.name}", 0, "unknown")
        links = [{"label": "orphan capture", "path": str(capture_dir)}]
        add_trace_record(
            unit,
            request_id=capture_dir.name,
            status="failed",
            reason="capture has no matching immutable request",
            links=links,
        )
    profiles = []
    for profile in CAPTURE_PROFILE_ORDER:
        profile_units = [row for row in units if row["capture_profile"] == profile]
        configured_requested = profile in requested_profiles
        observed_requested = any(
            unit["status"] in {"complete", "failed"} for unit in profile_units
        )
        requested = configured_requested or observed_requested
        counts = {
            status: sum(row["status"] == status for row in profile_units)
            for status in ("complete", "failed", "unavailable")
        }
        status = (
            "failed" if counts["failed"] else
            "unavailable" if requested and counts["unavailable"] else
            "complete" if counts["complete"] and (
                not requested or counts["complete"] == len(profile_units)
            ) else
            "not_requested" if not requested else "unavailable"
        )
        profiles.append({
            "profile": profile,
            "requested": requested,
            "status": status,
            "expected_units": len(profile_units),
            "complete_units": counts["complete"],
            "failed_units": counts["failed"],
            "unavailable_units": counts["unavailable"],
            "process_specialization": (
                "Process<true>" if profile in {"deep", "trace"}
                else "Process<false>" if profile in {"endpoint", "summary", "prefilter"}
                else None
            ),
            "quality_authority": (
                "production_quality_authority" if profile == "endpoint"
                else "eligible_only_after_endpoint_bit_exact_parity"
                if profile in {"summary", "prefilter"}
                else "diagnostic_only"
            ),
        })
    return {
        "schema_name": "openmvs.dmap.capture_profile_coverage",
        "schema_version": 1,
        "requested_profiles": requested_profiles,
        "capture_intents": [
            {
                "intent_sha256": intent["intent_sha256"],
                "activation_source": intent["activation_source"],
                "requested_profiles": intent["requested_profiles"],
                "expected_units": len(intent["units"]),
                "path": str(
                    root / "capture_intents" / f"{intent['intent_sha256']}.json"
                ),
            }
            for intent in capture_intents
        ],
        "profiles": profiles,
        "units": units,
    }


FRAME_RAW_ARTIFACTS = (
    "summary.json", "iteration.csv", "filtering.json", "view_support.csv", "map_manifest.json",
    "prefilter_manifest.json", "prefilter_capture_complete.json",
    "exact_observability.json", "exact_iteration.csv", "exact_view_summary.csv",
    "cpu_view_candidates.json", "cpu_view_estimation_selection.json",
    "postprocess_filters.json", "postprocess_filters.csv",
    "confidence_adjustment.json", "confidence_adjustment.csv", "filter_resource_plan.json",
)
RUN_RAW_ARTIFACTS = (
    "run_metadata.json", "scene_summary.json", "frame_selection.csv", "resource_plans.jsonl",
)


def build_reproducibility_artifact_index(
    config: dict[str, Any],
    root: Path,
    run_scenes: list[RunScene],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        kind: str,
        path: Path,
        *,
        run_scene: RunScene | None = None,
        mode: str = "experiment",
        frame: str = "",
    ) -> None:
        resolved = path.expanduser().resolve()
        identity = file_identity(resolved)
        try:
            experiment_relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            experiment_relative = os.path.relpath(resolved, root.resolve())
        rows.append({
            "kind": kind,
            "run": run_scene.label if run_scene else "",
            "role": run_scene.role if run_scene else "",
            "repeat": run_scene.repeat if run_scene else None,
            "scene_id": run_scene.scene_id if run_scene else "",
            "estimation_stage": run_scene.estimation_stage if run_scene else "",
            "geometric_iteration": run_scene.geometric_iteration if run_scene else None,
            "mode": mode,
            "frame": frame,
            "source_path": str(resolved),
            "experiment_relative_path": Path(experiment_relative).as_posix(),
            "exists": identity["exists"],
            "bytes": identity["size"],
            "sha256": identity["sha256"],
        })

    add("source_config", Path(str(config["_config_path"])))
    for file_name in (
        "00_experiment_lock.json", ENVIRONMENT_MANIFEST_FILE,
        "00_resolved_experiment.yaml",
        "00_storage_estimate.json", "00_suite.json",
    ):
        add(file_name, root / file_name)
    for intent in load_capture_intents(root):
        digest = str(intent["intent_sha256"])
        add("capture_intent", root / "capture_intents" / f"{digest}.json")
    phase = config.get("experiment_phase")
    if isinstance(phase, dict):
        phase_id = phase.get("phase_id")
        if isinstance(phase_id, str):
            phase_root = experiment_phase_evidence_dir(root, phase_id)
            for file_name in (
                "00_phase_lock.json",
                "01_resolved_phase.yaml",
                "02_phase_suite.json",
                "03_storage_estimate.json",
            ):
                add(f"phase/{file_name}", phase_root / file_name)
    seen_stage_roots: set[Path] = set()
    seen_mode_roots: set[Path] = set()
    for run_scene in run_scenes:
        stage_root = run_scene.instrumentation_dir.resolve()
        if stage_root not in seen_stage_roots:
            seen_stage_roots.add(stage_root)
            for file_name in RUN_RAW_ARTIFACTS:
                add(file_name, stage_root / file_name, run_scene=run_scene, mode="maps")
            for frame_dir in sorted(path for path in (stage_root / "depthmaps").glob("*") if path.is_dir()):
                for file_name in FRAME_RAW_ARTIFACTS:
                    add(file_name, frame_dir / file_name, run_scene=run_scene, mode="maps", frame=frame_dir.name)
        for mode, instrumentation_root in (
            ("maps", run_scene.instrumentation_dir),
            ("timing", run_scene.timing_dir),
        ):
            if instrumentation_root is None:
                continue
            mode_root = instrumentation_root.resolve().parent
            if mode_root in seen_mode_roots:
                continue
            seen_mode_roots.add(mode_root)
            for file_name in (
                "command.sh", "repro.json", "stdout.log", "stderr.log",
                integrity.CAPTURE_CLOSURE_FILE,
            ):
                add(file_name, mode_root / file_name, run_scene=run_scene, mode=mode)
        if run_scene.estimation_stage == "photometric":
            endpoint_root = run_scene.instrumentation_dir.parent.parent / "endpoint"
            if endpoint_root not in seen_mode_roots:
                seen_mode_roots.add(endpoint_root)
                for file_name in (
                    "command.sh", "repro.json", "stdout.log", "stderr.log", "endpoint_metadata.json",
                    integrity.CAPTURE_CLOSURE_FILE,
                ):
                    add(file_name, endpoint_root / file_name, run_scene=run_scene, mode="endpoint")
    return pd.DataFrame(rows)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_integer(value: Any) -> int | None:
    number = safe_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def view_probability_health_rows(
    summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]]]:
    """Validate and normalize the frame-level probability-health census."""

    raw = summary.get("view_probability_health")
    status: dict[str, Any] = {
        "schema_valid": False,
        "requested": False,
        "available": False,
        "accounting_valid": False,
        "unavailable_reason": "view_probability_health is absent from summary.json",
        "totals": {},
        "validation_errors": [],
    }
    if not isinstance(raw, dict):
        return status, {}
    status["requested"] = raw.get("requested") is True
    status["unavailable_reason"] = str(raw.get("unavailable_reason") or "")
    errors: list[str] = []
    if raw.get("schema_name") != "openmvs.dmap.view_probability_health":
        errors.append("invalid schema_name")
    if raw.get("schema_version") != 1:
        errors.append("unsupported schema_version")
    iterations = raw.get("iterations")
    totals = raw.get("totals")
    if not isinstance(iterations, list):
        errors.append("iterations must be an array")
        iterations = []
    if not isinstance(totals, dict):
        errors.append("totals must be an object")
        totals = {}
    if raw.get("available") is True and raw.get("accounting_valid") is not True:
        errors.append("available payload must declare accounting_valid=true")

    normalized: dict[tuple[int, int], dict[str, Any]] = {}
    sums = {name: 0 for name in VIEW_PROBABILITY_HEALTH_COUNTERS}
    for index, raw_row in enumerate(iterations):
        if not isinstance(raw_row, dict):
            errors.append(f"iterations[{index}] must be an object")
            continue
        level = _nonnegative_integer(raw_row.get("pyramid_level"))
        logical_iteration = _nonnegative_integer(raw_row.get("logical_iteration"))
        if level is None or logical_iteration is None:
            errors.append(f"iterations[{index}] has invalid identity")
            continue
        identity = (level, logical_iteration)
        if identity in normalized:
            errors.append(f"iterations[{index}] duplicates identity {identity}")
            continue
        row: dict[str, Any] = {}
        valid_counters = True
        for name in VIEW_PROBABILITY_HEALTH_COUNTERS:
            value = _nonnegative_integer(raw_row.get(name))
            if value is None:
                errors.append(f"iterations[{index}].{name} must be a nonnegative integer")
                valid_counters = False
                continue
            row[f"view_probability_{name}"] = value
            sums[name] += value
        if not valid_counters:
            continue
        processed = row["view_probability_processed"]
        finite = row["view_probability_finite_positive_events"]
        degenerate = row["view_probability_degenerate_events"]
        if finite + degenerate != processed:
            errors.append(f"iterations[{index}] event accounting is inconsistent")
        row["view_probability_healthy_ratio"] = finite / processed if processed else None
        row["view_probability_degenerate_ratio"] = degenerate / processed if processed else None
        row["view_probability_mean_positive_views"] = (
            row["view_probability_positive_view_count_sum"] / processed if processed else None
        )
        row["view_probability_unassigned_draws_per_event"] = (
            row["view_probability_unassigned_draws"] / processed if processed else None
        )
        normalized[identity] = row

    normalized_totals: dict[str, int] = {}
    for name in VIEW_PROBABILITY_HEALTH_COUNTERS:
        value = _nonnegative_integer(totals.get(name))
        if value is None:
            errors.append(f"totals.{name} must be a nonnegative integer")
            continue
        normalized_totals[name] = value
        if value != sums[name]:
            errors.append(f"totals.{name} does not match iteration rows")
    status.update({
        "schema_valid": not errors,
        "available": raw.get("available") is True and not errors and bool(normalized),
        "accounting_valid": raw.get("accounting_valid") is True and not errors,
        "totals": normalized_totals,
        "validation_errors": errors,
    })
    if errors:
        status["unavailable_reason"] = "malformed probability-health payload: " + "; ".join(errors)
        normalized = {}
    elif raw.get("available") is not True and not status["unavailable_reason"]:
        status["unavailable_reason"] = "probability-health census was not available"
    return status, normalized


def mean_support(summary: dict[str, Any], depthmap_dir: Path) -> float | None:
    histogram = summary.get("supporting_view_histogram")
    if isinstance(histogram, list) and histogram:
        total = sum(int(value) for value in histogram)
        return sum(index * int(value) for index, value in enumerate(histogram)) / total if total else None
    path = depthmap_dir / "view_support.csv"
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    total = sum(int(float(row["pixels"])) for row in rows)
    return (
        sum(int(float(row["supporting_view_count"])) * int(float(row["pixels"])) for row in rows) / total
        if total else None
    )


def instrumentation_csv(root: Path | None, name: str) -> Path | None:
    if root is None:
        return None
    candidates = (root / "instrumentation" / name, root / name)
    return next((path for path in candidates if path.is_file()), None)


@lru_cache(maxsize=1024)
def _dmap_valid_depth_coverage_cached(
    path_text: str,
    file_size: int,
    modified_ns: int,
) -> float:
    """Read only the depth payload needed for final-DMAP coverage.

    Size and mtime are cache-key inputs so a regenerated DMAP cannot reuse a
    stale report value for the same path.
    """

    del file_size, modified_ns
    path = Path(path_text)
    # Keep one validated decoder for both legacy DR and quantized D2 production
    # files.  D2 depth_exp must be applied before validity is measured.
    depth = np.asarray(annotation_fit.load_dmap(path)["depth_map"], dtype=np.float32)
    if depth.ndim != 2 or not depth.size:
        raise ValueError(f"invalid DMAP depth payload: {path}")
    valid = np.isfinite(depth) & (depth > 0.0)
    return float(np.count_nonzero(valid) / depth.size)


def dmap_valid_depth_coverage(
    path: Path, owning_root: Path | None = None
) -> float:
    """Return the valid-depth fraction from a terminal production DMAP."""

    source = _lexical_absolute_path(path)
    root = _lexical_absolute_path(owning_root or source.parent)
    source = owned_regular_artifact_path(
        root, source, description="terminal DMAP"
    )
    status = source.stat()
    return _dmap_valid_depth_coverage_cached(
        str(source), int(status.st_size), int(status.st_mtime_ns)
    )


def load_instrumentation(
    run_scene: RunScene,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    scene_summary = read_json(run_scene.instrumentation_dir / "scene_summary.json")
    scene_ignore_mask = nested_value(scene_summary, "aggregate", "ignore_mask", default={})
    if not isinstance(scene_ignore_mask, dict):
        scene_ignore_mask = {}
    for depthmap_dir in sorted((run_scene.instrumentation_dir / "depthmaps").glob("*")):
        if not depthmap_dir.is_dir():
            continue
        summary = read_json(depthmap_dir / "summary.json")
        if not summary:
            continue
        probability_health, probability_health_by_iteration = view_probability_health_rows(summary)
        final_cost = summary.get("final_cost") or {}
        total_pixels = int(summary.get("num_pixels_total", 0))
        rejected_keep_cost = int(
            summary.get("num_rejected_by_keep_cost_filter", summary.get("num_rejected_by_filter", 0))
        )
        ignore_mask = summary.get("ignore_mask") if isinstance(summary.get("ignore_mask"), dict) else {}
        ignore_mask_status = str(ignore_mask.get("status") or (
            "legacy_count_available" if not ignore_mask else "not_requested"
        ))
        mask_count_available = bool(ignore_mask.get(
            "rejection_count_available", ignore_mask_status != "unavailable"
        ))
        raw_mask_rejected = summary.get("num_rejected_by_ignore_mask")
        rejected_ignore_mask = (
            int(raw_mask_rejected or 0) if mask_count_available else None
        )
        image_id = int(summary.get("image_id", -1))
        endpoint_dmap = (
            run_scene.depth_map_dir / f"depth{image_id:04d}.dmap"
            if run_scene.depth_map_dir is not None and image_id >= 0 else None
        )
        endpoint_coverage = None
        endpoint_coverage_status = "unavailable"
        endpoint_coverage_reason = "terminal production DMAP is unavailable"
        if endpoint_dmap is not None and endpoint_dmap.is_file():
            try:
                endpoint_coverage = dmap_valid_depth_coverage(
                    endpoint_dmap, run_scene.depth_map_dir.parent
                )
                endpoint_coverage_status = "available"
                endpoint_coverage_reason = ""
            except (OSError, ValueError, IndexError) as exc:
                endpoint_coverage_status = "read_error"
                endpoint_coverage_reason = f"{type(exc).__name__}: {exc}"
        row = {
            "run": run_scene.label,
            "configured_run": run_scene.configured_label or run_scene.label,
            "diagnostic_only": run_scene.diagnostic_only,
            "capture_profile": run_scene.capture_profile,
            "role": run_scene.role,
            "repeat": run_scene.repeat,
            "scene_id": run_scene.scene_id,
            "estimation_stage": run_scene.estimation_stage,
            "geometric_iteration": run_scene.geometric_iteration,
            "image_id": image_id,
            "image_name": str(summary.get("image_name", "")),
            "safe_image_name": str(summary.get("safe_image_name", depthmap_dir.name)),
            "width": int(summary.get("width", 0)),
            "height": int(summary.get("height", 0)),
            "valid_ratio_before_filter": safe_float(summary.get("valid_ratio_before_filter")),
            "valid_ratio_after_keep_cost_filter": safe_float(
                summary.get("valid_ratio_after_keep_cost_filter", summary.get("valid_ratio_after_filter"))
            ),
            "valid_ratio_after_filter": safe_float(summary.get("valid_ratio_after_filter")),
            "endpoint_valid_depth_coverage": endpoint_coverage,
            "endpoint_valid_depth_coverage_status": endpoint_coverage_status,
            "endpoint_valid_depth_coverage_reason": endpoint_coverage_reason,
            "endpoint_valid_depth_coverage_source": (
                str(endpoint_dmap) if endpoint_dmap is not None else None
            ),
            "rejected_by_filter_ratio": safe_float(summary.get("rejected_by_filter_ratio")),
            "num_rejected_by_filter": int(summary.get("num_rejected_by_filter", 0)),
            "num_rejected_by_keep_cost_filter": rejected_keep_cost,
            "num_rejected_by_ignore_mask": rejected_ignore_mask,
            "ignore_mask_status": ignore_mask_status,
            "ignore_mask_requested": ignore_mask.get("requested"),
            "ignore_mask_loaded": ignore_mask.get("loaded"),
            "ignore_mask_rejection_count_available": mask_count_available,
            "ignore_mask_unavailable_reason": str(ignore_mask.get("unavailable_reason") or ""),
            "scene_ignore_mask_requested_frames": scene_ignore_mask.get("requested_frames"),
            "scene_ignore_mask_loaded_frames": scene_ignore_mask.get("loaded_frames"),
            "scene_ignore_mask_unavailable_frames": scene_ignore_mask.get("unavailable_frames"),
            "scene_ignore_mask_rejection_counts_available": scene_ignore_mask.get("rejection_counts_available"),
            "rejected_by_keep_cost_filter_ratio": (
                rejected_keep_cost / total_pixels if total_pixels > 0 else None
            ),
            "rejected_by_ignore_mask_ratio": (
                rejected_ignore_mask / total_pixels
                if rejected_ignore_mask is not None and total_pixels > 0 else None
            ),
            "final_cost_mean": safe_float(final_cost.get("mean")),
            "final_cost_median": safe_float(final_cost.get("median")),
            "final_cost_p90": safe_float(final_cost.get("p90")),
            "final_cost_p95": safe_float(final_cost.get("p95")),
            "mean_support": mean_support(summary, depthmap_dir),
            "depthmap_dir": str(depthmap_dir),
            "map_manifest": str(
                depthmap_dir / "map_manifest.json"
                if (depthmap_dir / "map_manifest.json").is_file()
                else depthmap_dir / "prefilter_manifest.json"
                if (depthmap_dir / "prefilter_manifest.json").is_file()
                else ""
            ) or None,
            "missing_maps": json.dumps(summary.get("missing_maps") or []),
            "candidate_accounting_mode": str(summary.get("candidate_accounting_mode", "legacy_or_exact")),
            "confidence_gap_mode": str(summary.get("confidence_gap_mode", "same_pass_or_unspecified")),
            "view_probability_health_schema_valid": probability_health["schema_valid"],
            "view_probability_health_requested": probability_health["requested"],
            "view_probability_health_available": probability_health["available"],
            "view_probability_health_accounting_valid": probability_health["accounting_valid"],
            "view_probability_health_unavailable_reason": probability_health["unavailable_reason"],
            "view_probability_health_totals_json": json.dumps(
                probability_health["totals"], sort_keys=True
            ),
            "view_probability_health_validation_errors_json": json.dumps(
                probability_health["validation_errors"]
            ),
        }
        for candidate in summary.get("candidate_acceptance") or []:
            name = re.sub(r"[^a-z0-9]+", "_", str(candidate.get("candidate_type", "unknown")).lower()).strip("_")
            row[f"candidate_{name}_tested"] = int(candidate.get("tested_count", 0))
            row[f"candidate_{name}_finite"] = int(candidate.get("finite_count", 0))
            row[f"candidate_{name}_accepted"] = int(candidate.get("accepted_count", 0))
            row[f"candidate_{name}_acceptance_rate"] = safe_float(candidate.get("acceptance_rate"))
        row = instrumentation_report.apply_candidate_accounting_contract(
            row, row["candidate_accounting_mode"]
        )
        frame_rows.append(row)
        iteration_path = depthmap_dir / "iteration.csv"
        observed_health_identities: set[tuple[int, int]] = set()
        if iteration_path.is_file():
            depthmap_iteration_rows: list[dict[str, Any]] = []
            with iteration_path.open(encoding="utf-8") as iteration_handle:
                for pass_row in csv.DictReader(iteration_handle):
                    converted: dict[str, Any] = {
                        "run": run_scene.label,
                        "configured_run": run_scene.configured_label or run_scene.label,
                        "capture_profile": run_scene.capture_profile,
                        "role": run_scene.role,
                        "repeat": run_scene.repeat,
                        "scene_id": run_scene.scene_id,
                        "estimation_stage": run_scene.estimation_stage,
                        "geometric_iteration": run_scene.geometric_iteration,
                        "image_id": int(float(pass_row.get("image_id", row["image_id"]))),
                        "candidate_accounting_mode": row["candidate_accounting_mode"],
                    }
                    for key, value in pass_row.items():
                        if key in {"image_name", "phase"}:
                            converted[key] = value
                        elif value not in (None, ""):
                            converted[key] = safe_float(value)
                    depthmap_iteration_rows.append(converted)
            logical_rows = instrumentation_report.aggregate_logical_iteration_rows(
                depthmap_iteration_rows
            )
            for logical_row in logical_rows:
                logical_row.update(
                    instrumentation_report.apply_candidate_accounting_contract(
                        logical_row, row["candidate_accounting_mode"]
                    )
                )
                identity = (
                    int(logical_row.get("scale_level", 0)),
                    int(logical_row.get("logical_iteration", -1)),
                )
                health_row = probability_health_by_iteration.get(identity)
                if health_row is not None:
                    logical_row.update(health_row)
                    observed_health_identities.add(identity)
            pass_rows.extend(logical_rows)
        for (level, logical_iteration), health_row in sorted(
            probability_health_by_iteration.items()
        ):
            if (level, logical_iteration) in observed_health_identities:
                continue
            pass_rows.append(instrumentation_report.apply_candidate_accounting_contract({
                "run": run_scene.label,
                "configured_run": run_scene.configured_label or run_scene.label,
                "capture_profile": run_scene.capture_profile,
                "role": run_scene.role,
                "repeat": run_scene.repeat,
                "scene_id": run_scene.scene_id,
                "estimation_stage": run_scene.estimation_stage,
                "geometric_iteration": run_scene.geometric_iteration,
                "image_id": int(summary.get("image_id", -1)),
                "image_name": str(summary.get("image_name", "")),
                "scale_level": level,
                "iteration": logical_iteration,
                "logical_iteration": logical_iteration,
                "stage": instrumentation_report.logical_iteration_label(logical_iteration),
                "phase": "iteration",
                "pass_index": logical_iteration + 1,
                "candidate_accounting_mode": row["candidate_accounting_mode"],
                **health_row,
            }, row["candidate_accounting_mode"]))
    timing_path = instrumentation_csv(run_scene.timing_dir, "timings.csv")
    if timing_path is not None:
        with timing_path.open(encoding="utf-8") as handle:
            for timing_row in csv.DictReader(handle):
                kernel_ms = safe_float(timing_row.get("kernel_ms"))
                if kernel_ms is None:
                    continue
                timing_rows.append({
                    "run": run_scene.label,
                    "configured_run": run_scene.configured_label or run_scene.label,
                    "capture_profile": run_scene.capture_profile,
                    "role": run_scene.role,
                    "repeat": run_scene.repeat,
                    "scene_id": run_scene.scene_id,
                    "estimation_stage": run_scene.estimation_stage,
                    "geometric_iteration": run_scene.geometric_iteration,
                    "image_id": int(float(timing_row.get("image_id", -1))),
                    "scale_number": int(float(timing_row.get("scale_number", 0))),
                    "pass_index": int(float(timing_row.get("pass_index", -1))),
                    "phase": str(timing_row.get("phase", "")),
                    "iteration": int(float(timing_row.get("iteration", -1))),
                    "kernel_ms": kernel_ms,
                    "timing_source": str(timing_path),
                })
    return frame_rows, pass_rows, timing_rows


def artifact_measurement_quality(artifact: instrumentation_report.MapArtifact) -> str:
    if artifact.measurement_quality:
        return artifact.measurement_quality
    return "derived_exact" if artifact.fidelity == "derived" else artifact.fidelity


def artifact_relative_path(depthmap_dir: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(depthmap_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def map_catalog_context(run_scene: RunScene, depthmap_dir: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    pyramid_level = None
    for source in (manifest, summary):
        for key in ("pyramid_level", "scale_level", "scale_number"):
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                candidate = int(float(value))
            except (TypeError, ValueError):
                continue
            if candidate >= 0:
                pyramid_level = candidate
                break
        if pyramid_level is not None:
            break
    return {
        "run": run_scene.label,
        "label": run_scene.label,
        "configured_run": run_scene.configured_label or run_scene.label,
        "capture_profile": run_scene.capture_profile,
        "run_role": run_scene.role,
        "repeat": run_scene.repeat,
        "scene_id": run_scene.scene_id,
        "estimation_stage": run_scene.estimation_stage,
        "geometric_iteration": run_scene.geometric_iteration,
        "frame": depthmap_dir.name,
        "image_id": int(summary.get("image_id", -1)),
        "image_name": str(summary.get("image_name", "")),
        "safe_image_name": str(summary.get("safe_image_name", depthmap_dir.name)),
        "manifest_schema_name": str(manifest.get("schema_name", "")),
        "manifest_schema_version": int(manifest.get("schema_version", summary.get("schema_version", 0)) or 0),
        "map_granularity": str(manifest.get("map_granularity", "")),
        "pyramid_level": pyramid_level,
        "manifest_path": str(
            depthmap_dir / "prefilter_manifest.json"
            if manifest.get("schema_name") == "openmvs.dmap.prefilter_manifest"
            else depthmap_dir / "map_manifest.json"
        ),
    }


def map_artifact_row(
    context: dict[str, Any],
    depthmap_dir: Path,
    artifact: instrumentation_report.MapArtifact,
) -> dict[str, Any]:
    exists = artifact.path.is_file()
    size = artifact.bytes
    if size is None and exists:
        size = artifact.path.stat().st_size
    logical_iteration = artifact.logical_iteration
    # Current CUDA captures store scale-transfer and static patch-eligibility
    # maps through the generic final-state helper. Their algorithmic scope is
    # fine-level initialization, so normalize the storage role at ingestion.
    if (
        logical_iteration is None
        and artifact.signal in OPTIONAL_MECHANISM_INITIALIZATION_SIGNALS
    ):
        logical_iteration = -1
    stage = artifact.stage
    if not stage and logical_iteration is not None:
        stage = "initialization" if logical_iteration < 0 else "iteration"
    role = artifact.role
    if (
        logical_iteration is not None
        and artifact.signal in OPTIONAL_MECHANISM_INITIALIZATION_SIGNALS
    ):
        role = "logical_state"
    if not role:
        role = artifact.temporal_scope
    return {
        **context,
        "signal": artifact.signal,
        "role": role,
        "logical_iteration": logical_iteration,
        "pyramid_level": (
            artifact.pyramid_level
            if artifact.pyramid_level is not None
            else context.get("pyramid_level")
        ),
        "stage": stage,
        "algorithm_stage": artifact.algorithm_stage,
        "pass_index": artifact.pass_index,
        "dtype": artifact.dtype,
        "measurement_quality": artifact_measurement_quality(artifact),
        "measurement_basis": artifact.measurement_basis,
        "proxy_target": artifact.proxy_target,
        "limitations": artifact.limitations,
        "semantics": artifact.semantics,
        "stage_index": artifact.stage_index,
        "source_view_index": artifact.source_view_index,
        "source_image_id": artifact.source_image_id,
        "source_image_name": artifact.source_image_name,
        "contribution_basis": artifact.contribution_basis,
        "channels_json": json.dumps(artifact.channels, sort_keys=True) if artifact.channels is not None else "",
        "encoding": artifact.encoding,
        "unavailable_value": artifact.unavailable_value,
        "gap_scope": artifact.gap_scope,
        "path": str(artifact.path),
        "relative_path": artifact_relative_path(depthmap_dir, artifact.path),
        "component_paths_json": "",
        "bytes": size,
        "exists": exists,
        "available": exists,
    }


def auxiliary_map_rows(
    run_scene: RunScene,
    depthmap_dir: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Catalog exact optional-filter maps emitted outside map_manifest.json."""
    rows: list[dict[str, Any]] = []
    specifications = (
        ("postprocess_filters.json", "postprocess_filter_state", "sequential_postprocess_stage"),
        ("confidence_adjustment.json", "confidence_adjustment_state", "confidence_adjustment_method"),
    )
    for file_name, role, granularity in specifications:
        artifact_path = depthmap_dir / file_name
        artifact = read_json(artifact_path)
        if not artifact:
            continue
        context = map_catalog_context(run_scene, depthmap_dir, summary, artifact)
        context["map_granularity"] = granularity
        context["manifest_path"] = str(artifact_path)
        stage_indices = {
            str(stage.get("name")): stage.get("stage_index")
            for stage in artifact.get("stages") or []
            if isinstance(stage, dict)
        }
        for parsed in instrumentation_report.parse_map_artifacts(depthmap_dir, artifact):
            row = map_artifact_row(context, depthmap_dir, parsed)
            row["role"] = role
            row["algorithm_stage"] = parsed.algorithm_stage or str(artifact.get("algorithm_stage", ""))
            row["stage"] = row["algorithm_stage"]
            if row.get("stage_index") is None:
                row["stage_index"] = stage_indices.get(row["algorithm_stage"])
            if not row.get("measurement_basis"):
                row["measurement_basis"] = (
                    "exact sequential before/after production depth-map state"
                    if file_name == "postprocess_filters.json"
                    else "exact production confidence-adjustment state"
                )
            rows.append(row)
    return rows


def compatibility_map_rows(
    run_scene: RunScene,
    depthmap_dir: Path,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Catalog per-pyramid compatibility maps declared by resource plans."""

    rows: list[dict[str, Any]] = []
    image_id = _nonnegative_integer(summary.get("image_id"))
    if image_id is None:
        return rows
    plans_path = run_scene.instrumentation_dir / "resource_plans.jsonl"
    seen_levels: set[int] = set()
    for plan in read_jsonl(plans_path):
        if _nonnegative_integer(plan.get("image_id")) != image_id:
            continue
        level = _nonnegative_integer(plan.get("pyramid_level"))
        if level is None or level == 0 or level in seen_levels:
            continue
        contract = plan.get("compatibility_map_contract")
        expected = (
            isinstance(contract, dict)
            and contract.get("update_source_map_expected") is True
        )
        if not expected and int(plan.get("schema_version", 0) or 0) < 4:
            expected = plan.get("compatibility_maps_requested") is True
        if not expected:
            continue
        seen_levels.add(level)
        path = (
            run_scene.instrumentation_dir
            / "instrumentation"
            / "maps"
            / f"depth{image_id:04d}_scale{level:02d}_update_source.png"
        )
        context = map_catalog_context(run_scene, depthmap_dir, summary, plan)
        context.update({
            "manifest_schema_name": str(plan.get("schema_name", "")),
            "manifest_schema_version": int(plan.get("schema_version", 0) or 0),
            "manifest_path": str(plans_path),
            "map_granularity": "pyramid_level_terminal_state",
            "pyramid_level": level,
        })
        artifact = instrumentation_report.MapArtifact(
            signal="candidate_source",
            path=path,
            dtype="uint8",
            role="final_state",
            semantics=(
                "display encoding of the last detected update source at this "
                "pyramid level; exact source family is unavailable"
            ),
            pyramid_level=level,
            stage="final_state",
            fidelity="proxy",
            measurement_quality="proxy",
            measurement_basis="post_pass_change_detection",
            proxy_target="exact_update_attribution",
            limitations=(
                "post-pass state changes do not identify the exact winning "
                "candidate"
            ),
            temporal_scope="final_state",
            bytes=path.stat().st_size if path.is_file() else None,
            schema_version=int(plan.get("schema_version", 0) or 0),
            encoding="source_code*32",
        )
        row = map_artifact_row(context, depthmap_dir, artifact)
        row["relative_path"] = path.relative_to(
            run_scene.instrumentation_dir
        ).as_posix()
        rows.append(row)
    return rows


def expected_logical_iterations(
    manifest: dict[str, Any],
    artifacts: list[instrumentation_report.MapArtifact],
) -> list[int]:
    num_states = int(manifest.get("num_logical_states", 0) or 0)
    if num_states > 0:
        return [-1, *range(max(0, num_states - 1))]
    if manifest.get("num_iterations") not in (None, ""):
        num_iterations = max(0, int(manifest["num_iterations"]))
        return [-1, *range(num_iterations)]
    observed = sorted({artifact.logical_iteration for artifact in artifacts if artifact.logical_iteration is not None})
    return observed or [-1]


def summary_logical_iterations(
    depthmap_dir: Path, scale_level: int | None = None
) -> list[int]:
    """Read complete logical iterations declared by a summary capture."""

    path = depthmap_dir / "iteration.csv"
    if not path.is_file():
        return []
    iterations: set[int] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_scale = next(
                (
                    row.get(key)
                    for key in ("pyramid_level", "scale_level", "scale_number")
                    if row.get(key) not in (None, "")
                ),
                None,
            )
            if scale_level is not None and raw_scale not in (None, ""):
                try:
                    if int(float(raw_scale)) != scale_level:
                        continue
                except ValueError:
                    continue
            try:
                value = float(row.get("iteration", ""))
                iteration = int(value)
            except (TypeError, ValueError):
                continue
            if value == iteration:
                iterations.add(iteration)
    return sorted(iterations)


def missing_signal_row(
    context: dict[str, Any],
    signal: str,
    logical_iteration: int | None,
    *,
    role: str = "logical_state",
    source_view_index: int | None = None,
) -> dict[str, Any]:
    presentation = instrumentation_report.LOGICAL_COST_SIGNAL_PRESENTATION.get(signal)
    fidelity = presentation[1] if presentation else "exact"
    measurement_quality = "derived_exact" if fidelity == "derived" else fidelity
    return {
        **context,
        "signal": signal,
        "role": role,
        "logical_iteration": logical_iteration,
        "stage": (
            "final_state" if logical_iteration is None
            else "initialization" if logical_iteration < 0
            else "iteration"
        ),
        "algorithm_stage": "",
        "pass_index": None,
        "dtype": "",
        "measurement_quality": measurement_quality,
        "measurement_basis": "",
        "proxy_target": "",
        "limitations": "",
        "semantics": "",
        "stage_index": (
            None if logical_iteration is None
            else 0 if logical_iteration < 0
            else logical_iteration + 1
        ),
        "source_view_index": source_view_index,
        "source_image_id": None,
        "source_image_name": "",
        "contribution_basis": "",
        "channels_json": "",
        "encoding": "",
        "unavailable_value": None,
        "gap_scope": "",
        "path": "",
        "relative_path": "",
        "component_paths_json": "",
        "bytes": None,
        "exists": False,
        "available": False,
    }


def low_texture_update_frame_execution(
    run_parameters: dict[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    """Resolve configured and actual per-frame hysteresis execution."""

    summary_parameters = summary.get("cuda_patchmatch_parameters")
    summary_parameters = (
        summary_parameters if isinstance(summary_parameters, dict) else {}
    )
    effective_parameters = dict(run_parameters)
    for key in ("low_texture_update_min_gain", "low_texture_update_gate"):
        if key in summary_parameters:
            effective_parameters[key] = summary_parameters[key]

    parameters_declared = {
        "low_texture_update_min_gain", "low_texture_update_gate",
    }.issubset(effective_parameters)
    min_gain = safe_float(effective_parameters.get("low_texture_update_min_gain"))
    gate = safe_float(effective_parameters.get("low_texture_update_gate"))
    configured = (
        parameters_declared
        and min_gain is not None and min_gain > 0.0
        and gate is not None and int(gate) != 0
    )

    prior_declared = "low_resolution_prior_available" in summary_parameters
    raw_prior = summary_parameters.get("low_resolution_prior_available")
    prior_available = raw_prior if isinstance(raw_prior, bool) else None
    prior_valid = not prior_declared or prior_available is not None
    iterations_declared = "estimation_iterations" in summary_parameters
    raw_iterations = summary_parameters.get("estimation_iterations")
    estimation_iterations = (
        raw_iterations
        if isinstance(raw_iterations, int) and not isinstance(raw_iterations, bool)
        else None
    )
    iterations_valid = (
        not iterations_declared
        or estimation_iterations is not None
        and 0 <= estimation_iterations <= 1024
    )
    execution_available = (
        configured
        and prior_valid
        and prior_available is not False
        and iterations_valid
        and estimation_iterations != 0
    )

    execution_unavailable_reason = None
    if configured and prior_available is False:
        execution_unavailable_reason = (
            "configured_but_not_executed: this frame/stage has no "
            "coarse-resolution prior"
        )
    elif configured and not prior_valid:
        execution_unavailable_reason = (
            "execution_availability_invalid: summary cuda_patchmatch_parameters."
            "low_resolution_prior_available is not boolean"
        )
    elif configured and not iterations_valid:
        execution_unavailable_reason = (
            "execution_availability_invalid: summary cuda_patchmatch_parameters."
            "estimation_iterations is not an integer in [0,1024]"
        )
    elif configured and estimation_iterations == 0:
        execution_unavailable_reason = (
            "configured_but_not_executed: estimation_iterations is zero"
        )

    return {
        "parameters_declared": parameters_declared,
        "configured": configured,
        "execution_available": execution_available,
        "low_resolution_prior_available": prior_available,
        "estimation_iterations": estimation_iterations,
        "prior_metadata_valid": prior_valid,
        "iterations_metadata_valid": iterations_valid,
        "execution_unavailable_reason": execution_unavailable_reason,
    }


def build_map_catalog(run_scenes: list[RunScene]) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    for run_scene in run_scenes:
        run_metadata = read_json(run_scene.instrumentation_dir / "run_metadata.json")
        cuda_parameters = run_metadata.get("cuda_patchmatch_parameters")
        cuda_parameters = cuda_parameters if isinstance(cuda_parameters, dict) else {}
        depthmaps_root = run_scene.instrumentation_dir / "depthmaps"
        for depthmap_dir in sorted(path for path in depthmaps_root.glob("*") if path.is_dir()):
            summary = read_json(depthmap_dir / "summary.json")
            hysteresis_execution = low_texture_update_frame_execution(
                cuda_parameters, summary
            )
            hysteresis_parameters_declared = bool(
                hysteresis_execution["parameters_declared"]
            )
            hysteresis_configured = bool(hysteresis_execution["configured"])
            hysteresis_execution_available = bool(
                hysteresis_execution["execution_available"]
            )
            hysteresis_execution_unavailable_reason = (
                hysteresis_execution["execution_unavailable_reason"]
            )
            manifest_path = depthmap_dir / "map_manifest.json"
            if not manifest_path.is_file():
                manifest_path = depthmap_dir / "prefilter_manifest.json"
            manifest = read_json(manifest_path)
            context = map_catalog_context(
                run_scene, depthmap_dir, summary, manifest if manifest else summary
            )
            if not manifest:
                context["manifest_path"] = str(depthmap_dir / "summary.json")
                context["map_granularity"] = "logical_iteration"
            artifacts = instrumentation_report.parse_map_artifacts(depthmap_dir, manifest)
            artifact_signal_ids = {artifact.signal for artifact in artifacts}
            adaptive_patch_signal_family = {
                *OPTIONAL_ADAPTIVE_PATCH_LOGICAL_STATE_SIGNALS,
                "adaptive_patch_support_mode",
                "adaptive_patch_activation",
            }
            jbu_signal_family = {
                "jbu_transfer_depth",
                "jbu_nearest_depth",
                "jbu_transfer_depth_delta",
                "jbu_fallback_status",
            }
            hierarchy_signal_family = set(OPTIONAL_MECHANISM_FINAL_STATE_SIGNALS)
            adaptive_patch_extension_declared = bool(
                artifact_signal_ids & adaptive_patch_signal_family
            )
            jbu_extension_declared = bool(artifact_signal_ids & jbu_signal_family)
            hierarchy_extension_declared = bool(
                artifact_signal_ids & hierarchy_signal_family
            )
            optional_logical_state_signals = (
                OPTIONAL_ADAPTIVE_PATCH_LOGICAL_STATE_SIGNALS
                if adaptive_patch_extension_declared
                else ()
            )
            optional_initialization_signals = (
                tuple(
                    signal
                    for signal in OPTIONAL_MECHANISM_INITIALIZATION_SIGNALS
                    if (
                        signal in adaptive_patch_signal_family
                        and adaptive_patch_extension_declared
                    )
                    or (
                        signal in jbu_signal_family
                        and jbu_extension_declared
                    )
                )
            )
            optional_final_state_signals = (
                OPTIONAL_MECHANISM_FINAL_STATE_SIGNALS
                if hierarchy_extension_declared
                else ()
            )
            low_texture_extension_declared = bool(
                hysteresis_parameters_declared
                or artifact_signal_ids & set(LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS)
            )
            optional_low_texture_event_signals = (
                LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
                if low_texture_extension_declared
                else ()
            )
            artifact_rows = [map_artifact_row(context, depthmap_dir, artifact) for artifact in artifacts]
            catalog_rows.extend(artifact_rows)
            catalog_rows.extend(auxiliary_map_rows(run_scene, depthmap_dir, summary))
            catalog_rows.extend(compatibility_map_rows(run_scene, depthmap_dir, summary))
            if not manifest:
                marker = read_json(depthmap_dir / "summary_complete.json")
                schema_version = int(summary.get("schema_version", 0) or 0)
                if (
                    schema_version < 4
                    or marker.get("schema_name") != "openmvs.dmap.summary_complete"
                    or marker.get("summary_complete") is not True
                ):
                    continue
                resource_plan = summary.get("resource_plan") or {}
                unavailable_reason = (
                    "summary_profile_maps_not_requested"
                    if resource_plan.get("maps_requested") is False
                    else str(resource_plan.get("decision") or "summary_capture_maps_unavailable")
                )
                exact_unavailable_reason = str(
                    resource_plan.get("exact_unavailable_reason")
                    or "exact capture unavailable in summary profile"
                )
                scale_level = context.get("pyramid_level")
                for logical_iteration in summary_logical_iterations(depthmap_dir, scale_level):
                    for signal in REQUIRED_LOGICAL_STATE_SIGNALS:
                        availability = missing_signal_row(context, signal, logical_iteration)
                        availability["required"] = False
                        availability["availability_reason"] = unavailable_reason
                        availability_rows.append(availability)
                    for signal in (
                        *MECHANISM_LOGICAL_STATE_SIGNALS,
                        *optional_logical_state_signals,
                    ):
                        availability = missing_signal_row(context, signal, logical_iteration)
                        availability["required"] = False
                        availability["availability_reason"] = (
                            f"exact_capture_unavailable: {exact_unavailable_reason}"
                        )
                        availability_rows.append(availability)
                    if logical_iteration == -1:
                        for signal in optional_initialization_signals:
                            availability = missing_signal_row(context, signal, logical_iteration)
                            availability["required"] = False
                            availability["availability_reason"] = unavailable_reason
                            availability_rows.append(availability)
                    for signal in (
                        *REQUIRED_LOGICAL_EVENT_SIGNALS,
                        *REPORT_DERIVED_LOGICAL_EVENT_SIGNALS,
                    ):
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role="logical_event"
                        )
                        availability["required"] = False
                        availability["availability_reason"] = unavailable_reason
                        availability_rows.append(availability)
                    for signal in optional_low_texture_event_signals:
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role="logical_event"
                        )
                        availability["required"] = False
                        if logical_iteration < 0:
                            availability["availability_reason"] = (
                                "not_applicable_at_initialization: the update gate is evaluated only "
                                "during iterative candidate updates"
                            )
                        elif hysteresis_parameters_declared and not hysteresis_configured:
                            availability["availability_reason"] = (
                                "mechanism_disabled: low_texture_update_min_gain must be positive "
                                "and low_texture_update_gate must be nonzero"
                            )
                        elif hysteresis_execution_unavailable_reason:
                            availability["availability_reason"] = str(
                                hysteresis_execution_unavailable_reason
                            )
                        else:
                            availability["availability_reason"] = (
                                f"exact_capture_unavailable: {exact_unavailable_reason}"
                            )
                        availability_rows.append(availability)
                    for signal, role in [
                        *[(value, "logical_state") for value in SCHEMA4_EXACT_STATE_SIGNALS],
                        *[(value, "logical_event") for value in SCHEMA4_EXACT_EVENT_SIGNALS],
                        *[(value, "logical_view_state") for value in SCHEMA4_EXACT_VIEW_SIGNALS],
                    ]:
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role=role
                        )
                        availability["required"] = False
                        availability["availability_reason"] = (
                            f"exact_capture_unavailable: {exact_unavailable_reason}"
                        )
                        availability_rows.append(availability)
                for signal in SUMMARY_PROFILE_FINAL_SIGNALS:
                    descriptor = component_registry.descriptor_from_row({"signal": signal})
                    availability = missing_signal_row(
                        context, signal, None,
                        role=(
                            "filtering_state"
                            if descriptor.mechanism == "filtering"
                            else "final_state"
                        ),
                    )
                    availability["required"] = False
                    availability["availability_reason"] = unavailable_reason
                    availability_rows.append(availability)
                for signal in optional_final_state_signals:
                    availability = missing_signal_row(context, signal, None)
                    availability["required"] = False
                    availability["availability_reason"] = unavailable_reason
                    availability_rows.append(availability)
                continue
            if manifest.get("schema_name") == "openmvs.dmap.prefilter_manifest":
                unavailable_reason = (
                    "prefilter_profile_retains_only_production_depth_before_filter"
                )
                exact_unavailable_reason = (
                    "exact_capture_not_requested_by_prefilter_profile"
                )
                for logical_iteration in summary_logical_iterations(
                    depthmap_dir, context.get("pyramid_level")
                ):
                    for signal in (
                        *REQUIRED_LOGICAL_STATE_SIGNALS,
                        *MECHANISM_LOGICAL_STATE_SIGNALS,
                        *optional_logical_state_signals,
                    ):
                        availability = missing_signal_row(
                            context, signal, logical_iteration
                        )
                        availability["required"] = False
                        availability["availability_reason"] = unavailable_reason
                        availability_rows.append(availability)
                    if logical_iteration == -1:
                        for signal in optional_initialization_signals:
                            availability = missing_signal_row(
                                context, signal, logical_iteration
                            )
                            availability["required"] = False
                            availability["availability_reason"] = unavailable_reason
                            availability_rows.append(availability)
                    for signal in (
                        *REQUIRED_LOGICAL_EVENT_SIGNALS,
                        *REPORT_DERIVED_LOGICAL_EVENT_SIGNALS,
                    ):
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role="logical_event"
                        )
                        availability["required"] = False
                        availability["availability_reason"] = unavailable_reason
                        availability_rows.append(availability)
                    for signal, role in [
                        *[(value, "logical_state") for value in SCHEMA4_EXACT_STATE_SIGNALS],
                        *[(value, "logical_event") for value in SCHEMA4_EXACT_EVENT_SIGNALS],
                        *[(value, "logical_view_state") for value in SCHEMA4_EXACT_VIEW_SIGNALS],
                        *[(value, "logical_event") for value in optional_low_texture_event_signals],
                    ]:
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role=role
                        )
                        availability["required"] = False
                        availability["availability_reason"] = (
                            f"exact_capture_unavailable: {exact_unavailable_reason}"
                        )
                        availability_rows.append(availability)
                for signal in SUMMARY_PROFILE_FINAL_SIGNALS:
                    if signal == "depth_final_before_filter":
                        continue
                    descriptor = component_registry.descriptor_from_row({"signal": signal})
                    availability = missing_signal_row(
                        context,
                        signal,
                        None,
                        role=(
                            "filtering_state"
                            if descriptor.mechanism == "filtering"
                            else "final_state"
                        ),
                    )
                    availability["required"] = False
                    availability["availability_reason"] = unavailable_reason
                    availability_rows.append(availability)
                for signal in optional_final_state_signals:
                    availability = missing_signal_row(context, signal, None)
                    availability["required"] = False
                    availability["availability_reason"] = unavailable_reason
                    availability_rows.append(availability)
                continue
            if context["manifest_schema_version"] < 3:
                continue

            by_signal_iteration: dict[tuple[str, int], list[tuple[instrumentation_report.MapArtifact, dict[str, Any]]]] = {}
            for artifact, row in zip(artifacts, artifact_rows):
                logical_iteration = row.get("logical_iteration")
                if logical_iteration is None:
                    continue
                by_signal_iteration.setdefault((artifact.signal, int(logical_iteration)), []).append((artifact, row))
            manifest_exact_capture = manifest.get("exact_capture") or {}
            exact_available = bool(manifest_exact_capture.get("available"))
            exact_reason = str(
                manifest_exact_capture.get("unavailable_reason")
                or "exact capture unavailable"
            )
            raw_order_contract = manifest_exact_capture.get("candidate_order_statistics") or {}
            raw_order_required = (
                isinstance(raw_order_contract, dict)
                and raw_order_contract.get("available") is True
            )
            enabled_hysteresis_exact_required = (
                hysteresis_execution_available and exact_available
            )
            for logical_iteration in expected_logical_iterations(manifest, artifacts):
                for signal in REQUIRED_LOGICAL_STATE_SIGNALS:
                    candidates = by_signal_iteration.get((signal, logical_iteration), [])
                    if candidates:
                        _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                        availability = dict(selected)
                        availability["required"] = True
                        availability["availability_reason"] = "available" if selected["exists"] else "declared_artifact_missing"
                    else:
                        availability = missing_signal_row(context, signal, logical_iteration)
                        availability["required"] = True
                        availability["availability_reason"] = "not_declared_in_manifest"
                    availability_rows.append(availability)
                for signal in (
                    *MECHANISM_LOGICAL_STATE_SIGNALS,
                    *optional_logical_state_signals,
                ):
                    candidates = by_signal_iteration.get((signal, logical_iteration), [])
                    if candidates:
                        _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                        availability = dict(selected)
                        availability["availability_reason"] = (
                            "available" if selected["exists"] else "declared_artifact_missing"
                        )
                    else:
                        availability = missing_signal_row(context, signal, logical_iteration)
                        availability["availability_reason"] = "not_declared_in_manifest"
                    availability["required"] = False
                    availability_rows.append(availability)
                for signal in optional_low_texture_event_signals:
                    candidates = by_signal_iteration.get((signal, logical_iteration), [])
                    if candidates:
                        _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                        availability = dict(selected)
                        availability["availability_reason"] = (
                            "available" if selected["exists"] else "declared_artifact_missing"
                        )
                    else:
                        availability = missing_signal_row(
                            context, signal, logical_iteration, role="logical_event"
                        )
                        if logical_iteration < 0:
                            availability["availability_reason"] = (
                                "not_applicable_at_initialization: the update gate is evaluated only "
                                "during iterative candidate updates"
                            )
                        elif hysteresis_parameters_declared and not hysteresis_configured:
                            availability["availability_reason"] = (
                                "mechanism_disabled: low_texture_update_min_gain must be positive "
                                "and low_texture_update_gate must be nonzero"
                            )
                        elif not hysteresis_parameters_declared:
                            availability["availability_reason"] = (
                                "capture_schema_does_not_declare_low_texture_update_hysteresis"
                            )
                        elif hysteresis_execution_unavailable_reason:
                            availability["availability_reason"] = str(
                                hysteresis_execution_unavailable_reason
                            )
                        elif not exact_available:
                            availability["availability_reason"] = (
                                f"exact_capture_unavailable: {exact_reason}"
                            )
                        else:
                            availability["availability_reason"] = "not_declared_in_manifest"
                    availability["required"] = bool(
                        logical_iteration >= 0
                        and (
                            raw_order_required
                            if signal in LOW_TEXTURE_UPDATE_RAW_ORDER_EXACT_EVENT_SIGNALS
                            else enabled_hysteresis_exact_required
                        )
                    )
                    availability_rows.append(availability)
                if logical_iteration == -1:
                    for signal in optional_initialization_signals:
                        candidates = by_signal_iteration.get((signal, logical_iteration), [])
                        if candidates:
                            _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                            availability = dict(selected)
                            availability["availability_reason"] = (
                                "available" if selected["exists"] else "declared_artifact_missing"
                            )
                        else:
                            availability = missing_signal_row(context, signal, logical_iteration)
                            availability["availability_reason"] = "not_declared_in_manifest"
                        availability["required"] = False
                        availability_rows.append(availability)
                if context["manifest_schema_version"] < 4:
                    continue
                exact_capture = manifest_exact_capture
                for signal, role in [
                    *[(value, "logical_state") for value in SCHEMA4_EXACT_STATE_SIGNALS],
                    *[(value, "logical_event") for value in SCHEMA4_EXACT_EVENT_SIGNALS],
                ]:
                    candidates = by_signal_iteration.get((signal, logical_iteration), [])
                    if candidates:
                        _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                        availability = dict(selected)
                        availability["required"] = exact_available
                        availability["availability_reason"] = "available" if selected["exists"] else "declared_artifact_missing"
                    else:
                        availability = missing_signal_row(context, signal, logical_iteration, role=role)
                        availability["required"] = exact_available
                        availability["availability_reason"] = (
                            "not_declared_in_manifest" if exact_available else f"exact_capture_unavailable: {exact_reason}"
                        )
                    availability_rows.append(availability)
                num_views = int(exact_capture.get("num_views", 0) or 0)
                for signal in SCHEMA4_EXACT_VIEW_SIGNALS:
                    candidates = by_signal_iteration.get((signal, logical_iteration), [])
                    source_view_indices: Iterable[int | None] = range(num_views) if num_views > 0 else (None,)
                    for source_view_index in source_view_indices:
                        view_candidates = [
                            item for item in candidates if item[0].source_view_index == source_view_index
                        ]
                        if view_candidates:
                            _artifact, selected = max(view_candidates, key=lambda item: bool(item[1]["exists"]))
                            availability = dict(selected)
                            availability["required"] = exact_available
                            availability["availability_reason"] = "available" if selected["exists"] else "declared_artifact_missing"
                        else:
                            availability = missing_signal_row(
                                context, signal, logical_iteration,
                                role="logical_view_state", source_view_index=source_view_index,
                            )
                            availability["required"] = exact_available
                            availability["availability_reason"] = (
                                "not_declared_in_manifest" if exact_available else f"exact_capture_unavailable: {exact_reason}"
                            )
                        availability_rows.append(availability)

            for signal in optional_final_state_signals:
                candidates = [
                    (artifact, row)
                    for artifact, row in zip(artifacts, artifact_rows)
                    if artifact.signal == signal and row.get("logical_iteration") is None
                ]
                if candidates:
                    _artifact, selected = max(candidates, key=lambda item: bool(item[1]["exists"]))
                    availability = dict(selected)
                    availability["availability_reason"] = (
                        "available" if selected["exists"] else "declared_artifact_missing"
                    )
                else:
                    availability = missing_signal_row(context, signal, None)
                    availability["availability_reason"] = "not_declared_in_manifest"
                availability["required"] = False
                availability_rows.append(availability)

    catalog = pd.DataFrame(catalog_rows, columns=MAP_CATALOG_COLUMNS)
    availability = pd.DataFrame(availability_rows, columns=SIGNAL_AVAILABILITY_COLUMNS)
    return catalog, availability


POSTPROCESS_STAGE_ORDER = ("remove_speckles", "fill_gaps")
CONFIDENCE_ADJUSTMENT_METHODS = (
    "adjust_confidence_fast", "adjust_confidence", "final_combined_confidence",
)
FILTER_RESOURCE_COMPONENTS = ("postprocess_filters", "confidence_adjustment")


def nested_value(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def cuda_resource_plan_row(
    context: dict[str, Any],
    plan: dict[str, Any],
    path: Path,
    *,
    required: bool,
    duplicate_count: int = 0,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not plan:
        if required:
            errors.append("required_cuda_resource_plan_missing")
    else:
        if plan.get("schema_name") != "openmvs.dmap.resource_plan":
            errors.append("unexpected_schema_name")
        schema_version = int(plan.get("schema_version", 0) or 0)
        if schema_version not in {1, 2, 3, 4}:
            errors.append("unsupported_schema_version")
        plan_image_id = _nonnegative_integer(plan.get("image_id"))
        if plan_image_id != _nonnegative_integer(context.get("image_id")):
            errors.append("reference_image_id_mismatch")
        pyramid_level = plan.get("pyramid_level")
        if schema_version >= 3 and (
            isinstance(pyramid_level, bool)
            or not isinstance(pyramid_level, int)
            or pyramid_level < 0
        ):
            errors.append("invalid_pyramid_level")
        preflight_required = (
            bool(plan.get("summary_available"))
            if schema_version >= 4
            else bool(plan.get("maps_available"))
        )
        if preflight_required and schema_version >= 2:
            if not bool(nested_value(plan, "storage_preflight", "attempted", default=False)):
                errors.append(
                    "capture_admitted_without_storage_preflight"
                    if schema_version >= 4
                    else "maps_admitted_without_storage_preflight"
                )
            if not bool(nested_value(plan, "storage_preflight", "succeeded", default=False)):
                errors.append(
                    "capture_admitted_after_failed_storage_preflight"
                    if schema_version >= 4
                    else "maps_admitted_after_failed_storage_preflight"
                )
        if schema_version >= 4:
            def nonnegative_integer(value: Any) -> int | None:
                return (
                    value
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    else None
                )

            current_storage = nonnegative_integer(nested_value(
                plan, "effective_estimate_bytes", "current_pyramid_storage"
            ))
            committed_storage = nonnegative_integer(nested_value(
                plan, "effective_estimate_bytes", "frame_storage_committed_before"
            ))
            priority_reserve = nonnegative_integer(nested_value(
                plan, "effective_estimate_bytes", "full_resolution_priority_reserve"
            ))
            frame_storage = nonnegative_integer(nested_value(
                plan, "effective_estimate_bytes", "frame_storage"
            ))
            if None in {
                current_storage, committed_storage, priority_reserve, frame_storage,
            }:
                errors.append("invalid_cumulative_storage_fields")
            elif frame_storage != committed_storage + current_storage:
                errors.append("cumulative_storage_arithmetic_mismatch")

            requested_storage = nonnegative_integer(nested_value(
                plan, "storage_preflight", "requested_bytes"
            ))
            requested_with_reserve = nonnegative_integer(nested_value(
                plan, "storage_preflight", "requested_plus_priority_reserve_bytes"
            ))
            priority_reservation = nonnegative_integer(nested_value(
                plan, "storage_preflight", "frame_priority_reservation_bytes"
            ))
            priority_consumed = nested_value(
                plan, "storage_preflight", "frame_priority_reservation_consumed"
            )
            if (
                requested_storage is None
                or requested_with_reserve is None
                or priority_reservation is None
                or priority_reserve is None
                or not isinstance(priority_consumed, bool)
            ):
                errors.append("invalid_storage_preflight_priority_fields")
            else:
                if requested_with_reserve != requested_storage + priority_reserve:
                    errors.append("storage_preflight_priority_arithmetic_mismatch")
                if priority_consumed and pyramid_level != 0:
                    errors.append("priority_reservation_consumed_before_fine_level")
                if (
                    plan.get("summary_available") is True
                    and isinstance(pyramid_level, int)
                    and pyramid_level > 0
                    and priority_reserve > 0
                    and priority_reservation != priority_reserve
                ):
                    errors.append("coarse_level_priority_reservation_not_held")
        for component, limit in (
            ("device", nested_value(plan, "limits_mib", "device")),
            ("host", nested_value(plan, "limits_mib", "host")),
            ("frame_storage", nested_value(plan, "limits_mib", "frame_storage")),
        ):
            effective = nested_value(plan, "effective_estimate_bytes", component)
            if effective is not None and limit not in (None, 0) and int(effective) > int(limit) * 1024**2:
                errors.append(f"effective_{component}_exceeds_limit")
        if duplicate_count > 1:
            warnings.append(f"{duplicate_count}_plans_for_frame_latest_selected")
    trace_unavailable_reason = (
        plan.get("trace_unavailable_reason")
        if plan.get("trace_requested") is True and plan.get("trace_available") is False
        else None
    )
    return {
        **context,
        "plan_kind": "cuda_patchmatch",
        "component": "cuda_patchmatch",
        "schema_name": plan.get("schema_name", "openmvs.dmap.resource_plan"),
        "schema_version": plan.get("schema_version"),
        "required": required,
        "available": bool(plan),
        "valid": (not errors) if plan else (False if required else None),
        "validation_errors_json": json.dumps(errors),
        "validation_warnings_json": json.dumps(warnings),
        "duplicate_count": duplicate_count,
        "plan_identity_complete": all(
            key in plan for key in (
                "image_id", "estimation_stage", "geometric_iteration",
                "pyramid_level", "width", "height",
            )
        ),
        "plan_image_id": plan.get("image_id"),
        "plan_estimation_stage": plan.get("estimation_stage"),
        "plan_geometric_iteration": plan.get("geometric_iteration"),
        "plan_pyramid_level": plan.get("pyramid_level"),
        "pyramid_level": plan.get("pyramid_level", context.get("pyramid_level")),
        "grid_width": plan.get("width"),
        "grid_height": plan.get("height"),
        "compatibility_maps_requested": plan.get("compatibility_maps_requested"),
        "compatibility_map_contract_json": json.dumps(
            plan.get("compatibility_map_contract") or {}, sort_keys=True
        ),
        "decision": plan.get("decision", "unavailable"),
        "reason": trace_unavailable_reason or nested_value(plan, "storage_preflight", "reason") or plan.get("exact_unavailable_reason") or (
            "resource plan unavailable" if not plan else ""
        ),
        "trace_requested": plan.get("trace_requested"),
        "trace_available": plan.get("trace_available"),
        "maps_requested": plan.get("maps_requested"),
        "maps_available": plan.get("maps_available"),
        "exact_requested": plan.get("exact_requested"),
        "exact_available": plan.get("exact_available"),
        "summary_available": plan.get("summary_available"),
        "effective_device_bytes": nested_value(plan, "effective_estimate_bytes", "device"),
        "effective_host_bytes": nested_value(plan, "effective_estimate_bytes", "host"),
        "effective_storage_bytes": nested_value(plan, "effective_estimate_bytes", "frame_storage"),
        "current_pyramid_storage_bytes": nested_value(plan, "effective_estimate_bytes", "current_pyramid_storage"),
        "frame_storage_committed_before_bytes": nested_value(plan, "effective_estimate_bytes", "frame_storage_committed_before"),
        "full_resolution_priority_reserve_bytes": nested_value(plan, "effective_estimate_bytes", "full_resolution_priority_reserve"),
        "device_limit_mib": nested_value(plan, "limits_mib", "device"),
        "host_limit_mib": nested_value(plan, "limits_mib", "host"),
        "storage_limit_mib": nested_value(plan, "limits_mib", "frame_storage"),
        "storage_preflight_attempted": nested_value(plan, "storage_preflight", "attempted"),
        "storage_preflight_succeeded": nested_value(plan, "storage_preflight", "succeeded"),
        "storage_available_bytes": nested_value(plan, "storage_preflight", "available_bytes"),
        "storage_reserved_before_bytes": nested_value(plan, "storage_preflight", "reserved_before_bytes", default=nested_value(plan, "storage_preflight", "reservation_bytes")),
        "storage_leased_bytes": nested_value(plan, "storage_preflight", "reservation_bytes"),
        "storage_requested_plus_priority_reserve_bytes": nested_value(plan, "storage_preflight", "requested_plus_priority_reserve_bytes"),
        "storage_frame_priority_reservation_bytes": nested_value(plan, "storage_preflight", "frame_priority_reservation_bytes"),
        "storage_frame_priority_reservation_consumed": nested_value(plan, "storage_preflight", "frame_priority_reservation_consumed"),
        "lease_released": None,
        "actual_map_count": None,
        "actual_declared_bytes": None,
        "actual_file_bytes": None,
        "estimate_covers_declared_bytes": None,
        "source_json": str(path),
    }


def filter_resource_plan_rows(
    frame_dir: Path,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    path = frame_dir / "filter_resource_plan.json"
    plan = read_json(path)
    postprocess = read_json(frame_dir / "postprocess_filters.json")
    confidence = read_json(frame_dir / "confidence_adjustment.json")
    component_artifacts = {
        "postprocess_filters": postprocess,
        "confidence_adjustment": confidence,
    }
    components = plan.get("components") if isinstance(plan.get("components"), dict) else {}
    rows: list[dict[str, Any]] = []
    for component_name in FILTER_RESOURCE_COMPONENTS:
        reference_declared = (
            str(nested_value(
                component_artifacts[component_name], "resource_plan", "path", default=""
            )) == path.name
        )
        component = components.get(component_name) if isinstance(components.get(component_name), dict) else {}
        errors: list[str] = []
        if reference_declared and not plan:
            errors.append("referenced_filter_resource_plan_missing")
        if plan and (reference_declared or component):
            if plan.get("schema_name") != "openmvs.dmap.filter_resource_plan":
                errors.append("unexpected_schema_name")
            if int(plan.get("schema_version", 0) or 0) not in {1, 2}:
                errors.append("unsupported_schema_version")
            if _nonnegative_integer(plan.get("reference_image_id")) != _nonnegative_integer(
                context.get("image_id")
            ):
                errors.append("reference_image_id_mismatch")
            if reference_declared and not component:
                errors.append("component_missing")
            if bool(nested_value(component, "effective_capabilities", "maps", default=False)):
                if not bool(nested_value(component, "storage_preflight", "attempted", default=False)):
                    errors.append("maps_admitted_without_storage_preflight")
                if not bool(nested_value(component, "storage_preflight", "succeeded", default=False)):
                    errors.append("maps_admitted_after_failed_storage_preflight")
                if not bool(nested_value(component, "storage_preflight", "lease_released", default=False)):
                    errors.append("storage_lease_not_released")
            if component.get("fatal"):
                errors.append("fatal_resource_decision")
            actual = component.get("actual_maps")
            if isinstance(actual, dict) and actual.get("estimate_covers_declared_bytes") is False:
                errors.append("estimate_does_not_cover_declared_bytes")
        actual = component.get("actual_maps") if isinstance(component.get("actual_maps"), dict) else {}
        rows.append({
            **context,
            "plan_kind": "optional_filter",
            "component": component_name,
            "schema_name": plan.get("schema_name", "openmvs.dmap.filter_resource_plan"),
            "schema_version": plan.get("schema_version"),
            "required": reference_declared,
            "available": bool(plan and component),
            "valid": (not errors) if plan and component else (False if reference_declared else None),
            "validation_errors_json": json.dumps(errors),
            "validation_warnings_json": "[]",
            "duplicate_count": 1 if plan else 0,
            "decision": component.get("decision", "unavailable"),
            "reason": component.get("reason") or (
                "filter resource plan unavailable" if not plan else "component unavailable"
            ),
            "maps_requested": nested_value(component, "requested_capabilities", "maps"),
            "maps_available": nested_value(component, "effective_capabilities", "maps"),
            "exact_requested": None,
            "exact_available": None,
            "summary_available": nested_value(component, "effective_capabilities", "summary"),
            "effective_device_bytes": 0,
            "effective_host_bytes": nested_value(component, "effective_estimate_bytes", "host"),
            "effective_storage_bytes": nested_value(component, "effective_estimate_bytes", "storage"),
            "device_limit_mib": 0,
            "host_limit_mib": nested_value(plan, "limits_mib", "host"),
            "storage_limit_mib": nested_value(plan, "limits_mib", "frame_storage"),
            "storage_preflight_attempted": nested_value(component, "storage_preflight", "attempted"),
            "storage_preflight_succeeded": nested_value(component, "storage_preflight", "succeeded"),
            "storage_available_bytes": nested_value(component, "storage_preflight", "available_bytes"),
            "storage_reserved_before_bytes": nested_value(component, "storage_preflight", "reserved_before_bytes"),
            "storage_leased_bytes": nested_value(component, "storage_preflight", "leased_bytes"),
            "lease_released": nested_value(component, "storage_preflight", "lease_released"),
            "actual_map_count": actual.get("map_count"),
            "actual_declared_bytes": actual.get("declared_bytes"),
            "actual_file_bytes": actual.get("file_bytes"),
            "estimate_covers_declared_bytes": actual.get("estimate_covers_declared_bytes"),
            "source_json": str(path),
        })
    return rows


def load_postprocess_observability(frame_dir: Path, context: dict[str, Any]) -> list[dict[str, Any]]:
    path = frame_dir / "postprocess_filters.json"
    artifact = read_json(path)
    declared = {
        str(row.get("name")): row
        for row in artifact.get("stages") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(POSTPROCESS_STAGE_ORDER):
        stage = declared.get(name, {})
        metrics = stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {}
        unavailable_reason = str(stage.get("unavailable_reason") or "")
        if not artifact:
            unavailable_reason = "artifact_not_produced"
        elif not stage:
            unavailable_reason = "stage_not_declared_in_artifact"
        enabled = bool(stage.get("enabled", False))
        executed = bool(stage.get("executed", False))
        success = stage.get("success")
        if unavailable_reason:
            status = "unavailable"
            quality = "unavailable"
        elif not enabled:
            status = "disabled"
            quality = "exact"
        elif executed and success is False:
            status = "failed"
            quality = "exact"
        elif executed:
            status = "available"
            quality = "exact"
        else:
            status = "unavailable"
            quality = "unavailable"
            unavailable_reason = "enabled_stage_not_executed"
        rows.append({
            **context,
            "schema_name": artifact.get("schema_name", "openmvs.dmap.postprocess_filters"),
            "schema_version": artifact.get("schema_version"),
            "algorithm_stage": artifact.get("algorithm_stage", "depth_map_optional_postprocess"),
            "stage_index": stage.get("stage_index", index),
            "stage_name": name,
            "enabled": enabled,
            "executed": executed,
            "success": success,
            "artifact_status": status,
            "measurement_quality": quality,
            "measurement_basis": stage.get("measurement_basis", ""),
            "unavailable_reason": unavailable_reason,
            "parameters_json": json.dumps(stage.get("parameters") or {}, sort_keys=True),
            "maps_requested": artifact.get("maps_requested"),
            "maps_enabled": artifact.get("maps_enabled"),
            "maps_unavailable_reason": artifact.get("maps_unavailable_reason", ""),
            "artifact_complete": artifact.get("complete"),
            "artifact_map_count": len(artifact.get("maps") or []),
            "write_error_count": len(artifact.get("write_errors") or []),
            **metrics,
            "source_json": str(path),
        })
    return rows


def load_confidence_adjustment_observability(
    frame_dir: Path,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    path = frame_dir / "confidence_adjustment.json"
    artifact = read_json(path)
    declared = {
        str(row.get("name")): row
        for row in artifact.get("methods") or []
        if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for method_name in CONFIDENCE_ADJUSTMENT_METHODS:
        method = declared.get(method_name, {})
        metrics = method.get("metrics") if isinstance(method.get("metrics"), dict) else {}
        unavailable_reason = str(method.get("unavailable_reason") or "")
        if not artifact:
            unavailable_reason = "artifact_not_produced"
        elif not method:
            unavailable_reason = "method_not_declared_in_artifact"
        enabled = bool(method.get("enabled", False))
        executed = bool(method.get("executed", False))
        output_available = bool(method.get("output_available", False))
        if output_available:
            status = "available"
            quality = str(method.get("quality") or "exact")
        elif not artifact or (unavailable_reason and enabled):
            status = "unavailable"
            quality = "unavailable"
        elif not enabled:
            status = "disabled"
            quality = "unavailable"
        else:
            status = "unavailable"
            quality = "unavailable"
            unavailable_reason = unavailable_reason or "output_map_unavailable"
        rows.append({
            **context,
            "schema_name": artifact.get("schema_name", "openmvs.dmap.confidence_adjustment"),
            "schema_version": artifact.get("schema_version"),
            "algorithm_stage": artifact.get("algorithm_stage", "depth_map_optional_confidence_adjustment"),
            "method": method_name,
            "enabled": enabled,
            "executed": executed,
            "output_available": output_available,
            "artifact_status": status,
            "measurement_quality": quality,
            "measurement_basis": method.get("basis", ""),
            "unavailable_reason": unavailable_reason,
            "input_available": artifact.get("input_available"),
            "depth_validity_unchanged": artifact.get("depth_validity_unchanged"),
            "final_combination": artifact.get("final_combination", ""),
            "neighbor_limit": artifact.get("neighbor_limit"),
            "neighbor_count": len(artifact.get("neighbors") or []),
            "parameters_json": json.dumps(artifact.get("parameters") or {}, sort_keys=True),
            "maps_requested": artifact.get("maps_requested"),
            "maps_enabled": artifact.get("maps_enabled"),
            "maps_unavailable_reason": artifact.get("maps_unavailable_reason", ""),
            "artifact_complete": artifact.get("complete"),
            "artifact_map_count": len(artifact.get("maps") or []),
            "write_error_count": len(artifact.get("write_errors") or []),
            **metrics,
            "source_json": str(path),
        })
    return rows


def load_schema4_observability(run_scenes: list[RunScene]) -> dict[str, pd.DataFrame]:
    table_rows: dict[str, list[dict[str, Any]]] = {
        "exact_observability": [], "exact_iterations": [], "exact_views": [],
        "cpu_view_candidates": [], "cpu_estimation_selection": [],
        "postprocess_filters": [], "confidence_adjustment": [],
        "cuda_resource_plans": [], "filter_resource_plans": [],
    }
    for run_scene in run_scenes:
        depthmaps_root = run_scene.instrumentation_dir / "depthmaps"
        cuda_plan_path = run_scene.instrumentation_dir / "resource_plans.jsonl"
        cuda_plans_by_image: dict[int, list[dict[str, Any]]] = {}
        for plan in read_jsonl(cuda_plan_path):
            parsed_image_id = _nonnegative_integer(plan.get("image_id"))
            image_id = parsed_image_id if parsed_image_id is not None else -1
            cuda_plans_by_image.setdefault(image_id, []).append(plan)
        for frame_dir in sorted(path for path in depthmaps_root.glob("*") if path.is_dir()):
            summary = read_json(frame_dir / "summary.json")
            context = {
                "run": run_scene.label, "role": run_scene.role, "repeat": run_scene.repeat,
                "scene_id": run_scene.scene_id, "frame": frame_dir.name,
                "estimation_stage": run_scene.estimation_stage,
                "geometric_iteration": run_scene.geometric_iteration,
                "image_id": int(summary.get("image_id", -1)),
            }
            manifest_path = frame_dir / "map_manifest.json"
            if not manifest_path.is_file():
                manifest_path = frame_dir / "prefilter_manifest.json"
            manifest = read_json(manifest_path)
            context["pyramid_level"] = map_catalog_context(
                run_scene, frame_dir, summary, manifest if manifest else summary
            ).get("pyramid_level")
            matching_cuda_plans = cuda_plans_by_image.get(context["image_id"], [])
            cuda_required = int(manifest.get("schema_version", summary.get("schema_version", 0)) or 0) >= 4
            if matching_cuda_plans:
                plans_per_level: dict[Any, int] = {}
                for plan in matching_cuda_plans:
                    level = plan.get("pyramid_level", context["pyramid_level"])
                    plans_per_level[level] = plans_per_level.get(level, 0) + 1
                for plan in matching_cuda_plans:
                    plan_context = dict(context)
                    plan_context["pyramid_level"] = plan.get(
                        "pyramid_level", context["pyramid_level"]
                    )
                    table_rows["cuda_resource_plans"].append(cuda_resource_plan_row(
                        plan_context,
                        plan,
                        cuda_plan_path,
                        required=cuda_required,
                        duplicate_count=plans_per_level[plan_context["pyramid_level"]],
                    ))
            else:
                table_rows["cuda_resource_plans"].append(cuda_resource_plan_row(
                    context, {}, cuda_plan_path, required=cuda_required,
                ))
            table_rows["filter_resource_plans"].extend(
                filter_resource_plan_rows(frame_dir, context)
            )
            for name, file_name in (
                ("exact_iterations", "exact_iteration.csv"),
                ("exact_views", "exact_view_summary.csv"),
            ):
                path = frame_dir / file_name
                if not path.is_file():
                    continue
                with path.open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        table_rows[name].append({
                            **context, **{key: value for key, value in row.items()},
                            "source_csv": str(path),
                        })
            exact_path = frame_dir / "exact_observability.json"
            exact_data = read_json(exact_path)
            if exact_data:
                table_rows["exact_observability"].append({
                    **context,
                    "schema_name": exact_data.get("schema_name"),
                    "schema_version": exact_data.get("schema_version"),
                    "num_logical_states": exact_data.get("num_logical_states"),
                    "num_views": exact_data.get("num_views"),
                    "states_json": json.dumps(exact_data.get("states") or [], sort_keys=True),
                    "source_json": str(exact_path),
                })
            candidate_path = frame_dir / "cpu_view_candidates.json"
            candidate_data = read_json(candidate_path)
            for candidate in candidate_data.get("candidates") or []:
                components = candidate.get("score_components") or {}
                table_rows["cpu_view_candidates"].append({
                    **context,
                    "schema_name": candidate_data.get("schema_name"),
                    "candidate_source": candidate_data.get("candidate_source"),
                    "ranking_succeeded": candidate_data.get("ranking_succeeded"),
                    "filter_succeeded": candidate_data.get("filter_succeeded"),
                    **{key: value for key, value in candidate.items() if key != "score_components"},
                    **{f"score_component_{key}": value for key, value in components.items()},
                    "source_json": str(candidate_path),
                })
            selection_path = frame_dir / "cpu_view_estimation_selection.json"
            selection_data = read_json(selection_path)
            selection_mode = selection_data.get("selection_mode")
            selection_schema_version = int(selection_data.get("schema_version") or 1)
            admission_policy = selection_data.get("admission_policy") or {}
            selection_parameters = selection_data.get("parameters") or {}
            if admission_policy:
                admission_policy_name = admission_policy.get("name")
                admission_policy_origin = admission_policy.get("policy_origin")
                score_threshold_applied = admission_policy.get("score_threshold_applied")
                configured_threshold_status = admission_policy.get("configured_score_threshold_status")
            else:
                legacy_ranked_prefix = selection_mode == "ranked_prefix"
                admission_policy_name = (
                    "ranked_prefix_with_score_cutoff" if legacy_ranked_prefix else "explicit_neighbor"
                )
                admission_policy_origin = "legacy_schema_v1_inference"
                score_threshold_applied = legacy_ranked_prefix
                configured_threshold_status = (
                    "applied" if legacy_ranked_prefix else "not_applicable_explicit_neighbor"
                )
            configured_effective_min_score = selection_parameters.get("configured_effective_min_score")
            if configured_effective_min_score is None and selection_schema_version == 1:
                configured_effective_min_score = selection_parameters.get("effective_min_score")
            for candidate in selection_data.get("candidates") or []:
                table_rows["cpu_estimation_selection"].append({
                    **context,
                    "schema_name": selection_data.get("schema_name"),
                    "selection_schema_version": selection_schema_version,
                    "selection_succeeded": selection_data.get("selection_succeeded"),
                    "selection_mode": selection_mode,
                    "admission_policy": admission_policy_name,
                    "admission_policy_origin": admission_policy_origin,
                    "score_threshold_applied": score_threshold_applied,
                    "configured_score_threshold_status": configured_threshold_status,
                    "view_min_score_absolute": selection_parameters.get("view_min_score_absolute"),
                    "view_min_score_ratio": selection_parameters.get("view_min_score_ratio"),
                    "effective_min_score": selection_parameters.get("effective_min_score"),
                    "configured_effective_min_score": configured_effective_min_score,
                    "ordered_cutoff_reason": (selection_data.get("ordered_cutoff") or {}).get("reason"),
                    **candidate,
                    "source_json": str(selection_path),
                })
            table_rows["postprocess_filters"].extend(
                load_postprocess_observability(frame_dir, context)
            )
            table_rows["confidence_adjustment"].extend(
                load_confidence_adjustment_observability(frame_dir, context)
            )
    tables: dict[str, pd.DataFrame] = {}
    for name, rows in table_rows.items():
        table = (
            pd.DataFrame(rows)
            if rows else pd.DataFrame(columns=SCHEMA4_EMPTY_TABLE_COLUMNS.get(name, ()))
        )
        for column in table.columns:
            if column in {"run", "role", "scene_id", "frame", "stage", "stage_name", "method", "algorithm_stage", "artifact_status", "measurement_quality", "measurement_basis", "unavailable_reason", "maps_unavailable_reason", "parameters_json", "final_combination", "plan_kind", "component", "decision", "reason", "validation_errors_json", "validation_warnings_json", "source_image_name", "candidate_image_name", "filter_decision", "initial_decision", "schema_name", "candidate_source", "selection_mode", "admission_policy", "admission_policy_origin", "configured_score_threshold_status", "ordered_cutoff_reason", "source_csv", "source_json"}:
                continue
            converted = pd.to_numeric(table[column], errors="coerce")
            nonempty = table[column].notna() & table[column].astype(str).ne("")
            if not nonempty.any() or converted[nonempty].notna().all():
                table[column] = converted
        tables[name] = table
    resource_tables = [
        tables[name]
        for name in ("cuda_resource_plans", "filter_resource_plans")
        if not tables[name].empty
    ]
    if resource_tables:
        resources = pd.concat(resource_tables, ignore_index=True, sort=False)
        tables["resource_plan_validation"] = resources[
            [
                column for column in RESOURCE_PLAN_VALIDATION_COLUMNS
                if column in resources.columns
            ]
        ].copy()
        for column in ("required", "available", "valid"):
            if column in tables["resource_plan_validation"]:
                def nullable_boolean(value: Any) -> Any:
                    if value is None or (
                        not isinstance(value, (list, tuple, dict, np.ndarray))
                        and bool(pd.isna(value))
                    ):
                        return pd.NA
                    if isinstance(value, (bool, np.bool_)):
                        return bool(value)
                    if isinstance(value, (int, float, np.integer, np.floating)) and value in {0, 1}:
                        return bool(value)
                    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                        return value.strip().lower() == "true"
                    raise ValueError(
                        f"resource-plan {column} contains non-boolean value {value!r}"
                    )

                tables["resource_plan_validation"][column] = (
                    tables["resource_plan_validation"][column]
                    .map(nullable_boolean)
                    .astype("boolean")
                )
    else:
        tables["resource_plan_validation"] = pd.DataFrame(
            columns=RESOURCE_PLAN_VALIDATION_COLUMNS
        )
    return tables


def build_exact_cost_evolution(map_catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if map_catalog.empty:
        return pd.DataFrame(columns=EXACT_COST_EVOLUTION_COLUMNS)
    signals = set(SCHEMA4_EXACT_STATE_SIGNALS) | {
        "cost_stored",
        "candidate_raw_best_cost_exact",
        "candidate_raw_runner_up_cost_exact",
        "gap_raw_best_runner_up_exact",
        "candidate_retained_minus_raw_best_exact",
    }
    selected = map_catalog[
        map_catalog["signal"].isin(signals)
        & map_catalog["available"].astype(bool)
        & map_catalog["logical_iteration"].notna()
        & (pd.to_numeric(map_catalog["manifest_schema_version"], errors="coerce") >= 4)
    ]
    for artifact in selected.to_dict("records"):
        path = Path(str(artifact.get("path", "")))
        if path.suffix.lower() != ".pfm" or not path.is_file():
            continue
        try:
            values = read_pfm(path)
        except Exception:
            continue
        if values.ndim == 3:
            values = values[..., 0]
        valid = np.isfinite(values)
        unavailable_value = safe_float(artifact.get("unavailable_value"))
        if unavailable_value is not None:
            valid &= values != unavailable_value
        sample = values[valid].astype(np.float64, copy=False)
        rows.append({
            **{key: artifact.get(key) for key in (
                "run", "run_role", "repeat", "scene_id", "frame", "image_id",
                "estimation_stage", "geometric_iteration", "signal", "logical_iteration", "stage",
                "pyramid_level", "measurement_quality", "measurement_basis",
            )},
            "pixels": int(values.size), "available_pixels": int(sample.size),
            "available_ratio": float(sample.size / values.size) if values.size else 0.0,
            "mean": float(sample.mean()) if sample.size else None,
            "median": float(np.median(sample)) if sample.size else None,
            "p90": float(np.percentile(sample, 90)) if sample.size else None,
            "min": float(sample.min()) if sample.size else None,
            "max": float(sample.max()) if sample.size else None,
            "source_map": str(path),
        })
    return (
        pd.DataFrame(rows)
        if rows else pd.DataFrame(columns=EXACT_COST_EVOLUTION_COLUMNS)
    )


def validate_run_instrumentation(run_scenes: list[RunScene]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    endpoint_directory_parity: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    terminal_identities = {
        (
            stage.label,
            stage.role,
            stage.repeat,
            stage.scene_id,
            stage.estimation_stage,
            stage.geometric_iteration,
        )
        for stage in select_terminal_run_scenes(run_scenes)
    }
    for run_scene in run_scenes:
        terminal_stage = (
            run_scene.label,
            run_scene.role,
            run_scene.repeat,
            run_scene.scene_id,
            run_scene.estimation_stage,
            run_scene.geometric_iteration,
        ) in terminal_identities
        if not terminal_stage:
            cross_parity_status = "not_applicable"
            cross_parity_reason = "cross-capture parity is evaluated only for the terminal estimation stage"
        elif not run_scene.cross_capture_parity_compatible:
            cross_parity_status = "not_applicable"
            cross_parity_reason = (
                run_scene.cross_capture_parity_reason
                or "capture production command signatures are incompatible"
            )
        else:
            cross_parity_status = "unavailable"
            cross_parity_reason = "matching capture depth map is unavailable"
        depthmaps_root = run_scene.instrumentation_dir / "depthmaps"
        for frame_dir in sorted(path for path in depthmaps_root.glob("*") if path.is_dir()):
            summary = read_json(frame_dir / "summary.json")
            image_id = int(summary.get("image_id", -1))
            map_capture = (
                (frame_dir / "map_manifest.json").is_file()
                or (frame_dir / "prefilter_manifest.json").is_file()
            )
            instrumented_dmap = None
            if terminal_stage and run_scene.depth_map_dir is not None and image_id >= 0:
                candidate = run_scene.depth_map_dir / f"depth{image_id:04d}.dmap"
                if candidate.is_file():
                    instrumented_dmap = candidate
            parity_dmap = None
            timing_depth_map_dir = instrumentation_depth_map_dir(run_scene.timing_dir)
            if (
                map_capture
                and terminal_stage
                and run_scene.cross_capture_parity_compatible
                and timing_depth_map_dir is not None
                and timing_depth_map_dir != run_scene.depth_map_dir
                and image_id >= 0
            ):
                candidate = timing_depth_map_dir / f"depth{image_id:04d}.dmap"
                if candidate.is_file():
                    parity_dmap = candidate
            if not map_capture:
                maps_summary_parity_status = "not_applicable"
                maps_summary_parity_reason = (
                    "summary-only capture has no maps; maps/summary parity requires a distinct maps capture"
                )
            else:
                maps_summary_parity_status = (
                    "checked" if parity_dmap is not None else cross_parity_status
                )
                maps_summary_parity_reason = (
                    "" if parity_dmap is not None else cross_parity_reason
                )
            try:
                result = instrumentation_validator.validate(
                    instrumentation_validator.Arguments(
                        frame_dir=frame_dir,
                        instrumented_dmap=instrumented_dmap,
                        reference_dmap=parity_dmap if instrumented_dmap is not None else None,
                    )
                )
                failed_checks = [
                    str(check.get("name", "unknown"))
                    for check in result.get("checks") or []
                    if not check.get("passed")
                ]
                endpoint_parity: dict[str, float] = {}
                endpoint_depth_map_dir = (
                    sibling_capture_depth_map_dir(run_scene.instrumentation_dir, "endpoint")
                    if run_scene.cross_capture_parity_compatible else None
                )
                endpoint_dmap = (
                    endpoint_depth_map_dir / f"depth{image_id:04d}.dmap"
                    if endpoint_depth_map_dir is not None and image_id >= 0 else None
                )
                endpoint_directory_result: dict[str, Any] = {}
                endpoint_key = (
                    run_scene.label, run_scene.role, run_scene.repeat, run_scene.scene_id
                )
                if (
                    terminal_stage
                    and run_scene.depth_map_dir is not None
                    and endpoint_depth_map_dir is not None
                    and endpoint_depth_map_dir.is_dir()
                ):
                    endpoint_directory_result = endpoint_directory_parity.setdefault(
                        endpoint_key,
                        compare_dmap_directories(
                            run_scene.depth_map_dir, endpoint_depth_map_dir
                        ),
                    )
                    if not endpoint_directory_result.get("bit_exact", False):
                        failed_checks.append("production_endpoint_dmap_set_bit_exact")
                if (
                    terminal_stage
                    and instrumented_dmap is not None
                    and endpoint_dmap is not None
                    and endpoint_dmap.is_file()
                ):
                    instrumented_values = instrumentation_validator.load_dmap(instrumented_dmap)
                    endpoint_values = instrumentation_validator.load_dmap(endpoint_dmap)
                    for key in ("depth_map", "normal_map", "confidence_map"):
                        if key in instrumented_values and key in endpoint_values:
                            endpoint_parity[key] = instrumentation_validator.max_abs_difference(
                                instrumented_values[key], endpoint_values[key]
                            )
                    if len(endpoint_parity) != 3 or any(value != 0.0 for value in endpoint_parity.values()):
                        failed_checks.append("production_endpoint_parity")
                endpoint_parity_status = (
                    "checked" if endpoint_parity else cross_parity_status
                )
                endpoint_parity_reason = (
                    "" if endpoint_parity else cross_parity_reason
                )
                endpoint_dmap_set_status = (
                    "checked" if endpoint_directory_result else cross_parity_status
                )
                endpoint_dmap_set_reason = (
                    "" if endpoint_directory_result else cross_parity_reason
                )
                quality_comparison_eligible = (
                    terminal_stage
                    and not run_scene.diagnostic_only
                    and endpoint_directory_result.get("bit_exact") is True
                )
                if run_scene.diagnostic_only:
                    quality_ineligible_reason = (
                        run_scene.diagnostic_only_reason
                        or "diagnostic capture profiles are not quality authorities"
                    )
                elif not terminal_stage:
                    quality_ineligible_reason = (
                        "quality comparison uses only the terminal estimation stage"
                    )
                elif not endpoint_directory_result:
                    quality_ineligible_reason = (
                        "complete production endpoint DMAP-set parity is unavailable"
                    )
                elif endpoint_directory_result.get("bit_exact") is not True:
                    quality_ineligible_reason = (
                        "terminal DMAP set is not bit-exact to the production endpoint"
                    )
                else:
                    quality_ineligible_reason = ""
                disposition = process_specialization_validation_disposition(
                    validator_valid=bool(result.get("valid")),
                    failed_checks=failed_checks,
                    diagnostic_only=run_scene.diagnostic_only,
                    allow_divergence=(
                        run_scene.allow_process_specialization_divergence_for_diagnostics
                    ),
                )
                rows.append({
                    "run": run_scene.label,
                    "role": run_scene.role,
                    "repeat": run_scene.repeat,
                    "scene_id": run_scene.scene_id,
                    "frame": frame_dir.name,
                    "image_id": image_id,
                    "estimation_stage": run_scene.estimation_stage,
                    "geometric_iteration": run_scene.geometric_iteration,
                    "schema_version": result.get("schema_version"),
                    "capture_kind": result.get("capture_kind", "maps"),
                    "maps_available": result.get("maps_available", map_capture),
                    "maps_unavailable_reason": result.get("maps_unavailable_reason", ""),
                    "exact_maps_available": result.get("exact_maps_available", False),
                    "exact_maps_unavailable_reason": result.get("exact_maps_unavailable_reason", ""),
                    "logical_iterations": result.get("logical_iterations") or [],
                    "terminal_stage": terminal_stage,
                    "cross_capture_parity_compatible": run_scene.cross_capture_parity_compatible,
                    "cross_capture_parity_reason": run_scene.cross_capture_parity_reason,
                    "diagnostic_only": run_scene.diagnostic_only,
                    "diagnostic_only_reason": run_scene.diagnostic_only_reason,
                    "allow_process_specialization_divergence_for_diagnostics": (
                        run_scene.allow_process_specialization_divergence_for_diagnostics
                    ),
                    **disposition,
                    "failed_checks": json.dumps(failed_checks),
                    "manifest_map_count": result.get("manifest_map_count"),
                    "instrumented_dmap_checked": result.get(
                        "instrumented_dmap_checked", instrumented_dmap is not None
                    ),
                    "maps_summary_parity_checked": parity_dmap is not None,
                    "maps_summary_parity_status": maps_summary_parity_status,
                    "maps_summary_parity_reason": maps_summary_parity_reason,
                    "parity_max_abs": json.dumps(result.get("parity_max_abs") or {}, sort_keys=True),
                    "endpoint_parity_checked": bool(endpoint_parity),
                    "endpoint_parity_status": endpoint_parity_status,
                    "endpoint_parity_reason": endpoint_parity_reason,
                    "endpoint_parity_max_abs": json.dumps(endpoint_parity, sort_keys=True),
                    "endpoint_dmap_set_checked": bool(endpoint_directory_result),
                    "endpoint_dmap_set_status": endpoint_dmap_set_status,
                    "endpoint_dmap_set_reason": endpoint_dmap_set_reason,
                    "endpoint_dmap_set_basis": endpoint_directory_result.get("first_basis"),
                    "endpoint_dmap_set_bit_exact": endpoint_directory_result.get("bit_exact"),
                    "endpoint_dmap_set_shared_count": endpoint_directory_result.get("shared_count"),
                    "endpoint_dmap_set_mismatched": json.dumps(
                        endpoint_directory_result.get("mismatched") or []
                    ),
                    "endpoint_dmap_set_missing": json.dumps({
                        "deep": endpoint_directory_result.get("missing_from_first") or [],
                        "endpoint": endpoint_directory_result.get("missing_from_second") or [],
                    }, sort_keys=True),
                    "quality_comparison_eligible": quality_comparison_eligible,
                    "quality_comparison_ineligible_reason": quality_ineligible_reason,
                    "logical_state_validation": json.dumps(
                        result.get("logical_state_validation") or {}, sort_keys=True
                    ),
                    "validation_warnings": json.dumps(
                        result.get("warnings") or [], sort_keys=True
                    ),
                    "validation_error": "",
                })
            except Exception as exc:
                rows.append({
                    "run": run_scene.label,
                    "role": run_scene.role,
                    "repeat": run_scene.repeat,
                    "scene_id": run_scene.scene_id,
                    "frame": frame_dir.name,
                    "image_id": image_id,
                    "estimation_stage": run_scene.estimation_stage,
                    "geometric_iteration": run_scene.geometric_iteration,
                    "schema_version": None,
                    "capture_kind": "maps" if map_capture else "summary_only",
                    "maps_available": map_capture,
                    "maps_unavailable_reason": "" if map_capture else "summary_capture_validation_failed",
                    "exact_maps_available": False,
                    "exact_maps_unavailable_reason": "validation_failed",
                    "logical_iterations": [],
                    "terminal_stage": terminal_stage,
                    "cross_capture_parity_compatible": run_scene.cross_capture_parity_compatible,
                    "cross_capture_parity_reason": run_scene.cross_capture_parity_reason,
                    "diagnostic_only": run_scene.diagnostic_only,
                    "diagnostic_only_reason": run_scene.diagnostic_only_reason,
                    "allow_process_specialization_divergence_for_diagnostics": (
                        run_scene.allow_process_specialization_divergence_for_diagnostics
                    ),
                    "valid": False,
                    "report_generation_allowed": False,
                    "production_qualification_status": "failed",
                    "process_specialization_divergence": False,
                    "failed_checks": "[]",
                    "manifest_map_count": None,
                    "instrumented_dmap_checked": instrumented_dmap is not None,
                    "maps_summary_parity_checked": parity_dmap is not None,
                    "maps_summary_parity_status": maps_summary_parity_status,
                    "maps_summary_parity_reason": maps_summary_parity_reason,
                    "parity_max_abs": "{}",
                    "endpoint_parity_checked": False,
                    "endpoint_parity_status": cross_parity_status,
                    "endpoint_parity_reason": cross_parity_reason,
                    "endpoint_parity_max_abs": "{}",
                    "endpoint_dmap_set_checked": False,
                    "endpoint_dmap_set_status": cross_parity_status,
                    "endpoint_dmap_set_reason": cross_parity_reason,
                    "endpoint_dmap_set_basis": None,
                    "endpoint_dmap_set_bit_exact": None,
                    "endpoint_dmap_set_shared_count": None,
                    "endpoint_dmap_set_mismatched": "[]",
                    "endpoint_dmap_set_missing": "{}",
                    "quality_comparison_eligible": False,
                    "quality_comparison_ineligible_reason": (
                        "instrumentation validation failed before production endpoint "
                        "parity could be established"
                    ),
                    "logical_state_validation": "{}",
                    "validation_warnings": "[]",
                    "validation_error": str(exc),
                })
    return pd.DataFrame(rows)


def summarize_instrumentation_validation(
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize strict validity separately from the diagnostic report gate."""

    invalid = [row for row in validation_records if not bool(row.get("valid"))]
    diagnostic_divergence = [
        row for row in invalid
        if bool(row.get("process_specialization_divergence"))
    ]
    fatal = [
        row for row in invalid
        if not bool(row.get("report_generation_allowed"))
    ]
    maps_summary_checked = [
        row for row in validation_records
        if bool(row.get("maps_summary_parity_checked"))
    ]
    maps_summary_bit_exact = 0
    for row in maps_summary_checked:
        try:
            failures = set(json.loads(str(row.get("failed_checks") or "[]")))
        except json.JSONDecodeError:
            failures = {"malformed_failed_checks"}
        if "production_output_parity" not in failures:
            maps_summary_bit_exact += 1
    quality_rows = [
        row for row in validation_records
        if bool(row.get("terminal_stage")) and not bool(row.get("diagnostic_only"))
    ]
    quality_eligible = [
        row for row in quality_rows
        if bool(row.get("quality_comparison_eligible"))
    ]
    endpoint_parity_qualified = (
        bool(quality_rows) and len(quality_eligible) == len(quality_rows)
    )
    if fatal:
        qualification_status = "failed"
    elif diagnostic_divergence:
        qualification_status = "failed_allowed_diagnostic_only"
    elif invalid:
        qualification_status = "failed"
    elif not endpoint_parity_qualified:
        qualification_status = "unqualified_endpoint_parity"
    else:
        qualification_status = "passed"
    return {
        "valid": bool(validation_records) and not invalid,
        "report_generation_allowed": bool(validation_records) and not fatal,
        "production_parity_qualified": (
            bool(validation_records) and not invalid and endpoint_parity_qualified
        ),
        "production_qualification_status": qualification_status,
        "invalid_frames": len(invalid),
        "fatal_frames": len(fatal),
        "diagnostic_process_specialization_divergence_frames": len(
            diagnostic_divergence
        ),
        "maps_summary_frames_bit_exact": maps_summary_bit_exact,
        "quality_comparison_frames": len(quality_rows),
        "quality_comparison_frames_eligible": len(quality_eligible),
    }


def dataframe_json_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in dataframe.to_dict("records"):
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            normalized[key] = None if missing else value
        records.append(normalized)
    return records


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"invalid PFM header: {path}")
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        width, height = [int(value) for value in dimensions.split()]
        scale = float(handle.readline().decode("ascii").strip())
        data = np.fromfile(handle, dtype="<f4" if scale < 0 else ">f4")
    pixels = width * height
    if pixels <= 0 or data.size % pixels:
        raise ValueError(f"invalid PFM payload size: {path}")
    channels = data.size // pixels
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.flipud(data.reshape(shape)).copy()


def scene_context(config: dict[str, Any], scene_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    scene_config = next(
        (
            row for row in config.get("scenes") or []
            if str(row.get("scan_id") or "") == scene_id
        ),
        {},
    )
    sidecar_raw = scene_config.get("annotation_sidecar")
    if sidecar_raw:
        sidecar_path = Path(str(sidecar_raw)).expanduser()
        if not sidecar_path.is_absolute():
            sidecar_path = Path(str(config["_config_path"])).parent / sidecar_path
        sidecar = annotation_fit.load_annotation_sidecar(sidecar_path.resolve(), scene_id)
        return (
            sidecar["scene"],
            sidecar["review"],
            sidecar["pipeline"],
            sidecar["image_mapping"],
        )
    if not config.get("dataset_root"):
        raise ValueError(
            f"annotation provider is not configured for scene {scene_id!r}; "
            "set scenes[].annotation_sidecar or dataset_root"
        )
    dataset_root = config_path(config, "dataset_root")
    cache_root = (
        config_path(config, "cache_root")
        if config.get("cache_root")
        else Path(str(config["_config_path"])).parent
    )
    db = annotation_fit.annotation_db_path(dataset_root)
    scan_rows = read_jsonl(db / "scans.jsonl") + read_jsonl(cache_root / scene_id / "annotations" / "scans.jsonl")
    review_rows = read_jsonl(db / "scan_reviewers.jsonl") + read_jsonl(cache_root / scene_id / "annotations" / "scan_reviewers.jsonl")
    pipeline_rows = read_jsonl(db / "scan_pipelines.jsonl") + read_jsonl(cache_root / scene_id / "annotations" / "scan_pipelines.jsonl")
    scan_row = annotation_fit.find_row(scan_rows, "id", scene_id)
    review_row = annotation_fit.find_row(review_rows, "scan_id", scene_id)
    pipeline_row = annotation_fit.find_row(pipeline_rows, "scan_id", scene_id)
    if not scan_row or not review_row or not pipeline_row:
        raise ValueError(f"missing annotation context for scene {scene_id}")
    mapping, _source = annotation_fit.load_image_mapping(
        scan_id=scene_id,
        dataset_root=dataset_root,
        cache_root=cache_root,
        explicit_mapping=None,
    )
    return scan_row, review_row, pipeline_row, mapping


def find_reference_dmap(
    run_scene: RunScene,
    image_id: int,
    scene_config: dict[str, Any] | None,
) -> Path | None:
    """Find calibration/reference data without borrowing another run's output."""
    candidates: list[Path] = []
    if run_scene.depth_map_dir:
        candidates.append(run_scene.depth_map_dir / f"depth{image_id:04d}.dmap")
    if scene_config:
        reference_dir = as_path(scene_config.get("reference_dmap_dir"))
        if reference_dir:
            candidates.append(reference_dir / f"depth{image_id:04d}.dmap")
    return next((path for path in candidates if path.is_file()), None)


def dmap_for_stage(reference: dict[str, Any], depth_path: Path | None) -> dict[str, Any]:
    dmap = dict(reference)
    if depth_path is not None:
        depth = read_pfm(depth_path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        dmap["depth_map"] = depth.astype(np.float32, copy=False)
        dmap["depth_height"], dmap["depth_width"] = depth.shape
    return dmap


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def make_annotation_fit_config(evaluation: dict[str, Any]) -> annotation_fit.FitConfig:
    source_width = evaluation.get("edge_ribbon_source_px")
    angular_width = evaluation.get("edge_ribbon_angle_mrad")
    legacy_depth_width = evaluation.get("edge_ribbon_px")
    if source_width is not None:
        # Source-image pixels are resolution-aware and take precedence during
        # migration from the legacy fixed depth-grid width.
        legacy_depth_width = None
        angular_width = None
    elif angular_width is not None:
        legacy_depth_width = None
    elif legacy_depth_width is None:
        source_width = 5.0
    fit = annotation_fit.FitConfig(
        ransac_threshold_m=float(evaluation.get("ransac_threshold_m", PRIMARY_RANSAC_THRESHOLD_M)),
        ransac_thresholds_m=tuple(float(value) for value in evaluation.get("ransac_thresholds_m", RANSAC_THRESHOLDS_M)),
        edge_ribbon_px=float(legacy_depth_width) if legacy_depth_width is not None else None,
        max_ransac_points=int(evaluation.get("max_ransac_points", 50000)),
        ransac_trials=int(evaluation.get("ransac_trials", 2000)),
        seed=int(evaluation.get("seed", 0)),
        min_confidence=None,
        plane_grid=int(evaluation.get("plane_grid", 32)),
        line_bins=int(evaluation.get("line_bins", 100)),
        max_visual_points=int(evaluation.get("max_visual_points", 6000)),
        edge_ribbon_source_px=float(source_width) if source_width is not None else None,
        edge_ribbon_angle_mrad=float(angular_width) if angular_width is not None else None,
    )
    annotation_fit.validate_fit_config(fit)
    return fit


def annotation_valid_points(
    *,
    dmap: dict[str, Any],
    valid_mask: np.ndarray,
    transformed_points: np.ndarray,
    kind: str,
    edge_ribbon_radius_px: float | None,
) -> np.ndarray:
    """Reconstruct candidate points for fixed-baseline-model evaluation."""

    if kind == "plane":
        mask = annotation_fit.polygon_mask(dmap["depth_map"].shape, transformed_points)
    elif kind == "edge" and edge_ribbon_radius_px is not None:
        mask, _t_map = annotation_fit.edge_ribbon_mask(
            dmap["depth_map"].shape,
            transformed_points,
            float(edge_ribbon_radius_px),
        )
    else:
        return np.empty((0, 3), dtype=np.float64)
    ys, xs = np.nonzero(mask & valid_mask)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    return annotation_fit.unproject_pixels(dmap, xs, ys)


def apply_baseline_model_cross_evaluation(
    candidate: dict[str, Any],
    points: np.ndarray,
    baseline_model: dict[str, Any],
    config: annotation_fit.FitConfig,
) -> None:
    """Evaluate one fixed baseline RANSAC model on candidate-run points."""

    prefix = "baseline_model_on_candidate_"
    kind = str(candidate.get("annotation_kind") or "")
    try:
        if kind == "plane":
            normal = np.asarray([
                baseline_model["plane_normal_x"],
                baseline_model["plane_normal_y"],
                baseline_model["plane_normal_z"],
            ], dtype=np.float64)
            point = np.asarray([
                baseline_model["model_point_x"],
                baseline_model["model_point_y"],
                baseline_model["model_point_z"],
            ], dtype=np.float64)
            residuals = annotation_fit.plane_residuals(points, normal, point)
        elif kind == "edge":
            direction = np.asarray([
                baseline_model["line_direction_x"],
                baseline_model["line_direction_y"],
                baseline_model["line_direction_z"],
            ], dtype=np.float64)
            point = np.asarray([
                baseline_model["model_point_x"],
                baseline_model["model_point_y"],
                baseline_model["model_point_z"],
            ], dtype=np.float64)
            residuals = annotation_fit.line_residuals(points, direction, point)
        else:
            raise ValueError(f"unsupported annotation kind {kind!r}")
    except (KeyError, TypeError, ValueError) as exc:
        candidate[f"{prefix}status"] = "unavailable_malformed_baseline_model"
        candidate[f"{prefix}reason"] = f"{type(exc).__name__}: {exc}"
        return
    if not len(points):
        candidate[f"{prefix}status"] = "unavailable_no_candidate_points"
        candidate[f"{prefix}reason"] = "candidate annotation mask has no valid depth"
        return
    finite = residuals[np.isfinite(residuals)]
    if not len(finite):
        candidate[f"{prefix}status"] = "unavailable_nonfinite_residuals"
        candidate[f"{prefix}reason"] = "fixed baseline model produced no finite residuals"
        return
    candidate[f"{prefix}status"] = "available"
    candidate[f"{prefix}reason"] = ""
    candidate[f"{prefix}schema_version"] = 1
    candidate[f"{prefix}source_run"] = baseline_model.get("run")
    candidate[f"{prefix}point_count"] = int(len(finite))
    for name, value in annotation_fit.residual_stats(
        finite, "all_residual"
    ).items():
        candidate[f"{prefix}{name}"] = value
    for name, value in annotation_fit.threshold_curve_stats(
        finite, config.ransac_thresholds_m
    ).items():
        candidate[f"{prefix}{name}"] = value
        if name.startswith("inlier_fraction_"):
            candidate[f"{prefix}effective_{name.replace('inlier_fraction_', 'inlier_coverage_')}"] = (
                float(candidate.get("coverage_fraction") or 0.0) * value
            )


def product_reference_annotation_inputs(
    config: dict[str, Any],
    frame_rows: list[dict[str, Any]],
) -> tuple[list[RunScene], list[dict[str, Any]]]:
    """Expose archived DMAPs as end-metric-only annotation evidence."""

    reference_config = config.get("product_reference") or {}
    if reference_config.get("enabled", True) is False:
        return [], []
    label = str(reference_config.get("label", "archived_product_reference"))
    required = bool(reference_config.get("required", False))
    scene_configs = {str(row["scan_id"]): row for row in config.get("scenes") or []}
    reference_scenes: list[RunScene] = []
    reference_frames: list[dict[str, Any]] = []
    seen_frames: set[tuple[str, int]] = set()
    for frame in frame_rows:
        scene_id = str(frame.get("scene_id", ""))
        image_id = int(frame.get("image_id", -1))
        key = (scene_id, image_id)
        if not scene_id or image_id < 0 or key in seen_frames:
            continue
        reference_dir = as_path(scene_configs.get(scene_id, {}).get("reference_dmap_dir"))
        reference_dmap = (
            reference_dir / f"depth{image_id:04d}.dmap"
            if reference_dir is not None else None
        )
        if reference_dmap is None or not reference_dmap.is_file():
            if required and reference_dir is not None:
                raise FileNotFoundError(
                    f"required product-reference DMAP does not exist: {reference_dmap}"
                )
            continue
        seen_frames.add(key)
        reference_frames.append({
            **frame,
            "run": label,
            "role": "reference",
            "repeat": 0,
            "depthmap_dir": "",
            "annotation_stages": ["post_filter"],
        })
    for scene_id in sorted({scene_id for scene_id, _image_id in seen_frames}):
        reference_dir = as_path(scene_configs[scene_id].get("reference_dmap_dir"))
        reference_scenes.append(RunScene(
            label=label,
            role="reference",
            repeat=0,
            scene_id=scene_id,
            instrumentation_dir=Path(),
            depth_map_dir=reference_dir,
            timing_dir=None,
        ))
    return reference_scenes, reference_frames


def evaluate_annotations(
    config: dict[str, Any],
    run_scenes: list[RunScene],
    frame_rows: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    evaluation = config.get("evaluation") or {}
    base_fit = make_annotation_fit_config(evaluation)
    scene_configs = {str(row["scan_id"]): row for row in config.get("scenes") or []}
    contexts: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]] = {}
    run_lookup = {(row.label, row.repeat, row.scene_id): row for row in run_scenes}
    results: list[dict[str, Any]] = []
    configured_runs = config.get("runs") or []
    baseline_labels = [
        str(row["label"]) for row in configured_runs if row.get("role") == "baseline"
    ]
    baseline_label = baseline_labels[0] if len(baseline_labels) == 1 else ""
    candidate_labels = {
        str(row["label"]) for row in configured_runs
        if str(row.get("label") or "") != baseline_label
    }
    baseline_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    baseline_models: dict[tuple[Any, ...], dict[str, Any]] = {}
    pending_cross_evaluations: dict[
        tuple[Any, ...], list[tuple[dict[str, Any], np.ndarray, annotation_fit.FitConfig]]
    ] = {}

    def cross_identity(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(row.get("repeat", 0)), str(row.get("scene_id") or ""),
            int(row.get("image_id", -1)), str(row.get("stage") or ""),
            str(row.get("annotation_kind") or ""), str(row.get("object_id") or ""),
            str(row.get("chunk_id") or ""),
        )

    for frame in frame_rows:
        scene_id = str(frame["scene_id"])
        try:
            if scene_id not in contexts:
                contexts[scene_id] = scene_context(config, scene_id)
            _scan, review, pipeline, mapping = contexts[scene_id]
        except Exception as exc:
            results.append({
                "run": frame["run"], "repeat": frame["repeat"], "scene_id": scene_id,
                "image_id": frame["image_id"], "fit_status": "scene_context_error", "error": str(exc),
            })
            continue
        image_to_frame = {annotation_fit.normalize_image_name(name): frame_id for frame_id, name in mapping.items()}
        image_name = annotation_fit.normalize_image_name(str(frame["image_name"]))
        frame_uuid = image_to_frame.get(image_name)
        if not frame_uuid:
            results.append({
                "run": frame["run"], "role": frame["role"], "repeat": frame["repeat"],
                "scene_id": scene_id, "image_id": frame["image_id"], "image_name": image_name,
                "fit_status": "missing_annotation_image_mapping",
                "error": "instrumented image is not present in the annotation frame mapping",
            })
            continue
        chunks = annotation_fit.rows_for_frame(review, frame_uuid)
        if not chunks:
            results.append({
                "run": frame["run"], "role": frame["role"], "repeat": frame["repeat"],
                "scene_id": scene_id, "image_id": frame["image_id"], "image_name": image_name,
                "frame_uuid": frame_uuid, "fit_status": "no_frame_annotations",
                "error": "the mapped frame has no line or plane annotation chunks",
            })
            continue
        run_scene = run_lookup[(str(frame["run"]), int(frame["repeat"]), scene_id)]
        reference_path = find_reference_dmap(
            run_scene, int(frame["image_id"]), scene_configs.get(scene_id)
        )
        if reference_path is None:
            results.append({
                "run": frame["run"], "repeat": frame["repeat"], "scene_id": scene_id,
                "image_id": frame["image_id"], "fit_status": "missing_reference_dmap",
            })
            continue
        reference = annotation_fit.load_dmap(reference_path)
        depthmap_dir = Path(str(frame["depthmap_dir"]))
        stages = {
            "post_filter": depthmap_dir / "maps" / "depth_final_after_filter.pfm",
            "pre_filter": depthmap_dir / "maps" / "depth_final_before_filter.pfm",
        }
        requested_stages = {
            str(value) for value in frame.get("annotation_stages", stages.keys())
        }
        for stage, stage_path in stages.items():
            if stage not in requested_stages:
                continue
            production_dmap = (
                run_scene.depth_map_dir / f"depth{int(frame['image_id']):04d}.dmap"
                if run_scene.depth_map_dir else None
            )
            if stage == "post_filter" and production_dmap is not None and production_dmap.is_file():
                dmap = annotation_fit.load_dmap(production_dmap)
            elif stage_path.is_file():
                dmap = dmap_for_stage(reference, stage_path)
            else:
                results.append({
                    "run": frame["run"], "role": frame["role"], "repeat": frame["repeat"],
                    "scene_id": scene_id, "image_id": frame["image_id"], "image_name": image_name,
                    "frame_uuid": frame_uuid, "stage": stage,
                    "fit_status": "missing_stage_depth_map",
                    "error": f"no depth map produced by this run for {stage}",
                })
                continue
            valid = annotation_fit.valid_depth_mask(dmap, None)
            for chunk_index, item in enumerate(chunks):
                chunk = item["chunk"]
                kind = str(item["kind"])
                chunk_id = str(chunk.get("id") or f"chunk_{chunk_index}")
                visual_dir = output_dir / "visualizations" / "annotations" / str(frame["run"]) / scene_id / f"{int(frame['image_id']):04d}" / stage
                fit_config = replace(base_fit, seed=stable_seed(base_fit.seed, scene_id, frame_uuid, chunk_id, stage))
                evaluation_points: np.ndarray | None = None
                try:
                    raw_points = annotation_fit.chunk_points(chunk, kind)
                    transformed = annotation_fit.annotation_points_to_depth_pixels(
                        raw_points, dmap, pipeline, str(evaluation.get("annotation_space", "distorted"))
                    )
                    edge_radius = None
                    edge_mode = None
                    if kind == "edge":
                        edge_radius, edge_mode = annotation_fit.edge_ribbon_radius_depth_px(
                            raw_segment=raw_points,
                            dmap=dmap,
                            pipeline_row=pipeline,
                            annotation_space=str(evaluation.get("annotation_space", "distorted")),
                            config=fit_config,
                        )
                    if str(frame["run"]) in candidate_labels:
                        evaluation_points = annotation_valid_points(
                            dmap=dmap,
                            valid_mask=valid,
                            transformed_points=transformed,
                            kind=kind,
                            edge_ribbon_radius_px=edge_radius,
                        )
                    result = annotation_fit.evaluate_chunk(
                        dmap=dmap,
                        valid_mask=valid,
                        transformed_points=transformed,
                        kind=kind,
                        config=fit_config,
                        save_mask_path=None,
                        visual_dir=visual_dir,
                        visual_prefix=f"{chunk_index:03d}_{kind}_{chunk_id[:8]}",
                        visual_title=f"{frame['run']} / {scene_id[:8]} / {image_name} / {kind} {chunk_id[:8]} / {stage}",
                        edge_ribbon_radius_px=edge_radius,
                        edge_ribbon_mode=edge_mode,
                    )
                except Exception as exc:
                    result = {
                        "fit_status": "failed_annotation_evaluation",
                        "failure_type": type(exc).__name__,
                        "failure_reason": str(exc),
                        "error": str(exc),
                    }
                result.update({
                    "run": frame["run"], "role": frame["role"], "repeat": frame["repeat"],
                    "scene_id": scene_id, "image_id": frame["image_id"], "image_name": image_name,
                    "frame_uuid": frame_uuid, "stage": stage, "annotation_kind": kind,
                    "object_id": str(item.get("object_id", "")), "chunk_id": chunk_id,
                    "reference_dmap": str(reference_path),
                })
                identity = cross_identity(result)
                run_label = str(result.get("run") or "")
                if run_label == baseline_label:
                    baseline_rows[identity] = result
                    if result.get("fit_status") == "ok":
                        baseline_models[identity] = result
                        for pending, points, pending_config in pending_cross_evaluations.pop(
                            identity, []
                        ):
                            apply_baseline_model_cross_evaluation(
                                pending, points, result, pending_config
                            )
                elif run_label in candidate_labels:
                    if evaluation_points is None or not len(evaluation_points):
                        result["baseline_model_on_candidate_status"] = (
                            "unavailable_no_candidate_points"
                        )
                        result["baseline_model_on_candidate_reason"] = (
                            "candidate annotation mask has no valid depth"
                        )
                    elif identity in baseline_models:
                        apply_baseline_model_cross_evaluation(
                            result, evaluation_points, baseline_models[identity], fit_config
                        )
                    elif identity in baseline_rows:
                        result["baseline_model_on_candidate_status"] = (
                            "unavailable_baseline_fit"
                        )
                        result["baseline_model_on_candidate_reason"] = (
                            "baseline fit status is "
                            f"{baseline_rows[identity].get('fit_status', 'unavailable')}"
                        )
                    else:
                        result["baseline_model_on_candidate_status"] = "pending_baseline_fit"
                        result["baseline_model_on_candidate_reason"] = (
                            "matching baseline fit has not been evaluated"
                        )
                        pending_cross_evaluations.setdefault(identity, []).append(
                            (result, evaluation_points, fit_config)
                        )
                results.append(result)
    for identity, pending_rows in pending_cross_evaluations.items():
        baseline_status = baseline_rows.get(identity, {}).get("fit_status", "unavailable")
        for pending, _points, _config in pending_rows:
            pending["baseline_model_on_candidate_status"] = "unavailable_baseline_fit"
            pending["baseline_model_on_candidate_reason"] = (
                f"matching baseline fit status is {baseline_status}"
            )
    return results


def add_model_stability(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    joined = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), how="inner")
    rows = []
    for _, row in joined.iterrows():
        out = {key: row[key] for key in keys}
        kind = str(row.get("annotation_kind", ""))
        out["baseline_current_fit_all_residual_p95_m"] = safe_float(
            row.get("all_residual_p95_m_baseline")
        )
        out["candidate_current_fit_all_residual_p95_m"] = safe_float(
            row.get("all_residual_p95_m_candidate")
        )
        for name in (
            "status", "reason", "schema_version", "source_run", "point_count",
            "all_residual_mean_m", "all_residual_median_m", "all_residual_p90_m",
            "all_residual_p95_m", "inlier_fraction_5mm", "inlier_fraction_10mm",
            "inlier_fraction_20mm", "inlier_fraction_50mm", "inlier_threshold_auc",
            "effective_inlier_coverage_5mm", "effective_inlier_coverage_10mm",
            "effective_inlier_coverage_20mm", "effective_inlier_coverage_50mm",
        ):
            raw_source = f"baseline_model_on_candidate_{name}"
            source = (
                f"{raw_source}_candidate"
                if f"{raw_source}_candidate" in row else raw_source
            )
            if source in row:
                out[raw_source] = row.get(source)
        if kind == "plane" and pd.notna(row.get("plane_normal_x_baseline")) and pd.notna(row.get("plane_normal_x_candidate")):
            nb = np.array([row["plane_normal_x_baseline"], row["plane_normal_y_baseline"], row["plane_normal_z_baseline"]], dtype=float)
            nc = np.array([row["plane_normal_x_candidate"], row["plane_normal_y_candidate"], row["plane_normal_z_candidate"]], dtype=float)
            nb /= max(np.linalg.norm(nb), 1e-12)
            nc /= max(np.linalg.norm(nc), 1e-12)
            if float(nb @ nc) < 0:
                nc = -nc
                candidate_d = -float(row["plane_d_candidate"])
            else:
                candidate_d = float(row["plane_d_candidate"])
            out["plane_normal_delta_deg"] = math.degrees(math.acos(float(np.clip(nb @ nc, -1.0, 1.0))))
            pb = np.array([row["model_point_x_baseline"], row["model_point_y_baseline"], row["model_point_z_baseline"]], dtype=float)
            out["plane_position_delta_m"] = abs(float(nc @ pb + candidate_d))
        if kind == "edge" and pd.notna(row.get("line_direction_x_baseline")) and pd.notna(row.get("line_direction_x_candidate")):
            db = np.array([row["line_direction_x_baseline"], row["line_direction_y_baseline"], row["line_direction_z_baseline"]], dtype=float)
            dc = np.array([row["line_direction_x_candidate"], row["line_direction_y_candidate"], row["line_direction_z_candidate"]], dtype=float)
            db /= max(np.linalg.norm(db), 1e-12)
            dc /= max(np.linalg.norm(dc), 1e-12)
            out["line_direction_delta_deg"] = math.degrees(math.acos(float(np.clip(abs(db @ dc), -1.0, 1.0))))
            pb = np.array([row["model_point_x_baseline"], row["model_point_y_baseline"], row["model_point_z_baseline"]], dtype=float)
            pc = np.array([row["model_point_x_candidate"], row["model_point_y_candidate"], row["model_point_z_candidate"]], dtype=float)
            anchor = 0.5 * (pb + pc)
            qb = pb + db * float((anchor - pb) @ db)
            qc = pc + dc * float((anchor - pc) @ dc)
            out["line_position_delta_m"] = float(np.linalg.norm(qb - qc))
            baseline_extent = float(row["line_extent_length_m_baseline"])
            candidate_extent = float(row["line_extent_length_m_candidate"])
            out["line_extent_length_m_baseline"] = baseline_extent
            out["line_extent_length_m_candidate"] = candidate_extent
            out["line_extent_delta_m"] = candidate_extent - baseline_extent
            out["line_extent_relative_delta"] = (
                out["line_extent_delta_m"] / max(abs(baseline_extent), 1e-12)
            )
        switch_reasons = []
        if safe_float(out.get("line_direction_delta_deg")) is not None and abs(
            float(out["line_direction_delta_deg"])
        ) >= MODEL_SWITCH_THRESHOLDS["line_direction_delta_deg"]:
            switch_reasons.append("line_direction")
        if safe_float(out.get("line_extent_delta_m")) is not None and abs(
            float(out["line_extent_delta_m"])
        ) >= MODEL_SWITCH_THRESHOLDS["line_extent_delta_m"]:
            switch_reasons.append("line_extent_absolute")
        if safe_float(out.get("line_extent_relative_delta")) is not None and abs(
            float(out["line_extent_relative_delta"])
        ) >= MODEL_SWITCH_THRESHOLDS["line_extent_relative_delta"]:
            switch_reasons.append("line_extent_relative")
        if safe_float(out.get("plane_normal_delta_deg")) is not None and abs(
            float(out["plane_normal_delta_deg"])
        ) >= MODEL_SWITCH_THRESHOLDS["plane_normal_delta_deg"]:
            switch_reasons.append("plane_normal")
        if safe_float(out.get("plane_position_delta_m")) is not None and abs(
            float(out["plane_position_delta_m"])
        ) >= MODEL_SWITCH_THRESHOLDS["plane_position_delta_m"]:
            switch_reasons.append("plane_offset")
        out["large_model_switch"] = bool(switch_reasons)
        out["model_switch_reasons_json"] = json.dumps(switch_reasons)
        rows.append(out)
    return pd.DataFrame(rows)


def annotate_model_switch_regression_proximity(
    stability: pd.DataFrame,
    accuracy_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Mark large model changes in candidate/scene cohorts that regress."""

    if stability.empty:
        return stability
    result = stability.copy()
    regressions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not accuracy_evidence.empty:
        for row in accuracy_evidence.to_dict("records"):
            loss = safe_float(row.get("normalized_regression_loss"))
            if loss is None or loss <= 1.0:
                continue
            key = (str(row.get("candidate") or ""), str(row.get("scene_id") or ""))
            regressions.setdefault(key, []).append({
                "metric": row.get("metric"),
                "normalized_regression_loss": loss,
            })
    near = []
    evidence = []
    for row in result.to_dict("records"):
        key = (str(row.get("candidate") or ""), str(row.get("scene_id") or ""))
        matching = regressions.get(key, [])
        is_near = bool(row.get("large_model_switch")) and bool(matching)
        near.append(is_near)
        evidence.append(json.dumps(matching if is_near else [], sort_keys=True))
    result["near_candidate_regression"] = near
    result["candidate_regression_evidence_json"] = evidence
    return result


def entity_scene_values(
    dataframe: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
    metric: str,
    entity_keys: list[str],
    relative_delta: bool = False,
) -> pd.DataFrame:
    required = [*entity_keys, "run", "repeat", metric]
    if dataframe.empty or any(column not in dataframe.columns for column in required):
        return pd.DataFrame()
    clean = dataframe[required].dropna(subset=[metric]).copy()
    baseline = clean[clean["run"] == baseline_label]
    candidate = clean[clean["run"] == candidate_label]
    if baseline.empty or candidate.empty:
        return pd.DataFrame()
    baseline_entity = baseline.groupby(entity_keys, as_index=False)[metric].mean().rename(columns={metric: "baseline_value"})
    candidate_entity = candidate.groupby(entity_keys, as_index=False)[metric].mean().rename(columns={metric: "candidate_value"})
    paired = baseline_entity.merge(candidate_entity, on=entity_keys, how="inner")
    paired["delta"] = paired["candidate_value"] - paired["baseline_value"]
    if relative_delta:
        denominator = paired["baseline_value"].abs()
        paired = paired[denominator > 1e-12].copy()
        paired["delta"] /= denominator[denominator > 1e-12]
    return paired.groupby("scene_id", as_index=False).agg(
        baseline_value=("baseline_value", "mean"),
        candidate_value=("candidate_value", "mean"),
        delta=("delta", "mean"),
        entities=("delta", "size"),
    )


def bootstrap_interval(values: np.ndarray, seed: int, samples: int = 10000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 512
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def baseline_repeatability(
    dataframe: pd.DataFrame,
    baseline_label: str,
    metric: str,
    entity_keys: list[str],
    *,
    relative_delta: bool = False,
) -> float:
    if dataframe.empty or metric not in dataframe.columns:
        return 0.0
    baseline = dataframe[dataframe["run"] == baseline_label].dropna(subset=[metric])
    repeats = sorted(int(value) for value in baseline["repeat"].unique())
    if len(repeats) < 2:
        return 0.0
    scene_repeat = baseline.groupby(["scene_id", "repeat"], as_index=False)[metric].mean()
    deltas = []
    for index, first in enumerate(repeats):
        for second in repeats[index + 1:]:
            left = scene_repeat[scene_repeat["repeat"] == first][["scene_id", metric]]
            right = scene_repeat[scene_repeat["repeat"] == second][["scene_id", metric]]
            paired = left.merge(right, on="scene_id", suffixes=("_a", "_b"))
            pair_deltas = np.abs(
                paired[f"{metric}_b"] - paired[f"{metric}_a"]
            )
            if relative_delta:
                denominator = paired[f"{metric}_a"].abs()
                pair_deltas = pair_deltas[denominator > 1e-12] / denominator[
                    denominator > 1e-12
                ]
            deltas.extend(pair_deltas.tolist())
    return float(np.percentile(deltas, 95)) if deltas else 0.0


def strict_accuracy_gate_fields(
    row: dict[str, Any],
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the explicitly configured strict candidate gate, if complete."""

    required = {
        "lost_baseline_fit_count_max",
        "median_normalized_noise_loss_lt",
        "worst_normalized_noise_loss_max",
    }
    if not isinstance(policy, dict) or not required.issubset(policy):
        return {
            "strict_accuracy_gate_available": False,
            "strict_accuracy_pass": None,
            "strict_accuracy_failed_checks": "[]",
        }
    failed = []
    lost = int(row.get("lost_baseline_fit_count", 0))
    median_loss = safe_float(row.get("median_normalized_noise_loss"))
    worst_loss = safe_float(row.get("worst_normalized_noise_loss"))
    if lost > int(policy["lost_baseline_fit_count_max"]):
        failed.append("lost_baseline_fit_count")
    if median_loss is None or not (
        median_loss < float(policy["median_normalized_noise_loss_lt"])
    ):
        failed.append("median_normalized_noise_loss")
    if worst_loss is None or not (
        worst_loss <= float(policy["worst_normalized_noise_loss_max"])
    ):
        failed.append("worst_normalized_noise_loss")
    return {
        "strict_accuracy_gate_available": True,
        "strict_accuracy_pass": not failed,
        "strict_accuracy_failed_checks": json.dumps(failed),
    }


def build_accuracy_first_ledger(
    annotations: pd.DataFrame,
    frames: pd.DataFrame,
    performance: pd.DataFrame,
    run_specs: list[dict[str, Any]],
    coverage_floor: float = 0.05,
    strict_candidate_gate: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an accuracy-first, availability-aware candidate ranking.

    Noise is evaluated only on baseline-successful structures that remain
    evaluable for the candidate. Coverage uses the baseline intent-to-treat
    population and assigns zero to an unavailable candidate fit. This keeps a
    sparse candidate from appearing accurate merely by changing the comparison
    population. Coverage warnings remain advisory and every candidate is kept.
    """

    baseline_labels = [str(run["label"]) for run in run_specs if run.get("role") == "baseline"]
    if len(baseline_labels) != 1:
        raise ValueError("exactly one run must have role=baseline")
    baseline_label = baseline_labels[0]
    candidates = [str(run["label"]) for run in run_specs if str(run["label"]) != baseline_label]
    if annotations.empty or not candidates:
        return pd.DataFrame(), pd.DataFrame()

    data = annotations.copy()
    if "stage" in data.columns:
        data = data[data["stage"] == "post_filter"]
    entity_keys = [
        "scene_id", "image_id", "annotation_kind", "object_id", "chunk_id",
    ]
    required = {"run", "repeat", "fit_status", *entity_keys}
    if not required.issubset(data.columns):
        return pd.DataFrame(), pd.DataFrame()

    baseline_rows = data[data["run"] == baseline_label]
    baseline_success = baseline_rows[baseline_rows["fit_status"] == "ok"]
    ledger_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    def entity_metric(
        rows: pd.DataFrame,
        metric: str,
        *,
        successful_only: bool,
        missing_as_zero: bool = False,
    ) -> pd.DataFrame:
        selected = rows[rows["fit_status"] == "ok"] if successful_only else rows
        if metric not in selected.columns or selected.empty:
            return pd.DataFrame(columns=[*entity_keys, metric])
        numeric = selected[[*entity_keys, metric]].copy()
        numeric[metric] = pd.to_numeric(numeric[metric], errors="coerce")
        if missing_as_zero:
            numeric[metric] = numeric[metric].fillna(0.0)
        else:
            numeric = numeric.dropna(subset=[metric])
        if numeric.empty:
            return pd.DataFrame(columns=[*entity_keys, metric])
        return numeric.groupby(entity_keys, as_index=False, dropna=False)[metric].mean()

    def identity_set(rows: pd.DataFrame) -> set[tuple[Any, ...]]:
        if rows.empty:
            return set()
        return {
            tuple(record[key] for key in entity_keys)
            for record in rows[entity_keys].drop_duplicates().to_dict("records")
        }

    baseline_success_ids = identity_set(baseline_success)
    baseline_scene_ids = sorted(str(value) for value in baseline_success["scene_id"].dropna().unique())

    for candidate_label in candidates:
        candidate_rows = data[data["run"] == candidate_label]
        # A partial campaign can contain configured candidates that have not run
        # yet. They have no comparison population and must not appear as zero-
        # coverage regressions in the accuracy ledger.
        if candidate_rows.empty:
            continue
        candidate_success = candidate_rows[candidate_rows["fit_status"] == "ok"]
        candidate_success_ids = identity_set(candidate_success)
        candidate_scene_ids = sorted(str(value) for value in candidate_rows["scene_id"].dropna().unique())
        common_scenes = sorted(set(baseline_scene_ids) & set(candidate_scene_ids))
        if not common_scenes:
            continue
        baseline_common_ids = {
            identity for identity in baseline_success_ids if str(identity[0]) in common_scenes
        }
        candidate_common_ids = {
            identity for identity in candidate_success_ids if str(identity[0]) in common_scenes
        }
        lost_ids = baseline_common_ids - candidate_common_ids
        gained_ids = candidate_common_ids - baseline_common_ids
        normalized_losses: list[float] = []
        primary_metrics_available = 0

        for metric, (direction, configured_tolerance, unit) in ACCURACY_PRIMARY_METRICS.items():
            baseline_metric = entity_metric(baseline_success, metric, successful_only=True).rename(
                columns={metric: "baseline_value"}
            )
            candidate_metric = entity_metric(candidate_success, metric, successful_only=True).rename(
                columns={metric: "candidate_value"}
            )
            paired = baseline_metric.merge(candidate_metric, on=entity_keys, how="inner")
            if common_scenes:
                paired = paired[paired["scene_id"].astype(str).isin(common_scenes)]
            if paired.empty:
                continue
            repeatability = baseline_repeatability(data, baseline_label, metric, entity_keys)
            for scene_id, scene_rows in paired.groupby("scene_id", sort=True):
                losses = []
                deltas = []
                tolerances = []
                for item in scene_rows.to_dict("records"):
                    baseline_value = float(item["baseline_value"])
                    candidate_value = float(item["candidate_value"])
                    tolerance = max(configured_tolerance, repeatability)
                    if metric == "all_residual_p95_m":
                        tolerance = max(tolerance, abs(baseline_value) * 0.05)
                    delta = candidate_value - baseline_value
                    loss = delta / tolerance if direction == "lower" else -delta / tolerance
                    losses.append(loss)
                    deltas.append(delta)
                    tolerances.append(tolerance)
                scene_loss = float(np.mean(losses))
                normalized_losses.append(scene_loss)
                primary_metrics_available += 1
                evidence_rows.append({
                    "candidate": candidate_label,
                    "scene_id": str(scene_id),
                    "metric": metric,
                    "direction": direction,
                    "unit": unit,
                    "paired_structures": int(len(scene_rows)),
                    "baseline_value": float(scene_rows["baseline_value"].mean()),
                    "candidate_value": float(scene_rows["candidate_value"].mean()),
                    "mean_delta": float(np.mean(deltas)),
                    "effective_tolerance": float(np.max(tolerances)),
                    "normalized_regression_loss": scene_loss,
                })

        has_regression = any(value > 1.0 for value in normalized_losses)
        has_improvement = any(value < -1.0 for value in normalized_losses)
        if not normalized_losses:
            paired_noise_class = "inconclusive"
        elif has_regression and has_improvement:
            paired_noise_class = "mixed"
        elif has_regression:
            paired_noise_class = "regressed"
        elif has_improvement:
            paired_noise_class = "improved"
        else:
            paired_noise_class = "equivalent"
        noise_class = "inconclusive" if lost_ids else paired_noise_class

        coverage_metrics = [
            metric for metric in (
                "effective_inlier_coverage_20mm", "effective_inlier_coverage",
                "spatial_coverage_fraction", "coverage_fraction",
            ) if metric in data.columns
        ]
        coverage_deltas: dict[str, float] = {}
        coverage_scene_min: dict[str, float] = {}
        for metric in coverage_metrics:
            baseline_metric = entity_metric(
                baseline_rows, metric, successful_only=False, missing_as_zero=True
            ).rename(
                columns={metric: "baseline_value"}
            )
            candidate_metric = entity_metric(
                candidate_rows, metric, successful_only=False, missing_as_zero=True
            ).rename(
                columns={metric: "candidate_value"}
            )
            population = baseline_metric.merge(candidate_metric, on=entity_keys, how="left")
            if common_scenes:
                population = population[population["scene_id"].astype(str).isin(common_scenes)]
            if population.empty:
                continue
            population["candidate_value"] = population["candidate_value"].fillna(0.0)
            population["delta"] = population["candidate_value"] - population["baseline_value"]
            scene_delta = population.groupby("scene_id", as_index=False)["delta"].mean()
            coverage_deltas[metric] = float(scene_delta["delta"].mean())
            coverage_scene_min[metric] = float(scene_delta["delta"].min())

        frame_coverage = entity_scene_values(
            frames, baseline_label, candidate_label, "valid_ratio_after_filter",
            ["scene_id", "image_id"],
        )
        if not frame_coverage.empty:
            coverage_deltas["valid_ratio_after_filter"] = float(frame_coverage["delta"].mean())
            coverage_scene_min["valid_ratio_after_filter"] = float(frame_coverage["delta"].min())

        endpoint_frame_coverage = entity_scene_values(
            frames, baseline_label, candidate_label, "endpoint_valid_depth_coverage",
            ["scene_id", "image_id"],
        )
        endpoint_delta = (
            float(endpoint_frame_coverage["delta"].mean())
            if not endpoint_frame_coverage.empty else math.nan
        )
        endpoint_scene_min_delta = (
            float(endpoint_frame_coverage["delta"].min())
            if not endpoint_frame_coverage.empty else math.nan
        )

        runtime = entity_scene_values(
            performance,
            baseline_label,
            candidate_label,
            "endpoint_elapsed_seconds",
            ["scene_id"],
            relative_delta=True,
        )
        runtime_delta = float(runtime["delta"].mean()) if not runtime.empty else math.nan
        advisory_metrics = {
            name: value for name, value in coverage_scene_min.items() if value < -abs(coverage_floor)
        }
        effective_delta = coverage_deltas.get(
            "effective_inlier_coverage_20mm",
            coverage_deltas.get("effective_inlier_coverage", math.nan),
        )
        spatial_delta = coverage_deltas.get("spatial_coverage_fraction", math.nan)
        valid_delta = coverage_deltas.get("valid_ratio_after_filter", math.nan)
        ledger_row = {
            "candidate": candidate_label,
            "noise_class": noise_class,
            "paired_noise_class": paired_noise_class,
            "noise_class_order": ACCURACY_CLASS_ORDER[noise_class],
            "worst_normalized_noise_loss": max(normalized_losses) if normalized_losses else math.inf,
            "median_normalized_noise_loss": float(np.median(normalized_losses)) if normalized_losses else math.inf,
            "primary_scene_metric_rows": primary_metrics_available,
            "scene_count": len(common_scenes),
            "baseline_successful_structures": len(baseline_common_ids),
            "paired_successful_structures": len(baseline_common_ids & candidate_common_ids),
            "lost_baseline_fit_count": len(lost_ids),
            "gained_fit_count": len(gained_ids),
            "availability_biased": bool(lost_ids),
            "coverage_floor": float(coverage_floor),
            "coverage_advisory": bool(advisory_metrics),
            "coverage_advisory_metrics": json.dumps(advisory_metrics, sort_keys=True),
            "effective_coverage_delta": effective_delta,
            "spatial_coverage_delta": spatial_delta,
            "valid_coverage_delta": valid_delta,
            "endpoint_valid_depth_coverage_delta": endpoint_delta,
            "endpoint_valid_depth_coverage_scene_min_delta": endpoint_scene_min_delta,
            "endpoint_coverage_advisory": bool(
                math.isfinite(endpoint_scene_min_delta)
                and endpoint_scene_min_delta < -abs(coverage_floor)
            ),
            "runtime_relative_delta": runtime_delta,
        }
        ledger_row.update(
            strict_accuracy_gate_fields(ledger_row, strict_candidate_gate)
        )
        ledger_rows.append(ledger_row)

    ledger = pd.DataFrame(ledger_rows)
    if not ledger.empty:
        ledger["_effective_sort"] = -pd.to_numeric(
            ledger["effective_coverage_delta"], errors="coerce"
        ).fillna(-math.inf)
        ledger["_spatial_sort"] = -pd.to_numeric(
            ledger["spatial_coverage_delta"], errors="coerce"
        ).fillna(-math.inf)
        ledger = ledger.sort_values([
            "noise_class_order", "worst_normalized_noise_loss",
            "median_normalized_noise_loss", "_effective_sort", "_spatial_sort", "candidate",
        ], kind="stable").reset_index(drop=True)
        ledger["accuracy_rank"] = np.arange(1, len(ledger) + 1)
        ledger = ledger.drop(columns=["_effective_sort", "_spatial_sort"])
    return ledger, pd.DataFrame(evidence_rows)


def build_comparisons(
    frames: pd.DataFrame,
    annotations: pd.DataFrame,
    performance: pd.DataFrame,
    run_specs: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    baseline_labels = [str(run["label"]) for run in run_specs if run.get("role") == "baseline"]
    if len(baseline_labels) != 1:
        raise ValueError("exactly one run must have role=baseline")
    baseline_label = baseline_labels[0]
    candidates = [str(run["label"]) for run in run_specs if str(run["label"]) != baseline_label]
    rows: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for candidate_label in candidates:
        for metric, spec in METRICS.items():
            data = frames if spec.level == "frame" else annotations if spec.level == "annotation" else performance
            if spec.level == "annotation" and "stage" in data.columns:
                data = data[data["stage"] == "post_filter"]
            entity_keys = ["scene_id", "image_id"] if spec.level == "frame" else [
                "scene_id", "image_id", "annotation_kind", "object_id", "chunk_id", "stage"
            ]
            if spec.level == "performance":
                entity_keys = ["scene_id"]
            scene_values = entity_scene_values(
                data,
                baseline_label,
                candidate_label,
                metric,
                entity_keys,
                relative_delta=spec.level == "performance",
            )
            if scene_values.empty:
                continue
            for scene_row in scene_values.to_dict("records"):
                rows.append({
                    "candidate": candidate_label, "metric": metric, "level": spec.level,
                    "direction": spec.direction, "unit": spec.unit, **scene_row,
                })
            values = scene_values["delta"].to_numpy(dtype=float)
            mean_delta = float(values.mean())
            ci_low, ci_high = bootstrap_interval(values, stable_seed(candidate_label, metric))
            noise = baseline_repeatability(
                data,
                baseline_label,
                metric,
                entity_keys,
                relative_delta=spec.level == "performance",
            )
            tolerance = max(spec.tolerance, noise)
            regression = (
                spec.direction == "higher" and mean_delta < -tolerance and ci_high < 0.0
            ) or (
                spec.direction == "lower" and mean_delta > tolerance and ci_low > 0.0
            )
            improvement = (
                spec.direction == "higher" and mean_delta > tolerance and ci_low > 0.0
            ) or (
                spec.direction == "lower" and mean_delta < -tolerance and ci_high < 0.0
            )
            if len(scene_values) < MIN_SCENES_FOR_GATE:
                status = "informational"
            elif spec.gate and regression:
                status = "fail"
            elif improvement:
                status = "improved"
            else:
                status = "pass"
            gates.append({
                "candidate": candidate_label,
                "metric": metric,
                "direction": spec.direction,
                "mean_delta": mean_delta,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "configured_tolerance": spec.tolerance,
                "repeatability_p95": noise,
                "effective_tolerance": tolerance,
                "scene_count": int(len(scene_values)),
                "gate": spec.gate,
                "status": status,
            })
    return pd.DataFrame(rows), gates


def pareto_summary(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted({str(row["candidate"]) for row in gates})
    objectives = [
        "valid_ratio_after_filter",
        "effective_inlier_coverage",
        "all_residual_p95_m",
        "endpoint_elapsed_seconds",
    ]
    vectors: dict[str, dict[str, float]] = {candidate: {} for candidate in candidates}
    for row in gates:
        metric = str(row["metric"])
        if metric not in objectives:
            continue
        value = float(row["mean_delta"])
        vectors[str(row["candidate"])][metric] = value if METRICS[metric].direction == "higher" else -value
    out = []
    for candidate in candidates:
        dominated_by = []
        for other in candidates:
            if other == candidate:
                continue
            common = sorted(set(vectors[candidate]) & set(vectors[other]))
            if common and all(vectors[other][metric] >= vectors[candidate][metric] for metric in common) and any(
                vectors[other][metric] > vectors[candidate][metric] for metric in common
            ):
                dominated_by.append(other)
        out.append({"candidate": candidate, "pareto": not dominated_by, "dominated_by": dominated_by, "objectives": vectors[candidate]})
    return out


def deterministic_findings(gates: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in gates:
        by_candidate.setdefault(str(row["candidate"]), {})[str(row["metric"])] = row
    findings: list[dict[str, str]] = []
    for candidate, metrics in by_candidate.items():
        coverage = metrics.get("valid_ratio_after_filter")
        geometry = metrics.get("effective_inlier_coverage")
        residual = metrics.get("all_residual_p95_m")
        cost = metrics.get("final_cost_median")
        rejection = metrics.get("rejected_by_filter_ratio")
        if coverage and rejection and coverage["mean_delta"] > 0 and rejection["mean_delta"] < 0:
            findings.append({"candidate": candidate, "strength": "supported", "finding": "Completeness gain coincides with fewer filter rejections."})
        if geometry and residual and geometry["mean_delta"] < 0 and residual["mean_delta"] > 0:
            findings.append({"candidate": candidate, "strength": "supported", "finding": "Geometric self-consistency regresses in both effective inlier coverage and residual tail."})
        if cost and residual and cost["mean_delta"] < 0 and residual["mean_delta"] > 0:
            findings.append({"candidate": candidate, "strength": "hypothesis", "finding": "Lower PatchMatch cost does not translate to better geometry; inspect confidence gaps, view churn, and low-texture regions."})
        failures = [name for name, row in metrics.items() if row.get("status") == "fail"]
        if failures:
            findings.append({"candidate": candidate, "strength": "gate", "finding": "Regression gates failed: " + ", ".join(sorted(failures)) + "."})
        if not any(row["candidate"] == candidate for row in findings):
            findings.append({"candidate": candidate, "strength": "neutral", "finding": "No registered mechanism relationship crossed its evidence threshold."})
    return findings


def write_dataframe(dataframe: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    dataframe.to_csv(csv_path, index=False)
    result = {"csv": str(csv_path), "rows": int(len(dataframe)), "parquet": None}
    try:
        dataframe.to_parquet(path, index=False)
        result["parquet"] = str(path)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        result["parquet_error"] = str(exc)
    return result


ENDPOINT_RUNTIME_COLUMNS = (
    "run",
    "role",
    "repeat",
    "scene_id",
    "endpoint_elapsed_seconds",
    "runtime_status",
    "runtime_reason",
    "runtime_authority",
    "runtime_scope",
    "runtime_source",
)


def load_endpoint_runtime(config: dict[str, Any], root: Path) -> pd.DataFrame:
    """Load decision-grade wall time only from completed production endpoints."""

    rows: list[dict[str, Any]] = []

    def add_unit(
        *,
        label: str,
        role: str,
        repeat: int,
        scene_id: str,
        endpoint_dir: Path | None,
    ) -> None:
        row: dict[str, Any] = {
            "run": label,
            "role": role,
            "repeat": repeat,
            "scene_id": scene_id,
            "endpoint_elapsed_seconds": math.nan,
            "runtime_status": "unavailable",
            "runtime_reason": "production endpoint capture is unavailable",
            "runtime_authority": "production_endpoint_wall_clock",
            "runtime_scope": "full_densify_process",
            "runtime_source": (
                str(endpoint_dir / "repro.json") if endpoint_dir is not None else ""
            ),
        }
        if endpoint_dir is None:
            rows.append(row)
            return
        complete, reason = validate_completed_run_mode(endpoint_dir, "endpoint")
        if not complete:
            row["runtime_reason"] = reason
            rows.append(row)
            return
        repro = read_json(endpoint_dir / "repro.json")
        metadata = read_json(endpoint_dir / "endpoint_metadata.json")
        elapsed = safe_float(repro.get("elapsed_seconds"))
        if repro.get("elapsed_clock") != "time.monotonic_ns":
            row["runtime_reason"] = (
                "endpoint repro does not declare the monotonic runtime clock"
            )
        elif metadata.get("runtime_authority") != "production_endpoint_wall_clock":
            row["runtime_reason"] = (
                "endpoint metadata does not declare production runtime authority"
            )
        elif elapsed is None or elapsed <= 0.0:
            row["runtime_reason"] = "endpoint elapsed_seconds is not finite and positive"
        else:
            row["endpoint_elapsed_seconds"] = elapsed
            row["runtime_status"] = "available"
            row["runtime_reason"] = "validated production endpoint wall clock"
        rows.append(row)

    configured_scene_ids = [str(scene_id) for scene_id in resolve_suite(config)]
    for run in config.get("runs") or []:
        label = validated_output_component(run.get("label"), "run label")
        role = str(run.get("role", "variant"))
        existing = run.get("existing") or {}
        if existing:
            for scene_id, existing_row in sorted(existing.items()):
                if not isinstance(existing_row, dict):
                    continue
                endpoint_dir = as_path(existing_row.get("endpoint_dir"))
                add_unit(
                    label=label,
                    role=role,
                    repeat=int(existing_row.get("repeat", 0)),
                    scene_id=str(scene_id),
                    endpoint_dir=endpoint_dir,
                )
            continue
        repeats = int(run.get("repeats", 3 if role == "baseline" else 1))
        for repeat in range(repeats):
            for scene_id in configured_scene_ids:
                endpoint_dir = contained_output_path(
                    root,
                    "runs",
                    label,
                    f"repeat_{repeat:02d}",
                    scene_id,
                    "endpoint",
                    description="endpoint runtime capture directory",
                )
                add_unit(
                    label=label,
                    role=role,
                    repeat=repeat,
                    scene_id=scene_id,
                    endpoint_dir=endpoint_dir,
                )
    return pd.DataFrame(rows, columns=ENDPOINT_RUNTIME_COLUMNS)


def aggregate_timing_frames(timings: pd.DataFrame) -> pd.DataFrame:
    if timings.empty:
        return pd.DataFrame()
    keys = ["run", "role", "repeat", "scene_id", "image_id"]
    return timings.groupby(keys, as_index=False).agg(
        kernel_ms=("kernel_ms", "sum"),
        timed_passes=("kernel_ms", "size"),
        timing_source=("timing_source", "first"),
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data available._"
    head_style = "background:#d9e8f2;color:#20384b;border-bottom:2px solid #8aa8bd;padding:7px 9px;text-align:left"
    cell_style = "border-bottom:1px solid #d6e0e8;padding:6px 9px;text-align:left;vertical-align:top"
    parts = [
        '<div style="overflow-x:auto;border:1px solid #bfd0de;border-radius:6px;margin:10px 0 18px">',
        '<table style="border-collapse:collapse;width:100%;font-size:13px;background:white"><thead><tr>',
        *[f'<th style="{head_style}">{html.escape(str(header))}</th>' for header in headers],
        "</tr></thead><tbody>",
    ]
    for index, row in enumerate(rows):
        parts.append(f'<tr style="background:{"#fff" if index % 2 == 0 else "#f2f7fa"}">')
        parts.extend(f'<td style="{cell_style}">{html.escape(str(cell))}</td>' for cell in row)
        parts.append("</tr>")
    parts.extend(["</tbody></table>", "</div>"])
    return "\n".join(parts)


def fmt(value: Any, digits: int = 4) -> str:
    number = safe_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def fmt_percent(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return "n/a" if number is None else f"{number * 100.0:.{digits}f}%"


def fmt_percentage_points(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    percentage_points = number * 100.0
    return f"{percentage_points:+.{digits}f} pp"


def candidate_accounting_count(
    rows: pd.DataFrame, key: str, *, available: bool,
) -> int | str:
    if not available:
        return "unavailable"
    values = pd.to_numeric(
        rows.get(key, pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    return int(values.sum()) if not values.empty else "n/a"


def candidate_accounting_value(
    row: dict[str, Any], key: str, *, integer: bool = False,
) -> int | str:
    if (
        row.get("candidate_accounting_mode")
        == instrumentation_report.CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
    ):
        return "unavailable"
    value = safe_float(row.get(key))
    if value is None:
        return "n/a"
    return int(value) if integer else fmt(value)


def fmt_millimetres(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    sign = "+" if signed else ""
    return f"{number * 1000.0:{sign}.{digits}f} mm"


def directional_status(baseline: Any, variant: Any, direction: Literal["higher", "lower"]) -> str:
    left = safe_float(baseline)
    right = safe_float(variant)
    if left is None or right is None:
        return "unavailable"
    delta = right - left
    if abs(delta) <= 1e-12:
        return "stable"
    return "improved" if (delta > 0) == (direction == "higher") else "regressed"


def annotation_metric_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No paired annotation metrics available._"
    parts = [
        '<div class="annotation-table-wrap">',
        '<table class="annotation-metric-table"><thead><tr>',
        "<th>metric</th><th>baseline</th><th>variant</th><th>delta</th><th>preferred direction</th><th>reading</th>",
        "</tr></thead><tbody>",
    ]
    for row in rows:
        status = str(row.get("status") or "unavailable")
        parts.extend([
            "<tr>",
            f"<td><strong>{html.escape(str(row.get('metric', '')))}</strong></td>",
            f"<td>{html.escape(str(row.get('baseline', 'n/a')))}</td>",
            f"<td>{html.escape(str(row.get('variant', 'n/a')))}</td>",
            f"<td>{html.escape(str(row.get('delta', 'n/a')))}</td>",
            f"<td>{html.escape(str(row.get('direction', '')))}</td>",
            f'<td><span class="annotation-change {html.escape(status)}">{html.escape(status)}</span></td>',
            "</tr>",
        ])
    parts.extend(["</tbody></table>", "</div>"])
    return "\n".join(parts)


def fmt_mib(value: Any) -> str:
    number = safe_float(value)
    return "n/a" if number is None else f"{number / 1024**2:.2f}"


def mask_rejection_display(row: dict[str, Any], *, ratio: bool) -> str:
    status = str(row.get("ignore_mask_status") or "legacy_count_available")
    if status == "not_requested":
        return "n/a"
    if status == "unavailable":
        reason = str(row.get("ignore_mask_unavailable_reason") or "mask unavailable")
        return f"unavailable ({reason})"
    key = "rejected_by_ignore_mask_ratio" if ratio else "num_rejected_by_ignore_mask"
    return fmt(row.get(key)) if ratio else (
        str(int(float(row[key]))) if row.get(key) is not None and not pd.isna(row.get(key)) else "unavailable"
    )


def aggregate_mask_rejection_display(group: pd.DataFrame) -> str:
    if "ignore_mask_status" not in group.columns:
        return fmt(group.get("rejected_by_ignore_mask_ratio", pd.Series(dtype=float)).mean())
    if group["ignore_mask_status"].astype(str).eq("unavailable").any():
        return "unavailable"
    loaded = group[~group["ignore_mask_status"].astype(str).eq("not_requested")]
    if loaded.empty:
        return "n/a"
    return fmt(loaded["rejected_by_ignore_mask_ratio"].mean())


def report_plots(
    frames: pd.DataFrame,
    annotations: pd.DataFrame,
    stability: pd.DataFrame,
    performance: pd.DataFrame,
    gates: list[dict[str, Any]],
    output_dir: Path,
) -> list[tuple[str, Path]]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output_dir / "visualizations" / "overall"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plots: list[tuple[str, Path]] = []
    gate_rows = [row for row in gates if row.get("gate")]
    if gate_rows:
        labels = [f"{row['candidate']} / {row['metric']}" for row in gate_rows]
        values = np.asarray([float(row["mean_delta"]) for row in gate_rows])
        low = values - np.asarray([float(row["ci95_low"]) for row in gate_rows])
        high = np.asarray([float(row["ci95_high"]) for row in gate_rows]) - values
        colors = ["#c43c39" if row["status"] == "fail" else "#2a9d8f" if row["status"] == "improved" else "#457b9d" for row in gate_rows]
        fig, ax = plt.subplots(figsize=(11, max(4.0, 0.42 * len(labels))))
        y = np.arange(len(labels))
        ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="none", ecolor="#6b7280", capsize=3)
        ax.scatter(values, y, c=colors, s=55, zorder=3)
        ax.axvline(0, color="#20242a", linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("candidate - baseline (scene-balanced mean, 95% CI)")
        ax.set_title("Regression gate effects")
        fig.tight_layout()
        path = plot_dir / "gate_effects.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(("Regression gate effects", path))
    threshold_columns = [f"inlier_fraction_{int(value * 1000)}mm" for value in RANSAC_THRESHOLDS_M]
    if not annotations.empty and all(column in annotations.columns for column in threshold_columns):
        post = annotations[annotations["stage"] == "post_filter"] if "stage" in annotations.columns else annotations
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for run, group in post.groupby("run"):
            means = [group[column].dropna().mean() for column in threshold_columns]
            ax.plot(np.asarray(RANSAC_THRESHOLDS_M) * 1000.0, means, marker="o", linewidth=2, label=str(run))
        ax.set_xlabel("inlier threshold (mm)")
        ax.set_ylabel("mean inlier fraction")
        ax.set_ylim(0, 1.02)
        ax.set_title("Plane/line fixed-model inlier curves")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = plot_dir / "annotation_threshold_curves.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(("Annotation threshold curves", path))
        grouped = post.groupby(["run", "annotation_kind"], as_index=False).agg(
            effective_inlier_coverage=("effective_inlier_coverage", "mean"),
            residual_p95_m=("all_residual_p95_m", "mean"),
        )
        if not grouped.empty:
            labels = [f"{row.run} / {row.annotation_kind}" for row in grouped.itertuples()]
            x = np.arange(len(labels))
            fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.55 * len(labels)), 4.8))
            axes[0].bar(x, grouped["effective_inlier_coverage"], color="#2a9d8f", edgecolor="white")
            axes[0].set_ylim(0.0, 1.0)
            axes[0].set_ylabel("coverage x fixed-model inlier fraction")
            axes[0].set_title("Effective annotation inlier coverage")
            axes[1].bar(x, grouped["residual_p95_m"] * 1000.0, color="#e76f51", edgecolor="white")
            axes[1].set_ylabel("P95 residual (mm; lower is better)")
            axes[1].set_title("Annotation residual tail")
            for axis in axes:
                axis.set_xticks(x, labels, rotation=25, ha="right")
                axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = plot_dir / "annotation_quality_summary.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            plots.append(("Annotation quality summary", path))
    if not stability.empty:
        groups = []
        angle_values = []
        position_values = []
        for (candidate, kind), group in stability.groupby(["candidate", "annotation_kind"]):
            angle_column = "plane_normal_delta_deg" if kind == "plane" else "line_direction_delta_deg"
            position_column = "plane_position_delta_m" if kind == "plane" else "line_position_delta_m"
            angles = group[angle_column].dropna().to_numpy(dtype=float) if angle_column in group.columns else np.asarray([])
            positions = group[position_column].dropna().to_numpy(dtype=float) * 1000.0 if position_column in group.columns else np.asarray([])
            if angles.size or positions.size:
                groups.append(f"{candidate} / {kind}")
                angle_values.append(angles if angles.size else np.asarray([np.nan]))
                position_values.append(positions if positions.size else np.asarray([np.nan]))
        if groups:
            fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.7 * len(groups)), 4.8))
            for axis, values, title, ylabel, color in (
                (axes[0], angle_values, "Fitted-model angular stability", "degrees", "#457b9d"),
                (axes[1], position_values, "Fitted-model positional stability", "millimeters", "#f4a261"),
            ):
                boxes = axis.boxplot(values, tick_labels=groups, patch_artist=True, showmeans=True)
                for box in boxes["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(0.75)
                axis.set_title(title)
                axis.set_ylabel(ylabel)
                axis.tick_params(axis="x", rotation=25)
                axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = plot_dir / "annotation_model_stability.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            plots.append(("Annotation model stability", path))
    if not frames.empty and {"run", "valid_ratio_after_filter", "final_cost_median"}.issubset(frames.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for run, group in frames.groupby("run"):
            axes[0].hist(group["valid_ratio_after_filter"].dropna(), bins=16, alpha=0.45, label=str(run))
            axes[1].hist(group["final_cost_median"].dropna(), bins=16, alpha=0.45, label=str(run))
        axes[0].set_title("Final valid-depth coverage")
        axes[0].set_xlabel("fraction")
        axes[1].set_title("Median final cost")
        axes[1].set_xlabel("cost; lower is better")
        for axis in axes:
            axis.legend(frameon=False)
        fig.tight_layout()
        path = plot_dir / "frame_distributions.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(("Frame distributions", path))
    if not performance.empty and {"run", "kernel_ms"}.issubset(performance.columns):
        labels = []
        values = []
        errors = []
        for run, group in performance.groupby("run"):
            labels.append(str(run))
            values.append(float(group["kernel_ms"].mean()))
            errors.append(float(group["kernel_ms"].std(ddof=1)) if len(group) > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * len(labels)), 4.4))
        ax.bar(labels, values, yerr=errors, color="#457b9d", capsize=4, edgecolor="white")
        ax.set_ylabel("summed PatchMatch kernel time per frame (ms)")
        ax.set_title("Summary-only timing run")
        fig.tight_layout()
        path = plot_dir / "kernel_timing.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(("PatchMatch kernel timing", path))
    return plots


def accuracy_first_plots(
    ledger: pd.DataFrame,
    evidence: pd.DataFrame,
    output_dir: Path,
) -> list[tuple[str, Path]]:
    if ledger.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_dir = output_dir / "visualizations" / "overall"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plots: list[tuple[str, Path]] = []

    if not evidence.empty:
        table = evidence.copy()
        table["column"] = table["scene_id"].astype(str).str.slice(0, 8) + " / " + table["metric"].astype(str)
        matrix = table.pivot_table(
            index="candidate", columns="column", values="normalized_regression_loss", aggfunc="mean"
        )
        order = [value for value in ledger.sort_values("accuracy_rank")["candidate"] if value in matrix.index]
        matrix = matrix.reindex(order)
        if not matrix.empty:
            values = matrix.to_numpy(dtype=float)
            masked = np.ma.masked_invalid(values)
            width = max(9.0, 1.45 * len(matrix.columns))
            height = max(4.2, 0.48 * len(matrix.index) + 1.8)
            fig, ax = plt.subplots(figsize=(width, height))
            image_plot = ax.imshow(masked, cmap="RdYlGn_r", vmin=-2.0, vmax=2.0, aspect="auto")
            ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
            for row_index in range(values.shape[0]):
                for column_index in range(values.shape[1]):
                    value = values[row_index, column_index]
                    ax.text(
                        column_index, row_index,
                        "n/a" if not np.isfinite(value) else f"{value:+.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(value) and abs(value) > 1.25 else "#20242a",
                    )
            ax.set_title("Accuracy-first normalized line-noise evidence")
            ax.set_xlabel("scene / metric (positive is worse; +1 crosses tolerance)")
            ax.set_ylabel("candidate")
            colorbar = fig.colorbar(image_plot, ax=ax, shrink=0.82)
            colorbar.set_label("regression loss / effective tolerance")
            fig.tight_layout()
            path = plot_dir / "accuracy_noise_heatmap.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            plots.append(("Accuracy-first noise heatmap", path))

    plot_data = ledger.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["worst_normalized_noise_loss", "effective_coverage_delta"]
    )
    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(9.2, 5.6))
        colors = {
            "improved": "#2a9d8f", "equivalent": "#457b9d", "mixed": "#e9c46a",
            "regressed": "#c43c39", "inconclusive": "#7a7f85",
        }
        for row in plot_data.to_dict("records"):
            x = float(row["effective_coverage_delta"]) * 100.0
            y = float(row["worst_normalized_noise_loss"])
            ax.scatter(x, y, s=75, color=colors.get(str(row["noise_class"]), "#7a7f85"), zorder=3)
            ax.annotate(str(row["candidate"]), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.axhline(1.0, color="#c43c39", linewidth=1.2, linestyle="--", label="noise tolerance")
        ax.axvline(0.0, color="#20242a", linewidth=1.0)
        ax.axvline(-float(plot_data["coverage_floor"].iloc[0]) * 100.0, color="#b7791f", linewidth=1.0, linestyle=":", label="coverage advisory")
        ax.set_xlabel("intent-to-treat effective coverage delta (percentage points)")
        ax.set_ylabel("worst paired noise loss / tolerance (lower is better)")
        ax.set_title("Accuracy and coverage tradeoffs")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = plot_dir / "accuracy_coverage_tradeoff.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plots.append(("Accuracy versus coverage", path))
    return plots


def exact_cost_trajectory_plots(
    exact_cost_evolution: pd.DataFrame,
    output_dir: Path,
) -> list[tuple[str, Path]]:
    if exact_cost_evolution.empty:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    required = {
        "run", "scene_id", "image_id", "estimation_stage", "geometric_iteration",
        "signal", "logical_iteration", "median", "p90",
    }
    if not required.issubset(exact_cost_evolution.columns):
        return []
    plot_dir = output_dir / "visualizations" / "overall"
    plot_dir.mkdir(parents=True, exist_ok=True)
    data = exact_cost_evolution.copy()
    data["geometric_iteration_group"] = pd.to_numeric(
        data["geometric_iteration"], errors="coerce"
    ).fillna(-1).astype(int)
    plots: list[tuple[str, Path]] = []
    groups = data.groupby(
        ["scene_id", "image_id", "estimation_stage", "geometric_iteration_group"],
        dropna=False,
    )
    for (scene_id, image_id, estimation_stage, geometric_iteration), frame in groups:
        signals = sorted(str(value) for value in frame["signal"].dropna().unique())
        if not signals:
            continue
        columns = 2
        rows = math.ceil(len(signals) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(13.5, max(4.2, rows * 3.2)), squeeze=False)
        colors = ("#355c7d", "#d1495b", "#2a9d8f", "#8f5d9f", "#d17b0f")
        for signal_index, signal in enumerate(signals):
            ax = axes.flat[signal_index]
            selected = frame[frame["signal"] == signal]
            for run_index, (run, run_rows) in enumerate(selected.groupby("run")):
                ordered = run_rows.sort_values("logical_iteration")
                x = ordered["logical_iteration"].to_numpy(dtype=float) + 1.0
                median = ordered["median"].to_numpy(dtype=float)
                p90 = ordered["p90"].to_numpy(dtype=float)
                color = colors[run_index % len(colors)]
                ax.plot(x, median, color=color, marker="o", linewidth=2.0, label=f"{run} median")
                ax.plot(x, p90, color=color, linestyle="--", linewidth=1.25, alpha=0.8, label=f"{run} P90")
                if len(x):
                    ax.scatter([x[-1]], [median[-1]], color=color, s=58, edgecolor="white", linewidth=0.8, zorder=4)
            states = sorted(set(int(value) for value in selected["logical_iteration"].dropna()))
            ax.set_xticks([value + 1 for value in states], ["init" if value < 0 else f"iter {value + 1}" for value in states])
            ax.set_title(signal.replace("_", " "), fontsize=9, fontweight="bold")
            ax.set_ylabel("production value")
            ax.grid(alpha=0.22)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=6, ncol=2)
        for index in range(len(signals), rows * columns):
            axes.flat[index].axis("off")
        stage_label = str(estimation_stage)
        if int(geometric_iteration) >= 0:
            stage_label += f" {int(geometric_iteration)}"
        fig.suptitle(
            f"Exact cost-component evolution / {scene_id} / image {int(image_id)} / {stage_label}",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.5, 0.002,
            "Solid: median. Dashed: P90. Enlarged marker: each run's final logical state.",
            ha="center", fontsize=8, color="#4b5563",
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        safe_stage = re.sub(r"[^a-z0-9]+", "_", stage_label.lower()).strip("_")
        path = plot_dir / f"exact_cost_trajectories_{scene_id}_{int(image_id):04d}_{safe_stage}.png"
        fig.savefig(path, dpi=175)
        plt.close(fig)
        plots.append((f"Exact cost-component trajectories / {scene_id} / image {int(image_id)} / {stage_label}", path))
    return plots


def mechanism_for_plot(title: str) -> str:
    lowered = title.lower()
    if "timing" in lowered or "runtime" in lowered or "kernel" in lowered:
        return "runtime"
    if "support" in lowered or "view" in lowered or "entropy" in lowered or "churn" in lowered:
        return "view"
    if "valid" in lowered or "reject" in lowered or "filter" in lowered or "pixel count" in lowered:
        return "filtering"
    if "candidate" in lowered or "changed" in lowered or "depth update" in lowered or "normal update" in lowered:
        return "update"
    if "cost" in lowered or "texture" in lowered or "improvement" in lowered:
        return "cost"
    return "overview"


def generate_diagnostic_panels(
    run_scenes: list[RunScene],
    output_dir: Path,
) -> tuple[
    dict[str, list[tuple[str, Path]]],
    dict[str, list[instrumentation_report.DiagnosticPanel]],
]:
    plots: dict[str, list[tuple[str, Path]]] = {}
    panels: dict[str, list[instrumentation_report.DiagnosticPanel]] = {}
    scenes = sorted({row.scene_id for row in run_scenes})
    for scene_id in scenes:
        scene_runs = [row for row in run_scenes if row.scene_id == scene_id and row.repeat == 0]
        if len(scene_runs) < 1:
            continue
        specs = [f"{row.label}={row.instrumentation_dir}" for row in scene_runs if row.instrumentation_dir.is_dir()]
        if not specs:
            continue
        runs = instrumentation_report.parse_runs(specs)
        scene_assets = output_dir / "visualizations" / "diagnostics" / scene_id
        instrumentation_report.prepare_assets_dir(scene_assets)
        scene_plots, diagnostic_panels = instrumentation_report.make_plots(runs, scene_assets)
        plots[scene_id] = scene_plots
        panels[scene_id] = diagnostic_panels
    return plots, panels


def relative_link(path: Path, report: Path) -> str:
    resolved = path.expanduser().resolve()
    report_root = report.parent.expanduser().resolve()
    try:
        relative = resolved.relative_to(report_root)
    except ValueError as exc:
        raise ValueError(
            f"report links must target report-owned files, not external evidence: {resolved}"
        ) from exc
    return relative.as_posix()


def markdown_evidence_reference(
    path: Path | str,
    label: str,
    report_path: Path,
    *,
    image: bool = False,
) -> str:
    """Link report-owned evidence; render external provenance as plain code."""

    source = Path(str(path)).expanduser()
    try:
        target = relative_link(source, report_path)
    except ValueError:
        return (
            f"`{label} (external evidence: {source})`"
            if not image else f"_{label} is external evidence and is not embedded._"
        )
    prefix = "!" if image else ""
    return f"{prefix}[{label}]({target})"


def html_evidence_reference(
    path: Path | str, label: str, report_path: Path
) -> str:
    source = Path(str(path)).expanduser()
    try:
        target = relative_link(source, report_path)
    except ValueError:
        return (
            f"<code>{html.escape(label)} (external evidence: "
            f"{html.escape(str(source))})</code>"
        )
    return f'<a href="{html.escape(target)}">{html.escape(label)}</a>'


def report_output_links(output: dict[str, Any], report_path: Path) -> str:
    links = []
    for kind in ("csv", "parquet", "json"):
        raw_path = output.get(kind)
        if raw_path:
            links.append(markdown_evidence_reference(raw_path, kind.upper(), report_path))
    return " | ".join(links) or "unavailable"


def production_qualification_warning_markdown(outputs: dict[str, Any]) -> list[str]:
    validation = outputs.get("instrumentation_validation") or {}
    if validation.get("production_qualification_status") != (
        "failed_allowed_diagnostic_only"
    ):
        return []
    frames = int(
        validation.get("diagnostic_process_specialization_divergence_frames", 0)
    )
    return [
        '<div class="production-qualification-warning"><strong>NOT '
        "PRODUCTION-PARITY QUALIFIED.</strong> The direct <code>Process&lt;true&gt;</code> "
        f"deep capture diverged from <code>Process&lt;false&gt;</code> on {frames} "
        "validated frame(s). The opt-in recovery policy permits this report only "
        "for diagnostic mechanics. Final quality and annotation metrics use the "
        "separately identified summary/endpoint <code>Process&lt;false&gt;</code> authority "
        "when one is present. Exact/deep "
        "maps must not be used as production output evidence; every failed parity "
        "check remains recorded in the validation JSON.</div>",
        "",
    ]


def evidence_context_markdown(evidence_context: dict[str, Any] | None) -> list[str]:
    """Render separate mechanics and external quality authorities near the report top."""

    if evidence_context is None:
        return []
    subject = evidence_context["subject"]
    mechanics = evidence_context["mechanics_authority"]
    quality = evidence_context["quality_authority"]

    def display(value: Any, digits: int = 3) -> str:
        if value is None:
            return "unavailable"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    def display_unit(value: Any, unit: str, digits: int = 3) -> str:
        separator = "" if unit == "%" else " "
        return (
            "unavailable"
            if value is None
            else f"{display(value, digits)}{separator}{unit}"
        )

    metric_rows = [
        [
            row["label"], display(row["value"]), row["unit"], row["status"],
            row["description"],
        ]
        for row in mechanics["summary_metrics"]
    ]
    coverage_rows = [
        [row["candidate"], row["coverage"].replace("_", " "), row["note"] or "-"]
        for row in mechanics["candidate_coverage"]
    ]
    candidate_rows = []
    for row in sorted(
        quality["candidates"], key=lambda item: (item["accuracy_rank"], item["candidate"])
    ):
        strict = row["strict_accuracy_pass"]
        availability = (
            f"biased / {row['lost_baseline_fit_count']} lost"
            if row["availability_biased"]
            else (
                f"{row['paired_successful_structures']}/"
                f"{row['baseline_successful_structures']} paired"
            )
        )
        candidate_rows.append([
            row["accuracy_rank"],
            row["candidate"],
            "pass" if strict is True else "fail" if strict is False else "unavailable",
            row["mechanics_coverage"].replace("_", " "),
            row["scene_count"],
            availability,
            display(row["median_normalized_noise_loss"]),
            display(row["worst_normalized_noise_loss"]),
            display_unit(row["residual_p95_delta_mm"], "mm"),
            display_unit(row["threshold_auc_delta_pp"], "pp"),
            display_unit(row["inlier_5mm_delta_pp"], "pp"),
            display_unit(row["effective_coverage_delta_pp"], "pp"),
            display_unit(row["spatial_coverage_delta_pp"], "pp"),
            display_unit(row["estimator_validity_delta_pp"], "pp"),
            display_unit(row["endpoint_validity_delta_pp"], "pp"),
            display_unit(row["runtime_delta_percent"], "%"),
        ])
    provenance_rows = []
    for authority_name, authority in (
        ("mechanics", mechanics), ("quality", quality),
    ):
        for source in authority["source_artifacts"]:
            cardinality = ", ".join(
                f"{key}={value}" for key, value in sorted(
                    (source.get("cardinality") or {}).items()
                )
            ) or "-"
            provenance_rows.append([
                authority_name, source["role"], source["bytes"], cardinality,
                source["sha256"], source.get("content_digest") or "-",
            ])

    return [
        "## Evidence Authority Contract", "",
        (
            '<div class="evidence-authority-intro"><strong>'
            + html.escape(subject["title"])
            + ".</strong> "
            + html.escape(subject["summary"])
            + " The two authorities below answer different questions and are never "
            "merged into one quality cohort.</div>"
        ), "",
        "### Diagnostic Mechanism Verdict", "",
        (
            f"**{html.escape(mechanics['headline'])}** "
            f"Status: `{mechanics['verdict']}`. Scope: "
            f"{html.escape(mechanics['scope'])}"
        ), "",
        (
            f"This authority is `{mechanics['process_specialization']}` evidence from "
            f"`{mechanics['experiment_id']}` and is explicitly **not quality eligible**."
        ), "",
        md_table(["mechanics metric", "value", "unit", "status", "interpretation"], metric_rows), "",
        "#### Candidate Mechanics Coverage", "",
        md_table(["candidate", "coverage", "note"], coverage_rows), "",
        "### External Production Quality Authority", "",
        (
            f"**{html.escape(quality['headline'])}** Scope: "
            f"{html.escape(quality['scope'])}"
        ), "",
        (
            f"Quality rows below come only from `{quality['experiment_id']}` "
            f"`{quality['process_specialization']}` evidence. They are a bound summary of "
            "an external production cohort, not annotations or comparisons recomputed from "
            "this report's diagnostic captures."
        ), "",
        md_table(
            [
                "rank", "candidate", "strict accuracy", "mechanics coverage", "scenes",
                "fit availability", "median loss / tolerance", "worst loss / tolerance",
                "residual P95 delta", "threshold AUC delta", "5 mm inlier delta",
                "effective coverage delta", "spatial coverage delta",
                "estimator validity delta", "terminal endpoint validity delta",
                "runtime delta",
            ],
            candidate_rows,
        ), "",
        (
            "Manual promotion is "
            + ("required" if quality["manual_promotion_required"] else "not required")
            + ". The sidecar contract always disables automatic promotion."
        ), "",
        '<details class="evidence-provenance"><summary>Bound authority provenance</summary><div>',
        "Raw external rows are intentionally not copied into current-report aggregates. "
        "Artifact content and semantic identities remain independently auditable:", "",
        md_table(
            ["authority", "artifact role", "bytes", "cardinality", "file SHA-256", "semantic digest"],
            provenance_rows,
        ),
        "</div></details>", "",
    ]


def build_markdown(
    config: dict[str, Any],
    report_path: Path,
    frames: pd.DataFrame,
    passes: pd.DataFrame,
    annotations: pd.DataFrame,
    stability: pd.DataFrame,
    performance: pd.DataFrame,
    comparisons: pd.DataFrame,
    gates: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    findings: list[dict[str, str]],
    plots: list[tuple[str, Path]],
    instrumentation_plots: dict[str, list[tuple[str, Path]]],
    diagnostic_panels: dict[str, list[instrumentation_report.DiagnosticPanel]],
    outputs: dict[str, Any],
    exact_cost_evolution: pd.DataFrame,
    exact_iterations: pd.DataFrame,
    exact_views: pd.DataFrame,
    cpu_view_candidates: pd.DataFrame,
    cpu_estimation_selection: pd.DataFrame,
    postprocess_filters: pd.DataFrame,
    confidence_adjustment: pd.DataFrame,
    cuda_resource_plans: pd.DataFrame,
    filter_resource_plans: pd.DataFrame,
    resource_plan_validation: pd.DataFrame,
    reproducibility_artifacts: pd.DataFrame,
    drilldowns: dict[str, Any] | None = None,
    accuracy_ledger: pd.DataFrame | None = None,
    accuracy_evidence: pd.DataFrame | None = None,
    evidence_context: dict[str, Any] | None = None,
    evidence_context_path: Path | None = None,
    published_report_dir: Path | None = None,
    low_texture_hysteresis: dict[str, Any] | None = None,
    accepted_gain_census: dict[str, Any] | None = None,
) -> str:
    accuracy_ledger = accuracy_ledger if accuracy_ledger is not None else pd.DataFrame()
    accuracy_evidence = accuracy_evidence if accuracy_evidence is not None else pd.DataFrame()
    low_texture_hysteresis = low_texture_hysteresis or {}
    accepted_gain_census = accepted_gain_census or {}
    style = """<style>
details.scene { margin:18px 0;border:1px solid #aebfcd;border-radius:7px;background:#fff }
details.scene>summary { cursor:pointer;padding:11px 14px;background:#dce9f2;color:#20384b;font-weight:700 }
details.frame { margin:8px 0;border:1px solid #c7d5df;border-radius:6px;background:#fff }
details.frame>summary { cursor:pointer;padding:8px 11px;background:#edf4f8;color:#294861;font-weight:650 }
details.mechanism { margin:10px 0;border-left:4px solid #6c93ad;background:#f8fbfd }
details.mechanism>summary { cursor:pointer;padding:9px 12px;background:#e7f0f5;color:#294861;font-weight:700 }
div.annotation-note { margin:10px 0 16px;padding:11px 13px;border-left:4px solid #2e6585;background:#e7f0f5;color:#304147 }
div.annotation-model-warning { margin:10px 0 16px;padding:11px 13px;border-left:4px solid #a66a12;background:#fff4dc;color:#69450f }
div.production-qualification-warning { margin:12px 0 18px;padding:13px 15px;border:2px solid #a83d3d;background:#f9e5e3;color:#722929;font-size:15px }
div.evidence-authority-intro { margin:12px 0 18px;padding:13px 15px;border-left:4px solid #2e6585;background:#e7f0f5;color:#304147 }
details.evidence-provenance { margin:10px 0 18px;border-left:4px solid #6c93ad;background:#f8fbfd }
details.evidence-provenance>summary { cursor:pointer;padding:9px 12px;background:#e7f0f5;color:#294861;font-weight:700 }
details.evidence-provenance>div { padding:8px 12px 12px }
div.annotation-scope { display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 14px }
div.annotation-scope>span { padding:4px 7px;border:1px solid #cad4d8;border-radius:4px;background:#f7f9f8;font-size:12px }
div.annotation-scope>span.available { color:#2c704e;border-color:#9dc2ad;background:#e2f0e8 }
details.annotation-comparison { margin:10px 0;border:1px solid #aebfcd;border-radius:6px;background:#fff }
details.annotation-comparison>summary { cursor:pointer;padding:10px 12px;background:#e4ecee;color:#304147;font-weight:700 }
details.annotation-comparison>div.annotation-comparison-body { padding:4px 12px 12px }
div.annotation-table-wrap { overflow-x:auto;border:1px solid #bfd0de;border-radius:5px;margin:10px 0 14px }
table.annotation-metric-table { border-collapse:collapse;width:100%;font-size:13px;background:#fff }
table.annotation-metric-table th { padding:7px 9px;border-bottom:2px solid #8aa8bd;background:#d9e8f2;color:#20384b;text-align:left }
table.annotation-metric-table td { padding:7px 9px;border-bottom:1px solid #d6e0e8;vertical-align:top }
table.annotation-metric-table tr:nth-child(even) { background:#f2f7fa }
span.annotation-change { display:inline-block;min-width:76px;padding:2px 6px;border-radius:4px;text-align:center;font-weight:700 }
span.annotation-change.improved { color:#2c704e;background:#e2f0e8 }
span.annotation-change.regressed { color:#a83d3d;background:#f9e5e3 }
span.annotation-change.stable { color:#2e6585;background:#e7f0f5 }
span.annotation-change.unavailable { color:#607078;background:#edf0ef }
details.annotation-detail { margin:9px 0;border-left:4px solid #d8b77d;background:#fffaf0 }
details.annotation-detail>summary { cursor:pointer;padding:9px 12px;color:#80531b;font-weight:700 }
details.annotation-detail>div { padding:0 12px 10px }
table:not(.annotation-metric-table) { border-collapse:collapse;width:100%;font-size:13px;background:#fff;margin:8px 0 16px }
table:not(.annotation-metric-table) th { padding:7px 9px;border-bottom:2px solid #8aa8bd;background:#d9e8f2;color:#20384b;text-align:left }
table:not(.annotation-metric-table) td { padding:7px 9px;border-bottom:1px solid #d6e0e8;vertical-align:top }
table:not(.annotation-metric-table) tr:nth-child(even) { background:#f2f7fa }
summary::marker { color:#4f7f9f }
</style>"""
    lines = [
        "# Depth-map Development Report", "",
        "Depth-map estimation and filtering only. Fusion, mesh reconstruction, refinement, and texturing are out of scope.", "",
        "Use the [interactive investigation interface](02_investigation.html) for synchronized maps, ranked regressions, logical-iteration controls, and pixel inspection. This Markdown document remains the canonical report.", "",
        style, "",
        *production_qualification_warning_markdown(outputs),
        *evidence_context_markdown(evidence_context),
        *dmap_report_model.render_investigation_guide_markdown().splitlines(), "",
        "## 1. Executive Summary", "",
    ]
    if not accuracy_ledger.empty:
        strict_rows = [
            row for row in accuracy_ledger.to_dict("records")
            if row.get("strict_accuracy_gate_available") is True
        ]
        if strict_rows:
            strict_passing = [
                row for row in strict_rows if row.get("strict_accuracy_pass") is True
            ]
            if strict_passing:
                passing_labels = ", ".join(
                    str(row.get("candidate")) for row in strict_passing
                )
                lines.extend([
                    f"**Strict accuracy gate: {len(strict_passing)} of {len(strict_rows)} "
                    f"candidates pass ({passing_labels}).**", "",
                ])
            else:
                lines.extend([
                    f'<div class="production-qualification-warning"><strong>Strict accuracy '
                    f'gate: 0 of {len(strict_rows)} candidates pass.</strong> No tested candidate '
                    'meets the configured accuracy-first promotion criteria.</div>', "",
                ])
        accuracy_rows = []
        for row in accuracy_ledger.to_dict("records"):
            availability = (
                f"biased: {int(row.get('lost_baseline_fit_count', 0))} lost"
                if bool(row.get("availability_biased")) else "paired"
            )
            coverage = "advisory loss" if bool(row.get("coverage_advisory")) else "within advisory floor"
            strict_status = (
                "pass" if row.get("strict_accuracy_pass") is True else
                "fail" if row.get("strict_accuracy_gate_available") is True else
                "not configured"
            )
            accuracy_rows.append([
                int(row.get("accuracy_rank", 0)), row.get("candidate"), strict_status,
                row.get("noise_class"), availability, int(row.get("scene_count", 0)),
                fmt(row.get("worst_normalized_noise_loss"), 2),
                fmt_percentage_points(row.get("effective_coverage_delta")),
                fmt_percentage_points(row.get("spatial_coverage_delta")),
                fmt_percentage_points(row.get("valid_coverage_delta")),
                fmt_percentage_points(row.get("endpoint_valid_depth_coverage_delta")),
                coverage, fmt_percent(row.get("runtime_relative_delta")),
            ])
        lines.extend([
            "### Accuracy-first Candidate Ledger", "",
            "Candidates are ordered lexicographically by paired geometric-noise evidence; coverage and production-endpoint runtime are secondary context. The estimator-stage validity delta preserves the legacy pre-postprocess meaning of `valid_coverage_delta`. Final endpoint coverage is read independently from the terminal production DMAP after optional speckle removal and gap filling. The five-percentage-point coverage floor is advisory and never hides a result. A lost baseline-successful fit makes the paired noise reading availability-biased and therefore inconclusive. These fits measure self-consistency, not absolute depth accuracy.", "",
            md_table(
                ["rank", "candidate", "strict gate", "noise class", "availability", "scenes", "worst loss / tolerance", "effective coverage delta", "spatial delta", "estimator validity delta", "final endpoint validity delta", "coverage note", "runtime delta"],
                accuracy_rows,
            ), "",
        ])
        if not accuracy_evidence.empty:
            evidence_rows = [[
                row.get("candidate"), row.get("scene_id"), row.get("metric"),
                row.get("direction"), int(row.get("paired_structures", 0)),
                fmt(row.get("baseline_value")), fmt(row.get("candidate_value")),
                fmt(row.get("mean_delta")), fmt(row.get("effective_tolerance")),
                fmt(row.get("normalized_regression_loss"), 2),
            ] for row in accuracy_evidence.to_dict("records")]
            lines.extend([
                '<details class="annotation-detail"><summary>Per-scene normalized noise evidence</summary><div>',
                md_table(
                    ["candidate", "scene", "metric", "better", "paired", "baseline", "candidate", "delta", "tolerance", "loss / tolerance"],
                    evidence_rows,
                ),
                "</div></details>", "",
            ])
    if not stability.empty and "large_model_switch" in stability.columns:
        switches = stability[
            stability["large_model_switch"].fillna(False).astype(bool)
        ]
        nearby = switches[
            switches.get(
                "near_candidate_regression", pd.Series(False, index=switches.index)
            ).fillna(False).astype(bool)
        ]
        if not switches.empty:
            lines.extend([
                '<div class="annotation-model-warning"><strong>Competing RANSAC models '
                f'detected.</strong> {len(switches)} paired annotation fit(s) cross the '
                f'large-model-change thresholds; {len(nearby)} occur in candidate/scene '
                'cohorts with a normalized regression above 1. Current-fit residuals '
                'refit each run independently and may therefore compare different '
                'structures. Inspect the fixed baseline-model-on-candidate-points '
                'residual before attributing these rows to lower noise.</div>', "",
            ])
    gate_rows = [
        [row["candidate"], row["metric"], row["status"], fmt(row["mean_delta"]), f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}]", fmt(row["effective_tolerance"]), row["scene_count"]]
        for row in gates
    ]
    lines.extend([md_table(["variant", "metric", "status", "delta", "95% CI", "tolerance", "scenes"], gate_rows), ""])
    lines.extend(["### Pareto Summary", "", md_table(
        ["variant", "Pareto", "dominated by"],
        [[row["candidate"], "yes" if row["pareto"] else "no", ", ".join(row["dominated_by"]) or "-"] for row in pareto],
    ), "", "### Automatic Mechanism Findings", ""])
    lines.extend(f"- **{row['candidate']} [{row['strength']}]:** {row['finding']}" for row in findings)
    report_policy_flags: list[str] = []
    if allow_process_specialization_divergence_for_diagnostics(config):
        report_policy_flags.append(
            "--allow-process-specialization-divergence-for-diagnostics"
        )
    if evidence_context_path is not None:
        report_policy_flags.extend([
            "--evidence-context", str(evidence_context_path.expanduser().absolute()),
        ])
    report_policy_flag = (
        " " + shlex.join(report_policy_flags) if report_policy_flags else ""
    )
    published_report_dir = (
        published_report_dir.expanduser().absolute()
        if published_report_dir is not None else report_path.parent
    )
    reproduced_report_dir = published_report_dir.with_name(
        published_report_dir.name + "_reproduced"
    )
    validate_context_flag = (
        " --evidence-context "
        + shlex.quote(str(evidence_context_path.expanduser().absolute()))
        if evidence_context_path is not None else ""
    )
    lines.extend([
        "", "## 2. Algorithm Description", "",
        "CUDA PatchMatch initializes and iteratively refines per-pixel depth/normal hypotheses, jointly selects source views, combines photometric, prior, and optional geometric costs, then filters weak hypotheses. Maps captures use the instrumented `Process<true>` specialization to record direct score components, candidate lifecycle, update attribution, winner gaps, and per-view mechanics in the hot kernel. Read-only post-pass snapshots retain final state and explicitly labeled equal-selected-view proxies. A direct signal is exact for that instrumented invocation; it is not production-equivalent quality evidence unless the complete DMAP set also passes the separate parity gate. Downstream fusion is not instrumented.",
        "",
        "Instrumentation is disabled unless `--dmap-instrumentation-dir` is non-empty. Disabled, summary, debug, and budget-degraded candidate kernels use `Process<false>`; exact maps use `Process<true>`. The enabled specialization has higher register/local-memory pressure and must remain diagnostic-only whenever kernel-resource, bit-exact production parity, resource-admission, or Compute Sanitizer gates are not all satisfied. Disabled runs do not allocate instrumentation buffers, launch diagnostic kernels, write instrumentation artifacts, or change instrumentation-related CUDA device state. Checkerboard phases are combined for analysis and remain separate only in timing diagnostics.",
        "", "## 3. Metrics Description", "",
        "- Estimator validity is the instrumentation snapshot after CUDA keep-cost and ignore-mask filtering, before optional CPU speckle removal and gap filling. Its baseline-relative ledger field remains `valid_coverage_delta` for compatibility.",
        "- Final endpoint valid-depth coverage is read from the terminal production DMAP after all configured optional depth-map postprocessing. Its ledger field is `endpoint_valid_depth_coverage_delta`; unavailable DMAPs remain null with an explicit status and reason.",
        "- CUDA filtering is decomposed into keep-cost rejection and ignore-mask rejection. Cost-function experiments should inspect the keep-cost term rather than attributing configured mask removal or later optional postprocessing to the objective.",
        "- Plane and line metrics are 3D self-consistency proxies derived from 2D annotations; they do not measure absolute depth bias.",
        "- Annotation models are fitted at 20 mm and evaluated at 5/10/20/50 mm without refitting.",
        "- Effective inlier coverage is annotation-mask coverage multiplied by the valid-point inlier fraction.",
        "- Runtime gates and Pareto ranking use only monotonic wall time from the uninstrumented production endpoint, averaged across repeats within each scene before scene-balanced comparison. CUDA-event kernel timings from summary captures are diagnostic only.",
        "- Headline deltas are paired and scene-balanced. Gates require both a practical tolerance and a paired 95% bootstrap interval excluding zero.",
        "", "## 4. Overall Scene Metrics", "",
    ])
    summary_rows = []
    for run, group in frames.groupby("run") if not frames.empty else []:
        summary_rows.append([
            run, group["scene_id"].nunique(), len(group), fmt(group["valid_ratio_after_filter"].mean()),
            fmt(group["endpoint_valid_depth_coverage"].mean())
            if "endpoint_valid_depth_coverage" in group else "n/a",
            fmt(group["rejected_by_filter_ratio"].mean()),
            fmt(group["rejected_by_keep_cost_filter_ratio"].mean()),
            aggregate_mask_rejection_display(group),
            fmt(group["final_cost_median"].mean()), fmt(group["mean_support"].mean(), 2),
        ])
    lines.extend([md_table(["run", "scenes", "frames", "estimator validity", "final endpoint validity", "rejected total", "keep-cost rejected", "mask rejected ratio", "cost median", "mean support"], summary_rows), ""])
    if not performance.empty:
        timing_rows = [
            [run, len(group), fmt(group["kernel_ms"].mean(), 2), fmt(group["kernel_ms"].median(), 2), fmt(group["kernel_ms"].std(ddof=1), 2)]
            for run, group in performance.groupby("run")
        ]
        lines.extend([
            "### Summary-only Runtime Diagnostics", "",
            "These CUDA-event kernel timings diagnose PatchMatch phases and do not enter runtime gates or Pareto ranking.", "",
            md_table(["run", "timed frames", "mean kernel ms", "median kernel ms", "stddev ms"], timing_rows), "",
        ])
    lines.extend(["## 5. Overall Visualizations", ""])
    for title, path in plots:
        lines.extend([f"### {title}", "", f"![{title}]({relative_link(path, report_path)})", ""])
    lines.extend([
        "## 6. Annotation Consistency", "",
        '<div class="annotation-note"><strong>How to read this section.</strong> Annotated planes and lines are reconstructed in 3D and fitted with deterministic RANSAC. Higher valid coverage, effective inlier coverage, and threshold AUC are better; lower residual P95 is better. These are geometric self-consistency proxies, not absolute depth-ground-truth errors.</div>', "",
    ])
    if (
        not annotations.empty
        and "role" in annotations.columns
        and bool((annotations["role"] == "reference").any())
    ):
        reference_labels = ", ".join(sorted(
            annotations.loc[annotations["role"] == "reference", "run"].dropna().astype(str).unique()
        ))
        lines.extend([
            '<div class="annotation-note"><strong>Archived product reference.</strong> '
            f"{html.escape(reference_labels)} is evaluated only as full-resolution final geometry. "
            "It has no current cost/view/update instrumentation and is excluded from regression gates; "
            "use it as an end-metric anchor rather than a controlled mechanics ablation.</div>", "",
        ])
    annotation_rows: list[list[Any]] = []
    annotation_metric_columns = {
        "run", "stage", "annotation_kind", "coverage_fraction", "spatial_coverage_fraction",
        "effective_inlier_coverage", "inlier_threshold_auc", "all_residual_p95_m",
    }
    post = pd.DataFrame()

    def successful_annotations(group: pd.DataFrame) -> pd.DataFrame:
        if "fit_status" not in group.columns:
            return group
        return group[group["fit_status"].fillna("unavailable") == "ok"]

    def annotation_mean(group: pd.DataFrame, column: str) -> float:
        fitted = successful_annotations(group)
        if column not in fitted.columns or fitted[column].dropna().empty:
            return math.nan
        return float(fitted[column].dropna().mean())

    def unique_annotation_structures(group: pd.DataFrame) -> int:
        columns = [
            column for column in ("scene_id", "image_id", "annotation_kind", "object_id", "chunk_id")
            if column in group.columns
        ]
        return int(len(group[columns].drop_duplicates())) if columns else int(len(group))

    if not annotations.empty and annotation_metric_columns.issubset(annotations.columns):
        post = annotations[annotations["stage"] == "post_filter"] if "stage" in annotations.columns else annotations
        for (run, kind), group in post.groupby(["run", "annotation_kind"]):
            fitted = successful_annotations(group)
            repeats = int(group["repeat"].nunique()) if "repeat" in group.columns else 1
            effective_column = "effective_inlier_coverage_20mm" if "effective_inlier_coverage_20mm" in group.columns else "effective_inlier_coverage"
            annotation_rows.append([
                run, kind, unique_annotation_structures(group), repeats, f"{len(fitted)}/{len(group)}",
                fmt_percent(annotation_mean(group, "coverage_fraction")),
                fmt_percent(annotation_mean(group, "spatial_coverage_fraction")),
                fmt_percent(annotation_mean(group, effective_column)),
                fmt_percent(annotation_mean(group, "inlier_threshold_auc")),
                fmt_millimetres(annotation_mean(group, "all_residual_p95_m")),
            ])
    if not post.empty:
        scope_columns = [
            column for column in ("scene_id", "image_id", "annotation_kind", "object_id", "chunk_id")
            if column in post.columns
        ]
        unique_structures = int(len(post[scope_columns].drop_duplicates())) if scope_columns else int(len(post))
        repeat_count = int(post["repeat"].nunique()) if "repeat" in post.columns else 1
        fitted_count = int(len(successful_annotations(post)))
        scene_count = int(post["scene_id"].nunique()) if "scene_id" in post.columns else 0
        scene_word = "scene" if scene_count == 1 else "scenes"
        kinds = ", ".join(sorted(str(value) for value in post["annotation_kind"].dropna().unique()))
        lines.extend([
            '<div class="annotation-scope">'
            f"<span>{scene_count} {scene_word}</span>"
            f"<span>{unique_structures} unique structures</span>"
            f"<span>{repeat_count} repeats</span>"
            f"<span>{html.escape(kinds or 'no geometry kinds')}</span>"
            f'<span class="available">{fitted_count}/{len(post)} successful post-filter evaluations</span>'
            "</div>", "",
        ])
    lines.extend([
        "### Evaluation Coverage", "",
        md_table(
            ["run", "kind", "unique structures", "repeats", "successful / evaluations", "valid depth coverage", "spatial coverage", "effective @20 mm", "threshold AUC", "residual P95"],
            annotation_rows,
        ), "",
        "### Baseline-relative Headline Metrics", "",
        "Values are means over successful post-filter evaluation rows. Colored readings show metric direction only; use the aggregate regression gates and repeated scenes for statistical decisions.", "",
    ])

    baseline_run = ""
    if not post.empty:
        if "role" in post.columns and not post[post["role"] == "baseline"].empty:
            baseline_run = str(post[post["role"] == "baseline"]["run"].iloc[0])
        else:
            baseline_run = str(sorted(post["run"].dropna().astype(str).unique())[0])
    candidate_runs = [str(value) for value in sorted(post["run"].dropna().astype(str).unique()) if str(value) != baseline_run] if not post.empty else []
    comparison_index = 0
    threshold_sections: list[str] = []
    for candidate in candidate_runs:
        for kind in sorted(str(value) for value in post["annotation_kind"].dropna().unique()):
            baseline_group = post[(post["run"].astype(str) == baseline_run) & (post["annotation_kind"].astype(str) == kind)]
            variant_group = post[(post["run"].astype(str) == candidate) & (post["annotation_kind"].astype(str) == kind)]
            if baseline_group.empty and variant_group.empty:
                continue
            effective_column = "effective_inlier_coverage_20mm" if "effective_inlier_coverage_20mm" in post.columns else "effective_inlier_coverage"
            metric_rows = []
            for label, column, direction, formatter, delta_formatter in [
                ("Valid depth coverage", "coverage_fraction", "higher", fmt_percent, fmt_percentage_points),
                ("Spatial coverage", "spatial_coverage_fraction", "higher", fmt_percent, fmt_percentage_points),
                ("Effective inliers @20 mm", effective_column, "higher", fmt_percent, fmt_percentage_points),
                ("Threshold AUC", "inlier_threshold_auc", "higher", fmt_percent, fmt_percentage_points),
                ("Residual P95", "all_residual_p95_m", "lower", fmt_millimetres, lambda value: fmt_millimetres(value, signed=True)),
            ]:
                baseline_value = annotation_mean(baseline_group, column)
                variant_value = annotation_mean(variant_group, column)
                baseline_number = safe_float(baseline_value)
                variant_number = safe_float(variant_value)
                delta = variant_number - baseline_number if baseline_number is not None and variant_number is not None else None
                metric_rows.append({
                    "metric": label,
                    "baseline": formatter(baseline_value),
                    "variant": formatter(variant_value),
                    "delta": delta_formatter(delta),
                    "direction": direction,
                    "status": directional_status(baseline_value, variant_value, direction),
                })
            baseline_repeats = int(baseline_group["repeat"].nunique()) if "repeat" in baseline_group.columns else 1
            variant_repeats = int(variant_group["repeat"].nunique()) if "repeat" in variant_group.columns else 1
            open_state = " open" if comparison_index == 0 else ""
            comparison_index += 1
            comparison_title = (
                f"{baseline_run} vs {candidate} / {kind}: "
                f"{unique_annotation_structures(baseline_group)} baseline and {unique_annotation_structures(variant_group)} variant structures; "
                f"{baseline_repeats}/{variant_repeats} repeats"
            )
            lines.extend([
                f'<details class="annotation-comparison"{open_state}>',
                f"<summary>{html.escape(comparison_title)}</summary>",
                '<div class="annotation-comparison-body">',
                annotation_metric_table(metric_rows),
                "</div>", "</details>", "",
            ])

            threshold_rows: list[list[Any]] = []
            for threshold in (5, 10, 20, 50):
                inlier_column = f"inlier_fraction_{threshold}mm"
                effective_threshold_column = f"effective_inlier_coverage_{threshold}mm"
                baseline_inliers = annotation_mean(baseline_group, inlier_column)
                variant_inliers = annotation_mean(variant_group, inlier_column)
                baseline_effective = annotation_mean(baseline_group, effective_threshold_column)
                variant_effective = annotation_mean(variant_group, effective_threshold_column)
                threshold_rows.append([
                    f"{threshold} mm",
                    fmt_percent(baseline_inliers), fmt_percent(variant_inliers),
                    fmt_percentage_points((safe_float(variant_inliers) or 0.0) - (safe_float(baseline_inliers) or 0.0))
                    if safe_float(baseline_inliers) is not None and safe_float(variant_inliers) is not None else "n/a",
                    fmt_percent(baseline_effective), fmt_percent(variant_effective),
                    fmt_percentage_points((safe_float(variant_effective) or 0.0) - (safe_float(baseline_effective) or 0.0))
                    if safe_float(baseline_effective) is not None and safe_float(variant_effective) is not None else "n/a",
                ])
            threshold_sections.extend([
                '<details class="annotation-detail">',
                f"<summary>{html.escape(baseline_run)} vs {html.escape(candidate)} / {html.escape(kind)}</summary>",
                "<div>",
                md_table(["threshold", "baseline inliers", "variant inliers", "delta", "baseline effective", "variant effective", "delta"], threshold_rows),
                "</div>", "</details>", "",
            ])

    unavailable_annotation_rows = []
    if not annotations.empty and "fit_status" in annotations.columns:
        unavailable = annotations[annotations["fit_status"].fillna("unavailable") != "ok"]
        unavailable_annotation_rows = [
            [
                row.get("run"), row.get("repeat"), row.get("scene_id"), row.get("image_id"),
                row.get("stage") or "not reached", row.get("fit_status") or "unavailable",
                row.get("error") or "annotation metric unavailable",
            ]
            for row in unavailable.to_dict("records")
        ]
    lines.extend([
        "### Fixed-model Inlier Threshold Sweep", "",
        "The model is fitted once at 20 mm; all four thresholds evaluate that same model without refitting. Expand a comparison to separate the valid-point inlier rate from effective coverage, which also accounts for missing depth.", "",
        *threshold_sections,
    ])
    if unavailable_annotation_rows:
        lines.extend([
            '<details class="annotation-detail">',
            f"<summary>Unavailable Annotation Evidence ({len(unavailable_annotation_rows)} records)</summary>",
            "<div>",
            "Missing mappings, annotations, calibration references, and run-owned stage depth maps are retained explicitly; no other run's depth map is substituted.", "",
            md_table(["run", "repeat", "scene", "frame", "stage", "status", "reason"], unavailable_annotation_rows),
            "</div>", "</details>", "",
        ])
    elif not annotations.empty:
        lines.extend(['<div class="annotation-scope"><span class="available">All annotation evaluation records are available.</span></div>', ""])
    stability_rows = []
    if not stability.empty:
        for (candidate, kind), group in stability.groupby(["candidate", "annotation_kind"]):
            angle_column = "plane_normal_delta_deg" if kind == "plane" else "line_direction_delta_deg"
            position_column = "plane_position_delta_m" if kind == "plane" else "line_position_delta_m"
            extent_column = "line_extent_delta_m"
            angles = group[angle_column].dropna() if angle_column in group.columns else pd.Series(dtype=float)
            positions = group[position_column].dropna() if position_column in group.columns else pd.Series(dtype=float)
            extents = group[extent_column].dropna() if extent_column in group.columns else pd.Series(dtype=float)
            large_switches = int(
                group.get("large_model_switch", pd.Series(False, index=group.index))
                .fillna(False).astype(bool).sum()
            )
            regression_switches = int(
                group.get("near_candidate_regression", pd.Series(False, index=group.index))
                .fillna(False).astype(bool).sum()
            )
            stability_rows.append([
                candidate, kind, len(group), large_switches, regression_switches,
                fmt(angles.mean(), 3) if not angles.empty else "n/a",
                fmt(angles.quantile(0.95), 3) if not angles.empty else "n/a",
                fmt(positions.mean() * 1000.0, 3) if not positions.empty else "n/a",
                fmt(positions.quantile(0.95) * 1000.0, 3) if not positions.empty else "n/a",
                fmt(extents.abs().mean() * 1000.0, 3) if not extents.empty else "n/a",
            ])
    if stability_rows:
        lines.extend([
            "### Fitted-model Stability vs Baseline", "",
            "Each run is fitted independently in the current-fit metrics above. A large model switch is flagged at 5 degrees of line/plane rotation, 25 mm or 25% line-extent change, or 10 mm plane offset. Fixed baseline-model-on-candidate-points columns in `model_stability.csv` retain a common model when candidate points are available. That baseline model is a comparison anchor, not ground truth.", "",
            '<details class="annotation-detail"><summary>Angular, position, and extent changes</summary><div>',
            md_table(
                ["candidate", "kind", "models", "large switches", "switches near regression", "angular mean deg", "angular P95 deg", "position mean mm", "position P95 mm", "extent change mean mm"],
                stability_rows,
            ),
            "</div></details>", "",
        ])
        switch_detail = stability[
            stability.get(
                "large_model_switch", pd.Series(False, index=stability.index)
            ).fillna(False).astype(bool)
        ]
        if not switch_detail.empty:
            switch_rows = []
            for row in switch_detail.to_dict("records"):
                switch_rows.append([
                    row.get("candidate"), str(row.get("scene_id") or "")[:8],
                    row.get("annotation_kind"), str(row.get("chunk_id") or "")[:8],
                    "yes" if bool(row.get("near_candidate_regression")) else "no",
                    row.get("model_switch_reasons_json"),
                    fmt_millimetres(row.get("baseline_current_fit_all_residual_p95_m")),
                    fmt_millimetres(row.get("candidate_current_fit_all_residual_p95_m")),
                    fmt_millimetres(row.get("baseline_model_on_candidate_all_residual_p95_m")),
                    row.get("baseline_model_on_candidate_status") or "unavailable",
                ])
            lines.extend([
                '<details class="annotation-detail"><summary>Large model-switch evidence</summary><div>',
                md_table(
                    ["candidate", "scene", "kind", "chunk", "near regression", "switch reasons", "baseline current-fit P95", "candidate current-fit P95", "fixed baseline-model P95 on candidate", "fixed-model status"],
                    switch_rows,
                ),
                "</div></details>", "",
            ])
        lines.extend([
            f"- **Machine-readable model stability and fixed-model cross-evaluation:** "
            f"{report_output_links(outputs.get('model_stability') or {}, report_path)}",
            "",
        ])
    else:
        lines.extend([
            "### Fitted-model Stability vs Baseline", "",
            "_No paired fitted-model stability evidence is available for this report._", "",
        ])
    lines.extend([
        "## 7. Algorithm Mechanics", "",
        "Use this index to move from a code change to the evidence that can confirm or reject its intended mechanism. Aggregate dashboards show whether behavior moved consistently; per-frame dashboards show where and why.", "",
        md_table(
            ["mechanism", "primary evidence", "development question"],
            [
                ["Cost Function and Convergence", "cost maps, components, closure residuals, distributions, evolution, improvement iterations", "Did the objective improve for the intended pixels, and did lower cost translate to better geometry?"],
                ["View Selection and Support", "support histograms, entropy, per-view weights/costs, transitions, churn", "Did the selector choose stable, informative views rather than merely changing support count?"],
                ["Update Dynamics", "changed ratios, accepted updates, depth/normal deltas, last-change maps", "Did hypotheses converge, stall, oscillate, or keep changing late?"],
                ["Filtering and Completeness", "validity, rejection counts/reasons, before/after and transition maps", "Did the change retain more correct structure or simply relax rejection?"],
                ["Runtime and Scalability", "production endpoint monotonic full-process wall time, paired per scene after repeat aggregation", "Did the change improve quality within an acceptable decision-grade runtime budget?"],
                ["Observer Timing Diagnostics", "summary CUDA-event per-pass and per-frame kernel timing", "Which estimator phases explain the runtime movement? These diagnostics do not drive runtime gates."],
                ["Geometric End Metrics", "plane/line coverage, fixed-model inliers, residual tails, model stability", "Did the final depth maps become more geometrically self-consistent?"],
                ["Cross-run Effects", "paired atlases, CDFs, scene-balanced deltas, gates, Pareto status", "Where does the variant differ from the baseline and is that difference repeatable?"],
                ["Final State Overview", "RGB, depth, normal, final cost, support, rejection, update state", "What scene content explains the metric and mechanism changes?"],
            ],
        ), "",
    ])
    plot_counts: dict[str, int] = {name: 0 for name in MECHANISM_ORDER}
    panel_counts: dict[str, int] = {name: 0 for name in MECHANISM_ORDER}
    for scene_plots in instrumentation_plots.values():
        for title, _path in scene_plots:
            plot_counts[mechanism_for_plot(title)] = plot_counts.get(mechanism_for_plot(title), 0) + 1
    for scene_panels in diagnostic_panels.values():
        for panel in scene_panels:
            panel_counts[panel.mechanism] = panel_counts.get(panel.mechanism, 0) + 1
    coverage_rows = []
    for mechanism in MECHANISM_ORDER:
        available = plot_counts.get(mechanism, 0) + panel_counts.get(mechanism, 0)
        coverage_rows.append([MECHANISM_LABELS[mechanism], "available" if available else "unavailable", plot_counts.get(mechanism, 0), panel_counts.get(mechanism, 0)])
    fitted_annotations = (
        int((annotations["fit_status"] == "ok").sum())
        if not annotations.empty and "fit_status" in annotations.columns else 0
    )
    coverage_rows.append(["Geometric End Metrics", "available" if fitted_annotations else "unavailable", 0, fitted_annotations])
    lines.extend(["### Instrumentation Coverage", "", md_table(["mechanism", "status", "aggregate plots", "frame/metric evidence"], coverage_rows), ""])
    map_catalog_output = outputs.get("map_catalog") or {}
    availability_output = outputs.get("signal_availability") or {}
    validation_output = outputs.get("instrumentation_validation") or {}
    lines.extend([
        "### Map Schema and Provenance", "",
        "Available maps are cataloged from each frame's `map_manifest.json` plus the versioned `postprocess_filters.json` and `confidence_adjustment.json` stage contracts; schema-v3/v4 map identity is never inferred from checkerboard filenames. For an atomically completed summary-only frame, `summary.json`, its resource plan, and final-scale logical rows in `iteration.csv` instead create explicit unavailable-signal inventory without inventing map artifacts. Schema-v4 map identity includes source-view and channel provenance for exact hot-kernel artifacts, while optional-filter maps retain their sequential algorithm-stage identity. Measurement quality, basis, proxy target, limitations, declared bytes, artifact existence, and unavailability reasons remain machine-readable. Schema-v2 maps are cataloged as declared but are not evaluated against the logical-state completeness contract.", "",
        md_table(
            ["artifact", "rows", "available/existing", "unavailable"],
            [
                ["map catalog", map_catalog_output.get("rows", 0), map_catalog_output.get("existing_rows", 0), "n/a"],
                ["required signal availability", availability_output.get("rows", 0), availability_output.get("available_rows", 0), availability_output.get("unavailable_rows", 0)],
                ["validated frames", validation_output.get("rows", 0), int(validation_output.get("rows", 0)) - int(validation_output.get("invalid_frames", 0)), validation_output.get("invalid_frames", 0)],
                ["production qualification", validation_output.get("production_qualification_status", "unavailable"), "yes" if validation_output.get("production_parity_qualified") else "no", f"diagnostic divergence frames: {validation_output.get('diagnostic_process_specialization_divergence_frames', 0)}"],
                ["validator warnings", validation_output.get("warning_count", 0), validation_output.get("warning_frames", 0), "advisory; evidence retained"],
                ["production endpoint DMAP sets", validation_output.get("endpoint_dmap_sets_checked", 0), validation_output.get("endpoint_dmap_sets_bit_exact", 0), "file-hash exact"],
                ["production endpoint DMAP files", validation_output.get("endpoint_dmaps_shared", 0), validation_output.get("endpoint_dmaps_shared", 0), "complete set"],
                ["deep/summary parity frames", validation_output.get("maps_summary_frames_checked", 0), validation_output.get("maps_summary_frames_bit_exact", 0), "numeric component parity"],
            ],
        ), "",
        f"- **Map catalog:** {report_output_links(map_catalog_output, report_path)}",
        f"- **Signal availability:** {report_output_links(availability_output, report_path)}",
        f"- **Frame/schema/parity validation:** {report_output_links(validation_output, report_path)}", "",
    ])
    validator_warnings = validation_output.get("warnings") or []
    if validator_warnings:
        warning_rows = []
        for warning in validator_warnings:
            maximum_weight = safe_float(warning.get("maximum_weight"))
            maximum_overshoot = safe_float(warning.get("maximum_overshoot"))
            warning_rows.append([
                warning.get("run"), warning.get("scene_id"), warning.get("frame"),
                warning.get("estimation_stage"),
                warning.get("geometric_iteration") if warning.get("geometric_iteration") is not None else "-",
                warning.get("signal"), warning.get("pixels"),
                f"{maximum_weight:.9g}" if maximum_weight is not None else "unavailable",
                f"{maximum_overshoot:.3g}" if maximum_overshoot is not None else "unavailable",
                warning.get("maximum_formula_ulp_error", "unavailable"), warning.get("code"),
            ])
        lines.extend([
            "#### Validator Warnings", "",
            "These captures pass validation, but retain known production numeric behavior as explicit evidence. Warnings are advisory only when the captured inputs reproduce the recorded value within the stated float32 tolerance; unexplained domain violations remain fatal.", "",
            md_table(
                ["run", "scene", "frame", "stage", "geom iter", "signal", "pixels", "max weight", "max overshoot", "max ULP error", "warning"],
                warning_rows,
            ), "",
        ])
    else:
        lines.extend(["#### Validator Warnings", "", "No validator warnings were emitted.", ""])
    lines.extend(["### Resource Admission and Storage Preflight", ""])
    resource_rows = []
    for table in (cuda_resource_plans, filter_resource_plans):
        for row in table.to_dict("records") if not table.empty else []:
            resource_rows.append([
                row.get("run"), row.get("scene_id"), row.get("image_id"), row.get("estimation_stage"),
                row.get("geometric_iteration") if not pd.isna(row.get("geometric_iteration")) else "-",
                row.get("pyramid_level") if not pd.isna(row.get("pyramid_level")) else "-",
                row.get("component"), row.get("decision"), row.get("trace_requested"),
                row.get("trace_available"), row.get("maps_requested"),
                row.get("maps_available"), fmt_mib(row.get("effective_device_bytes")),
                fmt_mib(row.get("effective_host_bytes")), fmt_mib(row.get("effective_storage_bytes")),
                fmt_mib(row.get("current_pyramid_storage_bytes")),
                fmt_mib(row.get("frame_storage_committed_before_bytes")),
                fmt_mib(row.get("full_resolution_priority_reserve_bytes")),
                fmt_mib(row.get("storage_frame_priority_reservation_bytes")),
                row.get("storage_frame_priority_reservation_consumed")
                if row.get("storage_frame_priority_reservation_consumed") is not None
                else "-",
                row.get("storage_preflight_succeeded"), row.get("lease_released"),
                row.get("actual_map_count") if not pd.isna(row.get("actual_map_count")) else "-",
                fmt_mib(row.get("actual_declared_bytes")),
                "unavailable" if pd.isna(row.get("valid")) else row.get("valid"),
                row.get("reason") or "-",
            ])
    if resource_rows:
        lines.extend([
            "Admission decisions are read from the CUDA PatchMatch planner and the per-frame optional-filter planner. Effective bytes are estimates admitted before observer allocation; actual declared bytes are reported separately where available. CUDA frame storage is cumulative across pyramid levels, and coarse levels hold the reported full-resolution reserve until the fine-level admission atomically consumes it.", "",
            md_table(
                ["run", "scene", "frame", "capture stage", "geo iter", "pyramid", "component", "decision", "trace requested", "trace admitted", "maps requested", "maps admitted", "device MiB", "host MiB", "frame MiB", "current pyramid MiB", "committed before MiB", "full-res reserve MiB", "held reserve MiB", "reserve consumed", "preflight", "lease released", "actual maps", "actual declared MiB", "valid", "reason"],
                resource_rows,
            ), "",
        ])
    else:
        lines.extend(["_Unavailable: no resource-plan observations were discovered._", ""])
    lines.extend([
        f"- **CUDA resource plans:** {report_output_links(outputs.get('cuda_resource_plans') or {}, report_path)}",
        f"- **Filter resource plans:** {report_output_links(outputs.get('filter_resource_plans') or {}, report_path)}",
        f"- **Resource-plan validation:** {report_output_links(outputs.get('resource_plan_validation') or {}, report_path)}", "",
    ])
    lines.extend(["### Exact Production Cost Evolution", ""])
    if exact_cost_evolution.empty:
        lines.extend(["_Unavailable: no schema-v4 exact hot-kernel cost maps were admitted for these captures._", ""])
    else:
        exact_cost_rows = [
            [
                row.get("run"), row.get("scene_id"), row.get("image_id"),
                row.get("estimation_stage"),
                instrumentation_report.logical_iteration_label(int(row.get("logical_iteration", -1))),
                row.get("signal"), fmt(row.get("available_ratio")), fmt(row.get("mean")),
                fmt(row.get("median")), fmt(row.get("p90")), row.get("measurement_quality"),
            ]
            for row in exact_cost_evolution.to_dict("records")
        ]
        lines.extend([
            "These statistics are read from exact production hot-kernel maps. Retained `cost_stored` remains alongside them as the production-state closure reference.", "",
            md_table(["run", "scene", "frame", "capture stage", "logical state", "signal", "available", "mean", "median", "P90", "quality"], exact_cost_rows), "",
        ])
        baseline_labels = exact_cost_evolution.loc[
            exact_cost_evolution["run_role"] == "baseline", "run"
        ].dropna().astype(str).unique().tolist()
        comparison_rows = []
        if baseline_labels:
            baseline_label = baseline_labels[0]
            keys = ["repeat", "scene_id", "image_id", "estimation_stage", "geometric_iteration", "logical_iteration", "signal"]
            baseline_values = exact_cost_evolution[exact_cost_evolution["run"] == baseline_label][keys + ["median", "p90"]]
            for candidate_label in sorted(set(exact_cost_evolution["run"].astype(str)) - {baseline_label}):
                candidate_values = exact_cost_evolution[exact_cost_evolution["run"] == candidate_label][keys + ["median", "p90"]]
                paired = baseline_values.merge(candidate_values, on=keys, suffixes=("_baseline", "_candidate"))
                for row in paired.to_dict("records"):
                    comparison_rows.append([
                        candidate_label, row["scene_id"], row["image_id"],
                        row["estimation_stage"],
                        instrumentation_report.logical_iteration_label(int(row["logical_iteration"])), row["signal"],
                        fmt(row["median_baseline"]), fmt(row["median_candidate"]),
                        fmt(row["median_candidate"] - row["median_baseline"]),
                        fmt(row["p90_candidate"] - row["p90_baseline"]),
                    ])
        lines.extend([
            "#### Baseline vs Variant Exact-cost Deltas", "",
            md_table(["variant", "scene", "frame", "capture stage", "state", "signal", "baseline median", "variant median", "median delta", "P90 delta"], comparison_rows), "",
        ])
        final_rows = []
        final_keys = ["run", "repeat", "scene_id", "image_id", "estimation_stage", "geometric_iteration", "signal"]
        final_costs = exact_cost_evolution.copy()
        final_costs["final_iteration"] = final_costs.groupby(final_keys, dropna=False)["logical_iteration"].transform("max")
        final_costs = final_costs[final_costs["logical_iteration"] == final_costs["final_iteration"]]
        if baseline_labels:
            baseline_label = baseline_labels[0]
            endpoint_keys = ["repeat", "scene_id", "image_id", "estimation_stage", "geometric_iteration", "signal"]
            baseline_endpoint = final_costs[final_costs["run"] == baseline_label][
                endpoint_keys + ["logical_iteration", "median", "p90"]
            ]
            for candidate_label in sorted(set(final_costs["run"].astype(str)) - {baseline_label}):
                candidate_endpoint = final_costs[final_costs["run"] == candidate_label][
                    endpoint_keys + ["logical_iteration", "median", "p90"]
                ]
                paired = baseline_endpoint.merge(
                    candidate_endpoint, on=endpoint_keys, suffixes=("_baseline", "_candidate")
                )
                for row in paired.to_dict("records"):
                    final_rows.append([
                        candidate_label, row["repeat"], row["scene_id"], row["image_id"], row["estimation_stage"], row["signal"],
                        instrumentation_report.logical_iteration_label(int(row["logical_iteration_baseline"])),
                        instrumentation_report.logical_iteration_label(int(row["logical_iteration_candidate"])),
                        fmt(row["median_baseline"]), fmt(row["median_candidate"]),
                        fmt(row["median_candidate"] - row["median_baseline"]),
                        fmt(row["p90_candidate"] - row["p90_baseline"]),
                    ])
        lines.extend([
            "#### Final State per Run", "",
            "This endpoint alignment compares each run's own final logical state, so iteration-count experiments remain directly comparable without losing same-iteration diagnostics.", "",
            md_table(
                ["variant", "repeat", "scene", "frame", "capture stage", "signal", "baseline endpoint", "variant endpoint", "baseline median", "variant median", "median delta", "P90 delta"],
                final_rows,
            ), "",
        ])
        trajectory_plots = [
            (title, path) for title, path in plots
            if title.startswith("Exact cost-component trajectories")
        ]
        if trajectory_plots:
            lines.extend(["#### Exact Component Trajectories", ""])
            for title, path in trajectory_plots:
                lines.extend([f"![{title}]({relative_link(path, report_path)})", ""])
    lines.extend(["### Exact Update Attribution and Winner Gap", ""])
    if exact_iterations.empty:
        lines.extend(["_Unavailable: `exact_iteration.csv` was not produced for these captures._", ""])
    else:
        source_columns = sorted(column for column in exact_iterations.columns if column.startswith("source_") and column != "source_csv")
        update_rows = []
        for row in exact_iterations.to_dict("records"):
            logical_iteration = safe_float(row.get("logical_iteration"))
            initialization = logical_iteration is None or logical_iteration < 0
            sources = ", ".join(
                f"{column.removeprefix('source_')}={int(safe_float(row.get(column)) or 0)}"
                for column in source_columns if (safe_float(row.get(column)) or 0) > 0
            ) or "none"
            update_rows.append([
                row.get("run"), row.get("scene_id"), row.get("image_id"), row.get("stage"),
                int(safe_float(row.get("tested_candidates")) or 0), int(safe_float(row.get("finite_candidates")) or 0),
                int(safe_float(row.get("accepted_candidates")) or 0), fmt(row.get("gap_mean")),
                "stored initialization assignments" if initialization else "sequential candidate acceptances",
                fmt(row.get("gap_p50")), fmt(row.get("gap_p90")), sources,
            ])
        update_explanation = (
            "For initialization, `accepted_candidates` means stored initialization assignments, "
            "including fallback/storage paths, so it can exceed the usable or finite-candidate "
            "count. Iterative rows count sequential candidate acceptances."
        )
        if low_texture_hysteresis.get("rows"):
            update_explanation += (
                " The optional low-texture extension can suppress a raw minimum; use its "
                "registered raw-order maps in those pixels."
            )
        lines.extend([
            update_explanation, "",
            md_table(
            ["run", "scene", "frame", "state", "tested", "finite", "accepted / stored", "meaning", "gap mean", "gap P50", "gap P90", "source counts"],
            [row[:7] + [row[8], row[7], *row[9:]] for row in update_rows],
        ), ""])
    hysteresis_rows = []
    for row in low_texture_hysteresis.get("rows") or []:
        logical_iteration = safe_float(row.get("logical_iteration"))
        hysteresis_rows.append([
            row.get("run"), row.get("scene_id"), row.get("image_id"),
            row.get("pyramid_level", row.get("scale_level", "-")),
            instrumentation_report.logical_iteration_label(int(logical_iteration))
            if logical_iteration is not None else "-",
            row.get("stage"),
            int(safe_float(row.get("eligible_pixels")) or 0),
            int(safe_float(row.get("propagation_accepted")) or 0),
            int(safe_float(row.get("propagation_rejected")) or 0),
            fmt_percent(row.get("propagation_rejection_rate")),
            int(safe_float(row.get("refinement_accepted")) or 0),
            int(safe_float(row.get("refinement_rejected")) or 0),
            fmt_percent(row.get("refinement_rejection_rate")),
            fmt(row.get("mean_required_gain"), 6),
            fmt(row.get("mean_best_proposed_gain"), 6),
        ])
    if hysteresis_rows:
        lines.extend([
            "### Low-texture Update Hysteresis (Optional Extension)",
            "",
            "Counts combine both checkerboards into one logical iteration. Rejection rates use accepted plus rejected legacy-improving gate-controlled proposals; refinement can contribute multiple sequential proposals at one pixel. Gain means divide by eligible pixels.",
            "",
            "When a raw minimum is suppressed, `gap_winner_runner_up_exact` and `candidate_runner_up_cost_exact` are unavailable rather than mixing retained and raw orderings. Inspect `candidate_raw_best_cost_exact`, `candidate_raw_runner_up_cost_exact`, `gap_raw_best_runner_up_exact`, `candidate_retained_minus_raw_best_exact`, and `candidate_raw_suppression_identity_exact` for the complete raw ordering and attribution.",
            "",
            md_table(
                [
                    "run", "scene", "frame", "pyramid", "logical iteration", "state", "eligible pixels",
                    "propagation accepted", "propagation rejected", "propagation rejection",
                    "refinement accepted", "refinement rejected", "refinement rejection",
                    "mean required gain", "mean best proposed gain",
                ],
                hysteresis_rows,
            ),
            "",
        ])
    gain_rows = []
    for row in accepted_gain_census.get("rows") or []:
        fractions = row.get("fractions_below") or {}
        quantiles = row.get("gain_quantiles") or {}
        gain_rows.append([
            row.get("run"), row.get("scene_id"), row.get("image_id"),
            instrumentation_report.logical_iteration_label(
                int(safe_float(row.get("logical_iteration")) or 0)
            ),
            fmt(row.get("variance_max"), 6),
            int(safe_float(row.get("low_texture_pixels")) or 0),
            int(safe_float(row.get("accepted_gain_pixels")) or 0),
            fmt_percent(row.get("accepted_fraction_of_low_texture")),
            fmt(quantiles.get("p50"), 6),
            fmt_percent(fractions.get("0.00025")),
            fmt_percent(fractions.get("0.0005")),
            fmt_percent(fractions.get("0.001")),
        ])
    if gain_rows:
        lines.extend([
            "#### Low-texture Accepted-gain Census (Optional Extension)",
            "",
            "This deterministic pre-change census measures positive exact iteration-entry incumbent minus retained-winner gains in pixels below the run's configured reference-variance threshold. It does not infer coarse-prior eligibility or sequential proposal attribution.",
            "",
            md_table(
                [
                    "run", "scene", "frame", "state", "variance max", "low-texture pixels",
                    "accepted gain pixels", "accepted / low texture", "gain P50",
                    "below 0.00025", "below 0.0005", "below 0.001",
                ],
                gain_rows,
            ),
            "",
        ])
    lines.extend(["### Exact Per-view Selection and Contribution", ""])
    if exact_views.empty:
        lines.extend(["_Unavailable: `exact_view_summary.csv` was not produced for these captures._", ""])
    else:
        decision_columns = sorted(column for column in exact_views.columns if column.startswith("decision_"))
        view_rows = []
        for row in exact_views.to_dict("records"):
            pixels = max(1.0, safe_float(row.get("pixels")) or 1.0)
            decisions = ", ".join(
                f"{column.removeprefix('decision_')}={int(safe_float(row.get(column)) or 0)}"
                for column in decision_columns if (safe_float(row.get(column)) or 0) > 0
            ) or "none"
            view_rows.append([
                row.get("run"), row.get("stage"), row.get("source_view_index"), row.get("source_image_id"),
                fmt((safe_float(row.get("selected_pixels")) or 0.0) / pixels), fmt(row.get("weight_mean")),
                fmt(row.get("probability_mean")), fmt(row.get("weighted_contribution_mean")),
                fmt(row.get("photometric_cost_mean")), fmt(row.get("geometric_cost_mean")), decisions,
            ])
        lines.extend([md_table(
            ["run", "state", "view", "image", "selected ratio", "weight", "probability", "contribution", "photo", "geometric", "decision counts"],
            view_rows,
        ), ""])
    lines.extend(["### CPU View Ranking and Estimation Selection", ""])
    if cpu_view_candidates.empty and cpu_estimation_selection.empty:
        lines.extend(["_Unavailable: CPU view-candidate ranking/selection artifacts were not produced for these captures._", ""])
    else:
        cpu_rows = []
        for row in cpu_view_candidates.to_dict("records"):
            cpu_rows.append([
                row.get("run"), row.get("image_id"), row.get("candidate_image_id"), row.get("raw_rank_zero_based"),
                fmt(row.get("ranking_score")), row.get("initial_decision"), row.get("filter_decision"),
                row.get("final_rank_zero_based"), row.get("accepted_after_filter"),
            ])
        lines.extend([md_table(
            ["run", "reference", "candidate", "raw rank", "score", "initial decision", "filter decision", "final rank", "accepted"],
            cpu_rows,
        ), ""])
        selection_rows = [[
            row.get("run"), row.get("image_id"), row.get("admission_policy"), row.get("score_threshold_applied"),
            fmt(row.get("configured_effective_min_score")), row.get("configured_score_threshold_status"),
            row.get("candidate_image_id"), row.get("filtered_rank_zero_based"), fmt(row.get("ranking_score")),
            fmt(row.get("score_ratio_to_best")), row.get("would_pass_configured_score_threshold"),
            row.get("selected_rank_zero_based"), row.get("decision"),
        ] for row in cpu_estimation_selection.to_dict("records")]
        lines.extend([md_table(
            ["run", "reference", "admission policy", "score cutoff applied", "configured cutoff", "cutoff status", "candidate", "filtered rank", "score", "ratio to best", "would pass configured cutoff", "selected rank", "estimation decision"],
            selection_rows,
        ), ""])
    lines.extend(["### Sequential Postprocess Filtering", ""])
    if postprocess_filters.empty:
        lines.extend(["_Unavailable: no postprocess-filter observation rows were discovered._", ""])
    else:
        filter_rows = [[
            row.get("run"), row.get("scene_id"), row.get("image_id"), row.get("estimation_stage"),
            row.get("geometric_iteration") if not pd.isna(row.get("geometric_iteration")) else "-",
            int(safe_float(row.get("stage_index")) or 0), row.get("stage_name"), row.get("artifact_status"),
            row.get("enabled"), row.get("executed"), row.get("success"),
            int(safe_float(row.get("input_valid_depth_pixels")) or 0) if safe_float(row.get("input_valid_depth_pixels")) is not None else "-",
            int(safe_float(row.get("output_valid_depth_pixels")) or 0) if safe_float(row.get("output_valid_depth_pixels")) is not None else "-",
            int(safe_float(row.get("removed_pixels")) or 0) if safe_float(row.get("removed_pixels")) is not None else "-",
            int(safe_float(row.get("added_pixels")) or 0) if safe_float(row.get("added_pixels")) is not None else "-",
            int(safe_float(row.get("depth_changed_pixels")) or 0) if safe_float(row.get("depth_changed_pixels")) is not None else "-",
            fmt(row.get("depth_abs_delta_mean_all_pixels")), row.get("unavailable_reason") or "-",
        ] for row in postprocess_filters.to_dict("records")]
        lines.extend([
            "Rows preserve execution order and compare each stage against the immediately preceding production state. Disabled rows are exact identities; unavailable rows are never interpreted as no-op stages.", "",
            md_table(
                ["run", "scene", "frame", "capture stage", "geo iter", "sequence", "stage", "status", "enabled", "executed", "success", "valid in", "valid out", "removed", "added", "depth changed", "mean |depth delta|", "unavailable reason"],
                filter_rows,
            ), "",
            f"- **Machine-readable stage rows:** {report_output_links(outputs.get('postprocess_filters') or {}, report_path)}", "",
        ])
    lines.extend(["### Confidence Adjustment", ""])
    if confidence_adjustment.empty:
        lines.extend(["_Unavailable: no confidence-adjustment observation rows were discovered._", ""])
    else:
        confidence_rows = [[
            row.get("run"), row.get("scene_id"), row.get("image_id"), row.get("estimation_stage"),
            row.get("geometric_iteration") if not pd.isna(row.get("geometric_iteration")) else "-",
            row.get("method"), row.get("artifact_status"), row.get("enabled"), row.get("executed"),
            row.get("output_available"),
            int(safe_float(row.get("input_positive_confidence_pixels")) or 0) if safe_float(row.get("input_positive_confidence_pixels")) is not None else "-",
            int(safe_float(row.get("output_positive_confidence_pixels")) or 0) if safe_float(row.get("output_positive_confidence_pixels")) is not None else "-",
            int(safe_float(row.get("changed_pixels")) or 0) if safe_float(row.get("changed_pixels")) is not None else "-",
            fmt(row.get("abs_delta_mean_all_pixels")), row.get("final_combination") or "-",
            row.get("unavailable_reason") or "-",
        ] for row in confidence_adjustment.to_dict("records")]
        lines.extend([
            md_table(
                ["run", "scene", "frame", "capture stage", "geo iter", "method", "status", "enabled", "executed", "output", "positive in", "positive out", "changed", "mean |confidence delta|", "combination", "unavailable reason"],
                confidence_rows,
            ), "",
            f"- **Machine-readable method rows:** {report_output_links(outputs.get('confidence_adjustment') or {}, report_path)}", "",
        ])
    lines.extend(["### View Probability Health", ""])
    health_rows = []
    if not passes.empty and "view_probability_processed" in passes.columns:
        health_passes = passes[
            pd.to_numeric(passes["view_probability_processed"], errors="coerce").notna()
        ]
        for (run, scale, logical_iteration), group in health_passes.groupby(
            ["run", "scale_level", "logical_iteration"], dropna=False
        ):
            processed = pd.to_numeric(group["view_probability_processed"], errors="coerce").sum()
            finite = pd.to_numeric(
                group["view_probability_finite_positive_events"], errors="coerce"
            ).sum()
            degenerate = pd.to_numeric(
                group["view_probability_degenerate_events"], errors="coerce"
            ).sum()
            positive_sum = pd.to_numeric(
                group["view_probability_positive_view_count_sum"], errors="coerce"
            ).sum()
            health_rows.append([
                run,
                int(scale),
                instrumentation_report.logical_iteration_label(int(logical_iteration)),
                int(processed),
                fmt_percent(finite / processed if processed else None),
                fmt_percent(degenerate / processed if processed else None),
                int(pd.to_numeric(group["view_probability_zero_mass_events"], errors="coerce").sum()),
                int(pd.to_numeric(group["view_probability_nonfinite_component_events"], errors="coerce").sum()),
                int(pd.to_numeric(group["view_probability_unassigned_draws"], errors="coerce").sum()),
                int(pd.to_numeric(group["view_probability_legacy_last_view_collapse_events"], errors="coerce").sum()),
                fmt(positive_sum / processed if processed else None, 2),
            ])
    if health_rows:
        lines.extend([
            "The observer census classifies raw pre-CDF view-selection mass for each complete logical iteration. These are aggregate event counts; unavailable per-pixel maps are not reconstructed from them.", "",
            md_table(
                ["run", "scale", "state", "processed", "healthy", "degenerate", "zero mass", "nonfinite component", "unassigned draws", "legacy collapse", "mean positive views"],
                health_rows,
            ), "",
            f"- **Machine-readable iteration rows:** {report_output_links(outputs.get('iterations') or {}, report_path)}", "",
        ])
    else:
        requested = (
            frames.get("view_probability_health_requested", pd.Series(dtype=bool))
            .fillna(False).astype(bool).any()
            if not frames.empty else False
        )
        reasons = sorted({
            str(value) for value in frames.get(
                "view_probability_health_unavailable_reason", pd.Series(dtype=str)
            ).dropna() if str(value)
        }) if not frames.empty else []
        lines.extend([
            "_Unavailable: no valid probability-health iteration rows were discovered._",
            "",
            f"Requested by at least one selected frame: `{str(bool(requested)).lower()}`. "
            + ("Reasons: " + "; ".join(reasons) if reasons else "No capture reason was recorded."),
            "",
        ])
    if not passes.empty and "logical_iteration" in passes.columns:
        unavailable_mode = instrumentation_report.CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
        candidate_accounting_unavailable = (
            not frames.empty
            and frames.get("candidate_accounting_mode", pd.Series(dtype=str))
            .eq(unavailable_mode).any()
        )
        pass_rows = []
        for (run, scale, iteration), group in passes.groupby(["run", "scale_level", "logical_iteration"]):
            accounting_available = not group.get(
                "candidate_accounting_mode", pd.Series(dtype=str)
            ).eq(unavailable_mode).any()
            pass_rows.append([
                run, instrumentation_report.logical_iteration_label(int(iteration)), int(scale),
                fmt(group.get("changed_ratio", pd.Series(dtype=float)).mean()) if accounting_available else "unavailable",
                candidate_accounting_count(group, "candidates_tested", available=accounting_available),
                candidate_accounting_count(group, "candidates_finite", available=accounting_available),
                candidate_accounting_count(group, "candidates_accepted", available=accounting_available),
                fmt(group.get("acceptance_rate", pd.Series(dtype=float)).mean()) if accounting_available else "unavailable",
                fmt(group.get("mean_cost", pd.Series(dtype=float)).mean()),
                fmt(group.get("mean_cost_delta", pd.Series(dtype=float)).mean()),
                fmt(group.get("mean_abs_depth_delta", pd.Series(dtype=float)).mean()),
                fmt(group.get("mean_normal_delta_deg", pd.Series(dtype=float)).mean()),
            ])
        lines.extend([md_table(["run", "stage", "scale", "changed", "tested", "finite", "accepted", "candidate acceptance", "mean cost", "cost improvement", "depth delta", "normal delta"], pass_rows), ""])
        if candidate_accounting_unavailable:
            lines.extend([
                "Changed-pixel ratios and candidate tested/finite/accepted/acceptance metrics are intentionally unavailable in robust post-pass snapshot mode; zeros are not inferred. Raw capture artifacts remain linked for provenance. Changed pixels retain a `changed_unknown` source; component maps and confidence gaps are device-side post-pass diagnostics.",
                "",
            ])
    else:
        lines.extend(["_No iteration-level data available._", ""])
    if not performance.empty:
        lines.extend([
            "Runtime gates and Pareto ranking use monotonic full-process wall time from the uninstrumented production endpoint. Repeats are averaged within each scene before scene-balanced relative deltas are computed; summary CUDA-event kernel timings remain diagnostic only.",
            "",
        ])
    lines.extend(["## 8. Per-scene Analysis", ""])
    for scene_id in sorted(set(frames["scene_id"])) if not frames.empty else []:
        scene_frames = frames[frames["scene_id"] == scene_id]
        scene_annotations = annotations[annotations["scene_id"] == scene_id] if not annotations.empty else pd.DataFrame()
        lines.extend([
            '<details class="scene">', f"<summary>{html.escape(scene_id)}</summary>", "",
            "### Scene Scorecard", "",
            md_table(
                ["run", "frames", "estimator validity", "final endpoint validity", "rejected total", "keep-cost rejected", "mask rejected ratio", "cost median", "mean support"],
                [[
                    run, len(group), fmt(group["valid_ratio_after_filter"].mean()),
                    fmt(group["endpoint_valid_depth_coverage"].mean())
                    if "endpoint_valid_depth_coverage" in group else "n/a",
                    fmt(group["rejected_by_filter_ratio"].mean()),
                    fmt(group["rejected_by_keep_cost_filter_ratio"].mean()),
                    aggregate_mask_rejection_display(group),
                    fmt(group["final_cost_median"].mean()),
                    fmt(group["mean_support"].mean(), 2),
                ] for run, group in scene_frames.groupby("run")],
            ), "",
        ])
        lines.extend([
            "### Ignore Mask Availability", "",
            md_table(
                ["run", "frame", "Ignore mask", "requested", "loaded", "Mask-rejected pixels", "unavailable reason"],
                [[
                    row.get("run"), row.get("image_id"), row.get("ignore_mask_status", "legacy_count_available"),
                    row.get("ignore_mask_requested"), row.get("ignore_mask_loaded"),
                    mask_rejection_display(row, ratio=False), row.get("ignore_mask_unavailable_reason") or "-",
                ] for row in scene_frames.to_dict("records")],
            ), "",
        ])
        scene_run_metadata: dict[str, instrumentation_report.RunData] = {}
        for panel in diagnostic_panels.get(scene_id, []):
            for run, _record in panel.records:
                scene_run_metadata[run.label] = run
        if scene_run_metadata:
            instrumentation_rows = []
            for label, run in sorted(scene_run_metadata.items()):
                instrument = run.run_metadata.get("instrumentation") or {}
                capabilities = instrument.get("capabilities") or {}
                effective_level = instrument.get("effective_level", instrument.get("level", "unknown"))
                instrumentation_rows.append([
                    label,
                    "yes" if instrument.get("enabled", True) else "no",
                    effective_level,
                    instrument.get("sample_rate", "n/a"),
                    instrument.get("image_list") or "all",
                    "yes" if capabilities.get("iteration_counters", capabilities.get("pass_counters", True)) else "no",
                    "yes" if capabilities.get("kernel_timings", True) else "no",
                    "yes" if capabilities.get("targeted_traces", effective_level in {"debug", "maps"}) else "no",
                    "yes" if capabilities.get("full_resolution_maps", instrument.get("write_maps", False)) else "no",
                    "yes" if capabilities.get("post_pass_component_rescore", True) else "no",
                ])
            lines.extend([
                "### Instrumentation Configuration", "",
                md_table(
                    ["run", "enabled", "level", "sample rate", "image selection", "counters", "timings", "traces", "full maps", "post-pass rescore"],
                    instrumentation_rows,
                ), "",
            ])
        if (
            not scene_annotations.empty
            and {"run", "stage", "annotation_kind"}.issubset(scene_annotations.columns)
        ):
            scene_annotation_summary_rows = []
            for (run, stage, kind), group in scene_annotations.groupby(["run", "stage", "annotation_kind"], dropna=False):
                fitted = successful_annotations(group)
                repeats = int(group["repeat"].nunique()) if "repeat" in group.columns else 1
                effective_column = "effective_inlier_coverage_20mm" if "effective_inlier_coverage_20mm" in group.columns else "effective_inlier_coverage"
                scene_annotation_summary_rows.append([
                    run, stage, kind, unique_annotation_structures(group), repeats, f"{len(fitted)}/{len(group)}",
                    fmt_percent(annotation_mean(group, "coverage_fraction")),
                    fmt_percent(annotation_mean(group, effective_column)),
                    fmt_percent(annotation_mean(group, "inlier_threshold_auc")),
                    fmt_millimetres(annotation_mean(group, "all_residual_p95_m")),
                ])
            lines.extend([
                "### Annotation Consistency", "",
                md_table(
                    ["run", "stage", "kind", "unique structures", "repeats", "successful / evaluations", "valid depth coverage", "effective @20 mm", "threshold AUC", "residual P95"],
                    scene_annotation_summary_rows,
                ), "",
                '<details class="annotation-detail">',
                f"<summary>Repeat-level annotation metrics ({len(scene_annotations)} evaluations)</summary>",
                "<div>",
                md_table(
                    ["run", "repeat", "frame", "stage", "kind", "chunk", "coverage", "5 mm", "10 mm", "20 mm", "50 mm", "effective @20 mm", "AUC", "P95", "status"],
                    [[
                        row.get("run"), row.get("repeat"), row.get("image_id"), row.get("stage"), row.get("annotation_kind"), str(row.get("chunk_id", ""))[:8],
                        fmt_percent(row.get("coverage_fraction")), fmt_percent(row.get("inlier_fraction_5mm")), fmt_percent(row.get("inlier_fraction_10mm")),
                        fmt_percent(row.get("inlier_fraction_20mm")), fmt_percent(row.get("inlier_fraction_50mm")),
                        fmt_percent(row.get("effective_inlier_coverage_20mm", row.get("effective_inlier_coverage"))),
                        fmt_percent(row.get("inlier_threshold_auc")), fmt_millimetres(row.get("all_residual_p95_m")), row.get("fit_status"),
                    ] for row in scene_annotations.to_dict("records")],
                ),
                "</div>", "</details>", "",
            ])
            post_scene_annotations = scene_annotations[scene_annotations["stage"] == "post_filter"] if "stage" in scene_annotations.columns else scene_annotations
            visual_rows = []
            for row in post_scene_annotations.to_dict("records"):
                overlay_raw = row.get("visual_overlay_svg")
                residual_raw = row.get("visual_residual_histogram_svg")
                overlay = Path(str(overlay_raw)) if overlay_raw and str(overlay_raw) != "nan" else None
                residual = Path(str(residual_raw)) if residual_raw and str(residual_raw) != "nan" else None
                if (overlay is not None and overlay.is_file()) or (residual is not None and residual.is_file()):
                    visual_rows.append((row, overlay, residual))
            if visual_rows:
                lines.extend(['<details class="mechanism">', f"<summary>Annotation Consistency Visuals ({len(visual_rows)} repeat-level evaluations)</summary>", ""])
                for row, overlay, residual in visual_rows:
                    title = f"{row.get('run')} / repeat {row.get('repeat')} / frame {row.get('image_id')} / {row.get('annotation_kind')} / {str(row.get('chunk_id', ''))[:8]}"
                    lines.extend([f"#### {html.escape(title)}", ""])
                    if overlay is not None and overlay.is_file():
                        lines.extend([f"![{title} overlay]({relative_link(overlay, report_path)})", ""])
                    if residual is not None and residual.is_file():
                        lines.extend([f"![{title} residual histogram]({relative_link(residual, report_path)})", ""])
                lines.extend(["</details>", ""])
        scene_stability = stability[stability["scene_id"] == scene_id] if not stability.empty else pd.DataFrame()
        if not scene_stability.empty:
            stability_detail_rows = []
            for row in scene_stability.to_dict("records"):
                kind = row.get("annotation_kind")
                angle = row.get("plane_normal_delta_deg") if kind == "plane" else row.get("line_direction_delta_deg")
                position = row.get("plane_position_delta_m") if kind == "plane" else row.get("line_position_delta_m")
                position_value = safe_float(position)
                extent_value = safe_float(row.get("line_extent_delta_m"))
                stability_detail_rows.append([
                    row.get("candidate"), row.get("image_id"), kind, str(row.get("chunk_id", ""))[:8],
                    "yes" if bool(row.get("large_model_switch")) else "no",
                    "yes" if bool(row.get("near_candidate_regression")) else "no",
                    fmt(angle, 3), fmt(position_value * 1000.0 if position_value is not None else None, 3),
                    fmt(abs(extent_value) * 1000.0 if extent_value is not None else None, 3) if kind == "edge" else "n/a",
                    fmt_millimetres(row.get("baseline_model_on_candidate_all_residual_p95_m")),
                    row.get("baseline_model_on_candidate_status") or "unavailable",
                ])
            lines.extend([
                "### Annotation Model Stability", "",
                '<details class="annotation-detail"><summary>Per-structure fitted-model deltas</summary><div>',
                md_table(["candidate", "frame", "kind", "chunk", "large switch", "near regression", "angular delta deg", "position delta mm", "extent delta mm", "fixed baseline-model P95", "fixed-model status"], stability_detail_rows),
                "</div></details>", "",
            ])
        scene_plot_groups: dict[str, list[tuple[str, Path]]] = {}
        for title, path in instrumentation_plots.get(scene_id, []):
            scene_plot_groups.setdefault(mechanism_for_plot(title), []).append((title, path))
        lines.extend([
            "### Complete Mechanism Dashboards", "",
            "This inventory intentionally retains the established instrumentation report rather than replacing it with only final-state panels.", "",
        ])
        for mechanism in MECHANISM_ORDER:
            scene_mechanism_plots = scene_plot_groups.get(mechanism, [])
            if not scene_mechanism_plots:
                continue
            open_state = " open" if mechanism in {"cost", "view"} else ""
            lines.extend([
                f'<details class="mechanism"{open_state}>',
                f"<summary>{MECHANISM_LABELS[mechanism]} ({len(scene_mechanism_plots)} aggregate/per-frame plots)</summary>", "",
                MECHANISM_DESCRIPTIONS[mechanism], "",
            ])
            for title, path in scene_mechanism_plots:
                lines.extend([f"#### {html.escape(title)}", "", f"![{title}]({relative_link(path, report_path)})", ""])
            lines.extend(["</details>", ""])
        panels_by_frame: dict[str, list[instrumentation_report.DiagnosticPanel]] = {}
        for panel in diagnostic_panels.get(scene_id, []):
            panels_by_frame.setdefault(panel.frame_key, []).append(panel)
        for frame_key, frame_panels in sorted(panels_by_frame.items(), key=lambda item: (int(item[0]) if str(item[0]).isdigit() else 1 << 30, str(item[0]))):
            label = frame_panels[0].records[0][1].summary.get("safe_image_name", frame_key) if frame_panels[0].records else frame_key
            lines.extend(['<details class="frame">', f"<summary>Frame {html.escape(str(frame_key))}: {html.escape(str(label))} diagnostics</summary>", ""])
            frame_mask = scene_frames["image_id"].astype(str) == str(frame_key)
            frame_rows = scene_frames[frame_mask]
            if not frame_rows.empty:
                lines.extend([
                    "#### Frame Metrics", "",
                    md_table(
                        ["run", "valid before", "after keep-cost", "estimator validity", "final endpoint validity", "endpoint status", "rejected total", "keep-cost rejected", "Ignore mask", "Mask-rejected pixels", "cost mean", "cost median", "cost P90", "cost P95", "mean support", "candidate accounting", "gap mode"],
                        [[
                            row.get("run"), fmt(row.get("valid_ratio_before_filter")), fmt(row.get("valid_ratio_after_keep_cost_filter")), fmt(row.get("valid_ratio_after_filter")),
                            fmt(row.get("endpoint_valid_depth_coverage")), row.get("endpoint_valid_depth_coverage_status", "unavailable"),
                            fmt(row.get("rejected_by_filter_ratio")),
                            fmt(row.get("rejected_by_keep_cost_filter_ratio")), row.get("ignore_mask_status", "legacy_count_available"), mask_rejection_display(row, ratio=False),
                            fmt(row.get("final_cost_mean")), fmt(row.get("final_cost_median")), fmt(row.get("final_cost_p90")), fmt(row.get("final_cost_p95")),
                            fmt(row.get("mean_support"), 2), row.get("candidate_accounting_mode", "unavailable"), row.get("confidence_gap_mode", "unavailable"),
                        ] for row in frame_rows.to_dict("records")],
                    ), "",
                ])
            if not passes.empty:
                frame_passes = passes[(passes["scene_id"] == scene_id) & (passes["image_id"].astype(str) == str(frame_key))]
                if not frame_passes.empty:
                    lines.extend([
                        "#### Iteration Metrics", "",
                        md_table(
                            ["run", "stage", "scale", "valid", "changed", "mean cost", "cost improvement", "depth update", "normal update", "tested", "finite", "accepted", "candidate acceptance"],
                            [[
                                row.get("run"), row.get("stage"), int(row.get("scale_level", 0)), fmt(row.get("valid_ratio")),
                                candidate_accounting_value(row, "changed_ratio"),
                                fmt(row.get("mean_cost")), fmt(row.get("mean_cost_delta")), fmt(row.get("mean_abs_depth_delta")), fmt(row.get("mean_normal_delta_deg")),
                                candidate_accounting_value(row, "candidates_tested", integer=True),
                                candidate_accounting_value(row, "candidates_finite", integer=True),
                                candidate_accounting_value(row, "candidates_accepted", integer=True),
                                candidate_accounting_value(row, "acceptance_rate"),
                            ] for row in frame_passes.sort_values(["run", "scale_level", "logical_iteration"]).to_dict("records")],
                        ), "",
                    ])
            frame_panel_groups: dict[str, list[instrumentation_report.DiagnosticPanel]] = {}
            for panel in frame_panels:
                frame_panel_groups.setdefault(panel.mechanism, []).append(panel)
            for mechanism in MECHANISM_ORDER:
                mechanism_panels = frame_panel_groups.get(mechanism, [])
                if not mechanism_panels:
                    continue
                lines.extend([
                    '<details class="mechanism">',
                    f"<summary>{MECHANISM_LABELS[mechanism]} ({len(mechanism_panels)} panels)</summary>", "",
                    MECHANISM_DESCRIPTIONS[mechanism], "",
                ])
                for panel in mechanism_panels:
                    lines.extend([
                        f"**{html.escape(panel.title)}**", "",
                        panel.caption, "" if panel.caption else "",
                        f"![{panel.title}]({relative_link(panel.path, report_path)})", "",
                    ])
                lines.extend(["</details>", ""])
            raw_records: dict[tuple[str, str], tuple[instrumentation_report.RunData, instrumentation_report.DepthMapRecord]] = {}
            for panel in frame_panels:
                for run, record in panel.records:
                    raw_records[(run.label, str(record.directory))] = (run, record)
            if raw_records:
                lines.extend(['<details class="mechanism">', "<summary>Raw Frame Artifacts</summary>", ""])
                for run, record in raw_records.values():
                    links = []
                    for file_name in FRAME_RAW_ARTIFACTS:
                        path = record.directory / file_name
                        if path.is_file():
                            links.append(markdown_evidence_reference(
                                path, file_name, report_path
                            ))
                    for file_name in RUN_RAW_ARTIFACTS:
                        path = run.path / file_name
                        if path.is_file():
                            links.append(markdown_evidence_reference(
                                path, file_name, report_path
                            ))
                    lines.append(f"- **{html.escape(run.label)}:** " + " | ".join(links))
                lines.extend(["", "</details>", ""])
            lines.extend(["</details>", ""])
        lines.extend(["</details>", ""])
    completed_traces = [
        entry for entry in (drilldowns or {}).get("entries") or []
        if entry.get("status") == "complete"
        and entry.get("capture_profile") == "trace"
        and (entry.get("trace_data") or {}).get("available")
    ]
    if completed_traces:
        lines.extend([
            "### Completed Targeted Pixel Traces", "",
            "These selected trace rows are synthesized from immutable `traces.jsonl` captures. New public-v1 trace requests run full-frame Process<true> maps with `write_maps=1`; there is no compact exact-trace kernel path. Source attribution is classified per row: `exact` requires exact-hot-kernel provenance plus valid schema-v4 exact-map completion evidence, while imported legacy/post-pass rows remain `proxy`. Treat every rerun as diagnostic evidence unless its separate parity and resource gates pass.", "",
        ])
        for entry in completed_traces:
            trace = entry["trace_data"]
            request_id = str(entry.get("request_sha256") or "unknown")
            lines.extend([
                '<details class="mechanism" open>',
                f"<summary>{html.escape(str(entry.get('scene_id')))} / image {entry.get('image_id')} / {trace.get('row_count')} logical-state rows / request {html.escape(request_id[:12])}</summary>", "",
            ])
            trace_rows = []
            for row in trace.get("rows") or []:
                cost = row.get("cost") or {}
                depth = row.get("depth") or {}
                normal = row.get("normal") or {}
                view = row.get("view") or {}
                logical_iteration = int(row.get("logical_iteration", -1))
                state_label = "initialization" if logical_iteration < 0 else f"iteration {logical_iteration + 1}"
                before_mask = view.get("selected_before_mask")
                after_mask = view.get("selected_mask")
                mask_label = (
                    f"{view.get('selected_count', 'n/a')} / "
                    f"{hex(int(before_mask)) if before_mask is not None else 'n/a'} -> "
                    f"{hex(int(after_mask)) if after_mask is not None else 'n/a'}"
                )
                source_quality = str(row.get("source_quality") or "proxy")
                measurement_basis = str(
                    row.get("measurement_basis") or "legacy_unclassified_trace"
                )
                request_identity = row.get("request_identity") or {}
                alias_coordinates = request_identity.get("alias_coordinates") or []
                requested_pixel = ", ".join(
                    f"({coordinate.get('x')}, {coordinate.get('y')})"
                    for coordinate in alias_coordinates
                    if isinstance(coordinate, dict)
                )
                if not requested_pixel and request_identity.get("x") is not None:
                    requested_pixel = (
                        f"({request_identity.get('x')}, {request_identity.get('y')})"
                    )
                if not requested_pixel:
                    requested_pixel = "unavailable"
                estimation_stage = str(row.get("estimation_stage") or "photometric")
                geometric_iteration = row.get("geometric_iteration")
                stage_label = (
                    "photometric"
                    if estimation_stage == "photometric"
                    else f"geometric {geometric_iteration}"
                )
                trace_rows.append([
                    row.get("run"), stage_label, row.get("pyramid_level"),
                    requested_pixel, f"({row.get('x')}, {row.get('y')})", state_label,
                    f"{row.get('source', 'unknown')} ({source_quality}; {measurement_basis})",
                    f"{fmt(cost.get('before'), 6)} -> {fmt(cost.get('after'), 6)}",
                    fmt(cost.get("improvement"), 6),
                    f"{fmt(depth.get('before'), 6)} -> {fmt(depth.get('after'), 6)}",
                    fmt(depth.get("absolute_change"), 6),
                    fmt(normal.get("angle_change_degrees"), 4), mask_label,
                ])
            lines.extend([
                md_table(
                    ["run", "stage", "pyramid level", "requested pixel(s)", "trace pixel", "state", "update source / quality", "cost before -> after", "improvement", "depth before -> after m", "absolute depth change m", "normal change deg", "selected views / masks"],
                    trace_rows,
                ), "",
            ])
            source_links = [
                markdown_evidence_reference(
                    source.get("source_path"),
                    (
                        f"{source.get('run')} "
                        f"{source.get('estimation_stage', 'photometric')}"
                        f"{'' if source.get('geometric_iteration') is None else ' ' + str(source.get('geometric_iteration'))} "
                        "traces.jsonl"
                    ),
                    report_path,
                )
                for source in trace.get("sources") or []
                if source.get("available") and source.get("source_path")
            ]
            if entry.get("executions"):
                source_links.append(markdown_evidence_reference(
                    entry["executions"], "execution manifest", report_path
                ))
            if source_links:
                lines.extend(["Raw evidence: " + " | ".join(source_links), ""])
            lines.extend(["</details>", ""])

    lines.extend([
        "## 9. Failure Analysis", "",
        "All missing scenes, frames, maps, annotation mappings, and failed RANSAC fits remain represented in the machine-readable tables. Inspect failed gates together with per-frame transition maps; internal deltas support mechanism hypotheses but do not establish causality.",
        "", "## 10. Reproducibility", "",
        "```bash",
        "# Capture any missing maps/timing modes (completed valid modes are skipped).",
        f"tools/dmap_observability.sh capture --config {shlex.quote(str(config['_config_path']))}",
        "",
        "# Reproduce this report under a fresh immutable directory with the same effective policy.",
        f"{shlex.quote(sys.executable)} scripts/python/dmap_dev.py report --config {shlex.quote(str(config['_config_path']))} --output-dir {shlex.quote(str(reproduced_report_dir))}{report_policy_flag}",
        "",
        "# Validate Markdown, links, inventory, resources, and the structured report model.",
        f"tools/dmap_observability.sh validate --config {shlex.quote(str(config['_config_path']))} --report-dir {shlex.quote(str(published_report_dir))}{validate_context_flag}",
        "",
        "# Execute the deterministic evidence-pixel drill-down when variants are present.",
        f"{shlex.quote(sys.executable)} scripts/python/dmap_dev.py trace-rerun --config {shlex.quote(str(config['_config_path']))} --execute",
        "```", "",
        "Machine-readable outputs:", "",
    ])
    for name, value in outputs.items():
        if isinstance(value, dict):
            target = value.get("parquet") or value.get("csv")
        else:
            target = value
        if target:
            lines.append(
                f"- `{name}`: "
                + markdown_evidence_reference(
                    target, Path(str(target)).name, report_path
                )
            )
    reproducibility_rows = []
    if not reproducibility_artifacts.empty:
        for row in reproducibility_artifacts[
            reproducibility_artifacts["exists"].astype(bool)
        ].to_dict("records"):
            source = Path(str(row["source_path"]))
            reproducibility_rows.append([
                html.escape(str(row.get("run") or "experiment")), html.escape(str(row.get("mode"))),
                html.escape(str(row.get("scene_id") or "-")), html.escape(str(row.get("frame") or "-")),
                html.escape(str(row.get("kind"))),
                html_evidence_reference(source, source.name, report_path),
            ])
    if reproducibility_rows:
        # This table intentionally contains links, so build it directly instead
        # of escaping the anchor text through md_table.
        lines.extend(["", "### Run and Source Artifact Index", ""])
        lines.append('<div style="overflow-x:auto"><table><thead><tr><th>run</th><th>mode</th><th>scene</th><th>frame</th><th>artifact</th><th>source</th></tr></thead><tbody>')
        for row in reproducibility_rows:
            lines.append("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>")
        lines.extend(["</tbody></table></div>", ""])
    lines.extend([
        "", "## 11. Recommendations", "",
        "- Treat failed quality gates as regression candidates and inspect their ranked scene/frame panels before changing additional parameters.",
        "- Calibrate practical tolerances from three baseline repeats on the development tier before using full-suite verdicts as merge gates.",
        "- Require a reproducible, build-bound CUDA qualification gate for each supported toolchain: sanitizer checks, declared observer resource budgets, production/observer parity, multiple worker counts, and representative development scenes.",
        "- Use the generated trace manifest for representative wins, regressions, newly invalid pixels, and stable controls. Keep direct `Process<true>` cohorts isolated as mechanics-only evidence wherever they fail parity or resource gates; do not let them enter quality rankings or promotion decisions.",
        "- Add texture/annotation-region stratification to the machine-readable frame table so completeness, ambiguity, and residual changes can be attributed to low-texture, plane, line, and boundary regions.",
        "- Package the lossless PFM/PNG bundle into one indexed compressed array artifact after the signal schema stabilizes; retain the manifest as the public contract.", "",
    ])
    return "\n".join(lines)


def read_u8_map(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        from PIL import Image

        data = np.asarray(Image.open(path))
    except Exception:
        return None
    return data[..., 0] if data.ndim == 3 else data


def trace_frame_maps(frame: pd.Series) -> dict[str, np.ndarray | None]:
    depthmap_dir = Path(str(frame["depthmap_dir"]))
    maps_dir = depthmap_dir / "maps"
    depth_path = maps_dir / "depth_final_after_filter.pfm"
    try:
        depth = read_pfm(depth_path) if depth_path.is_file() else None
    except Exception:
        depth = None
    valid_u8 = read_u8_map(maps_dir / "valid_after_filter.png")
    valid = valid_u8 > 0 if valid_u8 is not None else (np.isfinite(depth) & (depth > 0.0) if depth is not None else None)
    gap_path = maps_dir / "confidence_gap.pfm"
    try:
        gap = read_pfm(gap_path) if gap_path.is_file() else None
    except Exception:
        gap = None
    return {
        "depth": depth,
        "valid": valid,
        "gap": gap,
        "updates": read_u8_map(maps_dir / "accepted_update_count.png"),
    }


TRACE_SELECTION_BORDER_PIXELS = 6
TRACE_EXACT_CATEGORIES = (
    "exact_total_cost_delta",
    "exact_view_mask_xor",
    "exact_gap_collapse",
    "exact_gap_expansion",
    "lost_final_validity",
    "gained_final_validity",
    "relative_final_depth_delta",
    "stable_same_view_mask_control",
)


def select_trace_pixel(
    mask: np.ndarray,
    score: np.ndarray,
    largest: bool,
    used: set[tuple[int, int]],
    border: int = TRACE_SELECTION_BORDER_PIXELS,
) -> tuple[int, int] | None:
    eligible = np.asarray(mask, dtype=bool) & np.isfinite(score)
    if eligible.ndim != 2 or not eligible.any():
        return None
    if border < 0:
        raise ValueError("trace-pixel border must be non-negative")
    if border:
        if eligible.shape[0] <= 2 * border or eligible.shape[1] <= 2 * border:
            return None
        eligible[:border, :] = False
        eligible[-border:, :] = False
        eligible[:, :border] = False
        eligible[:, -border:] = False
    ys, xs = np.nonzero(eligible)
    values = score[ys, xs]
    # np.lexsort uses the last key as primary. Ties are intentionally resolved
    # in raster order so recommendations are stable across NumPy versions.
    primary = -values if largest else values
    order = np.lexsort((xs, ys, primary))
    for index in order:
        point = (int(xs[index]), int(ys[index]))
        if point not in used:
            used.add(point)
            return point
    return None


def trace_optional_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    integer = int(number)
    return integer if number == integer else None


def terminal_exact_trace_capture(frame: pd.Series) -> dict[str, Any]:
    """Load terminal schema-v4 maps without conflating estimation stages."""

    depthmap_dir = Path(str(frame.get("depthmap_dir", "")))
    manifest_path = depthmap_dir / "map_manifest.json"
    manifest = read_json(manifest_path)
    schema_version = trace_optional_int(manifest.get("schema_version")) or 0
    result: dict[str, Any] = {
        "depthmap_dir": str(depthmap_dir),
        "manifest_path": str(manifest_path),
        "schema_version": schema_version,
        "estimation_stage": str(manifest.get("estimation_stage", "")),
        "geometric_iteration": trace_optional_int(manifest.get("geometric_iteration")),
        "terminal_logical_iteration": None,
        "maps": {},
        "signal_errors": {},
        "view_mapping": {},
        "view_mapping_error": None,
        "capture_error": None,
    }
    if not manifest:
        result["capture_error"] = "map_manifest.json is missing or invalid"
        return result
    if manifest.get("schema_name") != "openmvs.dmap.map_manifest":
        result["capture_error"] = "map manifest schema_name is unsupported"
        return result
    if schema_version != 4:
        result["capture_error"] = (
            f"map manifest schema_version={schema_version} is unsupported; expected exactly 4"
        )
        return result
    manifest_width = trace_optional_int(manifest.get("width"))
    manifest_height = trace_optional_int(manifest.get("height"))
    frame_width = trace_optional_int(frame.get("width"))
    frame_height = trace_optional_int(frame.get("height"))
    if (
        manifest_width is None or manifest_width <= 0
        or manifest_height is None or manifest_height <= 0
    ):
        result["capture_error"] = "map manifest width/height is missing or non-positive"
        return result
    if frame_width != manifest_width or frame_height != manifest_height:
        result["capture_error"] = (
            "frame/manifest dimensions differ: "
            f"frame={frame_width}x{frame_height}, "
            f"manifest={manifest_width}x{manifest_height}"
        )
        return result
    result["width"] = manifest_width
    result["height"] = manifest_height
    maps_value = manifest.get("maps")
    if not isinstance(maps_value, list):
        result["capture_error"] = "map manifest maps must be a list"
        return result
    if manifest.get("complete") is not True:
        result["capture_error"] = "map manifest is not marked complete"
        return result
    write_errors = manifest.get("write_errors")
    if not isinstance(write_errors, list) or write_errors:
        result["capture_error"] = "map manifest write_errors is missing, invalid, or non-empty"
        return result
    expected_map_count = trace_optional_int(manifest.get("expected_map_count"))
    written_map_count = trace_optional_int(manifest.get("written_map_count"))
    if (
        expected_map_count != len(maps_value)
        or written_map_count != len(maps_value)
    ):
        result["capture_error"] = (
            "map manifest counters disagree with maps: "
            f"expected={expected_map_count}, written={written_map_count}, actual={len(maps_value)}"
        )
        return result
    exact_capture = manifest.get("exact_capture") or {}
    if exact_capture.get("available") is not True:
        result["capture_error"] = str(
            exact_capture.get("unavailable_reason") or "schema-v4 exact capture is unavailable"
        )
        return result

    artifacts = instrumentation_report.parse_map_artifacts(depthmap_dir, manifest)
    num_states = trace_optional_int(manifest.get("num_logical_states"))
    num_iterations = trace_optional_int(manifest.get("num_iterations"))
    if (
        num_states is None or num_states <= 0
        or num_iterations is None or num_iterations < 0
        or num_iterations != num_states - 1
    ):
        result["capture_error"] = (
            "schema-v4 logical counters are invalid or inconsistent: "
            f"num_iterations={num_iterations}, num_logical_states={num_states}"
        )
        return result
    terminal_iteration = num_iterations - 1
    result["terminal_logical_iteration"] = terminal_iteration
    if terminal_iteration is None:
        result["capture_error"] = "terminal logical iteration cannot be determined"
        return result

    def contained_artifact_path(path: Path) -> bool:
        try:
            path.resolve().relative_to(depthmap_dir.resolve())
        except ValueError:
            return False
        return True

    signal_specs = (
        ("cost_total_production_exact", terminal_iteration, {"exact"}),
        ("gap_winner_runner_up_exact", terminal_iteration, {"exact"}),
        ("selected_views_after_mask_exact", terminal_iteration, {"exact"}),
        ("depth_final_after_filter", None, {"exact"}),
        ("valid_after_filter", None, {"derived_exact"}),
    )
    for signal, logical_iteration, allowed_qualities in signal_specs:
        candidates = [
            artifact for artifact in artifacts
            if artifact.signal == signal
            and artifact.logical_iteration == logical_iteration
        ]
        if len(candidates) != 1:
            result["signal_errors"][signal] = (
                f"expected one terminal artifact, found {len(candidates)}"
            )
            continue
        artifact = candidates[0]
        if artifact.measurement_quality not in allowed_qualities:
            result["signal_errors"][signal] = (
                f"artifact quality is {artifact.measurement_quality!r}, expected one of "
                f"{sorted(allowed_qualities)}"
            )
            continue
        if not contained_artifact_path(artifact.path):
            result["signal_errors"][signal] = "terminal artifact path escapes depth-map directory"
            continue
        if not artifact.path.is_file():
            result["signal_errors"][signal] = "terminal artifact file is missing"
            continue
        try:
            values = dmap_report_model.read_diagnostic_map(artifact.path)
        except Exception as exc:
            result["signal_errors"][signal] = f"terminal artifact cannot be decoded: {exc}"
            continue
        if values.shape[:2] != (manifest_height, manifest_width):
            result["signal_errors"][signal] = (
                "artifact dimensions disagree with frame/manifest: "
                f"decoded={values.shape[1]}x{values.shape[0]}, "
                f"expected={manifest_width}x{manifest_height}"
            )
            continue
        if signal == "selected_views_after_mask_exact":
            if values.ndim != 3 or values.shape[2] != 4:
                result["signal_errors"][signal] = "terminal view mask is not uint8x4 RGBA"
                continue
            if artifact.encoding != "uint32 little-endian bytes in RGBA channels":
                result["signal_errors"][signal] = "terminal view-mask encoding is not canonical"
                continue
        elif values.ndim != 2:
            result["signal_errors"][signal] = "terminal scalar artifact is not two-dimensional"
            continue
        result["maps"][signal] = values

    view_artifacts = [
        artifact for artifact in artifacts
        if artifact.signal in SCHEMA4_EXACT_VIEW_SIGNALS
        and artifact.logical_iteration == terminal_iteration
    ]
    mapping_values: dict[int, set[tuple[int, str]]] = {}
    for artifact in view_artifacts:
        if artifact.measurement_quality != "exact":
            result["view_mapping_error"] = (
                "terminal source-view mapping is backed by a non-exact view artifact"
            )
            break
        if artifact.source_view_index is None or artifact.source_image_id is None:
            result["view_mapping_error"] = "terminal exact view metadata omits an index or image id"
            break
        mapping_values.setdefault(artifact.source_view_index, set()).add(
            (artifact.source_image_id, artifact.source_image_name)
        )
    if result["view_mapping_error"] is None:
        conflicting = {
            index: sorted(values)
            for index, values in mapping_values.items()
            if len({image_id for image_id, _name in values}) != 1
        }
        expected_views = trace_optional_int(exact_capture.get("num_views"))
        expected_indices = set(range(expected_views or 0))
        if conflicting:
            result["view_mapping_error"] = f"conflicting source image ids: {conflicting}"
        elif expected_views is None or expected_views <= 0:
            result["view_mapping_error"] = "exact_capture.num_views is missing or non-positive"
        elif set(mapping_values) != expected_indices:
            result["view_mapping_error"] = (
                f"source-view indices {sorted(mapping_values)} do not cover {sorted(expected_indices)}"
            )
        else:
            result["view_mapping"] = {
                index: {
                    "source_image_id": next(iter({item[0] for item in values})),
                    "source_image_names": sorted({item[1] for item in values if item[1]}),
                }
                for index, values in sorted(mapping_values.items())
            }
    return result


def decode_trace_view_mask(values: np.ndarray) -> np.ndarray:
    channels = np.asarray(values, dtype=np.uint32)
    return (
        channels[..., 0]
        | (channels[..., 1] << 8)
        | (channels[..., 2] << 16)
        | (channels[..., 3] << 24)
    )


def trace_mask_popcount(values: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(values, dtype=np.uint32)
    counts = np.zeros(unsigned.shape, dtype=np.uint8)
    for bit in range(32):
        counts += ((unsigned >> bit) & 1).astype(np.uint8)
    return counts


def exact_trace_pair_prerequisite(
    baseline: pd.Series,
    candidate: pd.Series,
    baseline_capture: dict[str, Any],
    candidate_capture: dict[str, Any],
) -> str | None:
    baseline_stage = str(baseline.get("estimation_stage", ""))
    candidate_stage = str(candidate.get("estimation_stage", ""))
    if baseline_stage != "geometric_consistency" or candidate_stage != "geometric_consistency":
        return (
            "exact recommendations require paired terminal geometric_consistency stages; "
            f"got baseline={baseline_stage!r}, candidate={candidate_stage!r}"
        )
    baseline_geometric = trace_optional_int(baseline.get("geometric_iteration"))
    candidate_geometric = trace_optional_int(candidate.get("geometric_iteration"))
    if baseline_geometric is None or candidate_geometric is None:
        return "terminal geometric iteration identity is missing"
    if baseline_geometric != candidate_geometric:
        return (
            "terminal geometric iterations differ: "
            f"baseline={baseline_geometric}, candidate={candidate_geometric}"
        )
    for label, stage, geometric_iteration, capture in (
        ("baseline", baseline_stage, baseline_geometric, baseline_capture),
        ("candidate", candidate_stage, candidate_geometric, candidate_capture),
    ):
        if capture["estimation_stage"] != stage:
            return (
                f"{label} frame/manifest estimation stage mismatch: "
                f"frame={stage!r}, manifest={capture['estimation_stage']!r}"
            )
        if capture["geometric_iteration"] != geometric_iteration:
            return (
                f"{label} frame/manifest geometric iteration mismatch: "
                f"frame={geometric_iteration}, manifest={capture['geometric_iteration']}"
            )
    for label, capture in (("baseline", baseline_capture), ("candidate", candidate_capture)):
        if capture["capture_error"]:
            return f"{label} exact capture unavailable: {capture['capture_error']}"
    return None


def trace_selection_record(
    category: str,
    *,
    status: str,
    measurement_quality: str,
    source_signals: list[str],
    scoring: str,
    reason: str | None = None,
    point: tuple[int, int] | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "category": category,
        "status": status,
        "measurement_quality": measurement_quality,
        "source_signals": source_signals,
        "scoring": scoring,
    }
    if reason:
        record["reason"] = reason
    if point is not None:
        record.update({"x": point[0], "y": point[1]})
    if score is not None and math.isfinite(score):
        record["score"] = float(score)
    return record


def exact_evidence_trace_recommendations(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, Any]:
    baseline_capture = terminal_exact_trace_capture(baseline)
    candidate_capture = terminal_exact_trace_capture(candidate)
    prerequisite = exact_trace_pair_prerequisite(
        baseline, candidate, baseline_capture, candidate_capture
    )
    context = {
        "baseline_manifest_schema_version": baseline_capture["schema_version"],
        "candidate_manifest_schema_version": candidate_capture["schema_version"],
        "baseline_manifest_estimation_stage": baseline_capture["estimation_stage"],
        "candidate_manifest_estimation_stage": candidate_capture["estimation_stage"],
        "baseline_manifest_geometric_iteration": baseline_capture["geometric_iteration"],
        "candidate_manifest_geometric_iteration": candidate_capture["geometric_iteration"],
        "baseline_terminal_logical_iteration": baseline_capture["terminal_logical_iteration"],
        "candidate_terminal_logical_iteration": candidate_capture["terminal_logical_iteration"],
        "baseline_source_view_mapping": baseline_capture["view_mapping"],
        "candidate_source_view_mapping": candidate_capture["view_mapping"],
    }
    if prerequisite:
        unavailable = [
            trace_selection_record(
                category,
                status="unavailable",
                measurement_quality="unavailable",
                source_signals=[],
                scoring="unavailable",
                reason=prerequisite,
            )
            for category in TRACE_EXACT_CATEGORIES
        ]
        return {
            "mode": "legacy_fallback",
            "fallback_reason": prerequisite,
            "selected": [],
            "availability": unavailable,
            "context": context,
        }

    used: set[tuple[int, int]] = set()
    selected: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []

    def add(
        category: str,
        mask: np.ndarray | None,
        score: np.ndarray | None,
        *,
        largest: bool,
        measurement_quality: str,
        source_signals: list[str],
        scoring: str,
        unavailable_reason: str | None = None,
    ) -> None:
        if unavailable_reason or mask is None or score is None:
            availability.append(trace_selection_record(
                category,
                status="unavailable",
                measurement_quality=measurement_quality,
                source_signals=source_signals,
                scoring=scoring,
                reason=unavailable_reason or "required paired maps are unavailable",
            ))
            return
        if mask.ndim != 2 or score.ndim != 2 or mask.shape != score.shape:
            availability.append(trace_selection_record(
                category,
                status="unavailable",
                measurement_quality=measurement_quality,
                source_signals=source_signals,
                scoring=scoring,
                reason="paired recommendation mask and score shapes differ",
            ))
            return
        point = select_trace_pixel(mask.copy(), score, largest, used)
        if point is None:
            availability.append(trace_selection_record(
                category,
                status="unavailable",
                measurement_quality=measurement_quality,
                source_signals=source_signals,
                scoring=scoring,
                reason=(
                    "no unused eligible pixel remains inside the six-pixel border"
                ),
            ))
            return
        value = float(score[point[1], point[0]])
        record = trace_selection_record(
            category,
            status="selected",
            measurement_quality=measurement_quality,
            source_signals=source_signals,
            scoring=scoring,
            point=point,
            score=value,
        )
        availability.append(record)
        selected.append({
            "x": point[0], "y": point[1], "selection": category,
            "measurement_quality": measurement_quality,
            "source_signals": source_signals,
            "score": value,
            "scoring": scoring,
        })

    def paired_exact(signal: str) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
        baseline_value = baseline_capture["maps"].get(signal)
        candidate_value = candidate_capture["maps"].get(signal)
        errors = []
        if baseline_value is None:
            errors.append(
                "baseline: " + baseline_capture["signal_errors"].get(signal, "artifact unavailable")
            )
        if candidate_value is None:
            errors.append(
                "candidate: " + candidate_capture["signal_errors"].get(signal, "artifact unavailable")
            )
        if errors:
            return None, None, "; ".join(errors)
        if baseline_value.shape != candidate_value.shape:
            return None, None, (
                f"paired {signal} shapes differ: {baseline_value.shape} vs {candidate_value.shape}"
            )
        return baseline_value, candidate_value, None

    baseline_cost, candidate_cost, cost_error = paired_exact(
        "cost_total_production_exact"
    )
    if cost_error is None:
        cost_delta = np.abs(candidate_cost - baseline_cost)
        cost_mask = np.isfinite(baseline_cost) & np.isfinite(candidate_cost) & (cost_delta > 0.0)
    else:
        cost_delta = cost_mask = None
    add(
        "exact_total_cost_delta", cost_mask, cost_delta, largest=True,
        measurement_quality="derived_exact",
        source_signals=["cost_total_production_exact"],
        scoring="largest absolute candidate-minus-baseline terminal exact total-cost delta",
        unavailable_reason=cost_error,
    )

    baseline_mask_rgba, candidate_mask_rgba, mask_error = paired_exact(
        "selected_views_after_mask_exact"
    )
    mapping_error = baseline_capture["view_mapping_error"] or candidate_capture["view_mapping_error"]
    baseline_mapping_ids = {
        index: value["source_image_id"]
        for index, value in baseline_capture["view_mapping"].items()
    }
    candidate_mapping_ids = {
        index: value["source_image_id"]
        for index, value in candidate_capture["view_mapping"].items()
    }
    if mapping_error is None and baseline_mapping_ids != candidate_mapping_ids:
        mapping_error = (
            "source-view index-to-image mapping differs: "
            f"baseline={baseline_mapping_ids}, candidate={candidate_mapping_ids}"
        )
    if mask_error is None and mapping_error is None:
        baseline_view_mask = decode_trace_view_mask(baseline_mask_rgba)
        candidate_view_mask = decode_trace_view_mask(candidate_mask_rgba)
        mask_xor = np.bitwise_xor(baseline_view_mask, candidate_view_mask)
        mask_churn = trace_mask_popcount(mask_xor).astype(float)
        changed_mask = mask_xor != 0
    else:
        baseline_view_mask = candidate_view_mask = mask_churn = changed_mask = None
    add(
        "exact_view_mask_xor", changed_mask, mask_churn, largest=True,
        measurement_quality="derived_exact",
        source_signals=["selected_views_after_mask_exact"],
        scoring="largest exact selected-source-view symmetric-difference popcount",
        unavailable_reason=mask_error or mapping_error,
    )

    baseline_gap, candidate_gap, gap_error = paired_exact("gap_winner_runner_up_exact")
    if gap_error is None:
        comparable_gap = (
            np.isfinite(baseline_gap) & np.isfinite(candidate_gap)
            & (baseline_gap >= 0.0) & (candidate_gap >= 0.0)
        )
        gap_collapse = baseline_gap - candidate_gap
        gap_expansion = candidate_gap - baseline_gap
    else:
        comparable_gap = gap_collapse = gap_expansion = None
    add(
        "exact_gap_collapse",
        comparable_gap & (gap_collapse > 0.0) if comparable_gap is not None else None,
        gap_collapse,
        largest=True,
        measurement_quality="derived_exact",
        source_signals=["gap_winner_runner_up_exact"],
        scoring="largest decrease in exact winner-versus-runner-up gap",
        unavailable_reason=gap_error,
    )
    add(
        "exact_gap_expansion",
        comparable_gap & (gap_expansion > 0.0) if comparable_gap is not None else None,
        gap_expansion,
        largest=True,
        measurement_quality="derived_exact",
        source_signals=["gap_winner_runner_up_exact"],
        scoring="largest increase in exact winner-versus-runner-up gap",
        unavailable_reason=gap_error,
    )

    baseline_valid = baseline_capture["maps"].get("valid_after_filter")
    candidate_valid = candidate_capture["maps"].get("valid_after_filter")
    valid_error = None
    if baseline_valid is None or candidate_valid is None:
        errors = []
        if baseline_valid is None:
            errors.append(
                "baseline: " + baseline_capture["signal_errors"].get(
                    "valid_after_filter", "final validity artifact unavailable"
                )
            )
        if candidate_valid is None:
            errors.append(
                "candidate: " + candidate_capture["signal_errors"].get(
                    "valid_after_filter", "final validity artifact unavailable"
                )
            )
        valid_error = "; ".join(errors)
    elif baseline_valid.shape != candidate_valid.shape:
        valid_error = (
            f"paired final validity shapes differ: {baseline_valid.shape} vs {candidate_valid.shape}"
        )
    if valid_error is None:
        baseline_valid = np.asarray(baseline_valid) > 0
        candidate_valid = np.asarray(candidate_valid) > 0
        height, width = baseline_valid.shape
        yy, xx = np.indices((height, width))
        center_distance = (xx - (width - 1) / 2.0) ** 2 + (yy - (height - 1) / 2.0) ** 2
    else:
        center_distance = None
    add(
        "lost_final_validity",
        baseline_valid & ~candidate_valid if valid_error is None else None,
        center_distance,
        largest=False,
        measurement_quality="derived_exact",
        source_signals=["valid_after_filter"],
        scoring="closest-to-center pixel that is valid only in the baseline",
        unavailable_reason=valid_error,
    )
    add(
        "gained_final_validity",
        ~baseline_valid & candidate_valid if valid_error is None else None,
        center_distance,
        largest=False,
        measurement_quality="derived_exact",
        source_signals=["valid_after_filter"],
        scoring="closest-to-center pixel that is valid only in the candidate",
        unavailable_reason=valid_error,
    )

    baseline_depth = baseline_capture["maps"].get("depth_final_after_filter")
    candidate_depth = candidate_capture["maps"].get("depth_final_after_filter")
    depth_error = valid_error
    if depth_error is None and (baseline_depth is None or candidate_depth is None):
        errors = []
        if baseline_depth is None:
            errors.append(
                "baseline: " + baseline_capture["signal_errors"].get(
                    "depth_final_after_filter", "final depth artifact unavailable"
                )
            )
        if candidate_depth is None:
            errors.append(
                "candidate: " + candidate_capture["signal_errors"].get(
                    "depth_final_after_filter", "final depth artifact unavailable"
                )
            )
        depth_error = "; ".join(errors)
    elif depth_error is None and (
        baseline_depth.shape != candidate_depth.shape
        or baseline_depth.shape != baseline_valid.shape
    ):
        depth_error = "paired final depth and validity map shapes differ"
    if depth_error is None:
        common_valid = (
            baseline_valid & candidate_valid
            & np.isfinite(baseline_depth) & np.isfinite(candidate_depth)
            & (baseline_depth > 0.0) & (candidate_depth > 0.0)
        )
        relative_depth = np.abs(candidate_depth - baseline_depth) / np.maximum(
            np.abs(baseline_depth), 1e-6
        )
    else:
        common_valid = relative_depth = None
    add(
        "relative_final_depth_delta", common_valid, relative_depth, largest=True,
        measurement_quality="derived_exact",
        source_signals=["depth_final_after_filter", "valid_after_filter"],
        scoring="largest absolute final-depth delta relative to baseline depth",
        unavailable_reason=depth_error,
    )

    stable_error = depth_error or mask_error or mapping_error
    if (
        stable_error is None
        and baseline_view_mask.shape != common_valid.shape
    ):
        stable_error = (
            "paired exact view-mask and final depth/validity shapes differ: "
            f"{baseline_view_mask.shape} vs {common_valid.shape}"
        )
    stable_mask = (
        common_valid & (baseline_view_mask == candidate_view_mask)
        if stable_error is None else None
    )
    add(
        "stable_same_view_mask_control", stable_mask, relative_depth, largest=False,
        measurement_quality="derived_exact",
        source_signals=[
            "selected_views_after_mask_exact", "depth_final_after_filter", "valid_after_filter",
        ],
        scoring="smallest relative final-depth delta with an identical exact source-view mask",
        unavailable_reason=stable_error,
    )
    return {
        "mode": "schema4_terminal_geometric_exact",
        "fallback_reason": None,
        "selected": selected,
        "availability": availability,
        "context": context,
    }


def legacy_evidence_trace_recommendations(
    baseline: pd.Series,
    candidate: pd.Series,
    fallback_reason: str,
) -> dict[str, Any]:
    baseline_maps = trace_frame_maps(baseline)
    candidate_maps = trace_frame_maps(candidate)
    width = int(candidate.get("width", 0))
    height = int(candidate.get("height", 0))
    used: set[tuple[int, int]] = set()
    selected: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []

    def add(
        label: str,
        mask: np.ndarray | None,
        score: np.ndarray | None,
        *,
        largest: bool = True,
        measurement_quality: str,
        source_signals: list[str],
        scoring: str,
    ) -> None:
        point = (
            select_trace_pixel(mask.copy(), score, largest, used)
            if mask is not None and score is not None and mask.shape == score.shape
            else None
        )
        if point is not None:
            value = float(score[point[1], point[0]])
            record = trace_selection_record(
                label,
                status="selected",
                measurement_quality=measurement_quality,
                source_signals=source_signals,
                scoring=scoring,
                point=point,
                score=value,
            )
            availability.append(record)
            selected.append({
                "x": point[0], "y": point[1], "selection": label,
                "measurement_quality": measurement_quality,
                "source_signals": source_signals,
                "score": value,
                "scoring": scoring,
            })
        else:
            availability.append(trace_selection_record(
                label,
                status="unavailable",
                measurement_quality=measurement_quality,
                source_signals=source_signals,
                scoring=scoring,
                reason="no unused eligible legacy pixel remains inside the six-pixel border",
            ))

    baseline_valid = baseline_maps["valid"]
    candidate_valid = candidate_maps["valid"]
    if baseline_valid is not None and candidate_valid is not None and baseline_valid.shape == candidate_valid.shape:
        height, width = baseline_valid.shape
        yy, xx = np.indices((height, width))
        center_score = -((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2)
        add(
            "newly_invalid", baseline_valid & ~candidate_valid, center_score,
            measurement_quality="derived_exact", source_signals=["valid_after_filter"],
            scoring="closest-to-center legacy final-validity loss",
        )
        add(
            "newly_valid", ~baseline_valid & candidate_valid, center_score,
            measurement_quality="derived_exact", source_signals=["valid_after_filter"],
            scoring="closest-to-center legacy final-validity gain",
        )
        common = baseline_valid & candidate_valid
        baseline_depth = baseline_maps["depth"]
        candidate_depth = candidate_maps["depth"]
        if baseline_depth is not None and candidate_depth is not None and baseline_depth.shape == candidate_depth.shape:
            relative = np.abs(candidate_depth - baseline_depth) / np.maximum(np.abs(baseline_depth), 1e-6)
            add(
                "maximum_relative_depth_disagreement", common, relative,
                measurement_quality="derived_exact",
                source_signals=["depth_final_after_filter", "valid_after_filter"],
                scoring="largest legacy relative final-depth disagreement",
            )
            stable_score = relative + np.sqrt((xx - width / 2.0) ** 2 + (yy - height / 2.0) ** 2) / max(width, height)
            add(
                "stable_control", common, stable_score, largest=False,
                measurement_quality="derived_exact",
                source_signals=["depth_final_after_filter", "valid_after_filter"],
                scoring="smallest legacy depth-and-center control score",
            )
        else:
            add(
                "stable_control", common, -center_score, largest=False,
                measurement_quality="heuristic",
                source_signals=["valid_after_filter"],
                scoring="closest-to-center common-valid legacy control",
            )
        gap = candidate_maps["gap"]
        if gap is not None and gap.shape == candidate_valid.shape:
            add(
                "lowest_nonnegative_confidence_gap_proxy",
                candidate_valid & np.isfinite(gap) & (gap >= 0.0), gap, largest=False,
                measurement_quality="proxy", source_signals=["confidence_gap"],
                scoring="smallest nonnegative legacy confidence-gap proxy",
            )
        else:
            add(
                "lowest_nonnegative_confidence_gap_proxy", None, None, largest=False,
                measurement_quality="proxy", source_signals=["confidence_gap"],
                scoring="smallest nonnegative legacy confidence-gap proxy",
            )
        updates = candidate_maps["updates"]
        if updates is not None and updates.shape == candidate_valid.shape:
            add(
                "highest_accepted_update_count", candidate_valid, updates.astype(float),
                measurement_quality="instrumented_exact",
                source_signals=["accepted_update_count"],
                scoring="largest legacy accepted-update count",
            )
        else:
            add(
                "highest_accepted_update_count", None, None,
                measurement_quality="instrumented_exact",
                source_signals=["accepted_update_count"],
                scoring="largest legacy accepted-update count",
            )
    if not selected and width > 0 and height > 0:
        center = (width // 2, height // 2)
        if (
            TRACE_SELECTION_BORDER_PIXELS <= center[0] < width - TRACE_SELECTION_BORDER_PIXELS
            and TRACE_SELECTION_BORDER_PIXELS <= center[1] < height - TRACE_SELECTION_BORDER_PIXELS
        ):
            selected.append({
                "x": center[0], "y": center[1], "selection": "center_fallback_no_maps",
                "measurement_quality": "heuristic", "source_signals": [],
                "scoring": "center pixel when no legacy evidence map is selectable",
            })
            availability.append(trace_selection_record(
                "center_fallback_no_maps",
                status="selected",
                measurement_quality="heuristic",
                source_signals=[],
                scoring="center pixel when no legacy evidence map is selectable",
                point=center,
            ))
    return {
        "mode": "legacy_fallback",
        "fallback_reason": fallback_reason,
        "selected": selected,
        "availability": availability,
        "context": {},
    }


def evidence_trace_recommendations(
    baseline: pd.Series,
    candidate: pd.Series,
) -> dict[str, Any]:
    exact = exact_evidence_trace_recommendations(baseline, candidate)
    if exact["mode"] != "legacy_fallback":
        return exact
    legacy = legacy_evidence_trace_recommendations(
        baseline, candidate, str(exact["fallback_reason"])
    )
    legacy["availability"] = [*exact["availability"], *legacy["availability"]]
    legacy["context"] = exact["context"]
    return legacy


def evidence_trace_pixels(baseline: pd.Series, candidate: pd.Series) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only selected pixel records."""

    return evidence_trace_recommendations(baseline, candidate)["selected"]


def trace_capture_frames(frames: pd.DataFrame, configured_run: str) -> pd.DataFrame:
    """Resolve one trace-evidence cohort for a configured run identity.

    ``run`` is a presentation label and may be qualified as ``[deep]`` when a
    Process<true> maps capture is isolated from the production quality cohort.
    New reports carry ``configured_run`` explicitly so trace selection never
    needs to reverse-engineer that display label.
    """

    required = {"run", "repeat", "scene_id", "image_id"}
    missing = sorted(required - set(frames.columns))
    if missing:
        raise ValueError(
            "trace selection frame table is missing required columns: "
            + ", ".join(missing)
        )
    identity_column = "configured_run" if "configured_run" in frames.columns else "run"
    identity = frames[identity_column].astype("string")
    matches = frames[
        identity.eq(configured_run)
        & pd.to_numeric(frames["repeat"], errors="coerce").eq(0)
    ].copy()
    if matches.empty:
        available = sorted({
            str(value)
            for value in frames[identity_column].dropna().tolist()
        })
        raise ValueError(
            f"trace selection cannot resolve configured run {configured_run!r} "
            f"through frame column {identity_column!r}; available identities: {available}"
        )

    chosen: list[pd.Series] = []
    for (scene_id, image_id), group in matches.groupby(
        ["scene_id", "image_id"], sort=True, dropna=False
    ):
        diagnostic = (
            group[group["diagnostic_only"].eq(True)]
            if "diagnostic_only" in group.columns
            else group.iloc[0:0]
        )
        if len(diagnostic) == 1:
            chosen.append(diagnostic.iloc[0])
            continue
        if len(diagnostic) > 1:
            labels = sorted(diagnostic["run"].astype(str).tolist())
            raise ValueError(
                "trace selection found multiple diagnostic cohorts for configured "
                f"run {configured_run!r}, scene {scene_id!r}, image {image_id!r}: {labels}"
            )
        if len(group) != 1:
            labels = sorted(group["run"].astype(str).tolist())
            raise ValueError(
                "trace selection cannot choose an unambiguous cohort for configured "
                f"run {configured_run!r}, scene {scene_id!r}, image {image_id!r}: {labels}"
            )
        chosen.append(group.iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def generate_trace_manifest(
    config: dict[str, Any],
    frames: pd.DataFrame,
    output_dir: Path,
) -> Path:
    baseline = next(str(run["label"]) for run in config.get("runs") or [] if run.get("role") == "baseline")
    variants = [str(run["label"]) for run in config.get("runs") or [] if str(run["label"]) != baseline]
    traces = []
    category_availability: list[dict[str, Any]] = []
    frames = select_terminal_frames(frames)
    baseline_frames = trace_capture_frames(frames, baseline)
    for variant in variants:
        candidate = trace_capture_frames(frames, variant)
        paired = baseline_frames.merge(
            candidate,
            on=["scene_id", "image_id"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        if paired.empty:
            raise ValueError(
                f"trace selection found no paired scene/frame between configured runs "
                f"{baseline!r} and {variant!r}"
            )
        paired["effect"] = (
            (paired["valid_ratio_after_filter_candidate"] - paired["valid_ratio_after_filter_baseline"]).abs() / 0.005
            + (paired["final_cost_median_candidate"] - paired["final_cost_median_baseline"]).abs() / 0.005
        )
        paired["effect"] = paired["effect"].fillna(0.0)
        largest = paired.sort_values(
            ["effect", "scene_id", "image_id"],
            ascending=[False, True, True], kind="stable",
        ).head(5)
        smallest = paired.sort_values(
            ["effect", "scene_id", "image_id"],
            ascending=[True, True, True], kind="stable",
        ).head(3)
        selected = pd.concat([largest, smallest]).drop_duplicates(["scene_id", "image_id"])
        for _, row in selected.iterrows():
            baseline_row = pd.Series({key[:-9]: value for key, value in row.items() if key.endswith("_baseline")})
            candidate_row = pd.Series({key[:-10]: value for key, value in row.items() if key.endswith("_candidate")})
            recommendation = evidence_trace_recommendations(baseline_row, candidate_row)
            context = {
                "variant": variant,
                "scene_id": row["scene_id"],
                "image_id": int(row["image_id"]),
                "selection_mode": recommendation["mode"],
                "fallback_reason": recommendation["fallback_reason"],
                **recommendation["context"],
            }
            remaining = max(0, 32 - sum(1 for item in traces if item["variant"] == variant))
            retained = recommendation["selected"][:remaining]
            retained_keys = {
                (pixel["selection"], int(pixel["x"]), int(pixel["y"]))
                for pixel in retained
            }
            for availability in recommendation["availability"]:
                availability = {**context, **availability}
                key = (
                    availability["category"],
                    int(availability.get("x", -1)),
                    int(availability.get("y", -1)),
                )
                if availability["status"] == "selected" and key not in retained_keys:
                    availability["status"] = "not_selected"
                    availability["reason"] = "per-variant 32-pixel trace budget exhausted"
                category_availability.append(availability)
            for pixel in retained:
                traces.append({
                    "variant": variant, "scene_id": row["scene_id"], "image_id": int(row["image_id"]),
                    "x": int(pixel["x"]), "y": int(pixel["y"]),
                    "label": f"{variant}_{pixel['selection']}", "selection": pixel["selection"],
                    "selection_mode": recommendation["mode"],
                    "measurement_quality": pixel["measurement_quality"],
                    "source_signals": pixel["source_signals"],
                    "score": pixel.get("score"),
                    "scoring": pixel["scoring"],
                })
    path = output_dir / "trace_rerun.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": SCHEMA_VERSION,
        "selection_method": (
            "explicit configured-run identity; preferred isolated diagnostic cohort; "
            "paired terminal geometric schema-v4 exact evidence when available; "
            "truthfully labeled legacy fallback; six-pixel border; deterministic "
            "score then y,x; maximum 32 pixels per variant"
        ),
        "exact_category_policy": list(TRACE_EXACT_CATEGORIES),
        "category_availability": category_availability,
        "trace_pixels": traces,
    }, sort_keys=False), encoding="utf-8")
    return path


def observability_run_arguments(
    config: dict[str, Any],
    run: dict[str, Any],
    scene: dict[str, Any],
    command_work_dir: Path,
    run_dir: Path,
    purpose: str,
) -> list[str]:
    """Resolve the same run/scene configuration used by broad sweep captures."""

    densify_args = [
        *(str(value) for value in config.get("default_densify_args") or []),
        *(str(value) for value in run.get("densify_args") or []),
    ]
    argument_overrides = validated_argument_overrides(
        scene.get("argument_overrides"),
        f"scene {scene.get('scan_id', '<unknown>')!r}",
    )
    for option, value in argument_overrides.items():
        densify_args = replace_argument_value(densify_args, option, value)
    validate_supported_densify_args(densify_args)

    ini_overrides = config_materialization.merge_ini_overrides(config, run, scene)
    if not ini_overrides:
        return densify_args
    configured_ini = Path(argument_value(densify_args, "--dense-config-file", "Densify.ini"))
    if not configured_ini.is_absolute():
        configured_ini = command_work_dir / configured_ini
    generated_ini = run_dir / "generated" / f"Densify.{purpose}.ini"
    metadata = config_materialization.render_ini_override(
        configured_ini, generated_ini, ini_overrides
    )
    metadata.update({"purpose": purpose, "run": str(run.get("label", ""))})
    write_json(run_dir / "generated" / f"Densify.{purpose}.json", metadata)
    return replace_argument_value(
        densify_args, "--dense-config-file", str(generated_ini.resolve())
    )


def execute_trace_reruns(
    config: dict[str, Any],
    root: Path,
    manifest_path: Path,
    allow_over_budget: bool = False,
) -> list[dict[str, Any]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    trace_rows = manifest.get("trace_pixels") or []
    baseline = next(str(run["label"]) for run in config.get("runs") or [] if run.get("role") == "baseline")
    run_specs = {str(run["label"]): run for run in config.get("runs") or []}
    scenes = {str(scene["scan_id"]): scene for scene in resolve_scenes(config, resolve_suite(config))}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trace_rows:
        scene_id = str(row["scene_id"])
        for run_label in (baseline, str(row["variant"])):
            pixel = {
                "image_id": int(row["image_id"]), "x": int(row["x"]), "y": int(row["y"]),
                "label": f"{row['variant']}__{row['selection']}__{run_label}",
            }
            bucket = grouped.setdefault((run_label, scene_id), [])
            if pixel not in bucket:
                bucket.append(pixel)
    densify_bin = densify_binary(config, instrumented=True)
    executions = []
    admission_pairs = [
        (run_specs[run_label], scenes[scene_id])
        for run_label, scene_id in sorted(grouped)
        if run_label in run_specs and scene_id in scenes
    ]
    admit_on_demand_storage(
        config,
        root,
        request_id=dmap_drilldown.file_digest(manifest_path),
        kind="targeted_trace",
        run_scene_pairs=admission_pairs,
        allow_over_budget=allow_over_budget,
    )
    for (run_label, scene_id), pixels in sorted(grouped.items()):
        scene = scenes.get(scene_id)
        run = run_specs.get(run_label)
        if scene is None or run is None or not scene.get("working_folder") or not scene.get("mvs_file"):
            raise FileNotFoundError(f"cannot resolve trace rerun input for {run_label}/{scene_id}")
        run_dir = contained_output_path(
            root,
            "trace_reruns",
            validated_output_component(run_label, "trace run label"),
            validated_output_component(scene_id, "trace scene_id"),
            description="trace rerun directory",
        )
        work_dir = run_dir / "work"
        local_mvs = prepare_locked_profile_workspace(
            config, root, scene, work_dir
        )
        command_work_dir = dmap_working_folder(local_mvs)
        trace_config = run_dir / "trace_config.json"
        write_json(trace_config, {"trace_pixels": pixels})
        image_list = ",".join(str(value) for value in sorted({pixel["image_id"] for pixel in pixels}))
        densify_args = observability_run_arguments(
            config, run, scene, command_work_dir, run_dir, "trace"
        )
        densify_args = without_value_arguments(
            densify_args,
            {
                "--dmap-instrumentation-config",
                "--dmap-instrumentation-dir",
                "--dmap-instrumentation-image-list",
                "--dmap-instrumentation-level",
                "--dmap-instrumentation-sample-rate",
                "--dmap-instrumentation-write-maps",
            },
        )
        command = [
            str(densify_bin), "--working-folder", str(command_work_dir), "--input-file", str(local_mvs),
            "--output-file", str(run_dir / "trace_dense.mvs"),
            "--dmap-instrumentation-dir", str(run_dir / "dmap_instrumentation"),
            "--dmap-instrumentation-config", str(trace_config),
            "--dmap-instrumentation-level", "maps", "--dmap-instrumentation-sample-rate", "1",
            "--dmap-instrumentation-image-list", image_list, "--dmap-instrumentation-write-maps", "1",
        ]
        if not has_argument(densify_args, "--fusion-mode"):
            command.extend(["--fusion-mode", "1"])
        command.extend(densify_args)
        result = execute_densify_command(
            command,
            REPO_ROOT,
            run_dir,
            False,
            experiment_root=root,
            runtime_role="observer",
        )
        result.update({"run": run_label, "scene_id": scene_id, "trace_pixels": len(pixels)})
        executions.append(result)
        if result["return_code"] != 0:
            raise RuntimeError(f"targeted trace rerun failed: {run_dir}")
        prepare_locked_profile_workspace(config, root, scene, work_dir)
        depth_dir = run_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for dmap in sorted(command_work_dir.glob("depth*.dmap")):
            hardlink_or_copy(str(dmap), str(depth_dir / dmap.name))
        integrity.write_capture_artifact_closure(run_dir, "trace")
        pixels_by_image: dict[int, list[dict[str, int]]] = {}
        for pixel in pixels:
            pixels_by_image.setdefault(int(pixel["image_id"]), []).append(pixel)
        validations = []
        for image_id, image_pixels in sorted(pixels_by_image.items()):
            valid, reason = drilldown_run_complete(
                run_dir, "trace", image_id, image_pixels
            )
            if not valid:
                raise RuntimeError(
                    f"targeted trace validation failed at {run_dir}: {reason}"
                )
            validations.append({"image_id": image_id, "validation": reason})
        result["validation"] = validations
    write_json(root / "trace_reruns" / "executions.json", executions)
    return executions


def git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(status.strip())
    except Exception:
        return None


def normalize_repeated_drilldown_args(args: list[str]) -> list[str]:
    """Let ``--pixel`` and ``--variant`` be repeated despite Tyro list parsing."""
    if not args or args[0] != "drilldown":
        return args

    def collapse(values: list[str], flag: str) -> list[str]:
        output: list[str] = []
        selected: list[str] = []
        index = 0
        while index < len(values):
            value = values[index]
            if value == flag:
                index += 1
                selection_start = len(selected)
                while index < len(values) and not values[index].startswith("--"):
                    selected.append(values[index])
                    index += 1
                if len(selected) == selection_start:
                    selected.append("")
                continue
            prefix = f"{flag}="
            if value.startswith(prefix):
                selected.append(value[len(prefix):])
            else:
                output.append(value)
            index += 1
        if selected:
            output.extend([flag, *selected])
        return output

    normalized = collapse(args, "--pixel")
    return collapse(normalized, "--variant")


def drilldown_run_complete(
    run_dir: Path,
    profile: str,
    image_id: int,
    trace_pixels: list[dict[str, int]],
) -> tuple[bool, str]:
    def as_integer(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    if profile in {"deep", "trace"}:
        maps_valid, maps_reason = validate_completed_run_mode(run_dir, "maps")
        if not maps_valid:
            return False, f"full-frame exact maps validation failed: {maps_reason}"
        if profile == "deep":
            return True, maps_reason
    repro = read_json(run_dir / "repro.json")
    if bool(repro.get("dry_run")) or as_integer(repro.get("return_code")) != 0:
        return False, "run did not complete successfully"
    command = repro.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) for value in command
    ):
        return False, "repro.json does not contain the resolved DensifyPointCloud command"
    controlled_config = run_dir / "generated" / "Densify.drilldown.cfg"
    configured_program_options = Path(
        argument_value(command, "--config-file", "")
    ).expanduser()
    if (
        not str(configured_program_options)
        or configured_program_options.resolve() != controlled_config.resolve()
        or controlled_config.is_symlink()
        or not controlled_config.is_file()
        or controlled_config.read_bytes() != b""
    ):
        return False, "resolved command is not bound to the empty drill-down program-options file"
    try:
        geometric_iterations = parse_argument(command, "--geometric-iters", 2)
        fusion_mode = parse_argument(command, "--fusion-mode", 0)
    except ValueError as exc:
        return False, f"resolved stage topology is malformed: {exc}"
    if geometric_iterations < 0:
        return False, "resolved --geometric-iters must be non-negative"
    expected_geometric_stages = (
        set(range(geometric_iterations)) if fusion_mode >= 0 else set()
    )
    instrumentation_dir = run_dir / "dmap_instrumentation"
    if read_json(instrumentation_dir / "run_metadata.json").get("schema_name") != "openmvs.dmap.run":
        return False, "run_metadata.json is missing or malformed"
    if read_json(instrumentation_dir / "scene_summary.json").get("schema_name") != "openmvs.dmap.scene_summary":
        return False, "scene_summary.json is missing or malformed"
    requested_order = [
        (int(pixel["x"]), int(pixel["y"])) for pixel in trace_pixels
    ]
    if not requested_order:
        return False, "trace request does not contain any pixels"
    requested = set(requested_order)

    geometric_root = instrumentation_dir / "geometric_iterations"
    observed_geometric_stages: set[int] = set()
    if geometric_root.exists() or geometric_root.is_symlink():
        if geometric_root.is_symlink() or not geometric_root.is_dir():
            return False, "geometric stage root is not a regular directory"
        for stage_dir in geometric_root.iterdir():
            if stage_dir.is_symlink() or not stage_dir.is_dir():
                return False, f"unexpected geometric stage artifact: {stage_dir.name}"
            match = re.fullmatch(r"iteration(\d+)", stage_dir.name)
            if match is None:
                return False, f"unexpected geometric stage directory: {stage_dir.name}"
            stage_index = int(match.group(1))
            if stage_index in observed_geometric_stages:
                return False, f"duplicate geometric stage index: {stage_index}"
            observed_geometric_stages.add(stage_index)
    if observed_geometric_stages != expected_geometric_stages:
        return False, (
            "geometric stage topology does not match the resolved command: "
            f"expected {sorted(expected_geometric_stages)}, observed "
            f"{sorted(observed_geometric_stages)}"
        )

    all_expected_states: set[tuple[str, int, int, int, int]] = set()
    stage_roots = instrumentation_stage_roots(instrumentation_dir)
    for estimation_stage, geometric_iteration, stage_root in stage_roots:
        stage_index = -1 if geometric_iteration is None else geometric_iteration
        stage_label = (
            "photometric"
            if geometric_iteration is None
            else f"geometric iteration {geometric_iteration}"
        )
        stage_metadata = read_json(stage_root / "run_metadata.json")
        stage_summary = read_json(stage_root / "scene_summary.json")
        if (
            stage_metadata.get("schema_name") != "openmvs.dmap.run"
            or stage_metadata.get("estimation_stage") != estimation_stage
            or (
                geometric_iteration is None
                and stage_metadata.get("geometric_iteration") is not None
            )
            or (
                geometric_iteration is not None
                and as_integer(stage_metadata.get("geometric_iteration")) != stage_index
            )
            or stage_summary.get("schema_name") != "openmvs.dmap.scene_summary"
            or stage_summary.get("estimation_stage") != estimation_stage
            or (
                geometric_iteration is None
                and stage_summary.get("geometric_iteration") is not None
            )
            or (
                geometric_iteration is not None
                and as_integer(stage_summary.get("geometric_iteration")) != stage_index
            )
        ):
            return False, f"{stage_label} metadata is missing or malformed"

        target_frames = [
            path.parent
            for path in sorted((stage_root / "depthmaps").glob("*/summary.json"))
            if as_integer(read_json(path).get("image_id")) == image_id
        ]
        if not target_frames:
            return False, f"no map capture was found for image {image_id} in {stage_label}"
        manifest_states: set[tuple[int, int]] = set()
        for frame_dir in target_frames:
            manifest = read_json(frame_dir / "map_manifest.json")
            exact_capture = manifest.get("exact_capture") or {}
            if exact_capture.get("requested") is not True or exact_capture.get("available") is not True:
                return False, f"exact Process<true> maps are unavailable in {stage_label}"
            num_iterations = as_integer(manifest.get("num_iterations"))
            num_logical_states = as_integer(manifest.get("num_logical_states"))
            if num_iterations < 0 or num_logical_states != num_iterations + 1:
                return False, f"logical-state topology is malformed in {stage_label}"
            levels = {
                as_integer(item.get("pyramid_level", item.get("scale_number", 0)), 0)
                for item in manifest.get("maps") or []
                if isinstance(item, dict)
            } or {as_integer(manifest.get("pyramid_level"), 0)}
            if any(level < 0 for level in levels):
                return False, f"pyramid-level topology is malformed in {stage_label}"
            manifest_states.update(
                (level, iteration)
                for level in levels
                for iteration in range(-1, num_iterations)
            )

        # PatchMatch scales requested full-resolution coordinates independently
        # at each pyramid level and first-wins deduplicates collisions.
        plans_by_level: dict[int, dict[str, Any]] = {}
        for plan in read_jsonl(stage_root / "resource_plans.jsonl"):
            if as_integer(plan.get("image_id")) != image_id:
                continue
            level = as_integer(plan.get("pyramid_level"))
            if level < 0 or level > 30 or level in plans_by_level:
                return False, f"trace resource-plan topology is malformed in {stage_label}"
            plans_by_level[level] = plan

        expected_coordinates: dict[tuple[int, int], tuple[int, int]] = {}
        expected_states: set[tuple[int, int, int]] = set()
        if plans_by_level:
            for level, plan in sorted(plans_by_level.items()):
                width = as_integer(plan.get("width"))
                height = as_integer(plan.get("height"))
                num_trace_pixels = as_integer(plan.get("num_trace_pixels"))
                num_logical_states = as_integer(plan.get("num_logical_states"))
                if width <= 0 or height <= 0 or num_trace_pixels < 0 or num_logical_states <= 0:
                    return False, f"trace resource-plan topology is malformed in {stage_label}"
                try:
                    selected = dmap_drilldown.trace_pyramid_layout(
                        requested_order, level, width=width, height=height
                    )
                except ValueError as exc:
                    return False, f"trace resource-plan topology is malformed: {exc}"
                if len(selected) != num_trace_pixels:
                    return False, (
                        f"trace resource plan declares {num_trace_pixels} pixels at pyramid "
                        f"level {level} in {stage_label}, but the request resolves to "
                        f"{len(selected)}"
                    )
                represented_requests = sum(len(slot.request_indices) for slot in selected)
                if level == 0 and represented_requests != len(requested_order):
                    return False, "one or more requested pixels are outside the full-resolution frame"
                if not selected:
                    if plan.get("trace_requested") is not False or plan.get("trace_available") is not False:
                        return False, f"empty trace resource plan is malformed in {stage_label}"
                    continue
                if plan.get("trace_requested") is not True or plan.get("trace_available") is not True:
                    return False, f"targeted trace is unavailable in {stage_label} at level {level}"
                for trace_index, slot in enumerate(selected):
                    expected_coordinates[(level, trace_index)] = slot.coordinate
                    expected_states.update(
                        (trace_index, level, iteration)
                        for iteration in range(-1, num_logical_states - 1)
                    )
            plan_states = {(level, iteration) for _, level, iteration in expected_states}
            manifest_levels = {level for level, _ in manifest_states}
            declared_plan_states = {
                state for state in plan_states if state[0] in manifest_levels
            }
            if manifest_states != declared_plan_states:
                return False, f"trace and exact-map logical-state topology disagree in {stage_label}"
        else:
            for level, iteration in manifest_states:
                if level != 0:
                    return False, "multiscale trace validation requires resource_plans.jsonl"
                for trace_index, slot in enumerate(
                    dmap_drilldown.trace_pyramid_layout(requested_order, level)
                ):
                    expected_coordinates[(level, trace_index)] = slot.coordinate
                    expected_states.add((trace_index, level, iteration))

        traces_path = stage_root / "instrumentation" / "traces.jsonl"
        if not traces_path.is_file():
            return False, f"targeted trace output is missing in {stage_label}"
        observed_states: set[tuple[int, int, int]] = set()
        for row in read_jsonl(traces_path):
            if as_integer(row.get("image_id")) != image_id:
                continue
            trace_index = as_integer(row.get("trace_index"))
            level = as_integer(row.get("pyramid_level", row.get("scale_number")))
            iteration = as_integer(row.get("logical_iteration", row.get("iteration")))
            coordinate = (as_integer(row.get("x")), as_integer(row.get("y")))
            state = (trace_index, level, iteration)
            if expected_coordinates.get((level, trace_index)) != coordinate:
                return False, (
                    f"trace output in {stage_label} does not bind its compact slot "
                    "to the requested pixel layout"
                )
            if state not in expected_states:
                return False, f"trace output in {stage_label} contains an undeclared state row"
            if state in observed_states:
                return False, f"trace output in {stage_label} contains a duplicate state row"
            observed_states.add(state)
        missing = expected_states - observed_states
        if missing:
            return False, (
                f"trace output in {stage_label} is missing {len(missing)} "
                "requested pixel/state rows"
            )
        all_expected_states.update(
            (estimation_stage, stage_index, trace_index, level, iteration)
            for trace_index, level, iteration in expected_states
        )

    logical_states = {
        (stage, stage_index, level, iteration)
        for stage, stage_index, _, level, iteration in all_expected_states
    }
    return True, (
        f"validated {len(requested)} targeted trace pixels across "
        f"{len(stage_roots)} stage(s) and {len(logical_states)} logical states"
    )


def drilldown_run_command(
    config: dict[str, Any],
    run: dict[str, Any],
    request: dict[str, Any],
    run_dir: Path,
    work_dir: Path,
    local_mvs: Path,
    trace_config: Path | None,
    program_options_config: Path,
    scene: dict[str, Any] | None = None,
) -> list[str]:
    densify_bin = densify_binary(config, instrumented=True)
    densify_args = observability_run_arguments(
        config, run, scene or {}, dmap_working_folder(local_mvs), run_dir, "drilldown"
    )
    densify_args = without_value_arguments(
        densify_args,
        {
            "--dmap-instrumentation-config",
            "--dmap-instrumentation-dir",
            "--dmap-instrumentation-image-list",
            "--dmap-instrumentation-level",
            "--dmap-instrumentation-sample-rate",
            "--dmap-instrumentation-write-maps",
            "--patch-match-cuda-instances",
            "--config-file",
        },
    )
    profile = str(request["capture_profile"])
    target = request["target"]
    command = [
        str(densify_bin),
        "--config-file", str(program_options_config.resolve()),
        # The second Boost notify() for an existing config file reapplies this
        # raw value after OpenMVS first normalizes it. Preserve the separator so
        # DMAP paths remain inside the staged working directory.
        "--working-folder", f"{dmap_working_folder(local_mvs)}{os.sep}",
        "--input-file", str(local_mvs),
        "--output-file", str(run_dir / "drilldown_dense.mvs"),
        "--patch-match-cuda-instances", "1",
        "--dmap-instrumentation-dir", str(run_dir / "dmap_instrumentation"),
        "--dmap-instrumentation-level", "maps",
        "--dmap-instrumentation-sample-rate", "1",
        "--dmap-instrumentation-image-list", str(int(target["image_id"])),
        "--dmap-instrumentation-write-maps", "1",
    ]
    if trace_config is not None:
        command.extend(["--dmap-instrumentation-config", str(trace_config)])
    if not has_argument(densify_args, "--fusion-mode"):
        command.extend(["--fusion-mode", "1"])
    command.extend(densify_args)
    return command


def refresh_drilldown_index(root: Path) -> Path:
    drilldown_root = root / "drilldowns"
    entries: list[dict[str, Any]] = []
    for request_path in sorted((drilldown_root / "requests").glob("*.yaml")):
        try:
            request = dmap_drilldown.load_request(request_path)
        except Exception as exc:
            entries.append({
                "request": str(request_path.relative_to(root)),
                "status": "invalid_request",
                "error": str(exc),
            })
            continue
        request_id = str(request["request_sha256"])
        capture_dir = drilldown_root / "captures" / request_id
        executions_path = capture_dir / "executions.json"
        executions_value = read_json(executions_path) if executions_path.is_file() else {}
        executions = executions_value.get("executions")
        requested_labels = [str(run["label"]) for run in request["runs"]]
        capture_request = capture_dir / "request.yaml"
        capture_request_bound = (
            not capture_request.is_symlink()
            and capture_request.is_file()
            and capture_request.read_bytes() == request_path.read_bytes()
        )
        execution_contract_valid = (
            executions_value.get("schema_name") == "openmvs.dmap.drilldown_executions"
            and executions_value.get("schema_version") == 1
            and executions_value.get("request_sha256") == request_id
            and isinstance(executions, list)
            and len(executions) == len(requested_labels)
            and {
                str(execution.get("run")) for execution in executions
                if isinstance(execution, dict)
            } == set(requested_labels)
            and all(
                isinstance(execution, dict)
                and execution.get("scene_id") == request["target"]["scene_id"]
                and execution.get("capture_profile") == request["capture_profile"]
                and not isinstance(execution.get("return_code"), bool)
                and isinstance(execution.get("return_code"), int)
                and execution.get("return_code") == 0
                for execution in executions or []
            )
        )
        runs_complete = False
        if capture_request_bound and execution_contract_valid:
            trace_pixels = dmap_drilldown.expand_trace_pixels(request)
            runs_complete = all(
                drilldown_run_complete(
                    capture_dir / "runs" / label / str(request["target"]["scene_id"]),
                    str(request["capture_profile"]),
                    int(request["target"]["image_id"]),
                    trace_pixels,
                )[0]
                for label in requested_labels
            )
        entries.append({
            "request_sha256": request_id,
            "capture_profile": request["capture_profile"],
            "scene_id": request["target"]["scene_id"],
            "image_id": request["target"]["image_id"],
            "trace_pixel_count": request["target"]["trace_pixel_count"],
            "run_labels": [run["label"] for run in request["runs"]],
            "status": (
                "complete" if runs_complete
                else "incomplete" if capture_dir.exists()
                else "requested"
            ),
            "request": str(request_path.relative_to(root)),
            "capture": str(capture_dir.relative_to(root)) if capture_dir.exists() else None,
            "executions": str(executions_path.relative_to(root)) if executions_path.is_file() else None,
        })
    index_path = drilldown_root / "index.json"
    write_json(index_path, {
        "schema_name": dmap_drilldown.INDEX_SCHEMA_NAME,
        "schema_version": 1,
        "entries": entries,
    })
    return index_path


def execute_drilldown_request(
    config: dict[str, Any],
    root: Path,
    request_path: Path,
    allow_over_budget: bool = False,
) -> list[dict[str, Any]]:
    request = dmap_drilldown.load_request(request_path)
    config_source = Path(str(config["_config_path"])).resolve()
    if dmap_drilldown.file_digest(config_source) != request["experiment"]["config_sha256"]:
        raise RuntimeError("experiment config changed after the drill-down request was created")
    scene_id = str(request["target"]["scene_id"])
    image_id = int(request["target"]["image_id"])
    profile = str(request["capture_profile"])
    trace_pixels = dmap_drilldown.expand_trace_pixels(request)
    scenes = {str(scene["scan_id"]): scene for scene in resolve_scenes(config, resolve_suite(config))}
    run_specs = {str(run["label"]): run for run in config.get("runs") or []}
    scene = scenes.get(scene_id)
    if scene is None or not scene.get("working_folder") or not scene.get("mvs_file"):
        raise FileNotFoundError(f"cannot resolve drill-down input scene {scene_id}")
    dmap_drilldown.validate_trace_row_admission(
        config, request, argument_overrides=scene.get("argument_overrides")
    )
    source_work = Path(str(scene["working_folder"])).expanduser().resolve()
    source_mvs = Path(str(scene["mvs_file"])).expanduser().resolve()
    densify_bin = densify_binary(config, instrumented=True)
    if not densify_bin.is_file():
        raise FileNotFoundError(f"DensifyPointCloud executable does not exist: {densify_bin}")
    plans: list[tuple[dict[str, Any], Path, bool]] = []
    request_id = str(request["request_sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", request_id) is None:
        raise ValueError("drill-down request_sha256 must be 64 lowercase hexadecimal characters")
    capture_root = contained_output_path(
        root,
        "drilldowns",
        "captures",
        request_id,
        description="drill-down capture directory",
    )
    for requested_run in request["runs"]:
        label = validated_output_component(
            requested_run.get("label"), "drill-down run label"
        )
        run = run_specs.get(label)
        if run is None:
            raise ValueError(f"drill-down run is no longer present in the config: {label}")
        run_dir = contained_output_path(
            capture_root,
            "runs",
            label,
            validated_output_component(scene_id, "drill-down scene_id"),
            description="drill-down run directory",
        )
        complete = False
        if run_dir.exists() and any(run_dir.iterdir()):
            complete, reason = drilldown_run_complete(run_dir, profile, image_id, trace_pixels)
            if not complete:
                raise RuntimeError(
                    f"refusing to overwrite incomplete drill-down run at {run_dir}: {reason}; "
                    "remove that run directory explicitly before retrying"
                )
        plans.append((run, run_dir, complete))
    pending_pairs = [(run, scene) for run, _run_dir, complete in plans if not complete]
    if pending_pairs:
        admit_on_demand_storage(
            config,
            root,
            request_id=request_id,
            kind=f"drilldown_{profile}",
            run_scene_pairs=pending_pairs,
            allow_over_budget=allow_over_budget,
        )
    capture_root.mkdir(parents=True, exist_ok=True)
    capture_request = capture_root / "request.yaml"
    if capture_request.exists():
        captured_request = yaml.safe_load(capture_request.read_text(encoding="utf-8")) or {}
        if captured_request != request:
            raise RuntimeError(f"capture request conflicts with immutable request {request_path}")
    else:
        shutil.copy2(request_path, capture_request)
    executions: list[dict[str, Any]] = []
    for run, run_dir, complete in plans:
        if complete:
            result = read_json(run_dir / "repro.json")
            result.update({
                "run": str(run["label"]),
                "scene_id": scene_id,
                "capture_profile": profile,
                "trace_pixels": len(trace_pixels),
                "reused": True,
            })
            executions.append(result)
            continue
        work_dir = run_dir / "work"
        local_mvs = prepare_locked_profile_workspace(
            config, root, scene, work_dir
        )
        dmap_drilldown.validate_no_implicit_program_options_file(
            dmap_working_folder(local_mvs)
        )
        program_options_config = run_dir / "generated" / "Densify.drilldown.cfg"
        if program_options_config.is_symlink():
            raise RuntimeError(
                f"controlled drill-down program-options path is a symlink: "
                f"{program_options_config}"
            )
        write_immutable_text(
            program_options_config,
            "",
            "controlled empty drill-down program-options file",
        )
        trace_config = None
        if profile == "trace":
            trace_config = run_dir / "trace_config.json"
            trace_rows = [
                {
                    "image_id": image_id,
                    "x": int(pixel["x"]),
                    "y": int(pixel["y"]),
                    "label": f"drilldown__{str(request['request_sha256'])[:12]}__{run['label']}__{index:04d}",
                }
                for index, pixel in enumerate(trace_pixels)
            ]
            write_json(trace_config, {"trace_pixels": trace_rows})
        command = drilldown_run_command(
            config,
            run,
            request,
            run_dir,
            work_dir,
            local_mvs,
            trace_config,
            program_options_config,
            scene,
        )
        result = execute_densify_command(
            command,
            REPO_ROOT,
            run_dir,
            False,
            experiment_root=root,
            runtime_role="observer",
        )
        result.update({
            "run": str(run["label"]),
            "scene_id": scene_id,
            "capture_profile": profile,
            "trace_pixels": len(trace_pixels),
            "reused": False,
        })
        if result["return_code"] != 0:
            raise RuntimeError(f"drill-down capture failed: {run_dir}")
        prepare_locked_profile_workspace(config, root, scene, work_dir)
        depth_dir = run_dir / "depth_maps"
        depth_dir.mkdir(exist_ok=True)
        for dmap in sorted(dmap_working_folder(local_mvs).glob("depth*.dmap")):
            hardlink_or_copy(str(dmap), str(depth_dir / dmap.name))
        integrity.write_capture_artifact_closure(run_dir, profile)
        valid, reason = drilldown_run_complete(run_dir, profile, image_id, trace_pixels)
        if not valid:
            raise RuntimeError(f"drill-down capture validation failed at {run_dir}: {reason}")
        result["validation"] = reason
        executions.append(result)
    write_json(capture_root / "executions.json", {
        "schema_name": "openmvs.dmap.drilldown_executions",
        "schema_version": 1,
        "request_sha256": request["request_sha256"],
        "executions": executions,
    })
    refresh_drilldown_index(root)
    return executions


def refresh_master_report(config_path_value: Path, report_dir: Path) -> None:
    wrapper = REPO_ROOT / "tools" / "dmap_observability.sh"
    report_dir = ensure_external_output_path(
        report_dir, "drilldown report directory"
    )
    command = [
        str(wrapper), "report",
        "--config", str(config_path_value.resolve()),
        "--report-dir", str(report_dir),
        "--python", sys.executable,
        "--rebuild-report",
    ]
    context_identity = (
        read_json(report_dir / "report_policy.json").get("evidence_context") or {}
    )
    context_path = context_identity.get("path")
    if context_path:
        command.extend(["--evidence-context", str(context_path)])
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
    )


def validate_report(path: Path, *, write_sidecar: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    report_policy = read_json(path.parent / "report_policy.json")
    report_tree_closure = integrity.validate_report_tree_closure(path.parent)
    published_root_raw = report_policy.get("published_output_dir")
    published_root = (
        Path(str(published_root_raw)).expanduser().absolute()
        if isinstance(published_root_raw, str) and published_root_raw
        else path.parent
    )
    finalizer_receipt_path = path.parent / "pre_publish_finalizer_receipt.json"
    finalizer_receipt_validation = {"present": False, "valid": True}
    if finalizer_receipt_path.exists() or finalizer_receipt_path.is_symlink():
        # Import lazily to avoid the dmap_sweep -> dmap_dev module cycle. Receipt
        # semantics are validated before this function can write a sidecar.
        import generate_attested_dmap_report

        finalizer_receipt_validation = (
            generate_attested_dmap_report.validate_finalizer_receipt(
                path.parent, published_root, require_receipt=True,
            )
        )

    def bundle_artifact(raw_path: Any) -> Path:
        """Resolve publication-relative inventory paths against this exact bundle."""

        artifact = Path(str(raw_path))
        if not artifact.is_absolute():
            return (path.parent / artifact).resolve()
        if published_root != path.parent:
            try:
                relative = artifact.relative_to(published_root)
            except ValueError:
                pass
            else:
                return (path.parent / relative).resolve()
        return artifact

    def published_artifact(local_path: Path) -> Path:
        try:
            relative = local_path.relative_to(path.parent)
        except ValueError:
            return local_path
        return published_root / relative

    report_model_path = path.parent / "report_model.json"
    report_model_schema_version = safe_float(read_json(report_model_path).get("schema_version"))
    guide_required = report_model_schema_version is not None and report_model_schema_version >= 2
    opening = len(re.findall(r"<details\b", text))
    closing = text.count("</details>")
    references: list[str] = []
    for markdown, html_ref in re.findall(r"!?\[[^]]*\]\(([^)]+)\)|(?:href|src)=[\"']([^\"']+)", text):
        references.append(unquote(markdown or html_ref))
    class LinkCollector(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.values: list[str] = []

        def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
            for key, value in attrs:
                if key in {"href", "src"} and value:
                    self.values.append(unquote(value))

    for html_name in ("01_development_report.html", "02_investigation.html"):
        html_file = path.parent / html_name
        if html_file.is_file():
            collector = LinkCollector()
            collector.feed(html_file.read_text(encoding="utf-8"))
            references.extend(collector.values)
    missing = []
    unsafe_references = []
    for reference in references:
        parsed = urlparse(reference)
        if parsed.scheme in {"http", "https", "data"} or reference.startswith("#"):
            continue
        local_value = parsed.path if parsed.scheme else reference.split("#", 1)[0]
        if (
            parsed.scheme == "file"
            or Path(local_value).is_absolute()
            or ".." in Path(local_value).parts
            or "\\" in reference
        ):
            unsafe_references.append(reference)
            continue
        target = (path.parent / local_value).resolve()
        try:
            target.relative_to(path.parent)
        except ValueError:
            unsafe_references.append(reference)
            continue
        if target.is_symlink() or not target.is_file():
            missing.append(reference)
    required = [
        "## 1. Executive Summary", "## 6. Annotation Consistency", "## 7. Algorithm Mechanics",
        "### Fixed-model Inlier Threshold Sweep", "### Fitted-model Stability vs Baseline",
        "### Instrumentation Coverage", "## 8. Per-scene Analysis", "## 10. Reproducibility", "## 11. Recommendations",
    ]
    if guide_required:
        required.insert(0, dmap_report_model.INVESTIGATION_GUIDE_HEADING)
    missing_sections = [section for section in required if section not in text]
    required_mechanisms = [
        "Cost Function and Convergence",
        "View Selection and Support",
        "Update Dynamics",
        "Filtering and Completeness",
        "Runtime and Scalability",
        "Geometric End Metrics",
        "Cross-run Effects",
        "Final State Overview",
    ]
    missing_mechanisms = [mechanism for mechanism in required_mechanisms if mechanism not in text]
    inventory_path = path.parent / "report_inventory.json"
    inventory_missing_paths: list[str] = []
    inventory_valid = inventory_path.is_file()
    resource_plan_validation_required = False
    if inventory_valid:
        inventory = read_json(inventory_path)
        inventory_mechanisms = set(str(value) for value in inventory.get("required_mechanisms") or [])
        if any(mechanism not in inventory_mechanisms for mechanism in required_mechanisms if mechanism != "Geometric End Metrics"):
            inventory_valid = False
        for scene in (inventory.get("scenes") or {}).values():
            for item in [*(scene.get("plots") or []), *(scene.get("panels") or [])]:
                artifact = bundle_artifact(item.get("path", ""))
                if not artifact.is_file():
                    inventory_missing_paths.append(str(artifact))
        for item in inventory.get("overall_plots") or []:
            artifact = bundle_artifact(item.get("path", ""))
            if not artifact.is_file():
                inventory_missing_paths.append(str(artifact))
        for item in inventory.get("geometry_artifacts") or []:
            for key in ("visual_overlay_svg", "visual_residual_histogram_svg"):
                if key not in item:
                    continue
                artifact = bundle_artifact(item[key])
                if not artifact.is_file():
                    inventory_missing_paths.append(str(artifact))
        for item in inventory.get("data_artifacts") or []:
            if item.get("name") == "resource_plan_validation":
                resource_plan_validation_required = True
            for key in ("csv", "parquet", "json"):
                raw_path = item.get(key)
                if not raw_path:
                    continue
                artifact = bundle_artifact(raw_path)
                if not artifact.is_file():
                    inventory_missing_paths.append(str(artifact))
        finalizer_inventory = inventory.get("pre_publish_finalizer")
        if finalizer_inventory is not None:
            if not (
                isinstance(finalizer_inventory, dict)
                and finalizer_inventory.get("schema_name")
                == "openmvs.dmap.pre_publish_finalizer_inventory"
                and finalizer_inventory.get("schema_version") == 1
                and finalizer_inventory.get("finalizer_receipt")
                == "pre_publish_finalizer_receipt.json"
                and isinstance(finalizer_inventory.get("approved_outputs"), list)
            ):
                inventory_valid = False
            else:
                receipt = bundle_artifact(
                    finalizer_inventory["finalizer_receipt"]
                )
                if receipt.is_symlink() or not receipt.is_file():
                    inventory_missing_paths.append(str(receipt))
                for output in finalizer_inventory["approved_outputs"]:
                    if not isinstance(output, dict) or set(output) != {
                        "path", "sha256", "bytes", "mode",
                    }:
                        inventory_valid = False
                        continue
                    artifact = bundle_artifact(output["path"])
                    if artifact.is_symlink() or not artifact.is_file():
                        inventory_missing_paths.append(str(artifact))
                        continue
                    if not (
                        artifact.stat().st_size == output["bytes"]
                        and stat.S_IMODE(artifact.stat().st_mode) == output["mode"]
                        and dmap_drilldown.file_digest(artifact) == output["sha256"]
                    ):
                        inventory_valid = False
        inventory_valid = inventory_valid and not inventory_missing_paths
    resource_plan_validation_path = path.parent / "resource_plan_validation.json"
    resource_plan_validation_valid = True
    if resource_plan_validation_required:
        resource_plan_validation_valid = (
            resource_plan_validation_path.is_file()
            and bool(read_json(resource_plan_validation_path).get("valid"))
        )
    report_tree_closure_result = report_tree_closure.as_dict()
    report_tree_closure_result["manifest_path"] = str(
        published_artifact(path.parent / integrity.REPORT_CLOSURE_FILE)
    )
    result = {
        "report": str(published_artifact(path)),
        "details_open": opening, "details_close": closing,
        "references": len(references), "missing_references": missing,
        "unsafe_external_or_traversal_references": unsafe_references,
        "missing_sections": missing_sections,
        "report_model_schema_version": report_model_schema_version,
        "investigation_guide_required": guide_required,
        "missing_mechanisms": missing_mechanisms,
        "inventory": str(published_artifact(inventory_path)),
        "inventory_valid": inventory_valid,
        "inventory_missing_paths": inventory_missing_paths,
        "resource_plan_validation_required": resource_plan_validation_required,
        "resource_plan_validation_valid": resource_plan_validation_valid,
        "report_tree_closure": report_tree_closure_result,
        "report_tree_closure_valid": report_tree_closure.valid,
        "report_tree_closure_status": report_tree_closure.status,
        "trusted_finalizer_receipt_present": finalizer_receipt_validation["present"],
        "trusted_finalizer_receipt_valid": finalizer_receipt_validation["valid"],
        "valid": opening == closing and not missing and not unsafe_references and not missing_sections and not missing_mechanisms and inventory_valid and resource_plan_validation_valid and report_tree_closure.valid,
    }
    if write_sidecar:
        write_json(path.with_suffix(".validation.json"), result)
    return result


def render_html(markdown: Path, html_path: Path) -> None:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        raise RuntimeError("pandoc is required for the canonical HTML companion")
    subprocess.run(
        [pandoc, "-f", "gfm", "-t", "html5", "--standalone", "--metadata", "title=Depth-map Development Report", str(markdown), "-o", str(html_path)],
        check=True,
    )


def existing_report_action(
    output_dir: Path,
    expected_policy: dict[str, Any],
) -> Literal["build", "skip"]:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return "build"
    report_path = output_dir / "01_development_report.md"
    model_path = output_dir / "report_model.json"
    required = (
        report_path,
        output_dir / "01_development_report.html",
        output_dir / "02_investigation.html",
        model_path,
        output_dir / "report_manifest.json",
        output_dir / "report_inventory.json",
        output_dir / "report_policy.json",
    )
    if all(path.is_file() for path in required):
        actual_policy = read_json(output_dir / "report_policy.json")
        if actual_policy != expected_policy:
            raise RuntimeError(
                f"refusing to reuse report directory {output_dir}: report policy mismatch; "
                "choose a new numbered --output-dir"
            )
        capture_evidence = expected_policy.get("capture_evidence") or {}
        if capture_evidence.get("reuse_eligible") is not True:
            raise RuntimeError(
                f"refusing to reuse report directory {output_dir}: capture evidence "
                "lacks verified artifact closure; rebuild into a staged report"
            )
        markdown_validation = validate_report(report_path, write_sidecar=False)
        model = read_json(model_path)
        model_validation = dmap_report_model.validate_report_model(model, output_dir)
        if markdown_validation.get("valid") and model_validation.get("valid"):
            return "skip"
    raise RuntimeError(
        f"refusing to overwrite incomplete or invalid report directory {output_dir}; "
        "choose a new numbered --output-dir or remove the directory explicitly"
    )


def build_report(
    config: dict[str, Any],
    root: Path,
    output_dir: Path,
    skip_diagnostics: bool,
    *,
    report_policy_activation_source: str = "effective_config",
    evidence_context_path: Path | None = None,
    published_output_dir: Path | None = None,
) -> Path:
    raw_output_dir = output_dir.expanduser().absolute()
    if raw_output_dir.is_symlink():
        raise ValueError(f"report output directory must not be a symlink: {raw_output_dir}")
    output_dir = ensure_external_output_path(output_dir, "report output directory")
    raw_published_output_dir = (
        published_output_dir.expanduser().absolute()
        if published_output_dir is not None
        else raw_output_dir
    )
    if raw_published_output_dir.is_symlink():
        raise ValueError(
            "published report directory must not be a symlink: "
            f"{raw_published_output_dir}"
        )
    published_output_dir = (
        ensure_external_output_path(
            published_output_dir, "published report directory"
        )
        if published_output_dir is not None else output_dir
    )
    if published_output_dir.parent.resolve() != output_dir.parent.resolve():
        raise ValueError(
            "staged and published report directories must be siblings for atomic promotion"
        )
    staged_publication = published_output_dir != output_dir

    def write_report_dataframe(
        dataframe: pd.DataFrame, path: Path,
    ) -> dict[str, Any]:
        materialized = (
            rebase_report_dataframe_paths(
                dataframe, output_dir, published_output_dir
            )
            if staged_publication else dataframe
        )
        return write_dataframe(materialized, path)

    evidence_context_identity = (
        file_identity(evidence_context_path)
        if evidence_context_path is not None else None
    )
    evidence_context = load_report_evidence_context(evidence_context_path)
    capture_profile_coverage = build_capture_profile_coverage(config, root)
    capture_evidence = build_capture_evidence_policy(
        root, capture_profile_coverage
    )
    report_policy = build_report_policy(
        config,
        root,
        skip_diagnostics=skip_diagnostics,
        capture_evidence=capture_evidence,
        activation_source=report_policy_activation_source,
        evidence_context_path=evidence_context_path,
        published_output_dir=(published_output_dir if staged_publication else None),
    )
    if (
        evidence_context_identity is not None
        and report_policy.get("evidence_context") != evidence_context_identity
    ):
        raise RuntimeError("report evidence context changed while loading")
    if existing_report_action(output_dir, report_policy) == "skip":
        if (
            evidence_context_identity is not None
            and file_identity(evidence_context_path) != evidence_context_identity
        ):
            raise RuntimeError("report evidence context changed while validating reuse")
        if build_capture_evidence_policy(
            root, build_capture_profile_coverage(config, root)
        ) != capture_evidence:
            raise RuntimeError("capture evidence changed while validating report reuse")
        return output_dir / "01_development_report.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report_policy.json", report_policy)
    run_scenes = discover_run_scenes(config, root)
    if not run_scenes:
        raise ValueError("no run artifacts found; configure runs[].existing or execute the experiment")
    auxiliary_profile_scenes = discover_auxiliary_profile_scenes(config, root)
    evidence_run_scenes = [*run_scenes, *auxiliary_profile_scenes]
    reproducibility_artifacts = build_reproducibility_artifact_index(
        config, root, evidence_run_scenes
    )
    frame_rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for run_scene in run_scenes:
        frames, passes, timings = load_instrumentation(run_scene)
        frame_rows.extend(frames)
        pass_rows.extend(passes)
        timing_rows.extend(timings)
    all_frames = pd.DataFrame(frame_rows)
    frames = select_terminal_frames(all_frames)
    passes = pd.DataFrame(pass_rows)
    timings = pd.DataFrame(timing_rows)
    performance = aggregate_timing_frames(timings)
    endpoint_performance = load_endpoint_runtime(config, root)
    map_catalog, signal_availability = build_map_catalog(evidence_run_scenes)
    schema4_tables = load_schema4_observability(evidence_run_scenes)
    exact_cost_evolution = build_exact_cost_evolution(map_catalog)
    resource_plan_validation = schema4_tables["resource_plan_validation"]
    resource_plan_validation_output = write_report_dataframe(
        resource_plan_validation, output_dir / "resource_plan_validation.parquet"
    )
    resource_plan_validation_json = output_dir / "resource_plan_validation.json"
    resource_validation_records = dataframe_json_records(resource_plan_validation)
    write_json(resource_plan_validation_json, {
        "schema_name": "openmvs.dmap.report_resource_plan_validation",
        "schema_version": 1,
        "valid": all(
            not bool(row.get("required")) or bool(row.get("valid"))
            for row in resource_validation_records
        ),
        "rows": resource_validation_records,
    })
    resource_plan_validation_output["json"] = str(resource_plan_validation_json)
    invalid_resource_plans = [
        row for row in resource_validation_records
        if bool(row.get("required")) and not bool(row.get("valid"))
    ]
    if invalid_resource_plans:
        labels = ", ".join(
            f"{row.get('run')}/{row.get('scene_id')}/{row.get('frame')}/{row.get('component')}"
            for row in invalid_resource_plans
        )
        raise ValueError(
            "required instrumentation resource-plan validation failed for "
            f"{labels}. See {resource_plan_validation_json}"
        )
    instrumentation_validation = validate_run_instrumentation(run_scenes)
    instrumentation_validation_output = write_report_dataframe(
        instrumentation_validation, output_dir / "instrumentation_validation.parquet"
    )
    validation_warning_records: list[dict[str, Any]] = []
    for row in instrumentation_validation.to_dict("records"):
        try:
            row_warnings = json.loads(str(row.get("validation_warnings") or "[]"))
        except json.JSONDecodeError:
            row_warnings = []
        for warning in row_warnings if isinstance(row_warnings, list) else []:
            if not isinstance(warning, dict):
                continue
            validation_warning_records.append({
                "run": row.get("run"),
                "role": row.get("role"),
                "repeat": row.get("repeat"),
                "scene_id": row.get("scene_id"),
                "frame": row.get("frame"),
                "image_id": row.get("image_id"),
                "estimation_stage": row.get("estimation_stage"),
                "geometric_iteration": row.get("geometric_iteration"),
                **warning,
            })
    terminal_validation = (
        instrumentation_validation[instrumentation_validation["terminal_stage"].astype(bool)]
        if not instrumentation_validation.empty and "terminal_stage" in instrumentation_validation.columns
        else pd.DataFrame()
    )
    endpoint_set_validation = terminal_validation
    if not endpoint_set_validation.empty and "endpoint_dmap_set_checked" in endpoint_set_validation:
        endpoint_set_validation = endpoint_set_validation[
            endpoint_set_validation["endpoint_dmap_set_checked"].fillna(False).astype(bool)
        ].drop_duplicates([
            column for column in ("run", "repeat", "scene_id")
            if column in endpoint_set_validation.columns
        ])
    validation_records = dataframe_json_records(instrumentation_validation)
    validation_summary = summarize_instrumentation_validation(validation_records)
    instrumentation_validation_output.update({
        "endpoint_dmap_sets_checked": int(len(endpoint_set_validation)),
        "endpoint_dmap_sets_bit_exact": int(
            endpoint_set_validation.get("endpoint_dmap_set_bit_exact", pd.Series(dtype=bool))
            .fillna(False).astype(bool).sum()
        ),
        "endpoint_dmaps_shared": int(
            pd.to_numeric(
                endpoint_set_validation.get("endpoint_dmap_set_shared_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "maps_summary_frames_checked": int(
            terminal_validation.get("maps_summary_parity_checked", pd.Series(dtype=bool))
            .fillna(False).astype(bool).sum()
        ),
        **validation_summary,
        "warning_count": len(validation_warning_records),
        "warning_frames": len({
            (
                row.get("run"), row.get("repeat"), row.get("scene_id"),
                row.get("estimation_stage"), row.get("geometric_iteration"), row.get("frame"),
            )
            for row in validation_warning_records
        }),
        "warnings": validation_warning_records,
    })
    instrumentation_validation_json = output_dir / "instrumentation_validation.json"
    write_json(instrumentation_validation_json, {
        "schema_name": "openmvs.dmap.report_frame_validation",
        "schema_version": 1,
        **validation_summary,
        "warnings": validation_warning_records,
        "rows": validation_records,
    })
    instrumentation_validation_output["json"] = str(instrumentation_validation_json)
    invalid_frames = [row for row in validation_records if not row.get("valid")]
    fatal_frames = [
        row for row in invalid_frames
        if not bool(row.get("report_generation_allowed"))
    ]
    if fatal_frames:
        labels = ", ".join(
            f"{row.get('run')}/{row.get('scene_id')}/{row.get('frame')}"
            for row in fatal_frames
        )
        raise ValueError(
            "instrumentation validation failed; report generation stopped for "
            f"{labels}. See {instrumentation_validation_json}"
        )
    metric_frames = production_quality_metric_rows(
        frames, config, instrumentation_validation
    )
    metric_performance = production_quality_metric_rows(
        endpoint_performance, config, instrumentation_validation
    )
    metric_labels = configured_run_labels(config)
    terminal_diagnostic_run_scenes = select_terminal_run_scenes(run_scenes)
    eligible_scene_identities = {
        (str(row["run"]), int(row["repeat"]), str(row["scene_id"]))
        for row in instrumentation_validation.to_dict("records")
        if bool(row.get("terminal_stage"))
        and not bool(row.get("diagnostic_only"))
        and bool(row.get("quality_comparison_eligible"))
    }
    endpoint_run_scenes = [
        run_scene for run_scene in terminal_diagnostic_run_scenes
        if run_scene.label in metric_labels
        and (run_scene.label, run_scene.repeat, run_scene.scene_id)
        in eligible_scene_identities
    ]
    product_reference_scenes, product_reference_frames = product_reference_annotation_inputs(
        config, metric_frames.to_dict("records")
    )
    annotation_rows = evaluate_annotations(
        config,
        [*endpoint_run_scenes, *product_reference_scenes],
        [*metric_frames.to_dict("records"), *product_reference_frames],
        output_dir,
    )
    annotations = pd.DataFrame(annotation_rows)
    comparisons, gates = build_comparisons(
        metric_frames, annotations, metric_performance, config.get("runs") or []
    )
    accuracy_policy = (config.get("evaluation") or {}).get("accuracy_first") or {}
    accuracy_ledger, accuracy_evidence = build_accuracy_first_ledger(
        annotations,
        metric_frames,
        metric_performance,
        config.get("runs") or [],
        coverage_floor=float(accuracy_policy.get("coverage_advisory_floor", 0.05)),
        strict_candidate_gate=(
            (config.get("decision_policy") or {}).get("strict_candidate_gate")
        ),
    )
    pareto = pareto_summary(gates)
    findings = deterministic_findings(gates)
    stability = pd.DataFrame()
    stability_required = {"run", "stage", "repeat", "annotation_kind", "object_id", "chunk_id"}
    if not annotations.empty and stability_required.issubset(annotations.columns):
        baseline_label = next(str(run["label"]) for run in config.get("runs") or [] if run.get("role") == "baseline")
        baseline = annotations[(annotations["run"] == baseline_label) & (annotations["stage"] == "post_filter")]
        stability_rows = []
        # Repeat is part of model identity; omitting it creates Cartesian joins
        # when baseline and candidate both contain repeated captures.
        keys = ["scene_id", "image_id", "repeat", "annotation_kind", "object_id", "chunk_id", "stage"]
        for run in config.get("runs") or []:
            label = str(run["label"])
            if label == baseline_label:
                continue
            candidate = annotations[(annotations["run"] == label) & (annotations["stage"] == "post_filter")]
            table = add_model_stability(baseline, candidate, keys)
            if not table.empty:
                table["candidate"] = label
                stability_rows.append(table)
        if stability_rows:
            stability = pd.concat(stability_rows, ignore_index=True)
            stability = annotate_model_switch_regression_proximity(
                stability, accuracy_evidence
            )
    map_catalog_output = write_report_dataframe(map_catalog, output_dir / "map_catalog.parquet")
    map_catalog_output["existing_rows"] = int(map_catalog["exists"].sum()) if not map_catalog.empty else 0
    signal_availability_output = write_report_dataframe(signal_availability, output_dir / "signal_availability.parquet")
    signal_availability_output["available_rows"] = int(signal_availability["available"].sum()) if not signal_availability.empty else 0
    signal_availability_output["unavailable_rows"] = int((~signal_availability["available"]).sum()) if not signal_availability.empty else 0
    signal_availability_json = output_dir / "signal_availability.json"
    write_json(signal_availability_json, {
        "schema_name": "openmvs.dmap.signal_availability",
        "schema_version": 2,
        "required_signals": list(REQUIRED_LOGICAL_STATE_SIGNALS),
        "schema4_exact_signals": {
            "logical_state": list(SCHEMA4_EXACT_STATE_SIGNALS),
            "logical_event": list(SCHEMA4_EXACT_EVENT_SIGNALS),
            "logical_view_state": list(SCHEMA4_EXACT_VIEW_SIGNALS),
        },
        "optional_exact_signals": {
            "low_texture_update_logical_event": list(
                LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
            ),
        },
        "rows": dataframe_json_records(signal_availability),
    })
    signal_availability_output["json"] = str(signal_availability_json)
    texture_region_summary = region_metrics.compute_texture_stratification(
        dataframe_json_records(map_catalog),
        dmap_report_model.read_diagnostic_map,
    )
    texture_region_metrics = pd.DataFrame(texture_region_summary["rows"])
    if texture_region_metrics.empty and not len(texture_region_metrics.columns):
        texture_region_metrics = pd.DataFrame(columns=TEXTURE_REGION_METRIC_COLUMNS)
    texture_region_output = write_report_dataframe(
        texture_region_metrics,
        output_dir / "texture_region_metrics.parquet",
    )
    texture_region_json = output_dir / "texture_region_metrics.json"
    write_json(texture_region_json, texture_region_summary)
    texture_region_output["json"] = str(texture_region_json)
    accepted_gain_census = region_metrics.compute_low_texture_accepted_gain_census(
        dataframe_json_records(map_catalog),
        dmap_report_model.read_diagnostic_map,
        variance_max_by_run=(
            dmap_report_model.configured_low_texture_variance_max_by_run(config)
        ),
    )
    accepted_gain_rows = []
    for row in accepted_gain_census["rows"]:
        quantiles = row.get("gain_quantiles") or {}
        fractions = row.get("fractions_below") or {}
        accepted_gain_rows.append({
            **{key: value for key, value in row.items() if key not in {"gain_quantiles", "fractions_below"}},
            **{f"gain_{key}": value for key, value in quantiles.items()},
            **{
                "fraction_below_000025": fractions.get("0.00025"),
                "fraction_below_00005": fractions.get("0.0005"),
                "fraction_below_0001": fractions.get("0.001"),
            },
        })
    accepted_gain_output = write_report_dataframe(
        pd.DataFrame(accepted_gain_rows),
        output_dir / "low_texture_accepted_gain_census.parquet",
    )
    accepted_gain_json = output_dir / "low_texture_accepted_gain_census.json"
    write_json(accepted_gain_json, accepted_gain_census)
    accepted_gain_output["json"] = str(accepted_gain_json)
    low_texture_hysteresis = (
        dmap_report_model.build_low_texture_update_hysteresis_metrics(
            schema4_tables["exact_iterations"]
        )
    )
    low_texture_hysteresis_output = write_report_dataframe(
        pd.DataFrame(low_texture_hysteresis["rows"]),
        output_dir / "low_texture_update_hysteresis.parquet",
    )
    low_texture_hysteresis_json = output_dir / "low_texture_update_hysteresis.json"
    write_json(low_texture_hysteresis_json, low_texture_hysteresis)
    low_texture_hysteresis_output["json"] = str(low_texture_hysteresis_json)
    component_registry_model = component_registry.build_registry([
        *dataframe_json_records(map_catalog),
        *dataframe_json_records(signal_availability),
        {"signal": "reference_rgb"},
    ])
    registry_errors = component_registry.validate_registry(component_registry_model)
    if registry_errors:
        raise ValueError("invalid component registry: " + "; ".join(registry_errors))
    component_registry_path = output_dir / "component_registry.json"
    write_json(component_registry_path, component_registry_model)
    capture_profile_coverage_path = output_dir / "capture_profile_coverage.json"
    write_json(capture_profile_coverage_path, capture_profile_coverage)
    experiment_lock = read_json(root / "00_experiment_lock.json")
    environment_identity = experiment_lock.get("environment_manifest")
    if not isinstance(environment_identity, dict):
        raise RuntimeError("experiment lock does not bind an environment manifest")
    environment_path = root / ENVIRONMENT_MANIFEST_FILE
    if file_identity(environment_path) != environment_identity:
        raise RuntimeError("environment manifest changed after experiment prepare")
    validate_environment_manifest(read_json(environment_path))
    report_environment_path = output_dir / ENVIRONMENT_MANIFEST_FILE
    shutil.copy2(environment_path, report_environment_path)
    outputs = {
        "report_policy": str(output_dir / "report_policy.json"),
        "report_tree_closure": str(output_dir / integrity.REPORT_CLOSURE_FILE),
        "frames": write_report_dataframe(frames, output_dir / "frames.parquet"),
        "iterations": write_report_dataframe(passes, output_dir / "iterations.parquet"),
        "timings": write_report_dataframe(timings, output_dir / "timings.parquet"),
        "performance": write_report_dataframe(performance, output_dir / "performance.parquet"),
        "endpoint_performance": write_report_dataframe(
            endpoint_performance, output_dir / "endpoint_performance.parquet"
        ),
        "annotations": write_report_dataframe(annotations, output_dir / "annotations.parquet"),
        "comparisons": write_report_dataframe(comparisons, output_dir / "comparisons.parquet"),
        "accuracy_ledger": write_report_dataframe(accuracy_ledger, output_dir / "accuracy_ledger.parquet"),
        "accuracy_evidence": write_report_dataframe(accuracy_evidence, output_dir / "accuracy_evidence.parquet"),
        "model_stability": write_report_dataframe(stability, output_dir / "model_stability.parquet"),
        "map_catalog": map_catalog_output,
        "signal_availability": signal_availability_output,
        "texture_region_metrics": texture_region_output,
        "low_texture_accepted_gain_census": accepted_gain_output,
        "low_texture_update_hysteresis": low_texture_hysteresis_output,
        "component_registry": str(component_registry_path),
        "capture_profile_coverage": str(capture_profile_coverage_path),
        "environment_manifest": str(report_environment_path),
        "instrumentation_validation": instrumentation_validation_output,
        "exact_observability": write_report_dataframe(schema4_tables["exact_observability"], output_dir / "exact_observability.parquet"),
        "exact_cost_evolution": write_report_dataframe(exact_cost_evolution, output_dir / "exact_cost_evolution.parquet"),
        "exact_iterations": write_report_dataframe(schema4_tables["exact_iterations"], output_dir / "exact_iterations.parquet"),
        "exact_views": write_report_dataframe(schema4_tables["exact_views"], output_dir / "exact_views.parquet"),
        "cpu_view_candidates": write_report_dataframe(schema4_tables["cpu_view_candidates"], output_dir / "cpu_view_candidates.parquet"),
        "cpu_estimation_selection": write_report_dataframe(schema4_tables["cpu_estimation_selection"], output_dir / "cpu_estimation_selection.parquet"),
        "postprocess_filters": write_report_dataframe(schema4_tables["postprocess_filters"], output_dir / "postprocess_filters.parquet"),
        "confidence_adjustment": write_report_dataframe(schema4_tables["confidence_adjustment"], output_dir / "confidence_adjustment.parquet"),
        "cuda_resource_plans": write_report_dataframe(schema4_tables["cuda_resource_plans"], output_dir / "cuda_resource_plans.parquet"),
        "filter_resource_plans": write_report_dataframe(schema4_tables["filter_resource_plans"], output_dir / "filter_resource_plans.parquet"),
        "resource_plan_validation": resource_plan_validation_output,
        "reproducibility_artifacts": write_report_dataframe(reproducibility_artifacts, output_dir / "reproducibility_artifacts.parquet"),
    }
    write_json(output_dir / "gates.json", {"schema_version": SCHEMA_VERSION, "gates": gates, "pareto": pareto})
    write_json(output_dir / "findings.json", {"schema_version": SCHEMA_VERSION, "findings": findings})
    plots = report_plots(frames, annotations, stability, performance, gates, output_dir)
    plots.extend(accuracy_first_plots(accuracy_ledger, accuracy_evidence, output_dir))
    plots.extend(exact_cost_trajectory_plots(exact_cost_evolution, output_dir))
    if skip_diagnostics:
        instrumentation_plots: dict[str, list[tuple[str, Path]]] = {}
        diagnostic_panels: dict[str, list[instrumentation_report.DiagnosticPanel]] = {}
    else:
        instrumentation_plots, diagnostic_panels = generate_diagnostic_panels(
            terminal_diagnostic_run_scenes, output_dir
        )
    inventory_scenes: dict[str, Any] = {}
    for scene_id in sorted(set(instrumentation_plots) | set(diagnostic_panels)):
        scene_plots = instrumentation_plots.get(scene_id, [])
        scene_panels = diagnostic_panels.get(scene_id, [])
        inventory_scenes[scene_id] = {
            "plots": [
                {"title": title, "path": str(path), "mechanism": mechanism_for_plot(title)}
                for title, path in scene_plots
            ],
            "panels": [
                {"frame": panel.frame_key, "title": panel.title, "path": str(panel.path), "mechanism": panel.mechanism}
                for panel in scene_panels
            ],
        }
    inventory_path = output_dir / "report_inventory.json"
    geometry_artifacts = []
    for row in annotations.to_dict("records") if not annotations.empty else []:
        paths = {}
        for column in ("visual_overlay_svg", "visual_residual_histogram_svg"):
            raw_path = row.get(column)
            if raw_path and str(raw_path) != "nan":
                paths[column] = str(Path(str(raw_path)))
        if paths:
            geometry_artifacts.append({
                "run": row.get("run"), "scene_id": row.get("scene_id"), "image_id": row.get("image_id"),
                "annotation_kind": row.get("annotation_kind"), "chunk_id": row.get("chunk_id"), "stage": row.get("stage"),
                **paths,
            })
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "required_mechanisms": [MECHANISM_LABELS[name] for name in MECHANISM_ORDER],
        "overall_plots": [{"title": title, "path": str(path)} for title, path in plots],
        "scenes": inventory_scenes,
        "geometry_rows": int(len(annotations)),
        "geometry_available_rows": (
            int((annotations["fit_status"] == "ok").sum())
            if not annotations.empty and "fit_status" in annotations.columns else 0
        ),
        "geometry_unavailable_rows": (
            int((annotations["fit_status"] != "ok").sum())
            if not annotations.empty and "fit_status" in annotations.columns else 0
        ),
        "geometry_artifacts": geometry_artifacts,
        "timing_rows": int(len(performance)),
        "data_artifacts": [
            {"name": "report_tree_closure", "json": outputs["report_tree_closure"]},
            {"name": "map_catalog", **map_catalog_output},
            {"name": "signal_availability", **signal_availability_output},
            {"name": "texture_region_metrics", **texture_region_output},
            {"name": "low_texture_accepted_gain_census", **accepted_gain_output},
            {"name": "low_texture_update_hysteresis", **low_texture_hysteresis_output},
            {"name": "component_registry", "json": str(component_registry_path)},
            {"name": "capture_profile_coverage", "json": str(capture_profile_coverage_path)},
            {"name": "environment_manifest", "json": outputs["environment_manifest"]},
            {"name": "instrumentation_validation", **instrumentation_validation_output},
            {"name": "endpoint_performance", **outputs["endpoint_performance"]},
            {"name": "exact_observability", **outputs["exact_observability"]},
            {"name": "exact_cost_evolution", **outputs["exact_cost_evolution"]},
            {"name": "exact_iterations", **outputs["exact_iterations"]},
            {"name": "exact_views", **outputs["exact_views"]},
            {"name": "cpu_view_candidates", **outputs["cpu_view_candidates"]},
            {"name": "cpu_estimation_selection", **outputs["cpu_estimation_selection"]},
            {"name": "postprocess_filters", **outputs["postprocess_filters"]},
            {"name": "confidence_adjustment", **outputs["confidence_adjustment"]},
            {"name": "cuda_resource_plans", **outputs["cuda_resource_plans"]},
            {"name": "filter_resource_plans", **outputs["filter_resource_plans"]},
            {"name": "resource_plan_validation", **outputs["resource_plan_validation"]},
            {"name": "reproducibility_artifacts", **outputs["reproducibility_artifacts"]},
            {"name": "accuracy_ledger", **outputs["accuracy_ledger"]},
            {"name": "accuracy_evidence", **outputs["accuracy_evidence"]},
            {"name": "model_stability", **outputs["model_stability"]},
        ],
    }
    write_json(
        inventory_path,
        dmap_report_model.paths_report_relative(inventory, output_dir),
    )
    outputs["report_inventory"] = str(inventory_path)
    trace_path = generate_trace_manifest(config, frames, output_dir)
    outputs["gates"] = str(output_dir / "gates.json")
    outputs["findings"] = str(output_dir / "findings.json")
    outputs["trace_rerun"] = str(trace_path)
    report_model = dmap_report_model.build_report_model(
        config=config,
        experiment_root=root,
        output_dir=output_dir,
        run_scenes=run_scenes,
        frames=frames,
        iterations=passes,
        performance=performance,
        endpoint_performance=metric_performance,
        annotations=annotations,
        comparisons=comparisons,
        accuracy_ledger=accuracy_ledger,
        accuracy_evidence=accuracy_evidence,
        model_stability=stability,
        gates=gates,
        pareto=pareto,
        findings=findings,
        map_catalog=map_catalog,
        signal_availability=signal_availability,
        inventory=inventory,
        metric_specs=METRICS,
        exact_cost_evolution=exact_cost_evolution,
        exact_observability=schema4_tables["exact_observability"],
        exact_iterations=schema4_tables["exact_iterations"],
        exact_views=schema4_tables["exact_views"],
        cpu_view_candidates=schema4_tables["cpu_view_candidates"],
        cpu_estimation_selection=schema4_tables["cpu_estimation_selection"],
        postprocess_filters=schema4_tables["postprocess_filters"],
        confidence_adjustment=schema4_tables["confidence_adjustment"],
        cuda_resource_plans=schema4_tables["cuda_resource_plans"],
        filter_resource_plans=schema4_tables["filter_resource_plans"],
        resource_plan_validation=schema4_tables["resource_plan_validation"],
        instrumentation_validation=instrumentation_validation,
        summary_signal_contract={
            "schema_name": "openmvs.dmap.summary_unavailable_signals",
            "schema_version": 1,
            "logical_signals": sorted({
                *REQUIRED_LOGICAL_STATE_SIGNALS,
                *REQUIRED_LOGICAL_EVENT_SIGNALS,
                *REPORT_DERIVED_LOGICAL_EVENT_SIGNALS,
                *SCHEMA4_EXACT_STATE_SIGNALS,
                *SCHEMA4_EXACT_EVENT_SIGNALS,
                *SCHEMA4_EXACT_VIEW_SIGNALS,
                *MECHANISM_LOGICAL_STATE_SIGNALS,
            }),
            "initialization_signals": [],
            "final_signals": list(SUMMARY_PROFILE_FINAL_SIGNALS),
        },
        capture_profile_coverage=capture_profile_coverage,
        evidence_context=evidence_context,
    )
    report_model["report_policy"] = dmap_report_model.paths_report_relative(
        report_policy, output_dir
    )
    report_model_path = output_dir / "report_model.json"
    dmap_report_model.write_report_model(report_model_path, report_model)
    investigation_path = output_dir / "02_investigation.html"
    ui_root = REPO_ROOT / "scripts" / "dmap_report_ui"
    dmap_report_model.render_investigation_html(
        investigation_path,
        report_model,
        ui_root / "investigation.html",
        ui_root / "investigation.css",
        ui_root / "investigation.js",
    )
    outputs["report_model"] = str(report_model_path)
    outputs["investigation_html"] = str(investigation_path)
    report_model_validation_path = output_dir / "report_model.validation.json"
    outputs["report_model_validation"] = str(report_model_validation_path)
    report_path = output_dir / "01_development_report.md"
    report_path.write_text(
        build_markdown(
            config, report_path, frames, passes, annotations, stability, performance, comparisons,
            gates, pareto, findings, plots, instrumentation_plots, diagnostic_panels, outputs,
            exact_cost_evolution, schema4_tables["exact_iterations"], schema4_tables["exact_views"],
            schema4_tables["cpu_view_candidates"], schema4_tables["cpu_estimation_selection"],
            schema4_tables["postprocess_filters"], schema4_tables["confidence_adjustment"],
            schema4_tables["cuda_resource_plans"], schema4_tables["filter_resource_plans"],
            schema4_tables["resource_plan_validation"],
            reproducibility_artifacts,
            report_model.get("drilldowns"),
            accuracy_ledger=accuracy_ledger,
            accuracy_evidence=accuracy_evidence,
            evidence_context=evidence_context,
            evidence_context_path=evidence_context_path,
            published_report_dir=published_output_dir,
            low_texture_hysteresis=report_model.get("mechanics", {}).get(
                "low_texture_update_hysteresis"
            ),
            accepted_gain_census=report_model.get("mechanics", {}).get(
                "accepted_gain_census"
            ),
        ),
        encoding="utf-8",
    )
    html_path = output_dir / "01_development_report.html"
    render_html(report_path, html_path)
    report_model_validation = dmap_report_model.validate_report_model(report_model, output_dir)
    write_json(report_model_validation_path, report_model_validation)
    if not report_model_validation["valid"]:
        raise RuntimeError(
            f"generated structured report model failed validation: {report_model_validation_path}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_config": config["_config_path"],
        "experiment_root": str(root),
        "published_output_dir": str(published_output_dir),
        "report_md": str(published_output_dir / report_path.name),
        "report_html": str(published_output_dir / html_path.name),
        "report_model": str(published_output_dir / report_model_path.name),
        "investigation_html": str(published_output_dir / investigation_path.name),
        "report_policy": report_policy,
        "report_policy_path": str(published_output_dir / "report_policy.json"),
        "report_tree_closure": str(
            published_output_dir / integrity.REPORT_CLOSURE_FILE
        ),
        "outputs": outputs,
        "plots": [{"title": title, "path": str(path)} for title, path in plots],
        "diagnostic_panels": {scene: [str(panel.path) for panel in panels] for scene, panels in diagnostic_panels.items()},
        "instrumentation_plots": {scene: [{"title": title, "path": str(path), "mechanism": mechanism_for_plot(title)} for title, path in scene_plots] for scene, scene_plots in instrumentation_plots.items()},
        "run_scenes": manifest_run_scene_rows(run_scenes),
    }
    if staged_publication:
        manifest = rebase_report_output_paths(
            manifest, output_dir, published_output_dir
        )
    write_json(output_dir / "report_manifest.json", manifest)
    if (
        evidence_context_identity is not None
        and file_identity(evidence_context_path) != evidence_context_identity
    ):
        raise RuntimeError("report evidence context changed during report generation")
    if build_capture_evidence_policy(
        root, build_capture_profile_coverage(config, root)
    ) != capture_evidence:
        raise RuntimeError("capture evidence changed during report generation")
    integrity.write_report_tree_closure(output_dir)
    validation = validate_report(report_path)
    if not validation["valid"]:
        raise RuntimeError(f"generated report validation failed: {validation}")
    return report_path


def main() -> int:
    command = tyro.cli(Command, args=normalize_repeated_drilldown_args(sys.argv[1:]))
    try:
        if isinstance(command, PrepareCommand):
            _config, root = prepare_experiment(command.config, command.allow_over_budget)
            print(root)
            return 0
        if isinstance(command, RunCommand):
            config, root = prepare_experiment(
                command.config, command.allow_over_budget, command.profile
            )
            run_experiment(
                config,
                root,
                command.dry_run,
                command.profile,
                command.allow_over_budget,
            )
            if not command.skip_report and not command.dry_run:
                print(build_report(config, root, root / "reports", False))
            return 0
        if isinstance(command, ReportCommand):
            config, root = prepare_experiment(command.config, True)
            activation_source = "effective_config"
            if command.allow_process_specialization_divergence_for_diagnostics:
                config = dict(config)
                instrumentation = dict(config.get("instrumentation") or {})
                instrumentation[
                    "allow_process_specialization_divergence_for_diagnostics"
                ] = True
                config["instrumentation"] = instrumentation
                activation_source = "explicit_report_cli"
            output = command.output_dir.expanduser().resolve() if command.output_dir else root / "reports"
            published_output = (
                command.published_output_dir.expanduser().absolute()
                if command.published_output_dir is not None else None
            )
            print(build_report(
                config,
                root,
                output,
                command.skip_diagnostics,
                report_policy_activation_source=activation_source,
                evidence_context_path=command.evidence_context,
                published_output_dir=published_output,
            ))
            return 0
        if isinstance(command, ArrayStoreCommand):
            config, root = prepare_experiment(command.config, True)
            result = array_store_workflow.materialize_experiment_array_stores(
                experiment_root=root,
                source_config=Path(str(config["_config_path"])),
                run_scenes=discover_run_scenes(config, root),
                runs=command.run,
                scenes=command.scene,
                frames=command.frame,
                chunk_size=command.chunk_size,
                shard_size=command.shard_size,
                zstd_level=command.zstd_level,
                allow_incomplete=command.allow_incomplete,
                max_uncompressed_bytes_per_store=command.max_uncompressed_bytes_per_store,
                verify_data=command.verify_data,
            )
            print(json.dumps(result.to_dict(), indent=2))
            return 0
        if isinstance(command, ValidateCommand):
            result = validate_report(command.report, write_sidecar=False)
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 1
        if isinstance(command, TraceRerunCommand):
            config, root = prepare_experiment(command.config, True)
            report_dir = root / "reports"
            frames_path = report_dir / "frames.csv"
            if not frames_path.is_file():
                build_report(config, root, report_dir, True)
            path = generate_trace_manifest(config, pd.read_csv(frames_path), report_dir)
            if command.execute:
                executions = execute_trace_reruns(
                    config, root, path, command.allow_over_budget
                )
                print(json.dumps({"manifest": str(path), "executions": executions}, indent=2))
                return 0
            print(path)
            return 0
        if isinstance(command, DrilldownCommand):
            if command.refresh_report and not command.execute:
                raise ValueError("--refresh-report requires --execute")
            if command.refresh_report and command.report_dir is None:
                raise ValueError("--refresh-report requires --report-dir")
            config, root = prepare_experiment(command.config, True)
            resolved_scenes = {
                str(scene["scan_id"]): scene
                for scene in resolve_scenes(config, resolve_suite(config))
            }
            selected_scene = resolved_scenes.get(command.scene)
            if selected_scene is None:
                raise ValueError(
                    f"scene {command.scene!r} is not selected by this experiment config"
                )
            selected_mvs = Path(str(selected_scene["mvs_file"])).expanduser().resolve()
            dmap_drilldown.validate_no_implicit_program_options_file(
                selected_mvs.parent
            )
            request = dmap_drilldown.build_request(
                config=config,
                config_path=Path(str(config["_config_path"])),
                scene_id=command.scene,
                image_id=command.frame,
                pixel_values=command.pixel,
                roi_value=command.roi,
                variants=command.variant,
                source_revision=git_hash(),
                source_dirty=git_dirty(),
                argument_overrides=selected_scene.get("argument_overrides"),
            )
            path = dmap_drilldown.write_immutable_request(root / "drilldowns", request)
            index_path = refresh_drilldown_index(root)
            output: dict[str, Any] = {
                "request": str(path),
                "request_sha256": request["request_sha256"],
                "capture_profile": request["capture_profile"],
                "target": request["target"],
                "runs": [run["label"] for run in request["runs"]],
                "index": str(index_path),
                "executed": False,
            }
            if command.execute:
                output["executions"] = execute_drilldown_request(
                    config, root, path, command.allow_over_budget
                )
                output["executed"] = True
                if command.refresh_report:
                    assert command.report_dir is not None
                    refresh_master_report(command.config, command.report_dir)
                    output["report"] = str(command.report_dir.expanduser().resolve())
            print(json.dumps(output, indent=2))
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
