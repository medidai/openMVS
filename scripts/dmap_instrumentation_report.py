#!/usr/bin/env python3
"""Generate a rich Markdown or HTML report for DensifyPointCloud depth-map instrumentation."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shlex
import shutil
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote


PNG_MAP_NAMES = (
    "cost_final.png",
    "candidate_source.png",
    "num_supporting_views.png",
    "last_changed_iter.png",
    "rejection_reason.png",
    "valid_before_filter.png",
    "valid_after_filter.png",
    "selected_view_count.png",
    "accepted_update_count.png",
)

PFM_MAP_NAMES = (
    "cost_final.pfm",
    "depth_final_after_filter.pfm",
    "normal_final.pfm",
    "depth_final_before_filter.pfm",
    "normal_final_before_filter.pfm",
    "cost_final_before_filter.pfm",
    "cost_photometric.pfm",
    "cost_photo_prior.pfm",
    "cost_geometric.pfm",
    "cost_total_components.pfm",
    "cost_depth_prior.pfm",
    "depth_prior_weight.pfm",
    "confidence_gap.pfm",
    "reference_variance.pfm",
    "view_entropy.pfm",
    "low_depth_prior.pfm",
    *(f"view_weight_{index}.pfm" for index in range(4)),
    *(f"view_cost_{index}.pfm" for index in range(4)),
    *(f"view_photometric_cost_{index}.pfm" for index in range(4)),
    *(f"view_geometric_cost_{index}.pfm" for index in range(4)),
)

REJECTION_KEYS = (
    "low_score",
    "insufficient_view_support",
    "geometric_inconsistency",
    "normal_inconsistency",
    "depth_range",
    "occlusion",
    "masked",
    "small_component",
    "unknown",
)

PLOT_COLORS = ("#355C7D", "#F67280", "#2A9D8F", "#F4A261", "#6C5B7B", "#C06C84", "#457B9D", "#E9C46A")

LOGICAL_COST_SIGNAL_PRESENTATION: dict[str, tuple[str, str, str]] = {
    "cost_stored": ("Stored production cost", "exact", "cost"),
    "confidence_stored": ("Production confidence from stored cost", "derived", "unit"),
    "cost_photo_raw_equal_selected_rescore_proxy": ("Raw photometric rescore", "proxy", "cost"),
    "cost_photo_prior_equal_selected_rescore_proxy": ("Photometric + prior rescore", "proxy", "cost"),
    "cost_geometric_equal_selected_rescore_proxy": ("Geometric rescore", "proxy", "cost"),
    "cost_total_equal_selected_rescore_proxy": ("Total equal-selected rescore", "proxy", "cost"),
    "cost_stored_minus_rescore": ("Stored minus rescore", "derived", "residual"),
    "depth_prior_disagreement_equal_selected_rescore_proxy": ("Depth-prior disagreement", "proxy", "nonnegative"),
    "depth_prior_weight_equal_selected_rescore_proxy": ("Depth-prior weight", "proxy", "unit"),
    "gap_local_neighbor_equal_selected_rescore_proxy": ("Local-neighbor confidence gap", "proxy", "nonnegative"),
    "reference_variance_equal_selected_rescore_proxy": ("Reference-patch variance", "proxy", "nonnegative"),
}
LOGICAL_EVENT_SIGNAL_IDS = {"depth_delta", "depth_relative_delta", "normal_angle_delta", "view_churn"}

MARKDOWN_REPORT_STYLE = """<style>
details.scene-details { margin: 24px 0; border: 1px solid #aebfcd; border-radius: 7px; background: #fff; }
details.scene-details > summary { cursor: pointer; padding: 12px 15px; background: #dce9f2; color: #20384b; font-size: 1.2rem; font-weight: 700; }
details.scene-details[open] > summary { border-bottom: 1px solid #aebfcd; }
details.frame-details { margin: 9px 0; border: 1px solid #c7d5df; border-radius: 6px; background: #fff; }
details.frame-details > summary { cursor: pointer; padding: 8px 11px; background: #edf4f8; color: #294861; font-weight: 650; }
details.frame-details[open] > summary { background: #dfeef6; border-bottom: 1px solid #c7d5df; }
details > summary::marker { color: #4f7f9f; }
</style>"""


@dataclass(frozen=True)
class MapArtifact:
    signal: str
    path: Path
    dtype: str = ""
    role: str = ""
    semantics: str = ""
    logical_iteration: int | None = None
    pyramid_level: int | None = None
    stage: str = ""
    pass_index: int | None = None
    fidelity: str = "unspecified"
    measurement_quality: str = ""
    measurement_basis: str = ""
    proxy_target: str = ""
    limitations: str = ""
    temporal_scope: str = "unspecified"
    bytes: int | None = None
    schema_version: int = 0
    stage_index: int | None = None
    source_view_index: int | None = None
    source_image_id: int | None = None
    source_image_name: str = ""
    contribution_basis: str = ""
    channels: dict[str, Any] | list[Any] | None = None
    encoding: str = ""
    unavailable_value: float | int | None = None
    gap_scope: str = ""
    algorithm_stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthMapRecord:
    directory: Path
    summary: dict[str, Any] = field(default_factory=dict)
    filtering: dict[str, Any] = field(default_factory=dict)
    iterations: list[dict[str, str]] = field(default_factory=list)
    view_support: list[dict[str, str]] = field(default_factory=list)
    png_maps: dict[str, Path] = field(default_factory=dict)
    pfm_maps: dict[str, Path] = field(default_factory=dict)
    preview_maps: dict[str, Path] = field(default_factory=dict)
    improvement_maps: dict[int, Path] = field(default_factory=dict)
    pass_maps: dict[str, dict[int, Path]] = field(default_factory=dict)
    map_manifest: dict[str, Any] = field(default_factory=dict)
    map_artifacts: list[MapArtifact] = field(default_factory=list)
    logical_state_maps: dict[str, dict[int, MapArtifact]] = field(default_factory=dict)
    logical_event_maps: dict[str, dict[int, MapArtifact]] = field(default_factory=dict)
    report_files: dict[str, Path] = field(default_factory=dict)
    reference_image: Path | None = None
    reference_thumbnail: Path | None = None
    panel_path: Path | None = None
    cost_panel_path: Path | None = None
    logical_cost_panel_path: Path | None = None
    view_panel_path: Path | None = None
    improvement_panel_path: Path | None = None
    pass_panel_paths: dict[str, Path] = field(default_factory=dict)
    improvement_vmax: float | None = None


@dataclass
class RunData:
    label: str
    path: Path
    run_metadata: dict[str, Any] = field(default_factory=dict)
    scene_summary: dict[str, Any] = field(default_factory=dict)
    depthmaps: list[DepthMapRecord] = field(default_factory=list)
    counters: list[dict[str, str]] = field(default_factory=list)
    timings: list[dict[str, str]] = field(default_factory=list)
    trace_count: int = 0
    report_files: dict[str, Path] = field(default_factory=dict)


@dataclass
class DiagnosticPanel:
    frame_key: str
    title: str
    path: Path
    records: list[tuple[RunData, DepthMapRecord]] = field(default_factory=list)
    mechanism: str = "overview"
    caption: str = ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def to_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isfinite(value):
        return value
    return default


def to_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt_float(value: Any, digits: int = 4, default: str = "") -> str:
    value = to_float(value)
    if value is None:
        return default
    return f"{value:.{digits}f}"


def fmt_pct(value: Any, digits: int = 2, default: str = "") -> str:
    value = to_float(value)
    if value is None:
        return default
    return f"{value * 100.0:.{digits}f}%"


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE = "unavailable_post_pass_snapshot"
CANDIDATE_ACCOUNTING_METRICS = (
    "changed_ratio",
    "candidates_tested",
    "candidates_finite",
    "candidates_accepted",
    "tested_candidates",
    "finite_candidates",
    "accepted_candidates",
    "acceptance_rate",
)


def candidate_accounting_available(value: DepthMapRecord | dict[str, Any] | str | None) -> bool:
    if isinstance(value, DepthMapRecord):
        mode = value.summary.get("candidate_accounting_mode")
    elif isinstance(value, dict):
        mode = value.get("candidate_accounting_mode")
    else:
        mode = value
    return str(mode or "") != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE


def apply_candidate_accounting_contract(
    row: dict[str, Any], mode: str | None = None,
) -> dict[str, Any]:
    """Carry accounting provenance and null metrics that the capture cannot observe."""

    result = dict(row)
    accounting_mode = str(mode or result.get("candidate_accounting_mode") or "")
    if accounting_mode:
        result["candidate_accounting_mode"] = accounting_mode
    if accounting_mode != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE:
        return result
    unavailable_keys = set(CANDIDATE_ACCOUNTING_METRICS)
    unavailable_keys.update(
        key for key in result
        if key.startswith("accepted_from_")
        or (
            key.startswith("candidate_")
            and key.endswith(("_tested", "_finite", "_accepted", "_acceptance_rate"))
        )
    )
    for key in unavailable_keys:
        result[key] = None
    return result


def logical_iteration_index(row: dict[str, Any]) -> int:
    """Map raw CUDA checkerboard phases to one user-facing PatchMatch iteration."""
    phase = str(row.get("phase", "")).strip().lower()
    iteration = to_int(row.get("iteration"), -1)
    if phase in {"init", "initialization"}:
        return -1
    if iteration >= 0:
        return iteration
    pass_index = to_int(row.get("pass_index"), -1)
    return -1 if pass_index <= 0 else (pass_index - 1) // 2


def logical_iteration_label(iteration: int) -> str:
    return "initialization" if iteration < 0 else f"iteration {iteration + 1}"


def aggregate_logical_iteration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse black/red rows while preserving state and update-event semantics."""
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scale = to_int(row.get("scale_level", row.get("scale_number", 0)), 0)
        grouped[(scale, logical_iteration_index(row))].append(row)

    count_keys = (
        "candidates_tested",
        "candidates_finite",
        "candidates_accepted",
        "accepted_from_init",
        "accepted_from_spatial_propagation",
        "accepted_from_view_propagation",
        "accepted_from_random_perturbation",
        "accepted_from_refinement",
        "accepted_from_prior_or_guidance",
    )
    result: list[dict[str, Any]] = []
    for (scale, iteration), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: to_int(row.get("pass_index"), -1))
        final = dict(ordered[-1])
        final["scale_level"] = scale
        final["iteration"] = iteration
        final["logical_iteration"] = iteration
        final["stage"] = logical_iteration_label(iteration)
        final["phase"] = "initialization" if iteration < 0 else "iteration"
        final["pass_index"] = 0 if iteration < 0 else iteration + 1
        accounting_mode = (
            CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
            if any(
                str(row.get("candidate_accounting_mode") or "")
                == CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
                for row in ordered
            )
            else str(final.get("candidate_accounting_mode") or "")
        )
        if accounting_mode:
            final["candidate_accounting_mode"] = accounting_mode
        if len(ordered) == 1 and ordered[0].get("raw_pass_indices") not in (None, ""):
            final["raw_pass_indices"] = ordered[0]["raw_pass_indices"]
        else:
            final["raw_pass_indices"] = ",".join(str(to_int(row.get("pass_index"), -1)) for row in ordered)
        if (
            iteration >= 0
            and len(ordered) > 1
            and accounting_mode != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
        ):
            changed = [to_float(row.get("changed_ratio"), 0.0) or 0.0 for row in ordered]
            final["changed_ratio"] = min(1.0, sum(changed))
            final["mean_cost_delta"] = sum(to_float(row.get("mean_cost_delta"), 0.0) or 0.0 for row in ordered)
            changed_total = sum(changed)
            for key in ("mean_abs_depth_delta", "mean_normal_delta_deg"):
                weighted = sum((to_float(row.get(key), 0.0) or 0.0) * weight for row, weight in zip(ordered, changed))
                final[key] = safe_div(weighted, changed_total)
            for key in count_keys:
                final[key] = sum(to_int(row.get(key), 0) for row in ordered)
            tested = to_int(final.get("candidates_tested"), 0)
            accepted = to_int(final.get("candidates_accepted"), 0)
            final["acceptance_rate"] = safe_div(accepted, tested)
        result.append(apply_candidate_accounting_contract(final, accounting_mode))
    return result


def record_logical_iterations(record: DepthMapRecord) -> list[dict[str, Any]]:
    rows = aggregate_logical_iteration_rows(record.iterations)
    scale = to_int(record.summary.get("scale_level"), 0)
    selected = [row for row in rows if to_int(row.get("scale_level"), scale) == scale]
    return selected or rows


def combine_checkerboard_maps(data_by_pass: dict[int, Any], np) -> dict[int, Any]:
    """Combine complementary black/red pixel maps into logical-iteration maps."""
    combined: dict[int, Any] = {}
    for pass_index, data in sorted(data_by_pass.items()):
        iteration = -1 if pass_index == 0 else (pass_index - 1) // 2
        values = np.asarray(data, dtype=float)
        if iteration not in combined:
            combined[iteration] = values.copy()
        else:
            combined[iteration] = combined[iteration] + values
    return combined


def frame_label(record: DepthMapRecord) -> str:
    summary = record.summary
    image_id = summary.get("image_id", "")
    safe_name = summary.get("safe_image_name", record.directory.name)
    return f"{image_id}:{safe_name}" if image_id != "" else str(safe_name)


def image_id_key(record: DepthMapRecord) -> str:
    return str(record.summary.get("image_id", record.directory.name))


def resolve_reference_image(run_path: Path, summary: dict[str, Any]) -> Path | None:
    raw_name = str(summary.get("image_name", "")).strip()
    if not raw_name:
        return None
    raw_path = Path(raw_name).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(run_path / raw_path)
    candidates.extend(
        [
            run_path.parent / "work" / "images" / raw_path.name,
            run_path / "images" / raw_path.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def signal_fidelity(signal: str, entry: dict[str, Any] | None = None) -> str:
    presentation = LOGICAL_COST_SIGNAL_PRESENTATION.get(signal)
    if presentation is not None:
        return presentation[1]
    entry = entry or {}
    for key in ("measurement_quality", "fidelity", "measurement_kind", "exactness"):
        value = str(entry.get(key, "")).strip().lower()
        if value in {"exact", "proxy"}:
            return value
        if value in {"derived", "derived_exact"}:
            return "derived"
    return "unspecified"


def artifact_iteration(entry: dict[str, Any]) -> int | None:
    value = entry.get("logical_iteration")
    if value not in (None, ""):
        return to_int(value)
    stage = str(entry.get("stage", "")).strip().lower()
    if stage in {"init", "initialization"}:
        return -1
    return None


def artifact_pyramid_level(entry: dict[str, Any], manifest: dict[str, Any]) -> int | None:
    """Read the canonical pyramid level while accepting legacy capture aliases."""

    for source in (entry, manifest):
        for key in ("pyramid_level", "scale_level", "scale_number"):
            value = source.get(key)
            if value in (None, ""):
                continue
            level = to_int(value, -1)
            return level if level >= 0 else None
    return None


def owned_map_artifact_path(depthmap_dir: Path, raw_path: str) -> Path:
    """Return a lexical map path only when every existing component is owned."""

    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe map artifact path: {raw_path!r}")
    owner = depthmap_dir.resolve()
    candidate = owner / relative
    current = owner
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return candidate
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"map artifact path contains a symlink: {raw_path}")
    try:
        candidate.resolve(strict=False).relative_to(owner)
    except ValueError as exc:
        raise ValueError(f"map artifact escapes its frame directory: {raw_path}") from exc
    if candidate.exists() and not stat.S_ISREG(candidate.lstat().st_mode):
        raise ValueError(f"map artifact is not a regular file: {raw_path}")
    return candidate


def parse_map_artifacts(depthmap_dir: Path, manifest: dict[str, Any]) -> list[MapArtifact]:
    schema_version = to_int(manifest.get("schema_version"), 0)
    artifacts: list[MapArtifact] = []
    for entry in manifest.get("maps") or []:
        if not isinstance(entry, dict):
            continue
        signal = str(entry.get("signal", "")).strip()
        raw_path = str(entry.get("path", "")).strip()
        if not signal or not raw_path:
            continue
        path = owned_map_artifact_path(depthmap_dir, raw_path)
        logical_iteration = artifact_iteration(entry)
        pass_value = entry.get("pass_index")
        pass_index = to_int(pass_value) if pass_value not in (None, "") else None
        temporal_scope = str(entry.get("temporal_scope", entry.get("scope", ""))).strip().lower()
        if logical_iteration is not None:
            temporal_scope = "logical_event" if signal in LOGICAL_EVENT_SIGNAL_IDS else "logical_state"
        elif not temporal_scope:
            temporal_scope = "pass_event" if pass_index is not None else "final_state"
        artifacts.append(
            MapArtifact(
                signal=signal,
                path=path,
                dtype=str(entry.get("dtype", "")),
                role=str(entry.get("role", "")),
                semantics=str(entry.get("semantics", "")),
                logical_iteration=logical_iteration,
                pyramid_level=artifact_pyramid_level(entry, manifest),
                stage=str(entry.get("stage", "")),
                pass_index=pass_index,
                fidelity=signal_fidelity(signal, entry),
                measurement_quality=str(entry.get("measurement_quality", entry.get("quality", ""))),
                measurement_basis=str(entry.get("measurement_basis", "")),
                proxy_target=str(entry.get("proxy_target", "")),
                limitations=str(entry.get("limitations", "")),
                temporal_scope=temporal_scope,
                bytes=to_int(entry.get("bytes")) if entry.get("bytes") not in (None, "") else None,
                schema_version=schema_version,
                stage_index=to_int(entry.get("stage_index")) if entry.get("stage_index") not in (None, "") else None,
                source_view_index=to_int(entry.get("source_view_index")) if entry.get("source_view_index") not in (None, "") else None,
                source_image_id=to_int(entry.get("source_image_id")) if entry.get("source_image_id") not in (None, "") else None,
                source_image_name=str(entry.get("source_image_name", "")),
                contribution_basis=str(entry.get("contribution_basis", "")),
                channels=entry.get("channels_memory_order", entry.get("channels")),
                encoding=str(entry.get("encoding", entry.get("integer_encoding", ""))),
                unavailable_value=entry.get("unavailable_value"),
                gap_scope=str(entry.get("gap_scope", "")),
                algorithm_stage=str(entry.get("algorithm_stage", "")),
                metadata=dict(entry),
            )
        )
    return artifacts


def register_map_artifact(record: DepthMapRecord, artifact: MapArtifact) -> None:
    record.map_artifacts.append(artifact)
    if artifact.logical_iteration is not None:
        target = record.logical_event_maps if (
            artifact.role == "logical_event" or artifact.signal in LOGICAL_EVENT_SIGNAL_IDS
        ) else record.logical_state_maps
        target.setdefault(artifact.signal, {}).setdefault(artifact.logical_iteration, artifact)


def register_legacy_file(record: DepthMapRecord, path: Path, signal: str, pass_index: int | None = None) -> None:
    key = (signal, None, pass_index, path.resolve())
    existing = {
        (artifact.signal, artifact.logical_iteration, artifact.pass_index, artifact.path)
        for artifact in record.map_artifacts
    }
    if key in existing:
        return
    register_map_artifact(
        record,
        MapArtifact(
            signal=signal,
            path=path.resolve(),
            dtype="float32" if path.suffix.lower() == ".pfm" else "uint8",
            pass_index=pass_index,
            fidelity="unspecified",
            temporal_scope="pass_event" if pass_index is not None else "final_state",
            schema_version=to_int(record.map_manifest.get("schema_version"), 0),
        ),
    )


def load_run(label: str, path: Path) -> RunData:
    run = RunData(label=label, path=path)
    run.run_metadata = read_json(path / "run_metadata.json")
    run.scene_summary = read_json(path / "scene_summary.json")
    run.counters = read_csv(path / "instrumentation" / "counters.csv")
    run.timings = read_csv(path / "instrumentation" / "timings.csv")
    run.trace_count = count_lines(path / "instrumentation" / "traces.jsonl")
    for depthmap_dir in sorted((path / "depthmaps").glob("*")):
        if not depthmap_dir.is_dir():
            continue
        summary = read_json(depthmap_dir / "summary.json")
        iterations = read_csv(depthmap_dir / "iteration.csv")
        accounting_mode = str(summary.get("candidate_accounting_mode", "legacy_or_exact"))
        iterations = [
            apply_candidate_accounting_contract(row, accounting_mode)
            for row in iterations
        ]
        record = DepthMapRecord(
            directory=depthmap_dir,
            summary=summary,
            filtering=read_json(depthmap_dir / "filtering.json"),
            iterations=iterations,
            view_support=read_csv(depthmap_dir / "view_support.csv"),
        )
        record.reference_image = resolve_reference_image(path, summary)
        record.map_manifest = read_json(depthmap_dir / "map_manifest.json")
        for artifact in parse_map_artifacts(depthmap_dir, record.map_manifest):
            register_map_artifact(record, artifact)
            if artifact.logical_iteration is None and artifact.pass_index is None:
                if artifact.path.name in PNG_MAP_NAMES and artifact.path.is_file():
                    record.png_maps[artifact.path.name] = artifact.path
                if artifact.path.name in PFM_MAP_NAMES and artifact.path.is_file():
                    record.pfm_maps[artifact.path.name] = artifact.path
            if artifact.schema_version < 3 and artifact.pass_index is not None and artifact.path.is_file():
                record.pass_maps.setdefault(artifact.signal, {})[artifact.pass_index] = artifact.path

        # Schema-v3 manifests are authoritative. Older captures retain the
        # filename fallback so existing reports remain reproducible.
        if to_int(record.map_manifest.get("schema_version"), 0) < 3:
            maps_dir = depthmap_dir / "maps"
            for name in PNG_MAP_NAMES:
                map_path = maps_dir / name
                if map_path.exists():
                    record.png_maps[name] = map_path
                    register_legacy_file(record, map_path, map_path.stem)
            for name in PFM_MAP_NAMES:
                map_path = maps_dir / name
                if map_path.exists():
                    record.pfm_maps[name] = map_path
                    register_legacy_file(record, map_path, map_path.stem)
        image_id = to_int(summary.get("image_id"), -1)
        scale_level = to_int(summary.get("scale_level"), 0)
        if image_id >= 0:
            improvement_dir = path / "instrumentation" / "improvements"
            pattern = f"depth{image_id:04d}_scale{scale_level:02d}_pass*_improvement.pfm"
            for improvement_path in sorted(improvement_dir.glob(pattern)):
                match = re.search(r"_pass(\d+)_improvement\.pfm$", improvement_path.name)
                if match:
                    record.improvement_maps[int(match.group(1))] = improvement_path
        if to_int(record.map_manifest.get("schema_version"), 0) < 3:
            pass_dir = depthmap_dir / "pass_maps"
            for pass_path in sorted(pass_dir.glob("pass*_*")):
                match = re.match(r"pass(\d+)_(.+)\.(pfm|png)$", pass_path.name)
                if match:
                    pass_index = int(match.group(1))
                    signal = match.group(2)
                    record.pass_maps.setdefault(signal, {})[pass_index] = pass_path
                    register_legacy_file(record, pass_path, signal, pass_index)
        run.depthmaps.append(record)
    return run


def parse_runs(values: list[str]) -> list[RunData]:
    runs: list[RunData] = []
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--runs entries must be name=/path, got: {item}")
        label, raw_path = item.split("=", 1)
        runs.append(load_run(label, Path(raw_path).expanduser().resolve()))
    return runs


def try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "figure.facecolor": "#fbfcfd",
                "axes.facecolor": "#ffffff",
                "axes.edgecolor": "#d8dee4",
                "axes.labelcolor": "#293241",
                "axes.titlecolor": "#1f2933",
                "axes.titleweight": "bold",
                "axes.titlesize": 13,
                "axes.labelsize": 10,
                "font.size": 10,
                "legend.fontsize": 9,
                "xtick.color": "#52616f",
                "ytick.color": "#52616f",
                "grid.color": "#d8dee4",
                "grid.linewidth": 0.8,
                "savefig.facecolor": "#fbfcfd",
                "savefig.bbox": "tight",
            }
        )
        return plt
    except Exception:
        return None


def try_import_numpy():
    try:
        import numpy as np

        return np
    except Exception:
        return None


def save_bar_plot(plt, out: Path, title: str, ylabel: str, labels: list[str], series: dict[str, list[float]]) -> Path | None:
    if not labels or not series:
        return None
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 0.6), 4.6))
    width = 0.82 / max(1, len(series))
    xs = list(range(len(labels)))
    has_available_value = False
    for idx, (name, values) in enumerate(series.items()):
        offsets = [x - 0.41 + width / 2 + idx * width for x in xs]
        plot_values = []
        for value in values:
            numeric = to_float(value)
            available = numeric is not None and math.isfinite(numeric)
            has_available_value |= available
            plot_values.append(numeric if available else math.nan)
        ax.bar(offsets, plot_values, width=width, label=name, color=PLOT_COLORS[idx % len(PLOT_COLORS)], alpha=0.92, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    if has_available_value:
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5, 0.5, "unavailable for this capture",
            transform=ax.transAxes, ha="center", va="center", color="#6b7280",
        )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def save_line_plot(plt, out: Path, title: str, ylabel: str, series: dict[str, list[tuple[int, float]]]) -> Path | None:
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    has_available_value = False
    for idx, (name, points) in enumerate(series.items()):
        points = sorted(
            (x, value) for x, value in points
            if to_float(value) is not None and math.isfinite(float(value))
        )
        if not points:
            continue
        has_available_value = True
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2.2, label=name, color=PLOT_COLORS[idx % len(PLOT_COLORS)])
    ax.set_title(title)
    logical_ticks = sorted({point[0] for points in series.values() for point in points})
    if logical_ticks:
        ax.set_xticks(logical_ticks)
        ax.set_xticklabels(["init" if value < 0 else str(value + 1) for value in logical_ticks])
    ax.set_xlabel("PatchMatch iteration")
    ax.set_ylabel(ylabel)
    if has_available_value:
        ax.legend(frameon=False)
    else:
        ax.text(
            0.5, 0.5, "unavailable for this capture",
            transform=ax.transAxes, ha="center", va="center", color="#6b7280",
        )
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def image_labels_for_run(run: RunData) -> list[str]:
    return [frame_label(record) for record in run.depthmaps]


def support_count(record: DepthMapRecord, count: int) -> int:
    for row in record.view_support:
        if to_int(row.get("supporting_view_count"), -1) == count:
            return to_int(row.get("pixels"))
    return 0


def support_ratio(record: DepthMapRecord, counts: set[int]) -> float:
    total = sum(to_int(row.get("pixels")) for row in record.view_support)
    if total <= 0:
        total = to_int(record.summary.get("num_pixels_total"))
    selected = sum(to_int(row.get("pixels")) for row in record.view_support if to_int(row.get("supporting_view_count"), -1) in counts)
    return safe_div(selected, total)


def mean_support(record: DepthMapRecord) -> float:
    total = sum(to_int(row.get("pixels")) for row in record.view_support)
    if not total:
        return 0.0
    weighted = sum(to_int(row.get("supporting_view_count")) * to_int(row.get("pixels")) for row in record.view_support)
    return weighted / total


def logical_iteration_value(record: DepthMapRecord, iteration: int, key: str) -> float:
    values = [to_float(row.get(key)) for row in record_logical_iterations(record) if to_int(row.get("logical_iteration"), -2) == iteration]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else math.nan


def run_logical_iterations(run: RunData) -> list[int]:
    iterations: set[int] = set()
    for record in run.depthmaps:
        for row in record_logical_iterations(record):
            iteration = to_int(row.get("logical_iteration"), -1)
            if iteration >= 0:
                iterations.add(iteration)
    return sorted(iterations)


def counters_by_image(run: RunData) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run.counters:
        grouped[str(row.get("image_id", ""))].append(row)
    return grouped


def timings_by_image(run: RunData) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in run.timings:
        grouped[str(row.get("image_id", ""))].append(row)
    return grouped


def last_pass_counter_rows(run: RunData) -> dict[str, dict[str, str]]:
    rows_by_image = counters_by_image(run)
    result: dict[str, dict[str, str]] = {}
    for image_id, rows in rows_by_image.items():
        if rows:
            result[image_id] = max(rows, key=lambda row: to_int(row.get("pass_index"), -1))
    return result


def counter_mean(row: dict[str, str] | None, sum_key: str, denom_key: str) -> float:
    if not row:
        return 0.0
    return safe_div(to_float(row.get(sum_key), 0.0) or 0.0, float(to_int(row.get(denom_key))))


def all_candidate_types(run: RunData) -> list[str]:
    keys = set()
    for record in run.depthmaps:
        if not candidate_accounting_available(record):
            continue
        for row in record.summary.get("candidate_acceptance", []):
            keys.add(str(row.get("candidate_type", "UNKNOWN")))
    return sorted(keys)


def candidate_value(record: DepthMapRecord, ctype: str, key: str) -> float:
    if not candidate_accounting_available(record):
        return math.nan
    for row in record.summary.get("candidate_acceptance", []):
        if str(row.get("candidate_type", "UNKNOWN")) == ctype:
            value = to_float(row.get(key))
            return value if value is not None else 0.0
    return math.nan


def read_pfm(path: Path, np):
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii", errors="replace").strip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"not a PFM: {path}")
        dims = handle.readline().decode("ascii", errors="replace").strip()
        while dims.startswith("#"):
            dims = handle.readline().decode("ascii", errors="replace").strip()
        width, height = [int(v) for v in dims.split()]
        scale = float(handle.readline().decode("ascii", errors="replace").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.fromfile(handle, dtype=dtype)
    pixel_count = width * height
    payload_channels = data.size // pixel_count if pixel_count and data.size % pixel_count == 0 else 0
    channels = 3 if header == "PF" else 1
    # OpenMVS can write a multi-channel TImage payload with a scalar Pf header.
    if payload_channels in {1, 3, 4}:
        channels = payload_channels
    expected = pixel_count * channels
    if data.size < expected:
        raise ValueError(f"PFM truncated: {path}")
    data = data[:expected]
    shape = (height, width, channels) if channels == 3 else (height, width)
    return np.flipud(data.reshape(shape))


def save_pfm_preview(plt, np, src: Path, dst: Path) -> bool:
    try:
        data = read_pfm(src, np)
    except Exception:
        return False
    if data.ndim == 3:
        finite = np.isfinite(data)
        data = np.where(finite, data, 0.0)
        if data.min() < -0.05 or data.max() > 1.05:
            data = (data + 1.0) * 0.5
        data = np.clip(data, 0.0, 1.0)
        plt.imsave(dst, data)
        return True
    finite = np.isfinite(data)
    if not finite.any():
        return False
    values = data[finite]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    norm = np.where(finite, norm, 0.0)
    cmap = "magma" if "cost" in src.name else "viridis"
    plt.imsave(dst, norm, cmap=cmap)
    return True


def make_reference_thumbnails(runs: list[RunData], assets: Path) -> None:
    try:
        from PIL import Image, ImageOps
    except Exception:
        for run in runs:
            for record in run.depthmaps:
                record.reference_thumbnail = record.reference_image
        return

    thumbnail_root = assets / "reference_images"
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for run in runs:
        run_dir = thumbnail_root / safe_asset_part(run.label)
        for record in run.depthmaps:
            if record.reference_image is None:
                continue
            out = run_dir / f"{safe_asset_part(record.directory.name)}_reference.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(record.reference_image) as source:
                    thumbnail = ImageOps.exif_transpose(source).convert("RGB")
                    thumbnail.thumbnail((1024, 1024), resampling)
                    thumbnail.save(out, format="JPEG", quality=90, optimize=True)
                record.reference_thumbnail = out
            except Exception:
                record.reference_thumbnail = record.reference_image


def make_map_previews(runs: list[RunData], assets: Path, plt) -> None:
    np = try_import_numpy()
    if plt is None or np is None:
        return
    preview_dir = assets / "map_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        for record in run.depthmaps:
            for name, path in record.pfm_maps.items():
                out = preview_dir / f"{run.label}_{record.directory.name}_{path.stem}.png"
                if save_pfm_preview(plt, np, path, out):
                    record.preview_maps[name] = out
            last_changed = read_png_u8(plt, np, record.png_maps.get("last_changed_iter.png"))
            logical_last_changed = decode_last_changed_iterations(record, np, last_changed)
            if logical_last_changed is not None:
                out = preview_dir / f"{run.label}_{record.directory.name}_last_changed_iteration.png"
                plt.imsave(out, logical_last_changed, cmap="cividis", vmin=0.0, vmax=1.0)
                record.preview_maps["last_changed_iteration.png"] = out


def improvement_iteration_label(iteration: int) -> str:
    return logical_iteration_label(iteration)


def save_improvement_panel(plt, np, run: RunData, record: DepthMapRecord, data_by_iteration: dict[int, Any], vmax: float, out: Path) -> Path | None:
    if not data_by_iteration:
        return None
    from matplotlib.colors import PowerNorm

    iterations = sorted(data_by_iteration)
    num_cols = min(6, len(iterations))
    num_rows = int(math.ceil(len(iterations) / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.0 * num_cols, 3.35 * num_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"{run.label} / {frame_label(record)} cost improvement by iteration", fontsize=14, fontweight="bold")
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    image_artist = None
    for ax, iteration in zip(axes.flat, iterations):
        data = data_by_iteration[iteration]
        display = np.where(np.isfinite(data) & (data >= 0.0), data, np.nan)
        image_artist = ax.imshow(display, cmap="magma", norm=norm, interpolation="nearest")
        ax.set_title(improvement_iteration_label(iteration), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(iterations):]:
        ax.axis("off")
    if image_artist is not None:
        fig.colorbar(
            image_artist,
            ax=list(axes.flat),
            shrink=0.82,
            pad=0.015,
            label=f"cost improvement (shared p99.5 clip = {vmax:.4f})",
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    record.improvement_vmax = vmax
    return out


def make_improvement_previews(runs: list[RunData], assets: Path, plt) -> None:
    np = try_import_numpy()
    if plt is None or np is None:
        return
    grouped: dict[tuple[str, int], list[tuple[RunData, DepthMapRecord]]] = defaultdict(list)
    for run in runs:
        for record in run.depthmaps:
            if not (
                record.logical_event_maps.get("cost_improvement_exact")
                or record.improvement_maps
            ):
                continue
            key = (image_id_key(record), to_int(record.summary.get("scale_level"), 0))
            grouped[key].append((run, record))

    preview_root = assets / "improvement_previews"
    for records in grouped.values():
        loaded: dict[int, dict[int, Any]] = {}
        sampled_values = []
        for _, record in records:
            record_data: dict[int, Any] = {}
            logical_artifacts = record.logical_event_maps.get(
                "cost_improvement_exact", {}
            )
            source_paths = (
                {
                    logical_iteration: artifact.path
                    for logical_iteration, artifact in logical_artifacts.items()
                }
                if logical_artifacts else record.improvement_maps
            )
            for index, path in source_paths.items():
                try:
                    data = read_pfm(path, np)
                except Exception:
                    continue
                if data.ndim != 2:
                    continue
                record_data[index] = data
                positive = data[np.isfinite(data) & (data > 0.0)]
                if positive.size > 200000:
                    positive = positive[:: int(math.ceil(positive.size / 200000))]
                if positive.size:
                    sampled_values.append(positive)
            loaded[id(record)] = (
                record_data
                if logical_artifacts
                else combine_checkerboard_maps(record_data, np)
            )
        if sampled_values:
            vmax = float(np.percentile(np.concatenate(sampled_values), 99.5))
        else:
            vmax = 1.0
        if not math.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        for run, record in records:
            out = preview_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_improvements.png"
            record.improvement_panel_path = save_improvement_panel(plt, np, run, record, loaded[id(record)], vmax, out)


def load_pass_map(path: Path, np, plt):
    try:
        if path.suffix.lower() == ".pfm":
            data = read_pfm(path, np)
        else:
            data = plt.imread(str(path))
            if data.ndim == 3:
                data = data[..., 0]
            if np.issubdtype(data.dtype, np.floating):
                data = data * 255.0
        return data if data.ndim == 2 else None
    except Exception:
        return None


def save_pass_map_panel(
    plt,
    np,
    run: RunData,
    record: DepthMapRecord,
    signal: str,
    data_by_iteration: dict[int, Any],
    limit: float,
    out: Path,
) -> Path | None:
    if not data_by_iteration:
        return None
    signed = signal in {"depth_delta", "depth_relative_delta"}
    iterations = sorted(data_by_iteration)
    num_cols = min(6, len(iterations))
    num_rows = int(math.ceil(len(iterations) / num_cols))
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(3.0 * num_cols, 3.3 * num_rows),
        constrained_layout=True, squeeze=False,
    )
    title = signal.replace("_", " ")
    fig.suptitle(f"{run.label} / {frame_label(record)} {title} by iteration", fontsize=14, fontweight="bold")
    artist = None
    for ax, iteration in zip(axes.flat, iterations):
        data = data_by_iteration[iteration]
        display = np.ma.masked_invalid(data)
        if signed:
            artist = ax.imshow(display, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        else:
            artist = ax.imshow(display, cmap="magma", vmin=0.0, vmax=limit, interpolation="nearest")
        ax.set_title(improvement_iteration_label(iteration), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in list(axes.flat)[len(iterations):]:
        ax.axis("off")
    if artist is not None:
        fig.colorbar(artist, ax=list(axes.flat), shrink=0.82, pad=0.015, label=f"{title} (shared p99 clip)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def make_pass_map_previews(runs: list[RunData], assets: Path, plt) -> None:
    np = try_import_numpy()
    if plt is None or np is None:
        return
    grouped: dict[tuple[str, int, str], list[tuple[RunData, DepthMapRecord]]] = defaultdict(list)
    for run in runs:
        for record in run.depthmaps:
            for signal in set(record.pass_maps) | set(record.logical_event_maps):
                if signal == "cost_improvement_exact":
                    continue
                key = (image_id_key(record), to_int(record.summary.get("scale_level"), 0), signal)
                grouped[key].append((run, record))
    panel_root = assets / "pass_map_previews"
    for (_image_id, _scale, signal), records in grouped.items():
        loaded: dict[int, dict[int, Any]] = {}
        samples = []
        for _, record in records:
            values = {}
            logical_artifacts = record.logical_event_maps.get(signal, {})
            source_paths = (
                {iteration: artifact.path for iteration, artifact in logical_artifacts.items()}
                if logical_artifacts else record.pass_maps.get(signal, {})
            )
            for index, path in source_paths.items():
                data = load_pass_map(path, np, plt)
                if data is None:
                    continue
                values[index] = data
                finite = np.abs(data[np.isfinite(data)]) if signal in {"depth_delta", "depth_relative_delta"} else data[np.isfinite(data) & (data >= 0.0)]
                if finite.size > 200000:
                    finite = finite[:: int(math.ceil(finite.size / 200000))]
                if finite.size:
                    samples.append(finite)
            # Schema-v3 logical maps already represent a complete iteration.
            # Only legacy phase-indexed event maps use checkerboard synthesis.
            loaded[id(record)] = values if logical_artifacts else combine_checkerboard_maps(values, np)
        limit = float(np.percentile(np.concatenate(samples), 99.0)) if samples else 1.0
        if not math.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        for run, record in records:
            out = panel_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_{signal}.png"
            panel = save_pass_map_panel(plt, np, run, record, signal, loaded[id(record)], limit, out)
            if panel is not None:
                record.pass_panel_paths[signal] = panel


def load_artifact_map(artifact: MapArtifact, np, plt):
    return load_pass_map(artifact.path, np, plt)


def logical_cost_limits(np, loaded_records: list[dict[str, dict[int, Any]]]) -> dict[str, tuple[float, float]]:
    samples_by_signal: dict[str, list[Any]] = defaultdict(list)
    shared_cost_samples = []
    for loaded in loaded_records:
        for signal, data_by_iteration in loaded.items():
            kind = LOGICAL_COST_SIGNAL_PRESENTATION[signal][2]
            for data in data_by_iteration.values():
                values = np.asarray(data).reshape(-1)
                valid = np.isfinite(values)
                if kind != "residual":
                    valid &= values >= 0.0
                values = values[valid]
                if values.size > 200000:
                    values = values[:: int(math.ceil(values.size / 200000))]
                if not values.size:
                    continue
                samples_by_signal[signal].append(values)
                if kind == "cost":
                    shared_cost_samples.append(values)

    def positive_high(samples: list[Any], percentile: float = 99.5) -> float:
        if not samples:
            return 1.0
        value = float(np.percentile(np.concatenate(samples), percentile))
        return value if math.isfinite(value) and value > 1e-8 else 1.0

    shared_cost_high = positive_high(shared_cost_samples)
    limits: dict[str, tuple[float, float]] = {}
    for signal, (_label, _fidelity, kind) in LOGICAL_COST_SIGNAL_PRESENTATION.items():
        if kind == "unit":
            limits[signal] = (0.0, 1.0)
        elif kind == "cost":
            limits[signal] = (0.0, shared_cost_high)
        elif kind == "residual":
            samples = samples_by_signal.get(signal, [])
            high = positive_high([np.abs(values) for values in samples], 99.0)
            limits[signal] = (-high, high)
        else:
            limits[signal] = (0.0, positive_high(samples_by_signal.get(signal, [])))
    return limits


def save_logical_cost_panel(
    plt,
    np,
    run: RunData,
    record: DepthMapRecord,
    loaded: dict[str, dict[int, Any]],
    limits: dict[str, tuple[float, float]],
    out: Path,
) -> Path | None:
    signals = [signal for signal in LOGICAL_COST_SIGNAL_PRESENTATION if loaded.get(signal)]
    iterations = sorted({iteration for signal in signals for iteration in loaded[signal]})
    if not signals or not iterations:
        return None

    fig, axes = plt.subplots(
        len(signals),
        len(iterations),
        figsize=(max(9.0, 2.75 * len(iterations)), max(5.0, 2.35 * len(signals))),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"{run.label} / {frame_label(record)} logical-iteration cost evolution",
        fontsize=15,
        fontweight="bold",
    )
    fidelity_colors = {"exact": "#1f6f50", "derived": "#355c7d", "proxy": "#9a6700", "unspecified": "#6b7280"}
    for row, signal in enumerate(signals):
        label, fidelity, kind = LOGICAL_COST_SIGNAL_PRESENTATION[signal]
        artist = None
        for column, iteration in enumerate(iterations):
            ax = axes[row, column]
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(logical_iteration_label(iteration), fontsize=9, fontweight="bold")
            if column == 0:
                ax.set_ylabel(
                    f"{label}\n[{fidelity.upper()}]",
                    fontsize=7.5,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                    color=fidelity_colors[fidelity],
                    fontweight="bold",
                )
            data = loaded[signal].get(iteration)
            if data is None:
                ax.set_facecolor("#f3f4f6")
                ax.text(0.5, 0.5, "unavailable", ha="center", va="center", color="#6b7280", fontsize=8)
                continue
            display = np.asarray(data)
            if kind != "residual":
                display = np.ma.masked_where(~np.isfinite(display) | (display < 0.0), display)
            else:
                display = np.ma.masked_invalid(display)
            low, high = limits[signal]
            cmap = "coolwarm" if kind == "residual" else ("magma" if kind == "cost" else "viridis")
            artist = ax.imshow(display, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        if artist is not None:
            colorbar = fig.colorbar(artist, ax=list(axes[row, :]), fraction=0.018, pad=0.012, aspect=35)
            colorbar.ax.tick_params(labelsize=6, length=2)
    fig.text(
        0.5,
        0.002,
        "EXACT: stored production value   |   DERIVED: exact arithmetic from exported values   |   PROXY: equal-selected-view post-state rescore",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4b5563",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def make_logical_cost_previews(runs: list[RunData], assets: Path, plt) -> None:
    np = try_import_numpy()
    if plt is None or np is None:
        return
    grouped: dict[tuple[str, int], list[tuple[RunData, DepthMapRecord]]] = defaultdict(list)
    for run in runs:
        for record in run.depthmaps:
            if any(record.logical_state_maps.get(signal) for signal in LOGICAL_COST_SIGNAL_PRESENTATION):
                key = (image_id_key(record), to_int(record.summary.get("scale_level"), 0))
                grouped[key].append((run, record))

    panel_root = assets / "logical_cost_previews"
    for records in grouped.values():
        loaded_by_record: dict[int, dict[str, dict[int, Any]]] = {}
        for _run, record in records:
            loaded: dict[str, dict[int, Any]] = {}
            for signal in LOGICAL_COST_SIGNAL_PRESENTATION:
                for iteration, artifact in sorted(record.logical_state_maps.get(signal, {}).items()):
                    data = load_artifact_map(artifact, np, plt)
                    if data is not None:
                        loaded.setdefault(signal, {})[iteration] = data
            loaded_by_record[id(record)] = loaded
        limits = logical_cost_limits(np, list(loaded_by_record.values()))
        for run, record in records:
            out = panel_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_logical_costs.png"
            record.logical_cost_panel_path = save_logical_cost_panel(
                plt, np, run, record, loaded_by_record[id(record)], limits, out
            )


def add_plot(plots: list[tuple[str, Path]], title: str, path: Path | None) -> None:
    if path:
        plots.append((title, path))


def title_with_run(run: RunData, title: str, include_run_label: bool) -> str:
    return f"{run.label}: {title}" if include_run_label else title


def add_per_frame_plots(run: RunData, assets: Path, plots: list[tuple[str, Path]], plt, include_run_label: bool) -> None:
    if not run.depthmaps:
        return
    run_dir = assets / "per_frame" / safe_asset_part(run.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    labels = image_labels_for_run(run)

    add_plot(
        plots,
        title_with_run(run, "per-frame validity ratios", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "validity_ratios_by_frame.png",
            title_with_run(run, "validity and filtering by frame", include_run_label),
            "ratio",
            labels,
            {
                "valid before filter": [to_float(record.summary.get("valid_ratio_before_filter"), 0.0) or 0.0 for record in run.depthmaps],
                "valid after filter": [to_float(record.summary.get("valid_ratio_after_filter"), 0.0) or 0.0 for record in run.depthmaps],
                "rejected by filter": [to_float(record.summary.get("rejected_by_filter_ratio"), 0.0) or 0.0 for record in run.depthmaps],
            },
        ),
    )
    add_plot(
        plots,
        title_with_run(run, "per-frame pixel counts", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "pixel_counts_by_frame.png",
            title_with_run(run, "valid/rejected pixel counts by frame", include_run_label),
            "pixels",
            labels,
            {
                "valid before": [float(to_int(record.summary.get("num_valid_before_filter"))) for record in run.depthmaps],
                "valid after": [float(to_int(record.summary.get("num_valid_after_filter"))) for record in run.depthmaps],
                "rejected": [float(to_int(record.summary.get("num_rejected_by_filter"))) for record in run.depthmaps],
            },
        ),
    )
    add_plot(
        plots,
        title_with_run(run, "per-frame final cost statistics", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "final_cost_stats_by_frame.png",
            title_with_run(run, "final cost distribution by frame", include_run_label),
            "cost",
            labels,
            {
                "mean": [to_float(record.summary.get("final_cost", {}).get("mean"), 0.0) or 0.0 for record in run.depthmaps],
                "median": [to_float(record.summary.get("final_cost", {}).get("median"), 0.0) or 0.0 for record in run.depthmaps],
                "p90": [to_float(record.summary.get("final_cost", {}).get("p90"), 0.0) or 0.0 for record in run.depthmaps],
                "p95": [to_float(record.summary.get("final_cost", {}).get("p95"), 0.0) or 0.0 for record in run.depthmaps],
            },
        ),
    )
    add_plot(
        plots,
        title_with_run(run, "per-frame support quality", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "support_quality_by_frame.png",
            title_with_run(run, "source-view support quality by frame", include_run_label),
            "ratio / normalized mean",
            labels,
            {
                "no support": [support_ratio(record, {0}) for record in run.depthmaps],
                "weak support <=1": [support_ratio(record, {0, 1}) for record in run.depthmaps],
                "strong support >=3": [support_ratio(record, {3, 4, 5, 6, 7, 8}) for record in run.depthmaps],
                "mean support / 4": [mean_support(record) / 4.0 for record in run.depthmaps],
            },
        ),
    )

    candidate_types = all_candidate_types(run)
    if candidate_types:
        add_plot(
            plots,
            title_with_run(run, "per-frame accepted candidate source", include_run_label),
            save_bar_plot(
                plt,
                run_dir / "candidate_accepts_by_frame.png",
                title_with_run(run, "accepted candidate source by frame", include_run_label),
                "accepted candidates",
                labels,
                {ctype: [candidate_value(record, ctype, "accepted_count") for record in run.depthmaps] for ctype in candidate_types},
            ),
        )
        add_plot(
            plots,
            title_with_run(run, "per-frame candidate acceptance rates", include_run_label),
            save_bar_plot(
                plt,
                run_dir / "candidate_acceptance_rates_by_frame.png",
                title_with_run(run, "candidate acceptance rates by frame", include_run_label),
                "acceptance rate",
                labels,
                {ctype: [candidate_value(record, ctype, "acceptance_rate") for record in run.depthmaps] for ctype in candidate_types},
            ),
        )

    logical_iterations = run_logical_iterations(run)
    if logical_iterations:
        for key, title, ylabel, name in (
            ("changed_ratio", "changed ratio", "ratio", "changed_ratio_by_frame.png"),
            ("mean_cost", "mean PatchMatch cost", "cost", "mean_cost_by_frame.png"),
            ("mean_cost_delta", "mean cost improvement", "cost delta", "mean_cost_delta_by_frame.png"),
            ("mean_abs_depth_delta", "mean absolute depth update", "depth units", "mean_abs_depth_delta_by_frame.png"),
            ("mean_normal_delta_deg", "mean normal update", "degrees", "mean_normal_delta_by_frame.png"),
        ):
            add_plot(
                plots,
                title_with_run(run, f"per-frame {title}", include_run_label),
                save_bar_plot(
                    plt,
                    run_dir / name,
                    title_with_run(run, f"{title} by frame and PatchMatch iteration", include_run_label),
                    ylabel,
                    labels,
                    {
                        logical_iteration_label(iteration): [logical_iteration_value(record, iteration, key) for record in run.depthmaps]
                        for iteration in logical_iterations
                    },
                ),
            )

    last_rows = last_pass_counter_rows(run)
    add_plot(
        plots,
        title_with_run(run, "per-frame component costs", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "component_costs_by_frame.png",
            title_with_run(run, "final-iteration cost components by frame", include_run_label),
            "mean cost",
            labels,
            {
                "aggregate": [counter_mean(last_rows.get(image_id_key(record)), "cost_sum", "processed") for record in run.depthmaps],
                "photometric": [counter_mean(last_rows.get(image_id_key(record)), "photometric_cost_sum", "component_samples") for record in run.depthmaps],
                "photo-prior": [counter_mean(last_rows.get(image_id_key(record)), "photo_prior_cost_sum", "component_samples") for record in run.depthmaps],
                "geometric": [counter_mean(last_rows.get(image_id_key(record)), "geometric_cost_sum", "component_samples") for record in run.depthmaps],
            },
        ),
    )

    counter_rows_by_image = counters_by_image(run)
    add_plot(
        plots,
        title_with_run(run, "per-frame low-texture and bad-cost rates", include_run_label),
        save_bar_plot(
            plt,
            run_dir / "low_texture_bad_cost_rates_by_frame.png",
            title_with_run(run, "low-texture and bad-cost rates by frame", include_run_label),
            "ratio",
            labels,
            {
                "low texture": [
                    safe_div(sum(to_int(row.get("low_texture")) for row in counter_rows_by_image.get(image_id_key(record), [])), sum(to_int(row.get("processed")) for row in counter_rows_by_image.get(image_id_key(record), [])))
                    for record in run.depthmaps
                ],
                "bad cost": [
                    safe_div(sum(to_int(row.get("bad_cost")) for row in counter_rows_by_image.get(image_id_key(record), [])), sum(to_int(row.get("processed")) for row in counter_rows_by_image.get(image_id_key(record), [])))
                    for record in run.depthmaps
                ],
            },
        ),
    )

    timing_rows_by_image = timings_by_image(run)
    timing_phases = sorted({row.get("phase", "") for row in run.timings if row.get("phase", "")})
    if timing_phases:
        add_plot(
            plots,
            title_with_run(run, "per-frame kernel timing", include_run_label),
            save_bar_plot(
                plt,
                run_dir / "kernel_ms_by_frame.png",
                title_with_run(run, "CUDA kernel time by frame", include_run_label),
                "milliseconds",
                labels,
                {
                    phase: [
                        sum(to_float(row.get("kernel_ms"), 0.0) or 0.0 for row in timing_rows_by_image.get(image_id_key(record), []) if row.get("phase") == phase)
                        for record in run.depthmaps
                    ]
                    for phase in timing_phases
                },
            ),
        )


def display_image_axis(plt, ax, path: Path | None, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    if path is None or not path.exists():
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center", color="#6b7280")
        ax.set_facecolor("#f3f4f6")
        return
    try:
        image = plt.imread(str(path))
    except Exception:
        ax.text(0.5, 0.5, "unreadable", ha="center", va="center", color="#6b7280")
        ax.set_facecolor("#f3f4f6")
        return
    ax.imshow(image)


def plot_iteration_axis(ax, record: DepthMapRecord) -> None:
    rows = record_logical_iterations(record)
    xs = [to_int(row.get("logical_iteration"), idx) for idx, row in enumerate(rows)]
    costs = [to_float(row.get("mean_cost"), 0.0) or 0.0 for row in rows]
    changed = [to_float(row.get("changed_ratio")) for row in rows]
    acceptance = [to_float(row.get("acceptance_rate")) for row in rows]
    ax.plot(xs, costs, marker="o", color=PLOT_COLORS[0], linewidth=2.0, label="mean cost")
    if any(value is not None for value in changed):
        ax.plot(xs, [value if value is not None else math.nan for value in changed], marker="o", color=PLOT_COLORS[1], linewidth=2.0, label="changed")
    if any(value is not None for value in acceptance):
        ax.plot(xs, [value if value is not None else math.nan for value in acceptance], marker="o", color=PLOT_COLORS[2], linewidth=2.0, label="candidate acceptance")
    if not candidate_accounting_available(record):
        ax.text(
            0.98, 0.04, "change and candidate accounting unavailable",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color="#6b7280",
        )
    ax.set_title("PatchMatch evolution", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["init" if value < 0 else str(value + 1) for value in xs])
    ax.set_xlabel("iteration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_candidate_axis(ax, record: DepthMapRecord) -> None:
    rows = record.summary.get("candidate_acceptance", [])
    if not candidate_accounting_available(record) or not rows:
        ax.axis("off")
        ax.set_title("Candidate-family attribution", fontsize=9)
        ax.text(
            0.5, 0.5,
            (
                "unavailable in robust\npost-pass snapshot mode"
                if not candidate_accounting_available(record)
                else "candidate-family counters not recorded"
            ),
            ha="center", va="center", color="#6b7280", fontsize=9,
        )
        return
    labels = [str(row.get("candidate_type", "UNKNOWN")).replace("_", "\n") for row in rows]
    values = [to_int(row.get("accepted_count")) for row in rows]
    xs = list(range(len(labels)))
    ax.bar(xs, values, color=[PLOT_COLORS[idx % len(PLOT_COLORS)] for idx in xs], edgecolor="white", linewidth=0.6)
    ax.set_title("Accepted candidates", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_support_axis(ax, record: DepthMapRecord) -> None:
    labels = [str(to_int(row.get("supporting_view_count"))) for row in record.view_support]
    values = [to_int(row.get("pixels")) for row in record.view_support]
    xs = list(range(len(labels)))
    ax.bar(xs, values, color=PLOT_COLORS[2], edgecolor="white", linewidth=0.6)
    ax.set_title("Supporting views", fontsize=9)
    ax.set_xlabel("views")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_summary_axis(ax, record: DepthMapRecord) -> None:
    final_cost = record.summary.get("final_cost", {})
    lines = [
        f"valid before: {fmt_pct(record.summary.get('valid_ratio_before_filter'))}",
        f"valid after:  {fmt_pct(record.summary.get('valid_ratio_after_filter'))}",
        f"rejected:     {fmt_pct(record.summary.get('rejected_by_filter_ratio'))}",
        f"cost mean:    {fmt_float(final_cost.get('mean'))}",
        f"cost median:  {fmt_float(final_cost.get('median'))}",
        f"cost p90/p95: {fmt_float(final_cost.get('p90'))} / {fmt_float(final_cost.get('p95'))}",
        f"mean support: {mean_support(record):.2f}",
    ]
    ax.axis("off")
    ax.set_title("Frame summary", fontsize=9)
    ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8, color="#293241")


def save_frame_panel(plt, run: RunData, record: DepthMapRecord, out: Path, include_run_label: bool) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"{run.label} / {frame_label(record)} depth-map diagnostics" if include_run_label else f"{frame_label(record)} depth-map diagnostics"
    map_items: list[tuple[str, Path | None]] = [
        ("reference RGB", record.reference_thumbnail or record.reference_image),
        ("final cost", record.png_maps.get("cost_final.png") or record.preview_maps.get("cost_final.pfm")),
        ("final depth", record.preview_maps.get("depth_final_after_filter.pfm")),
        ("final normal", record.preview_maps.get("normal_final.pfm")),
        ("valid after filter", record.png_maps.get("valid_after_filter.png")),
        ("candidate source", record.png_maps.get("candidate_source.png")),
        ("supporting views", record.png_maps.get("num_supporting_views.png")),
        ("last changed iteration", record.preview_maps.get("last_changed_iteration.png") or record.png_maps.get("last_changed_iter.png")),
        ("rejection reason", record.png_maps.get("rejection_reason.png")),
    ]
    optional_items = [
        ("accepted update count", record.png_maps.get("accepted_update_count.png")),
        ("photometric cost", record.preview_maps.get("cost_photometric.pfm")),
        ("geometric cost", record.preview_maps.get("cost_geometric.pfm")),
        ("reconstructed total", record.preview_maps.get("cost_total_components.pfm")),
        ("confidence gap", record.preview_maps.get("confidence_gap.pfm")),
        ("view-weight entropy", record.preview_maps.get("view_entropy.pfm")),
    ]
    map_items.extend((label, path) for label, path in optional_items if path is not None)
    map_rows = int(math.ceil(len(map_items) / 5.0))
    fig = plt.figure(figsize=(18, 3.5 * map_rows + 4.0), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    grid = fig.add_gridspec(map_rows + 1, 15, height_ratios=(*([1.0] * map_rows), 1.05))
    tile_axes = [fig.add_subplot(grid[row, col * 3:(col + 1) * 3]) for row in range(map_rows) for col in range(5)]
    chart_axes = [fig.add_subplot(grid[map_rows, col * 5:(col + 1) * 5]) for col in range(3)]
    for ax, (map_title, path) in zip(tile_axes, map_items):
        display_image_axis(plt, ax, path, map_title)
    for ax in tile_axes[len(map_items):]:
        ax.axis("off")
    plot_iteration_axis(chart_axes[0], record)
    plot_candidate_axis(chart_axes[1], record)
    plot_support_axis(chart_axes[2], record)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def read_record_pfm(record: DepthMapRecord, name: str, np):
    path = record.pfm_maps.get(name)
    if path is None:
        return None
    try:
        return read_pfm(path, np)
    except Exception:
        return None


def sampled_finite_values(np, data, require_nonnegative: bool = False):
    if data is None:
        return np.asarray([], dtype=float)
    values = np.asarray(data).reshape(-1)
    mask = np.isfinite(values)
    if require_nonnegative:
        mask &= values >= 0.0
    values = values[mask]
    if values.size > 200000:
        values = values[:: int(math.ceil(values.size / 200000))]
    return values


def save_cost_function_panel(plt, np, run: RunData, record: DepthMapRecord, out: Path) -> Path | None:
    cost_maps = [
        ("stored final cost", read_record_pfm(record, "cost_final.pfm", np)),
        ("raw photometric", read_record_pfm(record, "cost_photometric.pfm", np)),
        ("photo + depth prior", read_record_pfm(record, "cost_photo_prior.pfm", np)),
        ("geometric penalty", read_record_pfm(record, "cost_geometric.pfm", np)),
        ("reconstructed total", read_record_pfm(record, "cost_total_components.pfm", np)),
    ]
    if all(data is None for _, data in cost_maps[1:]):
        return None
    final_before = read_record_pfm(record, "cost_final_before_filter.pfm", np)
    photo_prior = cost_maps[2][1]
    geometric = cost_maps[3][1]
    reconstructed = cost_maps[4][1]
    component_residual = None
    if photo_prior is not None and geometric is not None and reconstructed is not None:
        component_residual = reconstructed - (photo_prior + geometric)
    final_residual = None
    if final_before is not None and reconstructed is not None:
        final_residual = final_before - reconstructed
    diagnostic_maps = [
        ("component closure residual", component_residual, "residual"),
        ("stored - reconstructed", final_residual, "residual"),
        ("raw depth-prior disagreement", read_record_pfm(record, "cost_depth_prior.pfm", np), "generic"),
        ("depth-prior blend weight", read_record_pfm(record, "depth_prior_weight.pfm", np), "weight"),
        ("reference-patch variance", read_record_pfm(record, "reference_variance.pfm", np), "generic"),
        ("confidence-gap proxy", read_record_pfm(record, "confidence_gap.pfm", np), "nonnegative"),
        ("low-resolution depth prior", read_record_pfm(record, "low_depth_prior.pfm", np), "generic"),
    ]
    map_specs: list[tuple[str, Any, str]] = [(title, data, "cost") for title, data in cost_maps]
    map_specs.extend((title, data, kind) for title, data, kind in diagnostic_maps if data is not None)
    num_cols = 5
    num_rows = int(math.ceil((1 + len(map_specs)) / num_cols))
    fig = plt.figure(figsize=(19, 3.45 * num_rows + 4.4), constrained_layout=True)
    fig.suptitle(f"{run.label} / {frame_label(record)} cost-function diagnostics", fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(num_rows + 1, 15, height_ratios=(*([1.0] * num_rows), 1.08))
    tile_axes = [fig.add_subplot(grid[row, col * 3:(col + 1) * 3]) for row in range(num_rows) for col in range(num_cols)]
    chart_axes = [fig.add_subplot(grid[num_rows, col * 5:(col + 1) * 5]) for col in range(3)]

    display_image_axis(plt, tile_axes[0], record.reference_thumbnail or record.reference_image, "reference RGB")
    cost_values = [sampled_finite_values(np, data) for _, data in cost_maps if data is not None]
    combined_costs = np.concatenate([values for values in cost_values if values.size]) if any(values.size for values in cost_values) else np.asarray([0.0, 1.0])
    cost_hi = float(np.percentile(combined_costs, 99.5))
    if not math.isfinite(cost_hi) or cost_hi <= 0.0:
        cost_hi = 1.0
    cost_artist = None
    cost_axes = []
    for ax, (title, data, kind) in zip(tile_axes[1:], map_specs):
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if data is None:
            show_unavailable_axis(ax, title)
            continue
        display = np.ma.masked_invalid(data)
        if kind == "cost":
            cost_artist = ax.imshow(display, cmap="magma", vmin=0.0, vmax=cost_hi, interpolation="nearest")
            cost_axes.append(ax)
        elif kind == "residual":
            limit = symmetric_limit(np, data, 99.5, 1e-6)
            ax.imshow(display, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        elif kind == "weight":
            ax.imshow(display, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        else:
            values = sampled_finite_values(np, data, require_nonnegative=kind == "nonnegative")
            if kind == "nonnegative":
                display = np.ma.masked_where(~np.isfinite(data) | (data < 0.0), data)
            hi = float(np.percentile(values, 99.5)) if values.size else 1.0
            if not math.isfinite(hi) or hi <= 0.0:
                hi = 1.0
            ax.imshow(display, cmap="viridis", vmin=0.0, vmax=hi, interpolation="nearest")
    for ax in tile_axes[1 + len(map_specs):]:
        ax.axis("off")
    add_horizontal_colorbar(fig, cost_artist, cost_axes, "cost (shared p99.5 clip)")

    plot_iteration_axis(chart_axes[0], record)
    histogram_axis = chart_axes[1]
    for index, (title, data) in enumerate(cost_maps):
        values = sampled_finite_values(np, data)
        if values.size:
            histogram_axis.hist(values, bins=80, range=(0.0, cost_hi), density=True, histtype="step", linewidth=1.6, color=PLOT_COLORS[index % len(PLOT_COLORS)], label=title)
    histogram_axis.set_title("Cost-component distributions", fontsize=9)
    histogram_axis.set_xlabel("cost")
    histogram_axis.set_ylabel("density")
    histogram_axis.legend(frameon=False, fontsize=6)
    histogram_axis.grid(alpha=0.2)

    variance = read_record_pfm(record, "reference_variance.pfm", np)
    gap = read_record_pfm(record, "confidence_gap.pfm", np)
    ambiguity_axis = chart_axes[2]
    if variance is None or gap is None:
        plot_summary_axis(ambiguity_axis, record)
    else:
        valid = np.isfinite(variance) & np.isfinite(gap) & (gap >= 0.0)
        x = variance[valid]
        y = gap[valid]
        if x.size > 150000:
            stride = int(math.ceil(x.size / 150000))
            x = x[::stride]
            y = y[::stride]
        if x.size:
            ambiguity_axis.hexbin(x, y, gridsize=55, bins="log", mincnt=1, cmap="viridis")
            ambiguity_axis.set_title("Texture variance vs confidence gap", fontsize=9)
            ambiguity_axis.set_xlabel("reference variance")
            ambiguity_axis.set_ylabel("gap proxy")
            ambiguity_axis.grid(alpha=0.15)
        else:
            show_unavailable_axis(ambiguity_axis, "Texture variance vs confidence gap")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def save_view_selection_panel(plt, np, run: RunData, record: DepthMapRecord, out: Path) -> Path | None:
    entropy = read_record_pfm(record, "view_entropy.pfm", np)
    selected_count = read_png_u8(plt, np, record.png_maps.get("selected_view_count.png"))
    view_weights = [read_record_pfm(record, f"view_weight_{index}.pfm", np) for index in range(4)]
    view_costs = [read_record_pfm(record, f"view_cost_{index}.pfm", np) for index in range(4)]
    view_photo = [read_record_pfm(record, f"view_photometric_cost_{index}.pfm", np) for index in range(4)]
    view_geom = [read_record_pfm(record, f"view_geometric_cost_{index}.pfm", np) for index in range(4)]
    support_path = record.png_maps.get("num_supporting_views.png")
    if entropy is None and selected_count is None and not any(data is not None for data in [*view_weights, *view_costs, *view_photo, *view_geom]):
        return None
    source_views = record.summary.get("selected_source_views") or []
    gap = read_record_pfm(record, "confidence_gap.pfm", np)
    num_cols = 5
    has_per_view = any(data is not None for data in [*view_weights, *view_costs, *view_photo, *view_geom])
    num_rows = 5 if has_per_view else 1
    fig = plt.figure(figsize=(19, 3.3 * num_rows + 5.0))
    fig.suptitle(f"{run.label} / {frame_label(record)} view-selection diagnostics", fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(
        num_rows + 2, 20,
        height_ratios=(*([1.0] * num_rows), 0.11, 1.08),
        left=0.04, right=0.96, top=0.94, bottom=0.04, hspace=0.34, wspace=0.28,
    )
    tile_axes = [fig.add_subplot(grid[row, col * 4:(col + 1) * 4]) for row in range(num_rows) for col in range(num_cols)]
    colorbar_axis = fig.add_subplot(grid[num_rows, 4:16])
    chart_axes = [fig.add_subplot(grid[num_rows + 1, col * 5:(col + 1) * 5]) for col in range(4)]
    display_image_axis(plt, tile_axes[0], record.reference_thumbnail or record.reference_image, "reference RGB")
    display_image_axis(plt, tile_axes[1], support_path, "final supporting views")

    all_cost_values = []
    for data in [*view_costs, *view_photo, *view_geom]:
        values = sampled_finite_values(np, data)
        if values.size:
            all_cost_values.append(values)
    cost_hi = float(np.percentile(np.concatenate(all_cost_values), 99.5)) if all_cost_values else 1.0
    if not math.isfinite(cost_hi) or cost_hi <= 0.0:
        cost_hi = 1.0
    cost_artist = None
    cost_axes = []

    def display_scalar(ax, title: str, data, kind: str) -> None:
        nonlocal cost_artist
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if data is None:
            show_unavailable_axis(ax, title)
            return
        if kind == "weight" or kind == "entropy":
            ax.imshow(np.ma.masked_invalid(data), cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        elif kind == "count":
            ax.imshow(data, cmap="cividis", vmin=0.0, vmax=4.0, interpolation="nearest")
        elif kind == "gap":
            values = sampled_finite_values(np, data, require_nonnegative=True)
            hi = float(np.percentile(values, 99.5)) if values.size else 1.0
            display = np.ma.masked_where(~np.isfinite(data) | (data < 0.0), data)
            ax.imshow(display, cmap="viridis", vmin=0.0, vmax=max(hi, 1e-6), interpolation="nearest")
        else:
            cost_artist = ax.imshow(np.ma.masked_invalid(data), cmap="magma", vmin=0.0, vmax=cost_hi, interpolation="nearest")
            cost_axes.append(ax)

    display_scalar(tile_axes[2], "selected-view count", selected_count, "count")
    display_scalar(tile_axes[3], "view-weight entropy", entropy, "entropy")
    display_scalar(tile_axes[4], "confidence-gap proxy", gap, "gap")
    if has_per_view:
        for index in range(4):
            row_axes = tile_axes[(index + 1) * num_cols:(index + 2) * num_cols]
            source = source_views[index] if index < len(source_views) else {}
            source_id = source.get("id", "unused")
            source_path = Path(str(source.get("name", ""))).expanduser() if source else None
            display_image_axis(plt, row_axes[0], source_path, f"slot {index} / image {source_id}: RGB")
            display_scalar(row_axes[1], f"slot {index}: weight", view_weights[index], "weight")
            display_scalar(row_axes[2], f"slot {index}: total cost", view_costs[index], "cost")
            display_scalar(row_axes[3], f"slot {index}: photometric", view_photo[index], "cost")
            display_scalar(row_axes[4], f"slot {index}: geometric", view_geom[index], "cost")
    if cost_artist is not None:
        colorbar = fig.colorbar(cost_artist, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("per-view cost (shared p99.5 clip)", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)
    else:
        colorbar_axis.axis("off")

    plot_support_axis(chart_axes[0], record)
    entropy_axis = chart_axes[1]
    entropy_values = sampled_finite_values(np, entropy)
    if entropy_values.size:
        entropy_axis.hist(entropy_values, bins=60, range=(0.0, 1.0), color=PLOT_COLORS[2], alpha=0.85)
        entropy_axis.set_title("View-weight entropy distribution", fontsize=9)
        entropy_axis.set_xlabel("normalized entropy")
        entropy_axis.set_ylabel("pixels")
        entropy_axis.grid(alpha=0.2)
    else:
        show_unavailable_axis(entropy_axis, "View-weight entropy distribution")

    slot_axis = chart_axes[2]
    slots = np.arange(4)
    mean_weights = [float(sampled_finite_values(np, data).mean()) if sampled_finite_values(np, data).size else 0.0 for data in view_weights]
    mean_costs = [float(sampled_finite_values(np, data).mean()) if sampled_finite_values(np, data).size else 0.0 for data in view_costs]
    if any(data is not None for data in view_weights):
        slot_axis.bar(slots, mean_weights, width=0.55, color=PLOT_COLORS[2], label="mean weight")
        cost_axis = slot_axis.twinx()
        cost_axis.plot(slots, mean_costs, marker="o", linewidth=1.8, color=PLOT_COLORS[1], label="mean cost")
        slot_axis.set_title("Per-slot weight and cost", fontsize=9)
        slot_axis.set_xlabel("source-view slot")
        slot_axis.set_ylabel("mean weight")
        cost_axis.set_ylabel("mean cost")
        slot_axis.set_xticks(slots)
        slot_axis.grid(axis="y", alpha=0.2)
    else:
        show_unavailable_axis(slot_axis, "Per-slot weight and cost")

    churn_axis = chart_axes[3]
    churn_maps = record.pass_maps.get("view_churn", {})
    loaded_churn = {
        pass_index: data
        for pass_index, path in sorted(churn_maps.items())
        if (data := load_pass_map(path, np, plt)) is not None
    }
    churn_series = []
    for iteration, data in combine_checkerboard_maps(loaded_churn, np).items():
        values = sampled_finite_values(np, data)
        churn_series.append((iteration, float(values.mean()) if values.size else 0.0))
    if churn_series:
        churn_axis.plot([row[0] for row in churn_series], [row[1] for row in churn_series], marker="o", linewidth=2.0, color=PLOT_COLORS[3])
        churn_axis.set_xticks([row[0] for row in churn_series])
        churn_axis.set_xticklabels(["init" if row[0] < 0 else str(row[0] + 1) for row in churn_series])
        churn_axis.set_title("Mean view churn by iteration", fontsize=9)
        churn_axis.set_xlabel("iteration")
        churn_axis.set_ylabel("views added/removed")
        churn_axis.grid(alpha=0.2)
    else:
        show_unavailable_axis(churn_axis, "Mean view churn by iteration")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def make_frame_panels(runs: list[RunData], assets: Path, plt) -> None:
    np = try_import_numpy()
    panel_root = assets / "frame_panels"
    include_run_label = len(runs) > 1
    for run in runs:
        for record in run.depthmaps:
            out = panel_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_overview.png"
            record.panel_path = save_frame_panel(plt, run, record, out, include_run_label)
            if np is not None:
                cost_out = panel_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_cost_function.png"
                record.cost_panel_path = save_cost_function_panel(plt, np, run, record, cost_out)
                view_out = panel_root / safe_asset_part(run.label) / f"{safe_asset_part(record.directory.name)}_view_selection.png"
                record.view_panel_path = save_view_selection_panel(plt, np, run, record, view_out)


def read_png_u8(plt, np, path: Path | None):
    if path is None or not path.is_file():
        return None
    try:
        data = plt.imread(str(path))
    except Exception:
        return None
    if data.ndim == 3:
        data = data[..., 0]
    if np.issubdtype(data.dtype, np.floating):
        data = np.rint(np.clip(data, 0.0, 1.0) * 255.0)
    return data.astype(np.uint8)


def decode_last_changed_iterations(record: DepthMapRecord, np, last_changed_u8):
    if last_changed_u8 is None:
        return None
    logical_rows = record_logical_iterations(record)
    raw_passes = [to_int(row.get("pass_index"), -1) for row in record.iterations]
    num_raw_passes = max(raw_passes, default=-1) + 1
    num_stages = len(logical_rows)
    if num_raw_passes <= 0 or num_stages <= 0:
        return last_changed_u8.astype(float) / 255.0
    encoded = last_changed_u8.astype(float)
    raw_index = np.rint(encoded * num_raw_passes / 255.0).astype(int) - 1
    stage_index = np.where(raw_index <= 0, 0, (raw_index + 1) // 2)
    return np.where(encoded > 0, (stage_index + 1) / float(num_stages), 0.0)


def read_record_diagnostics(plt, np, record: DepthMapRecord) -> dict[str, Any]:
    def read_map(name: str):
        path = record.pfm_maps.get(name)
        if path is None:
            return None
        try:
            return read_pfm(path, np)
        except Exception:
            return None

    cost = read_map("cost_final.pfm")
    depth = read_map("depth_final_after_filter.pfm")
    normal = read_map("normal_final.pfm")
    valid_u8 = read_png_u8(plt, np, record.png_maps.get("valid_after_filter.png"))
    if valid_u8 is not None:
        valid = valid_u8 > 127
    elif depth is not None:
        valid = np.isfinite(depth) & (depth > 0.0)
    else:
        valid = None
    candidate_u8 = read_png_u8(plt, np, record.png_maps.get("candidate_source.png"))
    support_u8 = read_png_u8(plt, np, record.png_maps.get("num_supporting_views.png"))
    last_changed_u8 = read_png_u8(plt, np, record.png_maps.get("last_changed_iter.png"))
    last_changed = decode_last_changed_iterations(record, np, last_changed_u8)
    return {
        "cost": cost,
        "depth": depth,
        "normal": normal,
        "valid": valid,
        "candidate": None if candidate_u8 is None else np.clip(np.rint(candidate_u8.astype(float) / 28.0), 0, 8).astype(int),
        "support": None if support_u8 is None else np.clip(np.rint(support_u8.astype(float) / 64.0), 0, 4).astype(int),
        "last_changed": last_changed,
    }


def finite_percentile(np, data, percentile: float, default: float = 0.0) -> float:
    if data is None:
        return default
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    if not values.size:
        return default
    return float(np.percentile(values, percentile))


def robust_shared_limits(np, arrays: list[Any], low_percentile: float, high_percentile: float, default: tuple[float, float]) -> tuple[float, float]:
    values = []
    for data in arrays:
        if data is None:
            continue
        finite = np.asarray(data)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            if finite.size > 250000:
                finite = finite[:: int(math.ceil(finite.size / 250000))]
            values.append(finite)
    if not values:
        return default
    combined = np.concatenate(values)
    lo, hi = np.percentile(combined, [low_percentile, high_percentile])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        return default
    return float(lo), float(hi)


def symmetric_limit(np, data, percentile: float = 99.0, default: float = 1.0) -> float:
    if data is None:
        return default
    values = np.abs(np.asarray(data))
    values = values[np.isfinite(values)]
    if not values.size:
        return default
    limit = float(np.percentile(values, percentile))
    return limit if math.isfinite(limit) and limit > 1e-8 else default


def show_unavailable_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f3f4f6")
    ax.text(0.5, 0.5, "unavailable", ha="center", va="center", color="#6b7280", fontsize=8)


def add_horizontal_colorbar(fig, artist, axes, label: str, ticks=None, ticklabels=None) -> None:
    if artist is None:
        return
    colorbar = fig.colorbar(artist, ax=axes, orientation="horizontal", fraction=0.055, pad=0.025, aspect=28, ticks=ticks)
    colorbar.set_label(label, fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)
    if ticklabels is not None:
        colorbar.ax.set_xticklabels(ticklabels, rotation=25, ha="right")


def plot_scalar_triplet(fig, plt, np, axes, baseline, modified, delta, titles: tuple[str, str, str], cmap_name: str, limits: tuple[float, float], delta_limit: float, value_label: str, delta_label: str) -> None:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#15181d")
    norm = Normalize(vmin=limits[0], vmax=limits[1])
    shared_artist = None
    for ax, data, title in zip(axes[:2], (baseline, modified), titles[:2]):
        if data is None:
            show_unavailable_axis(ax, title)
            continue
        shared_artist = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    delta_artist = None
    if delta is None:
        show_unavailable_axis(axes[2], titles[2])
    else:
        delta_cmap = plt.get_cmap("RdYlGn" if "better" in delta_label else "coolwarm").copy()
        delta_cmap.set_bad("#15181d")
        delta_norm = TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit)
        delta_artist = axes[2].imshow(np.ma.masked_invalid(delta), cmap=delta_cmap, norm=delta_norm, interpolation="nearest")
        axes[2].set_title(titles[2], fontsize=8)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    add_horizontal_colorbar(fig, shared_artist, axes[:2], value_label)
    add_horizontal_colorbar(fig, delta_artist, [axes[2]], delta_label)


def plot_categorical_triplet(fig, plt, axes, baseline, modified, transition, titles: tuple[str, str, str], colors: list[str], labels: list[str], transition_colors: list[str], transition_labels: list[str]) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(colors)
    norm = BoundaryNorm([value - 0.5 for value in range(len(colors) + 1)], cmap.N)
    shared_artist = None
    for ax, data, title in zip(axes[:2], (baseline, modified), titles[:2]):
        if data is None:
            show_unavailable_axis(ax, title)
            continue
        shared_artist = ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    transition_artist = None
    if transition is None:
        show_unavailable_axis(axes[2], titles[2])
    else:
        transition_cmap = ListedColormap(transition_colors)
        transition_norm = BoundaryNorm([value - 0.5 for value in range(len(transition_colors) + 1)], transition_cmap.N)
        transition_artist = axes[2].imshow(transition, cmap=transition_cmap, norm=transition_norm, interpolation="nearest")
        axes[2].set_title(titles[2], fontsize=8)
        axes[2].set_xticks([])
        axes[2].set_yticks([])
    add_horizontal_colorbar(fig, shared_artist, axes[:2], "category", list(range(len(labels))), labels)
    add_horizontal_colorbar(fig, transition_artist, [axes[2]], "transition", list(range(len(transition_labels))), transition_labels)


def normal_rgb(np, normal, valid):
    if normal is None or normal.ndim != 3 or normal.shape[2] < 3:
        return None
    lengths = np.linalg.norm(normal[..., :3], axis=2)
    usable = np.isfinite(lengths) & (lengths > 1e-8)
    if valid is not None:
        usable &= valid
    unit = np.zeros_like(normal[..., :3], dtype=float)
    unit[usable] = normal[..., :3][usable] / lengths[usable, None]
    rgb = np.clip(unit * 0.5 + 0.5, 0.0, 1.0)
    rgb[~usable] = 0.0
    return rgb


def normal_angle_delta(np, baseline_normal, modified_normal, common_valid):
    if baseline_normal is None or modified_normal is None or common_valid is None:
        return None
    baseline_length = np.linalg.norm(baseline_normal[..., :3], axis=2)
    modified_length = np.linalg.norm(modified_normal[..., :3], axis=2)
    usable = common_valid & np.isfinite(baseline_length) & np.isfinite(modified_length) & (baseline_length > 1e-8) & (modified_length > 1e-8)
    result = np.full(common_valid.shape, np.nan, dtype=float)
    dot = np.sum(baseline_normal[..., :3] * modified_normal[..., :3], axis=2)
    denom = baseline_length * modified_length
    result[usable] = np.degrees(np.arccos(np.clip(dot[usable] / denom[usable], -1.0, 1.0)))
    return result


def plot_cdf(ax, np, series: list[tuple[str, Any, str]], title: str, xlabel: str, xlimit: float | None = None) -> None:
    for label, data, color in series:
        if data is None:
            continue
        values = np.asarray(data)
        values = np.sort(values[np.isfinite(values)])
        if not values.size:
            continue
        cdf = np.arange(1, values.size + 1, dtype=float) / values.size
        ax.plot(values, cdf, label=label, color=color, linewidth=1.8)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel("CDF", fontsize=7)
    if xlimit is not None and xlimit > 0.0:
        ax.set_xlim(0.0, xlimit)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=6)
    if ax.lines:
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_comparison_diagnostic_panel(plt, np, baseline_run: RunData, baseline_record: DepthMapRecord, modified_run: RunData, modified_record: DepthMapRecord, out: Path) -> Path | None:
    baseline = read_record_diagnostics(plt, np, baseline_record)
    modified = read_record_diagnostics(plt, np, modified_record)
    baseline_valid = baseline["valid"]
    modified_valid = modified["valid"]
    if baseline_valid is None or modified_valid is None or baseline_valid.shape != modified_valid.shape:
        return None
    common_valid = baseline_valid & modified_valid

    baseline_cost = None if baseline["cost"] is None else np.where(baseline_valid & np.isfinite(baseline["cost"]), baseline["cost"], np.nan)
    modified_cost = None if modified["cost"] is None else np.where(modified_valid & np.isfinite(modified["cost"]), modified["cost"], np.nan)
    cost_delta = None if baseline_cost is None or modified_cost is None else np.where(common_valid, baseline_cost - modified_cost, np.nan)
    _, cost_hi = robust_shared_limits(np, [baseline_cost, modified_cost], 0.0, 99.5, (0.0, 1.2))
    cost_hi = max(cost_hi, 1e-4)
    cost_delta_limit = symmetric_limit(np, cost_delta, 99.0, max(cost_hi * 0.25, 1e-3))

    baseline_depth = None if baseline["depth"] is None else np.where(baseline_valid & np.isfinite(baseline["depth"]), baseline["depth"], np.nan)
    modified_depth = None if modified["depth"] is None else np.where(modified_valid & np.isfinite(modified["depth"]), modified["depth"], np.nan)
    depth_limits = robust_shared_limits(np, [baseline_depth, modified_depth], 1.0, 99.0, (0.0, 1.0))
    relative_depth_delta = None
    if baseline_depth is not None and modified_depth is not None:
        relative_depth_delta = np.where(common_valid, (modified_depth - baseline_depth) / np.maximum(np.abs(baseline_depth), 1e-6), np.nan)
    relative_depth_limit = symmetric_limit(np, relative_depth_delta, 99.0, 0.1)

    baseline_normal_rgb = normal_rgb(np, baseline["normal"], baseline_valid)
    modified_normal_rgb = normal_rgb(np, modified["normal"], modified_valid)
    angle_delta = normal_angle_delta(np, baseline["normal"], modified["normal"], common_valid)
    angle_limit = finite_percentile(np, angle_delta, 99.0, 90.0)
    angle_limit = max(angle_limit, 1.0)

    valid_transition = np.zeros(baseline_valid.shape, dtype=int)
    valid_transition[baseline_valid & modified_valid] = 1
    valid_transition[~baseline_valid & modified_valid] = 2
    valid_transition[baseline_valid & ~modified_valid] = 3

    support_delta = None
    if baseline["support"] is not None and modified["support"] is not None:
        support_delta = modified["support"].astype(float) - baseline["support"].astype(float)
    support_delta_limit = symmetric_limit(np, support_delta, 100.0, 4.0)

    source_transition = None
    if baseline["candidate"] is not None and modified["candidate"] is not None:
        source_transition = np.zeros(baseline_valid.shape, dtype=int)
        source_transition[common_valid & (baseline["candidate"] != modified["candidate"])] = 1
        source_transition[~baseline_valid & modified_valid] = 2
        source_transition[baseline_valid & ~modified_valid] = 3

    last_delta = None
    if baseline["last_changed"] is not None and modified["last_changed"] is not None:
        last_delta = modified["last_changed"] - baseline["last_changed"]
    last_delta_limit = symmetric_limit(np, last_delta, 99.0, 1.0)

    fig = plt.figure(figsize=(20, 21), constrained_layout=True)
    fig.suptitle(
        f"{frame_label(baseline_record)} depth-map diagnostics: {baseline_run.label} vs {modified_run.label}",
        fontsize=16,
        fontweight="bold",
    )
    outer = fig.add_gridspec(5, 2, height_ratios=(0.78, 1.0, 1.0, 1.0, 1.0))
    top = outer[0, :].subgridspec(1, 3, width_ratios=(1.0, 1.35, 1.35))
    reference_axis = fig.add_subplot(top[0, 0])
    display_image_axis(plt, reference_axis, modified_record.reference_thumbnail or baseline_record.reference_thumbnail or modified_record.reference_image or baseline_record.reference_image, "Reference RGB")
    summary_axis = fig.add_subplot(top[0, 1:])
    summary_axis.axis("off")

    gained = int(np.count_nonzero(~baseline_valid & modified_valid))
    lost = int(np.count_nonzero(baseline_valid & ~modified_valid))
    finite_cost_delta = None if cost_delta is None else cost_delta[np.isfinite(cost_delta)]
    lower_cost_ratio = float(np.count_nonzero(finite_cost_delta > 0.0) / finite_cost_delta.size) if finite_cost_delta is not None and finite_cost_delta.size else 0.0
    support_increased = int(np.count_nonzero(support_delta > 0.0)) if support_delta is not None else 0
    support_decreased = int(np.count_nonzero(support_delta < 0.0)) if support_delta is not None else 0
    absolute_relative_depth_delta = None if relative_depth_delta is None else np.abs(relative_depth_delta)
    baseline_final_cost = baseline_record.summary.get("final_cost", {})
    modified_final_cost = modified_record.summary.get("final_cost", {})
    summary_lines = [
        f"valid after       {fmt_pct(baseline_record.summary.get('valid_ratio_after_filter'))} -> {fmt_pct(modified_record.summary.get('valid_ratio_after_filter'))}",
        f"rejected pixels   {to_int(baseline_record.summary.get('num_rejected_by_filter')):,} -> {to_int(modified_record.summary.get('num_rejected_by_filter')):,}",
        f"gained / lost     {gained:,} / {lost:,}",
        f"median cost       {fmt_float(baseline_final_cost.get('median'))} -> {fmt_float(modified_final_cost.get('median'))}",
        f"p90 cost          {fmt_float(baseline_final_cost.get('p90'))} -> {fmt_float(modified_final_cost.get('p90'))}",
        f"lower cost        {lower_cost_ratio * 100.0:.2f}% of common-valid pixels",
        f"|relative depth|  p50 {finite_percentile(np, absolute_relative_depth_delta, 50.0):.4f}, p90 {finite_percentile(np, absolute_relative_depth_delta, 90.0):.4f}",
        f"normal angle      p50 {finite_percentile(np, angle_delta, 50.0):.2f} deg, p90 {finite_percentile(np, angle_delta, 90.0):.2f} deg",
        f"support +/-       {support_increased:,} / {support_decreased:,} pixels",
        "depth/normal deltas measure run disagreement, not ground-truth error",
    ]
    summary_axis.set_title("Comparison summary", fontsize=11, loc="left")
    summary_axis.text(0.01, 0.97, "\n".join(summary_lines), va="top", ha="left", family="monospace", fontsize=9, color="#293241")

    def group_axes(row: int, column: int):
        grid = outer[row, column].subgridspec(1, 3, wspace=0.04)
        return [fig.add_subplot(grid[0, index]) for index in range(3)]

    plot_scalar_triplet(
        fig, plt, np, group_axes(1, 0), baseline_cost, modified_cost, cost_delta,
        (f"{baseline_run.label}: final cost", f"{modified_run.label}: final cost", "cost delta"),
        "magma", (0.0, cost_hi), cost_delta_limit, "aggregate cost", "baseline - modified; positive = better",
    )
    plot_scalar_triplet(
        fig, plt, np, group_axes(1, 1), baseline_depth, modified_depth, relative_depth_delta,
        (f"{baseline_run.label}: final depth", f"{modified_run.label}: final depth", "relative depth delta"),
        "viridis", depth_limits, relative_depth_limit, "depth", "(modified - baseline) / baseline",
    )

    normal_axes = group_axes(2, 0)
    for ax, data, title in zip(normal_axes[:2], (baseline_normal_rgb, modified_normal_rgb), (f"{baseline_run.label}: normal", f"{modified_run.label}: normal")):
        if data is None:
            show_unavailable_axis(ax, title)
        else:
            ax.imshow(data, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    if angle_delta is None:
        show_unavailable_axis(normal_axes[2], "normal-angle delta")
    else:
        angle_cmap = plt.get_cmap("magma").copy()
        angle_cmap.set_bad("#15181d")
        angle_artist = normal_axes[2].imshow(np.ma.masked_invalid(angle_delta), cmap=angle_cmap, vmin=0.0, vmax=angle_limit, interpolation="nearest")
        normal_axes[2].set_title("normal-angle delta", fontsize=8)
        normal_axes[2].set_xticks([])
        normal_axes[2].set_yticks([])
        add_horizontal_colorbar(fig, angle_artist, [normal_axes[2]], "degrees")

    plot_categorical_triplet(
        fig, plt, group_axes(2, 1), baseline_valid.astype(int), modified_valid.astype(int), valid_transition,
        (f"{baseline_run.label}: valid after", f"{modified_run.label}: valid after", "filter transition"),
        ["#20242a", "#d8e2e8"], ["rejected", "valid"],
        ["#20242a", "#b8c5cc", "#2a9d8f", "#e76f51"], ["invalid both", "valid both", "gained", "lost"],
    )

    plot_scalar_triplet(
        fig, plt, np, group_axes(3, 0), baseline["support"], modified["support"], support_delta,
        (f"{baseline_run.label}: support", f"{modified_run.label}: support", "support delta"),
        "cividis", (0.0, 4.0), max(support_delta_limit, 1.0), "supporting views", "modified - baseline views",
    )
    candidate_colors = ["#20242a", "#457b9d", "#8ecae6", "#2a9d8f", "#90be6d", "#f4a261", "#e76f51", "#9b5de5", "#adb5bd"]
    plot_categorical_triplet(
        fig, plt, group_axes(3, 1), baseline["candidate"], modified["candidate"], source_transition,
        (f"{baseline_run.label}: source", f"{modified_run.label}: source", "source transition"),
        candidate_colors, ["unknown", "init", "init existing", "spatial", "view", "random", "refine", "prior", "other"],
        ["#5f6b73", "#e9c46a", "#2a9d8f", "#e76f51"], ["same", "changed", "gained", "lost"],
    )

    plot_scalar_triplet(
        fig, plt, np, group_axes(4, 0), baseline["last_changed"], modified["last_changed"], last_delta,
        (f"{baseline_run.label}: last update", f"{modified_run.label}: last update", "normalized update delta"),
        "cividis", (0.0, 1.0), max(last_delta_limit, 0.05), "normalized iteration position", "modified - baseline position",
    )

    distribution_axes = group_axes(4, 1)
    plot_cdf(
        distribution_axes[0], np,
        [(baseline_run.label, baseline_cost, PLOT_COLORS[0]), (modified_run.label, modified_cost, PLOT_COLORS[1])],
        "final-cost CDF", "cost", cost_hi,
    )
    plot_cdf(
        distribution_axes[1], np,
        [("absolute relative delta", None if relative_depth_delta is None else np.abs(relative_depth_delta), PLOT_COLORS[2])],
        "depth disagreement CDF", "absolute relative delta", relative_depth_limit,
    )
    transition_axis = distribution_axes[2]
    if baseline["support"] is None or modified["support"] is None:
        show_unavailable_axis(transition_axis, "support transitions")
    else:
        transition = np.zeros((5, 5), dtype=float)
        for before, after in zip(baseline["support"].ravel(), modified["support"].ravel()):
            transition[int(before), int(after)] += 1.0
        transition /= max(float(transition.sum()), 1.0)
        artist = transition_axis.imshow(transition, cmap="Blues", vmin=0.0, vmax=max(float(transition.max()), 1e-6))
        transition_axis.set_title("support transition share", fontsize=8)
        transition_axis.set_xlabel(modified_run.label, fontsize=7)
        transition_axis.set_ylabel(baseline_run.label, fontsize=7)
        transition_axis.set_xticks(range(5))
        transition_axis.set_yticks(range(5))
        transition_axis.tick_params(labelsize=6)
        for before in range(5):
            for after in range(5):
                share = transition[before, after]
                if share >= 0.005:
                    transition_axis.text(after, before, f"{share * 100.0:.1f}%", ha="center", va="center", fontsize=5, color="white" if share > transition.max() * 0.5 else "#1f2933")
        add_horizontal_colorbar(fig, artist, [transition_axis], "pixel share")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=165)
    plt.close(fig)
    return out


def make_diagnostic_panels(runs: list[RunData], assets: Path, plt) -> list[DiagnosticPanel]:
    np = try_import_numpy()
    if plt is None or np is None:
        return []
    grouped: dict[str, list[tuple[RunData, DepthMapRecord]]] = defaultdict(list)
    for run in runs:
        for record in run.depthmaps:
            grouped[image_id_key(record)].append((run, record))
    panels: list[DiagnosticPanel] = []
    panel_root = assets / "diagnostic_panels"
    for frame_key in sorted(grouped, key=lambda key: (to_int(key, 1 << 30), key)):
        records = grouped[frame_key]
        if len(records) > 1:
            baseline_run, baseline_record = records[0]
            for modified_run, modified_record in records[1:]:
                out = panel_root / f"{safe_asset_part(baseline_record.directory.name)}_{safe_asset_part(baseline_run.label)}_vs_{safe_asset_part(modified_run.label)}.png"
                path = save_comparison_diagnostic_panel(plt, np, baseline_run, baseline_record, modified_run, modified_record, out)
                if path:
                    panels.append(
                        DiagnosticPanel(
                            frame_key=frame_key,
                            title=f"{frame_label(baseline_record)}: {baseline_run.label} vs {modified_run.label}",
                            path=path,
                            records=[(baseline_run, baseline_record), (modified_run, modified_record)],
                            mechanism="comparison",
                            caption="Paired final-state comparison with shared scales, spatial deltas, transitions, and distribution diagnostics.",
                        )
                    )
        for run, record in records:
            paths: list[tuple[str, Path | None, str, str]] = [
                (f"{run.label} overview", record.panel_path, "overview", "Final state, filtering, support, candidate/update state, components, and convergence summary."),
                (f"{run.label} cost-function dashboard", record.cost_panel_path, "cost", "Cost formation, component closure, distributions, spatial maps, and iteration-by-iteration evolution."),
                (
                    f"{run.label} logical-iteration cost evolution",
                    record.logical_cost_panel_path,
                    "cost",
                    "Stored production cost/confidence and equal-selected-view component rescoring at initialization and every complete logical iteration. Exact, derived, and proxy signals are labeled explicitly.",
                ),
                (f"{run.label} view-selection dashboard", record.view_panel_path, "view", "Support, entropy, per-view weights and cost contributions, and selection churn."),
                (f"{run.label} cost-improvement iterations", record.improvement_panel_path, "cost", "Spatial cost reduction for initialization and every logical PatchMatch iteration using a shared robust scale."),
            ]
            paths.extend(
                (
                    f"{run.label} {signal.replace('_', ' ')} iterations",
                    path,
                    "view" if signal == "view_churn" else "update",
                    "Per-iteration view-set changes." if signal == "view_churn" else "Per-iteration hypothesis movement with a shared robust scale.",
                )
                for signal, path in sorted(record.pass_panel_paths.items())
            )
            for title, path, mechanism, caption in paths:
                if path is not None:
                    panels.append(
                        DiagnosticPanel(
                            frame_key=frame_key,
                            title=f"{frame_label(record)}: {title}",
                            path=path,
                            records=[(run, record)],
                            mechanism=mechanism,
                            caption=caption,
                        )
                    )
    return panels


def aggregate_filter_metric(run: RunData, key: str) -> float:
    total = 0.0
    count = 0
    for record in run.depthmaps:
        value = to_float(record.filtering.get(key))
        if value is None:
            value = to_float(record.summary.get(key))
        if value is not None:
            total += value
            count += 1
    return total / count if count else 0.0


def make_plots(runs: list[RunData], assets: Path) -> tuple[list[tuple[str, Path]], list[DiagnosticPanel]]:
    plt = try_import_matplotlib()
    if plt is None:
        return [], []
    assets.mkdir(parents=True, exist_ok=True)
    make_reference_thumbnails(runs, assets)
    make_map_previews(runs, assets, plt)
    make_improvement_previews(runs, assets, plt)
    make_pass_map_previews(runs, assets, plt)
    make_logical_cost_previews(runs, assets, plt)
    make_frame_panels(runs, assets, plt)
    diagnostic_panels = make_diagnostic_panels(runs, assets, plt)
    plots: list[tuple[str, Path]] = []
    include_run_label = len(runs) > 1

    labels = [run.label for run in runs]
    plot = save_bar_plot(
        plt,
        assets / "valid_and_rejected_ratios.png",
        "Depth-map validity and rejection ratios",
        "ratio",
        labels,
        {
            "valid before filter": [aggregate_filter_metric(run, "valid_ratio_before_filter") for run in runs],
            "valid after filter": [aggregate_filter_metric(run, "valid_ratio_after_filter") for run in runs],
            "rejected by filter": [aggregate_filter_metric(run, "rejected_by_filter_ratio") for run in runs],
        },
    )
    if plot:
        plots.append(("Valid/rejected ratios", plot))

    image_labels = [str(i) for i in range(max((len(run.depthmaps) for run in runs), default=0))]
    for run in runs:
        if len(run.depthmaps) == len(image_labels):
            image_labels = image_labels_for_run(run)
            break
    medians: dict[str, list[float]] = {}
    p90s: dict[str, list[float]] = {}
    for run in runs:
        medians[run.label] = []
        p90s[run.label] = []
        for idx in range(len(image_labels)):
            if idx < len(run.depthmaps):
                final_cost = run.depthmaps[idx].summary.get("final_cost", {})
                medians[run.label].append(to_float(final_cost.get("median"), 0.0) or 0.0)
                p90s[run.label].append(to_float(final_cost.get("p90"), 0.0) or 0.0)
            else:
                medians[run.label].append(0.0)
                p90s[run.label].append(0.0)
    plot = save_bar_plot(plt, assets / "final_cost_median_by_frame.png", "Final cost median by frame", "cost", image_labels, medians)
    if plot:
        plots.append(("Median final cost", plot))
    plot = save_bar_plot(plt, assets / "final_cost_p90_by_frame.png", "Final cost p90 by frame", "cost", image_labels, p90s)
    if plot:
        plots.append(("P90 final cost", plot))

    changed_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    cost_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    acceptance_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for run in runs:
        # Keep unavailable cohorts in the plot inventory so the chart states
        # that evidence is unavailable instead of silently disappearing.
        changed_series[run.label]
        acceptance_series[run.label]
        by_iteration_changed: dict[int, list[float]] = defaultdict(list)
        by_iteration_cost: dict[int, list[float]] = defaultdict(list)
        by_iteration_acceptance: dict[int, list[float]] = defaultdict(list)
        for record in run.depthmaps:
            for row in record_logical_iterations(record):
                iteration = to_int(row.get("logical_iteration"), -1)
                for source, target in (
                    (to_float(row.get("changed_ratio")), by_iteration_changed),
                    (to_float(row.get("acceptance_rate")), by_iteration_acceptance),
                ):
                    if source is not None:
                        target[iteration].append(source)
                cost = to_float(row.get("median_cost"))
                if cost is None:
                    cost = to_float(row.get("mean_cost"))
                if cost is not None:
                    by_iteration_cost[iteration].append(cost)
        for iteration, values in by_iteration_changed.items():
            changed_series[run.label].append((iteration, sum(values) / len(values)))
        for iteration, values in by_iteration_cost.items():
            cost_series[run.label].append((iteration, sum(values) / len(values)))
        for iteration, values in by_iteration_acceptance.items():
            acceptance_series[run.label].append((iteration, sum(values) / len(values)))
    for title, ylabel, name, series in (
        ("Changed ratio by PatchMatch iteration", "changed ratio", "changed_ratio_by_iteration.png", changed_series),
        ("Cost by PatchMatch iteration", "cost", "cost_by_iteration.png", cost_series),
        ("Candidate acceptance by PatchMatch iteration", "acceptance rate", "acceptance_rate_by_iteration.png", acceptance_series),
    ):
        plot = save_line_plot(plt, assets / name, title, ylabel, series)
        if plot:
            plots.append((title, plot))

    candidate_counts, candidate_rates = aggregate_candidate_data(runs)
    candidate_labels = sorted({key for counts in candidate_counts.values() for key in counts})
    plot = save_bar_plot(
        plt,
        assets / "candidate_source_distribution.png",
        "Accepted candidate source distribution",
        "accepted count",
        candidate_labels,
        {run.label: [float(candidate_counts[run.label][key]) for key in candidate_labels] for run in runs},
    )
    if plot:
        plots.append(("Candidate source distribution", plot))
    rate_labels = sorted({key for rates in candidate_rates.values() for key in rates})
    plot = save_bar_plot(
        plt,
        assets / "candidate_acceptance_rate.png",
        "Candidate acceptance rate by type",
        "acceptance rate",
        rate_labels,
        {run.label: [float(candidate_rates[run.label].get(key, 0.0)) for key in rate_labels] for run in runs},
    )
    if plot:
        plots.append(("Candidate acceptance rate", plot))

    reason_series: dict[str, list[float]] = {}
    for run in runs:
        counts = Counter()
        for record in run.depthmaps:
            counts.update({key: to_int(value) for key, value in record.filtering.get("rejection_reasons", {}).items()})
        reason_series[run.label] = [float(counts[key]) for key in REJECTION_KEYS]
    plot = save_bar_plot(plt, assets / "rejection_reason_distribution.png", "Filtering rejection reason distribution", "pixels", list(REJECTION_KEYS), reason_series)
    if plot:
        plots.append(("Rejection reason distribution", plot))

    support_labels = ["0", "1", "2", "3", "4"]
    support_series: dict[str, list[float]] = {}
    for run in runs:
        counts = Counter()
        for record in run.depthmaps:
            for row in record.view_support:
                counts[str(to_int(row.get("supporting_view_count")))] += to_int(row.get("pixels"))
        support_series[run.label] = [float(counts[key]) for key in support_labels]
    plot = save_bar_plot(plt, assets / "supporting_view_histogram.png", "Supporting-view count histogram", "pixels", support_labels, support_series)
    if plot:
        plots.append(("Supporting-view histogram", plot))

    for run in runs:
        add_per_frame_plots(run, assets, plots, plt, include_run_label)
    return plots, diagnostic_panels


def safe_asset_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe.strip("._") or "artifact"


def copy_report_file(src: Path, dst: Path) -> Path | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def prepare_assets_dir(assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    for generated_dir in ("map_previews", "reference_images", "improvement_previews", "pass_map_previews", "logical_cost_previews", "diagnostic_panels", "per_frame", "frame_panels", "artifacts"):
        path = assets / generated_dir
        if path.exists():
            shutil.rmtree(path)
    for generated_plot in assets.glob("*.png"):
        generated_plot.unlink()


def bundle_report_artifacts(runs: list[RunData], assets: Path) -> None:
    data_root = assets / "artifacts"
    for run in runs:
        run_root = data_root / safe_asset_part(run.label)
        for name in ("run_metadata.json", "scene_summary.json"):
            copied = copy_report_file(run.path / name, run_root / name)
            if copied:
                run.report_files[name] = copied
        for name in ("counters.csv", "timings.csv", "traces.jsonl"):
            copied = copy_report_file(run.path / "instrumentation" / name, run_root / "instrumentation" / name)
            if copied:
                run.report_files[f"instrumentation/{name}"] = copied
        for record in run.depthmaps:
            depthmap_root = run_root / "depthmaps" / safe_asset_part(record.directory.name)
            for name in ("summary.json", "filtering.json", "iteration.csv", "view_support.csv", "map_manifest.json"):
                copied = copy_report_file(record.directory / name, depthmap_root / name)
                if copied:
                    record.report_files[name] = copied
            maps_root = depthmap_root / "maps"
            bundled_png_maps: dict[str, Path] = {}
            for name, path in record.png_maps.items():
                bundled_png_maps[name] = copy_report_file(path, maps_root / name) or path
            record.png_maps = bundled_png_maps
            bundled_pfm_maps: dict[str, Path] = {}
            for name, path in record.pfm_maps.items():
                bundled_pfm_maps[name] = copy_report_file(path, maps_root / name) or path
            record.pfm_maps = bundled_pfm_maps
            for artifact in record.map_artifacts:
                if not artifact.path.is_file():
                    continue
                try:
                    relative = artifact.path.relative_to(record.directory.resolve())
                except ValueError:
                    relative = Path("maps") / artifact.path.name
                copy_report_file(artifact.path, depthmap_root / relative)


def aggregate_candidate_data(runs: list[RunData]):
    candidate_counts: dict[str, Counter[str]] = {}
    candidate_rates: dict[str, dict[str, float]] = {}
    for run in runs:
        counts: Counter[str] = Counter()
        tested: Counter[str] = Counter()
        accepted: Counter[str] = Counter()
        for record in run.depthmaps:
            if not candidate_accounting_available(record):
                continue
            for row in record.summary.get("candidate_acceptance", []):
                ctype = str(row.get("candidate_type", "UNKNOWN"))
                counts[ctype] += to_int(row.get("accepted_count"))
                tested[ctype] += to_int(row.get("tested_count"))
                accepted[ctype] += to_int(row.get("accepted_count"))
        candidate_counts[run.label] = counts
        candidate_rates[run.label] = {key: safe_div(accepted[key], tested[key]) for key in tested}
    return candidate_counts, candidate_rates


def html_table(headers: list[str], rows: list[list[Any]], css_class: str = "") -> str:
    cls = f" class='{css_class}'" if css_class else ""
    parts = [f"<table{cls}>", "<thead><tr>"]
    parts += [f"<th>{html.escape(header)}</th>" for header in headers]
    parts += ["</tr></thead><tbody>"]
    for row in rows:
        parts.append("<tr>")
        parts += [f"<td>{cell}</td>" if isinstance(cell, Html) else f"<td>{html.escape(str(cell))}</td>" for cell in row]
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


class Html(str):
    pass


class Md(str):
    pass


def link_href(path: Path, out: Path) -> str:
    try:
        href = path.relative_to(out.parent).as_posix()
    except ValueError:
        href = path.as_uri()
    return quote(href, safe="/:#?=&%")


def rel_link(path: Path, out: Path, label: str | None = None) -> Html:
    label = label or path.name
    href = link_href(path, out)
    return Html(f"<a href='{html.escape(str(href))}'>{html.escape(label)}</a>")


def image_tag(path: Path, out: Path, alt: str) -> Html:
    src = link_href(path, out)
    return Html(f"<img src='{html.escape(str(src))}' alt='{html.escape(alt)}'>")


def md_label(text: Any) -> str:
    return str(text).replace("[", "\\[").replace("]", "\\]")


def md_link(path: Path, out: Path, label: str | None = None) -> Md:
    label = label or path.name
    return Md(f"[{md_label(label)}]({link_href(path, out)})")


def md_image(path: Path, out: Path, alt: str) -> Md:
    return Md(f"![{md_label(alt)}]({link_href(path, out)})")


def md_cell_html(value: Any) -> str:
    if not isinstance(value, Md):
        return html.escape(str(value)).replace("\n", "<br>")
    raw = str(value)
    parts = []
    offset = 0
    for match in re.finditer(r"\[([^]]+)\]\(([^)]+)\)", raw):
        parts.append(html.escape(raw[offset:match.start()]))
        parts.append(
            f"<a href=\"{html.escape(match.group(2), quote=True)}\" "
            "style=\"color:#245b8f;font-weight:600;text-decoration:none\">"
            f"{html.escape(match.group(1))}</a>"
        )
        offset = match.end()
    parts.append(html.escape(raw[offset:]))
    return "".join(parts)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No data available._"
    parts = [
        '<div style="overflow-x:auto;margin:12px 0 20px;border:1px solid #bfd0de;border-radius:6px">',
        '<table style="border-collapse:collapse;width:100%;font-size:13px;background:#ffffff">',
        "<thead><tr>",
    ]
    header_style = "background:#d9e8f2;color:#20384b;border-bottom:2px solid #8aa8bd;padding:8px 10px;text-align:left;font-weight:700"
    parts += [f'<th style="{header_style}">{html.escape(header)}</th>' for header in headers]
    parts.append("</tr></thead><tbody>")
    for row_index, row in enumerate(rows):
        background = "#ffffff" if row_index % 2 == 0 else "#f2f7fa"
        parts.append(f'<tr style="background:{background}">')
        for column_index, cell in enumerate(row):
            accent = "border-left:3px solid #5f8dab;font-weight:600;color:#294861;" if column_index == 0 else ""
            cell_style = f"{accent}border-bottom:1px solid #d6e0e8;padding:7px 10px;text-align:left;vertical-align:top"
            parts.append(f'<td style="{cell_style}">{md_cell_html(cell)}</td>')
        parts.append("</tr>")
    parts += ["</tbody></table>", "</div>"]
    return "\n".join(parts)


def run_summary_rows(runs: list[RunData]) -> list[list[Any]]:
    rows = []
    for run in runs:
        aggregate = run.scene_summary.get("aggregate", {})
        rows.append(
            [
                run.label,
                run.path,
                len(run.depthmaps),
                to_int(aggregate.get("num_pixels_total")),
                fmt_pct(aggregate.get("valid_ratio_before_filter")),
                fmt_pct(aggregate.get("valid_ratio_after_filter")),
                fmt_pct(aggregate.get("rejected_ratio")),
                len(run.counters),
                len(run.timings),
                run.trace_count,
            ]
        )
    return rows


def per_depthmap_rows(run: RunData, out: Path) -> list[list[Any]]:
    rows = []
    for record in run.depthmaps:
        summary = record.summary
        final_cost = summary.get("final_cost", {})
        maps = len(record.png_maps) + len(record.pfm_maps)
        rows.append(
            [
                summary.get("image_id", ""),
                summary.get("safe_image_name", record.directory.name),
                summary.get("width", ""),
                summary.get("height", ""),
                fmt_pct(summary.get("valid_ratio_before_filter")),
                fmt_pct(summary.get("valid_ratio_after_filter")),
                fmt_pct(summary.get("rejected_by_filter_ratio")),
                fmt_float(final_cost.get("mean")),
                fmt_float(final_cost.get("median")),
                fmt_float(final_cost.get("p90")),
                fmt_float(final_cost.get("p95")),
                maps,
                rel_link(record.report_files.get("summary.json", record.directory / "summary.json"), out, "summary.json"),
            ]
        )
    return rows


def artifact_rows(run: RunData, out: Path, link_func) -> list[list[Any]]:
    rows = []
    for key in ("run_metadata.json", "scene_summary.json", "instrumentation/counters.csv", "instrumentation/timings.csv", "instrumentation/traces.jsonl"):
        path = run.report_files.get(key)
        if path:
            rows.append([key, link_func(path, out, Path(key).name)])
    for record in run.depthmaps:
        for key in ("summary.json", "filtering.json", "iteration.csv", "view_support.csv", "map_manifest.json"):
            path = record.report_files.get(key)
            if path:
                rows.append([f"{record.directory.name}/{key}", link_func(path, out, key)])
    return rows


def rejection_rows(run: RunData) -> list[list[Any]]:
    counts = Counter()
    for record in run.depthmaps:
        counts.update({key: to_int(value) for key, value in record.filtering.get("rejection_reasons", {}).items()})
    total = sum(counts.values())
    return [[key, counts[key], fmt_pct(safe_div(counts[key], total))] for key in REJECTION_KEYS]


def candidate_rows(run: RunData) -> list[list[Any]]:
    _, rates = aggregate_candidate_data([run])
    counts = Counter()
    tested = Counter()
    finite = Counter()
    unavailable_frames = 0
    for record in run.depthmaps:
        if not candidate_accounting_available(record):
            unavailable_frames += 1
            continue
        for row in record.summary.get("candidate_acceptance", []):
            ctype = str(row.get("candidate_type", "UNKNOWN"))
            counts[ctype] += to_int(row.get("accepted_count"))
            tested[ctype] += to_int(row.get("tested_count"))
            finite[ctype] += to_int(row.get("finite_count"))
    rows = [
        [key, tested[key], finite[key], counts[key], fmt_pct(rates[run.label].get(key))]
        for key in sorted(set(tested) | set(finite) | set(counts))
    ]
    if unavailable_frames:
        rows.append([
            f"unavailable post-pass frames ({unavailable_frames})",
            "unavailable", "unavailable", "unavailable", "unavailable",
        ])
    return rows


def iteration_rows(run: RunData) -> list[list[Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in run.depthmaps:
        for row in aggregate_logical_iteration_rows(record.iterations):
            grouped[(to_int(row.get("scale_level"), 0), to_int(row.get("logical_iteration"), -1))].append(row)
    rows = []
    for (scale, iteration), group in sorted(grouped.items()):
        accounting_available = all(candidate_accounting_available(row) for row in group)
        tested_values = [
            to_int(row.get("candidates_tested"))
            for row in group if row.get("candidates_tested") is not None
        ]
        accepted_values = [
            to_int(row.get("candidates_accepted"))
            for row in group if row.get("candidates_accepted") is not None
        ]
        finite_values = [
            to_int(row.get("candidates_finite"))
            for row in group if row.get("candidates_finite") is not None
        ]
        rows.append(
            [
                logical_iteration_label(iteration),
                scale,
                len(group),
                fmt_pct(avg(group, "valid_ratio")),
                fmt_pct(avg(group, "changed_ratio"), default="unavailable") if accounting_available else "unavailable",
                fmt_pct(avg(group, "acceptance_rate"), default="unavailable") if accounting_available else "unavailable",
                fmt_float(avg(group, "mean_cost")),
                fmt_float(avg(group, "mean_cost_delta")),
                fmt_float(avg(group, "mean_abs_depth_delta")),
                fmt_float(avg(group, "mean_normal_delta_deg")),
                sum(tested_values) if accounting_available and tested_values else "unavailable",
                sum(finite_values) if accounting_available and finite_values else "unavailable",
                sum(accepted_values) if accounting_available and accepted_values else "unavailable",
            ]
        )
    return rows


def avg(rows: list[dict[str, str]], key: str) -> float | None:
    values = [to_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def counter_rows(run: RunData) -> list[list[Any]]:
    final_by_entity: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in run.counters:
        scale = to_int(row.get("scale_number"), 0)
        iteration = logical_iteration_index(row)
        key = (str(row.get("image_id", "")), scale, iteration)
        if key not in final_by_entity or to_int(row.get("pass_index"), -1) > to_int(final_by_entity[key].get("pass_index"), -1):
            final_by_entity[key] = row
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for (_image_id, scale, iteration), row in final_by_entity.items():
        grouped[(scale, iteration)].append(row)
    rows = []
    for (scale, iteration), group in sorted(grouped.items()):
        processed = sum(to_int(row.get("processed")) for row in group)
        component_samples = sum(to_int(row.get("component_samples")) for row in group)
        rows.append(
            [
                logical_iteration_label(iteration),
                scale,
                processed,
                fmt_float(safe_div(sum(to_float(row.get("cost_sum"), 0.0) or 0.0 for row in group), processed)),
                fmt_float(safe_div(sum(to_float(row.get("photometric_cost_sum"), 0.0) or 0.0 for row in group), component_samples)),
                fmt_float(safe_div(sum(to_float(row.get("photo_prior_cost_sum"), 0.0) or 0.0 for row in group), component_samples)),
                fmt_float(safe_div(sum(to_float(row.get("geometric_cost_sum"), 0.0) or 0.0 for row in group), component_samples)),
                fmt_float(safe_div(sum(to_float(row.get("view_entropy_sum"), 0.0) or 0.0 for row in group), processed)),
                sum(to_int(row.get("low_texture")) for row in group),
                sum(to_int(row.get("bad_cost")) for row in group),
            ]
        )
    return rows


def timing_rows(run: RunData) -> list[list[Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in run.timings:
        grouped[to_int(row.get("pass_index"), -1)].append(row)
    rows = []
    for pass_index in sorted(grouped):
        group = grouped[pass_index]
        values = [to_float(row.get("kernel_ms"), 0.0) or 0.0 for row in group]
        rows.append([pass_index, group[0].get("phase", ""), group[0].get("iteration", ""), len(values), fmt_float(sum(values)), fmt_float(sum(values) / len(values))])
    return rows


def view_support_rows(run: RunData) -> list[list[Any]]:
    counts = Counter()
    for record in run.depthmaps:
        for row in record.view_support:
            counts[to_int(row.get("supporting_view_count"))] += to_int(row.get("pixels"))
    total = sum(counts.values())
    return [[support, counts[support], fmt_pct(safe_div(counts[support], total))] for support in sorted(counts)]


def diagnostic_summary_rows(panel: DiagnosticPanel) -> list[list[Any]]:
    rows = []
    for run, record in panel.records:
        final_cost = record.summary.get("final_cost", {})
        rows.append(
            [
                run.label,
                fmt_pct(record.summary.get("valid_ratio_after_filter")),
                f"{to_int(record.summary.get('num_rejected_by_filter')):,}",
                fmt_pct(record.summary.get("rejected_by_filter_ratio")),
                fmt_float(final_cost.get("median")),
                fmt_float(final_cost.get("p90")),
                fmt_float(mean_support(record), 2),
                len(record_logical_iterations(record)),
                ", ".join(map(str, record.summary.get("missing_maps", []))) or "-",
            ]
        )
    return rows


def diagnostic_artifact_items(record: DepthMapRecord) -> list[tuple[str, Path | None]]:
    return [
        ("summary.json", record.report_files.get("summary.json")),
        ("filtering.json", record.report_files.get("filtering.json")),
        ("iteration.csv", record.report_files.get("iteration.csv")),
        ("view_support.csv", record.report_files.get("view_support.csv")),
        ("cost PFM", record.pfm_maps.get("cost_final.pfm")),
        ("depth PFM", record.pfm_maps.get("depth_final_after_filter.pfm")),
        ("normal PFM", record.pfm_maps.get("normal_final.pfm")),
        ("valid PNG", record.png_maps.get("valid_after_filter.png")),
        ("rejection PNG", record.png_maps.get("rejection_reason.png")),
        ("candidate source PNG", record.png_maps.get("candidate_source.png")),
        ("support PNG", record.png_maps.get("num_supporting_views.png")),
        ("last iteration PNG", record.png_maps.get("last_changed_iter.png")),
    ]


def diagnostic_artifact_list_md(panel: DiagnosticPanel, out: Path) -> str:
    lines = ["**Raw artifacts**", ""]
    for run, record in panel.records:
        links = [str(md_link(path, out, label)) for label, path in diagnostic_artifact_items(record) if path is not None]
        lines.append(f"- **{md_label(run.label)}:** " + "; ".join(links))
    return "\n".join(lines)


def diagnostic_artifact_list_html(panel: DiagnosticPanel, out: Path) -> str:
    rows = []
    for run, record in panel.records:
        links = [str(rel_link(path, out, label)) for label, path in diagnostic_artifact_items(record) if path is not None]
        rows.append(f"<li><strong>{html.escape(run.label)}:</strong> " + "; ".join(links) + "</li>")
    return "<h4>Raw artifacts</h4><ul class='artifact-links'>" + "".join(rows) + "</ul>"


def diagnostic_comparison_summary(panel: DiagnosticPanel) -> str:
    if len(panel.records) < 2:
        return ""
    baseline_run, baseline = panel.records[0]
    modified_run, modified = panel.records[-1]
    baseline_cost = baseline.summary.get("final_cost", {})
    modified_cost = modified.summary.get("final_cost", {})
    comparisons = []

    baseline_valid = to_float(baseline.summary.get("valid_ratio_after_filter"))
    modified_valid = to_float(modified.summary.get("valid_ratio_after_filter"))
    if baseline_valid is not None and modified_valid is not None:
        comparisons.append(f"valid after {100.0 * (modified_valid - baseline_valid):+.2f} pp")

    rejected_delta = to_int(modified.summary.get("num_rejected_by_filter")) - to_int(baseline.summary.get("num_rejected_by_filter"))
    comparisons.append(f"rejected pixels {rejected_delta:+,}")

    for label, key in (("median cost", "median"), ("p90 cost", "p90")):
        baseline_value = to_float(baseline_cost.get(key))
        modified_value = to_float(modified_cost.get(key))
        if baseline_value is not None and modified_value is not None:
            comparisons.append(f"{label} {modified_value - baseline_value:+.4f}")

    comparisons.append(f"mean support {mean_support(modified) - mean_support(baseline):+.2f}")
    return f"{modified_run.label} - {baseline_run.label}: " + "; ".join(comparisons) + "."


def diagnostic_panel_gallery_md(panels: list[DiagnosticPanel], out: Path) -> str:
    if not panels:
        return "_No matched diagnostic panels were generated._"
    parts = [
        "Maps are aligned by image ID. Cost and depth use shared scales within each comparison. Cost delta is `baseline - modified`, so green regions have lower modified cost. Relative-depth and normal-angle panels measure run disagreement, not ground-truth error. Filter panels distinguish pixels gained and lost after filtering; last-update maps normalize logical iteration position independently for each run.",
        "",
    ]
    for panel in panels:
        comparison = diagnostic_comparison_summary(panel)
        parts += [
            '<details class="frame-details">',
            f"<summary><strong>{html.escape(panel.title)} diagnostic atlas</strong></summary>",
            "",
            str(md_image(panel.path, out, f"{panel.title} diagnostic atlas")),
            "",
            md_table(
                ["run", "valid after", "rejected px", "rejected", "cost median", "cost p90", "mean support", "iteration stages", "unavailable"],
                diagnostic_summary_rows(panel),
            ),
            "",
            f"**Comparison:** {comparison}" if comparison else "",
            "",
            diagnostic_artifact_list_md(panel, out),
            "",
            "</details>",
            "",
        ]
    return "\n".join(parts)


def diagnostic_panel_gallery_html(panels: list[DiagnosticPanel], out: Path) -> str:
    if not panels:
        return "<p class='note'>No matched diagnostic panels were generated.</p>"
    parts = [
        "<p class='note'>Maps are aligned by image ID. Cost and depth use shared scales within each comparison. Cost delta is baseline minus modified, so green regions have lower modified cost. Relative-depth and normal-angle panels measure run disagreement, not ground-truth error. Filter panels distinguish gained and lost pixels; last-update maps normalize logical iteration position independently for each run.</p>"
    ]
    for panel in panels:
        comparison = diagnostic_comparison_summary(panel)
        parts.append(
            f"<details class='diagnostic-atlas frame-details'><summary>{html.escape(panel.title)} diagnostic atlas</summary>"
            f"<figure>{image_tag(panel.path, out, panel.title)}<figcaption>{html.escape(panel.title)}</figcaption></figure>"
            f"{html_table(['run', 'valid after', 'rejected px', 'rejected', 'cost median', 'cost p90', 'mean support', 'iteration stages', 'unavailable'], diagnostic_summary_rows(panel))}"
            + (f"<p><strong>Comparison:</strong> {html.escape(comparison)}</p>" if comparison else "")
            + diagnostic_artifact_list_html(panel, out)
            + "</details>"
        )
    return "".join(parts)


def frame_panel_gallery(run: RunData, out: Path) -> str:
    cards = []
    for record in run.depthmaps:
        if not record.panel_path:
            continue
        title = frame_label(record)
        cards.append(
            f"<details class='frame-details'><summary>{html.escape(title)} overview</summary>"
            f"<figure>{image_tag(record.panel_path, out, title)}<figcaption>{html.escape(title)}</figcaption></figure>"
            "</details>"
        )
    return "<div class='frame-disclosures'>" + "".join(cards) + "</div>" if cards else "<p class='note'>No per-frame overview panels were generated.</p>"


def frame_panel_gallery_md(run: RunData, out: Path) -> str:
    parts = []
    for record in run.depthmaps:
        if not record.panel_path:
            continue
        title = frame_label(record)
        parts += [
            '<details class="frame-details">',
            f"<summary><strong>{html.escape(title)} overview</strong></summary>",
            "",
            str(md_image(record.panel_path, out, f"{title} overview panel")),
            "",
            "</details>",
            "",
        ]
    return "\n".join(parts) if parts else "_No per-frame overview panels were generated._"


def reference_thumbnail_status(run: RunData) -> str:
    resolved = sum(1 for record in run.depthmaps if record.reference_thumbnail and record.reference_thumbnail.is_file())
    return f"Reference thumbnails resolved: {resolved}/{len(run.depthmaps)}."


def improvement_panel_gallery(run: RunData, out: Path) -> str:
    cards = []
    for record in run.depthmaps:
        if not record.improvement_panel_path:
            continue
        title = frame_label(record)
        logical_count = len(record.logical_event_maps.get("cost_improvement_exact", {}))
        details = f"{logical_count or len({-1 if index == 0 else (index - 1) // 2 for index in record.improvement_maps})} iteration stages"
        if record.improvement_vmax is not None:
            details += f"; shared p99.5 clip {record.improvement_vmax:.4f}"
        cards.append(
            f"<details class='frame-details'><summary>{html.escape(title)} improvement maps</summary>"
            f"<figure>{image_tag(record.improvement_panel_path, out, title)}"
            f"<figcaption>{html.escape(title)}: {html.escape(details)}</figcaption></figure>"
            "</details>"
        )
    return "<div class='frame-disclosures'>" + "".join(cards) + "</div>" if cards else "<p class='note'>No per-iteration improvement maps were found.</p>"


def improvement_panel_gallery_md(run: RunData, out: Path) -> str:
    parts = []
    for record in run.depthmaps:
        if not record.improvement_panel_path:
            continue
        title = frame_label(record)
        logical_count = len(record.logical_event_maps.get("cost_improvement_exact", {}))
        details = f"{logical_count or len({-1 if index == 0 else (index - 1) // 2 for index in record.improvement_maps})} iteration stages"
        if record.improvement_vmax is not None:
            details += f"; shared p99.5 clip {record.improvement_vmax:.4f}"
        parts += [
            '<details class="frame-details">',
            f"<summary><strong>{html.escape(title)} improvement maps</strong> - {html.escape(details)}</summary>",
            "",
            str(md_image(record.improvement_panel_path, out, f"{title} per-iteration improvement maps")),
            "",
            "</details>",
            "",
        ]
    return "\n".join(parts) if parts else "_No per-iteration improvement maps were found._"


def metadata_block(run: RunData) -> str:
    meta = run.run_metadata
    limitations = meta.get("limitations", [])
    reporting = meta.get("reporting_contract", {})
    lines = [
        ["schema_version", meta.get("schema_version", "")],
        ["backend", meta.get("backend", "")],
        ["scope", meta.get("scope", "")],
        ["fusion_instrumented", meta.get("fusion_instrumented", "")],
        ["mesh_instrumented", meta.get("mesh_instrumented", "")],
        ["mechanics_granularity", reporting.get("mechanics_granularity", "logical PatchMatch iteration")],
        ["timing_granularity", reporting.get("timing_granularity", "checkerboard phase")],
    ]
    html_parts = [html_table(["field", "value"], lines, "compact")]
    if limitations:
        html_parts.append("<ul>" + "".join(f"<li>{html.escape(str(item))}</li>" for item in limitations) + "</ul>")
    return "".join(html_parts)


def metadata_block_md(run: RunData) -> str:
    meta = run.run_metadata
    limitations = meta.get("limitations", [])
    reporting = meta.get("reporting_contract", {})
    lines = [
        ["schema_version", meta.get("schema_version", "")],
        ["backend", meta.get("backend", "")],
        ["scope", meta.get("scope", "")],
        ["fusion_instrumented", meta.get("fusion_instrumented", "")],
        ["mesh_instrumented", meta.get("mesh_instrumented", "")],
        ["mechanics_granularity", reporting.get("mechanics_granularity", "logical PatchMatch iteration")],
        ["timing_granularity", reporting.get("timing_granularity", "checkerboard phase")],
    ]
    parts = [md_table(["field", "value"], lines)]
    if limitations:
        parts += ["", "Limitations:", "", *[f"- {item}" for item in limitations]]
    return "\n".join(parts)


def plot_html(plots: list[tuple[str, Path]], out: Path) -> str:
    cards = []
    for title, path in plots:
        cards.append(f"<figure>{image_tag(path, out, title)}<figcaption>{html.escape(title)}</figcaption></figure>")
    return "".join(cards) if cards else "<p class='note'>No plots were generated. Install matplotlib or check that instrumentation JSON/CSV files exist.</p>"


def plot_md(plots: list[tuple[str, Path]], out: Path) -> str:
    if not plots:
        return "_No plots were generated. Install matplotlib or check that instrumentation JSON/CSV files exist._"
    parts = []
    for title, path in plots:
        parts += [f"### {title}", "", str(md_image(path, out, title)), ""]
    return "\n".join(parts)


def instrumentation_roadmap_rows() -> list[list[str]]:
    return [
        [
            "P0",
            "Preserve pre-filter state",
            "Export final depth and aggregate cost immediately before filtering invalidates rejected pixels.",
            "Every rejected pixel remains inspectable; existing post-filter depth and validity maps remain unchanged.",
        ],
        [
            "P0",
            "Final score components and confidence gap",
            "Allocate one final-only float4 buffer at scale zero for raw photometric cost, photo-prior blend, geometric penalty, and second-best minus best cost.",
            "The exported components reconstruct the scorer within float tolerance; geometric-off maps are finite zero; unavailable gaps use an explicit sentinel.",
        ],
        [
            "P0",
            "Exact candidate accounting",
            "Count eligible, tested, finite, accepted, and rejected candidates by source and rejection reason in the CUDA pass that evaluates them.",
            "Accepted never exceeds tested, and no candidate type reports nonzero accepted with zero tested.",
        ],
        [
            "P1",
            "Region-aware diagnosis and frame ranking",
            "Aggregate cost, gap, support, rejection, and convergence over planar interiors, boundaries, texture bins, and available annotations; rank the largest regressions.",
            "The report links each ranked metric to the matching atlas and exposes both pixel counts and distributions.",
        ],
        [
            "P1",
            "Report manifest and validator",
            "Write report_manifest.json with schema, source runs, generated assets, and expected dimensions; audit links, disclosure balance, and required fields.",
            "A single validation command exits nonzero for a missing artifact, malformed summary, or unbalanced details block.",
        ],
        [
            "P2",
            "Fast iteration and regression verdicts",
            "Add frame/section selection, asset reuse, and configurable comparison thresholds for completeness, cost, convergence, runtime, and map coverage.",
            "Small targeted reports avoid rebuilding unrelated panels, while full reports produce a machine-readable pass/warn/fail verdict.",
        ],
    ]


def next_instrumentation_experiment() -> list[tuple[str, str]]:
    return [
        (
            "Hypothesis",
            "Pre-filter snapshots plus final score-component and confidence-gap maps will explain where extra PatchMatch iterations improve completeness, without changing estimated depth or normal values.",
        ),
        (
            "Code/config change",
            "Add the scale-zero, instrumentation-gated final-only float4 CUDA buffer; export pre-filter depth/cost; replace approximate candidate counts with exact evaluation-path counters. Keep the estimation and filtering algorithms unchanged.",
        ),
        (
            "Input",
            "Reuse the same selected-nine image IDs and paired one-iteration versus five-iteration configurations represented in this report.",
        ),
        (
            "Metrics",
            "Measure component reconstruction residuals, confidence-gap availability and quantiles, correlations with rejection/support/late updates, depth-normal output parity, and GPU memory/runtime overhead.",
        ),
        (
            "Expected failure modes",
            "Runner-up gaps are unavailable when fewer than two finite candidates share a scoring basis; geometric-off runs should be zero rather than missing; capturing after filtering would erase the regions this experiment is meant to diagnose.",
        ),
        (
            "Baseline comparison",
            "Compare against the current one-iteration/five-iteration artifacts and run instrumentation-disabled parity checks with identical inputs and parameters.",
        ),
        (
            "Decision rule",
            "Proceed to region-aware analysis only after depth/normal outputs show zero numerical delta, component residuals match the scorer's float tolerance, all expected maps validate, and overhead is measured and documented.",
        ),
    ]


def next_instrumentation_experiment_md() -> str:
    return "\n".join(f"- **{label}:** {description}" for label, description in next_instrumentation_experiment())


def next_instrumentation_experiment_html() -> str:
    return "<ul>" + "".join(
        f"<li><strong>{html.escape(label)}:</strong> {html.escape(description)}</li>"
        for label, description in next_instrumentation_experiment()
    ) + "</ul>"


def generate_html(runs: list[RunData], plots: list[tuple[str, Path]], diagnostic_panels: list[DiagnosticPanel], out: Path) -> str:
    run_sections = []
    include_run_label = len(runs) > 1
    for run in runs:
        section_title = f"Per-scene Analysis: {html.escape(run.label)}" if include_run_label else "Per-scene Analysis"
        run_sections.append(
            f"""
<details class="scene-details">
<summary>{section_title}</summary>
<div class="scene-content">
<h3>Metadata and Limitations</h3>
{metadata_block(run)}
<h3>Machine-readable Artifacts</h3>
{html_table(["artifact", "file"], artifact_rows(run, out, rel_link))}
<h3>Per-depthmap Summary</h3>
{html_table(["image", "safe name", "width", "height", "valid before", "valid after", "rejected", "cost mean", "cost median", "cost p90", "cost p95", "map files", "summary"], per_depthmap_rows(run, out))}
<h3>Filtering Rejection Summary</h3>
{html_table(["reason", "pixels", "share"], rejection_rows(run))}
<h3>Candidate Acceptance Summary</h3>
{html_table(["candidate type", "tested", "finite", "accepted", "acceptance rate"], candidate_rows(run))}
<h3>PatchMatch Iteration Summary</h3>
{html_table(["stage", "scale", "frames", "valid", "changed", "acceptance", "mean cost", "mean cost delta", "mean abs depth delta", "mean normal delta deg", "tested", "finite", "accepted"], iteration_rows(run))}
<h3>Cost-component Counter Summary</h3>
{html_table(["stage", "scale", "processed", "mean final cost", "mean photometric", "mean photo-prior", "mean geometric", "mean view entropy", "low texture", "bad cost"], counter_rows(run))}
<h3>Kernel Timing Summary</h3>
{html_table(["pass", "phase", "iteration", "rows", "total ms", "mean ms"], timing_rows(run))}
<h3>Supporting-view Histogram</h3>
{html_table(["supporting views", "pixels", "share"], view_support_rows(run))}
<h3>Per-frame Overview Panels</h3>
<p class="note">{html.escape(reference_thumbnail_status(run))}</p>
{frame_panel_gallery(run, out)}
<h3>Per-iteration Improvement Maps</h3>
<p class="note">Color uses gamma 0.5 and a shared per-frame p99.5 positive-value scale across compared runs. Initialization has no previous hypothesis, so its improvement is zero by definition. Each displayed iteration combines the complementary checkerboard updates.</p>
{improvement_panel_gallery(run, out)}
</div>
</details>
"""
        )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Rich depth-map instrumentation report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2933; background: #fbfcfd; }}
h1 {{ margin: 0 0 0.35rem; font-size: 32px; }}
h2 {{ margin-top: 34px; padding-top: 14px; border-top: 2px solid #d8dee4; }}
h3 {{ margin-top: 26px; }}
h4 {{ margin: 18px 0 8px; }}
a {{ color: #2454a6; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem; font-size: 13px; background: white; }}
th, td {{ border-bottom: 1px solid #d6e0e8; padding: 7px 10px; text-align: left; vertical-align: top; }}
th {{ background: #d9e8f2; color: #20384b; border-bottom: 2px solid #8aa8bd; font-weight: 700; }}
tbody tr:nth-child(even) td {{ background: #f2f7fa; }}
tbody tr:hover td {{ background: #fff5d9; }}
tbody td:first-child {{ border-left: 3px solid #5f8dab; color: #294861; font-weight: 600; }}
figure {{ margin: 14px 0 24px; background: white; border: 1px solid #d8dee4; border-radius: 7px; padding: 10px; }}
figure img {{ max-width: 100%; border-radius: 4px; }}
figcaption {{ color: #52616f; font-size: 13px; margin-top: 6px; }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; align-items: start; }}
.diagnostic-atlas {{ margin-bottom: 34px; }}
.diagnostic-atlas figure {{ padding: 6px; }}
.scene-details {{ margin: 28px 0; border: 1px solid #aebfcd; border-radius: 7px; background: #ffffff; }}
.scene-details > summary {{ cursor: pointer; padding: 13px 16px; background: #dce9f2; color: #20384b; font-size: 20px; font-weight: 700; }}
.scene-details[open] > summary {{ border-bottom: 1px solid #aebfcd; }}
.scene-content {{ padding: 2px 16px 18px; }}
.frame-disclosures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 9px; margin: 12px 0 20px; }}
.frame-disclosures .frame-details[open] {{ grid-column: 1 / -1; }}
.frame-details {{ margin: 9px 0; border: 1px solid #c7d5df; border-radius: 6px; background: #ffffff; }}
.frame-details > summary {{ cursor: pointer; padding: 9px 12px; background: #edf4f8; color: #294861; font-weight: 650; }}
.frame-details[open] > summary {{ background: #dfeef6; border-bottom: 1px solid #c7d5df; }}
.frame-details > figure, .frame-details > table {{ margin-left: 10px; margin-right: 10px; }}
summary::marker {{ color: #4f7f9f; }}
.note {{ color: #52616f; }}
.compact {{ max-width: 880px; }}
ul {{ margin-top: 0.5rem; }}
</style>
</head>
<body>
<h1>Rich Depth-map Instrumentation Report</h1>
<p class="note">Depth-map estimation and filtering diagnostics only. Fusion, mesh reconstruction, mesh refinement, and texturing are intentionally out of scope.</p>
<h2>Executive Summary</h2>
{html_table(["run", "path", "depth maps", "pixels", "valid before", "valid after", "rejected", "counter rows", "timing rows", "trace rows"], run_summary_rows(runs))}
<h2>Overall Visualizations</h2>
<div class="plots">{plot_html(plots, out)}</div>
<h2>Depth-map Diagnostic Panels</h2>
{diagnostic_panel_gallery_html(diagnostic_panels, out)}
<h2>Instrumentation Development Roadmap</h2>
{html_table(["priority", "deliverable", "implementation", "acceptance criteria"], instrumentation_roadmap_rows())}
<h3>Recommended Next Experiment</h3>
{next_instrumentation_experiment_html()}
{''.join(run_sections)}
</body>
</html>
"""


def generate_markdown(runs: list[RunData], plots: list[tuple[str, Path]], diagnostic_panels: list[DiagnosticPanel], out: Path) -> str:
    run_sections = []
    include_run_label = len(runs) > 1
    for run in runs:
        section_title = f"Per-scene Analysis: {run.label}" if include_run_label else "Per-scene Analysis"
        run_sections.append(
            "\n".join(
                [
                    '<details class="scene-details">',
                    f"<summary><strong>{html.escape(section_title)}</strong></summary>",
                    "",
                    "### Metadata and Limitations",
                    "",
                    metadata_block_md(run),
                    "",
                    "### Machine-readable Artifacts",
                    "",
                    md_table(["artifact", "file"], artifact_rows(run, out, md_link)),
                    "",
                    "### Per-depthmap Summary",
                    "",
                    md_table(
                        ["image", "safe name", "width", "height", "valid before", "valid after", "rejected", "cost mean", "cost median", "cost p90", "cost p95", "map files", "summary"],
                        per_depthmap_rows_md(run, out),
                    ),
                    "",
                    "### Filtering Rejection Summary",
                    "",
                    md_table(["reason", "pixels", "share"], rejection_rows(run)),
                    "",
                    "### Candidate Acceptance Summary",
                    "",
                    md_table(["candidate type", "tested", "finite", "accepted", "acceptance rate"], candidate_rows(run)),
                    "",
                    "### PatchMatch Iteration Summary",
                    "",
                    md_table(
                        ["stage", "scale", "frames", "valid", "changed", "acceptance", "mean cost", "mean cost delta", "mean abs depth delta", "mean normal delta deg", "tested", "finite", "accepted"],
                        iteration_rows(run),
                    ),
                    "",
                    "### Cost-component Counter Summary",
                    "",
                    md_table(
                        ["stage", "scale", "processed", "mean final cost", "mean photometric", "mean photo-prior", "mean geometric", "mean view entropy", "low texture", "bad cost"],
                        counter_rows(run),
                    ),
                    "",
                    "### Kernel Timing Summary",
                    "",
                    md_table(["pass", "phase", "iteration", "rows", "total ms", "mean ms"], timing_rows(run)),
                    "",
                    "### Supporting-view Histogram",
                    "",
                    md_table(["supporting views", "pixels", "share"], view_support_rows(run)),
                    "",
                    "### Per-frame Overview Panels",
                    "",
                    reference_thumbnail_status(run),
                    "",
                    frame_panel_gallery_md(run, out),
                    "",
                    "### Per-iteration Improvement Maps",
                    "",
                    "Color uses gamma 0.5 and a shared per-frame p99.5 positive-value scale across compared runs. Initialization has no previous hypothesis, so its improvement is zero by definition. Each displayed iteration combines the complementary checkerboard updates.",
                    "",
                    improvement_panel_gallery_md(run, out),
                    "",
                    "</details>",
                ]
            )
        )

    return "\n".join(
        [
            "# Rich Depth-map Instrumentation Report",
            "",
            "Depth-map estimation and filtering diagnostics only. Fusion, mesh reconstruction, mesh refinement, and texturing are intentionally out of scope.",
            "",
            MARKDOWN_REPORT_STYLE,
            "",
            "## Executive Summary",
            "",
            md_table(["run", "path", "depth maps", "pixels", "valid before", "valid after", "rejected", "counter rows", "timing rows", "trace rows"], run_summary_rows(runs)),
            "",
            "## Algorithm Description",
            "",
            "This report covers the CUDA PatchMatch depth-map stage in `DensifyPointCloud`. The instrumented pipeline initializes and refines per-image depth/normal hypotheses, scores candidates, records convergence counters by logical iteration, writes final diagnostic maps, and records filtering outcomes before any dense point-cloud fusion step. Checkerboard phases remain separate only in kernel timing diagnostics.",
            "",
            "## Metrics Description",
            "",
            "- `valid before filter`: fraction of pixels with an estimated depth before depth-map filtering.",
            "- `valid after filter`: fraction of pixels that survived depth-map filtering.",
            "- `rejected`: fraction of estimated pixels removed by depth-map filtering.",
            "- `final cost`: aggregate PatchMatch score retained for the final hypothesis; lower is better for the current OpenMVS scoring convention.",
            "- `changed ratio`: fraction of pixels updated during a complete logical PatchMatch iteration.",
            "- `acceptance rate`: accepted candidates divided by tested candidates for the pass or candidate type.",
            "- `supporting views`: number of source/support views recorded for pixels where that signal is available.",
            "- `per-frame overview panels`: composite diagnostics combining the reference RGB image, final maps, PatchMatch evolution, candidate acceptance, support histogram, and scalar summary for each depth map.",
            "- `per-iteration improvement maps`: spatial cost reduction across both complementary checkerboard updates in one logical iteration; higher values indicate larger accepted improvement.",
            "",
            "## Overall Scene Metrics",
            "",
            md_table(["run", "path", "depth maps", "pixels", "valid before", "valid after", "rejected", "counter rows", "timing rows", "trace rows"], run_summary_rows(runs)),
            "",
            "## Overall Visualizations",
            "",
            plot_md(plots, out),
            "",
            "## Depth-map Diagnostic Panels",
            "",
            diagnostic_panel_gallery_md(diagnostic_panels, out),
            "",
            "\n\n".join(run_sections),
            "",
            "## Failure Analysis",
            "",
            "Inspect the rejection-reason distribution, final-cost maps, valid-before/after-filter masks, candidate-source maps, support-view maps, and last-changed-iteration maps together. Recurring high-cost regions with low support usually indicate textureless, reflective, occluded, or poorly constrained areas; regions that are valid before filtering but absent after filtering identify the filter as the immediate completeness bottleneck.",
            "",
            "## Instrumentation Development Roadmap",
            "",
            md_table(["priority", "deliverable", "implementation", "acceptance criteria"], instrumentation_roadmap_rows()),
            "",
            "### Recommended Next Experiment",
            "",
            next_instrumentation_experiment_md(),
            "",
            "## Reproducibility",
            "",
            "Inputs are the instrumentation directories passed through `--runs`. Recreate this report by rerunning:",
            "",
            "```bash",
            shlex.join([sys.executable, "scripts/dmap_instrumentation_report.py"]) + " \\",
            "  --runs " + " ".join(f"{run.label}={run.path}" for run in runs) + " \\",
            f"  --out {out}",
            "```",
            "",
            "Generated plots, reference thumbnails, PFM previews, per-iteration improvement contact sheets, per-frame overview panels, and bundled JSON/CSV/map artifacts are stored next to the report in the report assets directory.",
            "",
            "## Recommendations",
            "",
            "- Implement the P0 observability bundle before changing PatchMatch scoring or filtering behavior.",
            "- Use ranked per-frame and region-level regressions to choose algorithm experiments instead of relying on scene aggregates alone.",
            "- Add a manifest-backed validator and regression verdict before scaling the benchmark to more scenes.",
            "",
        ]
    )


def per_depthmap_rows_md(run: RunData, out: Path) -> list[list[Any]]:
    rows = []
    for record in run.depthmaps:
        summary = record.summary
        final_cost = summary.get("final_cost", {})
        maps = len(record.png_maps) + len(record.pfm_maps)
        rows.append(
            [
                summary.get("image_id", ""),
                summary.get("safe_image_name", record.directory.name),
                summary.get("width", ""),
                summary.get("height", ""),
                fmt_pct(summary.get("valid_ratio_before_filter")),
                fmt_pct(summary.get("valid_ratio_after_filter")),
                fmt_pct(summary.get("rejected_by_filter_ratio")),
                fmt_float(final_cost.get("mean")),
                fmt_float(final_cost.get("median")),
                fmt_float(final_cost.get("p90")),
                fmt_float(final_cost.get("p95")),
                maps,
                md_link(record.report_files.get("summary.json", record.directory / "summary.json"), out, "summary.json"),
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Run specs in the form name=/path/to/instrumentation")
    parser.add_argument("--out", required=True, type=Path, help="Output report path. Use .md for Markdown or .html for HTML.")
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = out.parent / f"{out.stem}_assets"
    prepare_assets_dir(assets)
    runs = parse_runs(args.runs)
    plots, diagnostic_panels = make_plots(runs, assets)
    bundle_report_artifacts(runs, assets)
    if out.suffix.lower() in {".html", ".htm"}:
        out.write_text(generate_html(runs, plots, diagnostic_panels, out), encoding="utf-8")
    else:
        out.write_text(generate_markdown(runs, plots, diagnostic_panels, out), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
