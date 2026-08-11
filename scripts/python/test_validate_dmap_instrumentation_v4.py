#!/usr/bin/env python3
"""Focused schema-v4 exact-observability validator tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_dmap_instrumentation as validator
from test_validate_dmap_instrumentation_v3 import (
    HEIGHT,
    WIDTH,
    add_map,
    legacy_prior_weight,
    make_fixture,
    rewrite_entry,
    save_manifest,
    write_pfm,
    write_run_metadata,
)


def scalar(value: float) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value, dtype=np.float32)


def rgba_mask(value: int) -> np.ndarray:
    channels = np.array(
        [value & 255, (value >> 8) & 255, (value >> 16) & 255, (value >> 24) & 255],
        dtype=np.uint8,
    )
    return np.broadcast_to(channels, (HEIGHT, WIDTH, 4)).copy()


def rgb(first: int, second: int, third: int) -> np.ndarray:
    channels = np.array([first, second, third], dtype=np.uint8)
    return np.broadcast_to(channels, (HEIGHT, WIDTH, 3)).copy()


def rgba(first: int, second: int, third: int, fourth: int) -> np.ndarray:
    channels = np.array([first, second, third, fourth], dtype=np.uint8)
    return np.broadcast_to(channels, (HEIGHT, WIDTH, 4)).copy()


def v4_frame_dir(directory: str) -> Path:
    return Path(directory) / "stage" / "depthmaps" / "0001_0000"


def write_v4_run_metadata(frame_dir: Path, view_samples: object = 32) -> Path:
    path = write_run_metadata(frame_dir)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["cuda_patchmatch_parameters"]["view_samples"] = view_samples
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


_MISSING = object()


def rewrite_v4_view_samples_metadata(
    frame_dir: Path, view_samples: object = _MISSING
) -> None:
    path = frame_dir.parent.parent / "run_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    parameters = metadata["cuda_patchmatch_parameters"]
    if view_samples is _MISSING:
        parameters.pop("view_samples", None)
    else:
        parameters["view_samples"] = view_samples
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def add_exact_map(
    frame_dir: Path,
    manifest: dict,
    *,
    signal: str,
    iteration: int,
    values: np.ndarray,
    dtype: str = "float32",
    role: str = "logical_state",
    source_view_index: int | None = None,
) -> dict:
    stage_index = iteration + 1
    state = "initialization" if iteration == -1 else f"iteration_{iteration:02d}"
    suffix = ".png" if dtype.startswith("uint8") else ".pfm"
    view_suffix = "" if source_view_index is None else f"_view{source_view_index:02d}"
    metadata = {
        "role": role,
        "logical_iteration": iteration,
        "stage": "initialization" if iteration == -1 else "iteration",
        "stage_index": stage_index,
        "measurement_quality": "exact",
        "measurement_basis": "production_hot_kernel_view_record"
        if source_view_index is not None
        else "production_hot_kernel_candidate_record",
    }
    if source_view_index is not None:
        metadata.update(
            source_view_index=source_view_index,
            source_image_id=100 + source_view_index,
            source_image_name=f"source_{source_view_index}.jpg",
        )
    return add_map(
        frame_dir,
        manifest,
        signal,
        f"logical_states/{state}/{signal}{view_suffix}{suffix}",
        values,
        dtype=dtype,
        **metadata,
    )


def make_v4_fixture(frame_dir: Path, view_samples: int = 32) -> tuple[dict, dict]:
    if not 1 <= view_samples <= 63:
        raise ValueError("test fixture view_samples must be in [1,63]")
    manifest, summary = make_fixture(frame_dir, 3)
    manifest["schema_version"] = 4
    manifest["pyramid_level"] = 0
    manifest["measurement_model"] = validator.V4_MEASUREMENT_MODEL
    manifest["exact_capture"] = {
        "requested": True,
        "available": True,
        "unavailable_reason": "",
        "num_views": 2,
        "record_pixel_bytes": 48,
        "record_view_bytes": 32,
    }
    observer_sidecars = {
        "complete": True,
        "write_error_count": 0,
        "write_errors": [],
    }
    manifest["observer_sidecars"] = observer_sidecars.copy()
    summary["schema_version"] = 4
    summary.update({
        "schema_name": "openmvs.dmap.frame_summary",
        "image_id": 1,
        "image_name": "0000",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "observer_sidecars": observer_sidecars.copy(),
        "completion_marker": {
            "schema_name": "openmvs.dmap.capture_complete",
            "schema_version": 1,
            "path": "capture_complete.json",
            "maps_complete": True,
            "eligible": True,
        },
    })
    summary["candidate_accounting_mode"] = "exact_production_hot_kernel_full_frame"
    summary["confidence_gap_mode"] = "exact_process_pixel_winner_runner_up_full_frame"

    for iteration, stored in ((-1, 0.625), (0, 0.5)):
        exact_state = {
            "cost_photo_raw_production_exact": stored - 0.05,
            "cost_photo_prior_production_exact": stored - 0.125,
            "cost_geometric_production_exact": 0.125,
            "cost_total_production_exact": stored,
            "depth_prior_disagreement_production_exact": 0.1,
            "depth_prior_weight_production_exact": 0.25,
            "gap_winner_runner_up_exact": -1.0 if iteration == -1 else 0.1,
            "reference_variance_production_exact": 0.2,
        }
        for signal, value in exact_state.items():
            add_exact_map(
                frame_dir,
                manifest,
                signal=signal,
                iteration=iteration,
                values=scalar(value),
            )

        tested_mask = 1 if iteration == -1 else 7
        accepted_mask = 1 if iteration == -1 else 2
        runner = -1.0 if iteration == -1 else stored + 0.1
        exact_events: dict[str, np.ndarray] = {
            "candidate_stored_cost_before_exact": scalar(-1.0 if iteration == -1 else 0.625),
            "candidate_incumbent_cost_exact": scalar(stored + (0.0 if iteration == -1 else 0.2)),
            "candidate_winner_cost_exact": scalar(stored),
            "candidate_runner_up_cost_exact": scalar(runner),
            "candidate_tested_mask_exact": scalar(float(tested_mask)),
            "candidate_finite_mask_exact": scalar(float(tested_mask)),
            "candidate_accepted_mask_exact": scalar(float(accepted_mask)),
            "candidate_counts_exact": rgb(1 if iteration == -1 else 3, 1 if iteration == -1 else 3, 1),
            "candidate_identity_exact": rgb(0 if iteration == -1 else 1, 255 if iteration == -1 else 0, 1 if iteration == -1 else 2),
            "selected_view_counts_exact": rgb(0 if iteration == -1 else 2, 2, 2 if iteration == -1 else 2),
            "selected_views_before_mask_exact": rgba_mask(0 if iteration == -1 else 3),
            "selected_views_after_mask_exact": rgba_mask(3 if iteration == -1 else 5),
        }
        for signal, values in exact_events.items():
            dtype = (
                "uint8x4" if values.ndim == 3 and values.shape[-1] == 4
                else "uint8x3" if values.ndim == 3
                else "float32"
            )
            add_exact_map(
                frame_dir,
                manifest,
                signal=signal,
                iteration=iteration,
                values=values,
                dtype=dtype,
                role="logical_event",
            )

        for view in range(2):
            view_weight = (
                1 if iteration == -1
                else (view_samples + 1) // 2 if view == 0
                else view_samples // 2
            )
            photo = stored / 2.0 - 0.0625
            geometry = 0.0625
            cost_components = np.dstack([scalar(photo), scalar(geometry), scalar(stored / 2.0)])
            selection_metrics = np.dstack(
                [
                    scalar(-1.0 if iteration == -1 else 0.4),
                    scalar(-1.0 if iteration == -1 else 0.2),
                    scalar(-1.0 if iteration == -1 else 0.5),
                ]
            )
            view_maps = {
                "view_cost_components_exact": (cost_components, "float32x3"),
                "view_selection_metrics_exact": (selection_metrics, "float32x3"),
                "view_weighted_contribution_exact": (
                    scalar(
                        stored / 2.0
                        if iteration == -1
                        else stored * view_weight / view_samples
                    ),
                    "float32",
                ),
                "view_selection_state_exact": (
                    rgb(view_weight, view, 4 if iteration == -1 else 1),
                    "uint8x3",
                ),
                "view_agreement_state_exact": (rgb(0 if iteration == -1 else 4, 0, 15), "uint8x3"),
            }
            for signal, (values, dtype) in view_maps.items():
                add_exact_map(
                    frame_dir,
                    manifest,
                    signal=signal,
                    iteration=iteration,
                    values=values,
                    dtype=dtype,
                    role="logical_view_state",
                    source_view_index=view,
                )

    exact_iteration_header = (
        "logical_iteration,pyramid_level,"
        + ",".join(validator.V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS)
        + "\n"
    )
    exact_iteration_empty = "," * len(validator.V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS)
    table_specs = (
        (
            "openmvs.dmap.exact_iteration", 3, "exact_iteration.csv",
            exact_iteration_header
            + f"-1,0{exact_iteration_empty}\n"
            + f"0,0{exact_iteration_empty}\n",
        ),
        (
            "openmvs.dmap.exact_view_summary", 2, "exact_view_summary.csv",
            "logical_iteration,pyramid_level,view\n-1,0,0\n0,0,0\n",
        ),
        (
            "openmvs.dmap.exact_observability", 3, "exact_observability.json",
            '{"schema_name":"openmvs.dmap.exact_observability",'
            '"schema_version":3,"pyramid_level":0}\n',
        ),
    )
    manifest["tables"] = []
    for schema_name, schema_version, relative_path, payload in table_specs:
        path = frame_dir / relative_path
        path.write_text(payload)
        manifest["tables"].append(
            {
                "schema_name": schema_name,
                "schema_version": schema_version,
                "path": relative_path,
                "bytes": path.stat().st_size,
                "pyramid_level": 0,
            }
        )
    manifest.update(
        complete=True,
        expected_map_count=len(manifest["maps"]),
        written_map_count=len(manifest["maps"]),
        write_errors=[],
    )
    save_manifest(frame_dir, manifest)
    (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (frame_dir / "capture_complete.json").write_text(json.dumps({
        "schema_name": "openmvs.dmap.capture_complete",
        "schema_version": 1,
        "capture_kind": "maps",
        "image_id": summary["image_id"],
        "image_name": summary["image_name"],
        "estimation_stage": summary["estimation_stage"],
        "geometric_iteration": summary["geometric_iteration"],
        "maps_complete": True,
        "observer_sidecars_complete": True,
        "map_manifest": {
            "path": "map_manifest.json", "schema_version": 4, "complete": True,
        },
        "summary": {"path": "summary.json", "schema_version": 4},
    }, indent=2) + "\n")
    write_v4_run_metadata(frame_dir, view_samples)
    return manifest, summary


def add_filter_rejection_decomposition(frame_dir: Path, summary: dict) -> dict:
    ignore_mask = {
        "requested": True,
        "label": 0,
        "load_attempted": True,
        "loaded": True,
        "status": "loaded",
        "source_path": "reference.jpg.mask.png",
        "rejection_count_available": True,
        "unavailable_reason": "",
    }
    summary.update(
        num_valid_after_keep_cost_filter=3,
        num_rejected_by_keep_cost_filter=1,
        num_rejected_by_ignore_mask=1,
        num_valid_after_filter=2,
        num_rejected_by_filter=2,
        valid_ratio_after_keep_cost_filter=0.75,
        valid_ratio_after_filter=0.5,
        ignore_mask=ignore_mask,
        supporting_view_histogram=[2, 2, 0, 0, 0],
    )
    filtering = {
        "num_pixels_total": WIDTH * HEIGHT,
        "num_valid_before_filter": WIDTH * HEIGHT,
        "num_invalid_before_filter": 0,
        "num_valid_after_keep_cost_filter": 3,
        "num_rejected_by_keep_cost_filter": 1,
        "num_rejected_by_ignore_mask": 1,
        "num_valid_after_filter": 2,
        "num_rejected_by_filter": 2,
        "valid_ratio_before_filter": 1.0,
        "valid_ratio_after_keep_cost_filter": 0.75,
        "valid_ratio_after_filter": 0.5,
        "ignore_mask": ignore_mask,
        "rejection_reasons": {
            "low_score": 1,
            "insufficient_view_support": 0,
            "geometric_inconsistency": 0,
            "normal_inconsistency": 0,
            "depth_range": 0,
            "occlusion": 0,
            "masked": 1,
            "small_component": 0,
            "unknown": 0,
        },
    }
    (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (frame_dir / "filtering.json").write_text(json.dumps(filtering, indent=2) + "\n")
    return filtering


def check_by_name(result: dict, name: str) -> dict:
    return next(check for check in result["checks"] if check["name"] == name)


def add_core_terminal_maps(frame_dir: Path, manifest: dict) -> dict[str, np.ndarray]:
    values = {
        "depth_map": scalar(1.0),
        "normal_map": np.dstack([scalar(0.0), scalar(0.0), scalar(1.0)]),
        "confidence_map": scalar(0.5),
    }
    add_map(
        frame_dir, manifest, "depth_final_after_filter",
        "maps/depth_final_after_filter.pfm", values["depth_map"],
    )
    add_map(
        frame_dir, manifest, "normal_final", "maps/normal_final.pfm",
        values["normal_map"], dtype="float32x3",
    )
    add_map(
        frame_dir, manifest, "cost_final", "maps/cost_final.pfm",
        1.0 - values["confidence_map"],
    )
    manifest["expected_map_count"] = len(manifest["maps"])
    manifest["written_map_count"] = len(manifest["maps"])
    save_manifest(frame_dir, manifest)
    return values


def add_logical_cost_improvement_fixture(
    frame_dir: Path,
    manifest: dict,
    summary: dict,
) -> None:
    pass_one = scalar(0.125)
    pass_two = scalar(0.25)
    logical_values = {
        -1: scalar(0.0),
        0: pass_one + pass_two,
    }
    for iteration, values in logical_values.items():
        stage = "initialization" if iteration == -1 else "iteration"
        directory = "initialization" if iteration == -1 else f"iteration_{iteration:02d}"
        add_map(
            frame_dir,
            manifest,
            validator.LOGICAL_COST_IMPROVEMENT_SIGNAL,
            f"logical_states/{directory}/cost_improvement_exact.pfm",
            values,
            role="logical_event",
            logical_iteration=iteration,
            stage=stage,
            stage_index=iteration + 1,
            measurement_quality=validator.LOGICAL_COST_IMPROVEMENT_QUALITY,
            measurement_basis=validator.LOGICAL_COST_IMPROVEMENT_BASIS,
            aggregation=(
                "defined_zero"
                if iteration == -1
                else "sum_of_disjoint_checkerboard_passes"
            ),
            checkerboard_identity_exposed=False,
            valid_min=0.0,
            limitations=(
                "Nonnegative stored-cost reduction; cost increases are zero and "
                "view selection can change the cost basis."
            ),
        )
    summary.update(image_id=1, scale_level=0)
    (frame_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    improvement_dir = frame_dir.parent.parent / "instrumentation" / "improvements"
    write_pfm(improvement_dir / "depth0001_scale00_pass00_improvement.pfm", scalar(0.5))
    write_pfm(improvement_dir / "depth0001_scale00_pass01_improvement.pfm", pass_one)
    write_pfm(improvement_dir / "depth0001_scale00_pass02_improvement.pfm", pass_two)
    manifest["expected_map_count"] = len(manifest["maps"])
    manifest["written_map_count"] = len(manifest["maps"])
    save_manifest(frame_dir, manifest)


def optional_entry(
    frame_dir: Path,
    *,
    signal: str,
    relative_path: str,
    values: np.ndarray,
    algorithm_stage: str,
    dtype: str = "float32",
) -> dict:
    path = frame_dir / relative_path
    if path.suffix == ".png":
        path.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
    else:
        from test_validate_dmap_instrumentation_v3 import write_pfm
        write_pfm(path, values)
    return {
        "signal": signal,
        "path": relative_path,
        "dtype": dtype,
        "quality": "exact",
        "algorithm_stage": algorithm_stage,
        "semantics": f"test optional map for {signal}",
        "declared_bytes": int(np.asarray(values).size * np.asarray(values).dtype.itemsize),
        "file_bytes": path.stat().st_size,
        "file_size_available": True,
    }


def add_optional_stage_manifests(
    frame_dir: Path,
    *,
    depth: np.ndarray,
    confidence_after_postprocess: np.ndarray,
    confidence_final: np.ndarray | None,
    normal_changed_pixels: int | None = None,
    stages_enabled: bool = True,
    include_terminal_normal: bool = True,
) -> None:
    postprocess_maps: list[dict] = []
    if stages_enabled:
        postprocess_maps.extend([
            optional_entry(
                frame_dir,
                signal="00_remove_speckles_depth_after",
                relative_path="postprocess_filters/00_remove_speckles_depth_after.pfm",
                values=depth,
                algorithm_stage="remove_speckles",
            ),
            optional_entry(
                frame_dir,
                signal="00_remove_speckles_confidence_after",
                relative_path="postprocess_filters/00_remove_speckles_confidence_after.pfm",
                values=confidence_after_postprocess,
                algorithm_stage="remove_speckles",
            ),
        ])
        if include_terminal_normal:
            postprocess_maps.append(optional_entry(
                frame_dir,
                signal="00_remove_speckles_normal_after",
                relative_path="postprocess_filters/00_remove_speckles_normal_after.pfm",
                values=np.dstack([scalar(0.0), scalar(0.0), scalar(1.0)]),
                algorithm_stage="remove_speckles",
                dtype="float32x3",
            ))
    stage_metrics = {
        "depth_changed_pixels": int(np.count_nonzero(depth != 1.0)),
        "confidence_changed_pixels": int(
            np.count_nonzero(confidence_after_postprocess != 0.5)
        ),
    }
    stage_metrics["normal_changed_pixels"] = normal_changed_pixels or 0
    postprocess = {
        "schema_name": "openmvs.dmap.postprocess_filters",
        "schema_version": 2,
        "maps_requested": True,
        "maps_enabled": stages_enabled,
        "maps": postprocess_maps,
        "write_errors": [],
        "complete": True,
        "stages": [{
            "stage_index": 0,
            "name": "remove_speckles",
            "enabled": stages_enabled,
            "executed": stages_enabled,
            "success": True if stages_enabled else None,
            "metrics": stage_metrics,
        }],
    }
    (frame_dir / "postprocess_filters.json").write_text(
        json.dumps(postprocess, indent=2) + "\n"
    )

    confidence_maps: list[dict] = []
    if confidence_final is not None:
        confidence_maps.append(optional_entry(
            frame_dir,
            signal="confidence_final",
            relative_path="confidence_adjustment/confidence_final.pfm",
            values=confidence_final,
            algorithm_stage="confidence_adjustment",
        ))
    confidence_enabled = confidence_final is not None
    confidence = {
        "schema_name": "openmvs.dmap.confidence_adjustment",
        "schema_version": 1,
        "maps_requested": True,
        "maps_enabled": confidence_enabled,
        "maps": confidence_maps,
        "write_errors": [],
        "complete": True,
        "methods": [{
            "name": "final_combined_confidence",
            "enabled": confidence_enabled,
            "executed": confidence_enabled,
            "output_available": confidence_enabled,
            "metrics": {
                "changed_pixels": int(np.count_nonzero(
                    confidence_final != confidence_after_postprocess
                )) if confidence_enabled else 0,
            },
        }],
    }
    (frame_dir / "confidence_adjustment.json").write_text(
        json.dumps(confidence, indent=2) + "\n"
    )


class DMapInstrumentationSchemaV4Tests(unittest.TestCase):
    def test_logical_cost_improvement_validates_metadata_domain_and_legacy_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, summary = make_v4_fixture(frame_dir)
            add_logical_cost_improvement_fixture(frame_dir, manifest, summary)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            for name in (
                "logical_cost_improvement_coverage",
                "logical_cost_improvement_unique",
                "logical_cost_improvement_metadata",
                "logical_cost_improvement_layout",
                "logical_cost_improvement_domain",
                "logical_cost_improvement_initialization_zero",
                "logical_cost_improvement_legacy_pass_closure",
            ):
                self.assertTrue(check_by_name(result, name)["passed"], (name, result))
            validation = result["logical_event_validation"][
                validator.LOGICAL_COST_IMPROVEMENT_SIGNAL
            ]
            self.assertTrue(validation["available"])
            self.assertEqual(
                validation["legacy_pass_closure"]["per_iteration_max_abs"]["0"],
                0.0,
            )

    def test_logical_cost_improvement_rejects_negative_init_nonzero_and_missing_state(self) -> None:
        cases = (
            ("negative", 0, scalar(-0.125), "logical_cost_improvement_domain"),
            (
                "initialization_nonzero",
                -1,
                scalar(0.125),
                "logical_cost_improvement_initialization_zero",
            ),
            ("missing_state", 0, None, "logical_cost_improvement_coverage"),
        )
        for name, iteration, replacement, failed_check in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                frame_dir = v4_frame_dir(directory)
                manifest, summary = make_v4_fixture(frame_dir)
                add_logical_cost_improvement_fixture(frame_dir, manifest, summary)
                entry = next(
                    item for item in manifest["maps"]
                    if item.get("signal") == validator.LOGICAL_COST_IMPROVEMENT_SIGNAL
                    and item.get("logical_iteration") == iteration
                )
                path = frame_dir / entry["path"]
                if replacement is None:
                    path.unlink()
                    manifest["maps"].remove(entry)
                    manifest["expected_map_count"] = len(manifest["maps"])
                    manifest["written_map_count"] = len(manifest["maps"])
                else:
                    write_pfm(path, replacement)
                    entry["bytes"] = path.stat().st_size
                save_manifest(frame_dir, manifest)

                result = validator.validate(validator.Arguments(frame_dir=frame_dir))

                self.assertFalse(check_by_name(result, failed_check)["passed"], result)

    def test_logical_cost_improvement_rejects_inexact_or_phase_exposing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, summary = make_v4_fixture(frame_dir)
            add_logical_cost_improvement_fixture(frame_dir, manifest, summary)
            entry = next(
                item for item in manifest["maps"]
                if item.get("signal") == validator.LOGICAL_COST_IMPROVEMENT_SIGNAL
                and item.get("logical_iteration") == 0
            )
            entry["measurement_quality"] = "proxy"
            entry["checkerboard_identity_exposed"] = True
            entry["raw_pass_indices"] = [1, 2]
            save_manifest(frame_dir, manifest)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            metadata = check_by_name(result, "logical_cost_improvement_metadata")
            self.assertFalse(metadata["passed"])
            self.assertIn("raw_pass_indices", json.dumps(metadata["detail"]))

    def test_logical_cost_improvement_rejects_symlinked_legacy_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, summary = make_v4_fixture(frame_dir)
            add_logical_cost_improvement_fixture(frame_dir, manifest, summary)
            improvement_dir = frame_dir.parent.parent / "instrumentation" / "improvements"
            external = Path(directory) / "external_improvements"
            improvement_dir.rename(external)
            improvement_dir.symlink_to(external, target_is_directory=True)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            closure = check_by_name(
                result, "logical_cost_improvement_legacy_pass_closure"
            )
            self.assertFalse(closure["passed"])
            self.assertIn("symlink", json.dumps(closure["detail"]))

    def test_core_map_symlink_outside_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            relative_path = str(manifest["maps"][0]["path"])
            map_path = frame_dir / relative_path
            external = Path(directory) / f"external{map_path.suffix}"
            map_path.rename(external)
            map_path.symlink_to(external)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"], result)
            ownership = check_by_name(result, "v3_manifest_paths_safe_unique")
            self.assertFalse(ownership["passed"])
            self.assertIn("symlink", json.dumps(ownership["detail"]))

    def test_hysteresis_metadata_uses_per_frame_prior_execution_and_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            frame_dir.mkdir(parents=True)
            run_metadata_path = write_v4_run_metadata(frame_dir)
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            run_metadata["cuda_patchmatch_parameters"].update({
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            })
            run_metadata_path.write_text(
                json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
            )
            (frame_dir / "summary.json").write_text(json.dumps({
                "cuda_patchmatch_parameters": {
                    "low_resolution_prior_available": False,
                },
            }), encoding="utf-8")

            metadata = validator.low_texture_update_hysteresis_metadata(frame_dir)

            self.assertTrue(metadata["available"])
            self.assertTrue(metadata["configured"])
            self.assertFalse(metadata["execution_available"])
            self.assertFalse(metadata["enabled"])
            self.assertFalse(metadata["low_resolution_prior_available"])
            self.assertEqual(
                metadata["execution_availability_basis"],
                "per_frame_summary_low_resolution_prior",
            )
            self.assertIn("no coarse-resolution prior", metadata["execution_unavailable_reason"])

            (frame_dir / "summary.json").write_text(json.dumps({
                "cuda_patchmatch_parameters": {
                    "low_resolution_prior_available": True,
                    "estimation_iterations": 0,
                },
            }), encoding="utf-8")
            no_iterations = validator.low_texture_update_hysteresis_metadata(frame_dir)
            self.assertTrue(no_iterations["configured"])
            self.assertFalse(no_iterations["execution_available"])
            self.assertEqual(no_iterations["estimation_iterations"], 0)
            self.assertIn(
                "estimation_iterations is zero",
                no_iterations["execution_unavailable_reason"],
            )

            (frame_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            legacy = validator.low_texture_update_hysteresis_metadata(frame_dir)
            self.assertTrue(legacy["configured"])
            self.assertTrue(legacy["execution_available"])
            self.assertTrue(legacy["enabled"])
            self.assertIsNone(legacy["low_resolution_prior_available"])
            self.assertEqual(
                legacy["execution_availability_basis"],
                "legacy_configured_fallback",
            )

    def test_candidate_order_contract_closes_execution_fields_and_reason(self) -> None:
        reason = (
            "configured mechanism did not execute because this level/stage has no "
            "coarse-resolution prior"
        )
        metadata = {
            "available": True,
            "configured": True,
            "execution_available": False,
            "enabled": False,
            "execution_unavailable_reason": reason,
        }
        contract = {
            "schema_version": 1,
            "scope": "enabled_low_texture_hysteresis_iterative_states",
            "configured": True,
            "execution_available": False,
            "available": False,
            "unavailable_reason": reason,
            "retained_gap_unavailable_when_raw_best_suppressed": True,
            "raw_signals": sorted(
                validator.V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
            ),
        }
        valid, detail = validator.validate_v4_candidate_order_statistics_contract(
            {"available": True, "candidate_order_statistics": contract}, metadata
        )
        self.assertTrue(valid, detail)

        contract["execution_available"] = True
        valid, detail = validator.validate_v4_candidate_order_statistics_contract(
            {"available": True, "candidate_order_statistics": contract}, metadata
        )
        self.assertFalse(valid)
        self.assertTrue(any("execution_available" in error for error in detail["errors"]))

        contract["execution_available"] = False
        contract["unavailable_reason"] = ""
        valid, detail = validator.validate_v4_candidate_order_statistics_contract(
            {"available": True, "candidate_order_statistics": contract}, metadata
        )
        self.assertFalse(valid)
        self.assertTrue(any("unavailable_reason" in error for error in detail["errors"]))

    def test_exact_iteration_schema3_closes_hysteresis_table_to_maps_and_atomic_counters(self) -> None:
        logical_maps = {
            ("low_texture_update_eligible_exact", 0): ({}, Path("eligible.pfm"), scalar(1)),
            ("candidate_accepted_mask_exact", 0): (
                {}, Path("accepted.pfm"), scalar(float((1 << 1) | (1 << 9)))
            ),
            ("low_texture_update_rejected_count_exact", 0): (
                {}, Path("rejected_count.pfm"), scalar(1)
            ),
            ("low_texture_update_rejected_mask_exact", 0): (
                {}, Path("rejected_mask.pfm"), scalar(1)
            ),
            ("low_texture_update_required_gain_exact", 0): (
                {}, Path("required.pfm"), scalar(0.0005)
            ),
            ("low_texture_update_best_proposed_gain_exact", 0): (
                {}, Path("best.pfm"), scalar(0.0004)
            ),
        }
        rows = [
            {
                "logical_iteration": "-1", "pyramid_level": "0",
                **{field: "" for field in validator.V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS},
            },
            {
                "logical_iteration": "0", "pyramid_level": "0",
                "low_texture_gate_eligible": "4",
                "low_texture_propagation_accepted": "4",
                "low_texture_propagation_rejected": "4",
                "low_texture_refinement_accepted": "0",
                "low_texture_refinement_rejected": "0",
                "low_texture_required_gain_sum": "0.0020000000949949026",
                "low_texture_best_proposed_gain_sum": "0.0015999999595806003",
            },
        ]
        atomic_rows = [{
            "iteration": "0", "scale_level": "0",
            **{field: rows[1][field] for field in validator.V4_LOW_TEXTURE_UPDATE_COUNTER_FIELDS},
        }]
        checks: list[dict] = []
        result = validator.validate_v4_exact_iteration_table(
            rows=rows,
            columns=set(rows[0]),
            schema_version=3,
            logical_maps=logical_maps,
            expected_iterations={-1, 0},
            pyramid_level=0,
            hysteresis_metadata={
                "available": True, "enabled": True,
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            },
            atomic_rows=atomic_rows,
            tolerance=0.0,
            check=lambda name, passed, detail: checks.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )

        self.assertTrue(result["enabled"])
        self.assertTrue(all(row["passed"] for row in checks), checks)
        rows[1]["low_texture_propagation_rejected"] = "3"
        corrupted: list[dict] = []
        validator.validate_v4_exact_iteration_table(
            rows=rows, columns=set(rows[0]), schema_version=3,
            logical_maps=logical_maps, expected_iterations={-1, 0}, pyramid_level=0,
            hysteresis_metadata={
                "available": True, "enabled": True,
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            },
            atomic_rows=atomic_rows, tolerance=0.0,
            check=lambda name, passed, detail: corrupted.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )
        closure = next(
            row for row in corrupted
            if row["name"] == "v4_exact_iteration_low_texture_map_closure"
        )
        self.assertFalse(closure["passed"])

    def test_low_texture_hysteresis_optional_maps_validate_domains_and_formula(self) -> None:
        arrays = {
            "low_texture_update_eligible_exact": scalar(1),
            "low_texture_update_ambiguity_exact": scalar(0.5),
            "low_texture_update_required_gain_exact": scalar(0.0005),
            "low_texture_update_best_proposed_gain_exact": scalar(0.0004),
            "low_texture_update_rejected_mask_exact": scalar(1),
            "low_texture_update_would_have_won_source_exact": scalar(2),
            "low_texture_update_rejected_count_exact": scalar(1),
        }
        logical_maps = {
            (signal, 0): ({"signal": signal}, Path(f"{signal}.pfm"), values)
            for signal, values in arrays.items()
        }
        entries = [
            {
                "signal": signal, "logical_iteration": 0, "role": "logical_event",
                "measurement_quality": "exact", "measurement_basis": "hot kernel",
                "semantics": "complete logical iteration",
            }
            for signal in arrays
        ]
        checks: list[dict] = []

        result = validator.validate_v4_low_texture_update_hysteresis(
            entries=entries,
            logical_maps=logical_maps,
            num_iterations=1,
            metadata={
                "available": True, "enabled": True,
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            },
            tolerance=0.0,
            check=lambda name, passed, detail: checks.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )

        self.assertTrue(result["enabled"])
        self.assertTrue(all(row["passed"] for row in checks), checks)
        logical_maps[("low_texture_update_required_gain_exact", 0)] = (
            {}, Path("corrupt.pfm"), scalar(0.01)
        )
        corrupted: list[dict] = []
        validator.validate_v4_low_texture_update_hysteresis(
            entries=entries, logical_maps=logical_maps, num_iterations=1,
            metadata={
                "available": True, "enabled": True,
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            },
            tolerance=0.0,
            check=lambda name, passed, detail: corrupted.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )
        domains = next(
            row for row in corrupted if row["name"] == "v4_low_texture_update_domains"
        )
        self.assertFalse(domains["passed"])

    def test_low_texture_hysteresis_suppressed_raw_minimum_closes_separate_orders(self) -> None:
        arrays = {
            "low_texture_update_eligible_exact": scalar(1),
            "low_texture_update_ambiguity_exact": scalar(0.5),
            "low_texture_update_required_gain_exact": scalar(0.0005),
            "low_texture_update_best_proposed_gain_exact": scalar(0.1),
            "low_texture_update_rejected_mask_exact": scalar(1),
            "low_texture_update_would_have_won_source_exact": scalar(2),
            "low_texture_update_rejected_count_exact": scalar(1),
            "candidate_raw_best_cost_exact": scalar(0.4),
            "candidate_raw_runner_up_cost_exact": scalar(0.6),
            "gap_raw_best_runner_up_exact": scalar(0.2),
            "candidate_retained_minus_raw_best_exact": scalar(0.1),
            "candidate_raw_suppression_identity_exact": rgba(1, 2, 0, 2),
            "candidate_winner_cost_exact": scalar(0.5),
            "candidate_runner_up_cost_exact": scalar(-1.0),
            "gap_winner_runner_up_exact": scalar(-1.0),
            "candidate_identity_exact": rgb(0, 255, 0),
        }
        logical_maps = {
            (signal, 0): ({"signal": signal}, Path(f"{signal}.pfm"), values)
            for signal, values in arrays.items()
        }
        entries = [
            {
                "signal": signal,
                "logical_iteration": 0,
                "role": "logical_event",
                "measurement_quality": "exact",
                "measurement_basis": "production hot kernel",
                "semantics": "complete logical iteration",
            }
            for signal in (
                validator.V4_LOW_TEXTURE_UPDATE_EVENT_SIGNALS
                | validator.V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
            )
        ]
        checks: list[dict] = []
        metadata = {
            "available": True,
            "enabled": True,
            "low_texture_update_min_gain": 0.001,
            "low_texture_update_gate": 1,
            "raw_order_statistics_version": 1,
        }

        result = validator.validate_v4_low_texture_update_hysteresis(
            entries=entries,
            logical_maps=logical_maps,
            num_iterations=1,
            metadata=metadata,
            tolerance=0.0,
            check=lambda name, passed, detail: checks.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )

        self.assertTrue(all(row["passed"] for row in checks), checks)
        raw = result["raw_order_statistics"]["per_iteration"]["0"]
        self.assertEqual(raw["suppressed_raw_best_pixels"], WIDTH * HEIGHT)
        logical_maps[("gap_winner_runner_up_exact", 0)] = (
            {}, Path("mixed_retained_raw_gap.pfm"), scalar(0.2)
        )
        corrupted: list[dict] = []
        validator.validate_v4_low_texture_update_hysteresis(
            entries=entries,
            logical_maps=logical_maps,
            num_iterations=1,
            metadata=metadata,
            tolerance=0.0,
            check=lambda name, passed, detail: corrupted.append({
                "name": name, "passed": passed, "detail": detail,
            }),
        )
        closure = next(
            row for row in corrupted
            if row["name"] == "v4_low_texture_update_raw_order_closure"
        )
        self.assertFalse(closure["passed"])

    def test_optional_manifests_own_maps_and_supply_terminal_dmap_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            core = add_core_terminal_maps(frame_dir, manifest)
            terminal_depth = scalar(2.0)
            postprocess_confidence = scalar(0.6)
            terminal_confidence = scalar(0.8)
            add_optional_stage_manifests(
                frame_dir,
                depth=terminal_depth,
                confidence_after_postprocess=postprocess_confidence,
                confidence_final=terminal_confidence,
            )
            dmap = {
                "depth_map": terminal_depth,
                "normal_map": core["normal_map"],
                "confidence_map": terminal_confidence,
            }

            with mock.patch.object(validator, "load_dmap", return_value=dmap):
                result = validator.validate(validator.Arguments(
                    frame_dir=frame_dir,
                    instrumented_dmap=frame_dir / "depth0001.dmap",
                ))

            self.assertTrue(result["valid"], result)
            self.assertTrue(
                check_by_name(result, "v3_manifest_indexes_all_map_files")["passed"]
            )
            self.assertTrue(
                check_by_name(result, "postprocess_filters_indexes_all_map_files")["passed"]
            )
            self.assertTrue(
                check_by_name(result, "confidence_adjustment_indexes_all_map_files")["passed"]
            )
            self.assertEqual(
                result["dmap_terminal_state"]["depth_map"]["source"],
                "postprocess_filters/00_remove_speckles_depth_after.pfm",
            )
            self.assertEqual(
                result["dmap_terminal_state"]["confidence_map"]["source"],
                "confidence_adjustment/confidence_final.pfm",
            )
            self.assertEqual(result["dmap_consistency_max_abs"]["depth_map"], 0.0)
            self.assertEqual(
                result["dmap_consistency_max_abs"]["confidence_from_cost"], 0.0
            )

    def test_current_d2_terminal_consistency_uses_codec_quantization_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            core = add_core_terminal_maps(frame_dir, manifest)
            quantized = {
                "format": "D2",
                "depth_exponent": 0,
                "confidence_scale": 1.0,
                "depth_map": core["depth_map"] + np.float32(0.0009),
                "normal_map": core["normal_map"] + np.float32(0.00005),
                "confidence_map": core["confidence_map"] + np.float32(0.0019),
            }

            with mock.patch.object(validator, "load_dmap", return_value=quantized):
                result = validator.validate(validator.Arguments(
                    frame_dir=frame_dir,
                    instrumented_dmap=frame_dir / "depth0001.dmap",
                ))

            self.assertTrue(result["valid"], result)
            consistency = check_by_name(result, "instrumented_dmap_consistency")
            self.assertTrue(consistency["passed"])
            self.assertGreater(
                result["dmap_consistency_limits"]["depth_map"],
                result["dmap_consistency_max_abs"]["depth_map"],
            )

    def test_optional_manifest_rejects_unindexed_and_bad_byte_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            add_core_terminal_maps(frame_dir, manifest)
            add_optional_stage_manifests(
                frame_dir,
                depth=scalar(2.0),
                confidence_after_postprocess=scalar(0.6),
                confidence_final=scalar(0.8),
            )
            artifact_path = frame_dir / "postprocess_filters.json"
            artifact = json.loads(artifact_path.read_text())
            artifact["maps"][0]["file_bytes"] += 1
            artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
            from test_validate_dmap_instrumentation_v3 import write_pfm
            write_pfm(frame_dir / "postprocess_filters/unindexed.pfm", scalar(0.0))

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            self.assertFalse(
                check_by_name(result, "postprocess_filters_file_bytes")["passed"]
            )
            self.assertFalse(
                check_by_name(result, "postprocess_filters_indexes_all_map_files")["passed"]
            )
            self.assertTrue(
                check_by_name(result, "v3_manifest_indexes_all_map_files")["passed"]
            )

    def test_optional_normal_change_without_terminal_normal_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            core = add_core_terminal_maps(frame_dir, manifest)
            add_optional_stage_manifests(
                frame_dir,
                depth=scalar(2.0),
                confidence_after_postprocess=scalar(0.6),
                confidence_final=scalar(0.8),
                normal_changed_pixels=1,
                include_terminal_normal=False,
            )
            dmap = {
                "depth_map": scalar(2.0),
                "normal_map": core["normal_map"],
                "confidence_map": scalar(0.8),
            }

            with mock.patch.object(validator, "load_dmap", return_value=dmap):
                result = validator.validate(validator.Arguments(
                    frame_dir=frame_dir,
                    instrumented_dmap=frame_dir / "depth0001.dmap",
                ))

            self.assertFalse(result["valid"])
            availability = check_by_name(
                result, "instrumented_dmap_terminal_availability"
            )
            self.assertFalse(availability["passed"])
            self.assertFalse(
                result["dmap_terminal_state"]["normal_map"]["available"]
            )
            self.assertIn(
                "could not prove identity",
                result["dmap_terminal_state"]["normal_map"]["unavailable_reason"],
            )

    def test_disabled_optional_stages_fall_back_to_core_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            core = add_core_terminal_maps(frame_dir, manifest)
            add_optional_stage_manifests(
                frame_dir,
                depth=core["depth_map"],
                confidence_after_postprocess=core["confidence_map"],
                confidence_final=None,
                stages_enabled=False,
            )

            with mock.patch.object(validator, "load_dmap", return_value=core):
                result = validator.validate(validator.Arguments(
                    frame_dir=frame_dir,
                    instrumented_dmap=frame_dir / "depth0001.dmap",
                ))

            self.assertTrue(result["valid"], result)
            self.assertEqual(
                result["dmap_terminal_state"]["depth_map"]["source"],
                "maps/depth_final_after_filter.pfm",
            )

    def test_schema_v4_exact_fixture_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            make_v4_fixture(frame_dir)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            self.assertTrue(result["exact_validation"]["available"])
            self.assertTrue(
                check_by_name(result, "v4_exact_view_samples_metadata")["passed"]
            )
            self.assertEqual(
                result["exact_validation"]["view_samples_metadata"]["view_samples"],
                32,
            )

    def test_schema_v4_configured_hysteresis_without_prior_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, summary = make_v4_fixture(frame_dir)
            reason = (
                "configured mechanism did not execute because this level/stage has no "
                "coarse-resolution prior"
            )
            summary["cuda_patchmatch_parameters"] = {
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
                "low_resolution_prior_available": False,
            }
            (frame_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            run_metadata_path = frame_dir.parent.parent / "run_metadata.json"
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            run_metadata["cuda_patchmatch_parameters"].update({
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
            })
            run_metadata_path.write_text(
                json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
            )
            manifest["exact_capture"]["candidate_order_statistics"] = {
                "schema_version": 1,
                "scope": "enabled_low_texture_hysteresis_iterative_states",
                "configured": True,
                "execution_available": False,
                "available": False,
                "unavailable_reason": reason,
                "retained_gap_unavailable_when_raw_best_suppressed": True,
                "raw_signals": sorted(
                    validator.V4_LOW_TEXTURE_UPDATE_RAW_ORDER_EVENT_SIGNALS
                ),
            }
            save_manifest(frame_dir, manifest)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            hysteresis = result["exact_validation"]["low_texture_update_hysteresis"]
            self.assertFalse(hysteresis["enabled"])
            self.assertFalse(hysteresis["metadata"]["execution_available"])
            self.assertFalse(
                hysteresis["metadata"]["low_resolution_prior_available"]
            )
            self.assertTrue(
                check_by_name(
                    result, "v4_exact_candidate_order_statistics_contract"
                )["passed"]
            )

    def test_schema_v4_accepts_63_view_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            make_v4_fixture(frame_dir, view_samples=63)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            self.assertEqual(
                result["exact_validation"]["view_samples_metadata"]["view_samples"],
                63,
            )
            self.assertEqual(
                result["exact_validation"]["views"]["0"]["expected_weight_sum"],
                63.0,
            )
            self.assertEqual(
                result["exact_validation"]["views"]["0"]["invalid_weight_pixels"],
                0,
            )

    def test_schema_v4_rejects_missing_view_samples_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            make_v4_fixture(frame_dir)
            rewrite_v4_view_samples_metadata(frame_dir)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            metadata = result["exact_validation"]["view_samples_metadata"]
            self.assertFalse(result["valid"])
            self.assertFalse(
                check_by_name(result, "v4_exact_view_samples_metadata")["passed"]
            )
            self.assertFalse(metadata["available"])
            self.assertIn("missing", metadata["unavailable_reason"])

    def test_schema_v4_rejects_non_integer_view_samples_metadata(self) -> None:
        for value in ("63", 63.0, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                frame_dir = v4_frame_dir(directory)
                make_v4_fixture(frame_dir)
                rewrite_v4_view_samples_metadata(frame_dir, value)

                result = validator.validate(validator.Arguments(frame_dir=frame_dir))

                metadata = result["exact_validation"]["view_samples_metadata"]
                self.assertFalse(result["valid"])
                self.assertFalse(metadata["available"])
                self.assertIn("not an integer", metadata["unavailable_reason"])

    def test_schema_v4_rejects_out_of_range_view_samples_metadata(self) -> None:
        for value in (0, 64):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                frame_dir = v4_frame_dir(directory)
                make_v4_fixture(frame_dir)
                rewrite_v4_view_samples_metadata(frame_dir, value)

                result = validator.validate(validator.Arguments(frame_dir=frame_dir))

                metadata = result["exact_validation"]["view_samples_metadata"]
                self.assertFalse(result["valid"])
                self.assertFalse(metadata["available"])
                self.assertIn("[1,63]", metadata["unavailable_reason"])

    def test_schema_v4_warns_for_formula_backed_exact_prior_weight_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            variance = np.float32(-4.7447193e-6)
            for signal, value in (
                ("reference_variance_production_exact", variance),
                ("depth_prior_weight_production_exact", legacy_prior_weight(float(variance))),
            ):
                entry = next(
                    item for item in manifest["maps"]
                    if item["signal"] == signal and item["logical_iteration"] == 0
                )
                rewrite_entry(frame_dir, entry, scalar(float(value)))
            write_v4_run_metadata(frame_dir)
            save_manifest(frame_dir, manifest)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            domain = check_by_name(result, "v4_exact_depth_prior_weight_domain")
            self.assertTrue(result["valid"], result)
            self.assertTrue(domain["passed"], domain)
            self.assertEqual(domain["detail"]["fatal_invalid_domain_pixels"], 0)
            self.assertEqual(
                domain["detail"]["known_production_warning_pixels"], WIDTH * HEIGHT
            )
            self.assertEqual(len(result["warnings"]), 1)
            self.assertEqual(
                result["warnings"][0]["signal"], "depth_prior_weight_production_exact"
            )

    def test_schema_v4_rejects_unexplained_exact_prior_weight_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            for signal, value in (
                ("reference_variance_production_exact", -4.7447193e-6),
                ("depth_prior_weight_production_exact", 1.01),
            ):
                entry = next(
                    item for item in manifest["maps"]
                    if item["signal"] == signal and item["logical_iteration"] == 0
                )
                rewrite_entry(frame_dir, entry, scalar(value))
            write_v4_run_metadata(frame_dir)
            save_manifest(frame_dir, manifest)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            domain = check_by_name(result, "v4_exact_depth_prior_weight_domain")
            self.assertFalse(result["valid"])
            self.assertFalse(domain["passed"])
            self.assertEqual(domain["detail"]["fatal_invalid_domain_pixels"], WIDTH * HEIGHT)
            self.assertEqual(result["warnings"], [])

    def test_schema_v4_rejects_component_and_weight_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            manifest, _summary = make_v4_fixture(frame_dir)
            total = next(
                entry for entry in manifest["maps"]
                if entry["signal"] == "cost_total_production_exact"
                and entry["logical_iteration"] == 0
            )
            rewrite_entry(frame_dir, total, scalar(0.75))
            selection = next(
                entry for entry in manifest["maps"]
                if entry["signal"] == "view_selection_state_exact"
                and entry["logical_iteration"] == 0
                and entry["source_view_index"] == 0
            )
            rewrite_entry(frame_dir, selection, rgb(15, 0, 1))
            save_manifest(frame_dir, manifest)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            self.assertFalse(check_by_name(result, "v4_exact_arithmetic_closure")["passed"])
            self.assertFalse(check_by_name(result, "v4_exact_domains")["passed"])

    def test_schema_v4_accepts_filter_rejection_decomposition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            add_filter_rejection_decomposition(frame_dir, summary)

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            decomposition = check_by_name(result, "filter_rejection_decomposition")
            self.assertTrue(result["valid"], result)
            self.assertTrue(decomposition["passed"], decomposition)
            self.assertEqual(decomposition["detail"]["valid_after_keep_cost"], 3)
            self.assertEqual(decomposition["detail"]["rejected_keep_cost"], 1)
            self.assertEqual(decomposition["detail"]["rejected_ignore_mask"], 1)
            self.assertEqual(decomposition["detail"]["masked_reason_count"], 1)

    def test_schema_v4_accepts_requested_but_unavailable_ignore_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            filtering = add_filter_rejection_decomposition(frame_dir, summary)
            unavailable = {
                "requested": True,
                "label": 7,
                "load_attempted": True,
                "loaded": False,
                "status": "unavailable",
                "source_path": "missing.mask.png",
                "rejection_count_available": False,
                "unavailable_reason": "requested ignore mask label 7 could not be loaded",
            }
            summary.update(
                num_rejected_by_ignore_mask=None,
                num_valid_after_filter=3,
                num_rejected_by_filter=1,
                valid_ratio_after_filter=0.75,
                ignore_mask=unavailable,
                supporting_view_histogram=[1, 3, 0, 0, 0],
            )
            filtering.update(
                num_rejected_by_ignore_mask=None,
                num_valid_after_filter=3,
                num_rejected_by_filter=1,
                valid_ratio_after_filter=0.75,
                ignore_mask=unavailable,
                supporting_view_histogram=[1, 3, 0, 0, 0],
            )
            filtering["rejection_reasons"]["masked"] = None
            (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (frame_dir / "filtering.json").write_text(
                json.dumps(filtering, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            availability = check_by_name(result, "ignore_mask_availability")
            self.assertTrue(availability["passed"], availability)
            self.assertEqual(result["ignore_mask"]["status"], "unavailable")
            self.assertIsNone(result["ignore_mask"]["rejected_count"])

    def test_schema_v4_rejects_unavailable_ignore_mask_reported_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            filtering = add_filter_rejection_decomposition(frame_dir, summary)
            unavailable = {
                "requested": True,
                "label": 7,
                "load_attempted": True,
                "loaded": False,
                "status": "unavailable",
                "source_path": "missing.mask.png",
                "rejection_count_available": False,
                "unavailable_reason": "requested ignore mask label 7 could not be loaded",
            }
            summary.update(
                num_rejected_by_ignore_mask=0,
                num_valid_after_filter=3,
                num_rejected_by_filter=1,
                valid_ratio_after_filter=0.75,
                ignore_mask=unavailable,
            )
            filtering.update(
                num_rejected_by_ignore_mask=0,
                num_valid_after_filter=3,
                num_rejected_by_filter=1,
                valid_ratio_after_filter=0.75,
                ignore_mask=unavailable,
            )
            filtering["rejection_reasons"]["masked"] = 0
            (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (frame_dir / "filtering.json").write_text(
                json.dumps(filtering, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            availability = check_by_name(result, "ignore_mask_availability")
            self.assertFalse(availability["passed"], availability)

    def test_schema_v4_rejects_filter_count_domain_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            filtering = add_filter_rejection_decomposition(frame_dir, summary)
            filtering["num_pixels_total"] = -1
            (frame_dir / "filtering.json").write_text(
                json.dumps(filtering, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            domains = check_by_name(result, "filter_count_domains")
            self.assertFalse(domains["passed"], domains)

    def test_schema_v4_rejects_keep_ratio_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            filtering = add_filter_rejection_decomposition(frame_dir, summary)
            summary["valid_ratio_after_keep_cost_filter"] = 0.5
            filtering["valid_ratio_after_keep_cost_filter"] = 0.5
            (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (frame_dir / "filtering.json").write_text(
                json.dumps(filtering, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            closure = check_by_name(result, "filter_keep_ratio_closure")
            self.assertFalse(closure["passed"], closure)

    def test_schema_v4_rejects_support_from_finally_invalid_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            add_filter_rejection_decomposition(frame_dir, summary)
            # This has the right total but assigns one filtered pixel positive support.
            summary["supporting_view_histogram"] = [1, 3, 0, 0, 0]
            (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            support = check_by_name(result, "support_histogram_final_validity")
            self.assertFalse(support["passed"], support)

    def test_schema_v4_rejects_corrupted_masked_rejection_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            filtering = add_filter_rejection_decomposition(frame_dir, summary)
            # Preserve the reason total so this isolates masked-count attribution.
            filtering["rejection_reasons"]["low_score"] = 2
            filtering["rejection_reasons"]["masked"] = 0
            (frame_dir / "filtering.json").write_text(
                json.dumps(filtering, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            decomposition = check_by_name(result, "filter_rejection_decomposition")
            self.assertFalse(result["valid"])
            self.assertFalse(decomposition["passed"], decomposition)
            self.assertTrue(check_by_name(result, "summary_pixel_counts")["passed"])
            self.assertEqual(decomposition["detail"]["rejected_ignore_mask"], 1)
            self.assertEqual(decomposition["detail"]["rejection_reason_total"], 2)
            self.assertEqual(decomposition["detail"]["masked_reason_count"], 0)

    def test_schema_v4_accepts_explicit_budget_unavailability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, summary = make_fixture(frame_dir, 3)
            observer_sidecars = {
                "complete": True,
                "write_error_count": 0,
                "write_errors": [],
            }
            manifest.update(
                schema_version=4,
                measurement_model=validator.V4_MEASUREMENT_MODEL,
                observer_sidecars=observer_sidecars.copy(),
                exact_capture={
                    "requested": True,
                    "available": False,
                    "unavailable_reason": "exact capture exceeds frame budget",
                    "num_views": 2,
                },
            )
            summary.update({
                "schema_version": 4,
                "schema_name": "openmvs.dmap.frame_summary",
                "image_id": 1,
                "image_name": "0000",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
                "observer_sidecars": observer_sidecars.copy(),
                "completion_marker": {
                    "schema_name": "openmvs.dmap.capture_complete",
                    "schema_version": 1,
                    "path": "capture_complete.json",
                    "maps_complete": True,
                    "eligible": True,
                },
            })
            save_manifest(frame_dir, manifest)
            (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (frame_dir / "capture_complete.json").write_text(json.dumps({
                "schema_name": "openmvs.dmap.capture_complete",
                "schema_version": 1,
                "capture_kind": "maps",
                "image_id": summary["image_id"],
                "image_name": summary["image_name"],
                "estimation_stage": summary["estimation_stage"],
                "geometric_iteration": summary["geometric_iteration"],
                "maps_complete": True,
                "observer_sidecars_complete": True,
                "map_manifest": {
                    "path": "map_manifest.json",
                    "schema_version": 4,
                    "complete": True,
                },
                "summary": {"path": "summary.json", "schema_version": 4},
            }, indent=2) + "\n")

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertTrue(result["valid"], result)
            self.assertFalse(result["exact_validation"]["available"])

    def test_schema_v4_rejects_observer_sidecar_write_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            _manifest, summary = make_v4_fixture(frame_dir)
            summary["observer_sidecars"] = {
                "complete": False,
                "write_error_count": 1,
                "write_errors": [
                    {"artifact": "iteration.csv", "pyramid_level": 1}
                ],
            }
            (frame_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            self.assertFalse(
                check_by_name(result, "v4_observer_sidecars_complete")["passed"]
            )

    def test_schema_v4_rejects_marker_without_sidecar_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = v4_frame_dir(directory)
            make_v4_fixture(frame_dir)
            marker_path = frame_dir / "capture_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["observer_sidecars_complete"] = False
            marker_path.write_text(json.dumps(marker, indent=2) + "\n")

            result = validator.validate(validator.Arguments(frame_dir=frame_dir))

            self.assertFalse(result["valid"])
            self.assertFalse(
                check_by_name(result, "v4_capture_completion_marker")["passed"]
            )


if __name__ == "__main__":
    unittest.main()
