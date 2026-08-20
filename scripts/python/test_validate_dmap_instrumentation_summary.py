#!/usr/bin/env python3
"""Focused summary-only instrumentation validation tests."""

from __future__ import annotations

import csv
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


def reference_patch_layout() -> dict:
    axis = [-4, -2, 0, 2, 4]
    return {
        "schema_name": "openmvs.dmap.reference_patch_layout",
        "schema_version": 1,
        "kind": "fixed_cartesian_grid",
        "coordinate_domain": "reference_pyramid_pixels",
        "sample_position": "integer_offset_from_pixel_center",
        "texel_center_offset": 0.5,
        "texture_address_mode_configured": "wrap",
        "texture_address_mode_effective": "clamp",
        "texture_address_mode_effective_basis": (
            "cuda_runtime_unnormalized_wrap_is_clamped"
        ),
        "texture_coordinates_normalized": False,
        "texture_filter_mode": "linear",
        "half_window_pixels": 4,
        "step_pixels": 2,
        "sample_count": 25,
        "sample_offsets_pixels": [[x, y] for y in axis for x in axis],
        "layout_provenance": (
            "observer_contract_source_checked_against_cuda_scoring_constants"
        ),
        "sample_locations_captured_by_kernel": False,
        "sample_values_captured_by_kernel": False,
        "source_view_footprints_captured_by_kernel": False,
    }


def claim_reference_patch_layout(
    frame_dir: Path,
    summary: dict,
    *,
    capability: object = True,
    run_layout: object | None = None,
    summary_layout: object | None = None,
) -> None:
    layout = reference_patch_layout()
    run_value = layout if run_layout is None else run_layout
    summary_value = layout if summary_layout is None else summary_layout
    instrumentation_root = (
        frame_dir.parent.parent if frame_dir.parent.name == "depthmaps"
        else frame_dir.parent
    )
    metadata_path = instrumentation_root / "run_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file() else {
            "schema_name": "openmvs.dmap.run",
            "schema_version": 4,
        }
    )
    metadata.setdefault("instrumentation", {})["capabilities"] = {
        "reference_patch_layout_contract": capability,
        "reference_patch_sample_locations": False,
        "reference_patch_sample_values": False,
        "source_view_patch_footprints": False,
    }
    metadata.setdefault("cuda_patchmatch_parameters", {})[
        "reference_patch_layout"
    ] = run_value
    write_json(metadata_path, metadata)
    summary.setdefault("cuda_patchmatch_parameters", {})[
        "reference_patch_layout"
    ] = summary_value
    summary_path = frame_dir / "summary.json"
    write_json(summary_path, summary)
    for marker_name in (
        "summary_complete.json", "prefilter_capture_complete.json",
        "capture_complete.json",
    ):
        marker_path = frame_dir / marker_name
        if not marker_path.is_file():
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary_reference = marker.get("summary")
        if isinstance(summary_reference, dict) and "bytes" in summary_reference:
            summary_reference["bytes"] = summary_path.stat().st_size
            write_json(marker_path, marker)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_pfm(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "PF" if values.ndim == 3 else "Pf"
    with path.open("wb") as handle:
        handle.write(
            f"{header}\n{values.shape[1]} {values.shape[0]}\n-1.0\n".encode("ascii")
        )
        handle.write(np.flipud(values).astype("<f4", copy=False).tobytes())


def write_process_census(
    frame_dir: Path, image_id: int = 7, terminal_valid: int = 3,
) -> Path:
    instrumentation_root = (
        frame_dir.parent.parent if frame_dir.parent.name == "depthmaps" else frame_dir.parent
    )
    path = instrumentation_root / "instrumentation" / "counters.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_fields = [
        f"selected_views_{index}"
        for index in range(validator.PATCHMATCH_SELECTED_VIEW_BIN_COUNT)
    ]
    candidate_fields = [
        f"candidate_{kind}_{candidate}"
        for kind in ("tested", "finite", "accepted")
        for candidate in ("init", "propagation", "random", "refinement")
    ]
    fields = [
        "image_id", "scale_number", "width", "height", "pass_index", "phase",
        "iteration", "processed", "valid_depth", "invalid_depth", "accepted",
        "component_samples", *selected_fields, *candidate_fields,
    ]
    rows = []
    for iteration, valid, invalid, accepted, zero_selected in (
        (-1, 4, 0, 4, 0),
        (0, terminal_valid, 4 - terminal_valid, 1, 1),
    ):
        row = {field: 0 for field in fields}
        row.update({
            "image_id": image_id,
            "scale_number": 0,
            "width": 2,
            "height": 2,
            "pass_index": iteration + 1,
            "phase": "initialization" if iteration == -1 else "iteration",
            "iteration": iteration,
            "processed": 4,
            "valid_depth": valid,
            "invalid_depth": invalid,
            "accepted": accepted,
            "component_samples": 4 - zero_selected,
            "selected_views_0": zero_selected,
            "selected_views_2": 4 - zero_selected,
        })
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def rewrite_census_row(
    frame_dir: Path, iteration: int, *, terminal_valid: int = 3, **changes: object,
) -> None:
    path = write_process_census(frame_dir, terminal_valid=terminal_valid)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if int(row["iteration"]) == iteration:
            row.update({key: str(value) for key, value in changes.items()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_fixture(frame_dir: Path) -> tuple[dict, Path]:
    frame_dir.mkdir(parents=True)
    completion = {
        "schema_name": "openmvs.dmap.summary_complete",
        "schema_version": 1,
        "path": "summary_complete.json",
        "maps_complete": False,
        "eligible": True,
    }
    summary = {
        "schema_name": "openmvs.dmap.frame_summary",
        "schema_version": 4,
        "image_id": 7,
        "image_name": "images/0007.jpg",
        "safe_image_name": "0007",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "width": 2,
        "height": 2,
        "scale_level": 0,
        "cuda_patchmatch_parameters": {"estimation_iterations": 1},
        "num_pixels_total": 4,
        "num_valid_before_filter": 3,
        "num_invalid_before_filter": 1,
        "num_valid_after_filter": 2,
        "num_rejected_by_filter": 1,
        "valid_ratio_after_filter": 0.5,
        "candidate_accounting_mode": "unavailable_post_pass_snapshot",
        "confidence_gap_mode": "post_pass_current_plus_eight_neighbors",
        "unavailable_signals": sorted(validator.SUMMARY_UNAVAILABLE_EXACT_SIGNALS),
        "observer_sidecars": {
            "complete": True,
            "write_error_count": 0,
            "write_errors": [],
        },
        "completion_marker": completion,
        "resource_plan": {
            "decision": "summary",
            "summary_available": True,
            "maps_requested": False,
            "maps_available": False,
            "exact_requested": False,
            "exact_available": False,
            "exact_unavailable_reason": "exact capture was not requested",
        },
    }
    write_json(frame_dir / "summary.json", summary)
    write_json(frame_dir / "summary_complete.json", {
        "schema_name": "openmvs.dmap.summary_complete",
        "schema_version": 1,
        "capture_kind": "summary_only",
        "image_id": 7,
        "image_name": "images/0007.jpg",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "summary_complete": True,
        "maps_requested": False,
        "maps_complete": False,
        "observer_sidecars_complete": True,
        "summary": {"path": "summary.json", "schema_version": 4},
    })
    write_json(frame_dir / "filtering.json", {
        key: summary[key] for key in (
            "num_pixels_total", "num_valid_before_filter", "num_invalid_before_filter",
            "num_valid_after_filter", "num_rejected_by_filter",
        )
    })
    (frame_dir / "iteration.csv").write_text(
        "image_id,scale_level,iteration,num_pixels,valid_ratio,changed_ratio,phase,pass_index\n"
        "7,0,-1,4,1,1,initialization,0\n"
        "7,0,0,4,0.75,0.25,iteration,1\n",
        encoding="utf-8",
    )
    write_process_census(frame_dir)
    (frame_dir / "view_support.csv").write_text(
        "supporting_view_count,pixels\n0,2\n1,2\n",
        encoding="utf-8",
    )
    dmap = frame_dir.parent / "depth0007.dmap"
    dmap.touch()
    return summary, dmap


def make_exact_trace_fixture(root: Path, *, num_views: int = 2) -> tuple[Path, Path]:
    frame = root / "depthmaps" / "frame"
    summary, _dmap = make_summary_fixture(frame)
    summary["candidate_accounting_mode"] = (
        "exact_production_hot_kernel_counters_and_targeted_pixels"
    )
    summary["confidence_gap_mode"] = (
        "exact_process_pixel_winner_runner_up_at_targeted_pixels"
    )
    summary["unavailable_signals"] = []
    summary["resource_plan"].update({
        "trace_requested": True,
        "trace_available": True,
        "exact_trace_requested": True,
        "exact_trace_compatible": True,
        "exact_trace_available": True,
        "exact_trace_unavailable_reason": "",
        "exact_trace_record_layout": "selected_pixels_compact",
        "exact_record_pixel_stride": 1,
        "exact_candidate_record_count": 2,
        "exact_view_record_count": 2 * num_views,
        "num_trace_pixels": 1,
    })
    write_json(frame / "summary.json", summary)

    candidate = {
        key: 0 for key in validator.EXACT_TRACE_REQUIRED_CANDIDATE_FIELDS
    }
    candidate.update({
        "available": True,
        "candidate_tested_mask": 1,
        "candidate_finite_mask": 1,
        "candidate_production_valid_mask": 1,
        "candidate_accepted_mask": 1,
        "tested_count": 1,
        "finite_count": 1,
        "production_valid_count": 1,
        "accepted_count": 1,
        "low_texture_update_hysteresis": {"available": False},
    })
    rows = []
    for logical_state in range(2):
        views = []
        for source_view in range(num_views):
            view = {
                key: 0 for key in validator.EXACT_TRACE_REQUIRED_VIEW_FIELDS
            }
            view.update({
                "available": True,
                "source_view_index": source_view,
                "source_image_id": source_view + 10,
                "source_image_name": f"images/{source_view + 10:04d}.jpg",
            })
            views.append(view)
        component_count = min(
            num_views, validator.LEGACY_TRACE_COMPONENT_VIEW_LIMIT
        )
        rows.append({
            "schema_name": "openmvs.dmap.targeted_trace",
            "schema_version": 2,
            "image_id": 7,
            "scale_number": 0,
            "logical_state_index": logical_state,
            "trace_index": 0,
            "x": 1,
            "y": 1,
            "num_views": num_views,
            "process_specialization": "Process<true>",
            "measurement_quality": "exact",
            "measurement_basis": "production_hot_kernel_targeted_trace",
            "exact_hot_kernel_record": True,
            "view_component_count": component_count,
            "view_component_unavailable_reason": (
                "legacy trace aggregate stores at most four component views; "
                "use exact_views"
                if component_count < num_views else ""
            ),
            "view_costs": [0.0] * component_count,
            "view_photometric_costs": [0.0] * component_count,
            "view_geometric_costs": [0.0] * component_count,
            "exact_views_available": True,
            "exact_candidate": candidate,
            "exact_views": views,
            "exact_observability": {
                "available": True,
                "record_layout": "selected_pixels_compact",
                "candidate_record_available": True,
                "view_records_available": True,
                "view_record_count": num_views,
            },
        })
    trace_path = root / "instrumentation" / "traces.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return frame, trace_path


def make_prefilter_fixture(frame_dir: Path) -> tuple[dict, Path]:
    frame_dir.mkdir(parents=True)
    depth_path = frame_dir / "maps" / "depth_final_before_filter.pfm"
    write_pfm(depth_path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    completion = {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "path": "prefilter_capture_complete.json",
        "maps_complete": True,
        "prefilter_complete": True,
        "eligible": True,
    }
    summary = {
        "schema_name": "openmvs.dmap.frame_summary",
        "schema_version": 4,
        "image_id": 7,
        "image_name": "images/0007.jpg",
        "safe_image_name": "0007",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "width": 2,
        "height": 2,
        "scale_level": 0,
        "cuda_patchmatch_parameters": {"estimation_iterations": 1},
        "num_pixels_total": 4,
        "num_valid_before_filter": 4,
        "num_invalid_before_filter": 0,
        "num_valid_after_filter": 3,
        "num_rejected_by_filter": 1,
        "valid_ratio_after_filter": 0.75,
        "candidate_accounting_mode": "unavailable_post_pass_snapshot",
        "confidence_gap_mode": "post_pass_current_plus_eight_neighbors",
        "unavailable_signals": sorted(validator.SUMMARY_UNAVAILABLE_EXACT_SIGNALS),
        "observer_sidecars": {
            "complete": True,
            "write_error_count": 0,
            "write_errors": [],
        },
        "completion_marker": completion,
        "resource_plan": {
            "decision": "prefilter",
            "summary_available": True,
            "prefilter_requested": True,
            "prefilter_available": True,
            "maps_requested": False,
            "maps_available": False,
            "exact_requested": False,
            "exact_available": False,
            "exact_unavailable_reason": "exact capture was not requested",
        },
    }
    manifest = {
        "schema_name": "openmvs.dmap.prefilter_manifest",
        "schema_version": 1,
        "complete": True,
        "process_specialization": "Process<false>",
        "image_id": 7,
        "image_name": "images/0007.jpg",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "width": 2,
        "height": 2,
        "maps": [{
            "signal": "depth_final_before_filter",
            "path": "maps/depth_final_before_filter.pfm",
            "dtype": "float32",
            "measurement_quality": "exact",
            "measurement_basis": "production_pre_filter_snapshot",
            "bytes": depth_path.stat().st_size,
        }],
    }
    marker = {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "capture_kind": "prefilter",
        "eligible": True,
        "prefilter_complete": True,
        "maps_complete": True,
        "image_id": 7,
        "image_name": "images/0007.jpg",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "observer_sidecars_complete": True,
        "manifest": {"path": "prefilter_manifest.json", "schema_version": 1},
        "summary": {"path": "summary.json", "schema_version": 4},
    }
    summary_path = frame_dir / "summary.json"
    manifest_path = frame_dir / "prefilter_manifest.json"
    write_json(summary_path, summary)
    write_json(manifest_path, manifest)
    marker["manifest"]["bytes"] = manifest_path.stat().st_size
    marker["summary"]["bytes"] = summary_path.stat().st_size
    write_json(frame_dir / "prefilter_capture_complete.json", marker)
    write_process_census(frame_dir, terminal_valid=4)
    return summary, depth_path


class SummaryOnlyValidationTests(unittest.TestCase):
    def test_claimed_reference_patch_layout_is_validated_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _dmap = make_summary_fixture(frame)
            claim_reference_patch_layout(frame, summary)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertTrue(result["valid"], result)
            check = next(
                row for row in result["checks"]
                if row["name"] == "reference_patch_layout_contract"
            )
            self.assertTrue(check["passed"], check)
            self.assertEqual(check["detail"]["status"], "valid")

        cases = {
            "missing run layout": {"run_layout": {}},
            "summary mismatch": {"summary_layout": {
                **reference_patch_layout(), "sample_count": 24,
            }},
            "non-boolean capability": {"capability": "true"},
        }
        for label, options in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                frame = Path(directory) / "frame"
                summary, _dmap = make_summary_fixture(frame)
                claim_reference_patch_layout(frame, summary, **options)

                result = validator.validate(validator.Arguments(frame_dir=frame))

                self.assertFalse(result["valid"], result)
                check = next(
                    row for row in result["checks"]
                    if row["name"] == "reference_patch_layout_contract"
                )
                self.assertFalse(check["passed"], check)

    def test_valid_exact_targeted_trace_checks_compact_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame, trace_path = make_exact_trace_fixture(Path(directory))

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertTrue(result["valid"], result)
            self.assertTrue(result["exact_targeted_trace_available"])
            self.assertEqual(result["exact_targeted_trace_unavailable_reason"], "")
            self.assertEqual(result["exact_validation"]["scope"], "selected_pixels_compact")
            self.assertEqual(result["exact_validation"]["observed_candidate_records"], 2)
            self.assertEqual(result["exact_validation"]["observed_view_records"], 4)

            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            rows[0]["exact_views"][1]["source_view_index"] = 0
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            invalid = validator.validate(validator.Arguments(frame_dir=frame))
            self.assertFalse(invalid["valid"])
            failed = {row["name"] for row in invalid["checks"] if not row["passed"]}
            self.assertIn("summary_exact_targeted_trace", failed)

    def test_exact_trace_separates_legacy_components_from_all_runtime_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame, trace_path = make_exact_trace_fixture(
                Path(directory), num_views=15
            )

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertTrue(result["valid"], result)
            self.assertEqual(result["exact_validation"]["observed_view_records"], 30)

            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(rows[0]["view_component_count"], 4)
            self.assertEqual(len(rows[0]["exact_views"]), 15)

            rows[0]["view_component_unavailable_reason"] = ""
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            invalid = validator.validate(validator.Arguments(frame_dir=frame))
            self.assertFalse(invalid["valid"])
            self.assertIn(
                "record 0: legacy view-component contract is inconsistent",
                invalid["exact_validation"]["errors"],
            )

    def test_exact_trace_rejects_legacy_component_array_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame, trace_path = make_exact_trace_fixture(
                Path(directory), num_views=15
            )
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            rows[0]["view_costs"].append(0.0)
            trace_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            invalid = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(invalid["valid"])
            self.assertIn(
                "record 0: legacy view-component contract is inconsistent",
                invalid["exact_validation"]["errors"],
            )

    def test_exact_trace_requires_declared_record_availability(self) -> None:
        for field in ("candidate_record_available", "view_records_available"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                frame, trace_path = make_exact_trace_fixture(Path(directory))
                rows = [
                    json.loads(line) for line in trace_path.read_text().splitlines()
                ]
                rows[0]["exact_observability"][field] = False
                trace_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

                invalid = validator.validate(validator.Arguments(frame_dir=frame))

                self.assertFalse(invalid["valid"])
                self.assertIn(
                    "record 0: exact view count/availability contract is inconsistent",
                    invalid["exact_validation"]["errors"],
                )

    def test_valid_summary_capture_checks_retained_dmap_without_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            _summary, dmap = make_summary_fixture(frame)
            decoded = {
                "depth_map": np.ones((2, 2), dtype=np.float32),
                "normal_map": np.ones((2, 2, 3), dtype=np.float32),
                "confidence_map": np.ones((2, 2), dtype=np.float32),
            }

            with mock.patch.object(validator, "load_dmap", return_value=decoded):
                result = validator.validate(validator.Arguments(
                    frame_dir=frame, instrumented_dmap=dmap,
                ))

            self.assertTrue(result["valid"], result)
            self.assertEqual(result["capture_kind"], "summary_only")
            self.assertFalse(result["maps_available"])
            self.assertFalse(result["exact_maps_available"])
            self.assertEqual(
                result["maps_unavailable_reason"],
                "summary_profile_maps_not_requested",
            )
            self.assertTrue(result["instrumented_dmap_checked"])
            self.assertEqual(result["manifest_map_count"], 0)

    def test_valid_apd_summary_accepts_exact_hot_kernel_aggregates_without_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _dmap = make_summary_fixture(frame)
            summary["candidate_accounting_mode"] = (
                "exact_apd_working_objective_aggregate"
            )
            summary["confidence_gap_mode"] = (
                "exact_apd_working_winner_runner_up_aggregate"
            )
            summary["unavailable_signals"] = [
                "apd_exact_full_frame_pixel_mechanics"
            ]
            summary["resource_plan"].update({
                "apd_requested": True,
                "apd_summary_available": True,
                "apd_maps_available": False,
            })
            summary["apd_observability"] = {"enabled": True}
            write_json(frame / "summary.json", summary)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertTrue(result["valid"], result)
            check = next(
                row for row in result["checks"]
                if row["name"] == "summary_exact_maps_unavailable"
            )
            self.assertTrue(check["detail"]["apd_aggregate_available"])

    def test_summary_requires_atomic_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_summary_fixture(frame)
            marker = json.loads((frame / "summary_complete.json").read_text(encoding="utf-8"))
            marker["summary_complete"] = False
            write_json(frame / "summary_complete.json", marker)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("summary_completion_marker", failed)

    def test_summary_rejects_observer_sidecar_write_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _dmap = make_summary_fixture(frame)
            summary["observer_sidecars"] = {
                "complete": False,
                "write_error_count": 1,
                "write_errors": [{"artifact": "timings.csv", "pyramid_level": 1}],
            }
            write_json(frame / "summary.json", summary)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("summary_observer_sidecars_complete", failed)

    def test_summary_rejects_invalid_process_census_identities(self) -> None:
        mutations = (
            {"processed": 0, "valid_depth": 0, "invalid_depth": 0,
             "component_samples": 0, "selected_views_0": 0, "selected_views_2": 0},
            {"invalid_depth": 0},
            {"selected_views_2": 2},
            {"component_samples": 2},
            {"candidate_tested_init": 1},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                frame = Path(directory) / "frame"
                make_summary_fixture(frame)
                rewrite_census_row(frame, 0, **mutation)

                result = validator.validate(validator.Arguments(frame_dir=frame))

                self.assertFalse(result["valid"])
                failed = {row["name"] for row in result["checks"] if not row["passed"]}
                self.assertIn("summary_process_census", failed)

    def test_summary_rejects_missing_configured_logical_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_summary_fixture(frame)
            counters = frame.parent / "instrumentation" / "counters.csv"
            lines = counters.read_text(encoding="utf-8").splitlines()
            counters.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            iteration = frame / "iteration.csv"
            lines = iteration.read_text(encoding="utf-8").splitlines()
            iteration.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("summary_logical_iterations", failed)
            self.assertIn("summary_process_census", failed)

    def test_present_map_manifest_keeps_strict_map_validation_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_summary_fixture(frame)
            write_json(frame / "map_manifest.json", {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 4,
                "maps": [],
            })

            with mock.patch.object(validator, "validate_summary_only") as summary_validate:
                result = validator.validate(validator.Arguments(frame_dir=frame))

            summary_validate.assert_not_called()
            self.assertFalse(result["valid"])
            self.assertEqual(result["capture_kind"], "maps")


class PrefilterValidationTests(unittest.TestCase):
    def test_prefilter_enforces_claimed_reference_patch_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _depth_path = make_prefilter_fixture(frame)
            claim_reference_patch_layout(frame, summary)
            valid = validator.validate(validator.Arguments(frame_dir=frame))
            self.assertTrue(valid["valid"], valid)

            metadata_path = frame.parent / "run_metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["cuda_patchmatch_parameters"].pop(
                "reference_patch_layout"
            )
            write_json(metadata_path, metadata)
            invalid = validator.validate(validator.Arguments(frame_dir=frame))
            check = next(
                row for row in invalid["checks"]
                if row["name"] == "reference_patch_layout_contract"
            )
            self.assertFalse(check["passed"], check)

    def test_valid_prefilter_capture_is_bounded_and_process_false(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_prefilter_fixture(frame)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertTrue(result["valid"], result)
            self.assertTrue(result["prefilter_capture_available"])
            self.assertFalse(result["exact_maps_available"])
            self.assertEqual(result["process_specialization"], "Process<false>")

    def test_prefilter_rejects_unindexed_extra_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_prefilter_fixture(frame)
            write_pfm(frame / "maps" / "unexpected.pfm", np.ones((2, 2)))

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("prefilter_indexes_all_maps", failed)

    def test_prefilter_rejects_symlinked_depth_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame"
            _summary, depth_path = make_prefilter_fixture(frame)
            target = root / "external.pfm"
            target.write_bytes(depth_path.read_bytes())
            depth_path.unlink()
            depth_path.symlink_to(target)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("prefilter_map_file", failed)

    def test_prefilter_requires_resource_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _depth_path = make_prefilter_fixture(frame)
            summary["resource_plan"]["prefilter_available"] = False
            write_json(frame / "summary.json", summary)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            self.assertFalse(result["prefilter_capture_available"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("prefilter_resource_plan", failed)

    def test_prefilter_rejects_zero_process_census(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            make_prefilter_fixture(frame)
            rewrite_census_row(
                frame,
                0,
                terminal_valid=4,
                processed=0,
                valid_depth=0,
                invalid_depth=0,
                component_samples=0,
                selected_views_0=0,
                selected_views_2=0,
            )

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("prefilter_process_census", failed)

    def test_prefilter_requires_atomic_completion_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame"
            summary, _depth_path = make_prefilter_fixture(frame)
            summary["completion_marker"]["eligible"] = False
            write_json(frame / "summary.json", summary)

            result = validator.validate(validator.Arguments(frame_dir=frame))

            self.assertFalse(result["valid"])
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            self.assertIn("prefilter_summary_completion_reference", failed)


if __name__ == "__main__":
    unittest.main()
