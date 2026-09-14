#!/usr/bin/env python3
"""Focused schema-v5 DVP observability validator tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_dmap_instrumentation as validator


WIDTH = 2
HEIGHT = 2


def rgba_masks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    return np.stack(
        [
            values & 255,
            (values >> 8) & 255,
            (values >> 16) & 255,
            (values >> 24) & 255,
        ],
        axis=-1,
    ).astype(np.uint8)


def build_fixture(directory: str) -> tuple[
    Path,
    dict,
    dict,
    list[dict],
    dict[tuple[str, int], tuple[dict, Path, np.ndarray]],
]:
    frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
    instrumentation_dir = frame_dir.parent.parent / "instrumentation"
    instrumentation_dir.mkdir(parents=True)

    generated = np.array([[0, 1], [1, 2]], dtype=np.uint8)
    proposal_mask = np.array([[0, 1], [1, 3]], dtype=np.uint8)
    accepted = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    final_source = np.array([[0, 4], [12, 12]], dtype=np.uint8)
    final_winner = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    depth_retained = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    left_valid = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    right_valid = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    left_endpoint_count = np.array([[2, 2], [3, 3]], dtype=np.uint8)
    right_endpoint_count = np.array([[2, 3], [2, 3]], dtype=np.uint8)
    incumbent_depth = np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32)
    final_depth = np.array([[10.0, 11.0], [9.0, 8.0]], dtype=np.float32)
    incumbent_cost = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)
    candidate_0 = np.array([[-1.0, 0.4], [0.3, 0.4]], dtype=np.float32)
    candidate_1 = np.array([[-1.0, -1.0], [-1.0, 0.2]], dtype=np.float32)
    proposal_0 = np.array([[-1.0, 11.0], [9.0, 9.0]], dtype=np.float32)
    proposal_1 = np.array([[-1.0, -1.0], [-1.0, 8.0]], dtype=np.float32)
    winner_cost = np.array([[0.5, 0.4], [0.3, 0.2]], dtype=np.float32)
    runner_cost = np.array([[-1.0, 0.5], [0.5, 0.4]], dtype=np.float32)
    gap = np.array([[-1.0, 0.1], [0.2, 0.2]], dtype=np.float32)
    displacement = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    winner_ordinal = np.array([[255, 0], [0, 1]], dtype=np.uint8)
    native_fallback = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    unavailable_reason = np.array([[6, 0], [0, 0]], dtype=np.uint8)
    left_min = np.array([[-1.0, -1.0], [8.5, 7.5]], dtype=np.float32)
    left_max = np.array([[-1.0, -1.0], [9.5, 8.5]], dtype=np.float32)
    right_min = np.array([[-1.0, 10.5], [-1.0, 8.5]], dtype=np.float32)
    right_max = np.array([[-1.0, 11.5], [-1.0, 9.5]], dtype=np.float32)
    signed_0 = np.array([[0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    signed_1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    selected_views = np.full((HEIGHT, WIDTH), 7, dtype=np.uint32)

    arrays: dict[str, np.ndarray] = {
        "dvp_incumbent_depth": incumbent_depth,
        "dvp_final_depth": final_depth,
        "dvp_left_interval_minimum": left_min,
        "dvp_left_interval_maximum": left_max,
        "dvp_right_interval_minimum": right_min,
        "dvp_right_interval_maximum": right_max,
        "dvp_incumbent_cost": incumbent_cost,
        "dvp_winner_cost": winner_cost,
        "dvp_runner_up_cost": runner_cost,
        "dvp_winner_runner_up_gap": gap,
        "dvp_depth_displacement": displacement,
        "dvp_selected_source_views": rgba_masks(selected_views),
        "dvp_direction_source_views": rgba_masks(selected_views),
        "dvp_left_outer_count": left_endpoint_count,
        "dvp_left_inner_count": left_endpoint_count.copy(),
        "dvp_right_inner_count": right_endpoint_count,
        "dvp_right_outer_count": right_endpoint_count.copy(),
        "dvp_family": np.full((HEIGHT, WIDTH), 4, dtype=np.uint8),
        "dvp_unavailable_reason": unavailable_reason,
        "dvp_generated_count": generated,
        "dvp_tested_count": generated.copy(),
        "dvp_finite_count": generated.copy(),
        "dvp_accepted_count": generated.copy(),
        "dvp_tested_proposal_mask": proposal_mask.copy(),
        "dvp_finite_proposal_mask": proposal_mask.copy(),
        "dvp_accepted_proposal_mask": proposal_mask.copy(),
        "dvp_winner_ordinal": winner_ordinal,
        "dvp_accepted": accepted,
        "dvp_final_winner": final_winner,
        "dvp_final_update_source": final_source,
        "dvp_final_depth_retained": depth_retained,
        "dvp_native_depth_fallback": native_fallback,
        "dvp_left_interval_valid": left_valid,
        "dvp_right_interval_valid": right_valid,
    }
    for proposal, (depth, cost, signed) in enumerate(
        ((proposal_0, candidate_0, signed_0), (proposal_1, candidate_1, signed_1))
    ):
        generated_bit = (proposal_mask & (1 << proposal)) != 0
        support = np.where(generated_bit, 3, 0).astype(np.uint8)
        support_views = np.where(generated_bit, 7, 0).astype(np.uint32)
        arrays.update({
            f"dvp_proposal_depth_{proposal}": depth,
            f"dvp_candidate_cost_{proposal}": cost,
            f"dvp_mean_reprojection_error_{proposal}": np.full(
                (HEIGHT, WIDTH), -1.0, dtype=np.float32
            ),
            f"dvp_max_reprojection_error_{proposal}": np.full(
                (HEIGHT, WIDTH), -1.0, dtype=np.float32
            ),
            f"dvp_mean_relative_depth_error_{proposal}": np.full(
                (HEIGHT, WIDTH), -1.0, dtype=np.float32
            ),
            f"dvp_max_relative_depth_error_{proposal}": np.full(
                (HEIGHT, WIDTH), -1.0, dtype=np.float32
            ),
            f"dvp_signed_offset_{proposal}": signed,
            f"dvp_source_view_{proposal}": np.full(
                (HEIGHT, WIDTH), 255, dtype=np.uint8
            ),
            f"dvp_support_{proposal}": support,
            f"dvp_support_views_{proposal}": rgba_masks(support_views),
            f"dvp_occluded_views_{proposal}": rgba_masks(
                np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
            ),
        })

    entries: list[dict] = []
    logical_maps: dict[tuple[str, int], tuple[dict, Path, np.ndarray]] = {}
    for signal in sorted(validator.DVP_REQUIRED_SIGNALS):
        dtype = (
            "float32" if signal in validator.DVP_REQUIRED_FLOAT_SIGNALS
            else "uint8x4" if signal in validator.DVP_REQUIRED_RGBA_SIGNALS
            else "uint8"
        )
        entry = {
            "signal": signal,
            "path": f"dvp_states/iteration01/{signal}.png",
            "dtype": dtype,
            "logical_iteration": 0,
            "role": "dvp_logical_iteration_update",
            "stage": "iteration",
            "dvp_schema_name": validator.DVP_SCHEMA_NAME,
            "dvp_schema_version": validator.DVP_SCHEMA_VERSION,
            "measurement_quality": "exact",
            "measurement_basis": "active_process_pixel_candidate_path",
        }
        entries.append(entry)
        logical_maps[(signal, 0)] = (entry, Path(entry["path"]), arrays[signal])

    final_source_counts = {
        name: int(np.count_nonzero(final_source == code))
        for name, code in validator.DVP_FINAL_SOURCE_CODES.items()
    }
    unavailable_names = (
        "none", "family_disabled", "geometry_unavailable",
        "invalid_reference_depth", "no_selected_source_view",
        "invalid_epipolar_direction", "insufficient_endpoint_support",
        "invalid_interval_order", "no_finite_global_candidate",
        "global_reprojection_gate", "global_support_gate",
        "global_occlusion_rejected",
    )
    summary_iteration = {
        "logical_iteration": 0,
        "attempted_pixels": 4,
        "proposal_available_pixels": 3,
        "native_depth_fallback_pixels": 1,
        "proposals": {
            "generated": 4,
            "tested": 4,
            "finite": 4,
            "accepted_events": 4,
            "final_winner_pixels": 2,
            "final_depth_retained_pixels": 3,
            "final_update_source_counts": final_source_counts,
            "support_mean": 3.0,
            "occluded_candidates": 0,
        },
        "intervals": {
            "left_valid_pixels": 2,
            "right_valid_pixels": 2,
            "endpoint_support_mean": {
                "left_outer": 2.5,
                "left_inner": 2.5,
                "right_inner": 2.5,
                "right_outer": 2.5,
            },
        },
        "views": {
            "selected_source_count_mean": 3.0,
            "direction_source_count_mean": 3.0,
        },
        "costs": {
            "samples": 4,
            "incumbent_mean": 0.5,
            "winner_mean": 0.35,
            "improvement_mean": 0.15,
            "winner_runner_up_gap_mean": 1.0 / 6.0,
        },
        "geometry": {
            "depth_displacement_mean": 1.0,
            "reprojection_error_mean_px": None,
            "relative_depth_error_mean": None,
        },
        "unavailable_reason_counts": {
            name: int(np.count_nonzero(unavailable_reason == code))
            for code, name in enumerate(unavailable_names)
        },
    }
    capture = {
        "schema_name": validator.DVP_SCHEMA_NAME,
        "schema_version": validator.DVP_SCHEMA_VERSION,
        "requested": True,
        "maps_available": True,
        "summary_available": True,
        "targeted_trace_available": True,
        "family": "dvp_eq11_interval_v1",
        "num_iterations": 1,
        "proposal_candidate_slots": [22, 23],
        "endpoint_samples_full_frame": False,
        "endpoint_samples_targeted_trace": True,
        "promotion_eligible": True,
        "author_code_exact_equivalence": False,
        "update_record_bytes": 148,
        "counter_record_bytes": 280,
        "aggregate_sum_precision": (
            "float64 device atomic; counts are uint32 device atomic"
        ),
    }
    observability = {
        "schema_name": validator.DVP_OBSERVABILITY_SCHEMA_NAME,
        "schema_version": validator.DVP_SCHEMA_VERSION,
        "requested": True,
        "enabled": True,
        "summary_available": True,
        "maps_available": True,
        "targeted_trace_available": True,
        "family": "dvp_eq11_interval_v1",
        "promotion_eligible": True,
        "author_code_exact_equivalence": False,
        "measurement_basis": (
            "exact active Process<true> candidate generation, scoring, "
            "acceptance, and retained outcome"
        ),
        "parameters": {"alpha": 1.0, "beta": 4.0, "mu": 3},
        "provenance": {
            "official_repository_commit": (
                "5b2b31c02941cfa28808b4e37f06a87e8dc0f2d9"
            ),
            "colleague_donor_commit": (
                "c6f0f2c0fe2e31ec68c54cac681cc3b1665d2f7b"
            ),
        },
        "update_record_bytes": 148,
        "counter_record_bytes": 280,
        "trace_record_bytes": 700,
        "aggregate_sum_precision": (
            "float64 device atomic; counts are uint32 device atomic"
        ),
        "iterations": [summary_iteration],
    }
    manifest = {"dvp_capture": capture}
    summary = {
        "image_id": 1,
        "scale_level": 0,
        "dvp_observability": observability,
    }

    csv_path = instrumentation_dir / "dvp_iteration.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_id", "scale_number", "logical_iteration",
            "attempted_pixels", "proposal_available_pixels",
            "native_depth_fallback_pixels", "proposals_generated",
            "proposals_tested", "proposals_finite", "proposals_accepted",
            "final_winner_pixels", "final_depth_retained_pixels",
        ])
        writer.writeheader()
        writer.writerow({
            "image_id": 1,
            "scale_number": 0,
            "logical_iteration": 0,
            "attempted_pixels": 4,
            "proposal_available_pixels": 3,
            "native_depth_fallback_pixels": 1,
            "proposals_generated": 4,
            "proposals_tested": 4,
            "proposals_finite": 4,
            "proposals_accepted": 4,
            "final_winner_pixels": 2,
            "final_depth_retained_pixels": 3,
        })

    endpoint_samples = {
        "left_outer": [10.0, 10.1, 10.2],
        "left_inner": [10.3, 10.4, 10.5],
        "right_inner": [10.6, 10.7, 10.8],
        "right_outer": [10.9, 11.0, 11.1],
    }
    endpoint_offsets = {
        "left_outer": -5.0,
        "left_inner": -1.0,
        "right_inner": 1.0,
        "right_outer": 5.0,
    }
    trace = {
        "schema_name": "openmvs.dmap.dvp_trace",
        "schema_version": 1,
        "measurement_quality": "exact",
        "measurement_basis": "active_process_pixel_candidate_path",
        "image_id": 1,
        "scale_number": 0,
        "logical_iteration": 0,
        "trace_index": 0,
        "x": 1,
        "y": 0,
        "family": "dvp_eq11_interval_v1",
        "endpoint_samples": endpoint_samples,
        "endpoint_sample_views": {name: 7 for name in endpoint_samples},
        "endpoint_records": {
            name: [
                {
                    "view": view,
                    "offset_px": endpoint_offsets[name],
                    "back_projected_depth": depth,
                }
                for view, depth in enumerate(samples)
            ]
            for name, samples in endpoint_samples.items()
        },
        "proposals": [{
            "ordinal": 0,
            "tested": True,
            "finite": True,
            "sequentially_accepted": True,
            "depth": 11.0,
            "candidate_cost": 0.4,
        }],
        "decision": {
            "generated_count": 1,
            "tested_count": 1,
            "finite_count": 1,
            "sequential_acceptance_count": 1,
            "accepted_proposal_mask": 1,
            "final_update_source": "refine_normal",
            "final_update_source_code": 4,
        },
    }
    (instrumentation_dir / "dvp_traces.jsonl").write_text(
        json.dumps(trace) + "\n", encoding="utf-8"
    )
    return frame_dir, manifest, summary, entries, logical_maps


class DVPValidatorTests(unittest.TestCase):
    def validate_fixture(
        self,
        frame_dir: Path,
        manifest: dict,
        summary: dict,
        entries: list[dict],
        logical_maps: dict[tuple[str, int], tuple[dict, Path, np.ndarray]],
    ) -> tuple[dict, list[dict]]:
        checks: list[dict] = []

        def check(name: str, passed: bool, detail: object) -> None:
            checks.append({"name": name, "passed": passed, "detail": detail})

        result = validator.validate_dvp_maps(
            frame_dir=frame_dir,
            manifest=manifest,
            summary=summary,
            entries=entries,
            logical_maps=logical_maps,
            width=WIDTH,
            height=HEIGHT,
            tolerance=0.0,
            check=check,
        )
        return result, checks

    def test_eq11_maps_counters_csv_and_trace_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            result, checks = self.validate_fixture(*fixture)
            self.assertTrue(result["available"])
            self.assertEqual(result["expected_map_count"], 56)
            self.assertEqual(result["trace_rows"], 1)
            self.assertTrue(all(row["passed"] for row in checks), checks)

    def test_per_proposal_acceptance_mask_is_not_an_aggregate_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir, manifest, summary, entries, logical_maps = build_fixture(directory)
            logical_maps[("dvp_accepted_proposal_mask", 0)][2][1, 1] = 1
            result, checks = self.validate_fixture(
                frame_dir, manifest, summary, entries, logical_maps
            )
            self.assertEqual(
                result["iterations"]["0"]["domain_errors"]["accepted_mask_count"],
                1,
            )
            self.assertFalse(next(
                row["passed"] for row in checks
                if row["name"] == "dvp_iteration_0_domains"
            ))

    def test_endpoint_view_identity_must_close_to_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            trace_path = fixture[0].parent.parent / "instrumentation" / "dvp_traces.jsonl"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["endpoint_records"]["right_inner"][1]["view"] = 3
            trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
            _result, checks = self.validate_fixture(*fixture)
            trace_check = next(
                row for row in checks if row["name"] == "dvp_targeted_trace"
            )
            self.assertFalse(trace_check["passed"])
            self.assertTrue(any(
                "endpoint group right_inner does not close" in error
                for error in trace_check["detail"]["errors"]
            ))


def build_visible_normal_fixture(directory: str) -> tuple[
    Path,
    dict,
    dict,
    list[dict],
    dict[tuple[str, int], tuple[dict, Path, np.ndarray]],
]:
    frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
    (frame_dir.parent.parent / "instrumentation").mkdir(parents=True)
    shape = (HEIGHT, WIDTH)
    arrays: dict[str, np.ndarray] = {}
    for signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_SIGNALS:
        if signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_FLOAT_SIGNALS:
            arrays[signal] = np.zeros(shape, dtype=np.float32)
        elif signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_WORD_SIGNALS:
            fill = 65535 if "selected_retry" in signal else 0
            arrays[signal] = np.full(shape, fill, dtype=np.uint16)
        elif signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_RGBA_SIGNALS:
            arrays[signal] = rgba_masks(np.ones(shape, dtype=np.uint32))
        else:
            arrays[signal] = np.zeros(shape, dtype=np.uint8)

    byte_values = {
        "support_count": 1,
        "direction_count": 2,
        "mode": 1,
        "current_valid": 1,
        "current_feasible": 1,
        "current_reason": 0,
        "current_rejected_direction": 255,
        "propagation_tested_mask": 1,
        "propagation_valid_mask": 1,
        "propagation_feasible_mask": 1,
        "propagation_rejected_mask": 0,
        "propagation_native_best": 0,
        "propagation_constrained_best": 0,
        "propagation_selected": 0,
        "propagation_reason": 2,
        "propagation_fallback": 0,
        "propagation_applied_constraint": 0,
        "propagation_accepted": 0,
        "refinement_native_tested_mask": 3,
        "refinement_native_valid_mask": 3,
        "refinement_native_feasible_mask": 3,
        "refinement_retry_success_mask": 0,
        "refinement_exhaustion_fallback_mask": 0,
        "refinement_applied_retry_mask": 0,
        "refinement_accepted_mask": 0,
    }
    for name, value in byte_values.items():
        arrays[f"dvp_visible_normal_{name}"][:] = value

    entries: list[dict] = []
    logical_maps: dict[tuple[str, int], tuple[dict, Path, np.ndarray]] = {}
    for signal in sorted(validator.DVP_VISIBLE_NORMAL_REQUIRED_SIGNALS):
        dtype = (
            "float32"
            if signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_FLOAT_SIGNALS
            else "uint16"
            if signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_WORD_SIGNALS
            else "uint8x4"
            if signal in validator.DVP_VISIBLE_NORMAL_REQUIRED_RGBA_SIGNALS
            else "uint8"
        )
        entry = {
            "signal": signal,
            "path": f"dvp_visible_normal_states/iteration01/{signal}.png",
            "dtype": dtype,
            "logical_iteration": 0,
            "role": "dvp_visible_normal_logical_iteration_update",
            "stage": "iteration",
            "dvp_visible_normal_schema_name": validator.DVP_VISIBLE_NORMAL_SCHEMA_NAME,
            "dvp_visible_normal_schema_version": (
                validator.DVP_VISIBLE_NORMAL_SCHEMA_VERSION
            ),
            "measurement_quality": "exact",
            "measurement_basis": "active_process_pixel_visible_normal_decision",
        }
        entries.append(entry)
        logical_maps[(signal, 0)] = (entry, Path(entry["path"]), arrays[signal])

    capture = {
        "schema_name": validator.DVP_VISIBLE_NORMAL_SCHEMA_NAME,
        "schema_version": validator.DVP_VISIBLE_NORMAL_SCHEMA_VERSION,
        "mode": "shadow",
        "requested": True,
        "stage_active": True,
        "summary_available": True,
        "maps_available": True,
        "targeted_trace_available": False,
        "num_iterations": 1,
        "counter_record_bytes": 160,
        "update_record_bytes": 84,
        "trace_record_bytes": 344,
        "main_rng_schedule": "native_unchanged",
        "retry_rng": "bounded_local_copy",
        "propagation_fallback": "native_best",
        "refinement_fallback": "native_proposal",
        "actual_normal_vectors_full_frame": False,
        "actual_normal_vectors_targeted_trace": True,
    }
    observability = {
        "schema_name": validator.DVP_VISIBLE_NORMAL_OBSERVABILITY_SCHEMA_NAME,
        "schema_version": validator.DVP_VISIBLE_NORMAL_SCHEMA_VERSION,
        "enabled": True,
        "mode": "shadow",
        "requested": True,
        "stage_active": True,
        "summary_available": True,
        "maps_available": True,
        "targeted_trace_available": False,
        "counter_record_bytes": 160,
        "update_record_bytes": 84,
        "trace_record_bytes": 344,
        "measurement_quality": "exact",
        "measurement_basis": (
            "same_stream_Process_true_logical_iteration_proposal_decisions"
        ),
        "evaluation_reason_enum": {
            str(code): name
            for code, name in enumerate(
                validator.DVP_VISIBLE_NORMAL_EVALUATION_REASON_NAMES
            )
        },
        "proposal_reason_enum": {
            str(code): name
            for code, name in enumerate(
                validator.DVP_VISIBLE_NORMAL_PROPOSAL_REASON_NAMES
            )
        },
        "propagation_reason_enum": {
            str(code): name
            for code, name in enumerate(
                validator.DVP_VISIBLE_NORMAL_PROPAGATION_REASON_NAMES
            )
        },
        "iterations": [{
            "logical_iteration": 0,
            "pixels": 4,
            "selected_support_views": 4,
            "current": {"valid": 4, "feasible": 4, "rejected": 0, "invalid": 0},
            "propagation": {"tested": 4, "valid": 4, "feasible": 4, "rejected": 0},
            "refinement": {
                "native_tested": 8,
                "native_valid": 8,
                "native_feasible": 8,
                "native_rejected": 0,
                "retries_tested": 0,
                "retry_success": 0,
                "exhaustion": 0,
                "fallback": 0,
                "applied_retry": 0,
                "accepted": 0,
            },
        }],
    }
    return (
        frame_dir,
        {"dvp_visible_normal_capture": capture},
        {"dvp_visible_normal_observability": observability},
        entries,
        logical_maps,
    )


class DVPVisibleNormalValidatorTests(unittest.TestCase):
    def validate_fixture(self, fixture: tuple) -> tuple[dict, list[dict]]:
        checks: list[dict] = []

        def check(name: str, passed: bool, detail: object) -> None:
            checks.append({"name": name, "passed": passed, "detail": detail})

        result = validator.validate_dvp_visible_normal_maps(
            frame_dir=fixture[0],
            manifest=fixture[1],
            summary=fixture[2],
            entries=fixture[3],
            logical_maps=fixture[4],
            width=WIDTH,
            height=HEIGHT,
            check=check,
        )
        return result, checks

    def test_visible_normal_maps_close_to_aggregate_counters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result, checks = self.validate_fixture(
                build_visible_normal_fixture(directory)
            )
            self.assertTrue(result["available"])
            self.assertEqual(result["expected_map_count"], 47)
            self.assertTrue(all(row["passed"] for row in checks), checks)

    def test_visible_normal_feasible_mask_must_be_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_visible_normal_fixture(directory)
            fixture[4][("dvp_visible_normal_propagation_valid_mask", 0)][2][:] = 0
            _result, checks = self.validate_fixture(fixture)
            domain = next(
                row for row in checks
                if row["name"] == "dvp_visible_normal_iteration_0_domains"
            )
            self.assertFalse(domain["passed"])
            self.assertIn(
                "propagation feasible mask is not a valid subset",
                domain["detail"]["errors"],
            )


class APDCandidateGapSemanticsTests(unittest.TestCase):
    def validate_gap(
        self,
        *,
        working: list[float],
        runner: list[float],
        gap: list[float],
        constraints: list[bool] | None,
    ) -> tuple[dict, list[str]]:
        constraint_mask = (
            None if constraints is None else np.asarray([constraints], dtype=bool)
        )
        return validator.validate_apd_candidate_gap_semantics(
            working=np.asarray([working], dtype=np.float32),
            runner=np.asarray([runner], dtype=np.float32),
            gap=np.asarray([gap], dtype=np.float32),
            tolerance=2.0e-6,
            constraint_mask=constraint_mask,
        )

    def test_unconstrained_gap_closes_to_selected_winner(self) -> None:
        result, errors = self.validate_gap(
            working=[0.2, 0.3],
            runner=[0.5, -1.0],
            gap=[0.3, -1.0],
            constraints=None,
        )
        self.assertEqual(errors, [])
        self.assertEqual(result["selected_winner_gap_unavailable_pixels"], 0)
        self.assertAlmostEqual(result["strict_selected_closure_max_abs"], 0.0)

    def test_constraint_covered_divergence_is_explicitly_unavailable(self) -> None:
        result, errors = self.validate_gap(
            working=[0.4],
            runner=[0.5],
            gap=[0.3],
            constraints=[True],
        )
        self.assertEqual(errors, [])
        self.assertEqual(result["selected_winner_gap_unavailable_pixels"], 1)
        self.assertEqual(result["uncovered_divergence_pixels"], 0)
        self.assertEqual(
            result["semantics"],
            "unconstrained_evaluated_candidate_set_best_vs_runner_up",
        )

    def test_divergence_without_constraint_evidence_fails(self) -> None:
        result, errors = self.validate_gap(
            working=[0.4],
            runner=[0.5],
            gap=[0.3],
            constraints=[False],
        )
        self.assertEqual(result["uncovered_divergence_pixels"], 1)
        self.assertTrue(any(
            "outside a visible-normal propagation selection constraint" in error
            for error in errors
        ))

    def test_refinement_retry_does_not_cover_selected_gap_divergence(self) -> None:
        propagation_applied_constraint = np.asarray([[0]], dtype=np.uint8)
        refinement_applied_retry = np.asarray([[1]], dtype=np.uint8)
        self.assertEqual(int(refinement_applied_retry[0, 0]), 1)

        result, errors = validator.validate_apd_candidate_gap_semantics(
            working=np.asarray([[0.4]], dtype=np.float32),
            runner=np.asarray([[0.5]], dtype=np.float32),
            gap=np.asarray([[0.3]], dtype=np.float32),
            tolerance=2.0e-6,
            constraint_mask=propagation_applied_constraint == 1,
        )

        self.assertEqual(result["constraint_pixels"], 0)
        self.assertEqual(result["uncovered_divergence_pixels"], 1)
        self.assertTrue(any(
            "outside a visible-normal propagation selection constraint" in error
            for error in errors
        ))

    def test_selected_cost_below_evaluated_candidate_best_fails(self) -> None:
        result, errors = self.validate_gap(
            working=[0.1],
            runner=[0.5],
            gap=[0.3],
            constraints=[True],
        )
        self.assertEqual(result["invalid_candidate_order_pixels"], 1)
        self.assertTrue(any("violates ordering" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
