#!/usr/bin/env python3
"""Synthetic schema-v4 report integration tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_dev
import dmap_report_model


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def write_pfm(path: Path, values: np.ndarray) -> None:
    data = np.asarray(values, dtype=np.float32)
    color = data.ndim == 3
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"PF\n" if color else b"Pf\n")
        handle.write(f"{data.shape[1]} {data.shape[0]}\n-1.0\n".encode("ascii"))
        np.flipud(data).astype("<f4").tofile(handle)


class Schema4ReportIntegrationTest(unittest.TestCase):

    def test_schema3_cuda_resource_plan_requires_pyramid_level(self) -> None:
        context = {
            "run": "candidate", "role": "variant", "repeat": 0,
            "scene_id": "scene-a", "frame": "0007_frame",
            "estimation_stage": "photometric", "geometric_iteration": None,
            "pyramid_level": 0, "image_id": 0,
        }
        plan = {
            "schema_name": "openmvs.dmap.resource_plan",
            "schema_version": 3,
            "image_id": 0,
            "pyramid_level": 2,
            "summary_available": True,
            "maps_available": False,
        }
        row = dmap_dev.cuda_resource_plan_row(
            context, plan, Path("resource_plans.jsonl"), required=True,
        )
        self.assertTrue(row["valid"])
        self.assertEqual(row["pyramid_level"], 2)

        invalid = dict(plan)
        invalid.pop("pyramid_level")
        invalid_row = dmap_dev.cuda_resource_plan_row(
            context, invalid, Path("resource_plans.jsonl"), required=True,
        )
        self.assertFalse(invalid_row["valid"])
        self.assertIn("invalid_pyramid_level", invalid_row["validation_errors_json"])

    def test_schema4_cuda_resource_plan_validates_cumulative_priority_reservation(self) -> None:
        context = {
            "run": "candidate", "role": "variant", "repeat": 0,
            "scene_id": "scene-a", "frame": "0007_frame",
            "estimation_stage": "photometric", "geometric_iteration": None,
            "pyramid_level": 2, "image_id": 7,
        }
        plan = {
            "schema_name": "openmvs.dmap.resource_plan",
            "schema_version": 4,
            "image_id": 7,
            "pyramid_level": 2,
            "summary_available": True,
            "maps_available": False,
            "limits_mib": {"device": 0, "host": 0, "frame_storage": 1},
            "effective_estimate_bytes": {
                "device": 1024,
                "host": 2048,
                "frame_storage": 6144,
                "current_pyramid_storage": 4096,
                "frame_storage_committed_before": 2048,
                "full_resolution_priority_reserve": 8192,
            },
            "storage_preflight": {
                "attempted": True,
                "succeeded": True,
                "available_bytes": 100000,
                "reserved_before_bytes": 0,
                "requested_bytes": 4096,
                "requested_plus_priority_reserve_bytes": 12288,
                "reservation_bytes": 4096,
                "frame_priority_reservation_bytes": 8192,
                "frame_priority_reservation_consumed": False,
            },
        }
        row = dmap_dev.cuda_resource_plan_row(
            context, plan, Path("resource_plans.jsonl"), required=True,
        )
        self.assertTrue(row["valid"], row["validation_errors_json"])
        self.assertEqual(row["full_resolution_priority_reserve_bytes"], 8192)

        invalid = json.loads(json.dumps(plan))
        invalid["storage_preflight"]["frame_priority_reservation_bytes"] = 0
        invalid["storage_preflight"]["requested_plus_priority_reserve_bytes"] = 4096
        invalid_row = dmap_dev.cuda_resource_plan_row(
            context, invalid, Path("resource_plans.jsonl"), required=True,
        )
        self.assertFalse(invalid_row["valid"])
        self.assertIn(
            "storage_preflight_priority_arithmetic_mismatch",
            invalid_row["validation_errors_json"],
        )
        self.assertIn(
            "coarse_level_priority_reservation_not_held",
            invalid_row["validation_errors_json"],
        )

    def test_nested_geometric_stage_discovery_preserves_stage_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instrumentation = root / "instrumentation"
            (instrumentation / "depthmaps").mkdir(parents=True)
            nested = instrumentation / "geometric_iterations" / "iteration02"
            (nested / "depthmaps").mkdir(parents=True)
            base = dmap_dev.RunScene("run", "baseline", 0, "scene", instrumentation, None, None)

            stages = dmap_dev.expand_instrumentation_stages(base)

            self.assertEqual(len(stages), 2)
            self.assertEqual(stages[0].estimation_stage, "photometric")
            self.assertEqual(stages[1].estimation_stage, "geometric_consistency")
            self.assertEqual(stages[1].geometric_iteration, 2)
            self.assertEqual(stages[1].instrumentation_dir, nested)

    def test_terminal_stage_selection_prefers_last_geometric_iteration(self) -> None:
        base = dmap_dev.RunScene(
            "run", "baseline", 0, "scene", Path("/photo"), None, None,
        )
        geometric_one = replace(
            base,
            instrumentation_dir=Path("/geom1"),
            estimation_stage="geometric_consistency",
            geometric_iteration=1,
        )
        geometric_three = replace(
            base,
            instrumentation_dir=Path("/geom3"),
            estimation_stage="geometric_consistency",
            geometric_iteration=3,
        )

        selected = dmap_dev.select_terminal_run_scenes([geometric_one, base, geometric_three])

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].instrumentation_dir, Path("/geom3"))

    def test_terminal_frame_selection_preserves_photometric_only_runs(self) -> None:
        frames = pd.DataFrame([
            {"run": "base", "repeat": 0, "scene_id": "scene", "image_id": 7,
             "estimation_stage": "photometric", "geometric_iteration": None, "value": 1},
            {"run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 7,
             "estimation_stage": "photometric", "geometric_iteration": None, "value": 2},
            {"run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 7,
             "estimation_stage": "geometric_consistency", "geometric_iteration": 1, "value": 3},
            {"run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 7,
             "estimation_stage": "geometric_consistency", "geometric_iteration": 4, "value": 4},
        ])

        selected = dmap_dev.select_terminal_frames(frames).sort_values("run")

        self.assertEqual(selected["value"].tolist(), [1, 4])

    def make_capture(self, root: Path) -> tuple[dmap_dev.RunScene, Path]:
        instrumentation = root / "instrumentation"
        frame = instrumentation / "depthmaps" / "0007_frame"
        state = "logical_states/state00_initialization"
        entries = [
            {
                "signal": "cost_total_production_exact", "path": f"{state}/cost_total_production_exact.pfm",
                "dtype": "float32", "role": "logical_state", "logical_iteration": -1,
                "stage": "initialization", "stage_index": 0, "measurement_quality": "exact",
                "measurement_basis": "production_hot_kernel_contribution_basis",
            },
            {
                "signal": "gap_winner_runner_up_exact", "path": f"{state}/gap_winner_runner_up_exact.pfm",
                "dtype": "float32", "role": "logical_state", "logical_iteration": -1,
                "stage": "initialization", "measurement_quality": "exact", "unavailable_value": -1.0,
                "gap_scope": "one complete logical update invocation per pixel",
            },
            {
                "signal": "candidate_winner_cost_exact", "path": f"{state}/candidate_winner_cost_exact.pfm",
                "dtype": "float32", "role": "logical_event", "logical_iteration": -1,
                "stage": "initialization", "measurement_quality": "exact",
            },
        ]
        for view, image_id in ((0, 20), (1, 21)):
            entries.append({
                "signal": "view_cost_components_exact",
                "path": f"{state}/view_cost_components_exact_view{view:02d}_id{image_id:04d}.pfm",
                "dtype": "float32x3", "role": "logical_view_state", "logical_iteration": -1,
                "stage": "initialization", "measurement_quality": "exact",
                "measurement_basis": "production_hot_kernel_view_record",
                "source_view_index": view, "source_image_id": image_id,
                "source_image_name": f"images/{image_id:04d}.jpg",
                "contribution_basis": "deterministic_initialization_top_k",
                "channels_memory_order": {"0": "photo_after_prior", "1": "geometric", "2": "total"},
            })
        write_json(frame / "summary.json", {
            "schema_version": 4, "image_id": 7, "image_name": "images/0007.jpg",
            "safe_image_name": "frame", "width": 3, "height": 2, "scale_level": 0,
        })
        write_json(frame / "map_manifest.json", {
            "schema_name": "openmvs.dmap.map_manifest", "schema_version": 4,
            "pyramid_level": 0,
            "num_iterations": 0, "num_logical_states": 1, "map_granularity": "logical_iteration",
            "exact_capture": {"requested": True, "available": True, "num_views": 2},
            "maps": entries,
        })
        scalar = np.asarray([[0.4, 0.3, 0.2], [0.5, 0.6, 0.7]], dtype=np.float32)
        for entry in entries:
            path = frame / entry["path"]
            if entry["dtype"] == "float32x3":
                write_pfm(path, np.dstack([scalar, scalar * 0.1, scalar * 1.1]))
            else:
                write_pfm(path, scalar)
        (frame / "exact_iteration.csv").write_text(
            "logical_iteration,pyramid_level,stage,pixels,tested_candidates,finite_candidates,accepted_candidates,gap_available_pixels,gap_mean,gap_p50,gap_p90,view_churn_pixels,low_texture_gate_eligible,low_texture_propagation_accepted,low_texture_propagation_rejected,low_texture_refinement_accepted,low_texture_refinement_rejected,low_texture_required_gain_sum,low_texture_best_proposed_gain_sum,source_0,source_3\n"
            "-1,0,initialization,6,78,72,6,6,0.2,0.1,0.4,2,,,,,,,,4,2\n"
            "0,0,iteration,6,78,72,3,6,0.2,0.1,0.4,2,5,2,1,3,2,0.0025,0.003,4,2\n", encoding="utf-8",
        )
        (frame / "exact_view_summary.csv").write_text(
            "logical_iteration,pyramid_level,stage,source_view_index,source_image_id,source_image_name,pixels,selected_pixels,finite_cost_pixels,probability_available_pixels,weight_mean,probability_mean,weighted_contribution_mean,photometric_cost_mean,geometric_cost_mean,total_cost_mean,decision_4\n"
            "-1,0,initialization,0,20,images/0020.jpg,6,5,6,0,1,0,0.2,0.3,0.0,0.3,5\n", encoding="utf-8",
        )
        write_json(frame / "exact_observability.json", {
            "schema_name": "openmvs.dmap.exact_observability", "schema_version": 3,
            "pyramid_level": 0,
            "num_logical_states": 1, "num_views": 2,
            "states": [{"logical_iteration": -1, "candidate_tested": 78}],
        })
        write_json(frame / "cpu_view_candidates.json", {
            "schema_name": "openmvs.dmap.cpu_view_candidates", "candidate_source": "computed_sparse_visibility",
            "ranking_succeeded": True, "filter_succeeded": True,
            "candidates": [{
                "candidate_image_id": 20, "ranking_score": 2.5, "raw_rank_zero_based": 0,
                "initial_decision": "scored", "filter_decision": "retained_threshold_pass",
                "final_rank_zero_based": 0, "accepted_after_filter": True,
                "score_components": {"available": True, "angle_weight_sum": 1.2},
            }],
        })
        write_json(frame / "cpu_view_estimation_selection.json", {
            "schema_name": "openmvs.dmap.cpu_view_estimation_selection", "schema_version": 2,
            "selection_succeeded": True,
            "selection_mode": "ranked_prefix", "ordered_cutoff": {"reason": "requested_neighbor_limit"},
            "admission_policy": {
                "name": "ranked_prefix_without_score_cutoff",
                "policy_origin": "fixture_ranked_prefix_policy",
                "score_threshold_applied": False,
                "configured_score_threshold_status": "reported_not_applied_to_patchmatch",
            },
            "parameters": {
                "view_min_score_absolute": 3.0, "view_min_score_ratio": 0.03,
                "effective_min_score": None, "configured_effective_min_score": 3.0,
            },
            "candidates": [{
                "candidate_image_id": 20, "filtered_rank_zero_based": 0, "ranking_score": 2.5,
                "score_ratio_to_best": 1.0, "selected": True, "selected_rank_zero_based": 0,
                "would_pass_configured_score_threshold": False,
                "decision": "selected",
            }],
        })
        postprocess_maps = [
            {
                "signal": "00_remove_speckles_depth_delta",
                "path": "postprocess_filters/00_remove_speckles_depth_delta.pfm",
                "dtype": "float32", "quality": "exact", "algorithm_stage": "remove_speckles",
                "semantics": "signed depth_after-depth_before",
            },
            {
                "signal": "00_remove_speckles_validity_transition",
                "path": "postprocess_filters/00_remove_speckles_validity_transition.png",
                "dtype": "uint8", "quality": "exact", "algorithm_stage": "remove_speckles",
                "semantics": "0 unchanged invalid, 1 unchanged valid, 2 removed, 3 added",
            },
        ]
        write_json(frame / "postprocess_filters.json", {
            "schema_name": "openmvs.dmap.postprocess_filters", "schema_version": 1,
            "algorithm_stage": "depth_map_optional_postprocess", "estimation_stage": "photometric",
            "geometric_iteration": None,
            "resource_plan": {"path": "filter_resource_plan.json", "schema_name": "openmvs.dmap.filter_resource_plan", "schema_version": 1},
            "maps_requested": True, "maps_enabled": True, "maps": postprocess_maps,
            "write_errors": [], "complete": True,
            "stages": [
                {
                    "stage_index": 0, "name": "remove_speckles", "enabled": True,
                    "executed": True, "success": True,
                    "measurement_basis": "exact sequential before/after state", "parameters": {},
                    "metrics": {
                        "total_pixels": 6, "input_valid_depth_pixels": 6,
                        "output_valid_depth_pixels": 5, "removed_pixels": 1, "added_pixels": 0,
                        "depth_changed_pixels": 1, "depth_abs_delta_mean_all_pixels": 0.1,
                    },
                },
                {
                    "stage_index": 1, "name": "fill_gaps", "enabled": False,
                    "executed": False, "success": None,
                    "measurement_basis": "exact identity because stage was disabled", "parameters": {},
                    "metrics": {
                        "total_pixels": 6, "input_valid_depth_pixels": 5,
                        "output_valid_depth_pixels": 5, "removed_pixels": 0, "added_pixels": 0,
                        "depth_changed_pixels": 0, "depth_abs_delta_mean_all_pixels": 0.0,
                    },
                },
            ],
        })
        write_pfm(frame / postprocess_maps[0]["path"], scalar * 0.01)
        Image.fromarray(np.asarray([[0, 1, 2], [3, 1, 0]], dtype=np.uint8)).save(
            frame / postprocess_maps[1]["path"]
        )

        confidence_maps = [
            {
                "signal": "confidence_final_delta",
                "path": "confidence_adjustment/confidence_final_delta.pfm",
                "dtype": "float32", "quality": "exact", "algorithm_stage": "confidence_adjustment",
                "semantics": "signed adjusted minus input confidence",
            },
            {
                "signal": "confidence_final_transition",
                "path": "confidence_adjustment/confidence_final_transition.png",
                "dtype": "uint8", "quality": "exact", "algorithm_stage": "confidence_adjustment",
                "semantics": "confidence positivity transition",
            },
        ]
        confidence_metrics = {
            "total_pixels": 6, "valid_depth_pixels": 5,
            "input_positive_confidence_pixels": 6, "output_positive_confidence_pixels": 5,
            "changed_pixels": 2, "became_positive_pixels": 0, "became_zero_pixels": 1,
            "abs_delta_mean_all_pixels": 0.05, "abs_delta_max": 0.2,
        }
        write_json(frame / "confidence_adjustment.json", {
            "schema_name": "openmvs.dmap.confidence_adjustment", "schema_version": 1,
            "algorithm_stage": "depth_map_optional_confidence_adjustment", "estimation_stage": "photometric",
            "geometric_iteration": None, "status": "complete", "input_available": True,
            "depth_validity_unchanged": True, "final_combination": "fast_only",
            "neighbor_limit": 8, "neighbors": [{"image_id": 20}], "parameters": {},
            "resource_plan": {"path": "filter_resource_plan.json", "schema_name": "openmvs.dmap.filter_resource_plan", "schema_version": 1},
            "maps_requested": True, "maps_enabled": True, "maps": confidence_maps, "write_errors": [], "complete": True,
            "methods": [
                {
                    "name": "adjust_confidence_fast", "enabled": True, "executed": True,
                    "output_available": True, "quality": "exact", "basis": "fast production method",
                    "unavailable_reason": None, "metrics": confidence_metrics,
                },
                {
                    "name": "adjust_confidence", "enabled": False, "executed": False,
                    "output_available": False, "quality": "unavailable", "basis": "full production method",
                    "unavailable_reason": "method disabled by nOptimize", "metrics": None,
                },
                {
                    "name": "final_combined_confidence", "enabled": True, "executed": True,
                    "output_available": True, "quality": "exact", "basis": "fast_only",
                    "unavailable_reason": None, "metrics": confidence_metrics,
                },
            ],
        })
        write_pfm(frame / confidence_maps[0]["path"], scalar * 0.02)
        Image.fromarray(np.asarray([[0, 1, 2], [3, 1, 0]], dtype=np.uint8)).save(
            frame / confidence_maps[1]["path"]
        )
        write_jsonl(instrumentation / "resource_plans.jsonl", [{
            "schema_name": "openmvs.dmap.resource_plan", "schema_version": 2,
            "image_id": 7, "image_name": "images/0007.jpg", "estimation_stage": "photometric",
            "geometric_iteration": None, "decision": "exact_maps", "maps_requested": True,
            "maps_available": True, "exact_requested": True, "exact_available": True,
            "summary_available": True,
            "effective_estimate_bytes": {"device": 1024, "host": 2048, "frame_storage": 4096},
            "limits_mib": {"device": 1, "host": 1, "frame_storage": 1},
            "storage_preflight": {
                "attempted": True, "succeeded": True, "available_bytes": 100000,
                "reserved_before_bytes": 0, "reservation_bytes": 4096,
            },
        }])
        component = {
            "requested_capabilities": {"summary": True, "maps": True, "filter_active": True},
            "effective_capabilities": {"summary": True, "maps": True},
            "effective_estimate_bytes": {"host": 2048, "storage": 4096},
            "storage_preflight": {
                "attempted": True, "succeeded": True, "available_bytes": 100000,
                "reserved_before_bytes": 0, "leased_bytes": 4096, "lease_released": True,
            },
            "decision": "maps_admitted", "reason": "fits", "fatal": False,
            "actual_maps": {
                "map_count": 2, "declared_bytes": 128, "file_bytes": 96,
                "estimate_covers_declared_bytes": True,
            },
        }
        write_json(frame / "filter_resource_plan.json", {
            "schema_name": "openmvs.dmap.filter_resource_plan", "schema_version": 1,
            "reference_image_id": 7, "limits_mib": {"host": 1, "frame_storage": 1},
            "components": {
                "postprocess_filters": component,
                "confidence_adjustment": component,
            },
        })
        return dmap_dev.RunScene(
            "schema4", "baseline", 0, "scene-a", instrumentation, None, None
        ), frame

    def make_mechanism_capture(self, root: Path) -> dmap_dev.RunScene:
        instrumentation = root / "mechanism_instrumentation"
        frame = instrumentation / "depthmaps" / "0007_frame"
        maps = frame / "maps"
        scalar = np.asarray([[0.4, 0.3], [0.2, 0.1]], dtype=np.float32)
        specifications = (
            ("jbu_transfer_depth", "exact"),
            ("jbu_nearest_depth", "derived_exact"),
            ("jbu_transfer_depth_delta", "derived_exact"),
            ("adaptive_patch_activation", "derived_exact"),
            ("adaptive_patch_support_mode", "derived_exact"),
            ("hierarchy_entry_cost", "exact"),
            ("hierarchy_proposed_cost", "proxy"),
            ("hierarchy_improvement_margin", "derived_exact"),
            ("hierarchy_update_status", "exact"),
        )
        entries = []
        for signal, quality in specifications:
            path = maps / f"{signal}.pfm"
            values = (
                np.asarray([[0, 1], [1, 0]], dtype=np.float32)
                if signal.endswith(("activation", "support_mode", "update_status"))
                else scalar
            )
            write_pfm(path, values)
            entries.append({
                "signal": signal,
                "path": f"maps/{signal}.pfm",
                "dtype": "float32",
                "role": "final_state",
                "measurement_quality": quality,
                "measurement_basis": "production_input" if signal.startswith("jbu_") else "production_gate_state",
            })

        health_iterations = []
        for level in (2, 1, 0):
            for logical_iteration in (0, 1):
                health_iterations.append({
                    "pyramid_level": level,
                    "logical_iteration": logical_iteration,
                    "processed": 8,
                    "finite_positive_events": 7,
                    "zero_mass_events": 1,
                    "nonfinite_component_events": 1,
                    "negative_component_events": 0,
                    "nonfinite_sum_events": 0,
                    "degenerate_events": 1,
                    "unassigned_draws": 2,
                    "legacy_last_view_collapse_events": 1,
                    "positive_view_count_sum": 21,
                })
        totals = {
            name: sum(row[name] for row in health_iterations)
            for name in dmap_dev.VIEW_PROBABILITY_HEALTH_COUNTERS
        }
        write_json(frame / "summary.json", {
            "schema_version": 4,
            "image_id": 7,
            "image_name": "images/0007.jpg",
            "safe_image_name": "frame",
            "scale_level": 0,
            "width": 2,
            "height": 2,
            "view_probability_health": {
                "schema_name": "openmvs.dmap.view_probability_health",
                "schema_version": 1,
                "requested": True,
                "available": True,
                "accounting_valid": True,
                "iterations": health_iterations,
                "totals": totals,
            },
        })
        write_json(frame / "map_manifest.json", {
            "schema_name": "openmvs.dmap.map_manifest",
            "schema_version": 4,
            "num_iterations": 2,
            "num_logical_states": 3,
            "map_granularity": "logical_iteration",
            "exact_capture": {
                "requested": False,
                "available": False,
                "unavailable_reason": "mechanism observer uses Process<false>",
                "num_views": 0,
            },
            "maps": entries,
        })
        header = (
            "image_id,image_name,scale_level,iteration,num_pixels,valid_ratio,changed_ratio,"
            "candidates_tested,candidates_accepted,acceptance_rate,mean_cost,phase,pass_index\n"
        )
        rows = []
        for level in (2, 1, 0):
            rows.append(f"7,images/0007.jpg,{level},-1,4,1,0,0,0,0,0.5,initialization,0\n")
            rows.append(f"7,images/0007.jpg,{level},0,4,1,0.25,8,2,0.25,0.4,iteration,1\n")
            rows.append(f"7,images/0007.jpg,{level},1,4,1,0.125,8,1,0.125,0.3,iteration,2\n")
        (frame / "iteration.csv").write_text(header + "".join(rows), encoding="utf-8")
        return dmap_dev.RunScene(
            "mechanisms", "variant", 0, "scene-a", instrumentation, None, None
        )

    def test_cpp_shaped_mechanism_artifacts_and_probability_health_reach_model(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            root = Path(directory)
            scene = self.make_mechanism_capture(root)

            frame_rows, iteration_rows, _timing_rows = dmap_dev.load_instrumentation(scene)
            catalog, availability = dmap_dev.build_map_catalog([scene])

            self.assertTrue(frame_rows[0]["view_probability_health_available"])
            health_rows = [
                row for row in iteration_rows
                if row.get("view_probability_processed") is not None
            ]
            self.assertEqual(
                {(int(row["scale_level"]), int(row["logical_iteration"])) for row in health_rows},
                {(level, iteration) for level in (0, 1, 2) for iteration in (0, 1)},
            )
            self.assertEqual(health_rows[0]["view_probability_degenerate_ratio"], 0.125)
            self.assertEqual(health_rows[0]["view_probability_mean_positive_views"], 2.625)

            by_signal = {row["signal"]: row for row in catalog.to_dict("records")}
            for signal in ("jbu_transfer_depth", "adaptive_patch_support_mode"):
                self.assertEqual(by_signal[signal]["logical_iteration"], -1)
                self.assertEqual(by_signal[signal]["role"], "logical_state")
            for signal in dmap_dev.OPTIONAL_MECHANISM_FINAL_STATE_SIGNALS:
                self.assertTrue(pd.isna(by_signal[signal]["logical_iteration"]))
                self.assertEqual(by_signal[signal]["role"], "final_state")
            self.assertTrue(availability[
                (availability["signal"] == "jbu_transfer_depth")
                & (availability["logical_iteration"] == -1)
            ].iloc[0]["available"])
            self.assertTrue(availability[
                availability["signal"] == "hierarchy_update_status"
            ].iloc[0]["available"])

            output = root / "report"
            output.mkdir()
            model = dmap_report_model.build_report_model(
                config={"name": "mechanism contract"},
                experiment_root=root,
                output_dir=output,
                run_scenes=[scene],
                frames=pd.DataFrame(frame_rows),
                iterations=pd.DataFrame(iteration_rows),
                performance=pd.DataFrame(),
                annotations=pd.DataFrame(),
                comparisons=pd.DataFrame(),
                gates=[],
                pareto=[],
                findings=[],
                map_catalog=catalog,
                signal_availability=availability,
                inventory={"scenes": {}, "overall_plots": []},
                metric_specs={},
            )
            report_frame = model["scenes"][0]["frames"][0]
            self.assertEqual(report_frame["pyramid_levels"], [0, 1, 2])
            self.assertTrue(report_frame["run_frames"][0]["view_probability_health"]["available"])
            hierarchy = next(
                row for row in report_frame["maps"]
                if row["signal"] == "hierarchy_update_status" and row["available"]
            )
            self.assertEqual(hierarchy["category_legend"]["1"], "crossed configured margin")
            self.assertEqual(
                hierarchy["preview"]["scale_mode"], "categorical_registered_codes"
            )
            multiscale_recipe = next(
                recipe for recipe in model["investigation_guide"]["recipes"]
                if recipe["key"] == "multiscale"
            )
            self.assertIn(
                "hierarchy_update_status", multiscale_recipe["action"]["signals"]
            )
            ui = Path(__file__).resolve().parents[1] / "dmap_report_ui"
            dmap_report_model.render_investigation_html(
                output / "02_investigation.html",
                model,
                ui / "investigation.html",
                ui / "investigation.css",
                ui / "investigation.js",
            )
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
            ):
                (output / name).touch()
            validation = dmap_report_model.validate_report_model(model, output)
            self.assertTrue(validation["valid"], validation)

            chromium = shutil.which("chromium") or shutil.which("google-chrome")
            if chromium:
                url = (output / "02_investigation.html").as_uri() + "#" + urlencode({
                    "baseline": "mechanisms",
                    "variant": "mechanisms",
                    "scene": report_frame["scene_id"],
                    "frame": report_frame["id"],
                    "captureStage": "photometric",
                    "pyramidLevel": "0",
                    "alignment": "final",
                    "mapPreset": "custom",
                    "signals": "hierarchy_update_status",
                })
                browser = subprocess.run(
                    [
                        chromium,
                        "--headless",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--allow-file-access-from-files",
                        "--run-all-compositor-stages-before-draw",
                        "--virtual-time-budget=2500",
                        f"--user-data-dir={root / 'chromium-profile'}",
                        "--dump-dom",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(browser.returncode, 0, browser.stderr)
                self.assertIn("crossed configured margin", browser.stdout)
                self.assertIn("View probability health", browser.stdout)
                self.assertIn("hierarchy_update_status for mechanisms", browser.stdout)

    def test_exact_disabled_hysteresis_signals_report_manifest_reason(self) -> None:
        with tempfile.TemporaryDirectory(dir=SCRIPT_DIR) as directory:
            root = Path(directory)
            scene = self.make_mechanism_capture(root)
            write_json(scene.instrumentation_dir / "run_metadata.json", {
                "cuda_patchmatch_parameters": {
                    "low_texture_update_min_gain": 0.0005,
                    "low_texture_update_gate": 1,
                },
            })

            _catalog, availability = dmap_dev.build_map_catalog([scene])

            iterative = availability[
                (availability["logical_iteration"] == 0)
                & availability["signal"].isin(
                    dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
                )
            ]
            self.assertEqual(
                set(iterative["signal"]),
                set(dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS),
            )
            self.assertEqual(
                set(iterative["availability_reason"]),
                {"exact_capture_unavailable: mechanism observer uses Process<false>"},
            )
            self.assertFalse(iterative["required"].any())

    def test_schema4_catalog_tables_and_report_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, _frame = self.make_capture(root)
            catalog, availability = dmap_dev.build_map_catalog([scene])

            view_rows = catalog[catalog["signal"] == "view_cost_components_exact"]
            self.assertEqual(set(view_rows["source_view_index"]), {0, 1})
            self.assertEqual(set(view_rows["source_image_id"]), {20, 21})
            self.assertIn("photo_after_prior", view_rows.iloc[0]["channels_json"])
            self.assertEqual(
                len(availability),
                41 + len(dmap_dev.MECHANISM_LOGICAL_STATE_SIGNALS),
            )
            filter_maps = catalog[catalog["role"] == "postprocess_filter_state"]
            self.assertEqual(set(filter_maps["algorithm_stage"]), {"remove_speckles"})
            self.assertEqual(set(filter_maps["stage_index"]), {0})
            self.assertTrue((filter_maps["measurement_quality"] == "exact").all())
            confidence_maps = catalog[catalog["role"] == "confidence_adjustment_state"]
            self.assertEqual(set(confidence_maps["algorithm_stage"]), {"confidence_adjustment"})
            missing = availability[
                (availability["signal"] == "candidate_runner_up_cost_exact")
                & (availability["logical_iteration"] == -1)
            ].iloc[0]
            self.assertTrue(missing["required"])
            self.assertEqual(missing["availability_reason"], "not_declared_in_manifest")
            mechanism_missing = availability[
                (availability["signal"] == "view_probability_mass")
                & (availability["logical_iteration"] == -1)
            ].iloc[0]
            self.assertFalse(mechanism_missing["required"])
            self.assertEqual(mechanism_missing["availability_reason"], "not_declared_in_manifest")
            self.assertTrue(availability[
                availability["signal"].isin(
                    dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
                )
            ].empty)
            self.assertTrue(
                availability[availability["signal"] == "jbu_transfer_depth_delta"].empty
            )

            tables = dmap_dev.load_schema4_observability([scene])
            self.assertEqual(tables["exact_iterations"].iloc[0]["tested_candidates"], 78)
            self.assertEqual(set(tables["exact_iterations"]["pyramid_level"]), {0})
            hysteresis = dmap_report_model.build_low_texture_update_hysteresis_metrics(
                tables["exact_iterations"]
            )
            self.assertEqual(hysteresis["rows"][0]["eligible_pixels"], 5)
            self.assertEqual(hysteresis["rows"][0]["propagation_rejected"], 1)
            self.assertIn("exact hot-kernel pixel-record aggregation", hysteresis["rows"][0]["measurement_basis"])
            self.assertEqual(tables["exact_observability"].iloc[0]["num_views"], 2)
            self.assertEqual(tables["exact_views"].iloc[0]["source_image_id"], 20)
            self.assertEqual(tables["cpu_view_candidates"].iloc[0]["score_component_angle_weight_sum"], 1.2)
            self.assertEqual(tables["cpu_estimation_selection"].iloc[0]["decision"], "selected")
            self.assertEqual(
                tables["cpu_estimation_selection"].iloc[0]["admission_policy"],
                "ranked_prefix_without_score_cutoff",
            )
            self.assertEqual(
                tables["cpu_estimation_selection"].iloc[0]["admission_policy_origin"],
                "fixture_ranked_prefix_policy",
            )
            self.assertFalse(tables["cpu_estimation_selection"].iloc[0]["score_threshold_applied"])
            self.assertFalse(tables["cpu_estimation_selection"].iloc[0]["would_pass_configured_score_threshold"])
            self.assertEqual(tables["cpu_estimation_selection"].iloc[0]["configured_effective_min_score"], 3.0)
            self.assertEqual(list(tables["postprocess_filters"]["stage_name"]), ["remove_speckles", "fill_gaps"])
            self.assertEqual(list(tables["postprocess_filters"]["artifact_status"]), ["available", "disabled"])
            self.assertEqual(tables["postprocess_filters"].iloc[0]["removed_pixels"], 1)
            self.assertEqual(list(tables["confidence_adjustment"]["artifact_status"]), ["available", "disabled", "available"])
            self.assertTrue(tables["cuda_resource_plans"].iloc[0]["valid"])
            self.assertEqual(
                set(tables["filter_resource_plans"]["component"]),
                {"postprocess_filters", "confidence_adjustment"},
            )
            self.assertTrue(tables["filter_resource_plans"]["valid"].all())
            self.assertTrue(tables["resource_plan_validation"]["valid"].all())

            exact_cost = dmap_dev.build_exact_cost_evolution(catalog)
            self.assertEqual(set(exact_cost["signal"]), {"cost_total_production_exact", "gap_winner_runner_up_exact"})
            output = root / "report"
            output.mkdir()
            resource_output = dmap_dev.write_dataframe(
                tables["resource_plan_validation"],
                output / "resource_plan_validation.parquet",
            )
            if resource_output["parquet"] is None:
                self.assertIn("Unable to find a usable engine", resource_output["parquet_error"])
                self.assertNotIn("tried to convert to boolean", resource_output["parquet_error"])
            else:
                self.assertNotIn("parquet_error", resource_output)
            frames = pd.DataFrame([{
                "run": "schema4", "role": "baseline", "repeat": 0, "scene_id": "scene-a",
                "image_id": 7, "image_name": "images/0007.jpg", "safe_image_name": "frame",
            }])
            model = dmap_report_model.build_report_model(
                config={"name": "schema4", "_config_path": str(root / "config.yaml")},
                experiment_root=root, output_dir=output, run_scenes=[scene], frames=frames,
                iterations=pd.DataFrame(), performance=pd.DataFrame(), annotations=pd.DataFrame(),
                comparisons=pd.DataFrame(), gates=[], pareto=[], findings=[], map_catalog=catalog,
                signal_availability=availability, inventory={"scenes": {}, "overall_plots": []},
                metric_specs=dmap_dev.METRICS, exact_cost_evolution=exact_cost,
                exact_observability=tables["exact_observability"],
                exact_iterations=tables["exact_iterations"], exact_views=tables["exact_views"],
                cpu_view_candidates=tables["cpu_view_candidates"],
                cpu_estimation_selection=tables["cpu_estimation_selection"],
                postprocess_filters=tables["postprocess_filters"],
                confidence_adjustment=tables["confidence_adjustment"],
                cuda_resource_plans=tables["cuda_resource_plans"],
                filter_resource_plans=tables["filter_resource_plans"],
                resource_plan_validation=tables["resource_plan_validation"],
            )
            self.assertTrue(model["mechanics"]["availability"]["exact_hot_kernel"])
            self.assertTrue(model["mechanics"]["availability"]["cpu_view_ranking"])
            self.assertTrue(model["mechanics"]["availability"]["postprocess_filters"])
            self.assertTrue(model["mechanics"]["availability"]["postprocess_filter_contract"])
            self.assertTrue(model["mechanics"]["availability"]["confidence_adjustment"])
            self.assertTrue(model["mechanics"]["availability"]["confidence_adjustment_contract"])
            self.assertTrue(model["mechanics"]["availability"]["resource_plans_valid"])
            self.assertIn(
                "low_texture_update_hysteresis",
                {recipe["key"] for recipe in model["investigation_guide"]["recipes"]},
            )
            extension = next(
                row for row in model["contract"]["optional_extensions"]
                if row["extension_id"] == "low_texture_update_hysteresis"
            )
            self.assertFalse(extension["base_schema_required"])
            self.assertTrue(extension["counter_evidence_declared"])
            report_frame = model["scenes"][0]["frames"][0]
            self.assertEqual(report_frame["pyramid_levels"], [0])
            self.assertNotIn(
                "unspecified", report_frame["logical_iterations_by_pyramid_level"]
            )
            view_map = next(
                item for item in model["scenes"][0]["frames"][0]["maps"]
                if item["signal"] == "view_cost_components_exact" and item["source_view_index"] == 0
            )
            self.assertEqual(view_map["mechanism"], "view_selection")
            self.assertEqual(len(view_map["preview"]["channels"]), 3)
            self.assertEqual(view_map["preview"]["channels"][2]["label"], "total")
            filter_map = next(
                item for item in model["scenes"][0]["frames"][0]["maps"]
                if item["signal"] == "00_remove_speckles_depth_delta"
            )
            self.assertEqual(filter_map["mechanism"], "filtering")
            self.assertEqual(filter_map["algorithm_stage"], "remove_speckles")
            transition_map = next(
                item for item in model["scenes"][0]["frames"][0]["maps"]
                if item["signal"] == "00_remove_speckles_validity_transition"
            )
            self.assertEqual(transition_map["preview"]["scale_mode"], "discrete_transition_codes")
            drilldowns = {"entries": [{
                "status": "complete", "capture_profile": "trace", "scene_id": "scene-a",
                "image_id": 7, "request_sha256": "a" * 64, "executions": "../executions.json",
                "trace_data": {
                    "available": True, "row_count": 1,
                    "sources": [{"run": "variant", "available": True, "source_path": "../traces.jsonl"}],
                    "rows": [{
                        "run": "variant", "x": 4, "y": 5, "logical_iteration": 0,
                        "source": "changed_unknown",
                        "cost": {"before": 0.8, "after": 0.4, "improvement": 0.4},
                        "depth": {"before": 1.0, "after": 1.1, "absolute_change": 0.1},
                        "normal": {"angle_change_degrees": 2.0},
                        "view": {"selected_count": 2, "selected_before_mask": 3, "selected_mask": 5},
                    }],
                },
            }]}
            markdown = dmap_dev.build_markdown(
                {"_config_path": str(root / "config.yaml")}, output / "report.md",
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                pd.DataFrame(), [], [], [], [], {}, {}, {}, exact_cost,
                tables["exact_iterations"], tables["exact_views"],
                tables["cpu_view_candidates"], tables["cpu_estimation_selection"],
                tables["postprocess_filters"], tables["confidence_adjustment"],
                tables["cuda_resource_plans"], tables["filter_resource_plans"],
                tables["resource_plan_validation"], pd.DataFrame(), drilldowns,
                low_texture_hysteresis=hysteresis,
            )
            self.assertIn("### Exact Production Cost Evolution", markdown)
            self.assertIn("cost_total_production_exact", markdown)
            self.assertIn("### Exact Update Attribution and Winner Gap", markdown)
            self.assertIn(
                "### Low-texture Update Hysteresis (Optional Extension)", markdown
            )
            self.assertIn("logical iteration", markdown)
            self.assertNotIn("#### Low-texture Accepted-gain Census", markdown)
            self.assertNotIn("this capture did not produce enabled low-texture update hysteresis counters", markdown)
            self.assertIn("33.33%", markdown)
            self.assertIn("### Exact Per-view Selection and Contribution", markdown)
            self.assertIn("### CPU View Ranking and Estimation Selection", markdown)
            self.assertIn("ranked_prefix_without_score_cutoff", markdown)
            self.assertIn("would pass configured cutoff", markdown)
            self.assertIn("### Sequential Postprocess Filtering", markdown)
            self.assertIn("### Confidence Adjustment", markdown)
            self.assertIn("### Resource Admission and Storage Preflight", markdown)
            self.assertIn("#### Final State per Run", markdown)
            self.assertIn("stored initialization assignments", markdown)
            self.assertIn("[interactive investigation interface](02_investigation.html)", markdown)
            self.assertIn("tools/dmap_observability.sh capture", markdown)
            self.assertIn("scripts/python/dmap_dev.py report", markdown)
            self.assertIn("--output-dir", markdown)
            self.assertIn("tools/dmap_observability.sh validate", markdown)
            self.assertIn("### Completed Targeted Pixel Traces", markdown)
            self.assertIn("full-frame Process<true> maps", markdown)
            self.assertIn("no compact exact-trace kernel path", markdown)
            self.assertIn("Source attribution is classified per row", markdown)
            self.assertIn(
                "changed_unknown (proxy; legacy_unclassified_trace)", markdown
            )
            self.assertIn(
                "variant photometric traces.jsonl (external evidence:", markdown
            )
            self.assertNotIn("](../traces.jsonl)", markdown)
            self.assertLess(
                markdown.index(dmap_report_model.INVESTIGATION_GUIDE_HEADING),
                markdown.index("## 1. Executive Summary"),
            )
            for recipe in (
                "### Debug a cost-function change",
                "### Debug a propagation change",
                "### Debug a patch or photometric-scoring change",
                "### Debug a multiscale change",
            ):
                self.assertIn(recipe, markdown)
            self.assertIn(
                "Pyramid level selects algorithm state; Shared and Local select only preview color normalization.",
                markdown,
            )

    def test_schema1_cpu_selection_policy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, frame = self.make_capture(root)
            write_json(frame / "cpu_view_estimation_selection.json", {
                "schema_name": "openmvs.dmap.cpu_view_estimation_selection",
                "schema_version": 1,
                "selection_succeeded": True,
                "selection_mode": "ranked_prefix",
                "ordered_cutoff": {"reason": "score_below_effective_minimum"},
                "parameters": {"effective_min_score": 2.0},
                "candidates": [{
                    "candidate_image_id": 20,
                    "filtered_rank_zero_based": 0,
                    "ranking_score": 2.5,
                    "selected": True,
                    "selected_rank_zero_based": 0,
                    "decision": "selected",
                }],
            })
            tables = dmap_dev.load_schema4_observability([scene])
            row = tables["cpu_estimation_selection"].iloc[0]
            self.assertEqual(row["admission_policy"], "ranked_prefix_with_score_cutoff")
            self.assertEqual(row["admission_policy_origin"], "legacy_schema_v1_inference")
            self.assertTrue(row["score_threshold_applied"])
            self.assertEqual(row["configured_effective_min_score"], 2.0)

    def test_missing_optional_filter_artifacts_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, frame = self.make_capture(root)
            (frame / "postprocess_filters.json").unlink()
            (frame / "confidence_adjustment.json").unlink()

            tables = dmap_dev.load_schema4_observability([scene])

            self.assertEqual(len(tables["postprocess_filters"]), 2)
            self.assertTrue((tables["postprocess_filters"]["artifact_status"] == "unavailable").all())
            self.assertEqual(set(tables["postprocess_filters"]["unavailable_reason"]), {"artifact_not_produced"})
            self.assertEqual(len(tables["confidence_adjustment"]), 3)
            self.assertTrue((tables["confidence_adjustment"]["artifact_status"] == "unavailable").all())

    def test_referenced_filter_resource_plan_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene, frame = self.make_capture(Path(directory))
            (frame / "filter_resource_plan.json").unlink()

            tables = dmap_dev.load_schema4_observability([scene])
            filter_validation = tables["resource_plan_validation"][
                tables["resource_plan_validation"]["plan_kind"] == "optional_filter"
            ]

            self.assertTrue(filter_validation["required"].all())
            self.assertTrue((~filter_validation["valid"]).all())
            self.assertTrue(
                filter_validation["validation_errors_json"].str.contains(
                    "referenced_filter_resource_plan_missing", regex=False
                ).all()
            )

    def test_filter_resource_plan_preserves_zero_reference_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _scene, frame = self.make_capture(Path(directory))
            plan_path = frame / "filter_resource_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["reference_image_id"] = 0
            write_json(plan_path, plan)

            rows = dmap_dev.filter_resource_plan_rows(frame, {"image_id": 0})

            self.assertTrue(all(row["valid"] for row in rows), rows)

    def test_unreferenced_absent_filter_component_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene, frame = self.make_capture(Path(directory))
            (frame / "confidence_adjustment.json").unlink()
            plan_path = frame / "filter_resource_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            del plan["components"]["confidence_adjustment"]
            write_json(plan_path, plan)

            tables = dmap_dev.load_schema4_observability([scene])
            rows = tables["filter_resource_plans"].set_index("component")

            self.assertTrue(rows.loc["postprocess_filters", "required"])
            self.assertTrue(rows.loc["postprocess_filters", "valid"])
            self.assertFalse(rows.loc["confidence_adjustment", "required"])
            self.assertFalse(rows.loc["confidence_adjustment", "available"])
            self.assertTrue(pd.isna(rows.loc["confidence_adjustment", "valid"]))
            self.assertEqual(
                json.loads(rows.loc["confidence_adjustment", "validation_errors_json"]), []
            )

    def test_referenced_missing_filter_component_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene, frame = self.make_capture(Path(directory))
            plan_path = frame / "filter_resource_plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            del plan["components"]["confidence_adjustment"]
            write_json(plan_path, plan)

            tables = dmap_dev.load_schema4_observability([scene])
            rows = tables["filter_resource_plans"].set_index("component")

            self.assertTrue(rows.loc["confidence_adjustment", "required"])
            self.assertFalse(rows.loc["confidence_adjustment", "available"])
            self.assertFalse(rows.loc["confidence_adjustment", "valid"])
            self.assertIn(
                "component_missing",
                json.loads(rows.loc["confidence_adjustment", "validation_errors_json"]),
            )


if __name__ == "__main__":
    unittest.main()
