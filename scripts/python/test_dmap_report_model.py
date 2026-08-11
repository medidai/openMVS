#!/usr/bin/env python3
"""Focused tests for the structured DMAP report model and investigation UI."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from PIL import Image
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_report_model


def write_pfm(path: Path, values: np.ndarray) -> None:
    data = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    color = data.ndim == 3 and data.shape[2] == 3
    height, width = data.shape[:2]
    with path.open("wb") as handle:
        handle.write(b"PF\n" if color else b"Pf\n")
        handle.write(f"{width} {height}\n-1.0\n".encode("ascii"))
        np.flipud(data).astype("<f4").tofile(handle)


def trace_row(iteration: int, *, x: int = 11, y: int = 13) -> dict:
    return {
        "image_id": 7,
        "scale_number": 0,
        "trace_index": 0,
        "label": "test-pixel",
        "x": x,
        "y": y,
        "logical_iteration": iteration,
        "source": "init" if iteration == -1 else "propagation",
        "source_quality": "exact",
        "measurement_basis": "exact_hot_kernel",
        "selected_view_count": 3,
        "selected_views_mask": 7,
        "selected_views_before_mask": 3,
        "depth_before": 1.0,
        "depth_after": 1.1,
        "cost_before": 0.8,
        "cost_after": 0.6,
        "cost_improvement": 0.2,
        "depth_abs_change": 0.1,
        "depth_rel_change": 0.1,
        "normal_angle_change": 2.5,
        "view_entropy": 0.75,
        "photometric_cost_after": 0.5,
        "photo_prior_cost_after": 0.55,
        "depth_prior_cost_after": 0.05,
        "depth_prior_weight_after": 0.25,
        "geometric_cost_after": 0.05,
        "ref_variance": 0.2,
        "low_depth": 0.9,
        "raw_pass_indices": [0] if iteration == -1 else [1, 2],
        "neighbor_costs": [0.6, 0.7],
        "view_costs": [0.5, 0.6, 0.7],
        "view_photometric_costs": [0.45, 0.55, 0.65],
        "view_geometric_costs": [0.05, 0.05, 0.05],
        "view_weights": [1, 1, 1],
        "bad_reasons": [0, 0, 0],
    }


def make_evidence_context() -> dict:
    context = {
        "schema_name": dmap_report_model.EVIDENCE_CONTEXT_SCHEMA_NAME,
        "schema_version": dmap_report_model.EVIDENCE_CONTEXT_SCHEMA_VERSION,
        "context_sha256": "",
        "subject": {
            "experiment_id": "experiment-72",
            "title": "Experiment 72 authority split",
            "summary": "Diagnostic mechanics are interpreted beside external production quality.",
        },
        "mechanics_authority": {
            "experiment_id": "experiment-72",
            "authority_role": "diagnostic_mechanics",
            "process_specialization": "Process<true>",
            "quality_eligible": False,
            "status": "valid",
            "verdict": "supported",
            "headline": "Mechanism behavior is captured",
            "scope": "One diagnostic scene with complete logical-iteration maps.",
            "summary_metrics": [{
                "key": "captured_frames", "label": "Captured frames", "value": 9,
                "unit": "frames", "status": "pass",
                "description": "Validated diagnostic frames.",
            }],
            "candidate_coverage": [{
                "candidate": "external-variant", "coverage": "quality_only",
                "note": "No matching deep capture.",
            }],
            "source_artifacts": [{
                "role": "mechanics_validation", "sha256": "a" * 64,
                "bytes": 123, "schema_name": "openmvs.test.mechanics",
                "schema_version": 1, "content_digest": "b" * 64,
                "cardinality": {"frames": 9},
            }],
        },
        "quality_authority": {
            "experiment_id": "experiment-71",
            "authority_role": "production_quality",
            "process_specialization": "Process<false>",
            "quality_eligible": True,
            "status": "terminal_valid",
            "headline": "External accuracy-first ranking",
            "scope": "Two production scenes and terminal DMAP endpoints.",
            "manual_promotion_required": True,
            "candidates": [{
                "candidate": "external-variant", "mechanics_coverage": "quality_only",
                "accuracy_rank": 1, "strict_accuracy_pass": True, "scene_count": 2,
                "primary_scene_metric_rows": 8, "baseline_successful_structures": 5,
                "paired_successful_structures": 5, "lost_baseline_fit_count": 0,
                "availability_biased": False, "median_normalized_noise_loss": -0.2,
                "worst_normalized_noise_loss": 0.1, "residual_p95_delta_mm": -0.4,
                "threshold_auc_delta_pp": 0.8, "inlier_5mm_delta_pp": 0.5,
                "effective_coverage_delta_pp": 0.3, "spatial_coverage_delta_pp": 0.2,
                "estimator_validity_delta_pp": 1.5, "endpoint_validity_delta_pp": 0.9,
                "runtime_delta_percent": 4.0,
            }],
            "source_artifacts": [{
                "role": "production_candidate_ledger", "sha256": "c" * 64,
                "bytes": 456, "schema_name": "openmvs.test.quality",
                "schema_version": 2, "content_digest": "d" * 64,
                "cardinality": {"candidates": 1, "scenes": 2},
            }],
        },
        "separation_contract": dict(
            dmap_report_model.EVIDENCE_CONTEXT_SEPARATION_CONTRACT
        ),
    }
    context["context_sha256"] = dmap_report_model.evidence_context_digest(context)
    return context


def write_completed_trace_drilldown(
    root: Path,
    rows: list[dict],
    *,
    run: str = "base",
    scene_id: str = "scene-a",
    exact_capture: bool = True,
) -> tuple[str, Path]:
    requested_coordinate = (rows[0]["x"], rows[0]["y"]) if rows else (11, 13)
    request_value = {
        "schema_name": dmap_report_model.dmap_drilldown.SCHEMA_NAME,
        "schema_version": dmap_report_model.dmap_drilldown.SCHEMA_VERSION,
        "experiment": {
            "experiment_id": "test", "config_name": "experiment.yaml",
            "config_sha256": "a" * 64, "source_revision": None,
            "source_dirty": None,
        },
        "capture_profile": "trace",
        "target": {
            "scene_id": scene_id, "image_id": 7,
            "pixels": [{"x": requested_coordinate[0], "y": requested_coordinate[1]}],
            "roi": None, "trace_pixel_count": 1,
        },
        "runs": [{
            "label": run, "role": "baseline", "densify_args": [],
            "ini_overrides": {},
        }],
        "capture": {
            "instrumentation_level": "maps", "write_maps": True,
            "patch_match_cuda_instances": 1,
            "process_specialization": "Process<true>",
            "compact_exact_trace_available": False,
            "storage_policy": "full-frame exact maps plus selected trace rows",
        },
    }
    request_value["request_sha256"] = dmap_report_model.dmap_drilldown.request_digest(
        request_value
    )
    request_sha256 = request_value["request_sha256"]
    capture = root / "drilldowns" / "captures" / request_sha256
    traces = (
        capture / "runs" / run / scene_id / "dmap_instrumentation"
        / "instrumentation" / "traces.jsonl"
    )
    traces.parent.mkdir(parents=True, exist_ok=True)
    traces.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    run_root = capture / "runs" / run / scene_id
    (run_root / "repro.json").write_text(json.dumps({
        "command": [
            "/tmp/DensifyPointCloudDMapObserve",
            "--dmap-instrumentation-level", "maps",
            "--dmap-instrumentation-write-maps", "1",
        ],
        "return_code": 0,
        "dry_run": False,
    }), encoding="utf-8")
    if exact_capture:
        frame = run_root / "dmap_instrumentation" / "depthmaps" / "depth0007"
        frame.mkdir(parents=True)
        completion_reference = {
            "schema_name": "openmvs.dmap.capture_complete",
            "schema_version": 1,
            "path": "capture_complete.json",
            "maps_complete": True,
            "eligible": True,
        }
        (frame / "summary.json").write_text(json.dumps({
            "schema_name": "openmvs.dmap.frame_summary",
            "schema_version": 4,
            "image_id": 7,
            "image_name": "image0007.jpg",
            "estimation_stage": "photometric",
            "geometric_iteration": None,
            "completion_marker": completion_reference,
        }), encoding="utf-8")
        (frame / "map_manifest.json").write_text(json.dumps({
            "schema_name": "openmvs.dmap.map_manifest",
            "schema_version": 4,
            "complete": True,
            "pyramid_level": 0,
            "num_iterations": 1,
            "num_logical_states": 2,
            "write_errors": [],
            "observer_sidecars": {"complete": True},
            "exact_capture": {"requested": True, "available": True},
        }), encoding="utf-8")
        (frame / "capture_complete.json").write_text(json.dumps({
            "schema_name": "openmvs.dmap.capture_complete",
            "schema_version": 1,
            "capture_kind": "maps",
            "image_id": 7,
            "image_name": "image0007.jpg",
            "estimation_stage": "photometric",
            "geometric_iteration": None,
            "maps_complete": True,
            "observer_sidecars_complete": True,
            "map_manifest": {
                "path": "map_manifest.json", "schema_version": 4, "complete": True,
            },
            "summary": {"path": "summary.json", "schema_version": 4},
        }), encoding="utf-8")
    executions = capture / "executions.json"
    executions.write_text(json.dumps({
        "schema_name": "openmvs.dmap.drilldown_executions",
        "schema_version": 1,
        "request_sha256": request_sha256,
        "executions": [{
            "run": run,
            "scene_id": scene_id,
            "capture_profile": "trace",
            "return_code": 0,
            "validation": "validated trace pixels",
            "reused": False,
        }],
    }), encoding="utf-8")
    request = root / "drilldowns" / "requests" / f"{request_sha256}.yaml"
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(yaml.safe_dump(request_value, sort_keys=False), encoding="utf-8")
    (capture / "request.yaml").write_bytes(request.read_bytes())
    index = root / "drilldowns" / "index.json"
    index.write_text(json.dumps({
        "schema_name": "openmvs.dmap.drilldown_index",
        "schema_version": 1,
        "entries": [{
            "request_sha256": request_sha256,
            "capture_profile": "trace",
            "scene_id": scene_id,
            "image_id": 7,
            "trace_pixel_count": 1,
            "run_labels": [run],
            "status": "complete",
            "request": str(request.relative_to(root)),
            "capture": str(capture.relative_to(root)),
            "executions": str(executions.relative_to(root)),
        }],
    }), encoding="utf-8")
    return request_sha256, traces


class DMapReportModelTests(unittest.TestCase):
    def test_reference_thumbnail_rejects_symlink_and_malformed_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            report.mkdir()
            external = root / "outside-secret.jpg"
            external.write_bytes(b"not an image: private payload")
            linked = root / "linked.jpg"
            linked.symlink_to(external)

            self.assertIsNone(
                dmap_report_model.materialize_reference_thumbnail(
                    linked,
                    report,
                    "scene-a",
                    7,
                    allowed_roots=[root],
                )
            )
            self.assertFalse((report / "interactive" / "reference_images").exists())

    def test_identity_integer_preserves_zero(self) -> None:
        self.assertEqual(dmap_report_model._identity_integer(0), 0)
        self.assertEqual(dmap_report_model._identity_integer(None), -1)

    def test_investigation_guide_uses_registry_without_base_extension_clutter(self) -> None:
        registry = {
            "signals": [{
                "signal_id": "texture_reliability_experiment",
                "mechanism": "texture",
                "default_visible": True,
            }]
        }

        guide = dmap_report_model.investigation_guide_for_registry(registry)
        recipes = {row["key"]: row for row in guide["recipes"]}

        self.assertNotIn("low_texture_update_hysteresis", recipes)
        self.assertIn(
            "texture_reliability_experiment",
            recipes["texture"]["action"]["signals"],
        )
        self.assertEqual(
            dmap_report_model.optional_extension_contracts(registry), []
        )
        improvement_registry = dmap_report_model.component_registry.build_registry([
            {"signal": "cost_improvement_exact"}
        ])
        improvement_guide = dmap_report_model.investigation_guide_for_registry(
            improvement_registry
        )
        improvement_recipes = {
            row["key"]: row for row in improvement_guide["recipes"]
        }
        self.assertIn(
            "cost_improvement_exact",
            improvement_recipes["cost"]["action"]["signals"],
        )
        self.assertIn(
            "cost_improvement_exact",
            improvement_recipes["propagation"]["action"]["signals"],
        )
        self.assertIn("cost_improvement_exact", dmap_report_model.DEFAULT_SIGNALS)

        extension_registry = {
            "signals": [{
                "signal_id": "low_texture_update_eligible_exact",
                "mechanism": "candidate_update",
                "default_visible": True,
            }]
        }
        extension_guide = dmap_report_model.investigation_guide_for_registry(
            extension_registry
        )
        self.assertIn(
            "low_texture_update_hysteresis",
            {row["key"] for row in extension_guide["recipes"]},
        )
        extension_contracts = dmap_report_model.optional_extension_contracts(
            extension_registry
        )
        self.assertEqual(
            extension_contracts[0]["extension_id"],
            "low_texture_update_hysteresis",
        )
        self.assertFalse(extension_contracts[0]["base_schema_required"])

    def validate_drilldowns(
        self, drilldowns: dict, output_dir: Path
    ) -> dict:
        return dmap_report_model.validate_report_model({
            "schema_name": dmap_report_model.SCHEMA_NAME,
            "schema_version": 1,
            "drilldowns": drilldowns,
            "runs": [],
            "scenes": [],
            "aggregates": {"regressions": []},
        }, output_dir)

    def build_availability_model(
        self,
        root: Path,
        postprocess_filters: pd.DataFrame,
        confidence_adjustment: pd.DataFrame,
    ) -> dict:
        output = root / "report"
        output.mkdir()
        return dmap_report_model.build_report_model(
            config={"name": "availability"},
            experiment_root=root,
            output_dir=output,
            run_scenes=[],
            frames=pd.DataFrame(),
            iterations=pd.DataFrame(),
            performance=pd.DataFrame(),
            annotations=pd.DataFrame(),
            comparisons=pd.DataFrame(),
            gates=[],
            pareto=[],
            findings=[],
            map_catalog=pd.DataFrame(),
            signal_availability=pd.DataFrame(),
            inventory={"scenes": {}, "overall_plots": []},
            metric_specs={},
            postprocess_filters=postprocess_filters,
            confidence_adjustment=confidence_adjustment,
        )

    def test_evidence_context_is_versioned_content_attested_and_strict(self) -> None:
        context = make_evidence_context()

        validation = dmap_report_model.validate_evidence_context(context)

        self.assertTrue(validation["valid"], validation)
        self.assertEqual(validation["candidate_count"], 1)
        mutated = json.loads(json.dumps(context))
        mutated["subject"]["summary"] = "mutated after attestation"
        invalid = dmap_report_model.validate_evidence_context(mutated)
        self.assertFalse(invalid["valid"])
        self.assertIn("context_sha256 does not match content", invalid["errors"])
        extended = json.loads(json.dumps(context))
        extended["unsupported"] = True
        extended["context_sha256"] = dmap_report_model.evidence_context_digest(extended)
        invalid = dmap_report_model.validate_evidence_context(extended)
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("unsupported unsupported" in error for error in invalid["errors"]))

        for required in ("content_digest", "cardinality"):
            with self.subTest(missing_source_field=required):
                incomplete = json.loads(json.dumps(context))
                del incomplete["mechanics_authority"]["source_artifacts"][0][required]
                incomplete["context_sha256"] = dmap_report_model.evidence_context_digest(
                    incomplete
                )
                invalid = dmap_report_model.validate_evidence_context(incomplete)
                self.assertFalse(invalid["valid"])
                self.assertTrue(any(required in error for error in invalid["errors"]))
        empty_cardinality = json.loads(json.dumps(context))
        empty_cardinality["quality_authority"]["source_artifacts"][0]["cardinality"] = {}
        empty_cardinality["context_sha256"] = dmap_report_model.evidence_context_digest(
            empty_cardinality
        )
        invalid = dmap_report_model.validate_evidence_context(empty_cardinality)
        self.assertFalse(invalid["valid"])
        self.assertTrue(any("nonempty object" in error for error in invalid["errors"]))

    def test_evidence_context_captured_candidates_require_exact_run_domain(self) -> None:
        context = make_evidence_context()
        context["mechanics_authority"]["candidate_coverage"][0]["coverage"] = "captured"
        context["quality_authority"]["candidates"][0]["mechanics_coverage"] = "captured"
        context["context_sha256"] = dmap_report_model.evidence_context_digest(context)

        def domain_check(label: str | None, evidence: dict | None = context) -> dict:
            runs = [] if label is None else [{"label": label}]
            result = dmap_report_model.validate_report_model({
                "schema_name": dmap_report_model.SCHEMA_NAME,
                "schema_version": 1,
                "evidence_context": evidence,
                "runs": runs,
                "scenes": [],
                "aggregates": {"regressions": []},
            }, Path("."))
            return next(
                row for row in result["checks"]
                if row["name"] == "evidence_context_captured_candidate_domain"
            )

        self.assertTrue(domain_check("external-variant [deep]")["passed"])
        self.assertTrue(domain_check("external-variant")["passed"])
        wrong_suffix = domain_check("external-variant [deep] extra")
        self.assertFalse(wrong_suffix["passed"])
        self.assertEqual(
            wrong_suffix["detail"]["missing_captured_candidates"],
            ["external-variant"],
        )
        self.assertTrue(domain_check(None, None)["passed"])

    def test_filter_availability_distinguishes_execution_contract_and_unavailability(self) -> None:
        cases = (
            ("executed", "available", True, True, True),
            ("disabled", "disabled", False, False, True),
            ("unavailable", "unavailable", False, False, False),
        )
        for name, artifact_status, executed, expected_execution, expected_contract in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                row = pd.DataFrame([{
                    "artifact_status": artifact_status,
                    "enabled": executed,
                    "executed": executed,
                }])
                model = self.build_availability_model(Path(directory), row, row.copy())
                availability = model["mechanics"]["availability"]

                self.assertEqual(availability["postprocess_filters"], expected_execution)
                self.assertEqual(availability["confidence_adjustment"], expected_execution)
                self.assertEqual(availability["postprocess_filter_contract"], expected_contract)
                self.assertEqual(availability["confidence_adjustment_contract"], expected_contract)

    def test_summary_only_unavailable_signals_are_registered_and_model_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            signal_availability = pd.DataFrame([
                {
                    "run": "summary", "label": "summary", "run_role": "baseline",
                    "repeat": 0, "scene_id": "scene-a", "frame": "0007_frame",
                    "image_id": 7, "signal": "gap_winner_runner_up_exact",
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "role": "logical_state", "logical_iteration": -1,
                    "stage": "initialization", "measurement_quality": "exact",
                    "source_view_index": None, "available": False, "required": False,
                    "availability_reason": "exact_capture_unavailable: exact capture was not requested",
                },
                {
                    "run": "summary", "label": "summary", "run_role": "baseline",
                    "repeat": 0, "scene_id": "scene-a", "frame": "0007_frame",
                    "image_id": 7, "signal": "depth_final_after_filter",
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "role": "final_state", "logical_iteration": None,
                    "stage": "final_state", "measurement_quality": "exact",
                    "source_view_index": None, "available": False, "required": False,
                    "availability_reason": "summary_profile_maps_not_requested",
                },
                {
                    "run": "summary", "label": "summary", "run_role": "baseline",
                    "repeat": 0, "scene_id": "scene-a", "frame": "0007_frame",
                    "image_id": 7, "signal": "view_probability_mass",
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "role": "logical_state", "logical_iteration": -1,
                    "stage": "initialization", "measurement_quality": "exact",
                    "source_view_index": None, "available": False, "required": False,
                    "availability_reason": "exact_capture_unavailable: exact capture was not requested",
                },
                {
                    "run": "summary", "label": "summary", "run_role": "baseline",
                    "repeat": 0, "scene_id": "scene-a", "frame": "0007_frame",
                    "image_id": 7, "signal": "jbu_transfer_depth_delta",
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "role": "logical_state", "logical_iteration": -1,
                    "stage": "initialization", "measurement_quality": "exact",
                    "source_view_index": None, "available": False, "required": False,
                    "availability_reason": "exact_capture_unavailable: exact capture was not requested",
                },
            ])

            model = dmap_report_model.build_report_model(
                config={"name": "summary only"}, experiment_root=root,
                output_dir=output,
                run_scenes=[SimpleNamespace(
                    label="summary", role="baseline", repeat=0, scene_id="scene-a",
                )],
                frames=pd.DataFrame([{
                    "run": "summary", "role": "baseline", "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "safe_image_name": "0007", "image_name": "",
                }]),
                iterations=pd.DataFrame([{
                    "run": "summary", "repeat": 0, "scene_id": "scene-a",
                    "image_id": 7, "logical_iteration": -1,
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "phase": "initialization",
                }]),
                performance=pd.DataFrame(),
                annotations=pd.DataFrame(), comparisons=pd.DataFrame(), gates=[],
                pareto=[], findings=[], map_catalog=pd.DataFrame(),
                signal_availability=signal_availability,
                inventory={"scenes": {}, "overall_plots": []}, metric_specs={},
                instrumentation_validation=pd.DataFrame([{
                    "run": "summary", "repeat": 0, "scene_id": "scene-a",
                    "image_id": 7,
                    "estimation_stage": "photometric", "geometric_iteration": None,
                    "terminal_stage": True, "valid": True,
                    "capture_kind": "summary_only", "maps_available": False,
                    "exact_maps_available": False,
                    "logical_iterations": [-1],
                    "maps_summary_parity_checked": False,
                }]),
                summary_signal_contract={
                    "schema_name": "openmvs.dmap.summary_unavailable_signals",
                    "schema_version": 1,
                    "logical_signals": ["gap_winner_runner_up_exact", "view_probability_mass"],
                    "initialization_signals": ["jbu_transfer_depth_delta"],
                    "final_signals": ["depth_final_after_filter"],
                },
            )

            registry = {
                row["signal_id"]: row
                for row in model["component_registry"]["signals"]
            }
            self.assertIn("gap_winner_runner_up_exact", registry)
            self.assertEqual(registry["view_probability_mass"]["component_id"], "probability_health")
            self.assertEqual(registry["jbu_transfer_depth_delta"]["component_id"], "pyramid_depth_transfer")
            artifact = next(
                row
                for scene in model["scenes"]
                for frame in scene["frames"]
                for row in frame["maps"]
                if row["signal"] == "gap_winner_runner_up_exact"
            )
            self.assertFalse(artifact["available"])
            self.assertEqual(
                artifact["unavailable_reason"],
                "exact_capture_unavailable: exact capture was not requested",
            )
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
            ):
                (output / name).touch()
            validation = dmap_report_model.validate_report_model(model, output)
            self.assertTrue(validation["valid"], validation)
            self.assertFalse(
                model["capture_validation"]["production_parity_qualified"]
            )
            self.assertEqual(
                model["capture_validation"]["production_qualification_status"],
                "unqualified_endpoint_parity",
            )
            self.assertFalse(model["runs"][0]["quality_comparison_eligible"])
            model["capture_validation"]["rows"].append({
                "run": "summary", "repeat": 0, "scene_id": "scene-a",
                "image_id": 7, "capture_kind": "summary_only",
                "estimation_stage": "geometric_consistency", "geometric_iteration": 0,
                "logical_iterations": [-1], "valid": True,
            })
            missing_stage = dmap_report_model.validate_report_model(model, output)
            stage_check = next(
                row for row in missing_stage["checks"]
                if row["name"] == "summary_unavailable_signal_inventory"
            )
            self.assertFalse(stage_check["passed"])
            self.assertTrue(any(
                "missing logical signals" in error
                for error in stage_check["detail"]["errors"]
            ))
            model["capture_validation"]["rows"].pop()
            frame_maps = model["scenes"][0]["frames"][0]["maps"]
            model["scenes"][0]["frames"][0]["maps"] = [
                row for row in frame_maps
                if row["signal"] != "depth_final_after_filter"
            ]
            incomplete = dmap_report_model.validate_report_model(model, output)
            inventory_check = next(
                row for row in incomplete["checks"]
                if row["name"] == "summary_unavailable_signal_inventory"
            )
            self.assertFalse(inventory_check["passed"])

    def test_signed_delta_scale_is_symmetric_about_zero(self) -> None:
        values = np.asarray([[-3.0, -1.0], [0.5, 2.0]], dtype=np.float32)

        low, high = dmap_report_model._local_scale("confidence_final_delta", values)

        self.assertAlmostEqual(low, -high)
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)

    def test_blank_catalog_path_never_resolves_to_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            maps, _signals, _budget = dmap_report_model.build_map_assets(
                pd.DataFrame([{
                    "signal": "cost_stored", "path": "", "available": False,
                    "availability_reason": "summary_profile_maps_not_requested",
                }]),
                pd.DataFrame(),
                output,
            )

            self.assertEqual(len(maps), 1)
            self.assertIsNone(maps[0]["source_path"])
            self.assertFalse(maps[0]["available"])
            self.assertEqual(
                maps[0]["unavailable_reason"],
                "summary_profile_maps_not_requested",
            )

    def test_performance_regression_uses_relative_delta(self) -> None:
        timings = pd.DataFrame([
            {"run": "base", "scene_id": "scene", "endpoint_elapsed_seconds": 10.0},
            {"run": "variant", "scene_id": "scene", "endpoint_elapsed_seconds": 12.0},
        ])
        specs = {"endpoint_elapsed_seconds": SimpleNamespace(level="performance", direction="lower", tolerance=0.05, unit="relative")}

        rows = dmap_report_model._paired_regressions(timings, "base", ["variant"], specs, "performance")

        self.assertAlmostEqual(rows[0]["raw_delta"], 2.0)
        self.assertAlmostEqual(rows[0]["delta"], 0.2)
        self.assertEqual(rows[0]["status"], "regressed")
        self.assertIsNone(rows[0]["image_id"])
        self.assertIsNone(rows[0]["frame_deep_link_id"])
        self.assertEqual(rows[0]["navigation_level"], "scene")

    def test_paired_regressions_skip_unavailable_metric_pairs(self) -> None:
        frames = pd.DataFrame([
            {"run": "base", "scene_id": "scene", "image_id": 0, "cost": None},
            {"run": "variant", "scene_id": "scene", "image_id": 0, "cost": None},
            {"run": "base", "scene_id": "scene", "image_id": 1, "cost": 0.4},
            {"run": "variant", "scene_id": "scene", "image_id": 1, "cost": 0.3},
        ])
        specs = {
            "cost": SimpleNamespace(
                level="frame", direction="lower", tolerance=0.0, unit="cost"
            )
        }

        rows = dmap_report_model._paired_regressions(
            frames, "base", ["variant"], specs, "frame"
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], 1)
        self.assertEqual(rows[0]["status"], "improved")

    def test_pixel_budget_selects_baseline_variant_cohorts_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(6, dtype=np.float32).reshape(2, 3)
            paths = [root / "base.pfm", root / "variant.pfm"]
            for path in paths:
                write_pfm(path, values)
            catalog = pd.DataFrame([
                {"run": run, "repeat": 0, "scene_id": "scene", "frame": "frame", "image_id": 1,
                 "signal": "cost_stored", "logical_iteration": -1, "role": "logical_state",
                 "stage": "initialization", "measurement_quality": "exact", "available": True,
                 "path": str(path), "relative_path": path.name}
                for run, path in zip(("base", "variant"), paths)
            ])

            maps, _signals, budget = dmap_report_model.build_map_assets(
                catalog, pd.DataFrame(), root / "report", pixel_budget_bytes=values.nbytes
            )

            self.assertEqual(budget["selected_artifacts"], 0)
            self.assertTrue(all(not row["pixel_data"]["available"] for row in maps))

    def test_lossless_browser_encoding_round_trips_scalar_and_vector_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scalar = np.asarray([[1.25, -3.5], [np.nan, np.inf]], dtype=np.float32)
            scalar_path = root / "scalar.js"
            scalar_metadata = dmap_report_model._write_pixel_encoding(scalar_path, scalar)
            decoded_scalar = dmap_report_model.decode_pixel_encoding(scalar_path, 1, 0, 1)
            self.assertEqual(scalar_metadata["encoding"], dmap_report_model.PIXEL_ENCODING)
            self.assertEqual(decoded_scalar, [-3.5])

            vector = np.asarray([[[0.25, -0.5, 1.0], [2.0, 4.0, 8.0]]], dtype=np.float32)
            vector_path = root / "vector.js"
            dmap_report_model._write_pixel_encoding(vector_path, vector)
            self.assertEqual(
                dmap_report_model.decode_pixel_encoding(vector_path, 0, 0, 3),
                [0.25, -0.5, 1.0],
            )

    def test_exact_candidate_pngs_are_lossless_numeric_pixel_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = np.asarray([[[0, 255, 1], [9, 7, 3]]], dtype=np.uint8)
            counts = np.asarray([[[1, 0, 1], [13, 3, 2]]], dtype=np.uint8)
            suppression = np.asarray([[[0, 255, 0, 0], [9, 7, 3, 6]]], dtype=np.uint8)
            paths = {
                "candidate_identity_exact": root / "candidate_identity_exact.png",
                "candidate_counts_exact": root / "candidate_counts_exact.png",
                "candidate_raw_suppression_identity_exact": (
                    root / "candidate_raw_suppression_identity_exact.png"
                ),
            }
            Image.fromarray(identity, mode="RGB").save(paths["candidate_identity_exact"])
            Image.fromarray(counts, mode="RGB").save(paths["candidate_counts_exact"])
            Image.fromarray(suppression, mode="RGBA").save(
                paths["candidate_raw_suppression_identity_exact"]
            )
            channels = {
                "candidate_identity_exact": {"R": "winner_slot", "G": "runner_up_slot", "B": "update_source"},
                "candidate_counts_exact": {"R": "tested_count", "G": "finite_count", "B": "accepted_count"},
                "candidate_raw_suppression_identity_exact": {
                    "R": "raw_best_slot", "G": "raw_runner_up_slot",
                    "B": "retained_winner_slot", "A": "suppressed_raw_best_source",
                },
            }
            catalog = pd.DataFrame([{
                "run": "base", "repeat": 0, "scene_id": "scene", "frame": "frame", "image_id": 1,
                "signal": signal, "logical_iteration": 0, "role": "logical_event", "stage": "iteration",
                "measurement_quality": "exact", "available": True, "path": str(path),
                "relative_path": path.name, "channels_json": json.dumps(channels[signal]),
            } for signal, path in paths.items()])
            cost_path = root / "cost_total_production_exact.pfm"
            write_pfm(cost_path, np.asarray([[0.8, 0.4]], dtype=np.float32))
            catalog.loc[len(catalog)] = {
                "run": "base", "repeat": 0, "scene_id": "scene", "frame": "frame", "image_id": 1,
                "signal": "cost_total_production_exact", "logical_iteration": 0,
                "role": "logical_state", "stage": "iteration", "measurement_quality": "exact",
                "available": True, "path": str(cost_path), "relative_path": cost_path.name,
            }

            maps, _signals, budget = dmap_report_model.build_map_assets(
                catalog,
                pd.DataFrame(),
                root / "report",
                pixel_budget_bytes=(identity.size + counts.size + suppression.size) * 4,
            )

            self.assertEqual(budget["selected_artifacts"], 3)
            by_signal = {row["signal"]: row for row in maps}
            self.assertFalse(by_signal["cost_total_production_exact"]["pixel_data"]["available"])
            for signal, expected in (
                ("candidate_identity_exact", [9.0, 7.0, 3.0]),
                ("candidate_counts_exact", [13.0, 3.0, 2.0]),
                ("candidate_raw_suppression_identity_exact", [9.0, 7.0, 3.0, 6.0]),
            ):
                pixel_data = by_signal[signal]["pixel_data"]
                self.assertTrue(pixel_data["available"])
                self.assertEqual(pixel_data["channel_labels"], list(channels[signal].values()))
                payload = root / "report" / pixel_data["script_path"]
                self.assertEqual(
                    dmap_report_model.decode_pixel_encoding(payload, 1, 0, len(expected)), expected
                )

            suppression_artifact = by_signal["candidate_raw_suppression_identity_exact"]
            self.assertEqual(len(suppression_artifact["preview"]["channels"]), 4)
            self.assertNotEqual(
                suppression_artifact["preview"]["local"],
                suppression_artifact["source_path"],
            )
            for preview in suppression_artifact["preview"]["channels"]:
                preview_image = Image.open(root / "report" / preview["local"]).convert("RGBA")
                self.assertEqual(preview_image.getchannel("A").getextrema(), (255, 255))

    def test_map_assets_canonicalize_pyramid_aliases_without_level_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "level0.pfm", root / "level1.pfm"]
            for level, path in enumerate(paths):
                write_pfm(path, np.full((2, 2), level + 0.5, dtype=np.float32))
            catalog = pd.DataFrame([
                {
                    "run": "base", "repeat": 0, "scene_id": "scene", "frame": "frame",
                    "image_id": 1, "signal": "hierarchy_improvement_margin",
                    "logical_iteration": 0, "role": "logical_state", "stage": "iteration",
                    "measurement_quality": "exact", "available": True, "path": str(path),
                    "relative_path": path.name, alias: level,
                }
                for level, (alias, path) in enumerate(zip(("scale_level", "scale_number"), paths))
            ])

            maps, signals, _budget = dmap_report_model.build_map_assets(
                catalog, pd.DataFrame(), root / "report"
            )

            self.assertEqual({row["pyramid_level"] for row in maps}, {0, 1})
            self.assertEqual(len({row["id"] for row in maps}), 2)
            signal = next(row for row in signals if row["name"] == "hierarchy_improvement_margin")
            self.assertEqual(signal["mechanism"], "multiscale")
            self.assertEqual(signal["component_id"], "hierarchy_gate")

    def test_registered_mechanism_status_png_has_categorical_preview_and_exact_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "view_probability_health_status.png"
            Image.fromarray(np.asarray([[0, 1], [2, 3]], dtype=np.uint8), mode="L").save(source)
            catalog = pd.DataFrame([{
                "run": "base", "repeat": 0, "scene_id": "scene", "frame": "frame",
                "image_id": 1, "signal": "view_probability_health_status",
                "logical_iteration": 0, "role": "logical_state", "stage": "iteration",
                "measurement_quality": "exact", "available": True, "path": str(source),
                "relative_path": source.name, "pyramid_level": 0,
            }])

            maps, _signals, _budget = dmap_report_model.build_map_assets(
                catalog, pd.DataFrame(), root / "report"
            )

            artifact = maps[0]
            self.assertEqual(artifact["preview"]["scale_mode"], "categorical_registered_codes")
            self.assertTrue(artifact["pixel_data"]["available"])
            self.assertEqual(
                dmap_report_model.decode_pixel_encoding(
                    root / "report" / artifact["pixel_data"]["script_path"], 1, 1, 1
                ),
                [3.0],
            )

    def test_registered_mechanism_status_pfm_has_legend_and_categorical_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "hierarchy_update_status.pfm"
            write_pfm(source, np.asarray([[0, 1], [1, 0]], dtype=np.float32))
            catalog = pd.DataFrame([{
                "run": "base", "repeat": 0, "scene_id": "scene", "frame": "frame",
                "image_id": 1, "signal": "hierarchy_update_status",
                "logical_iteration": None, "role": "final_state", "stage": "final_state",
                "measurement_quality": "exact", "available": True, "path": str(source),
                "relative_path": source.name, "pyramid_level": 0,
            }])

            maps, _signals, _budget = dmap_report_model.build_map_assets(
                catalog, pd.DataFrame(), root / "report"
            )

            artifact = maps[0]
            self.assertEqual(
                artifact["preview"]["scale_mode"], "categorical_registered_codes"
            )
            self.assertEqual(
                artifact["category_legend"],
                {"0": "retained scale-entry hypothesis", "1": "crossed configured margin"},
            )
            self.assertTrue(artifact["pixel_data"]["available"])

    def test_terminal_frame_embeds_all_stage_and_pyramid_iteration_mechanics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            frames = pd.DataFrame([{
                "run": "summary", "role": "baseline", "repeat": 0,
                "scene_id": "scene-a", "image_id": 7,
                "safe_image_name": "0007", "image_name": "",
                "estimation_stage": "geometric_consistency",
                "geometric_iteration": 3,
                "view_probability_health_schema_valid": True,
                "view_probability_health_requested": True,
                "view_probability_health_available": True,
                "view_probability_health_accounting_valid": True,
                "view_probability_health_unavailable_reason": "",
                "view_probability_health_totals_json": "{}",
            }])
            iterations = pd.DataFrame([
                {
                    "run": "summary", "role": "baseline", "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "estimation_stage": estimation_stage,
                    "geometric_iteration": geometric_iteration,
                    "scale_level": level, "logical_iteration": 0,
                    "stage": "iteration 1",
                    "view_probability_processed": 10,
                    "view_probability_finite_positive_events": 9,
                    "view_probability_degenerate_events": 1,
                }
                for estimation_stage, geometric_iteration, level in (
                    ("photometric", None, 2),
                    ("photometric", None, 1),
                    ("photometric", None, 0),
                    ("geometric_consistency", 3, 0),
                )
            ])
            availability = pd.DataFrame([{
                "run": "summary", "label": "summary",
                "run_role": "baseline", "repeat": 0,
                "scene_id": "scene-a", "frame": "0007", "image_id": 7,
                "estimation_stage": "photometric", "geometric_iteration": None,
                "signal": "cost_stored", "role": "logical_state",
                "logical_iteration": 0, "stage": "iteration",
                "measurement_quality": "exact", "available": False,
                "availability_reason": "summary_profile_maps_not_requested",
                "pyramid_level": 0,
            }])
            model = dmap_report_model.build_report_model(
                config={"name": "terminal frame mechanics"},
                experiment_root=root,
                output_dir=output,
                run_scenes=[SimpleNamespace(
                    label="summary", role="baseline", repeat=0,
                    scene_id="scene-a",
                )],
                frames=frames,
                iterations=iterations,
                performance=pd.DataFrame(),
                annotations=pd.DataFrame(),
                comparisons=pd.DataFrame(),
                gates=[], pareto=[], findings=[],
                map_catalog=pd.DataFrame(),
                signal_availability=availability,
                inventory={"scenes": {}, "overall_plots": []},
                metric_specs={},
            )

            report_frame = model["scenes"][0]["frames"][0]
            embedded = report_frame["run_frames"][0]["iterations"]
            self.assertEqual(report_frame["pyramid_levels"], [0, 1, 2])
            self.assertEqual(
                report_frame["logical_iterations_by_pyramid_level"],
                {"0": [0], "1": [0], "2": [0]},
            )
            self.assertEqual(len(embedded), 4)
            self.assertEqual(
                {(row["estimation_stage"], row["geometric_iteration"], row["pyramid_level"])
                 for row in embedded},
                {
                    ("photometric", None, 2),
                    ("photometric", None, 1),
                    ("photometric", None, 0),
                    ("geometric_consistency", 3, 0),
                },
            )
            self.assertEqual(
                len({row["deep_link_id"] for row in embedded}), len(embedded)
            )
            self.assertEqual(
                set(report_frame["map_groups"]["by_pyramid_level"]), {"0"}
            )
            self.assertFalse(report_frame["maps"][0]["available"])
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
            ):
                (output / name).touch()
            validation = dmap_report_model.validate_report_model(model, output)
            self.assertTrue(validation["valid"], validation)

    def test_report_model_nulls_only_unavailable_legacy_candidate_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            frames = pd.DataFrame([
                {
                    "run": run, "role": role, "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "safe_image_name": "0007", "image_name": "",
                    "estimation_stage": "photometric",
                    "geometric_iteration": None,
                    "candidate_accounting_mode": mode,
                }
                for run, role, mode in (
                    (
                        "legacy", "baseline",
                        "unavailable_post_pass_snapshot",
                    ),
                    (
                        "deep", "variant",
                        "exact_production_hot_kernel",
                    ),
                )
            ])
            frames["candidate_spatial_tested"] = [0, 100]
            frames["candidate_spatial_finite"] = [0, 90]
            frames["candidate_spatial_accepted"] = [0, 20]
            frames["candidate_spatial_acceptance_rate"] = [0.0, 0.2]
            iterations = pd.DataFrame([
                {
                    "run": run, "role": role, "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "estimation_stage": "photometric",
                    "geometric_iteration": None, "scale_level": 0,
                    "logical_iteration": 0, "stage": "iteration 1",
                    "candidate_accounting_mode": mode,
                    "changed_ratio": changed,
                    "candidates_tested": tested,
                    "candidates_finite": finite,
                    "candidates_accepted": accepted,
                    "acceptance_rate": rate,
                }
                for run, role, mode, changed, tested, finite, accepted, rate in (
                    (
                        "legacy", "baseline",
                        "unavailable_post_pass_snapshot", 0.0, 0, 0, 0, 0.0,
                    ),
                    (
                        "deep", "variant", "exact_production_hot_kernel",
                        0.25, 100, 90, 20, 0.2,
                    ),
                )
            ])
            model = dmap_report_model.build_report_model(
                config={"name": "accounting availability"},
                experiment_root=root,
                output_dir=output,
                run_scenes=[
                    SimpleNamespace(
                        label="legacy", role="baseline", repeat=0,
                        scene_id="scene-a",
                    ),
                    SimpleNamespace(
                        label="deep", role="variant", repeat=0,
                        scene_id="scene-a",
                    ),
                ],
                frames=frames,
                iterations=iterations,
                performance=pd.DataFrame(), annotations=pd.DataFrame(),
                comparisons=pd.DataFrame(), gates=[], pareto=[], findings=[],
                map_catalog=pd.DataFrame(), signal_availability=pd.DataFrame(),
                inventory={"scenes": {}, "overall_plots": []},
                metric_specs={},
                exact_iterations=pd.DataFrame([{
                    "run": "deep", "repeat": 0, "scene_id": "scene-a",
                    "image_id": 7, "logical_iteration": 0,
                    "tested_candidates": 100, "finite_candidates": 90,
                    "accepted_candidates": 20,
                }]),
            )

            report_frame = model["scenes"][0]["frames"][0]
            self.assertEqual(
                model["contract"]["candidate_accounting_contract"]["presentation"],
                "null and explicitly labeled unavailable; never zero-filled",
            )
            by_run = {row["run"]: row for row in report_frame["run_frames"]}
            legacy = by_run["legacy"]["iterations"][0]
            exact = by_run["deep"]["iterations"][0]
            self.assertEqual(
                by_run["legacy"]["candidate_accounting_mode"],
                "unavailable_post_pass_snapshot",
            )
            for key in dmap_report_model.CANDIDATE_ACCOUNTING_METRICS:
                self.assertIn(key, legacy)
                self.assertIsNone(legacy[key], key)
                self.assertIn(key, by_run["legacy"]["metrics"])
                self.assertIsNone(by_run["legacy"]["metrics"][key], key)
            for key in (
                "candidate_spatial_tested", "candidate_spatial_finite",
                "candidate_spatial_accepted",
                "candidate_spatial_acceptance_rate",
            ):
                self.assertIsNone(by_run["legacy"]["metrics"][key], key)
            self.assertEqual(by_run["deep"]["metrics"]["candidate_spatial_tested"], 100)
            self.assertEqual(exact["changed_ratio"], 0.25)
            self.assertEqual(exact["candidates_tested"], 100)
            self.assertEqual(exact["candidates_finite"], 90)
            self.assertEqual(exact["candidates_accepted"], 20)
            self.assertEqual(
                model["mechanics"]["exact_iterations"][0]["tested_candidates"],
                100,
            )
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
            ):
                (output / name).touch()
            validation = dmap_report_model.validate_report_model(model, output)
            self.assertTrue(validation["valid"], validation)

            legacy["changed_ratio"] = 0.0
            invalid = dmap_report_model.validate_report_model(model, output)
            accounting_check = next(
                row for row in invalid["checks"]
                if row["name"] == "candidate_accounting_availability"
            )
            self.assertFalse(accounting_check["passed"])
            self.assertIn("changed_ratio", accounting_check["detail"]["errors"][0])

    def test_model_contains_runs_frames_iterations_maps_unavailable_and_regressions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports"
            output.mkdir()
            image = root / "reference.png"
            Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8), mode="RGB").save(image)
            baseline_map = root / "baseline.pfm"
            variant_map = root / "variant.pfm"
            write_pfm(baseline_map, np.asarray([[0.8, 0.7, 0.6], [0.5, 0.4, 0.3]], dtype=np.float32))
            write_pfm(variant_map, np.asarray([[0.7, 0.6, 0.5], [0.4, 0.3, 0.2]], dtype=np.float32))

            frames = pd.DataFrame([
                {"run": "base", "role": "baseline", "repeat": 0, "scene_id": "scene-a", "image_id": 7,
                 "image_name": str(image), "safe_image_name": "0007", "valid_ratio_after_filter": 0.90,
                 "endpoint_valid_depth_coverage": 0.88,
                 "endpoint_valid_depth_coverage_status": "available",
                 "endpoint_valid_depth_coverage_source": str(baseline_map)},
                {"run": "variant", "role": "variant", "repeat": 0, "scene_id": "scene-a", "image_id": 7,
                 "image_name": str(image), "safe_image_name": "0007", "valid_ratio_after_filter": 0.80,
                 "endpoint_valid_depth_coverage": 0.75,
                 "endpoint_valid_depth_coverage_status": "available",
                 "endpoint_valid_depth_coverage_source": str(variant_map)},
            ])
            iterations = pd.DataFrame([
                {"run": run, "role": role, "repeat": 0, "scene_id": "scene-a", "image_id": 7,
                 "logical_iteration": -1, "stage": "initialization", "mean_cost": cost}
                for run, role, cost in (("base", "baseline", 0.8), ("variant", "variant", 0.7))
            ])
            catalog = pd.DataFrame([
                {"run": run, "label": run, "run_role": role, "repeat": 0, "scene_id": "scene-a",
                 "frame": "0007", "image_id": 7, "signal": "cost_stored", "role": "logical_state",
                 "logical_iteration": -1, "stage": "initialization", "measurement_quality": "exact",
                 "measurement_basis": "production_cost_snapshot", "available": True, "exists": True,
                 "path": str(path), "relative_path": path.name, "scale_level": 0}
                for run, role, path in (("base", "baseline", baseline_map), ("variant", "variant", variant_map))
            ])
            availability = pd.DataFrame([{
                "run": "variant", "label": "variant", "run_role": "variant", "repeat": 0,
                "scene_id": "scene-a", "frame": "0007", "image_id": 7,
                "signal": "gap_local_neighbor_equal_selected_rescore_proxy", "role": "logical_state",
                "logical_iteration": -1, "stage": "initialization", "measurement_quality": "proxy",
                "available": False, "availability_reason": "not_declared_in_manifest",
                "scale_number": 0,
            }])
            specs = {
                "valid_ratio_after_filter": SimpleNamespace(
                    level="frame", direction="higher", tolerance=0.005, unit="fraction"
                ),
                "endpoint_elapsed_seconds": SimpleNamespace(
                    level="performance", direction="lower", tolerance=0.05, unit="relative"
                ),
                "kernel_ms": SimpleNamespace(
                    level="performance", direction="lower", tolerance=0.05, unit="relative"
                ),
            }
            run_scenes = [
                SimpleNamespace(label="base", role="baseline", repeat=0, scene_id="scene-a"),
                SimpleNamespace(label="variant", role="variant", repeat=0, scene_id="scene-a"),
            ]
            model = dmap_report_model.build_report_model(
                config={"name": "test report", "_config_path": str(root / "config.yaml")},
                experiment_root=root,
                output_dir=output,
                run_scenes=run_scenes,
                frames=frames,
                iterations=iterations,
                performance=pd.DataFrame([
                    {"run": "base", "scene_id": "scene-a", "image_id": 7, "kernel_ms": 10.0},
                    {"run": "variant", "scene_id": "scene-a", "image_id": 7, "kernel_ms": 30.0},
                ]),
                endpoint_performance=pd.DataFrame([
                    {"run": "base", "scene_id": "scene-a", "endpoint_elapsed_seconds": 10.0},
                    {"run": "variant", "scene_id": "scene-a", "endpoint_elapsed_seconds": 12.0},
                ]),
                annotations=pd.DataFrame(),
                comparisons=pd.DataFrame(),
                accuracy_ledger=pd.DataFrame([{
                    "candidate": "variant", "accuracy_rank": 1,
                    "noise_class": "regressed", "availability_biased": False,
                }]),
                accuracy_evidence=pd.DataFrame([{
                    "candidate": "variant", "scene_id": "scene-a",
                    "metric": "all_residual_p95_m", "normalized_regression_loss": 2.0,
                }]),
                model_stability=pd.DataFrame([{
                    "candidate": "variant", "scene_id": "scene-a",
                    "large_model_switch": True,
                    "near_candidate_regression": True,
                    "baseline_model_on_candidate_status": "available",
                }]),
                gates=[], pareto=[], findings=[],
                map_catalog=catalog,
                signal_availability=availability,
                inventory={"scenes": {}, "overall_plots": []},
                metric_specs=specs,
                instrumentation_validation=pd.DataFrame([
                    {
                        "run": run,
                        "repeat": 0,
                        "scene_id": "scene-a",
                        "terminal_stage": True,
                        "valid": True,
                        "endpoint_dmap_set_checked": True,
                        "endpoint_dmap_set_bit_exact": True,
                        "endpoint_dmap_set_shared_count": 9,
                        "maps_summary_parity_checked": True,
                        "quality_comparison_eligible": True,
                    }
                    for run in ("base", "variant")
                ]),
            )

            self.assertEqual(model["schema_name"], dmap_report_model.SCHEMA_NAME)
            self.assertEqual(model["schema_version"], 3)
            self.assertFalse(model["contract"]["capture_profile_contract"]["light"]["available"])
            self.assertFalse(model["contract"]["capture_profile_contract"]["summary"]["compute_light"])
            self.assertEqual(model["capture_validation"]["endpoint_dmap_sets_bit_exact"], 2)
            self.assertEqual(model["capture_validation"]["endpoint_dmaps_shared"], 18)
            self.assertEqual(
                model["component_registry"]["schema_name"],
                "openmvs.dmap.component_registry",
            )
            registry = {
                row["signal_id"]: row
                for row in model["component_registry"]["signals"]
            }
            self.assertEqual(registry["cost_stored"]["mechanism"], "cost")
            self.assertTrue(model["aggregates"]["mechanism_impact"])
            self.assertEqual(model["aggregates"]["accuracy_ledger"][0]["accuracy_rank"], 1)
            self.assertEqual(model["aggregates"]["accuracy_evidence"][0]["scene_id"], "scene-a")
            self.assertEqual(
                model["aggregates"]["annotation_model_switch_summary"]["large_switches"],
                1,
            )
            guide = model["investigation_guide"]
            self.assertEqual(guide["schema_name"], "openmvs.dmap.investigation_guide")
            self.assertEqual(
                {recipe["key"] for recipe in guide["recipes"]},
                {"overview", "cost", "propagation", "view_selection", "texture", "patch", "deformable_patch", "multiscale"},
            )
            self.assertEqual(model["contract"]["optional_extensions"], [])
            self.assertEqual(len(model["runs"]), 2)
            report_frame = model["scenes"][0]["frames"][0]
            self.assertTrue(report_frame["reference"]["available"])
            self.assertFalse(report_frame["reference"]["path"].startswith(".."))
            report_reference = output / report_frame["reference"]["path"]
            self.assertTrue(report_reference.is_file())
            with Image.open(report_reference) as thumbnail:
                self.assertEqual(thumbnail.size, (3, 2))
            reference_signal = next(
                row for row in model["signals"] if row["name"] == "reference_rgb"
            )
            self.assertEqual(reference_signal["available_artifacts"], 1)
            self.assertEqual(reference_signal["unavailable_artifacts"], 0)
            ownership = next(
                row for row in dmap_report_model.validate_report_model(model, output)["checks"]
                if row["name"] == "reference_images_report_owned"
            )
            self.assertTrue(ownership["passed"], ownership)
            report_frame["reference"]["path"] = "../reference.png"
            outside = next(
                row for row in dmap_report_model.validate_report_model(model, output)["checks"]
                if row["name"] == "reference_images_report_owned"
            )
            self.assertFalse(outside["passed"])
            self.assertEqual(
                report_frame["run_frames"][0]["endpoint_valid_depth_coverage"]["value"],
                0.88,
            )
            self.assertEqual(report_frame["logical_iterations"], [-1])
            self.assertEqual(report_frame["pyramid_levels"], [0])
            self.assertEqual(
                report_frame["logical_iterations_by_pyramid_level"],
                {"0": [-1], "unspecified": [-1]},
            )
            self.assertIn("0", report_frame["map_groups"]["by_pyramid_level"])
            self.assertTrue(any(item["available"] for item in report_frame["maps"]))
            self.assertTrue(any(not item["available"] for item in report_frame["maps"]))
            regressions = model["aggregates"]["regressions"]
            self.assertTrue(any(row["status"] == "regressed" for row in regressions))
            self.assertIn("endpoint_elapsed_seconds", {row["metric"] for row in regressions})
            self.assertNotIn("kernel_ms", {row["metric"] for row in regressions})
            self.assertEqual(
                model["aggregates"]["runtime"]["authority"],
                "production_endpoint_wall_clock",
            )
            self.assertEqual(
                model["aggregates"]["runtime"]["observer_timing_authority"],
                "diagnostic_only",
            )
            exact_maps = [item for item in report_frame["maps"] if item.get("pixel_data", {}).get("available")]
            self.assertEqual(len(exact_maps), 2)
            self.assertTrue((output / exact_maps[0]["pixel_data"]["script_path"]).is_file())

    def test_diagnostic_runs_remain_inspectable_but_are_excluded_from_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            run_scenes = [
                SimpleNamespace(
                    label="base", role="baseline", repeat=0, scene_id="scene-a",
                    diagnostic_only=False, diagnostic_only_reason="",
                ),
                SimpleNamespace(
                    label="variant", role="variant", repeat=0, scene_id="scene-a",
                    diagnostic_only=False, diagnostic_only_reason="",
                ),
                SimpleNamespace(
                    label="base [deep]", role="baseline", repeat=0,
                    scene_id="scene-a", diagnostic_only=True,
                    diagnostic_only_reason="Process<true> mechanics capture",
                ),
                SimpleNamespace(
                    label="variant [deep]", role="variant", repeat=0,
                    scene_id="scene-a", diagnostic_only=True,
                    diagnostic_only_reason="Process<true> mechanics capture",
                ),
            ]
            frames = pd.DataFrame([
                {
                    "run": run, "role": role, "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "safe_image_name": "0007", "image_name": "",
                    "valid_ratio_after_filter": value,
                }
                for run, role, value in (
                    ("base", "baseline", 0.9),
                    ("variant", "variant", 0.8),
                    ("base [deep]", "baseline", 0.2),
                    ("variant [deep]", "variant", 0.1),
                )
            ])
            annotations = pd.DataFrame([
                {
                    "run": run, "repeat": 0, "scene_id": "scene-a",
                    "image_id": 7, "stage": "post_filter", "fit_status": "ok",
                }
                for run in ("variant", "variant [deep]")
            ])
            comparisons = pd.DataFrame([
                {"candidate": "variant", "metric": "coverage", "delta": -0.1},
                {"candidate": "variant [deep]", "metric": "coverage", "delta": -0.8},
            ])
            candidate_rows = [
                {"candidate": "variant", "marker": "quality"},
                {"candidate": "variant [deep]", "marker": "diagnostic"},
            ]
            model = dmap_report_model.build_report_model(
                config={"name": "diagnostic isolation"},
                experiment_root=root,
                output_dir=output,
                run_scenes=run_scenes,
                frames=frames,
                iterations=pd.DataFrame(),
                performance=pd.DataFrame(),
                annotations=annotations,
                comparisons=comparisons,
                accuracy_ledger=pd.DataFrame(candidate_rows),
                accuracy_evidence=pd.DataFrame(candidate_rows),
                gates=candidate_rows,
                pareto=[
                    {"candidate": "variant", "pareto": False,
                     "dominated_by": ["variant [deep]"]},
                    {"candidate": "variant [deep]", "pareto": True,
                     "dominated_by": []},
                ],
                findings=candidate_rows,
                map_catalog=pd.DataFrame(),
                signal_availability=pd.DataFrame(),
                inventory={"scenes": {}, "overall_plots": []},
                metric_specs={
                    "valid_ratio_after_filter": SimpleNamespace(
                        level="frame", direction="higher", tolerance=0.005,
                        unit="fraction",
                    )
                },
                exact_iterations=pd.DataFrame([{
                    "run": "variant [deep]", "repeat": 0,
                    "scene_id": "scene-a", "image_id": 7,
                    "logical_iteration": 0, "tested_candidates": 12,
                }]),
                instrumentation_validation=pd.DataFrame([
                    {
                        "run": run,
                        "repeat": 0,
                        "scene_id": "scene-a",
                        "terminal_stage": True,
                        "diagnostic_only": False,
                        "valid": True,
                        "quality_comparison_eligible": True,
                    }
                    for run in ("base", "variant")
                ]),
                evidence_context=make_evidence_context(),
            )

            runs = {row["label"]: row for row in model["runs"]}
            self.assertTrue(runs["variant [deep]"]["diagnostic_only"])
            self.assertEqual(
                runs["variant [deep]"]["diagnostic_only_reason"],
                "Process<true> mechanics capture",
            )
            self.assertFalse(
                runs["variant [deep]"]["quality_comparison_eligible"]
            )
            self.assertTrue(runs["variant"]["quality_comparison_eligible"])

            aggregates = model["aggregates"]
            for key in (
                "accuracy_ledger", "accuracy_evidence", "gates", "pareto",
                "findings", "comparisons", "regressions",
            ):
                self.assertEqual(
                    {row.get("candidate") for row in aggregates[key]},
                    {"variant"},
                    key,
                )
            self.assertEqual(aggregates["pareto"][0]["dominated_by"], [])
            self.assertEqual(
                {row["run"] for row in model["mechanics"]["exact_iterations"]},
                {"variant [deep]"},
            )
            self.assertEqual(
                model["evidence_context"]["quality_authority"]["candidates"][0]["candidate"],
                "external-variant",
            )
            self.assertNotIn(
                "external-variant",
                {row.get("candidate") for row in aggregates["accuracy_ledger"]},
            )
            report_frame = model["scenes"][0]["frames"][0]
            self.assertIn(
                "variant [deep]",
                {row["run"] for row in report_frame["run_frames"]},
            )
            self.assertNotIn(
                "variant [deep]",
                {row["run"] for row in report_frame["annotations"]},
            )
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
            ):
                (output / name).write_text("test\n", encoding="utf-8")
            validation = dmap_report_model.validate_report_model(model, output)
            self.assertTrue(validation["valid"], validation)

    def test_investigation_html_is_self_contained_and_links_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "02_investigation.html"
            ui = Path(__file__).resolve().parents[1] / "dmap_report_ui"
            model = {
                "schema_name": dmap_report_model.SCHEMA_NAME, "schema_version": 1,
                "experiment": {"name": "test"}, "runs": [], "scenes": [], "signals": [],
                "aggregates": {"gates": [], "regressions": []},
                "map_catalog_summary": {"artifacts": 0, "available": 0, "unavailable": 0,
                                        "pixel_data": {"selected_artifacts": 0, "eligible_artifacts": 0, "omitted_artifacts": 0}},
            }
            dmap_report_model.render_investigation_html(
                output, model, ui / "investigation.html", ui / "investigation.css", ui / "investigation.js"
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("01_development_report.md", text)
            self.assertIn("report_model.json", text)
            self.assertIn("openmvs.dmap.development_report", text)
            self.assertIn('id="source-view-select"', text)
            self.assertIn('id="channel-select"', text)
            self.assertIn('id="capture-stage-select"', text)
            self.assertIn('id="pyramid-level-select"', text)
            self.assertIn('id="alignment-select"', text)
            self.assertIn('id="map-preset-select"', text)
            self.assertIn('id="mechanism-select"', text)
            self.assertIn('id="component-select"', text)
            self.assertIn('id="mechanism-impact"', text)
            self.assertIn('id="accuracy-ledger-table"', text)
            self.assertIn("renderAccuracyLedger", text)
            self.assertIn('id="evidence-context-section"', text)
            self.assertIn("renderEvidenceContext", text)
            self.assertIn("external quality rows are content-attested summaries", text.lower())
            self.assertIn("row.endpoint_valid_depth_coverage_delta", text)
            self.assertIn('id="deep-frame-request"', text)
            self.assertIn('id="trace-pixel-request"', text)
            self.assertIn('id="drilldown-detail"', text)
            self.assertIn("renderCompletedTrace", text)
            self.assertIn("full-frame Process&lt;true&gt; maps rerun", text)
            self.assertIn("classified per row", text)
            self.assertIn("valid schema-v4 exact-map completion evidence", text)
            self.assertIn('id="annotation-stage-select"', text)
            self.assertIn('id="annotation-kind-select"', text)
            self.assertIn('id="guide-open"', text)
            self.assertIn('id="guide-dialog"', text)
            self.assertIn("Investigation guide", text)
            self.assertIn("applyGuideRecipe", text)
            self.assertIn("|| Object.keys(action).length > 0", text)
            self.assertIn('state.mapPreset = "custom"', text)
            self.assertIn("Sequential filtering", text)
            self.assertIn("Final state per run", text)
            self.assertIn("pyramidLevelOf", text)
            self.assertIn("pyramidLevelMatches", text)
            self.assertIn("pickAlignedArtifact", text)
            self.assertIn("View probability health", text)
            self.assertIn("Arithmetic subtraction is not meaningful for categorical status codes", text)
            self.assertIn("category_legend", text)
            self.assertIn("Pyramid level selects algorithm state", text)
            self.assertIn("renderCaptureProfileCoverage();", text)
            self.assertIn("renderMechanics(); renderMaps();", text)
            self.assertIn("Exact cost, candidate, and view decisions", text)
            self.assertIn("Delta: variant - baseline", text)
            self.assertIn("renderDeltaCanvas", text)
            self.assertIn("exportDrilldownRequest", text)
            self.assertIn("3D self-consistency proxy", text)
            self.assertIn("annotation-summary-grid", text)
            self.assertIn("data-annotation-structure", text)
            self.assertIn("row.run === run", text)
            self.assertIn("rowRepeat === Number(repeat)", text)
            self.assertIn("effective_inlier_coverage_20mm", text)
            self.assertIn("formatPercentagePoints", text)
            self.assertIn("formatMillimetres", text)
            self.assertIn('img[data-src]', text)
            self.assertIn("const COLUMN_HELP", text)
            self.assertIn("installColumnTooltips", text)
            self.assertIn('id = "column-tooltip"', text)

            self.assertIn('button[data-sort]', text)
            self.assertIn("percentage points", text)
            self.assertIn("Direction-corrected raw delta", text)
            self.assertIn("Dimensionless worst primary scene/metric loss", text)
            self.assertIn("diagnostic mechanics only", text)
            self.assertIn("const qualityRuns", text)
            self.assertIn(
                "Diagnostic-only cohorts are excluded from automatic quality regression ranking",
                text,
            )
            self.assertIn(
                "Annotation quality comparisons exclude diagnostic-only",
                text,
            )
            for column in (
                "Rank", "Noise", "Availability", "Worst loss / tolerance",
                "Effective coverage", "Estimator validity", "Terminal endpoint validity",
                "Regression score", "Delta", "Quality",
                "Effective @20 mm", "Threshold AUC", "Residual P95",
                "Gap P50", "Score", "Mean |depth delta|", "Device MiB",
            ):
                self.assertIn(f'"{column}":', text)
            self.assertNotIn("annotation-grid", text)
            self.assertNotIn("Raw passes:", text)
            self.assertNotIn("__DMAP_REPORT_", text)
            self.assertNotIn("<script src=", text)

    def test_guide_recipe_keys_keep_patch_actions_distinct(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is unavailable")
        source = (
            Path(__file__).resolve().parents[1]
            / "dmap_report_ui"
            / "investigation.js"
        ).read_text(encoding="utf-8")
        functions = source.split("function normalizeGuideRecipeKey", 1)[1].split(
            "function guideItemParts", 1
        )[0]
        script = f"""
const guideRecipeExactKeys = new Set([
  "patch", "deformable_patch", "low_texture_update_hysteresis"
]);
function normalizeGuideRecipeKey{functions}
const keys = [
  guideRecipeKey("patch"),
  guideRecipeKey("deformable_patch"),
  guideRecipeKey("low_texture_update_hysteresis"),
];
if (JSON.stringify(keys) !== JSON.stringify([
  "patch", "deformable_patch", "low_texture_update_hysteresis"
])) process.exit(1);
const actions = new Map(keys.map((key, index) => [key, index]));
if (actions.size !== 3 || actions.get("patch") === actions.get("deformable_patch")) process.exit(2);
if (guideRecipeKey("Debug patch scoring") !== "patch") process.exit(3);
"""
        result = subprocess.run(
            [node, "-e", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

        render = source.split("function renderGuide()", 1)[1].split(
            "function renderGuideContext", 1
        )[0]
        self.assertLess(
            render.index("guideRecipeExactKeys.add(normalizeGuideRecipeKey(rawKey))"),
            render.index("recipes.map((recipe, index) => guideSectionHtml"),
        )

    def test_ui_links_only_report_owned_evidence(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is unavailable")
        source = (
            Path(__file__).resolve().parents[1]
            / "dmap_report_ui" / "investigation.js"
        ).read_text(encoding="utf-8")
        functions = source.split("function safeEvidenceHref", 1)[1].split(
            "function captureEvidenceLinks", 1
        )[0]
        script = f"""
function escapeHtml(value) {{ return String(value); }}
function safeEvidenceHref{functions}
if (safeEvidenceHref("../captures/raw.json")) process.exit(1);
if (safeEvidenceHref("a/%2e%2e/raw.json")) process.exit(2);
if (safeEvidenceHref("/tmp/raw.json")) process.exit(3);
if (safeEvidenceHref("interactive/maps/map.png") !== "interactive/maps/map.png") process.exit(4);
if (evidenceReference("../captures/raw.json", "raw").includes("<a")) process.exit(5);
if (!evidenceReference("interactive/maps/map.png", "map").includes("<a")) process.exit(6);
"""
        result = subprocess.run(
            [node, "-e", script], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_raster_previews_are_materialized_inside_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            source = root / "capture" / "mask.png"
            source.parent.mkdir()
            Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(source)
            catalog = pd.DataFrame([{
                "run": "base", "label": "base", "run_role": "baseline",
                "repeat": 0, "scene_id": "scene-a", "frame": "0001",
                "image_id": 1, "signal": "valid_after_filter",
                "role": "final_state", "logical_iteration": None,
                "measurement_quality": "exact", "measurement_basis": "fixture",
                "available": True, "exists": True, "path": str(source),
                "relative_path": "maps/mask.png",
            }])

            maps, _signals, _budget = dmap_report_model.build_map_assets(
                catalog, pd.DataFrame(), output
            )

            preview = maps[0]["preview"]["local"]
            self.assertNotIn("..", Path(preview).parts)
            self.assertTrue((output / preview).is_file())
            self.assertEqual(maps[0]["preview"]["scale_mode"], "report_owned_source_raster")

    def test_ui_prefers_available_terminal_artifacts_without_same_iteration_fallback(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "dmap_report_ui" / "investigation.js"
        ).read_text(encoding="utf-8")
        function = source.split("function pickAlignedArtifact", 1)[1].split(
            "function selectedArtifact", 1
        )[0]

        self.assertLess(
            function.index('preferred.find((map) => map.logical_iteration == null && map.available)'),
            function.index("finalMaps[0]"),
        )
        self.assertIn(
            "return logical.find((map) => map.available) || logical[0] || null;",
            function,
        )
        self.assertNotIn("logical[0] || preferred.find", function)

    def test_nested_provenance_paths_are_report_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "04_report"
            output.mkdir(parents=True)
            source = root / "runs" / "base" / "run_metadata.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}\n", encoding="utf-8")
            inventory = {
                "overall_plots": [],
                "scenes": {},
                "data_artifacts": [{
                    "name": "instrumentation_validation",
                    "warnings": [{"parameter_source": {"path": str(source)}}],
                }],
            }

            portable = dmap_report_model._portable_inventory(inventory, output)
            path = portable["data_artifacts"][0]["warnings"][0][
                "parameter_source"
            ]["path"]

            self.assertFalse(Path(path).is_absolute())
            self.assertEqual((output / path).resolve(), source.resolve())

            policy = dmap_report_model.paths_report_relative({
                "source_config": {"path": str(root / "config.yaml")},
                "resolved_experiment": {"path": str(root / "resolved.yaml")},
            }, output)
            self.assertFalse(Path(policy["source_config"]["path"]).is_absolute())
            self.assertFalse(Path(policy["resolved_experiment"]["path"]).is_absolute())

    def test_capture_declared_component_is_rendered_without_filename_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report"
            output.mkdir()
            source = root / "custom.pfm"
            write_pfm(source, np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
            catalog = pd.DataFrame([{
                "run": "variant",
                "label": "variant",
                "run_role": "variant",
                "repeat": 0,
                "scene_id": "scene-a",
                "frame": "0007",
                "image_id": 7,
                "signal": "experimental_flatness_penalty",
                "signal_label": "Flatness penalty",
                "component_id": "flatness",
                "mechanism": "texture",
                "quantity": "contribution",
                "units": "cost",
                "preferred_direction": "lower",
                "minimum_profile": "light",
                "measurement_kind": "map",
                "colormap": "magma",
                "default_visible": True,
                "role": "logical_state",
                "logical_iteration": 0,
                "stage": "iteration",
                "measurement_quality": "exact",
                "measurement_basis": "test fixture",
                "available": True,
                "exists": True,
                "path": str(source),
                "relative_path": source.name,
            }])

            maps, signals, _budget = dmap_report_model.build_map_assets(
                catalog,
                pd.DataFrame(),
                output,
            )

            custom_map = next(row for row in maps if row["signal"] == "experimental_flatness_penalty")
            custom_signal = next(row for row in signals if row["name"] == "experimental_flatness_penalty")
            self.assertEqual(custom_map["mechanism"], "texture")
            self.assertEqual(custom_map["component_id"], "flatness")
            self.assertEqual(custom_signal["label"], "Flatness penalty")
            self.assertTrue(custom_signal["default"])
            self.assertTrue((output / custom_map["preview"]["shared"]).is_file())

    def test_drilldown_index_is_embedded_with_report_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            request = root / "drilldowns" / "requests" / ("a" * 64 + ".yaml")
            request.parent.mkdir(parents=True)
            request.write_text("schema_version: 1\n", encoding="utf-8")
            index = root / "drilldowns" / "index.json"
            index.write_text(json.dumps({
                "schema_name": "openmvs.dmap.drilldown_index",
                "schema_version": 1,
                "entries": [{
                    "request_sha256": "a" * 64,
                    "capture_profile": "trace",
                    "scene_id": "scene-a",
                    "image_id": 7,
                    "trace_pixel_count": 1,
                    "status": "requested",
                    "request": str(request.relative_to(root)),
                }],
            }), encoding="utf-8")

            embedded = dmap_report_model.load_drilldown_index(root, output)

            self.assertTrue(embedded["available"])
            self.assertEqual(embedded["entries"][0]["status"], "requested")
            self.assertTrue(embedded["entries"][0]["request"].startswith("../../drilldowns/"))

    def test_completed_trace_drilldown_embeds_normalized_logical_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            write_completed_trace_drilldown(root, [trace_row(-1), trace_row(0)])

            embedded = dmap_report_model.load_drilldown_index(root, output)

            entry = embedded["entries"][0]
            execution_metadata = entry["execution_metadata"]
            trace_data = entry["trace_data"]
            self.assertTrue(execution_metadata["available"])
            self.assertEqual(
                execution_metadata["schema_name"],
                "openmvs.dmap.drilldown_executions",
            )
            self.assertTrue(execution_metadata["source_path"].startswith("../../drilldowns/"))
            self.assertEqual(trace_data["schema_name"], "openmvs.dmap.completed_trace_rows")
            self.assertEqual(trace_data["schema_version"], 2)
            self.assertTrue(trace_data["available"])
            self.assertTrue(trace_data["complete"])
            self.assertEqual(trace_data["row_count"], 2)
            self.assertEqual(trace_data["source_count"], 1)
            self.assertTrue(trace_data["sources"][0]["contained"])
            self.assertTrue(trace_data["sources"][0]["source_path"].startswith("../../drilldowns/"))
            initialization, iteration = trace_data["rows"]
            self.assertEqual(initialization["logical_iteration"], -1)
            self.assertEqual(initialization["stage"], "initialization")
            self.assertEqual(iteration["logical_iteration"], 0)
            self.assertEqual(iteration["display_iteration"], 1)
            self.assertEqual(iteration["cost"]["after"], 0.6)
            self.assertEqual(iteration["depth"]["after"], 1.1)
            self.assertEqual(iteration["normal"]["angle_change_degrees"], 2.5)
            self.assertEqual(iteration["view"]["selected_mask"], 7)
            self.assertNotIn("raw_pass_indices", iteration["arrays"])
            self.assertEqual(iteration["source"], "propagation")
            self.assertEqual(iteration["source_quality"], "exact")
            self.assertEqual(iteration["measurement_basis"], "exact_hot_kernel")
            self.assertTrue(trace_data["all_source_attribution_exact"])
            self.assertTrue(
                trace_data["sources"][0]["exact_capture_evidence"]["valid"]
            )
            self.assertTrue(iteration["trace_source_path"].startswith("../../drilldowns/"))
            validation = self.validate_drilldowns(embedded, output)
            self.assertTrue(validation["valid"], validation)

    def test_completed_legacy_debug_trace_keeps_proxy_source_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            legacy = trace_row(0)
            legacy.pop("source_quality")
            legacy.pop("measurement_basis")
            write_completed_trace_drilldown(root, [legacy], exact_capture=False)

            embedded = dmap_report_model.load_drilldown_index(root, output)

            trace_data = embedded["entries"][0]["trace_data"]
            self.assertFalse(trace_data["all_source_attribution_exact"])
            self.assertEqual(trace_data["source_quality_counts"], {"exact": 0, "proxy": 1})
            self.assertEqual(trace_data["rows"][0]["source_quality"], "proxy")
            self.assertEqual(trace_data["rows"][0]["measurement_basis"], "post_pass_proxy")
            self.assertEqual(trace_data["rows"][0]["source_quality_origin"], "legacy_default")
            self.assertFalse(trace_data["sources"][0]["exact_capture_evidence"]["valid"])
            self.assertTrue(self.validate_drilldowns(embedded, output)["valid"])

    def test_completed_trace_reports_parse_errors_and_validator_rejects_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            _request_sha256, traces = write_completed_trace_drilldown(
                root, [trace_row(-1)]
            )
            with traces.open("a", encoding="utf-8") as handle:
                handle.write("{not-json}\n")

            embedded = dmap_report_model.load_drilldown_index(root, output)

            trace_data = embedded["entries"][0]["trace_data"]
            self.assertTrue(trace_data["available"])
            self.assertFalse(trace_data["complete"])
            self.assertEqual(trace_data["row_count"], 1)
            self.assertEqual(trace_data["errors"][0]["kind"], "trace_parse_error")
            self.assertIn("line 2", trace_data["errors"][0]["message"])
            validation = self.validate_drilldowns(embedded, output)
            self.assertFalse(validation["valid"])
            check = next(
                row for row in validation["checks"] if row["name"] == "drilldown_index"
            )
            self.assertFalse(check["passed"])

    def test_completed_trace_rejects_execution_identity_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            request_sha256, _traces = write_completed_trace_drilldown(
                root, [trace_row(-1)]
            )
            executions = (
                root / "drilldowns" / "captures" / request_sha256 / "executions.json"
            )
            value = json.loads(executions.read_text(encoding="utf-8"))
            value["executions"][0]["run"] = "../../outside"
            executions.write_text(json.dumps(value), encoding="utf-8")

            embedded = dmap_report_model.load_drilldown_index(root, output)

            entry = embedded["entries"][0]
            self.assertFalse(entry["trace_data"]["available"])
            self.assertTrue(
                any(
                    error["kind"] == "execution_metadata_error"
                    for error in entry["embedding_errors"]
                ),
                entry["embedding_errors"],
            )
            self.assertTrue(entry["embedding_errors"])
            self.assertFalse(self.validate_drilldowns(embedded, output)["valid"])

    def test_completed_trace_requires_every_requested_logical_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            write_completed_trace_drilldown(root, [trace_row(0)])

            embedded = dmap_report_model.load_drilldown_index(root, output)

            trace_data = embedded["entries"][0]["trace_data"]
            self.assertTrue(trace_data["available"])
            self.assertFalse(trace_data["complete"])
            coverage_error = next(
                error for error in trace_data["errors"]
                if error["kind"] == "trace_coverage_error"
            )
            self.assertEqual(len(coverage_error["missing_states"]), 1)
            self.assertEqual(
                coverage_error["missing_states"][0]["logical_iteration"], -1
            )
            self.assertFalse(self.validate_drilldowns(embedded, output)["valid"])

    def test_completed_trace_does_not_infer_exact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            rows = [trace_row(-1), trace_row(0)]
            for row in rows:
                row.pop("source_quality")
                row.pop("measurement_basis")
            write_completed_trace_drilldown(root, rows)

            embedded = dmap_report_model.load_drilldown_index(root, output)

            trace_data = embedded["entries"][0]["trace_data"]
            self.assertTrue(trace_data["complete"])
            self.assertEqual(trace_data["source_quality_counts"], {"exact": 0, "proxy": 2})
            self.assertTrue(
                trace_data["sources"][0]["exact_capture_evidence"]["valid"]
            )
            self.assertEqual(
                {row["source_quality_origin"] for row in trace_data["rows"]},
                {"legacy_default"},
            )

    def test_completed_trace_binds_capture_to_immutable_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            request_sha256, _traces = write_completed_trace_drilldown(
                root, [trace_row(-1), trace_row(0)]
            )
            captured_request = (
                root / "drilldowns" / "captures" / request_sha256 / "request.yaml"
            )
            captured_request.write_text("different: true\n", encoding="utf-8")

            embedded = dmap_report_model.load_drilldown_index(root, output)

            entry = embedded["entries"][0]
            self.assertTrue(any(
                error["kind"] == "capture_request_binding_error"
                for error in entry["embedding_errors"]
            ))
            self.assertFalse(self.validate_drilldowns(embedded, output)["valid"])

    def test_completed_trace_row_limit_is_explicit_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dmap_report_model, "MAX_COMPLETED_TRACE_ROWS", 2
        ):
            root = Path(directory)
            output = root / "reports" / "03_master"
            output.mkdir(parents=True)
            write_completed_trace_drilldown(
                root,
                [trace_row(-1), trace_row(0), trace_row(1)],
            )

            embedded = dmap_report_model.load_drilldown_index(root, output)

            trace_data = embedded["entries"][0]["trace_data"]
            self.assertEqual(trace_data["row_limit"], 2)
            self.assertEqual(trace_data["row_count"], 2)
            self.assertTrue(trace_data["truncated"])
            self.assertFalse(trace_data["complete"])
            self.assertEqual(trace_data["errors"], [])
            self.assertTrue(self.validate_drilldowns(embedded, output)["valid"])

    def test_low_texture_hysteresis_metrics_use_declared_denominators(self) -> None:
        rows = pd.DataFrame([
            {
                "run": "variant", "repeat": 0, "scene_id": "scene-a", "image_id": 7,
                "logical_iteration": -1, "low_texture_gate_eligible": 100,
            },
            {
                "run": "variant", "repeat": 0, "scene_id": "scene-a", "image_id": 7,
                "logical_iteration": 1, "scale_number": 0,
                "low_texture_gate_eligible": 100,
                "low_texture_propagation_accepted": 30,
                "low_texture_propagation_rejected": 10,
                "low_texture_refinement_accepted": 20,
                "low_texture_refinement_rejected": 40,
                "low_texture_required_gain_sum": 0.05,
                "low_texture_best_proposed_gain_sum": 0.08,
            },
        ])

        result = dmap_report_model.build_low_texture_update_hysteresis_metrics(rows)

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["pyramid_level"], 0)
        self.assertEqual(row["eligible_pixels"], 100)
        self.assertAlmostEqual(row["propagation_rejection_rate"], 0.25)
        self.assertAlmostEqual(row["refinement_rejection_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(row["proposal_rejection_rate"], 0.5)
        self.assertAlmostEqual(row["mean_required_gain"], 0.0005)
        self.assertAlmostEqual(row["mean_best_proposed_gain"], 0.0008)

    def test_model_validator_rejects_checkerboard_rows_and_missing_unavailable_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("01_development_report.md", "01_development_report.html", "02_investigation.html", "report_model.json"):
                (root / name).touch()
            model = {
                "schema_name": dmap_report_model.SCHEMA_NAME,
                "schema_version": dmap_report_model.SCHEMA_VERSION,
                "entrypoints": {
                    "markdown": "01_development_report.md", "static_html": "01_development_report.html",
                    "investigation_html": "02_investigation.html", "model": "report_model.json",
                },
                "runs": [{"deep_link_id": "run-1"}],
                "scenes": [{
                    "deep_link_id": "scene-1",
                    "frames": [{
                        "id": "frame-1", "deep_link_id": "frame-1", "reference": {"available": False},
                        "run_frames": [{"iterations": [{"deep_link_id": "iter-red", "phase": "red"}]}],
                        "maps": [{"id": "map-1", "deep_link_id": "map-1", "available": False}],
                    }],
                }],
                "aggregates": {"regressions": []},
            }

            result = dmap_report_model.validate_report_model(model, root)
            failed = {row["name"] for row in result["checks"] if not row["passed"]}
            guide_check = next(row for row in result["checks"] if row["name"] == "investigation_guide")

            self.assertFalse(result["valid"])
            self.assertFalse(guide_check["passed"])
            self.assertEqual(
                set(guide_check["detail"]["missing_recipe_keys"]),
                {"overview", "cost", "propagation", "view_selection", "texture", "patch", "deformable_patch", "multiscale"},
            )
            self.assertIn("logical_iterations_only", failed)
            self.assertIn("unavailable_maps_explicit", failed)

            model["schema_version"] = 1
            legacy = dmap_report_model.validate_report_model(model, root)
            legacy_guide = next(row for row in legacy["checks"] if row["name"] == "investigation_guide")
            self.assertTrue(legacy_guide["passed"])
            self.assertFalse(legacy_guide["detail"]["required"])


if __name__ == "__main__":
    unittest.main()
