#!/usr/bin/env python3
"""Focused schema-v3 tests for the depth-map instrumentation validator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_dmap_instrumentation as validator


WIDTH = 2
HEIGHT = 2
NUM_ITERATIONS = 1
NUM_PASSES = 1 + 2 * NUM_ITERATIONS


def write_pfm(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "PF" if values.ndim == 3 else "Pf"
    with path.open("wb") as handle:
        handle.write(f"{header}\n{values.shape[1]} {values.shape[0]}\n-1.0\n".encode("ascii"))
        handle.write(np.flipud(values).astype("<f4", copy=False).tobytes())


def write_map(frame_dir: Path, relative_path: str, values: np.ndarray) -> Path:
    path = frame_dir / relative_path
    if path.suffix == ".png":
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)
    else:
        write_pfm(path, values)
    return path


def add_map(
    frame_dir: Path,
    manifest: dict,
    signal: str,
    relative_path: str,
    values: np.ndarray,
    dtype: str = "float32",
    **metadata,
) -> dict:
    path = write_map(frame_dir, relative_path, values)
    entry = {
        "signal": signal,
        "path": relative_path,
        "dtype": dtype,
        "semantics": f"test map for {signal}",
        "bytes": path.stat().st_size,
        **metadata,
    }
    if metadata.get("role") in {"logical_state", "logical_event"}:
        entry["uncompressed_bytes"] = WIDTH * HEIGHT * 4
    manifest["maps"].append(entry)
    return entry


def logical_values(iteration: int) -> dict[str, np.ndarray]:
    if iteration == -1:
        cost_stored = 0.625
        photo_prior = 0.25
        geometric = 0.125
    else:
        cost_stored = 0.5
        photo_prior = 0.125
        geometric = 0.125
    total = photo_prior + geometric
    scalar = lambda value: np.full((HEIGHT, WIDTH), value, dtype=np.float32)
    return {
        "cost_stored": scalar(cost_stored),
        "confidence_stored": scalar(max(1.0 - cost_stored, 0.0)),
        "cost_photo_raw_equal_selected_rescore_proxy": scalar(photo_prior + 0.125),
        "cost_photo_prior_equal_selected_rescore_proxy": scalar(photo_prior),
        "cost_geometric_equal_selected_rescore_proxy": scalar(geometric),
        "cost_total_equal_selected_rescore_proxy": scalar(total),
        "cost_stored_minus_rescore": scalar(cost_stored - total),
        "depth_prior_disagreement_equal_selected_rescore_proxy": scalar(0.125),
        "depth_prior_weight_equal_selected_rescore_proxy": scalar(0.25),
        "gap_local_neighbor_equal_selected_rescore_proxy": scalar(0.125),
        "reference_variance_equal_selected_rescore_proxy": scalar(0.25),
    }


def make_fixture(frame_dir: Path, schema_version: int) -> tuple[dict, dict]:
    scalar = lambda value: np.full((HEIGHT, WIDTH), value, dtype=np.float32)
    manifest: dict = {
        "schema_version": schema_version,
        "width": WIDTH,
        "height": HEIGHT,
        "num_passes": NUM_PASSES,
        "measurement_model": (
            validator.V3_MEASUREMENT_MODEL
            if schema_version == 3
            else "production PatchMatch plus post-pass snapshot diagnostics"
        ),
        "maps": [],
    }
    summary = {
        "schema_version": schema_version,
        "width": WIDTH,
        "height": HEIGHT,
        "candidate_accounting_mode": "unavailable_post_pass_snapshot",
        "confidence_gap_mode": "post_pass_current_plus_eight_neighbors",
        "num_pixels_total": WIDTH * HEIGHT,
        "num_valid_before_filter": WIDTH * HEIGHT,
        "num_invalid_before_filter": 0,
        "num_valid_after_filter": WIDTH * HEIGHT - 1,
        "num_rejected_by_filter": 1,
        "valid_ratio_after_filter": 0.75,
        "candidate_acceptance": [],
    }

    final_maps = {
        "depth_final_before_filter": scalar(1.0),
        "normal_final_before_filter": np.dstack([scalar(0.0), scalar(0.0), scalar(1.0)]),
        "cost_final_before_filter": scalar(0.5),
        "cost_photometric": scalar(0.25),
        "cost_photo_prior": scalar(0.25),
        "cost_geometric": scalar(0.25),
        "cost_total_components": scalar(0.5),
        "confidence_gap": scalar(0.125),
        "reference_variance": scalar(0.25),
        "view_entropy": scalar(0.5),
    }
    for signal, values in final_maps.items():
        dtype = "float32x3" if values.ndim == 3 else "float32"
        add_map(frame_dir, manifest, signal, f"maps/{signal}.pfm", values, dtype=dtype)
    for signal in ("selected_view_count", "accepted_update_count"):
        add_map(
            frame_dir,
            manifest,
            signal,
            f"maps/{signal}.png",
            np.ones((HEIGHT, WIDTH), dtype=np.uint8),
            dtype="uint8",
        )

    if schema_version == 2:
        for signal in ("depth_delta", "depth_relative_delta", "normal_angle_delta", "view_churn"):
            for pass_index in range(NUM_PASSES):
                suffix = "png" if signal == "view_churn" else "pfm"
                add_map(
                    frame_dir,
                    manifest,
                    signal,
                    f"pass_maps/pass{pass_index:02d}_{signal}.{suffix}",
                    np.zeros((HEIGHT, WIDTH), dtype=np.uint8 if suffix == "png" else np.float32),
                    dtype="uint8" if suffix == "png" else "float32",
                    pass_index=pass_index,
                )

    if schema_version == 3:
        manifest.update({
            "schema_name": validator.V3_SCHEMA_NAME,
            "num_iterations": NUM_ITERATIONS,
            "num_logical_states": NUM_ITERATIONS + 1,
            "map_granularity": "logical_iteration",
        })
        for iteration in (-1, 0):
            stage = "initialization" if iteration == -1 else "iteration"
            directory = "initialization" if iteration == -1 else f"iteration_{iteration:02d}"
            for signal, values in logical_values(iteration).items():
                quality = validator.V3_MEASUREMENT_QUALITY[signal]
                if signal == "cost_stored":
                    basis = "production_cost_snapshot"
                elif signal == "confidence_stored":
                    basis = "max(1-cost,0)"
                elif signal in validator.V3_PROXY_LOGICAL_STATE_SIGNALS:
                    basis = validator.V3_PROXY_MEASUREMENT_BASIS
                else:
                    basis = "production_cost_snapshot_minus_equal_selected_view_binary_post_pass_rescore"
                metadata = {
                    "role": "logical_state",
                    "logical_iteration": iteration,
                    "stage": stage,
                    "measurement_quality": quality,
                    "measurement_basis": basis,
                }
                if signal in validator.V3_PROXY_LOGICAL_STATE_SIGNALS:
                    metadata.update({
                        "proxy_target": signal.removesuffix("_equal_selected_rescore_proxy"),
                        "limitations": "Post-pass equal-selected-view rescore; not a hot-kernel exact value.",
                    })
                add_map(
                    frame_dir,
                    manifest,
                    signal,
                    f"logical_states/{directory}/{signal}.pfm",
                    values,
                    **metadata,
                )
            for signal in validator.V3_REQUIRED_LOGICAL_EVENT_SIGNALS:
                suffix = "png" if signal == "view_churn" else "pfm"
                add_map(
                    frame_dir,
                    manifest,
                    signal,
                    f"logical_states/{directory}/{signal}.{suffix}",
                    np.zeros(
                        (HEIGHT, WIDTH),
                        dtype=np.uint8 if signal == "view_churn" else np.float32,
                    ),
                    dtype="uint8" if signal == "view_churn" else "float32",
                    role="logical_event",
                    logical_iteration=iteration,
                    stage=stage,
                    measurement_quality="exact",
                    measurement_basis="post_pass_production_state_difference",
                )
        manifest.update({
            "complete": True,
            "expected_map_count": len(manifest["maps"]),
            "written_map_count": len(manifest["maps"]),
            "write_errors": [],
        })

    (frame_dir / "map_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return manifest, summary


def save_manifest(frame_dir: Path, manifest: dict) -> None:
    (frame_dir / "map_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_run_metadata(frame_dir: Path, decay_scale: float = 0.02) -> Path:
    path = frame_dir.parent.parent / "run_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "schema_name": "openmvs.dmap.run",
            "schema_version": 4,
            "cuda_patchmatch_parameters": {
                "low_texture_decay_scale": decay_scale,
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def legacy_prior_weight(reference_variance: float, decay_scale: float = 0.02) -> np.float32:
    exponent = np.float32(reference_variance) * (
        np.float32(-1.0) / np.float32(decay_scale)
    )
    return np.float32(np.exp(np.float64(exponent)))


def find_logical_entry(manifest: dict, signal: str, iteration: int) -> dict:
    return next(
        entry for entry in manifest["maps"]
        if entry.get("signal") == signal and entry.get("logical_iteration") == iteration
    )


def rewrite_entry(frame_dir: Path, entry: dict, values: np.ndarray) -> None:
    path = write_map(frame_dir, entry["path"], values)
    entry["bytes"] = path.stat().st_size


def check_by_name(result: dict, name: str) -> dict:
    return next(check for check in result["checks"] if check["name"] == name)


class DMapInstrumentationSchemaV3Tests(unittest.TestCase):
    def validate_fixture(self, frame_dir: Path) -> dict:
        return validator.validate(validator.Arguments(frame_dir=frame_dir))

    def run_validator_cli(
        self, frame_dir: Path, output: Path
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(validator.__file__).resolve()),
                "--frame-dir",
                str(frame_dir),
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        def reject_nonstandard_constant(value: str) -> None:
            raise ValueError(f"non-standard JSON constant: {value}")

        stdout_result = json.loads(
            completed.stdout, parse_constant=reject_nonstandard_constant
        )
        output_result = json.loads(
            output.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
        )
        self.assertEqual(stdout_result, output_result)
        return completed, stdout_result

    def test_cli_emits_machine_readable_failure_for_invalid_map_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "frame"
            manifest, _summary = make_fixture(frame_dir, 3)
            total = next(
                entry
                for entry in manifest["maps"]
                if entry["signal"] == "cost_total_components"
            )
            rewrite_entry(
                frame_dir, total, np.full((1, WIDTH), 0.5, dtype=np.float32)
            )
            save_manifest(frame_dir, manifest)

            completed, result = self.run_validator_cli(
                frame_dir, Path(directory) / "validation.json"
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["component_total_max_abs_residual"])
            self.assertFalse(check_by_name(result, "map_shapes")["passed"])
            replaced = result["machine_readable_diagnostics"][
                "nonfinite_values_represented_as_null"
            ]
            self.assertIn("$.component_total_max_abs_residual", replaced)

    def test_cli_emits_machine_readable_failure_for_nonfinite_map_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "frame"
            manifest, _summary = make_fixture(frame_dir, 3)
            total = next(
                entry
                for entry in manifest["maps"]
                if entry["signal"] == "cost_total_components"
            )
            rewrite_entry(
                frame_dir,
                total,
                np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32),
            )
            save_manifest(frame_dir, manifest)

            completed, result = self.run_validator_cli(
                frame_dir, Path(directory) / "validation.json"
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["component_total_max_abs_residual"])
            component_check = check_by_name(result, "component_reconstruction")
            self.assertFalse(component_check["passed"])
            self.assertIsNone(component_check["detail"]["max_abs"])
            replaced = result["machine_readable_diagnostics"][
                "nonfinite_values_represented_as_null"
            ]
            self.assertTrue(
                any(path.endswith(".detail.max_abs") for path in replaced),
                replaced,
            )

    def test_schema_v2_fixture_remains_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            make_fixture(frame_dir, 2)

            result = self.validate_fixture(frame_dir)

            self.assertTrue(result["valid"], result)
            self.assertEqual(result["schema_version"], 2)
            self.assertNotIn("logical_state_validation", result)

    def test_schema_v3_accepts_nonzero_stored_vs_proxy_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            make_fixture(frame_dir, 3)

            result = self.validate_fixture(frame_dir)

            self.assertTrue(result["valid"], result)
            self.assertNotIn("raw_map_capture_complete", {row["name"] for row in result["checks"]})
            diagnostic = result["logical_state_validation"]["stored_vs_proxy_max_abs_diagnostic_only"]
            self.assertGreater(diagnostic["-1"], 0.0)
            self.assertGreater(diagnostic["0"], 0.0)

    def test_schema_v3_requires_unique_complete_logical_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            missing = find_logical_entry(manifest, "depth_delta", 0)
            manifest["maps"].remove(missing)
            duplicate = dict(find_logical_entry(manifest, "normal_angle_delta", -1))
            manifest["maps"].append(duplicate)
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(result["valid"])
            self.assertFalse(check_by_name(result, "v3_logical_event_coverage")["passed"])
            self.assertFalse(check_by_name(result, "v3_logical_event_unique")["passed"])

    def test_schema_v3_rejects_event_phase_and_nonzero_initialization_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            depth = find_logical_entry(manifest, "depth_delta", -1)
            depth["checkerboard_phase"] = "black"
            rewrite_entry(frame_dir, depth, np.ones((HEIGHT, WIDTH), dtype=np.float32))
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(result["valid"])
            self.assertFalse(check_by_name(result, "v3_logical_event_phase_free")["passed"])
            self.assertFalse(
                check_by_name(result, "v3_logical_event_initialization_zero")["passed"]
            )
            self.assertFalse(check_by_name(result, "initialization_delta_zero")["passed"])

    def test_schema_v3_rejects_missing_and_duplicate_logical_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            missing = find_logical_entry(manifest, "cost_stored", 0)
            manifest["maps"].remove(missing)
            duplicate = dict(find_logical_entry(manifest, "confidence_stored", -1))
            manifest["maps"].append(duplicate)
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(result["valid"])
            self.assertFalse(check_by_name(result, "v3_logical_state_coverage")["passed"])
            self.assertFalse(check_by_name(result, "v3_logical_state_unique")["passed"])

    def test_schema_v3_rejects_proxy_exactness_and_phase_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            entry = find_logical_entry(
                manifest, "gap_local_neighbor_equal_selected_rescore_proxy", 0
            )
            entry["measurement_quality"] = "exact"
            entry["phase"] = "black"
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(check_by_name(result, "v3_logical_state_metadata")["passed"])
            self.assertFalse(check_by_name(result, "v3_logical_state_phase_free")["passed"])

    def test_schema_v3_rejects_confidence_gap_and_weight_domain_violations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            bad_values = {
                "confidence_stored": 1.25,
                "gap_local_neighbor_equal_selected_rescore_proxy": -0.5,
                "depth_prior_weight_equal_selected_rescore_proxy": -0.25,
            }
            for signal, value in bad_values.items():
                entry = find_logical_entry(manifest, signal, 0)
                rewrite_entry(frame_dir, entry, np.full((HEIGHT, WIDTH), value, dtype=np.float32))
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(check_by_name(result, "v3_confidence_domain")["passed"])
            self.assertFalse(check_by_name(result, "v3_gap_domain")["passed"])
            self.assertFalse(check_by_name(result, "v3_depth_prior_weight_domain")["passed"])

    def test_schema_v3_warns_for_formula_backed_legacy_prior_weight_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
            manifest, _summary = make_fixture(frame_dir, 3)
            variance = np.float32(-4.7447193e-6)
            weight = legacy_prior_weight(float(variance))
            self.assertGreater(float(weight), 1.0)
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "reference_variance_equal_selected_rescore_proxy", 0
                ),
                np.full((HEIGHT, WIDTH), variance, dtype=np.float32),
            )
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "depth_prior_weight_equal_selected_rescore_proxy", 0
                ),
                np.full((HEIGHT, WIDTH), weight, dtype=np.float32),
            )
            write_run_metadata(frame_dir)
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            domain = check_by_name(result, "v3_depth_prior_weight_domain")
            self.assertTrue(result["valid"], result)
            self.assertTrue(domain["passed"], domain)
            self.assertEqual(domain["detail"]["invalid_domain_pixels"], WIDTH * HEIGHT)
            self.assertEqual(domain["detail"]["fatal_invalid_domain_pixels"], 0)
            self.assertEqual(
                domain["detail"]["known_production_warning_pixels"], WIDTH * HEIGHT
            )
            self.assertEqual(len(result["warnings"]), 1)
            self.assertEqual(
                result["warnings"][0]["code"],
                validator.NEGATIVE_VARIANCE_PRIOR_WEIGHT_WARNING,
            )
            self.assertEqual(
                result["warnings"][0]["signal"],
                "depth_prior_weight_equal_selected_rescore_proxy",
            )

    def test_schema_v3_rejects_legacy_overshoot_without_decay_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
            manifest, _summary = make_fixture(frame_dir, 3)
            variance = np.float32(-4.7447193e-6)
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "reference_variance_equal_selected_rescore_proxy", 0
                ),
                np.full((HEIGHT, WIDTH), variance, dtype=np.float32),
            )
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "depth_prior_weight_equal_selected_rescore_proxy", 0
                ),
                np.full(
                    (HEIGHT, WIDTH), legacy_prior_weight(float(variance)), dtype=np.float32
                ),
            )
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            domain = check_by_name(result, "v3_depth_prior_weight_domain")
            self.assertFalse(result["valid"])
            self.assertFalse(domain["passed"])
            self.assertEqual(domain["detail"]["fatal_invalid_domain_pixels"], WIDTH * HEIGHT)
            self.assertEqual(result["warnings"], [])

    def test_schema_v3_rejects_unexplained_positive_prior_weight_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
            manifest, _summary = make_fixture(frame_dir, 3)
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "reference_variance_equal_selected_rescore_proxy", 0
                ),
                np.full((HEIGHT, WIDTH), -4.7447193e-6, dtype=np.float32),
            )
            rewrite_entry(
                frame_dir,
                find_logical_entry(
                    manifest, "depth_prior_weight_equal_selected_rescore_proxy", 0
                ),
                np.full((HEIGHT, WIDTH), 1.01, dtype=np.float32),
            )
            write_run_metadata(frame_dir)
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            domain = check_by_name(result, "v3_depth_prior_weight_domain")
            self.assertFalse(result["valid"])
            self.assertFalse(domain["passed"])
            self.assertEqual(domain["detail"]["fatal_invalid_domain_pixels"], WIDTH * HEIGHT)
            self.assertEqual(result["warnings"], [])

    def test_schema_v3_validates_proxy_closure_without_equating_stored_cost(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            total_entry = find_logical_entry(
                manifest, "cost_total_equal_selected_rescore_proxy", 0
            )
            residual_entry = find_logical_entry(manifest, "cost_stored_minus_rescore", 0)
            rewrite_entry(frame_dir, total_entry, np.full((HEIGHT, WIDTH), 0.375, dtype=np.float32))
            rewrite_entry(frame_dir, residual_entry, np.full((HEIGHT, WIDTH), 0.125, dtype=np.float32))
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(check_by_name(result, "v3_proxy_component_closure")["passed"])
            self.assertTrue(check_by_name(result, "v3_stored_minus_rescore_definition")["passed"])

    def test_schema_v3_allows_float32_component_roundoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            for iteration, reconstructed in ((-1, 0.375), (0, 0.25)):
                total_entry = find_logical_entry(
                    manifest, "cost_total_equal_selected_rescore_proxy", iteration
                )
                rewrite_entry(
                    frame_dir,
                    total_entry,
                    np.full(
                        (HEIGHT, WIDTH), reconstructed + 5.0e-7, dtype=np.float32
                    ),
                )
            final_total = next(
                entry
                for entry in manifest["maps"]
                if entry["signal"] == "cost_total_components"
            )
            rewrite_entry(
                frame_dir,
                final_total,
                np.full((HEIGHT, WIDTH), 0.5000005, dtype=np.float32),
            )
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertTrue(check_by_name(result, "v3_proxy_component_closure")["passed"])
            self.assertTrue(check_by_name(result, "component_reconstruction")["passed"])

    def test_schema_v3_rejects_incorrect_residual_definition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            residual_entry = find_logical_entry(manifest, "cost_stored_minus_rescore", 0)
            rewrite_entry(frame_dir, residual_entry, np.zeros((HEIGHT, WIDTH), dtype=np.float32))
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(check_by_name(result, "v3_stored_minus_rescore_definition")["passed"])

    def test_schema_v3_rejects_bad_layout_bytes_and_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            layout_entry = find_logical_entry(manifest, "reference_variance_equal_selected_rescore_proxy", 0)
            layout_entry["dtype"] = "float64"
            layout_entry["uncompressed_bytes"] = 1
            missing_entry = find_logical_entry(manifest, "cost_photo_raw_equal_selected_rescore_proxy", 0)
            missing_entry["path"] = "logical_states/iteration_00/missing.pfm"
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(check_by_name(result, "map_paths")["passed"])
            self.assertFalse(check_by_name(result, "v3_logical_state_layout")["passed"])
            self.assertFalse(check_by_name(result, "v3_logical_state_bytes")["passed"])

    def test_schema_v3_rejects_incomplete_manifest_and_unindexed_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            manifest, _summary = make_fixture(frame_dir, 3)
            manifest["complete"] = False
            manifest["write_errors"] = ["cost_stored"]
            write_pfm(frame_dir / "logical_states" / "stale_unindexed.pfm", np.zeros((HEIGHT, WIDTH)))
            save_manifest(frame_dir, manifest)

            result = self.validate_fixture(frame_dir)

            self.assertFalse(result["valid"])
            self.assertFalse(check_by_name(result, "v3_manifest_complete")["passed"])
            self.assertFalse(check_by_name(result, "v3_manifest_indexes_all_map_files")["passed"])


if __name__ == "__main__":
    unittest.main()
