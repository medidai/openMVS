from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.python import dmap_dev


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class MapCatalogTest(unittest.TestCase):
    def make_scene(self, root: Path, manifest: dict[str, object]) -> tuple[dmap_dev.RunScene, Path]:
        instrumentation = root / "instrumentation"
        frame = instrumentation / "depthmaps" / "0007_frame"
        write_json(frame / "summary.json", {
            "schema_version": manifest.get("schema_version", 0),
            "image_id": 7,
            "image_name": "images/0007.jpg",
            "safe_image_name": "frame",
        })
        write_json(frame / "map_manifest.json", manifest)
        scene = dmap_dev.RunScene(
            label="variant",
            role="variant",
            repeat=2,
            scene_id="scene-a",
            instrumentation_dir=instrumentation,
            depth_map_dir=None,
            timing_dir=None,
        )
        return scene, frame

    def test_schema3_catalog_and_required_signal_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = [
                {
                    "signal": "cost_stored",
                    "path": "logical_states/state00_initialization/cost_stored.pfm",
                    "dtype": "float32",
                    "role": "logical_state",
                    "logical_iteration": -1,
                    "stage": "initialization",
                    "measurement_quality": "exact",
                    "measurement_basis": "production_cost",
                    "bytes": 16,
                },
                {
                    "signal": "cost_stored",
                    "path": "logical_states/state01_iteration01/cost_stored.pfm",
                    "dtype": "float32",
                    "role": "logical_state",
                    "logical_iteration": 0,
                    "stage": "iteration",
                    "measurement_quality": "exact",
                    "measurement_basis": "production_cost",
                    "bytes": 16,
                },
                {
                    "signal": "cost_total_equal_selected_rescore_proxy",
                    "path": "logical_states/state01_iteration01/cost_total_proxy.pfm",
                    "dtype": "float32",
                    "role": "logical_state",
                    "logical_iteration": 0,
                    "stage": "iteration",
                    "measurement_quality": "proxy",
                    "measurement_basis": "equal_selected_view_binary_post_pass_rescore",
                    "proxy_target": "production score",
                    "limitations": "equal view weights",
                    "bytes": 16,
                },
                {
                    "signal": "depth_delta",
                    "path": "logical_states/state01_iteration01/depth_delta.pfm",
                    "dtype": "float32",
                    "role": "logical_event",
                    "logical_iteration": 0,
                    "stage": "iteration",
                    "measurement_quality": "exact",
                    "measurement_basis": "accepted update",
                    "bytes": 16,
                },
            ]
            scene, frame = self.make_scene(root, {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 3,
                "num_iterations": 1,
                "num_logical_states": 2,
                "map_granularity": "logical_iteration",
                "maps": entries,
            })
            for entry in (entries[0], entries[2], entries[3]):
                path = frame / str(entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"0123456789abcdef")

            catalog, availability = dmap_dev.build_map_catalog([scene])

            self.assertEqual(len(catalog), 4)
            self.assertEqual(set(catalog["run"]), {"variant"})
            self.assertEqual(set(catalog["label"]), {"variant"})
            self.assertEqual(set(catalog["repeat"]), {2})
            self.assertEqual(set(catalog["scene_id"]), {"scene-a"})
            self.assertEqual(set(catalog["frame"]), {"0007_frame"})
            self.assertEqual(
                len(availability),
                (
                    len(dmap_dev.REQUIRED_LOGICAL_STATE_SIGNALS)
                    + len(dmap_dev.MECHANISM_LOGICAL_STATE_SIGNALS)
                ) * 2,
            )

            initialization = availability[
                (availability["signal"] == "cost_stored") & (availability["logical_iteration"] == -1)
            ].iloc[0]
            self.assertTrue(initialization["available"])
            self.assertEqual(initialization["measurement_quality"], "exact")
            self.assertEqual(initialization["availability_reason"], "available")

            missing_file = availability[
                (availability["signal"] == "cost_stored") & (availability["logical_iteration"] == 0)
            ].iloc[0]
            self.assertFalse(missing_file["available"])
            self.assertEqual(missing_file["availability_reason"], "declared_artifact_missing")

            undeclared = availability[
                (availability["signal"] == "gap_local_neighbor_equal_selected_rescore_proxy")
                & (availability["logical_iteration"] == 0)
            ].iloc[0]
            self.assertFalse(undeclared["available"])
            self.assertEqual(undeclared["measurement_quality"], "proxy")
            self.assertEqual(undeclared["availability_reason"], "not_declared_in_manifest")

            self.assertTrue(
                availability[
                    availability["signal"].isin(
                        dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
                    )
                ].empty
            )

    def test_schema2_is_cataloged_without_v3_completeness_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "signal": "cost_final",
                "path": "maps/cost_final.pfm",
                "dtype": "float32",
                "semantics": "legacy final cost",
            }
            scene, frame = self.make_scene(root, {"schema_version": 2, "maps": [entry]})
            path = frame / str(entry["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

            catalog, availability = dmap_dev.build_map_catalog([scene])

            self.assertEqual(len(catalog), 1)
            self.assertTrue(catalog.iloc[0]["exists"])
            self.assertTrue(availability.empty)
            self.assertEqual(list(availability.columns), list(dmap_dev.SIGNAL_AVAILABILITY_COLUMNS))

    def test_summary_only_catalogs_registered_maps_as_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instrumentation = root / "instrumentation"
            frame = instrumentation / "depthmaps" / "0007_frame"
            write_json(frame / "summary.json", {
                "schema_name": "openmvs.dmap.frame_summary",
                "schema_version": 4,
                "image_id": 7,
                "image_name": "images/0007.jpg",
                "safe_image_name": "frame",
                "scale_level": 0,
                "resource_plan": {
                    "summary_available": True,
                    "maps_requested": False,
                    "maps_available": False,
                    "exact_available": False,
                    "exact_unavailable_reason": "exact capture was not requested",
                },
            })
            write_json(frame / "summary_complete.json", {
                "schema_name": "openmvs.dmap.summary_complete",
                "schema_version": 1,
                "summary_complete": True,
                "maps_complete": False,
            })
            (frame / "iteration.csv").write_text(
                "image_id,scale_level,iteration,phase,pass_index\n"
                "7,2,-1,initialization,0\n"
                "7,2,0,iteration,1\n"
                "7,0,-1,initialization,0\n"
                "7,0,0,iteration,1\n"
                "7,0,1,iteration,2\n",
                encoding="utf-8",
            )
            scene = dmap_dev.RunScene(
                label="summary", role="variant", repeat=0, scene_id="scene-a",
                instrumentation_dir=instrumentation, depth_map_dir=None, timing_dir=instrumentation,
            )

            catalog, availability = dmap_dev.build_map_catalog([scene])

            signals_per_iteration = (
                len(dmap_dev.REQUIRED_LOGICAL_STATE_SIGNALS)
                + len(dmap_dev.REQUIRED_LOGICAL_EVENT_SIGNALS)
                + len(dmap_dev.REPORT_DERIVED_LOGICAL_EVENT_SIGNALS)
                + len(dmap_dev.SCHEMA4_EXACT_STATE_SIGNALS)
                + len(dmap_dev.SCHEMA4_EXACT_EVENT_SIGNALS)
                + len(dmap_dev.SCHEMA4_EXACT_VIEW_SIGNALS)
                + len(dmap_dev.MECHANISM_LOGICAL_STATE_SIGNALS)
            )
            self.assertTrue(catalog.empty)
            self.assertEqual(
                len(availability),
                signals_per_iteration * 3
                + len(dmap_dev.SUMMARY_PROFILE_FINAL_SIGNALS),
            )
            self.assertEqual(
                set(availability["logical_iteration"].dropna().astype(int)), {-1, 0, 1}
            )
            self.assertFalse(availability["available"].any())
            self.assertFalse(availability["required"].any())
            self.assertEqual(
                set(availability[
                    availability["signal"].isin(dmap_dev.SCHEMA4_EXACT_STATE_SIGNALS)
                    | availability["signal"].isin(dmap_dev.SCHEMA4_EXACT_EVENT_SIGNALS)
                    | availability["signal"].isin(dmap_dev.SCHEMA4_EXACT_VIEW_SIGNALS)
                ]["availability_reason"]),
                {"exact_capture_unavailable: exact capture was not requested"},
            )
            hysteresis = availability[
                availability["signal"].isin(dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS)
            ]
            self.assertTrue(hysteresis.empty)
            nonexact = availability[
                ~availability["availability_reason"].str.startswith("exact_capture_unavailable:")
                & ~availability["signal"].isin(dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS)
            ]
            self.assertEqual(
                set(nonexact["availability_reason"]),
                {"summary_profile_maps_not_requested"},
            )
            for signal in (
                "depth_final_after_filter", "normal_final", "rejection_reason",
                "validity_transition", "selected_view_count", "candidate_source",
            ):
                selected = availability[availability["signal"] == signal]
                self.assertEqual(len(selected), 1, signal)
                self.assertTrue(selected["logical_iteration"].isna().all(), signal)
            self.assertEqual(
                set(availability["manifest_schema_name"]),
                {"openmvs.dmap.frame_summary"},
            )
            self.assertTrue(
                availability["manifest_path"].str.endswith("summary.json").all()
            )

            output = root / "report"
            output.mkdir()
            map_rows, signal_rows, _budget = dmap_dev.dmap_report_model.build_map_assets(
                catalog, availability, output
            )
            self.assertEqual(len(map_rows), len(availability))
            self.assertTrue(map_rows)
            self.assertTrue(all(not row["available"] for row in map_rows))
            self.assertIn(
                "cost_stored", {row["name"] for row in signal_rows}
            )

    def test_catalogs_declared_coarse_update_source_compatibility_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, frame = self.make_scene(root, {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 4,
                "num_logical_states": 1,
                "maps": [],
                "exact_capture": {"requested": False, "available": False},
            })
            plans = scene.instrumentation_dir / "resource_plans.jsonl"
            plans.write_text(json.dumps({
                "schema_name": "openmvs.dmap.resource_plan",
                "schema_version": 4,
                "image_id": 7,
                "pyramid_level": 1,
                "compatibility_maps_requested": True,
                "compatibility_map_contract": {
                    "update_source_map_expected": True,
                    "cost_map_expected": False,
                    "cost_map_unavailable_reason": (
                        "production confidence maps are retained at pyramid level 0 only"
                    ),
                },
            }) + "\n", encoding="utf-8")
            update_map = (
                scene.instrumentation_dir / "instrumentation" / "maps"
                / "depth0007_scale01_update_source.png"
            )
            update_map.parent.mkdir(parents=True)
            update_map.write_bytes(b"png")

            catalog, _availability = dmap_dev.build_map_catalog([scene])

            row = catalog[
                (catalog["signal"] == "candidate_source")
                & (catalog["pyramid_level"] == 1)
            ].iloc[0]
            self.assertTrue(row["available"])
            self.assertEqual(row["measurement_quality"], "proxy")
            self.assertEqual(row["measurement_basis"], "post_pass_change_detection")
            self.assertEqual(row["encoding"], "source_code*32")
            self.assertEqual(row["manifest_schema_name"], "openmvs.dmap.resource_plan")
            self.assertEqual(
                row["relative_path"],
                "instrumentation/maps/depth0007_scale01_update_source.png",
            )

    def test_configured_hysteresis_without_coarse_prior_is_not_required_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reason = (
                "configured mechanism did not execute because this level/stage has no "
                "coarse-resolution prior"
            )
            scene, frame = self.make_scene(root, {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 4,
                "num_iterations": 1,
                "exact_capture": {
                    "available": True,
                    "candidate_order_statistics": {
                        "configured": True,
                        "execution_available": False,
                        "available": False,
                        "unavailable_reason": reason,
                    },
                },
                "maps": [],
            })
            write_json(scene.instrumentation_dir / "run_metadata.json", {
                "cuda_patchmatch_parameters": {
                    "low_texture_update_min_gain": 0.001,
                    "low_texture_update_gate": 1,
                },
            })
            summary = json.loads((frame / "summary.json").read_text(encoding="utf-8"))
            summary["cuda_patchmatch_parameters"] = {
                "low_texture_update_min_gain": 0.001,
                "low_texture_update_gate": 1,
                "low_resolution_prior_available": False,
            }
            write_json(frame / "summary.json", summary)

            _catalog, availability = dmap_dev.build_map_catalog([scene])

            iterative = availability[
                availability["signal"].isin(
                    dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS
                )
                & (availability["logical_iteration"] == 0)
            ]
            self.assertEqual(
                len(iterative), len(dmap_dev.LOW_TEXTURE_UPDATE_EXACT_EVENT_SIGNALS)
            )
            self.assertFalse(iterative["required"].any())
            self.assertEqual(
                set(iterative["availability_reason"]),
                {
                    "configured_but_not_executed: this frame/stage has no "
                    "coarse-resolution prior"
                },
            )

    def test_frame_execution_keeps_legacy_configured_fallback(self) -> None:
        configured = {
            "low_texture_update_min_gain": 0.001,
            "low_texture_update_gate": 1,
        }
        legacy = dmap_dev.low_texture_update_frame_execution(configured, {})
        self.assertTrue(legacy["configured"])
        self.assertTrue(legacy["execution_available"])
        self.assertIsNone(legacy["low_resolution_prior_available"])

        current = dmap_dev.low_texture_update_frame_execution(configured, {
            "cuda_patchmatch_parameters": {
                "low_resolution_prior_available": False,
            },
        })
        self.assertTrue(current["configured"])
        self.assertFalse(current["execution_available"])
        self.assertIn("configured_but_not_executed", current["execution_unavailable_reason"])

        no_iterations = dmap_dev.low_texture_update_frame_execution(configured, {
            "cuda_patchmatch_parameters": {
                "low_resolution_prior_available": True,
                "estimation_iterations": 0,
            },
        })
        self.assertTrue(no_iterations["configured"])
        self.assertFalse(no_iterations["execution_available"])
        self.assertEqual(no_iterations["estimation_iterations"], 0)
        self.assertIn(
            "estimation_iterations is zero",
            no_iterations["execution_unavailable_reason"],
        )

        executed = dmap_dev.low_texture_update_frame_execution(configured, {
            "cuda_patchmatch_parameters": {
                "low_resolution_prior_available": True,
            },
        })
        self.assertTrue(executed["configured"])
        self.assertTrue(executed["execution_available"])
        self.assertIsNone(executed["execution_unavailable_reason"])

    def test_run_validation_records_failed_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene, _frame = self.make_scene(Path(directory), {"schema_version": 3, "maps": []})
            fake_result = {
                "schema_version": 3,
                "valid": False,
                "manifest_map_count": 0,
                "checks": [
                    {"name": "v3_manifest_complete", "passed": False},
                    {"name": "dimensions", "passed": True},
                ],
                "parity_max_abs": {},
            }
            with mock.patch.object(dmap_dev.instrumentation_validator, "validate", return_value=fake_result):
                validation = dmap_dev.validate_run_instrumentation([scene])

            self.assertEqual(len(validation), 1)
            self.assertFalse(validation.iloc[0]["valid"])
            self.assertEqual(json.loads(validation.iloc[0]["failed_checks"]), ["v3_manifest_complete"])


if __name__ == "__main__":
    unittest.main()
