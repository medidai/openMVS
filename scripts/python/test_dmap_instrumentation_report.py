from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import dmap_instrumentation_report as report  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_pfm(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype="<f4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"Pf\n")
        handle.write(f"{values.shape[1]} {values.shape[0]}\n".encode("ascii"))
        handle.write(b"-1.0\n")
        np.flipud(values).tofile(handle)


class ManifestLoadingTest(unittest.TestCase):
    def make_run(self, root: Path, manifest: dict[str, object]) -> Path:
        run = root / "run"
        frame = run / "depthmaps" / "0001_frame"
        write_json(frame / "summary.json", {"schema_version": manifest.get("schema_version"), "image_id": 1, "scale_level": 0})
        write_json(frame / "map_manifest.json", manifest)
        return run

    def test_schema3_manifest_is_authoritative_and_indexes_logical_states(self) -> None:
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
                    "measurement_basis": "production hypothesis state",
                },
                {
                    "signal": "cost_stored",
                    "path": "logical_states/state01_iteration01/cost_stored.pfm",
                    "dtype": "float32",
                    "role": "logical_state",
                    "logical_iteration": 0,
                    "stage": "iteration",
                    "measurement_quality": "exact",
                },
                {
                    "signal": "cost_total_equal_selected_rescore_proxy",
                    "path": "logical_states/state01_iteration01/cost_total_proxy.pfm",
                    "dtype": "float32",
                    "role": "logical_state",
                    "logical_iteration": 0,
                    "measurement_quality": "proxy",
                    "proxy_target": "production selected-view score",
                    "limitations": "equal selected-view weights",
                },
                {
                    "signal": "cost_stored",
                    "path": "phase_maps/pass01_cost_stored.pfm",
                    "dtype": "float32",
                    "role": "phase_state",
                    "pass_index": 1,
                    "stage": "iteration",
                    "measurement_quality": "exact",
                },
                {
                    "signal": "view_churn",
                    "path": "logical_states/state01_iteration01/view_churn.png",
                    "dtype": "uint8",
                    "role": "logical_event",
                    "logical_iteration": 0,
                    "stage": "iteration",
                    "measurement_quality": "exact",
                },
            ]
            run_path = self.make_run(root, {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 3,
                "map_granularity": "logical_iteration",
                "maps": entries,
            })
            frame = run_path / "depthmaps" / "0001_frame"
            for entry in entries:
                path = frame / str(entry["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            unlisted = frame / "maps" / "cost_final.pfm"
            unlisted.parent.mkdir(parents=True, exist_ok=True)
            unlisted.touch()

            record = report.load_run("variant", run_path).depthmaps[0]

            self.assertEqual(set(record.logical_state_maps["cost_stored"]), {-1, 0})
            self.assertEqual(record.logical_state_maps["cost_stored"][-1].fidelity, "exact")
            proxy = record.logical_state_maps["cost_total_equal_selected_rescore_proxy"][0]
            self.assertEqual(proxy.fidelity, "proxy")
            self.assertEqual(proxy.proxy_target, "production selected-view score")
            self.assertEqual(set(record.logical_event_maps["view_churn"]), {0})
            self.assertNotIn("cost_stored", record.pass_maps)
            self.assertNotIn("cost_final.pfm", record.pfm_maps)

    def test_schema2_retains_filename_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = self.make_run(root, {"schema_version": 2, "maps": []})
            frame = run_path / "depthmaps" / "0001_frame"
            final_cost = frame / "maps" / "cost_final.pfm"
            final_cost.parent.mkdir(parents=True, exist_ok=True)
            final_cost.touch()
            event = frame / "pass_maps" / "pass01_depth_delta.pfm"
            event.parent.mkdir(parents=True, exist_ok=True)
            event.touch()

            record = report.load_run("legacy", run_path).depthmaps[0]

            self.assertEqual(record.pfm_maps["cost_final.pfm"], final_cost)
            self.assertEqual(record.pass_maps["depth_delta"][1], event)
            self.assertTrue(any(artifact.signal == "cost_final" for artifact in record.map_artifacts))

    def test_unavailable_candidate_accounting_is_explicit_in_tables_and_plot(self) -> None:
        plt = report.try_import_matplotlib()
        if plt is None:
            self.skipTest("matplotlib unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = self.make_run(root, {"schema_version": 3, "maps": []})
            frame = run_path / "depthmaps" / "0001_frame"
            write_json(frame / "summary.json", {
                "schema_version": 3,
                "image_id": 1,
                "scale_level": 0,
                "candidate_accounting_mode": "unavailable_post_pass_snapshot",
                "candidate_acceptance": [{
                    "candidate_type": "SPATIAL_PROPAGATION",
                    "tested_count": 0,
                    "accepted_count": 0,
                    "acceptance_rate": 0.0,
                }],
            })
            (frame / "iteration.csv").write_text(
                "image_id,scale_level,iteration,pass_index,phase,valid_ratio,"
                "changed_ratio,candidates_tested,candidates_finite,"
                "candidates_accepted,acceptance_rate,mean_cost\n"
                "1,0,0,1,black,0.9,0,0,0,0,0,0.6\n"
                "1,0,0,2,red,0.9,0,0,0,0,0,0.5\n",
                encoding="utf-8",
            )

            run = report.load_run("legacy", run_path)
            record = run.depthmaps[0]
            logical = report.record_logical_iterations(record)[0]

            self.assertEqual(
                logical["candidate_accounting_mode"],
                "unavailable_post_pass_snapshot",
            )
            for key in report.CANDIDATE_ACCOUNTING_METRICS:
                self.assertIn(key, logical)
                self.assertIsNone(logical[key], key)
            raw = report.read_csv(frame / "iteration.csv")
            self.assertEqual(raw[0]["changed_ratio"], "0")
            self.assertEqual(raw[0]["candidates_tested"], "0")

            table = report.iteration_rows(run)[0]
            self.assertEqual(table[4], "unavailable")
            self.assertEqual(table[5], "unavailable")
            self.assertEqual(table[10], "unavailable")
            self.assertEqual(table[11], "unavailable")
            self.assertEqual(table[12], "unavailable")
            self.assertIn("unavailable post-pass frames", report.candidate_rows(run)[0][0])
            markdown = report.generate_markdown(
                [run], [], [], root / "report.md"
            )
            self.assertIn(">candidate type</th>", markdown)
            self.assertIn(">finite</th>", markdown)
            self.assertIn("unavailable post-pass frames", markdown)
            self.assertGreaterEqual(markdown.count("unavailable"), 6)

            figure, axis = plt.subplots()
            report.plot_iteration_axis(axis, record)
            self.assertEqual(
                [line.get_label() for line in axis.lines], ["mean cost"]
            )
            self.assertTrue(any(
                "accounting unavailable" in item.get_text()
                for item in axis.texts
            ))
            plt.close(figure)

            figure, axis = plt.subplots()
            report.plot_candidate_axis(axis, record)
            self.assertEqual(len(axis.patches), 0)
            self.assertTrue(any(
                "post-pass snapshot mode" in item.get_text()
                for item in axis.texts
            ))
            plt.close(figure)

    def test_schema4_parser_preserves_source_view_and_channel_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "signal": "view_cost_components_exact",
                "path": "logical_states/state01_iteration01/view_cost_components_exact_view02_id0042.pfm",
                "dtype": "float32x3",
                "role": "logical_view_state",
                "logical_iteration": 0,
                "stage": "iteration",
                "stage_index": 1,
                "measurement_quality": "exact",
                "measurement_basis": "production_hot_kernel_view_record",
                "source_view_index": 2,
                "source_image_id": 42,
                "source_image_name": "images/0042.jpg",
                "contribution_basis": "production_32_draw_monte_carlo",
                "channels_memory_order": {"0": "photo_after_prior", "1": "geometric", "2": "total"},
                "unavailable_value": -1.0,
            }
            frame = root / "frame"
            artifact = report.parse_map_artifacts(frame, {"schema_version": 4, "maps": [entry]})[0]

            self.assertEqual(artifact.source_view_index, 2)
            self.assertEqual(artifact.source_image_id, 42)
            self.assertEqual(artifact.source_image_name, "images/0042.jpg")
            self.assertEqual(artifact.stage_index, 1)
            self.assertEqual(artifact.contribution_basis, "production_32_draw_monte_carlo")
            self.assertEqual(artifact.channels["1"], "geometric")
            self.assertEqual(artifact.unavailable_value, -1.0)
            self.assertEqual(artifact.metadata["role"], "logical_view_state")

    def test_parser_canonicalizes_entry_and_manifest_pyramid_level_aliases(self) -> None:
        frame = Path("/tmp/frame")
        entries = [
            {"signal": "jbu_transfer_depth", "path": "level2.pfm", "scale_number": 2},
            {"signal": "jbu_nearest_depth", "path": "level1.pfm"},
            {"signal": "cost_stored", "path": "invalid.pfm", "pyramid_level": -1},
        ]

        artifacts = report.parse_map_artifacts(
            frame,
            {"schema_version": 4, "scale_level": 1, "maps": entries},
        )

        self.assertEqual(artifacts[0].pyramid_level, 2)
        self.assertEqual(artifacts[1].pyramid_level, 1)
        self.assertIsNone(artifacts[2].pyramid_level)

    def test_logical_cost_atlas_labels_and_renders_manifest_states(self) -> None:
        plt = report.try_import_matplotlib()
        if plt is None:
            self.skipTest("matplotlib unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for iteration, state in ((-1, "state00_initialization"), (0, "state01_iteration01")):
                for signal, quality in (
                    ("cost_stored", "exact"),
                    ("cost_total_equal_selected_rescore_proxy", "proxy"),
                    ("cost_stored_minus_rescore", "derived_exact"),
                ):
                    entries.append({
                        "signal": signal,
                        "path": f"logical_states/{state}/{signal}.pfm",
                        "dtype": "float32",
                        "role": "logical_state",
                        "logical_iteration": iteration,
                        "stage": "initialization" if iteration < 0 else "iteration",
                        "measurement_quality": quality,
                    })
            run_path = self.make_run(root, {
                "schema_name": "openmvs.dmap.map_manifest",
                "schema_version": 3,
                "map_granularity": "logical_iteration",
                "maps": entries,
            })
            frame = run_path / "depthmaps" / "0001_frame"
            for index, entry in enumerate(entries):
                values = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32) + index * 0.01
                if entry["signal"] == "cost_stored_minus_rescore":
                    values -= 0.35
                write_pfm(frame / str(entry["path"]), values)
            run = report.load_run("variant", run_path)

            report.make_logical_cost_previews([run], root / "assets", plt)

            panel = run.depthmaps[0].logical_cost_panel_path
            self.assertIsNotNone(panel)
            self.assertTrue(panel.is_file())
            self.assertGreater(panel.stat().st_size, 0)
            self.assertEqual(report.signal_fidelity("cost_stored_minus_rescore", {"measurement_quality": "derived_exact"}), "derived")
            self.assertEqual(report.signal_fidelity("confidence_stored", {"measurement_quality": "derived_exact"}), "derived")
            panels = report.make_diagnostic_panels([run], root / "assets", plt)
            self.assertTrue(any(item.path == panel and item.mechanism == "cost" for item in panels))

    def test_improvement_panel_prefers_exact_logical_event_over_legacy_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "cost_improvement_exact.pfm"
            legacy_a = root / "pass01_improvement.pfm"
            legacy_b = root / "pass02_improvement.pfm"
            write_pfm(exact, np.full((2, 3), 2.0, dtype=np.float32))
            write_pfm(legacy_a, np.full((2, 3), 11.0, dtype=np.float32))
            write_pfm(legacy_b, np.full((2, 3), 13.0, dtype=np.float32))
            record = report.DepthMapRecord(
                directory=root,
                summary={"image_id": 1, "scale_level": 0},
                improvement_maps={1: legacy_a, 2: legacy_b},
                logical_event_maps={
                    "cost_improvement_exact": {
                        0: report.MapArtifact(
                            signal="cost_improvement_exact",
                            path=exact,
                            role="logical_event",
                            logical_iteration=0,
                            measurement_quality="derived_exact",
                        )
                    }
                },
            )
            run = report.RunData(label="base", path=root, depthmaps=[record])
            captured: list[dict[int, np.ndarray]] = []

            def save_panel(_plt, _np, _run, _record, data, _vmax, out):
                captured.append(data)
                return out

            with mock.patch.object(
                report, "save_improvement_panel", side_effect=save_panel
            ):
                report.make_improvement_previews([run], root / "assets", object())

            self.assertEqual(list(captured[0]), [0])
            np.testing.assert_array_equal(
                captured[0][0], np.full((2, 3), 2.0, dtype=np.float32)
            )
            source = Path(report.__file__).read_text(encoding="utf-8")
            self.assertIn('if signal == "cost_improvement_exact":', source)


if __name__ == "__main__":
    unittest.main()
