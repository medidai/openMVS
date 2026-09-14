#!/usr/bin/env python3
"""Adversarial tests for capture and report artifact closures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dmap_observability import integrity


class ArtifactClosureTests(unittest.TestCase):
    def _new_capture(self, root: Path, *, managed: bool = True) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        if managed:
            (root / "00_experiment_lock.json").write_text(
                json.dumps({
                    "schema_name": "openmvs.dmap.experiment_lock",
                    "schema_version": 3,
                }),
                encoding="utf-8",
            )
        capture = root / "runs" / "base" / "repeat_00" / "scene" / "deep"
        (capture / "depth_maps").mkdir(parents=True)
        (capture / "dmap_instrumentation" / "depthmaps" / "0001").mkdir(parents=True)
        (capture / "work").mkdir()
        (capture / "command.sh").write_text("densify --observe\n", encoding="utf-8")
        (capture / "repro.json").write_text('{"return_code":0}\n', encoding="utf-8")
        (capture / "depth_maps" / "depth0001.dmap").write_bytes(b"terminal-dmap")
        (capture / "dmap_instrumentation" / "run_metadata.json").write_text(
            '{"schema_name":"openmvs.dmap.run"}\n', encoding="utf-8"
        )
        (capture / "dmap_instrumentation" / "depthmaps" / "0001" / "summary.json").write_text(
            '{"image_id":1}\n', encoding="utf-8"
        )
        (capture / "work" / "depth0001.dmap").write_bytes(b"duplicate-work-copy")
        return capture

    def _new_report(self, root: Path, *, managed: bool = True) -> Path:
        report = root / "reports" / "master"
        (report / "visualizations").mkdir(parents=True)
        if managed:
            (report / "report_policy.json").write_text(
                json.dumps({
                    "schema_name": "openmvs.dmap.report_policy",
                    "schema_version": 2,
                    "integrity_contract": {
                        "capture_artifact_closure_schema_version": 1,
                        "report_tree_closure_schema_version": 1,
                    },
                }),
                encoding="utf-8",
            )
        (report / "01_development_report.md").write_text("# Report\n", encoding="utf-8")
        (report / "01_development_report.html").write_text("<h1>Report</h1>\n", encoding="utf-8")
        (report / "02_investigation.html").write_text("<main>frame 1</main>\n", encoding="utf-8")
        (report / "report_model.json").write_text('{"schema_version":3}\n', encoding="utf-8")
        (report / "visualizations" / "cost.svg").write_text("<svg/>\n", encoding="utf-8")
        return report

    def test_capture_closure_covers_terminal_outputs_and_excludes_only_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = self._new_capture(Path(directory))
            manifest_path = integrity.write_capture_artifact_closure(capture, "deep")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = {row["path"] for row in manifest["files"]}

            self.assertIn("depth_maps/depth0001.dmap", paths)
            self.assertIn("dmap_instrumentation/run_metadata.json", paths)
            self.assertIn("command.sh", paths)
            self.assertNotIn("work/depth0001.dmap", paths)
            self.assertEqual(manifest["exclusions"], list(integrity.CAPTURE_EXCLUSIONS))
            validation = integrity.validate_capture_artifact_closure(capture, "deep")
            self.assertTrue(validation.valid, validation.reason)
            self.assertEqual(validation.status, "verified")

    def test_capture_closure_rejects_same_size_tamper_and_mode_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = self._new_capture(Path(directory))
            integrity.write_capture_artifact_closure(capture, "deep")
            target = capture / "depth_maps" / "depth0001.dmap"
            target.write_bytes(b"altered--dmap")
            tampered = integrity.validate_capture_artifact_closure(capture, "deep")
            self.assertFalse(tampered.valid)
            self.assertIn("changed=['depth_maps/depth0001.dmap']", tampered.reason)

            target.write_bytes(b"terminal-dmap")
            integrity.write_capture_artifact_closure(capture, "deep")
            os.chmod(target, 0o600)
            mode_changed = integrity.validate_capture_artifact_closure(capture, "deep")
            self.assertFalse(mode_changed.valid)
            self.assertIn("changed=['depth_maps/depth0001.dmap']", mode_changed.reason)

    def test_new_capture_requires_manifest_but_legacy_is_explicitly_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            managed = self._new_capture(root / "managed")
            required = integrity.validate_capture_artifact_closure(managed, "deep")
            self.assertFalse(required.valid)
            self.assertTrue(required.required)
            self.assertEqual(required.status, "invalid")

            legacy = self._new_capture(root / "legacy", managed=False)
            unverified = integrity.validate_capture_artifact_closure(legacy, "deep")
            self.assertTrue(unverified.valid)
            self.assertFalse(unverified.required)
            self.assertEqual(unverified.status, "legacy-unverified")
            self.assertIn("unverified", unverified.reason)

    def test_report_closure_rejects_same_size_investigation_html_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._new_report(Path(directory))
            integrity.write_report_tree_closure(report)
            target = report / "02_investigation.html"
            original = target.read_bytes()
            replacement = original.replace(b"frame 1", b"frame X")
            self.assertEqual(len(original), len(replacement))
            target.write_bytes(replacement)

            validation = integrity.validate_report_tree_closure(report)
            self.assertFalse(validation.valid)
            self.assertEqual(validation.status, "invalid")
            self.assertIn("changed=['02_investigation.html']", validation.reason)

    def test_report_closure_rejects_unexpected_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._new_report(Path(directory))
            integrity.write_report_tree_closure(report)
            (report / "unreviewed_debug_dump.bin").write_bytes(b"unexpected")

            validation = integrity.validate_report_tree_closure(report)
            self.assertFalse(validation.valid)
            self.assertIn("unexpected=['unreviewed_debug_dump.bin']", validation.reason)

    def test_mutable_validation_sidecar_is_explicitly_outside_report_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._new_report(Path(directory))
            integrity.write_report_tree_closure(report)
            sidecar = report / "01_development_report.validation.json"
            sidecar.write_text('{"valid":true}\n', encoding="utf-8")
            first = integrity.validate_report_tree_closure(report)
            sidecar.write_text('{"valid":false,"rerun":1}\n', encoding="utf-8")
            second = integrity.validate_report_tree_closure(report)

            self.assertTrue(first.valid, first.reason)
            self.assertTrue(second.valid, second.reason)
            manifest = json.loads(
                (report / integrity.REPORT_CLOSURE_FILE).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["exclusions"], list(integrity.REPORT_EXCLUSIONS))

    def test_present_invalid_legacy_manifest_is_never_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._new_report(Path(directory), managed=False)
            integrity.write_report_tree_closure(report)
            manifest_path = report / integrity.REPORT_CLOSURE_FILE
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["exclusions"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            validation = integrity.validate_report_tree_closure(report)
            self.assertFalse(validation.valid)
            self.assertFalse(validation.required)
            self.assertEqual(validation.status, "invalid")

    def test_symlinks_are_rejected_even_when_they_replace_an_excluded_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._new_report(Path(directory))
            target = report / "01_development_report.md"
            (report / "01_development_report.validation.json").symlink_to(target)
            with self.assertRaisesRegex(integrity.IntegrityError, "symlink"):
                integrity.write_report_tree_closure(report)


if __name__ == "__main__":
    unittest.main()
