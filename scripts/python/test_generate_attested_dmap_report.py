#!/usr/bin/env python3
"""Focused tests for attested standalone DMAP report generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_sweep
import generate_attested_dmap_report as attested_report
from dmap_observability import portable_bundle


class AttestedReportTests(unittest.TestCase):
    def make_composite_report_fixture(
        self, root: Path, report: Path,
    ) -> tuple[Path, dict[str, object]]:
        config_path = root / "composite-config.yaml"
        runs = [
            {"label": f"run-{index}", "role": "baseline" if index == 0 else "variant"}
            for index in range(7)
        ]
        scenes = [{"scan_id": f"scene-{index}"} for index in range(4)]
        config = {
            "default_densify_args": ["--geometric-iters", "4"],
            "runs": runs,
            "scenes": scenes,
            "sweep": {"stages": [{"name": "maps", "argument_overrides": {}}]},
        }
        import yaml

        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        jobs: dict[str, dict[str, object]] = {}
        manifest_rows: list[dict[str, object]] = []
        for run in runs:
            for scene in scenes:
                label = str(run["label"])
                scene_id = str(scene["scan_id"])
                job_id = f"{label}-{scene_id}"
                run_dir = root / "runs" / job_id
                jobs[job_id] = {
                    "status": "complete", "run": label, "repeat": 0,
                    "scene_id": scene_id, "run_dir": str(run_dir), "stage": "maps",
                }
                instrumentation = run_dir / "dmap_instrumentation"
                common = {
                    "label": label, "repeat": 0, "scene_id": scene_id,
                    "role": run["role"], "depth_map_dir": str(run_dir / "depth_maps"),
                    "diagnostic_only": False, "diagnostic_only_reason": "",
                }
                manifest_rows.append({
                    **common,
                    "instrumentation_dir": str(instrumentation),
                })
                for iteration in range(4):
                    manifest_rows.append({
                        **common,
                        "instrumentation_dir": str(
                            instrumentation / "geometric_iterations" / f"iteration{iteration:02d}"
                        ),
                    })
        config_identity = {
            "path": str(config_path), "exists": True,
            "size": config_path.stat().st_size,
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
        ledger: dict[str, object] = {
            "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
            "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
            "identity": {
                "sha256": "a" * 64,
                "source_schedules": [{"config_identity": config_identity}],
            },
            "composite": {
                "schema_name": attested_report.predecessor_composite.COMPOSITE_SCHEMA_NAME,
                "schema_version": attested_report.predecessor_composite.COMPOSITE_SCHEMA_VERSION,
            },
            "composite_sha256": "b" * 64,
            "status": "complete", "sessions": [], "jobs": jobs,
        }
        schedule = root / "composite-schedule.json"
        schedule.write_text(json.dumps(ledger), encoding="utf-8")
        report.mkdir(parents=True)
        artifacts: dict[str, object] = {
            "report_manifest.json": {
                "schema_version": 2, "source_config": str(config_path),
                "run_scenes": manifest_rows,
            },
            "report_model.json": {
                "runs": [
                    {
                        "label": run["label"], "repeat": 0,
                        "scenes": [scene["scan_id"] for scene in scenes],
                    }
                    for run in runs
                ],
            },
            "report_inventory.json": {"schema_version": 2},
            "report_policy.json": {"schema_version": 1},
        }
        for name, value in artifacts.items():
            (report / name).write_text(
                json.dumps(value, sort_keys=True) + "\n", encoding="utf-8",
            )
        (report / "01_development_report.md").write_text("# Composite\n", encoding="utf-8")
        (report / "accuracy_ledger.csv").write_text("candidate,rank\ncontrol,1\n", encoding="utf-8")
        return schedule, ledger

    def test_atomic_staged_report_promotion_archives_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            result = attested_report.promote_staged_report(
                staging, canonical, archive
            )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertEqual(
                (archive / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertFalse(staging.exists())
            self.assertTrue(result["replaced_nonempty_report"])
            self.assertEqual(result["archive"], str(archive))

    def test_promotion_requires_receipt_declared_by_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")
            (staging / "report_inventory.json").write_text(
                json.dumps({attested_report.FINALIZER_INVENTORY_KEY: {}}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "required but missing"):
                attested_report.promote_staged_report(
                    staging, canonical, archive,
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old",
            )
            self.assertEqual(
                (staging / "identity.txt").read_text(encoding="utf-8"), "new",
            )
            self.assertFalse(archive.exists())

    def test_promotion_transaction_can_require_receipt_without_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "required but missing"):
                attested_report.promote_staged_report(
                    staging, canonical, root / "archive",
                    require_finalizer_receipt=True,
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old",
            )
            self.assertTrue(staging.is_dir())

    def test_empty_target_promotion_rolls_back_receipt_deleted_inside_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            staging.mkdir()
            receipt_path = staging / attested_report.FINALIZER_RECEIPT_FILE
            receipt_path.write_text('{"sentinel":true}\n', encoding="utf-8")
            (staging / "report_inventory.json").write_text(
                json.dumps({attested_report.FINALIZER_INVENTORY_KEY: {}}) + "\n",
                encoding="utf-8",
            )
            validation = {
                "present": True,
                "valid": True,
                "receipt": {"canonical_tree_after": None},
            }
            real_replace = os.replace
            replace_calls = 0

            def replace_then_delete_receipt(source: Path, target: Path) -> None:
                nonlocal replace_calls
                replace_calls += 1
                real_replace(source, target)
                if replace_calls == 1:
                    (target / attested_report.FINALIZER_RECEIPT_FILE).unlink()

            def validate_receipt(
                report: Path, _published: Path, **_kwargs: object,
            ) -> dict[str, object]:
                if not (report / attested_report.FINALIZER_RECEIPT_FILE).is_file():
                    raise RuntimeError("trusted finalizer receipt is required but missing")
                return validation

            with (
                mock.patch.object(
                    attested_report, "validate_finalizer_receipt",
                    side_effect=validate_receipt,
                ),
                mock.patch.object(
                    attested_report.os, "replace",
                    side_effect=replace_then_delete_receipt,
                ),
                self.assertRaisesRegex(RuntimeError, "rolled back"),
            ):
                attested_report.promote_staged_report(
                    staging, canonical, None, require_finalizer_receipt=True,
                )

            self.assertFalse(canonical.exists())
            self.assertTrue(staging.is_dir())
            self.assertFalse(receipt_path.exists())

    def test_atomic_promotion_archive_collision_leaves_both_inputs_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            archive.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "archive already exists"):
                attested_report.promote_staged_report(
                    staging, canonical, archive
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertEqual(
                (staging / "identity.txt").read_text(encoding="utf-8"), "new"
            )

    def test_atomic_promotion_archive_failure_rolls_back_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            with mock.patch.object(
                attested_report,
                "_atomic_rename_noreplace",
                side_effect=OSError("injected archive failure"),
            ), self.assertRaisesRegex(RuntimeError, "rolled back"):
                attested_report.promote_staged_report(
                    staging, canonical, archive
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertEqual(
                (staging / "identity.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertFalse(archive.exists())

    def test_archive_collision_created_after_precheck_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")
            real_noreplace = attested_report._atomic_rename_noreplace

            def collide(left: Path, right: Path) -> None:
                if right == archive:
                    archive.mkdir()
                    (archive / "identity.txt").write_text(
                        "concurrent", encoding="utf-8"
                    )
                real_noreplace(left, right)

            with mock.patch.object(
                attested_report,
                "_atomic_rename_noreplace",
                side_effect=collide,
            ), self.assertRaisesRegex(RuntimeError, "rolled back"):
                attested_report.promote_staged_report(
                    staging, canonical, archive
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertEqual(
                (staging / "identity.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertEqual(
                (archive / "identity.txt").read_text(encoding="utf-8"),
                "concurrent",
            )

    def test_atomic_promotion_without_predecessor_is_single_rename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            staging.mkdir()
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            result = attested_report.promote_staged_report(
                staging, canonical, None
            )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertFalse(staging.exists())
            self.assertFalse(result["replaced_nonempty_report"])
            self.assertIsNone(result["archive"])

    def test_promotion_rejects_authoritative_staging_path_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "report_manifest.json").write_text(
                json.dumps({"report": str(staging / "report.md")}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "vanished staging path"):
                attested_report.promote_staged_report(
                    staging, canonical, archive
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertTrue(staging.is_dir())
            self.assertFalse(archive.exists())

    def test_post_exchange_path_validation_failure_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "03_master"
            staging = root / "03_master_staging"
            archive = root / "03_master_archive_20260719_120000"
            canonical.mkdir()
            staging.mkdir()
            (canonical / "identity.txt").write_text("old", encoding="utf-8")
            (staging / "identity.txt").write_text("new", encoding="utf-8")

            with mock.patch.object(
                attested_report,
                "require_no_report_path_reference",
                side_effect=[None, RuntimeError("injected post-exchange failure")],
            ), self.assertRaisesRegex(RuntimeError, "rolled back"):
                attested_report.promote_staged_report(
                    staging, canonical, archive
                )

            self.assertEqual(
                (canonical / "identity.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertEqual(
                (staging / "identity.txt").read_text(encoding="utf-8"), "new"
            )
            self.assertFalse(archive.exists())

    def test_staged_publication_binding_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "03_master_staging"
            published = root / "03_master"
            staging.mkdir()
            (staging / "report_policy.json").write_text(json.dumps({
                "published_output_dir": str(published),
            }), encoding="utf-8")
            (staging / "report_manifest.json").write_text(json.dumps({
                "published_output_dir": str(published),
            }), encoding="utf-8")

            attested_report.require_published_output_binding(staging, published)

            (staging / "report_manifest.json").write_text(json.dumps({
                "published_output_dir": str(root / "wrong"),
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "manifest does not bind"):
                attested_report.require_published_output_binding(staging, published)

            (staging / "report_policy.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "binding is unreadable"):
                attested_report.require_published_output_binding(staging, published)

    def test_trusted_finalizer_spec_is_stable_and_explicitly_unsandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            published = root / "03_master"
            finalizer = root / "finalizer.sh"
            finalizer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            finalizer.chmod(0o755)
            digest = hashlib.sha256(finalizer.read_bytes()).hexdigest()

            spec = attested_report.trusted_finalizer_spec(
                finalizer, digest, ["campaign-finalize"],
                ["parity.json", "parity.csv"], published,
                {"BOUND_VALUE": "exact", "PATH": "/usr/bin:/bin"},
            )

            rendered = json.dumps(spec, sort_keys=True)
            self.assertNotIn("/proc/", rendered)
            self.assertNotIn("staging_", rendered)
            self.assertEqual(spec["isolation"], "none")
            self.assertTrue(spec["arbitrary_code_execution"])
            self.assertIn("${STAGED_REPORT_DIR}", spec["logical_argv"])
            self.assertIn("${STAGED_REPORT_DIR}", spec["replay_command"])
            self.assertEqual(
                spec["environment"],
                {"BOUND_VALUE": "exact", "PATH": "/usr/bin:/bin"},
            )
            self.assertTrue(spec["replay_command"].startswith("env -i "))
            self.assertEqual(
                spec["approved_outputs"], ["parity.csv", "parity.json"]
            )
            self.assertEqual(
                spec["replay_scope"],
                "invocation_only_runtime_dependencies_unbound",
            )
            self.assertEqual(
                spec["runtime_dependencies"]["coverage"], "invocation_only",
            )

    def test_bounded_runtime_manifest_detects_non_yaml_dependency_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finalizer = root / "finalizer.sh"
            finalizer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            finalizer.chmod(0o755)
            dependency = root / "non_yaml_extension.so"
            dependency.write_bytes(b"native-extension-v1")
            manifest = attested_report.build_runtime_dependency_manifest(
                [attested_report.runtime_dependency_file(
                    "python_module", "numpy.core._multiarray_umath", dependency,
                )],
                [],
                coverage="loaded_nonstdlib_python_modules_and_declared_files",
            )
            digest = hashlib.sha256(finalizer.read_bytes()).hexdigest()

            spec = attested_report.trusted_finalizer_spec(
                finalizer, digest, [], ["output.json"], root / "report",
                runtime_dependencies=manifest,
            )
            self.assertEqual(
                spec["replay_scope"],
                "invocation_and_bounded_runtime_dependencies",
            )

            dependency.write_bytes(b"native-extension-v2")
            with self.assertRaisesRegex(RuntimeError, "runtime dependency drifted"):
                attested_report.trusted_finalizer_spec(
                    finalizer, digest, [], ["output.json"], root / "report",
                    runtime_dependencies=manifest,
                )

    def test_trusted_finalizer_rejects_symlink_and_wrong_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.sh"
            target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            target.chmod(0o755)
            finalizer = root / "finalizer.sh"
            finalizer.symlink_to(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "non-symlink"):
                attested_report.trusted_finalizer_spec(
                    finalizer, digest, [], ["output.json"], root / "report"
                )
            with self.assertRaisesRegex(RuntimeError, "trust pin"):
                attested_report.trusted_finalizer_spec(
                    target, "0" * 64, [], ["output.json"], root / "report"
                )
            with self.assertRaisesRegex(RuntimeError, "protected report controls"):
                attested_report.trusted_finalizer_spec(
                    target, digest, [], ["report_policy.json"], root / "report"
                )

    def test_trusted_finalizer_detects_real_canonical_mutation_without_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "reports/staging"
            canonical = root / "reports/canonical"
            staging.mkdir(parents=True)
            canonical.mkdir()
            (staging / "report_inventory.json").write_text("{}\n", encoding="utf-8")
            sentinel = canonical / "sentinel.txt"
            sentinel.write_text("before\n", encoding="utf-8")
            finalizer = root / "mutating-finalizer.sh"
            finalizer.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "staged= published=\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --staged-report-dir) staged=$2; shift 2 ;;\n"
                "    --published-report-dir) published=$2; shift 2 ;;\n"
                "    *) exit 64 ;;\n"
                "  esac\n"
                "done\n"
                "printf 'mutated\\n' > \"$published/sentinel.txt\"\n",
                encoding="utf-8",
            )
            finalizer.chmod(0o755)
            digest = hashlib.sha256(finalizer.read_bytes()).hexdigest()
            source = {"sha256": "c" * 64}
            with mock.patch.object(
                attested_report.dmap_sweep,
                "validate_report_source_provenance",
                return_value=(True, "valid source", source),
            ):
                with self.assertRaisesRegex(RuntimeError, "mutated the canonical"):
                    attested_report.run_trusted_finalizer_transaction(
                        staging, canonical, finalizer, digest, [], ["output.json"],
                    )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "mutated\n")
            self.assertFalse((staging / attested_report.FINALIZER_RECEIPT_FILE).exists())
            self.assertEqual(
                json.loads((staging / "report_inventory.json").read_text()), {}
            )

    def test_receipt_bearing_reuse_is_byte_exact_on_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            config.write_text("schema_version: 2\n", encoding="utf-8")
            schedule = root / "schedule.json"
            ledger = {
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {"sha256": "a" * 64},
                "status": "complete", "sessions": [],
                "jobs": {"job-a": {"status": "complete"}},
            }
            schedule.write_text(json.dumps(ledger), encoding="utf-8")
            report = root / "report"
            report.mkdir()
            (report / "existing-report.md").write_text("# existing\n", encoding="utf-8")
            (report / "report_policy.json").write_text("{}\n", encoding="utf-8")
            (report / attested_report.FINALIZER_RECEIPT_FILE).write_text(
                '{"receipt":"bound"}\n', encoding="utf-8",
            )

            def snapshot() -> dict[str, bytes]:
                return {
                    path.relative_to(report).as_posix(): path.read_bytes()
                    for path in sorted(report.rglob("*")) if path.is_file()
                }

            before = snapshot()
            source = {"sha256": "c" * 64}
            bound_recovery = {
                "path": str(schedule.resolve()),
                "sha256": dmap_sweep.sha256_file(schedule),
                "schedule_identity_sha256": ledger["identity"]["sha256"],
                "evidence_digest": dmap_sweep.report_evidence_digest(ledger),
            }
            with mock.patch.object(
                attested_report.dmap_sweep, "valid_report",
                return_value=(True, "valid report"),
            ), mock.patch.object(
                attested_report.dmap_sweep, "validate_report_source_provenance",
                return_value=(True, "valid source", source),
            ), mock.patch.object(
                attested_report, "validate_finalizer_receipt",
                return_value={
                    "present": True, "valid": True,
                    "receipt": {"recovery_binding": bound_recovery},
                },
            ), mock.patch.object(
                attested_report, "write_recovery_binding",
            ) as recovery_writer:
                self.assertEqual(attested_report.run(attested_report.Arguments(
                    config=config, output_dir=report, parent_schedule=schedule,
                )), 0)
            recovery_writer.assert_not_called()
            self.assertEqual(snapshot(), before)

            with mock.patch.object(
                attested_report.dmap_sweep, "valid_report",
                return_value=(True, "valid report"),
            ), mock.patch.object(
                attested_report.dmap_sweep, "validate_report_source_provenance",
                return_value=(True, "valid source", source),
            ), mock.patch.object(
                attested_report, "validate_finalizer_receipt",
                side_effect=RuntimeError("receipt drift"),
            ):
                with self.assertRaisesRegex(RuntimeError, "receipt drift"):
                    attested_report.run(attested_report.Arguments(
                        config=config, output_dir=report, parent_schedule=schedule,
                    ))
            self.assertEqual(snapshot(), before)

    def test_finalizer_environment_is_exactly_bound_and_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "reports/staging"
            canonical = root / "reports/canonical"
            staging.mkdir(parents=True)
            canonical.mkdir()
            (staging / "report_inventory.json").write_text("{}\n", encoding="utf-8")
            finalizer = root / "environment-finalizer.sh"
            finalizer.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "staged=\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --staged-report-dir) staged=$2; shift 2 ;;\n"
                "    --published-report-dir) shift 2 ;;\n"
                "    *) exit 64 ;;\n"
                "  esac\n"
                "done\n"
                "printf '%s|%s\\n' \"${BOUND_VALUE:-missing}\" \"${UNBOUND_VALUE:-absent}\" > \"$staged/environment.txt\"\n",
                encoding="utf-8",
            )
            finalizer.chmod(0o755)
            digest = hashlib.sha256(finalizer.read_bytes()).hexdigest()
            source = {"sha256": "c" * 64}
            snapshot = mock.Mock(archive_sha256=source["sha256"])
            with mock.patch.object(
                attested_report.dmap_sweep,
                "validate_report_source_provenance",
                return_value=(True, "valid source", source),
            ), mock.patch.object(
                attested_report.dmap_sweep,
                "create_report_source_snapshot",
                return_value=snapshot,
            ):
                receipt = attested_report.run_trusted_finalizer_transaction(
                    staging, canonical, finalizer, digest, [], ["environment.txt"],
                    environment={"BOUND_VALUE": "exact"},
                )
                self.assertEqual(
                    (staging / "environment.txt").read_text(encoding="utf-8"),
                    "exact|absent\n",
                )
                self.assertEqual(
                    receipt["finalizer"]["environment"], {"BOUND_VALUE": "exact"},
                )
                drifted = attested_report.trusted_finalizer_spec(
                    finalizer, digest, [], ["environment.txt"], canonical,
                    {"BOUND_VALUE": "drifted"},
                )
                with self.assertRaisesRegex(
                    RuntimeError, "replay specification differs",
                ):
                    attested_report.validate_finalizer_receipt(
                        staging, canonical, expected_spec=drifted,
                        require_receipt=True,
                    )

    def test_trusted_finalizer_real_recovery_publication_transaction(self) -> None:
        if dmap_sweep.source_snapshot.zstandard is None:
            raise unittest.SkipTest("zstandard is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / "source.txt").write_text("bound source\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test",
                "-c", "user.email=test@example.invalid", "commit", "-q",
                "-m", "initial",
            ], cwd=repository, check=True)

            canonical = root / "reports" / "03_master"
            staging = root / "reports" / "03_master_staging_unique"
            archive = root / "reports" / "03_master_archive"
            canonical.mkdir(parents=True)
            staging.mkdir()
            (canonical / "nested").mkdir()
            (canonical / "nested" / "identity.txt").write_text(
                "old canonical\n", encoding="utf-8",
            )
            old_tree = attested_report.recursive_tree_identity(canonical)
            policy = {
                "schema_version": 1, "published_output_dir": str(canonical),
            }
            manifest = {
                "schema_version": 1, "published_output_dir": str(canonical),
                "report_policy": policy,
            }
            (staging / "report_policy.json").write_text(
                json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8",
            )
            (staging / "report_manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8",
            )
            (staging / "report_inventory.json").write_text(
                json.dumps({"schema_version": 1}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            policy_before = (staging / "report_policy.json").read_bytes()
            manifest_before = (staging / "report_manifest.json").read_bytes()

            schedule = root / "schedule.json"
            ledger = {
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {
                    "sha256": "a" * 64,
                    "source_provenance": {"sha256": "b" * 64},
                },
                "status": "incomplete",
                "sessions": [],
                "jobs": {
                    "job-a": {"status": "complete"},
                    "job-b": {"status": "pending"},
                },
                "report": {},
            }
            schedule.write_text(json.dumps(ledger), encoding="utf-8")
            finalizer = root / "finalizer.sh"
            finalizer.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "[ \"$1\" = campaign-finalize ]\n"
                "shift\n"
                "staged= published=\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  case \"$1\" in\n"
                "    --staged-report-dir) staged=$2; shift 2 ;;\n"
                "    --published-report-dir) published=$2; shift 2 ;;\n"
                "    *) exit 64 ;;\n"
                "  esac\n"
                "done\n"
                "[ -n \"$staged\" ] && [ -n \"$published\" ]\n"
                "printf '{\"valid\":true}\\n' > \"$staged/parity.json\"\n"
                "printf 'metric,value\\nvalid,1\\n' > \"$staged/parity.csv\"\n",
                encoding="utf-8",
            )
            finalizer.chmod(0o755)
            finalizer_sha256 = hashlib.sha256(finalizer.read_bytes()).hexdigest()
            non_yaml_dependency = root / "non_yaml_runtime_extension.so"
            non_yaml_dependency.write_bytes(b"runtime-extension-v1")
            runtime_manifest = attested_report.build_runtime_dependency_manifest(
                [attested_report.runtime_dependency_file(
                    "python_module", "numpy.runtime_extension",
                    non_yaml_dependency,
                )],
                [],
                coverage="loaded_nonstdlib_python_modules_and_declared_files",
            )

            snapshots = root / "snapshots"
            snapshots.mkdir()
            with mock.patch.object(dmap_sweep, "REPO_ROOT", repository):
                before = dmap_sweep.create_report_source_snapshot(
                    snapshots, "before.tar.zst"
                )
                after = dmap_sweep.create_report_source_snapshot(
                    snapshots, "after.tar.zst"
                )
                report_source = dmap_sweep.publish_report_source_provenance(
                    staging, before, after,
                )
                loaded = attested_report.load_parent_schedule(schedule)
                attested_report.write_recovery_binding(
                    staging, schedule, loaded, report_source,
                )

                receipt = attested_report.run_trusted_finalizer_transaction(
                    staging, canonical, finalizer, finalizer_sha256,
                    ["campaign-finalize"], ["parity.json", "parity.csv"],
                    schedule, runtime_dependencies=runtime_manifest,
                )
                staged_validation = attested_report.validate_finalizer_receipt(
                    staging, canonical, require_receipt=True,
                )
                receipt_path = staging / attested_report.FINALIZER_RECEIPT_FILE
                receipt_bytes = receipt_path.read_bytes()
                receipt_path.unlink()
                with self.assertRaisesRegex(RuntimeError, "required but missing"):
                    attested_report.promote_staged_report(
                        staging, canonical, archive,
                    )
                self.assertEqual(
                    attested_report.recursive_tree_identity(canonical), old_tree,
                )
                self.assertTrue(staging.is_dir())
                receipt_path.write_bytes(receipt_bytes)

                canonical_identity = canonical / "nested" / "identity.txt"
                canonical_identity.write_text("concurrent drift\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "canonical report drifted"):
                    attested_report.promote_staged_report(
                        staging, canonical, archive,
                    )
                self.assertTrue(staging.is_dir())
                self.assertFalse(archive.exists())
                canonical_identity.write_text("old canonical\n", encoding="utf-8")

                real_exchange = attested_report._atomic_exchange
                exchange_calls = 0

                def delete_receipt_during_exchange(left: Path, right: Path) -> None:
                    nonlocal exchange_calls
                    exchange_calls += 1
                    if exchange_calls == 1:
                        (left / attested_report.FINALIZER_RECEIPT_FILE).unlink()
                    real_exchange(left, right)

                with mock.patch.object(
                    attested_report, "_atomic_exchange",
                    side_effect=delete_receipt_during_exchange,
                ), self.assertRaisesRegex(RuntimeError, "required but missing"):
                    attested_report.promote_staged_report(
                        staging, canonical, archive,
                        require_finalizer_receipt=True,
                    )
                self.assertEqual(
                    attested_report.recursive_tree_identity(canonical), old_tree,
                )
                self.assertTrue(staging.is_dir())
                self.assertFalse(archive.exists())
                receipt_path.write_bytes(receipt_bytes)

                exchange_calls = 0

                def mutate_predecessor_during_exchange(left: Path, right: Path) -> None:
                    nonlocal exchange_calls
                    exchange_calls += 1
                    if exchange_calls == 1:
                        (right / "nested" / "identity.txt").write_text(
                            "exchange-window drift\n", encoding="utf-8",
                        )
                    real_exchange(left, right)

                with mock.patch.object(
                    attested_report, "_atomic_exchange",
                    side_effect=mutate_predecessor_during_exchange,
                ), self.assertRaisesRegex(RuntimeError, "canonical report drifted"):
                    attested_report.promote_staged_report(
                        staging, canonical, archive,
                        require_finalizer_receipt=True,
                    )
                self.assertTrue(staging.is_dir())
                self.assertFalse(archive.exists())
                canonical_identity.write_text("old canonical\n", encoding="utf-8")
                result = attested_report.promote_staged_report(
                    staging, canonical, archive, require_finalizer_receipt=True,
                )
                published_validation = attested_report.validate_finalizer_receipt(
                    canonical, canonical, require_receipt=True,
                )

            self.assertTrue(staged_validation["valid"])
            self.assertTrue(published_validation["valid"])
            self.assertTrue(result["replaced_nonempty_report"])
            self.assertEqual(
                attested_report.recursive_tree_identity(archive), old_tree,
            )
            self.assertEqual(
                receipt["canonical_tree_before"], receipt["canonical_tree_after"],
            )
            self.assertEqual(receipt["canonical_tree_before"], old_tree)
            self.assertEqual((canonical / "report_policy.json").read_bytes(), policy_before)
            self.assertEqual(
                (canonical / "report_manifest.json").read_bytes(), manifest_before,
            )
            inventory = json.loads(
                (canonical / "report_inventory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                inventory[attested_report.FINALIZER_INVENTORY_KEY]
                ["finalizer_receipt"],
                attested_report.FINALIZER_RECEIPT_FILE,
            )
            self.assertEqual(
                [row["path"] for row in inventory[
                    attested_report.FINALIZER_INVENTORY_KEY
                ]["approved_outputs"]],
                ["parity.csv", "parity.json"],
            )
            rendered_receipt = json.dumps(receipt, sort_keys=True)
            self.assertNotIn(str(root / "reports" / "03_master_staging_unique"), rendered_receipt)
            self.assertNotIn("/proc/self/fd/", rendered_receipt)
            self.assertIn(
                "${STAGED_REPORT_DIR}", receipt["finalizer"]["replay_command"],
            )
            self.assertEqual(
                receipt["finalizer"]["replay_scope"],
                "invocation_and_bounded_runtime_dependencies",
            )
            mismatched_replay = attested_report.trusted_finalizer_spec(
                finalizer, finalizer_sha256, ["different-finalizer-action"],
                ["parity.json", "parity.csv"], canonical,
            )
            with self.assertRaisesRegex(RuntimeError, "replay specification differs"):
                attested_report.validate_finalizer_receipt(
                    canonical, canonical, expected_spec=mismatched_replay,
                    require_receipt=True,
                )
            source_valid, _reason, validated_source = (
                dmap_sweep.validate_report_source_provenance(canonical)
            )
            self.assertTrue(source_valid)
            self.assertEqual(validated_source["sha256"], report_source["sha256"])

            parity_bytes = (canonical / "parity.json").read_bytes()
            (canonical / "parity.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "output drifted"):
                attested_report.validate_finalizer_receipt(
                    canonical, canonical, require_receipt=True,
                )
            (canonical / "parity.json").write_bytes(parity_bytes)
            attested_report.validate_finalizer_receipt(
                canonical, canonical, require_receipt=True,
            )

            non_yaml_dependency.write_bytes(b"runtime-extension-v2")
            with self.assertRaisesRegex(RuntimeError, "runtime dependency drifted"):
                attested_report.validate_finalizer_receipt(
                    canonical, canonical, require_receipt=True,
                )
            non_yaml_dependency.write_bytes(b"runtime-extension-v1")
            attested_report.validate_finalizer_receipt(
                canonical, canonical, require_receipt=True,
            )

            finalizer.unlink()
            schedule.unlink()
            portable_stage = root / "portable-stage"
            portable_bundle.stage_review_tree(canonical, portable_stage)
            portable_value = json.loads(
                (portable_stage / attested_report.FINALIZER_RECEIPT_FILE)
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                portable_value["schema_name"],
                portable_bundle.PORTABLE_FINALIZER_RECEIPT_SCHEMA_NAME,
            )
            self.assertNotIn(str(finalizer), json.dumps(portable_value))
            self.assertNotIn(str(schedule), json.dumps(portable_value))
            portable_receipt_path = (
                portable_stage / attested_report.FINALIZER_RECEIPT_FILE
            )
            portable_receipt_bytes = portable_receipt_path.read_bytes()
            portable_value["source_finalizer_sha256"] = "0" * 64
            portable_receipt_path.write_text(
                json.dumps(portable_value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                portable_bundle.BundleError, "self-digest is invalid",
            ):
                portable_bundle.validate_staged_review_tree(portable_stage)
            portable_receipt_path.write_bytes(portable_receipt_bytes)
            bundle_path = root / "share" / "review.tar.zst"
            portable_bundle.create_review_bundle(
                portable_stage, bundle_path, stage_report=False,
            )
            bundle_validation = portable_bundle.validate_review_bundle(bundle_path)
            self.assertGreater(bundle_validation.review_file_count, 0)

    def test_composite_finalizer_rebinds_inventory_before_receipt_tree(self) -> None:
        if dmap_sweep.source_snapshot.zstandard is None:
            raise unittest.SkipTest("zstandard is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repository"
            repository.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / "source.txt").write_text("bound source\n", encoding="utf-8")
            subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
            subprocess.run([
                "git", "-c", "user.name=Test", "-c",
                "user.email=test@example.invalid", "commit", "-q",
                "-m", "initial",
            ], cwd=repository, check=True)
            staging = root / "reports/composite-staging"
            canonical = root / "reports/composite"
            canonical.mkdir(parents=True)
            schedule, ledger = self.make_composite_report_fixture(root, staging)
            finalizer = root / "composite-finalizer.sh"
            finalizer.write_text(
                "#!/bin/sh\nset -eu\nstaged=\n"
                "while [ \"$#\" -gt 0 ]; do case \"$1\" in "
                "--staged-report-dir) staged=$2; shift 2 ;; "
                "--published-report-dir) shift 2 ;; *) exit 64 ;; esac; done\n"
                "printf '{\"valid\":true}\\n' > \"$staged/parity.json\"\n",
                encoding="utf-8",
            )
            finalizer.chmod(0o755)
            finalizer_sha = hashlib.sha256(finalizer.read_bytes()).hexdigest()
            snapshots = root / "snapshots"
            snapshots.mkdir()
            with mock.patch.object(dmap_sweep, "REPO_ROOT", repository):
                before = dmap_sweep.create_report_source_snapshot(
                    snapshots, "before.tar.zst",
                )
                after = dmap_sweep.create_report_source_snapshot(
                    snapshots, "after.tar.zst",
                )
                report_source = dmap_sweep.publish_report_source_provenance(
                    staging, before, after,
                )
                attested_report.write_recovery_binding(
                    staging, schedule, ledger, report_source,
                )
                initial_binding = json.loads((
                    staging / attested_report.predecessor_composite.ARTIFACT_BINDING_FILE
                ).read_text(encoding="utf-8"))
                receipt = attested_report.run_trusted_finalizer_transaction(
                    staging, canonical, finalizer, finalizer_sha, [], ["parity.json"],
                    schedule,
                )
                validation = attested_report.validate_finalizer_receipt(
                    staging, canonical, require_receipt=True,
                )
            refreshed = json.loads((
                staging / attested_report.predecessor_composite.ARTIFACT_BINDING_FILE
            ).read_text(encoding="utf-8"))
            current_inventory_sha = hashlib.sha256(
                (staging / "report_inventory.json").read_bytes()
            ).hexdigest()
            self.assertNotEqual(
                initial_binding["core_artifacts"]["report_inventory.json"]["sha256"],
                current_inventory_sha,
            )
            self.assertEqual(
                refreshed["core_artifacts"]["report_inventory.json"]["sha256"],
                current_inventory_sha,
            )
            self.assertTrue(validation["valid"])
            self.assertEqual(
                receipt["published_protected_tree"],
                attested_report.recursive_tree_identity(
                    staging, ["parity.json", attested_report.FINALIZER_RECEIPT_FILE],
                ),
            )
            finalizer.unlink()
            schedule.unlink()
            portable_validation = attested_report.validate_finalizer_receipt(
                staging, canonical, require_receipt=True,
                verify_live_bindings=False,
            )
            self.assertTrue(portable_validation["valid"])

    def test_evidence_context_identity_is_regular_file_bound_and_immutable(self) -> None:
        from test_dmap_report_model import make_evidence_context

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = root / "context.json"
            context.write_text(
                json.dumps(make_evidence_context(), sort_keys=True) + "\n",
                encoding="utf-8",
            )

            identity = attested_report.evidence_context_identity(context)

            self.assertEqual(identity["path"], str(context.resolve()))
            self.assertEqual(identity["size"], context.stat().st_size)
            self.assertEqual(len(identity["sha256"]), 64)
            attested_report.require_evidence_context_unchanged(context, identity)

            context.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed while the report"):
                attested_report.require_evidence_context_unchanged(context, identity)

    def test_evidence_context_identity_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(RuntimeError, "regular non-symlink"):
                attested_report.evidence_context_identity(link)

    def test_recovery_binding_records_parent_evidence_and_distinct_reporter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "schedule.json"
            ledger = {
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {
                    "sha256": "a" * 64,
                    "source_provenance": {"sha256": "b" * 64},
                },
                "status": "incomplete",
                "jobs": {
                    "job-a": {"status": "complete"},
                    "job-b": {"status": "pending"},
                },
                "report": {},
            }
            schedule.write_text(json.dumps(ledger), encoding="utf-8")
            loaded = attested_report.load_parent_schedule(schedule)
            report = root / "report"
            report.mkdir()

            recovery = attested_report.write_recovery_binding(
                report, schedule, loaded, {"sha256": "c" * 64}
            )

            binding = json.loads((report / "sweep_report_binding.json").read_text())
            self.assertEqual(
                binding["schema_version"], dmap_sweep.REPORT_BINDING_SCHEMA_VERSION
            )
            self.assertEqual(binding["schedule_identity_sha256"], "a" * 64)
            self.assertEqual(binding["completed_job_ids"], ["job-a"])
            self.assertEqual(binding["report_source_sha256"], "c" * 64)
            self.assertEqual(recovery["source_relation"], "distinct_attested_reporter")
            self.assertEqual(recovery["completed_job_ids"], ["job-a"])
            self.assertEqual(
                json.loads((report / "report_recovery_manifest.json").read_text()),
                recovery,
            )

    def test_parent_schedule_schema_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / "schedule.json"
            schedule.write_text(json.dumps({"schema_name": "wrong"}), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                attested_report.load_parent_schedule(schedule)

    def test_composite_recovery_writes_generation_time_artifact_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "composite.json"
            ledger = {
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {"sha256": "a" * 64},
                "composite": {
                    "schema_name": attested_report.predecessor_composite.COMPOSITE_SCHEMA_NAME,
                    "schema_version": attested_report.predecessor_composite.COMPOSITE_SCHEMA_VERSION,
                },
                "status": "complete",
                "jobs": {"job-a": {"status": "complete"}},
            }
            schedule.write_text(json.dumps(ledger), encoding="utf-8")
            report = root / "report"
            report.mkdir()
            anchored = {
                "schema_name": attested_report.predecessor_composite.ARTIFACT_BINDING_SCHEMA_NAME,
                "schema_version": 1,
                "binding_sha256": "d" * 64,
            }
            with mock.patch.object(
                attested_report.predecessor_composite,
                "build_report_artifact_binding",
                return_value=anchored,
            ) as builder:
                attested_report.write_recovery_binding(
                    report, schedule, ledger, {"sha256": "c" * 64}
                )
            builder.assert_called_once_with(
                report, schedule, ledger, "c" * 64
            )
            self.assertEqual(
                json.loads(
                    (
                        report
                        / attested_report.predecessor_composite.ARTIFACT_BINDING_FILE
                    ).read_text()
                ),
                anchored,
            )

    def test_reused_composite_report_cannot_mint_missing_artifact_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = root / "composite.json"
            ledger = {
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {"sha256": "a" * 64},
                "composite": {
                    "schema_name": attested_report.predecessor_composite.COMPOSITE_SCHEMA_NAME,
                    "schema_version": attested_report.predecessor_composite.COMPOSITE_SCHEMA_VERSION,
                },
                "status": "complete",
                "jobs": {"job-a": {"status": "complete"}},
            }
            schedule.write_text(json.dumps(ledger), encoding="utf-8")
            report = root / "report"
            report.mkdir()

            with mock.patch.object(
                attested_report.predecessor_composite,
                "validate_report_artifact_binding",
                return_value=(False, "binding is missing", {}),
            ) as validator, mock.patch.object(
                attested_report.predecessor_composite,
                "build_report_artifact_binding",
            ) as builder:
                with self.assertRaisesRegex(RuntimeError, "generation-time artifact"):
                    attested_report.write_recovery_binding(
                        report,
                        schedule,
                        ledger,
                        {"sha256": "c" * 64},
                        create_composite_artifact_binding=False,
                    )

            validator.assert_called_once_with(
                report, schedule, ledger, "c" * 64
            )
            builder.assert_not_called()
            self.assertFalse(
                (
                    report
                    / attested_report.predecessor_composite.ARTIFACT_BINDING_FILE
                ).exists()
            )
            self.assertFalse((report / "sweep_report_binding.json").exists())
            self.assertFalse((report / "report_recovery_manifest.json").exists())

    def test_reuse_path_rejects_missing_or_mismatched_composite_binding(self) -> None:
        for reason in ("binding is missing", "generation binding is mismatched"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "config.yaml"
                config.write_text("schema_version: 2\n", encoding="utf-8")
                schedule = root / "composite.json"
                ledger = {
                    "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                    "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                    "identity": {"sha256": "a" * 64},
                    "composite": {
                        "schema_name": attested_report.predecessor_composite.COMPOSITE_SCHEMA_NAME,
                        "schema_version": attested_report.predecessor_composite.COMPOSITE_SCHEMA_VERSION,
                    },
                    "status": "complete",
                    "sessions": [],
                    "jobs": {"job-a": {"status": "complete"}},
                }
                schedule.write_text(json.dumps(ledger), encoding="utf-8")
                report = root / "report"
                report.mkdir()
                (report / "existing-report.md").write_text(
                    "# Existing report\n", encoding="utf-8"
                )
                source = {"sha256": "c" * 64}

                with mock.patch.object(
                    attested_report.dmap_sweep,
                    "valid_report",
                    return_value=(True, "valid report"),
                ), mock.patch.object(
                    attested_report.dmap_sweep,
                    "validate_report_source_provenance",
                    return_value=(True, "valid source", source),
                ), mock.patch.object(
                    attested_report.predecessor_composite,
                    "validate_report_artifact_binding",
                    return_value=(False, reason, {}),
                ) as validator, mock.patch.object(
                    attested_report.predecessor_composite,
                    "build_report_artifact_binding",
                ) as builder:
                    with self.assertRaisesRegex(
                        RuntimeError, "generation-time artifact"
                    ):
                        attested_report.run(
                            attested_report.Arguments(
                                config=config,
                                output_dir=report,
                                parent_schedule=schedule,
                            )
                        )

                validator.assert_called_once_with(
                    report, schedule, ledger, source["sha256"]
                )
                builder.assert_not_called()
                self.assertFalse(
                    (
                        report
                        / attested_report.predecessor_composite.ARTIFACT_BINDING_FILE
                    ).exists()
                )

    def test_running_parent_schedule_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schedule = Path(directory) / "schedule.json"
            schedule.write_text(json.dumps({
                "schema_name": dmap_sweep.SCHEDULE_SCHEMA_NAME,
                "schema_version": dmap_sweep.SCHEDULE_SCHEMA_VERSION,
                "identity": {"sha256": "a" * 64},
                "status": "running",
                "jobs": {"job-a": {"status": "complete"}},
            }), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "must be finalized"):
                attested_report.load_parent_schedule(schedule)


if __name__ == "__main__":
    unittest.main()
