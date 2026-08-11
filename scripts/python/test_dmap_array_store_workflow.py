#!/usr/bin/env python3
"""Focused tests for experiment-level observability array-store orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts.python.dmap_observability import array_store, array_store_workflow


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frame(depthmaps_root: Path, frame: str, image_id: int, *, manifest: bool = True) -> Path:
    frame_dir = depthmaps_root / frame
    frame_dir.mkdir(parents=True)
    (frame_dir / "summary.json").write_text(
        json.dumps({"image_id": image_id}) + "\n", encoding="utf-8"
    )
    if manifest:
        (frame_dir / "map_manifest.json").write_text(
            json.dumps(
                {
                    "schema_name": "openmvs.dmap.map_manifest",
                    "schema_version": 4,
                    "complete": True,
                    "maps": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return frame_dir


def run_scene(instrumentation_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        label="baseline",
        role="baseline",
        repeat=0,
        scene_id="scene-a",
        instrumentation_dir=instrumentation_dir,
        estimation_stage="photometric",
        geometric_iteration=None,
    )


def fake_convert(manifest_path: Path, output_path: Path, **_kwargs: object) -> object:
    output_path.mkdir(parents=True)
    catalog = {
        "schema_name": array_store.SCHEMA_NAME,
        "schema_version": array_store.SCHEMA_VERSION,
        "source_manifest": {"sha256": sha256(manifest_path)},
        "artifact_count": 2,
        "available_count": 2,
        "unavailable_count": 0,
        "uncompressed_bytes": 128,
        "complete": True,
    }
    (output_path / array_store.MANIFEST_NAME).write_text(
        json.dumps(catalog, sort_keys=True) + "\n", encoding="utf-8"
    )
    return SimpleNamespace(store_path=output_path)


def fake_validation(store_path: Path, **kwargs: object) -> array_store.ValidationResult:
    catalog = json.loads((store_path / array_store.MANIFEST_NAME).read_text(encoding="utf-8"))
    source_frame_dir = Path(str(kwargs["source_frame_dir"]))
    source_valid = catalog["source_manifest"]["sha256"] == sha256(
        source_frame_dir / "map_manifest.json"
    )
    errors = () if source_valid else ("source map manifest checksum mismatch",)
    return array_store.ValidationResult(
        store_path=store_path,
        valid=source_valid,
        errors=errors,
        warnings=(),
        artifact_count=2,
        available_count=2,
        unavailable_count=0,
        uncompressed_bytes=128,
        stored_bytes=sum(item.stat().st_size for item in store_path.rglob("*") if item.is_file()),
    )


class ArrayStoreWorkflowTest(unittest.TestCase):
    def test_filtered_conversions_merge_into_relative_deterministic_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            experiment = base / "experiment"
            config = base / "config.yaml"
            config.write_text("schema_version: 2\n", encoding="utf-8")
            instrumentation = base / "external-capture"
            depthmaps = instrumentation / "depthmaps"
            write_frame(depthmaps, "0001_camera", 1)
            write_frame(depthmaps, "0002_camera", 2)
            scene = run_scene(instrumentation)

            with (
                mock.patch.object(
                    array_store_workflow.array_store,
                    "convert_map_manifest",
                    side_effect=fake_convert,
                ) as convert,
                mock.patch.object(
                    array_store_workflow.array_store,
                    "validate_store",
                    side_effect=fake_validation,
                ),
            ):
                first = array_store_workflow.materialize_experiment_array_stores(
                    experiment_root=experiment,
                    source_config=config,
                    run_scenes=[scene],
                    frames=["1"],
                )
                self.assertEqual(first.created_count, 1)
                self.assertEqual(first.indexed_count, 1)
                second = array_store_workflow.materialize_experiment_array_stores(
                    experiment_root=experiment,
                    source_config=config,
                    run_scenes=[scene],
                    frames=["0002_camera"],
                )
                self.assertEqual(second.created_count, 1)
                self.assertEqual(second.indexed_count, 2)
                index_before = second.index_path.read_bytes()
                third = array_store_workflow.materialize_experiment_array_stores(
                    experiment_root=experiment,
                    source_config=config,
                    run_scenes=[scene],
                    frames=["1"],
                )

            self.assertEqual(convert.call_count, 2)
            self.assertEqual(third.created_count, 0)
            self.assertEqual(third.validated_count, 1)
            self.assertEqual(index_before, third.index_path.read_bytes())
            index = json.loads(third.index_path.read_text(encoding="utf-8"))
            self.assertEqual(index["schema_name"], array_store_workflow.SCHEMA_NAME)
            self.assertEqual(index["schema_version"], 1)
            self.assertEqual(index["store_count"], 2)
            self.assertTrue(index["complete"])
            for row in index["stores"]:
                self.assertFalse(Path(row["store"]).is_absolute())
                self.assertFalse(Path(row["source_manifest"]).is_absolute())
                self.assertTrue((experiment / row["store"]).is_dir())
            self.assertEqual(
                list(third.index_path.parent.glob(f".{third.index_path.name}.*.tmp")),
                [],
            )

    def test_invalid_existing_store_fails_without_republishing_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            experiment = base / "experiment"
            config = base / "config.yaml"
            config.write_text("schema_version: 2\n", encoding="utf-8")
            instrumentation = base / "capture"
            frame_dir = write_frame(instrumentation / "depthmaps", "0001_camera", 1)
            scene = run_scene(instrumentation)
            with (
                mock.patch.object(
                    array_store_workflow.array_store,
                    "convert_map_manifest",
                    side_effect=fake_convert,
                ),
                mock.patch.object(
                    array_store_workflow.array_store,
                    "validate_store",
                    side_effect=fake_validation,
                ),
            ):
                result = array_store_workflow.materialize_experiment_array_stores(
                    experiment_root=experiment,
                    source_config=config,
                    run_scenes=[scene],
                )
            index_before = result.index_path.read_bytes()
            manifest = json.loads((frame_dir / "map_manifest.json").read_text(encoding="utf-8"))
            manifest["complete"] = False
            (frame_dir / "map_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with mock.patch.object(
                array_store_workflow.array_store,
                "validate_store",
                side_effect=fake_validation,
            ):
                with self.assertRaisesRegex(
                    array_store_workflow.ArrayStoreWorkflowError,
                    "source manifest checksum",
                ):
                    array_store_workflow.materialize_experiment_array_stores(
                        experiment_root=experiment,
                        source_config=config,
                        run_scenes=[scene],
                    )

            self.assertEqual(result.index_path.read_bytes(), index_before)

    def test_discovery_rejects_unknown_selectors_and_missing_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instrumentation = Path(directory) / "capture"
            write_frame(instrumentation / "depthmaps", "0001_camera", 1, manifest=False)
            scene = run_scene(instrumentation)
            with self.assertRaisesRegex(
                array_store_workflow.ArrayStoreWorkflowError,
                "run selector",
            ):
                array_store_workflow.discover_frame_sources([scene], runs=["typo"])
            with self.assertRaisesRegex(
                array_store_workflow.ArrayStoreWorkflowError,
                "frame selector",
            ):
                array_store_workflow.discover_frame_sources([scene], frames=["99"])
            with self.assertRaisesRegex(
                array_store_workflow.ArrayStoreWorkflowError,
                "no authoritative map_manifest",
            ):
                array_store_workflow.discover_frame_sources([scene], frames=["1"])

    def test_optional_dependency_error_remains_actionable_and_index_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            experiment = base / "experiment"
            config = base / "config.yaml"
            config.write_text("schema_version: 2\n", encoding="utf-8")
            instrumentation = base / "capture"
            write_frame(instrumentation / "depthmaps", "0001_camera", 1)
            with mock.patch.object(
                array_store_workflow.array_store,
                "convert_map_manifest",
                side_effect=array_store.OptionalDependencyError(
                    "Zarr v3 support is required; install requirements-dmap-array-store.txt"
                ),
            ):
                with self.assertRaisesRegex(
                    array_store.OptionalDependencyError,
                    "requirements-dmap-array-store.txt",
                ):
                    array_store_workflow.materialize_experiment_array_stores(
                        experiment_root=experiment,
                        source_config=config,
                        run_scenes=[run_scene(instrumentation)],
                    )
            self.assertFalse((experiment / array_store_workflow.INDEX_RELATIVE).exists())


if __name__ == "__main__":
    unittest.main()
