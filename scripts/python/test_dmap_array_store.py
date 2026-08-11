#!/usr/bin/env python3
"""Tests for the canonical depth-map observability array store."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.python.dmap_observability import array_store


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


HAS_ZARR = has_module("zarr")
HAS_PILLOW = has_module("PIL.Image")


def tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def write_fixture(
    root: Path, *, missing: bool = False
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    frame = root / "frame"
    frame.mkdir()
    cost = np.asarray(
        [[0.5, 0.25, np.nan, 1.0], [0.75, 0.1, 0.4, 0.2], [0.6, 0.3, 0.8, 0.9]],
        dtype=np.float32,
    )
    labels = np.arange(12, dtype=np.uint8).reshape(3, 4)
    array_store.write_pfm(frame / "logical_states" / "cost_textureless.pfm", cost)
    eligible = (np.arange(12).reshape(3, 4) % 2).astype(np.float32)
    array_store.write_pfm(
        frame / "events" / "low_texture_update_eligible_exact.pfm", eligible
    )
    if not missing:
        Image = __import__("PIL.Image", fromlist=["Image"])
        Image.fromarray(labels).save(frame / "events" / "update_source.png")
    maps = [
        {
            "signal": "cost_textureless_experiment",
            "path": "logical_states/cost_textureless.pfm",
            "bytes": (frame / "logical_states" / "cost_textureless.pfm").stat().st_size,
            "dtype": "float32",
            "role": "logical_state",
            "logical_iteration": 2,
            "stage": "iteration",
            "measurement_quality": "exact",
            "mechanism": "texture",
            "quantity": "cost_contribution",
            "units": "cost",
            "component_id": "textureless",
            "description": "Synthetic textureless cost contribution.",
        },
        {
            "signal": "candidate_update_source",
            "path": "events/update_source.png",
            "bytes": (frame / "events" / "update_source.png").stat().st_size if not missing else 0,
            "dtype": "uint8",
            "role": "logical_event",
            "logical_iteration": 2,
            "stage": "iteration",
            "measurement_quality": "exact",
        },
        {
            "signal": "low_texture_update_eligible_exact",
            "path": "events/low_texture_update_eligible_exact.pfm",
            "bytes": (frame / "events" / "low_texture_update_eligible_exact.pfm").stat().st_size,
            "dtype": "float32",
            "role": "logical_event",
            "logical_iteration": 2,
            "stage": "iteration",
            "measurement_quality": "exact",
        },
    ]
    manifest = {
        "schema_name": "openmvs.dmap.map_manifest",
        "schema_version": 4,
        "complete": not missing,
        "width": 4,
        "height": 3,
        "map_granularity": "logical_iteration",
        "maps": maps,
    }
    manifest_path = frame / "map_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, cost, labels, eligible


@unittest.skipUnless(HAS_ZARR and HAS_PILLOW, "Zarr v3 and Pillow are optional test dependencies")
class ArrayStoreIntegrationTest(unittest.TestCase):
    def test_convert_read_validate_export_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, cost, labels, eligible = write_fixture(root)
            source_checksums = {
                path.relative_to(manifest_path.parent): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in manifest_path.parent.rglob("*")
                if path.is_file()
            }
            first = root / "first.zarr"
            second = root / "second.zarr"

            result = array_store.convert_map_manifest(
                manifest_path,
                first,
                chunk_size=2,
                shard_size=4,
            )
            array_store.convert_map_manifest(
                manifest_path,
                second,
                chunk_size=2,
                shard_size=4,
            )

            self.assertEqual(result.artifact_count, 3)
            self.assertEqual(result.unavailable_count, 0)
            self.assertEqual(tree_digest(first), tree_digest(second))
            self.assertEqual(
                source_checksums,
                {
                    path.relative_to(manifest_path.parent): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in manifest_path.parent.rglob("*")
                    if path.is_file()
                },
            )
            validation = array_store.validate_store(
                first,
                source_frame_dir=manifest_path.parent,
            )
            self.assertTrue(validation.valid, validation.errors)
            self.assertEqual(validation.available_count, 3)

            store = array_store.ArrayStore(first)
            cost_rows = store.find(
                signal="cost_textureless_experiment",
                logical_iteration=2,
            )
            self.assertEqual(len(cost_rows), 1)
            descriptor = cost_rows[0]["signal_descriptor"]
            self.assertEqual(descriptor["mechanism"], "texture")
            self.assertEqual(descriptor["component_id"], "textureless")
            self.assertEqual(descriptor["description"], "Synthetic textureless cost contribution.")
            np.testing.assert_array_equal(store.read(cost_rows[0]["artifact_id"]), cost)
            np.testing.assert_array_equal(store.read("map_000001"), labels)
            hysteresis_rows = store.find(signal="low_texture_update_eligible_exact")
            self.assertEqual(len(hysteresis_rows), 1)
            self.assertEqual(
                hysteresis_rows[0]["signal_descriptor"]["component_id"],
                "low_texture_update_hysteresis",
            )
            np.testing.assert_array_equal(store.read(hysteresis_rows[0]["artifact_id"]), eligible)

            pfm_path = root / "export" / "cost.pfm"
            pfm_result = array_store.export_artifact(first, "map_000000", pfm_path)
            self.assertTrue(pfm_result.exact)
            np.testing.assert_array_equal(array_store.read_pfm(pfm_path), cost)
            png_path = root / "export" / "cost.png"
            png_result = array_store.export_artifact(
                first,
                "map_000000",
                png_path,
                minimum=0.0,
                maximum=1.0,
            )
            self.assertFalse(png_result.exact)
            self.assertEqual(png_result.transform, "linear_min_max_to_uint8")
            self.assertEqual(png_result.invalid_pixels, 1)
            exact_png = root / "export" / "labels.png"
            label_result = array_store.export_artifact(first, "map_000001", exact_png)
            self.assertTrue(label_result.exact)
            Image = __import__("PIL.Image", fromlist=["Image"])
            with Image.open(exact_png) as image:
                np.testing.assert_array_equal(np.asarray(image), labels)

            zarr_metadata = json.loads(
                (first / "arrays" / "map_000000" / "zarr.json").read_text(encoding="utf-8")
            )
            self.assertEqual(zarr_metadata["zarr_format"], 3)
            self.assertEqual(zarr_metadata["codecs"][0]["name"], "sharding_indexed")
            inner_codecs = zarr_metadata["codecs"][0]["configuration"]["codecs"]
            self.assertEqual(inner_codecs[-1]["name"], "zstd")
            self.assertTrue(inner_codecs[-1]["configuration"]["checksum"])

    def test_refuses_overwrite_and_leaves_existing_output_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _, _ = write_fixture(root)
            output = root / "existing.zarr"
            output.mkdir()
            marker = output / "owner.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(array_store.ArrayStoreError, "refusing to overwrite"):
                array_store.convert_map_manifest(manifest_path, output)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_incomplete_source_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _, _ = write_fixture(root, missing=True)
            output = root / "incomplete.zarr"

            with self.assertRaisesRegex(array_store.ArrayStoreError, "source map manifest is incomplete"):
                array_store.convert_map_manifest(manifest_path, output)
            result = array_store.convert_map_manifest(
                manifest_path,
                output,
                allow_incomplete=True,
            )

            self.assertEqual(result.unavailable_count, 1)
            validation = array_store.validate_store(output)
            self.assertTrue(validation.valid, validation.errors)
            store = array_store.ArrayStore(output)
            unavailable = store.find(availability="unavailable")
            self.assertEqual(len(unavailable), 1)
            self.assertEqual(unavailable[0]["unavailable_reason"], "source map does not exist")
            with self.assertRaisesRegex(array_store.ArrayStoreError, "is unavailable"):
                store.read(unavailable[0]["artifact_id"])

    def test_validator_detects_catalog_checksum_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _, _ = write_fixture(root)
            output = root / "store.zarr"
            array_store.convert_map_manifest(manifest_path, output)
            catalog_path = output / array_store.MANIFEST_NAME
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            catalog["artifacts"][0]["array_sha256"] = "0" * 64
            catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

            validation = array_store.validate_store(output)

            self.assertFalse(validation.valid)
            self.assertTrue(any("array checksum mismatch" in error for error in validation.errors))

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _, _ = write_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["maps"][0]["path"] = "../outside.pfm"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(array_store.ArrayStoreError, "remain within"):
                array_store.convert_map_manifest(manifest_path, root / "store.zarr")


class ArrayStoreDependencyTest(unittest.TestCase):
    def test_reads_openmvs_multichannel_pfm_convention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.pfm"
            values = np.asarray(
                [
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
                ],
                dtype=np.float32,
            )
            with path.open("wb") as stream:
                stream.write(b"Pf\n2 2\n-3.000000\n")
                np.flipud(values).astype("<f4").tofile(stream)

            decoded = array_store.read_pfm(path)

            np.testing.assert_array_equal(decoded, values)

    def test_zarr_dependency_error_is_actionable(self) -> None:
        with mock.patch.object(array_store.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(
                array_store.OptionalDependencyError,
                "requirements-dmap-array-store.txt",
            ):
                array_store._require_zarr()

    def test_pillow_dependency_error_is_actionable(self) -> None:
        with mock.patch.object(array_store.importlib, "import_module", side_effect=ImportError("missing")):
            with self.assertRaisesRegex(
                array_store.OptionalDependencyError,
                "Pillow is required",
            ):
                array_store._require_pillow()


if __name__ == "__main__":
    unittest.main()
