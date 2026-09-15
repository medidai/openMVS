#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Preserve the caller's environment, including any pre-existing overrides.
with mock.patch.dict(os.environ):
    import cv2
import numpy as np


MODULE_PATH = Path(__file__).with_name("validate_dvp_depth_edge_prior.py")
SPEC = importlib.util.spec_from_file_location("validate_dvp_depth_edge_prior", MODULE_PATH)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DVPDepthEdgePriorValidatorTest(unittest.TestCase):
    def build_bundle(self, root: Path) -> tuple[Path, Path]:
        source_dir = root / "images"
        bundle_dir = root / "prior"
        frame_dir = bundle_dir / "frames" / "depth0001"
        source_dir.mkdir()
        frame_dir.mkdir(parents=True)
        source_path = source_dir / "frame.jpg"
        source_pixels = np.zeros((80, 120, 3), dtype=np.uint8)
        source_pixels[:, 60:] = 180
        assert cv2.imwrite(str(source_path), source_pixels)

        stages: dict[str, dict[str, str]] = {}
        labels = np.ones((43, 64), dtype=np.uint16)
        for stage_name in validator.STAGES:
            map_name = f"regions_{stage_name}.png"
            map_path = frame_dir / map_name
            assert cv2.imwrite(str(map_path), labels)
            stages[stage_name] = {"label_map": map_name, "sha256": sha256_file(map_path)}
        manifest = {
            "schema": validator.SCHEMA,
            "schema_version": validator.SCHEMA_VERSION,
            "complete": True,
            "depth_id": 1,
            "direct_depth_use": "forbidden_topology_only",
            "source_image": {
                "name": source_path.name,
                "width": 120,
                "height": 80,
                "sha256": sha256_file(source_path),
            },
            "processing_image": {
                "width": 64,
                "height": 43,
                "scale": 64 / 120,
                "requested_resolution_level": 0,
                "effective_resolution_level": 0,
                "min_resolution": 32,
                "max_resolution": 64,
                "prepared_max_resolution": 64,
                "resize_interpolation": "cv::INTER_AREA",
                "geometry_contract": "OpenMVS_Image_RecomputeMaxResolution_then_ReloadImage",
            },
            "model": {
                "revision": "a" * 40,
                "checkpoint_sha256": "b" * 64,
            },
            "parameters": {"eta": 300},
            "region_component_filtering": {
                "policy": "components_with_area_lte_eta_become_boundary",
                "minimum_region_size_exclusive": 300,
                "raw_region_count": 1,
                "retained_region_count": 1,
                "discarded_region_count": 0,
                "retained_region_pixels": 43 * 64,
                "discarded_region_pixels": 0,
            },
            "stages": stages,
        }
        manifest_path = frame_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        bundle = {
            "schema": validator.SCHEMA,
            "schema_version": validator.SCHEMA_VERSION,
            "complete": True,
            "image_preparation": {
                "resolution_level": 0,
                "min_resolution": 32,
                "max_resolution": 64,
            },
            "frame_count": 1,
            "frames": [
                {
                    "depth_id": 1,
                    "manifest": "frames/depth0001/manifest.json",
                    "manifest_sha256": sha256_file(manifest_path),
                }
            ],
        }
        (bundle_dir / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        return bundle_dir, source_dir

    def test_valid_processing_geometry_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir, source_dir = self.build_bundle(Path(directory))
            result = validator.validate(validator.Args(bundle_dir, source_dir))
            self.assertTrue(result["valid"], result["errors"])

    def test_raw_resolution_labels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir, source_dir = self.build_bundle(Path(directory))
            manifest_path = bundle_dir / "frames" / "depth0001" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["processing_image"]["width"] = 120
            manifest["processing_image"]["height"] = 80
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            bundle_path = bundle_dir / "bundle.json"
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            bundle["frames"][0]["manifest_sha256"] = sha256_file(manifest_path)
            bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
            result = validator.validate(validator.Args(bundle_dir, source_dir))
            self.assertFalse(result["valid"])
            self.assertTrue(
                any("does not match OpenMVS" in error for error in result["errors"]),
                result["errors"],
            )


if __name__ == "__main__":
    unittest.main()
