#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_drilldown


class DrilldownRequestTests(unittest.TestCase):
    def make_config(self, path: Path) -> dict[str, object]:
        config: dict[str, object] = {
            "schema_version": 2,
            "experiment_id": "experiment",
            "instrumentation": {
                "max_trace_pixels_per_request": 16,
                "expected_width": 20,
                "expected_height": 10,
            },
            "runs": [
                {"label": "base", "role": "baseline", "densify_args": ["--iters", "1"]},
                {
                    "label": "variant_b",
                    "role": "variant",
                    "densify_args": ["--iters", "5"],
                    "ini_overrides": {"PatchMatch CUDA View Samples": "24"},
                },
                {"label": "variant_a", "role": "variant", "densify_args": ["--iters", "3"]},
            ],
        }
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return config

    def build(self, config: dict[str, object], path: Path, **kwargs: object) -> dict[str, object]:
        return dmap_drilldown.build_request(
            config=config,
            config_path=path,
            scene_id="scene",
            image_id=7,
            source_revision="0123456789abcdef",
            source_dirty=False,
            **kwargs,
        )

    def test_repeated_pixels_are_deduplicated_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)

            first = self.build(
                config, config_path,
                pixel_values=["10,4", "2,3", "10,4"],
                variants=["variant_b"],
            )
            second = self.build(
                config, config_path,
                pixel_values=["2,3", "10,4"],
                variants=["variant_b"],
            )

            self.assertEqual(first, second)
            self.assertEqual(first["capture_profile"], "trace")
            self.assertEqual(first["capture"]["instrumentation_level"], "maps")
            self.assertTrue(first["capture"]["write_maps"])
            self.assertEqual(first["capture"]["process_specialization"], "Process<true>")
            self.assertFalse(first["capture"]["compact_exact_trace_available"])
            self.assertEqual(first["target"]["pixels"], [{"x": 2, "y": 3}, {"x": 10, "y": 4}])
            self.assertEqual([run["label"] for run in first["runs"]], ["base", "variant_b"])
            self.assertEqual(
                first["runs"][1]["ini_overrides"],
                {"PatchMatch CUDA View Samples": "24"},
            )

    def test_no_pixel_selection_requests_full_frame_deep_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)

            request = self.build(config, config_path)

            self.assertEqual(request["capture_profile"], "deep")
            self.assertEqual(request["target"]["trace_pixel_count"], 0)
            self.assertTrue(request["capture"]["write_maps"])
            self.assertEqual(
                [run["label"] for run in request["runs"]],
                ["base", "variant_a", "variant_b"],
            )

    def test_roi_expands_in_row_major_order_and_enforces_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            request = self.build(config, config_path, roi_value="3,5,2,2")

            self.assertEqual(
                dmap_drilldown.expand_trace_pixels(request),
                [
                    {"x": 3, "y": 5}, {"x": 4, "y": 5},
                    {"x": 3, "y": 6}, {"x": 4, "y": 6},
                ],
            )
            with self.assertRaisesRegex(ValueError, "exceeding the configured limit"):
                self.build(config, config_path, roi_value="0,0,5,5")

    def test_pixel_and_roi_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)

            with self.assertRaisesRegex(ValueError, "not both"):
                self.build(config, config_path, pixel_values=["1,2"], roi_value="0,0,2,2")

    def test_scene_specific_extent_precedes_global_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            instrumentation = config["instrumentation"]
            assert isinstance(instrumentation, dict)
            instrumentation["expected_extent_by_scene"] = {
                "scene": {"width": 8, "height": 6},
                "wide_scene": {"width": 24, "height": 12},
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            request = self.build(config, config_path, pixel_values=["7,5"])
            self.assertEqual(request["target"]["pixels"], [{"x": 7, "y": 5}])
            with self.assertRaisesRegex(ValueError, "configured 8x6"):
                self.build(config, config_path, pixel_values=["8,5"])
            with self.assertRaisesRegex(ValueError, "configured 8x6"):
                self.build(config, config_path, roi_value="6,4,3,2")

            wide_request = dmap_drilldown.build_request(
                config=config,
                config_path=config_path,
                scene_id="wide_scene",
                image_id=7,
                pixel_values=["23,11"],
                source_revision="0123456789abcdef",
                source_dirty=False,
            )
            self.assertEqual(
                wide_request["target"]["pixels"],
                [{"x": 23, "y": 11}],
            )

    def test_missing_scene_specific_extent_uses_global_extent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            instrumentation = config["instrumentation"]
            assert isinstance(instrumentation, dict)
            instrumentation["expected_extent_by_scene"] = {
                "another_scene": {"width": 8, "height": 6},
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            request = self.build(config, config_path, pixel_values=["19,9"])
            self.assertEqual(request["target"]["pixels"], [{"x": 19, "y": 9}])
            with self.assertRaisesRegex(ValueError, "configured 20x10"):
                self.build(config, config_path, pixel_values=["20,9"])

    def test_malformed_scene_specific_extent_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            instrumentation = config["instrumentation"]
            assert isinstance(instrumentation, dict)
            instrumentation["expected_extent_by_scene"] = {"scene": [8, 6]}
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"expected_extent_by_scene\['scene'\]"):
                self.build(config, config_path, pixel_values=["1,1"])

    def test_request_file_is_immutable_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config = self.make_config(config_path)
            request = self.build(config, config_path, pixel_values=["1,2"])

            first = dmap_drilldown.write_immutable_request(root / "drilldowns", request)
            second = dmap_drilldown.write_immutable_request(root / "drilldowns", request)

            self.assertEqual(first, second)
            self.assertEqual(dmap_drilldown.load_request(first), request)
            damaged = yaml.safe_load(first.read_text(encoding="utf-8"))
            damaged["target"]["image_id"] = 8
            first.write_text(yaml.safe_dump(damaged, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                dmap_drilldown.load_request(first)

    def test_invalid_pixel_and_variant_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)

            with self.assertRaisesRegex(ValueError, "expected X,Y"):
                self.build(config, config_path, pixel_values=["1:2"])
            with self.assertRaisesRegex(ValueError, "unknown drill-down variant"):
                self.build(config, config_path, variants=["missing"])
            with self.assertRaisesRegex(ValueError, "outside the configured"):
                self.build(config, config_path, pixel_values=["20,2"])


if __name__ == "__main__":
    unittest.main()
