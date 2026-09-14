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
            "scenes": [{"scan_id": "scene"}],
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

    def test_trace_layout_matches_half_up_scaling_bounds_and_first_wins_dedup(self) -> None:
        requested = [(120, 80), (119, 80)]

        self.assertEqual(
            dmap_drilldown.trace_pyramid_layout(
                requested, 1, width=60, height=100
            ),
            [],
        )
        self.assertEqual(
            dmap_drilldown.trace_pyramid_layout(
                requested, 1, width=61, height=100
            ),
            [dmap_drilldown.TracePyramidSlot(
                request_indices=(0, 1),
                requested_coordinates=((120, 80), (119, 80)),
                coordinate=(60, 40),
            )],
        )

    def test_trace_layout_matches_float32_boundaries_and_rejects_unsafe_int32_edge(self) -> None:
        self.assertEqual(
            dmap_drilldown.trace_pyramid_layout(
                [(16_777_216, 0), (16_777_217, 0)], 0
            ),
            [dmap_drilldown.TracePyramidSlot(
                request_indices=(0, 1),
                requested_coordinates=((16_777_216, 0), (16_777_217, 0)),
                coordinate=(16_777_216, 0),
            )],
        )
        # 16,777,218 / 2 is 8,388,609, but adding 0.5f at that
        # magnitude rounds to the next even float exactly as C++ does.
        self.assertEqual(
            dmap_drilldown.trace_pyramid_layout([(16_777_218, 0)], 1)[0].coordinate,
            (8_388_610, 0),
        )
        with self.assertRaisesRegex(ValueError, "safe C\\+\\+ float32-to-int"):
            dmap_drilldown.trace_pyramid_layout(
                [(dmap_drilldown.MAX_SAFE_TRACE_COORDINATE + 1, 0)], 0
            )
        with self.assertRaisesRegex(ValueError, "safe C\\+\\+ float32-to-int"):
            dmap_drilldown.parse_pixel(
                f"{dmap_drilldown.MAX_SAFE_TRACE_COORDINATE + 1},0"
            )
        with self.assertRaisesRegex(ValueError, "safe C\\+\\+ float32-to-int"):
            dmap_drilldown.parse_roi(
                f"{dmap_drilldown.MAX_SAFE_TRACE_COORDINATE},0,2,1"
            )

        oversized_roi = {
            "target": {
                "roi": {"x": 0, "y": 0, "width": 4097, "height": 4097}
            }
        }
        with self.assertRaisesRegex(ValueError, "more than 4096 pixels"):
            dmap_drilldown.expand_trace_pixels(oversized_roi)

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

    def test_trace_request_is_admitted_by_worst_case_report_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            instrumentation = config["instrumentation"]
            assert isinstance(instrumentation, dict)
            instrumentation["max_trace_pixels_per_request"] = 500
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            pixel_values = [f"{index % 20},{index // 20}" for index in range(129)]

            request = self.build(
                config, config_path,
                pixel_values=pixel_values[:128],
                variants=["variant_b"],
            )
            self.assertEqual(request["capture"]["trace_row_upper_bound"], 4096)
            self.assertEqual(
                request["capture"]["trace_row_limit"],
                dmap_drilldown.MAX_TRACE_REPORT_ROWS,
            )
            self.assertEqual(
                dmap_drilldown.validate_trace_row_admission(config, request), 4096
            )
            legacy_request = yaml.safe_load(yaml.safe_dump(request))
            legacy_request["capture"].pop("trace_row_upper_bound")
            legacy_request["capture"].pop("trace_row_limit")
            self.assertEqual(
                dmap_drilldown.validate_trace_row_admission(config, legacy_request),
                4096,
            )
            with self.assertRaisesRegex(ValueError, "up to 4128 rows"):
                self.build(
                    config, config_path,
                    pixel_values=pixel_values,
                    variants=["variant_b"],
                )

            instrumentation["max_trace_rows_per_request"] = 5000
            with self.assertRaisesRegex(ValueError, r"must be in \[1,4096\]"):
                self.build(
                    config, config_path,
                    pixel_values=["1,1"],
                    variants=["variant_b"],
                )

    def test_zero_photometric_iters_still_accounts_for_geometric_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            config["instrumentation"] = {
                "max_trace_pixels_per_request": 500,
                "expected_width": 400,
                "expected_height": 1,
            }
            config["runs"] = [
                {
                    "label": "base",
                    "role": "baseline",
                    "densify_args": [
                        "--iters", "0", "--geometric-iters", "2",
                        "--sub-resolution-levels", "2",
                    ],
                },
                {
                    "label": "variant",
                    "role": "variant",
                    "densify_args": [
                        "--iters", "0", "--geometric-iters", "2",
                        "--sub-resolution-levels", "2",
                    ],
                },
            ]
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            pixels = [f"{index},0" for index in range(350)]

            self.assertEqual(
                dmap_drilldown.trace_row_upper_bound(
                    config, config["runs"], len(pixels)
                ),
                4900,
            )
            with self.assertRaisesRegex(ValueError, "up to 4900 rows"):
                self.build(
                    config,
                    config_path,
                    pixel_values=pixels,
                    variants=["variant"],
                )

    def test_scene_argument_overrides_are_included_in_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            scene = config["scenes"][0]
            assert isinstance(scene, dict)
            scene["argument_overrides"] = {
                "--iters": "10",
                "--geometric-iters": "3",
                "--sub-resolution-levels": "1",
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            request = self.build(
                config,
                config_path,
                pixel_values=["1,1"],
                variants=["variant_b"],
            )
            self.assertEqual(request["capture"]["trace_row_upper_bound"], 56)
            self.assertEqual(
                dmap_drilldown.validate_trace_row_admission(config, request), 56
            )
            self.assertEqual(
                dmap_drilldown.validate_trace_row_admission(
                    config,
                    request,
                    argument_overrides=scene["argument_overrides"],
                ),
                56,
            )

            changed_overrides = dict(scene["argument_overrides"])
            changed_overrides["--iters"] = "20"
            with self.assertRaisesRegex(ValueError, "declaration does not match"):
                dmap_drilldown.validate_trace_row_admission(
                    config,
                    request,
                    argument_overrides=changed_overrides,
                )

            scene["argument_overrides"]["--iters"] = "1022"
            with self.assertRaisesRegex(ValueError, "up to 4104 rows"):
                self.build(
                    config,
                    config_path,
                    pixel_values=["1,1"],
                    variants=["variant_b"],
                )

    def test_resolved_scene_overrides_can_be_supplied_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            request = self.build(
                config,
                config_path,
                pixel_values=["1,1"],
                variants=["variant_b"],
                argument_overrides={
                    "--iters": "10",
                    "--geometric-iters": "3",
                    "--sub-resolution-levels": "1",
                },
            )
            self.assertEqual(request["capture"]["trace_row_upper_bound"], 56)

    def test_request_run_specs_are_bound_to_current_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config = self.make_config(config_path)
            request = self.build(
                config,
                config_path,
                pixel_values=["1,1"],
                variants=["variant_b"],
            )
            self.assertEqual(
                dmap_drilldown.validate_request_runs(config, request), request["runs"]
            )

            forged = yaml.safe_load(yaml.safe_dump(request))
            forged["runs"][1]["densify_args"] = ["--iters", "1"]
            with self.assertRaisesRegex(ValueError, "run specs do not match"):
                dmap_drilldown.validate_request_runs(config, forged)
            with self.assertRaisesRegex(ValueError, "run specs do not match"):
                dmap_drilldown.validate_trace_row_admission(config, forged)

    def test_external_program_options_file_is_rejected_for_trace_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"

            config = self.make_config(config_path)
            config["default_densify_args"] = ["--config-file", "Densify.cfg"]
            with self.assertRaisesRegex(ValueError, "uses --config-file"):
                self.build(config, config_path, pixel_values=["1,1"])

            config = self.make_config(config_path)
            config["runs"][0]["densify_args"] = ["--config-file=Densify.cfg"]
            with self.assertRaisesRegex(ValueError, "uses --config-file"):
                self.build(config, config_path, pixel_values=["1,1"])

            config = self.make_config(config_path)
            scene = config["scenes"][0]
            assert isinstance(scene, dict)
            scene["argument_overrides"] = {"--config-file": "Densify.cfg"}
            with self.assertRaisesRegex(ValueError, "uses --config-file"):
                self.build(config, config_path, pixel_values=["1,1"])

    def test_implicit_observer_config_is_rejected_when_working_folder_is_known(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            scene_root = root / "scene"
            scene_root.mkdir()
            mvs_path = scene_root / "scene.mvs"
            mvs_path.touch()
            config = self.make_config(config_path)
            scene = config["scenes"][0]
            assert isinstance(scene, dict)
            scene.update({
                "working_folder": str(scene_root),
                "mvs_file": str(mvs_path),
            })
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            expected = scene_root / f"{dmap_drilldown.OBSERVER_APP_NAME}.cfg"
            self.assertEqual(
                dmap_drilldown.validate_no_implicit_program_options_file(scene_root),
                expected,
            )
            expected.write_text("iters=500\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "implicit program-options file"):
                dmap_drilldown.validate_no_implicit_program_options_file(scene_root)
            with self.assertRaisesRegex(ValueError, "implicit program-options file"):
                self.build(config, config_path, pixel_values=["1,1"])

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

            request = self.build(config, config_path, pixel_values=["1,2"])
            request["target"]["trace_pixel_count"] = 2
            payload = dict(request)
            payload.pop("request_sha256")
            request["request_sha256"] = dmap_drilldown.request_digest(payload)
            with self.assertRaisesRegex(ValueError, "trace_pixel_count"):
                dmap_drilldown.validate_request(request)


if __name__ == "__main__":
    unittest.main()
