#!/usr/bin/env python3
"""Focused tests for line/plane DMAP annotation evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import report_dmap_annotation_fit as annotation_fit


def fit_config(**overrides: object) -> annotation_fit.FitConfig:
    values: dict[str, object] = {
        "ransac_threshold_m": 0.02,
        "ransac_thresholds_m": (0.005, 0.010, 0.020, 0.050),
        "edge_ribbon_px": 5.0,
        "max_ransac_points": 5000,
        "ransac_trials": 300,
        "seed": 7,
        "min_confidence": None,
        "plane_grid": 16,
        "line_bins": 50,
        "max_visual_points": 1000,
        "edge_ribbon_source_px": None,
        "edge_ribbon_angle_mrad": None,
    }
    values.update(overrides)
    return annotation_fit.FitConfig(**values)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class DMapAnnotationFitTests(unittest.TestCase):
    def test_load_dmap_decodes_current_d2_quantized_codec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth0000.dmap"
            image_name = b"images/00000.jpg"
            payload = b"".join([
                b"D2",
                np.asarray([15], dtype=np.uint8).tobytes(),
                np.asarray([2], dtype=np.int8).tobytes(),
                np.asarray([2, 1, 2, 1], dtype="<u4").tobytes(),
                np.asarray([0.5, 8.0, 2.0], dtype="<f4").tobytes(),
                np.asarray([len(image_name)], dtype="<u2").tobytes(),
                image_name,
                np.asarray([2], dtype="<u4").tobytes(),
                np.asarray([0, 3], dtype="<u4").tobytes(),
                np.eye(3, dtype="<f8").tobytes(),
                np.eye(3, dtype="<f8").tobytes(),
                np.asarray([1.0, 2.0, 3.0], dtype="<f8").tobytes(),
                np.asarray([1.0, 2.0], dtype="<f2").tobytes(),
                np.asarray([[0, 0], [-32768, -32768]], dtype="<i2").tobytes(),
                np.asarray([0, 255], dtype=np.uint8).tobytes(),
                np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8).tobytes(),
            ])
            path.write_bytes(payload)

            decoded = annotation_fit.load_dmap(path)

            self.assertEqual(decoded["format"], "D2")
            self.assertEqual(decoded["reference_view_id"], 0)
            self.assertEqual(decoded["neighbor_view_ids"], [3])
            np.testing.assert_array_equal(
                decoded["depth_map"], np.asarray([[4.0, 8.0]], dtype=np.float32)
            )
            np.testing.assert_allclose(
                decoded["normal_map"],
                np.asarray([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32),
            )
            np.testing.assert_allclose(
                decoded["confidence_map"], np.asarray([[0.0, 2.0]], dtype=np.float32)
            )
            np.testing.assert_array_equal(
                decoded["views_map"],
                np.asarray([[[0, 1, 2, 3], [4, 5, 6, 7]]], dtype=np.uint8),
            )

            path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(ValueError, "truncated DMAP views map"):
                annotation_fit.load_dmap(path)

    def test_public_annotation_sidecar_is_self_contained_and_scene_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotations.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_name": "openmvs.dmap.annotation_sidecar",
                        "schema_version": 1,
                        "scene_id": "scene-a",
                        "image_mapping": {"frame-a": "images/0001.jpg"},
                        "frames": [
                            {
                                "id": "frame-a",
                                "imageResolution": {"width": 640, "height": 480},
                            }
                        ],
                        "annotations": {
                            "controlEdges": [],
                            "controlPlanes": [],
                        },
                        "camera_final": {"width": 640, "height": 480},
                    }
                ),
                encoding="utf-8",
            )

            sidecar = annotation_fit.load_annotation_sidecar(path, "scene-a")

            self.assertEqual(sidecar["scene"]["id"], "scene-a")
            self.assertEqual(sidecar["image_mapping"], {"frame-a": "images/0001.jpg"})
            self.assertEqual(sidecar["review"]["frames"][0]["id"], "frame-a")
            self.assertEqual(sidecar["pipeline"]["camera_final"]["width"], 640)
            with self.assertRaisesRegex(ValueError, "does not match"):
                annotation_fit.load_annotation_sidecar(path, "scene-b")

    def test_required_thresholds_are_validated(self) -> None:
        annotation_fit.validate_fit_config(fit_config())
        with self.assertRaisesRegex(ValueError, "5/10/20/50"):
            annotation_fit.validate_fit_config(
                fit_config(ransac_thresholds_m=(0.005, 0.010, 0.020))
            )

    def test_source_pixel_ribbon_scales_to_depth_resolution(self) -> None:
        config = fit_config(edge_ribbon_px=None, edge_ribbon_source_px=8.0)
        dmap = {
            "K": np.asarray([[500.0, 0.0, 500.0], [0.0, 500.0, 375.0], [0.0, 0.0, 1.0]]),
            "depth_width": 1000,
            "depth_height": 750,
        }
        pipeline = {
            "camera_final": {"width": 4000, "height": 3000},
            "camera_undistorted": {"width": 4000, "height": 3000},
        }
        raw_segment = np.asarray([[1000.0, 1500.0], [3000.0, 1500.0]])

        radius, mode = annotation_fit.edge_ribbon_radius_depth_px(
            raw_segment=raw_segment,
            dmap=dmap,
            pipeline_row=pipeline,
            annotation_space="final",
            config=config,
        )

        self.assertAlmostEqual(radius, 2.0)
        self.assertEqual(mode, "annotation_source_pixels")

    def test_angular_ribbon_uses_dmap_focal_length(self) -> None:
        config = fit_config(edge_ribbon_px=None, edge_ribbon_angle_mrad=10.0)
        dmap = {"K": np.diag([400.0, 900.0, 1.0])}

        radius, mode = annotation_fit.edge_ribbon_radius_depth_px(
            raw_segment=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            dmap=dmap,
            pipeline_row={},
            annotation_space="final",
            config=config,
        )

        self.assertAlmostEqual(radius, np.tan(0.01) * 600.0)
        self.assertEqual(mode, "angular_mrad")

    def test_ransac_is_deterministic_for_same_annotation_seed(self) -> None:
        rng = np.random.default_rng(19)
        xy = rng.uniform(-1.0, 1.0, (500, 2))
        z = 0.3 * xy[:, 0] - 0.1 * xy[:, 1] + 2.0 + rng.normal(0.0, 0.003, 500)
        points = np.column_stack([xy, z])
        points[:20] += rng.normal(0.0, 0.2, (20, 3))
        seed = annotation_fit.stable_annotation_seed(
            11,
            scan_id="scan",
            frame_id="frame",
            kind="plane",
            object_id="object",
            chunk_id="chunk",
        )
        config = fit_config(seed=seed, max_ransac_points=250)

        first = annotation_fit.ransac_plane(points, config)
        second = annotation_fit.ransac_plane(points, config)

        self.assertEqual(first, second)
        self.assertEqual(first["fit_status"], "ok")
        for threshold_mm in (5, 10, 20, 50):
            self.assertIn(f"inlier_fraction_{threshold_mm}mm", first)

    def test_missing_annotation_row_keeps_identity_and_metric_schema(self) -> None:
        row = annotation_fit.missing_annotations_row(
            scan_id="scan-a",
            frame_id="frame-b",
            image_name="0001.jpg",
            thresholds_m=(0.005, 0.010, 0.020, 0.050),
            identity={"run_id": "variant", "repeat_id": "2", "stage": "postfilter"},
        )

        self.assertFalse(row["annotation_present"])
        self.assertEqual(row["fit_status"], "missing_annotations_for_frame")
        self.assertEqual(row["run_id"], "variant")
        self.assertEqual(row["repeat_id"], "2")
        self.assertEqual(row["stage"], "postfilter")
        self.assertIsNone(row["inlier_fraction_50mm"])

    def test_run_report_emits_missing_row_in_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "report"
            args = annotation_fit.build_parser().parse_args(
                [
                    "--dmap",
                    str(Path(temporary) / "depth0001.dmap"),
                    "--scan-id",
                    "scan-a",
                    "--output-dir",
                    str(output_dir),
                    "--run-id",
                    "baseline",
                    "--repeat-id",
                    "0",
                    "--stage",
                    "post_filter",
                ]
            )
            dmap = {
                "image_name": "0001.jpg",
                "reference_view_id": 1,
                "image_width": 4,
                "image_height": 3,
                "depth_width": 4,
                "depth_height": 3,
                "depth_map": np.ones((3, 4), dtype=np.float32),
                "K": np.eye(3),
                "R": np.eye(3),
                "C": np.zeros(3),
            }
            scan = {"id": "scan-a"}
            review = {"scan_id": "scan-a", "annotations": {}}
            pipeline = {
                "scan_id": "scan-a",
                "camera_final": {"width": 4, "height": 3},
            }
            with (
                mock.patch.object(annotation_fit, "load_dmap", return_value=dmap),
                mock.patch.object(annotation_fit, "load_scan_context", return_value=(scan, review, pipeline)),
                mock.patch.object(
                    annotation_fit,
                    "load_image_mapping",
                    return_value=({"frame-a": "0001.jpg"}, "fixture"),
                ),
            ):
                result = annotation_fit.run_report(args)

            row = result["annotations"][0]
            persisted = json.loads((output_dir / "metrics.json").read_text())["annotations"][0]
            csv_text = (output_dir / "annotation_metrics.csv").read_text()

        self.assertEqual(row["fit_status"], "missing_annotations_for_frame")
        self.assertEqual(persisted["run_id"], "baseline")
        self.assertIn("missing_annotations_for_frame", csv_text)

    def test_corpus_manifest_reports_ready_and_malformed_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            cache = Path(temporary) / "cache"
            scan_id = "scan-1"
            write_jsonl(
                root / "db" / "scans.jsonl",
                [{"id": scan_id}],
            )
            write_jsonl(
                root / "db" / "scan_pipelines.jsonl",
                [
                    {
                        "scan_id": scan_id,
                        "camera_distorted": {"camera_model": "RADIAL"},
                        "camera_final": {"width": 1000, "height": 750},
                    }
                ],
            )
            write_jsonl(
                root / "db" / "scan_reviewers.jsonl",
                [
                    {
                        "scan_id": scan_id,
                        "frames": [
                            {"id": "frame-ready", "imageResolution": {"width": 4000, "height": 3000}},
                            {"id": "frame-bad", "imageResolution": {"width": 4000, "height": 3000}},
                        ],
                        "annotations": {
                            "controlEdges": [
                                {
                                    "id": "edge-object",
                                    "chunks": [
                                        {
                                            "id": "edge-chunk",
                                            "frameId": "frame-ready",
                                            "start": {"x": 10.0, "y": 20.0},
                                            "end": {"x": 30.0, "y": 20.0},
                                        }
                                    ],
                                }
                            ],
                            "controlPlanes": [
                                {
                                    "id": "plane-object",
                                    "chunks": [
                                        {"id": "bad-chunk", "frameId": "frame-bad", "points": []}
                                    ],
                                }
                            ],
                        },
                    }
                ],
            )
            mapping = cache / scan_id / "image_id_mapping.json"
            mapping.parent.mkdir(parents=True)
            mapping.write_text("{}\n")
            (mapping.parent / "depth0001.dmap").write_bytes(b"fixture")

            manifest = annotation_fit.build_corpus_annotation_manifest(root, cache)

        self.assertEqual(manifest["summary"]["annotated_scans"], 1)
        self.assertEqual(manifest["summary"]["annotated_frames"], 2)
        self.assertEqual(manifest["summary"]["edge_chunks"], 1)
        self.assertEqual(manifest["summary"]["plane_chunks"], 1)
        self.assertEqual(manifest["summary"]["malformed_chunks"], 1)
        by_frame = {row["frame_id"]: row for row in manifest["frames"]}
        self.assertTrue(by_frame["frame-ready"]["evaluation_ready"])
        self.assertEqual(by_frame["frame-bad"]["audit_status"], "contains_malformed_chunks")


if __name__ == "__main__":
    unittest.main()
