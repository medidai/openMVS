#!/usr/bin/env python3
"""Focused regression tests for the depth-map development report tooling."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_dev
import validate_dmap_disabled
import validate_dmap_instrumentation


def write_depth_only_dmap(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    depth_min = float(finite.min()) if finite.size else 0.0
    depth_max = float(finite.max()) if finite.size else 0.0
    image_name = b"image.jpg"
    with path.open("wb") as handle:
        handle.write(b"DR")
        handle.write(np.asarray([1, 0], dtype=np.uint8).tobytes())
        handle.write(np.asarray([width, height], dtype=np.uint32).tobytes())
        handle.write(np.asarray([width, height], dtype=np.uint32).tobytes())
        handle.write(np.asarray([depth_min, depth_max], dtype=np.float32).tobytes())
        handle.write(np.asarray([len(image_name)], dtype=np.uint16).tobytes())
        handle.write(image_name)
        handle.write(np.asarray([1], dtype=np.uint32).tobytes())
        handle.write(np.asarray([0], dtype=np.uint32).tobytes())
        handle.write(np.eye(3, dtype=np.float64).tobytes())
        handle.write(np.eye(3, dtype=np.float64).tobytes())
        handle.write(np.zeros(3, dtype=np.float64).tobytes())
        handle.write(depth.tobytes())


def write_quantized_depth_only_dmap(
    path: Path, depth: np.ndarray, *, depth_exponent: int = 2
) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    height, width = depth.shape
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    depth_min = float(finite.min()) if finite.size else 0.0
    depth_max = float(finite.max()) if finite.size else 0.0
    image_name = b"image.jpg"
    scale = np.float32(2.0**depth_exponent)
    with path.open("wb") as handle:
        handle.write(b"D2")
        handle.write(np.asarray([1], dtype=np.uint8).tobytes())
        handle.write(np.asarray([depth_exponent], dtype=np.int8).tobytes())
        handle.write(np.asarray([width, height], dtype="<u4").tobytes())
        handle.write(np.asarray([width, height], dtype="<u4").tobytes())
        handle.write(np.asarray([depth_min, depth_max], dtype="<f4").tobytes())
        handle.write(np.asarray([1.0], dtype="<f4").tobytes())
        handle.write(np.asarray([len(image_name)], dtype="<u2").tobytes())
        handle.write(image_name)
        handle.write(np.asarray([1], dtype="<u4").tobytes())
        handle.write(np.asarray([0], dtype="<u4").tobytes())
        handle.write(np.eye(3, dtype="<f8").tobytes())
        handle.write(np.eye(3, dtype="<f8").tobytes())
        handle.write(np.zeros(3, dtype="<f8").tobytes())
        handle.write((depth / scale).astype("<f2").tobytes())


def write_pfm(path: Path, values: np.ndarray) -> None:
    data = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"PF\n" if data.ndim == 3 else b"Pf\n")
        handle.write(f"{data.shape[1]} {data.shape[0]}\n-1.0\n".encode("ascii"))
        np.flipud(data).astype("<f4").tofile(handle)


def write_fake_densify_binary(
    path: Path, *, observer: bool, version: str = "v1"
) -> None:
    observer_help = "" if not observer else """
printf '%s\n' '--dmap-instrumentation-dir arg'
printf '%s\n' '--dmap-instrumentation-level arg'
printf '%s\n' '--dmap-instrumentation-write-maps arg'
"""
    path.write_text(
        "#!/bin/sh\n"
        f"# fixture {version}\n"
        "printf '%s\\n' 'DensifyPointCloud options'\n"
        + observer_help
        + "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def make_exact_trace_frame(
    frame: Path,
    *,
    total_cost: np.ndarray,
    exact_gap: np.ndarray,
    view_mask: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    source_image_ids: tuple[int, ...] = (20, 21),
    estimation_stage: str = "geometric_consistency",
    geometric_iteration: int | None = 2,
) -> None:
    state = "logical_states/state01_iteration01"
    maps = [
        {
            "signal": "cost_total_production_exact",
            "path": f"{state}/cost_total_production_exact.pfm",
            "dtype": "float32", "role": "logical_state", "logical_iteration": 0,
            "stage": "iteration", "stage_index": 1, "measurement_quality": "exact",
            "measurement_basis": "production_hot_kernel_contribution_basis",
        },
        {
            "signal": "gap_winner_runner_up_exact",
            "path": f"{state}/gap_winner_runner_up_exact.pfm",
            "dtype": "float32", "role": "logical_state", "logical_iteration": 0,
            "stage": "iteration", "stage_index": 1, "measurement_quality": "exact",
            "measurement_basis": "production_hot_kernel_candidate_order",
            "unavailable_value": -1.0,
        },
        {
            "signal": "selected_views_after_mask_exact",
            "path": f"{state}/selected_views_after_mask_exact.png",
            "dtype": "uint8x4", "role": "logical_event", "logical_iteration": 0,
            "stage": "iteration", "stage_index": 1, "measurement_quality": "exact",
            "measurement_basis": "production_selected_view_mask",
            "encoding": "uint32 little-endian bytes in RGBA channels",
        },
        {
            "signal": "depth_final_after_filter",
            "path": "maps/depth_final_after_filter.pfm",
            "dtype": "float32", "role": "final_state", "measurement_quality": "exact",
            "measurement_basis": "production_post_filter_state",
        },
        {
            "signal": "valid_after_filter",
            "path": "maps/valid_after_filter.png",
            "dtype": "uint8", "role": "data", "measurement_quality": "derived_exact",
            "measurement_basis": "depth_final_after_filter>0",
        },
    ]
    for source_view_index, source_image_id in enumerate(source_image_ids):
        maps.append({
            "signal": "view_cost_components_exact",
            "path": f"{state}/view_cost_components_exact_view{source_view_index:02d}.pfm",
            "dtype": "float32x3", "role": "logical_view_state", "logical_iteration": 0,
            "stage": "iteration", "stage_index": 1, "measurement_quality": "exact",
            "measurement_basis": "production_hot_kernel_view_record",
            "source_view_index": source_view_index,
            "source_image_id": source_image_id,
            "source_image_name": f"images/{source_image_id:04d}.jpg",
        })
    manifest = {
        "schema_name": "openmvs.dmap.map_manifest", "schema_version": 4,
        "estimation_stage": estimation_stage,
        "geometric_iteration": geometric_iteration,
        "width": int(total_cost.shape[1]),
        "height": int(total_cost.shape[0]),
        "num_iterations": 1, "num_logical_states": 2,
        "exact_capture": {
            "requested": True, "available": True, "num_views": len(source_image_ids),
        },
        "maps": maps,
        "write_errors": [],
        "expected_map_count": len(maps),
        "written_map_count": len(maps),
        "complete": True,
    }
    dmap_dev.write_json(frame / "map_manifest.json", manifest)
    write_pfm(frame / state / "cost_total_production_exact.pfm", total_cost)
    write_pfm(frame / state / "gap_winner_runner_up_exact.pfm", exact_gap)
    rgba = np.empty((*view_mask.shape, 4), dtype=np.uint8)
    unsigned = np.asarray(view_mask, dtype=np.uint32)
    for channel in range(4):
        rgba[..., channel] = ((unsigned >> (8 * channel)) & 255).astype(np.uint8)
    mask_path = frame / state / "selected_views_after_mask_exact.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(mask_path)
    write_pfm(frame / "maps" / "depth_final_after_filter.pfm", depth)
    valid_path = frame / "maps" / "valid_after_filter.png"
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(valid, dtype=np.uint8) * 255), mode="L").save(valid_path)


def exact_trace_frame_row(
    frame: Path,
    *,
    run: str = "run",
    estimation_stage: str = "geometric_consistency",
    geometric_iteration: int | None = 2,
) -> pd.Series:
    return pd.Series({
        "run": run, "repeat": 0, "scene_id": "scene", "image_id": 7,
        "estimation_stage": estimation_stage,
        "geometric_iteration": geometric_iteration,
        "depthmap_dir": str(frame), "width": 20, "height": 20,
        "valid_ratio_after_filter": 1.0, "final_cost_median": 0.5,
    })


def make_discovery_capture(
    scene: Path,
    mode: str,
    *,
    max_resolution: int,
    frame_width: int,
) -> tuple[Path, Path]:
    capture = scene / mode
    instrumentation = capture / "dmap_instrumentation"
    frame = instrumentation / "depthmaps" / "0001"
    frame.mkdir(parents=True)
    dmap_dev.write_json(instrumentation / "run_metadata.json", {"schema_version": 4})
    dmap_dev.write_json(frame / "summary.json", {
        "schema_version": 4,
        "image_id": 1,
        "image_name": "0001.jpg",
        "safe_image_name": "0001",
        "width": frame_width,
        "height": 480,
        "num_pixels_total": frame_width * 480,
        "num_rejected_by_filter": 0,
        "valid_ratio_after_filter": 1.0,
        "final_cost": {"median": 0.25},
    })
    depth_maps = capture / "depth_maps"
    depth_maps.mkdir()
    work = capture / "work"
    work.mkdir()
    (work / "Densify.ini").write_text("[Densify]\nNCC Threshold Keep = 0.9\n", encoding="utf-8")
    command = [
        f"/build/{mode}/DensifyPointCloudDMapObserve",
        "--working-folder", str(capture / "work"),
        "--input-file", str(capture / "work" / "scene.mvs"),
        "--output-file", str(capture / "dense.mvs"),
        "--dmap-instrumentation-dir", str(instrumentation),
        "--dmap-instrumentation-level", "maps" if mode == "maps" else "summary",
        "--dmap-instrumentation-sample-rate", "1",
        "--dmap-instrumentation-write-maps", "1" if mode == "maps" else "0",
        "--dense-config-file", str(capture / "work" / "Densify.ini"),
        "--fusion-mode", "1",
        "--resolution-level", "1",
        "--min-resolution", "640",
        "--max-resolution", str(max_resolution),
        "--sub-resolution-levels", "2",
        "--iters", "3",
    ]
    (capture / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    return instrumentation, depth_maps


def make_prefilter_capture(scene: Path) -> tuple[Path, Path]:
    capture = scene / "prefilter"
    instrumentation = capture / "dmap_instrumentation"
    frame = instrumentation / "depthmaps" / "0001"
    frame.mkdir(parents=True)
    depth_path = frame / "maps" / "depth_final_before_filter.pfm"
    write_pfm(depth_path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    dmap_dev.write_json(instrumentation / "run_metadata.json", {
        "schema_name": "openmvs.dmap.run",
        "schema_version": 4,
    })
    summary_path = frame / "summary.json"
    manifest_path = frame / "prefilter_manifest.json"
    completion_reference = {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "path": "prefilter_capture_complete.json",
        "maps_complete": True,
        "prefilter_complete": True,
        "eligible": True,
    }
    dmap_dev.write_json(summary_path, {
        "schema_name": "openmvs.dmap.frame_summary",
        "schema_version": 4,
        "image_id": 1,
        "image_name": "images/0001.jpg",
        "safe_image_name": "0001",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "width": 2,
        "height": 2,
        "num_pixels_total": 4,
        "num_rejected_by_filter": 1,
        "valid_ratio_after_filter": 0.75,
        "final_cost": {"median": 0.25},
        "completion_marker": completion_reference,
    })
    dmap_dev.write_json(manifest_path, {
        "schema_name": "openmvs.dmap.prefilter_manifest",
        "schema_version": 1,
        "complete": True,
        "process_specialization": "Process<false>",
        "image_id": 1,
        "image_name": "images/0001.jpg",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "width": 2,
        "height": 2,
        "maps": [{
            "signal": "depth_final_before_filter",
            "path": "maps/depth_final_before_filter.pfm",
            "dtype": "float32",
            "role": "final_state",
            "measurement_quality": "exact",
            "measurement_basis": "production_pre_filter_snapshot",
            "semantics": "Production depth state immediately before filtering.",
            "bytes": depth_path.stat().st_size,
        }],
    })
    dmap_dev.write_json(frame / "prefilter_capture_complete.json", {
        "schema_name": "openmvs.dmap.prefilter_capture_complete",
        "schema_version": 1,
        "capture_kind": "prefilter",
        "eligible": True,
        "prefilter_complete": True,
        "maps_complete": True,
        "observer_sidecars_complete": True,
        "image_id": 1,
        "image_name": "images/0001.jpg",
        "estimation_stage": "photometric",
        "geometric_iteration": None,
        "manifest": {
            "path": manifest_path.name,
            "schema_version": 1,
            "bytes": manifest_path.stat().st_size,
        },
        "summary": {
            "path": summary_path.name,
            "schema_version": 4,
            "bytes": summary_path.stat().st_size,
        },
    })
    (frame / "iteration.csv").write_text(
        "image_id,iteration,phase,pass_index\n"
        "1,-1,initialization,0\n"
        "1,0,iteration,1\n",
        encoding="utf-8",
    )
    depth_maps = capture / "depth_maps"
    depth_maps.mkdir()
    return instrumentation, depth_maps


class DMapDevelopmentReportTests(unittest.TestCase):
    def test_explicit_scene_and_annotation_sidecar_need_no_private_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar_path = root / "annotations.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "schema_name": "openmvs.dmap.annotation_sidecar",
                        "schema_version": 1,
                        "scene_id": "scene-a",
                        "image_mapping": {"frame-a": "images/0001.jpg"},
                        "frames": [{"id": "frame-a"}],
                        "annotations": {
                            "controlEdges": [],
                            "controlPlanes": [],
                        },
                        "camera_final": {"width": 640, "height": 480},
                    }
                ),
                encoding="utf-8",
            )
            scene_root = root / "scene"
            scene_root.mkdir()
            (scene_root / "scene.mvs").write_bytes(b"scene")
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config = {
                "_config_path": str(root / "experiment.yaml"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "suite": {"name": "smoke"},
                "scenes": [
                    {
                        "scan_id": "scene-a",
                        "working_folder": str(root / "scene"),
                        "mvs_file": str(root / "scene/scene.mvs"),
                        "annotation_sidecar": str(sidecar_path),
                    }
                ],
            }

            suite = dmap_dev.resolve_suite(config)
            scenes = dmap_dev.resolve_scenes(config, suite)
            _scene, review, pipeline, mapping = dmap_dev.scene_context(
                config, "scene-a"
            )

            self.assertEqual(suite, ["scene-a"])
            self.assertEqual(scenes[0]["mvs_file"], str(root / "scene/scene.mvs"))
            self.assertEqual(review["scan_id"], "scene-a")
            self.assertEqual(pipeline["camera_final"]["width"], 640)
            self.assertEqual(mapping, {"frame-a": "images/0001.jpg"})
            lock = dmap_dev.build_experiment_lock(config, scenes)
            self.assertEqual(
                lock["scenes"][0]["annotation_sidecar"]["sha256"],
                hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
            )

    def test_report_evidence_context_is_loaded_bound_and_rendered_separately(self) -> None:
        from test_dmap_report_model import make_evidence_context

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            resolved_path = root / "00_resolved_experiment.yaml"
            context_path = root / "evidence_context.json"
            config_path.write_text("schema_version: 2\n", encoding="utf-8")
            resolved_path.write_text("schema_version: 2\n", encoding="utf-8")
            context = make_evidence_context()
            context_path.write_text(
                json.dumps(context, sort_keys=True) + "\n", encoding="utf-8"
            )

            loaded = dmap_dev.load_report_evidence_context(context_path)
            policy = dmap_dev.build_report_policy(
                {"_config_path": str(config_path)}, root,
                skip_diagnostics=False,
                capture_evidence={"reuse_eligible": True},
                evidence_context_path=context_path,
                published_output_dir=root / "canonical_report",
            )
            markdown = "\n".join(dmap_dev.evidence_context_markdown(loaded))

            self.assertEqual(loaded, context)
            self.assertEqual(
                policy["evidence_context"], dmap_dev.file_identity(context_path)
            )
            self.assertEqual(
                policy["published_output_dir"], str(root / "canonical_report")
            )
            self.assertIn("Diagnostic Mechanism Verdict", markdown)
            self.assertIn("External Production Quality Authority", markdown)
            self.assertIn("not annotations or comparisons recomputed", markdown)
            self.assertIn("estimator validity delta", markdown)
            self.assertIn("terminal endpoint validity delta", markdown)
            self.assertIn("Raw external rows are intentionally not copied", markdown)

            mutated = dict(context)
            mutated["subject"] = dict(context["subject"])
            mutated["subject"]["summary"] = "changed without a new digest"
            context_path.write_text(json.dumps(mutated), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context_sha256 does not match"):
                dmap_dev.load_report_evidence_context(context_path)

            target = root / "context-target.json"
            target.write_text(json.dumps(context), encoding="utf-8")
            link = root / "context-link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                dmap_dev.load_report_evidence_context(link)

    def test_staged_report_paths_rebase_only_report_owned_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "03_master_staging"
            published = root / "03_master"
            external = root / "captures" / "summary.json"
            value = {
                "report": str(staging / "01_development_report.md"),
                "nested": [{"asset": str(staging / "assets" / "plot.png")}],
                "external": str(external),
                "lookalike": str(root / "03_master_staging-other" / "asset.png"),
                "relative": "assets/plot.png",
            }

            rebased = dmap_dev.rebase_report_output_paths(
                value, staging, published
            )

            self.assertEqual(
                rebased["report"], str(published / "01_development_report.md")
            )
            self.assertEqual(
                rebased["nested"][0]["asset"],
                str(published / "assets" / "plot.png"),
            )
            self.assertEqual(rebased["external"], str(external))
            self.assertEqual(rebased["lookalike"], value["lookalike"])
            self.assertEqual(rebased["relative"], "assets/plot.png")

            dataframe = pd.DataFrame([
                {
                    "visual_overlay_svg": str(staging / "visualizations" / "a.svg"),
                    "reference_dmap": str(external),
                    "score": 1.0,
                },
            ])
            rebased_dataframe = dmap_dev.rebase_report_dataframe_paths(
                dataframe, staging, published
            )
            self.assertEqual(
                rebased_dataframe.iloc[0]["visual_overlay_svg"],
                str(published / "visualizations" / "a.svg"),
            )
            self.assertEqual(
                rebased_dataframe.iloc[0]["reference_dmap"], str(external)
            )
            self.assertEqual(
                dataframe.iloc[0]["visual_overlay_svg"],
                str(staging / "visualizations" / "a.svg"),
            )

    def test_summary_capture_is_reportable_without_deep_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            summary = scene / "timing" / "dmap_instrumentation"
            summary.mkdir(parents=True)
            (summary / "run_metadata.json").write_text("{}\n", encoding="utf-8")
            (scene / "timing" / "depth_maps").mkdir()

            discovered = dmap_dev.discover_run_scenes(
                {"runs": [{"label": "base", "role": "baseline"}]}, root
            )

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].instrumentation_dir, summary)
            self.assertEqual(discovered[0].depth_map_dir, scene / "timing" / "depth_maps")
            self.assertEqual(discovered[0].timing_dir, summary)

    def test_prefilter_capture_is_discovered_and_marks_deep_signals_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            instrumentation, depth_maps = make_prefilter_capture(scene)

            discovered = dmap_dev.discover_run_scenes(
                {"runs": [{"label": "base", "role": "baseline"}]}, root
            )
            catalog, availability = dmap_dev.build_map_catalog(discovered)

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].instrumentation_dir, instrumentation)
            self.assertEqual(discovered[0].depth_map_dir, depth_maps)
            self.assertEqual(
                dmap_dev.validate_prefilter_frame(
                    instrumentation / "depthmaps" / "0001"
                ),
                (True, "validated bounded prefilter depth snapshot"),
            )
            self.assertEqual(catalog["signal"].tolist(), ["depth_final_before_filter"])
            self.assertTrue(bool(catalog.iloc[0]["available"]))
            self.assertEqual(catalog.iloc[0]["measurement_quality"], "exact")
            cost_rows = availability[
                (availability["signal"] == "cost_total_production_exact")
                & (availability["logical_iteration"] == 0)
            ]
            self.assertEqual(len(cost_rows), 1)
            self.assertIn(
                "exact_capture_not_requested_by_prefilter_profile",
                cost_rows.iloc[0]["availability_reason"],
            )
            improvement_rows = availability[
                availability["signal"] == "cost_improvement_exact"
            ]
            self.assertEqual(
                sorted(improvement_rows["logical_iteration"].tolist()),
                [-1, 0],
            )
            self.assertTrue(
                improvement_rows["availability_reason"].str.contains(
                    "prefilter_profile_retains_only"
                ).all()
            )
            self.assertFalse(
                bool((availability["signal"] == "depth_final_before_filter").any())
            )

    def test_prefilter_validator_identifies_summary_only_runtime_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            dmap_dev.write_json(frame / "summary.json", {
                "schema_name": "openmvs.dmap.frame_summary",
                "schema_version": 4,
            })
            dmap_dev.write_json(frame / "summary_complete.json", {
                "schema_name": "openmvs.dmap.summary_complete",
                "schema_version": 1,
                "capture_kind": "summary_only",
                "summary_complete": True,
                "maps_complete": False,
            })

            valid, reason = dmap_dev.validate_prefilter_frame(frame)

            self.assertFalse(valid)
            self.assertIn("emitted summary-only evidence", reason)
            self.assertIn("prefilter manifest and completion marker are absent", reason)

    def test_prefilter_validator_rejects_symlinked_map_parent_and_frame_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instrumentation, _depth_maps = make_prefilter_capture(root / "scene")
            frame = instrumentation / "depthmaps" / "0001"
            maps = frame / "maps"
            external_maps = root / "external-maps"
            maps.rename(external_maps)
            maps.symlink_to(external_maps, target_is_directory=True)

            valid, reason = dmap_dev.validate_prefilter_frame(frame)

            self.assertFalse(valid)
            self.assertIn("contains a symlink", reason)

            maps.unlink()
            external_maps.rename(maps)
            external_frame = root / "external-frame"
            frame.rename(external_frame)
            frame.symlink_to(external_frame, target_is_directory=True)

            valid, reason = dmap_dev.validate_prefilter_frame(frame)

            self.assertFalse(valid)
            self.assertIn("contains a symlink", reason)

    def test_manifest_rows_preserve_complete_estimation_stage_identity(self) -> None:
        root = Path("/tmp/report-stage-identity")
        rows = [
            dmap_dev.RunScene(
                label="base", role="baseline", repeat=0, scene_id="scene-a",
                instrumentation_dir=root / "dmap_instrumentation",
                depth_map_dir=root / "depth_maps", timing_dir=root / "dmap_instrumentation",
            ),
            dmap_dev.RunScene(
                label="base", role="baseline", repeat=0, scene_id="scene-a",
                instrumentation_dir=(
                    root / "dmap_instrumentation/geometric_iterations/iteration03"
                ),
                depth_map_dir=root / "depth_maps", timing_dir=root / "dmap_instrumentation",
                estimation_stage="geometric_consistency", geometric_iteration=3,
            ),
        ]

        manifest = dmap_dev.manifest_run_scene_rows(rows)

        self.assertEqual(
            [
                (row["estimation_stage"], row["geometric_iteration"])
                for row in manifest
            ],
            [("photometric", None), ("geometric_consistency", 3)],
        )

    def test_stage_discovery_accepts_unbounded_geometric_iteration_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instrumentation = Path(directory) / "dmap_instrumentation"
            for name in ("iteration100", "iteration02"):
                (instrumentation / "geometric_iterations" / name).mkdir(parents=True)
            run_scene = dmap_dev.RunScene(
                label="base",
                role="baseline",
                repeat=0,
                scene_id="scene",
                instrumentation_dir=instrumentation,
                depth_map_dir=None,
                timing_dir=instrumentation,
            )

            expanded = dmap_dev.expand_instrumentation_stages(run_scene)

            self.assertEqual(
                [row.geometric_iteration for row in expanded],
                [None, 2, 100],
            )
            self.assertEqual(
                [stage[1] for stage in dmap_dev.instrumentation_stage_roots(instrumentation)],
                [None, 2, 100],
            )

    def test_same_production_signature_keeps_summary_quality_and_isolates_deep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            maps_instrumentation, maps_depth = make_discovery_capture(
                scene, "maps", max_resolution=3200, frame_width=3200
            )
            timing_instrumentation, _timing_depth = make_discovery_capture(
                scene, "timing", max_resolution=3200, frame_width=3200
            )
            config = {"runs": [{"label": "base", "role": "baseline"}]}

            discovered = dmap_dev.discover_run_scenes(config, root)

            self.assertEqual(len(discovered), 2)
            by_label = {row.label: row for row in discovered}
            quality = by_label["base"]
            diagnostic = by_label["base [deep]"]
            self.assertEqual(quality.instrumentation_dir, timing_instrumentation)
            self.assertEqual(quality.capture_profile, "summary")
            self.assertFalse(quality.diagnostic_only)
            self.assertEqual(diagnostic.instrumentation_dir, maps_instrumentation)
            self.assertEqual(diagnostic.depth_map_dir, maps_depth)
            self.assertEqual(diagnostic.timing_dir, timing_instrumentation)
            self.assertEqual(diagnostic.capture_profile, "deep")
            self.assertTrue(diagnostic.diagnostic_only)
            self.assertTrue(diagnostic.cross_capture_parity_compatible)
            self.assertEqual(
                dmap_dev.normalized_capture_command_signature(scene / "maps"),
                dmap_dev.normalized_capture_command_signature(scene / "timing"),
            )

    def test_deep_only_capture_is_never_quality_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            make_discovery_capture(scene, "maps", max_resolution=3200, frame_width=3200)
            config = {"runs": [{"label": "base", "role": "baseline"}]}

            discovered = dmap_dev.discover_run_scenes(config, root)

            self.assertEqual(len(discovered), 1)
            self.assertEqual(discovered[0].label, "base [deep]")
            self.assertEqual(discovered[0].configured_label, "base")
            self.assertEqual(discovered[0].capture_profile, "deep")
            self.assertTrue(discovered[0].diagnostic_only)
            rows, _passes, _timings = dmap_dev.load_instrumentation(discovered[0])
            self.assertTrue(rows)
            self.assertTrue(dmap_dev.configured_metric_rows(pd.DataFrame(rows), config).empty)

    def test_profile_coexistence_preserves_prefilter_as_auxiliary_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            make_discovery_capture(scene, "timing", max_resolution=3200, frame_width=3200)
            make_discovery_capture(scene, "maps", max_resolution=3200, frame_width=3200)
            make_prefilter_capture(scene)
            config = {"runs": [{"label": "base", "role": "baseline"}]}

            analysis = dmap_dev.discover_run_scenes(config, root)
            auxiliary = dmap_dev.discover_auxiliary_profile_scenes(config, root)
            catalog, _availability = dmap_dev.build_map_catalog([*analysis, *auxiliary])

            self.assertEqual({row.capture_profile for row in analysis}, {"summary", "deep"})
            self.assertEqual(len(auxiliary), 1)
            self.assertEqual(auxiliary[0].capture_profile, "prefilter")
            prefilter = catalog[catalog["capture_profile"] == "prefilter"]
            self.assertEqual(prefilter["signal"].tolist(), ["depth_final_before_filter"])
            self.assertTrue(prefilter.iloc[0]["manifest_path"].endswith("prefilter_manifest.json"))
            frame_rows = []
            for run_scene in analysis:
                rows, _passes, _timings = dmap_dev.load_instrumentation(run_scene)
                frame_rows.extend(rows)
            metric_rows = dmap_dev.configured_metric_rows(
                dmap_dev.select_terminal_frames(pd.DataFrame(frame_rows)), config
            )
            self.assertEqual(metric_rows["capture_profile"].tolist(), ["summary"])

    def test_observed_deep_capture_is_requested_in_profile_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dmap_dev, "validate_completed_run_mode", return_value=(True, "valid")
        ):
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            for mode in ("endpoint", "timing", "prefilter", "maps"):
                (scene / mode).mkdir(parents=True)
            config = {
                "capture_profiles": ["endpoint", "summary", "prefilter"],
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            self.assertEqual(
                coverage["requested_profiles"], ["endpoint", "summary", "prefilter"]
            )
            profiles = {row["profile"]: row for row in coverage["profiles"]}
            self.assertTrue(profiles["deep"]["requested"])
            self.assertEqual(profiles["deep"]["status"], "complete")

    def test_completed_targeted_trace_is_complete_profile_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dmap_dev, "drilldown_run_complete", return_value=(True, "valid trace")
        ):
            root = Path(directory)
            (root / "runs" / "base" / "repeat_00" / "scene-a").mkdir(parents=True)
            request_value = {
                "schema_name": dmap_dev.dmap_drilldown.SCHEMA_NAME,
                "schema_version": dmap_dev.dmap_drilldown.SCHEMA_VERSION,
                "experiment": {
                    "experiment_id": "test", "config_name": "experiment.yaml",
                    "config_sha256": "a" * 64, "source_revision": None,
                    "source_dirty": None,
                },
                "capture_profile": "trace",
                "target": {
                    "scene_id": "scene-a", "image_id": 7,
                    "pixels": [{"x": 11, "y": 13}], "roi": None,
                    "trace_pixel_count": 1,
                },
                "runs": [{
                    "label": "base", "role": "baseline", "densify_args": [],
                    "ini_overrides": {},
                }],
                "capture": {
                    "instrumentation_level": "maps", "write_maps": True,
                    "patch_match_cuda_instances": 1,
                    "process_specialization": "Process<true>",
                    "compact_exact_trace_available": False,
                    "storage_policy": "full-frame exact maps plus selected trace rows",
                },
            }
            request_value["request_sha256"] = dmap_dev.dmap_drilldown.request_digest(
                request_value
            )
            request_path = dmap_dev.dmap_drilldown.write_immutable_request(
                root / "drilldowns", request_value
            )
            capture = root / "drilldowns" / "captures" / request_value["request_sha256"]
            capture.mkdir(parents=True)
            (capture / "request.yaml").write_bytes(request_path.read_bytes())
            dmap_dev.write_json(capture / "executions.json", {
                "schema_name": "openmvs.dmap.drilldown_executions",
                "schema_version": 1,
                "request_sha256": request_value["request_sha256"],
                "executions": [{
                    "run": "base", "scene_id": "scene-a",
                    "capture_profile": "trace", "return_code": 0,
                }],
            })
            traces = (
                capture / "runs" / "base" / "scene-a" / "dmap_instrumentation"
                / "instrumentation" / "traces.jsonl"
            )
            traces.parent.mkdir(parents=True)
            traces.write_text("{}\n", encoding="utf-8")
            config = {
                "capture_profiles": ["endpoint", "summary", "prefilter"],
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            trace_profile = next(
                row for row in coverage["profiles"] if row["profile"] == "trace"
            )
            trace_units = [
                row for row in coverage["units"] if row["capture_profile"] == "trace"
            ]
            self.assertTrue(trace_profile["requested"])
            self.assertEqual(trace_profile["status"], "complete")
            self.assertEqual(len(trace_units), 1)
            self.assertEqual(trace_units[0]["status"], "complete")
            self.assertEqual(trace_units[0]["frames"][0]["image_id"], 7)
            self.assertEqual(
                {link["label"] for link in trace_units[0]["evidence_links"]},
                {"immutable request", "captured request", "executions", "traces"},
            )

    def test_failed_and_malformed_trace_requests_remain_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests = root / "drilldowns" / "requests"
            requests.mkdir(parents=True)
            (requests / "malformed.yaml").write_text("runs: [\n", encoding="utf-8")
            request_value = {
                "schema_name": dmap_dev.dmap_drilldown.SCHEMA_NAME,
                "schema_version": dmap_dev.dmap_drilldown.SCHEMA_VERSION,
                "experiment": {
                    "experiment_id": "test", "config_name": "experiment.yaml",
                    "config_sha256": "a" * 64, "source_revision": None,
                    "source_dirty": None,
                },
                "capture_profile": "trace",
                "target": {
                    "scene_id": "scene-a", "image_id": 7,
                    "pixels": [{"x": 11, "y": 13}], "roi": None,
                    "trace_pixel_count": 1,
                },
                "runs": [{
                    "label": "base", "role": "baseline", "densify_args": [],
                    "ini_overrides": {},
                }],
                "capture": {
                    "instrumentation_level": "maps", "write_maps": True,
                    "patch_match_cuda_instances": 1,
                    "process_specialization": "Process<true>",
                    "compact_exact_trace_available": False,
                    "storage_policy": "full-frame exact maps plus selected trace rows",
                },
            }
            request_value["request_sha256"] = dmap_dev.dmap_drilldown.request_digest(
                request_value
            )
            request_path = dmap_dev.dmap_drilldown.write_immutable_request(
                root / "drilldowns", request_value
            )
            capture = root / "drilldowns" / "captures" / request_value["request_sha256"]
            capture.mkdir(parents=True)
            (capture / "request.yaml").write_bytes(request_path.read_bytes())
            dmap_dev.write_json(capture / "executions.json", {
                "schema_name": "openmvs.dmap.drilldown_executions",
                "schema_version": 1,
                "request_sha256": request_value["request_sha256"],
                "executions": [{
                    "run": "base", "scene_id": "scene-a",
                    "capture_profile": "trace", "return_code": 7,
                }],
            })
            config = {
                "capture_profiles": ["summary"],
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            trace_units = [
                row for row in coverage["units"]
                if row["capture_profile"] == "trace"
            ]
            self.assertEqual({row["status"] for row in trace_units}, {"failed"})
            self.assertTrue(any(
                "malformed immutable request" in row["reason"]
                for row in trace_units
            ))
            failed = next(row for row in trace_units if row["configured_run"] == "base")
            self.assertIn("failed execution", failed["reason"])
            trace_profile = next(
                row for row in coverage["profiles"] if row["profile"] == "trace"
            )
            self.assertTrue(trace_profile["requested"])
            self.assertEqual(trace_profile["status"], "failed")

    def test_deep_capture_cannot_substitute_for_missing_prefilter_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            dmap_dev, "validate_completed_run_mode",
            side_effect=lambda path, mode: (
                (True, "deep complete") if mode == "maps"
                else (False, "prefilter manifest missing")
            ),
        ):
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            (scene / "maps").mkdir(parents=True)
            (scene / "prefilter").mkdir()
            config = {
                "capture_profiles": ["prefilter"],
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            profiles = {row["profile"]: row for row in coverage["profiles"]}
            self.assertEqual(profiles["deep"]["status"], "complete")
            self.assertEqual(profiles["prefilter"]["status"], "failed")

    def test_profile_coverage_keeps_missing_run_repeat_and_scene_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "capture_profiles": ["endpoint", "summary", "prefilter"],
                "suite": {"scan_ids": ["scene-a", "scene-b"]},
                "runs": [{
                    "label": "base", "role": "baseline", "repeats": 2,
                }],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            endpoint = [
                row for row in coverage["units"]
                if row["capture_profile"] == "endpoint"
            ]
            self.assertEqual(len(endpoint), 4)
            self.assertEqual(
                {(row["repeat"], row["scene_id"]) for row in endpoint},
                {(0, "scene-a"), (0, "scene-b"), (1, "scene-a"), (1, "scene-b")},
            )
            self.assertEqual({row["status"] for row in endpoint}, {"unavailable"})
            summary = next(
                row for row in coverage["profiles"] if row["profile"] == "summary"
            )
            self.assertEqual(summary["expected_units"], 4)
            self.assertEqual(summary["unavailable_units"], 4)

    def test_profile_coverage_represents_imported_evidence_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instrumentation = root / "imported" / "dmap_instrumentation"
            instrumentation.mkdir(parents=True)
            config = {
                "capture_profiles": ["summary"],
                "suite": {"scan_ids": ["scene-a"]},
                "runs": [{
                    "label": "archive", "role": "baseline", "repeats": 1,
                    "existing": {"scene-a": {
                        "instrumentation_dir": str(instrumentation),
                        "capture_profile": "summary",
                    }},
                }],
            }

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            imported = [
                row for row in coverage["units"]
                if row["configured_run"] == "archive"
                and row["capture_profile"] == "summary"
            ]
            self.assertEqual(len(imported), 1)
            self.assertEqual(imported[0]["status"], "failed")
            self.assertIn("missing command.sh", imported[0]["reason"])

    def test_opt_in_specialization_policy_splits_same_signature_deep_cohort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "base" / "repeat_00" / "scene-a"
            maps_instrumentation, maps_depth = make_discovery_capture(
                scene, "maps", max_resolution=3200, frame_width=3200
            )
            timing_instrumentation, timing_depth = make_discovery_capture(
                scene, "timing", max_resolution=3200, frame_width=3200
            )
            config = {
                "instrumentation": {
                    "allow_process_specialization_divergence_for_diagnostics": True,
                },
                "runs": [{"label": "base", "role": "baseline"}],
            }

            discovered = dmap_dev.discover_run_scenes(config, root)
            by_label = {row.label: row for row in discovered}

            self.assertEqual(set(by_label), {"base", "base [deep]"})
            quality = by_label["base"]
            diagnostic = by_label["base [deep]"]
            self.assertEqual(quality.instrumentation_dir, timing_instrumentation)
            self.assertEqual(quality.depth_map_dir, timing_depth)
            self.assertFalse(quality.diagnostic_only)
            self.assertEqual(diagnostic.instrumentation_dir, maps_instrumentation)
            self.assertEqual(diagnostic.depth_map_dir, maps_depth)
            self.assertEqual(diagnostic.timing_dir, timing_instrumentation)
            self.assertTrue(diagnostic.cross_capture_parity_compatible)
            self.assertTrue(diagnostic.diagnostic_only)
            self.assertTrue(
                diagnostic.allow_process_specialization_divergence_for_diagnostics
            )
            self.assertEqual(diagnostic.configured_label, "base")
            diagnostic_frames, _passes, _timings = dmap_dev.load_instrumentation(
                diagnostic
            )
            self.assertEqual(diagnostic_frames[0]["run"], "base [deep]")
            self.assertEqual(diagnostic_frames[0]["configured_run"], "base")
            self.assertTrue(diagnostic_frames[0]["diagnostic_only"])
            frames = pd.DataFrame([
                {"run": "base", "value": "quality"},
                {"run": "base [deep]", "value": "mechanics"},
            ])
            self.assertEqual(
                dmap_dev.configured_metric_rows(frames, config)["value"].tolist(),
                ["quality"],
            )

    def test_opt_in_deep_label_is_stable_across_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene_name in ("scene-a", "scene-b"):
                scene = root / "runs" / "base" / "repeat_00" / scene_name
                make_discovery_capture(scene, "maps", max_resolution=3200, frame_width=3200)
                make_discovery_capture(scene, "timing", max_resolution=3200, frame_width=3200)
            config = {
                "instrumentation": {
                    "allow_process_specialization_divergence_for_diagnostics": True,
                },
                "runs": [{"label": "base", "role": "baseline"}],
            }

            discovered = dmap_dev.discover_run_scenes(config, root)

            self.assertEqual(
                [row.label for row in discovered if row.diagnostic_only],
                ["base [deep]", "base [deep]"],
            )

    def test_specialization_policy_requires_a_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            dmap_dev.discover_run_scenes({
                "instrumentation": {
                    "allow_process_specialization_divergence_for_diagnostics": "true",
                },
                "runs": [],
            }, Path("unused"))

    def test_mismatched_resolution_splits_quality_and_diagnostic_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "candidate" / "repeat_00" / "scene-a"
            maps_instrumentation, maps_depth = make_discovery_capture(
                scene, "maps", max_resolution=640, frame_width=640
            )
            timing_instrumentation, timing_depth = make_discovery_capture(
                scene, "timing", max_resolution=3200, frame_width=3200
            )
            config = {"runs": [{"label": "candidate", "role": "variant"}]}

            discovered = dmap_dev.discover_run_scenes(config, root)
            by_label = {row.label: row for row in discovered}

            self.assertEqual(set(by_label), {"candidate", "candidate [deep]"})
            quality = by_label["candidate"]
            diagnostic = by_label["candidate [deep]"]
            self.assertEqual(quality.instrumentation_dir, timing_instrumentation)
            self.assertEqual(quality.depth_map_dir, timing_depth)
            self.assertTrue(quality.cross_capture_parity_compatible)
            self.assertEqual(diagnostic.instrumentation_dir, maps_instrumentation)
            self.assertEqual(diagnostic.depth_map_dir, maps_depth)
            self.assertEqual(diagnostic.timing_dir, maps_instrumentation)
            self.assertEqual(diagnostic.role, "variant")
            self.assertFalse(diagnostic.cross_capture_parity_compatible)
            self.assertIn("--max-resolution", diagnostic.cross_capture_parity_reason)
            self.assertEqual(
                {row.label for row in dmap_dev.select_terminal_run_scenes(discovered)},
                {"candidate", "candidate [deep]"},
            )

            frame_rows = []
            for run_scene in discovered:
                rows, _passes, _timings = dmap_dev.load_instrumentation(run_scene)
                frame_rows.extend(rows)
            frames = dmap_dev.select_terminal_frames(pd.DataFrame(frame_rows))
            metric_frames = dmap_dev.configured_metric_rows(frames, config)
            self.assertEqual(metric_frames["run"].tolist(), ["candidate"])
            self.assertEqual(metric_frames["width"].tolist(), [3200])
            self.assertEqual(
                frames.loc[frames["run"] == "candidate [deep]", "width"].tolist(), [640]
            )

    def test_mismatched_dense_config_content_splits_capture_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "candidate" / "repeat_00" / "scene-a"
            make_discovery_capture(scene, "maps", max_resolution=3200, frame_width=3200)
            make_discovery_capture(scene, "timing", max_resolution=3200, frame_width=3200)
            (scene / "maps" / "work" / "Densify.ini").write_text(
                "[Densify]\nNCC Threshold Keep = 0.8\n", encoding="utf-8"
            )

            discovered = dmap_dev.discover_run_scenes(
                {"runs": [{"label": "candidate", "role": "variant"}]}, root
            )

            self.assertEqual({row.label for row in discovered}, {"candidate", "candidate [deep]"})
            diagnostic = next(row for row in discovered if row.label.endswith("[deep]"))
            self.assertIn("--dense-config-file-sha256", diagnostic.cross_capture_parity_reason)

    def test_missing_command_provenance_splits_capture_cohorts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "candidate" / "repeat_00" / "scene-a"
            make_discovery_capture(
                scene, "maps", max_resolution=3200, frame_width=3200
            )
            make_discovery_capture(
                scene, "timing", max_resolution=3200, frame_width=3200
            )
            (scene / "maps" / "command.sh").unlink()

            discovered = dmap_dev.discover_run_scenes(
                {"runs": [{"label": "candidate", "role": "variant"}]}, root
            )

            diagnostic = next(
                row for row in discovered if row.label.endswith("[deep]")
            )
            self.assertFalse(diagnostic.cross_capture_parity_compatible)
            self.assertIn(
                "deep maps command provenance",
                diagnostic.cross_capture_parity_reason,
            )

    def test_unsafe_configured_label_is_rejected_before_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "runs" / "candidate" / "repeat_00" / "scene-a"
            make_discovery_capture(scene, "maps", max_resolution=640, frame_width=640)
            make_discovery_capture(scene, "timing", max_resolution=3200, frame_width=3200)
            config = {"runs": [
                {"label": "candidate", "role": "variant"},
                {"label": "candidate [deep]", "role": "variant"},
            ]}

            with self.assertRaisesRegex(ValueError, "portable path component"):
                dmap_dev.discover_run_scenes(config, root)

    def test_incompatible_deep_validation_marks_cross_parity_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene-a"
            maps_instrumentation = scene / "maps" / "dmap_instrumentation"
            frame = maps_instrumentation / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            dmap_dev.write_json(frame / "summary.json", {"image_id": 1})
            dmap_dev.write_json(frame / "map_manifest.json", {})
            maps_depth = scene / "maps" / "depth_maps"
            timing_depth = scene / "timing" / "depth_maps"
            endpoint_depth = scene / "endpoint" / "depth_maps"
            for path, payload in (
                (maps_depth / "depth0001.dmap", b"deep"),
                (timing_depth / "depth0001.dmap", b"summary"),
                (endpoint_depth / "depth0001.dmap", b"endpoint"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            timing_instrumentation = scene / "timing" / "dmap_instrumentation"
            timing_instrumentation.mkdir(parents=True)
            reason = (
                "deep maps and full summary captures have incompatible production command "
                "signatures: --max-resolution maps=['640'] timing=['3200']; "
                "cross-capture maps/summary and production-endpoint parity are not applicable"
            )
            run_scene = dmap_dev.RunScene(
                "candidate [deep]", "variant", 0, "scene-a",
                maps_instrumentation, maps_depth, maps_instrumentation,
                cross_capture_parity_compatible=False,
                cross_capture_parity_reason=reason,
            )
            validation_result = {
                "valid": True,
                "schema_version": 4,
                "checks": [],
                "manifest_map_count": 12,
                "parity_max_abs": {},
                "warnings": [{
                    "code": "negative_reference_variance_prior_weight_overshoot",
                    "signal": "depth_prior_weight_production_exact",
                    "pixels": 11,
                }],
            }

            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate", return_value=validation_result
            ) as validate:
                result = dmap_dev.validate_run_instrumentation([run_scene])

            self.assertEqual(len(result), 1)
            row = result.iloc[0]
            self.assertTrue(row["valid"])
            self.assertTrue(row["instrumented_dmap_checked"])
            self.assertFalse(row["maps_summary_parity_checked"])
            self.assertEqual(row["maps_summary_parity_status"], "not_applicable")
            self.assertIn("incompatible production command", row["maps_summary_parity_reason"])
            self.assertFalse(row["endpoint_parity_checked"])
            self.assertEqual(row["endpoint_parity_status"], "not_applicable")
            self.assertFalse(row["endpoint_dmap_set_checked"])
            self.assertEqual(row["endpoint_dmap_set_status"], "not_applicable")
            self.assertIsNone(row["endpoint_dmap_set_bit_exact"])
            self.assertEqual(
                json.loads(row["validation_warnings"])[0]["pixels"], 11
            )
            arguments = validate.call_args.args[0]
            self.assertEqual(arguments.instrumented_dmap, maps_depth / "depth0001.dmap")
            self.assertIsNone(arguments.reference_dmap)

    def test_opt_in_specialization_divergence_is_recorded_but_reportable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene-a"
            maps_instrumentation = scene / "maps" / "dmap_instrumentation"
            frame = maps_instrumentation / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            dmap_dev.write_json(frame / "summary.json", {"image_id": 1})
            dmap_dev.write_json(frame / "map_manifest.json", {})
            timing_instrumentation = scene / "timing" / "dmap_instrumentation"
            timing_instrumentation.mkdir(parents=True)
            maps_depth = scene / "maps" / "depth_maps"
            timing_depth = scene / "timing" / "depth_maps"
            endpoint_depth = scene / "endpoint" / "depth_maps"
            for path, payload in (
                (maps_depth / "depth0001.dmap", b"deep"),
                (timing_depth / "depth0001.dmap", b"summary"),
                (endpoint_depth / "depth0001.dmap", b"endpoint"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            run_scene = dmap_dev.RunScene(
                "candidate [deep]", "variant", 0, "scene-a",
                maps_instrumentation, maps_depth, timing_instrumentation,
                diagnostic_only=True,
                diagnostic_only_reason="Process<true> diagnostic cohort",
                allow_process_specialization_divergence_for_diagnostics=True,
            )
            validation_result = {
                "valid": False,
                "schema_version": 4,
                "checks": [{"name": "production_output_parity", "passed": False}],
                "manifest_map_count": 12,
                "parity_max_abs": {"depth_map": 1.0},
            }
            deep_values = {
                "depth_map": np.zeros((1, 1), dtype=np.float32),
                "normal_map": np.zeros((1, 1, 3), dtype=np.float32),
                "confidence_map": np.zeros((1, 1), dtype=np.float32),
            }
            endpoint_values = {
                key: np.ones_like(value) for key, value in deep_values.items()
            }

            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate",
                return_value=validation_result,
            ) as validate, mock.patch.object(
                dmap_dev.instrumentation_validator, "load_dmap",
                side_effect=lambda path: (
                    deep_values if Path(path) == maps_depth / "depth0001.dmap"
                    else endpoint_values
                ),
            ):
                result = dmap_dev.validate_run_instrumentation([run_scene])

            row = result.iloc[0]
            self.assertFalse(row["valid"])
            self.assertTrue(row["report_generation_allowed"])
            self.assertTrue(row["process_specialization_divergence"])
            self.assertEqual(
                row["production_qualification_status"],
                "failed_allowed_diagnostic_only",
            )
            self.assertEqual(
                set(json.loads(row["failed_checks"])),
                dmap_dev.PROCESS_SPECIALIZATION_PARITY_FAILURES,
            )
            self.assertTrue(row["maps_summary_parity_checked"])
            self.assertTrue(row["endpoint_parity_checked"])
            self.assertTrue(row["endpoint_dmap_set_checked"])
            arguments = validate.call_args.args[0]
            self.assertEqual(arguments.instrumented_dmap, maps_depth / "depth0001.dmap")
            self.assertEqual(arguments.reference_dmap, timing_depth / "depth0001.dmap")

    def test_specialization_waiver_never_allows_non_parity_failure(self) -> None:
        result = dmap_dev.process_specialization_validation_disposition(
            validator_valid=False,
            failed_checks=["production_output_parity", "map_contract"],
            diagnostic_only=True,
            allow_divergence=True,
        )

        self.assertFalse(result["valid"])
        self.assertFalse(result["report_generation_allowed"])
        self.assertFalse(result["process_specialization_divergence"])
        self.assertEqual(result["production_qualification_status"], "failed")

    def test_specialization_waiver_never_applies_without_explicit_opt_in(self) -> None:
        result = dmap_dev.process_specialization_validation_disposition(
            validator_valid=False,
            failed_checks=["production_output_parity"],
            diagnostic_only=True,
            allow_divergence=False,
        )

        self.assertFalse(result["report_generation_allowed"])
        self.assertEqual(result["production_qualification_status"], "failed")

    def test_specialization_divergence_summary_remains_invalid_and_warns(self) -> None:
        rows = [{
            "valid": True,
            "report_generation_allowed": True,
            "process_specialization_divergence": False,
            "maps_summary_parity_checked": False,
            "failed_checks": "[]",
        }, {
            "valid": False,
            "report_generation_allowed": True,
            "process_specialization_divergence": True,
            "maps_summary_parity_checked": True,
            "failed_checks": json.dumps(list(
                dmap_dev.PROCESS_SPECIALIZATION_PARITY_FAILURES
            )),
        }]

        summary = dmap_dev.summarize_instrumentation_validation(rows)
        warning = dmap_dev.production_qualification_warning_markdown({
            "instrumentation_validation": summary,
        })

        self.assertFalse(summary["valid"])
        self.assertTrue(summary["report_generation_allowed"])
        self.assertFalse(summary["production_parity_qualified"])
        self.assertEqual(
            summary["production_qualification_status"],
            "failed_allowed_diagnostic_only",
        )
        self.assertEqual(
            summary["diagnostic_process_specialization_divergence_frames"], 1
        )
        self.assertEqual(summary["maps_summary_frames_bit_exact"], 0)
        self.assertIn("NOT PRODUCTION-PARITY QUALIFIED", warning[0])
        self.assertIn("mechanics", warning[0])

    def test_fatal_validation_dominates_allowed_diagnostic_status(self) -> None:
        rows = [{
            "valid": False,
            "report_generation_allowed": True,
            "process_specialization_divergence": True,
            "maps_summary_parity_checked": False,
            "failed_checks": json.dumps(["production_output_parity"]),
        }, {
            "valid": False,
            "report_generation_allowed": False,
            "process_specialization_divergence": False,
            "maps_summary_parity_checked": False,
            "failed_checks": json.dumps(["map_contract"]),
        }]

        summary = dmap_dev.summarize_instrumentation_validation(rows)

        self.assertFalse(summary["report_generation_allowed"])
        self.assertEqual(summary["production_qualification_status"], "failed")

    def test_summary_validation_does_not_compare_timing_capture_to_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            timing = Path(directory) / "scene-a" / "timing"
            instrumentation = timing / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            dmap_dev.write_json(frame / "summary.json", {"image_id": 1})
            dmap_dev.write_json(frame / "summary_complete.json", {
                "schema_name": "openmvs.dmap.summary_complete",
                "summary_complete": True,
            })
            depth_map = timing / "depth_maps" / "depth0001.dmap"
            depth_map.parent.mkdir(parents=True)
            depth_map.write_bytes(b"summary")
            run_scene = dmap_dev.RunScene(
                "candidate", "variant", 0, "scene-a", instrumentation,
                timing / "depth_maps", instrumentation,
            )
            validation_result = {
                "valid": True,
                "schema_version": 4,
                "capture_kind": "summary_only",
                "checks": [],
                "manifest_map_count": 0,
                "maps_available": False,
                "maps_unavailable_reason": "summary_profile_maps_not_requested",
                "exact_maps_available": False,
                "exact_maps_unavailable_reason": "exact capture was not requested",
                "instrumented_dmap_checked": True,
                "parity_max_abs": {},
            }

            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate",
                return_value=validation_result,
            ) as validate:
                result = dmap_dev.validate_run_instrumentation([run_scene])

            row = result.iloc[0]
            self.assertTrue(row["valid"])
            self.assertEqual(row["capture_kind"], "summary_only")
            self.assertFalse(row["maps_available"])
            self.assertFalse(row["exact_maps_available"])
            self.assertTrue(row["instrumented_dmap_checked"])
            self.assertFalse(row["maps_summary_parity_checked"])
            self.assertEqual(row["maps_summary_parity_status"], "not_applicable")
            self.assertIn("summary-only capture", row["maps_summary_parity_reason"])
            self.assertFalse(row["quality_comparison_eligible"])
            summary = dmap_dev.summarize_instrumentation_validation(
                dmap_dev.dataframe_json_records(result)
            )
            self.assertFalse(summary["production_parity_qualified"])
            self.assertEqual(
                summary["production_qualification_status"],
                "unqualified_endpoint_parity",
            )
            arguments = validate.call_args.args[0]
            self.assertEqual(arguments.instrumented_dmap, depth_map)
            self.assertIsNone(arguments.reference_dmap)

    def test_quality_metrics_require_bit_exact_endpoint_parity(self) -> None:
        frames = pd.DataFrame([
            {"run": "base", "repeat": 0, "scene_id": "scene", "image_id": 1, "value": 1},
            {"run": "base", "repeat": 0, "scene_id": "scene", "image_id": 2, "value": 2},
            {"run": "diagnostic", "repeat": 0, "scene_id": "scene", "image_id": 1, "value": 3},
        ])
        validation = pd.DataFrame([
            {
                "run": "base", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "terminal_stage": True, "diagnostic_only": False,
                "quality_comparison_eligible": True,
            },
            {
                "run": "base", "repeat": 0, "scene_id": "scene", "image_id": 2,
                "terminal_stage": True, "diagnostic_only": False,
                "quality_comparison_eligible": False,
            },
        ])
        config = {"runs": [{"label": "base"}, {"label": "diagnostic"}]}

        selected = dmap_dev.production_quality_metric_rows(
            frames, config, validation
        )

        self.assertEqual(selected["image_id"].tolist(), [1])
        self.assertEqual(selected["run"].tolist(), ["base"])

    def test_deep_completion_rejects_legacy_or_degraded_exact_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "maps"
            instrumentation = run_dir / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            (run_dir / "command.sh").write_text("true\n", encoding="utf-8")
            dmap_dev.write_json(run_dir / "repro.json", {
                "return_code": 0, "dry_run": False,
                "command": [
                    "DensifyPointCloudDMapObserve", "--iters", "0",
                    "--geometric-iters", "0", "--sub-resolution-levels", "0",
                    "--fusion-mode", "1",
                ],
            })
            dmap_dev.write_json(instrumentation / "run_metadata.json", {
                "schema_name": "openmvs.dmap.run",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(instrumentation / "scene_summary.json", {
                "schema_name": "openmvs.dmap.scene_summary",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(frame / "summary.json", {
                "schema_version": 4, "image_id": 1,
            })
            dmap_dev.write_json(frame / "map_manifest.json", {
                "schema_version": 3,
                "exact_capture": {"requested": False, "available": False},
            })
            depth = run_dir / "depth_maps" / "depth0001.dmap"
            depth.parent.mkdir()
            depth.write_bytes(b"dmap")
            (instrumentation / "resource_plans.jsonl").write_text(
                json.dumps({
                    "schema_name": "openmvs.dmap.resource_plan",
                    "image_id": 1,
                    "pyramid_level": 0,
                    "num_logical_states": 1,
                }) + "\n",
                encoding="utf-8",
            )

            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate",
                return_value={"valid": True, "checks": []},
            ):
                valid, reason = dmap_dev.validate_completed_run_mode(
                    run_dir, "maps"
                )

            self.assertFalse(valid)
            self.assertIn("schema-v4 exact Process<true>", reason)

    def test_deep_completion_accepts_documented_multiscale_resource_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "maps"
            instrumentation = run_dir / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0001"
            frame.mkdir(parents=True)
            (run_dir / "command.sh").write_text("true\n", encoding="utf-8")
            dmap_dev.write_json(run_dir / "repro.json", {
                "return_code": 0, "dry_run": False,
                "command": [
                    "DensifyPointCloudDMapObserve", "--iters", "0",
                    "--geometric-iters", "0", "--sub-resolution-levels", "1",
                    "--fusion-mode", "1",
                ],
            })
            dmap_dev.write_json(instrumentation / "run_metadata.json", {
                "schema_name": "openmvs.dmap.run",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(instrumentation / "scene_summary.json", {
                "schema_name": "openmvs.dmap.scene_summary",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(frame / "summary.json", {
                "schema_version": 4, "image_id": 1,
            })
            dmap_dev.write_json(frame / "map_manifest.json", {
                "schema_version": 4,
                "exact_capture": {"requested": True, "available": True},
            })
            depth = run_dir / "depth_maps" / "depth0001.dmap"
            depth.parent.mkdir()
            depth.write_bytes(b"dmap")

            def plan(level: int, *, fine: bool) -> dict[str, object]:
                return {
                    "schema_name": "openmvs.dmap.resource_plan",
                    "schema_version": 4,
                    "image_id": 1,
                    "pyramid_level": level,
                    "num_logical_states": 1,
                    "width": 2,
                    "height": 2,
                    "summary_available": True,
                    "compatibility_maps_requested": not fine,
                    "compatibility_map_contract": {
                        "update_source_map_expected": not fine,
                        "cost_map_expected": False,
                        "cost_map_unavailable_reason": (
                            "not_requested" if fine else
                            "production confidence maps are retained at pyramid level 0 only"
                        ),
                    },
                    "maps_requested": fine,
                    "maps_available": fine,
                    "exact_requested": fine,
                    "exact_available": fine,
                    "limits_mib": {"device": 0, "host": 0, "frame_storage": 0},
                    "effective_estimate_bytes": {
                        "device": 0,
                        "host": 0,
                        "frame_storage": 100,
                        "current_pyramid_storage": 100,
                        "frame_storage_committed_before": 0,
                        "full_resolution_priority_reserve": 0,
                    },
                    "storage_preflight": {
                        "attempted": True,
                        "succeeded": True,
                        "requested_bytes": 100,
                        "requested_plus_priority_reserve_bytes": 100,
                        "frame_priority_reservation_bytes": 0,
                        "frame_priority_reservation_consumed": False,
                    },
                }

            plans_path = instrumentation / "resource_plans.jsonl"
            fine_plan = plan(0, fine=True)
            coarse_plan = plan(1, fine=False)
            plans_path.write_text(
                "\n".join(json.dumps(row) for row in (coarse_plan, fine_plan)) + "\n",
                encoding="utf-8",
            )
            compatibility_map = (
                instrumentation
                / "instrumentation"
                / "maps"
                / "depth0001_scale01_update_source.png"
            )
            compatibility_map.parent.mkdir(parents=True)
            Image.fromarray(
                np.full((2, 2), 32, dtype=np.uint8), mode="L"
            ).save(compatibility_map)

            closure = mock.Mock(valid=True, status="complete", reason="fixture")
            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate",
                return_value={"valid": True, "checks": []},
            ), mock.patch.object(
                dmap_dev.integrity, "validate_capture_artifact_closure",
                return_value=closure,
            ):
                valid, reason = dmap_dev.validate_completed_run_mode(run_dir, "maps")
                self.assertTrue(valid, reason)

                compatibility_map.unlink()
                valid, reason = dmap_dev.validate_completed_run_mode(run_dir, "maps")
                self.assertFalse(valid)
                self.assertIn("coarse compatibility update-source map", reason)

                compatibility_map.write_bytes(b"not-a-png")
                valid, reason = dmap_dev.validate_completed_run_mode(run_dir, "maps")
                self.assertFalse(valid)
                self.assertIn("cannot be decoded", reason)

                Image.fromarray(
                    np.full((2, 2), 32, dtype=np.uint8), mode="L"
                ).save(compatibility_map)
                coarse_plan["compatibility_map_contract"] = {
                    "update_source_map_expected": True,
                    "cost_map_expected": True,
                    "cost_map_unavailable_reason": "",
                }
                plans_path.write_text(
                    "\n".join(json.dumps(row) for row in (coarse_plan, fine_plan)) + "\n",
                    encoding="utf-8",
                )
                valid, reason = dmap_dev.validate_completed_run_mode(run_dir, "maps")

            self.assertFalse(valid)
            self.assertIn("coarse compatibility resource tier", reason)

    def test_capture_topology_requires_declared_geometric_stages_and_levels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instrumentation = Path(directory) / "dmap_instrumentation"
            command = [
                "DensifyPointCloudDMapObserve", "--iters", "2",
                "--geometric-iters", "1", "--sub-resolution-levels", "1",
                "--fusion-mode", "1",
            ]

            def write_stage(
                root: Path,
                stage: str,
                geometric_iteration: int | None,
                plans: list[tuple[int, int]],
            ) -> None:
                dmap_dev.write_json(root / "run_metadata.json", {
                    "schema_name": "openmvs.dmap.run",
                    "estimation_stage": stage,
                    "geometric_iteration": geometric_iteration,
                })
                dmap_dev.write_json(root / "scene_summary.json", {
                    "schema_name": "openmvs.dmap.scene_summary",
                    "estimation_stage": stage,
                    "geometric_iteration": geometric_iteration,
                })
                dmap_dev.write_json(root / "depthmaps" / "0007" / "summary.json", {
                    "image_id": 7,
                })
                (root / "resource_plans.jsonl").write_text(
                    "".join(
                        json.dumps({
                            "schema_name": "openmvs.dmap.resource_plan",
                            "image_id": 7,
                            "pyramid_level": level,
                            "num_logical_states": logical_states,
                        }) + "\n"
                        for level, logical_states in plans
                    ),
                    encoding="utf-8",
                )

            write_stage(instrumentation, "photometric", None, [(0, 3)])
            valid, reason = dmap_dev.validate_instrumentation_capture_topology(
                instrumentation, command
            )
            self.assertFalse(valid)
            self.assertIn("geometric stage topology", reason)

            geometric = instrumentation / "geometric_iterations" / "iteration00"
            write_stage(geometric, "geometric_consistency", 0, [(0, 2)])
            valid, reason = dmap_dev.validate_instrumentation_capture_topology(
                instrumentation, command
            )
            self.assertFalse(valid)
            self.assertIn("photometric pyramid topology", reason)

            write_stage(instrumentation, "photometric", None, [(0, 3), (1, 3)])
            valid, reason = dmap_dev.validate_instrumentation_capture_topology(
                instrumentation, command
            )
            self.assertTrue(valid, reason)

            valid, reason = dmap_dev.validate_instrumentation_capture_topology(
                instrumentation, command, require_timings=True
            )
            self.assertFalse(valid)
            self.assertIn("photometric stage has no timing rows", reason)

            timing_header = (
                "image_id,scale_number,pass_index,phase,iteration,kernel_ms\n"
            )
            (instrumentation / "timings.csv").write_text(
                timing_header + "".join(
                    f"7,{level},{pass_index},phase,0,1.0\n"
                    for level in (0, 1)
                    for pass_index in range(5)
                ),
                encoding="utf-8",
            )
            (geometric / "timings.csv").write_text(
                timing_header + "".join(
                    f"7,0,{pass_index},phase,0,1.0\n"
                    for pass_index in range(3)
                ),
                encoding="utf-8",
            )
            valid, reason = dmap_dev.validate_instrumentation_capture_topology(
                instrumentation, command, require_timings=True
            )
            self.assertTrue(valid, reason)

    def test_entity_scene_values_balances_scenes_after_pairing_entities(self) -> None:
        data = pd.DataFrame([
            {"scene_id": "a", "image_id": 1, "run": "base", "repeat": 0, "metric": 1.0},
            {"scene_id": "a", "image_id": 2, "run": "base", "repeat": 0, "metric": 1.0},
            {"scene_id": "b", "image_id": 1, "run": "base", "repeat": 0, "metric": 2.0},
            {"scene_id": "a", "image_id": 1, "run": "variant", "repeat": 0, "metric": 3.0},
            {"scene_id": "a", "image_id": 2, "run": "variant", "repeat": 0, "metric": 3.0},
            {"scene_id": "b", "image_id": 1, "run": "variant", "repeat": 0, "metric": 1.0},
        ])

        result = dmap_dev.entity_scene_values(
            data, "base", "variant", "metric", ["scene_id", "image_id"]
        ).set_index("scene_id")

        self.assertEqual(result.loc["a", "entities"], 2)
        self.assertAlmostEqual(result.loc["a", "delta"], 2.0)
        self.assertAlmostEqual(result.loc["b", "delta"], -1.0)
        self.assertAlmostEqual(result["delta"].mean(), 0.5)

    def test_relative_delta_uses_each_paired_baseline_value(self) -> None:
        data = pd.DataFrame([
            {"scene_id": "a", "image_id": 1, "run": "base", "repeat": 0, "kernel_ms": 10.0},
            {"scene_id": "b", "image_id": 1, "run": "base", "repeat": 0, "kernel_ms": 20.0},
            {"scene_id": "a", "image_id": 1, "run": "variant", "repeat": 0, "kernel_ms": 11.0},
            {"scene_id": "b", "image_id": 1, "run": "variant", "repeat": 0, "kernel_ms": 18.0},
        ])

        result = dmap_dev.entity_scene_values(
            data,
            "base",
            "variant",
            "kernel_ms",
            ["scene_id", "image_id"],
            relative_delta=True,
        ).set_index("scene_id")

        self.assertAlmostEqual(result.loc["a", "delta"], 0.1)
        self.assertAlmostEqual(result.loc["b", "delta"], -0.1)

    def test_bootstrap_interval_is_deterministic_and_handles_one_scene(self) -> None:
        singleton = dmap_dev.bootstrap_interval(np.asarray([0.25]), seed=7)
        first = dmap_dev.bootstrap_interval(np.asarray([-1.0, 0.0, 2.0]), seed=7, samples=1000)
        second = dmap_dev.bootstrap_interval(np.asarray([-1.0, 0.0, 2.0]), seed=7, samples=1000)

        self.assertEqual(singleton, (0.25, 0.25))
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], np.mean([-1.0, 0.0, 2.0]))
        self.assertGreaterEqual(first[1], np.mean([-1.0, 0.0, 2.0]))

    def test_comparison_rows_preserve_candidate_label(self) -> None:
        frames = pd.DataFrame([
            {
                "scene_id": scene,
                "image_id": 1,
                "run": run,
                "repeat": 0,
                "valid_ratio_after_filter": value,
            }
            for scene in ("a", "b")
            for run, value in (("base", 0.8), ("candidate", 0.9))
        ])
        comparisons, gates = dmap_dev.build_comparisons(
            frames,
            pd.DataFrame(),
            pd.DataFrame(),
            [
                {"label": "base", "role": "baseline"},
                {"label": "candidate", "role": "variant"},
            ],
        )

        self.assertEqual(set(comparisons["candidate"]), {"candidate"})
        self.assertIn("candidate_value", comparisons.columns)
        self.assertEqual({row["candidate"] for row in gates}, {"candidate"})
        self.assertTrue(all(row["status"] == "informational" for row in gates))

    def test_accuracy_ledger_does_not_reward_missing_fits(self) -> None:
        rows = []
        for chunk in ("a", "b"):
            rows.append({
                "run": "base", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "annotation_kind": "edge", "object_id": "edge", "chunk_id": chunk,
                "stage": "post_filter", "fit_status": "ok",
                "all_residual_p95_m": 0.010, "inlier_threshold_auc": 0.70,
                "inlier_fraction_5mm": 0.40, "effective_inlier_coverage_20mm": 0.40,
                "spatial_coverage_fraction": 0.60, "coverage_fraction": 0.70,
            })
            rows.append({
                "run": "improved", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "annotation_kind": "edge", "object_id": "edge", "chunk_id": chunk,
                "stage": "post_filter", "fit_status": "ok",
                "all_residual_p95_m": 0.007, "inlier_threshold_auc": 0.74,
                "inlier_fraction_5mm": 0.44, "effective_inlier_coverage_20mm": 0.41,
                "spatial_coverage_fraction": 0.61, "coverage_fraction": 0.71,
            })
        rows.extend([
            {
                "run": "sparse", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "annotation_kind": "edge", "object_id": "edge", "chunk_id": "a",
                "stage": "post_filter", "fit_status": "ok",
                "all_residual_p95_m": 0.001, "inlier_threshold_auc": 0.99,
                "inlier_fraction_5mm": 0.99, "effective_inlier_coverage_20mm": 0.20,
                "spatial_coverage_fraction": 0.30, "coverage_fraction": 0.35,
            },
            {
                "run": "sparse", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "annotation_kind": "edge", "object_id": "edge", "chunk_id": "b",
                "stage": "post_filter", "fit_status": "skipped_no_valid_depth",
                "all_residual_p95_m": None, "inlier_threshold_auc": None,
                "inlier_fraction_5mm": None, "effective_inlier_coverage_20mm": None,
                "spatial_coverage_fraction": 0.0, "coverage_fraction": 0.0,
            },
        ])
        frames = pd.DataFrame([
            {"run": run, "repeat": 0, "scene_id": "scene", "image_id": 1,
             "valid_ratio_after_filter": value}
            for run, value in (("base", 0.8), ("improved", 0.81), ("sparse", 0.3))
        ])
        ledger, evidence = dmap_dev.build_accuracy_first_ledger(
            pd.DataFrame(rows), frames, pd.DataFrame(), [
                {"label": "base", "role": "baseline"},
                {"label": "improved", "role": "variant"},
                {"label": "sparse", "role": "variant"},
            ],
        )

        indexed = ledger.set_index("candidate")
        self.assertEqual(indexed.loc["improved", "noise_class"], "improved")
        self.assertEqual(indexed.loc["improved", "accuracy_rank"], 1)
        self.assertEqual(indexed.loc["sparse", "noise_class"], "inconclusive")
        self.assertTrue(indexed.loc["sparse", "availability_biased"])
        self.assertEqual(indexed.loc["sparse", "lost_baseline_fit_count"], 1)
        self.assertTrue(indexed.loc["sparse", "coverage_advisory"])
        self.assertEqual(set(evidence["candidate"]), {"improved", "sparse"})

    def test_accuracy_ledger_omits_candidates_without_common_post_filter_evidence(self) -> None:
        def annotation(run: str, scene: str, residual: float) -> dict[str, object]:
            return {
                "run": run,
                "repeat": 0,
                "scene_id": scene,
                "image_id": 1,
                "annotation_kind": "edge",
                "object_id": "edge",
                "chunk_id": "chunk",
                "stage": "post_filter",
                "fit_status": "ok",
                "all_residual_p95_m": residual,
                "inlier_threshold_auc": 0.80,
                "inlier_fraction_5mm": 0.60,
                "effective_inlier_coverage_20mm": 0.70,
                "spatial_coverage_fraction": 0.80,
                "coverage_fraction": 0.90,
            }

        ledger, evidence = dmap_dev.build_accuracy_first_ledger(
            pd.DataFrame([
                annotation("base", "shared", 0.010),
                annotation("observed", "shared", 0.009),
                annotation("disjoint", "other", 0.008),
            ]),
            pd.DataFrame(),
            pd.DataFrame(),
            [
                {"label": "base", "role": "baseline"},
                {"label": "observed", "role": "variant"},
                {"label": "disjoint", "role": "variant"},
                {"label": "never-run", "role": "variant"},
            ],
        )

        self.assertEqual(ledger["candidate"].tolist(), ["observed"])
        self.assertEqual(set(evidence["candidate"]), {"observed"})

    def test_accuracy_ledger_separates_estimator_and_endpoint_coverage(self) -> None:
        annotations = pd.DataFrame([
            {
                "run": run, "repeat": 0, "scene_id": "scene", "image_id": 1,
                "annotation_kind": "edge", "object_id": "edge", "chunk_id": "chunk",
                "stage": "post_filter", "fit_status": "ok",
                "all_residual_p95_m": residual,
                "inlier_threshold_auc": auc,
                "inlier_fraction_5mm": inliers,
                "effective_inlier_coverage_20mm": 0.5,
                "spatial_coverage_fraction": 0.6,
                "coverage_fraction": 0.7,
            }
            for run, residual, auc, inliers in (
                ("base", 0.010, 0.80, 0.60),
                ("candidate", 0.014, 0.77, 0.55),
            )
        ])
        frames = pd.DataFrame([
            {
                "run": "base", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "valid_ratio_after_filter": 0.80,
                "endpoint_valid_depth_coverage": 0.78,
            },
            {
                "run": "candidate", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "valid_ratio_after_filter": 0.80,
                "endpoint_valid_depth_coverage": 0.68,
            },
        ])

        ledger, _evidence = dmap_dev.build_accuracy_first_ledger(
            annotations,
            frames,
            pd.DataFrame(),
            [
                {"label": "base", "role": "baseline"},
                {"label": "candidate", "role": "variant"},
            ],
            strict_candidate_gate={
                "lost_baseline_fit_count_max": 0,
                "median_normalized_noise_loss_lt": 0.0,
                "worst_normalized_noise_loss_max": 1.0,
            },
        )

        row = ledger.iloc[0]
        self.assertEqual(row["valid_coverage_delta"], 0.0)
        self.assertAlmostEqual(row["endpoint_valid_depth_coverage_delta"], -0.10)
        self.assertTrue(row["endpoint_coverage_advisory"])
        self.assertTrue(row["strict_accuracy_gate_available"])
        self.assertFalse(row["strict_accuracy_pass"])

    def test_terminal_dmap_coverage_counts_final_valid_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth0001.dmap"
            write_depth_only_dmap(path, np.asarray([
                [1.0, 0.0, np.nan],
                [2.0, -1.0, 3.0],
            ], dtype=np.float32))

            coverage = dmap_dev.dmap_valid_depth_coverage(path)

        self.assertEqual(coverage, 0.5)

    def test_terminal_dmap_coverage_decodes_quantized_d2_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth0001.dmap"
            write_quantized_depth_only_dmap(path, np.asarray([
                [4.0, 0.0, np.nan],
                [8.0, -4.0, 12.0],
            ], dtype=np.float32), depth_exponent=2)

            coverage = dmap_dev.dmap_valid_depth_coverage(path)

        self.assertEqual(coverage, 0.5)

    def test_terminal_dmap_coverage_rejects_malformed_and_symlink_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.dmap"
            malformed.write_bytes(b"D2\x01\x00")
            with self.assertRaisesRegex(ValueError, "truncated DMAP"):
                dmap_dev.dmap_valid_depth_coverage(malformed)

            target = root / "target.dmap"
            write_quantized_depth_only_dmap(target, np.ones((1, 1), dtype=np.float32))
            link = root / "depth0001.dmap"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                dmap_dev.dmap_valid_depth_coverage(link)

            real_depth_maps = root / "real-depth-maps"
            real_depth_maps.mkdir()
            nested_target = real_depth_maps / "depth0002.dmap"
            write_quantized_depth_only_dmap(
                nested_target, np.ones((1, 1), dtype=np.float32)
            )
            capture = root / "capture"
            capture.mkdir()
            (capture / "depth_maps").symlink_to(real_depth_maps, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                dmap_dev.dmap_valid_depth_coverage(
                    capture / "depth_maps" / "depth0002.dmap", capture
                )

            real_capture = root / "real-capture"
            real_capture.mkdir()
            root_target = real_capture / "depth0003.dmap"
            write_quantized_depth_only_dmap(
                root_target, np.ones((1, 1), dtype=np.float32)
            )
            capture_link = root / "capture-link"
            capture_link.symlink_to(real_capture, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                dmap_dev.dmap_valid_depth_coverage(
                    capture_link / "depth0003.dmap", capture_link
                )

            real_parent = root / "real-parent"
            nested_capture = real_parent / "capture"
            nested_capture.mkdir(parents=True)
            parent_target = nested_capture / "depth0004.dmap"
            write_quantized_depth_only_dmap(
                parent_target, np.ones((1, 1), dtype=np.float32)
            )
            parent_link = root / "parent-link"
            parent_link.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                dmap_dev.dmap_valid_depth_coverage(
                    parent_link / "capture" / "depth0004.dmap",
                    parent_link / "capture",
                )

    def test_empty_structured_tables_keep_csv_headers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            empty = dmap_dev.build_exact_cost_evolution(pd.DataFrame())
            output = dmap_dev.write_dataframe(
                empty, Path(directory) / "exact_cost_evolution.parquet"
            )

            header = Path(output["csv"]).read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("logical_iteration", header)
        self.assertIn("source_map", header)

    def test_timing_frames_sum_passes_per_frame(self) -> None:
        timings = pd.DataFrame([
            {"run": "base", "role": "baseline", "repeat": 0, "scene_id": "a", "image_id": 1,
             "kernel_ms": 1.5, "timing_source": "summary"},
            {"run": "base", "role": "baseline", "repeat": 0, "scene_id": "a", "image_id": 1,
             "kernel_ms": 2.5, "timing_source": "summary"},
        ])

        result = dmap_dev.aggregate_timing_frames(timings).iloc[0]

        self.assertAlmostEqual(result["kernel_ms"], 4.0)
        self.assertEqual(result["timed_passes"], 2)
        self.assertEqual(result["timing_source"], "summary")

    def test_only_endpoint_wall_time_is_a_performance_gate_and_pareto_objective(self) -> None:
        endpoint_performance = pd.DataFrame([
            {
                "run": run,
                "role": "baseline" if run == "base" else "variant",
                "repeat": 0,
                "scene_id": scene,
                "endpoint_elapsed_seconds": elapsed,
            }
            for scene in ("a", "b", "c", "d", "e")
            for run, elapsed in (("base", 10.0), ("candidate", 12.0))
        ])

        _comparisons, gates = dmap_dev.build_comparisons(
            pd.DataFrame(),
            pd.DataFrame(),
            endpoint_performance,
            [
                {"label": "base", "role": "baseline"},
                {"label": "candidate", "role": "variant"},
            ],
        )
        runtime_gate = next(
            row for row in gates
            if row["metric"] == "endpoint_elapsed_seconds"
        )
        pareto = dmap_dev.pareto_summary(gates)

        self.assertNotIn("kernel_ms", dmap_dev.METRICS)
        self.assertEqual(runtime_gate["status"], "fail")
        self.assertAlmostEqual(runtime_gate["mean_delta"], 0.2)
        self.assertIn(
            "endpoint_elapsed_seconds", pareto[0]["objectives"]
        )

    def test_endpoint_runtime_rejects_non_monotonic_repro(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = (
                root / "runs" / "base" / "repeat_00" / "scene-a" / "endpoint"
            )
            endpoint.mkdir(parents=True)
            dmap_dev.write_json(endpoint / "repro.json", {
                "elapsed_seconds": 1.25,
                "elapsed_clock": "wall_clock",
            })
            dmap_dev.write_json(endpoint / "endpoint_metadata.json", {
                "runtime_authority": "production_endpoint_wall_clock",
            })
            config = {
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
                "suite": {"scan_ids": ["scene-a"]},
                "scenes": [{"scan_id": "scene-a"}],
            }

            with mock.patch.object(
                dmap_dev,
                "validate_completed_run_mode",
                return_value=(True, "validated"),
            ):
                result = dmap_dev.load_endpoint_runtime(config, root).iloc[0]

            self.assertEqual(result["runtime_status"], "unavailable")
            self.assertIn("monotonic", result["runtime_reason"])
            self.assertTrue(math.isnan(result["endpoint_elapsed_seconds"]))

    def test_checkerboard_rows_collapse_to_one_logical_iteration(self) -> None:
        rows = [
            {"scale_level": 0, "pass_index": 0, "phase": "init", "iteration": -1,
             "valid_ratio": 0.9, "changed_ratio": 0.9, "mean_cost": 0.8},
            {"scale_level": 0, "pass_index": 1, "phase": "black", "iteration": 0,
             "valid_ratio": 0.91, "changed_ratio": 0.20, "mean_cost": 0.6,
             "mean_cost_delta": 0.10, "mean_abs_depth_delta": 2.0,
             "mean_normal_delta_deg": 4.0, "candidates_tested": 10,
             "candidates_finite": 8, "candidates_accepted": 2,
             "candidate_accounting_mode": "exact_production_hot_kernel"},
            {"scale_level": 0, "pass_index": 2, "phase": "red", "iteration": 0,
             "valid_ratio": 0.92, "changed_ratio": 0.30, "mean_cost": 0.5,
             "mean_cost_delta": 0.15, "mean_abs_depth_delta": 4.0,
             "mean_normal_delta_deg": 8.0, "candidates_tested": 20,
             "candidates_finite": 15, "candidates_accepted": 6,
             "candidate_accounting_mode": "exact_production_hot_kernel"},
        ]

        aggregated = dmap_dev.instrumentation_report.aggregate_logical_iteration_rows(rows)

        self.assertEqual([row["stage"] for row in aggregated], ["initialization", "iteration 1"])
        iteration = aggregated[1]
        self.assertAlmostEqual(iteration["valid_ratio"], 0.92)
        self.assertAlmostEqual(iteration["mean_cost"], 0.5)
        self.assertAlmostEqual(iteration["changed_ratio"], 0.5)
        self.assertAlmostEqual(iteration["mean_cost_delta"], 0.25)
        self.assertAlmostEqual(iteration["mean_abs_depth_delta"], 3.2)
        self.assertAlmostEqual(iteration["mean_normal_delta_deg"], 6.4)
        self.assertEqual(iteration["candidates_tested"], 30)
        self.assertEqual(iteration["candidates_finite"], 23)
        self.assertEqual(iteration["candidates_accepted"], 8)
        self.assertEqual(
            dmap_dev.instrumentation_report.aggregate_logical_iteration_rows(aggregated),
            aggregated,
        )

    def test_unavailable_post_pass_accounting_stays_null_after_aggregation(self) -> None:
        rows = [
            {
                "scale_level": 0, "pass_index": pass_index, "phase": phase,
                "iteration": 0, "valid_ratio": 0.9,
                "changed_ratio": 0.0, "mean_cost": mean_cost,
                "candidates_tested": 0, "candidates_finite": 0,
                "candidates_accepted": 0, "acceptance_rate": 0.0,
                "accepted_from_spatial_propagation": 0,
                "candidate_accounting_mode": "unavailable_post_pass_snapshot",
            }
            for pass_index, phase, mean_cost in (
                (1, "black", 0.6), (2, "red", 0.5),
            )
        ]
        raw_rows = [dict(row) for row in rows]

        aggregated = dmap_dev.instrumentation_report.aggregate_logical_iteration_rows(rows)

        self.assertEqual(rows, raw_rows)
        self.assertEqual(len(aggregated), 1)
        iteration = aggregated[0]
        self.assertEqual(
            iteration["candidate_accounting_mode"],
            "unavailable_post_pass_snapshot",
        )
        for key in (
            "changed_ratio", "candidates_tested", "candidates_finite",
            "candidates_accepted", "tested_candidates", "finite_candidates",
            "accepted_candidates", "acceptance_rate",
            "accepted_from_spatial_propagation",
        ):
            self.assertIn(key, iteration)
            self.assertIsNone(iteration[key], key)
        self.assertAlmostEqual(iteration["mean_cost"], 0.5)
        self.assertEqual(
            dmap_dev.instrumentation_report.aggregate_logical_iteration_rows(
                aggregated
            ),
            aggregated,
        )

    def test_load_instrumentation_propagates_unavailable_accounting_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instrumentation = Path(directory) / "instrumentation"
            frame = instrumentation / "depthmaps" / "0001"
            dmap_dev.write_json(frame / "summary.json", {
                "image_id": 1,
                "image_name": "0001.jpg",
                "safe_image_name": "0001",
                "candidate_accounting_mode": "unavailable_post_pass_snapshot",
                "candidate_acceptance": [{
                    "candidate_type": "SPATIAL_PROPAGATION",
                    "tested_count": 0, "finite_count": 0,
                    "accepted_count": 0, "acceptance_rate": 0.0,
                }],
            })
            frame.joinpath("iteration.csv").write_text(
                "image_id,scale_level,iteration,pass_index,phase,changed_ratio,"
                "candidates_tested,candidates_finite,candidates_accepted,"
                "acceptance_rate,mean_cost\n"
                "1,0,0,1,black,0,0,0,0,0,0.6\n"
                "1,0,0,2,red,0,0,0,0,0,0.5\n",
                encoding="utf-8",
            )
            run = dmap_dev.RunScene(
                "legacy", "baseline", 0, "scene-a", instrumentation, None, None
            )

            frames, iterations, _timings = dmap_dev.load_instrumentation(run)

            self.assertEqual(
                frames[0]["candidate_accounting_mode"],
                "unavailable_post_pass_snapshot",
            )
            for key in (
                "candidate_spatial_propagation_tested",
                "candidate_spatial_propagation_finite",
                "candidate_spatial_propagation_accepted",
                "candidate_spatial_propagation_acceptance_rate",
            ):
                self.assertIn(key, frames[0])
                self.assertIsNone(frames[0][key], key)
            self.assertEqual(len(iterations), 1)
            self.assertEqual(
                iterations[0]["candidate_accounting_mode"],
                "unavailable_post_pass_snapshot",
            )
            for key in dmap_dev.instrumentation_report.CANDIDATE_ACCOUNTING_METRICS:
                self.assertIn(key, iterations[0])
                self.assertIsNone(iterations[0][key], key)
            self.assertEqual(
                dmap_dev.candidate_accounting_value(
                    iterations[0], "changed_ratio"
                ),
                "unavailable",
            )
            self.assertEqual(
                dmap_dev.candidate_accounting_count(
                    pd.DataFrame(iterations), "candidates_finite",
                    available=False,
                ),
                "unavailable",
            )

    def test_checkerboard_maps_combine_by_logical_iteration(self) -> None:
        maps = {
            0: np.zeros((2, 2), dtype=np.float32),
            1: np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            2: np.asarray([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32),
        }

        combined = dmap_dev.instrumentation_report.combine_checkerboard_maps(maps, np)

        self.assertEqual(set(combined), {-1, 0})
        np.testing.assert_array_equal(combined[0], np.asarray([[1.0, 3.0], [4.0, 2.0]]))

    def test_trace_rerun_removes_inherited_instrumentation_arguments(self) -> None:
        args = [
            "--iters", "5",
            "--dmap-instrumentation-image-list", "1",
            "--dmap-instrumentation-level=maps",
            "--number-views", "3",
        ]

        result = dmap_dev.without_value_arguments(
            args,
            {"--dmap-instrumentation-image-list", "--dmap-instrumentation-level"},
        )

        self.assertEqual(result, ["--iters", "5", "--number-views", "3"])

    def test_endpoint_argument_filter_removes_observer_control(self) -> None:
        args = [
            "--iters", "5",
            "--dmap-instrumentation-write-maps", "0",
            "--number-views", "3",
        ]

        result = dmap_dev.without_value_arguments(
            args, dmap_dev.OBSERVER_VALUE_ARGUMENTS,
        )

        self.assertEqual(result, ["--iters", "5", "--number-views", "3"])

    def test_retired_observer_controls_fail_before_capture(self) -> None:
        for argument in dmap_dev.REMOVED_INSTRUMENTATION_ARGUMENTS:
            with self.subTest(argument=argument):
                with self.assertRaisesRegex(ValueError, "removed"):
                    dmap_dev.validate_supported_densify_args([argument, "1"])

    def test_configured_densify_arguments_cannot_override_orchestration(self) -> None:
        for argument in sorted(dmap_dev.ORCHESTRATION_OWNED_DENSIFY_ARGUMENTS):
            for values in ([argument, "value"], [f"{argument}=value"]):
                with self.subTest(argument=argument, values=values):
                    with self.assertRaisesRegex(ValueError, "orchestration-owned"):
                        dmap_dev.validate_supported_densify_args(values)
        with self.assertRaisesRegex(ValueError, "positional"):
            dmap_dev.validate_supported_densify_args(["scene.mvs"])
        with self.assertRaisesRegex(ValueError, "requires one value"):
            dmap_dev.validate_supported_densify_args(["--iters"])
        with self.assertRaisesRegex(ValueError, "orchestration-owned"):
            dmap_dev.validated_argument_overrides(
                {"--output-file": "/tmp/escape.mvs"}, "scene"
            )

    def test_workflow_owned_sample_rate_is_configurable_and_validated(self) -> None:
        self.assertEqual(dmap_dev.instrumentation_sample_rate({}), 1.0)
        self.assertEqual(
            dmap_dev.instrumentation_sample_rate({
                "instrumentation": {"sample_rate": 0.25},
            }),
            0.25,
        )
        for value in (True, 0, -0.1, 1.1, float("nan"), "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "sample_rate"
            ):
                dmap_dev.instrumentation_sample_rate({
                    "instrumentation": {"sample_rate": value},
                })

    def test_config_identity_validation_rejects_escape_duplicates_and_bad_runs(self) -> None:
        base = {
            "schema_version": 2,
            "experiment_id": "experiment",
            "output_root": "/tmp/dmap-output",
            "suite": {"name": "smoke", "scan_ids": ["scene-a"]},
            "scenes": [{"scan_id": "scene-a", "name": "scene-a"}],
            "runs": [{"label": "baseline", "role": "baseline", "repeats": 1}],
        }
        mutations = (
            ("experiment traversal", lambda value: value.update(experiment_id="../escape")),
            ("absolute run", lambda value: value["runs"][0].update(label="/tmp/run")),
            ("scene traversal", lambda value: value["scenes"][0].update(scan_id="../scene")),
            ("scene name separator", lambda value: value["scenes"][0].update(name="bad/name")),
            ("duplicate run", lambda value: value["runs"].append(dict(value["runs"][0]))),
            ("duplicate scene", lambda value: value["scenes"].append(dict(value["scenes"][0]))),
            ("no baseline", lambda value: value["runs"][0].update(role="variant")),
            ("bad role", lambda value: value["runs"][0].update(role="control")),
            ("bad repeats", lambda value: value["runs"][0].update(repeats=0)),
            ("duplicate suite scene", lambda value: value["suite"].update(scan_ids=["scene-a", "scene-a"])),
        )
        for label, mutate in mutations:
            config = json.loads(json.dumps(base))
            mutate(config)
            with self.subTest(label=label), self.assertRaises(ValueError):
                dmap_dev.validate_experiment_config(config)

    def test_contained_output_path_rejects_existing_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "experiment"
            outside = parent / "outside"
            root.mkdir()
            outside.mkdir()
            (root / "runs").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "escapes experiment root"):
                dmap_dev.contained_output_path(
                    root, "runs", "baseline", description="test run directory"
                )

    def test_executable_boundary_attests_distinct_cli_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            config = {
                "_config_path": str(root / "experiment.yaml"),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
            }

            boundary = dmap_dev.attest_densify_executable_boundary(config)

            self.assertTrue(boundary["valid"])
            self.assertEqual(
                boundary["production"]["cli_probe"]["observer_flags"], []
            )
            self.assertTrue(
                dmap_dev.REQUIRED_OBSERVER_CLI_FLAGS.issubset(
                    boundary["observer"]["cli_probe"]["observer_flags"]
                )
            )
            same_binary = dict(config, densify_observe_bin=str(production))
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                dmap_dev.attest_densify_executable_boundary(same_binary)
            write_fake_densify_binary(observer, observer=False, version="no-observer")
            with self.assertRaisesRegex(RuntimeError, "missing required"):
                dmap_dev.attest_densify_executable_boundary(config)

    def test_cli_probe_identity_ignores_dynamic_banner_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "DensifyPointCloud"
            binary.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"dynamic process $$\"\n"
                "printf '%s\\n' '--gpu-device arg'\n"
                "exit 1\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)

            first = dmap_dev.probe_densify_cli(binary)
            second = dmap_dev.probe_densify_cli(binary)

            self.assertEqual(first, second)
            self.assertIn("option_surface_sha256", first)
            self.assertNotIn("captured_sha256", first)

    def test_prepare_validation_failures_do_not_create_output_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            scene_file = source / "scene.mvs"
            scene_file.write_bytes(b"scene")
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)

            def write_config(
                name: str,
                *,
                experiment_id: str = "experiment",
                scene_path: Path = scene_file,
            ) -> tuple[Path, Path]:
                output_root = root / f"output-{name}"
                config_path = root / f"{name}.yaml"
                config_path.write_text(
                    yaml.safe_dump({
                        "schema_version": 2,
                        "experiment_id": experiment_id,
                        "output_root": str(output_root),
                        "densify_bin": str(production),
                        "densify_observe_bin": str(observer),
                        "suite": {"name": "smoke", "scan_ids": ["scene-a"]},
                        "scenes": [{
                            "scan_id": "scene-a",
                            "working_folder": str(source),
                            "mvs_file": str(scene_path),
                        }],
                        "runs": [{
                            "label": "baseline", "role": "baseline", "repeats": 1,
                        }],
                    }, sort_keys=False),
                    encoding="utf-8",
                )
                return config_path, output_root

            write_fake_densify_binary(observer, observer=False, version="wrong-role")
            config_path, output_root = write_config("bad-binary")
            with self.assertRaisesRegex(RuntimeError, "missing required"):
                dmap_dev.prepare_experiment(config_path, False)
            self.assertFalse(output_root.exists())

            write_fake_densify_binary(observer, observer=True)
            config_path, output_root = write_config(
                "missing-input", scene_path=source / "missing.mvs"
            )
            with self.assertRaisesRegex(FileNotFoundError, "mvs_file"):
                dmap_dev.prepare_experiment(config_path, False)
            self.assertFalse(output_root.exists())

            config_path, output_root = write_config(
                "unsafe-path", experiment_id="../escape"
            )
            with self.assertRaisesRegex(ValueError, "experiment_id"):
                dmap_dev.prepare_experiment(config_path, False)
            self.assertFalse(output_root.exists())

    def test_scene_inputs_must_exist_and_mvs_must_be_inside_working_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "experiment.yaml"),
                "scenes": [{
                    "scan_id": "scene-a",
                    "working_folder": str(root / "missing"),
                    "mvs_file": str(root / "missing" / "scene.mvs"),
                }],
            }
            with self.assertRaisesRegex(FileNotFoundError, "working_folder"):
                dmap_dev.resolve_scenes(config, ["scene-a"])
            work = root / "work"
            work.mkdir()
            outside = root / "outside.mvs"
            outside.write_bytes(b"scene")
            config["scenes"][0].update(
                working_folder=str(work), mvs_file=str(outside)
            )
            with self.assertRaisesRegex(ValueError, "below working_folder"):
                dmap_dev.resolve_scenes(config, ["scene-a"])

    def test_observability_binary_is_separate_from_production_endpoint(self) -> None:
        config = {"densify_bin": "/opt/openmvs/bin/DensifyPointCloud"}

        self.assertEqual(
            dmap_dev.densify_binary(config, instrumented=False),
            Path("/opt/openmvs/bin/DensifyPointCloud"),
        )
        self.assertEqual(
            dmap_dev.densify_binary(config, instrumented=True),
            Path("/opt/openmvs/bin/DensifyPointCloudDMapObserve"),
        )

        explicit = dict(config, densify_observe_bin="/debug/bin/DMapObserve")
        self.assertEqual(
            dmap_dev.densify_binary(explicit, instrumented=True),
            Path("/debug/bin/DMapObserve"),
        )

        defaults = {"_config_path": str(Path("/tmp/dmap-config.yaml"))}
        self.assertEqual(
            dmap_dev.densify_binary(defaults, instrumented=False),
            dmap_dev.DEFAULT_DENSIFY_BIN.resolve(),
        )
        self.assertEqual(
            dmap_dev.densify_binary(defaults, instrumented=True),
            dmap_dev.DEFAULT_DENSIFY_OBSERVE_BIN.resolve(),
        )

    def test_generated_outputs_must_resolve_outside_source_tree(self) -> None:
        config_path = dmap_dev.REPO_ROOT / "experiment.yaml"
        with self.assertRaisesRegex(ValueError, "define 'output_root'"):
            dmap_dev.experiment_root({"_config_path": str(config_path)})

        with self.assertRaisesRegex(ValueError, "outside the OpenMVS source tree"):
            dmap_dev.experiment_root({
                "_config_path": str(config_path),
                "output_root": str(dmap_dev.REPO_ROOT / "generated"),
                "experiment_id": "capture",
            })

        with self.assertRaisesRegex(ValueError, "outside the OpenMVS source tree"):
            dmap_dev.ensure_external_output_path(
                dmap_dev.REPO_ROOT / "reports", "report output directory"
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolved = dmap_dev.experiment_root({
                "_config_path": str(config_path),
                "output_root": str(root),
                "experiment_id": "capture",
            })
            self.assertEqual(resolved, (root / "capture").resolve())

    def test_drilldown_report_refresh_rejects_source_tree_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the OpenMVS source tree"):
            dmap_dev.refresh_master_report(
                Path("/tmp/config.yaml"), dmap_dev.REPO_ROOT / "reports"
            )

    def test_capture_profiles_are_explicit_deduplicated_and_backward_compatible(self) -> None:
        self.assertEqual(dmap_dev.capture_profiles({}), ["deep", "summary"])
        self.assertEqual(
            dmap_dev.capture_profiles(
                {"capture_profiles": ["endpoint", "summary", "prefilter", "deep"]}
            ),
            ["endpoint", "summary", "prefilter", "deep"],
        )
        self.assertEqual(
            dmap_dev.capture_profiles({}, ["deep", "endpoint", "deep"]),
            ["deep", "endpoint"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported capture profile"):
            dmap_dev.capture_profiles({}, ["light"])

    def test_prefilter_storage_estimate_is_bounded_and_excludes_deep_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "experiment_id": "prefilter-storage",
                "output_root": str(root / "outputs"),
                "capture_profiles": ["endpoint", "prefilter"],
                "instrumentation": {
                    "expected_width": 1920,
                    "expected_height": 2560,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                    "maps_capability": False,
                },
                "default_densify_args": ["--iters", "10", "--geometric-iters", "4"],
                "runs": [{"label": "control", "role": "baseline", "repeats": 1}],
            }

            estimate = dmap_dev.estimate_storage(
                config, [{"scan_id": "scene", "expected_frames": 1}]
            )
            run = estimate["runs"][0]

            self.assertFalse(estimate["maps_capability"])
            self.assertTrue(estimate["prefilter_capability"])
            self.assertFalse(estimate["exact_observability"])
            self.assertEqual(run["prefilter_bytes_per_stage_raw"], 19_726_336)
            self.assertEqual(run["deep_bytes_per_stage_raw"], 0)
            self.assertEqual(run["bytes_per_frame_raw"], 5 * 19_726_336)
            self.assertLess(estimate["estimated_gib"], 0.1)

    def test_inline_densify_options_drive_storage_and_invalid_values_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "experiment_id": "inline-storage",
                "output_root": str(root / "outputs"),
                "capture_profiles": ["summary"],
                "instrumentation": {
                    "expected_width": 16,
                    "expected_height": 12,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 10,
                },
                "default_densify_args": ["--number-views=32"],
                "runs": [{
                    "label": "control", "role": "baseline", "repeats": 1,
                    "densify_args": ["--iters=100", "--geometric-iters=2"],
                }],
            }

            estimate = dmap_dev.estimate_storage(
                config,
                [{"scan_id": "scene", "expected_frames": 1}],
                requested_profiles=["deep"],
            )
            run = estimate["runs"][0]

            self.assertEqual(run["iterations"], 100)
            self.assertEqual(run["number_views"], 32)
            self.assertEqual(run["estimation_stages"], 3)
            self.assertTrue(estimate["maps_capability"])
            self.assertEqual(
                estimate["effective_capture_profiles"], ["summary", "deep"]
            )
            config["runs"][0]["densify_args"] = ["--iters=not-a-number"]
            with self.assertRaisesRegex(ValueError, "--iters must be an integer"):
                dmap_dev.estimate_storage(
                    config,
                    [{"scan_id": "scene", "expected_frames": 1}],
                    requested_profiles=["deep"],
                )

    def test_inline_fusion_mode_is_detected_without_duplicate_option(self) -> None:
        self.assertTrue(dmap_dev.has_argument(["--fusion-mode=1"], "--fusion-mode"))
        self.assertFalse(dmap_dev.has_argument(["--other=1"], "--fusion-mode"))

    def test_exact_observability_config_toggle_is_rejected(self) -> None:
        config = {
            "capture_profiles": ["deep"],
            "instrumentation": {"exact_observability": False},
        }

        with self.assertRaisesRegex(ValueError, "always use the exact"):
            dmap_dev.estimate_storage(config, [{"scan_id": "scene"}])

    def test_nested_mvs_staging_preserves_relative_sfm_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            scene = source / "mvs" / "scene.mvs"
            image = source / "sfm" / "dense" / "images" / "0000.jpg"
            scene.parent.mkdir(parents=True)
            image.parent.mkdir(parents=True)
            scene.write_bytes(b"scene")
            image.write_bytes(b"image")

            work = root / "work"
            dmap_dev.prepare_working_folder(source, work)
            staged_scene = dmap_dev.stage_mvs_input(source, scene, work)

            self.assertEqual(staged_scene, work / "mvs" / "scene.mvs")
            self.assertEqual(dmap_dev.dmap_working_folder(staged_scene), work / "mvs")
            self.assertTrue((staged_scene.parent / "../sfm/dense/images/0000.jpg").resolve().is_file())
            self.assertTrue((work / "mvs").stat().st_mode & 0o200)

    def test_complete_dmap_set_comparison_is_file_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep = root / "deep"
            endpoint = root / "endpoint"
            deep.mkdir()
            endpoint.mkdir()
            (deep / "depth0001.dmap").write_bytes(b"same")
            (endpoint / "depth0001.dmap").write_bytes(b"same")

            exact = dmap_dev.compare_dmap_directories(deep, endpoint)
            self.assertTrue(exact["bit_exact"])
            self.assertEqual(exact["shared_count"], 1)

            (endpoint / "depth0001.dmap").write_bytes(b"different")
            (endpoint / "depth0002.dmap").write_bytes(b"extra")
            changed = dmap_dev.compare_dmap_directories(deep, endpoint)
            self.assertFalse(changed["bit_exact"])
            self.assertEqual(changed["mismatched"], ["depth0001.dmap"])
            self.assertEqual(changed["missing_from_first"], ["depth0002.dmap"])

    def test_complete_dmap_set_comparison_uses_verified_precompaction_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "timing"
            endpoint = root / "endpoint"
            summary_depth = summary / "depth_maps"
            endpoint_depth = endpoint / "depth_maps"
            summary_depth.mkdir(parents=True)
            endpoint_depth.mkdir(parents=True)
            for name, payload in (("depth0001.dmap", b"one"), ("depth0002.dmap", b"two")):
                (summary_depth / name).write_bytes(payload)
                (endpoint_depth / name).write_bytes(payload)

            complete = dmap_dev.dmap_set_identity(summary_depth)
            (summary_depth / "depth0002.dmap").unlink()
            retained = dmap_dev.dmap_set_identity(summary_depth)
            manifest = {
                "schema_name": "openmvs.dmap.retention_manifest",
                "schema_version": 2,
                "complete": True,
                "lossy": True,
                "full_dmap_set_retained": False,
                "complete_pre_compaction_dmap_set": complete,
                "retained_dmap_set": retained,
                "removed": [{
                    "path": "depth_maps/depth0002.dmap",
                    "bytes": len(b"two"),
                    "sha256": dmap_dev.hashlib.sha256(b"two").hexdigest(),
                    "reason": "not_instrumented_image_id",
                    "lossy": True,
                }],
            }
            dmap_dev.write_json(summary / "retention_manifest.json", manifest)
            completion = {
                "schema_name": "openmvs.dmap.compacted_completion",
                "schema_version": 2,
                "complete": True,
                "complete_pre_compaction_dmap_count": complete["count"],
                "complete_pre_compaction_dmap_set_sha256": complete["sha256"],
                "retained_dmap_count": retained["count"],
                "retained_dmap_set_sha256": retained["sha256"],
                "retention_manifest": "retention_manifest.json",
                "retention_manifest_sha256": dmap_dev.dmap_drilldown.file_digest(
                    summary / "retention_manifest.json"
                ),
            }
            dmap_dev.write_json(summary / "compacted_completion.json", completion)

            exact = dmap_dev.compare_dmap_directories(summary_depth, endpoint_depth)

            self.assertTrue(exact["bit_exact"])
            self.assertEqual(exact["shared_count"], 2)
            self.assertEqual(exact["first_basis"], "verified_pre_compaction_manifest")
            self.assertEqual(exact["second_basis"], "physical_files")

            (endpoint_depth / "depth0002.dmap").write_bytes(b"changed")
            changed = dmap_dev.compare_dmap_directories(summary_depth, endpoint_depth)
            self.assertFalse(changed["bit_exact"])
            self.assertEqual(changed["mismatched"], ["depth0002.dmap"])

    def test_complete_dmap_set_comparison_rejects_tampered_retention_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "timing"
            endpoint = root / "endpoint"
            summary_depth = summary / "depth_maps"
            endpoint_depth = endpoint / "depth_maps"
            summary_depth.mkdir(parents=True)
            endpoint_depth.mkdir(parents=True)
            for depth_dir in (summary_depth, endpoint_depth):
                (depth_dir / "depth0001.dmap").write_bytes(b"same")
            identity = dmap_dev.dmap_set_identity(summary_depth)
            manifest = {
                "schema_name": "openmvs.dmap.retention_manifest",
                "complete": True,
                "lossy": False,
                "full_dmap_set_retained": True,
                "complete_pre_compaction_dmap_set": identity,
                "retained_dmap_set": identity,
                "removed": [],
            }
            dmap_dev.write_json(summary / "retention_manifest.json", manifest)
            dmap_dev.write_json(summary / "compacted_completion.json", {
                "schema_name": "openmvs.dmap.compacted_completion",
                "complete": True,
                "complete_pre_compaction_dmap_count": identity["count"],
                "complete_pre_compaction_dmap_set_sha256": identity["sha256"],
                "retained_dmap_count": identity["count"],
                "retained_dmap_set_sha256": identity["sha256"],
                "retention_manifest": "retention_manifest.json",
                "retention_manifest_sha256": dmap_dev.dmap_drilldown.file_digest(
                    summary / "retention_manifest.json"
                ),
            })
            manifest["unexpected"] = "tamper"
            dmap_dev.write_json(summary / "retention_manifest.json", manifest)

            with self.assertRaisesRegex(ValueError, "retention manifest digest mismatch"):
                dmap_dev.compare_dmap_directories(summary_depth, endpoint_depth)

    def test_complete_dmap_set_comparison_rejects_rebound_semantic_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "timing"
            endpoint = root / "endpoint"
            summary_depth = summary / "depth_maps"
            endpoint_depth = endpoint / "depth_maps"
            summary_depth.mkdir(parents=True)
            endpoint_depth.mkdir(parents=True)
            for name, payload in (("depth0001.dmap", b"one"), ("depth0002.dmap", b"two")):
                (summary_depth / name).write_bytes(payload)
                (endpoint_depth / name).write_bytes(payload)
            complete = dmap_dev.dmap_set_identity(summary_depth)
            (summary_depth / "depth0002.dmap").unlink()
            retained = dmap_dev.dmap_set_identity(summary_depth)
            tampered_rows = [dict(row) for row in complete["files"]]
            tampered_rows[1]["sha256"] = dmap_dev.hashlib.sha256(b"forged").hexdigest()
            tampered = {
                "count": len(tampered_rows),
                "sha256": dmap_dev.stable_json_digest(tampered_rows),
                "files": tampered_rows,
            }
            manifest = {
                "schema_name": "openmvs.dmap.retention_manifest",
                "schema_version": 2,
                "complete": True,
                "lossy": True,
                "full_dmap_set_retained": False,
                "complete_pre_compaction_dmap_set": tampered,
                "retained_dmap_set": retained,
                "removed": [{
                    "path": "depth_maps/depth0002.dmap",
                    "bytes": len(b"two"),
                    "sha256": dmap_dev.hashlib.sha256(b"two").hexdigest(),
                    "reason": "not_instrumented_image_id",
                    "lossy": True,
                }],
            }
            dmap_dev.write_json(summary / "retention_manifest.json", manifest)
            dmap_dev.write_json(summary / "compacted_completion.json", {
                "schema_name": "openmvs.dmap.compacted_completion",
                "schema_version": 2,
                "complete": True,
                "complete_pre_compaction_dmap_count": tampered["count"],
                "complete_pre_compaction_dmap_set_sha256": tampered["sha256"],
                "retained_dmap_count": retained["count"],
                "retained_dmap_set_sha256": retained["sha256"],
                "retention_manifest": "retention_manifest.json",
                "retention_manifest_sha256": dmap_dev.dmap_drilldown.file_digest(
                    summary / "retention_manifest.json"
                ),
            })

            with self.assertRaisesRegex(ValueError, "cannot reconstruct the full set"):
                dmap_dev.compare_dmap_directories(summary_depth, endpoint_depth)

    def test_complete_dmap_set_comparison_rejects_unsupported_retention_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "timing"
            endpoint = root / "endpoint"
            for capture in (summary, endpoint):
                depth = capture / "depth_maps"
                depth.mkdir(parents=True)
                (depth / "depth0001.dmap").write_bytes(b"same")
            identity = dmap_dev.dmap_set_identity(summary / "depth_maps")
            manifest = {
                "schema_name": "openmvs.dmap.retention_manifest",
                "schema_version": 99,
                "complete": True,
                "lossy": False,
                "full_dmap_set_retained": True,
                "complete_pre_compaction_dmap_set": identity,
                "retained_dmap_set": identity,
                "removed": [],
            }
            dmap_dev.write_json(summary / "retention_manifest.json", manifest)
            dmap_dev.write_json(summary / "compacted_completion.json", {
                "schema_name": "openmvs.dmap.compacted_completion",
                "schema_version": 2,
                "complete": True,
                "retention_manifest": "retention_manifest.json",
                "retention_manifest_sha256": dmap_dev.dmap_drilldown.file_digest(
                    summary / "retention_manifest.json"
                ),
            })

            with self.assertRaisesRegex(ValueError, "unsupported retention schema"):
                dmap_dev.compare_dmap_directories(
                    summary / "depth_maps", endpoint / "depth_maps"
                )

    def test_instrumentation_validation_uses_retained_full_set_for_endpoint_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene-a"
            timing = scene / "timing"
            instrumentation = timing / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0001_0000"
            frame.mkdir(parents=True)
            dmap_dev.write_json(frame / "summary.json", {"image_id": 1})
            timing_depth = timing / "depth_maps"
            endpoint_depth = scene / "endpoint" / "depth_maps"
            timing_depth.mkdir()
            endpoint_depth.mkdir(parents=True)
            for name, payload in (("depth0001.dmap", b"one"), ("depth0002.dmap", b"two")):
                (timing_depth / name).write_bytes(payload)
                (endpoint_depth / name).write_bytes(payload)
            complete = dmap_dev.dmap_set_identity(timing_depth)
            (timing_depth / "depth0002.dmap").unlink()
            retained = dmap_dev.dmap_set_identity(timing_depth)
            manifest = {
                "schema_name": "openmvs.dmap.retention_manifest",
                "schema_version": 2,
                "complete": True,
                "lossy": True,
                "full_dmap_set_retained": False,
                "complete_pre_compaction_dmap_set": complete,
                "retained_dmap_set": retained,
                "removed": [{
                    "path": "depth_maps/depth0002.dmap",
                    "bytes": len(b"two"),
                    "sha256": dmap_dev.hashlib.sha256(b"two").hexdigest(),
                    "reason": "not_instrumented_image_id",
                    "lossy": True,
                }],
            }
            dmap_dev.write_json(timing / "retention_manifest.json", manifest)
            dmap_dev.write_json(timing / "compacted_completion.json", {
                "schema_name": "openmvs.dmap.compacted_completion",
                "schema_version": 2,
                "complete": True,
                "complete_pre_compaction_dmap_count": complete["count"],
                "complete_pre_compaction_dmap_set_sha256": complete["sha256"],
                "retained_dmap_count": retained["count"],
                "retained_dmap_set_sha256": retained["sha256"],
                "retention_manifest": "retention_manifest.json",
                "retention_manifest_sha256": dmap_dev.dmap_drilldown.file_digest(
                    timing / "retention_manifest.json"
                ),
            })
            run_scene = dmap_dev.RunScene(
                "candidate", "variant", 0, "scene-a", instrumentation,
                timing_depth, instrumentation,
            )
            validation_result = {
                "valid": True,
                "schema_version": 4,
                "capture_kind": "summary_only",
                "checks": [],
                "maps_available": False,
                "exact_maps_available": False,
                "logical_iterations": [-1, 0],
                "instrumented_dmap_checked": True,
                "parity_max_abs": {},
            }
            dmap_values = {
                name: np.zeros((1, 1), dtype=np.float32)
                for name in ("depth_map", "normal_map", "confidence_map")
            }

            with mock.patch.object(
                dmap_dev.instrumentation_validator, "validate", return_value=validation_result
            ), mock.patch.object(
                dmap_dev.instrumentation_validator, "load_dmap", return_value=dmap_values
            ):
                result = dmap_dev.validate_run_instrumentation([run_scene])

            row = result.iloc[0]
            self.assertTrue(row["valid"])
            self.assertTrue(row["endpoint_parity_checked"])
            self.assertEqual(
                row["endpoint_parity_max_abs"],
                '{"confidence_map": 0.0, "depth_map": 0.0, "normal_map": 0.0}',
            )
            self.assertTrue(row["endpoint_dmap_set_bit_exact"])
            self.assertEqual(row["endpoint_dmap_set_shared_count"], 2)
            self.assertEqual(
                row["endpoint_dmap_set_basis"], "verified_pre_compaction_manifest"
            )

    def test_product_reference_is_annotation_only_and_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            (reference / "depth0001.dmap").write_bytes(b"archived")
            config = {
                "product_reference": {
                    "enabled": True,
                    "required": True,
                    "label": "archived",
                },
                "scenes": [{
                    "scan_id": "scene",
                    "reference_dmap_dir": str(reference),
                }],
            }
            frames = [{
                "run": "baseline",
                "role": "baseline",
                "repeat": 0,
                "scene_id": "scene",
                "image_id": 1,
                "image_name": "0000.jpg",
                "depthmap_dir": "/deep",
            }]

            scenes, reference_frames = dmap_dev.product_reference_annotation_inputs(
                config, frames
            )

            self.assertEqual(len(scenes), 1)
            self.assertEqual(scenes[0].depth_map_dir, reference)
            self.assertEqual(reference_frames[0]["run"], "archived")
            self.assertEqual(reference_frames[0]["annotation_stages"], ["post_filter"])

            (reference / "depth0001.dmap").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "required product-reference"):
                dmap_dev.product_reference_annotation_inputs(config, frames)

    def test_drilldown_cli_accepts_repeated_pixels_and_variants(self) -> None:
        result = dmap_dev.normalize_repeated_drilldown_args([
            "drilldown", "--config", "config.yaml", "--scene", "scene", "--frame", "7",
            "--pixel", "3,4", "--variant", "variant-b", "--pixel", "1,2",
            "--variant", "variant-a", "--execute",
        ])

        self.assertEqual(result.count("--pixel"), 1)
        self.assertEqual(result.count("--variant"), 1)
        pixel_index = result.index("--pixel")
        variant_index = result.index("--variant")
        self.assertEqual(result[pixel_index + 1:variant_index], ["3,4", "1,2"])
        self.assertEqual(result[variant_index + 1:], ["variant-b", "variant-a"])

    def test_drilldown_command_overrides_inherited_capture_selection(self) -> None:
        config = {
            "densify_bin": "/tmp/DensifyPointCloud",
            "default_densify_args": [
                "--iters", "5", "--patch-match-cuda-instances", "4",
                "--dmap-instrumentation-image-list", "99",
            ],
        }
        request = {
            "capture_profile": "trace",
            "target": {"image_id": 7},
        }
        command = dmap_dev.drilldown_run_command(
            config,
            {"densify_args": []},
            request,
            Path("/tmp/run"),
            Path("/tmp/work"),
            Path("/tmp/work/scene.mvs"),
            Path("/tmp/run/trace_config.json"),
            Path("/tmp/run/generated/Densify.drilldown.cfg"),
        )

        self.assertEqual(command[command.index("--patch-match-cuda-instances") + 1], "1")
        self.assertEqual(command[command.index("--dmap-instrumentation-level") + 1], "maps")
        self.assertEqual(command[command.index("--dmap-instrumentation-image-list") + 1], "7")
        self.assertEqual(command[command.index("--dmap-instrumentation-write-maps") + 1], "1")
        self.assertEqual(
            command[command.index("--config-file") + 1],
            "/tmp/run/generated/Densify.drilldown.cfg",
        )
        self.assertNotIn("99", command)

    def test_trace_completion_requires_exact_maps_and_every_requested_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "trace"
            instrumentation = run_dir / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0007"
            frame.mkdir(parents=True)
            dmap_dev.write_json(run_dir / "repro.json", {
                "return_code": 0,
                "dry_run": False,
                "command": [
                    "DensifyPointCloudDMapObserve", "--config-file",
                    str(run_dir / "generated" / "Densify.drilldown.cfg"),
                    "--geometric-iters", "0",
                    "--fusion-mode", "1",
                ],
            })
            dmap_dev.write_immutable_text(
                run_dir / "generated" / "Densify.drilldown.cfg", "", "test config"
            )
            dmap_dev.write_json(instrumentation / "run_metadata.json", {
                "schema_name": "openmvs.dmap.run",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(instrumentation / "scene_summary.json", {
                "schema_name": "openmvs.dmap.scene_summary",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(frame / "summary.json", {"image_id": 7})
            dmap_dev.write_json(frame / "map_manifest.json", {
                "exact_capture": {"requested": True, "available": True},
                "num_iterations": 1,
                "num_logical_states": 2,
                "pyramid_level": 0,
            })
            traces_path = instrumentation / "instrumentation" / "traces.jsonl"
            traces_path.parent.mkdir()
            traces_path.write_text(
                '{"image_id":7,"trace_index":0,"x":2,"y":3,"scale_number":0,"logical_iteration":-1}\n'
                '{"image_id":7,"trace_index":0,"x":2,"y":3,"scale_number":0,"logical_iteration":0}\n'
                '{"image_id":7,"trace_index":1,"x":5,"y":8,"scale_number":0,"logical_iteration":-1}\n'
                '{"image_id":7,"trace_index":1,"x":5,"y":8,"scale_number":0,"logical_iteration":0}\n',
                encoding="utf-8",
            )
            pixels = [{"x": 2, "y": 3}, {"x": 5, "y": 8}]

            with mock.patch.object(
                dmap_dev, "validate_completed_run_mode", return_value=(True, "maps valid")
            ):
                self.assertEqual(
                    dmap_dev.drilldown_run_complete(
                        run_dir, "trace", 7, pixels
                    ),
                    (True, "validated 2 targeted trace pixels across 1 stage(s) and 2 logical states"),
                )
                controlled_config = run_dir / "generated" / "Densify.drilldown.cfg"
                controlled_config.write_text("iters=99\n", encoding="utf-8")
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("empty drill-down program-options file", reason)
                controlled_config.write_text("", encoding="utf-8")
                dmap_dev.write_json(frame / "map_manifest.json", {
                    "exact_capture": {"requested": True, "available": False},
                    "num_iterations": 1,
                    "num_logical_states": 2,
                    "pyramid_level": 0,
                })
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("exact Process<true> maps are unavailable", reason)
                dmap_dev.write_json(frame / "map_manifest.json", {
                    "exact_capture": {"requested": True, "available": True},
                    "num_iterations": 1,
                    "num_logical_states": 2,
                    "pyramid_level": 0,
                })
                traces_path.write_text(
                    '{"image_id":7,"trace_index":0,"x":2,"y":3,"scale_number":0,"logical_iteration":-1}\n'
                    '{"image_id":7,"trace_index":0,"x":2,"y":3,"scale_number":0,"logical_iteration":0}\n',
                    encoding="utf-8",
                )
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("missing 2 requested pixel/state rows", reason)

    def test_trace_completion_validates_scaled_deduplicated_pyramid_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "trace"
            instrumentation = run_dir / "dmap_instrumentation"
            frame = instrumentation / "depthmaps" / "0007"
            frame.mkdir(parents=True)
            dmap_dev.write_json(run_dir / "repro.json", {
                "return_code": 0,
                "dry_run": False,
                "command": [
                    "DensifyPointCloudDMapObserve", "--config-file",
                    str(run_dir / "generated" / "Densify.drilldown.cfg"),
                    "--geometric-iters", "1",
                    "--fusion-mode", "1",
                ],
            })
            dmap_dev.write_immutable_text(
                run_dir / "generated" / "Densify.drilldown.cfg", "", "test config"
            )
            dmap_dev.write_json(instrumentation / "run_metadata.json", {
                "schema_name": "openmvs.dmap.run",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(instrumentation / "scene_summary.json", {
                "schema_name": "openmvs.dmap.scene_summary",
                "estimation_stage": "photometric",
                "geometric_iteration": None,
            })
            dmap_dev.write_json(frame / "summary.json", {"image_id": 7})
            dmap_dev.write_json(frame / "map_manifest.json", {
                "exact_capture": {"requested": True, "available": True},
                "num_iterations": 1,
                "num_logical_states": 2,
                "pyramid_level": 0,
            })
            plans = [
                {
                    "image_id": 7,
                    "pyramid_level": 1,
                    "width": 4,
                    "height": 4,
                    "num_trace_pixels": 2,
                    "num_logical_states": 2,
                    "trace_requested": True,
                    "trace_available": True,
                },
                {
                    "image_id": 7,
                    "pyramid_level": 0,
                    "width": 8,
                    "height": 8,
                    "num_trace_pixels": 3,
                    "num_logical_states": 2,
                    "trace_requested": True,
                    "trace_available": True,
                },
            ]
            (instrumentation / "resource_plans.jsonl").write_text(
                "".join(json.dumps(plan) + "\n" for plan in plans),
                encoding="utf-8",
            )
            traces_path = instrumentation / "instrumentation" / "traces.jsonl"
            traces_path.parent.mkdir()
            rows = [
                {"image_id": 7, "trace_index": trace_index, "x": x, "y": y,
                 "scale_number": level, "logical_iteration": iteration}
                for level, coordinates in (
                    (1, [(1, 1), (3, 2)]),
                    (0, [(1, 2), (2, 2), (6, 4)]),
                )
                for trace_index, (x, y) in enumerate(coordinates)
                for iteration in (-1, 0)
            ]
            traces_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            geometric = instrumentation / "geometric_iterations" / "iteration00"
            geometric_frame = geometric / "depthmaps" / "0007"
            geometric_frame.mkdir(parents=True)
            dmap_dev.write_json(geometric / "run_metadata.json", {
                "schema_name": "openmvs.dmap.run",
                "estimation_stage": "geometric_consistency",
                "geometric_iteration": 0,
            })
            dmap_dev.write_json(geometric / "scene_summary.json", {
                "schema_name": "openmvs.dmap.scene_summary",
                "estimation_stage": "geometric_consistency",
                "geometric_iteration": 0,
            })
            dmap_dev.write_json(geometric_frame / "summary.json", {"image_id": 7})
            dmap_dev.write_json(geometric_frame / "map_manifest.json", {
                "exact_capture": {"requested": True, "available": True},
                "num_iterations": 1,
                "num_logical_states": 2,
                "pyramid_level": 0,
            })
            (geometric / "resource_plans.jsonl").write_text(
                json.dumps({
                    "image_id": 7,
                    "pyramid_level": 0,
                    "width": 8,
                    "height": 8,
                    "num_trace_pixels": 3,
                    "num_logical_states": 2,
                    "trace_requested": True,
                    "trace_available": True,
                }) + "\n",
                encoding="utf-8",
            )
            geometric_traces = geometric / "instrumentation" / "traces.jsonl"
            geometric_traces.parent.mkdir()
            geometric_rows = [
                {"image_id": 7, "trace_index": trace_index, "x": x, "y": y,
                 "scale_number": 0, "logical_iteration": iteration}
                for trace_index, (x, y) in enumerate([(1, 2), (2, 2), (6, 4)])
                for iteration in (-1, 0)
            ]
            geometric_traces.write_text(
                "".join(json.dumps(row) + "\n" for row in geometric_rows),
                encoding="utf-8",
            )
            pixels = [
                {"x": 1, "y": 2}, {"x": 2, "y": 2},
                {"x": 1, "y": 2}, {"x": 6, "y": 4},
            ]

            with mock.patch.object(
                dmap_dev, "validate_completed_run_mode", return_value=(True, "maps valid")
            ):
                self.assertEqual(
                    dmap_dev.drilldown_run_complete(run_dir, "trace", 7, pixels),
                    (True, "validated 3 targeted trace pixels across 2 stage(s) and 6 logical states"),
                )
                rows[0]["x"] = 2
                traces_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("does not bind its compact slot", reason)
                rows[0]["x"] = 1
                traces_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                geometric.rename(geometric.with_name("iteration01"))
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("geometric stage topology", reason)
                geometric.with_name("iteration01").rename(
                    geometric.with_name("not-an-iteration")
                )
                valid, reason = dmap_dev.drilldown_run_complete(
                    run_dir, "trace", 7, pixels
                )
                self.assertFalse(valid)
                self.assertIn("unexpected geometric stage directory", reason)

    def test_reused_drilldown_preserves_execution_contract_and_complete_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "experiment.yaml"
            config_path.write_text("schema_version: 2\n", encoding="utf-8")
            request_id = "a" * 64
            request = {
                "request_sha256": request_id,
                "capture_profile": "trace",
                "experiment": {
                    "config_sha256": dmap_dev.dmap_drilldown.file_digest(config_path),
                },
                "target": {
                    "scene_id": "scene-a",
                    "image_id": 7,
                    "pixels": [{"x": 11, "y": 13}, {"x": 17, "y": 19}],
                    "trace_pixel_count": 2,
                },
                "runs": [{
                    "label": "base",
                    "role": "baseline",
                    "densify_args": [],
                    "ini_overrides": {},
                }, {
                    "label": "candidate",
                    "role": "variant",
                    "densify_args": [],
                    "ini_overrides": {},
                }],
            }
            request_path = root / "drilldowns" / "requests" / f"{request_id}.yaml"
            request_path.parent.mkdir(parents=True)
            request_path.write_text(
                yaml.safe_dump(request, sort_keys=True), encoding="utf-8"
            )
            for label in ("base", "candidate"):
                run_dir = (
                    root / "drilldowns" / "captures" / request_id
                    / "runs" / label / "scene-a"
                )
                run_dir.mkdir(parents=True)
                dmap_dev.write_json(run_dir / "repro.json", {
                    "return_code": 0,
                    "dry_run": False,
                    "command": ["DensifyPointCloudDMapObserve"],
                })
            observer = root / "DensifyPointCloudDMapObserve"
            observer.write_bytes(b"observer")
            config = {
                "_config_path": str(config_path),
                "runs": [
                    {"label": "base", "role": "baseline"},
                    {"label": "candidate", "role": "variant"},
                ],
            }
            scene = {
                "scan_id": "scene-a",
                "working_folder": str(root / "source"),
                "mvs_file": str(root / "source" / "scene.mvs"),
            }

            with mock.patch.object(
                dmap_dev.dmap_drilldown, "load_request", return_value=request
            ), mock.patch.object(
                dmap_dev, "resolve_scenes", return_value=[scene]
            ), mock.patch.object(
                dmap_dev, "resolve_suite", return_value=["scene-a"]
            ), mock.patch.object(
                dmap_dev, "densify_binary", return_value=observer
            ), mock.patch.object(
                dmap_dev, "drilldown_run_complete", return_value=(True, "valid")
            ):
                executions = dmap_dev.execute_drilldown_request(
                    config, root, request_path
                )

            self.assertEqual(executions[0]["capture_profile"], "trace")
            self.assertEqual(executions[0]["trace_pixels"], 2)
            self.assertTrue(executions[0]["reused"])
            self.assertEqual(len(executions), 2)
            execution_manifest = dmap_dev.read_json(
                root / "drilldowns" / "captures" / request_id / "executions.json"
            )
            self.assertEqual(
                execution_manifest["executions"][0]["capture_profile"], "trace"
            )
            index = dmap_dev.read_json(root / "drilldowns" / "index.json")
            self.assertEqual(index["entries"][0]["status"], "complete")

    def test_drilldown_command_materializes_run_and_scene_ini_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            source_ini = work / "Densify.ini"
            source_ini.write_text(
                "[Densify]\nPatchMatch CUDA View Samples = 32\nSpeckle Size = 100\n",
                encoding="utf-8",
            )
            source_before = source_ini.read_bytes()
            run_dir = root / "capture"
            request = {"capture_profile": "deep", "target": {"image_id": 7}}
            command = dmap_dev.drilldown_run_command(
                {
                    "densify_bin": str(root / "DensifyPointCloud"),
                    "default_densify_args": [
                        "--dense-config-file", "Densify.ini",
                        "--max-resolution", "2560",
                    ],
                    "sweep": {"ini_overrides": {"Speckle Size": "150"}},
                },
                {
                    "label": "view_samples24",
                    "densify_args": [],
                    "ini_overrides": {"PatchMatch CUDA View Samples": "24"},
                },
                request,
                run_dir,
                work,
                work / "scene.mvs",
                None,
                run_dir / "generated" / "Densify.drilldown.cfg",
                {
                    "argument_overrides": {"--max-resolution": "1920"},
                    "ini_overrides": {"Speckle Size": "200"},
                },
            )

            generated = Path(
                command[command.index("--dense-config-file") + 1]
            )
            self.assertEqual(source_ini.read_bytes(), source_before)
            self.assertEqual(generated, run_dir / "generated" / "Densify.drilldown.ini")
            rendered = generated.read_text(encoding="utf-8")
            self.assertIn("PatchMatch CUDA View Samples = 24", rendered)
            self.assertIn("Speckle Size = 200", rendered)
            self.assertEqual(
                command[command.index("--max-resolution") + 1], "1920"
            )
            metadata = dmap_dev.read_json(
                run_dir / "generated" / "Densify.drilldown.json"
            )
            self.assertEqual(metadata["overrides"]["Speckle Size"], "200")
            self.assertEqual(metadata["run"], "view_samples24")

    def test_map_difference_requires_matching_shape_and_finite_domain(self) -> None:
        values = np.asarray([[1.0, np.nan], [3.0, 4.0]])

        self.assertEqual(
            validate_dmap_instrumentation.max_abs_difference(values, values.copy()),
            0.0,
        )
        self.assertTrue(np.isinf(validate_dmap_instrumentation.max_abs_difference(values, values[:, :1])))
        changed_domain = values.copy()
        changed_domain[0, 1] = 2.0
        self.assertTrue(np.isinf(validate_dmap_instrumentation.max_abs_difference(values, changed_domain)))

    def test_mechanism_plot_categories_cover_cost_view_filtering_update_and_runtime(self) -> None:
        cases = {
            "Cost by PatchMatch iteration": "cost",
            "per-frame low-texture and bad-cost rates": "cost",
            "Supporting-view histogram": "view",
            "per-frame candidate acceptance rates": "update",
            "Valid/rejected ratios": "filtering",
            "per-frame kernel timing": "runtime",
        }

        self.assertEqual(
            {title: dmap_dev.mechanism_for_plot(title) for title in cases},
            cases,
        )

    def test_disabled_contract_helpers_compare_exact_dmaps(self) -> None:
        values = np.asarray([[1.0, 2.0]], dtype=np.float32)

        self.assertEqual(validate_dmap_disabled.dmap_image_id(Path("depth0043.dmap")), 43)
        self.assertIsNone(validate_dmap_disabled.dmap_image_id(Path("cost0043.pfm")))
        self.assertEqual(validate_dmap_disabled.max_abs_difference(values, values.copy()), 0.0)

    def test_trace_pixel_excludes_border_and_avoids_reuse(self) -> None:
        mask = np.ones((16, 16), dtype=bool)
        score = np.zeros((16, 16), dtype=float)
        score[0, 0] = 100.0
        score[6, 6] = 10.0
        used: set[tuple[int, int]] = set()

        first = dmap_dev.select_trace_pixel(mask.copy(), score, True, used)
        second = dmap_dev.select_trace_pixel(mask.copy(), score, True, used)

        self.assertEqual(first, (6, 6))
        self.assertEqual(second, (7, 6))

    def test_exact_trace_recommendations_cover_all_eight_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            baseline_cost = np.full(shape, 0.5, dtype=np.float32)
            candidate_cost = baseline_cost.copy()
            candidate_cost[1, 1] = 100.0  # Highest delta is excluded by the border.
            candidate_cost[6, 6] = 2.0
            baseline_gap = np.full(shape, 0.5, dtype=np.float32)
            candidate_gap = baseline_gap.copy()
            candidate_gap[6, 8] = 0.1
            candidate_gap[6, 9] = 1.0
            baseline_mask = np.ones(shape, dtype=np.uint32)
            candidate_mask = baseline_mask.copy()
            candidate_mask[6, 7] = 2
            baseline_depth = np.full(shape, 10.0, dtype=np.float32)
            candidate_depth = baseline_depth.copy()
            candidate_depth[6, 12] = 20.0
            baseline_valid = np.ones(shape, dtype=bool)
            candidate_valid = np.ones(shape, dtype=bool)
            candidate_valid[6, 10] = False
            baseline_valid[6, 11] = False

            baseline_frame = root / "baseline"
            candidate_frame = root / "candidate"
            make_exact_trace_frame(
                baseline_frame, total_cost=baseline_cost, exact_gap=baseline_gap,
                view_mask=baseline_mask, depth=baseline_depth, valid=baseline_valid,
            )
            make_exact_trace_frame(
                candidate_frame, total_cost=candidate_cost, exact_gap=candidate_gap,
                view_mask=candidate_mask, depth=candidate_depth, valid=candidate_valid,
            )

            result = dmap_dev.evidence_trace_recommendations(
                exact_trace_frame_row(baseline_frame, run="baseline"),
                exact_trace_frame_row(candidate_frame, run="candidate"),
            )

            self.assertEqual(result["mode"], "schema4_terminal_geometric_exact")
            self.assertEqual(
                [row["selection"] for row in result["selected"]],
                list(dmap_dev.TRACE_EXACT_CATEGORIES),
            )
            self.assertEqual(len({(row["x"], row["y"]) for row in result["selected"]}), 8)
            self.assertTrue(all(row["x"] >= 6 and row["y"] >= 6 for row in result["selected"]))
            self.assertEqual(result["selected"][0]["x"], 6)
            self.assertEqual(result["selected"][0]["y"], 6)
            self.assertTrue(all(row["status"] == "selected" for row in result["availability"]))
            gap_rows = [
                row for row in result["availability"] if "gap" in row["category"]
            ]
            self.assertEqual(
                {tuple(row["source_signals"]) for row in gap_rows},
                {("gap_winner_runner_up_exact",)},
            )
            self.assertNotIn("confidence_gap", json.dumps(gap_rows))

    def test_exact_view_mask_requires_source_index_mapping_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            scalar = np.full(shape, 0.5, dtype=np.float32)
            depth = np.full(shape, 10.0, dtype=np.float32)
            valid = np.ones(shape, dtype=bool)
            baseline_mask = np.ones(shape, dtype=np.uint32)
            candidate_mask = baseline_mask.copy()
            candidate_mask[6, 6] = 2
            make_exact_trace_frame(
                root / "baseline", total_cost=scalar, exact_gap=scalar,
                view_mask=baseline_mask, depth=depth, valid=valid,
                source_image_ids=(20, 21),
            )
            make_exact_trace_frame(
                root / "candidate", total_cost=scalar, exact_gap=scalar,
                view_mask=candidate_mask, depth=depth, valid=valid,
                source_image_ids=(30, 21),
            )

            result = dmap_dev.evidence_trace_recommendations(
                exact_trace_frame_row(root / "baseline", run="baseline"),
                exact_trace_frame_row(root / "candidate", run="candidate"),
            )
            availability = {row["category"]: row for row in result["availability"]}

            self.assertEqual(result["mode"], "schema4_terminal_geometric_exact")
            self.assertEqual(availability["exact_view_mask_xor"]["status"], "unavailable")
            self.assertIn("mapping differs", availability["exact_view_mask_xor"]["reason"])
            self.assertEqual(
                availability["stable_same_view_mask_control"]["status"], "unavailable"
            )
            self.assertEqual(
                result["context"]["baseline_source_view_mapping"][0]["source_image_id"], 20
            )
            self.assertEqual(
                result["context"]["candidate_source_view_mapping"][0]["source_image_id"], 30
            )

    def test_exact_trace_rejects_wrong_dimensions_counters_and_future_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            scalar = np.full(shape, 0.5, dtype=np.float32)
            depth = np.full(shape, 10.0, dtype=np.float32)
            valid = np.ones(shape, dtype=bool)
            for run in ("baseline", "candidate"):
                make_exact_trace_frame(
                    root / run, total_cost=scalar, exact_gap=scalar,
                    view_mask=np.ones(shape, dtype=np.uint32), depth=depth, valid=valid,
                )
            baseline_row = exact_trace_frame_row(root / "baseline", run="baseline")
            candidate_row = exact_trace_frame_row(root / "candidate", run="candidate")

            wrong_size = np.full((30, 30), 0.9, dtype=np.float32)
            write_pfm(
                root / "candidate" / "logical_states" / "state01_iteration01"
                / "cost_total_production_exact.pfm",
                wrong_size,
            )
            dimensions = dmap_dev.evidence_trace_recommendations(
                baseline_row, candidate_row
            )
            by_category = {
                row["category"]: row for row in dimensions["availability"]
            }
            self.assertEqual(
                by_category["exact_total_cost_delta"]["status"], "unavailable"
            )
            self.assertIn(
                "artifact dimensions disagree",
                by_category["exact_total_cost_delta"]["reason"],
            )
            self.assertTrue(all(
                row["x"] < 20 and row["y"] < 20 for row in dimensions["selected"]
            ))

            manifest_path = root / "candidate" / "map_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["num_iterations"] = 5
            dmap_dev.write_json(manifest_path, manifest)
            counters = dmap_dev.evidence_trace_recommendations(
                baseline_row, candidate_row
            )
            self.assertEqual(counters["mode"], "legacy_fallback")
            self.assertIn("logical counters", str(counters["fallback_reason"]))

            manifest["num_iterations"] = 1
            manifest["schema_version"] = 5
            dmap_dev.write_json(manifest_path, manifest)
            future = dmap_dev.evidence_trace_recommendations(
                baseline_row, candidate_row
            )
            self.assertEqual(future["mode"], "legacy_fallback")
            self.assertIn("expected exactly 4", str(future["fallback_reason"]))

            manifest["schema_version"] = 4
            dmap_dev.write_json(manifest_path, manifest)
            wrong_frame = candidate_row.copy()
            wrong_frame["width"] = 19
            frame_dimensions = dmap_dev.evidence_trace_recommendations(
                baseline_row, wrong_frame
            )
            self.assertEqual(frame_dimensions["mode"], "legacy_fallback")
            self.assertIn("frame/manifest dimensions differ", str(
                frame_dimensions["fallback_reason"]
            ))

    def test_trace_view_mask_decodes_all_rgba_bytes_little_endian(self) -> None:
        values = np.asarray([[[0xD4, 0xC3, 0xB2, 0xA1]]], dtype=np.uint8)

        decoded = dmap_dev.decode_trace_view_mask(values)

        self.assertEqual(int(decoded[0, 0]), 0xA1B2C3D4)

    def test_legacy_trace_fallback_labels_confidence_gap_as_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            rows = []
            for run in ("baseline", "candidate"):
                frame = root / run
                dmap_dev.write_json(frame / "map_manifest.json", {
                    "schema_name": "openmvs.dmap.map_manifest", "schema_version": 3,
                    "maps": [],
                })
                write_pfm(frame / "maps" / "depth_final_after_filter.pfm", np.full(shape, 10.0))
                write_pfm(frame / "maps" / "confidence_gap.pfm", np.full(shape, 0.2))
                valid_path = frame / "maps" / "valid_after_filter.png"
                valid_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full(shape, 255, dtype=np.uint8), mode="L").save(valid_path)
                rows.append(exact_trace_frame_row(frame, run=run))

            result = dmap_dev.evidence_trace_recommendations(rows[0], rows[1])
            proxy = next(
                row for row in result["selected"]
                if row["selection"] == "lowest_nonnegative_confidence_gap_proxy"
            )

            self.assertEqual(result["mode"], "legacy_fallback")
            self.assertEqual(proxy["measurement_quality"], "proxy")
            self.assertEqual(proxy["source_signals"], ["confidence_gap"])
            exact_unavailable = [
                row for row in result["availability"]
                if row["category"] in dmap_dev.TRACE_EXACT_CATEGORIES
            ]
            self.assertEqual(len(exact_unavailable), 8)
            self.assertTrue(all(row["status"] == "unavailable" for row in exact_unavailable))

    def test_trace_manifest_reselects_terminal_geometric_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            scalar = np.full(shape, 0.5, dtype=np.float32)
            depth = np.full(shape, 10.0, dtype=np.float32)
            valid = np.ones(shape, dtype=bool)
            rows = []
            for run in ("baseline", "candidate"):
                for stage, geometric_iteration in (("photometric", None), ("geometric_consistency", 3)):
                    frame = root / run / stage
                    make_exact_trace_frame(
                        frame, total_cost=scalar, exact_gap=scalar,
                        view_mask=np.ones(shape, dtype=np.uint32), depth=depth, valid=valid,
                        estimation_stage=stage, geometric_iteration=geometric_iteration,
                    )
                    row = exact_trace_frame_row(
                        frame, run=run, estimation_stage=stage,
                        geometric_iteration=geometric_iteration,
                    )
                    rows.append(row.to_dict())
            output = root / "report"
            output.mkdir()

            path = dmap_dev.generate_trace_manifest(
                {
                    "runs": [
                        {"label": "baseline", "role": "baseline"},
                        {"label": "candidate", "role": "variant"},
                    ]
                },
                pd.DataFrame(rows),
                output,
            )
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))

            exact_rows = [
                row for row in manifest["category_availability"]
                if row["category"] in dmap_dev.TRACE_EXACT_CATEGORIES
            ]
            self.assertTrue(exact_rows)
            self.assertEqual({row["selection_mode"] for row in exact_rows}, {
                "schema4_terminal_geometric_exact"
            })
            self.assertEqual({row["baseline_terminal_logical_iteration"] for row in exact_rows}, {0})
            self.assertTrue(all("terminal geometric" not in str(row.get("reason")) for row in exact_rows))

    def test_trace_manifest_pairs_profile_qualified_rows_by_configured_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (20, 20)
            baseline_cost = np.full(shape, 0.5, dtype=np.float32)
            candidate_cost = baseline_cost.copy()
            candidate_cost[8, 8] = 0.9
            gap = np.full(shape, 0.2, dtype=np.float32)
            depth = np.full(shape, 10.0, dtype=np.float32)
            valid = np.ones(shape, dtype=bool)
            rows = []
            for configured_run, cost in (
                ("baseline", baseline_cost), ("candidate", candidate_cost)
            ):
                frame = root / configured_run / "deep"
                make_exact_trace_frame(
                    frame, total_cost=cost, exact_gap=gap,
                    view_mask=np.ones(shape, dtype=np.uint32), depth=depth, valid=valid,
                    estimation_stage="geometric_consistency", geometric_iteration=3,
                )
                row = exact_trace_frame_row(
                    frame, run=f"{configured_run} [deep]",
                    estimation_stage="geometric_consistency", geometric_iteration=3,
                ).to_dict()
                rows.append({
                    **row,
                    "configured_run": configured_run,
                    "diagnostic_only": True,
                })
                # The Process<false> quality cohort shares the configured run
                # identity but is not a source of exact trace recommendations.
                rows.append({
                    **row,
                    "run": configured_run,
                    "depthmap_dir": str(root / configured_run / "summary"),
                    "configured_run": configured_run,
                    "diagnostic_only": False,
                })
            output = root / "report"
            output.mkdir()

            path = dmap_dev.generate_trace_manifest(
                {
                    "runs": [
                        {"label": "baseline", "role": "baseline"},
                        {"label": "candidate", "role": "variant"},
                    ]
                },
                pd.DataFrame(rows),
                output,
            )
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))

            self.assertTrue(manifest["trace_pixels"])
            self.assertEqual({row["variant"] for row in manifest["trace_pixels"]}, {
                "candidate"
            })
            exact_rows = [
                row for row in manifest["category_availability"]
                if row["category"] in dmap_dev.TRACE_EXACT_CATEGORIES
            ]
            self.assertEqual(len(exact_rows), len(dmap_dev.TRACE_EXACT_CATEGORIES))
            self.assertEqual({row["selection_mode"] for row in exact_rows}, {
                "schema4_terminal_geometric_exact"
            })
            self.assertIn(
                "explicit configured-run identity", manifest["selection_method"]
            )

    def test_trace_manifest_rejects_unmapped_profile_qualified_rows(self) -> None:
        frames = pd.DataFrame([{
            "run": "baseline [deep]", "repeat": 0, "scene_id": "scene",
            "image_id": 7, "estimation_stage": "geometric_consistency",
            "geometric_iteration": 3,
        }])

        with self.assertRaisesRegex(ValueError, "cannot resolve configured run 'baseline'"):
            dmap_dev.generate_trace_manifest(
                {
                    "runs": [
                        {"label": "baseline", "role": "baseline"},
                        {"label": "candidate", "role": "variant"},
                    ]
                },
                frames,
                Path("unused"),
            )

    def test_trace_capture_frames_rejects_ambiguous_cohorts(self) -> None:
        frames = pd.DataFrame([
            {
                "run": label, "configured_run": "baseline",
                "diagnostic_only": False, "repeat": 0, "scene_id": "scene",
                "image_id": 7, "estimation_stage": "geometric_consistency",
                "geometric_iteration": 3,
            }
            for label in ("baseline-a", "baseline-b")
        ])

        with self.assertRaisesRegex(ValueError, "cannot choose an unambiguous cohort"):
            dmap_dev.trace_capture_frames(frames, "baseline")

    def test_reference_dmap_does_not_fall_back_to_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own = root / "own"
            other = root / "other"
            other.mkdir()
            (other / "depth0007.dmap").touch()
            run = dmap_dev.RunScene("candidate", "variant", 0, "scene", root, own, None)

            self.assertIsNone(dmap_dev.find_reference_dmap(run, 7, None))
            self.assertEqual(
                dmap_dev.find_reference_dmap(run, 7, {"reference_dmap_dir": str(other)}),
                other / "depth0007.dmap",
            )

    def test_model_stability_pairs_repeats_without_cartesian_join(self) -> None:
        base_rows = []
        candidate_rows = []
        for repeat in (0, 1):
            common = {
                "scene_id": "scene", "image_id": 7, "repeat": repeat, "annotation_kind": "plane",
                "object_id": "object", "chunk_id": "chunk", "stage": "post_filter",
                "plane_normal_x": 0.0, "plane_normal_y": 0.0, "plane_normal_z": 1.0,
                "plane_d": -1.0, "model_point_x": 0.0, "model_point_y": 0.0, "model_point_z": 1.0,
            }
            base_rows.append(common)
            candidate_rows.append({**common, "plane_d": -1.0 - 0.01 * repeat})
        keys = ["scene_id", "image_id", "repeat", "annotation_kind", "object_id", "chunk_id", "stage"]

        result = dmap_dev.add_model_stability(pd.DataFrame(base_rows), pd.DataFrame(candidate_rows), keys)

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["repeat"]), {0, 1})

    def test_model_switch_is_flagged_near_regression_with_fixed_model_metric(self) -> None:
        keys = [
            "scene_id", "image_id", "repeat", "annotation_kind",
            "object_id", "chunk_id", "stage",
        ]
        common = {
            "scene_id": "scene", "image_id": 7, "repeat": 0,
            "annotation_kind": "edge", "object_id": "edge", "chunk_id": "chunk",
            "stage": "post_filter", "all_residual_p95_m": 0.01,
            "line_direction_x": 1.0, "line_direction_y": 0.0,
            "line_direction_z": 0.0, "model_point_x": 0.0,
            "model_point_y": 0.0, "model_point_z": 1.0,
            "line_extent_length_m": 1.0,
        }
        candidate = {
            **common,
            "line_direction_x": 0.0,
            "line_direction_y": 1.0,
            "line_extent_length_m": 0.3,
            "all_residual_p95_m": 0.005,
            "baseline_model_on_candidate_status": "available",
            "baseline_model_on_candidate_all_residual_p95_m": 0.2,
        }

        stability = dmap_dev.add_model_stability(
            pd.DataFrame([common]), pd.DataFrame([candidate]), keys
        )
        stability["candidate"] = "variant"
        result = dmap_dev.annotate_model_switch_regression_proximity(
            stability,
            pd.DataFrame([{
                "candidate": "variant", "scene_id": "scene",
                "metric": "inlier_fraction_5mm", "normalized_regression_loss": 1.5,
            }]),
        )

        row = result.iloc[0]
        self.assertTrue(row["large_model_switch"])
        self.assertTrue(row["near_candidate_regression"])
        self.assertIn("line_direction", row["model_switch_reasons_json"])
        self.assertEqual(
            row["baseline_model_on_candidate_all_residual_p95_m"], 0.2
        )

    def test_fixed_baseline_plane_model_is_evaluated_on_candidate_points(self) -> None:
        candidate = {"annotation_kind": "plane", "coverage_fraction": 0.5}
        baseline = {
            "run": "base", "plane_normal_x": 0.0, "plane_normal_y": 0.0,
            "plane_normal_z": 1.0, "model_point_x": 0.0,
            "model_point_y": 0.0, "model_point_z": 1.0,
        }
        points = np.asarray([
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.01], [0.0, 1.0, 0.99],
        ])

        dmap_dev.apply_baseline_model_cross_evaluation(
            candidate,
            points,
            baseline,
            dmap_dev.make_annotation_fit_config({}),
        )

        self.assertEqual(candidate["baseline_model_on_candidate_status"], "available")
        self.assertAlmostEqual(
            candidate["baseline_model_on_candidate_all_residual_p95_m"], 0.01
        )
        self.assertEqual(candidate["baseline_model_on_candidate_source_run"], "base")

    def test_annotation_evaluation_records_missing_mapping(self) -> None:
        frame_rows = [{
            "run": "candidate", "role": "variant", "repeat": 0, "scene_id": "scene",
            "image_id": 7, "image_name": "images/0007.jpg", "depthmap_dir": "/missing",
        }]
        run_scene = dmap_dev.RunScene("candidate", "variant", 0, "scene", Path("/tmp/instrumentation"), None, None)
        context = ({}, {"annotations": {}}, {}, {"frame-1": "images/0008.jpg"})
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(dmap_dev, "scene_context", return_value=context):
            config = {"_config_path": str(Path(directory) / "config.yaml"), "evaluation": {}, "scenes": []}
            rows = dmap_dev.evaluate_annotations(config, [run_scene], frame_rows, Path(directory))

        self.assertEqual(rows[0]["fit_status"], "missing_annotation_image_mapping")
        self.assertIn("not present", rows[0]["error"])

    def test_annotation_fit_prefers_resolution_aware_source_ribbon(self) -> None:
        preferred = dmap_dev.make_annotation_fit_config({
            "edge_ribbon_source_px": 4.0,
            "edge_ribbon_px": 9.0,
        })
        legacy = dmap_dev.make_annotation_fit_config({"edge_ribbon_px": 6.0})
        default = dmap_dev.make_annotation_fit_config({})

        self.assertEqual(preferred.edge_ribbon_source_px, 4.0)
        self.assertIsNone(preferred.edge_ribbon_px)
        self.assertEqual(legacy.edge_ribbon_px, 6.0)
        self.assertIsNone(legacy.edge_ribbon_source_px)
        self.assertEqual(default.edge_ribbon_source_px, 5.0)
        self.assertIsNone(default.edge_ribbon_px)

    def test_annotation_presentation_formats_units_and_metric_direction(self) -> None:
        self.assertEqual(dmap_dev.fmt_percent(0.8134), "81.34%")
        self.assertEqual(dmap_dev.fmt_percentage_points(0.0456), "+4.56 pp")
        self.assertEqual(dmap_dev.fmt_millimetres(0.05141), "51.41 mm")
        self.assertEqual(dmap_dev.fmt_millimetres(-0.00764, signed=True), "-7.64 mm")
        self.assertEqual(dmap_dev.directional_status(0.81, 0.86, "higher"), "improved")
        self.assertEqual(dmap_dev.directional_status(0.051, 0.044, "lower"), "improved")
        self.assertEqual(dmap_dev.directional_status(0.81, 0.80, "higher"), "regressed")
        self.assertEqual(dmap_dev.directional_status(0.81, 0.81, "higher"), "stable")
        self.assertEqual(dmap_dev.directional_status(None, 0.81, "higher"), "unavailable")

    def test_annotation_metric_table_marks_directional_status(self) -> None:
        rendered = dmap_dev.annotation_metric_table([{
            "metric": "Effective inliers @20 mm",
            "baseline": "81.34%",
            "variant": "85.90%",
            "delta": "+4.56 pp",
            "direction": "higher",
            "status": "improved",
        }])

        self.assertIn("Effective inliers @20 mm", rendered)
        self.assertIn("+4.56 pp", rendered)
        self.assertIn('annotation-change improved', rendered)

    def test_ignore_mask_states_preserve_unavailable_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instrumentation = Path(directory) / "instrumentation"
            states = [
                (1, {"requested": False, "loaded": None, "status": "not_requested",
                     "rejection_count_available": True, "unavailable_reason": ""}, 0),
                (2, {"requested": True, "loaded": True, "status": "loaded",
                     "rejection_count_available": True, "unavailable_reason": ""}, 5),
                (3, {"requested": True, "loaded": False, "status": "unavailable",
                     "rejection_count_available": False, "unavailable_reason": "decode failed"}, None),
            ]
            for image_id, ignore_mask, rejected in states:
                dmap_dev.write_json(
                    instrumentation / "depthmaps" / f"{image_id:04d}" / "summary.json",
                    {
                        "schema_version": 4, "image_id": image_id, "image_name": f"{image_id:04d}.jpg",
                        "safe_image_name": f"{image_id:04d}", "num_pixels_total": 100,
                        "num_rejected_by_filter": 5, "num_rejected_by_keep_cost_filter": 5,
                        "num_rejected_by_ignore_mask": rejected, "ignore_mask": ignore_mask,
                    },
                )
            run = dmap_dev.RunScene("run", "baseline", 0, "scene", instrumentation, None, None)

            frames, _iterations, _timings = dmap_dev.load_instrumentation(run)
            by_id = {row["image_id"]: row for row in frames}

            self.assertEqual(by_id[1]["ignore_mask_status"], "not_requested")
            self.assertEqual(by_id[1]["num_rejected_by_ignore_mask"], 0)
            self.assertEqual(dmap_dev.mask_rejection_display(by_id[1], ratio=False), "n/a")
            self.assertEqual(by_id[2]["num_rejected_by_ignore_mask"], 5)
            self.assertEqual(dmap_dev.mask_rejection_display(by_id[2], ratio=False), "5")
            self.assertIsNone(by_id[3]["num_rejected_by_ignore_mask"])
            self.assertIsNone(by_id[3]["rejected_by_ignore_mask_ratio"])
            self.assertIn("decode failed", dmap_dev.mask_rejection_display(by_id[3], ratio=False))

    def test_schema4_storage_estimate_matches_conservative_cpp_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "experiment_id": "storage",
                "output_root": str(root / "outputs"),
                "instrumentation": {
                    "expected_width": 708, "expected_height": 944,
                    "expected_frames_per_scene": 1, "max_artifact_gb": 10,
                },
                "default_densify_args": ["--number-views", "3"],
                "runs": [
                    {"label": "iter1", "role": "baseline", "repeats": 1, "densify_args": ["--iters", "1"]},
                    {"label": "iter5", "role": "variant", "repeats": 1, "densify_args": ["--iters", "5"]},
                ],
            }

            estimate = dmap_dev.estimate_storage(config, [{"scan_id": "scene", "expected_frames": 1}])
            runs = {row["label"]: row for row in estimate["runs"]}

            self.assertEqual(runs["iter1"]["filter_bytes_per_frame_raw"], 93_704_448)
            self.assertFalse(runs["iter1"]["low_texture_update_hysteresis"])
            self.assertEqual(runs["iter1"]["exact_bytes_per_pixel"], 358)
            self.assertEqual(runs["iter1"]["exact_artifact_count"], 70)
            self.assertEqual(runs["iter1"]["exact_fixed_bytes"], 352_256)
            self.assertEqual(runs["iter1"]["legacy_bytes_per_pixel"], 299)
            self.assertEqual(runs["iter1"]["bytes_per_frame_raw"], 533_163_968)
            self.assertEqual(runs["iter5"]["legacy_bytes_per_pixel"], 591)
            self.assertEqual(runs["iter5"]["bytes_per_frame_raw"], 1_207_436_224)
            self.assertEqual(estimate["estimated_bytes"], 1_740_600_192)
            self.assertIn("159+6P+61L", estimate["storage_model"])

            default_config = dict(config)
            default_config["runs"] = []
            default_estimate = dmap_dev.estimate_storage(
                default_config, [{"scan_id": "scene", "expected_frames": 1}]
            )
            self.assertEqual(
                default_estimate["runs"][0]["legacy_bytes_per_pixel"], 445
            )

            enabled = dict(config)
            enabled["runs"] = [{
                "label": "iter13", "role": "variant", "repeats": 1,
                "densify_args": ["--iters", "12"],
                "ini_overrides": {
                    "PatchMatch CUDA Low Texture Update Min Gain": "0.0005",
                    "PatchMatch CUDA Low Texture Update Gate": "3",
                },
            }]
            enabled_estimate = dmap_dev.estimate_storage(
                enabled, [{"scan_id": "scene", "expected_frames": 1}]
            )
            enabled_run = enabled_estimate["runs"][0]
            self.assertTrue(enabled_run["low_texture_update_hysteresis"])
            self.assertEqual(enabled_run["exact_bytes_per_pixel"], 2_903)
            self.assertEqual(enabled_run["exact_artifact_count"], 599)
            self.assertGreater(
                enabled_run["exact_bytes_per_pixel"], 13 * (121 + 34 * 3)
            )

    def test_storage_estimate_rejects_mismatched_hysteresis_enablement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "experiment_id": "invalid-hysteresis-storage",
                "output_root": str(root / "outputs"),
                "instrumentation": {"expected_width": 16, "expected_height": 12},
                "runs": [{
                    "label": "invalid", "role": "variant",
                    "ini_overrides": {
                        "PatchMatch CUDA Low Texture Update Min Gain": "0.0005",
                        "PatchMatch CUDA Low Texture Update Gate": "0",
                    },
                }],
            }

            with self.assertRaisesRegex(ValueError, "enabled or disabled together"):
                dmap_dev.estimate_storage(
                    config, [{"scan_id": "scene", "expected_frames": 1}]
                )

    def test_storage_estimate_accounts_for_each_geometric_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "_config_path": str(root / "config.yaml"),
                "experiment_id": "geometric-storage",
                "output_root": str(root / "outputs"),
                "instrumentation": {
                    "expected_width": 16,
                    "expected_height": 12,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "default_densify_args": ["--number-views", "3"],
                "runs": [{
                    "label": "geometric",
                    "role": "baseline",
                    "repeats": 1,
                    "densify_args": ["--iters", "1", "--geometric-iters", "4"],
                }],
            }

            estimate = dmap_dev.estimate_storage(
                config, [{"scan_id": "scene", "expected_frames": 1}]
            )
            run = estimate["runs"][0]

            self.assertEqual(run["geometric_iterations"], 4)
            self.assertEqual(run["estimation_stages"], 5)
            self.assertEqual(run["bytes_per_frame_raw"], 5 * run["bytes_per_stage_raw"])

    def test_frozen_scene_input_is_independent_and_detects_copy_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            image = source / "images" / "0001.jpg"
            image.write_bytes(b"original-image")
            mvs = source / "scene.mvs"
            mvs.write_bytes(b"scene")
            (source / "Densify.ini").write_text("[Densify]\n", encoding="utf-8")
            (source / "depth0001.dmap").write_bytes(b"generated")
            scene = {
                "scan_id": "scene-a",
                "working_folder": str(source),
                "mvs_file": str(mvs),
            }
            config = {"input_snapshot": {"max_files": 100, "max_bytes": 4096}}
            snapshot = dmap_dev.configured_scene_input_snapshot(config, scene)
            record = {
                "scan_id": "scene-a",
                "mvs_file": dmap_dev.file_identity(mvs),
                "staged_input_snapshot": snapshot,
            }

            frozen = dmap_dev.prepare_frozen_scene_input(
                config, root / "experiment", scene, record
            )

            self.assertEqual((frozen / "images" / "0001.jpg").read_bytes(), b"original-image")
            self.assertFalse((frozen / "depth0001.dmap").exists())
            self.assertNotEqual(
                image.stat().st_ino, (frozen / "images" / "0001.jpg").stat().st_ino
            )
            image.write_bytes(b"changed-image!")
            self.assertEqual((frozen / "images" / "0001.jpg").read_bytes(), b"original-image")

            race_root = root / "race-experiment"
            image.write_bytes(b"original-image")
            original_copy = dmap_dev.shutil.copy2
            raced = False

            def mutate_before_copy(source_path: str, destination_path: str, *args: object, **kwargs: object) -> str:
                nonlocal raced
                if Path(source_path) == image and not raced:
                    raced = True
                    image.write_bytes(b"same-size-change")
                return str(original_copy(source_path, destination_path, *args, **kwargs))

            with mock.patch.object(
                dmap_dev.shutil, "copy2", side_effect=mutate_before_copy
            ), self.assertRaisesRegex(RuntimeError, "does not match the locked input snapshot"):
                dmap_dev.prepare_frozen_scene_input(
                    config, race_root, scene, record
                )
            self.assertFalse(dmap_dev.frozen_scene_work_dir(race_root, "scene-a").exists())

            symlink_root = root / "symlink-experiment"
            redirected = root / "redirected"
            redirected.mkdir()
            frozen_parent = symlink_root / "frozen_inputs" / "scene-a"
            frozen_parent.mkdir(parents=True)
            (frozen_parent / "work").symlink_to(redirected, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "path contains a symlink"):
                dmap_dev.prepare_frozen_scene_input(
                    config, symlink_root, scene, record
                )

    def test_dry_run_uses_plan_tree_and_revalidates_live_inputs_per_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            source = Path(directory) / "source"
            (source / "images").mkdir(parents=True)
            image = source / "images" / "0001.jpg"
            image.write_bytes(b"original-image")
            mvs = source / "scene.mvs"
            mvs.write_bytes(b"scene")
            (source / "Densify.ini").write_text("[Densify]\n", encoding="utf-8")
            scene = {
                "scan_id": "scene-a",
                "name": "scene-a",
                "working_folder": str(source),
                "mvs_file": str(mvs),
            }
            config = {
                "_config_path": str(Path(directory) / "config.yaml"),
                "output_root": str(Path(directory) / "outputs"),
                "experiment_id": "experiment",
                "suite": {"scan_ids": ["scene-a"]},
                "scenes": [scene],
                "capture_profiles": ["endpoint", "summary", "prefilter"],
                "instrumentation": {"sample_rate": 0.25},
                "densify_bin": str(Path(directory) / "DensifyPointCloud"),
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }
            record = {
                "scan_id": "scene-a",
                "mvs_file": dmap_dev.file_identity(mvs),
                "staged_input_snapshot": dmap_dev.configured_scene_input_snapshot(
                    config, scene
                ),
            }
            dmap_dev.write_json(root / "00_experiment_lock.json", {
                "schema_name": "openmvs.dmap.experiment_lock",
                "schema_version": 3,
                "scenes": [record],
            })
            dmap_dev.prepare_frozen_scene_input(config, root, scene, record)
            original_execute = dmap_dev.execute_command
            calls = 0

            def execute_and_mutate(*args: object, **kwargs: object) -> dict:
                nonlocal calls
                result = original_execute(*args, **kwargs)
                calls += 1
                if calls == 2:
                    image.write_bytes(b"changed-image!")
                return result

            with mock.patch.object(
                dmap_dev, "execute_command", side_effect=execute_and_mutate
            ), self.assertRaisesRegex(RuntimeError, "staged input snapshot changed"):
                dmap_dev.run_experiment(config, root, dry_run=True)

            final_run = root / "runs" / "base" / "repeat_00" / "scene-a" / "endpoint"
            self.assertFalse((root / "runs").exists())
            self.assertEqual(dmap_dev.existing_run_mode_action(final_run, "endpoint"), "run")
            self.assertTrue(
                (root / "plans" / "dry_run" / "base" / "repeat_00" / "scene-a"
                 / "endpoint" / "command.sh").is_file()
            )
            summary_command = (
                root / "plans" / "dry_run" / "base" / "repeat_00" / "scene-a"
                / "summary" / "command.sh"
            ).read_text(encoding="utf-8")
            self.assertIn("--dmap-instrumentation-sample-rate 0.25", summary_command)

    def test_capture_revalidates_hardlinked_inputs_after_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            source = Path(directory) / "source"
            (source / "images").mkdir(parents=True)
            image = source / "images" / "0001.jpg"
            image.write_bytes(b"original-image")
            mvs = source / "scene.mvs"
            mvs.write_bytes(b"scene")
            (source / "Densify.ini").write_text("[Densify]\n", encoding="utf-8")
            scene = {
                "scan_id": "scene-a", "name": "scene-a",
                "working_folder": str(source), "mvs_file": str(mvs),
            }
            config = {
                "_config_path": str(Path(directory) / "config.yaml"),
                "output_root": str(Path(directory)),
                "experiment_id": "experiment",
                "suite": {"scan_ids": ["scene-a"]},
                "scenes": [scene],
                "capture_profiles": ["endpoint"],
                "densify_bin": str(Path(directory) / "DensifyPointCloud"),
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
                "instrumentation": {
                    "expected_width": 16, "expected_height": 12,
                    "expected_frames_per_scene": 1, "max_artifact_gb": 1,
                },
            }
            record = {
                "scan_id": "scene-a",
                "mvs_file": dmap_dev.file_identity(mvs),
                "staged_input_snapshot": dmap_dev.configured_scene_input_snapshot(
                    config, scene
                ),
            }
            dmap_dev.write_json(root / "00_experiment_lock.json", {
                "schema_name": "openmvs.dmap.experiment_lock",
                "schema_version": 3,
                "scenes": [record],
            })
            dmap_dev.prepare_frozen_scene_input(config, root, scene, record)

            def corrupt_staged_input(
                command: list[str], _cwd: Path, _output: Path, _dry_run: bool, **_kwargs
            ) -> dict:
                local_mvs = Path(command[command.index("--input-file") + 1])
                staged_image = local_mvs.parent / "images" / "0001.jpg"
                staged_image.chmod(staged_image.stat().st_mode | 0o200)
                staged_image.write_bytes(b"changed-image!")
                return {"return_code": 0}

            with mock.patch.object(
                dmap_dev, "execute_densify_command", side_effect=corrupt_staged_input
            ), self.assertRaisesRegex(
                RuntimeError, "post-run frozen scene.*locked input snapshot"
            ):
                dmap_dev.run_experiment(config, root, dry_run=False)

    def test_on_demand_trace_storage_fails_closed_and_binds_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            root.mkdir()
            config = {
                "_config_path": str(Path(directory) / "config.yaml"),
                "output_root": str(Path(directory)),
                "experiment_id": "experiment",
                "capture_profiles": ["summary"],
                "instrumentation": {
                    "expected_width": 64,
                    "expected_height": 48,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 0.000001,
                },
            }
            runs = [
                {"label": "base", "role": "baseline", "repeats": 1},
                {"label": "variant", "role": "variant", "repeats": 1},
            ]
            scene = {"scan_id": "scene"}
            request_id = "a" * 64

            with self.assertRaisesRegex(RuntimeError, "storage admission failed"):
                dmap_dev.admit_on_demand_storage(
                    config,
                    root,
                    request_id=request_id,
                    kind="drilldown_trace",
                    run_scene_pairs=[(run, scene) for run in runs],
                    allow_over_budget=False,
                )

            plan = dmap_dev.read_json(
                root / "storage_admissions" / "drilldown_trace" / request_id / "plan.json"
            )
            self.assertEqual(plan["request_sha256"], request_id)
            self.assertEqual(len(plan["estimates"]), 2)
            admitted = dmap_dev.admit_on_demand_storage(
                config,
                root,
                request_id=request_id,
                kind="drilldown_trace",
                run_scene_pairs=[(run, scene) for run in runs],
                allow_over_budget=True,
            )
            self.assertTrue(admitted["receipt"]["admitted"])
            self.assertTrue(admitted["receipt"]["allow_over_budget"])

    def test_experiment_lock_rejects_changed_config_binary_or_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            annotations = dataset / "annotations"
            annotations.mkdir(parents=True)
            (annotations / "import_manifest.json").write_text("{}\n", encoding="utf-8")
            source = root / "source"
            (source / "mvs").mkdir(parents=True)
            (source / "mvs" / "scene.mvs").write_bytes(b"scene")
            (source / "mvs" / "Densify.ini").write_text("[Densify]\n", encoding="utf-8")
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True, version="v1")
            snapshot = root / "snapshot_manifest.json"
            snapshot.write_text('{"snapshot_sha256":"one"}\n', encoding="utf-8")
            config_path = root / "config.yaml"

            def write_config(hypothesis: str) -> None:
                config_path.write_text(yaml.safe_dump({
                    "schema_version": 2,
                    "experiment_id": "locked",
                    "output_root": str(root / "outputs"),
                    "dataset_root": str(dataset),
                    "input_snapshot_manifest": str(snapshot),
                    "densify_bin": str(production),
                    "densify_observe_bin": str(observer),
                    "hypothesis": hypothesis,
                    "suite": {"name": "smoke", "scan_ids": ["scene"]},
                    "scenes": [{
                        "scan_id": "scene",
                        "working_folder": str(source),
                        "mvs_file": str(source / "mvs" / "scene.mvs"),
                    }],
                    "instrumentation": {
                        "expected_width": 16,
                        "expected_height": 12,
                        "expected_frames_per_scene": 1,
                        "max_artifact_gb": 1,
                    },
                    "runs": [{
                        "label": "baseline", "role": "baseline", "repeats": 1,
                    }],
                }, sort_keys=False), encoding="utf-8")

            write_config("first")
            dmap_dev.prepare_experiment(config_path, False)
            dmap_dev.prepare_experiment(config_path, False)
            environment_path = (
                root / "outputs" / "locked" / dmap_dev.ENVIRONMENT_MANIFEST_FILE
            )
            environment = dmap_dev.read_json(environment_path)
            lock = dmap_dev.read_json(
                root / "outputs" / "locked" / "00_experiment_lock.json"
            )
            self.assertEqual(
                environment["schema_name"],
                dmap_dev.ENVIRONMENT_MANIFEST_SCHEMA_NAME,
            )
            self.assertFalse(environment["network_access_used"])
            self.assertEqual(
                lock["environment_manifest"],
                dmap_dev.file_identity(environment_path),
            )

            write_fake_densify_binary(observer, observer=True, version="v2")
            with self.assertRaisesRegex(RuntimeError, "use a new experiment_id"):
                dmap_dev.prepare_experiment(config_path, False)
            write_fake_densify_binary(observer, observer=True, version="v1")

            write_config("changed")
            with self.assertRaisesRegex(RuntimeError, "use a new experiment_id"):
                dmap_dev.prepare_experiment(config_path, False)

    def test_generated_phase_preserves_and_binds_parent_experiment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            annotations = dataset / "annotations"
            annotations.mkdir(parents=True)
            (annotations / "import_manifest.json").write_text("{}\n", encoding="utf-8")
            source = root / "source"
            (source / "mvs").mkdir(parents=True)
            (source / "mvs/scene.mvs").write_bytes(b"scene")
            (source / "mvs/Densify.ini").write_text("[Densify]\n", encoding="utf-8")
            production = root / "DensifyPointCloud"
            observer = root / "DensifyPointCloudDMapObserve"
            write_fake_densify_binary(production, observer=False)
            write_fake_densify_binary(observer, observer=True)
            snapshot = root / "snapshot_manifest.json"
            snapshot.write_text('{"snapshot_sha256":"one"}\n', encoding="utf-8")
            config_path = root / "config.yaml"
            base_config = {
                "schema_version": 2,
                "experiment_id": "phase-locked",
                "output_root": str(root / "outputs"),
                "dataset_root": str(dataset),
                "input_snapshot_manifest": str(snapshot),
                "densify_bin": str(production),
                "densify_observe_bin": str(observer),
                "hypothesis": "parent",
                "suite": {"name": "smoke", "scan_ids": ["scene"]},
                "scenes": [{
                    "scan_id": "scene",
                    "working_folder": str(source),
                    "mvs_file": str(source / "mvs/scene.mvs"),
                }],
                "instrumentation": {
                    "expected_width": 16,
                    "expected_height": 12,
                    "expected_frames_per_scene": 1,
                    "max_artifact_gb": 1,
                },
                "runs": [
                    {"label": "baseline", "role": "baseline", "repeats": 1},
                    {"label": "candidate", "role": "variant", "repeats": 1},
                ],
            }
            config_path.write_text(
                yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8"
            )
            _config, experiment_root = dmap_dev.prepare_experiment(config_path, False)
            root_lock_path = experiment_root / "00_experiment_lock.json"
            root_resolved_path = experiment_root / "00_resolved_experiment.yaml"
            root_lock_before = root_lock_path.read_bytes()
            root_resolved_before = root_resolved_path.read_bytes()

            selection_path = experiment_root / "validations/selection.json"
            selection_path.parent.mkdir(parents=True)
            selection_text = '{"selected_run":"candidate","valid":true}\n'
            selection_path.write_text(selection_text, encoding="utf-8")
            phase = {
                "schema_name": dmap_dev.EXPERIMENT_PHASE_SCHEMA_NAME,
                "schema_version": dmap_dev.EXPERIMENT_PHASE_SCHEMA_VERSION,
                "phase_id": "confirmation",
                "parent_experiment_id": "phase-locked",
                "parent": {
                    "source_config": dmap_dev.file_identity(config_path),
                    "experiment_lock": dmap_dev.file_identity(root_lock_path),
                    "resolved_experiment": dmap_dev.file_identity(root_resolved_path),
                    "suite": dmap_dev.file_identity(experiment_root / "00_suite.json"),
                },
                "selection_manifest": dmap_dev.file_identity(selection_path),
            }
            phase["lineage_sha256"] = dmap_dev.stable_json_digest(phase)
            phase_config = dict(base_config)
            phase_config["hypothesis"] = "generated confirmation"
            phase_config["manual_confirmation"] = {"resolution": {
                "selected_run": "candidate",
                "selection_manifest": str(selection_path),
                "selection_manifest_sha256": dmap_dev.file_identity(selection_path)["sha256"],
            }}
            phase_config["experiment_phase"] = phase
            phase_path = experiment_root / "configs/confirmation.yaml"
            phase_path.parent.mkdir(parents=True)
            phase_path.write_text(
                yaml.safe_dump(phase_config, sort_keys=False), encoding="utf-8"
            )

            dmap_dev.prepare_experiment(phase_path, False)
            dmap_dev.prepare_experiment(phase_path, False)
            evidence = dmap_dev.experiment_phase_evidence_dir(
                experiment_root, "confirmation"
            )
            self.assertTrue((evidence / "00_phase_lock.json").is_file())
            self.assertTrue((evidence / "01_resolved_phase.yaml").is_file())
            self.assertTrue((evidence / "02_phase_suite.json").is_file())
            self.assertTrue((evidence / "03_storage_estimate.json").is_file())
            self.assertEqual(root_lock_path.read_bytes(), root_lock_before)
            self.assertEqual(root_resolved_path.read_bytes(), root_resolved_before)
            policy = dmap_dev.build_report_policy(
                dmap_dev.load_config(phase_path),
                experiment_root,
                skip_diagnostics=False,
                capture_evidence={"reuse_eligible": True},
            )
            self.assertEqual(
                policy["resolved_experiment"],
                dmap_dev.file_identity(evidence / "01_resolved_phase.yaml"),
            )
            self.assertEqual(
                policy["experiment_phase"]["phase_lock"],
                dmap_dev.file_identity(evidence / "00_phase_lock.json"),
            )

            selection_path.write_text(
                '{"selected_run":"other","valid":true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "selection manifest changed"):
                dmap_dev.prepare_experiment(phase_path, False)
            selection_path.write_text(selection_text, encoding="utf-8")
            phase_config["hypothesis"] = "mutated phase"
            phase_path.write_text(
                yaml.safe_dump(phase_config, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "generated experiment phase identity"):
                dmap_dev.prepare_experiment(phase_path, False)

    def test_report_validation_checks_details_sections_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "asset.png").write_bytes(b"test")
            report = root / "report.md"
            report.write_text(
                "\n".join([
                    "## 1. Executive Summary",
                    "## 6. Annotation Consistency",
                    "### Fixed-model Inlier Threshold Sweep",
                    "### Fitted-model Stability vs Baseline",
                    "## 7. Algorithm Mechanics",
                    "### Instrumentation Coverage",
                    "Cost Function and Convergence",
                    "View Selection and Support",
                    "Update Dynamics",
                    "Filtering and Completeness",
                    "Runtime and Scalability",
                    "Geometric End Metrics",
                    "Cross-run Effects",
                    "Final State Overview",
                    "## 8. Per-scene Analysis",
                    "<details><summary>Scene</summary>",
                    "![asset](asset.png)",
                    "</details>",
                    "## 10. Reproducibility",
                    "## 11. Recommendations",
                ]),
                encoding="utf-8",
            )
            dmap_dev.write_json(root / "report_inventory.json", {
                "schema_version": 2,
                "required_mechanisms": [dmap_dev.MECHANISM_LABELS[name] for name in dmap_dev.MECHANISM_ORDER],
                "scenes": {},
                "overall_plots": [{"path": "asset.png"}],
            })

            result = dmap_dev.validate_report(report)

            self.assertTrue(result["valid"])
            self.assertEqual(result["details_open"], 1)
            self.assertEqual(result["references"], 1)

            dmap_dev.write_json(root / "report_model.json", {"schema_version": 1})
            legacy_result = dmap_dev.validate_report(report)
            self.assertTrue(legacy_result["valid"])
            self.assertFalse(legacy_result["investigation_guide_required"])

            dmap_dev.write_json(root / "report_model.json", {"schema_version": 2})
            missing_guide = dmap_dev.validate_report(report)
            self.assertFalse(missing_guide["valid"])
            self.assertTrue(missing_guide["investigation_guide_required"])
            self.assertIn(
                dmap_dev.dmap_report_model.INVESTIGATION_GUIDE_HEADING,
                missing_guide["missing_sections"],
            )

            report.write_text(
                dmap_dev.dmap_report_model.INVESTIGATION_GUIDE_HEADING + "\n" + report.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            self.assertTrue(dmap_dev.validate_report(report)["valid"])
            report.write_text(
                report.read_text(encoding="utf-8")
                + "\n[external raw capture](../capture/repro.json)\n",
                encoding="utf-8",
            )
            unsafe = dmap_dev.validate_report(report)
            self.assertFalse(unsafe["valid"])
            self.assertEqual(
                unsafe["unsafe_external_or_traversal_references"],
                ["../capture/repro.json"],
            )

    def test_staged_report_validation_resolves_relative_inventory_in_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "01_master_staging"
            published = root / "01_master"
            staging.mkdir()
            published.mkdir()
            (staging / "asset.png").write_bytes(b"new")
            (published / "asset.png").write_bytes(b"old")
            report = staging / "report.md"
            report.write_text(
                "\n".join([
                    dmap_dev.dmap_report_model.INVESTIGATION_GUIDE_HEADING,
                    "## 1. Executive Summary",
                    "## 6. Annotation Consistency",
                    "### Fixed-model Inlier Threshold Sweep",
                    "### Fitted-model Stability vs Baseline",
                    "## 7. Algorithm Mechanics",
                    "### Instrumentation Coverage",
                    "Cost Function and Convergence",
                    "View Selection and Support",
                    "Update Dynamics",
                    "Filtering and Completeness",
                    "Runtime and Scalability",
                    "Geometric End Metrics",
                    "Cross-run Effects",
                    "Final State Overview",
                    "## 8. Per-scene Analysis",
                    "## 10. Reproducibility",
                    "## 11. Recommendations",
                ]),
                encoding="utf-8",
            )
            dmap_dev.write_json(staging / "report_model.json", {"schema_version": 2})
            dmap_dev.write_json(staging / "report_policy.json", {
                "published_output_dir": str(published),
            })
            dmap_dev.write_json(staging / "report_inventory.json", {
                "schema_version": 2,
                "required_mechanisms": [
                    dmap_dev.MECHANISM_LABELS[name]
                    for name in dmap_dev.MECHANISM_ORDER
                ],
                "scenes": {},
                "overall_plots": [{"path": "asset.png"}],
            })

            result = dmap_dev.validate_report(report)

            self.assertTrue(result["valid"])
            self.assertEqual(result["report"], str(published / "report.md"))
            self.assertNotIn(
                str(staging),
                (staging / "report.validation.json").read_text(encoding="utf-8"),
            )

            (staging / "asset.png").unlink()
            result = dmap_dev.validate_report(report)
            self.assertFalse(result["valid"])
            self.assertIn(str(staging / "asset.png"), result["inventory_missing_paths"])

    def test_receipt_validation_precedes_sidecar_mutation_and_cli_mode_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.md"
            report.write_text("# incomplete fixture\n", encoding="utf-8")
            dmap_dev.write_json(root / "report_policy.json", {})
            (root / "pre_publish_finalizer_receipt.json").write_text(
                '{"fixture":true}\n', encoding="utf-8",
            )
            sidecar = root / "report.validation.json"
            sidecar.write_bytes(b"immutable-sidecar\x00")
            with mock.patch(
                "generate_attested_dmap_report.validate_finalizer_receipt",
                side_effect=RuntimeError("receipt semantic drift"),
            ):
                with self.assertRaisesRegex(RuntimeError, "receipt semantic drift"):
                    dmap_dev.validate_report(report, write_sidecar=True)
            self.assertEqual(sidecar.read_bytes(), b"immutable-sidecar\x00")

            with mock.patch(
                "generate_attested_dmap_report.validate_finalizer_receipt",
                return_value={"present": True, "valid": True},
            ):
                result = dmap_dev.validate_report(report, write_sidecar=False)
            self.assertTrue(result["trusted_finalizer_receipt_present"])
            self.assertTrue(result["trusted_finalizer_receipt_valid"])
            self.assertEqual(sidecar.read_bytes(), b"immutable-sidecar\x00")

    def test_executive_summary_states_zero_strict_passes_and_model_switch_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text("schema_version: 2\n", encoding="utf-8")
            model_stability_csv = root / "model_stability.csv"
            model_stability_csv.write_text("candidate,large_model_switch\nvariant,true\n", encoding="utf-8")
            ledger = pd.DataFrame([{
                "accuracy_rank": 1,
                "candidate": "variant",
                "noise_class": "regressed",
                "availability_biased": False,
                "lost_baseline_fit_count": 0,
                "scene_count": 2,
                "worst_normalized_noise_loss": 1.5,
                "effective_coverage_delta": 0.0,
                "spatial_coverage_delta": 0.0,
                "valid_coverage_delta": 0.0,
                "endpoint_valid_depth_coverage_delta": -0.1,
                "coverage_advisory": True,
                "runtime_relative_delta": 0.0,
                "strict_accuracy_gate_available": True,
                "strict_accuracy_pass": False,
            }])
            stability = pd.DataFrame([{
                "candidate": "variant", "scene_id": "scene",
                "annotation_kind": "edge", "chunk_id": "chunk",
                "large_model_switch": True, "near_candidate_regression": True,
                "line_direction_delta_deg": 20.0,
                "line_position_delta_m": 0.01,
                "line_extent_delta_m": 0.4,
                "model_switch_reasons_json": '["line_direction"]',
                "baseline_model_on_candidate_status": "available",
                "baseline_model_on_candidate_all_residual_p95_m": 0.2,
            }])

            markdown = dmap_dev.build_markdown(
                config={"_config_path": str(config_path)},
                report_path=root / "report.md",
                frames=pd.DataFrame(),
                passes=pd.DataFrame(),
                annotations=pd.DataFrame(),
                stability=stability,
                performance=pd.DataFrame(),
                comparisons=pd.DataFrame(),
                gates=[],
                pareto=[],
                findings=[],
                plots=[],
                instrumentation_plots={},
                diagnostic_panels={},
                outputs={
                    "model_stability": {
                        "csv": str(model_stability_csv), "rows": 1, "parquet": None,
                    }
                },
                exact_cost_evolution=pd.DataFrame(),
                exact_iterations=pd.DataFrame(),
                exact_views=pd.DataFrame(),
                cpu_view_candidates=pd.DataFrame(),
                cpu_estimation_selection=pd.DataFrame(),
                postprocess_filters=pd.DataFrame(),
                confidence_adjustment=pd.DataFrame(),
                cuda_resource_plans=pd.DataFrame(),
                filter_resource_plans=pd.DataFrame(),
                resource_plan_validation=pd.DataFrame(),
                reproducibility_artifacts=pd.DataFrame(),
                accuracy_ledger=ledger,
                accuracy_evidence=pd.DataFrame(),
                evidence_context_path=root / "evidence_context.json",
                published_report_dir=root / "canonical_report",
            )

        self.assertIn("Strict accuracy gate: 0 of 1 candidates pass", markdown)
        self.assertIn("Competing RANSAC models detected", markdown)
        self.assertIn("final endpoint validity delta", markdown)
        self.assertIn(f"--output-dir {root / 'canonical_report_reproduced'}", markdown)
        self.assertIn(f"--report-dir {root / 'canonical_report'}", markdown)
        self.assertGreaterEqual(
            markdown.count(f"--evidence-context {root / 'evidence_context.json'}"),
            2,
        )

    def test_external_raw_artifacts_are_plain_text_not_report_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "reports" / "01_report.md"
            artifact = root / "runs" / "base" / "summary.json"

            reference = dmap_dev.markdown_evidence_reference(
                artifact, "summary.json", report
            )

            self.assertIn("external evidence", reference)
            self.assertNotIn("](../", reference)

    def test_runtime_boundary_rejects_loader_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "DensifyPointCloud"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)

            for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "LD_AUDIT"):
                with self.subTest(variable=variable), mock.patch.dict(
                    "os.environ", {variable: "/tmp/alternate-openmvs"}
                ), self.assertRaisesRegex(RuntimeError, "must be empty"):
                    dmap_dev.runtime_boundary_identity(executable)

    def test_capture_runtime_boundary_receipt_binds_pre_and_post_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            binary_dir = Path(directory) / "bin"
            binary_dir.mkdir()
            executable = binary_dir / "DensifyPointCloud"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            (binary_dir / "libMVS.so").write_bytes(b"locked-library")
            boundary = dmap_dev.runtime_boundary_identity(executable)
            dmap_dev.write_json(root / "00_experiment_lock.json", {
                "schema_name": dmap_dev.EXPERIMENT_LOCK_SCHEMA_NAME,
                "schema_version": dmap_dev.EXPERIMENT_LOCK_SCHEMA_VERSION,
                "runtime_boundaries": {
                    "production": boundary,
                    "observer": boundary,
                },
            })
            run_dir = root / "runs" / "base" / "endpoint"

            result = dmap_dev.execute_densify_command(
                [str(executable)],
                root,
                run_dir,
                False,
                experiment_root=root,
                runtime_role="production",
            )

            self.assertTrue(result["runtime_boundary_receipt"]["valid"])
            valid, reason = dmap_dev.validate_runtime_boundary_receipt(
                run_dir, dmap_dev.read_json(run_dir / "repro.json"), "endpoint"
            )
            self.assertTrue(valid, reason)

            original_execute = dmap_dev.execute_command

            def execute_then_relink(*args, **kwargs):
                value = original_execute(*args, **kwargs)
                (binary_dir / "libMVS.so").write_bytes(b"changed-library")
                return value

            changed_dir = root / "runs" / "changed" / "endpoint"
            with mock.patch.object(
                dmap_dev, "execute_command", side_effect=execute_then_relink
            ), self.assertRaisesRegex(RuntimeError, "changed while the subprocess"):
                dmap_dev.execute_densify_command(
                    [str(executable)],
                    root,
                    changed_dir,
                    False,
                    experiment_root=root,
                    runtime_role="production",
                )
            changed_repro = dmap_dev.read_json(changed_dir / "repro.json")
            self.assertFalse(changed_repro["runtime_boundary_receipt"]["valid"])

    def test_cli_only_capture_intent_survives_absent_capture_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("name: test\n", encoding="utf-8")
            config = {
                "_config_path": str(config_path),
                "experiment_id": "experiment",
                "capture_profiles": ["summary"],
                "suite": {"scan_ids": ["scene-a"]},
                "runs": [{"label": "base", "role": "baseline", "repeats": 1}],
            }
            dmap_dev.write_json(root / "00_experiment_lock.json", {
                "schema_name": dmap_dev.EXPERIMENT_LOCK_SCHEMA_NAME,
                "schema_version": dmap_dev.EXPERIMENT_LOCK_SCHEMA_VERSION,
            })
            intent = dmap_dev.build_capture_intent(
                config,
                root,
                [{"scan_id": "scene-a"}],
                ["deep"],
                activation_source="cli_override",
            )
            dmap_dev.persist_capture_intent(root, intent)

            coverage = dmap_dev.build_capture_profile_coverage(config, root)

            deep = next(
                row for row in coverage["profiles"] if row["profile"] == "deep"
            )
            self.assertTrue(deep["requested"])
            self.assertEqual(deep["status"], "unavailable")
            self.assertEqual(coverage["capture_intents"][0]["activation_source"], "cli_override")

    def test_existing_run_mode_skips_only_validated_complete_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "maps"
            self.assertEqual(dmap_dev.existing_run_mode_action(run_dir, "maps"), "run")
            run_dir.mkdir()
            (run_dir / "partial").touch()
            with mock.patch.object(dmap_dev, "validate_completed_run_mode", return_value=(True, "ok")):
                self.assertEqual(dmap_dev.existing_run_mode_action(run_dir, "maps"), "skip")
            with mock.patch.object(dmap_dev, "validate_completed_run_mode", return_value=(False, "bad")):
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    dmap_dev.existing_run_mode_action(run_dir, "maps")

    def test_existing_report_skips_only_when_both_validators_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            expected_policy = {
                "schema_name": dmap_dev.REPORT_POLICY_SCHEMA_NAME,
                "schema_version": dmap_dev.REPORT_POLICY_SCHEMA_VERSION,
                "allow_process_specialization_divergence_for_diagnostics": True,
                "capture_evidence": {"reuse_eligible": True},
            }
            for name in (
                "01_development_report.md", "01_development_report.html", "02_investigation.html",
                "report_model.json", "report_manifest.json", "report_inventory.json",
            ):
                (output / name).write_text("{}", encoding="utf-8")
            dmap_dev.write_json(output / "report_policy.json", expected_policy)
            with (
                mock.patch.object(dmap_dev, "validate_report", return_value={"valid": True}),
                mock.patch.object(dmap_dev.dmap_report_model, "validate_report_model", return_value={"valid": True}),
            ):
                self.assertEqual(
                    dmap_dev.existing_report_action(output, expected_policy), "skip"
                )
            with (
                mock.patch.object(dmap_dev, "validate_report", return_value={"valid": False}),
                mock.patch.object(dmap_dev.dmap_report_model, "validate_report_model", return_value={"valid": True}),
            ):
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    dmap_dev.existing_report_action(output, expected_policy)
            strict_policy = {
                **expected_policy,
                "allow_process_specialization_divergence_for_diagnostics": False,
            }
            with self.assertRaisesRegex(RuntimeError, "report policy mismatch"):
                dmap_dev.existing_report_action(output, strict_policy)

    def test_new_capture_evidence_cannot_reuse_a_stale_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closure = root / "runs/baseline/repeat_00/scene/maps/capture_artifact_closure.json"
            closure.parent.mkdir(parents=True)
            closure.write_text("{}\n", encoding="utf-8")
            unavailable = {
                "schema_name": "openmvs.dmap.capture_profile_coverage",
                "schema_version": 1,
                "profiles": [],
                "units": [{
                    "configured_run": "baseline",
                    "repeat": 0,
                    "scene_id": "scene",
                    "capture_profile": "deep",
                    "status": "unavailable",
                    "evidence_links": [],
                }],
            }
            complete = json.loads(json.dumps(unavailable))
            complete["units"][0].update({
                "status": "complete",
                "evidence_links": [{
                    "label": "artifact closure",
                    "path": str(closure),
                }],
            })
            old_evidence = dmap_dev.build_capture_evidence_policy(root, unavailable)
            new_evidence = dmap_dev.build_capture_evidence_policy(root, complete)
            self.assertNotEqual(old_evidence, new_evidence)
            self.assertTrue(new_evidence["reuse_eligible"])
            self.assertNotIn(str(root), json.dumps(new_evidence))

            output = root / "reports"
            output.mkdir()
            old_policy = {
                "schema_name": dmap_dev.REPORT_POLICY_SCHEMA_NAME,
                "schema_version": dmap_dev.REPORT_POLICY_SCHEMA_VERSION,
                "capture_evidence": old_evidence,
            }
            new_policy = {**old_policy, "capture_evidence": new_evidence}
            for name in (
                "01_development_report.md", "01_development_report.html",
                "02_investigation.html", "report_model.json",
                "report_manifest.json", "report_inventory.json",
            ):
                (output / name).write_text("{}", encoding="utf-8")
            dmap_dev.write_json(output / "report_policy.json", old_policy)

            with self.assertRaisesRegex(RuntimeError, "report policy mismatch"):
                dmap_dev.existing_report_action(output, new_policy)

    def test_unclosed_complete_capture_disables_report_reuse(self) -> None:
        coverage = {
            "schema_name": "openmvs.dmap.capture_profile_coverage",
            "schema_version": 1,
            "profiles": [],
            "units": [{
                "configured_run": "legacy",
                "repeat": 0,
                "scene_id": "scene",
                "capture_profile": "deep",
                "status": "complete",
                "evidence_links": [],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            evidence = dmap_dev.build_capture_evidence_policy(
                Path(directory), coverage
            )
        self.assertFalse(evidence["reuse_eligible"])
        self.assertEqual(
            evidence["complete_units_without_verified_closure"],
            ["legacy/0/scene/deep"],
        )

    def test_exact_cost_trajectory_plot_marks_per_run_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for run, iterations in (("base", (-1, 0)), ("variant", (-1, 0, 1, 2))):
                for logical_iteration in iterations:
                    rows.append({
                        "run": run, "scene_id": "scene", "image_id": 7,
                        "estimation_stage": "photometric", "geometric_iteration": None,
                        "signal": "cost_total_production_exact", "logical_iteration": logical_iteration,
                        "median": 0.8 - 0.1 * (logical_iteration + 1),
                        "p90": 0.9 - 0.08 * (logical_iteration + 1),
                    })

            plots = dmap_dev.exact_cost_trajectory_plots(pd.DataFrame(rows), Path(directory))

            self.assertEqual(len(plots), 1)
            self.assertIn("Exact cost-component trajectories", plots[0][0])
            self.assertTrue(plots[0][1].is_file())


if __name__ == "__main__":
    unittest.main()
