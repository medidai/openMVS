#!/usr/bin/env python3
"""Focused tests for exact DVP visibility observability schema v1."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_dmap_instrumentation as validator


WIDTH = 2
HEIGHT = 2
NUM_VIEWS = 3
MODE = "depth_gated_restore_v1"


def rgba_masks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    return np.stack(
        [
            values & 255,
            (values >> 8) & 255,
            (values >> 16) & 255,
            (values >> 24) & 255,
        ],
        axis=-1,
    ).astype(np.uint8)


def popcount(values: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda value: int(value).bit_count())(
        values.astype(np.uint32)
    ).astype(np.uint8)


def build_fixture(directory: str) -> tuple[
    Path,
    dict,
    dict,
    list[dict],
    dict[tuple[str, int], tuple[dict, Path, np.ndarray]],
]:
    frame_dir = Path(directory) / "stage" / "depthmaps" / "0001_0000"
    instrumentation_dir = frame_dir.parent.parent / "instrumentation"
    instrumentation_dir.mkdir(parents=True)

    previous = np.array([[1, 3], [0, 5]], dtype=np.uint32)
    restored = np.array([[2, 0], [1, 2]], dtype=np.uint32)
    resolved = previous | restored
    rejected = np.uint32(7) & ~resolved
    next_mask = np.array([[3, 1], [5, 7]], dtype=np.uint32)
    added = next_mask & ~previous
    removed = previous & ~next_mask
    candidate_tested = np.array(
        [[1 << 22, (1 << 22) | (1 << 23)], [0, 1 << 23]], dtype=np.uint32
    )
    candidate_finite = np.array(
        [[1 << 22, 1 << 23], [0, 1 << 23]], dtype=np.uint32
    )

    masks = {
        "previous": previous,
        "resolved": resolved,
        "next": next_mask,
        "active_support": resolved.copy(),
        "restored": restored,
        "rejected": rejected,
        "added": added,
        "removed": removed,
        "candidate_tested": candidate_tested,
        "candidate_finite": candidate_finite,
    }
    counts = {
        "previous_visible": popcount(previous),
        "resolved_visible": popcount(resolved),
        "next_visible": popcount(next_mask),
        "active_support": popcount(resolved),
        "restored": popcount(restored),
        "rejected": popcount(rejected),
        "added": popcount(added),
        "removed": popcount(removed),
    }
    reason_maps = {
        reason: np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for reason in validator.DVP_VISIBILITY_REASON_NAMES
    }
    reason_maps["retained_previous_weight"] = counts["previous_visible"].copy()
    reason_maps["invalid_observed_source_depth"] = counts["rejected"].copy()
    reason_maps["restored_depth_gated"] = counts["restored"].copy()

    arrays: dict[str, np.ndarray] = {}
    arrays.update({
        f"dvp_visibility_{name}_mask": rgba_masks(value)
        for name, value in masks.items()
    })
    arrays.update({
        f"dvp_visibility_{name}_count": value
        for name, value in counts.items()
    })
    arrays.update({
        "dvp_visibility_previous_weight_sum": counts[
            "previous_visible"
        ].astype(np.uint16),
        "dvp_visibility_resolved_weight_sum": counts[
            "resolved_visible"
        ].astype(np.uint16),
        "dvp_visibility_next_weight_sum": counts[
            "next_visible"
        ].astype(np.uint16),
        "dvp_visibility_active_support_weight_sum": counts[
            "active_support"
        ].astype(np.uint16),
        "dvp_visibility_denominator": counts[
            "active_support"
        ].astype(np.uint16),
        "dvp_visibility_mode": np.full((HEIGHT, WIDTH), 2, dtype=np.uint8),
        "dvp_visibility_denominator_defined": (
            counts["active_support"] > 0
        ).astype(np.uint8),
        "dvp_visibility_support_matches_resolved": np.ones(
            (HEIGHT, WIDTH), dtype=np.uint8
        ),
        "dvp_visibility_transition_status": np.zeros(
            (HEIGHT, WIDTH), dtype=np.uint8
        ),
    })
    arrays.update({
        f"dvp_visibility_reason_{reason}": value
        for reason, value in reason_maps.items()
    })

    entries: list[dict] = []
    logical_maps: dict[tuple[str, int], tuple[dict, Path, np.ndarray]] = {}
    for signal in sorted(validator.DVP_VISIBILITY_REQUIRED_SIGNALS):
        dtype = (
            "uint8x4"
            if signal in validator.DVP_VISIBILITY_REQUIRED_RGBA_SIGNALS
            else "uint16"
            if signal in validator.DVP_VISIBILITY_REQUIRED_WORD_SIGNALS
            else "uint8"
        )
        entry = {
            "signal": signal,
            "path": f"dvp_visibility_states/iteration01/{signal}.png",
            "dtype": dtype,
            "logical_iteration": 0,
            "role": "dvp_visibility_logical_iteration_state",
            "stage": "iteration",
            "dvp_visibility_schema_name": validator.DVP_VISIBILITY_SCHEMA_NAME,
            "dvp_visibility_schema_version": (
                validator.DVP_VISIBILITY_SCHEMA_VERSION
            ),
            "measurement_quality": "exact",
            "measurement_basis": (
                "same_stream_logical_iteration_visibility_transition"
            ),
        }
        entries.append(entry)
        logical_maps[(signal, 0)] = (entry, Path(entry["path"]), arrays[signal])

    exact_counts = {
        "pixels": WIDTH * HEIGHT,
        "previous_visible_views": int(np.sum(counts["previous_visible"])),
        "resolved_visible_views": int(np.sum(counts["resolved_visible"])),
        "next_visible_views": int(np.sum(counts["next_visible"])),
        "active_support_views": int(np.sum(counts["active_support"])),
        "restored_views": int(np.sum(counts["restored"])),
        "rejected_views": int(np.sum(counts["rejected"])),
        "added_views": int(np.sum(counts["added"])),
        "removed_views": int(np.sum(counts["removed"])),
        "previous_weight_sum": int(np.sum(counts["previous_visible"])),
        "resolved_weight_sum": int(np.sum(counts["resolved_visible"])),
        "next_weight_sum": int(np.sum(counts["next_visible"])),
        "active_support_weight_sum": int(np.sum(counts["active_support"])),
        "denominator_sum": int(np.sum(counts["active_support"])),
        "zero_denominator_pixels": int(np.count_nonzero(
            counts["active_support"] == 0
        )),
        "support_mismatch_pixels": 0,
        "changed_pixels": int(np.count_nonzero((added | removed) != 0)),
        "invalid_transition_pixels": 0,
        "candidate_tested_pixels": int(np.count_nonzero(candidate_tested)),
        "candidate_finite_pixels": int(np.count_nonzero(candidate_finite)),
        "candidate_tested_count": int(np.sum(popcount(candidate_tested))),
        "candidate_finite_count": int(np.sum(popcount(candidate_finite))),
    }
    reason_counts = {
        reason: int(np.sum(values)) for reason, values in reason_maps.items()
    }
    summary_iteration = {
        "logical_iteration": 0,
        "pixels": exact_counts["pixels"],
        "mode": MODE,
        "views": {
            "previous_visible": exact_counts["previous_visible_views"],
            "resolved_visible": exact_counts["resolved_visible_views"],
            "next_visible": exact_counts["next_visible_views"],
            "active_support": exact_counts["active_support_views"],
            "restored": exact_counts["restored_views"],
            "rejected": exact_counts["rejected_views"],
            "added": exact_counts["added_views"],
            "removed": exact_counts["removed_views"],
        },
        "weights": {
            "previous_sum": exact_counts["previous_weight_sum"],
            "resolved_sum": exact_counts["resolved_weight_sum"],
            "next_sum": exact_counts["next_weight_sum"],
            "active_support_sum": exact_counts["active_support_weight_sum"],
            "denominator_sum": exact_counts["denominator_sum"],
            "zero_denominator_pixels": exact_counts["zero_denominator_pixels"],
        },
        "transitions": {
            "support_mismatch_pixels": 0,
            "changed_pixels": exact_counts["changed_pixels"],
            "invalid_transition_pixels": 0,
        },
        "candidates": {
            "tested_pixels": exact_counts["candidate_tested_pixels"],
            "finite_pixels": exact_counts["candidate_finite_pixels"],
            "tested_count": exact_counts["candidate_tested_count"],
            "finite_count": exact_counts["candidate_finite_count"],
        },
        "reason_counts": reason_counts,
    }
    capture = {
        "schema_name": validator.DVP_VISIBILITY_SCHEMA_NAME,
        "schema_version": validator.DVP_VISIBILITY_SCHEMA_VERSION,
        "mode": MODE,
        "requested": True,
        "stage_active": True,
        "summary_available": True,
        "maps_available": True,
        "targeted_trace_available": True,
        "num_iterations": 1,
        "num_views": NUM_VIEWS,
        "counter_record_bytes": 160,
        "update_record_bytes": 76,
        "trace_record_bytes": 260,
        "aggregate_weight_sum_precision": "uint64 device atomic",
        "immutable_candidate_support": True,
        "per_view_weights_full_frame": False,
        "per_view_weights_targeted_trace": True,
        "per_view_reason_full_frame": "per-pixel reason counts and lossless masks",
        "per_view_reason_targeted_trace": True,
    }
    observability = {
        "schema_name": validator.DVP_VISIBILITY_OBSERVABILITY_SCHEMA_NAME,
        "schema_version": validator.DVP_VISIBILITY_SCHEMA_VERSION,
        "enabled": True,
        "mode": MODE,
        "requested": True,
        "stage_active": True,
        "summary_available": True,
        "maps_available": True,
        "targeted_trace_available": True,
        "num_views": NUM_VIEWS,
        "counter_record_bytes": 160,
        "update_record_bytes": 76,
        "trace_record_bytes": 260,
        "aggregate_weight_sum_precision": "uint64 device atomic",
        "measurement_quality": "exact",
        "measurement_basis": (
            "same-stream Process<true> logical-iteration visibility transition "
            "and active candidate support"
        ),
        "reason_enum": {
            str(code): name
            for code, name in enumerate(validator.DVP_VISIBILITY_REASON_NAMES)
        },
        "iterations": [summary_iteration],
    }
    manifest = {"dvp_visibility_capture": capture}
    summary = {
        "image_id": 1,
        "scale_level": 0,
        "dvp_visibility_observability": observability,
    }

    csv_fields = [
        "image_id", "scale_number", "logical_iteration", "mode",
        *exact_counts.keys(),
        *(f"reason_{reason}" for reason in validator.DVP_VISIBILITY_REASON_NAMES),
    ]
    with (instrumentation_dir / "dvp_visibility_iteration.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow({
            "image_id": 1,
            "scale_number": 0,
            "logical_iteration": 0,
            "mode": MODE,
            **exact_counts,
            **{f"reason_{name}": value for name, value in reason_counts.items()},
        })

    trace_x = 0
    trace_y = 0
    trace_previous = [1, 0, 0]
    trace_resolved = [1, 1, 0]
    trace_next = [1, 1, 0]
    trace_active = trace_resolved.copy()
    trace_reason_codes = [1, 11, 5]
    trace_views = []
    for view in range(NUM_VIEWS):
        bit = 1 << view
        trace_views.append({
            "view_index": view,
            "previous_weight": trace_previous[view],
            "resolved_weight": trace_resolved[view],
            "next_weight": trace_next[view],
            "active_support_weight": trace_active[view],
            "normalized_active_support_weight": trace_active[view] / 2.0,
            "reason_code": trace_reason_codes[view],
            "reason": validator.DVP_VISIBILITY_REASON_NAMES[
                trace_reason_codes[view]
            ],
            "restored": bool(int(restored[trace_y, trace_x]) & bit),
            "rejected": bool(int(rejected[trace_y, trace_x]) & bit),
            "added_next": bool(int(added[trace_y, trace_x]) & bit),
            "removed_next": bool(int(removed[trace_y, trace_x]) & bit),
        })
    trace = {
        "schema_name": validator.DVP_VISIBILITY_TRACE_SCHEMA_NAME,
        "schema_version": validator.DVP_VISIBILITY_SCHEMA_VERSION,
        "measurement_quality": "exact",
        "measurement_basis": "same_stream_logical_iteration_visibility_transition",
        "image_id": 1,
        "scale_number": 0,
        "logical_iteration": 0,
        "trace_index": 0,
        "label": "fixture",
        "x": trace_x,
        "y": trace_y,
        "mode": MODE,
        "state": {
            "previous_mask": 1,
            "resolved_mask": 3,
            "next_mask": 3,
            "active_support_mask": 3,
            "restored_mask": 2,
            "rejected_mask": 4,
            "added_mask": 2,
            "removed_mask": 0,
            "transition_status_code": 0,
            "transition_status": "valid",
            "active_support_matches_resolved": True,
        },
        "weights": {
            "previous_sum": 1,
            "resolved_sum": 2,
            "next_sum": 2,
            "active_support_sum": 2,
            "denominator": 2,
            "denominator_defined": True,
        },
        "candidates": {
            "tested_mask": int(candidate_tested[trace_y, trace_x]),
            "finite_mask": int(candidate_finite[trace_y, trace_x]),
        },
        "reason_counts": {
            reason: trace_reason_codes.count(code)
            for code, reason in enumerate(validator.DVP_VISIBILITY_REASON_NAMES)
        },
        "views": trace_views,
    }
    (instrumentation_dir / "dvp_visibility_traces.jsonl").write_text(
        json.dumps(trace) + "\n", encoding="utf-8"
    )
    return frame_dir, manifest, summary, entries, logical_maps


class DVPVisibilityValidatorTests(unittest.TestCase):
    def validate_fixture(
        self,
        frame_dir: Path,
        manifest: dict,
        summary: dict,
        entries: list[dict],
        logical_maps: dict[tuple[str, int], tuple[dict, Path, np.ndarray]],
    ) -> tuple[dict, list[dict]]:
        checks: list[dict] = []

        def check(name: str, passed: bool, detail: object) -> None:
            checks.append({"name": name, "passed": passed, "detail": detail})

        result = validator.validate_dvp_visibility_maps(
            frame_dir=frame_dir,
            manifest=manifest,
            summary=summary,
            entries=entries,
            logical_maps=logical_maps,
            width=WIDTH,
            height=HEIGHT,
            check=check,
        )
        return result, checks

    def test_maps_counters_csv_and_trace_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            result, checks = self.validate_fixture(*fixture)
            self.assertTrue(result["available"])
            self.assertEqual(result["expected_map_count"], 39)
            self.assertEqual(result["trace_rows"], 1)
            self.assertTrue(all(row["passed"] for row in checks), checks)

    def test_candidate_mask_uses_candidate_not_view_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = build_fixture(directory)
            result, checks = self.validate_fixture(*fixture)
            self.assertEqual(
                result["iterations"]["0"]["counts"]["candidate_tested_count"],
                4,
            )
            domains = next(
                row for row in checks
                if row["name"] == "dvp_visibility_iteration_0_domains"
            )
            self.assertTrue(domains["passed"], domains)

    def test_reason_coverage_failure_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame_dir, manifest, summary, entries, logical_maps = build_fixture(
                directory
            )
            logical_maps[("dvp_visibility_reason_invalid_observed_source_depth", 0)][
                2
            ][0, 0] = 0
            result, checks = self.validate_fixture(
                frame_dir, manifest, summary, entries, logical_maps
            )
            self.assertEqual(
                result["iterations"]["0"]["domain_errors"]["reason_coverage"],
                1,
            )
            domains = next(
                row for row in checks
                if row["name"] == "dvp_visibility_iteration_0_domains"
            )
            self.assertFalse(domains["passed"])


if __name__ == "__main__":
    unittest.main()
