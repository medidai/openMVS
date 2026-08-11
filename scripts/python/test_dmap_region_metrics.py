from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from scripts.python.dmap_observability import region_metrics


class RegionMetricsTest(unittest.TestCase):
    def test_identity_preserves_zero_image_and_geometric_iteration(self) -> None:
        identity = region_metrics._identity({
            "run": "candidate",
            "repeat": 0,
            "scene_id": "scene-a",
            "image_id": 0,
            "geometric_iteration": 0,
        })
        self.assertEqual(identity[3], 0)
        self.assertEqual(identity[5], 0)

    def test_texture_bins_partition_finite_pixels(self) -> None:
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        values[0, 0] = np.nan

        bins, thresholds = region_metrics.texture_bins(values)

        self.assertEqual(int(np.count_nonzero(bins >= 0)), 11)
        self.assertEqual(set(int(value) for value in np.unique(bins[bins >= 0])), {0, 1, 2})
        self.assertLessEqual(thresholds["low_mid"], thresholds["mid_high"])

    def test_registered_metric_is_summarized_by_texture_region(self) -> None:
        arrays = {
            "texture": np.arange(9, dtype=np.float32).reshape(3, 3),
            "cost": np.asarray([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.float32),
        }
        rows = [
            {
                "run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "signal": "texture_score", "path": "texture", "available": True,
                "logical_iteration": 0,
            },
            {
                "run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 1,
                "signal": "custom_texture_cost", "component_id": "texture_penalty",
                "mechanism": "texture", "quantity": "contribution", "path": "cost",
                "available": True, "logical_iteration": 0,
            },
        ]

        result = region_metrics.compute_texture_stratification(
            rows,
            lambda path: arrays[str(path)],
        )

        self.assertEqual(result["failures"], [])
        cost_rows = [row for row in result["rows"] if row["signal"] == "custom_texture_cost"]
        self.assertEqual({row["region"] for row in cost_rows}, {"low", "mid", "high"})
        self.assertEqual(sum(row["region_pixels"] for row in cost_rows), 9)
        self.assertGreater(cost_rows[0]["mean"], cost_rows[-1]["mean"])

    def test_unavailable_sentinel_is_excluded_from_region_metric(self) -> None:
        arrays = {
            "texture": np.arange(9, dtype=np.float32).reshape(3, 3),
            "gain": np.asarray([[-1, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.float32),
        }
        result = region_metrics.compute_texture_stratification(
            [
                {
                    "run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 1,
                    "signal": "texture_score", "path": "texture", "available": True,
                    "logical_iteration": 0,
                },
                {
                    "run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 1,
                    "signal": "low_texture_update_best_proposed_gain_exact", "path": "gain",
                    "available": True, "logical_iteration": 0, "unavailable_value": -1,
                },
            ],
            lambda path: arrays[str(path)],
        )

        gain_rows = [
            row for row in result["rows"]
            if row["signal"] == "low_texture_update_best_proposed_gain_exact"
        ]
        self.assertEqual(sum(row["valid_pixels"] for row in gain_rows), 8)

    def test_low_texture_accepted_gain_census_is_deterministic(self) -> None:
        incumbent = np.ones((2, 3), dtype=np.float64)
        winner = incumbent - np.asarray(
            [[0.0001, 0.0003, 0.0007], [0.002, 0.25, 0.0]], dtype=np.float64
        )
        variance = np.asarray(
            [[0.1, 0.1, 0.1], [0.1, 0.5, 0.5]], dtype=np.float64
        )
        arrays = {"incumbent": incumbent, "winner": winner, "variance": variance}
        common = {
            "run": "variant", "repeat": 0, "scene_id": "scene", "image_id": 1,
            "available": True, "logical_iteration": 2,
        }
        rows = [
            {**common, "signal": "candidate_incumbent_cost_exact", "path": "incumbent"},
            {**common, "signal": "candidate_winner_cost_exact", "path": "winner"},
            {**common, "signal": "reference_variance_production_exact", "path": "variance"},
        ]

        result = region_metrics.compute_low_texture_accepted_gain_census(
            rows,
            lambda path: arrays[str(path)],
            variance_max_by_run={"variant": 0.2},
        )

        self.assertEqual(result["failures"], [])
        self.assertEqual(len(result["rows"]), 1)
        census = result["rows"][0]
        self.assertEqual(census["low_texture_pixels"], 4)
        self.assertEqual(census["accepted_gain_pixels"], 4)
        self.assertAlmostEqual(census["gain_quantiles"]["p50"], 0.0005)
        self.assertAlmostEqual(census["fractions_below"]["0.00025"], 0.25)
        self.assertAlmostEqual(census["fractions_below"]["0.0005"], 0.5)
        self.assertAlmostEqual(census["fractions_below"]["0.001"], 0.75)


if __name__ == "__main__":
    unittest.main()
