#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("generate_dvp_depth_edge_prior.py")
SPEC = importlib.util.spec_from_file_location("generate_dvp_depth_edge_prior", MODULE_PATH)
assert SPEC and SPEC.loader
prior = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prior
# OpenCV's import can change loader settings for child processes. Keep these
# unit tests from changing the environment seen by unrelated runtime tests.
with mock.patch.dict(os.environ):
    SPEC.loader.exec_module(prior)


class DVPDepthEdgePriorTest(unittest.TestCase):
    def test_openmvs_processing_geometry_matches_densify_preparation(self) -> None:
        geometry = prior.compute_openmvs_processing_geometry(
            width=3019,
            height=4026,
            resolution_level=0,
            min_resolution=640,
            max_resolution=2560,
        )
        self.assertEqual((geometry.width, geometry.height), (1920, 2560))
        self.assertEqual(geometry.effective_resolution_level, 0)
        self.assertEqual(geometry.prepared_max_resolution, 2560)

        half = prior.compute_openmvs_processing_geometry(
            width=3019,
            height=4026,
            resolution_level=1,
            min_resolution=640,
            max_resolution=2560,
        )
        self.assertEqual((half.width, half.height), (1510, 2013))
        self.assertEqual(half.effective_resolution_level, 1)
        self.assertEqual(half.prepared_max_resolution, 2013)

    def test_openmvs_processing_geometry_honors_minimum_and_unlimited_maximum(self) -> None:
        fallback = prior.compute_openmvs_processing_geometry(
            width=1000,
            height=800,
            resolution_level=3,
            min_resolution=640,
            max_resolution=2560,
        )
        self.assertEqual((fallback.width, fallback.height), (1000, 800))
        self.assertEqual(fallback.effective_resolution_level, 0)

        unlimited = prior.compute_openmvs_processing_geometry(
            width=3019,
            height=4026,
            resolution_level=0,
            min_resolution=640,
            max_resolution=0,
        )
        self.assertEqual((unlimited.width, unlimited.height), (3019, 4026))
        self.assertEqual(unlimited.prepared_max_resolution, 0)

    def test_roberts_regions_separate_image_edge(self) -> None:
        gray = np.zeros((32, 40), dtype=np.uint8)
        gray[:, 20:] = 200
        edges, magnitude = prior.roberts_edges(gray, 4.0)
        labels, metadata = prior.connected_regions(edges, 8)
        self.assertGreater(float(magnitude[:, 19:21].max()), 4.0)
        self.assertNotEqual(int(labels[10, 5]), 0)
        self.assertNotEqual(int(labels[10, 30]), 0)
        self.assertNotEqual(int(labels[10, 5]), int(labels[10, 30]))
        self.assertEqual(metadata["raw_region_count"], 2)
        self.assertEqual(metadata["retained_region_count"], 2)

    def test_small_components_are_dropped_before_uint16_compaction(self) -> None:
        edges = np.ones((768, 768), dtype=bool)
        edges[::2, ::2] = False
        labels, metadata = prior.connected_regions(
            edges, connectivity=8, minimum_region_size=1
        )
        self.assertGreater(metadata["raw_region_count"], prior.MAX_REGION_LABEL)
        self.assertEqual(metadata["retained_region_count"], 0)
        self.assertEqual(metadata["discarded_region_count"], metadata["raw_region_count"])
        self.assertEqual(int(labels.max()), 0)

    def test_exact_plane_fit_and_determinism(self) -> None:
        height, width = 40, 50
        yy, xx = np.mgrid[:height, :width]
        depth = (0.2 + 0.15 * xx / (width - 1) + 0.1 * yy / (height - 1)).astype(np.float32)
        mask = np.ones((height, width), dtype=bool)
        first = prior.fit_plane(depth, mask, 1e-4, 64, 11)
        second = prior.fit_plane(depth, mask, 1e-4, 64, 11)
        self.assertTrue(first.valid)
        self.assertGreater(first.inlier_ratio, 0.999)
        np.testing.assert_array_equal(first.normal, second.normal)
        self.assertEqual(first.offset, second.offset)

    def test_erosion_can_split_depth_discontinuity(self) -> None:
        height, width = 48, 64
        yy, xx = np.mgrid[:height, :width]
        depth = np.where(
            xx < width // 2,
            0.1 + 0.5 * xx / (width - 1),
            0.8 + 0.3 * yy / (height - 1),
        ).astype(np.float32)
        labels = np.ones((height, width), dtype=np.uint16)
        parameters = prior.PriorParameters(
            eta=20,
            sigma=2.0,
            gamma=0.0,
            epsilon=0.03,
            ransac_threshold=0.01,
            ransac_trials=96,
            seed=3,
        )
        parents = prior.fit_region_planes(depth, labels, parameters)
        eroded, metadata = prior.erode_regions(depth, labels, parents, parameters)
        accepted = [item for item in metadata["split_decisions"] if item["accepted"]]
        self.assertTrue(accepted)
        self.assertGreaterEqual(len(np.unique(eroded[eroded != 0])), 2)
        self.assertGreater(int((eroded == 0).sum()), 0)

    def test_dilation_merges_coplanar_regions_and_reassignment_fills_boundary(self) -> None:
        height, width = 32, 48
        yy, xx = np.mgrid[:height, :width]
        depth = (0.25 + 0.1 * xx / (width - 1) + 0.05 * yy / (height - 1)).astype(np.float32)
        labels = np.zeros((height, width), dtype=np.uint16)
        labels[:, : width // 2] = 1
        labels[:, width // 2 + 1 :] = 2
        parameters = prior.PriorParameters(
            eta=10,
            sigma=0.5,
            kappa=0.7,
            delta=0.8,
            ransac_threshold=1e-3,
            ransac_trials=64,
            seed=5,
        )
        planes = prior.fit_region_planes(depth, labels, parameters)
        dilated, metadata = prior.dilate_regions(labels, planes, parameters)
        self.assertTrue(any(item["accepted"] for item in metadata["merge_decisions"]))
        self.assertEqual(len(np.unique(dilated[dilated != 0])), 1)
        dilated_planes = prior.fit_region_planes(depth, dilated, parameters)
        reassigned, reassignment = prior.reassign_boundary_pixels(
            depth, dilated, dilated_planes, parameters
        )
        self.assertGreater(reassignment["assigned_boundary_pixels"], 0)
        self.assertLess(int((reassigned == 0).sum()), int((dilated == 0).sum()))

    def test_full_topology_is_reproducible_and_cumulative(self) -> None:
        image = np.zeros((36, 52, 3), dtype=np.uint8)
        image[:, 26:] = (180, 180, 180)
        yy, xx = np.mgrid[:36, :52]
        depth = (0.2 + 0.3 * xx / 51 + 0.1 * yy / 35).astype(np.float32)
        parameters = prior.PriorParameters(eta=20, ransac_trials=32, seed=17)
        first, first_metadata = prior.generate_topology(image, depth, parameters)
        second, second_metadata = prior.generate_topology(image, depth, parameters)
        self.assertEqual(tuple(first), prior.STAGE_NAMES)
        for stage in prior.STAGE_NAMES:
            self.assertEqual(first[stage].dtype, np.uint16)
            np.testing.assert_array_equal(first[stage], second[stage])
        np.testing.assert_array_equal(first["roberts_regions"], first["dav2_planarized"])
        self.assertEqual(
            first_metadata["component_filtering"], second_metadata["component_filtering"]
        )
        self.assertEqual(
            first_metadata["component_filtering"]["policy"],
            "components_with_area_lte_eta_become_boundary",
        )


if __name__ == "__main__":
    unittest.main()
