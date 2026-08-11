from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PATCHMATCH = REPO_ROOT / "libs" / "MVS" / "PatchMatchCUDA.cpp"
PATCHMATCH_CUDA = REPO_ROOT / "libs" / "MVS" / "PatchMatchCUDA.cu"
PATCHMATCH_INLINE = REPO_ROOT / "libs" / "MVS" / "PatchMatchCUDA.inl"
SCENE_DENSIFY = REPO_ROOT / "libs" / "MVS" / "SceneDensify.cpp"
ARCHITECTURE = REPO_ROOT / "docs" / "dmap_observability" / "02_architecture.md"


class CppResourceAccountingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patchmatch = PATCHMATCH.read_text(encoding="utf-8")
        cls.patchmatch_cuda = PATCHMATCH_CUDA.read_text(encoding="utf-8")
        cls.patchmatch_inline = PATCHMATCH_INLINE.read_text(encoding="utf-8")
        cls.scene_densify = SCENE_DENSIFY.read_text(encoding="utf-8")
        cls.architecture = ARCHITECTURE.read_text(encoding="utf-8")

    def test_terminal_and_export_peaks_are_named_and_asserted(self) -> None:
        self.assertIn("DMAP_LEGACY_TERMINAL_BYTES_PER_PIXEL == 97u", self.patchmatch)
        self.assertIn("DMAP_MAP_EXPORT_HOST_SCRATCH_BYTES_PER_PIXEL == 21u", self.patchmatch)
        self.assertIn(
            "DMAP_LEGACY_LOGICAL_STATE_STORAGE_BYTES_PER_PIXEL == 61u",
            self.patchmatch,
        )
        self.assertIn("DMAP_INSTRUMENT_FIXED_HOST_BYTES", self.patchmatch)

    def test_reference_patch_layout_metadata_uses_kernel_source_of_truth(self) -> None:
        def macro_value(source: str, name: str) -> int:
            match = re.search(rf"^#define {name} (\d+)$", source, re.MULTILINE)
            self.assertIsNotNone(match, f"missing integer macro {name}")
            return int(match.group(1))

        self.assertEqual(
            macro_value(self.patchmatch_inline, "PATCHMATCHCUDA_PATCH_HALF_WINDOW"),
            macro_value(self.patchmatch_cuda, "nSizeHalfWindow"),
        )
        self.assertEqual(
            macro_value(self.patchmatch_inline, "PATCHMATCHCUDA_PATCH_STEP"),
            macro_value(self.patchmatch_cuda, "nSizeStep"),
        )
        self.assertIn("texDesc.addressMode[0] = cudaAddressModeWrap;", self.patchmatch)
        self.assertIn("texDesc.addressMode[1] = cudaAddressModeWrap;", self.patchmatch)
        self.assertIn("texDesc.normalizedCoords = 0;", self.patchmatch)
        self.assertIn("texDesc.filterMode = cudaFilterModeLinear;", self.patchmatch)
        for field in (
            '"openmvs.dmap.reference_patch_layout"',
            '"sample_offsets_pixels"',
            '"texture_address_mode_configured", "wrap"',
            '"texture_address_mode_effective", "clamp"',
            '"texture_coordinates_normalized", false',
            '"texture_filter_mode", "linear"',
            '"sample_locations_captured_by_kernel", false',
            '"sample_values_captured_by_kernel", false',
            '"source_view_footprints_captured_by_kernel", false',
        ):
            self.assertIn(field, self.patchmatch)

    def test_trace_resources_are_planned_before_materialization(self) -> None:
        estimate = self.patchmatch[self.patchmatch.index("void PatchMatch::EstimateDepthMap") :]
        selection = estimate.index("instrumentTraceSelection = SelectInstrumentTracePixels")
        planning = estimate.index("instrumentExtendedMaps = PlanInstrumentResources")
        preflight = estimate.index("ApplyInstrumentStoragePreflight")
        materialize = estimate.index("activeTracePixels = MaterializeInstrumentTracePixels")
        self.assertLess(selection, planning)
        self.assertLess(planning, preflight)
        self.assertLess(preflight, materialize)
        self.assertIn("trace ? plan.traceDeviceBytes : 0u", self.patchmatch)
        self.assertIn("trace ? plan.traceHostBytes : 0u", self.patchmatch)

    def test_resource_plan_is_cumulative_and_reserves_level_zero(self) -> None:
        for field in (
            '"schema_version", 4',
            '"current_pyramid_storage"',
            '"frame_storage_committed_before"',
            '"full_resolution_priority_reserve"',
            '"frame_priority_reservation_consumed"',
        ):
            self.assertIn(field, self.patchmatch)
        self.assertIn("instrumentFramePriorityReservation", self.patchmatch)

    def test_coarse_compatibility_cost_omission_is_explicit_and_nonfatal(self) -> None:
        self.assertIn('"compatibility_map_contract"', self.patchmatch)
        self.assertIn('"update_source_map_expected"', self.patchmatch)
        self.assertIn('"cost_map_expected", false', self.patchmatch)
        self.assertIn(
            "production confidence maps are retained at pyramid level 0 only",
            self.patchmatch,
        )
        export = self.patchmatch[self.patchmatch.index(
            "if (instrumentEnabled && writeInstrumentMaps)"
        ) :]
        self.assertIn(
            "if (scaleNumber == 0 &&\n"
            "\t\t\t\t\t!SaveInstrumentCostMap",
            export,
        )
        self.assertIn(
            'RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, '
            '"legacy_cost_maps", (int)scaleNumber)',
            export,
        )

    def test_postprocess_peak_counts_all_retained_stage_snapshots(self) -> None:
        self.assertIn(
            "DMapSaturatingMul(pixels, DMapSaturatingMul(postprocessStages, 20u))",
            self.scene_densify,
        )
        self.assertIn(
            "DMapSaturatingAdd(DMapSaturatingMul(postprocessStages, 20u), 10u)",
            self.scene_densify,
        )

    def test_manifest_files_are_bound_before_summary_publication(self) -> None:
        self.assertIn("std::filesystem::symlink_status", self.patchmatch)
        self.assertIn("std::filesystem::is_regular_file", self.patchmatch)
        self.assertIn("size == 0", self.patchmatch)
        manifest_write = self.patchmatch.index(
            'mapManifestWritten = WriteJsonFile(depthMapDir + _T("map_manifest.json")'
        )
        summary_write = self.patchmatch.index(
            'summaryWritten(WriteJsonFile(depthMapDir + _T("summary.json")'
        )
        self.assertLess(manifest_write, summary_write)
        self.assertIn('"pending_map_publication"', self.patchmatch)

    def test_successful_summary_uses_canonical_completion_reference(self) -> None:
        evidence_bound = self.patchmatch.index(
            "const bool mapEvidenceComplete(mapManifestComplete && mapManifestBound)"
        )
        final_reference = self.patchmatch.index(
            'summary["completion_marker"] = {', evidence_bound
        )
        summary_write = self.patchmatch.index(
            'summaryWritten(WriteJsonFile(depthMapDir + _T("summary.json")'
        )
        self.assertLess(evidence_bound, final_reference)
        self.assertLess(final_reference, summary_write)
        for field in (
            '{"path", "capture_complete.json"}',
            '{"path", "prefilter_capture_complete.json"}',
            '{"path", "summary_complete.json"}',
            '{"maps_complete", true}',
            '{"eligible", true}',
        ):
            self.assertIn(field, self.patchmatch[final_reference:summary_write])
        self.assertNotIn('["completion_marker"]["manifest_bytes"]', self.patchmatch)

    def test_process_false_resource_non_equivalence_is_explicit(self) -> None:
        self.assertIn('"observer_process_false_resource_equivalent", false', self.patchmatch)
        self.assertIn(
            '"measurement_status", "unavailable_without_build_bound_receipt"',
            self.patchmatch,
        )
        self.assertIn('"measurement_values", nullptr', self.patchmatch)
        for stale_key in (
            "observer_process_false_register_delta",
            "observer_process_false_stack_delta_bytes",
            "production_black_red_registers",
            "production_black_red_stack_bytes",
            "exact_black_red_registers_photometric",
            "exact_black_red_registers_geometric",
            "exact_black_red_stack_bytes",
            "exact_initialization_registers_photometric",
            "exact_initialization_registers_geometric",
            "exact_initialization_stack_bytes",
        ):
            self.assertNotIn(stale_key, self.patchmatch)
        self.assertIn("does not imply that the observer and production", self.architecture)
        self.assertIn("build-bound resource receipt", self.architecture)

    def test_logical_cost_improvement_reuses_pass_maps_with_exact_semantics(self) -> None:
        self.assertIn(
            "passCostImprovements.size() >= area * (size_t)numPasses",
            self.patchmatch,
        )
        self.assertIn(
            "costImprovement[i] = stateIndex == 0 ? 0.f : passCostImprovements",
            self.patchmatch,
        )
        self.assertIn(
            'saveEvent("cost_improvement_exact", "cost_improvement_exact.pfm"',
            self.patchmatch,
        )
        self.assertIn(
            '"derived_exact", "sum_of_disjoint_checkerboard_production_cost_reductions"',
            self.patchmatch,
        )
        self.assertIn('{"checkerboard_identity_exposed", false}', self.patchmatch)
        self.assertIn("cost increases are zero", self.patchmatch)
        self.assertIn("SaveInstrumentImprovementMaps", self.patchmatch)

    def test_device_budget_scope_and_allocation_failure_are_explicit(self) -> None:
        self.assertIn('"device_budget_scope"', self.patchmatch)
        self.assertIn("explicit observer-owned buffers only", self.patchmatch)
        self.assertIn('"allocation_failure_behavior"', self.patchmatch)
        self.assertIn("runtime stack-pool growth", self.architecture)
        self.assertIn("does not guarantee", self.architecture)
        self.assertIn("terminates the capture", self.architecture)

    def test_exact_depth_prior_weight_mirrors_production_formula(self) -> None:
        cuda_source = (REPO_ROOT / "libs" / "MVS" / "PatchMatchCUDA.cu").read_text(
            encoding="utf-8"
        )
        production_formula = (
            "const float factorDeltaDepth(__expf(cache.varRef * smoothSigmaDepth));"
        )
        self.assertEqual(cuda_source.count(production_formula), 2)
        self.assertNotIn(
            "factorDeltaDepth(__expf(max(cache.varRef, 0.f) * smoothSigmaDepth))",
            cuda_source,
        )


if __name__ == "__main__":
    unittest.main()
