from __future__ import annotations

import unittest

from scripts.python.dmap_observability import component_registry


class ComponentRegistryTest(unittest.TestCase):
    def test_declared_component_overrides_fallback_semantics(self) -> None:
        registry = component_registry.build_registry([
            {
                "signal": "cost_textureless_experiment",
                "signal_label": "Textureless penalty",
                "component_id": "textureless_penalty",
                "mechanism": "texture",
                "quantity": "contribution",
                "units": "cost",
                "domain": "nonnegative",
                "preferred_direction": "lower",
                "minimum_profile": "light",
                "measurement_kind": "map",
                "colormap": "magma",
                "default_visible": True,
            }
        ])

        self.assertEqual(component_registry.validate_registry(registry), [])
        descriptor = component_registry.registry_index(registry)["cost_textureless_experiment"]
        self.assertEqual(descriptor["label"], "Textureless penalty")
        self.assertEqual(descriptor["mechanism"], "texture")
        self.assertEqual(descriptor["component_id"], "textureless_penalty")
        self.assertTrue(descriptor["default_visible"])

    def test_unknown_signal_has_deterministic_generic_descriptor(self) -> None:
        descriptor = component_registry.descriptor_from_row({"signal": "patch_deformation_condition"})

        self.assertEqual(descriptor.mechanism, "patch")
        self.assertEqual(descriptor.minimum_profile, "light")
        self.assertEqual(descriptor.measurement_kind, "map")

    def test_conflicting_declarations_fail_closed(self) -> None:
        rows = [
            {"signal": "new_score", "mechanism": "cost"},
            {"signal": "new_score", "mechanism": "texture"},
        ]

        with self.assertRaisesRegex(ValueError, "conflicting descriptors"):
            component_registry.build_registry(rows)

    def test_observed_profiles_do_not_conflict_for_the_same_signal(self) -> None:
        registry = component_registry.build_registry([
            {"signal": "depth_final_before_filter", "capture_profile": "prefilter"},
            {"signal": "depth_final_before_filter", "capture_profile": "deep"},
            {"signal": "depth_final_before_filter", "capture_profile": "trace"},
        ])

        descriptor = component_registry.registry_index(registry)[
            "depth_final_before_filter"
        ]
        self.assertEqual(descriptor["minimum_profile"], "light")

    def test_validator_rejects_unknown_contract_values(self) -> None:
        registry = component_registry.build_registry([{"signal": "cost_stored"}])
        registry["signals"][0]["minimum_profile"] = "everything"

        self.assertIn(
            "signals[0].minimum_profile is invalid",
            component_registry.validate_registry(registry),
        )

    def test_mechanism_screen_signals_have_stable_component_semantics(self) -> None:
        expected = {
            "view_probability_mass": ("view_selection", "probability_health", "probability_mass"),
            "view_probability_health_status": ("view_selection", "probability_health", "probability_health_status"),
            "jbu_transfer_depth_delta": ("multiscale", "pyramid_depth_transfer", "depth_transfer_delta"),
            "jbu_fallback_status": ("multiscale", "pyramid_depth_transfer", "fallback_status"),
            "adaptive_patch_support_mode": ("patch", "adaptive_support", "support_mode"),
            "adaptive_patch_fallback_status": ("patch", "adaptive_support", "fallback_status"),
            "hierarchy_improvement_margin": ("multiscale", "hierarchy_gate", "hierarchy_improvement_margin"),
            "hierarchy_update_status": ("multiscale", "hierarchy_gate", "hierarchy_update_status"),
            "low_texture_update_eligible_exact": ("candidate_update", "low_texture_update_hysteresis", "eligibility"),
            "low_texture_update_ambiguity_exact": ("candidate_update", "low_texture_update_hysteresis", "ambiguity"),
            "low_texture_update_required_gain_exact": ("candidate_update", "low_texture_update_hysteresis", "required_gain"),
            "low_texture_update_best_proposed_gain_exact": ("candidate_update", "low_texture_update_hysteresis", "best_proposed_gain"),
            "low_texture_update_rejected_mask_exact": ("candidate_update", "low_texture_update_hysteresis", "rejected_update_mask"),
            "low_texture_update_would_have_won_source_exact": ("candidate_update", "low_texture_update_hysteresis", "would_have_won_source"),
            "low_texture_update_rejected_count_exact": ("candidate_update", "low_texture_update_hysteresis", "rejected_update_count"),
            "candidate_raw_best_cost_exact": ("candidate_update", "low_texture_update_hysteresis", "raw_best_cost"),
            "candidate_raw_runner_up_cost_exact": ("candidate_update", "low_texture_update_hysteresis", "raw_runner_up_cost"),
            "gap_raw_best_runner_up_exact": ("candidate_update", "low_texture_update_hysteresis", "raw_best_runner_up_gap"),
            "candidate_retained_minus_raw_best_exact": ("candidate_update", "low_texture_update_hysteresis", "retained_minus_raw_best"),
            "candidate_raw_suppression_identity_exact": ("candidate_update", "low_texture_update_hysteresis", "raw_suppression_identity"),
        }

        for signal, values in expected.items():
            with self.subTest(signal=signal):
                descriptor = component_registry.descriptor_from_row({"signal": signal})
                self.assertEqual(
                    (descriptor.mechanism, descriptor.component_id, descriptor.quantity),
                    values,
                )
                self.assertEqual(descriptor.minimum_profile, "deep")
                self.assertTrue(descriptor.description)


if __name__ == "__main__":
    unittest.main()
