#!/usr/bin/env python3
"""Structured report model and local investigation assets for DMAP experiments."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

try:
    from dmap_observability import (
        component_registry, integrity, reference_patch_layout, region_metrics,
    )
    import dmap_drilldown
except ImportError:  # Imported as scripts.python.dmap_report_model in unit tests.
    from scripts.python.dmap_observability import (
        component_registry, integrity, reference_patch_layout, region_metrics,
    )
    from scripts.python import dmap_drilldown


SCHEMA_NAME = "openmvs.dmap.development_report"
SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
VIEWER_SCHEMA_VERSION = 3
PIXEL_DATA_BUDGET_BYTES = 768 * 1024 * 1024
PIXEL_ENCODING = "float32_le_gzip_base64_js_v1"
LOSSLESS_RASTER_PIXEL_SIGNALS = {
    "candidate_identity_exact",
    "candidate_raw_suppression_identity_exact",
    "candidate_counts_exact",
    "selected_view_counts_exact",
    "selected_views_before_mask_exact",
    "selected_views_after_mask_exact",
    "view_probability_health_status",
    "view_probability_legacy_last_view_collapse",
    "jbu_fallback_status",
    "adaptive_patch_support_mode",
    "adaptive_patch_activation",
    "adaptive_patch_fallback_status",
    "hierarchy_update_status",
}
BUILTIN_CATEGORY_LEGENDS = {
    "view_probability_legacy_last_view_collapse": {"0": "no collapse", "1": "legacy last-view collapse"},
    "adaptive_patch_support_mode": {"0": "fixed 9x9 eligibility", "1": "eligible for adaptive 13x13"},
    "adaptive_patch_activation": {"0": "inactive", "1": "adaptive rule eligible"},
    "hierarchy_update_status": {"0": "retained scale-entry hypothesis", "1": "crossed configured margin"},
    "low_texture_update_eligible_exact": {"0": "not eligible", "1": "eligible"},
    "low_texture_update_rejected_mask_exact": {
        "0": "no gated rejection", "1": "propagation", "2": "refinement",
        "3": "propagation and refinement",
    },
    "low_texture_update_would_have_won_source_exact": {
        "0": "none", "2": "propagation", "3": "depth refinement",
        "4": "normal refinement", "5": "random-normal refinement",
        "6": "surface-normal refinement",
    },
}
CRITICAL_PIXEL_SIGNALS = {
    "candidate_identity_exact",
    "candidate_counts_exact",
    "depth_final_after_filter",
    "normal_final",
    "cost_final",
    "confidence_final",
    "view_cost_components_exact",
    "view_selection_metrics_exact",
    "view_weighted_contribution_exact",
    "low_texture_update_eligible_exact",
    "low_texture_update_ambiguity_exact",
    "low_texture_update_required_gain_exact",
    "low_texture_update_best_proposed_gain_exact",
    "low_texture_update_rejected_mask_exact",
    "low_texture_update_would_have_won_source_exact",
    "low_texture_update_rejected_count_exact",
    "candidate_raw_best_cost_exact",
    "candidate_raw_runner_up_cost_exact",
    "gap_raw_best_runner_up_exact",
    "candidate_retained_minus_raw_best_exact",
    "candidate_raw_suppression_identity_exact",
}
MAX_SCALE_SAMPLES_PER_MAP = 65_536
PREVIEW_MAX_DIMENSION = 480
INVESTIGATION_GUIDE_HEADING = "## How to Investigate a Change"
COMPLETED_TRACE_SCHEMA_NAME = "openmvs.dmap.completed_trace_rows"
COMPLETED_TRACE_SCHEMA_VERSION = 2
MAX_COMPLETED_TRACE_ROWS = dmap_drilldown.MAX_TRACE_REPORT_ROWS
MAX_TRACE_ARRAY_VALUES = 64
MAX_TRACE_JSONL_LINE_BYTES = 1024 * 1024
CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE = "unavailable_post_pass_snapshot"
CANDIDATE_ACCOUNTING_METRICS = (
    "changed_ratio",
    "candidates_tested",
    "candidates_finite",
    "candidates_accepted",
    "tested_candidates",
    "finite_candidates",
    "accepted_candidates",
    "acceptance_rate",
)
LOW_TEXTURE_UPDATE_HYSTERESIS_SCHEMA_NAME = (
    "openmvs.dmap.low_texture_update_hysteresis_metrics"
)
LOW_TEXTURE_UPDATE_HYSTERESIS_SCHEMA_VERSION = 1
LOW_TEXTURE_UPDATE_COUNTER_FIELDS = (
    "low_texture_gate_eligible",
    "low_texture_propagation_accepted",
    "low_texture_propagation_rejected",
    "low_texture_refinement_accepted",
    "low_texture_refinement_rejected",
    "low_texture_required_gain_sum",
    "low_texture_best_proposed_gain_sum",
)
EVIDENCE_CONTEXT_SCHEMA_NAME = "openmvs.dmap.report_evidence_context"
EVIDENCE_CONTEXT_SCHEMA_VERSION = 1
EVIDENCE_CONTEXT_DIGEST_FIELD = "context_sha256"
EVIDENCE_CONTEXT_FIELDS = frozenset({
    "schema_name", "schema_version", EVIDENCE_CONTEXT_DIGEST_FIELD,
    "subject", "mechanics_authority", "quality_authority",
    "separation_contract",
})
EVIDENCE_CONTEXT_METRIC_STATUSES = frozenset({
    "pass", "fail", "warning", "info", "unavailable",
})
EVIDENCE_CONTEXT_MECHANICS_VERDICTS = frozenset({
    "supported", "rejected", "inconclusive",
})
EVIDENCE_CONTEXT_MECHANICS_COVERAGE = frozenset({
    "captured", "quality_only", "not_captured", "not_applicable",
})
EVIDENCE_CONTEXT_CANDIDATE_FIELDS = (
    "candidate",
    "mechanics_coverage",
    "accuracy_rank",
    "strict_accuracy_pass",
    "scene_count",
    "primary_scene_metric_rows",
    "baseline_successful_structures",
    "paired_successful_structures",
    "lost_baseline_fit_count",
    "availability_biased",
    "median_normalized_noise_loss",
    "worst_normalized_noise_loss",
    "residual_p95_delta_mm",
    "threshold_auc_delta_pp",
    "inlier_5mm_delta_pp",
    "effective_coverage_delta_pp",
    "spatial_coverage_delta_pp",
    "estimator_validity_delta_pp",
    "endpoint_validity_delta_pp",
    "runtime_delta_percent",
)
EVIDENCE_CONTEXT_INTEGER_CANDIDATE_FIELDS = frozenset({
    "accuracy_rank",
    "scene_count",
    "primary_scene_metric_rows",
    "baseline_successful_structures",
    "paired_successful_structures",
    "lost_baseline_fit_count",
})
EVIDENCE_CONTEXT_NUMERIC_CANDIDATE_FIELDS = frozenset({
    "median_normalized_noise_loss",
    "worst_normalized_noise_loss",
    "residual_p95_delta_mm",
    "threshold_auc_delta_pp",
    "inlier_5mm_delta_pp",
    "effective_coverage_delta_pp",
    "spatial_coverage_delta_pp",
    "estimator_validity_delta_pp",
    "endpoint_validity_delta_pp",
    "runtime_delta_percent",
})
EVIDENCE_CONTEXT_SEPARATION_CONTRACT = {
    "mechanics_do_not_establish_quality": True,
    "external_quality_rows_are_not_current_report_rows": True,
    "current_report_aggregates_remain_capture_local": True,
    "automatic_promotion": False,
}
EVIDENCE_CONTEXT_DEEP_RUN_SUFFIX = " [deep]"


def evidence_context_run_candidate(value: Any) -> str:
    """Normalize only the report's exact diagnostic run suffix."""

    label = str(value)
    return (
        label[:-len(EVIDENCE_CONTEXT_DEEP_RUN_SUFFIX)]
        if label.endswith(EVIDENCE_CONTEXT_DEEP_RUN_SUFFIX)
        else label
    )

INVESTIGATION_GUIDE = {
    "schema_name": "openmvs.dmap.investigation_guide",
    "schema_version": 1,
    "title": "How to investigate a change",
    "introduction": (
        "Start with a concrete hypothesis, compare one baseline with one variant, and move from "
        "aggregate regressions to a frame, a logical iteration, and finally an exact pixel. Keep "
        "quality and runtime separate, and treat internal mechanics as explanations to verify rather "
        "than proof of final geometric improvement."
    ),
    "quick_start": {
        "title": "Baseline-to-pixel workflow",
        "summary": "Use this sequence for every algorithm experiment before following a specialized recipe.",
        "steps": [
            "Select Baseline, Variant, and their repeats; then choose the Scene, Frame, and Capture stage.",
            "Use Final state per run for endpoint quality. Use Same logical iteration when isolating where behavior first diverges.",
            "Start with Shared map scaling so both runs use the same color range. Local only rescales each preview and can hide magnitude differences.",
            "Sort Ranked regressions and use Inspect to navigate from an aggregate metric to the responsible frame.",
            "Click the same spatial feature in any map. The shared crosshair and Pixel inspection table expose exact values when a numeric payload was retained.",
            "Confirm the mechanism-level hypothesis against final coverage, filtering, annotation consistency, runtime, and repeatability before accepting the change.",
            "The URL hash stores the selected controls and crosshair, so copy the complete browser URL when sharing an investigation state.",
        ],
    },
    "recipes": [
        {
            "key": "overview",
            "title": "Triage an unfamiliar regression",
            "summary": "Determine whether the first visible failure is cost, candidate motion, view support, filtering, or geometry.",
            "steps": [
                "Choose Final state per run and the Overview preset, then inspect the highest-ranked regression.",
                "Check reference RGB, final depth and normal, total cost, confidence gap, support, validity, and rejection reason at the same pixel.",
                "Switch to Same logical iteration and move from Initialization forward until the two runs first diverge.",
                "Use the corresponding specialized recipe once the first diverging mechanism is identified.",
            ],
            "look_for": [
                "A cost change without a depth/normal change, which may be calibration rather than useful geometry.",
                "A validity loss caused by filtering rather than PatchMatch estimation.",
                "Low winner gap, high view churn, or weak support around the failing region.",
            ],
            "cautions": [
                "A synchronized internal change supports a mechanism hypothesis but does not establish causality.",
            ],
            "action_label": "Open overview triage",
            "action": {
                "alignment": "final",
                "map_preset": "overview",
                "target": "regression-heading",
            },
        },
        {
            "key": "cost",
            "title": "Debug a cost-function change",
            "summary": "Test whether the new objective improves candidate ordering and final geometry instead of merely shifting score scale.",
            "steps": [
                "Use Same logical iteration and the Cost and candidate mechanics preset; compare Initialization first, then each complete iteration.",
                "Inspect exact raw photometric, photo-plus-prior, geometric, and total production costs together with prior disagreement, prior weight, and winner gap.",
                "Use Shared scaling and the crosshair to determine whether changed cost regions align with RGB texture, depth discontinuities, or annotation structures.",
                "Return to Final state per run and check coverage, keep-cost rejection, annotation residuals, runtime, and repeatability.",
            ],
            "look_for": [
                "Lower total cost accompanied by larger winner gaps and improved annotation consistency.",
                "A component unexpectedly dominating the total or changing where its input should be inactive.",
                "Lower scores with unchanged or worse geometry, suggesting score rescaling or candidate mis-ordering.",
            ],
            "cautions": [
                "Compare exact production-basis signals with exact signals; equal-selected-view rescoring remains explicitly labeled proxy evidence.",
                "Configured ignore-mask rejection is separate from keep-cost rejection and must not be attributed to the objective.",
                "Lower cost is better. Stored confidence is derived exactly as max(1 - stored cost, 0); a small winner gap indicates candidate ambiguity.",
                "cost_improvement_exact is positive-only: cost increases map to zero, and view-set changes can change the stored-cost basis. Read it beside stored cost and selected views.",
                "Interpret geometric cost only in a Geometric N capture. It is identically zero in a photometric-only capture.",
            ],
            "action_label": "Set up cost inspection",
            "action": {
                "alignment": "same",
                "map_preset": "cost",
                "target": "maps-heading",
                "signals": [
                    "reference_rgb",
                    "cost_stored",
                    "confidence_stored",
                    "cost_photo_raw_production_exact",
                    "cost_photo_prior_production_exact",
                    "cost_geometric_production_exact",
                    "cost_total_production_exact",
                    "cost_improvement_exact",
                    "depth_prior_disagreement_production_exact",
                    "depth_prior_weight_production_exact",
                    "gap_winner_runner_up_exact",
                ],
            },
        },
        {
            "key": "propagation",
            "title": "Debug a propagation change",
            "summary": "Determine whether propagation tests better hypotheses, wins in the intended regions, and reduces downstream ambiguity.",
            "steps": [
                "Use Same logical iteration and compare the first iteration where propagation differs.",
                "Inspect exact candidate identity, tested/finite/accepted masks, incumbent/winner costs, winner gap, and selected-view transitions.",
                "Corroborate the maps with Candidate update attribution and winner gap source counts; iteration counts are sequential acceptances, not only final winners.",
                "Check depth/normal deltas, view churn, and later cost improvement before evaluating final coverage and geometry.",
            ],
            "look_for": [
                "Propagation acceptances spreading from reliable textured or geometrically supported regions.",
                "Winner slots 1-8 identify propagation neighbors; slots 9-12 identify depth, normal, random-normal, and surface-normal refinement candidates.",
                "More tested candidates but fewer finite or accepted candidates, indicating invalid proposals or stricter scoring interaction.",
                "Large propagation wins followed by churn or reversal in later iterations.",
            ],
            "cautions": [
                "Initialization reports stored assignments; later iterations report sequential incumbent improvements. Do not compare those counts as identical events.",
                "Candidate identity is a packed three-channel map: winner slot, runner-up slot, and final update source. Use the tile/channel caption and provenance rather than treating its RGB values as color.",
            ],
            "action_label": "Set up propagation inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "target": "maps-heading",
                "signals": [
                    "reference_rgb",
                    "candidate_identity_exact",
                    "candidate_counts_exact",
                    "candidate_tested_mask_exact",
                    "candidate_finite_mask_exact",
                    "candidate_accepted_mask_exact",
                    "candidate_incumbent_cost_exact",
                    "candidate_winner_cost_exact",
                    "cost_improvement_exact",
                    "gap_winner_runner_up_exact",
                    "depth_delta",
                    "normal_angle_delta",
                    "view_churn",
                ],
            },
        },
        {
            "key": "view_selection",
            "title": "Debug a view-selection change",
            "summary": "Separate candidate ranking and healthy probability mass from stochastic selection, reliability weighting, and per-view cost contribution.",
            "steps": [
                "Use Same logical iteration and select the pyramid level where the view rule executes.",
                "Inspect pre-CDF probability mass, positive-view count, health status, unassigned draws, and predicted legacy last-view collapse.",
                "Use Source view to compare exact selection state, probability, reliability weight, photometric/geometric cost, and weighted contribution.",
                "Return to final-state support, filtering, annotation consistency, and repeatability before accepting the view change.",
            ],
            "look_for": [
                "Zero or nonfinite probability mass colocated with last-view collapse or weak geometric evidence.",
                "A candidate that improves probability health without concentrating support on one unreliable source view.",
                "Ranking changes that improve final annotation residuals rather than only changing selected-view count.",
            ],
            "cautions": [
                "Health maps describe pre-CDF mechanics; they do not by themselves prove that a fallback improves geometry.",
                "Broad summary captures retain counters but declare unavailable exact per-pixel hot-path maps explicitly.",
            ],
            "action_label": "Set up view-selection inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "target": "maps-heading",
                "mechanisms": ["view_selection"],
                "signals": [
                    "reference_rgb", "view_probability_mass",
                    "view_probability_positive_count", "view_probability_health_status",
                    "view_probability_unassigned_draw_count",
                    "view_probability_legacy_last_view_collapse",
                    "view_selection_state_exact", "view_weighted_contribution_exact",
                ],
            },
        },
        {
            "key": "texture",
            "title": "Debug a textureless-region change",
            "summary": "Verify that a texture score activates in the intended regions and improves geometry instead of only changing confidence or completeness.",
            "steps": [
                "Compare the registered texture score, activation, threshold, weight, and affected cost components at the same logical iteration.",
                "Use Texture-stratified outcomes to compare cost, winner gap, updates, coverage, filtering, and annotation consistency in low, mid, and high texture regions.",
                "Inspect newly valid, newly invalid, and stable-control pixels with synchronized RGB and delta maps.",
                "Export an exact pixel trace when activation or component closure cannot be explained from the broad maps.",
            ],
            "look_for": [
                "Activation concentrated in low-texture areas without spreading across detailed boundaries.",
                "Larger winner gaps and better annotation consistency in low-texture regions.",
                "Coverage gains that survive filtering without increasing residual tails.",
            ],
            "cautions": [
                "Texture quantiles are computed per matched frame and logical state; compare corresponding bins rather than their raw threshold values alone.",
                "A lower cost or higher coverage is not sufficient when line/plane residuals regress.",
            ],
            "action_label": "Set up texture inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "target": "mechanics-heading",
                "mechanisms": ["texture", "cost", "candidate_update"],
                "signals": [
                    "reference_rgb", "texture_score", "texture_activation",
                    "reference_variance_production_exact", "cost_total_production_exact",
                    "gap_winner_runner_up_exact", "depth_delta", "rejection_reason",
                ],
            },
        },
        {
            "key": "patch",
            "title": "Debug a patch or photometric-scoring change",
            "summary": "Separate patch support and variance failures from view projection, candidate generation, and final aggregation effects.",
            "steps": [
                "Capture a baseline before rebuilding the patch/scoring code, then compare with Same logical iteration.",
                "Inspect reference-patch variance, exact raw photometric cost, prior blend/weight, finite-candidate masks, and winner gap.",
                "Use Source view and Channel to inspect per-view photometric, geometric, and total cost channels plus their weighted contribution.",
                "When the component registry declares patch-support signals, inspect their eligibility, activation, valid-sample fraction, and fallback status alongside the core cost evidence.",
                "Use targeted trace reruns for representative low-variance, boundary, newly invalid, and stable-control pixels when aggregate maps are insufficient.",
            ],
            "look_for": [
                "Low reference variance aligned with bad/finite-candidate changes.",
                "Reference variance should remain constant across iterations within a run; use an A/B change as evidence of changed patch layout or weighting.",
                "One source view dominating or becoming invalid after a patch-window or sampling change.",
                "Better photometric separation without increased boundary bleeding, depth noise, or annotation residuals.",
            ],
            "cautions": [
                "The reference loupe derives fixed-grid positions from a validated observer layout contract and exact pyramid dimensions; it does not expose CUDA sample values, weights, ZNCC terms, or source-view footprints.",
                "CUDA clamps these unnormalized texture samples at borders even though the descriptor requests wrap; inspect the loupe's clamped-sample markers rather than assuming out-of-bounds rejection.",
                "For multi-channel maps, the tile caption is authoritative; the global Channel selector can contain generic labels contributed by another signal.",
            ],
            "action_label": "Set up patch inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "showPatchLayout": True,
                "target": "maps-heading",
                "signals": [
                    "reference_rgb",
                    "reference_variance_production_exact",
                    "candidate_finite_mask_exact",
                    "cost_photo_raw_production_exact",
                    "cost_photo_prior_production_exact",
                    "depth_prior_weight_production_exact",
                    "gap_winner_runner_up_exact",
                    "view_cost_components_exact",
                    "view_selection_metrics_exact",
                    "view_weighted_contribution_exact",
                    "adaptive_patch_support_mode",
                    "adaptive_patch_activation",
                    "adaptive_patch_valid_sample_fraction",
                    "adaptive_patch_fallback_status",
                ],
            },
        },
        {
            "key": "deformable_patch",
            "title": "Debug a deformable-patch change",
            "summary": "Separate deformation selection and support failures from the downstream photometric and geometric effect.",
            "steps": [
                "Compare deformation magnitude, condition, valid-sample fraction, border failures, and score delta at the same logical iteration.",
                "Check candidate ordering, per-view contribution, winner gap, depth/normal changes, and boundary validity around affected patches.",
                "Trace representative low-texture, boundary, improved, regressed, and stable-control pixels.",
                "In the trace explorer, reconstruct the score from saved footprints, samples, weights, intensities, and residuals.",
            ],
            "look_for": [
                "Small, well-conditioned deformations that improve separation without reducing valid support.",
                "Large or unstable deformations concentrated at image borders or depth discontinuities.",
                "Per-view deformation effects that agree with the final selected-view contribution.",
            ],
            "cautions": [
                "Full patch samples are trace-level evidence; broad frame maps contain deformation summaries only.",
                "A better patch score is useful only when candidate ordering and final geometry improve.",
            ],
            "action_label": "Set up deformable-patch inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "target": "maps-heading",
                "mechanisms": ["patch", "cost", "view_selection"],
                "signals": [
                    "reference_rgb", "patch_deformation_magnitude",
                    "patch_valid_sample_fraction", "patch_score_delta",
                    "cost_photo_raw_production_exact", "gap_winner_runner_up_exact",
                    "view_weighted_contribution_exact",
                ],
            },
        },
        {
            "key": "multiscale",
            "title": "Debug a multiscale change",
            "summary": "Identify which pyramid level introduces a gain or regression before attributing the final full-resolution result.",
            "steps": [
                "Hold resolution level and all unrelated parameters fixed; vary only sub-resolution levels or the scale-specific rule being tested.",
                "Use Pyramid level to compare per-level iteration counters, cost/update summaries, registered transfer signals, and checkerboard timing rows.",
                "At each captured fine-level initialization, inspect any coarse-to-fine transfer signals declared by the component registry; do not infer an unavailable transfer rule from filenames or resolution alone.",
                "Use Final state for any registered hierarchy or scale-transition summaries, and keep terminal summaries distinct from per-iteration proposal evidence.",
                "Check Final state per run, annotation consistency, coverage, runtime, and storage before accepting an additional scale.",
            ],
            "look_for": [
                "The first pyramid level where cost, support, propagation acceptance, or validity diverges.",
                "Coarse-level improvements that disappear or reverse during scale-0 refinement.",
                "Runtime growth or memory/storage admission that outweighs the final geometric gain.",
            ],
            "cautions": [
                "Pyramid level selects algorithm state; Shared and Local select only preview color normalization.",
                "An Unspecified level means the capture did not retain level identity; do not infer level attribution from filenames or image dimensions.",
                "Geometric-consistency estimation disables the sub-resolution pyramid, so test photometric multiscale and geometric-stage changes separately.",
            ],
            "action_label": "Open scale-0 comparison",
            "action": {
                "alignment": "final",
                "map_preset": "custom",
                "target": "maps-heading",
                "mechanisms": ["multiscale"],
                "signals": [
                    "reference_rgb", "cost_total_production_exact",
                    "gap_winner_runner_up_exact", "depth_delta",
                    "normal_angle_delta", "view_churn",
                    "depth_final_after_filter",
                ],
            },
        },
    ],
}

DEFAULT_SIGNALS = (
    "reference_rgb",
    "cost_total_production_exact",
    "cost_improvement_exact",
    "cost_photo_raw_production_exact",
    "cost_photo_prior_production_exact",
    "cost_geometric_production_exact",
    "gap_winner_runner_up_exact",
    "candidate_identity_exact",
    "candidate_counts_exact",
    "view_weighted_contribution_exact",
    "view_selection_state_exact",
    "cost_stored",
    "confidence_stored",
    "cost_photo_raw_equal_selected_rescore_proxy",
    "cost_geometric_equal_selected_rescore_proxy",
    "cost_stored_minus_rescore",
    "gap_local_neighbor_equal_selected_rescore_proxy",
    "depth_delta",
    "normal_angle_delta",
    "view_churn",
    "depth_final_after_filter",
)

LOW_TEXTURE_UPDATE_EXTENSION_SIGNALS = frozenset({
    "low_texture_update_eligible_exact",
    "low_texture_update_ambiguity_exact",
    "low_texture_update_required_gain_exact",
    "low_texture_update_best_proposed_gain_exact",
    "low_texture_update_rejected_mask_exact",
    "low_texture_update_would_have_won_source_exact",
    "low_texture_update_rejected_count_exact",
    "candidate_raw_best_cost_exact",
    "candidate_raw_runner_up_cost_exact",
    "gap_raw_best_runner_up_exact",
    "candidate_retained_minus_raw_best_exact",
    "candidate_raw_suppression_identity_exact",
})


def investigation_guide_for_registry(
    registry: Mapping[str, Any],
    *,
    low_texture_update_declared: bool = False,
) -> dict[str, Any]:
    """Tailor recipes to signals declared by this report's component registry."""

    guide = copy.deepcopy(INVESTIGATION_GUIDE)
    recipe_mechanisms = {
        "cost": {"cost", "candidate_update"},
        "propagation": {"propagation", "candidate_update"},
        "view_selection": {"view_selection"},
        "texture": {"texture", "cost", "candidate_update"},
        "patch": {"patch", "cost", "view_selection", "candidate_update"},
        "deformable_patch": {"patch", "cost", "view_selection"},
        "multiscale": {"multiscale"},
    }
    descriptors = {
        str(row.get("signal_id")): row
        for row in registry.get("signals") or []
        if isinstance(row, Mapping) and row.get("signal_id")
    }
    for recipe in guide["recipes"]:
        action = recipe.get("action") or {}
        mechanisms = set(action.get("mechanisms") or []) | recipe_mechanisms.get(
            str(recipe.get("key")), set()
        )
        if not mechanisms:
            continue
        declared = [
            signal_id
            for signal_id, descriptor in descriptors.items()
            if descriptor.get("mechanism") in mechanisms
            and bool(descriptor.get("default_visible"))
        ]
        action["signals"] = list(dict.fromkeys([
            *(action.get("signals") or []),
            *sorted(declared),
        ]))

    declared_hysteresis = sorted(
        LOW_TEXTURE_UPDATE_EXTENSION_SIGNALS & descriptors.keys()
    )
    if low_texture_update_declared or declared_hysteresis:
        guide["recipes"].append({
            "key": "low_texture_update_hysteresis",
            "title": "Debug low-texture update hysteresis",
            "summary": (
                "Inspect the optional registered update gate without treating it as a base "
                "OpenMVS mechanism."
            ),
            "steps": [
                "Use Same logical iteration and find the first state where the gate changes retained candidates.",
                "Compare declared eligibility, required-gain, rejected-update, and raw-order signals with the core candidate and cost maps.",
                "Return to final annotation consistency, coverage, filtering, runtime, and repeatability before accepting the change.",
            ],
            "look_for": [
                "Rejected proposals concentrated where the registered required gain exceeds the proposed improvement.",
                "Reduced update churn without lost propagation or worse geometric residual tails.",
            ],
            "cautions": [
                "This recipe is present only because the capture declared extension signals.",
                "Proposal counts and retained-winner evidence must be interpreted using the extension schema carried by the report.",
            ],
            "action_label": "Set up hysteresis inspection",
            "action": {
                "alignment": "same",
                "map_preset": "custom",
                "target": "mechanics-heading",
                "mechanisms": ["texture", "candidate_update", "propagation", "cost"],
                "signals": [
                    "reference_rgb",
                    "reference_variance_production_exact",
                    "candidate_identity_exact",
                    "gap_winner_runner_up_exact",
                    "depth_delta",
                    "normal_angle_delta",
                    "view_churn",
                    *declared_hysteresis,
                ],
            },
        })
    return guide


def optional_extension_contracts(
    registry: Mapping[str, Any],
    *,
    low_texture_update_declared: bool = False,
) -> list[dict[str, Any]]:
    """Describe known optional schemas only when the capture declares their signals."""

    signal_ids = {
        str(row.get("signal_id"))
        for row in registry.get("signals") or []
        if isinstance(row, Mapping) and row.get("signal_id")
    }
    contracts: list[dict[str, Any]] = []
    low_texture_signals = sorted(LOW_TEXTURE_UPDATE_EXTENSION_SIGNALS & signal_ids)
    if low_texture_update_declared or low_texture_signals:
        contracts.append({
            "extension_id": "low_texture_update_hysteresis",
            "schema_version": 1,
            "signals": low_texture_signals,
            "counter_evidence_declared": low_texture_update_declared,
            "base_schema_required": False,
            "maps": "exact logical events for iterations only; initialization is unavailable",
            "proposal_outcomes": (
                "accepted and rejected are mutually exclusive only for proposals that improve "
                "the legacy incumbent"
            ),
            "proposal_rate_denominator": (
                "accepted plus hysteresis-rejected gate-controlled proposals"
            ),
            "gain_mean_denominator": "eligible pixels",
        })
    return contracts

PATH_COLUMNS = {
    "depthmap_dir", "map_manifest", "manifest_path", "path", "timing_source",
    "reference_dmap", "visual_overlay_svg", "visual_residual_histogram_svg",
    "source_csv", "source_json", "source_map",
}


def signal_mechanism(signal: str) -> str:
    return component_registry.descriptor_from_row({"signal": signal}).mechanism


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def json_value(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible values."""
    if isinstance(value, np.generic):
        value = value.item()
    if _missing(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def evidence_context_digest(value: dict[str, Any]) -> str:
    """Return the canonical self-digest for a report evidence context."""

    payload = dict(value)
    payload.pop(EVIDENCE_CONTEXT_DIGEST_FIELD, None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_evidence_context(value: Any) -> dict[str, Any]:
    """Validate the optional external-authority presentation envelope.

    The envelope carries only derived headline evidence and content identities.
    Raw external quality rows remain outside the current report's aggregates.
    """

    errors: list[str] = []
    if not isinstance(value, dict):
        return {
            "schema_name": EVIDENCE_CONTEXT_SCHEMA_NAME,
            "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
            "valid": False,
            "errors": ["evidence context must be a JSON object"],
        }

    def require_string(record: dict[str, Any], key: str, location: str) -> None:
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f"{location}.{key} must be a nonempty string")

    def validate_sources(raw: Any, location: str) -> None:
        if not isinstance(raw, list) or not raw:
            errors.append(f"{location} must be a nonempty list")
            return
        roles: set[str] = set()
        allowed = {
            "role", "sha256", "bytes", "schema_name", "schema_version",
            "content_digest", "cardinality",
        }
        for index, source in enumerate(raw):
            item_location = f"{location}[{index}]"
            if not isinstance(source, dict):
                errors.append(f"{item_location} must be an object")
                continue
            extras = sorted(set(source) - allowed)
            if extras:
                errors.append(
                    f"{item_location} has unsupported fields: {', '.join(extras)}"
                )
            missing = sorted({"role", "sha256", "bytes", "content_digest", "cardinality"} - set(source))
            if missing:
                errors.append(
                    f"{item_location} is missing required fields: {', '.join(missing)}"
                )
            require_string(source, "role", item_location)
            role = source.get("role")
            if isinstance(role, str):
                if role in roles:
                    errors.append(f"{location} contains duplicate role {role!r}")
                roles.add(role)
            digest = source.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                errors.append(f"{item_location}.sha256 must be lowercase SHA-256")
            size = source.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                errors.append(f"{item_location}.bytes must be a nonnegative integer")
            if "schema_name" in source and source["schema_name"] is not None:
                require_string(source, "schema_name", item_location)
            if "schema_version" in source and source["schema_version"] is not None:
                version = source["schema_version"]
                if isinstance(version, bool) or not isinstance(version, int) or version < 1:
                    errors.append(
                        f"{item_location}.schema_version must be a positive integer"
                    )
            content_digest = source.get("content_digest")
            if (
                not isinstance(content_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", content_digest) is None
            ):
                errors.append(
                    f"{item_location}.content_digest must be lowercase SHA-256"
                )
            cardinality = source.get("cardinality")
            if not isinstance(cardinality, dict) or not cardinality:
                errors.append(f"{item_location}.cardinality must be a nonempty object")
            else:
                for name, count in cardinality.items():
                    if not isinstance(name, str) or not name.strip():
                        errors.append(
                            f"{item_location}.cardinality keys must be nonempty strings"
                        )
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        errors.append(
                            f"{item_location}.cardinality.{name} must be a nonnegative integer"
                        )

    if set(value) != EVIDENCE_CONTEXT_FIELDS:
        missing = sorted(EVIDENCE_CONTEXT_FIELDS - set(value))
        extras = sorted(set(value) - EVIDENCE_CONTEXT_FIELDS)
        errors.append(
            "evidence context fields are invalid"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unsupported {', '.join(extras)}" if extras else "")
        )
    if value.get("schema_name") != EVIDENCE_CONTEXT_SCHEMA_NAME:
        errors.append(
            f"schema_name must be {EVIDENCE_CONTEXT_SCHEMA_NAME!r}"
        )
    if value.get("schema_version") != EVIDENCE_CONTEXT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {EVIDENCE_CONTEXT_SCHEMA_VERSION}"
        )
    digest = value.get(EVIDENCE_CONTEXT_DIGEST_FIELD)
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        errors.append(f"{EVIDENCE_CONTEXT_DIGEST_FIELD} must be lowercase SHA-256")
    else:
        try:
            expected_digest = evidence_context_digest(value)
        except (TypeError, ValueError) as exc:
            errors.append(f"evidence context is not canonical JSON: {exc}")
        else:
            if digest != expected_digest:
                errors.append(f"{EVIDENCE_CONTEXT_DIGEST_FIELD} does not match content")

    subject = value.get("subject")
    if not isinstance(subject, dict):
        errors.append("subject must be an object")
        subject = {}
    elif set(subject) != {"experiment_id", "title", "summary"}:
        errors.append("subject fields must be experiment_id, title, and summary")
    for key in ("experiment_id", "title", "summary"):
        require_string(subject, key, "subject")

    mechanics = value.get("mechanics_authority")
    mechanics_fields = {
        "experiment_id", "authority_role", "process_specialization",
        "quality_eligible", "status", "verdict", "headline", "scope",
        "summary_metrics", "candidate_coverage", "source_artifacts",
    }
    if not isinstance(mechanics, dict):
        errors.append("mechanics_authority must be an object")
        mechanics = {}
    elif set(mechanics) != mechanics_fields:
        missing = sorted(mechanics_fields - set(mechanics))
        extras = sorted(set(mechanics) - mechanics_fields)
        errors.append(
            "mechanics_authority fields are invalid"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unsupported {', '.join(extras)}" if extras else "")
        )
    for key in ("experiment_id", "process_specialization", "headline", "scope"):
        require_string(mechanics, key, "mechanics_authority")
    if mechanics.get("experiment_id") != subject.get("experiment_id"):
        errors.append("mechanics_authority.experiment_id must match subject.experiment_id")
    if mechanics.get("authority_role") != "diagnostic_mechanics":
        errors.append("mechanics_authority.authority_role must be diagnostic_mechanics")
    if mechanics.get("quality_eligible") is not False:
        errors.append("mechanics_authority.quality_eligible must be false")
    if mechanics.get("status") != "valid":
        errors.append("mechanics_authority.status must be valid")
    if mechanics.get("verdict") not in EVIDENCE_CONTEXT_MECHANICS_VERDICTS:
        errors.append("mechanics_authority.verdict is unsupported")
    metrics = mechanics.get("summary_metrics")
    if not isinstance(metrics, list) or not metrics:
        errors.append("mechanics_authority.summary_metrics must be a nonempty list")
    else:
        metric_keys: set[str] = set()
        for index, metric in enumerate(metrics):
            location = f"mechanics_authority.summary_metrics[{index}]"
            if not isinstance(metric, dict):
                errors.append(f"{location} must be an object")
                continue
            expected_fields = {"key", "label", "value", "unit", "status", "description"}
            if set(metric) != expected_fields:
                errors.append(f"{location} fields are invalid")
            for key in ("key", "label", "unit", "description"):
                require_string(metric, key, location)
            key = metric.get("key")
            if isinstance(key, str):
                if key in metric_keys:
                    errors.append(f"summary_metrics contains duplicate key {key!r}")
                metric_keys.add(key)
            if metric.get("status") not in EVIDENCE_CONTEXT_METRIC_STATUSES:
                errors.append(f"{location}.status is unsupported")
            metric_value = metric.get("value")
            if not (
                metric_value is None
                or isinstance(metric_value, (str, bool, int))
                or (isinstance(metric_value, float) and math.isfinite(metric_value))
            ):
                errors.append(f"{location}.value must be a finite JSON scalar or null")
            if metric.get("status") == "unavailable" and metric_value is not None:
                errors.append(f"{location}.value must be null when status is unavailable")
    coverage = mechanics.get("candidate_coverage")
    coverage_by_candidate: dict[str, str] = {}
    if not isinstance(coverage, list):
        errors.append("mechanics_authority.candidate_coverage must be a list")
    else:
        coverage_candidates: set[str] = set()
        for index, row in enumerate(coverage):
            location = f"mechanics_authority.candidate_coverage[{index}]"
            if not isinstance(row, dict) or set(row) != {"candidate", "coverage", "note"}:
                errors.append(f"{location} fields are invalid")
                continue
            require_string(row, "candidate", location)
            if not isinstance(row.get("note"), str):
                errors.append(f"{location}.note must be a string")
            if row.get("coverage") not in EVIDENCE_CONTEXT_MECHANICS_COVERAGE:
                errors.append(f"{location}.coverage is unsupported")
            candidate = row.get("candidate")
            if isinstance(candidate, str):
                if candidate in coverage_candidates:
                    errors.append(f"candidate_coverage contains duplicate {candidate!r}")
                coverage_candidates.add(candidate)
                if isinstance(row.get("coverage"), str):
                    coverage_by_candidate[candidate] = row["coverage"]
    validate_sources(
        mechanics.get("source_artifacts"),
        "mechanics_authority.source_artifacts",
    )

    quality = value.get("quality_authority")
    quality_fields = {
        "experiment_id", "authority_role", "process_specialization",
        "quality_eligible", "status", "headline", "scope",
        "manual_promotion_required", "candidates", "source_artifacts",
    }
    if not isinstance(quality, dict):
        errors.append("quality_authority must be an object")
        quality = {}
    elif set(quality) != quality_fields:
        missing = sorted(quality_fields - set(quality))
        extras = sorted(set(quality) - quality_fields)
        errors.append(
            "quality_authority fields are invalid"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unsupported {', '.join(extras)}" if extras else "")
        )
    for key in ("experiment_id", "process_specialization", "headline", "scope"):
        require_string(quality, key, "quality_authority")
    if quality.get("authority_role") != "production_quality":
        errors.append("quality_authority.authority_role must be production_quality")
    if quality.get("quality_eligible") is not True:
        errors.append("quality_authority.quality_eligible must be true")
    if quality.get("status") != "terminal_valid":
        errors.append("quality_authority.status must be terminal_valid")
    if not isinstance(quality.get("manual_promotion_required"), bool):
        errors.append("quality_authority.manual_promotion_required must be boolean")
    candidates = quality.get("candidates")
    candidate_names: set[str] = set()
    if not isinstance(candidates, list) or not candidates:
        errors.append("quality_authority.candidates must be a nonempty list")
    else:
        expected_fields = set(EVIDENCE_CONTEXT_CANDIDATE_FIELDS)
        for index, row in enumerate(candidates):
            location = f"quality_authority.candidates[{index}]"
            if not isinstance(row, dict):
                errors.append(f"{location} must be an object")
                continue
            if set(row) != expected_fields:
                missing = sorted(expected_fields - set(row))
                extras = sorted(set(row) - expected_fields)
                errors.append(
                    f"{location} fields are invalid"
                    + (f"; missing {', '.join(missing)}" if missing else "")
                    + (f"; unsupported {', '.join(extras)}" if extras else "")
                )
            require_string(row, "candidate", location)
            candidate = row.get("candidate")
            if isinstance(candidate, str):
                if candidate in candidate_names:
                    errors.append(f"quality candidates contains duplicate {candidate!r}")
                candidate_names.add(candidate)
            if row.get("mechanics_coverage") not in EVIDENCE_CONTEXT_MECHANICS_COVERAGE:
                errors.append(f"{location}.mechanics_coverage is unsupported")
            if row.get("strict_accuracy_pass") is not None and not isinstance(
                row.get("strict_accuracy_pass"), bool
            ):
                errors.append(f"{location}.strict_accuracy_pass must be boolean or null")
            if not isinstance(row.get("availability_biased"), bool):
                errors.append(f"{location}.availability_biased must be boolean")
            for key in EVIDENCE_CONTEXT_INTEGER_CANDIDATE_FIELDS:
                item = row.get(key)
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    errors.append(f"{location}.{key} must be a nonnegative integer")
            if isinstance(row.get("accuracy_rank"), int) and row["accuracy_rank"] < 1:
                errors.append(f"{location}.accuracy_rank must be positive")
            for key in EVIDENCE_CONTEXT_NUMERIC_CANDIDATE_FIELDS:
                item = row.get(key)
                if item is not None and (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                ):
                    errors.append(f"{location}.{key} must be finite numeric or null")
    if candidate_names != set(coverage_by_candidate):
        errors.append(
            "mechanics candidate coverage must exactly match quality candidates"
        )
    elif isinstance(candidates, list):
        for row in candidates:
            if not isinstance(row, dict) or not isinstance(row.get("candidate"), str):
                continue
            candidate = row["candidate"]
            if coverage_by_candidate.get(candidate) != row.get("mechanics_coverage"):
                errors.append(
                    f"candidate {candidate!r} has inconsistent mechanics coverage"
                )
    validate_sources(
        quality.get("source_artifacts"),
        "quality_authority.source_artifacts",
    )

    if value.get("separation_contract") != EVIDENCE_CONTEXT_SEPARATION_CONTRACT:
        errors.append("separation_contract does not match the required isolation contract")

    return {
        "schema_name": EVIDENCE_CONTEXT_SCHEMA_NAME,
        "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
        "valid": not errors,
        "errors": errors,
        "candidate_count": len(candidates) if isinstance(candidates, list) else 0,
        "mechanics_metric_count": len(metrics) if isinstance(metrics, list) else 0,
    }


def _optional_boolean(value: Any) -> bool | None:
    if _missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return json_value(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return json_value(parsed) if isinstance(parsed, dict) else {}


def records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []
    return [json_value(row) for row in dataframe.to_dict("records")]


def configured_low_texture_variance_max_by_run(
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Extract explicit per-run variance thresholds without assuming defaults."""

    result: dict[str, float] = {}
    for raw in config.get("runs") or []:
        if not isinstance(raw, dict) or not raw.get("label"):
            continue
        overrides = raw.get("ini_overrides")
        if not isinstance(overrides, dict):
            continue
        value = overrides.get("PatchMatch CUDA Low Texture Variance Max")
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(threshold) or threshold <= 0.0:
            continue
        label = str(raw["label"])
        result[label] = threshold
        result[f"{label}{EVIDENCE_CONTEXT_DEEP_RUN_SUFFIX}"] = threshold
    return result


def build_low_texture_update_hysteresis_metrics(
    exact_iterations: pd.DataFrame,
) -> dict[str, Any]:
    """Derive proposal rejection rates and per-eligible-pixel gain means."""

    output: list[dict[str, Any]] = []

    def finite(value: Any) -> float | None:
        normalized = json_value(value)
        if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
            return None
        return float(normalized) if math.isfinite(float(normalized)) else None

    for raw in records(exact_iterations):
        logical_iteration = _model_integer(raw.get("logical_iteration"))
        if logical_iteration is None or logical_iteration < 0:
            continue
        if not any(finite(raw.get(field)) is not None for field in LOW_TEXTURE_UPDATE_COUNTER_FIELDS):
            continue
        values = {
            field: finite(raw.get(field)) or 0.0
            for field in LOW_TEXTURE_UPDATE_COUNTER_FIELDS
        }
        eligible = values["low_texture_gate_eligible"]
        propagation_total = (
            values["low_texture_propagation_accepted"]
            + values["low_texture_propagation_rejected"]
        )
        refinement_total = (
            values["low_texture_refinement_accepted"]
            + values["low_texture_refinement_rejected"]
        )
        accepted = (
            values["low_texture_propagation_accepted"]
            + values["low_texture_refinement_accepted"]
        )
        rejected = (
            values["low_texture_propagation_rejected"]
            + values["low_texture_refinement_rejected"]
        )
        proposals = accepted + rejected
        identity_fields = (
            "run", "repeat", "scene_id", "image_id", "estimation_stage",
            "geometric_iteration", "pyramid_level", "scale_level", "scale_number",
            "logical_iteration", "stage",
        )
        result = canonical_pyramid_record({
            key: raw.get(key) for key in identity_fields if key in raw
        })
        result.update({
            "logical_iteration": logical_iteration,
            "eligible_pixels": int(eligible),
            "propagation_accepted": int(values["low_texture_propagation_accepted"]),
            "propagation_rejected": int(values["low_texture_propagation_rejected"]),
            "propagation_rejection_rate": (
                values["low_texture_propagation_rejected"] / propagation_total
                if propagation_total else None
            ),
            "refinement_accepted": int(values["low_texture_refinement_accepted"]),
            "refinement_rejected": int(values["low_texture_refinement_rejected"]),
            "refinement_rejection_rate": (
                values["low_texture_refinement_rejected"] / refinement_total
                if refinement_total else None
            ),
            "accepted_proposals": int(accepted),
            "rejected_proposals": int(rejected),
            "proposal_rejection_rate": rejected / proposals if proposals else None,
            "mean_required_gain": (
                values["low_texture_required_gain_sum"] / eligible if eligible else None
            ),
            "mean_best_proposed_gain": (
                values["low_texture_best_proposed_gain_sum"] / eligible if eligible else None
            ),
            "measurement_quality": "exact",
            "measurement_basis": (
                "complete logical-iteration exact hot-kernel pixel-record aggregation; "
                "gain sums divided by eligible pixels"
            ),
        })
        output.append(result)
    return {
        "schema_name": LOW_TEXTURE_UPDATE_HYSTERESIS_SCHEMA_NAME,
        "schema_version": LOW_TEXTURE_UPDATE_HYSTERESIS_SCHEMA_VERSION,
        "proposal_rate_denominator": (
            "legacy-improving gate-controlled proposals; refinement can contribute multiple "
            "sequential proposals per pixel"
        ),
        "gain_mean_denominator": "eligible pixels in the complete logical iteration",
        "rows": output,
    }


def _contained_relative_path(base: Path, raw_path: Any) -> tuple[Path | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path:
        return None, "path is missing"
    relative = Path(raw_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None, "path must be normalized, relative, and contained"
    base = base.resolve()
    candidate = base / relative
    current = base
    for part in relative.parts:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            return None, f"path cannot be inspected: {exc}"
        if stat.S_ISLNK(status.st_mode):
            return None, "symlink components are not allowed"
    return candidate, None


def _safe_path_component(value: Any, name: str) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, f"{name} is missing"
    if value in {".", ".."} or "/" in value or "\\" in value:
        return None, f"{name} is not a safe path component"
    return value, None


def _trace_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _model_integer(value: Any) -> int | None:
    """Normalize integral dataframe/JSON scalars used by report identities."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    numeric = float(value)
    return int(numeric) if math.isfinite(numeric) and numeric.is_integer() else None


def _identity_integer(value: Any, default: int = -1) -> int:
    normalized = _model_integer(value)
    return default if normalized is None else normalized


def canonical_pyramid_level(row: dict[str, Any]) -> int | None:
    """Normalize the report's pyramid level while accepting capture aliases."""

    for key in ("pyramid_level", "scale_level", "scale_number"):
        value = row.get(key)
        if value in (None, ""):
            continue
        level = _model_integer(value)
        if level is not None and level >= 0:
            return level
    return None


def canonical_pyramid_record(row: dict[str, Any]) -> dict[str, Any]:
    """Return a model row with aliases removed and the canonical field present."""

    result = dict(row)
    result["pyramid_level"] = canonical_pyramid_level(result)
    result.pop("scale_level", None)
    result.pop("scale_number", None)
    return result


def candidate_accounting_record(
    row: dict[str, Any], mode: str | None = None,
) -> dict[str, Any]:
    """Normalize legacy accounting absence without manufacturing zero evidence."""

    result = dict(row)
    accounting_mode = str(mode or result.get("candidate_accounting_mode") or "")
    if accounting_mode:
        result["candidate_accounting_mode"] = accounting_mode
    if accounting_mode != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE:
        return result
    unavailable_keys = set(CANDIDATE_ACCOUNTING_METRICS)
    unavailable_keys.update(
        key for key in result
        if key.startswith("accepted_from_")
        or (
            key.startswith("candidate_")
            and key.endswith(("_tested", "_finite", "_accepted", "_acceptance_rate"))
        )
    )
    for key in unavailable_keys:
        result[key] = None
    return result


def _trace_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _trace_array(
    raw: dict[str, Any], key: str, *, integer: bool = False
) -> tuple[list[int | float | None], bool]:
    value = raw.get(key)
    if not isinstance(value, list):
        return [], False
    truncated = len(value) > MAX_TRACE_ARRAY_VALUES
    normalize = _trace_integer if integer else _trace_number
    return [normalize(item) for item in value[:MAX_TRACE_ARRAY_VALUES]], truncated


def _trace_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _trace_command_option(command: list[Any], name: str) -> str | None:
    for index, raw in enumerate(command):
        value = str(raw)
        if value == name:
            return str(command[index + 1]) if index + 1 < len(command) else None
        prefix = f"{name}="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _trace_stage_roots(
    instrumentation_root: Path,
) -> list[tuple[str, int | None, str, Path]]:
    """Return the photometric root and every concrete geometric stage root."""

    stages = [("photometric", None, ".", instrumentation_root)]
    geometric_root = instrumentation_root / "geometric_iterations"
    if not geometric_root.is_dir() or geometric_root.is_symlink():
        return stages
    geometric_stages: list[tuple[int, Path]] = []
    for candidate in geometric_root.iterdir():
        match = re.fullmatch(r"iteration(\d+)", candidate.name)
        if match and candidate.is_dir() and not candidate.is_symlink():
            geometric_stages.append((int(match.group(1)), candidate))
    for geometric_iteration, candidate in sorted(geometric_stages):
        stages.append((
            "geometric_consistency",
            geometric_iteration,
            f"geometric_iterations/{candidate.name}",
            candidate,
        ))
    return stages


def _trace_stage_capture_evidence(
    *,
    stage_root: Path,
    instrumentation_root: Path,
    target_image_id: int | None,
    estimation_stage: str,
    geometric_iteration: int | None,
    relative_root: str,
    command_valid: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    matching_frames = 0
    valid_frames = 0
    manifest_states: set[tuple[int, int]] = set()
    trace_plans: list[dict[str, Any]] = []
    resource_plan_errors: list[str] = []
    plans_path = stage_root / "resource_plans.jsonl"
    if plans_path.is_file() and not plans_path.is_symlink():
        seen_levels: set[int] = set()
        try:
            with plans_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if len(line.encode("utf-8")) > MAX_TRACE_JSONL_LINE_BYTES:
                        resource_plan_errors.append(
                            f"resource_plans.jsonl line {line_number} exceeds the line-size limit"
                        )
                        break
                    plan = json.loads(line)
                    if not isinstance(plan, dict):
                        raise ValueError(f"line {line_number} is not a JSON object")
                    if _trace_integer(plan.get("image_id")) != target_image_id:
                        continue
                    level = canonical_pyramid_level(plan)
                    width = _trace_integer(plan.get("width"))
                    height = _trace_integer(plan.get("height"))
                    num_trace_pixels = _trace_integer(plan.get("num_trace_pixels"))
                    num_logical_states = _trace_integer(plan.get("num_logical_states"))
                    identity_valid = (
                        plan.get("estimation_stage") in (None, estimation_stage)
                        and (
                            "geometric_iteration" not in plan
                            or plan.get("geometric_iteration") == geometric_iteration
                        )
                    )
                    valid = (
                        plan.get("schema_name") == "openmvs.dmap.resource_plan"
                        and _trace_integer(plan.get("schema_version")) == 4
                        and identity_valid
                        and level is not None and level not in seen_levels
                        and width is not None and width > 0
                        and height is not None and height > 0
                        and num_trace_pixels is not None and num_trace_pixels >= 0
                        and num_logical_states is not None and num_logical_states > 0
                        and (
                            (num_trace_pixels > 0
                             and plan.get("trace_requested") is True
                             and plan.get("trace_available") is True)
                            or (num_trace_pixels == 0
                                and plan.get("trace_requested") is False
                                and plan.get("trace_available") is False)
                        )
                    )
                    if not valid:
                        resource_plan_errors.append(
                            f"resource_plans.jsonl line {line_number} has invalid trace topology"
                        )
                        continue
                    seen_levels.add(level)
                    trace_plans.append({
                        "estimation_stage": estimation_stage,
                        "geometric_iteration": geometric_iteration,
                        "pyramid_level": level,
                        "width": width,
                        "height": height,
                        "num_trace_pixels": num_trace_pixels,
                        "num_logical_states": num_logical_states,
                    })
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
            resource_plan_errors.append(f"resource_plans.jsonl is invalid: {exc}")
        if not trace_plans:
            resource_plan_errors.append("no resource plan matches the requested image")

    depthmaps_root = stage_root / "depthmaps"
    summary_paths = (
        sorted(depthmaps_root.glob("*/summary.json"))
        if depthmaps_root.is_dir() and not depthmaps_root.is_symlink()
        else []
    )
    for summary_path in summary_paths:
        try:
            summary_path.resolve().relative_to(instrumentation_root.resolve())
        except ValueError:
            continue
        summary = _trace_json_object(summary_path)
        if _trace_integer(summary.get("image_id")) != target_image_id:
            continue
        matching_frames += 1
        frame_dir = summary_path.parent
        manifest = _trace_json_object(frame_dir / "map_manifest.json")
        marker = _trace_json_object(frame_dir / "capture_complete.json")
        exact_capture = manifest.get("exact_capture") or {}
        manifest_binding = marker.get("map_manifest") or {}
        summary_binding = marker.get("summary") or {}
        summary_completion = summary.get("completion_marker") or {}
        num_iterations = _trace_integer(manifest.get("num_iterations"))
        num_logical_states = _trace_integer(manifest.get("num_logical_states"))
        map_levels = {
            level for level in (
                canonical_pyramid_level(item)
                for item in manifest.get("maps") or []
                if isinstance(item, dict)
            ) if level is not None
        }
        manifest_level = canonical_pyramid_level(manifest)
        if manifest_level is not None:
            map_levels.add(manifest_level)
        if not map_levels:
            map_levels.add(0)
        topology_valid = (
            num_iterations is not None
            and num_iterations >= 0
            and num_logical_states == num_iterations + 1
            and all(level >= 0 for level in map_levels)
        )
        stage_identity_valid = (
            summary.get("estimation_stage") == estimation_stage
            and summary.get("geometric_iteration") == geometric_iteration
            and marker.get("estimation_stage") == estimation_stage
            and marker.get("geometric_iteration") == geometric_iteration
        )
        frame_valid = (
            summary.get("schema_name") == "openmvs.dmap.frame_summary"
            and summary.get("schema_version") == 4
            and manifest.get("schema_name") == "openmvs.dmap.map_manifest"
            and manifest.get("schema_version") == 4
            and manifest.get("complete") is True
            and not (manifest.get("write_errors") or [])
            and exact_capture.get("requested") is True
            and exact_capture.get("available") is True
            and (manifest.get("observer_sidecars") or {}).get("complete") is True
            and marker.get("schema_name") == "openmvs.dmap.capture_complete"
            and marker.get("schema_version") == 1
            and marker.get("capture_kind") == "maps"
            and marker.get("maps_complete") is True
            and marker.get("observer_sidecars_complete") is True
            and marker.get("image_id") == summary.get("image_id")
            and marker.get("image_name") == summary.get("image_name")
            and stage_identity_valid
            and manifest_binding.get("path") == "map_manifest.json"
            and manifest_binding.get("schema_version") == 4
            and manifest_binding.get("complete") is True
            and summary_binding.get("path") == "summary.json"
            and summary_binding.get("schema_version") == 4
            and summary_completion == {
                "schema_name": "openmvs.dmap.capture_complete",
                "schema_version": 1,
                "path": "capture_complete.json",
                "maps_complete": True,
                "eligible": True,
            }
            and topology_valid
        )
        if frame_valid:
            valid_frames += 1
            manifest_states.update(
                (level, iteration)
                for level in map_levels
                for iteration in range(-1, num_iterations)
            )
        else:
            errors.append(
                f"{summary_path.relative_to(instrumentation_root)} lacks valid schema-v4 exact completion evidence"
            )

    plan_declared_states = {
        (plan["pyramid_level"], iteration)
        for plan in trace_plans
        for iteration in range(-1, plan["num_logical_states"] - 1)
    }
    plan_trace_states = {
        state
        for plan in trace_plans
        if plan["num_trace_pixels"] > 0
        for state in (
            (plan["pyramid_level"], iteration)
            for iteration in range(-1, plan["num_logical_states"] - 1)
        )
    }
    manifest_level_zero = {state for state in manifest_states if state[0] == 0}
    plan_level_zero = {state for state in plan_declared_states if state[0] == 0}
    if trace_plans and manifest_level_zero != plan_level_zero:
        resource_plan_errors.append(
            "resource plan and exact map manifest disagree on level-0 logical states"
        )
    # Exact map-manifest requirements are never replaced by a resource plan.
    # Plans add only the coarse states that cannot be represented by level-0 maps.
    expected_states = manifest_states | {
        state for state in plan_trace_states if state[0] > 0
    }
    errors.extend(resource_plan_errors)
    if matching_frames == 0:
        errors.append("no schema-v4 map frame matches the requested image")
    return {
        "valid": (
            command_valid and matching_frames > 0
            and valid_frames == matching_frames and not resource_plan_errors
        ),
        "measurement_basis": "schema_v4_exact_map_completion",
        "estimation_stage": estimation_stage,
        "geometric_iteration": geometric_iteration,
        "relative_root": relative_root,
        "matching_frame_count": matching_frames,
        "valid_frame_count": valid_frames,
        "command_maps_write_maps": command_valid,
        "declared_exact_topology": bool(manifest_states),
        "expected_states": [
            {
                "estimation_stage": estimation_stage,
                "geometric_iteration": geometric_iteration,
                "pyramid_level": level,
                "logical_iteration": iteration,
            }
            for level, iteration in sorted(expected_states)
        ],
        "manifest_states": [
            {"pyramid_level": level, "logical_iteration": iteration}
            for level, iteration in sorted(manifest_states)
        ],
        "trace_plans": sorted(trace_plans, key=lambda plan: plan["pyramid_level"]),
        "errors": errors,
    }


def _exact_trace_capture_evidence(
    run_root: Path, target_image_id: int | None
) -> dict[str, Any]:
    """Attest every estimation stage in a completed Process<true> maps run."""

    repro = _trace_json_object(run_root / "repro.json")
    command = repro.get("command") if isinstance(repro.get("command"), list) else []
    command_valid = (
        repro.get("dry_run") is False
        and _trace_integer(repro.get("return_code")) == 0
        and _trace_command_option(command, "--dmap-instrumentation-level") == "maps"
        and _trace_command_option(command, "--dmap-instrumentation-write-maps") == "1"
    )
    instrumentation_root = run_root / "dmap_instrumentation"
    topology_errors: list[str] = []
    try:
        geometric_iterations = int(
            _trace_command_option(command, "--geometric-iters") or "2"
        )
        fusion_mode = int(_trace_command_option(command, "--fusion-mode") or "0")
        if geometric_iterations < 0:
            raise ValueError("--geometric-iters is negative")
    except ValueError as exc:
        geometric_iterations = 0
        fusion_mode = 0
        topology_errors.append(f"resolved stage topology is malformed: {exc}")
    expected_geometric = (
        set(range(geometric_iterations)) if fusion_mode >= 0 else set()
    )
    geometric_root = instrumentation_root / "geometric_iterations"
    observed_geometric: list[int] = []
    if geometric_root.exists() or geometric_root.is_symlink():
        if geometric_root.is_symlink() or not geometric_root.is_dir():
            topology_errors.append("geometric stage root is not a regular directory")
        else:
            for candidate in geometric_root.iterdir():
                match = re.fullmatch(r"iteration(\d+)", candidate.name)
                if candidate.is_symlink() or not candidate.is_dir():
                    topology_errors.append(
                        f"unexpected geometric stage artifact: {candidate.name}"
                    )
                elif match is None:
                    topology_errors.append(
                        f"unexpected geometric stage directory: {candidate.name}"
                    )
                else:
                    observed_geometric.append(int(match.group(1)))
    if len(observed_geometric) != len(set(observed_geometric)):
        topology_errors.append("duplicate geometric stage index")
    if set(observed_geometric) != expected_geometric:
        topology_errors.append(
            "geometric stage topology does not match the resolved command: "
            f"expected {sorted(expected_geometric)}, observed "
            f"{sorted(set(observed_geometric))}"
        )
    stages = [
        _trace_stage_capture_evidence(
            stage_root=stage_root,
            instrumentation_root=instrumentation_root,
            target_image_id=target_image_id,
            estimation_stage=estimation_stage,
            geometric_iteration=geometric_iteration,
            relative_root=relative_root,
            command_valid=command_valid,
        )
        for estimation_stage, geometric_iteration, relative_root, stage_root
        in _trace_stage_roots(instrumentation_root)
    ]
    if topology_errors and stages:
        stages[0]["valid"] = False
        stages[0]["errors"] = [
            *(stages[0].get("errors") or []), *topology_errors,
        ]
    errors = ([] if command_valid else [
        "repro.json does not attest a successful maps/write-maps=1 run"
    ])
    errors.extend(
        f"{stage['relative_root']}: {error}"
        for stage in stages
        for error in stage.get("errors") or []
    )
    expected_states = [
        state for stage in stages for state in stage.get("expected_states") or []
    ]
    trace_plans = [
        plan for stage in stages for plan in stage.get("trace_plans") or []
    ]
    return {
        "valid": command_valid and bool(stages) and all(
            stage.get("valid") is True for stage in stages
        ),
        "measurement_basis": "schema_v4_exact_map_completion",
        "matching_frame_count": sum(
            int(stage.get("matching_frame_count", 0)) for stage in stages
        ),
        "valid_frame_count": sum(
            int(stage.get("valid_frame_count", 0)) for stage in stages
        ),
        "command_maps_write_maps": command_valid,
        "expected_states": expected_states,
        "trace_plans": trace_plans,
        "stages": stages,
        "errors": errors,
    }


def _trace_source_provenance(
    raw: dict[str, Any], exact_capture_evidence: dict[str, Any]
) -> tuple[str, str, str, str | None]:
    quality = raw.get("source_quality")
    basis = raw.get("measurement_basis")
    if quality is not None or basis is not None:
        if quality == "proxy" and basis == "post_pass_proxy":
            return "proxy", "post_pass_proxy", "trace_row_declaration", None
        if quality == "exact" and basis == "exact_hot_kernel":
            if exact_capture_evidence.get("valid") is True:
                return "exact", "exact_hot_kernel", "trace_row_declaration", None
            return (
                "proxy",
                "exact_hot_kernel_unverified",
                "conservative_downgrade",
                None,
            )
        return "proxy", "invalid_trace_provenance", "conservative_downgrade", (
            "source_quality and measurement_basis must be the exact/hot-kernel or proxy/post-pass pair"
        )
    return "proxy", "post_pass_proxy", "legacy_default", None


def _normalize_completed_trace_row(
    raw: dict[str, Any],
    *,
    run: str,
    scene_id: str,
    estimation_stage: str,
    geometric_iteration: int | None,
    trace_source_path: str,
    exact_capture_evidence: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    image_id = _trace_integer(raw.get("image_id"))
    x = _trace_integer(raw.get("x"))
    y = _trace_integer(raw.get("y"))
    logical_iteration = _trace_integer(
        raw.get("logical_iteration", raw.get("iteration"))
    )
    if image_id is None or image_id < 0:
        return None, "image_id must be a non-negative integer"
    if x is None or x < 0 or y is None or y < 0:
        return None, "x and y must be non-negative integers"
    if logical_iteration is None or logical_iteration < -1:
        return None, "logical_iteration must be an integer greater than or equal to -1"

    arrays: dict[str, Any] = {}
    truncated_arrays: list[str] = []
    for key, integer in (
        ("neighbor_costs", False),
        ("view_costs", False),
        ("view_photometric_costs", False),
        ("view_geometric_costs", False),
        ("view_weights", False),
        ("bad_reasons", True),
    ):
        arrays[key], truncated = _trace_array(raw, key, integer=integer)
        if truncated:
            truncated_arrays.append(key)

    source_quality, measurement_basis, quality_origin, provenance_error = (
        _trace_source_provenance(raw, exact_capture_evidence)
    )
    if provenance_error:
        return None, provenance_error
    row = {
        "run": run,
        "scene_id": scene_id,
        "image_id": image_id,
        "estimation_stage": estimation_stage,
        "geometric_iteration": geometric_iteration,
        "trace_index": _trace_integer(raw.get("trace_index")),
        "label": str(raw.get("label") or ""),
        "x": x,
        "y": y,
        "pyramid_level": canonical_pyramid_level(raw),
        "logical_iteration": logical_iteration,
        "stage": "initialization" if logical_iteration == -1 else "iteration",
        "display_iteration": 0 if logical_iteration == -1 else logical_iteration + 1,
        "source": str(raw.get("source") or "unknown"),
        "source_quality": source_quality,
        "measurement_basis": measurement_basis,
        "source_quality_origin": quality_origin,
        "trace_source_path": trace_source_path,
        "cost": {
            "before": _trace_number(raw.get("cost_before")),
            "after": _trace_number(raw.get("cost_after")),
            "improvement": _trace_number(raw.get("cost_improvement")),
            "photometric_after": _trace_number(raw.get("photometric_cost_after")),
            "photo_prior_after": _trace_number(raw.get("photo_prior_cost_after")),
            "depth_prior_after": _trace_number(raw.get("depth_prior_cost_after")),
            "depth_prior_weight_after": _trace_number(raw.get("depth_prior_weight_after")),
            "geometric_after": _trace_number(raw.get("geometric_cost_after")),
            "reference_variance": _trace_number(raw.get("ref_variance")),
        },
        "depth": {
            "before": _trace_number(raw.get("depth_before")),
            "after": _trace_number(raw.get("depth_after")),
            "absolute_change": _trace_number(raw.get("depth_abs_change")),
            "relative_change": _trace_number(raw.get("depth_rel_change")),
            "low_resolution_prior": _trace_number(raw.get("low_depth")),
        },
        "normal": {
            "angle_change_degrees": _trace_number(raw.get("normal_angle_change")),
        },
        "view": {
            "selected_count": _trace_integer(raw.get("selected_view_count")),
            "selected_mask": _trace_integer(raw.get("selected_views_mask")),
            "selected_before_mask": _trace_integer(raw.get("selected_views_before_mask")),
            "entropy": _trace_number(raw.get("view_entropy")),
        },
        "arrays": arrays,
        "truncated_arrays": truncated_arrays,
    }
    return row, None


def _completed_trace_payload(
    *,
    entry: dict[str, Any],
    capture_root: Path,
    executions: list[dict[str, Any]],
    request: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    truncated = False
    target_image_id = _trace_integer(entry.get("image_id"))
    requested_pixels = dmap_drilldown.expand_trace_pixels(request)
    requested_coordinates = [
        (int(pixel["x"]), int(pixel["y"])) for pixel in requested_pixels
    ]

    def closure_record(validation: integrity.ClosureValidation) -> dict[str, Any]:
        return {
            "valid": validation.valid,
            "status": validation.status,
            "required": validation.required,
            "reason": validation.reason,
            "source_path": relative_path(validation.manifest_path, output_dir),
            "file_count": validation.file_count,
            "total_bytes": validation.total_bytes,
            "files_sha256": validation.files_sha256,
        }

    for execution_index, execution in enumerate(executions):
        run, run_error = _safe_path_component(execution.get("run"), "run")
        scene_id, scene_error = _safe_path_component(execution.get("scene_id"), "scene_id")
        if run_error or scene_error:
            message = "; ".join(value for value in (run_error, scene_error) if value)
            errors.append({
                "kind": "unsafe_execution_identity",
                "execution_index": execution_index,
                "message": message,
            })
            sources.append({
                "run": run,
                "scene_id": scene_id,
                "image_id": target_image_id,
                "estimation_stage": None,
                "geometric_iteration": None,
                "stage_key": None,
                "source_path": None,
                "contained": False,
                "available": False,
                "row_count": 0,
                "truncated": False,
                "error": message,
                "source_quality_counts": {"exact": 0, "proxy": 0},
                "exact_capture_evidence": {},
                "coverage": {},
            })
            continue
        run_root = (capture_root / "runs" / run / scene_id).resolve()
        execution_row_start = len(rows)
        execution_source_start = len(sources)
        closure_before = integrity.validate_capture_artifact_closure(
            run_root, "trace", required=False
        )
        closure_evidence = closure_record(closure_before)
        exact_capture = _exact_trace_capture_evidence(run_root, target_image_id)
        if closure_before.status != "verified":
            closure_reason = (
                "trace capture artifact closure is not verified: "
                f"{closure_before.reason}"
            )
            for stage in exact_capture.get("stages") or []:
                stage["valid"] = False
                stage["errors"] = [
                    *(stage.get("errors") or []), closure_reason,
                ]
        for stage_evidence in exact_capture.get("stages") or []:
            estimation_stage = str(stage_evidence.get("estimation_stage") or "photometric")
            geometric_iteration = _trace_integer(stage_evidence.get("geometric_iteration"))
            relative_root = str(stage_evidence.get("relative_root") or ".")
            stage_key = (
                f"geometric_consistency:{geometric_iteration}"
                if estimation_stage == "geometric_consistency"
                else "photometric"
            )
            stage_root = (
                run_root / "dmap_instrumentation"
                if relative_root == "."
                else run_root / "dmap_instrumentation" / relative_root
            )
            trace_path = (stage_root / "instrumentation" / "traces.jsonl").resolve()
            source: dict[str, Any] = {
                "run": run,
                "scene_id": scene_id,
                "image_id": target_image_id,
                "estimation_stage": estimation_stage,
                "geometric_iteration": geometric_iteration,
                "stage_key": stage_key,
                "source_path": None,
                "contained": False,
                "available": False,
                "row_count": 0,
                "truncated": False,
                "error": None,
                "source_quality_counts": {"exact": 0, "proxy": 0},
                "exact_capture_evidence": stage_evidence,
                "artifact_closure": closure_evidence,
                "coverage": {},
            }

            def add_source_error(kind: str, message: str, **details: Any) -> None:
                source["error"] = source.get("error") or message
                errors.append({
                    "kind": kind,
                    "execution_index": execution_index,
                    "source_path": source.get("source_path"),
                    "estimation_stage": estimation_stage,
                    "geometric_iteration": geometric_iteration,
                    "message": message,
                    **details,
                })

            try:
                trace_path.relative_to(capture_root.resolve())
            except ValueError:
                add_source_error("unsafe_trace_path", "trace source escapes its capture root")
                sources.append(source)
                continue
            source_path = relative_path(trace_path, output_dir)
            source["source_path"] = source_path
            source["contained"] = True
            if closure_before.status == "invalid":
                add_source_error(
                    "trace_artifact_closure_error",
                    "trace capture artifact closure is invalid: "
                    f"{closure_before.reason}",
                )
                sources.append(source)
                continue
            if (
                closure_before.status != "verified"
                and stage_evidence.get("declared_exact_topology") is True
            ):
                add_source_error(
                    "trace_artifact_closure_error",
                    "exact trace artifacts require a verified capture artifact closure",
                )
                sources.append(source)
                continue
            if (
                stage_evidence.get("declared_exact_topology") is True
                and stage_evidence.get("valid") is not True
            ):
                add_source_error(
                    "trace_topology_evidence_error",
                    "declared exact trace topology is invalid: "
                    + "; ".join(stage_evidence.get("errors") or ["unknown error"]),
                )

            trace_plans = {
                plan["pyramid_level"]: plan
                for plan in stage_evidence.get("trace_plans") or []
            }
            trace_layouts: dict[int, list[dmap_drilldown.TracePyramidSlot]] = {}

            def trace_layout(
                pyramid_level: int | None,
            ) -> list[dmap_drilldown.TracePyramidSlot]:
                if pyramid_level is None or pyramid_level < 0:
                    return []
                if pyramid_level not in trace_layouts:
                    plan = trace_plans.get(pyramid_level)
                    trace_layouts[pyramid_level] = dmap_drilldown.trace_pyramid_layout(
                        requested_coordinates,
                        pyramid_level,
                        width=plan.get("width") if plan else None,
                        height=plan.get("height") if plan else None,
                    )
                return trace_layouts[pyramid_level]

            for level, plan in trace_plans.items():
                resolved_count = len(trace_layout(level))
                if resolved_count != plan["num_trace_pixels"]:
                    add_source_error(
                        "trace_resource_plan_identity_error",
                        f"resource plan level {level} declares {plan['num_trace_pixels']} "
                        f"trace pixels but the immutable request selects {resolved_count}",
                    )
            if not trace_path.is_file() or trace_path.is_symlink():
                add_source_error("trace_unavailable", "trace source is unavailable")
                sources.append(source)
                continue
            source["available"] = True
            source_row_start = len(rows)
            declared_states = {
                (state["pyramid_level"], state["logical_iteration"])
                for state in stage_evidence.get("expected_states") or []
            }
            try:
                with trace_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if len(rows) >= MAX_COMPLETED_TRACE_ROWS:
                            truncated = True
                            source["truncated"] = True
                            break
                        if len(line.encode("utf-8")) > MAX_TRACE_JSONL_LINE_BYTES:
                            add_source_error(
                                "trace_line_too_large",
                                f"line {line_number} exceeds the JSONL line-size limit",
                                line=line_number,
                            )
                            break
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError as exc:
                            add_source_error(
                                "trace_parse_error", f"line {line_number}: {exc}",
                                line=line_number,
                            )
                            break
                        if not isinstance(raw, dict):
                            add_source_error(
                                "trace_parse_error",
                                f"line {line_number}: expected a JSON object",
                                line=line_number,
                            )
                            break
                        normalized, normalize_error = _normalize_completed_trace_row(
                            raw,
                            run=run,
                            scene_id=scene_id,
                            estimation_stage=estimation_stage,
                            geometric_iteration=geometric_iteration,
                            trace_source_path=source_path,
                            exact_capture_evidence=stage_evidence,
                        )
                        if normalize_error:
                            add_source_error(
                                "trace_row_error",
                                f"line {line_number}: {normalize_error}",
                                line=line_number,
                            )
                            break
                        if target_image_id is not None and normalized["image_id"] != target_image_id:
                            add_source_error(
                                "trace_identity_error",
                                f"line {line_number}: image_id {normalized['image_id']} does not "
                                f"match requested image {target_image_id}",
                                line=line_number,
                            )
                            break
                        row_state = (
                            normalized.get("pyramid_level"),
                            normalized["logical_iteration"],
                        )
                        if declared_states and row_state not in declared_states:
                            add_source_error(
                                "trace_undeclared_state_error",
                                f"line {line_number}: trace row declares state {row_state} "
                                "outside the exact manifest/resource-plan topology",
                                line=line_number,
                            )
                            break
                        row_trace_index = normalized.get("trace_index")
                        level_layout = trace_layout(normalized.get("pyramid_level"))
                        if (
                            row_trace_index is None
                            or row_trace_index < 0
                            or row_trace_index >= len(level_layout)
                            or level_layout[row_trace_index].coordinate
                            != (normalized["x"], normalized["y"])
                        ):
                            add_source_error(
                                "trace_request_identity_error",
                                f"line {line_number}: trace_index does not identify the immutable "
                                "requested pixel at this pyramid level",
                                line=line_number,
                            )
                            break
                        slot = level_layout[row_trace_index]
                        normalized["request_identity"] = {
                            "request_index": slot.request_indices[0],
                            "x": slot.requested_coordinates[0][0],
                            "y": slot.requested_coordinates[0][1],
                            "alias_request_indices": list(slot.request_indices),
                            "alias_coordinates": [
                                {"x": coordinate[0], "y": coordinate[1]}
                                for coordinate in slot.requested_coordinates
                            ],
                            "trace_x": slot.coordinate[0],
                            "trace_y": slot.coordinate[1],
                        }
                        rows.append(normalized)
                        source["row_count"] += 1
                        source["source_quality_counts"][normalized["source_quality"]] += 1
            except (OSError, UnicodeError) as exc:
                add_source_error("trace_read_error", str(exc))

            source_rows = rows[source_row_start:]
            observed_request_indices = {
                request_index
                for row in source_rows
                for request_index in trace_layout(row.get("pyramid_level"))[
                    int(row["trace_index"])
                ].request_indices
            }
            missing_pixels = {
                coordinate
                for request_index, coordinate in enumerate(requested_coordinates)
                if request_index not in observed_request_indices
            }
            missing_states: list[dict[str, Any]] = []
            if declared_states:
                observed = {
                    (row.get("pyramid_level"), row["trace_index"], row["logical_iteration"])
                    for row in source_rows
                }
                missing_states = [
                    {
                        "estimation_stage": estimation_stage,
                        "geometric_iteration": geometric_iteration,
                        "x": coordinate[0],
                        "y": coordinate[1],
                        "trace_x": slot.coordinate[0],
                        "trace_y": slot.coordinate[1],
                        "trace_index": trace_index,
                        "pyramid_level": level,
                        "logical_iteration": iteration,
                    }
                    for level, iteration in sorted(declared_states)
                    for trace_index, slot in enumerate(trace_layout(level))
                    for coordinate in slot.requested_coordinates[:1]
                    if (level, trace_index, iteration) not in observed
                ]
            source["coverage"] = {
                "requested_pixel_count": len(requested_coordinates),
                "observed_pixel_count": len(observed_request_indices),
                "expected_state_count_per_pixel": len(declared_states),
                "missing_pixel_count": len(missing_pixels),
                "unexpected_pixel_count": 0,
                "missing_state_count": len(missing_states),
            }
            if missing_pixels or missing_states:
                add_source_error(
                    "trace_coverage_error",
                    "trace coverage does not match the immutable request: "
                    f"missing_pixels={len(missing_pixels)}, unexpected_pixels=0, "
                    f"missing_states={len(missing_states)}",
                    missing_states=missing_states[:64],
                )
            state_counts: dict[tuple[int | None, int, int], int] = {}
            for row in source_rows:
                state = (
                    row.get("pyramid_level"), int(row["trace_index"]),
                    int(row["logical_iteration"]),
                )
                state_counts[state] = state_counts.get(state, 0) + 1
            duplicate_states = [state for state, count in state_counts.items() if count > 1]
            if duplicate_states:
                add_source_error(
                    "trace_duplicate_state_error",
                    f"trace contains {len(duplicate_states)} duplicate logical states",
                )
            sources.append(source)
        if closure_before.status == "verified":
            closure_after = integrity.validate_capture_artifact_closure(
                run_root, "trace", required=True
            )
            closure_stable = (
                closure_after.valid
                and closure_after.status == "verified"
                and closure_after.files_sha256 == closure_before.files_sha256
                and closure_after.file_count == closure_before.file_count
                and closure_after.total_bytes == closure_before.total_bytes
            )
            if not closure_stable:
                del rows[execution_row_start:]
                message = (
                    "trace capture artifact closure changed or became invalid while "
                    f"building the report: {closure_after.reason}"
                )
                for source in sources[execution_source_start:]:
                    source["available"] = False
                    source["row_count"] = 0
                    source["source_quality_counts"] = {"exact": 0, "proxy": 0}
                    source["coverage"] = {}
                    source["error"] = source.get("error") or message
                errors.append({
                    "kind": "trace_artifact_closure_changed",
                    "execution_index": execution_index,
                    "message": message,
                })
    rows.sort(key=lambda row: (
        row["run"], row["scene_id"], row["image_id"],
        0 if row.get("estimation_stage") == "photometric" else 1,
        row.get("geometric_iteration") if row.get("geometric_iteration") is not None else -1,
        row.get("pyramid_level") if row.get("pyramid_level") is not None else -1,
        row["y"], row["x"], row.get("trace_index", -1),
        row["logical_iteration"],
    ))
    unavailable_reason = None
    if not rows:
        unavailable_reason = (
            errors[0]["message"] if errors else "completed capture contains no trace rows"
        )
    quality_counts = {"exact": 0, "proxy": 0}
    for row in rows:
        quality_counts[row["source_quality"]] += 1
    return {
        "schema_name": COMPLETED_TRACE_SCHEMA_NAME,
        "schema_version": COMPLETED_TRACE_SCHEMA_VERSION,
        "available": bool(rows),
        "complete": bool(rows) and not errors and not truncated,
        "row_limit": MAX_COMPLETED_TRACE_ROWS,
        "row_count": len(rows),
        "source_count": len(sources),
        "source_quality_counts": quality_counts,
        "all_source_attribution_exact": bool(rows) and quality_counts["proxy"] == 0,
        "truncated": truncated,
        "unavailable_reason": unavailable_reason,
        "sources": sources,
        "rows": rows,
        "errors": errors,
    }


def load_drilldown_index(experiment_root: Path, output_dir: Path) -> dict[str, Any]:
    path = experiment_root / "drilldowns" / "index.json"
    if not path.is_file():
        return {
            "schema_name": "openmvs.dmap.drilldown_index",
            "schema_version": 1,
            "available": False,
            "entries": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_name": "openmvs.dmap.drilldown_index",
            "schema_version": 1,
            "available": False,
            "entries": [],
            "error": str(exc),
        }
    if not isinstance(value, dict):
        return {
            "schema_name": "openmvs.dmap.drilldown_index",
            "schema_version": 1,
            "available": False,
            "entries": [],
            "error": "drilldown index must be a JSON object",
        }
    entries = []
    for raw in value.get("entries") or []:
        if not isinstance(raw, dict):
            entries.append({
                "status": "invalid_request",
                "error": "drilldown index entry must be a JSON object",
            })
            continue
        row = dict(raw)
        path_errors: list[dict[str, Any]] = []
        resolved_paths: dict[str, Path] = {}
        for key in ("request", "capture", "executions"):
            if row.get(key):
                resolved, error = _contained_relative_path(experiment_root, row[key])
                if error:
                    path_errors.append({"kind": "unsafe_index_path", "field": key, "message": error})
                    row[key] = None
                else:
                    assert resolved is not None
                    resolved_paths[key] = resolved
                    row[key] = relative_path(resolved, output_dir)
        row["embedding_errors"] = path_errors
        if row.get("status") == "complete":
            capture_root = resolved_paths.get("capture")
            executions_path = resolved_paths.get("executions")
            request_path = resolved_paths.get("request")
            request_sha256 = str(row.get("request_sha256") or "")
            request_value: dict[str, Any] = {}
            request_metadata: dict[str, Any] = {
                "available": False,
                "source_path": row.get("request"),
                "request_sha256": None,
                "error": None,
            }
            if request_path is None or not request_path.is_file():
                request_metadata["error"] = "completed entry has no safe immutable request"
            else:
                try:
                    request_value = dmap_drilldown.load_request(request_path)
                except (OSError, UnicodeError, ValueError) as exc:
                    request_metadata["error"] = str(exc)
                else:
                    target = request_value.get("target") or {}
                    expected_labels = [
                        str(item.get("label"))
                        for item in request_value.get("runs") or []
                        if isinstance(item, dict)
                    ]
                    entry_labels = [str(value) for value in row.get("run_labels") or []]
                    identity_valid = (
                        request_value.get("request_sha256") == request_sha256
                        and request_value.get("capture_profile") == row.get("capture_profile")
                        and target.get("scene_id") == row.get("scene_id")
                        and _trace_integer(target.get("image_id")) == _trace_integer(row.get("image_id"))
                        and _trace_integer(target.get("trace_pixel_count"))
                        == _trace_integer(row.get("trace_pixel_count"))
                        and expected_labels == entry_labels
                    )
                    if not identity_valid:
                        request_metadata["error"] = (
                            "immutable request identity does not match the drilldown index entry"
                        )
                    else:
                        request_metadata.update({
                            "available": True,
                            "request_sha256": request_sha256,
                            "capture_profile": request_value.get("capture_profile"),
                            "scene_id": target.get("scene_id"),
                            "image_id": target.get("image_id"),
                            "trace_pixel_count": target.get("trace_pixel_count"),
                            "run_labels": expected_labels,
                            "pixels": dmap_drilldown.expand_trace_pixels(request_value),
                        })
            if request_metadata["error"]:
                path_errors.append({
                    "kind": "request_metadata_error",
                    "message": request_metadata["error"],
                })
            row["request_metadata"] = request_metadata
            capture_base = (experiment_root / "drilldowns" / "captures").resolve()
            if capture_root is None:
                path_errors.append({"kind": "capture_unavailable", "message": "completed entry has no safe capture root"})
            else:
                try:
                    capture_root.relative_to(capture_base)
                except ValueError:
                    path_errors.append({"kind": "unsafe_capture_path", "message": "capture root is outside drilldowns/captures"})
                    capture_root = None
                if capture_root is not None and request_sha256 and capture_root.name != request_sha256:
                    path_errors.append({"kind": "capture_identity_error", "message": "capture directory does not match request_sha256"})
                    capture_root = None
                if capture_root is not None and request_metadata["available"]:
                    captured_request, capture_request_error = _contained_relative_path(
                        capture_root, "request.yaml"
                    )
                    if (
                        capture_request_error
                        or captured_request is None
                        or not captured_request.is_file()
                        or request_path is None
                        or captured_request.read_bytes() != request_path.read_bytes()
                    ):
                        path_errors.append({
                            "kind": "capture_request_binding_error",
                            "message": "capture request is missing, unsafe, or differs from the immutable request",
                        })
                        capture_root = None
            execution_metadata: dict[str, Any] = {
                "available": False,
                "source_path": row.get("executions"),
                "schema_name": None,
                "schema_version": None,
                "request_sha256": None,
                "execution_count": 0,
                "executions": [],
                "error": None,
            }
            execution_rows: list[dict[str, Any]] = []
            if capture_root is not None and executions_path is not None:
                try:
                    executions_path.relative_to(capture_root)
                except ValueError:
                    execution_metadata["error"] = "executions metadata is outside its capture root"
                if execution_metadata["error"] is None and not executions_path.is_file():
                    execution_metadata["error"] = "executions metadata is unavailable"
                if execution_metadata["error"] is None:
                    try:
                        execution_value = json.loads(executions_path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError) as exc:
                        execution_metadata["error"] = str(exc)
                    else:
                        if not isinstance(execution_value, dict):
                            execution_metadata["error"] = "executions metadata must be a JSON object"
                        elif (
                            execution_value.get("schema_name") != "openmvs.dmap.drilldown_executions"
                            or execution_value.get("schema_version") != 1
                            or execution_value.get("request_sha256") != request_sha256
                            or not isinstance(execution_value.get("executions"), list)
                            or not all(isinstance(item, dict) for item in execution_value.get("executions") or [])
                        ):
                            execution_metadata["error"] = "executions metadata contract is invalid"
                        else:
                            raw_executions = execution_value["executions"]
                            for execution in raw_executions:
                                normalized_execution = {
                                    key: json_value(execution.get(key))
                                    for key in (
                                        "run", "scene_id", "capture_profile", "return_code",
                                        "reused", "dry_run", "validation", "elapsed_seconds", "duration_seconds",
                                        "wall_time_seconds", "command_sha256",
                                    )
                                    if key in execution
                                }
                                execution_rows.append(normalized_execution)
                            requested_labels = request_metadata.get("run_labels") or []
                            observed_labels = [
                                str(item.get("run")) for item in execution_rows
                            ]
                            execution_identity_valid = (
                                request_metadata.get("available") is True
                                and len(observed_labels) == len(set(observed_labels))
                                and set(observed_labels) == set(requested_labels)
                                and all(
                                    item.get("scene_id") == request_metadata.get("scene_id")
                                    and item.get("capture_profile")
                                    == request_metadata.get("capture_profile")
                                    and _trace_integer(item.get("return_code")) == 0
                                    for item in execution_rows
                                )
                            )
                            if not execution_identity_valid:
                                execution_metadata["error"] = (
                                    "executions do not exactly cover the immutable requested runs"
                                )
                            else:
                                execution_metadata.update({
                                    "available": True,
                                    "schema_name": execution_value["schema_name"],
                                    "schema_version": execution_value["schema_version"],
                                    "request_sha256": execution_value["request_sha256"],
                                    "execution_count": len(execution_rows),
                                    "executions": execution_rows,
                                })
            elif execution_metadata["error"] is None:
                execution_metadata["error"] = "completed entry has no safe executions metadata"
            if execution_metadata["error"]:
                path_errors.append({
                    "kind": "execution_metadata_error",
                    "message": execution_metadata["error"],
                })
            row["execution_metadata"] = execution_metadata
            if row.get("capture_profile") == "trace" and capture_root is not None and execution_metadata["available"]:
                row["trace_data"] = _completed_trace_payload(
                    entry=row,
                    capture_root=capture_root,
                    executions=execution_rows,
                    request=request_value,
                    output_dir=output_dir,
                )
                path_errors.extend(row["trace_data"]["errors"])
            else:
                reason = (
                    f"capture profile {row.get('capture_profile')} does not produce targeted traces"
                    if row.get("capture_profile") != "trace"
                    else "trace data unavailable because capture metadata is invalid"
                )
                row["trace_data"] = {
                    "schema_name": COMPLETED_TRACE_SCHEMA_NAME,
                    "schema_version": COMPLETED_TRACE_SCHEMA_VERSION,
                    "available": False,
                    "complete": row.get("capture_profile") != "trace" and not path_errors,
                    "row_limit": MAX_COMPLETED_TRACE_ROWS,
                    "row_count": 0,
                    "source_count": 0,
                    "source_quality_counts": {"exact": 0, "proxy": 0},
                    "all_source_attribution_exact": False,
                    "truncated": False,
                    "unavailable_reason": reason,
                    "sources": [],
                    "rows": [],
                    "errors": [],
                }
        entries.append(row)
    return {
        "schema_name": value.get("schema_name", "openmvs.dmap.drilldown_index"),
        "schema_version": value.get("schema_version", 1),
        "available": True,
        "index_path": relative_path(path, output_dir),
        "entries": entries,
    }


def render_investigation_guide_markdown() -> str:
    """Render the versioned investigation guide for the canonical report."""
    guide = INVESTIGATION_GUIDE
    quick_start = guide["quick_start"]
    lines = [
        INVESTIGATION_GUIDE_HEADING,
        "",
        str(guide["introduction"]),
        "",
        f"### {quick_start['title']}",
        "",
        str(quick_start["summary"]),
        "",
    ]
    lines.extend(f"{index}. {step}" for index, step in enumerate(quick_start["steps"], start=1))
    for recipe in guide["recipes"]:
        lines.extend(["", f"### {recipe['title']}", "", str(recipe["summary"]), "", "**Workflow**", ""])
        lines.extend(f"{index}. {step}" for index, step in enumerate(recipe["steps"], start=1))
        lines.extend(["", "**Look for**", ""])
        lines.extend(f"- {item}" for item in recipe["look_for"])
        lines.extend(["", "**Interpretation cautions**", ""])
        lines.extend(f"- {item}" for item in recipe["cautions"])
    return "\n".join(lines)


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:44] or "item"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{slug}-{digest}"


def relative_path(path: Path | str | None, output_dir: Path) -> str | None:
    if not path:
        return None
    source = Path(str(path)).expanduser()
    try:
        return Path(os.path.relpath(source.resolve(), output_dir.resolve())).as_posix()
    except (OSError, ValueError):
        return source.resolve().as_uri()


def materialize_reference_thumbnail(
    source_value: Path | str | None,
    output_dir: Path,
    scene_id: str,
    image_id: int,
    *,
    allowed_roots: Iterable[Path] = (),
) -> str | None:
    """Create a report-owned browser image instead of linking into a run tree."""

    if not source_value:
        return None
    roots = [Path(root).expanduser().resolve() for root in allowed_roots]
    if not roots:
        return None
    raw_source = Path(str(source_value)).expanduser()
    candidates = [raw_source] if raw_source.is_absolute() else [root / raw_source for root in roots]
    source: Path | None = None
    for candidate in candidates:
        try:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve(strict=True)
            metadata = resolved.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            if not any(resolved.is_relative_to(root) for root in roots):
                continue
        except (OSError, ValueError):
            continue
        source = resolved
        break
    if source is None:
        return None
    destination_root = output_dir / "interactive" / "reference_images"
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / f"{stable_id('reference', scene_id, image_id)}.jpg"
    try:
        from PIL import Image, ImageOps

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        with Image.open(source) as image:
            thumbnail = ImageOps.exif_transpose(image).convert("RGB")
            thumbnail.thumbnail((1024, 1024), resampling)
            thumbnail.save(destination, format="JPEG", quality=90, optimize=True)
    except (OSError, ValueError):
        return None
    return relative_path(destination, output_dir)


def reference_source_roots(
    config: Mapping[str, Any], experiment_root: Path, frame_rows: Iterable[Mapping[str, Any]]
) -> list[Path]:
    """Collect explicitly owned roots that may supply a report thumbnail."""

    roots = [experiment_root.resolve()]
    config_path = Path(str(config.get("_config_path") or Path.cwd() / "experiment.yaml"))
    config_parent = config_path.expanduser().resolve().parent
    for scene in config.get("scenes") or []:
        if not isinstance(scene, Mapping) or not scene.get("working_folder"):
            continue
        root = Path(str(scene["working_folder"])).expanduser()
        roots.append((config_parent / root).resolve() if not root.is_absolute() else root.resolve())
    for row in frame_rows:
        raw_depthmap_dir = row.get("depthmap_dir")
        if not raw_depthmap_dir:
            continue
        depthmap_dir = Path(str(raw_depthmap_dir)).expanduser().resolve()
        instrumentation_root = depthmap_dir.parent.parent
        capture_root = instrumentation_root.parent
        roots.extend((instrumentation_root, capture_root, capture_root / "work"))
    return list(dict.fromkeys(roots))


REPORT_PATH_KEYS = frozenset({
    "path", "source_path", "script_path", "markdown", "static_html",
    "investigation_html", "model", "root", "source_config", "reference_dmap",
    "visual_overlay_svg", "visual_residual_histogram_svg", "csv", "parquet",
    "json", "source_image_name", "candidate_image_name", "reference_image_name",
    "trace_source_path", "request", "capture", "executions",
    "published_output_dir",
})


def paths_report_relative(value: Any, output_dir: Path) -> Any:
    """Return a JSON value with every declared nested path made report-relative."""

    normalized = json_value(value)

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if key in REPORT_PATH_KEYS and isinstance(child, str) and (
                    Path(child).is_absolute() or child.startswith("file://")
                ):
                    source = child.removeprefix("file://")
                    result[key] = relative_path(source, output_dir)
                else:
                    result[key] = visit(child)
            return result
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(normalized)


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = handle.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"invalid PFM header: {path}")
        dimensions = handle.readline().decode("ascii").strip()
        while dimensions.startswith("#"):
            dimensions = handle.readline().decode("ascii").strip()
        width, height = [int(value) for value in dimensions.split()]
        scale = float(handle.readline().decode("ascii").strip())
        values = np.fromfile(handle, dtype="<f4" if scale < 0 else ">f4")
    pixels = width * height
    if pixels <= 0 or values.size % pixels:
        raise ValueError(f"invalid PFM payload size: {path}")
    channels = values.size // pixels
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.flipud(values.reshape(shape)).astype(np.float32, copy=False)


def read_diagnostic_map(path: Path) -> np.ndarray:
    """Read a scalar/vector map supported by the report's region analyzer."""

    if path.suffix.lower() == ".pfm":
        return read_pfm(path)
    from PIL import Image

    return np.asarray(Image.open(path))


def _valid_mask(signal: str, data: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(data), axis=2) if data.ndim == 3 else np.isfinite(data)
    lowered = signal.lower()
    descriptor = component_registry.descriptor_from_row({"signal": signal})
    if data.ndim == 3:
        return finite
    signed = descriptor.signed or "normal" in lowered
    if descriptor.quantity == "depth" or lowered.startswith("depth_final") or lowered == "depth":
        return finite & (data > 0)
    if descriptor.domain in {"positive", "nonnegative", "zero_to_one", "boolean"} and not signed:
        return finite & (data >= 0)
    return finite


def _scale_sample(signal: str, data: np.ndarray) -> np.ndarray:
    if data.ndim == 3:
        values = data[np.all(np.isfinite(data), axis=2)].reshape(-1)
    else:
        values = data[_valid_mask(signal, data)].reshape(-1)
    if values.size > MAX_SCALE_SAMPLES_PER_MAP:
        stride = max(1, values.size // MAX_SCALE_SAMPLES_PER_MAP)
        values = values[::stride][:MAX_SCALE_SAMPLES_PER_MAP]
    return values.astype(np.float64, copy=False)


def _local_scale(signal: str, data: np.ndarray) -> tuple[float, float]:
    lowered = signal.lower()
    descriptor = component_registry.descriptor_from_row({"signal": signal})
    if data.ndim == 3 and descriptor.quantity == "normal":
        return -1.0, 1.0
    sample = _scale_sample(signal, data)
    if sample.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(sample, [1.0, 99.0])
    signed = descriptor.signed
    if signed:
        magnitude = max(abs(float(low)), abs(float(high)), 1e-6)
        return -magnitude, magnitude
    if descriptor.domain == "zero_to_one":
        low, high = max(0.0, float(low)), min(1.0, float(high))
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        center = float(np.median(sample)) if sample.size else 0.0
        width = max(abs(center) * 0.01, 1e-6)
        return center - width, center + width
    return float(low), float(high)


def _colormap(signal: str) -> str:
    descriptor = component_registry.descriptor_from_row({"signal": signal})
    # Matplotlib does not expose the categorical name used by the browser
    # descriptor; retain a deterministic continuous fallback for previews.
    return "tab20" if descriptor.colormap == "tab20" else descriptor.colormap


def _category_legend(signal: str) -> dict[str, str]:
    descriptor = component_registry.descriptor_from_row({"signal": signal})
    if signal in BUILTIN_CATEGORY_LEGENDS:
        return dict(BUILTIN_CATEGORY_LEGENDS[signal])
    if descriptor.domain == "boolean":
        return {"0": "false", "1": "true"}
    return {}


def _write_preview(
    path: Path,
    signal: str,
    data: np.ndarray,
    limits: tuple[float, float],
) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = component_registry.descriptor_from_row({"signal": signal})
    if descriptor.measurement_kind == "enum" and data.ndim == 2:
        from matplotlib import colormaps

        finite = np.isfinite(data)
        codes = np.zeros(data.shape, dtype=np.int64)
        codes[finite] = np.rint(data[finite]).astype(np.int64)
        palette = (colormaps["tab20"](np.arange(20) / 19.0) * 255.0).astype(np.uint8)
        rgba = palette[np.mod(codes, len(palette))]
        rgba[..., 3] = (finite * 255).astype(np.uint8)
    elif data.ndim == 3 and data.shape[2] >= 3 and "normal" in signal.lower():
        rgb = np.clip((data[..., :3] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        alpha = (np.all(np.isfinite(data[..., :3]), axis=2) * 255).astype(np.uint8)
        rgba = np.dstack((rgb, alpha))
    else:
        from matplotlib import colormaps

        scalar = data[..., 0] if data.ndim == 3 else data
        low, high = limits
        normalized = np.clip((scalar - low) / max(high - low, 1e-12), 0.0, 1.0)
        rgba = (colormaps[_colormap(signal)](normalized) * 255.0).astype(np.uint8)
        rgba[..., 3] = (_valid_mask(signal, scalar) * 255).astype(np.uint8)
    image = Image.fromarray(rgba, mode="RGBA")
    if max(image.size) > PREVIEW_MAX_DIMENSION:
        scale = PREVIEW_MAX_DIMENSION / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            resample=(
                Image.Resampling.NEAREST
                if descriptor.measurement_kind == "enum"
                else Image.Resampling.LANCZOS
            ),
        )
    image.save(path, format="PNG", compress_level=6)


def _write_pixel_encoding(path: Path, data: np.ndarray, payload_id: str | None = None) -> dict[str, Any]:
    """Encode exact float32 bytes in a lazily loaded file://-compatible script."""
    values = np.asarray(data, dtype="<f4")
    if values.ndim == 2:
        values = values[..., None]
    height, width, channels = values.shape
    identifier = payload_id or stable_id("pixel", path)
    compressed = gzip.compress(values.tobytes(order="C"), compresslevel=6, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": encoded, "width": width, "height": height, "channels": channels}
    script = (
        "window.__DMAP_PIXEL_PAYLOADS=window.__DMAP_PIXEL_PAYLOADS||{};"
        f"window.__DMAP_PIXEL_PAYLOADS[{json.dumps(identifier)}]={json.dumps(payload, separators=(',', ':'))};\n"
    )
    path.write_text(script, encoding="ascii")
    return {
        "available": True,
        "script_path": path.as_posix(),
        "payload_id": identifier,
        "encoding": PIXEL_ENCODING,
        "value_dtype": "float32",
        "byte_order": "little",
        "width": width,
        "height": height,
        "channels": channels,
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": int(values.nbytes),
        "exact": True,
    }


def decode_pixel_encoding(path: Path, x: int, y: int, channels: int) -> list[float]:
    """Reference decoder used by tests; the browser implements the same contract."""
    script = path.read_text(encoding="ascii")
    match = re.search(r"PAYLOADS\[[^]]+\]=(\{.*\});", script)
    if not match:
        raise ValueError(f"invalid pixel payload script: {path}")
    payload = json.loads(match.group(1))
    values = np.frombuffer(gzip.decompress(base64.b64decode(payload["data"])), dtype="<f4")
    offset = (y * int(payload["width"]) + x) * channels
    return [float(value) for value in values[offset:offset + channels]]


def _artifact_priority(row: dict[str, Any]) -> tuple[Any, ...]:
    signal = str(row.get("signal", ""))
    descriptor = component_registry.descriptor_from_row(row)
    try:
        signal_rank = DEFAULT_SIGNALS.index(signal)
    except ValueError:
        signal_rank = len(DEFAULT_SIGNALS) if descriptor.default_visible else len(DEFAULT_SIGNALS) + 1
    return (
        0 if signal in CRITICAL_PIXEL_SIGNALS else 1,
        0 if row.get("role") == "logical_state" else 1,
        0 if row.get("measurement_quality") in {"exact", "derived_exact"} else 1,
        signal_rank,
        str(row.get("run", "")),
        int(row.get("repeat", 0) or 0),
        canonical_pyramid_level(row) if canonical_pyramid_level(row) is not None else -1,
        int(row.get("logical_iteration", -10) or -10),
        str(row.get("path", "")),
    )


def _channel_label(configured: dict[Any, Any], index: int) -> str:
    for key in (str(index), index):
        if key in configured:
            return str(configured[key])
    color_keys = ("R", "G", "B", "A")
    if index < len(color_keys):
        for key in (color_keys[index], color_keys[index].lower()):
            if key in configured:
                return str(configured[key])
    return f"channel {index}"


def build_map_assets(
    map_catalog: pd.DataFrame,
    signal_availability: pd.DataFrame,
    output_dir: Path,
    pixel_budget_bytes: int = PIXEL_DATA_BUDGET_BYTES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Create deterministic previews and lossless, bounded browser pixel payloads."""
    map_rows = records(map_catalog)
    map_rows = [canonical_pyramid_record(row) for row in map_rows]
    registry = component_registry.build_registry(map_rows)
    registry_by_signal = component_registry.registry_index(registry)
    asset_root = output_dir / "interactive" / "maps"
    stats: dict[str, dict[str, Any]] = {}
    shared_ranges: dict[str, list[float]] = {}
    shared_channel_ranges: dict[tuple[str, int], list[float]] = {}
    failures: list[dict[str, str]] = []

    for row in map_rows:
        raw_source = str(row.get("path") or "").strip()
        source = Path(raw_source) if raw_source else None
        artifact_id = stable_id(
            "map", row.get("run"), row.get("repeat"), row.get("scene_id"), row.get("frame"),
            row.get("signal"), row.get("logical_iteration"), row.get("role"), row.get("relative_path"),
            row.get("source_view_index"), row.get("estimation_stage"), row.get("geometric_iteration"),
            row.get("pyramid_level"),
        )
        row["id"] = artifact_id
        row["deep_link_id"] = artifact_id
        descriptor = registry_by_signal[str(row.get("signal", ""))]
        row["mechanism"] = descriptor["mechanism"]
        row["component_id"] = descriptor.get("component_id")
        row["quantity"] = descriptor["quantity"]
        row["units"] = descriptor["units"]
        row["value_domain"] = descriptor["domain"]
        row["preferred_direction"] = descriptor["preferred_direction"]
        row["minimum_profile"] = descriptor["minimum_profile"]
        row["measurement_kind"] = descriptor["measurement_kind"]
        row["colormap"] = descriptor["colormap"]
        row["signed"] = descriptor["signed"]
        row["category_legend"] = _category_legend(str(row.get("signal", "")))
        raw_channels = row.pop("channels_json", "")
        if raw_channels:
            try:
                row["channels"] = json.loads(str(raw_channels))
            except json.JSONDecodeError:
                row["channels"] = {"0": str(raw_channels)}
        row["source_path"] = relative_path(source, output_dir) if source is not None else None
        if row.get("source_image_name") and Path(str(row["source_image_name"])).is_absolute():
            row["source_image_name"] = relative_path(row["source_image_name"], output_dir)
        row.pop("path", None)
        row.pop("manifest_path", None)
        row.pop("image_name", None)
        if not row.get("available") or source is None or not source.is_file():
            row["available"] = False
            row["unavailable_reason"] = (
                row.get("availability_reason")
                or row.get("unavailable_reason")
                or "source artifact is missing"
            )
            continue
        if source.suffix.lower() != ".pfm":
            if "transition" in str(row.get("signal", "")).lower():
                try:
                    from PIL import Image

                    codes = np.asarray(Image.open(source))
                    if codes.ndim > 2:
                        codes = codes[..., 0]
                    palette = np.asarray([
                        [35, 42, 49, 255],
                        [73, 145, 118, 255],
                        [210, 67, 65, 255],
                        [43, 145, 190, 255],
                    ], dtype=np.uint8)
                    preview_path = asset_root / f"{row['id']}_transition.png"
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(palette[np.clip(codes.astype(np.int64), 0, 3)], mode="RGBA").save(preview_path)
                    preview = relative_path(preview_path, output_dir)
                    row["preview"] = {
                        "local": preview,
                        "shared": preview,
                        "scale_mode": "discrete_transition_codes",
                    }
                    row["transition_legend"] = {
                        "0": "unchanged absent/zero", "1": "unchanged present/positive",
                        "2": "removed/became zero", "3": "added/became positive",
                    }
                    continue
                except Exception as exc:
                    failures.append({"artifact_id": artifact_id, "error": f"transition preview: {exc}"})
            try:
                from PIL import Image

                preview_path = asset_root / f"{artifact_id}_source.png"
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as source_image:
                    source_image.save(preview_path, format="PNG")
                report_preview = relative_path(preview_path, output_dir)
                row["preview"] = {
                    "local": report_preview,
                    "shared": report_preview,
                    "scale_mode": "report_owned_source_raster",
                }
            except Exception as exc:
                row["available"] = False
                row["unavailable_reason"] = f"raster preview materialization failed: {exc}"
                failures.append({
                    "artifact_id": artifact_id,
                    "error": f"raster preview materialization: {exc}",
                })
                continue
            if str(row.get("signal", "")) not in LOSSLESS_RASTER_PIXEL_SIGNALS:
                continue
            try:
                from PIL import Image

                data = np.asarray(Image.open(source))
                if data.ndim not in {2, 3}:
                    raise ValueError(f"unsupported raster shape {data.shape}")
                limits = _local_scale(str(row.get("signal", "")), data)
                channel_count = data.shape[2] if data.ndim == 3 else 1
                configured_channels = row.get("channels") if isinstance(row.get("channels"), dict) else {}
                channel_stats = []
                for channel in range(channel_count):
                    channel_data = data[..., channel] if data.ndim == 3 else data
                    channel_limits = _local_scale(str(row.get("signal", "")), channel_data)
                    channel_stats.append({
                        "index": channel,
                        "label": _channel_label(configured_channels, channel),
                        "local": channel_limits,
                    })
                    channel_bounds = shared_channel_ranges.setdefault(
                        (str(row.get("signal", "")), channel),
                        [channel_limits[0], channel_limits[1]],
                    )
                    channel_bounds[0] = min(channel_bounds[0], channel_limits[0])
                    channel_bounds[1] = max(channel_bounds[1], channel_limits[1])
                stats[artifact_id] = {
                    "source": source,
                    "shape": list(data.shape),
                    # Browser payloads normalize every numeric source to float32.
                    "source_bytes": int(data.size * np.dtype(np.float32).itemsize),
                    "local": limits,
                    "channels": channel_stats,
                    "source_raster": True,
                }
                if data.ndim == 2 and descriptor["measurement_kind"] == "enum":
                    categorical_path = asset_root / f"{artifact_id}_categorical.png"
                    _write_preview(
                        categorical_path,
                        str(row.get("signal", "")),
                        data,
                        limits,
                    )
                    categorical_preview = relative_path(categorical_path, output_dir)
                    row["preview"] = {
                        "local": categorical_preview,
                        "shared": categorical_preview,
                        "scale_mode": "categorical_registered_codes",
                    }
            except Exception as exc:
                failures.append({"artifact_id": artifact_id, "error": f"lossless raster decode: {exc}"})
            continue
        try:
            data = read_pfm(source)
            limits = _local_scale(str(row.get("signal", "")), data)
            channel_count = data.shape[2] if data.ndim == 3 else 1
            configured_channels = row.get("channels") if isinstance(row.get("channels"), dict) else {}
            channel_stats = []
            for channel in range(channel_count):
                channel_data = data[..., channel] if data.ndim == 3 else data
                channel_limits = (
                    (-1.0, 1.0)
                    if "normal" in str(row.get("signal", "")).lower()
                    else _local_scale(str(row.get("signal", "")), channel_data)
                )
                label = _channel_label(configured_channels, channel)
                channel_stats.append({"index": channel, "label": label, "local": channel_limits})
                channel_bounds = shared_channel_ranges.setdefault(
                    (str(row.get("signal", "")), channel), [channel_limits[0], channel_limits[1]]
                )
                channel_bounds[0] = min(channel_bounds[0], channel_limits[0])
                channel_bounds[1] = max(channel_bounds[1], channel_limits[1])
            stats[artifact_id] = {
                "source": source,
                "shape": list(data.shape),
                "source_bytes": int(data.nbytes),
                "local": limits,
                "channels": channel_stats,
            }
            bounds = shared_ranges.setdefault(str(row.get("signal", "")), [limits[0], limits[1]])
            bounds[0] = min(bounds[0], limits[0])
            bounds[1] = max(bounds[1], limits[1])
        except Exception as exc:
            row["available"] = False
            row["unavailable_reason"] = f"PFM decode failed: {exc}"
            failures.append({"artifact_id": artifact_id, "error": str(exc)})

    numeric_candidates = [row for row in map_rows if row.get("id") in stats]
    numeric_cohorts: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in numeric_candidates:
        key = (
            row.get("scene_id"), row.get("image_id"), row.get("frame"), row.get("signal"),
            row.get("logical_iteration"), row.get("role"), row.get("stage"),
            row.get("estimation_stage"), row.get("geometric_iteration"), row.get("pyramid_level"),
        )
        numeric_cohorts.setdefault(key, []).append(row)
    ordered_cohorts = sorted(
        numeric_cohorts.values(), key=lambda cohort: min(_artifact_priority(row) for row in cohort)
    )
    pixel_selected: set[str] = set()
    source_bytes_selected = 0
    for cohort in ordered_cohorts:
        source_bytes = sum(int(stats[str(row["id"])]["source_bytes"]) for row in cohort)
        if source_bytes_selected + source_bytes <= pixel_budget_bytes:
            pixel_selected.update(str(row["id"]) for row in cohort)
            source_bytes_selected += source_bytes

    for row in map_rows:
        artifact_id = str(row.get("id", ""))
        if artifact_id not in stats:
            continue
        info = stats[artifact_id]
        try:
            data = read_diagnostic_map(info["source"]) if info.get("source_raster") else read_pfm(info["source"])
            if info.get("source_raster"):
                row["shape"] = info["shape"]
                if data.ndim == 3:
                    row["preview"] = {
                        "local": None,
                        "shared": None,
                        "scale_mode": (
                            "categorical_registered_codes"
                            if row.get("measurement_kind") == "enum"
                            else "generated_channel_preview"
                        ),
                        "colormap": _colormap(str(row.get("signal", ""))),
                        "channels": [],
                    }
                    for channel in info["channels"]:
                        index = int(channel["index"])
                        channel_local_path = asset_root / f"{artifact_id}_ch{index}_local.png"
                        channel_shared_path = asset_root / f"{artifact_id}_ch{index}_shared.png"
                        channel_shared = tuple(
                            shared_channel_ranges[(str(row.get("signal", "")), index)]
                        )
                        _write_preview(
                            channel_local_path,
                            str(row.get("signal", "")),
                            data[..., index],
                            tuple(channel["local"]),
                        )
                        _write_preview(
                            channel_shared_path,
                            str(row.get("signal", "")),
                            data[..., index],
                            channel_shared,
                        )
                        preview = {
                            "index": index,
                            "label": channel["label"],
                            "local": relative_path(channel_local_path, output_dir),
                            "shared": relative_path(channel_shared_path, output_dir),
                            "local_scale": {
                                "low": channel["local"][0],
                                "high": channel["local"][1],
                            },
                            "shared_scale": {
                                "low": channel_shared[0],
                                "high": channel_shared[1],
                            },
                        }
                        row["preview"]["channels"].append(preview)
                    first_preview = row["preview"]["channels"][0]
                    row["preview"].update({
                        "local": first_preview["local"],
                        "shared": first_preview["shared"],
                        "local_scale": first_preview["local_scale"],
                        "shared_scale": first_preview["shared_scale"],
                    })
                if artifact_id in pixel_selected:
                    encoded_path = asset_root / f"{artifact_id}_float32.js"
                    encoding = _write_pixel_encoding(encoded_path, data, artifact_id)
                    encoding["script_path"] = relative_path(encoded_path, output_dir)
                    encoding["channel_labels"] = [channel["label"] for channel in info["channels"]]
                    row["pixel_data"] = encoding
                else:
                    row["pixel_data"] = {
                        "available": False,
                        "reason": "numeric payload omitted by the interactive pixel-data storage budget",
                        "source_bytes": info["source_bytes"],
                    }
                continue
            local_path = asset_root / f"{artifact_id}_local.png"
            shared_path = asset_root / f"{artifact_id}_shared.png"
            _write_preview(local_path, str(row.get("signal", "")), data, tuple(info["local"]))
            shared = tuple(shared_ranges[str(row.get("signal", ""))])
            _write_preview(shared_path, str(row.get("signal", "")), data, shared)
            row["preview"] = {
                "local": relative_path(local_path, output_dir),
                "shared": relative_path(shared_path, output_dir),
                "local_scale": {"low": info["local"][0], "high": info["local"][1]},
                "shared_scale": {"low": shared[0], "high": shared[1]},
                "colormap": _colormap(str(row.get("signal", ""))),
                "channels": [],
            }
            if row.get("measurement_kind") == "enum" and data.ndim == 2:
                row["preview"]["scale_mode"] = "categorical_registered_codes"
            if data.ndim == 3:
                for channel in info["channels"]:
                    index = int(channel["index"])
                    channel_local_path = asset_root / f"{artifact_id}_ch{index}_local.png"
                    channel_shared_path = asset_root / f"{artifact_id}_ch{index}_shared.png"
                    channel_shared = tuple(shared_channel_ranges[(str(row.get("signal", "")), index)])
                    _write_preview(
                        channel_local_path, str(row.get("signal", "")), data[..., index], tuple(channel["local"])
                    )
                    _write_preview(
                        channel_shared_path, str(row.get("signal", "")), data[..., index], channel_shared
                    )
                    row["preview"]["channels"].append({
                        "index": index, "label": channel["label"],
                        "local": relative_path(channel_local_path, output_dir),
                        "shared": relative_path(channel_shared_path, output_dir),
                        "local_scale": {"low": channel["local"][0], "high": channel["local"][1]},
                        "shared_scale": {"low": channel_shared[0], "high": channel_shared[1]},
                    })
            row["shape"] = info["shape"]
            if artifact_id in pixel_selected:
                encoded_path = asset_root / f"{artifact_id}_float32.js"
                encoding = _write_pixel_encoding(encoded_path, data, artifact_id)
                encoding["script_path"] = relative_path(encoded_path, output_dir)
                encoding["channel_labels"] = [channel["label"] for channel in info["channels"]]
                row["pixel_data"] = encoding
            else:
                row["pixel_data"] = {
                    "available": False,
                    "reason": "numeric payload omitted by the interactive pixel-data storage budget",
                    "source_bytes": info["source_bytes"],
                }
        except Exception as exc:
            row["available"] = False
            row["unavailable_reason"] = f"interactive asset generation failed: {exc}"
            failures.append({"artifact_id": artifact_id, "error": str(exc)})

    existing_keys = {
        (
            row.get("run"), row.get("repeat"), row.get("scene_id"), row.get("frame"),
            row.get("signal"), row.get("logical_iteration"), row.get("source_view_index"),
            row.get("estimation_stage"), row.get("geometric_iteration"), row.get("pyramid_level"),
        )
        for row in map_rows
    }
    for row in records(signal_availability):
        key = (
            row.get("run"), row.get("repeat"), row.get("scene_id"), row.get("frame"),
            row.get("signal"), row.get("logical_iteration"), row.get("source_view_index"),
            row.get("estimation_stage"), row.get("geometric_iteration"), canonical_pyramid_level(row),
        )
        if bool(row.get("available")) or key in existing_keys:
            continue
        artifact_id = stable_id("map-unavailable", *key)
        map_rows.append({
            "id": artifact_id,
            "deep_link_id": artifact_id,
            "run": row.get("run"),
            "label": row.get("label"),
            "run_role": row.get("run_role"),
            "repeat": row.get("repeat"),
            "scene_id": row.get("scene_id"),
            "estimation_stage": row.get("estimation_stage"),
            "geometric_iteration": row.get("geometric_iteration"),
            "pyramid_level": canonical_pyramid_level(row),
            "frame": row.get("frame"),
            "image_id": row.get("image_id"),
            "signal": row.get("signal"),
            "logical_iteration": row.get("logical_iteration"),
            "source_view_index": row.get("source_view_index"),
            "source_image_id": row.get("source_image_id"),
            "source_image_name": row.get("source_image_name"),
            "mechanism": registry_by_signal.get(
                str(row.get("signal", "")),
                component_registry.descriptor_from_row(row).to_dict(),
            )["mechanism"],
            "stage": row.get("stage"),
            "role": row.get("role"),
            "measurement_quality": row.get("measurement_quality"),
            "measurement_basis": row.get("measurement_basis"),
            "proxy_target": row.get("proxy_target"),
            "semantics": row.get("semantics"),
            "limitations": row.get("limitations"),
            "available": False,
            "unavailable_reason": row.get("availability_reason") or "signal unavailable",
        })

    signal_rows: list[dict[str, Any]] = []
    for signal in sorted({str(row.get("signal")) for row in map_rows if row.get("signal")}):
        selected = [row for row in map_rows if row.get("signal") == signal]
        descriptor = registry_by_signal.get(
            signal,
            component_registry.descriptor_from_row({"signal": signal}).to_dict(),
        )
        qualities = sorted({
            str(row.get("measurement_quality"))
            for row in selected
            if row.get("available") and row.get("measurement_quality")
        })
        if not qualities and selected:
            qualities = ["unavailable"]
        signal_rows.append({
            "id": stable_id("signal", signal),
            "name": signal,
            "label": descriptor["label"],
            "measurement_qualities": qualities,
            "available_artifacts": sum(bool(row.get("available")) for row in selected),
            "unavailable_artifacts": sum(not bool(row.get("available")) for row in selected),
            "shared_scale": (
                {"low": shared_ranges[signal][0], "high": shared_ranges[signal][1]}
                if signal in shared_ranges else None
            ),
            "default": bool(descriptor["default_visible"]),
            "mechanism": descriptor["mechanism"],
            "component_id": descriptor.get("component_id"),
            "quantity": descriptor["quantity"],
            "units": descriptor["units"],
            "domain": descriptor["domain"],
            "preferred_direction": descriptor["preferred_direction"],
            "minimum_profile": descriptor["minimum_profile"],
            "measurement_kind": descriptor["measurement_kind"],
            "colormap": descriptor["colormap"],
            "description": descriptor["description"],
        })
    signal_rows.insert(0, {
        "id": stable_id("signal", "reference_rgb"),
        "name": "reference_rgb",
        "label": "reference RGB",
        "measurement_qualities": ["source"],
        "available_artifacts": 0,
        "unavailable_artifacts": 0,
        "shared_scale": None,
        "default": True,
        "mechanism": "input",
        "component_id": None,
        "quantity": "reference_image",
        "units": "unitless",
        "domain": "finite",
        "preferred_direction": "contextual",
        "minimum_profile": "light",
        "measurement_kind": "image",
        "colormap": "viridis",
        "description": "Reference RGB image for spatial interpretation.",
    })
    preview_paths: set[str] = set()
    for row in map_rows:
        preview = row.get("preview") or {}
        values = [preview.get("local"), preview.get("shared")]
        for channel in preview.get("channels") or []:
            values.extend([channel.get("local"), channel.get("shared")])
        preview_paths.update(
            str(value) for value in values
            if value and not str(value).startswith(("file:", "http:", "https:"))
        )
    pixel_paths = {
        str((row.get("pixel_data") or {}).get("script_path"))
        for row in map_rows
        if (row.get("pixel_data") or {}).get("available")
    }
    preview_output_bytes = sum(
        (output_dir / path).stat().st_size for path in preview_paths if (output_dir / path).is_file()
    )
    pixel_output_bytes = sum(
        (output_dir / path).stat().st_size for path in pixel_paths if (output_dir / path).is_file()
    )
    budget = {
        "limit_bytes": pixel_budget_bytes,
        "source_bytes_selected": source_bytes_selected,
        "encoded_pixel_output_bytes": pixel_output_bytes,
        "preview_output_bytes": preview_output_bytes,
        "interactive_map_output_bytes": pixel_output_bytes + preview_output_bytes,
        "selected_artifacts": len(pixel_selected),
        "eligible_artifacts": len(stats),
        "omitted_artifacts": len(stats) - len(pixel_selected),
        "policy": "critical candidate/final/view mechanics first, then logical-state and exact/derived signals; baseline/variant artifact cohorts are atomic",
        "encoding": PIXEL_ENCODING,
        "failures": failures,
    }
    return map_rows, signal_rows, budget


def _run_id(label: Any, repeat: Any) -> str:
    return stable_id("run", label, int(repeat or 0))


def _scalar_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key in PATH_COLUMNS or key in {
            "run", "role", "repeat", "scene_id", "image_name", "safe_image_name", "missing_maps",
            "candidate_accounting_mode", "confidence_gap_mode",
        }:
            continue
        normalized = json_value(value)
        if isinstance(normalized, (int, float, bool)) or normalized is None:
            result[key] = normalized
    return result


def _portable_record(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = dict(row)
    for key in PATH_COLUMNS:
        if result.get(key):
            result[key] = relative_path(result[key], output_dir)
    for key in ("source_image_name", "candidate_image_name", "reference_image_name"):
        if result.get(key) and Path(str(result[key])).is_absolute():
            result[key] = relative_path(result[key], output_dir)
    return result


def normalize_reference_patch_layout(
    value: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate the observer patch contract source-checked against CUDA layout constants."""
    return reference_patch_layout.normalize(value)


def reference_patch_layout_records(
    run_scenes: Iterable[Any], output_dir: Path,
) -> list[dict[str, Any]]:
    """Load one explicit patch-layout availability record per run/stage/scene."""

    records_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for run_scene in run_scenes:
        identity = (
            str(run_scene.label), int(run_scene.repeat), str(run_scene.scene_id),
            str(getattr(run_scene, "estimation_stage", "photometric")),
            getattr(run_scene, "geometric_iteration", None),
        )
        instrumentation_dir = getattr(run_scene, "instrumentation_dir", None)
        metadata_path = (
            Path(instrumentation_dir) / "run_metadata.json"
            if instrumentation_dir is not None else None
        )
        record: dict[str, Any] = {
            "run": identity[0],
            "repeat": identity[1],
            "scene_id": identity[2],
            "estimation_stage": identity[3],
            "geometric_iteration": identity[4],
            "capture_profile": str(getattr(run_scene, "capture_profile", "summary")),
            "contract_claimed": False,
            "contract_valid": True,
            "capability_status": "legacy_unclaimed",
            "available": False,
            "layout": None,
            "measurement_quality": "unavailable",
            "measurement_basis": (
                "run_metadata.cuda_patchmatch_parameters.reference_patch_layout"
            ),
            "visualization_quality": "unavailable",
            "unavailable_reason": (
                "run_metadata.json is missing"
                if metadata_path is not None
                else "instrumentation directory is unavailable"
            ),
            "limitations": (
                "Reference-grid positions can be derived only from a declared fixed "
                "layout and a validated pyramid extent. CUDA sample values and "
                "source-view footprints are separate capabilities."
            ),
        }
        if (
            metadata_path is not None
            and metadata_path.is_file()
            and not metadata_path.is_symlink()
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                record["contract_valid"] = False
                record["capability_status"] = "invalid_metadata"
                record["unavailable_reason"] = f"run_metadata.json could not be read: {exc}"
            else:
                instrumentation = metadata.get("instrumentation")
                capabilities = (
                    instrumentation.get("capabilities")
                    if isinstance(instrumentation, dict) else None
                )
                capabilities = capabilities if isinstance(capabilities, dict) else {}
                capability_name = "reference_patch_layout_contract"
                if capability_name not in capabilities:
                    record["unavailable_reason"] = (
                        "reference patch layout capability was not declared"
                    )
                elif not isinstance(capabilities.get(capability_name), bool):
                    record["contract_claimed"] = None
                    record["contract_valid"] = False
                    record["capability_status"] = "invalid"
                    record["unavailable_reason"] = (
                        "reference patch layout capability is not boolean"
                    )
                elif capabilities[capability_name] is False:
                    record["capability_status"] = "disabled"
                    record["unavailable_reason"] = (
                        "reference patch layout capability was not claimed"
                    )
                else:
                    record["contract_claimed"] = True
                    record["capability_status"] = "claimed"
                    if (
                        metadata.get("schema_name") != "openmvs.dmap.run"
                        or _model_integer(metadata.get("schema_version")) != 4
                    ):
                        record["contract_valid"] = False
                        record["capability_status"] = "invalid"
                        record["unavailable_reason"] = (
                            "claimed layout requires openmvs.dmap.run schema v4"
                        )
                    else:
                        capability_fields_valid = all(
                            capabilities.get(name) is False for name in (
                                "reference_patch_sample_locations",
                                "reference_patch_sample_values",
                                "source_view_patch_footprints",
                            )
                        )
                        parameters = metadata.get("cuda_patchmatch_parameters")
                        layout, reason = normalize_reference_patch_layout(
                            parameters.get("reference_patch_layout")
                            if isinstance(parameters, dict) else None
                        )
                        if layout is None or not capability_fields_valid:
                            record["contract_valid"] = False
                            record["capability_status"] = "invalid"
                            record["unavailable_reason"] = (
                                reason if layout is None else
                                "schema-v1 patch sample capabilities must be false"
                            )
                        else:
                            record.update({
                                "available": True,
                                "layout": layout,
                                "measurement_quality": "exact",
                                "visualization_quality": "derived_exact",
                                "unavailable_reason": None,
                            })
        previous = records_by_identity.get(identity)
        if previous is not None and previous != record:
            raise ValueError(
                "conflicting reference patch layouts for "
                f"{identity[0]}/{identity[2]}/{identity[3]}"
            )
        records_by_identity[identity] = record
    return [records_by_identity[key] for key in sorted(
        records_by_identity,
        key=lambda value: (
            value[0], value[1], value[2], value[3],
            -1 if value[4] is None else int(value[4]),
        ),
    )]


def _cuda_resource_plan_records(
    dataframe: pd.DataFrame,
    *,
    experiment_root: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve schema-v4 coarse compatibility-map availability contracts."""

    del experiment_root  # Catalog rows already bind validated raw-plan fields.
    output: list[dict[str, Any]] = []
    availability: list[dict[str, Any]] = []
    for source_row in records(dataframe):
        row = dict(source_row)
        target_image = _model_integer(row.get("image_id"))
        target_level = canonical_pyramid_level(row)
        target_stage = str(row.get("estimation_stage") or "photometric")
        target_geometric = _model_integer(row.get("geometric_iteration"))
        plan_geometric = row.get("plan_geometric_iteration")
        geometric_identity_matches = (
            plan_geometric is None
            if target_geometric is None
            else _model_integer(plan_geometric) == target_geometric
        )
        row_authoritative = (
            row.get("available") is True
            and row.get("valid") is True
            and row.get("schema_name") == "openmvs.dmap.resource_plan"
            and _model_integer(row.get("schema_version")) == 4
            and _model_integer(row.get("duplicate_count")) == 1
            and row.get("plan_identity_complete") is True
            and target_image is not None
            and target_level is not None
            and _model_integer(row.get("plan_image_id")) == target_image
            and canonical_pyramid_level({
                "pyramid_level": row.get("plan_pyramid_level")
            }) == target_level
            and row.get("plan_estimation_stage") == target_stage
            and geometric_identity_matches
        )
        plan_width = _trace_integer(row.get("grid_width"))
        plan_height = _trace_integer(row.get("grid_height"))
        extent_available = (
            row_authoritative
            and plan_width is not None and plan_width > 0
            and plan_height is not None and plan_height > 0
        )
        if not row_authoritative:
            extent_reason = (
                "resource plan catalog row is not unique valid schema-v4 evidence "
                "with matching frame, stage, and pyramid identity"
            )
        else:
            extent_reason = "resource plan width/height is missing or invalid"
        row["grid_extent"] = {
            "available": extent_available,
            "width": plan_width if extent_available else None,
            "height": plan_height if extent_available else None,
            "measurement_quality": "exact" if extent_available else "unavailable",
            "measurement_basis": "resource_plan.width_height",
            "unavailable_reason": (
                None if extent_available else extent_reason
            ),
        }
        contract = _json_object(row.get("compatibility_map_contract_json"))
        if row_authoritative and contract:
            row["compatibility_map_contract"] = json_value(contract)
            level = canonical_pyramid_level(row)
            if level is not None and level > 0:
                cost_expected = contract.get("cost_map_expected")
                cost_reason = str(contract.get("cost_map_unavailable_reason") or "")
                entry = {
                    "run": row.get("run"),
                    "repeat": row.get("repeat"),
                    "scene_id": row.get("scene_id"),
                    "image_id": row.get("image_id"),
                    "estimation_stage": row.get("estimation_stage"),
                    "geometric_iteration": row.get("geometric_iteration"),
                    "pyramid_level": level,
                    "compatibility_maps_requested": row.get(
                        "compatibility_maps_requested"
                    ),
                    "update_source_map_expected": contract.get(
                        "update_source_map_expected"
                    ),
                    "cost_map_expected": cost_expected,
                    "cost_map_available": False if cost_expected is False else None,
                    "cost_map_unavailable_reason": cost_reason or None,
                    "measurement_basis": (
                        "resource_plan.compatibility_map_contract"
                    ),
                    "source_json": row.get("source_json"),
                }
                availability.append(json_value(entry))
        output.append(_portable_record(row, output_dir))
    availability.sort(key=lambda row: (
        str(row.get("run")), int(row.get("repeat") or 0),
        str(row.get("scene_id")), _identity_integer(row.get("image_id")),
        str(row.get("estimation_stage")),
        _identity_integer(row.get("geometric_iteration")),
        _identity_integer(row.get("pyramid_level")),
    ))
    return output, availability


def _exclude_diagnostic_quality_rows(
    dataframe: pd.DataFrame,
    diagnostic_labels: set[str],
) -> pd.DataFrame:
    """Remove mechanics-only cohorts from aggregate quality evidence."""
    if dataframe.empty or not diagnostic_labels:
        return dataframe
    selected = pd.Series(True, index=dataframe.index, dtype=bool)
    for column in ("candidate", "run", "baseline", "variant"):
        if column in dataframe.columns:
            selected &= ~dataframe[column].fillna("").astype(str).isin(diagnostic_labels)
    return dataframe[selected].copy()


def _exclude_diagnostic_quality_records(
    rows: Iterable[dict[str, Any]],
    diagnostic_labels: set[str],
) -> list[dict[str, Any]]:
    """Filter candidate-oriented quality records without touching mechanics."""
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if any(
            str(row.get(column, "")) in diagnostic_labels
            for column in ("candidate", "run", "baseline", "variant")
        ):
            continue
        item = dict(row)
        if isinstance(item.get("dominated_by"), list):
            item["dominated_by"] = [
                value for value in item["dominated_by"]
                if str(value) not in diagnostic_labels
            ]
        filtered.append(item)
    return filtered


def _paired_regressions(
    dataframe: pd.DataFrame,
    baseline: str,
    candidates: Iterable[str],
    metric_specs: dict[str, Any],
    level: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_columns = ["scene_id"] if level == "performance" else ["scene_id", "image_id"]
    if dataframe.empty or not {"run", *identity_columns}.issubset(dataframe.columns):
        return rows
    for candidate in candidates:
        for metric, spec in metric_specs.items():
            if getattr(spec, "level", None) != level or metric not in dataframe.columns:
                continue
            metric_rows = dataframe[["run", *identity_columns, metric]].copy()
            metric_rows[metric] = pd.to_numeric(metric_rows[metric], errors="coerce")
            metric_rows = metric_rows[np.isfinite(metric_rows[metric])]
            if metric_rows.empty:
                continue
            grouped = metric_rows.groupby(
                ["run", *identity_columns], as_index=False
            )[metric].mean()
            left = grouped[grouped["run"] == baseline].rename(columns={metric: "baseline_value"})
            right = grouped[grouped["run"] == candidate].rename(columns={metric: "candidate_value"})
            paired = left.merge(right, on=identity_columns, how="inner")
            for item in records(paired):
                raw_delta = float(item["candidate_value"] - item["baseline_value"])
                if level == "performance" and abs(float(item["baseline_value"])) > 1e-12:
                    delta = raw_delta / abs(float(item["baseline_value"]))
                else:
                    delta = raw_delta
                tolerance = float(getattr(spec, "tolerance", 0.0))
                direction = str(getattr(spec, "direction", "lower"))
                regression_score = -delta if direction == "higher" else delta
                if regression_score > tolerance:
                    status = "regressed"
                elif regression_score < -tolerance:
                    status = "improved"
                else:
                    status = "stable"
                scene_id = str(item["scene_id"])
                image_id = (
                    None if level == "performance"
                    else _model_integer(item["image_id"])
                )
                if level != "performance" and image_id is None:
                    continue
                rows.append({
                    "id": stable_id(
                        "regression", candidate, metric, scene_id,
                        "scene" if image_id is None else image_id,
                    ),
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "level": level,
                    "direction": direction,
                    "unit": str(getattr(spec, "unit", "")),
                    "baseline_value": item["baseline_value"],
                    "candidate_value": item["candidate_value"],
                    "delta": delta,
                    "raw_delta": raw_delta,
                    "regression_score": regression_score,
                    "tolerance": tolerance,
                    "status": status,
                    "scene_id": scene_id,
                    "image_id": image_id,
                    "scene_deep_link_id": stable_id("scene", scene_id),
                    "frame_deep_link_id": (
                        None if image_id is None
                        else stable_id("frame", scene_id, image_id)
                    ),
                    "navigation_level": (
                        "scene" if image_id is None else "frame"
                    ),
                })
    return rows


def _portable_inventory(inventory: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    result = json_value(inventory)
    for item in result.get("overall_plots", []):
        item["path"] = relative_path(item.get("path"), output_dir)
    for scene in result.get("scenes", {}).values():
        for item in [*(scene.get("plots") or []), *(scene.get("panels") or [])]:
            item["path"] = relative_path(item.get("path"), output_dir)
    for item in result.get("geometry_artifacts", []):
        for key in ("visual_overlay_svg", "visual_residual_histogram_svg"):
            if item.get(key):
                item[key] = relative_path(item[key], output_dir)
    for item in result.get("data_artifacts", []):
        for key in ("csv", "parquet", "json"):
            if item.get(key):
                item[key] = relative_path(item[key], output_dir)
    return paths_report_relative(result, output_dir)


def build_report_model(
    *,
    config: dict[str, Any],
    experiment_root: Path,
    output_dir: Path,
    run_scenes: list[Any],
    frames: pd.DataFrame,
    iterations: pd.DataFrame,
    performance: pd.DataFrame,
    annotations: pd.DataFrame,
    comparisons: pd.DataFrame,
    gates: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    map_catalog: pd.DataFrame,
    signal_availability: pd.DataFrame,
    inventory: dict[str, Any],
    metric_specs: dict[str, Any],
    endpoint_performance: pd.DataFrame | None = None,
    exact_cost_evolution: pd.DataFrame | None = None,
    exact_observability: pd.DataFrame | None = None,
    exact_iterations: pd.DataFrame | None = None,
    exact_views: pd.DataFrame | None = None,
    cpu_view_candidates: pd.DataFrame | None = None,
    cpu_estimation_selection: pd.DataFrame | None = None,
    postprocess_filters: pd.DataFrame | None = None,
    confidence_adjustment: pd.DataFrame | None = None,
    cuda_resource_plans: pd.DataFrame | None = None,
    filter_resource_plans: pd.DataFrame | None = None,
    resource_plan_validation: pd.DataFrame | None = None,
    instrumentation_validation: pd.DataFrame | None = None,
    accuracy_ledger: pd.DataFrame | None = None,
    accuracy_evidence: pd.DataFrame | None = None,
    model_stability: pd.DataFrame | None = None,
    summary_signal_contract: dict[str, Any] | None = None,
    capture_profile_coverage: dict[str, Any] | None = None,
    evidence_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    endpoint_performance = (
        endpoint_performance if endpoint_performance is not None else pd.DataFrame()
    )
    accuracy_ledger = accuracy_ledger if accuracy_ledger is not None else pd.DataFrame()
    accuracy_evidence = accuracy_evidence if accuracy_evidence is not None else pd.DataFrame()
    model_stability = model_stability if model_stability is not None else pd.DataFrame()
    exact_cost_evolution = exact_cost_evolution if exact_cost_evolution is not None else pd.DataFrame()
    exact_observability = exact_observability if exact_observability is not None else pd.DataFrame()
    exact_iterations = exact_iterations if exact_iterations is not None else pd.DataFrame()
    exact_views = exact_views if exact_views is not None else pd.DataFrame()
    cpu_view_candidates = cpu_view_candidates if cpu_view_candidates is not None else pd.DataFrame()
    cpu_estimation_selection = cpu_estimation_selection if cpu_estimation_selection is not None else pd.DataFrame()
    postprocess_filters = postprocess_filters if postprocess_filters is not None else pd.DataFrame()
    confidence_adjustment = confidence_adjustment if confidence_adjustment is not None else pd.DataFrame()
    cuda_resource_plans = cuda_resource_plans if cuda_resource_plans is not None else pd.DataFrame()
    filter_resource_plans = filter_resource_plans if filter_resource_plans is not None else pd.DataFrame()
    resource_plan_validation = resource_plan_validation if resource_plan_validation is not None else pd.DataFrame()
    instrumentation_validation = instrumentation_validation if instrumentation_validation is not None else pd.DataFrame()
    (
        cuda_resource_plan_records,
        coarse_compatibility_map_availability,
    ) = _cuda_resource_plan_records(
        cuda_resource_plans,
        experiment_root=experiment_root,
        output_dir=output_dir,
    )
    map_rows, signal_rows, pixel_budget = build_map_assets(
        map_catalog, signal_availability, output_dir
    )
    reference_patch_layouts = reference_patch_layout_records(
        run_scenes, output_dir
    )
    component_registry_model = component_registry.build_registry([
        *records(map_catalog),
        *records(signal_availability),
        {"signal": "reference_rgb"},
    ])
    registry_errors = component_registry.validate_registry(component_registry_model)
    if registry_errors:
        raise ValueError("invalid component registry: " + "; ".join(registry_errors))
    texture_stratification = region_metrics.compute_texture_stratification(
        records(map_catalog),
        read_diagnostic_map,
    )
    accepted_gain_census = region_metrics.compute_low_texture_accepted_gain_census(
        records(map_catalog),
        read_diagnostic_map,
        variance_max_by_run=configured_low_texture_variance_max_by_run(config),
    )
    low_texture_hysteresis = build_low_texture_update_hysteresis_metrics(
        exact_iterations
    )
    low_texture_update_declared = bool(low_texture_hysteresis.get("rows"))
    investigation_guide_model = investigation_guide_for_registry(
        component_registry_model,
        low_texture_update_declared=low_texture_update_declared,
    )
    extension_contracts = optional_extension_contracts(
        component_registry_model,
        low_texture_update_declared=low_texture_update_declared,
    )
    drilldowns = load_drilldown_index(experiment_root, output_dir)
    quality_eligibility_by_run: dict[tuple[str, int], tuple[bool, str]] = {}
    validation_quality = instrumentation_validation
    if not validation_quality.empty and "terminal_stage" in validation_quality.columns:
        validation_quality = validation_quality[
            validation_quality["terminal_stage"].fillna(False).astype(bool)
        ]
    if not validation_quality.empty and "diagnostic_only" in validation_quality.columns:
        validation_quality = validation_quality[
            ~validation_quality["diagnostic_only"].fillna(False).astype(bool)
        ]
    if not validation_quality.empty and {"run", "repeat"}.issubset(
        validation_quality.columns
    ):
        for (run_label, repeat), group in validation_quality.groupby(
            ["run", "repeat"], dropna=False
        ):
            eligible = (
                "quality_comparison_eligible" in group.columns
                and bool(
                    group["quality_comparison_eligible"]
                    .fillna(False)
                    .astype(bool)
                    .all()
                )
            )
            reasons = sorted({
                str(reason)
                for reason in group.get(
                    "quality_comparison_ineligible_reason", pd.Series(dtype=str)
                ).dropna()
                if str(reason)
            })
            quality_eligibility_by_run[(str(run_label), int(repeat))] = (
                eligible,
                " | ".join(reasons),
            )
    runs_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    diagnostic_reasons_by_key: dict[tuple[str, int], set[str]] = {}
    for item in run_scenes:
        key = (str(item.label), int(item.repeat))
        diagnostic_only = bool(getattr(item, "diagnostic_only", False))
        diagnostic_reason = str(getattr(item, "diagnostic_only_reason", "") or "")
        parity_eligible, parity_reason = quality_eligibility_by_run.get(
            key, (False, "complete production endpoint parity is unavailable")
        )
        entry = runs_by_key.setdefault(key, {
            "id": _run_id(*key),
            "deep_link_id": _run_id(*key),
            "label": key[0],
            "role": str(item.role),
            "repeat": key[1],
            "diagnostic_only": diagnostic_only,
            "diagnostic_only_reason": "",
            "quality_comparison_eligible": (
                not diagnostic_only and parity_eligible
            ),
            "quality_comparison_ineligible_reason": (
                "" if not diagnostic_only and parity_eligible
                else diagnostic_reason or parity_reason
            ),
            "scenes": [],
            "capture_profiles": [],
        })
        if bool(entry["diagnostic_only"]) != diagnostic_only:
            raise ValueError(
                f"run {key[0]!r} repeat {key[1]} mixes diagnostic-only and quality cohorts"
            )
        if diagnostic_reason:
            diagnostic_reasons_by_key.setdefault(key, set()).add(diagnostic_reason)
        entry["scenes"].append(str(item.scene_id))
        capture_profile = str(getattr(item, "capture_profile", "summary"))
        if capture_profile in {"deep", "trace"} and not diagnostic_only:
            raise ValueError(
                f"run {key[0]!r} exposes {capture_profile} as quality-eligible"
            )
        entry["capture_profiles"].append(capture_profile)
    run_rows = sorted(runs_by_key.values(), key=lambda row: (row["role"] != "baseline", row["label"], row["repeat"]))
    for row in run_rows:
        row["scenes"] = sorted(set(row["scenes"]))
        row["capture_profiles"] = sorted(set(row["capture_profiles"]))
        key = (str(row["label"]), int(row["repeat"]))
        row["diagnostic_only_reason"] = " | ".join(sorted(diagnostic_reasons_by_key.get(key, set())))

    diagnostic_labels = {
        str(row["label"]) for row in run_rows if bool(row["diagnostic_only"])
    }
    quality_accuracy_ledger = _exclude_diagnostic_quality_rows(
        accuracy_ledger, diagnostic_labels
    )
    quality_accuracy_evidence = _exclude_diagnostic_quality_rows(
        accuracy_evidence, diagnostic_labels
    )
    quality_comparisons = _exclude_diagnostic_quality_rows(
        comparisons, diagnostic_labels
    )
    quality_gates = _exclude_diagnostic_quality_records(gates, diagnostic_labels)
    quality_pareto = _exclude_diagnostic_quality_records(pareto, diagnostic_labels)
    quality_findings = _exclude_diagnostic_quality_records(findings, diagnostic_labels)

    terminal_validation = instrumentation_validation
    if not terminal_validation.empty and "terminal_stage" in terminal_validation.columns:
        terminal_validation = terminal_validation[
            terminal_validation["terminal_stage"].fillna(False).astype(bool)
        ]
    endpoint_sets = terminal_validation
    if not endpoint_sets.empty and "endpoint_dmap_set_checked" in endpoint_sets.columns:
        endpoint_sets = endpoint_sets[
            endpoint_sets["endpoint_dmap_set_checked"].fillna(False).astype(bool)
        ]
        endpoint_sets = endpoint_sets.drop_duplicates(
            [column for column in ("run", "repeat", "scene_id") if column in endpoint_sets.columns]
        )
    invalid_validation = instrumentation_validation[
        ~instrumentation_validation.get("valid", pd.Series(
            False, index=instrumentation_validation.index, dtype=bool
        )).fillna(False).astype(bool)
    ]
    fatal_validation = invalid_validation[
        ~invalid_validation.get("report_generation_allowed", pd.Series(
            False, index=invalid_validation.index, dtype=bool
        )).fillna(False).astype(bool)
    ]
    diagnostic_divergence = invalid_validation[
        invalid_validation.get("process_specialization_divergence", pd.Series(
            False, index=invalid_validation.index, dtype=bool
        )).fillna(False).astype(bool)
    ]
    quality_validation = terminal_validation
    if not quality_validation.empty and "diagnostic_only" in quality_validation.columns:
        quality_validation = quality_validation[
            ~quality_validation["diagnostic_only"].fillna(False).astype(bool)
        ]
    eligible_quality_validation = quality_validation[
        quality_validation.get("quality_comparison_eligible", pd.Series(
            False, index=quality_validation.index, dtype=bool
        )).fillna(False).astype(bool)
    ]
    endpoint_parity_qualified = (
        bool(len(quality_validation))
        and len(eligible_quality_validation) == len(quality_validation)
    )
    production_parity_qualified = (
        bool(len(instrumentation_validation))
        and not len(invalid_validation)
        and endpoint_parity_qualified
    )
    if len(fatal_validation):
        production_qualification_status = "failed"
    elif len(diagnostic_divergence):
        production_qualification_status = "failed_allowed_diagnostic_only"
    elif len(invalid_validation):
        production_qualification_status = "failed"
    elif not endpoint_parity_qualified:
        production_qualification_status = "unqualified_endpoint_parity"
    else:
        production_qualification_status = "passed"
    capture_validation = {
        "validated_frames": int(len(instrumentation_validation)),
        "all_frames_valid": bool(len(instrumentation_validation)) and bool(
            instrumentation_validation.get("valid", pd.Series(dtype=bool)).fillna(False).astype(bool).all()
        ),
        "endpoint_dmap_sets_checked": int(len(endpoint_sets)),
        "endpoint_dmap_sets_bit_exact": int(
            endpoint_sets.get("endpoint_dmap_set_bit_exact", pd.Series(dtype=bool))
            .fillna(False).astype(bool).sum()
        ),
        "endpoint_dmaps_shared": int(
            pd.to_numeric(
                endpoint_sets.get("endpoint_dmap_set_shared_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "maps_summary_frames_checked": int(
            terminal_validation.get("maps_summary_parity_checked", pd.Series(dtype=bool))
            .fillna(False).astype(bool).sum()
        ),
        "report_generation_allowed": bool(len(instrumentation_validation)) and bool(
            instrumentation_validation.get(
                "report_generation_allowed",
                instrumentation_validation.get("valid", pd.Series(dtype=bool)),
            ).fillna(False).astype(bool).all()
        ),
        "production_parity_qualified": production_parity_qualified,
        "diagnostic_process_specialization_divergence_frames": int(
            instrumentation_validation.get(
                "process_specialization_divergence", pd.Series(dtype=bool)
            ).fillna(False).astype(bool).sum()
        ),
        "production_qualification_status": production_qualification_status,
        "quality_comparison_frames": int(len(quality_validation)),
        "quality_comparison_frames_eligible": int(
            len(eligible_quality_validation)
        ),
        "rows": [_portable_record(row, output_dir) for row in records(instrumentation_validation)],
    }

    frame_data = [
        candidate_accounting_record(
            row, str(row.get("candidate_accounting_mode") or "")
        )
        for row in records(frames)
    ]
    candidate_accounting_modes = {
        (
            str(row.get("run")),
            int(row.get("repeat", 0) or 0),
            str(row.get("scene_id")),
            _identity_integer(row.get("image_id")),
        ): str(row.get("candidate_accounting_mode") or "")
        for row in frame_data
        if row.get("candidate_accounting_mode")
    }
    iteration_data = []
    for row in records(iterations):
        row.pop("image_name", None)
        key = (
            str(row.get("run")),
            int(row.get("repeat", 0) or 0),
            str(row.get("scene_id")),
            _identity_integer(row.get("image_id")),
        )
        canonical = canonical_pyramid_record(row)
        canonical = candidate_accounting_record(
            canonical,
            str(canonical.get("candidate_accounting_mode") or "")
            or candidate_accounting_modes.get(key),
        )
        iteration_data.append(_portable_record(canonical, output_dir))
    quality_annotations = _exclude_diagnostic_quality_rows(
        annotations, diagnostic_labels
    )
    annotation_data = [_portable_record(row, output_dir) for row in records(quality_annotations)]
    map_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item in map_rows:
        map_by_frame.setdefault((str(item.get("scene_id")), _identity_integer(item.get("image_id"))), []).append(item)
    scenes: list[dict[str, Any]] = []
    all_frame_keys = sorted({(str(row.get("scene_id")), _identity_integer(row.get("image_id"))) for row in frame_data})
    for scene_id in sorted({key[0] for key in all_frame_keys}):
        scene_frames: list[dict[str, Any]] = []
        for _scene_id, image_id in [key for key in all_frame_keys if key[0] == scene_id]:
            selected_frames = [row for row in frame_data if str(row.get("scene_id")) == scene_id and _identity_integer(row.get("image_id")) == image_id]
            selected_iterations = [row for row in iteration_data if str(row.get("scene_id")) == scene_id and _identity_integer(row.get("image_id")) == image_id]
            selected_annotations = [row for row in annotation_data if str(row.get("scene_id")) == scene_id and _identity_integer(row.get("image_id")) == image_id]
            reference = next((str(row.get("image_name")) for row in selected_frames if row.get("image_name")), None)
            run_frames: list[dict[str, Any]] = []
            for frame_row in selected_frames:
                label = str(frame_row.get("run"))
                repeat = int(frame_row.get("repeat", 0) or 0)
                run_iterations = [
                    row for row in selected_iterations
                    if str(row.get("run")) == label
                    and int(row.get("repeat", 0) or 0) == repeat
                ]
                run_frames.append({
                    "run_id": _run_id(label, repeat),
                    "run": label,
                    "role": frame_row.get("role"),
                    "repeat": repeat,
                    "estimation_stage": frame_row.get("estimation_stage"),
                    "geometric_iteration": frame_row.get("geometric_iteration"),
                    "candidate_accounting_mode": frame_row.get("candidate_accounting_mode"),
                    "frame_key": frame_row.get("safe_image_name") or frame_row.get("image_id"),
                    "metrics": _scalar_metrics(frame_row),
                    "endpoint_valid_depth_coverage": {
                        "value": frame_row.get("endpoint_valid_depth_coverage"),
                        "status": frame_row.get(
                            "endpoint_valid_depth_coverage_status", "unavailable"
                        ),
                        "reason": frame_row.get(
                            "endpoint_valid_depth_coverage_reason"
                        ) or None,
                        "source": relative_path(
                            frame_row.get("endpoint_valid_depth_coverage_source"),
                            output_dir,
                        ) if frame_row.get("endpoint_valid_depth_coverage_source") else None,
                        "algorithm_endpoint": "terminal_production_dmap_after_optional_postprocess",
                    },
                    "ignore_mask": {
                        "status": frame_row.get("ignore_mask_status"),
                        "requested": frame_row.get("ignore_mask_requested"),
                        "loaded": frame_row.get("ignore_mask_loaded"),
                        "rejection_count_available": frame_row.get("ignore_mask_rejection_count_available"),
                        "rejected_pixels": frame_row.get("num_rejected_by_ignore_mask"),
                        "unavailable_reason": frame_row.get("ignore_mask_unavailable_reason") or None,
                    },
                    "view_probability_health": {
                        "schema_valid": _optional_boolean(
                            frame_row.get("view_probability_health_schema_valid")
                        ),
                        "requested": _optional_boolean(
                            frame_row.get("view_probability_health_requested")
                        ),
                        "available": _optional_boolean(
                            frame_row.get("view_probability_health_available")
                        ),
                        "accounting_valid": _optional_boolean(
                            frame_row.get("view_probability_health_accounting_valid")
                        ),
                        "unavailable_reason": (
                            frame_row.get("view_probability_health_unavailable_reason") or None
                        ),
                        "totals": _json_object(
                            frame_row.get("view_probability_health_totals_json")
                        ),
                    },
                    "iterations": [
                        {
                            **row,
                            "deep_link_id": stable_id(
                                "iteration",
                                label,
                                repeat,
                                scene_id,
                                image_id,
                                row.get("estimation_stage"),
                                _model_integer(row.get("geometric_iteration")),
                                row.get("pyramid_level"),
                                row.get("logical_iteration"),
                            ),
                        }
                        for row in run_iterations
                    ],
                })
            frame_map_rows = map_by_frame.get((scene_id, image_id), [])
            frame_iteration_rows = selected_iterations
            logical_iterations = sorted({
                int(row["logical_iteration"])
                for row in [*frame_map_rows, *frame_iteration_rows]
                if row.get("logical_iteration") is not None
            })
            pyramid_levels = sorted({
                int(row["pyramid_level"])
                for row in [*frame_map_rows, *frame_iteration_rows]
                if row.get("pyramid_level") is not None
            })
            logical_iterations_by_pyramid_level = {
                str(level): sorted({
                    int(row["logical_iteration"])
                    for row in [*frame_map_rows, *frame_iteration_rows]
                    if row.get("pyramid_level") == level
                    and row.get("logical_iteration") is not None
                })
                for level in pyramid_levels
            }
            if any(
                row.get("pyramid_level") is None
                for row in [*frame_map_rows, *frame_iteration_rows]
            ):
                logical_iterations_by_pyramid_level["unspecified"] = sorted({
                    int(row["logical_iteration"])
                    for row in [*frame_map_rows, *frame_iteration_rows]
                    if row.get("pyramid_level") is None
                    and row.get("logical_iteration") is not None
                })
            frame_id = stable_id("frame", scene_id, image_id)
            frame_maps = sorted(frame_map_rows, key=_artifact_priority)
            report_reference = materialize_reference_thumbnail(
                reference,
                output_dir,
                scene_id,
                image_id,
                allowed_roots=reference_source_roots(
                    config, experiment_root, selected_frames
                ),
            )
            mechanism_groups: dict[str, list[str]] = {}
            source_view_groups: dict[str, list[str]] = {}
            pyramid_level_groups: dict[str, list[str]] = {}
            for artifact in frame_maps:
                mechanism_groups.setdefault(str(artifact.get("mechanism", "state")), []).append(str(artifact["id"]))
                if artifact.get("source_view_index") is not None:
                    source_view_groups.setdefault(str(artifact["source_view_index"]), []).append(str(artifact["id"]))
                pyramid_key = (
                    str(artifact["pyramid_level"])
                    if artifact.get("pyramid_level") is not None
                    else "unspecified"
                )
                pyramid_level_groups.setdefault(pyramid_key, []).append(str(artifact["id"]))
            scene_frames.append({
                "id": frame_id,
                "deep_link_id": frame_id,
                "scene_id": scene_id,
                "image_id": image_id,
                "label": Path(reference).name if reference else f"image {image_id}",
                "reference": {
                    "available": report_reference is not None,
                    "path": report_reference,
                    "unavailable_reason": None if report_reference else "reference image unavailable",
                },
                "logical_iterations": logical_iterations,
                "pyramid_levels": pyramid_levels,
                "logical_iterations_by_pyramid_level": logical_iterations_by_pyramid_level,
                "run_frames": run_frames,
                "maps": frame_maps,
                "map_groups": {
                    "by_mechanism": mechanism_groups,
                    "by_source_view": source_view_groups,
                    "by_pyramid_level": pyramid_level_groups,
                },
                "annotations": selected_annotations,
            })
        scene_deep_id = stable_id("scene", scene_id)
        scenes.append({
            "id": scene_id,
            "deep_link_id": scene_deep_id,
            "label": scene_id,
            "frames": scene_frames,
            "frame_count": len(scene_frames),
        })

    reference_available = sum(
        bool((frame.get("reference") or {}).get("available"))
        for scene in scenes for frame in scene.get("frames") or []
    )
    reference_total = sum(len(scene.get("frames") or []) for scene in scenes)
    for signal in signal_rows:
        if signal.get("name") == "reference_rgb":
            signal["available_artifacts"] = reference_available
            signal["unavailable_artifacts"] = reference_total - reference_available
            break

    quality_run_rows = [
        row for row in run_rows if bool(row["quality_comparison_eligible"])
    ]
    baseline_labels = sorted({
        row["label"] for row in quality_run_rows if row["role"] == "baseline"
    })
    baseline = baseline_labels[0] if baseline_labels else (
        quality_run_rows[0]["label"] if quality_run_rows else ""
    )
    candidates = sorted({
        row["label"] for row in quality_run_rows if row["label"] != baseline
    })
    regressions = _paired_regressions(frames, baseline, candidates, metric_specs, "frame")
    regressions.extend(_paired_regressions(
        endpoint_performance, baseline, candidates, metric_specs, "performance"
    ))
    post_annotations = annotations
    if not annotations.empty and "stage" in annotations.columns:
        post_annotations = annotations[annotations["stage"] == "post_filter"]
    regressions.extend(_paired_regressions(post_annotations, baseline, candidates, metric_specs, "annotation"))
    regressions.sort(key=lambda row: (-float(row["regression_score"]), row["metric"], row["scene_id"], row["image_id"]))

    mechanism_impact = []
    for mechanism in sorted({str(row.get("mechanism", "state")) for row in map_rows}):
        selected = [row for row in map_rows if str(row.get("mechanism", "state")) == mechanism]
        mechanism_impact.append({
            "mechanism": mechanism,
            "signals": sorted({str(row.get("signal")) for row in selected if row.get("signal")}),
            "available_artifacts": sum(bool(row.get("available")) for row in selected),
            "unavailable_artifacts": sum(not bool(row.get("available")) for row in selected),
            "exact_artifacts": sum(
                bool(row.get("available"))
                and row.get("measurement_quality") in {"exact", "derived_exact"}
                for row in selected
            ),
            "proxy_artifacts": sum(
                bool(row.get("available")) and row.get("measurement_quality") == "proxy"
                for row in selected
            ),
            "runs": sorted({str(row.get("run")) for row in selected if row.get("run")}),
            "frames": len({(str(row.get("scene_id")), _identity_integer(row.get("image_id"))) for row in selected}),
            "pyramid_levels": sorted({
                int(row["pyramid_level"])
                for row in selected if row.get("pyramid_level") is not None
            }),
        })

    model = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "viewer_schema_version": VIEWER_SCHEMA_VERSION,
        "investigation_guide": json_value(investigation_guide_model),
        "contract": {
            "canonical_output": "Markdown",
            "logical_state_granularity": "initialization and complete PatchMatch iterations",
            "comparison_alignment_modes": ["same logical iteration", "final state per run"],
            "pyramid_level_field": "pyramid_level",
            "pyramid_level_aliases_accepted": ["scale_level", "scale_number"],
            "checkerboard_policy": "checkerboard phases are exposed only for timings",
            "measurement_quality_values": ["exact", "derived_exact", "proxy", "unavailable"],
            "capture_profiles": ["endpoint", "summary", "prefilter", "deep", "trace"],
            "capture_profile_contract": {
                "endpoint": {
                    "available": True,
                    "executable": "DensifyPointCloud",
                    "device_observer_work": False,
                },
                "summary": {
                    "available": True,
                    "storage_light": True,
                    "compute_light": False,
                    "reason": "full-frame snapshots and diagnostic rescoring still execute",
                },
                "light": {
                    "available": False,
                    "reason": "a no-extra-kernel logical-iteration profile is not implemented",
                },
                "prefilter": {
                    "available": True,
                    "runtime_level": "summary",
                    "process_specialization": "Process<false>",
                    "storage_policy": "bounded terminal pre-filter depth snapshot",
                },
                "deep": {"available": True, "runtime_level": "maps"},
                "trace": {
                    "available": True,
                    "runtime_level": "maps",
                    "process_specialization": "Process<true>",
                    "compact_exact_trace_available": False,
                    "storage_policy": "full-frame exact maps plus selected trace rows",
                },
            },
            "component_registry": component_registry.SCHEMA_NAME,
            "unavailable_signals_are_explicit": True,
            "candidate_accounting_contract": {
                "unavailable_mode": CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE,
                "unavailable_metrics": list(CANDIDATE_ACCOUNTING_METRICS),
                "presentation": "null and explicitly labeled unavailable; never zero-filled",
                "raw_capture_artifacts_preserved": True,
                "exact_production_hot_kernel_rows_preserved": True,
            },
            "optional_extensions": json_value(extension_contracts),
            "annotation_model_switch_contract": {
                "schema_version": 1,
                "line_direction_delta_deg": 5.0,
                "line_extent_delta_m": 0.025,
                "line_extent_relative_delta": 0.25,
                "plane_normal_delta_deg": 5.0,
                "plane_position_delta_m": 0.010,
                "current_fit_warning": (
                    "baseline and candidate current-fit metrics may represent competing "
                    "RANSAC models when a large switch is flagged"
                ),
                "fixed_model_metric_prefix": "baseline_model_on_candidate_",
            },
            "endpoint_valid_depth_coverage_contract": {
                "schema_version": 1,
                "frame_metric": "endpoint_valid_depth_coverage",
                "ledger_delta": "endpoint_valid_depth_coverage_delta",
                "source": "terminal production DMAP after configured optional postprocess",
                "validity": "finite depth greater than zero",
                "legacy_estimator_delta_preserved_as": "valid_coverage_delta",
                "unavailable_values": "null with status and reason",
            },
            "runtime_authority_contract": {
                "schema_version": 1,
                "decision_authority": "production_endpoint_wall_clock",
                "clock": "time.monotonic_ns",
                "aggregation_unit": "scene_repeat",
                "scope": "full_densify_process",
                "observer_kernel_timings": "diagnostic_only",
            },
            "summary_unavailable_signal_contract": json_value(
                summary_signal_contract or {
                    "schema_name": "openmvs.dmap.summary_unavailable_signals",
                    "schema_version": 1,
                    "logical_signals": [],
                    "initialization_signals": [],
                    "final_signals": [],
                }
            ),
            "paths_are_relative_to": "report directory",
        },
        "capture_validation": capture_validation,
        "capture_profile_coverage": paths_report_relative(
            capture_profile_coverage or {
                "schema_name": "openmvs.dmap.capture_profile_coverage",
                "schema_version": 1,
                "requested_profiles": [],
                "profiles": [
                    {
                        "profile": profile,
                        "requested": False,
                        "status": "not_requested",
                        "expected_units": 0,
                        "complete_units": 0,
                        "failed_units": 0,
                        "unavailable_units": 0,
                        "process_specialization": (
                            "Process<true>" if profile in {"deep", "trace"}
                            else "Process<false>"
                        ),
                        "quality_authority": "not_recorded",
                    }
                    for profile in ("endpoint", "summary", "prefilter", "deep", "trace")
                ],
                "units": [],
            },
            output_dir,
        ),
        "experiment": {
            "name": str(config.get("name") or experiment_root.name),
            "root": relative_path(experiment_root, output_dir),
            "source_config": relative_path(config.get("_config_path"), output_dir),
        },
        "entrypoints": {
            "markdown": "01_development_report.md",
            "static_html": "01_development_report.html",
            "investigation_html": "02_investigation.html",
            "model": "report_model.json",
        },
        "runs": run_rows,
        "component_registry": component_registry_model,
        "drilldowns": drilldowns,
        "signals": signal_rows,
        "scenes": scenes,
        "aggregates": {
            "runtime": {
                "authority": "production_endpoint_wall_clock",
                "clock": "time.monotonic_ns",
                "scope": "full_densify_process",
                "aggregation_unit": "scene_repeat",
                "endpoint_rows": records(endpoint_performance),
                "observer_kernel_diagnostics": records(performance),
                "observer_timing_authority": "diagnostic_only",
            },
            "accuracy_ledger": records(quality_accuracy_ledger),
            "accuracy_evidence": records(quality_accuracy_evidence),
            "annotation_model_stability": records(model_stability),
            "annotation_model_switch_summary": {
                "paired_models": int(len(model_stability)),
                "large_switches": int(
                    model_stability.get(
                        "large_model_switch", pd.Series(False, index=model_stability.index)
                    ).fillna(False).astype(bool).sum()
                ),
                "large_switches_near_candidate_regression": int(
                    model_stability.get(
                        "near_candidate_regression", pd.Series(False, index=model_stability.index)
                    ).fillna(False).astype(bool).sum()
                ),
                "fixed_baseline_model_cross_evaluations": int(
                    (
                        model_stability.get(
                            "baseline_model_on_candidate_status",
                            pd.Series("", index=model_stability.index),
                        ).astype(str) == "available"
                    ).sum()
                ),
            },
            "gates": json_value(quality_gates),
            "pareto": json_value(quality_pareto),
            "findings": json_value(quality_findings),
            "comparisons": records(quality_comparisons),
            "regressions": regressions,
            "mechanism_impact": mechanism_impact,
        },
        "mechanics": {
            "reference_patch_layouts": reference_patch_layouts,
            "observer_kernel_timing_diagnostics": records(performance),
            "texture_stratification": texture_stratification,
            "accepted_gain_census": accepted_gain_census,
            "low_texture_update_hysteresis": low_texture_hysteresis,
            "exact_cost_evolution": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(exact_cost_evolution)],
            "exact_observability": [
                _portable_record(canonical_pyramid_record({key: value for key, value in row.items() if key != "states_json"}), output_dir)
                for row in records(exact_observability)
            ],
            "exact_iterations": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(exact_iterations)],
            "exact_views": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(exact_views)],
            "cpu_view_candidates": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(cpu_view_candidates)],
            "cpu_estimation_selection": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(cpu_estimation_selection)],
            "postprocess_filters": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(postprocess_filters)],
            "confidence_adjustment": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(confidence_adjustment)],
            "cuda_resource_plans": [
                canonical_pyramid_record(row) for row in cuda_resource_plan_records
            ],
            "coarse_compatibility_map_availability": (
                coarse_compatibility_map_availability
            ),
            "filter_resource_plans": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(filter_resource_plans)],
            "resource_plan_validation": [_portable_record(canonical_pyramid_record(row), output_dir) for row in records(resource_plan_validation)],
            "availability": {
                "reference_patch_layout": any(
                    row.get("available") is True
                    for row in reference_patch_layouts
                ),
                "exact_hot_kernel": not exact_iterations.empty,
                "low_texture_update_hysteresis": bool(
                    low_texture_hysteresis.get("rows")
                ),
                "exact_per_view": not exact_views.empty,
                "cpu_view_ranking": not cpu_view_candidates.empty,
                "cpu_estimation_selection": not cpu_estimation_selection.empty,
                "postprocess_filters": (
                    not postprocess_filters.empty
                    and bool(postprocess_filters.get(
                        "executed", pd.Series(False, index=postprocess_filters.index)
                    ).fillna(False).astype(bool).any())
                ),
                "postprocess_filter_contract": (
                    not postprocess_filters.empty
                    and bool((postprocess_filters.get("artifact_status", pd.Series(dtype=str)) != "unavailable").any())
                ),
                "confidence_adjustment": (
                    not confidence_adjustment.empty
                    and bool(confidence_adjustment.get(
                        "executed", pd.Series(False, index=confidence_adjustment.index)
                    ).fillna(False).astype(bool).any())
                ),
                "confidence_adjustment_contract": (
                    not confidence_adjustment.empty
                    and bool((confidence_adjustment.get("artifact_status", pd.Series(dtype=str)) != "unavailable").any())
                ),
                "resource_plans_valid": (
                    not resource_plan_validation.empty
                    and bool((
                        ~resource_plan_validation.get("required", pd.Series(False, index=resource_plan_validation.index)).fillna(False).astype(bool)
                        | resource_plan_validation.get("valid", pd.Series(False, index=resource_plan_validation.index)).fillna(False).astype(bool)
                    ).all())
                ),
            },
        },
        "map_catalog_summary": {
            "artifacts": len(map_rows),
            "available": sum(bool(row.get("available")) for row in map_rows),
            "unavailable": sum(not bool(row.get("available")) for row in map_rows),
            "pixel_data": pixel_budget,
        },
        "inventory": _portable_inventory(inventory, output_dir),
    }
    if evidence_context is not None:
        validation = validate_evidence_context(evidence_context)
        if not validation["valid"]:
            raise ValueError(
                "invalid report evidence context: " + "; ".join(validation["errors"])
            )
        model["evidence_context"] = json_value(evidence_context)
    return model


def write_report_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(model), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def validate_report_model(model: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Validate model structure, deep links, local artifacts, and iteration semantics."""
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": json_value(detail)})

    model_schema_version = model.get("schema_version")
    add(
        "schema_identity",
        model.get("schema_name") == SCHEMA_NAME and model_schema_version in SUPPORTED_SCHEMA_VERSIONS,
        {
            "schema_name": model.get("schema_name"),
            "schema_version": model_schema_version,
            "supported_schema_versions": list(SUPPORTED_SCHEMA_VERSIONS),
        },
    )
    evidence_context = model.get("evidence_context")
    evidence_validation = (
        validate_evidence_context(evidence_context)
        if evidence_context is not None
        else {
            "schema_name": EVIDENCE_CONTEXT_SCHEMA_NAME,
            "schema_version": EVIDENCE_CONTEXT_SCHEMA_VERSION,
            "valid": True,
            "errors": [],
            "absent": True,
        }
    )
    add(
        "evidence_context",
        evidence_validation["valid"],
        evidence_validation,
    )
    run_candidate_domain = {
        evidence_context_run_candidate(row.get("label"))
        for row in model.get("runs") or []
        if isinstance(row, dict) and str(row.get("label") or "").strip()
    }
    captured_candidates = {
        evidence_context_run_candidate(row.get("candidate"))
        for row in (
            ((evidence_context or {}).get("mechanics_authority") or {}).get(
                "candidate_coverage"
            ) or []
        )
        if isinstance(row, dict) and row.get("coverage") == "captured"
        and str(row.get("candidate") or "").strip()
    }
    missing_captured_candidates = sorted(captured_candidates - run_candidate_domain)
    add(
        "evidence_context_captured_candidate_domain",
        not missing_captured_candidates,
        {
            "normalization": "strip one exact trailing ' [deep]' suffix only",
            "run_candidate_domain": sorted(run_candidate_domain),
            "captured_candidates": sorted(captured_candidates),
            "missing_captured_candidates": missing_captured_candidates,
        },
    )
    registry = model.get("component_registry") or {}
    registry_required = isinstance(model_schema_version, int) and model_schema_version >= 3
    registry_errors = component_registry.validate_registry(registry) if registry else ["component registry missing"]
    add(
        "component_registry",
        not registry_required or not registry_errors,
        {
            "required": registry_required,
            "schema_name": registry.get("schema_name") if isinstance(registry, dict) else None,
            "schema_version": registry.get("schema_version") if isinstance(registry, dict) else None,
            "signal_count": len(registry.get("signals") or []) if isinstance(registry, dict) else 0,
            "errors": registry_errors,
        },
    )
    registry_signal_ids = {
        str(row.get("signal_id")) for row in registry.get("signals") or []
        if isinstance(row, dict) and row.get("signal_id")
    } if isinstance(registry, dict) else set()
    declared_signal_ids = {
        str(row.get("name")) for row in model.get("signals") or []
        if isinstance(row, dict) and row.get("name")
    }
    add(
        "component_registry_covers_signals",
        declared_signal_ids <= registry_signal_ids,
        {"missing": sorted(declared_signal_ids - registry_signal_ids)},
    )
    mechanics = model.get("mechanics") or {}
    patch_layout_rows = mechanics.get("reference_patch_layouts") or []
    patch_layout_errors: list[str] = []
    patch_layout_identities: set[tuple[Any, ...]] = set()
    if not isinstance(patch_layout_rows, list):
        patch_layout_errors.append("reference_patch_layouts is not a list")
        patch_layout_rows = []
    for index, row in enumerate(patch_layout_rows):
        if not isinstance(row, dict):
            patch_layout_errors.append(f"rows[{index}] is not an object")
            continue
        identity = (
            row.get("run"), _identity_integer(row.get("repeat")),
            row.get("scene_id"), row.get("estimation_stage"),
            _model_integer(row.get("geometric_iteration")),
        )
        if identity in patch_layout_identities:
            patch_layout_errors.append(f"rows[{index}] duplicates identity {identity}")
        patch_layout_identities.add(identity)
        if row.get("contract_valid") is not True:
            patch_layout_errors.append(f"rows[{index}] has an invalid claimed contract")
        if row.get("available") is True:
            if row.get("contract_claimed") is not True:
                patch_layout_errors.append(
                    f"rows[{index}] exposes an unclaimed layout as available"
                )
            normalized, reason = normalize_reference_patch_layout(row.get("layout"))
            if normalized is None:
                patch_layout_errors.append(f"rows[{index}] is malformed: {reason}")
            elif normalized != row.get("layout"):
                patch_layout_errors.append(f"rows[{index}] layout is not normalized")
            if row.get("measurement_quality") != "exact":
                patch_layout_errors.append(f"rows[{index}] available layout is not exact")
            if row.get("visualization_quality") != "derived_exact":
                patch_layout_errors.append(
                    f"rows[{index}] available visualization is not derived_exact"
                )
            if row.get("unavailable_reason") is not None:
                patch_layout_errors.append(
                    f"rows[{index}] available layout has an unavailable reason"
                )
        elif not str(row.get("unavailable_reason") or "").strip():
            patch_layout_errors.append(
                f"rows[{index}] unavailable layout has no reason"
            )
    grid_extent_errors: list[str] = []
    for index, row in enumerate(mechanics.get("cuda_resource_plans") or []):
        if not isinstance(row, dict) or "grid_extent" not in row:
            continue
        extent = row.get("grid_extent")
        if not isinstance(extent, dict):
            grid_extent_errors.append(f"cuda_resource_plans[{index}] grid_extent is invalid")
            continue
        if extent.get("available") is True:
            width = _model_integer(extent.get("width"))
            height = _model_integer(extent.get("height"))
            if width is None or width <= 0 or height is None or height <= 0:
                grid_extent_errors.append(
                    f"cuda_resource_plans[{index}] available grid extent is non-positive"
                )
            if extent.get("measurement_quality") != "exact":
                grid_extent_errors.append(
                    f"cuda_resource_plans[{index}] available grid extent is not exact"
                )
        elif not str(extent.get("unavailable_reason") or "").strip():
            grid_extent_errors.append(
                f"cuda_resource_plans[{index}] unavailable grid extent has no reason"
            )
    add(
        "reference_patch_layout_contract",
        not patch_layout_errors and not grid_extent_errors,
        {
            "layout_records": len(patch_layout_rows),
            "available_layouts": sum(
                row.get("available") is True
                for row in patch_layout_rows if isinstance(row, dict)
            ),
            "layout_errors": patch_layout_errors,
            "grid_extent_errors": grid_extent_errors,
        },
    )
    coarse_availability = mechanics.get(
        "coarse_compatibility_map_availability"
    ) or []
    coarse_errors: list[str] = []
    coarse_identities: set[tuple[Any, ...]] = set()
    for index, row in enumerate(coarse_availability):
        if not isinstance(row, dict):
            coarse_errors.append(f"rows[{index}] is not an object")
            continue
        identity = (
            row.get("run"), _identity_integer(row.get("repeat")),
            row.get("scene_id"), _identity_integer(row.get("image_id")),
            row.get("estimation_stage"),
            _identity_integer(row.get("geometric_iteration")),
            _identity_integer(row.get("pyramid_level")),
        )
        if identity in coarse_identities:
            coarse_errors.append(f"rows[{index}] duplicates {identity!r}")
        coarse_identities.add(identity)
        if (
            identity[-1] <= 0
            or row.get("measurement_basis")
            != "resource_plan.compatibility_map_contract"
            or not isinstance(row.get("update_source_map_expected"), bool)
            or not isinstance(row.get("cost_map_expected"), bool)
            or (
                row.get("cost_map_expected") is False
                and (
                    row.get("cost_map_available") is not False
                    or not str(row.get("cost_map_unavailable_reason") or "").strip()
                )
            )
        ):
            coarse_errors.append(f"rows[{index}] has an invalid availability contract")
    schema_v4_coarse_plans = [
        row for row in mechanics.get("cuda_resource_plans") or []
        if _identity_integer(row.get("schema_version")) == 4
        and (canonical_pyramid_level(row) or 0) > 0
    ]
    declared_coarse_contracts = [
        row for row in schema_v4_coarse_plans
        if isinstance(row.get("compatibility_map_contract"), dict)
    ]
    if len(declared_coarse_contracts) != len(schema_v4_coarse_plans):
        coarse_errors.append(
            "one or more schema-v4 coarse resource plans lost their compatibility contract"
        )
    if len(coarse_availability) != len(schema_v4_coarse_plans):
        coarse_errors.append(
            "coarse availability cardinality does not match preserved resource-plan contracts"
        )
    add(
        "coarse_compatibility_map_availability",
        not coarse_errors,
        {
            "rows": len(coarse_availability),
            "declared_contracts": len(declared_coarse_contracts),
            "schema_v4_coarse_plans": len(schema_v4_coarse_plans),
            "errors": coarse_errors,
        },
    )
    profile_coverage = model.get("capture_profile_coverage") or {}
    profile_coverage_required = (
        isinstance(model_schema_version, int) and model_schema_version >= 3
    )
    profile_errors: list[str] = []
    canonical_profiles = ("endpoint", "summary", "prefilter", "deep", "trace")
    profile_rows = profile_coverage.get("profiles") or []
    coverage_units = profile_coverage.get("units") or []
    if profile_coverage_required or profile_coverage:
        if (
            profile_coverage.get("schema_name")
            != "openmvs.dmap.capture_profile_coverage"
            or profile_coverage.get("schema_version") != 1
        ):
            profile_errors.append("invalid capture-profile coverage schema")
        if [
            row.get("profile") for row in profile_rows if isinstance(row, dict)
        ] != list(canonical_profiles):
            profile_errors.append(
                "capture-profile summary must contain the canonical ordered profiles"
            )
    valid_profile_statuses = {"complete", "failed", "unavailable", "not_requested"}
    unit_identities: set[tuple[str, int, str, str]] = set()
    for index, unit in enumerate(coverage_units):
        if not isinstance(unit, dict):
            profile_errors.append(f"units[{index}] is not an object")
            continue
        identity = (
            str(unit.get("configured_run")),
            _identity_integer(unit.get("repeat")),
            str(unit.get("scene_id")),
            str(unit.get("capture_profile")),
        )
        if identity in unit_identities:
            profile_errors.append(f"units[{index}] duplicates {identity!r}")
        unit_identities.add(identity)
        if identity[3] not in canonical_profiles or unit.get("status") not in valid_profile_statuses:
            profile_errors.append(f"units[{index}] has invalid profile/status")
        closure = unit.get("artifact_closure")
        if closure is not None and not (
            isinstance(closure, dict)
            and closure.get("status") in {
                "verified", "legacy-unverified", "invalid", "unavailable",
            }
            and isinstance(closure.get("required"), bool)
            and isinstance(closure.get("reason"), str)
            and (
                closure.get("path") is None
                or isinstance(closure.get("path"), str)
            )
        ):
            profile_errors.append(f"units[{index}] has malformed artifact closure")
        if (
            isinstance(closure, dict)
            and unit.get("status") == "complete"
            and closure.get("status") in {"invalid", "unavailable"}
        ):
            profile_errors.append(
                f"units[{index}] is complete with an invalid artifact closure"
            )
        links = unit.get("evidence_links") or []
        frames_value = unit.get("frames") or []
        if not all(
            isinstance(link, dict) and isinstance(link.get("label"), str)
            and isinstance(link.get("path"), str)
            for link in links
        ) or not all(isinstance(frame, dict) for frame in frames_value):
            profile_errors.append(f"units[{index}] has malformed evidence links")
        if identity[3] == "prefilter" and unit.get("status") == "complete":
            for frame_index, frame in enumerate(frames_value):
                labels = {
                    str(link.get("label")) for link in frame.get("evidence_links") or []
                    if isinstance(link, dict)
                }
                required = {
                    "prefilter_manifest.json", "prefilter_capture_complete.json",
                    "depth_final_before_filter",
                }
                if not required <= labels:
                    profile_errors.append(
                        f"units[{index}].frames[{frame_index}] lacks exact prefilter evidence"
                    )
    for profile_index, profile_row in enumerate(profile_rows):
        if not isinstance(profile_row, dict):
            continue
        profile = str(profile_row.get("profile"))
        selected = [unit for unit in coverage_units if unit.get("capture_profile") == profile]
        expected_counts = {
            "expected_units": len(selected),
            "complete_units": sum(unit.get("status") == "complete" for unit in selected),
            "failed_units": sum(unit.get("status") == "failed" for unit in selected),
            "unavailable_units": sum(unit.get("status") == "unavailable" for unit in selected),
        }
        if any(profile_row.get(key) != value for key, value in expected_counts.items()):
            profile_errors.append(f"profiles[{profile_index}] aggregate counts are inconsistent")
        if profile_row.get("status") not in valid_profile_statuses:
            profile_errors.append(f"profiles[{profile_index}] status is invalid")
    for run_index, run in enumerate(model.get("runs") or []):
        capture_profiles = set(run.get("capture_profiles") or [])
        if capture_profiles & {"deep", "trace"} and run.get("quality_comparison_eligible") is not False:
            profile_errors.append(f"runs[{run_index}] makes deep/trace evidence quality eligible")
    add(
        "capture_profile_coverage",
        not profile_errors,
        {"profiles": len(profile_rows), "units": len(coverage_units), "errors": profile_errors},
    )
    drilldowns = model.get("drilldowns") or {}
    drilldown_errors = []
    if drilldowns and drilldowns.get("schema_name") != "openmvs.dmap.drilldown_index":
        drilldown_errors.append("invalid schema_name")
    for index, row in enumerate(drilldowns.get("entries") or []):
        if row.get("status") not in {"requested", "incomplete", "complete", "invalid_request"}:
            drilldown_errors.append(f"entries[{index}].status is invalid")
        if row.get("status") != "invalid_request" and not row.get("request_sha256"):
            drilldown_errors.append(f"entries[{index}].request_sha256 is missing")
        if row.get("status") != "complete":
            continue
        if row.get("embedding_errors"):
            drilldown_errors.append(f"entries[{index}] has embedding errors")
        request_metadata = row.get("request_metadata") or {}
        request_pixels = request_metadata.get("pixels") or []
        if (
            request_metadata.get("available") is not True
            or request_metadata.get("request_sha256") != row.get("request_sha256")
            or request_metadata.get("capture_profile") != row.get("capture_profile")
            or request_metadata.get("scene_id") != row.get("scene_id")
            or _trace_integer(request_metadata.get("image_id"))
            != _trace_integer(row.get("image_id"))
            or _trace_integer(request_metadata.get("trace_pixel_count"))
            != _trace_integer(row.get("trace_pixel_count"))
            or request_metadata.get("run_labels") != row.get("run_labels")
            or len(request_pixels) != _trace_integer(row.get("trace_pixel_count"))
            or any(
                _trace_integer(pixel.get("x")) is None
                or _trace_integer(pixel.get("y")) is None
                for pixel in request_pixels if isinstance(pixel, dict)
            )
            or not all(isinstance(pixel, dict) for pixel in request_pixels)
            or request_metadata.get("error")
        ):
            drilldown_errors.append(f"entries[{index}].request_metadata is invalid")
        execution_metadata = row.get("execution_metadata") or {}
        if (
            execution_metadata.get("available") is not True
            or execution_metadata.get("schema_name") != "openmvs.dmap.drilldown_executions"
            or execution_metadata.get("schema_version") != 1
            or execution_metadata.get("request_sha256") != row.get("request_sha256")
            or execution_metadata.get("execution_count") != len(execution_metadata.get("executions") or [])
            or not execution_metadata.get("executions")
            or any(
                not isinstance(execution.get("run"), str)
                or not isinstance(execution.get("scene_id"), str)
                or _trace_integer(execution.get("return_code")) != 0
                for execution in execution_metadata.get("executions") or []
            )
            or execution_metadata.get("error")
        ):
            drilldown_errors.append(f"entries[{index}].execution_metadata is invalid")
        trace_data = row.get("trace_data") or {}
        if (
            trace_data.get("schema_name") != COMPLETED_TRACE_SCHEMA_NAME
            or trace_data.get("schema_version") != COMPLETED_TRACE_SCHEMA_VERSION
            or trace_data.get("row_limit") != MAX_COMPLETED_TRACE_ROWS
            or trace_data.get("row_count") != len(trace_data.get("rows") or [])
            or trace_data.get("source_count") != len(trace_data.get("sources") or [])
            or _trace_integer(trace_data.get("row_count")) is None
            or int(trace_data.get("row_count")) > MAX_COMPLETED_TRACE_ROWS
            or trace_data.get("source_quality_counts") != {
                quality: sum(
                    1 for trace in trace_data.get("rows") or []
                    if trace.get("source_quality") == quality
                )
                for quality in ("exact", "proxy")
            }
            or trace_data.get("all_source_attribution_exact") != (
                bool(trace_data.get("rows"))
                and all(
                    trace.get("source_quality") == "exact"
                    for trace in trace_data.get("rows") or []
                )
            )
        ):
            drilldown_errors.append(f"entries[{index}].trace_data contract is invalid")
            continue
        if row.get("capture_profile") != "trace":
            if trace_data.get("available") or not trace_data.get("unavailable_reason"):
                drilldown_errors.append(f"entries[{index}].trace_data must be explicitly unavailable")
            continue
        if (
            trace_data.get("available") is not True
            or not trace_data.get("rows")
            or trace_data.get("errors")
            or (trace_data.get("truncated") and trace_data.get("row_count") != MAX_COMPLETED_TRACE_ROWS)
        ):
            drilldown_errors.append(f"entries[{index}].trace_data is unavailable or malformed")
        source_paths = {
            str(source.get("source_path"))
            for source in trace_data.get("sources") or []
            if source.get("available") and source.get("contained") and source.get("source_path")
        }
        source_by_path = {
            str(source.get("source_path")): source
            for source in trace_data.get("sources") or []
            if source.get("source_path")
        }
        for source_index, source in enumerate(trace_data.get("sources") or []):
            evidence = source.get("exact_capture_evidence") or {}
            closure = source.get("artifact_closure") or {}
            coverage = source.get("coverage") or {}
            source_rows = [
                trace for trace in trace_data.get("rows") or []
                if str(trace.get("trace_source_path")) == str(source.get("source_path"))
            ]
            if (
                not source.get("contained")
                or not source.get("available")
                or not source.get("source_path")
                or source.get("error")
                or _trace_integer(source.get("row_count")) is None
                or int(source.get("row_count")) < 0
                or source.get("row_count") != len(source_rows)
                or source.get("source_quality_counts") != {
                    quality: sum(
                        1 for trace in source_rows
                        if trace.get("source_quality") == quality
                    )
                    for quality in ("exact", "proxy")
                }
                or not isinstance(evidence.get("valid"), bool)
                or evidence.get("measurement_basis")
                != "schema_v4_exact_map_completion"
                or _trace_integer(evidence.get("matching_frame_count")) is None
                or _trace_integer(evidence.get("valid_frame_count")) is None
                or not isinstance(evidence.get("command_maps_write_maps"), bool)
                or not isinstance(evidence.get("errors"), list)
                or not isinstance(evidence.get("expected_states"), list)
                or closure.get("status") not in {
                    "verified", "legacy-unverified"
                }
                or not isinstance(closure.get("valid"), bool)
                or not isinstance(closure.get("required"), bool)
                or not isinstance(closure.get("reason"), str)
                or (
                    closure.get("status") == "verified"
                    and (
                        closure.get("valid") is not True
                        or not closure.get("source_path")
                        or _trace_integer(closure.get("file_count")) is None
                        or _trace_integer(closure.get("total_bytes")) is None
                        or not isinstance(closure.get("files_sha256"), str)
                        or len(closure.get("files_sha256")) != 64
                    )
                )
                or (
                    evidence.get("declared_exact_topology") is True
                    and closure.get("status") != "verified"
                )
                or source.get("estimation_stage") not in {
                    "photometric", "geometric_consistency"
                }
                or source.get("estimation_stage") != evidence.get("estimation_stage")
                or source.get("geometric_iteration") != evidence.get("geometric_iteration")
                or source.get("stage_key") != (
                    f"geometric_consistency:{source.get('geometric_iteration')}"
                    if source.get("estimation_stage") == "geometric_consistency"
                    else "photometric"
                )
                or (
                    source.get("estimation_stage") == "photometric"
                    and source.get("geometric_iteration") is not None
                )
                or (
                    source.get("estimation_stage") == "geometric_consistency"
                    and (
                        _trace_integer(source.get("geometric_iteration")) is None
                        or int(source.get("geometric_iteration")) < 0
                    )
                )
                or coverage.get("requested_pixel_count") != len(request_pixels)
                or coverage.get("observed_pixel_count") != len(request_pixels)
                or coverage.get("missing_pixel_count") != 0
                or coverage.get("unexpected_pixel_count") != 0
                or coverage.get("missing_state_count") != 0
            ):
                drilldown_errors.append(
                    f"entries[{index}].trace_data.sources[{source_index}] is invalid"
                )
        for trace_index, trace in enumerate(trace_data.get("rows") or []):
            arrays = trace.get("arrays") or {}
            trace_source = source_by_path.get(str(trace.get("trace_source_path"))) or {}
            exact_evidence = trace_source.get("exact_capture_evidence") or {}
            request_identity = trace.get("request_identity") or {}
            expected_states = {
                (state.get("pyramid_level"), state.get("logical_iteration"))
                for state in exact_evidence.get("expected_states") or []
                if isinstance(state, dict)
            }
            trace_valid = (
                isinstance(trace.get("run"), str) and bool(trace.get("run"))
                and isinstance(trace.get("scene_id"), str) and bool(trace.get("scene_id"))
                and _trace_integer(trace.get("image_id")) is not None
                and _trace_integer(trace.get("x")) is not None
                and _trace_integer(trace.get("y")) is not None
                and _trace_integer(trace.get("logical_iteration")) is not None
                and int(trace.get("logical_iteration")) >= -1
                and trace.get("estimation_stage") == trace_source.get("estimation_stage")
                and trace.get("geometric_iteration") == trace_source.get("geometric_iteration")
                and _trace_integer(trace.get("pyramid_level")) is not None
                and int(trace.get("pyramid_level")) >= 0
                and _trace_integer(trace.get("trace_index")) is not None
                and int(trace.get("trace_index")) >= 0
                and (
                    not expected_states
                    or (
                        trace.get("pyramid_level"), trace.get("logical_iteration")
                    ) in expected_states
                )
                and trace.get("stage") in {"initialization", "iteration"}
                and trace.get("source_quality") in {"exact", "proxy"}
                and isinstance(trace.get("measurement_basis"), str)
                and bool(trace.get("measurement_basis"))
                and trace.get("source_quality_origin") in {
                    "trace_row_declaration",
                    "v4_map_completion_inference",
                    "legacy_default",
                    "conservative_downgrade",
                }
                and (
                    trace.get("source_quality") != "exact"
                    or exact_evidence.get("valid") is True
                )
                and isinstance(trace.get("cost"), dict)
                and isinstance(trace.get("depth"), dict)
                and isinstance(trace.get("normal"), dict)
                and isinstance(trace.get("view"), dict)
                and isinstance(arrays, dict)
                and _trace_integer(request_identity.get("request_index")) is not None
                and _trace_integer(request_identity.get("x")) is not None
                and _trace_integer(request_identity.get("y")) is not None
                and request_identity.get("trace_x") == trace.get("x")
                and request_identity.get("trace_y") == trace.get("y")
                and isinstance(request_identity.get("alias_request_indices"), list)
                and isinstance(request_identity.get("alias_coordinates"), list)
                and len(request_identity.get("alias_request_indices"))
                == len(request_identity.get("alias_coordinates"))
                and str(trace.get("trace_source_path")) in source_paths
                and all(
                    isinstance(values, list) and len(values) <= MAX_TRACE_ARRAY_VALUES
                    for values in arrays.values()
                )
            )
            if not trace_valid:
                drilldown_errors.append(
                    f"entries[{index}].trace_data.rows[{trace_index}] is invalid"
                )
    add(
        "drilldown_index",
        not drilldown_errors,
        {"entries": len(drilldowns.get("entries") or []), "errors": drilldown_errors},
    )
    guide = model.get("investigation_guide") or {}
    guide_required = isinstance(model_schema_version, int) and model_schema_version >= 2
    recipe_keys = {
        str(recipe.get("key")) for recipe in guide.get("recipes") or []
        if isinstance(recipe, dict) and recipe.get("key")
    }
    required_recipe_keys = {
        "overview", "cost", "propagation", "view_selection", "texture", "patch", "deformable_patch", "multiscale",
    }
    guide_valid = (
        guide.get("schema_name") == "openmvs.dmap.investigation_guide"
        and guide.get("schema_version") == 1
        and isinstance(guide.get("quick_start"), dict)
        and required_recipe_keys.issubset(recipe_keys)
    )
    add(
        "investigation_guide",
        not guide_required or guide_valid,
        {
            "required": guide_required,
            "schema_name": guide.get("schema_name"),
            "schema_version": guide.get("schema_version"),
            "recipe_keys": sorted(recipe_keys),
            "missing_recipe_keys": sorted(required_recipe_keys - recipe_keys),
        },
    )
    runs = model.get("runs") or []
    scenes = model.get("scenes") or []
    diagnostic_contract_declared = any(
        "diagnostic_only" in row or "quality_comparison_eligible" in row
        for row in runs
    )
    diagnostic_contract_errors: list[str] = []
    if diagnostic_contract_declared:
        for row in runs:
            label = f"{row.get('label')} repeat {row.get('repeat')}"
            diagnostic_only = row.get("diagnostic_only")
            quality_eligible = row.get("quality_comparison_eligible")
            if not isinstance(diagnostic_only, bool):
                diagnostic_contract_errors.append(
                    f"{label}: diagnostic_only must be boolean"
                )
            if not isinstance(quality_eligible, bool):
                diagnostic_contract_errors.append(
                    f"{label}: quality_comparison_eligible must be boolean"
                )
            if isinstance(diagnostic_only, bool) and isinstance(quality_eligible, bool):
                if diagnostic_only and quality_eligible:
                    diagnostic_contract_errors.append(
                        f"{label}: diagnostic run cannot be quality eligible"
                    )
            if diagnostic_only and not str(row.get("diagnostic_only_reason", "")).strip():
                diagnostic_contract_errors.append(
                    f"{label}: diagnostic-only reason is missing"
                )
    add(
        "diagnostic_run_contract",
        not diagnostic_contract_errors,
        {
            "declared": diagnostic_contract_declared,
            "diagnostic_runs": sum(bool(row.get("diagnostic_only")) for row in runs),
            "errors": diagnostic_contract_errors,
        },
    )
    frame_rows = [frame for scene in scenes for frame in scene.get("frames") or []]
    maps = [item for frame in frame_rows for item in frame.get("maps") or []]
    pyramid_errors: list[str] = []
    for row in maps:
        if row.get("pyramid_level") is None:
            continue
        level = _model_integer(row.get("pyramid_level"))
        if level is None or level < 0:
            pyramid_errors.append(f"{row.get('id')}: pyramid_level must be a nonnegative integer")
    for frame in frame_rows:
        if "pyramid_levels" not in frame:
            continue
        declared = frame.get("pyramid_levels")
        if not isinstance(declared, list) or any(
            _model_integer(level) is None or int(level) < 0 for level in declared
        ):
            pyramid_errors.append(f"{frame.get('id')}: pyramid_levels is invalid")
            continue
        frame_algorithm_rows = [
            *(frame.get("maps") or []),
            *[
                iteration
                for run_frame in frame.get("run_frames") or []
                for iteration in run_frame.get("iterations") or []
            ],
        ]
        observed = sorted({
            int(row["pyramid_level"])
            for row in frame_algorithm_rows
            if row.get("pyramid_level") is not None
        })
        if [int(level) for level in declared] != observed:
            pyramid_errors.append(
                f"{frame.get('id')}: pyramid_levels does not match its map/iteration inventory"
            )
        expected_iterations_by_level = {
            str(level): sorted({
                int(row["logical_iteration"])
                for row in frame_algorithm_rows
                if canonical_pyramid_level(row) == level
                and _model_integer(row.get("logical_iteration")) is not None
            })
            for level in observed
        }
        if any(canonical_pyramid_level(row) is None for row in frame_algorithm_rows):
            expected_iterations_by_level["unspecified"] = sorted({
                int(row["logical_iteration"])
                for row in frame_algorithm_rows
                if canonical_pyramid_level(row) is None
                and _model_integer(row.get("logical_iteration")) is not None
            })
        if frame.get("logical_iterations_by_pyramid_level") != expected_iterations_by_level:
            pyramid_errors.append(
                f"{frame.get('id')}: logical iterations by pyramid level do not match its map/iteration inventory"
            )
        groups = (frame.get("map_groups") or {}).get("by_pyramid_level")
        if not isinstance(groups, dict):
            pyramid_errors.append(f"{frame.get('id')}: by_pyramid_level map group is missing")
    add(
        "pyramid_level_contract",
        not pyramid_errors,
        {
            "canonical_field": "pyramid_level",
            "aliases": ["scale_level", "scale_number"],
            "errors": pyramid_errors,
        },
    )
    candidate_accounting_errors: list[str] = []
    for frame in frame_rows:
        for run_frame in frame.get("run_frames") or []:
            frame_mode = str(run_frame.get("candidate_accounting_mode") or "")
            label = (
                f"{run_frame.get('run')}/{frame.get('scene_id')}/"
                f"{frame.get('image_id')}"
            )
            if frame_mode == CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE:
                metrics = run_frame.get("metrics") or {}
                missing_metrics = [
                    key for key in CANDIDATE_ACCOUNTING_METRICS
                    if key not in metrics
                ]
                populated_metrics = [
                    key for key, value in metrics.items()
                    if value is not None
                    and (
                        key in CANDIDATE_ACCOUNTING_METRICS
                        or key.startswith("accepted_from_")
                        or (
                            key.startswith("candidate_")
                            and key.endswith((
                                "_tested", "_finite", "_accepted",
                                "_acceptance_rate",
                            ))
                        )
                    )
                ]
                if missing_metrics:
                    candidate_accounting_errors.append(
                        f"{label}: unavailable frame accounting fields are not explicit: "
                        + ", ".join(missing_metrics)
                    )
                if populated_metrics:
                    candidate_accounting_errors.append(
                        f"{label}: unavailable frame accounting fields contain values: "
                        + ", ".join(sorted(populated_metrics))
                    )
            for iteration in run_frame.get("iterations") or []:
                mode = str(iteration.get("candidate_accounting_mode") or frame_mode)
                if (
                    frame_mode == CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
                    and mode != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE
                ):
                    candidate_accounting_errors.append(
                        f"{label}: iteration does not inherit unavailable candidate accounting"
                    )
                    continue
                if mode != CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE:
                    continue
                missing = [
                    key for key in CANDIDATE_ACCOUNTING_METRICS
                    if key not in iteration
                ]
                populated = [
                    key for key, value in iteration.items()
                    if value is not None
                    and (
                        key in CANDIDATE_ACCOUNTING_METRICS
                        or key.startswith("accepted_from_")
                        or (
                            key.startswith("candidate_")
                            and key.endswith((
                                "_tested", "_finite", "_accepted",
                                "_acceptance_rate",
                            ))
                        )
                    )
                ]
                if missing:
                    candidate_accounting_errors.append(
                        f"{label}: unavailable accounting fields are not explicit: "
                        + ", ".join(missing)
                    )
                if populated:
                    candidate_accounting_errors.append(
                        f"{label}: unavailable accounting fields contain values: "
                        + ", ".join(sorted(populated))
                    )
    add(
        "candidate_accounting_availability",
        not candidate_accounting_errors,
        {
            "unavailable_mode": CANDIDATE_ACCOUNTING_UNAVAILABLE_MODE,
            "unavailable_metrics": list(CANDIDATE_ACCOUNTING_METRICS),
            "errors": candidate_accounting_errors,
        },
    )
    health_errors: list[str] = []
    for frame in frame_rows:
        for run_frame in frame.get("run_frames") or []:
            health = run_frame.get("view_probability_health") or {}
            if not any(value is not None for value in health.values()):
                continue
            label = (
                f"{run_frame.get('run')}/{frame.get('scene_id')}/"
                f"{frame.get('image_id')}"
            )
            for key in ("schema_valid", "requested", "available", "accounting_valid"):
                if health.get(key) is not None and not isinstance(health.get(key), bool):
                    health_errors.append(f"{label}: {key} must be boolean or null")
            if health.get("available") is True:
                if health.get("schema_valid") is not True or health.get("accounting_valid") is not True:
                    health_errors.append(f"{label}: available health evidence is not schema/accounting valid")
                evidence_rows = [
                    row for row in run_frame.get("iterations") or []
                    if row.get("view_probability_processed") is not None
                ]
                if not evidence_rows:
                    health_errors.append(f"{label}: available health evidence has no iteration rows")
                identities: set[tuple[str, int | None, int, int]] = set()
                for row in evidence_rows:
                    level = canonical_pyramid_level(row)
                    logical_iteration = _model_integer(row.get("logical_iteration"))
                    if level is None or logical_iteration is None or logical_iteration < 0:
                        health_errors.append(f"{label}: health iteration identity is invalid")
                        continue
                    identity = (
                        str(row.get("estimation_stage") or "unspecified"),
                        _model_integer(row.get("geometric_iteration")),
                        level,
                        logical_iteration,
                    )
                    if identity in identities:
                        health_errors.append(f"{label}: duplicate health iteration {identity}")
                    identities.add(identity)
                    processed = _model_integer(row.get("view_probability_processed"))
                    finite = _model_integer(row.get("view_probability_finite_positive_events"))
                    degenerate = _model_integer(row.get("view_probability_degenerate_events"))
                    if None in (processed, finite, degenerate) or finite + degenerate != processed:
                        health_errors.append(f"{label}: health event accounting is invalid at {identity}")
            elif health.get("requested") is True and not health.get("unavailable_reason"):
                health_errors.append(f"{label}: requested unavailable health evidence has no reason")
    add(
        "view_probability_health_contract",
        not health_errors,
        {"errors": health_errors},
    )
    map_signal_ids = {
        str(row.get("signal")) for row in maps if row.get("signal")
    }
    add(
        "component_registry_covers_maps",
        map_signal_ids <= registry_signal_ids,
        {"missing": sorted(map_signal_ids - registry_signal_ids)},
    )
    deep_links = [
        *[row.get("deep_link_id") for row in runs],
        *[row.get("deep_link_id") for row in scenes],
        *[row.get("deep_link_id") for row in frame_rows],
        *[row.get("deep_link_id") for row in maps],
    ]
    deep_links = [str(value) for value in deep_links if value]
    add("deep_link_ids_unique", len(deep_links) == len(set(deep_links)), {"ids": len(deep_links), "unique": len(set(deep_links))})

    unavailable_without_reason = [row.get("id") for row in maps if not row.get("available") and not row.get("unavailable_reason")]
    add("unavailable_maps_explicit", not unavailable_without_reason, unavailable_without_reason)
    summary_contract = (model.get("contract") or {}).get(
        "summary_unavailable_signal_contract"
    ) or {}
    summary_rows = [
        row for row in (model.get("capture_validation") or {}).get("rows") or []
        if row.get("capture_kind") == "summary_only"
    ]
    summary_inventory_errors: list[str] = []
    logical_contract = {
        str(value) for value in summary_contract.get("logical_signals") or []
    }
    initialization_contract = {
        str(value) for value in summary_contract.get("initialization_signals") or []
    }
    final_contract = {
        str(value) for value in summary_contract.get("final_signals") or []
    }
    contract_valid = (
        summary_contract.get("schema_name")
        == "openmvs.dmap.summary_unavailable_signals"
        and summary_contract.get("schema_version") == 1
        and bool(logical_contract)
        and bool(final_contract)
    )
    if summary_rows and not contract_valid:
        summary_inventory_errors.append("summary unavailable-signal contract is missing or empty")
    if contract_valid:
        for validation_row in summary_rows:
            scene_id = str(validation_row.get("scene_id", ""))
            image_id = _model_integer(validation_row.get("image_id"))
            run = str(validation_row.get("run", ""))
            repeat = _model_integer(validation_row.get("repeat"))
            estimation_stage = validation_row.get("estimation_stage")
            geometric_iteration = _model_integer(
                validation_row.get("geometric_iteration")
            )
            matching_frames = [
                frame for frame in frame_rows
                if str(frame.get("scene_id", "")) == scene_id
                and _model_integer(frame.get("image_id")) == image_id
            ]
            matching_maps = [
                row for frame in matching_frames for row in frame.get("maps") or []
                if str(row.get("run", "")) == run
                and _model_integer(row.get("repeat")) == repeat
                and row.get("estimation_stage") == estimation_stage
                and _model_integer(row.get("geometric_iteration"))
                == geometric_iteration
            ]
            raw_iterations = validation_row.get("logical_iterations")
            if isinstance(raw_iterations, str):
                try:
                    raw_iterations = json.loads(raw_iterations)
                except json.JSONDecodeError:
                    raw_iterations = []
            run_iterations = {
                normalized
                for value in raw_iterations or []
                if (normalized := _model_integer(value)) is not None
            }
            if not run_iterations:
                run_iterations = {
                    normalized
                    for row in matching_maps
                    if row.get("logical_iteration") is not None
                    if (normalized := _model_integer(row.get("logical_iteration"))) is not None
                }
            identity = f"{run}/repeat_{(repeat or 0):02d}/{scene_id}/{image_id}"
            if not matching_frames:
                summary_inventory_errors.append(f"{identity}: report frame missing")
                continue
            if not run_iterations:
                summary_inventory_errors.append(f"{identity}: logical iterations missing")
            keys = {
                (str(row.get("signal", "")), _model_integer(row.get("logical_iteration")))
                for row in matching_maps
            }
            missing_logical = sorted(
                f"{signal}@{iteration}"
                for iteration in run_iterations
                for signal in logical_contract
                if (signal, iteration) not in keys
            )
            missing_final = sorted(
                signal for signal in final_contract if (signal, None) not in keys
            )
            missing_initialization = sorted(
                signal for signal in initialization_contract
                if (signal, -1) not in keys
            ) if -1 in run_iterations else []
            policy_errors = [
                str(row.get("signal")) for row in matching_maps
                if (
                    (str(row.get("signal")) in logical_contract)
                    or (str(row.get("signal")) in initialization_contract)
                    or (str(row.get("signal")) in final_contract)
                )
                and (bool(row.get("available")) or not row.get("unavailable_reason"))
            ]
            if missing_logical:
                summary_inventory_errors.append(
                    f"{identity}: missing logical signals {', '.join(missing_logical[:12])}"
                )
            if missing_final:
                summary_inventory_errors.append(
                    f"{identity}: missing final signals {', '.join(missing_final[:12])}"
                )
            if missing_initialization:
                summary_inventory_errors.append(
                    f"{identity}: missing initialization signals {', '.join(missing_initialization[:12])}"
                )
            if policy_errors:
                summary_inventory_errors.append(
                    f"{identity}: summary signals not explicitly unavailable {', '.join(policy_errors[:12])}"
                )
    add(
        "summary_unavailable_signal_inventory",
        not summary_inventory_errors,
        {
            "summary_frames": len(summary_rows),
            "logical_signal_count": len(logical_contract),
            "initialization_signal_count": len(initialization_contract),
            "final_signal_count": len(final_contract),
            "errors": summary_inventory_errors,
        },
    )
    forbidden_iterations = []
    for frame in frame_rows:
        for run_frame in frame.get("run_frames") or []:
            for iteration in run_frame.get("iterations") or []:
                phase = str(iteration.get("phase", "")).lower()
                if phase in {"black", "red"}:
                    forbidden_iterations.append(iteration.get("deep_link_id"))
    for row in maps:
        stage = str(row.get("stage", "")).lower()
        if "black" in stage or "red" in stage:
            forbidden_iterations.append(row.get("id"))
    add("logical_iterations_only", not forbidden_iterations, forbidden_iterations)

    reference_ownership_errors: list[dict[str, str]] = []
    report_root = output_dir.resolve()
    for frame in frame_rows:
        reference_record = frame.get("reference") or {}
        if not reference_record.get("available"):
            continue
        value = str(reference_record.get("path") or "")
        if not value or value.startswith(("file://", "http://", "https://")):
            reference_ownership_errors.append({
                "frame": str(frame.get("id")), "path": value,
                "reason": "reference image must be a report-owned relative path",
            })
            continue
        target = (output_dir / value).resolve()
        try:
            target.relative_to(report_root)
        except ValueError:
            reference_ownership_errors.append({
                "frame": str(frame.get("id")), "path": value,
                "reason": "reference image resolves outside the report directory",
            })
    add(
        "reference_images_report_owned",
        not reference_ownership_errors,
        reference_ownership_errors,
    )

    missing_paths: list[dict[str, str]] = []

    def check_path(owner: str, raw_path: Any) -> None:
        if not raw_path:
            return
        value = str(raw_path)
        if value.startswith(("http://", "https://", "data:")):
            return
        if value.startswith("file://"):
            target = Path(value.removeprefix("file://"))
        else:
            target = (output_dir / value).resolve()
        if not target.is_file():
            missing_paths.append({"owner": owner, "path": value})

    for name, path in (model.get("entrypoints") or {}).items():
        check_path(f"entrypoint:{name}", path)
    for frame in frame_rows:
        if (frame.get("reference") or {}).get("available"):
            check_path(str(frame.get("id")), frame["reference"].get("path"))
    for row in maps:
        if not row.get("available"):
            continue
        check_path(str(row.get("id")), row.get("source_path"))
        preview = row.get("preview") or {}
        check_path(str(row.get("id")), preview.get("local"))
        check_path(str(row.get("id")), preview.get("shared"))
        for channel in preview.get("channels") or []:
            check_path(str(row.get("id")), channel.get("local"))
            check_path(str(row.get("id")), channel.get("shared"))
        pixel_data = row.get("pixel_data") or {}
        if pixel_data.get("available"):
            check_path(str(row.get("id")), pixel_data.get("script_path"))
    for index, row in enumerate(drilldowns.get("entries") or []):
        execution_metadata = row.get("execution_metadata") or {}
        if execution_metadata.get("available"):
            check_path(
                f"drilldown:{index}:executions", execution_metadata.get("source_path")
            )
        for source_index, source in enumerate((row.get("trace_data") or {}).get("sources") or []):
            if source.get("available"):
                check_path(
                    f"drilldown:{index}:trace:{source_index}", source.get("source_path")
                )
                closure = source.get("artifact_closure") or {}
                if closure.get("status") == "verified":
                    check_path(
                        f"drilldown:{index}:trace-closure:{source_index}",
                        closure.get("source_path"),
                    )
    add("referenced_artifacts_exist", not missing_paths, missing_paths)

    malformed_pixel_data = []
    for row in maps:
        pixel_data = row.get("pixel_data") or {}
        if not pixel_data.get("available"):
            continue
        if (
            pixel_data.get("encoding") != PIXEL_ENCODING
            or not pixel_data.get("exact")
            or int(pixel_data.get("width", 0)) <= 0
            or int(pixel_data.get("height", 0)) <= 0
            or int(pixel_data.get("channels", 0)) <= 0
        ):
            malformed_pixel_data.append(row.get("id"))
    add("pixel_data_contract", not malformed_pixel_data, malformed_pixel_data)

    nonportable_paths: list[dict[str, str]] = []

    def audit_paths(value: Any, location: str = "model") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{location}.{key}"
                if key in REPORT_PATH_KEYS and isinstance(item, str) and (
                    Path(item).is_absolute() or item.startswith("file://")
                ):
                    nonportable_paths.append({"location": child, "path": item})
                audit_paths(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                audit_paths(item, f"{location}[{index}]")

    audit_paths(model)
    add("paths_report_relative", not nonportable_paths, nonportable_paths)

    report_root = output_dir.resolve()
    broken_served_paths: list[dict[str, str]] = []

    def require_served_path(value: Any, location: str) -> None:
        if not isinstance(value, str) or not value:
            return
        if value.startswith(("#", "data:", "blob:")) or re.match(
            r"^[a-z][a-z0-9+.-]*:", value, flags=re.IGNORECASE
        ):
            return
        candidate_value = value.split("#", 1)[0].split("?", 1)[0]
        parts = Path(candidate_value).parts
        # The UI renders external provenance as plain text. Only report-owned
        # local paths are links or fetch targets.
        if Path(candidate_value).is_absolute() or ".." in parts or "\\" in value:
            return
        candidate = (report_root / candidate_value).resolve()
        try:
            candidate.relative_to(report_root)
        except ValueError:
            broken_served_paths.append({"location": location, "path": value})
            return
        if not candidate.is_file() or candidate.is_symlink():
            broken_served_paths.append({"location": location, "path": value})

    for key, value in (model.get("entrypoints") or {}).items():
        require_served_path(value, f"entrypoints.{key}")
    for scene_index, scene in enumerate(scenes):
        for frame_index, frame in enumerate(scene.get("frames") or []):
            require_served_path(
                (frame.get("reference") or {}).get("path"),
                f"scenes[{scene_index}].frames[{frame_index}].reference.path",
            )
            for map_index, artifact in enumerate(frame.get("maps") or []):
                preview = artifact.get("preview") or {}

                def require_preview_paths(item: Any, location: str) -> None:
                    if isinstance(item, dict):
                        for key, child in item.items():
                            require_preview_paths(child, f"{location}.{key}")
                    elif isinstance(item, list):
                        for index, child in enumerate(item):
                            require_preview_paths(child, f"{location}[{index}]")
                    elif isinstance(item, str) and item.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".webp", ".svg")
                    ):
                        require_served_path(item, location)

                require_preview_paths(
                    preview,
                    f"scenes[{scene_index}].frames[{frame_index}].maps[{map_index}].preview",
                )
                require_served_path(
                    (artifact.get("pixel_data") or {}).get("script_path"),
                    f"scenes[{scene_index}].frames[{frame_index}].maps[{map_index}].pixel_data.script_path",
                )
    coverage = model.get("capture_profile_coverage") or {}
    for unit_index, unit in enumerate(coverage.get("units") or []):
        for link_index, link in enumerate(unit.get("evidence_links") or []):
            require_served_path(
                link.get("path"),
                f"capture_profile_coverage.units[{unit_index}].evidence_links[{link_index}]",
            )
        for frame_index, frame in enumerate(unit.get("frames") or []):
            for link_index, link in enumerate(frame.get("evidence_links") or []):
                require_served_path(
                    link.get("path"),
                    f"capture_profile_coverage.units[{unit_index}].frames[{frame_index}].evidence_links[{link_index}]",
                )
    for entry_index, entry in enumerate((model.get("drilldowns") or {}).get("entries") or []):
        for key in ("request", "executions"):
            require_served_path(entry.get(key), f"drilldowns.entries[{entry_index}].{key}")
        trace = entry.get("trace") or {}
        for source_index, source in enumerate(trace.get("sources") or []):
            require_served_path(
                source.get("source_path"),
                f"drilldowns.entries[{entry_index}].trace.sources[{source_index}]",
            )
    add("served_local_evidence_paths", not broken_served_paths, broken_served_paths)

    frame_deep_links = {str(frame.get("deep_link_id")) for frame in frame_rows}
    scene_deep_links = {str(scene.get("deep_link_id")) for scene in scenes}
    broken_regression_targets = [
        row.get("id") for row in (model.get("aggregates") or {}).get("regressions") or []
        if (
            str(row.get("scene_deep_link_id")) not in scene_deep_links
            or (
                row.get("navigation_level") != "scene"
                and str(row.get("frame_deep_link_id")) not in frame_deep_links
            )
            or (
                row.get("navigation_level") == "scene"
                and row.get("frame_deep_link_id") is not None
            )
        )
    ]
    add("regression_navigation_targets", not broken_regression_targets, broken_regression_targets)
    valid = all(bool(row["passed"]) for row in checks)
    return {
        "schema_name": "openmvs.dmap.development_report_validation",
        "schema_version": 1,
        "report_model_schema_version": model.get("schema_version"),
        "valid": valid,
        "checks": checks,
        "run_count": len(runs),
        "scene_count": len(scenes),
        "frame_count": len(frame_rows),
        "map_count": len(maps),
    }


def render_investigation_html(
    output_path: Path,
    model: dict[str, Any],
    template_path: Path,
    css_path: Path,
    javascript_path: Path,
) -> None:
    template = template_path.read_text(encoding="utf-8")
    model_json = json.dumps(json_value(model), sort_keys=True, separators=(",", ":"), allow_nan=False)
    model_json = model_json.replace("</", "<\\/")
    rendered = template.replace("__DMAP_REPORT_CSS__", css_path.read_text(encoding="utf-8"))
    rendered = rendered.replace("__DMAP_REPORT_MODEL__", model_json)
    rendered = rendered.replace("__DMAP_REPORT_JS__", javascript_path.read_text(encoding="utf-8"))
    output_path.write_text(rendered, encoding="utf-8")
