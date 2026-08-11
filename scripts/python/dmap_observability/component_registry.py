"""Versioned signal descriptors for depth-map observability and reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping


SCHEMA_NAME = "openmvs.dmap.component_registry"
SCHEMA_VERSION = 1

CAPTURE_PROFILES = {"summary", "light", "deep", "trace"}
MEASUREMENT_KINDS = {"scalar", "enum", "vector", "map", "table", "trace", "image"}
PREFERRED_DIRECTIONS = {"higher", "lower", "neutral", "contextual"}
MECHANISMS = {
    "input",
    "state",
    "cost",
    "texture",
    "patch",
    "candidate_update",
    "propagation",
    "view_selection",
    "multiscale",
    "filtering",
    "geometry",
    "runtime",
}


@dataclass(frozen=True)
class SignalDescriptor:
    """Display and validation semantics for one machine-readable signal."""

    signal_id: str
    label: str
    mechanism: str
    quantity: str
    units: str = "unitless"
    domain: str = "finite"
    preferred_direction: str = "contextual"
    minimum_profile: str = "light"
    measurement_kind: str = "map"
    colormap: str = "viridis"
    signed: bool = False
    default_visible: bool = False
    component_id: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _descriptor(
    signal_id: str,
    mechanism: str,
    quantity: str,
    *,
    label: str | None = None,
    units: str = "unitless",
    domain: str = "finite",
    preferred_direction: str = "contextual",
    minimum_profile: str = "light",
    measurement_kind: str = "map",
    colormap: str = "viridis",
    signed: bool = False,
    default_visible: bool = False,
    component_id: str | None = None,
    description: str = "",
) -> SignalDescriptor:
    return SignalDescriptor(
        signal_id=signal_id,
        label=label or signal_id.replace("_", " "),
        mechanism=mechanism,
        quantity=quantity,
        units=units,
        domain=domain,
        preferred_direction=preferred_direction,
        minimum_profile=minimum_profile,
        measurement_kind=measurement_kind,
        colormap=colormap,
        signed=signed,
        default_visible=default_visible,
        component_id=component_id,
        description=description,
    )


BUILTIN_DESCRIPTORS = {
    item.signal_id: item
    for item in (
        _descriptor("reference_rgb", "input", "reference_image", measurement_kind="image", default_visible=True),
        _descriptor("depth", "state", "depth", units="m", domain="positive", colormap="turbo"),
        _descriptor("depth_final_after_filter", "state", "depth", units="m", domain="positive", colormap="turbo", default_visible=True),
        _descriptor("normal_final", "state", "normal", domain="unit_vector", measurement_kind="vector", default_visible=True),
        _descriptor("cost_stored", "cost", "stored_value", preferred_direction="lower", colormap="magma", default_visible=True, component_id="total"),
        _descriptor(
            "cost_improvement_exact", "cost", "logical_improvement",
            label="Logical cost improvement", units="cost", domain="nonnegative",
            preferred_direction="higher", minimum_profile="deep", colormap="viridis",
            default_visible=True, component_id="total",
            description=(
                "Derived-exact positive per-pixel cost reduction summed across the complementary "
                "checkerboard passes of one complete logical PatchMatch iteration. Cost "
                "increases are represented as zero, and view-set changes can change the stored-cost basis."
            ),
        ),
        _descriptor("confidence_stored", "cost", "confidence", domain="zero_to_one", preferred_direction="higher", colormap="viridis", default_visible=True),
        _descriptor("cost_total_production_exact", "cost", "total", preferred_direction="lower", minimum_profile="deep", colormap="magma", default_visible=True, component_id="total"),
        _descriptor("cost_photo_raw_production_exact", "cost", "raw_value", preferred_direction="lower", minimum_profile="deep", colormap="magma", default_visible=True, component_id="photometric"),
        _descriptor("cost_photo_prior_production_exact", "cost", "contribution", preferred_direction="lower", minimum_profile="deep", colormap="magma", default_visible=True, component_id="photo_prior"),
        _descriptor("cost_geometric_production_exact", "cost", "contribution", preferred_direction="lower", minimum_profile="deep", colormap="magma", default_visible=True, component_id="geometric"),
        _descriptor("cost_depth_prior_production_exact", "cost", "contribution", preferred_direction="lower", minimum_profile="deep", colormap="magma", component_id="depth_prior"),
        _descriptor("depth_prior_weight_production_exact", "cost", "weight", domain="zero_to_one", minimum_profile="deep", colormap="viridis", component_id="depth_prior"),
        _descriptor("reference_variance_production_exact", "texture", "reference_variance", domain="nonnegative", minimum_profile="light", colormap="cividis", default_visible=True, component_id="texture"),
        _descriptor("texture_score", "texture", "score", domain="nonnegative", minimum_profile="light", colormap="cividis", default_visible=True, component_id="texture"),
        _descriptor("texture_activation", "texture", "activation", domain="boolean", minimum_profile="light", measurement_kind="enum", colormap="cividis", component_id="texture"),
        _descriptor(
            "gap_winner_runner_up_exact", "candidate_update", "winner_runner_up_gap",
            domain="nonnegative", preferred_direction="higher", minimum_profile="deep",
            colormap="viridis", default_visible=True,
            description=(
                "Runner-up minus retained production winner when the retained winner is also the "
                "raw minimum. Unavailable when low-texture hysteresis suppresses that raw minimum."
            ),
        ),
        _descriptor(
            "candidate_raw_best_cost_exact", "candidate_update", "raw_best_cost",
            label="Raw best candidate cost", units="cost", domain="nonnegative",
            preferred_direction="lower", minimum_profile="deep", colormap="magma",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Lowest finite raw candidate cost observed on the gated sequential trajectory. "
                "This remains available when hysteresis retains a more expensive production winner."
            ),
        ),
        _descriptor(
            "candidate_raw_runner_up_cost_exact", "candidate_update", "raw_runner_up_cost",
            label="Raw runner-up candidate cost", units="cost", domain="nonnegative",
            preferred_direction="lower", minimum_profile="deep", colormap="magma",
            component_id="low_texture_update_hysteresis",
            description=(
                "Second-lowest finite raw candidate cost on the observed gated trajectory; "
                "the unavailable sentinel is retained when fewer than two candidates are finite."
            ),
        ),
        _descriptor(
            "gap_raw_best_runner_up_exact", "candidate_update", "raw_best_runner_up_gap",
            label="Raw best-to-runner gap", units="cost", domain="nonnegative",
            preferred_direction="higher", minimum_profile="deep", colormap="viridis",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Raw runner-up minus raw best cost on the observed gated trajectory, independent "
                "of whether hysteresis retained that raw best candidate."
            ),
        ),
        _descriptor(
            "candidate_retained_minus_raw_best_exact", "candidate_update", "retained_minus_raw_best",
            label="Retained minus raw-best cost", units="cost", domain="nonnegative",
            preferred_direction="lower", minimum_profile="deep", colormap="magma",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Retained production winner cost minus raw best cost. A positive value identifies "
                "a raw minimum suppressed by low-texture hysteresis."
            ),
        ),
        _descriptor(
            "candidate_raw_suppression_identity_exact", "candidate_update", "raw_suppression_identity",
            label="Raw suppression identity", domain="nonnegative",
            minimum_profile="deep", measurement_kind="enum", colormap="tab20",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Lossless RGBA identity tuple: raw-best slot, raw runner-up slot, retained-winner "
                "slot, and suppressed raw-best source enum (zero when the raw best was retained)."
            ),
        ),
        _descriptor("candidate_identity_exact", "candidate_update", "identity", minimum_profile="deep", measurement_kind="enum", colormap="tab20", default_visible=True),
        _descriptor("candidate_counts_exact", "candidate_update", "count", domain="nonnegative", minimum_profile="deep", colormap="viridis", default_visible=True),
        _descriptor("candidate_accepted_mask_exact", "candidate_update", "accepted_mask", minimum_profile="deep", measurement_kind="enum", colormap="tab20"),
        _descriptor("candidate_finite_mask_exact", "candidate_update", "finite_mask", minimum_profile="deep", measurement_kind="enum", colormap="tab20"),
        _descriptor(
            "low_texture_update_eligible_exact", "candidate_update", "eligibility",
            label="Low-texture update eligibility", domain="boolean",
            minimum_profile="deep", measurement_kind="enum", colormap="tab20",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Exact eligibility of this pixel for low-texture update hysteresis: a valid "
                "coarse prior, variance below the configured threshold, and an enabled gate."
            ),
        ),
        _descriptor(
            "low_texture_update_ambiguity_exact", "candidate_update", "ambiguity",
            label="Low-texture update ambiguity", domain="zero_to_one",
            minimum_profile="deep", colormap="cividis", default_visible=True,
            component_id="low_texture_update_hysteresis",
            description=(
                "Exact per-pixel ambiguity used by the low-texture update gate. "
                "Zero is unambiguous and one is maximally ambiguous under the configured variance threshold."
            ),
        ),
        _descriptor(
            "low_texture_update_required_gain_exact", "candidate_update", "required_gain",
            label="Required low-texture update gain", units="cost", domain="nonnegative",
            minimum_profile="deep", colormap="magma", default_visible=True,
            component_id="low_texture_update_hysteresis",
            description=(
                "Exact minimum incumbent-cost improvement required at an eligible pixel after "
                "scaling the configured hysteresis margin by ambiguity."
            ),
        ),
        _descriptor(
            "low_texture_update_best_proposed_gain_exact", "candidate_update", "best_proposed_gain",
            label="Best gate-controlled proposed gain", units="cost", domain="nonnegative",
            preferred_direction="higher", minimum_profile="deep", colormap="viridis",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Largest positive legacy incumbent-cost improvement proposed by a gate-controlled "
                "candidate at this pixel during the complete logical iteration."
            ),
        ),
        _descriptor(
            "low_texture_update_rejected_mask_exact", "candidate_update", "rejected_update_mask",
            label="Low-texture rejected update families", domain="nonnegative",
            minimum_profile="deep", measurement_kind="enum", colormap="tab20",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Exact family bit mask for legacy-improving proposals suppressed by hysteresis: "
                "bit 1 propagation and bit 2 refinement."
            ),
        ),
        _descriptor(
            "low_texture_update_would_have_won_source_exact", "candidate_update", "would_have_won_source",
            label="Best suppressed proposal source", domain="nonnegative",
            minimum_profile="deep", measurement_kind="enum", colormap="tab20",
            default_visible=True, component_id="low_texture_update_hysteresis",
            description=(
                "Exact PatchMatch source enum for the largest-gain legacy-accepted proposal "
                "suppressed on the gated trajectory at this pixel; zero means no suppressed "
                "proposal. This is not a counterfactual replay of the final ungated winner."
            ),
        ),
        _descriptor(
            "low_texture_update_rejected_count_exact", "candidate_update", "rejected_update_count",
            label="Low-texture rejected update count", units="proposals", domain="nonnegative",
            preferred_direction="contextual", minimum_profile="deep", colormap="cividis",
            component_id="low_texture_update_hysteresis",
            description=(
                "Exact number of legacy-improving propagation or refinement proposals suppressed "
                "by the low-texture update gate at this pixel during the logical iteration."
            ),
        ),
        _descriptor("candidate_origin_offset_exact", "propagation", "origin_offset", units="px", minimum_profile="deep", measurement_kind="vector", colormap="coolwarm", signed=True),
        _descriptor("candidate_terminal_survival_exact", "propagation", "terminal_survival", domain="boolean", minimum_profile="deep", measurement_kind="enum", colormap="cividis"),
        _descriptor("view_weighted_contribution_exact", "view_selection", "weighted_contribution", preferred_direction="lower", minimum_profile="deep", colormap="magma", default_visible=True),
        _descriptor("view_selection_state_exact", "view_selection", "decision", minimum_profile="deep", measurement_kind="enum", colormap="tab20", default_visible=True),
        _descriptor("selected_view_count", "view_selection", "selected_count", domain="nonnegative", colormap="cividis"),
        _descriptor(
            "view_probability_mass", "view_selection", "probability_mass",
            label="View probability mass", domain="nonnegative", minimum_profile="deep",
            colormap="viridis", default_visible=True, component_id="probability_health",
            description="Raw finite selection-score mass before CDF normalization. Zero mass makes stochastic view selection degenerate.",
        ),
        _descriptor(
            "view_probability_positive_count", "view_selection", "positive_probability_count",
            label="Positive view probability count", units="views", domain="nonnegative",
            minimum_profile="deep", colormap="cividis", component_id="probability_health",
            description="Number of source views with a finite, strictly positive pre-CDF selection score.",
        ),
        _descriptor(
            "view_probability_health_status", "view_selection", "probability_health_status",
            label="View probability health", minimum_profile="deep", measurement_kind="enum",
            colormap="tab20", default_visible=True, component_id="probability_health",
            description="Categorical pre-CDF health classification; use the capture-provided channel/code legend for exact status meanings.",
        ),
        _descriptor(
            "view_probability_unassigned_draw_count", "view_selection", "unassigned_draw_count",
            label="Unassigned view draws", units="draws", domain="nonnegative",
            preferred_direction="lower", minimum_profile="deep", colormap="magma",
            component_id="probability_health",
            description="Monte Carlo draws that were not assigned to a valid CDF interval.",
        ),
        _descriptor(
            "view_probability_legacy_last_view_collapse", "view_selection", "legacy_last_view_collapse",
            label="Predicted legacy last-view collapse", domain="boolean",
            preferred_direction="lower", minimum_profile="deep", measurement_kind="enum",
            colormap="tab20", component_id="probability_health",
            description="Whether the legacy zero/nonfinite-mass path is predicted to collapse selection onto the final source-view slot.",
        ),
        _descriptor("depth_delta", "candidate_update", "depth_delta", units="m", colormap="coolwarm", signed=True, default_visible=True),
        _descriptor("depth_relative_delta", "candidate_update", "relative_depth_delta", colormap="coolwarm", signed=True),
        _descriptor("normal_angle_delta", "candidate_update", "normal_angle_delta", units="deg", domain="nonnegative", colormap="magma", default_visible=True),
        _descriptor("view_churn", "view_selection", "churn", domain="nonnegative", colormap="cividis", default_visible=True),
        _descriptor("rejection_reason", "filtering", "rejection_reason", minimum_profile="light", measurement_kind="enum", colormap="tab20", default_visible=True),
        _descriptor("validity_transition", "filtering", "validity_transition", minimum_profile="light", measurement_kind="enum", colormap="tab20"),
        _descriptor("patch_deformation_magnitude", "patch", "deformation_magnitude", units="px", domain="nonnegative", minimum_profile="deep", colormap="magma"),
        _descriptor("patch_valid_sample_fraction", "patch", "valid_sample_fraction", domain="zero_to_one", minimum_profile="deep", colormap="viridis"),
        _descriptor("patch_score_delta", "patch", "score_delta", minimum_profile="deep", colormap="coolwarm", signed=True),
        _descriptor(
            "adaptive_patch_support_mode", "patch", "support_mode",
            label="Adaptive patch eligibility mode", minimum_profile="deep",
            measurement_kind="enum", colormap="tab20", default_visible=True,
            component_id="adaptive_support",
            description="Static reference-pixel eligibility for adaptive support. It does not prove that every source-view hypothesis used the expanded footprint.",
        ),
        _descriptor(
            "adaptive_patch_activation", "patch", "activation",
            label="Adaptive patch activation", domain="boolean", minimum_profile="deep",
            measurement_kind="enum", colormap="tab20", component_id="adaptive_support",
            description="Whether the adaptive low-texture patch rule was requested at this pixel.",
        ),
        _descriptor(
            "adaptive_patch_valid_sample_fraction", "patch", "valid_sample_fraction",
            label="Adaptive patch valid sample fraction", domain="zero_to_one",
            preferred_direction="higher", minimum_profile="deep", colormap="viridis",
            component_id="adaptive_support",
            description="Fraction of the requested adaptive patch samples that were finite and in bounds.",
        ),
        _descriptor(
            "adaptive_patch_fallback_status", "patch", "fallback_status",
            label="Adaptive patch fallback status", minimum_profile="deep",
            measurement_kind="enum", colormap="tab20", component_id="adaptive_support",
            description="Categorical reason the adaptive footprint was used, skipped, or replaced by the standard footprint.",
        ),
        _descriptor("scale_transfer_depth_delta", "multiscale", "depth_transfer_delta", units="m", minimum_profile="light", colormap="coolwarm", signed=True),
        _descriptor("scale_transfer_cost_delta", "multiscale", "cost_transfer_delta", minimum_profile="light", colormap="coolwarm", signed=True),
        _descriptor(
            "jbu_transfer_depth", "multiscale", "transferred_depth",
            label="JBU transferred depth", units="m", domain="positive", minimum_profile="deep",
            colormap="turbo", component_id="pyramid_depth_transfer",
            description="Fine-level depth initialized by joint-bilateral upsampling from the preceding coarse level.",
        ),
        _descriptor(
            "jbu_nearest_depth", "multiscale", "nearest_transferred_depth",
            label="Nearest transferred depth", units="m", domain="positive", minimum_profile="deep",
            colormap="turbo", component_id="pyramid_depth_transfer",
            description="Nearest-neighbor coarse-depth transfer retained as the paired JBU reference.",
        ),
        _descriptor(
            "jbu_transfer_depth_delta", "multiscale", "depth_transfer_delta",
            label="JBU minus nearest depth", units="m", minimum_profile="deep",
            colormap="coolwarm", signed=True, default_visible=True,
            component_id="pyramid_depth_transfer",
            description="Joint-bilateral transferred depth minus nearest-neighbor transferred depth at the same fine-level pixel.",
        ),
        _descriptor(
            "jbu_fallback_status", "multiscale", "fallback_status",
            label="JBU fallback status", minimum_profile="deep", measurement_kind="enum",
            colormap="tab20", component_id="pyramid_depth_transfer",
            description="Categorical JBU availability/fallback result, including invalid coarse support and nearest-neighbor fallback.",
        ),
        _descriptor(
            "hierarchy_entry_cost", "multiscale", "hierarchy_entry_cost",
            label="Hierarchy entry cost", preferred_direction="lower", minimum_profile="deep",
            colormap="magma", component_id="hierarchy_gate",
            description="Stored objective value at entry to the current pyramid level.",
        ),
        _descriptor(
            "hierarchy_proposed_cost", "multiscale", "hierarchy_proposed_cost",
            label="Hierarchy final retained cost proxy", preferred_direction="lower", minimum_profile="deep",
            colormap="magma", component_id="hierarchy_gate",
            description="Final retained post-gate cost proxy. Rejected intermediate proposal costs are not retained by the current capture.",
        ),
        _descriptor(
            "hierarchy_improvement_margin", "multiscale", "hierarchy_improvement_margin",
            label="Hierarchy retained improvement", units="cost", preferred_direction="higher",
            minimum_profile="deep", colormap="coolwarm", signed=True, default_visible=True,
            component_id="hierarchy_gate",
            description="Entry cost minus final retained cost. Rejected intermediate proposal margins remain unavailable.",
        ),
        _descriptor(
            "hierarchy_update_status", "multiscale", "hierarchy_update_status",
            label="Hierarchy update status", minimum_profile="deep", measurement_kind="enum",
            colormap="tab20", default_visible=True, component_id="hierarchy_gate",
            description="Terminal gate state: retained the scale-entry hypothesis or crossed the configured margin at least once.",
        ),
    )
}


def infer_descriptor(signal_id: str) -> SignalDescriptor:
    """Return deterministic fallback semantics for a signal unknown to this version."""

    lowered = signal_id.lower()
    if lowered.startswith("view_") or lowered.startswith("selected_view"):
        mechanism = "view_selection"
    elif "texture" in lowered or "variance" in lowered:
        mechanism = "texture"
    elif "patch" in lowered or "deform" in lowered:
        mechanism = "patch"
    elif "scale" in lowered or "pyramid" in lowered or "coarse" in lowered:
        mechanism = "multiscale"
    elif lowered.startswith("candidate_") or "winner_runner" in lowered:
        mechanism = "candidate_update"
    elif "propagat" in lowered or "origin" in lowered:
        mechanism = "propagation"
    elif (
        "filter" in lowered
        or "rejection" in lowered
        or lowered.startswith("valid_")
        or "speckle" in lowered
        or "fill_gap" in lowered
    ):
        mechanism = "filtering"
    elif lowered.startswith("cost_") or "prior" in lowered or "confidence" in lowered or "gap" in lowered:
        mechanism = "cost"
    elif "delta" in lowered or "churn" in lowered or "changed" in lowered:
        mechanism = "candidate_update"
    elif lowered == "reference_rgb":
        mechanism = "input"
    else:
        mechanism = "state"

    signed = any(token in lowered for token in ("delta", "minus", "residual"))
    if signed:
        colormap = "coolwarm"
    elif "depth" in lowered and "prior" not in lowered:
        colormap = "turbo"
    elif "cost" in lowered or "residual" in lowered:
        colormap = "magma"
    elif mechanism == "view_selection":
        colormap = "cividis"
    else:
        colormap = "viridis"
    preferred = "lower" if any(token in lowered for token in ("cost", "error", "residual")) else "contextual"
    domain = "finite" if signed else "nonnegative"
    return _descriptor(
        signal_id,
        mechanism,
        "value",
        domain=domain,
        preferred_direction=preferred,
        colormap=colormap,
        signed=signed,
    )


def descriptor_from_row(row: Mapping[str, Any]) -> SignalDescriptor:
    """Merge a capture-declared descriptor over the built-in/fallback descriptor."""

    signal_id = str(row.get("signal_id") or row.get("signal") or "").strip()
    if not signal_id:
        raise ValueError("signal descriptor is missing signal_id")
    base = BUILTIN_DESCRIPTORS.get(signal_id, infer_descriptor(signal_id))
    values: dict[str, Any] = {}
    aliases = {
        "label": ("signal_label", "label") if row.get("signal_id") else ("signal_label",),
        "mechanism": ("mechanism",),
        "quantity": ("quantity", "component_quantity"),
        "units": ("units",),
        "domain": ("domain", "value_domain"),
        "preferred_direction": ("preferred_direction",),
        # capture_profile records where one observation was collected; it does
        # not redefine the signal's intrinsic minimum capture capability.
        "minimum_profile": ("minimum_profile",),
        "measurement_kind": ("measurement_kind", "display_type"),
        "colormap": ("colormap",),
        "signed": ("signed",),
        "default_visible": ("default_visible",),
        "component_id": ("component_id",),
        # Map-level semantics may include stage/view-specific wording.  Only an
        # explicit registry description is stable enough for contract merging.
        "description": ("description",),
    }
    for target, candidates in aliases.items():
        for candidate in candidates:
            value = row.get(candidate)
            if value is not None and value != "":
                values[target] = value
                break
    return replace(base, **values)


def build_registry(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the authoritative registry carried by a report model."""

    descriptors: dict[str, SignalDescriptor] = {}
    for row in rows:
        descriptor = descriptor_from_row(row)
        previous = descriptors.get(descriptor.signal_id)
        if previous is not None and previous != descriptor:
            raise ValueError(f"conflicting descriptors for signal '{descriptor.signal_id}'")
        descriptors[descriptor.signal_id] = descriptor
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "signals": [descriptors[key].to_dict() for key in sorted(descriptors)],
    }


def validate_registry(registry: Mapping[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the registry is valid."""

    errors: list[str] = []
    if registry.get("schema_name") != SCHEMA_NAME:
        errors.append("invalid schema_name")
    if registry.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    seen: set[str] = set()
    for index, raw in enumerate(registry.get("signals") or []):
        location = f"signals[{index}]"
        signal_id = str(raw.get("signal_id") or "")
        if not signal_id:
            errors.append(f"{location}.signal_id is required")
        elif signal_id in seen:
            errors.append(f"duplicate signal_id '{signal_id}'")
        seen.add(signal_id)
        if raw.get("mechanism") not in MECHANISMS:
            errors.append(f"{location}.mechanism is invalid")
        if raw.get("minimum_profile") not in CAPTURE_PROFILES:
            errors.append(f"{location}.minimum_profile is invalid")
        if raw.get("measurement_kind") not in MEASUREMENT_KINDS:
            errors.append(f"{location}.measurement_kind is invalid")
        if raw.get("preferred_direction") not in PREFERRED_DIRECTIONS:
            errors.append(f"{location}.preferred_direction is invalid")
    return errors


def registry_index(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["signal_id"]): dict(row) for row in registry.get("signals") or [] if row.get("signal_id")}
