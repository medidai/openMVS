"""Deterministic region-stratified metrics for depth-map diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from . import component_registry


SCHEMA_NAME = "openmvs.dmap.region_metrics"
SCHEMA_VERSION = 1
GAIN_CENSUS_SCHEMA_NAME = "openmvs.dmap.low_texture_accepted_gain_census"
GAIN_CENSUS_SCHEMA_VERSION = 1
GAIN_CENSUS_SIGNALS = (
    "candidate_incumbent_cost_exact",
    "candidate_winner_cost_exact",
    "reference_variance_production_exact",
)
DEFAULT_GAIN_THRESHOLDS = (0.00025, 0.0005, 0.001)
TEXTURE_SIGNAL_PRIORITY = (
    "texture_score",
    "reference_variance_production_exact",
    "reference_variance",
)
DEFAULT_METRIC_QUANTITIES = {
    "total",
    "stored_value",
    "confidence",
    "winner_runner_up_gap",
    "depth_delta",
    "relative_depth_delta",
    "normal_angle_delta",
    "selected_count",
    "churn",
    "contribution",
    "probability_mass",
    "positive_probability_count",
    "unassigned_draw_count",
    "depth_transfer_delta",
    "valid_sample_fraction",
    "hierarchy_entry_cost",
    "hierarchy_proposed_cost",
    "hierarchy_improvement_margin",
    "eligibility",
    "ambiguity",
    "required_gain",
    "best_proposed_gain",
    "rejected_update_count",
}


def _finite_scalar(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"expected a scalar image, got shape {values.shape}")
    return values.astype(np.float64, copy=False)


def _available_values(values: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    """Mask finite values plus any capture-declared unavailable sentinel."""

    available = np.isfinite(values)
    sentinel = row.get("unavailable_value")
    try:
        sentinel_value = float(sentinel)
    except (TypeError, ValueError):
        return available
    if np.isfinite(sentinel_value):
        available &= values != sentinel_value
    return available


def _pyramid_level(row: Mapping[str, Any]) -> int | None:
    for key in ("pyramid_level", "scale_level", "scale_number"):
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            level = int(float(value))
        except (TypeError, ValueError):
            continue
        return level if level >= 0 else None
    return None


def _identity_integer(value: Any, default: int = -1) -> int:
    if value in (None, "") or isinstance(value, bool):
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return int(numeric) if np.isfinite(numeric) and numeric.is_integer() else default


def texture_bins(texture: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Partition finite texture values into deterministic low/mid/high thirds."""

    values = _finite_scalar(texture)
    finite = np.isfinite(values)
    sample = values[finite]
    if not sample.size:
        return np.full(values.shape, -1, dtype=np.int8), {"low_mid": float("nan"), "mid_high": float("nan")}
    low_mid, mid_high = np.quantile(sample, [1.0 / 3.0, 2.0 / 3.0])
    bins = np.full(values.shape, -1, dtype=np.int8)
    bins[finite & (values <= low_mid)] = 0
    bins[finite & (values > low_mid) & (values <= mid_high)] = 1
    bins[finite & (values > mid_high)] = 2
    return bins, {"low_mid": float(low_mid), "mid_high": float(mid_high)}


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("run"),
        int(row.get("repeat", 0) or 0),
        row.get("scene_id"),
        _identity_integer(row.get("image_id")),
        row.get("estimation_stage") or "photometric",
        _identity_integer(row.get("geometric_iteration")),
        _pyramid_level(row),
        row.get("logical_iteration"),
    )


def compute_texture_stratification(
    rows: Iterable[Mapping[str, Any]],
    read_map: Callable[[Path], np.ndarray],
    *,
    metric_quantities: set[str] | None = None,
) -> dict[str, Any]:
    """Compute long-form metric summaries grouped by per-frame texture thirds."""

    metric_quantities = metric_quantities or DEFAULT_METRIC_QUANTITIES
    available = [
        dict(row)
        for row in rows
        if row.get("available", row.get("exists", False))
        and row.get("path")
        and row.get("source_view_index") in {None, "", -1}
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in available:
        grouped.setdefault(_identity(row), []).append(row)

    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    labels = ("low", "mid", "high")
    for identity, selected in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        by_signal = {str(row.get("signal")): row for row in selected if row.get("signal")}
        texture_row = next((by_signal[signal] for signal in TEXTURE_SIGNAL_PRIORITY if signal in by_signal), None)
        if texture_row is None:
            texture_row = next((
                row for row in selected
                if component_registry.descriptor_from_row(row).mechanism == "texture"
                and component_registry.descriptor_from_row(row).quantity in {"score", "reference_variance"}
            ), None)
        if texture_row is None:
            continue
        try:
            texture = _finite_scalar(read_map(Path(str(texture_row["path"]))))
            texture_for_bins = np.where(
                _available_values(texture, texture_row), texture, np.nan
            )
            bins, thresholds = texture_bins(texture_for_bins)
        except Exception as exc:
            failures.append({"identity": list(identity), "signal": texture_row.get("signal"), "error": str(exc)})
            continue

        metric_rows = []
        for row in selected:
            descriptor = component_registry.descriptor_from_row(row)
            if row is texture_row or descriptor.quantity not in metric_quantities:
                continue
            try:
                values = _finite_scalar(read_map(Path(str(row["path"]))))
            except Exception as exc:
                failures.append({"identity": list(identity), "signal": row.get("signal"), "error": str(exc)})
                continue
            if values.shape != texture.shape:
                failures.append({
                    "identity": list(identity),
                    "signal": row.get("signal"),
                    "error": f"shape {values.shape} does not match texture shape {texture.shape}",
                })
                continue
            metric_rows.append((row, descriptor, values))

        for bin_index, bin_label in enumerate(labels):
            region = bins == bin_index
            region_count = int(np.count_nonzero(region))
            texture_values = texture[region & _available_values(texture, texture_row)]
            base = {
                "run": identity[0],
                "repeat": identity[1],
                "scene_id": identity[2],
                "image_id": identity[3],
                "estimation_stage": identity[4],
                "geometric_iteration": identity[5],
                "pyramid_level": identity[6],
                "logical_iteration": identity[7],
                "region_type": "texture_quantile",
                "region": bin_label,
                "region_pixels": region_count,
                "region_fraction": float(region_count / max(1, np.count_nonzero(bins >= 0))),
                "texture_signal": texture_row.get("signal"),
                "texture_threshold_low_mid": thresholds["low_mid"],
                "texture_threshold_mid_high": thresholds["mid_high"],
                "texture_mean": float(np.mean(texture_values)) if texture_values.size else None,
            }
            output.append({**base, "signal": texture_row.get("signal"), "mechanism": "texture", "quantity": "score", "valid_pixels": int(texture_values.size), "mean": base["texture_mean"], "median": float(np.median(texture_values)) if texture_values.size else None, "p90": float(np.quantile(texture_values, 0.9)) if texture_values.size else None})
            for row, descriptor, values in metric_rows:
                metric_values = values[region & _available_values(values, row)]
                output.append({
                    **base,
                    "signal": row.get("signal"),
                    "mechanism": descriptor.mechanism,
                    "quantity": descriptor.quantity,
                    "valid_pixels": int(metric_values.size),
                    "mean": float(np.mean(metric_values)) if metric_values.size else None,
                    "median": float(np.median(metric_values)) if metric_values.size else None,
                    "p90": float(np.quantile(metric_values, 0.9)) if metric_values.size else None,
                })

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "binning": "per-run/frame/stage/pyramid-level/logical-state finite-value thirds",
        "rows": output,
        "failures": failures,
    }


def compute_low_texture_accepted_gain_census(
    rows: Iterable[Mapping[str, Any]],
    read_map: Callable[[Path], np.ndarray],
    *,
    variance_max_by_run: Mapping[str, float],
    gain_thresholds: tuple[float, ...] = DEFAULT_GAIN_THRESHOLDS,
) -> dict[str, Any]:
    """Measure positive retained gains in configured low-texture regions.

    The census is a deterministic pre-change diagnostic. It identifies pixels
    where the exact final winner improves on the iteration-entry incumbent and
    the reference variance is below the run's configured threshold. It does not
    infer coarse-prior availability or claim that every accepted proposal would
    be controlled by a future gate.
    """

    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        if (
            not row.get("available", row.get("exists", False))
            or not row.get("path")
            or row.get("source_view_index") not in {None, "", -1}
            or row.get("signal") not in GAIN_CENSUS_SIGNALS
        ):
            continue
        identity = _identity(row)
        logical_iteration = identity[-1]
        try:
            if logical_iteration is None or int(logical_iteration) < 0:
                continue
        except (TypeError, ValueError):
            continue
        grouped.setdefault(identity, {})[str(row["signal"])] = row

    thresholds = tuple(sorted({float(value) for value in gain_thresholds if float(value) > 0.0}))
    output: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for identity, signals in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        missing = sorted(set(GAIN_CENSUS_SIGNALS) - set(signals))
        if missing:
            failures.append({"identity": list(identity), "missing_signals": missing})
            continue
        run = str(identity[0])
        variance_max = variance_max_by_run.get(run)
        try:
            variance_max = float(variance_max) if variance_max is not None else None
        except (TypeError, ValueError):
            variance_max = None
        if variance_max is None or not np.isfinite(variance_max) or variance_max <= 0.0:
            failures.append({
                "identity": list(identity),
                "error": "configured positive low-texture variance threshold is unavailable",
            })
            continue
        try:
            incumbent = _finite_scalar(read_map(Path(str(signals[GAIN_CENSUS_SIGNALS[0]]["path"]))))
            winner = _finite_scalar(read_map(Path(str(signals[GAIN_CENSUS_SIGNALS[1]]["path"]))))
            variance = _finite_scalar(read_map(Path(str(signals[GAIN_CENSUS_SIGNALS[2]]["path"]))))
        except Exception as exc:
            failures.append({"identity": list(identity), "error": str(exc)})
            continue
        if incumbent.shape != winner.shape or incumbent.shape != variance.shape:
            failures.append({
                "identity": list(identity),
                "error": (
                    f"shape mismatch: incumbent={incumbent.shape}, winner={winner.shape}, "
                    f"variance={variance.shape}"
                ),
            })
            continue
        valid = (
            _available_values(incumbent, signals[GAIN_CENSUS_SIGNALS[0]])
            & _available_values(winner, signals[GAIN_CENSUS_SIGNALS[1]])
            & _available_values(variance, signals[GAIN_CENSUS_SIGNALS[2]])
        )
        low_texture = valid & (variance < variance_max)
        gains = incumbent - winner
        accepted = low_texture & (gains > 0.0)
        accepted_gains = gains[accepted]
        quantiles = {
            f"p{percentile}": (
                float(np.quantile(accepted_gains, percentile / 100.0))
                if accepted_gains.size else None
            )
            for percentile in (10, 25, 50, 75, 90, 95)
        }
        fractions = {
            format(threshold, ".8g"): (
                float(np.mean(accepted_gains < threshold))
                if accepted_gains.size else None
            )
            for threshold in thresholds
        }
        output.append({
            "run": identity[0],
            "repeat": identity[1],
            "scene_id": identity[2],
            "image_id": identity[3],
            "estimation_stage": identity[4],
            "geometric_iteration": identity[5],
            "pyramid_level": identity[6],
            "logical_iteration": identity[7],
            "variance_signal": GAIN_CENSUS_SIGNALS[2],
            "variance_max": variance_max,
            "valid_pixels": int(np.count_nonzero(valid)),
            "low_texture_pixels": int(np.count_nonzero(low_texture)),
            "accepted_gain_pixels": int(accepted_gains.size),
            "accepted_fraction_of_low_texture": (
                float(accepted_gains.size / np.count_nonzero(low_texture))
                if np.count_nonzero(low_texture) else None
            ),
            "gain_quantiles": quantiles,
            "fractions_below": fractions,
        })

    return {
        "schema_name": GAIN_CENSUS_SCHEMA_NAME,
        "schema_version": GAIN_CENSUS_SCHEMA_VERSION,
        "definition": (
            "positive candidate_incumbent_cost_exact - candidate_winner_cost_exact "
            "where reference_variance_production_exact is below the run's configured threshold"
        ),
        "limitations": (
            "diagnostic retained-winner census; coarse-prior eligibility and sequential proposal "
            "attribution are not inferred"
        ),
        "gain_thresholds": list(thresholds),
        "rows": output,
        "failures": failures,
    }
