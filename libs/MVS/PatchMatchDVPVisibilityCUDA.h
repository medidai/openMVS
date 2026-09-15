/*
 * PatchMatchDVPVisibilityCUDA.h
 *
 * Shared host/device contract for persistent DVP-MVS visibility weights.
 * The paper-compatible 2D rule and the depth-gated OpenMVS adaptation remain
 * distinct so CPU fixtures, CUDA oracles, and production kernels cannot drift.
 */

#pragma once

#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#define PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE __host__ __device__
#else
#define PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE
#endif

namespace MVS {
namespace CUDA {

static constexpr uint32_t DVP_VISIBILITY_STATE_VERSION = 1u;
static constexpr float DVP_VISIBILITY_REPROJECTION_THRESHOLD = 2.f;
static constexpr float DVP_VISIBILITY_RELATIVE_DEPTH_THRESHOLD = 0.01f;
static constexpr uint8_t DVP_VISIBILITY_RESTORED_WEIGHT = 1u;
static constexpr unsigned DVP_VISIBILITY_MAX_VIEWS = 32u;

enum class DVPVisibilityMode : uint8_t {
	DISABLED = 0,
	PAPER_2D_RESTORE_V1 = 1,
	DEPTH_GATED_RESTORE_V1 = 2,
};

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE constexpr bool DVPVisibilityModeEnabled(unsigned mode)
{
	return mode == static_cast<unsigned>(DVPVisibilityMode::PAPER_2D_RESTORE_V1) ||
		mode == static_cast<unsigned>(DVPVisibilityMode::DEPTH_GATED_RESTORE_V1);
}

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE constexpr bool DVPVisibilityFinite(float value)
{
	return value >= -FLT_MAX && value <= FLT_MAX;
}

enum class DVPVisibilityConfigStatus : uint8_t {
	VALID = 0,
	INVALID_MODE,
	REQUIRES_FULL_APD,
	REQUIRES_GEOMETRIC_CONSISTENCY,
	INVALID_REPROJECTION_THRESHOLD,
	INVALID_RELATIVE_DEPTH_THRESHOLD,
};

struct DVPVisibilityConfig {
	unsigned mode = static_cast<unsigned>(DVPVisibilityMode::DISABLED);
	float reprojectionThreshold = DVP_VISIBILITY_REPROJECTION_THRESHOLD;
	float relativeDepthThreshold = DVP_VISIBILITY_RELATIVE_DEPTH_THRESHOLD;
	bool fullAPD = false;
	bool geometricConsistency = false;
};

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE constexpr DVPVisibilityConfigStatus
ValidateDVPVisibilityConfig(const DVPVisibilityConfig& config)
{
	if (config.mode > static_cast<unsigned>(DVPVisibilityMode::DEPTH_GATED_RESTORE_V1))
		return DVPVisibilityConfigStatus::INVALID_MODE;
	if (!DVPVisibilityFinite(config.reprojectionThreshold) ||
		config.reprojectionThreshold <= 0.f)
	{
		return DVPVisibilityConfigStatus::INVALID_REPROJECTION_THRESHOLD;
	}
	if (!DVPVisibilityFinite(config.relativeDepthThreshold) ||
		config.relativeDepthThreshold < 0.f)
	{
		return DVPVisibilityConfigStatus::INVALID_RELATIVE_DEPTH_THRESHOLD;
	}
	if (!DVPVisibilityModeEnabled(config.mode))
		return DVPVisibilityConfigStatus::VALID;
	if (!config.fullAPD)
		return DVPVisibilityConfigStatus::REQUIRES_FULL_APD;
	if (!config.geometricConsistency)
		return DVPVisibilityConfigStatus::REQUIRES_GEOMETRIC_CONSISTENCY;
	return DVPVisibilityConfigStatus::VALID;
}

enum class DVPVisibilityReason : uint8_t {
	MODE_DISABLED = 0,
	RETAINED_PREVIOUS_WEIGHT,
	INVALID_REFERENCE_DEPTH,
	FORWARD_OUT_OF_BOUNDS,
	INVALID_EXPECTED_SOURCE_DEPTH,
	INVALID_OBSERVED_SOURCE_DEPTH,
	BACKWARD_OUT_OF_BOUNDS,
	ROUND_TRIP_REJECTED,
	NEARER_OCCLUDER_REJECTED,
	DEPTH_DISAGREEMENT_REJECTED,
	RESTORED_PAPER_2D,
	RESTORED_DEPTH_GATED,
};

// Geometry is computed once from the accepted t-1 reference state. Candidate
// depth is deliberately absent, ensuring every candidate consumes one support
// set rather than selecting evidence through its own geometry.
struct DVPVisibilityObservation {
	float referenceDepth = -1.f;
	float expectedSourceDepth = -1.f;
	float observedSourceDepth = -1.f;
	float roundTripError = -1.f;
	bool forwardInside = false;
	bool backwardInside = false;
};

struct DVPVisibilityDecision {
	float relativeDepthError = -1.f;
	uint8_t previousWeight = 0u;
	uint8_t weight = 0u;
	DVPVisibilityReason reason = DVPVisibilityReason::MODE_DISABLED;
	bool visible = false;
	bool restored = false;
};

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE inline DVPVisibilityDecision ResolveDVPVisibility(
	const DVPVisibilityConfig& config,
	uint8_t previousWeight,
	const DVPVisibilityObservation& observation)
{
	DVPVisibilityDecision result;
	result.previousWeight = previousWeight;
	result.weight = previousWeight;
	result.visible = previousWeight > 0u;
	if (!DVPVisibilityModeEnabled(config.mode))
		return result;
	if (previousWeight > 0u) {
		result.reason = DVPVisibilityReason::RETAINED_PREVIOUS_WEIGHT;
		return result;
	}
	if (!DVPVisibilityFinite(observation.referenceDepth) || observation.referenceDepth <= 0.f) {
		result.reason = DVPVisibilityReason::INVALID_REFERENCE_DEPTH;
		return result;
	}
	if (!observation.forwardInside) {
		result.reason = DVPVisibilityReason::FORWARD_OUT_OF_BOUNDS;
		return result;
	}
	if (!DVPVisibilityFinite(observation.expectedSourceDepth) ||
		observation.expectedSourceDepth <= 0.f)
	{
		result.reason = DVPVisibilityReason::INVALID_EXPECTED_SOURCE_DEPTH;
		return result;
	}
	if (!DVPVisibilityFinite(observation.observedSourceDepth) ||
		observation.observedSourceDepth <= 0.f)
	{
		result.reason = DVPVisibilityReason::INVALID_OBSERVED_SOURCE_DEPTH;
		return result;
	}
	if (!observation.backwardInside) {
		result.reason = DVPVisibilityReason::BACKWARD_OUT_OF_BOUNDS;
		return result;
	}
	if (!DVPVisibilityFinite(observation.roundTripError) ||
		observation.roundTripError > config.reprojectionThreshold)
	{
		result.reason = DVPVisibilityReason::ROUND_TRIP_REJECTED;
		return result;
	}
	if (config.mode == static_cast<unsigned>(DVPVisibilityMode::PAPER_2D_RESTORE_V1)) {
		result.weight = DVP_VISIBILITY_RESTORED_WEIGHT;
		result.reason = DVPVisibilityReason::RESTORED_PAPER_2D;
		result.visible = true;
		result.restored = true;
		return result;
	}
	result.relativeDepthError =
		(observation.observedSourceDepth > observation.expectedSourceDepth ?
			observation.observedSourceDepth-observation.expectedSourceDepth :
			observation.expectedSourceDepth-observation.observedSourceDepth) /
		(observation.expectedSourceDepth > 1e-6f ? observation.expectedSourceDepth : 1e-6f);
	if (observation.observedSourceDepth <
		observation.expectedSourceDepth*(1.f-config.relativeDepthThreshold))
	{
		result.reason = DVPVisibilityReason::NEARER_OCCLUDER_REJECTED;
		return result;
	}
	if (!DVPVisibilityFinite(result.relativeDepthError) ||
		result.relativeDepthError > config.relativeDepthThreshold)
	{
		result.reason = DVPVisibilityReason::DEPTH_DISAGREEMENT_REJECTED;
		return result;
	}
	result.weight = DVP_VISIBILITY_RESTORED_WEIGHT;
	result.reason = DVPVisibilityReason::RESTORED_DEPTH_GATED;
	result.visible = true;
	result.restored = true;
	return result;
}

struct DVPVisibilityWeightSummary {
	uint32_t sum = 0u;
	uint8_t visibleCount = 0u;
	bool valid = false;
};

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE inline DVPVisibilityWeightSummary
SummarizeDVPVisibilityWeights(const uint8_t* weights, unsigned count)
{
	DVPVisibilityWeightSummary result;
	if (!weights || count == 0u || count > DVP_VISIBILITY_MAX_VIEWS)
		return result;
	for (unsigned view=0u; view<count; ++view) {
		result.sum += weights[view];
		result.visibleCount += weights[view] > 0u ? 1u : 0u;
	}
	result.valid = result.sum > 0u;
	return result;
}

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE constexpr float DVPVisibilityNormalizedWeight(
	uint8_t weight,
	uint32_t denominator)
{
	return denominator > 0u ? static_cast<float>(weight)/static_cast<float>(denominator) : 0.f;
}

struct DVPVisibilityStateHeader {
	uint32_t version = DVP_VISIBILITY_STATE_VERSION;
	uint32_t width = 0u;
	uint32_t height = 0u;
	uint32_t numViews = 0u;
	uint32_t logicalIteration = 0u;
};

enum class DVPVisibilityTransitionStatus : uint8_t {
	VALID = 0,
	INVALID_VERSION,
	INVALID_GEOMETRY,
	INVALID_VIEW_COUNT,
	INVALID_LOGICAL_ITERATION,
};

PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE constexpr DVPVisibilityTransitionStatus
ValidateDVPVisibilityTransition(
	const DVPVisibilityStateHeader& previous,
	const DVPVisibilityStateHeader& next)
{
	if (previous.version != DVP_VISIBILITY_STATE_VERSION ||
		next.version != DVP_VISIBILITY_STATE_VERSION)
	{
		return DVPVisibilityTransitionStatus::INVALID_VERSION;
	}
	if (previous.width == 0u || previous.height == 0u ||
		previous.width != next.width || previous.height != next.height)
	{
		return DVPVisibilityTransitionStatus::INVALID_GEOMETRY;
	}
	if (previous.numViews == 0u || previous.numViews > DVP_VISIBILITY_MAX_VIEWS ||
		previous.numViews != next.numViews)
	{
		return DVPVisibilityTransitionStatus::INVALID_VIEW_COUNT;
	}
	if (next.logicalIteration != previous.logicalIteration+1u)
		return DVPVisibilityTransitionStatus::INVALID_LOGICAL_ITERATION;
	return DVPVisibilityTransitionStatus::VALID;
}

struct DVPVisibilityCUDAOracleResult {
	DVPVisibilityDecision retained;
	DVPVisibilityDecision paperOccluder;
	DVPVisibilityDecision depthOccluder;
	DVPVisibilityDecision depthAgreement;
	DVPVisibilityDecision backwardOutOfBounds;
	DVPVisibilityWeightSummary weightSummary;
	DVPVisibilityTransitionStatus transition = DVPVisibilityTransitionStatus::INVALID_VERSION;
	float normalizedWeightSum = -1.f;
	uint32_t passedChecks = 0u;
};

} // namespace CUDA
} // namespace MVS

#undef PATCHMATCH_DVP_VISIBILITY_HOST_DEVICE
