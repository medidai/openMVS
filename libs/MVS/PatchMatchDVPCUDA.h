/*
 * PatchMatchDVPCUDA.h
 *
 * Shared host/device contract for the DVP-MVS epipolar depth proposals.
 * Paper mechanics and explicitly named historical compatibility behavior live
 * here so CPU fixtures, CUDA oracles, and production kernels use one policy.
 */

#pragma once

#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#define PATCHMATCH_DVP_HOST_DEVICE __host__ __device__
#else
#define PATCHMATCH_DVP_HOST_DEVICE
#endif

namespace MVS {
namespace CUDA {

static constexpr unsigned DVP_MAX_SOURCE_VIEWS = 32u;
static constexpr unsigned DVP_MAX_PROPOSALS = 2u;
static constexpr float DVP_EPIPOLAR_ALPHA = 1.f;
static constexpr float DVP_EPIPOLAR_BETA = 4.f;
static constexpr unsigned DVP_EPIPOLAR_MU = 3u;
static constexpr unsigned DVP_GLOBAL_SEARCH_RADIUS = 160u;
static constexpr float DVP_GLOBAL_REPROJECTION_THRESHOLD = 2.f;
static constexpr float DVP_GLOBAL_RELATIVE_DEPTH_THRESHOLD = 0.01f;

enum class DVPEpipolarFamily : uint8_t {
	DISABLED = 0,
	HISTORICAL_GLOBAL_V0 = 1,
	GLOBAL_SEARCH_GATED_V1 = 2,
	HISTORICAL_MIDPOINT_V1 = 3,
	DVP_EQ11_INTERVAL_V1 = 4,
};

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPEpipolarFamilyEnabled(unsigned family)
{
	return family >= static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) &&
		family <= static_cast<unsigned>(DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1);
}

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPFamilyUsesGlobalSearch(unsigned family)
{
	return family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) ||
		family == static_cast<unsigned>(DVPEpipolarFamily::GLOBAL_SEARCH_GATED_V1);
}

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPFamilyUsesPaperIntervals(unsigned family)
{
	return family == static_cast<unsigned>(DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1);
}

enum class DVPConfigStatus : uint8_t {
	VALID = 0,
	INVALID_FAMILY,
	INVALID_OFFSETS,
	INVALID_SUPPORT,
	INVALID_SEARCH_RADIUS,
	INVALID_REPROJECTION_THRESHOLD,
	INVALID_DEPTH_THRESHOLD,
};

struct DVPConfig {
	unsigned family = static_cast<unsigned>(DVPEpipolarFamily::DISABLED);
	float alpha = DVP_EPIPOLAR_ALPHA;
	float beta = DVP_EPIPOLAR_BETA;
	unsigned mu = DVP_EPIPOLAR_MU;
	unsigned searchRadius = DVP_GLOBAL_SEARCH_RADIUS;
	float reprojectionThreshold = DVP_GLOBAL_REPROJECTION_THRESHOLD;
	float relativeDepthThreshold = DVP_GLOBAL_RELATIVE_DEPTH_THRESHOLD;
};

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPFinite(float value)
{
	return value >= -FLT_MAX && value <= FLT_MAX;
}

PATCHMATCH_DVP_HOST_DEVICE constexpr DVPConfigStatus ValidateDVPConfig(const DVPConfig& config)
{
	if (config.family > static_cast<unsigned>(DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1))
		return DVPConfigStatus::INVALID_FAMILY;
	if (!DVPFinite(config.alpha) || !DVPFinite(config.beta) ||
		!DVPFinite(config.alpha+config.beta) ||
		config.alpha <= 0.f || config.beta <= 0.f)
		return DVPConfigStatus::INVALID_OFFSETS;
	if (config.mu == 0u || config.mu > DVP_MAX_SOURCE_VIEWS)
		return DVPConfigStatus::INVALID_SUPPORT;
	if (config.searchRadius == 0u || config.searchRadius > 1024u)
		return DVPConfigStatus::INVALID_SEARCH_RADIUS;
	if (!DVPFinite(config.reprojectionThreshold) || config.reprojectionThreshold <= 0.f)
		return DVPConfigStatus::INVALID_REPROJECTION_THRESHOLD;
	if (!DVPFinite(config.relativeDepthThreshold) || config.relativeDepthThreshold < 0.f)
		return DVPConfigStatus::INVALID_DEPTH_THRESHOLD;
	return DVPConfigStatus::VALID;
}

enum class DVPEpipolarUnavailableReason : uint8_t {
	NONE = 0,
	FAMILY_DISABLED,
	GEOMETRY_UNAVAILABLE,
	INVALID_REFERENCE_DEPTH,
	NO_SELECTED_SOURCE_VIEW,
	INVALID_EPIPOLAR_DIRECTION,
	INSUFFICIENT_ENDPOINT_SUPPORT,
	INVALID_INTERVAL_ORDER,
	NO_FINITE_GLOBAL_CANDIDATE,
	GLOBAL_REPROJECTION_GATE,
	GLOBAL_SUPPORT_GATE,
	GLOBAL_OCCLUSION_REJECTED,
};

struct DVPEndpointSamples {
	float leftOuter[DVP_MAX_SOURCE_VIEWS] = {};
	float leftInner[DVP_MAX_SOURCE_VIEWS] = {};
	float rightInner[DVP_MAX_SOURCE_VIEWS] = {};
	float rightOuter[DVP_MAX_SOURCE_VIEWS] = {};
	uint8_t leftOuterCount = 0u;
	uint8_t leftInnerCount = 0u;
	uint8_t rightInnerCount = 0u;
	uint8_t rightOuterCount = 0u;
	uint32_t selectedSourceViews = 0u;
	uint32_t leftOuterViews = 0u;
	uint32_t leftInnerViews = 0u;
	uint32_t rightInnerViews = 0u;
	uint32_t rightOuterViews = 0u;
};

PATCHMATCH_DVP_HOST_DEVICE inline bool DVPAppendEndpoint(
	float* values, uint8_t& count, float value)
{
	if (!values || !DVPFinite(value) || count >= DVP_MAX_SOURCE_VIEWS)
		return false;
	values[count++] = value;
	return true;
}

PATCHMATCH_DVP_HOST_DEVICE inline void DVPSortAscending(float* values, unsigned count)
{
	for (unsigned i=1u; i<count; ++i) {
		const float value(values[i]);
		unsigned j(i);
		while (j > 0u && values[j-1u] > value) {
			values[j] = values[j-1u];
			--j;
		}
		values[j] = value;
	}
}

PATCHMATCH_DVP_HOST_DEVICE inline bool DVPNthSmallest(
	float* values, unsigned count, unsigned order, float& value)
{
	value = -1.f;
	if (!values || order == 0u || count < order)
		return false;
	DVPSortAscending(values, count);
	value = values[order-1u];
	return DVPFinite(value);
}

PATCHMATCH_DVP_HOST_DEVICE inline bool DVPNthLargest(
	float* values, unsigned count, unsigned order, float& value)
{
	value = -1.f;
	if (!values || order == 0u || count < order)
		return false;
	DVPSortAscending(values, count);
	value = values[count-order];
	return DVPFinite(value);
}

struct DVPDepthInterval {
	float minimum = -1.f;
	float maximum = -1.f;
	float outerStatistic = -1.f;
	float innerStatistic = -1.f;
	uint8_t outerSupport = 0u;
	uint8_t innerSupport = 0u;
	uint8_t order = 0u;
	bool valid = false;
};

struct DVPIntervalSet {
	DVPDepthInterval left;
	DVPDepthInterval right;
	DVPEpipolarUnavailableReason reason = DVPEpipolarUnavailableReason::NONE;
};

PATCHMATCH_DVP_HOST_DEVICE inline DVPIntervalSet DVPBuildPaperIntervals(
	DVPEndpointSamples samples, unsigned mu)
{
	DVPIntervalSet result;
	if (mu == 0u || mu > DVP_MAX_SOURCE_VIEWS) {
		result.reason = DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT;
		return result;
	}
	result.left.outerSupport = samples.leftOuterCount;
	result.left.innerSupport = samples.leftInnerCount;
	result.left.order = static_cast<uint8_t>(mu);
	result.right.outerSupport = samples.rightOuterCount;
	result.right.innerSupport = samples.rightInnerCount;
	result.right.order = static_cast<uint8_t>(mu);
	const bool leftSupported(
		samples.leftOuterCount >= mu && samples.leftInnerCount >= mu);
	const bool rightSupported(
		samples.rightInnerCount >= mu && samples.rightOuterCount >= mu);

	float leftOuter(-1.f), leftInner(-1.f);
	if (leftSupported &&
		DVPNthSmallest(samples.leftOuter, samples.leftOuterCount, mu, leftOuter) &&
		DVPNthLargest(samples.leftInner, samples.leftInnerCount, mu, leftInner))
	{
		result.left.outerStatistic = leftOuter;
		result.left.innerStatistic = leftInner;
		result.left.minimum = leftOuter;
		result.left.maximum = leftInner;
		result.left.valid = leftOuter < leftInner;
	}

	float rightInner(-1.f), rightOuter(-1.f);
	if (rightSupported &&
		DVPNthLargest(samples.rightInner, samples.rightInnerCount, mu, rightInner) &&
		DVPNthSmallest(samples.rightOuter, samples.rightOuterCount, mu, rightOuter))
	{
		result.right.outerStatistic = rightOuter;
		result.right.innerStatistic = rightInner;
		result.right.minimum = rightInner;
		result.right.maximum = rightOuter;
		result.right.valid = rightInner < rightOuter;
	}

	if (!result.left.valid && !result.right.valid)
		result.reason = !leftSupported && !rightSupported ?
			DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT :
			DVPEpipolarUnavailableReason::INVALID_INTERVAL_ORDER;
	return result;
}

// Safe donor-compatibility midpoint policy. It preserves the historical
// endpoint ordering but deliberately rejects under-supported statistics.
PATCHMATCH_DVP_HOST_DEVICE inline DVPIntervalSet DVPBuildHistoricalMidpointIntervals(
	DVPEndpointSamples samples, unsigned mu)
{
	DVPIntervalSet result;
	if (mu == 0u || mu > DVP_MAX_SOURCE_VIEWS) {
		result.reason = DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT;
		return result;
	}
	result.left.outerSupport = samples.leftOuterCount;
	result.left.innerSupport = samples.leftInnerCount;
	result.left.order = static_cast<uint8_t>(mu);
	result.right.outerSupport = samples.rightOuterCount;
	result.right.innerSupport = samples.rightInnerCount;
	result.right.order = static_cast<uint8_t>(mu);
	if (samples.leftOuterCount < mu || samples.leftInnerCount < mu ||
		samples.rightInnerCount < mu || samples.rightOuterCount < mu)
	{
		result.reason = DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT;
		return result;
	}

	float leftHistoricalInner(-1.f), leftHistoricalOuter(-1.f);
	if (DVPNthSmallest(samples.leftInner, samples.leftInnerCount, mu, leftHistoricalInner) &&
		DVPNthLargest(samples.leftOuter, samples.leftOuterCount, mu, leftHistoricalOuter))
	{
		result.left.innerStatistic = leftHistoricalInner;
		result.left.outerStatistic = leftHistoricalOuter;
		result.left.minimum = leftHistoricalInner < leftHistoricalOuter ?
			leftHistoricalInner : leftHistoricalOuter;
		result.left.maximum = leftHistoricalInner > leftHistoricalOuter ?
			leftHistoricalInner : leftHistoricalOuter;
		result.left.valid = result.left.minimum < result.left.maximum;
	}

	float rightHistoricalInner(-1.f), rightHistoricalOuter(-1.f);
	if (DVPNthSmallest(samples.rightInner, samples.rightInnerCount, mu, rightHistoricalInner) &&
		DVPNthLargest(samples.rightOuter, samples.rightOuterCount, mu, rightHistoricalOuter))
	{
		result.right.innerStatistic = rightHistoricalInner;
		result.right.outerStatistic = rightHistoricalOuter;
		result.right.minimum = rightHistoricalInner < rightHistoricalOuter ?
			rightHistoricalInner : rightHistoricalOuter;
		result.right.maximum = rightHistoricalInner > rightHistoricalOuter ?
			rightHistoricalInner : rightHistoricalOuter;
		result.right.valid = result.right.minimum < result.right.maximum;
	}

	if (!result.left.valid && !result.right.valid) {
		result.reason = DVPEpipolarUnavailableReason::INVALID_INTERVAL_ORDER;
	}
	return result;
}

PATCHMATCH_DVP_HOST_DEVICE inline float DVPIntervalMidpoint(const DVPDepthInterval& interval)
{
	return interval.valid ? 0.5f*(interval.minimum+interval.maximum) : -1.f;
}

PATCHMATCH_DVP_HOST_DEVICE inline float DVPInterpolateInterval(
	const DVPDepthInterval& interval, float unitSample)
{
	if (!interval.valid || !DVPFinite(unitSample))
		return -1.f;
	const float clamped(unitSample < 0.f ? 0.f : (unitSample > 1.f ? 1.f : unitSample));
	return interval.minimum+clamped*(interval.maximum-interval.minimum);
}

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPIntervalContains(
	const DVPDepthInterval& interval, float depth)
{
	return interval.valid && DVPFinite(depth) &&
		depth >= interval.minimum && depth <= interval.maximum;
}

struct DVPProposalDecision {
	float incumbentCost = -1.f;
	float candidateCosts[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float winnerCost = -1.f;
	float runnerUpCost = -1.f;
	float winnerRunnerUpGap = -1.f;
	uint8_t proposalCount = 0u;
	uint8_t finiteCount = 0u;
	uint8_t winnerOrdinal = 0xffu;
	bool accepted = false;
};

PATCHMATCH_DVP_HOST_DEVICE constexpr bool DVPValidProposalCost(float value)
{
	return DVPFinite(value) && value >= 0.f;
}

PATCHMATCH_DVP_HOST_DEVICE inline DVPProposalDecision ResolveDVPProposalDecision(
	float incumbentCost, const float* candidateCosts, unsigned proposalCount)
{
	DVPProposalDecision result;
	result.incumbentCost = incumbentCost;
	result.winnerCost = incumbentCost;
	result.proposalCount = static_cast<uint8_t>(
		proposalCount < DVP_MAX_PROPOSALS ? proposalCount : DVP_MAX_PROPOSALS);
	float best(incumbentCost);
	float second(FLT_MAX);
	if (!DVPValidProposalCost(best))
		best = FLT_MAX;
	for (unsigned i=0u; i<result.proposalCount; ++i) {
		const float candidate(candidateCosts ? candidateCosts[i] : FLT_MAX);
		result.candidateCosts[i] = DVPValidProposalCost(candidate) ? candidate : -1.f;
		if (!DVPValidProposalCost(candidate))
			continue;
		++result.finiteCount;
		if (candidate < best) {
			second = best;
			best = candidate;
			result.winnerOrdinal = static_cast<uint8_t>(i);
		} else if (candidate < second) {
			second = candidate;
		}
	}
	result.accepted = result.winnerOrdinal != 0xffu;
	result.winnerCost = best < FLT_MAX ? best : -1.f;
	result.runnerUpCost = second < FLT_MAX ? second : -1.f;
	result.winnerRunnerUpGap = result.runnerUpCost >= 0.f && result.winnerCost >= 0.f ?
		result.runnerUpCost-result.winnerCost : -1.f;
	return result;
}

struct DVPCUDAOracleResult {
	float roundTripPixelError = -1.f;
	float roundTripDepthError = -1.f;
	float rngControlFirst = -1.f;
	float rngControlNext = -1.f;
	float rngProposalFirst = -1.f;
	float rngProposalSecond = -1.f;
	float rngDVPNext = -1.f;
	DVPIntervalSet paperIntervals;
	DVPProposalDecision proposalDecision;
	uint32_t passedChecks = 0u;
};

} // namespace CUDA
} // namespace MVS

#undef PATCHMATCH_DVP_HOST_DEVICE
