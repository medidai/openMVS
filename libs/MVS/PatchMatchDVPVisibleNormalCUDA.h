/*
 * PatchMatchDVPVisibleNormalCUDA.h
 *
 * Shared host/device contract for DVP-MVS visible-normal constraints.
 * All normals and camera-to-point directions use reference-camera
 * coordinates. The contract is independent of source depth-map sampling.
 */

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE __host__ __device__
#else
#define PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE
#endif

namespace MVS {
namespace CUDA {

static constexpr unsigned DVP_VISIBLE_NORMAL_MAX_SOURCE_VIEWS = 32u;
static constexpr unsigned DVP_VISIBLE_NORMAL_MAX_DIRECTIONS =
	DVP_VISIBLE_NORMAL_MAX_SOURCE_VIEWS+1u;
static constexpr unsigned DVP_VISIBLE_NORMAL_MAX_ATTEMPTS = 1000u;
static constexpr unsigned DVP_VISIBLE_NORMAL_DEFAULT_ATTEMPTS = 200u;
static constexpr unsigned DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES = 8u;
static constexpr unsigned DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS = 2u;
static constexpr float DVP_VISIBLE_NORMAL_DEFAULT_DOT_TOLERANCE = 0.f;
static constexpr uint8_t DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE = 0xffu;

enum class DVPVisibleNormalMode : uint8_t {
	DISABLED = 0,
	SHADOW = 1,
	REFINEMENT = 2,
	PROPAGATION = 3,
	FULL = 4,
};

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr bool
DVPVisibleNormalModeEnabled(unsigned mode)
{
	return mode >= static_cast<unsigned>(DVPVisibleNormalMode::SHADOW) &&
		mode <= static_cast<unsigned>(DVPVisibleNormalMode::FULL);
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr bool
DVPVisibleNormalRefinementEnabled(unsigned mode)
{
	return mode == static_cast<unsigned>(DVPVisibleNormalMode::REFINEMENT) ||
		mode == static_cast<unsigned>(DVPVisibleNormalMode::FULL);
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr bool
DVPVisibleNormalPropagationEnabled(unsigned mode)
{
	return mode == static_cast<unsigned>(DVPVisibleNormalMode::PROPAGATION) ||
		mode == static_cast<unsigned>(DVPVisibleNormalMode::FULL);
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr bool
DVPVisibleNormalFinite(float value)
{
	return value >= -FLT_MAX && value <= FLT_MAX;
}

enum class DVPVisibleNormalConfigStatus : uint8_t {
	VALID = 0,
	INVALID_MODE,
	INVALID_DOT_TOLERANCE,
	INVALID_ATTEMPTS,
	REQUIRES_FULL_APD,
};

struct DVPVisibleNormalConfig {
	unsigned mode = static_cast<unsigned>(DVPVisibleNormalMode::DISABLED);
	float dotTolerance = DVP_VISIBLE_NORMAL_DEFAULT_DOT_TOLERANCE;
	unsigned attempts = DVP_VISIBLE_NORMAL_DEFAULT_ATTEMPTS;
	bool fullAPD = false;
};

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr DVPVisibleNormalConfigStatus
ValidateDVPVisibleNormalConfig(const DVPVisibleNormalConfig& config)
{
	if (config.mode > static_cast<unsigned>(DVPVisibleNormalMode::FULL))
		return DVPVisibleNormalConfigStatus::INVALID_MODE;
	if (!DVPVisibleNormalFinite(config.dotTolerance) ||
		config.dotTolerance < 0.f || config.dotTolerance > 1.f)
	{
		return DVPVisibleNormalConfigStatus::INVALID_DOT_TOLERANCE;
	}
	if (config.attempts == 0u || config.attempts > DVP_VISIBLE_NORMAL_MAX_ATTEMPTS)
		return DVPVisibleNormalConfigStatus::INVALID_ATTEMPTS;
	if (DVPVisibleNormalModeEnabled(config.mode) && !config.fullAPD)
		return DVPVisibleNormalConfigStatus::REQUIRES_FULL_APD;
	return DVPVisibleNormalConfigStatus::VALID;
}

struct DVPVisibleNormalVector {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
};

struct DVPVisibleNormalRotation {
	float values[9] = {
		1.f, 0.f, 0.f,
		0.f, 1.f, 0.f,
		0.f, 0.f, 1.f,
	};
};

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr DVPVisibleNormalVector
DVPVisibleNormalSubtract(
	const DVPVisibleNormalVector& first,
	const DVPVisibleNormalVector& second)
{
	return {
		first.x-second.x,
		first.y-second.y,
		first.z-second.z,
	};
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr float DVPVisibleNormalDot(
	const DVPVisibleNormalVector& first,
	const DVPVisibleNormalVector& second)
{
	return first.x*second.x+first.y*second.y+first.z*second.z;
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr float DVPVisibleNormalNormSquared(
	const DVPVisibleNormalVector& vector)
{
	return DVPVisibleNormalDot(vector, vector);
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline bool DVPVisibleNormalNormalize(
	const DVPVisibleNormalVector& input,
	DVPVisibleNormalVector& output)
{
	output = DVPVisibleNormalVector{};
	const float normSquared(DVPVisibleNormalNormSquared(input));
	if (!DVPVisibleNormalFinite(normSquared) || normSquared <= 1e-12f)
		return false;
	const float inverseNorm(1.f/sqrtf(normSquared));
	output = {input.x*inverseNorm, input.y*inverseNorm, input.z*inverseNorm};
	return DVPVisibleNormalFinite(output.x) && DVPVisibleNormalFinite(output.y) &&
		DVPVisibleNormalFinite(output.z);
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr DVPVisibleNormalVector
DVPVisibleNormalTransform(
	const DVPVisibleNormalRotation& rotation,
	const DVPVisibleNormalVector& vector)
{
	return {
		rotation.values[0]*vector.x+rotation.values[1]*vector.y+
			rotation.values[2]*vector.z,
		rotation.values[3]*vector.x+rotation.values[4]*vector.y+
			rotation.values[5]*vector.z,
		rotation.values[6]*vector.x+rotation.values[7]*vector.y+
			rotation.values[8]*vector.z,
	};
}

// OpenMVS poses use X_camera = R * (X_world-C). Express a source-camera
// center in the reference-camera frame without consulting a source depth map.
PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE constexpr DVPVisibleNormalVector
DVPVisibleNormalSourceCenterInReference(
	const DVPVisibleNormalRotation& referenceRotation,
	const DVPVisibleNormalVector& referenceCenterWorld,
	const DVPVisibleNormalVector& sourceCenterWorld)
{
	return DVPVisibleNormalTransform(
		referenceRotation,
		DVPVisibleNormalSubtract(sourceCenterWorld, referenceCenterWorld));
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline bool
DVPVisibleNormalCameraToPointDirection(
	const DVPVisibleNormalVector& pointInReference,
	const DVPVisibleNormalVector& cameraCenterInReference,
	DVPVisibleNormalVector& direction)
{
	return DVPVisibleNormalNormalize(
		DVPVisibleNormalSubtract(pointInReference, cameraCenterInReference), direction);
}

enum class DVPVisibleNormalEvaluationReason : uint8_t {
	VALID = 0,
	INVALID_NORMAL,
	INVALID_DIRECTION_COUNT,
	INVALID_DIRECTION,
	INVALID_DOT_TOLERANCE,
	HEMISPHERE_REJECTED,
};

struct DVPVisibleNormalEvaluation {
	DVPVisibleNormalVector normal;
	float maxDot = -FLT_MAX;
	float maxViolation = -FLT_MAX;
	uint8_t directionCount = 0u;
	uint8_t rejectedDirection = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	DVPVisibleNormalEvaluationReason reason =
		DVPVisibleNormalEvaluationReason::INVALID_NORMAL;
	bool valid = false;
	bool feasible = false;
};

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline DVPVisibleNormalEvaluation
EvaluateDVPVisibleNormal(
	const DVPVisibleNormalVector& normal,
	const DVPVisibleNormalVector* directions,
	unsigned directionCount,
	float dotTolerance)
{
	DVPVisibleNormalEvaluation result;
	if (!DVPVisibleNormalFinite(dotTolerance) || dotTolerance < 0.f || dotTolerance > 1.f) {
		result.reason = DVPVisibleNormalEvaluationReason::INVALID_DOT_TOLERANCE;
		return result;
	}
	if (!directions || directionCount == 0u ||
		directionCount > DVP_VISIBLE_NORMAL_MAX_DIRECTIONS)
	{
		result.reason = DVPVisibleNormalEvaluationReason::INVALID_DIRECTION_COUNT;
		return result;
	}
	if (!DVPVisibleNormalNormalize(normal, result.normal))
		return result;
	result.directionCount = static_cast<uint8_t>(directionCount);
	for (unsigned index=0u; index<directionCount; ++index) {
		DVPVisibleNormalVector direction;
		if (!DVPVisibleNormalNormalize(directions[index], direction)) {
			result.reason = DVPVisibleNormalEvaluationReason::INVALID_DIRECTION;
			result.rejectedDirection = static_cast<uint8_t>(index);
			return result;
		}
		const float dot(DVPVisibleNormalDot(result.normal, direction));
		if (dot > result.maxDot)
			result.maxDot = dot;
		if (dot > dotTolerance &&
			result.rejectedDirection == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE)
		{
			result.rejectedDirection = static_cast<uint8_t>(index);
		}
	}
	result.maxViolation = result.maxDot-dotTolerance;
	result.valid = true;
	result.feasible = result.rejectedDirection == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	result.reason = result.feasible ? DVPVisibleNormalEvaluationReason::VALID :
		DVPVisibleNormalEvaluationReason::HEMISPHERE_REJECTED;
	return result;
}

enum class DVPVisibleNormalProposalReason : uint8_t {
	INVALID_ARGUMENT = 0,
	NATIVE_FEASIBLE,
	CONSTRAINED_RETRY,
	RETRY_EXHAUSTED_NATIVE_FALLBACK,
};

struct DVPVisibleNormalProposalDecision {
	DVPVisibleNormalVector selected;
	DVPVisibleNormalEvaluation nativeEvaluation;
	DVPVisibleNormalEvaluation selectedEvaluation;
	DVPVisibleNormalProposalReason reason =
		DVPVisibleNormalProposalReason::INVALID_ARGUMENT;
	uint16_t retriesTested = 0u;
	uint16_t selectedRetry = 0xffffu;
	bool valid = false;
	bool fallback = false;
};

// The caller generates retryNormals from a copied RNG state. The native
// proposal always remains available, so exhaustion is explicit and cannot
// silently remove a PatchMatch candidate.
PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline DVPVisibleNormalProposalDecision
ResolveDVPVisibleNormalProposal(
	const DVPVisibleNormalVector& nativeNormal,
	const DVPVisibleNormalVector* retryNormals,
	unsigned retryCount,
	const DVPVisibleNormalVector* directions,
	unsigned directionCount,
	float dotTolerance)
{
	DVPVisibleNormalProposalDecision result;
	result.selected = nativeNormal;
	result.nativeEvaluation = EvaluateDVPVisibleNormal(
		nativeNormal, directions, directionCount, dotTolerance);
	result.selectedEvaluation = result.nativeEvaluation;
	if (retryCount > DVP_VISIBLE_NORMAL_MAX_ATTEMPTS ||
		(retryCount > 0u && !retryNormals) || !result.nativeEvaluation.valid)
	{
		return result;
	}
	result.valid = true;
	if (result.nativeEvaluation.feasible) {
		result.reason = DVPVisibleNormalProposalReason::NATIVE_FEASIBLE;
		return result;
	}
	for (unsigned retry=0u; retry<retryCount; ++retry) {
		const DVPVisibleNormalEvaluation evaluation(EvaluateDVPVisibleNormal(
			retryNormals[retry], directions, directionCount, dotTolerance));
		++result.retriesTested;
		if (!evaluation.valid)
			continue;
		if (evaluation.feasible) {
			result.selected = retryNormals[retry];
			result.selectedEvaluation = evaluation;
			result.selectedRetry = static_cast<uint16_t>(retry);
			result.reason = DVPVisibleNormalProposalReason::CONSTRAINED_RETRY;
			return result;
		}
	}
	result.reason = DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK;
	result.fallback = true;
	return result;
}

enum class DVPVisibleNormalPropagationReason : uint8_t {
	INVALID_ARGUMENT = 0,
	NO_VALID_CANDIDATE,
	CONSTRAINED_FEASIBLE,
	NO_FEASIBLE_NATIVE_FALLBACK,
};

struct DVPVisibleNormalPropagationDecision {
	float selectedCost = FLT_MAX;
	uint8_t nativeBest = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t constrainedBest = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t selected = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t validCount = 0u;
	uint8_t feasibleCount = 0u;
	DVPVisibleNormalPropagationReason reason =
		DVPVisibleNormalPropagationReason::INVALID_ARGUMENT;
	bool valid = false;
	bool fallback = false;
};

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline bool DVPVisibleNormalValidCost(float cost)
{
	return DVPVisibleNormalFinite(cost) && cost >= 0.f && cost < FLT_MAX;
}

PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE inline DVPVisibleNormalPropagationDecision
ResolveDVPVisibleNormalPropagation(
	const float* costs,
	const uint8_t* validCandidates,
	const uint8_t* feasibleCandidates,
	unsigned candidateCount)
{
	DVPVisibleNormalPropagationDecision result;
	if (!costs || !validCandidates || !feasibleCandidates || candidateCount == 0u ||
		candidateCount > DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES)
	{
		return result;
	}
	float nativeCost(FLT_MAX);
	float constrainedCost(FLT_MAX);
	for (unsigned index=0u; index<candidateCount; ++index) {
		if (!validCandidates[index] || !DVPVisibleNormalValidCost(costs[index]))
			continue;
		++result.validCount;
		if (costs[index] < nativeCost) {
			nativeCost = costs[index];
			result.nativeBest = static_cast<uint8_t>(index);
		}
		if (!feasibleCandidates[index])
			continue;
		++result.feasibleCount;
		if (costs[index] < constrainedCost) {
			constrainedCost = costs[index];
			result.constrainedBest = static_cast<uint8_t>(index);
		}
	}
	result.valid = true;
	if (result.nativeBest == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE) {
		result.reason = DVPVisibleNormalPropagationReason::NO_VALID_CANDIDATE;
		return result;
	}
	if (result.constrainedBest != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE) {
		result.selected = result.constrainedBest;
		result.selectedCost = constrainedCost;
		result.reason = DVPVisibleNormalPropagationReason::CONSTRAINED_FEASIBLE;
		return result;
	}
	result.selected = result.nativeBest;
	result.selectedCost = nativeCost;
	result.reason = DVPVisibleNormalPropagationReason::NO_FEASIBLE_NATIVE_FALLBACK;
	result.fallback = true;
	return result;
}

struct DVPVisibleNormalCUDAOracleResult {
	DVPVisibleNormalVector sourceCenter;
	DVPVisibleNormalVector sourceDirection;
	DVPVisibleNormalEvaluation referenceFeasible;
	DVPVisibleNormalEvaluation multiViewFeasible;
	DVPVisibleNormalEvaluation rejected;
	DVPVisibleNormalEvaluation contradictory;
	DVPVisibleNormalProposalDecision retryDecision;
	DVPVisibleNormalProposalDecision fallbackDecision;
	DVPVisibleNormalPropagationDecision propagationDecision;
	DVPVisibleNormalPropagationDecision propagationFallback;
	uint32_t passedChecks = 0u;
};

} // namespace CUDA
} // namespace MVS

#undef PATCHMATCH_DVP_VISIBLE_NORMAL_HOST_DEVICE
