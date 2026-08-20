/*
 * PatchMatchAPDCUDA.h
 *
 * Paper-mechanics contract for Adaptive Patch Deformation (CVPR 2023).
 * The helpers are shared by CPU fixtures and CUDA kernels so policy cannot
 * silently drift between the oracle and production implementation.
 */

#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define PATCHMATCH_APD_HOST_DEVICE __host__ __device__
#else
#define PATCHMATCH_APD_HOST_DEVICE
#endif

namespace MVS {
namespace CUDA {

static constexpr unsigned APD_PROFILE_RADIUS = 30u;
static constexpr unsigned APD_PROFILE_SIZE = 2u*APD_PROFILE_RADIUS+1u;
static constexpr unsigned APD_SECTOR_COUNT = 32u;
static constexpr unsigned APD_MAX_ANCHORS = 8u;
static constexpr unsigned APD_RANSAC_TRIALS = 50u;
static constexpr unsigned APD_MIN_INLIERS = 6u;
// A symmetric integer lattice cannot contain exactly 100 samples about its
// center. OpenMVS uses the documented 100x100 paper neighborhood as the
// deterministic compatibility window [-50,+50] (101x101), recorded as such.
static constexpr int APD_NEAREST_SEARCH_RADIUS = 50;
static constexpr unsigned APD_SPOKE_ATTEMPTS_PER_RADIUS = 4u;
static constexpr float APD_PROFILE_MAX_COST = 0.50f;
static constexpr float APD_SINGLE_MINIMUM_MAX_COST = 0.15f;
static constexpr float APD_MINIMUM_SEPARATION = 0.20f;
static constexpr float APD_RANSAC_NORMALIZED_THRESHOLD_MAX = 0.010f;
static constexpr float APD_RANSAC_NORMALIZED_THRESHOLD_MIN = 0.005f;
static constexpr float APD_CENTER_WEIGHT = 0.25f;
static constexpr float APD_ANCHOR_WEIGHT = 0.75f;
static constexpr float APD_BAD_COST = 1.20f;
static constexpr float APD_MAX_NCC_COST = 2.f;
static constexpr float APD_VIEW_PRIOR_SELECTED = 0.9f;
static constexpr float APD_VIEW_PRIOR_REJECTED = 0.1f;
static constexpr float APD_VIEW_COST_THRESHOLD = 0.8f;
static constexpr float APD_VIEW_COST_ITERATION_SIGMA = 4.f;
static constexpr float APD_VIEW_AGREEMENT_SIGMA = 0.3f;
static constexpr float APD_VIEW_FALLBACK_SIGMA = 0.4f;
static constexpr unsigned APD_VIEW_MIN_AGREEMENT = 3u;
static constexpr unsigned APD_VIEW_MAX_BAD = 2u;
static constexpr int APD_PATCH_RADIUS = 5;
static constexpr int APD_CENTER_PATCH_INCREMENT = 2;
static constexpr int APD_ANCHOR_PATCH_INCREMENT = 5;
static constexpr unsigned APD_CENTER_PATCH_AXIS_SAMPLES = 6u;
static constexpr unsigned APD_CENTER_PATCH_SAMPLES = 36u;
static constexpr unsigned APD_ANCHOR_PATCH_AXIS_SAMPLES = 3u;
static constexpr unsigned APD_ANCHOR_PATCH_SAMPLES = 9u;
static constexpr int APD_FINAL_REFINEMENT_RADIUS = 5;
static constexpr float APD_FINAL_REFINEMENT_MIN_IMPROVEMENT = 0.10f;
static constexpr float APD_PI = 3.14159265358979323846f;
static constexpr uint32_t APD_MULTISCALE_STATE_VERSION = 1u;
static constexpr float APD_RANSAC_NORMALIZED_THRESHOLD_LEVEL_STEP = 0.00125f;

// APD's outer clock is deliberately distinct from PatchMatch's inner
// checkerboard iteration. levelIndex advances from coarsest to finest, while
// stageIndex also advances through later geometric-consistency passes.
struct APDStageClock {
	unsigned levelIndex = 0u;
	unsigned levelCount = 1u;
	unsigned stageIndex = 0u;
	unsigned logicalIteration = 0u;
	bool hasTransferredState = false;
	bool geometricConsistency = false;
};

enum class APDStageClockStatus : uint8_t {
	VALID = 0,
	INVALID_LEVEL_COUNT,
	INVALID_LEVEL_INDEX,
	INVALID_STAGE_INDEX,
};

struct APDStageSchedule {
	bool valid = false;
	bool conventional = true;
	bool consumesTransferredState = false;
	unsigned reliabilityEta = 6u;
	float ransacNormalizedThreshold = APD_RANSAC_NORMALIZED_THRESHOLD_MAX;
};

// This header accompanies every host-side reliability/anchor provenance map.
// The maps themselves remain runtime-only state and are never folded into the
// stable DMAP file format.
struct APDMultiscaleStateHeader {
	uint32_t version = APD_MULTISCALE_STATE_VERSION;
	uint32_t width = 0u;
	uint32_t height = 0u;
	uint32_t sourceLevelIndex = 0u;
	uint32_t sourceStageIndex = 0u;
};

enum class APDMultiscaleTransferStatus : uint8_t {
	VALID = 0,
	UNAVAILABLE_NO_SOURCE,
	INVALID_VERSION,
	INVALID_SOURCE_SIZE,
	INVALID_DESTINATION_SIZE,
	INVALID_LEVEL,
	INVALID_STAGE,
};

PATCHMATCH_APD_HOST_DEVICE constexpr bool APDFinite(float value)
{
	return value >= -FLT_MAX && value <= FLT_MAX;
}

PATCHMATCH_APD_HOST_DEVICE constexpr APDStageClockStatus ValidateAPDStageClock(
	const APDStageClock& clock)
{
	if (clock.levelCount == 0u)
		return APDStageClockStatus::INVALID_LEVEL_COUNT;
	if (clock.levelIndex >= clock.levelCount)
		return APDStageClockStatus::INVALID_LEVEL_INDEX;
	if (clock.stageIndex < clock.levelIndex)
		return APDStageClockStatus::INVALID_STAGE_INDEX;
	return APDStageClockStatus::VALID;
}

PATCHMATCH_APD_HOST_DEVICE inline APDStageSchedule ResolveAPDStageSchedule(
	const APDStageClock& clock)
{
	APDStageSchedule schedule;
	if (ValidateAPDStageClock(clock) != APDStageClockStatus::VALID)
		return schedule;
	schedule.valid = true;
	schedule.conventional = !clock.hasTransferredState;
	schedule.consumesTransferredState = clock.hasTransferredState;
	const unsigned reliabilityStage(clock.stageIndex < 2u ? clock.stageIndex : 2u);
	schedule.reliabilityEta = 6u-2u*reliabilityStage;
	const float unclampedThreshold(
		APD_RANSAC_NORMALIZED_THRESHOLD_MAX-
		static_cast<float>(clock.levelIndex)*APD_RANSAC_NORMALIZED_THRESHOLD_LEVEL_STEP);
	schedule.ransacNormalizedThreshold =
		unclampedThreshold > APD_RANSAC_NORMALIZED_THRESHOLD_MIN ?
		unclampedThreshold : APD_RANSAC_NORMALIZED_THRESHOLD_MIN;
	return schedule;
}

// Mirrors OpenCV INTER_NEAREST's source-pixel choice for the explicit APD
// provenance maps. 64-bit arithmetic keeps large-image products defined.
PATCHMATCH_APD_HOST_DEVICE constexpr unsigned APDNearestSourceCoordinate(
	unsigned destinationCoordinate,
	unsigned sourceExtent,
	unsigned destinationExtent)
{
	if (sourceExtent == 0u || destinationExtent == 0u)
		return 0u;
	const uint64_t sourceCoordinate(
		(static_cast<uint64_t>(destinationCoordinate)*sourceExtent)/destinationExtent);
	return static_cast<unsigned>(
		sourceCoordinate < sourceExtent ? sourceCoordinate : sourceExtent-1u);
}

PATCHMATCH_APD_HOST_DEVICE constexpr APDMultiscaleTransferStatus ValidateAPDMultiscaleTransfer(
	const APDMultiscaleStateHeader& state,
	const APDStageClock& destination,
	unsigned destinationWidth,
	unsigned destinationHeight)
{
	if (state.version != APD_MULTISCALE_STATE_VERSION)
		return APDMultiscaleTransferStatus::INVALID_VERSION;
	if (state.width == 0u || state.height == 0u)
		return APDMultiscaleTransferStatus::INVALID_SOURCE_SIZE;
	if (destinationWidth == 0u || destinationHeight == 0u)
		return APDMultiscaleTransferStatus::INVALID_DESTINATION_SIZE;
	if (ValidateAPDStageClock(destination) != APDStageClockStatus::VALID ||
		state.sourceLevelIndex >= destination.levelCount ||
		state.sourceLevelIndex > destination.levelIndex ||
		(destination.levelIndex-state.sourceLevelIndex) > 1u)
	{
		return APDMultiscaleTransferStatus::INVALID_LEVEL;
	}
	// The runtime mailbox carries exactly one stage. Accepting any older state
	// would make a skipped or stale transfer look valid while applying the wrong
	// reliability and anchor provenance to the destination stage.
	if (state.sourceStageIndex < state.sourceLevelIndex ||
		destination.stageIndex == 0u ||
		state.sourceStageIndex != destination.stageIndex-1u)
		return APDMultiscaleTransferStatus::INVALID_STAGE;
	if (state.sourceLevelIndex == destination.levelIndex &&
		(state.width != destinationWidth || state.height != destinationHeight))
	{
		return APDMultiscaleTransferStatus::INVALID_DESTINATION_SIZE;
	}
	if (state.sourceLevelIndex < destination.levelIndex &&
		(state.width > destinationWidth || state.height > destinationHeight))
	{
		return APDMultiscaleTransferStatus::INVALID_DESTINATION_SIZE;
	}
	return APDMultiscaleTransferStatus::VALID;
}

enum class APDMode : uint8_t {
	DISABLED = 0,
	DEFORMABLE_COST = 1,
};

enum class APDConfigStatus : uint8_t {
	VALID = 0,
	INVALID_MODE,
	INVALID_PROFILE_POLICY,
	INVALID_ANCHOR_POLICY,
	INVALID_RANSAC_POLICY,
	INVALID_SCORE_POLICY,
};

struct APDConfig {
	unsigned mode = static_cast<unsigned>(APDMode::DISABLED);
	unsigned profileRadius = APD_PROFILE_RADIUS;
	unsigned profileSize = APD_PROFILE_SIZE;
	unsigned sectorCount = APD_SECTOR_COUNT;
	unsigned maxAnchors = APD_MAX_ANCHORS;
	unsigned ransacTrials = APD_RANSAC_TRIALS;
	unsigned minimumInliers = APD_MIN_INLIERS;
	float profileMaxCost = APD_PROFILE_MAX_COST;
	float singleMinimumMaxCost = APD_SINGLE_MINIMUM_MAX_COST;
	float minimumSeparation = APD_MINIMUM_SEPARATION;
	float ransacNormalizedThresholdMax = APD_RANSAC_NORMALIZED_THRESHOLD_MAX;
	float ransacNormalizedThresholdMin = APD_RANSAC_NORMALIZED_THRESHOLD_MIN;
	float centerWeight = APD_CENTER_WEIGHT;
	float anchorWeight = APD_ANCHOR_WEIGHT;
};

PATCHMATCH_APD_HOST_DEVICE constexpr APDConfigStatus ValidateAPDConfig(const APDConfig& config)
{
	if (config.mode > static_cast<unsigned>(APDMode::DEFORMABLE_COST))
		return APDConfigStatus::INVALID_MODE;
	if (config.profileRadius != APD_PROFILE_RADIUS || config.profileSize != APD_PROFILE_SIZE ||
		config.profileMaxCost != APD_PROFILE_MAX_COST ||
		config.singleMinimumMaxCost != APD_SINGLE_MINIMUM_MAX_COST ||
		config.minimumSeparation != APD_MINIMUM_SEPARATION)
		return APDConfigStatus::INVALID_PROFILE_POLICY;
	if (config.sectorCount != APD_SECTOR_COUNT || config.maxAnchors != APD_MAX_ANCHORS)
		return APDConfigStatus::INVALID_ANCHOR_POLICY;
	if (config.ransacTrials != APD_RANSAC_TRIALS ||
		config.minimumInliers != APD_MIN_INLIERS ||
		config.minimumInliers > config.sectorCount ||
		config.ransacNormalizedThresholdMax != APD_RANSAC_NORMALIZED_THRESHOLD_MAX ||
		config.ransacNormalizedThresholdMin != APD_RANSAC_NORMALIZED_THRESHOLD_MIN)
		return APDConfigStatus::INVALID_RANSAC_POLICY;
	if (config.centerWeight != APD_CENTER_WEIGHT || config.anchorWeight != APD_ANCHOR_WEIGHT ||
		config.centerWeight+config.anchorWeight != 1.f)
		return APDConfigStatus::INVALID_SCORE_POLICY;
	return APDConfigStatus::VALID;
}

enum class APDReliabilityClass : uint8_t {
	UNKNOWN = 0,
	UNRELIABLE = 1,
	RELIABLE = 2,
};

enum class APDUpdateStage : uint8_t {
	ALL = 0,
	RELIABLE = 1,
	NON_RELIABLE = 2,
};

// Every classified pixel is dispatched exactly once: reliable pixels run in
// the early stage, while unreliable and unclassifiable pixels run later. The
// latter retain the native fallback when no deformable support is available.
PATCHMATCH_APD_HOST_DEVICE constexpr bool APDShouldProcess(
	APDReliabilityClass reliability,
	APDUpdateStage stage)
{
	return stage == APDUpdateStage::ALL ||
		(stage == APDUpdateStage::RELIABLE && reliability == APDReliabilityClass::RELIABLE) ||
		(stage == APDUpdateStage::NON_RELIABLE && reliability != APDReliabilityClass::RELIABLE);
}

enum class APDProfileReason : uint8_t {
	UNKNOWN_INVALID_INPUT = 0,
	UNKNOWN_NONFINITE_COST,
	UNRELIABLE_NO_LOCAL_MINIMUM,
	UNRELIABLE_GLOBAL_MINIMUM_OUTSIDE_ETA,
	UNRELIABLE_GLOBAL_MINIMUM_COST_TOO_HIGH,
	UNRELIABLE_SINGLE_MINIMUM_COST_NOT_STRICTLY_BELOW_T2,
	UNRELIABLE_MULTI_MINIMUM_SEPARATION_NOT_ABOVE_T3,
	RELIABLE_SINGLE_MINIMUM,
	RELIABLE_SEPARATED_MINIMA,
};

enum class APDSeparationConvention : uint8_t {
	PAPER = 0,
	RMS_COMPATIBILITY = 1,
};

struct APDProfileSummary {
	APDReliabilityClass reliability = APDReliabilityClass::UNKNOWN;
	APDProfileReason reason = APDProfileReason::UNKNOWN_INVALID_INPUT;
	unsigned finiteCount = 0;
	unsigned localMinimumCount = 0;
	unsigned globalMinimumIndex = APD_PROFILE_SIZE;
	unsigned globalMinimumPlateauStart = APD_PROFILE_SIZE;
	unsigned globalMinimumPlateauEnd = APD_PROFILE_SIZE;
	int globalMinimumOffset = 0;
	float globalMinimumCost = APD_BAD_COST;
	float separation = 0.f;
};

PATCHMATCH_APD_HOST_DEVICE inline float ComputeAPDMinimumSeparation(
	float squaredDifferenceSum,
	unsigned localMinimumCount,
	APDSeparationConvention convention = APDSeparationConvention::PAPER)
{
	if (localMinimumCount < 2u || !(squaredDifferenceSum >= 0.f) ||
		!APDFinite(squaredDifferenceSum))
		return 0.f;
	const float denominator(static_cast<float>(localMinimumCount-1u));
	return convention == APDSeparationConvention::PAPER ?
		::sqrtf(squaredDifferenceSum)/denominator :
		::sqrtf(squaredDifferenceSum/denominator);
}

// Flat runs are treated as one local minimum. Endpoint runs are eligible when
// their only exterior neighbor is higher; an entirely flat profile is not.
// When the global minimum spans or repeats across samples, the representative
// closest to the profile center is used, with the lower index breaking ties.
PATCHMATCH_APD_HOST_DEVICE inline APDProfileSummary SummarizeAPDProfile(
	const float* costs,
	unsigned count,
	unsigned eta,
	APDSeparationConvention convention = APDSeparationConvention::PAPER)
{
	APDProfileSummary summary;
	if (!costs || count != APD_PROFILE_SIZE || eta > APD_PROFILE_RADIUS)
		return summary;
	float globalMinimumCost(FLT_MAX);
	for (unsigned i=0; i<count; ++i) {
		if (!APDFinite(costs[i])) {
			summary.reason = APDProfileReason::UNKNOWN_NONFINITE_COST;
			return summary;
		}
		++summary.finiteCount;
		if (costs[i] < globalMinimumCost)
			globalMinimumCost = costs[i];
	}
	summary.globalMinimumCost = globalMinimumCost;
	unsigned bestDistance(APD_PROFILE_SIZE);
	for (unsigned i=0; i<count; ++i) {
		if (costs[i] != globalMinimumCost)
			continue;
		const unsigned distance(i > APD_PROFILE_RADIUS ? i-APD_PROFILE_RADIUS : APD_PROFILE_RADIUS-i);
		if (distance < bestDistance || (distance == bestDistance && i < summary.globalMinimumIndex)) {
			bestDistance = distance;
			summary.globalMinimumIndex = i;
		}
	}
	unsigned minimumStart(summary.globalMinimumIndex);
	unsigned minimumEnd(summary.globalMinimumIndex);
	while (minimumStart > 0u && costs[minimumStart-1u] == globalMinimumCost)
		--minimumStart;
	while (minimumEnd+1u < count && costs[minimumEnd+1u] == globalMinimumCost)
		++minimumEnd;
	summary.globalMinimumPlateauStart = minimumStart;
	summary.globalMinimumPlateauEnd = minimumEnd;
	summary.globalMinimumOffset = static_cast<int>(summary.globalMinimumIndex)-
		static_cast<int>(APD_PROFILE_RADIUS);

	float squaredDifferenceSum(0.f);
	for (unsigned runStart=0; runStart<count;) {
		unsigned runEnd(runStart);
		while (runEnd+1u < count && costs[runEnd+1u] == costs[runStart])
			++runEnd;
		const bool hasLeft(runStart > 0u);
		const bool hasRight(runEnd+1u < count);
		const bool leftHigher(!hasLeft || costs[runStart-1u] > costs[runStart]);
		const bool rightHigher(!hasRight || costs[runEnd+1u] > costs[runStart]);
		if ((hasLeft || hasRight) && leftHigher && rightHigher) {
			++summary.localMinimumCount;
			const float difference(costs[runStart]-globalMinimumCost);
			squaredDifferenceSum += difference*difference;
		}
		runStart = runEnd+1u;
	}

	if (summary.globalMinimumOffset < -static_cast<int>(eta) ||
		summary.globalMinimumOffset > static_cast<int>(eta))
	{
		summary.reliability = APDReliabilityClass::UNRELIABLE;
		summary.reason = APDProfileReason::UNRELIABLE_GLOBAL_MINIMUM_OUTSIDE_ETA;
		return summary;
	}
	if (summary.globalMinimumCost > APD_PROFILE_MAX_COST) {
		summary.reliability = APDReliabilityClass::UNRELIABLE;
		summary.reason = APDProfileReason::UNRELIABLE_GLOBAL_MINIMUM_COST_TOO_HIGH;
		return summary;
	}
	if (summary.localMinimumCount == 0u) {
		summary.reliability = APDReliabilityClass::UNRELIABLE;
		summary.reason = APDProfileReason::UNRELIABLE_NO_LOCAL_MINIMUM;
		return summary;
	}
	if (summary.localMinimumCount == 1u) {
		if (summary.globalMinimumCost < APD_SINGLE_MINIMUM_MAX_COST) {
			summary.reliability = APDReliabilityClass::RELIABLE;
			summary.reason = APDProfileReason::RELIABLE_SINGLE_MINIMUM;
		} else {
			summary.reliability = APDReliabilityClass::UNRELIABLE;
			summary.reason = APDProfileReason::UNRELIABLE_SINGLE_MINIMUM_COST_NOT_STRICTLY_BELOW_T2;
		}
		return summary;
	}
	summary.separation = ComputeAPDMinimumSeparation(
		squaredDifferenceSum, summary.localMinimumCount, convention);
	if (summary.separation > APD_MINIMUM_SEPARATION) {
		summary.reliability = APDReliabilityClass::RELIABLE;
		summary.reason = APDProfileReason::RELIABLE_SEPARATED_MINIMA;
	} else {
		summary.reliability = APDReliabilityClass::UNRELIABLE;
		summary.reason = APDProfileReason::UNRELIABLE_MULTI_MINIMUM_SEPARATION_NOT_ABOVE_T3;
	}
	return summary;
}

struct APDWeightedCostAccumulator {
	float weightedCostSum = 0.f;
	float weightSum = 0.f;
	unsigned contributingViews = 0u;
	unsigned rejectedViews = 0u;
};

PATCHMATCH_APD_HOST_DEVICE inline bool AccumulateAPDViewCost(
	APDWeightedCostAccumulator& accumulator,
	bool selected,
	float weight,
	float cost,
	float maximumCost = APD_MAX_NCC_COST)
{
	if (!selected)
		return true;
	if (!(weight > 0.f) || !APDFinite(weight) || !(maximumCost > 0.f) ||
		!APDFinite(maximumCost) || !(cost >= 0.f) || cost > maximumCost || !APDFinite(cost))
	{
		++accumulator.rejectedViews;
		return false;
	}
	accumulator.weightedCostSum += weight*cost;
	accumulator.weightSum += weight;
	++accumulator.contributingViews;
	return APDFinite(accumulator.weightedCostSum) && APDFinite(accumulator.weightSum);
}

PATCHMATCH_APD_HOST_DEVICE inline float FinishAPDViewCost(
	const APDWeightedCostAccumulator& accumulator,
	float unavailableCost = APD_BAD_COST)
{
	return accumulator.contributingViews > 0u && accumulator.weightSum > 0.f &&
		APDFinite(accumulator.weightedCostSum) && APDFinite(accumulator.weightSum) ?
		accumulator.weightedCostSum/accumulator.weightSum : unavailableCost;
}

enum class APDPatchKind : uint8_t {
	CENTER = 0,
	ANCHOR = 1,
};

struct APDPatchOffset {
	int x = 0;
	int y = 0;
	bool valid = false;
};

PATCHMATCH_APD_HOST_DEVICE constexpr unsigned APDPatchAxisSampleCount(APDPatchKind kind)
{
	return kind == APDPatchKind::CENTER ? APD_CENTER_PATCH_AXIS_SAMPLES : APD_ANCHOR_PATCH_AXIS_SAMPLES;
}

PATCHMATCH_APD_HOST_DEVICE constexpr unsigned APDPatchSampleCount(APDPatchKind kind)
{
	return kind == APDPatchKind::CENTER ? APD_CENTER_PATCH_SAMPLES : APD_ANCHOR_PATCH_SAMPLES;
}

PATCHMATCH_APD_HOST_DEVICE constexpr int APDPatchIncrement(APDPatchKind kind)
{
	return kind == APDPatchKind::CENTER ? APD_CENTER_PATCH_INCREMENT : APD_ANCHOR_PATCH_INCREMENT;
}

PATCHMATCH_APD_HOST_DEVICE constexpr APDPatchOffset MakeAPDPatchOffset(
	APDPatchKind kind,
	unsigned sampleIndex)
{
	const unsigned axisCount(APDPatchAxisSampleCount(kind));
	return sampleIndex >= axisCount*axisCount ? APDPatchOffset{} : APDPatchOffset{
		-APD_PATCH_RADIUS+static_cast<int>(sampleIndex%axisCount)*APDPatchIncrement(kind),
		-APD_PATCH_RADIUS+static_cast<int>(sampleIndex/axisCount)*APDPatchIncrement(kind),
		true,
	};
}

PATCHMATCH_APD_HOST_DEVICE constexpr bool APDReferencePatchFits(
	int x,
	int y,
	int width,
	int height)
{
	return width > 0 && height > 0 && x >= APD_PATCH_RADIUS && y >= APD_PATCH_RADIUS &&
		x+APD_PATCH_RADIUS < width && y+APD_PATCH_RADIUS < height;
}

struct APDPoint2 {
	float x = 0.f;
	float y = 0.f;
};

struct APDPoint3 {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
};

struct APDPlane {
	float x = 0.f;
	float y = 0.f;
	float z = 0.f;
	float w = 0.f;
};

PATCHMATCH_APD_HOST_DEVICE inline bool APDPlaneDepthAtRay(
	const APDPlane& plane,
	const APDPoint3& rayAtUnitDepth,
	float depthMin,
	float depthMax,
	float& depth)
{
	const float denominator(
		plane.x*rayAtUnitDepth.x+plane.y*rayAtUnitDepth.y+plane.z*rayAtUnitDepth.z);
	if (!(depthMin > 0.f) || !(depthMax >= depthMin) ||
		!APDFinite(denominator) || !(denominator > FLT_EPSILON || denominator < -FLT_EPSILON))
		return false;
	depth = -plane.w/denominator;
	return depth >= depthMin && depth <= depthMax && APDFinite(depth);
}

struct APDSectorDirection {
	float x = 0.f;
	float y = 0.f;
	bool valid = false;
};

PATCHMATCH_APD_HOST_DEVICE inline APDSectorDirection MakeAPDSectorDirection(
	unsigned sector,
	unsigned sectorCount,
	float unitOffset)
{
	APDSectorDirection direction;
	if (sectorCount < 3u || sector >= sectorCount || !(unitOffset >= 0.f) ||
		!(unitOffset < 1.f) || !APDFinite(unitOffset))
		return direction;
	const float angle(2.f*APD_PI*(static_cast<float>(sector)+unitOffset)/
		static_cast<float>(sectorCount));
	direction.x = ::cosf(angle);
	direction.y = ::sinf(angle);
	direction.valid = APDFinite(direction.x) && APDFinite(direction.y);
	return direction;
}

PATCHMATCH_APD_HOST_DEVICE inline unsigned APDSectorForDirection(
	float x,
	float y,
	unsigned sectorCount)
{
	if (sectorCount < 3u || !APDFinite(x) || !APDFinite(y) || !(x*x+y*y > FLT_MIN))
		return sectorCount;
	float angle(::atan2f(y, x));
	if (angle < 0.f)
		angle += 2.f*APD_PI;
	unsigned sector(static_cast<unsigned>(angle*static_cast<float>(sectorCount)/(2.f*APD_PI)));
	return sector < sectorCount ? sector : 0u;
}

PATCHMATCH_APD_HOST_DEVICE inline float APDSquaredDistance(
	const APDPoint2& first,
	const APDPoint2& second)
{
	const float dx(first.x-second.x);
	const float dy(first.y-second.y);
	return dx*dx+dy*dy;
}

PATCHMATCH_APD_HOST_DEVICE inline float APDCross(
	const APDPoint2& first,
	const APDPoint2& second,
	const APDPoint2& origin)
{
	return (first.x-origin.x)*(second.y-origin.y)-
		(first.y-origin.y)*(second.x-origin.x);
}

PATCHMATCH_APD_HOST_DEVICE inline bool APDTriangleContainsPoint(
	const APDPoint2& first,
	const APDPoint2& second,
	const APDPoint2& third,
	const APDPoint2& point)
{
	if (!APDFinite(first.x) || !APDFinite(first.y) ||
		!APDFinite(second.x) || !APDFinite(second.y) ||
		!APDFinite(third.x) || !APDFinite(third.y) ||
		!APDFinite(point.x) || !APDFinite(point.y))
		return false;
	const float triangleArea(APDCross(second, third, first));
	if (!(triangleArea > FLT_EPSILON || triangleArea < -FLT_EPSILON))
		return false;
	const float firstCross(APDCross(first, second, point));
	const float secondCross(APDCross(second, third, point));
	const float thirdCross(APDCross(third, first, point));
	return (firstCross >= 0.f && secondCross >= 0.f && thirdCross >= 0.f) ||
		(firstCross <= 0.f && secondCross <= 0.f && thirdCross <= 0.f);
}

PATCHMATCH_APD_HOST_DEVICE inline bool FitAPDPlane(
	const APDPoint3& first,
	const APDPoint3& second,
	const APDPoint3& third,
	APDPlane& plane)
{
	const float firstX(first.x-third.x);
	const float firstY(first.y-third.y);
	const float firstZ(first.z-third.z);
	const float secondX(second.x-third.x);
	const float secondY(second.y-third.y);
	const float secondZ(second.z-third.z);
	const float normalX(firstY*secondZ-secondY*firstZ);
	const float normalY(secondX*firstZ-firstX*secondZ);
	const float normalZ(firstX*secondY-secondX*firstY);
	const float squaredNorm(normalX*normalX+normalY*normalY+normalZ*normalZ);
	if (!(squaredNorm > FLT_MIN) || !APDFinite(squaredNorm))
		return false;
	const float inverseNorm(1.f/::sqrtf(squaredNorm));
	plane.x = normalX*inverseNorm;
	plane.y = normalY*inverseNorm;
	plane.z = normalZ*inverseNorm;
	plane.w = -(plane.x*first.x+plane.y*first.y+plane.z*first.z);
	return APDFinite(plane.x) && APDFinite(plane.y) && APDFinite(plane.z) && APDFinite(plane.w);
}

PATCHMATCH_APD_HOST_DEVICE inline float APDNormalizedPlaneResidual(
	const APDPlane& plane,
	const APDPoint3& point,
	float depthRange)
{
	if (!(depthRange > 0.f) || !APDFinite(depthRange))
		return FLT_MAX;
	const float residual(plane.x*point.x+plane.y*point.y+plane.z*point.z+plane.w);
	if (!APDFinite(residual))
		return FLT_MAX;
	return (residual < 0.f ? -residual : residual)/depthRange;
}

PATCHMATCH_APD_HOST_DEVICE inline bool APDPlaneInlier(
	const APDPlane& plane,
	const APDPoint3& point,
	float depthRange,
	float normalizedThreshold = APD_RANSAC_NORMALIZED_THRESHOLD_MIN)
{
	return normalizedThreshold > 0.f && APDFinite(normalizedThreshold) &&
		APDNormalizedPlaneResidual(plane, point, depthRange) < normalizedThreshold;
}

PATCHMATCH_APD_HOST_DEVICE inline float APDRansacNormalizedThreshold(
	unsigned logicalIteration,
	unsigned logicalIterationCount)
{
	if (logicalIterationCount <= 1u)
		return APD_RANSAC_NORMALIZED_THRESHOLD_MAX;
	const unsigned clampedIteration(logicalIteration < logicalIterationCount ?
		logicalIteration : logicalIterationCount-1u);
	const float progress(static_cast<float>(clampedIteration)/
		static_cast<float>(logicalIterationCount-1u));
	return APD_RANSAC_NORMALIZED_THRESHOLD_MAX-
		progress*(APD_RANSAC_NORMALIZED_THRESHOLD_MAX-APD_RANSAC_NORMALIZED_THRESHOLD_MIN);
}

PATCHMATCH_APD_HOST_DEVICE constexpr uint32_t APDHash(uint32_t value)
{
	value ^= value >> 16;
	value *= 0x7FEB352Du;
	value ^= value >> 15;
	value *= 0x846CA68Bu;
	value ^= value >> 16;
	return value;
}

PATCHMATCH_APD_HOST_DEVICE inline uint32_t APDStageSeed(const APDStageClock& clock)
{
	if (ValidateAPDStageClock(clock) != APDStageClockStatus::VALID)
		return 0u;
	return APDHash(
		APDHash(clock.stageIndex+1u) ^
		APDHash((clock.levelIndex+1u)*0x9E3779B9u) ^
		APDHash((clock.logicalIteration+1u)*0x85EBCA6Bu) ^
		(clock.geometricConsistency ? 0xC2B2AE35u : 0x27D4EB2Fu));
}

struct APDRansacTriplet {
	unsigned first = 0u;
	unsigned second = 0u;
	unsigned third = 0u;
	bool valid = false;
};

PATCHMATCH_APD_HOST_DEVICE inline APDRansacTriplet MakeAPDRansacTriplet(
	uint32_t pixelSeed,
	uint32_t stageSeed,
	unsigned trial,
	unsigned candidateCount)
{
	APDRansacTriplet triplet;
	if (candidateCount < 3u || candidateCount > APD_SECTOR_COUNT || trial >= APD_RANSAC_TRIALS)
		return triplet;
	uint32_t state(APDHash(pixelSeed ^ APDHash(stageSeed+0x9E3779B9u) ^
		APDHash(trial+0x85EBCA6Bu)));
	triplet.first = state%candidateCount;
	state = APDHash(state+0x9E3779B9u);
	triplet.second = state%(candidateCount-1u);
	if (triplet.second >= triplet.first)
		++triplet.second;
	state = APDHash(state+0x9E3779B9u);
	triplet.third = state%(candidateCount-2u);
	const unsigned lower(triplet.first < triplet.second ? triplet.first : triplet.second);
	const unsigned upper(triplet.first < triplet.second ? triplet.second : triplet.first);
	if (triplet.third >= lower)
		++triplet.third;
	if (triplet.third >= upper)
		++triplet.third;
	if (triplet.second < triplet.first) {
		const unsigned swap(triplet.first);
		triplet.first = triplet.second;
		triplet.second = swap;
	}
	if (triplet.third < triplet.second) {
		const unsigned swap(triplet.second);
		triplet.second = triplet.third;
		triplet.third = swap;
	}
	if (triplet.second < triplet.first) {
		const unsigned swap(triplet.first);
		triplet.first = triplet.second;
		triplet.second = swap;
	}
	triplet.valid = true;
	return triplet;
}

struct APDModelQuality {
	bool valid = false;
	unsigned candidateCount = 0u;
	unsigned inlierCount = 0u;
	unsigned outlierCount = 0u;
	float centerResidual = FLT_MAX;
	float meanInlierResidual = FLT_MAX;
	APDRansacTriplet sample;
};

PATCHMATCH_APD_HOST_DEVICE inline APDModelQuality MakeAPDModelQuality(
	unsigned candidateCount,
	unsigned inlierCount,
	bool centerEnclosed,
	float centerResidual,
	float meanInlierResidual,
	const APDRansacTriplet& sample)
{
	APDModelQuality quality;
	quality.candidateCount = candidateCount;
	quality.inlierCount = inlierCount;
	quality.outlierCount = candidateCount >= inlierCount ? candidateCount-inlierCount : candidateCount;
	quality.centerResidual = centerResidual;
	quality.meanInlierResidual = meanInlierResidual;
	quality.sample = sample;
	quality.valid = sample.valid && centerEnclosed && candidateCount >= APD_MIN_INLIERS &&
		inlierCount >= APD_MIN_INLIERS && inlierCount <= candidateCount &&
		centerResidual >= 0.f && APDFinite(centerResidual) &&
		meanInlierResidual >= 0.f && APDFinite(meanInlierResidual);
	return quality;
}

PATCHMATCH_APD_HOST_DEVICE inline bool PreferAPDModel(
	const APDModelQuality& candidate,
	const APDModelQuality& incumbent)
{
	if (candidate.valid != incumbent.valid)
		return candidate.valid;
	if (!candidate.valid)
		return false;
	if (candidate.outlierCount != incumbent.outlierCount)
		return candidate.outlierCount < incumbent.outlierCount;
	if (candidate.centerResidual != incumbent.centerResidual)
		return candidate.centerResidual < incumbent.centerResidual;
	if (candidate.sample.first != incumbent.sample.first)
		return candidate.sample.first < incumbent.sample.first;
	if (candidate.sample.second != incumbent.sample.second)
		return candidate.sample.second < incumbent.sample.second;
	return candidate.sample.third < incumbent.sample.third;
}

struct APDAnchorRank {
	bool inlier = false;
	float normalizedPlaneResidual = FLT_MAX;
	unsigned candidateIndex = APD_SECTOR_COUNT;
};

enum class APDViewSelectionMode : uint8_t {
	NATIVE = 0,
	ANCHOR_EVIDENCE,
	PREVIOUS_WEIGHTS_FALLBACK,
	SELECTED_MASK_FALLBACK,
	FIRST_VIEW_FALLBACK,
};

// One source-view score built from the immutable reliable-anchor hypotheses
// of a weak pixel. The OpenMVS-native threshold schedule is retained as an
// explicit compatibility behavior; the evidence source is APD's anchors.
struct APDAnchorViewEvidence {
	float prior = 0.f;
	float samplingScore = 0.f;
	unsigned supportCount = 0u;
	unsigned agreeCount = 0u;
	unsigned badCount = 0u;
	bool valid = false;
};

PATCHMATCH_APD_HOST_DEVICE inline float APDViewCostThreshold(unsigned logicalIteration)
{
	const float iteration(static_cast<float>(logicalIteration));
	return APD_VIEW_COST_THRESHOLD*::expf(
		iteration*iteration/(-2.f*APD_VIEW_COST_ITERATION_SIGMA*APD_VIEW_COST_ITERATION_SIGMA));
}

PATCHMATCH_APD_HOST_DEVICE inline APDAnchorViewEvidence ComputeAPDAnchorViewEvidence(
	const float* anchorViewCosts,
	unsigned viewStride,
	const uint32_t* anchorSelectedViews,
	uint32_t validAnchorMask,
	unsigned anchorCount,
	unsigned view,
	float costThreshold)
{
	APDAnchorViewEvidence evidence;
	if (!anchorViewCosts || !anchorSelectedViews || viewStride == 0u || view >= viewStride ||
		anchorCount == 0u || anchorCount > APD_MAX_ANCHORS || !(costThreshold > 0.f) ||
		!APDFinite(costThreshold))
		return evidence;
	float agreementWeightSum(0.f);
	for (unsigned anchor=0; anchor<anchorCount; ++anchor) {
		if ((validAnchorMask & (1u << anchor)) == 0u)
			continue;
		++evidence.supportCount;
		evidence.prior += (anchorSelectedViews[anchor] & (1u << view)) != 0u ?
			APD_VIEW_PRIOR_SELECTED : APD_VIEW_PRIOR_REJECTED;
		const float cost(anchorViewCosts[anchor*viewStride+view]);
		if (!APDFinite(cost) || cost >= APD_BAD_COST) {
			++evidence.badCount;
			continue;
		}
		if (cost < costThreshold) {
			agreementWeightSum += ::expf(
				cost*cost/(-2.f*APD_VIEW_AGREEMENT_SIGMA*APD_VIEW_AGREEMENT_SIGMA));
			++evidence.agreeCount;
		}
	}
	if (evidence.supportCount == 0u || !APDFinite(evidence.prior))
		return evidence;
	if (evidence.agreeCount >= APD_VIEW_MIN_AGREEMENT &&
		evidence.badCount <= APD_VIEW_MAX_BAD)
	{
		evidence.samplingScore = evidence.prior*agreementWeightSum/
			static_cast<float>(evidence.agreeCount);
	} else if (evidence.badCount <= APD_VIEW_MAX_BAD) {
		evidence.samplingScore = evidence.prior*::expf(
			costThreshold*costThreshold/
			(-2.f*APD_VIEW_FALLBACK_SIGMA*APD_VIEW_FALLBACK_SIGMA));
	}
	evidence.valid = evidence.samplingScore > 0.f && APDFinite(evidence.samplingScore);
	return evidence;
}

struct APDAnchorCandidateRank {
	float workingCost = FLT_MAX;
	uint32_t anchorIndex = ~uint32_t(0);
	unsigned slot = APD_MAX_ANCHORS;
	bool valid = false;
};

PATCHMATCH_APD_HOST_DEVICE inline bool PreferAPDAnchorCandidate(
	const APDAnchorCandidateRank& candidate,
	const APDAnchorCandidateRank& incumbent)
{
	if (candidate.valid != incumbent.valid)
		return candidate.valid;
	if (!candidate.valid)
		return false;
	if (candidate.workingCost != incumbent.workingCost)
		return candidate.workingCost < incumbent.workingCost;
	return candidate.slot < incumbent.slot;
}

PATCHMATCH_APD_HOST_DEVICE inline bool PreferAPDAnchor(
	const APDAnchorRank& candidate,
	const APDAnchorRank& incumbent)
{
	if (candidate.inlier != incumbent.inlier)
		return candidate.inlier;
	if (!candidate.inlier)
		return false;
	if (candidate.normalizedPlaneResidual != incumbent.normalizedPlaneResidual)
		return candidate.normalizedPlaneResidual < incumbent.normalizedPlaneResidual;
	return candidate.candidateIndex < incumbent.candidateIndex;
}

struct APDAnchorCostAccumulator {
	float costSum = 0.f;
	unsigned anchorCount = 0u;
	unsigned validSupportCount = 0u;
	unsigned invalidSupportCount = 0u;
};

PATCHMATCH_APD_HOST_DEVICE inline bool AccumulateAPDAnchorCost(
	APDAnchorCostAccumulator& accumulator,
	bool supportValid,
	float sampledCost,
	float badCost = APD_BAD_COST)
{
	if (!(badCost > 0.f) || !APDFinite(badCost))
		return false;
	if (supportValid && (!(sampledCost >= 0.f) || sampledCost > APD_MAX_NCC_COST ||
		!APDFinite(sampledCost)))
		return false;
	accumulator.costSum += supportValid ? sampledCost : badCost;
	++accumulator.anchorCount;
	if (supportValid)
		++accumulator.validSupportCount;
	else
		++accumulator.invalidSupportCount;
	return APDFinite(accumulator.costSum);
}

enum class APDScoreFallback : uint8_t {
	NONE = 0,
	MECHANISM_DISABLED,
	PIXEL_NOT_UNRELIABLE,
	INVALID_CENTER_SUPPORT,
	INVALID_ANCHOR_CONSENSUS,
	NO_ANCHORS,
	NONFINITE_INPUT,
};

struct APDScoreDecision {
	float workingCost = APD_BAD_COST;
	float anchorMeanCost = 0.f;
	unsigned anchorCount = 0u;
	bool usedDeformableCost = false;
	APDScoreFallback fallback = APDScoreFallback::NONFINITE_INPUT;
};

struct APDPersistentScoreDecision {
	float cost = APD_BAD_COST;
	bool valid = false;
	bool usedWinnerRescore = false;
};

struct APDFinalRefinementDecision {
	float depth = 0.f;
	float cost = APD_BAD_COST;
	float improvement = 0.f;
	int offset = 0;
	bool valid = false;
	bool accepted = false;
};

PATCHMATCH_APD_HOST_DEVICE inline APDFinalRefinementDecision ResolveAPDFinalRefinement(
	float incumbentDepth,
	float incumbentCost,
	float candidateDepth,
	float candidateCost,
	int candidateOffset,
	float minimumImprovement = APD_FINAL_REFINEMENT_MIN_IMPROVEMENT)
{
	APDFinalRefinementDecision decision;
	decision.depth = candidateDepth;
	decision.cost = candidateCost;
	decision.offset = candidateOffset;
	decision.valid = incumbentDepth > 0.f && candidateDepth > 0.f &&
		incumbentCost >= 0.f && candidateCost >= 0.f &&
		APDFinite(incumbentDepth) && APDFinite(candidateDepth) &&
		APDFinite(incumbentCost) && APDFinite(candidateCost) &&
		minimumImprovement >= 0.f && APDFinite(minimumImprovement);
	if (!decision.valid)
		return decision;
	decision.improvement = incumbentCost-candidateCost;
	decision.accepted = decision.improvement > minimumImprovement;
	return decision;
}

// Deformable scores rank candidates only inside the active update. Any winner
// persisted to the production depth map carries a fresh conventional score.
PATCHMATCH_APD_HOST_DEVICE inline APDPersistentScoreDecision ResolveAPDPersistentScore(
	bool candidateAccepted,
	float incumbentNativeCost,
	float winnerNativeRescore)
{
	APDPersistentScoreDecision decision;
	const float selectedCost(candidateAccepted ? winnerNativeRescore : incumbentNativeCost);
	decision.cost = selectedCost;
	decision.valid = selectedCost >= 0.f && selectedCost <= APD_MAX_NCC_COST && APDFinite(selectedCost);
	decision.usedWinnerRescore = candidateAccepted;
	return decision;
}

PATCHMATCH_APD_HOST_DEVICE inline APDScoreDecision EvaluateAPDWorkingScore(
	bool mechanismEnabled,
	APDReliabilityClass reliability,
	bool centerSupportValid,
	bool anchorConsensusValid,
	float centerCost,
	const APDAnchorCostAccumulator& anchors,
	float badCost = APD_BAD_COST)
{
	APDScoreDecision decision;
	if (!(badCost > 0.f) || !APDFinite(badCost) || !(centerCost >= 0.f) ||
		centerCost > APD_MAX_NCC_COST || !APDFinite(centerCost))
		return decision;
	decision.workingCost = centerSupportValid ? centerCost : badCost;
	decision.anchorCount = anchors.anchorCount;
	if (!centerSupportValid) {
		decision.fallback = APDScoreFallback::INVALID_CENTER_SUPPORT;
		return decision;
	}
	if (!mechanismEnabled) {
		decision.fallback = APDScoreFallback::MECHANISM_DISABLED;
		return decision;
	}
	if (reliability != APDReliabilityClass::UNRELIABLE) {
		decision.fallback = APDScoreFallback::PIXEL_NOT_UNRELIABLE;
		return decision;
	}
	if (!anchorConsensusValid) {
		decision.fallback = APDScoreFallback::INVALID_ANCHOR_CONSENSUS;
		return decision;
	}
	if (anchors.anchorCount == 0u) {
		decision.fallback = APDScoreFallback::NO_ANCHORS;
		return decision;
	}
	if (!(anchors.costSum >= 0.f) || !APDFinite(anchors.costSum))
		return decision;
	decision.anchorMeanCost = anchors.costSum/static_cast<float>(anchors.anchorCount);
	decision.workingCost = APD_CENTER_WEIGHT*centerCost+
		APD_ANCHOR_WEIGHT*decision.anchorMeanCost;
	decision.usedDeformableCost = true;
	decision.fallback = APDScoreFallback::NONE;
	return decision;
}

} // namespace CUDA
} // namespace MVS

#undef PATCHMATCH_APD_HOST_DEVICE
