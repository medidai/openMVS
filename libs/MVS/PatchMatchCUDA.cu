/*
* PatchMatchCUDA.cu
*
* Copyright (c) 2014-2021 SEACAVE
*
* Author(s):
*
*	  cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*	  You are required to preserve legal notices and author attributions in
*	  that material or in the Appropriate Legal Notices displayed by works
*	  containing it.
*/

#include "PatchMatchCUDA.inl"
#include "PatchMatchAPDCUDA.h"
#include "PatchMatchDVPCUDA.h"
#ifdef _USE_DMAP_INSTRUMENTATION
#include <vector>
#endif

// static max supported views
#define MAX_VIEWS 32

// samples used to perform views selection
#define NUM_SAMPLES 32

// unified "bad cost" sentinel: returned by ScorePlane when the patch
// cannot be evaluated against a view (out-of-frame, texture-less, or
// degenerate variance), used as the view-pruning threshold in the
// multi-hypothesis joint view selection, and as the all-views-rejected
// fallback in AggregateMultiViewScores. Keeping the three meanings on
// a single name makes the alignment explicit and tunable from one place.
#define fBadCost 1.2f

// patch window radius
#define nSizeHalfWindow 4

// patch stepping
#define nSizeStep 2

static_assert(MVS::CUDA::APD_BAD_COST == fBadCost, "APD and production bad-cost sentinels must match");
static_assert(MVS::CUDA::APD_SECTOR_COUNT == MAX_VIEWS, "APD candidate storage assumes 32 sectors");
#ifdef _USE_DMAP_INSTRUMENTATION
static_assert(MVS::CUDA::APD_PROFILE_SIZE == MVS::CUDA::PM_APD_INSTRUMENT_PROFILE_SAMPLES,
	"APD trace profile schema must match the paper profile");
static_assert(MVS::CUDA::APD_SECTOR_COUNT == MVS::CUDA::PM_APD_INSTRUMENT_SECTORS,
	"APD trace sector schema must match the paper sectors");
static_assert(MVS::CUDA::APD_MAX_ANCHORS == MVS::CUDA::PM_APD_INSTRUMENT_ANCHORS,
	"APD trace anchor schema must match the paper anchor cap");
#endif

// Launch-bounds tuning. Default uses 256 threads/block with 2 resident
// blocks/SM, letting the warp scheduler interleave across blocks while
// one is stalled on tex2D latency (~1.8% per-view kernel time vs the
// historical 512/1 config). Set PATCHMATCHCUDA_LB_256_2=0 to fall back.
#ifndef PATCHMATCHCUDA_LB_256_2
#define PATCHMATCHCUDA_LB_256_2 1
#endif

#if PATCHMATCHCUDA_LB_256_2
#define PATCHMATCHCUDA_BLOCK_H_DIV 4   // BLOCK_H = BLOCK_W / 4 = 8
#define PATCHMATCHCUDA_LAUNCH_BOUNDS __launch_bounds__(256, 2)
// The opt-in visible-normal path has a substantially larger live state than
// native APD. Keep the default specialization at two blocks/SM, but allow the
// active specialization one block/SM instead of weakening every APD launch.
#define PATCHMATCHCUDA_APD_LAUNCH_BOUNDS(VISIBLE_NORMAL) \
	__launch_bounds__(256, (VISIBLE_NORMAL) ? 1 : 2)
#ifdef _USE_DMAP_INSTRUMENTATION
#define PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS __launch_bounds__(256, 1)
#endif
#else
#define PATCHMATCHCUDA_BLOCK_H_DIV 2   // BLOCK_H = BLOCK_W / 2 = 16
#define PATCHMATCHCUDA_LAUNCH_BOUNDS __launch_bounds__(512, 1)
#define PATCHMATCHCUDA_APD_LAUNCH_BOUNDS(VISIBLE_NORMAL) __launch_bounds__(512, 1)
#ifdef _USE_DMAP_INSTRUMENTATION
#define PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS __launch_bounds__(512, 1)
#endif
#endif


namespace MVS {

namespace CUDA {

#define ImagePixels cudaTextureObject_t
#define RandState curandState

// nvcc rejects `__constant__ Camera[...]` with newer toolchains because the
// Eigen-backed Camera type is treated as needing dynamic initialization. Keep
// constant memory as aligned byte storage and reinterpret it at use sites.
struct alignas(Camera) CameraConstStorage {
	unsigned char bytes[sizeof(Camera)];
};
static_assert(sizeof(CameraConstStorage) == sizeof(Camera), "Camera constant storage must preserve Camera size");
static_assert(alignof(CameraConstStorage) == alignof(Camera), "Camera constant storage must preserve Camera alignment");

// Cameras and runtime params live in __constant__ memory: warp-broadcast
// reads through the constant cache replace per-thread parameter-stack /
// global-memory traffic. Updated via UploadCameras() / UploadParams()
// before each pyramid-level kernel launch.
//
// IMPORTANT: these symbols are module-global per device, so PatchMatchCUDA
// is single-instance / single-in-flight per device. The current densify
// pipeline guarantees this (one image at a time per device) and the C++
// side enforces it at runtime via an atomic in-flight counter (see
// PatchMatchCUDA::EstimateDepthMap). If multi-stream / multi-instance
// parallel use is ever added, switch to per-instance device buffers
// passed explicitly to kernels.
__constant__ CameraConstStorage g_cameraStorage[MAX_VIEWS + 1];
#define g_cameras reinterpret_cast<const Camera*>(g_cameraStorage)
__constant__ PatchMatch::Params g_params;

// set/check a bit
__device__ constexpr void SetBit(unsigned& input, unsigned i) {
	input |= (1u << i);
}
__device__ constexpr int IsBitSet(unsigned input, unsigned i) {
	return (input >> i) & 1u;
}

// Read-only-cache loaders for planes[]. Safe within a checkerboard pass
// because every offset in `dirs` and `neighborPositions` has odd Manhattan
// parity from the current pixel, so the cells we read are not written in
// this launch -- the __ldg() read-only contract holds and L1 bandwidth
// is freed for the texture work.
__device__ __forceinline__ Point4 LoadPlaneLDG(const Point4* p) {
	const float* f = p->data();
	Point4 r;
	r.x() = __ldg(f + 0);
	r.y() = __ldg(f + 1);
	r.z() = __ldg(f + 2);
	r.w() = __ldg(f + 3);
	return r;
}
__device__ __forceinline__ float LoadPlaneWLDG(const Point4* p) {
	return __ldg(p->data() + 3);
}

// sort the given values array using bubble sort algorithm
__device__ inline void Sort(const float* values, float* sortedValues, int n) {
	for (int i = 0; i < n; ++i)
		sortedValues[i] = values[i];
	do {
		int newn = 0;
		for (int i = 1; i < n; ++i) {
			if (sortedValues[i-1] > sortedValues[i]) {
				Swap(sortedValues[i-1], sortedValues[i]);
				newn = i;
			}
		}
		n = newn;
	} while(n);
}

// find the index of the minimum value in the given values array
__device__ inline int FindMinIndex(const float* values, const int n) {
	float minValue = values[0];
	int minValueIdx = 0;
	for (int i = 1; i < n; ++i) {
		if (minValue > values[i]) {
			minValue = values[i];
			minValueIdx = i;
		}
	}
	return minValueIdx;
}

// convert Probability Density Function (PDF) to Cumulative Distribution Function (CDF)
__device__ inline void PDF2CDF(float* probs, const int numProbs) {
	float probSum = 0.f;
	for (int i = 0; i < numProbs; ++i)
		probSum += probs[i];
	const float invProbSum = 1.f / probSum;
	float sumProb = 0.f;
	for (int i = 0; i < numProbs-1; ++i) {
		sumProb += probs[i] * invProbSum;
		probs[i] = sumProb;
	}
	probs[numProbs-1] = 1.f;
}
#ifdef _USE_DMAP_INSTRUMENTATION
__device__ __forceinline__ int CountSelectedViews(const unsigned selectedViews)
{
	return __popc(selectedViews);
}

__device__ __forceinline__ int UpdateMagnitudeBinDepth(const float value)
{
	if (!isfinite(value))
		return PM_INSTRUMENT_NUM_UPDATE_BINS - 1;
	if (value < 0.001f)
		return 0;
	if (value < 0.005f)
		return 1;
	if (value < 0.01f)
		return 2;
	if (value < 0.02f)
		return 3;
	if (value < 0.05f)
		return 4;
	if (value < 0.1f)
		return 5;
	if (value < 0.25f)
		return 6;
	return 7;
}

__device__ __forceinline__ int UpdateMagnitudeBinNormal(const float degrees)
{
	if (!isfinite(degrees))
		return PM_INSTRUMENT_NUM_UPDATE_BINS - 1;
	if (degrees < 1.f)
		return 0;
	if (degrees < 2.f)
		return 1;
	if (degrees < 5.f)
		return 2;
	if (degrees < 10.f)
		return 3;
	if (degrees < 20.f)
		return 4;
	if (degrees < 45.f)
		return 5;
	if (degrees < 90.f)
		return 6;
	return 7;
}

__device__ __forceinline__ float ComputeNormalAngleDegrees(const Point3& before, const Point3& after)
{
	const float beforeNorm = before.norm();
	const float afterNorm = after.norm();
	if (beforeNorm <= FLT_EPSILON || afterNorm <= FLT_EPSILON)
		return 0.f;
	const float dot = max(-1.f, min(1.f, before.dot(after) / (beforeNorm * afterNorm)));
	return acosf(dot) * (180.f / (float)M_PI);
}

__device__ __forceinline__ float ComputeViewEntropy(const unsigned* viewWeights, const int numViews, const float denominator)
{
	if (denominator <= 0.f || numViews <= 1)
		return 0.f;
	float entropy = 0.f;
	for (int imgId = 0; imgId < numViews && imgId < PM_INSTRUMENT_MAX_VIEWS; ++imgId) {
		if (!viewWeights[imgId])
			continue;
		const float p = (float)viewWeights[imgId] / denominator;
		entropy -= p * log2f(p);
	}
	return entropy / log2f((float)numViews);
}

__device__ __forceinline__ void TrackCandidateCost(const float cost, float& best, float& second, int& count)
{
	if (!isfinite(cost) || cost >= fBadCost)
		return;
	++count;
	if (cost < best) {
		second = best;
		best = cost;
	} else if (cost < second) {
		second = cost;
	}
}

__device__ __forceinline__ bool IsFiniteCandidateCost(const float cost)
{
	return isfinite(cost) && cost < fBadCost;
}

__device__ __forceinline__ void TrackExactCandidate(
	const int slot,
	const float cost,
	uint32_t& testedMask,
	uint32_t& finiteMask,
	uint8_t& testedCount,
	uint8_t& finiteCount,
	float& bestCost,
	uint8_t& bestSlot,
	float& runnerUpCost,
	uint8_t& runnerUpSlot)
{
	if (slot < 0 || slot >= PM_INSTRUMENT_EXACT_NUM_CANDIDATES)
		return;
	const uint32_t bit(1u << slot);
	testedMask |= bit;
	++testedCount;
	if (!IsFiniteCandidateCost(cost))
		return;
	finiteMask |= bit;
	++finiteCount;
	if (cost < bestCost) {
		runnerUpCost = bestCost;
		runnerUpSlot = bestSlot;
		bestCost = cost;
		bestSlot = (uint8_t)slot;
	} else if (cost < runnerUpCost) {
		runnerUpCost = cost;
		runnerUpSlot = (uint8_t)slot;
	}
}

__device__ __forceinline__ uint32_t PackExactViewMetadata(
	const unsigned weight,
	const unsigned rank,
	const unsigned agreeCount,
	const unsigned badCount,
	const int decision,
	const bool selected,
	const bool contributionBasis,
	const bool finite,
	const bool probabilityAvailable)
{
	uint32_t metadata =
		((min(weight, 63u) & PM_EXACT_VIEW_WEIGHT_MASK) << PM_EXACT_VIEW_WEIGHT_SHIFT) |
		((min(rank, 63u) & PM_EXACT_VIEW_RANK_MASK) << PM_EXACT_VIEW_RANK_SHIFT) |
		((min(agreeCount, 15u) & PM_EXACT_VIEW_AGREE_MASK) << PM_EXACT_VIEW_AGREE_SHIFT) |
		((min(badCount, 15u) & PM_EXACT_VIEW_BAD_MASK) << PM_EXACT_VIEW_BAD_SHIFT) |
		(((uint32_t)decision & PM_EXACT_VIEW_DECISION_MASK) << PM_EXACT_VIEW_DECISION_SHIFT);
	if (selected)
		metadata |= PM_EXACT_VIEW_SELECTED_BIT;
	if (contributionBasis)
		metadata |= PM_EXACT_VIEW_CONTRIBUTION_BIT;
	if (finite)
		metadata |= PM_EXACT_VIEW_FINITE_BIT;
	if (probabilityAvailable)
		metadata |= PM_EXACT_VIEW_PROBABILITY_BIT;
	return metadata;
}

__device__ __forceinline__ void InstrumentCandidate(
	PatchMatchInstrumentCounters* counters,
	const PatchMatchInstrumentKernelParams& instr,
	const int candidateType,
	const float cost,
	const bool accepted)
{
	if (!instr.enabled || !counters || instr.passIndex < 0 ||
		candidateType < 0 || candidateType >= PM_INSTRUMENT_NUM_CANDIDATE_TYPES)
		return;
	PatchMatchInstrumentCounters& counter = counters[instr.passIndex];
	atomicAdd(&counter.candidateTested[candidateType], 1u);
	if (isfinite(cost) && cost < fBadCost)
		atomicAdd(&counter.candidateFinite[candidateType], 1u);
	if (accepted)
		atomicAdd(&counter.candidateAccepted[candidateType], 1u);
}

__device__ __noinline__ void InstrumentPixel(
	PatchMatchInstrumentCounters* counters,
	PatchMatchInstrumentTraceRecord* traceRecords,
	const int32_t* traceMap,
	uint8_t* updateSources,
	const PatchMatchInstrumentKernelParams& instr,
	const int pixelIndex,
	const Point2i& p,
	const float depthBefore,
	const float depthAfter,
	const float costBefore,
	const float costAfter,
	const PatchMatchInstrumentCostComponents& costComponents,
	const float refVariance,
	const float lowDepth,
	const unsigned selectedViewsBefore,
	const unsigned selectedViews,
	const int source,
	const float depthAbsChange,
	const float depthRelChange,
	const float normalAngleChange,
	const float viewEntropy,
	const float confidenceGap,
	const float* neighborCosts,
	const unsigned* viewWeights)
{
	const float costImprovement = costBefore > costAfter ? costBefore - costAfter : 0.f;
	const float depthDelta = (depthBefore > 0.f && depthAfter > 0.f) ? depthAfter - depthBefore : 0.f;
	const float depthRelDelta = depthBefore > 0.f ? depthDelta / max(depthBefore, FLT_EPSILON) : 0.f;
	const unsigned addedViews = selectedViews & ~selectedViewsBefore;
	const unsigned removedViews = selectedViewsBefore & ~selectedViews;
	const int viewChurnCount = CountSelectedViews(addedViews) + CountSelectedViews(removedViews);
	if (!instr.proxyOnly && instr.enabled && counters && instr.passIndex >= 0) {
		PatchMatchInstrumentCounters& counter = counters[instr.passIndex];
		atomicAdd(&counter.processed, 1u);
		if (depthAfter > 0.f)
			atomicAdd(&counter.validDepth, 1u);
		else
			atomicAdd(&counter.invalidDepth, 1u);
		if (costAfter >= fBadCost)
			atomicAdd(&counter.badCost, 1u);
		if (lowDepth > 0.f)
			atomicAdd(&counter.lowResPrior, 1u);
		if (refVariance < 0.0025f)
			atomicAdd(&counter.lowTexture, 1u);
		if (source == PM_SOURCE_PROPAGATE || source == PM_SOURCE_APD_ANCHOR_PROPAGATE)
			atomicAdd(&counter.propagationWins, 1u);
		if ((source >= PM_SOURCE_REFINE_DEPTH && source <= PM_SOURCE_REFINE_SURFACE_NORMAL) ||
			source == PM_SOURCE_APD_FITTED_PLANE || source == PM_SOURCE_APD_FINAL_REFINEMENT)
			atomicAdd(&counter.refinementWins, 1u);
		if (source != PM_SOURCE_NONE)
			atomicAdd(&counter.accepted, 1u);
		const int selectedViewCount = min(CountSelectedViews(selectedViews), PM_INSTRUMENT_MAX_VIEWS);
		atomicAdd(&counter.selectedViewBins[selectedViewCount], 1u);
		if (source >= 0 && source < PM_INSTRUMENT_NUM_SOURCES)
			atomicAdd(&counter.updateSource[source], 1u);
		atomicAdd(&counter.costBeforeSum, costBefore);
		atomicAdd(&counter.costSum, costAfter);
		atomicAdd(&counter.costSqSum, costAfter * costAfter);
		atomicAdd(&counter.costImprovementSum, costImprovement);
		atomicAdd(&counter.viewEntropySum, viewEntropy);
		if (instr.passIndex > 0) {
			if (addedViews || removedViews) {
				atomicAdd(&counter.viewChurn, 1u);
				atomicAdd(&counter.viewAddedSum, (uint32_t)CountSelectedViews(addedViews));
				atomicAdd(&counter.viewRemovedSum, (uint32_t)CountSelectedViews(removedViews));
			}
			if (source != PM_SOURCE_NONE) {
				atomicAdd(&counter.updateMagnitudeSamples, 1u);
				atomicAdd(&counter.depthAbsChangeSum, depthAbsChange);
				atomicAdd(&counter.depthRelChangeSum, depthRelChange);
				atomicAdd(&counter.normalAngleSum, normalAngleChange);
				atomicAdd(&counter.depthRelChangeBins[UpdateMagnitudeBinDepth(depthRelChange)], 1u);
				atomicAdd(&counter.normalAngleBins[UpdateMagnitudeBinNormal(normalAngleChange)], 1u);
			}
		}
		if (costComponents.sampleCount > 0) {
			atomicAdd(&counter.componentSamples, 1u);
			atomicAdd(&counter.depthPriorSamples, costComponents.depthPriorSamples);
			atomicAdd(&counter.photometricCostSum, costComponents.photometricCost);
			atomicAdd(&counter.photoPriorCostSum, costComponents.photoPriorCost);
			atomicAdd(&counter.depthPriorCostSum, costComponents.depthPriorCost);
			atomicAdd(&counter.depthPriorWeightSum, costComponents.depthPriorWeight);
			atomicAdd(&counter.geometricCostSum, costComponents.geometricCost);
			for (int i = 0; i < PM_INSTRUMENT_NUM_BAD_REASONS; ++i)
				if (costComponents.badReason[i])
					atomicAdd(&counter.badReason[i], costComponents.badReason[i]);
			for (int i = 0; i < PM_INSTRUMENT_MAP_VIEWS; ++i) {
				if (costComponents.viewWeightSum[i] <= 0.f)
					continue;
				atomicAdd(&counter.viewWeightSum[i], costComponents.viewWeightSum[i]);
				atomicAdd(&counter.viewBadCost[i], costComponents.viewBadCost[i]);
				atomicAdd(&counter.viewCostWeightedSum[i], costComponents.viewCostWeightedSum[i]);
				atomicAdd(&counter.viewPhotometricCostWeightedSum[i], costComponents.viewPhotometricCostWeightedSum[i]);
				atomicAdd(&counter.viewGeometricCostWeightedSum[i], costComponents.viewGeometricCostWeightedSum[i]);
			}
		}
	}
	if (!instr.proxyOnly && updateSources && source != PM_SOURCE_NONE)
		updateSources[pixelIndex] = (uint8_t)source;
	if (!instr.proxyOnly && instr.passUpdateSources && instr.passIndex >= 0 && instr.area > 0)
		instr.passUpdateSources[(size_t)instr.passIndex * instr.area + pixelIndex] = (uint8_t)source;
	if (!instr.proxyOnly && instr.improvementMaps && instr.passIndex >= 0 && instr.area > 0)
		instr.improvementMaps[(size_t)instr.passIndex * instr.area + pixelIndex] = costImprovement;
	if (!instr.proxyOnly && instr.passDepthDeltas && instr.passIndex >= 0 && instr.area > 0)
		instr.passDepthDeltas[(size_t)instr.passIndex * instr.area + pixelIndex] = depthDelta;
	if (!instr.proxyOnly && instr.passDepthRelDeltas && instr.passIndex >= 0 && instr.area > 0)
		instr.passDepthRelDeltas[(size_t)instr.passIndex * instr.area + pixelIndex] = depthRelDelta;
	if (!instr.proxyOnly && instr.passNormalAngleDeltas && instr.passIndex >= 0 && instr.area > 0)
		instr.passNormalAngleDeltas[(size_t)instr.passIndex * instr.area + pixelIndex] = normalAngleChange;
	if (!instr.proxyOnly && instr.passViewChurn && instr.passIndex >= 0 && instr.area > 0)
		instr.passViewChurn[(size_t)instr.passIndex * instr.area + pixelIndex] = (uint8_t)min(viewChurnCount, 255);
	if (instr.logicalStateIndex >= 0 && instr.area > 0) {
		const size_t logicalIndex((size_t)instr.logicalStateIndex * instr.area + pixelIndex);
		if (instr.logicalStoredCosts)
			instr.logicalStoredCosts[logicalIndex] = costAfter;
		if (instr.logicalScorePrimary)
			instr.logicalScorePrimary[logicalIndex] = make_float4(
			costComponents.photometricCost,
			costComponents.photoPriorCost,
			costComponents.geometricCost,
			costComponents.totalCost);
		if (instr.logicalScoreSecondary)
			instr.logicalScoreSecondary[logicalIndex] = make_float4(
			costComponents.depthPriorCost,
			costComponents.depthPriorWeight,
			confidenceGap,
			refVariance);
	}
	if (!instr.proxyOnly && instr.exactLogicalStateIndex >= 0 &&
		instr.exactLogicalStateIndex < instr.numLogicalStates && instr.area > 0) {
		const size_t logicalIndex((size_t)instr.exactLogicalStateIndex * instr.area + pixelIndex);
		if (instr.exactLogicalScorePrimary)
			instr.exactLogicalScorePrimary[logicalIndex] = make_float4(
				costComponents.photometricCost,
				costComponents.photoPriorCost,
				costComponents.geometricCost,
				costComponents.totalCost);
		if (instr.exactLogicalScoreSecondary)
			instr.exactLogicalScoreSecondary[logicalIndex] = make_float4(
				costComponents.depthPriorCost,
				costComponents.depthPriorWeight,
				confidenceGap,
				refVariance);
	}
	if (instr.finalViewEntropy)
		instr.finalViewEntropy[pixelIndex] = viewEntropy;
	if (instr.finalLowDepth)
		instr.finalLowDepth[pixelIndex] = lowDepth;
	if (instr.finalSelectedViews)
		instr.finalSelectedViews[pixelIndex] = selectedViews;
	if (!instr.proxyOnly && instr.acceptedUpdateCount && source != PM_SOURCE_NONE) {
		const uint8_t previous = instr.acceptedUpdateCount[pixelIndex];
		instr.acceptedUpdateCount[pixelIndex] = previous == 255 ? 255 : (uint8_t)(previous + 1);
	}
	if (instr.finalViewWeights || instr.finalViewCosts || instr.finalViewPhotometricCosts || instr.finalViewGeometricCosts) {
		float weights[PM_INSTRUMENT_MAP_VIEWS] = {};
		float costs[PM_INSTRUMENT_MAP_VIEWS] = {};
		float photoCosts[PM_INSTRUMENT_MAP_VIEWS] = {};
		float geometricCosts[PM_INSTRUMENT_MAP_VIEWS] = {};
		for (int i = 0; i < PM_INSTRUMENT_MAP_VIEWS; ++i) {
			weights[i] = viewWeights ? (float)viewWeights[i] : 0.f;
			const float componentWeight = costComponents.viewWeightSum[i];
			if (componentWeight > 0.f) {
				costs[i] = costComponents.viewCostWeightedSum[i] / componentWeight;
				photoCosts[i] = costComponents.viewPhotometricCostWeightedSum[i] / componentWeight;
				geometricCosts[i] = costComponents.viewGeometricCostWeightedSum[i] / componentWeight;
			}
		}
		if (instr.finalViewWeights)
			instr.finalViewWeights[pixelIndex] = make_float4(weights[0], weights[1], weights[2], weights[3]);
		if (instr.finalViewCosts)
			instr.finalViewCosts[pixelIndex] = make_float4(costs[0], costs[1], costs[2], costs[3]);
		if (instr.finalViewPhotometricCosts)
			instr.finalViewPhotometricCosts[pixelIndex] = make_float4(photoCosts[0], photoCosts[1], photoCosts[2], photoCosts[3]);
		if (instr.finalViewGeometricCosts)
			instr.finalViewGeometricCosts[pixelIndex] = make_float4(geometricCosts[0], geometricCosts[1], geometricCosts[2], geometricCosts[3]);
	}
	if (instr.proxyOnly || !instr.sampled || !traceRecords || !traceMap || instr.numTracePixels <= 0 || instr.numPasses <= 0)
		return;
	const int traceIndex = traceMap[pixelIndex];
	if (traceIndex < 0 || traceIndex >= instr.numTracePixels)
		return;
	PatchMatchInstrumentTraceRecord& record = traceRecords[traceIndex * instr.numPasses + instr.passIndex];
	record.valid = 1;
	record.imageID = instr.imageID;
	record.scaleNumber = instr.scaleNumber;
	record.passIndex = instr.passIndex;
	record.phase = instr.phase;
	record.iteration = instr.iteration;
	record.x = p.x();
	record.y = p.y();
	record.source = source;
	record.selectedViewCount = CountSelectedViews(selectedViews);
	record.selectedViews = selectedViews;
	record.depthBefore = depthBefore;
	record.depthAfter = depthAfter;
	record.costBefore = costBefore;
	record.costAfter = costAfter;
	record.costImprovement = costImprovement;
	record.depthAbsChange = depthAbsChange;
	record.depthRelChange = depthRelChange;
	record.normalAngleChange = normalAngleChange;
	record.viewEntropy = viewEntropy;
	record.photometricCostAfter = costComponents.photometricCost;
	record.photoPriorCostAfter = costComponents.photoPriorCost;
	record.depthPriorCostAfter = costComponents.depthPriorCost;
	record.depthPriorWeightAfter = costComponents.depthPriorWeight;
	record.geometricCostAfter = costComponents.geometricCost;
	record.refVariance = refVariance;
	record.lowDepth = lowDepth;
	record.selectedViewsBefore = selectedViewsBefore;
	for (int i = 0; i < PM_INSTRUMENT_NUM_NEIGHBORS; ++i)
		record.neighborCosts[i] = neighborCosts ? neighborCosts[i] : -1.f;
	for (int i = 0; i < PM_INSTRUMENT_NUM_BAD_REASONS; ++i)
		record.badReason[i] = costComponents.badReason[i];
	for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i) {
		record.viewWeights[i] = viewWeights ? viewWeights[i] : 0u;
		if (i < PM_INSTRUMENT_MAP_VIEWS) {
			record.viewCosts[i] = costComponents.viewWeightSum[i] > 0.f ? costComponents.viewCostWeightedSum[i] / costComponents.viewWeightSum[i] : 0.f;
			record.viewPhotometricCosts[i] = costComponents.viewWeightSum[i] > 0.f ? costComponents.viewPhotometricCostWeightedSum[i] / costComponents.viewWeightSum[i] : 0.f;
			record.viewGeometricCosts[i] = costComponents.viewWeightSum[i] > 0.f ? costComponents.viewGeometricCostWeightedSum[i] / costComponents.viewWeightSum[i] : 0.f;
		}
	}
}
#endif
/*----------------------------------------------------------------*/


// generate a random unit vector (Marsaglia's method on the unit sphere);
// algebraically unit-length: |n|^2 = 4q1^2(1-s) + 4q2^2(1-s) + (1-2s)^2 = 1
__device__ inline Point3 GenerateRandomUnitVector(RandState* randState)
{
	float q1, q2, s;
	do {
		q1 = 2.f * curand_uniform(randState) - 1.f;
		q2 = 2.f * curand_uniform(randState) - 1.f;
		s = q1 * q1 + q2 * q2;
	} while (s >= 1.f);
	const float sq = sqrtf(1.f - s);
	return Point3(
		2.f * q1 * sq,
		2.f * q2 * sq,
		1.f - 2.f * s);
}

// generate a random normal in the camera-facing half-space
__device__ inline Point3 GenerateRandomNormal(const CUDA::Camera& camera, const Point2i& p, RandState* randState)
{
	const Point3 normal = GenerateRandomUnitVector(randState);
	const Point3 viewDirection = camera.model.ViewDirection(p);
	return normal.dot(viewDirection) > 0.f ? Point3(-normal) : normal;
}

// randomly perturb a normal (algorithmically unit-preserving);
// Rodrigues rotation around a Marsaglia-unit axis by a random small angle
__device__ inline Point3 GeneratePerturbedNormal(const CUDA::Camera& camera, const Point2i& p, const Point3& normal, RandState* randState, const float perturbation)
{
	// random angle in [-perturbation/2, +perturbation/2]
	const float theta = (curand_uniform(randState) - 0.5f) * perturbation;
	float sinT, cosT;
	__sincosf(theta, &sinT, &cosT);

	// rodrigues' rotation formula
	const Point3 axis = GenerateRandomUnitVector(randState);
	const float aDotN = axis.dot(normal);
	const Point3 axCrossN = axis.cross(normal);
	const Point3 normalPerturbed = normal * cosT + axCrossN * sinT + axis * (aDotN * (1.f - cosT));

	// keep the perturbed normal in the camera-facing half-space
	const Point3 viewDirection = camera.model.ViewDirection(p);
	return normalPerturbed.dot(viewDirection) >= 0.f ? normal : normalPerturbed;
}

// OpenMVS 2.3 used Euler perturbation and consumed three random draws here.
__device__ inline Point3 GeneratePerturbedNormal23(const CUDA::Camera& camera, const Point2i& p, const Point3& normal, RandState* randState, const float perturbation)
{
	const Point3 viewDirection = camera.model.ViewDirection(p);
	const float a1 = (curand_uniform(randState) - 0.5f) * perturbation;
	const float a2 = (curand_uniform(randState) - 0.5f) * perturbation;
	const float a3 = (curand_uniform(randState) - 0.5f) * perturbation;
	const float sinA1 = sin(a1);
	const float sinA2 = sin(a2);
	const float sinA3 = sin(a3);
	const float cosA1 = cos(a1);
	const float cosA2 = cos(a2);
	const float cosA3 = cos(a3);
	Matrix3 rotation; rotation <<
		cosA2 * cosA3,
		cosA3 * sinA1 * sinA2 - cosA1 * sinA3,
		sinA1 * sinA3 + cosA1 * cosA3 * sinA2,
		cosA2 * sinA3,
		cosA1 * cosA3 + sinA1 * sinA2 * sinA3,
		cosA1 * sinA2 * sinA3 - cosA3 * sinA1,
		-sinA2,
		cosA2 * sinA1,
		cosA1 * cosA2;
	const Point3 normalPerturbed = rotation * normal.topLeftCorner<3,1>();
	if (normalPerturbed.dot(viewDirection) >= 0.f)
		return normal;
	return normalPerturbed.normalized();
}

// randomly perturb a depth, sampling uniformly from the intersection of the
// perturbation window [(1-p)d, (1+p)d] with the valid range [fDepthMin, fDepthMax]
__device__ inline float GeneratePerturbedDepth(float depth, RandState* randState, const float perturbation)
{
	const float lo = fmaxf((1.f - perturbation) * depth, g_params.fDepthMin);
	const float hi = fminf((1.f + perturbation) * depth, g_params.fDepthMax);
	return lo + curand_uniform(randState) * (hi - lo);
}

__device__ inline DVPVisibleNormalVector ToDVPVisibleNormalVector(const Point3& vector)
{
	return {vector.x(), vector.y(), vector.z()};
}

__device__ inline Point3 FromDVPVisibleNormalVector(const DVPVisibleNormalVector& vector)
{
	return Point3(vector.x, vector.y, vector.z);
}

// Evaluate camera-to-candidate directions directly in the reference-camera
// frame. Streaming avoids a 33-vector per-thread array in the register-bound
// production PatchMatch kernel. Source depth maps are deliberately not used.
__device__ inline DVPVisibleNormalEvaluation EvaluateDVPVisibleNormalRuntime(
	const Point2i& pixel,
	float depth,
	unsigned selectedSourceViews,
	const Point3& normal,
	float dotTolerance,
	uint8_t& selectedSourceCount)
{
	selectedSourceCount = 0u;
	DVPVisibleNormalEvaluation result;
	if (!DVPVisibleNormalFinite(dotTolerance) || dotTolerance < 0.f ||
		dotTolerance > 1.f)
	{
		result.reason = DVPVisibleNormalEvaluationReason::INVALID_DOT_TOLERANCE;
		return result;
	}
	if (!isfinite(depth) || depth <= 0.f) {
		result.reason = DVPVisibleNormalEvaluationReason::INVALID_DIRECTION_COUNT;
		return result;
	}
	if (!DVPVisibleNormalNormalize(ToDVPVisibleNormalVector(normal), result.normal))
		return result;
	const Point3 pointReference(
		g_cameras[0].model.TransformPointI2C(pixel.cast<float>(), depth));
	uint8_t directionIndex(0u);
	DVPVisibleNormalVector direction;
	if (!DVPVisibleNormalCameraToPointDirection(
		ToDVPVisibleNormalVector(pointReference), DVPVisibleNormalVector{}, direction))
	{
		result.reason = DVPVisibleNormalEvaluationReason::INVALID_DIRECTION;
		result.rejectedDirection = directionIndex;
		return result;
	}
	float dot(DVPVisibleNormalDot(result.normal, direction));
	result.maxDot = dot;
	if (dot > dotTolerance)
		result.rejectedDirection = directionIndex;
	++directionIndex;
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!IsBitSet(selectedSourceViews, view))
			continue;
		++selectedSourceCount;
		const Point3 sourceCenterReference(
			g_cameras[0].pose.R*(g_cameras[view+1].pose.C-g_cameras[0].pose.C));
		if (!DVPVisibleNormalCameraToPointDirection(
			ToDVPVisibleNormalVector(pointReference),
			ToDVPVisibleNormalVector(sourceCenterReference), direction))
		{
			result.directionCount = directionIndex+1u;
			result.reason = DVPVisibleNormalEvaluationReason::INVALID_DIRECTION;
			result.rejectedDirection = directionIndex;
			return result;
		}
		dot = DVPVisibleNormalDot(result.normal, direction);
		if (dot > result.maxDot)
			result.maxDot = dot;
		if (dot > dotTolerance &&
			result.rejectedDirection == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE)
		{
			result.rejectedDirection = directionIndex;
		}
		++directionIndex;
	}
	result.directionCount = directionIndex;
	result.maxViolation = result.maxDot-dotTolerance;
	result.valid = true;
	result.feasible = result.rejectedDirection == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	result.reason = result.feasible ? DVPVisibleNormalEvaluationReason::VALID :
		DVPVisibleNormalEvaluationReason::HEMISPHERE_REJECTED;
	return result;
}

enum class DVPVisibleNormalRuntimeStatus : uint8_t {
	INVALID = 0,
	REJECTED,
	FEASIBLE,
};

// Process<false> needs only validity and feasibility. Avoid materializing the
// diagnostic evaluation record in the register-bound production kernel.
__device__ inline DVPVisibleNormalRuntimeStatus ClassifyDVPVisibleNormalRuntime(
	const Point2i& pixel,
	float depth,
	unsigned selectedSourceViews,
	const Point3& normal,
	float dotTolerance)
{
	if (!DVPVisibleNormalFinite(dotTolerance) || dotTolerance < 0.f ||
		dotTolerance > 1.f || !isfinite(depth) || depth <= 0.f)
	{
		return DVPVisibleNormalRuntimeStatus::INVALID;
	}
	DVPVisibleNormalVector normalizedNormal;
	if (!DVPVisibleNormalNormalize(ToDVPVisibleNormalVector(normal), normalizedNormal))
		return DVPVisibleNormalRuntimeStatus::INVALID;
	const Point3 pointReference(
		g_cameras[0].model.TransformPointI2C(pixel.cast<float>(), depth));
	DVPVisibleNormalVector direction;
	if (!DVPVisibleNormalCameraToPointDirection(
		ToDVPVisibleNormalVector(pointReference), DVPVisibleNormalVector{}, direction))
	{
		return DVPVisibleNormalRuntimeStatus::INVALID;
	}
	bool rejected(DVPVisibleNormalDot(normalizedNormal, direction) > dotTolerance);
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!IsBitSet(selectedSourceViews, view))
			continue;
		const Point3 sourceCenterReference(
			g_cameras[0].pose.R*(g_cameras[view+1].pose.C-g_cameras[0].pose.C));
		if (!DVPVisibleNormalCameraToPointDirection(
			ToDVPVisibleNormalVector(pointReference),
			ToDVPVisibleNormalVector(sourceCenterReference), direction))
		{
			return DVPVisibleNormalRuntimeStatus::INVALID;
		}
		if (DVPVisibleNormalDot(normalizedNormal, direction) > dotTolerance)
			rejected = true;
	}
	return rejected ? DVPVisibleNormalRuntimeStatus::REJECTED :
		DVPVisibleNormalRuntimeStatus::FEASIBLE;
}

__device__ inline Point3 ResolveDVPVisibleNormalRuntimeProduction(
	const CUDA::Camera& camera,
	const Point2i& pixel,
	float depth,
	unsigned selectedSourceViews,
	const Point3& incumbentNormal,
	const Point3& nativeNormal,
	RandState* retryState,
	bool randomNormal,
	float perturbation,
	float dotTolerance,
	unsigned maxAttempts)
{
	const DVPVisibleNormalRuntimeStatus nativeStatus(
		ClassifyDVPVisibleNormalRuntime(
			pixel, depth, selectedSourceViews, nativeNormal, dotTolerance));
	if (!retryState || nativeStatus != DVPVisibleNormalRuntimeStatus::REJECTED ||
		maxAttempts == 0u || maxAttempts > DVP_VISIBLE_NORMAL_MAX_ATTEMPTS)
	{
		return nativeNormal;
	}
	for (unsigned attempt=0u; attempt<maxAttempts; ++attempt) {
		const Point3 retryNormal(randomNormal ?
			GenerateRandomNormal(camera, pixel, retryState) :
			GeneratePerturbedNormal(
				camera, pixel, incumbentNormal, retryState, perturbation));
		if (ClassifyDVPVisibleNormalRuntime(
			pixel, depth, selectedSourceViews, retryNormal, dotTolerance) ==
			DVPVisibleNormalRuntimeStatus::FEASIBLE)
		{
			return retryNormal;
		}
	}
	return nativeNormal;
}

__device__ inline DVPVisibleNormalProposalDecision ResolveDVPVisibleNormalRuntimeProposal(
	const CUDA::Camera& camera,
	const Point2i& pixel,
	float depth,
	unsigned selectedSourceViews,
	const Point3& incumbentNormal,
	const Point3& nativeNormal,
	RandState* retryState,
	bool randomNormal,
	float perturbation,
	float dotTolerance,
	unsigned maxAttempts)
{
	DVPVisibleNormalProposalDecision result;
	result.selected = ToDVPVisibleNormalVector(nativeNormal);
	uint8_t selectedSourceCount(0u);
	result.nativeEvaluation = EvaluateDVPVisibleNormalRuntime(
		pixel, depth, selectedSourceViews, nativeNormal, dotTolerance,
		selectedSourceCount);
	result.selectedEvaluation = result.nativeEvaluation;
	if (!retryState || !result.nativeEvaluation.valid ||
		maxAttempts == 0u || maxAttempts > DVP_VISIBLE_NORMAL_MAX_ATTEMPTS)
	{
		return result;
	}
	result.valid = true;
	if (result.nativeEvaluation.feasible) {
		result.reason = DVPVisibleNormalProposalReason::NATIVE_FEASIBLE;
		return result;
	}
	for (unsigned attempt=0u; attempt<maxAttempts; ++attempt) {
		const Point3 retryNormal(randomNormal ?
			GenerateRandomNormal(camera, pixel, retryState) :
			GeneratePerturbedNormal(
				camera, pixel, incumbentNormal, retryState, perturbation));
		const DVPVisibleNormalVector retry(ToDVPVisibleNormalVector(retryNormal));
		const DVPVisibleNormalEvaluation evaluation(EvaluateDVPVisibleNormalRuntime(
			pixel, depth, selectedSourceViews, retryNormal, dotTolerance,
			selectedSourceCount));
		++result.retriesTested;
		if (!evaluation.valid || !evaluation.feasible)
			continue;
		result.selected = retry;
		result.selectedEvaluation = evaluation;
		result.selectedRetry = static_cast<uint16_t>(attempt);
		result.reason = DVPVisibleNormalProposalReason::CONSTRAINED_RETRY;
		return result;
	}
	result.reason = DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK;
	result.fallback = true;
	return result;
}

// DVP can emit two interval candidates in place of the one native depth proposal.
// Sample them from a fork so later PatchMatch proposals keep the control RNG schedule.
struct DVPDepthProposalRNG {
	RandState proposalState;

	__device__ explicit DVPDepthProposalRNG(const RandState* randState)
		: proposalState(*randState) {}
	__device__ float Uniform() { return curand_uniform(&proposalState); }
	__device__ static void ConsumeNativeDepthSlot(RandState* randState)
	{
		(void)curand_uniform(randState);
	}
};

// OpenMVS 2.3 sampled the complete perturbation window without range clamping.
__device__ inline float GeneratePerturbedDepth23(float depth, RandState* randState, const float perturbation)
{
	const float lo = (1.f - perturbation) * depth;
	const float hi = (1.f + perturbation) * depth;
	return lo + curand_uniform(randState) * (hi - lo);
}

// interpolate given pixel's estimate to the current position
__device__ inline float InterpolatePixel(const CUDA::Camera& camera, const Point2i& p, const Point2i& np, float depth, const Point3& normal)
{
	float depthNew;
	if (p.x() == np.x()) {
		const float nx1 = (p.y() - camera.model.p.y()) / camera.model.f.y();
		const float denom = normal.z() + nx1 * normal.y();
		if (fabsf(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.y() - camera.model.p.y()) / camera.model.f.y();
		const float nom = depth * (normal.z() + x1 * normal.y());
		depthNew = nom / denom;
	} else if (p.y() == np.y()) {
		const float nx1 = (p.x() - camera.model.p.x()) / camera.model.f.x();
		const float denom = normal.z() + nx1 * normal.x();
		if (fabsf(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.x() - camera.model.p.x()) / camera.model.f.x();
		const float nom = depth * (normal.z() + x1 * normal.x());
		depthNew = nom / denom;
	} else {
		const float planeD = normal.dot(camera.model.TransformPointI2C(np.cast<float>(), depth));
		depthNew = planeD / normal.dot(camera.model.TransformPointI2C(p.cast<float>()));
	}
	return (depthNew >= g_params.fDepthMin && depthNew <= g_params.fDepthMax) ? depthNew : depth;
}

// compute normal to the surface given the 4 neighbors
__device__ inline Point3 ComputeDepthGradient(const LinearCameraModel& model, float depth, const Point2i& pos, const Point4& ndepth) {
	constexpr float2 nposg[4] = {{0,-1}, {0,1}, {-1,0}, {1,0}};
	Point2 dg(0,0);
	// add neighbor depths at the gradient locations
	for (int i=0; i<4; ++i)
		dg += Point2(nposg[i].x,nposg[i].y) * (ndepth[i] - depth);
	// compute depth gradient
	const Point2 d = dg*0.5f;
	// compute normal from depth gradient
	return Point3(
		model.f.x()*d.x(),
		model.f.y()*d.y(),
		(model.p.x()-pos.x())*d.x()+(model.p.y()-pos.y())*d.y()-depth).normalized();
}

// compose tho homography matrix that transforms a point from reference to source camera through the given plane
__device__ inline Matrix3 ComputeHomography(const CUDA::Camera& refCamera, const CUDA::Camera& trgCamera, const Point2& p, const Point4& plane)
{
	const Point3 X = refCamera.model.TransformPointI2C(p, plane.w());
	const Point3 normal = plane.topLeftCorner<3,1>();
	// guard against plane passing through (or near) the reference camera center:
	// normal.dot(X) -> 0 makes t infinite and the resulting H NaN
	const float denom = normal.dot(X);
	const float safeDenom = fabsf(denom) < FLT_EPSILON ? copysignf(FLT_EPSILON, denom) : denom;
	const Point3 t = (refCamera.pose.C - trgCamera.pose.C) / safeDenom;
	const Matrix3 H = trgCamera.pose.R * (refCamera.pose.R.transpose() + t*normal.transpose());
	return trgCamera.model.K() * H * refCamera.model.K().inverse();
}

// weight a neighbor texel based on color similarity and distance to the center texel
__device__ inline float ComputeBilateralWeight4(int idx, float pix, float centerPix)
{
	// spatial Gaussian for the 5x5 patch (sample positions in {-4,-2,0,2,4};
	// sigmaSpatial = -1/18) precomputed: exp(-(dx*dx + dy*dy) / 18);
	// row-major over (i, j) with i as the outer index
	static constexpr float spatialLUT[25] = {
		0.169013f, 0.329193f, 0.411112f, 0.329193f, 0.169013f,
		0.329193f, 0.641180f, 0.800737f, 0.641180f, 0.329193f,
		0.411112f, 0.800737f, 1.000000f, 0.800737f, 0.411112f,
		0.329193f, 0.641180f, 0.800737f, 0.641180f, 0.329193f,
		0.169013f, 0.329193f, 0.411112f, 0.329193f, 0.169013f,
	};
	constexpr float sigmaColor = -1.f / (2.f * 25.f/255.f*25.f/255.f);
	const float colorDistSq = Square(pix - centerPix);
	return spatialLUT[idx] * __expf(colorDistSq * sigmaColor);
}
__device__ inline float ComputeBilateralWeight(int xDist, int yDist, float pix, float centerPix)
{
	constexpr float sigmaSpatial = -1.f / (2.f * (nSizeHalfWindow-1)*(nSizeHalfWindow-1));
	constexpr float sigmaColor = -1.f / (2.f * 25.f/255.f*25.f/255.f);
	const float spatialDistSq = float(xDist * xDist + yDist * yDist);
	const float colorDistSq = Square(pix - centerPix);
	return __expf(spatialDistSq * sigmaSpatial + colorDistSq * sigmaColor);
}

// compute the geometric consistency weight
__device__ inline float GeometricConsistencyWeight(const ImagePixels depthImage, const CUDA::Camera& refCamera, const CUDA::Camera& trgCamera, const Point4& plane, const Point2i& p)
{
	if (depthImage == NULL)
		return 0.f;
	constexpr float maxDist = 4.f;
	const Point3 forwardPoint = refCamera.TransformPointI2W(p.cast<float>(), plane.w());
	const Point2 trgPt = trgCamera.TransformPointW2I(forwardPoint);
	const float trgDepth = tex2D<float>(depthImage, trgPt.x() + 0.5f, trgPt.y() + 0.5f);
	if (trgDepth == 0.f)
		return maxDist;
	const Point3 trgX = trgCamera.TransformPointI2W(trgPt, trgDepth);
	const Point2 backwardPoint = refCamera.TransformPointW2I(trgX);
	const Point2 diff = p.cast<float>() - backwardPoint;
	const float distSq = diff.squaredNorm();
	return min(maxDist, sqrtf(distSq + sqrtf(distSq)*2.f));
}

// number of samples in the (2*halfWin/step + 1)^2 reference patch
#define N_PATCH_SAMPLES ((2 * nSizeHalfWindow / nSizeStep + 1) * (2 * nSizeHalfWindow / nSizeStep + 1))

// Per-pixel reference-patch state. Depends only on the reference image at p,
// so it is invariant across source views and plane hypotheses. Compute once
// at the top of ProcessPixel / InitializePixelScore and reuse for every
// ScorePlane call (eliminates ~13*nNumViews redundant ref tex2D fetches and
// 25*13*nNumViews bilateral-weight evaluations per pixel).
struct RefPatchCache {
	float weight[N_PATCH_SAMPLES];        // bilateral weight per patch sample
	float weightRefPix[N_PATCH_SAMPLES];  // weight * refPix per sample
	float sumRef;                         // Σ weight * refPix
	float bilateralWeightSum;             // Σ weight
	float varRef;                         // sumRefRef*Σw - sumRef^2
};

__device__ inline void ComputeRefPatchCache(const ImagePixels refImage, const Point2i& p, RefPatchCache& cache)
{
	const float refCenterPix = tex2D<float>(refImage, p.x() + 0.5f, p.y() + 0.5f);
	float sumRef = 0.f, sumRefRef = 0.f, bws = 0.f;
	int idx = 0;
	#pragma unroll
	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		#pragma unroll
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep) {
			const float refPix = tex2D<float>(refImage, p.x() + j + 0.5f, p.y() + i + 0.5f);
			#if nSizeHalfWindow == 4
			const float w = ComputeBilateralWeight4(idx, refPix, refCenterPix);
			#else
			const float w = ComputeBilateralWeight(j, i, refPix, refCenterPix);
			#endif
			const float wRef = w * refPix;
			cache.weight[idx] = w;
			cache.weightRefPix[idx] = wRef;
			sumRef += wRef;
			sumRefRef += wRef * refPix;
			bws += w;
			++idx;
		}
	}
	cache.sumRef = sumRef;
	cache.bilateralWeightSum = bws;
	cache.varRef = sumRefRef * bws - sumRef * sumRef;
}
#ifdef _USE_DMAP_INSTRUMENTATION
struct PlaneScoreComponents {
	float photometricCost = fBadCost;
	float photoPriorCost = fBadCost;
	float depthPriorCost = 0.f;
	float depthPriorWeight = 0.f;
	float geometricCost = 0.f;
	float totalCost = fBadCost;
	uint32_t depthPriorUsed = 0;
	uint32_t badReason = PM_BAD_NONE;
};

// compute photometric score using weighted ZNCC; uses precomputed reference cache
__device__ PlaneScoreComponents ScorePlaneComponents(const RefPatchCache& cache, const CUDA::Camera& refCamera, const ImagePixels trgImage, const CUDA::Camera& trgCamera, const Point2i& p, const Point4& plane, const float lowDepth)
{
	PlaneScoreComponents components;
	Matrix3 H = ComputeHomography(refCamera, trgCamera, p.cast<float>(), plane);
	// inline hnormalized() as RCP + 2 FMAs (the +0.5 tex2D pixel-center bias rides into the FMA)
	// replaces 2 IEEE divisions per sample in the 25-sample patch walk; hottest inner loop, ~-29% per-view kernel time
	{
		const Point3 ptH = H * p.cast<float>().homogeneous();
		if (!isfinite(ptH.z()) || fabsf(ptH.z()) < FLT_EPSILON) {
			components.badReason = PM_BAD_INVALID_PROJECTION;
			return components;
		}
		const float invZ = __fdividef(1.f, ptH.z());
		const float ptX = ptH.x() * invZ;
		const float ptY = ptH.y() * invZ;
		if (!isfinite(ptX) || !isfinite(ptY)) {
			components.badReason = PM_BAD_INVALID_PROJECTION;
			return components;
		}
		if (ptX >= trgCamera.size.x() || ptX < 0.f || ptY >= trgCamera.size.y() || ptY < 0.f) {
			components.badReason = PM_BAD_OUT_OF_BOUNDS;
			return components;
		}
	}
	Point3 X = H * Point2(p.x()-nSizeHalfWindow, p.y()-nSizeHalfWindow).homogeneous();
	Point3 baseX(X);
	H *= float(nSizeStep);

	float sumTrg = 0.f, sumTrgTrg = 0.f, sumRefTrg = 0.f;
	int idx = 0;
	#pragma unroll
	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		#pragma unroll
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep) {
			const float invZ = __fdividef(1.f, X.z());
			const float trgPx = X.x() * invZ + 0.5f;
			const float trgPy = X.y() * invZ + 0.5f;
			const float trgPix = tex2D<float>(trgImage, trgPx, trgPy);
			const float w = cache.weight[idx];
			const float wTrg = w * trgPix;
			sumTrg += wTrg;
			sumTrgTrg += wTrg * trgPix;
			sumRefTrg += cache.weightRefPix[idx] * trgPix;
			++idx;
			X += H.col(0);
		}
		baseX += H.col(1);
		X = baseX;
	}

	if (lowDepth <= 0 && cache.varRef < 1e-8f) {
		components.badReason = PM_BAD_LOW_REF_VARIANCE;
		return components;
	}
	const float varTrg = sumTrgTrg * cache.bilateralWeightSum - sumTrg * sumTrg;
	const float varRefTrg = cache.varRef * varTrg;
	if (varRefTrg < 1e-16f) {
		components.badReason = PM_BAD_LOW_TARGET_VARIANCE;
		return components;
	}
	const float covarTrgRef = sumRefTrg * cache.bilateralWeightSum - cache.sumRef * sumTrg;
	float ncc = 1.f - covarTrgRef * rsqrtf(varRefTrg);
	components.photometricCost = max(0.f, min(2.f, ncc));

	// apply depth prior weight based on patch textureless;
	// hard-cap the prior on medium to well-textured patches:
	// 0.0025 is the optimum tested on several GT datasets
	if (lowDepth > 0 && cache.varRef < 0.0025f) {
		const float depth(plane.w());
		const float deltaDepth(min((fabsf(lowDepth-depth) / lowDepth), 0.5f));
		constexpr float smoothSigmaDepth(-1.f / (1.f * 0.02f)); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
		// Mirror ScorePlane exactly, including its roundoff behavior, because this
		// value is exported as the production-exact depth-prior weight.
		const float factorDeltaDepth(__expf(cache.varRef * smoothSigmaDepth));
		ncc = (1.f-factorDeltaDepth)*ncc + factorDeltaDepth*deltaDepth;
		components.depthPriorCost = deltaDepth;
		components.depthPriorWeight = factorDeltaDepth;
		components.depthPriorUsed = 1;
	}
	components.photoPriorCost = max(0.f, min(2.f, ncc));
	components.totalCost = components.photoPriorCost;
	return components;
}

#endif

// compute photometric score using weighted ZNCC; uses precomputed reference cache
__device__ float ScorePlane(const RefPatchCache& cache, const CUDA::Camera& refCamera, const ImagePixels trgImage, const CUDA::Camera& trgCamera, const Point2i& p, const Point4& plane, const float lowDepth)
{
	Matrix3 H = ComputeHomography(refCamera, trgCamera, p.cast<float>(), plane);
	// inline hnormalized() as RCP + 2 FMAs (the +0.5 tex2D pixel-center bias rides into the FMA)
	// replaces 2 IEEE divisions per sample in the 25-sample patch walk; hottest inner loop, ~-29% per-view kernel time
	{
		const Point3 ptH = H * p.cast<float>().homogeneous();
		const float invZ = __fdividef(1.f, ptH.z());
		const float ptX = ptH.x() * invZ;
		const float ptY = ptH.y() * invZ;
		if (ptX >= trgCamera.size.x() || ptX < 0.f || ptY >= trgCamera.size.y() || ptY < 0.f)
			return fBadCost;
	}
	Point3 X = H * Point2(p.x()-nSizeHalfWindow, p.y()-nSizeHalfWindow).homogeneous();
	Point3 baseX(X);
	H *= float(nSizeStep);

	float sumTrg = 0.f, sumTrgTrg = 0.f, sumRefTrg = 0.f;
	int idx = 0;
	#pragma unroll
	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		#pragma unroll
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep) {
			const float invZ = __fdividef(1.f, X.z());
			const float trgPx = X.x() * invZ + 0.5f;
			const float trgPy = X.y() * invZ + 0.5f;
			const float trgPix = tex2D<float>(trgImage, trgPx, trgPy);
			const float w = cache.weight[idx];
			const float wTrg = w * trgPix;
			sumTrg += wTrg;
			sumTrgTrg += wTrg * trgPix;
			sumRefTrg += cache.weightRefPix[idx] * trgPix;
			++idx;
			X += H.col(0);
		}
		baseX += H.col(1);
		X = baseX;
	}

	if (lowDepth <= 0 && cache.varRef < 1e-8f)
		return fBadCost;
	const float varTrg = sumTrgTrg * cache.bilateralWeightSum - sumTrg * sumTrg;
	const float varRefTrg = cache.varRef * varTrg;
	if (varRefTrg < 1e-16f)
		return fBadCost;
	const float covarTrgRef = sumRefTrg * cache.bilateralWeightSum - cache.sumRef * sumTrg;
	float ncc = 1.f - covarTrgRef * rsqrtf(varRefTrg);

	// apply depth prior weight based on patch textureless;
	// hard-cap the prior on medium to well-textured patches:
	// 0.0025 is the optimum tested on several GT datasets
	if (lowDepth > 0 && (g_params.bCompat23 || cache.varRef < 0.0025f)) {
		const float depth(plane.w());
		const float deltaDepth(min((fabsf(lowDepth-depth) / lowDepth), 0.5f));
		constexpr float smoothSigmaDepth(-1.f / (1.f * 0.02f)); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
		const float factorDeltaDepth(__expf(cache.varRef * smoothSigmaDepth));
		ncc = (1.f-factorDeltaDepth)*ncc + factorDeltaDepth*deltaDepth;
	}
	return max(0.f, min(2.f, ncc));
}

// Paper APD uses an unweighted 11x11 support sampled asymmetrically: the
// center window walks by theta=2 (36 samples), while every anchor window walks
// by w/2=5 (9 samples). Invalid support contributes the OpenMVS bad-cost
// sentinel; this is an explicit integration adaptation from the released
// implementation's value 2 so candidate validity remains conventional.
__device__ inline float ScorePlaneAPDPatch(
	const Matrix3& homography,
	const ImagePixels refImage,
	const ImagePixels trgImage,
	const CUDA::Camera& trgCamera,
	const Point2i& center,
	const APDPatchKind patchKind)
{
	if (!APDReferencePatchFits(
		center.x(), center.y(), g_cameras[0].size.x(), g_cameras[0].size.y()))
		return fBadCost;
	float sumRef(0.f), sumRefRef(0.f), sumTrg(0.f), sumTrgTrg(0.f), sumRefTrg(0.f);
	const unsigned sampleCount(APDPatchSampleCount(patchKind));
	for (unsigned sample=0; sample<sampleCount; ++sample) {
		const APDPatchOffset offset(MakeAPDPatchOffset(patchKind, sample));
		const Point2i refPoint(center.x()+offset.x, center.y()+offset.y);
		const Point3 projectedH(homography*refPoint.cast<float>().homogeneous());
		if (!isfinite(projectedH.z()) || fabsf(projectedH.z()) < FLT_EPSILON)
			return fBadCost;
		const float inverseZ(__fdividef(1.f, projectedH.z()));
		const float targetX(projectedH.x()*inverseZ);
		const float targetY(projectedH.y()*inverseZ);
		if (!isfinite(targetX) || !isfinite(targetY) || targetX < 0.f || targetY < 0.f ||
			targetX >= static_cast<float>(trgCamera.size.x()-1) ||
			targetY >= static_cast<float>(trgCamera.size.y()-1))
			return fBadCost;
		const float refPixel(tex2D<float>(refImage, refPoint.x()+0.5f, refPoint.y()+0.5f));
		const float targetPixel(tex2D<float>(trgImage, targetX+0.5f, targetY+0.5f));
		sumRef += refPixel;
		sumRefRef += refPixel*refPixel;
		sumTrg += targetPixel;
		sumTrgTrg += targetPixel*targetPixel;
		sumRefTrg += refPixel*targetPixel;
	}
	const float inverseCount(1.f/static_cast<float>(sampleCount));
	const float meanRef(sumRef*inverseCount);
	const float meanTrg(sumTrg*inverseCount);
	const float varianceRef(sumRefRef*inverseCount-meanRef*meanRef);
	const float varianceTrg(sumTrgTrg*inverseCount-meanTrg*meanTrg);
	constexpr float minimumVariance(1e-5f);
	if (!(varianceRef >= minimumVariance) || !(varianceTrg >= minimumVariance))
		return fBadCost;
	const float covariance(sumRefTrg*inverseCount-meanRef*meanTrg);
	return max(0.f, min(2.f, 1.f-covariance*rsqrtf(varianceRef*varianceTrg)));
}

struct APDViewScoreComponents {
	float centerCost = fBadCost;
	float anchorMeanCost = fBadCost;
	float deformablePhotometricCost = fBadCost;
	float geometricCost = 0.f;
	float totalCost = fBadCost;
	uint8_t validAnchorSupportCount = 0u;
	uint8_t invalidAnchorSupportCount = 0u;
};

struct APDAggregateScoreComponents {
	float centerCost = -1.f;
	float anchorMeanCost = -1.f;
	float deformablePhotometricCost = -1.f;
	float geometricCost = -1.f;
};

template <bool GEOM, bool CAPTURE_COMPONENTS = false, bool VISIBILITY = false>
__device__ inline float ScorePlaneAPD(
	const ImagePixels* images,
	const ImagePixels* depthImages,
	const Point2i& p,
	const Point4& plane,
	const uint32_t* anchors,
	const uint8_t anchorCount,
	const int targetView,
	APDViewScoreComponents* capturedComponents = nullptr,
	const uint8_t* visibilityWeights = nullptr)
{
	const CUDA::Camera& targetCamera(g_cameras[targetView+1]);
	const Matrix3 homography(ComputeHomography(g_cameras[0], targetCamera, p.cast<float>(), plane));
	const float centerCost(ScorePlaneAPDPatch(
		homography, images[0], images[targetView+1], targetCamera, p, APDPatchKind::CENTER));
	if (centerCost >= fBadCost || !isfinite(centerCost)) {
		if constexpr (CAPTURE_COMPONENTS) {
			if (capturedComponents)
				*capturedComponents = APDViewScoreComponents{};
		}
		return fBadCost;
	}
	APDAnchorCostAccumulator anchorCosts;
	const int width(g_cameras[0].size.x());
	const int area(width*g_cameras[0].size.y());
	for (unsigned slot=0; slot<min(static_cast<unsigned>(anchorCount), APD_MAX_ANCHORS); ++slot) {
		const uint32_t anchorIndex(anchors[slot]);
		if constexpr (VISIBILITY) {
			if (!visibilityWeights || anchorIndex >= static_cast<uint32_t>(area) ||
				visibilityWeights[(size_t)anchorIndex*MAX_VIEWS+targetView] == 0u)
			{
				continue;
			}
		}
		if (anchorIndex >= static_cast<uint32_t>(area)) {
			AccumulateAPDAnchorCost(anchorCosts, false, 0.f);
			continue;
		}
		const Point2i anchor(
			static_cast<int>(anchorIndex%static_cast<uint32_t>(width)),
			static_cast<int>(anchorIndex/static_cast<uint32_t>(width)));
		const float anchorCost(ScorePlaneAPDPatch(
			homography, images[0], images[targetView+1], targetCamera,
			anchor, APDPatchKind::ANCHOR));
		AccumulateAPDAnchorCost(anchorCosts, anchorCost < fBadCost, anchorCost);
	}
	const bool anchorConsensusValid(VISIBILITY ?
		anchorCosts.anchorCount >= APD_MIN_INLIERS : anchorCount >= APD_MIN_INLIERS);
	const APDScoreDecision score(EvaluateAPDWorkingScore(
		true, APDReliabilityClass::UNRELIABLE, true,
		anchorConsensusValid, centerCost, anchorCosts));
	float geometricCost(0.f);
	if constexpr (GEOM)
		geometricCost = 0.1f*GeometricConsistencyWeight(
			depthImages[targetView], g_cameras[0], targetCamera, plane, p);
	const float cost(score.workingCost+geometricCost);
	if constexpr (CAPTURE_COMPONENTS) {
		if (capturedComponents) {
			// A visibility-gated view can lose anchor consensus and correctly
			// fall back to its center cost. Record the effective anchor leg so
			// the aggregate 0.25/0.75 decomposition remains exact.
			const float effectiveAnchorCost(
				score.usedDeformableCost ? score.anchorMeanCost : centerCost);
			capturedComponents->centerCost = centerCost;
			capturedComponents->anchorMeanCost = effectiveAnchorCost;
			capturedComponents->deformablePhotometricCost = score.workingCost;
			capturedComponents->geometricCost = geometricCost;
			capturedComponents->totalCost = cost;
			capturedComponents->validAnchorSupportCount = static_cast<uint8_t>(
				min(anchorCosts.validSupportCount, 255u));
			capturedComponents->invalidAnchorSupportCount = static_cast<uint8_t>(
				min(anchorCosts.invalidSupportCount, 255u));
		}
	}
	return cost;
}

// Keep register allocation within the calling kernel's launch bounds.
template <bool GEOM, bool CAPTURE_COMPONENTS = false, bool VISIBILITY = false>
__device__ __forceinline__ void MultiViewScorePlaneAPD(
	const ImagePixels* images,
	const ImagePixels* depthImages,
	const Point2i& p,
	const Point4& plane,
	const uint32_t* anchors,
	const uint8_t anchorCount,
	float* costVector,
	APDViewScoreComponents* capturedComponents = nullptr,
	const uint8_t* visibilityWeights = nullptr)
{
	for (int view=0; view<g_params.nNumViews; ++view) {
		APDViewScoreComponents* components(nullptr);
		if constexpr (CAPTURE_COMPONENTS)
			components = capturedComponents ? capturedComponents+view : nullptr;
		costVector[view] = ScorePlaneAPD<GEOM, CAPTURE_COMPONENTS, VISIBILITY>(
			images, depthImages, p, plane, anchors, anchorCount, view, components,
			visibilityWeights);
	}
}

__device__ inline float AggregateAPDViewScores(
	const unsigned* viewWeights,
	const float* costVector,
	const int numViews)
{
	float weightedCost(0.f);
	unsigned weightSum(0u);
	for (int view=0; view<numViews; ++view) {
		weightedCost += static_cast<float>(viewWeights[view])*costVector[view];
		weightSum += viewWeights[view];
	}
	return weightSum > 0u ? weightedCost/static_cast<float>(weightSum) : fBadCost;
}

__device__ inline unsigned APDCostRank(const float* costs, const int view, const int numViews)
{
	unsigned rank(0u);
	for (int other=0; other<numViews; ++other)
		if (costs[other] < costs[view] || (costs[other] == costs[view] && other < view))
			++rank;
	return rank;
}

// compute photometric score for all neighbor images;
// GEOM-templated so geom-consistency loop is dead-code eliminated when off
template <bool GEOM>
__device__ inline void MultiViewScorePlane(const RefPatchCache& cache, const ImagePixels* images, const ImagePixels* depthImages, const Point2i& p, const Point4& plane, const float lowDepth, float* costVector)
{
	const int nNumViews = g_params.nNumViews;
	for (int imgId = 1; imgId <= nNumViews; ++imgId)
		costVector[imgId-1] = ScorePlane(cache, g_cameras[0], images[imgId], g_cameras[imgId], p, plane, lowDepth);
	if (GEOM) {
		for (int imgId = 0; imgId < nNumViews; ++imgId)
			costVector[imgId] += 0.1f * GeometricConsistencyWeight(depthImages[imgId], g_cameras[0], g_cameras[imgId+1], plane, p);
	}
}
// same as above, but interpolate the plane to current pixel position
template <bool GEOM>
__device__ inline float MultiViewScoreNeighborPlane(const RefPatchCache& cache, const ImagePixels* images, const ImagePixels* depthImages, const Point2i& p, const Point2i& np, Point4 plane, const float lowDepth, float* costVector)
{
	plane.w() = InterpolatePixel(g_cameras[0], p, np, plane.w(), plane.topLeftCorner<3,1>());
	MultiViewScorePlane<GEOM>(cache, images, depthImages, p, plane, lowDepth, costVector);
	return plane.w();
}

// aggregate photometric scores from MC-sampled views into one per-pixel
// cost: the MC-weighted mean over views with viewWeights > 0. Sentinel
// views (cost == fBadCost from ScorePlane: out-of-frame, occlusion, or
// degenerate variance) are included at their raw cost, pulling the mean
// upward and disadvantaging plane hypotheses that fail to project many
// views. NUM_SAMPLES = sum(viewWeights[]) by construction in
// ProcessPixel (NUM_SAMPLES MC draws each increment one viewWeights[])
__device__ inline float AggregateMultiViewScores(const unsigned* viewWeights, const float* costVector, int numViews)
{
	float cost = 0;
	for (int imgId = 0; imgId < numViews; ++imgId)
		if (viewWeights[imgId])
			cost += viewWeights[imgId] * costVector[imgId];
	return cost / float(NUM_SAMPLES);
}

template <bool GEOM, bool APD, bool CAPTURE_APD_COMPONENTS = false, bool VISIBILITY = false>
__device__ inline float ScorePatchMatchCandidate(
	const RefPatchCache& refCache,
	const ImagePixels* images,
	const ImagePixels* depthImages,
	const Point2i& p,
	const Point4& plane,
	const float lowDepth,
	const bool apdActive,
	const uint32_t* anchors,
	const uint8_t anchorCount,
	const unsigned* viewWeights,
	float* costVector,
	APDAggregateScoreComponents* capturedAggregate = nullptr,
	const uint8_t* visibilityWeights = nullptr)
{
	if constexpr (APD) {
		if (apdActive) {
			if constexpr (CAPTURE_APD_COMPONENTS) {
				APDViewScoreComponents viewComponents[MAX_VIEWS];
				MultiViewScorePlaneAPD<GEOM, true, VISIBILITY>(
					images, depthImages, p, plane, anchors, anchorCount, costVector,
					viewComponents, visibilityWeights);
				const float cost(AggregateAPDViewScores(
					viewWeights, costVector, g_params.nNumViews));
				if (capturedAggregate) {
					float centerSum(0.f);
					float anchorSum(0.f);
					float deformableSum(0.f);
					float geometricSum(0.f);
					unsigned weightSum(0u);
					for (int view=0; view<g_params.nNumViews; ++view) {
						const unsigned weight(viewWeights[view]);
						centerSum += static_cast<float>(weight)*viewComponents[view].centerCost;
						anchorSum += static_cast<float>(weight)*viewComponents[view].anchorMeanCost;
						deformableSum += static_cast<float>(weight)*viewComponents[view].deformablePhotometricCost;
						geometricSum += static_cast<float>(weight)*viewComponents[view].geometricCost;
						weightSum += weight;
					}
					if (weightSum > 0u) {
						const float inverseWeight(1.f/static_cast<float>(weightSum));
						capturedAggregate->centerCost = centerSum*inverseWeight;
						capturedAggregate->anchorMeanCost = anchorSum*inverseWeight;
						capturedAggregate->deformablePhotometricCost = deformableSum*inverseWeight;
						capturedAggregate->geometricCost = geometricSum*inverseWeight;
					}
				}
				return cost;
			} else {
				MultiViewScorePlaneAPD<GEOM, false, VISIBILITY>(
					images, depthImages, p, plane, anchors, anchorCount, costVector,
					nullptr, visibilityWeights);
				return AggregateAPDViewScores(
					viewWeights, costVector, g_params.nNumViews);
			}
		}
	}
	MultiViewScorePlane<GEOM>(refCache, images, depthImages, p, plane, lowDepth, costVector);
	if constexpr (VISIBILITY)
		return AggregateAPDViewScores(viewWeights, costVector, g_params.nNumViews);
	return AggregateMultiViewScores(viewWeights, costVector, g_params.nNumViews);
}

struct DVPGlobalCandidate {
	float depth = -1.f;
	float reprojectionError = FLT_MAX;
	float meanReprojectionError = -1.f;
	float maxReprojectionError = -1.f;
	float meanRelativeDepthError = -1.f;
	float maxRelativeDepthError = -1.f;
	int16_t signedOffset = 0;
	uint8_t sourceView = 0xffu;
	uint8_t support = 0u;
	uint32_t supportViews = 0u;
	uint32_t occludedViews = 0u;
	bool valid = false;
};

struct DVPProposalSet {
	float depths[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	int16_t signedOffsets[DVP_MAX_PROPOSALS] = {};
	uint8_t sourceViews[DVP_MAX_PROPOSALS] = {0xffu, 0xffu};
	uint8_t supports[DVP_MAX_PROPOSALS] = {};
	uint32_t supportViews[DVP_MAX_PROPOSALS] = {};
	uint32_t occludedViews[DVP_MAX_PROPOSALS] = {};
	float meanReprojectionErrors[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float maxReprojectionErrors[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float meanRelativeDepthErrors[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float maxRelativeDepthErrors[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	DVPEndpointSamples endpointSamples;
	DVPIntervalSet intervals;
	DVPEpipolarUnavailableReason reason = DVPEpipolarUnavailableReason::NONE;
	uint32_t selectedSourceViews = 0u;
	uint32_t directionSourceViews = 0u;
	uint8_t count = 0u;
	bool nativeDepthFallback = false;
};

__device__ __forceinline__ bool DVPPointFinite(const Point2& point)
{
	return isfinite(point.x()) && isfinite(point.y());
}

__device__ __forceinline__ bool DVPPointInside(const Point2& point, const Camera& camera)
{
	return DVPPointFinite(point) && point.x() >= 0.f && point.y() >= 0.f &&
		point.x() < static_cast<float>(camera.size.x()) &&
		point.y() < static_cast<float>(camera.size.y());
}

__device__ __forceinline__ bool DVPSourceDepthToReferenceEvidence(
	const ImagePixels* depthImages, int view, const Point2& sourcePixel,
	float& referenceDepth)
{
	referenceDepth = -1.f;
	if (!depthImages || view < 0 || view >= g_params.nNumViews ||
		depthImages[view] == 0)
		return false;
	const Camera& sourceCamera(g_cameras[view+1]);
	if (!DVPPointInside(sourcePixel, sourceCamera))
		return false;
	const float sourceDepth(tex2D<float>(
		depthImages[view], sourcePixel.x()+0.5f, sourcePixel.y()+0.5f));
	if (!isfinite(sourceDepth) || sourceDepth <= 0.f)
		return false;
	const Point3 world(sourceCamera.TransformPointI2W(sourcePixel, sourceDepth));
	const Point3 referenceCameraPoint(
		g_cameras[0].pose.R*(world-g_cameras[0].pose.C));
	referenceDepth = referenceCameraPoint.z();
	if (!isfinite(referenceDepth) || referenceDepth < g_params.fDepthMin ||
		referenceDepth > g_params.fDepthMax)
		return false;
	const Point2 referencePixel(g_cameras[0].TransformPointW2I(world));
	if (!DVPPointFinite(referencePixel))
		return false;
	return true;
}

__device__ __forceinline__ bool DVPEpipolarDirection(
	const Point2i& pixel, float depth, int view, Point2& sourcePixel, Point2& direction)
{
	if (!isfinite(depth) || depth < g_params.fDepthMin || depth > g_params.fDepthMax ||
		view < 0 || view >= g_params.nNumViews)
		return false;
	const Camera& referenceCamera(g_cameras[0]);
	const Camera& sourceCamera(g_cameras[view+1]);
	const Point2 referencePixel(pixel.cast<float>());
	sourcePixel = sourceCamera.TransformPointW2I(
		referenceCamera.TransformPointI2W(referencePixel, depth));
	if (!DVPPointInside(sourcePixel, sourceCamera))
		return false;
	const float depthStep(max(depth*0.01f, 1e-4f));
	const float nearDepth(max(g_params.fDepthMin, depth-depthStep));
	const float farDepth(min(g_params.fDepthMax, depth+depthStep));
	if (!(nearDepth < farDepth))
		return false;
	const Point2 sourceNear(sourceCamera.TransformPointW2I(
		referenceCamera.TransformPointI2W(referencePixel, nearDepth)));
	const Point2 sourceFar(sourceCamera.TransformPointW2I(
		referenceCamera.TransformPointI2W(referencePixel, farDepth)));
	direction = sourceFar-sourceNear;
	const float squaredNorm(direction.squaredNorm());
	if (!DVPPointFinite(sourceNear) || !DVPPointFinite(sourceFar) ||
		!isfinite(squaredNorm) || squaredNorm <= 1e-8f)
		return false;
	direction *= rsqrtf(squaredNorm);
	return DVPPointFinite(direction);
}

__device__ __forceinline__ bool DVPAppendEndpointForView(
	float* values, uint8_t& count, uint32_t& views, float value, unsigned view)
{
	if (!DVPAppendEndpoint(values, count, value))
		return false;
	SetBit(views, view);
	return true;
}

template <bool GEOM>
__device__ DVPProposalSet DVPCollectEndpointProposals(
	const ImagePixels* depthImages, const Point2i& pixel, float depth,
	const unsigned* viewWeights, unsigned family)
{
	DVPProposalSet result;
	if constexpr (!GEOM) {
		result.reason = DVPEpipolarUnavailableReason::GEOMETRY_UNAVAILABLE;
		return result;
	}
	if (!depthImages) {
		result.reason = DVPEpipolarUnavailableReason::GEOMETRY_UNAVAILABLE;
		return result;
	}
	if (!isfinite(depth) || depth < g_params.fDepthMin || depth > g_params.fDepthMax) {
		result.reason = DVPEpipolarUnavailableReason::INVALID_REFERENCE_DEPTH;
		return result;
	}
	const float alpha(g_params.fDVPEpipolarAlpha);
	const float outer(alpha+g_params.fDVPEpipolarBeta);
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!viewWeights || viewWeights[view] == 0u)
			continue;
		SetBit(result.selectedSourceViews, static_cast<unsigned>(view));
		Point2 sourcePixel, direction;
		if (!DVPEpipolarDirection(pixel, depth, view, sourcePixel, direction))
			continue;
		SetBit(result.directionSourceViews, static_cast<unsigned>(view));
		float sampleDepth(-1.f);
		if (DVPSourceDepthToReferenceEvidence(depthImages, view,
			sourcePixel-direction*outer, sampleDepth))
		{
			DVPAppendEndpointForView(result.endpointSamples.leftOuter,
				result.endpointSamples.leftOuterCount, result.endpointSamples.leftOuterViews,
				sampleDepth, static_cast<unsigned>(view));
		}
		if (DVPSourceDepthToReferenceEvidence(depthImages, view,
			sourcePixel-direction*alpha, sampleDepth))
		{
			DVPAppendEndpointForView(result.endpointSamples.leftInner,
				result.endpointSamples.leftInnerCount, result.endpointSamples.leftInnerViews,
				sampleDepth, static_cast<unsigned>(view));
		}
		if (DVPSourceDepthToReferenceEvidence(depthImages, view,
			sourcePixel+direction*alpha, sampleDepth))
		{
			DVPAppendEndpointForView(result.endpointSamples.rightInner,
				result.endpointSamples.rightInnerCount, result.endpointSamples.rightInnerViews,
				sampleDepth, static_cast<unsigned>(view));
		}
		if (DVPSourceDepthToReferenceEvidence(depthImages, view,
			sourcePixel+direction*outer, sampleDepth))
		{
			DVPAppendEndpointForView(result.endpointSamples.rightOuter,
				result.endpointSamples.rightOuterCount, result.endpointSamples.rightOuterViews,
				sampleDepth, static_cast<unsigned>(view));
		}
	}
	result.endpointSamples.selectedSourceViews = result.selectedSourceViews;
	if (result.selectedSourceViews == 0u) {
		result.reason = DVPEpipolarUnavailableReason::NO_SELECTED_SOURCE_VIEW;
		return result;
	}
	if (result.directionSourceViews == 0u) {
		result.reason = DVPEpipolarUnavailableReason::INVALID_EPIPOLAR_DIRECTION;
		return result;
	}
	if (family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_MIDPOINT_V1)) {
		result.intervals = DVPBuildHistoricalMidpointIntervals(
			result.endpointSamples, g_params.nDVPEpipolarMu);
		if (result.intervals.left.valid) {
			result.depths[result.count] = DVPIntervalMidpoint(result.intervals.left);
			result.signedOffsets[result.count] = -1;
			result.supports[result.count] = static_cast<uint8_t>(g_params.nDVPEpipolarMu);
			result.supportViews[result.count++] = result.endpointSamples.leftOuterViews &
				result.endpointSamples.leftInnerViews;
		}
		if (result.intervals.right.valid && result.count < DVP_MAX_PROPOSALS) {
			result.depths[result.count] = DVPIntervalMidpoint(result.intervals.right);
			result.signedOffsets[result.count] = 1;
			result.supports[result.count] = static_cast<uint8_t>(g_params.nDVPEpipolarMu);
			result.supportViews[result.count++] = result.endpointSamples.rightInnerViews &
				result.endpointSamples.rightOuterViews;
		}
	} else {
		result.intervals = DVPBuildPaperIntervals(
			result.endpointSamples, g_params.nDVPEpipolarMu);
	}
	result.reason = result.intervals.reason;
	return result;
}

__device__ __forceinline__ DVPGlobalCandidate DVPFindBestGlobalCandidateInView(
	const ImagePixels* depthImages, const Point2i& pixel, float depth,
	int view, unsigned searchRadius, bool& directionValid)
{
	DVPGlobalCandidate result;
	directionValid = false;
	Point2 sourcePixel, direction;
	if (!DVPEpipolarDirection(pixel, depth, view, sourcePixel, direction))
		return result;
	directionValid = true;
	Point2 lastPixel(-FLT_MAX, -FLT_MAX);
	const Point2 referencePixel(pixel.cast<float>());
	for (int offset=-static_cast<int>(searchRadius);
		offset<=static_cast<int>(searchRadius); ++offset)
	{
		const Point2 candidate(sourcePixel+direction*static_cast<float>(offset));
		const Point2 rounded(floorf(candidate.x()+0.5f), floorf(candidate.y()+0.5f));
		if (rounded.x() == lastPixel.x() && rounded.y() == lastPixel.y())
			continue;
		lastPixel = rounded;
		float candidateDepth(-1.f);
		if (!DVPSourceDepthToReferenceEvidence(
			depthImages, view, rounded, candidateDepth))
			continue;
		const Point3 candidateWorld(
			g_cameras[0].TransformPointI2W(referencePixel, candidateDepth));
		const Point2 sourceReprojection(g_cameras[view+1].TransformPointW2I(candidateWorld));
		const float error((sourceReprojection-rounded).norm());
		if (!isfinite(error))
			continue;
		if (!result.valid || error < result.reprojectionError ||
			(error == result.reprojectionError && abs(offset) < abs(result.signedOffset)))
		{
			result.valid = true;
			result.depth = candidateDepth;
			result.reprojectionError = error;
			result.signedOffset = static_cast<int16_t>(offset);
			result.sourceView = static_cast<uint8_t>(view);
		}
	}
	return result;
}

__device__ inline DVPGlobalCandidate DVPValidateGlobalCandidate(
	const ImagePixels* depthImages, const Point2i& pixel,
	const unsigned* viewWeights, DVPGlobalCandidate candidate)
{
	if (!candidate.valid ||
		candidate.reprojectionError > g_params.fDVPReprojectionThreshold)
	{
		candidate.valid = false;
		return candidate;
	}
	const Point2 referencePixel(pixel.cast<float>());
	const Point3 candidateWorld(
		g_cameras[0].TransformPointI2W(referencePixel, candidate.depth));
	float reprojectionSum(0.f), reprojectionMax(0.f);
	float relativeDepthSum(0.f), relativeDepthMax(0.f);
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!viewWeights || viewWeights[view] == 0u || !depthImages || depthImages[view] == 0)
			continue;
		const Camera& sourceCamera(g_cameras[view+1]);
		const Point3 expectedSource(sourceCamera.pose.R*(candidateWorld-sourceCamera.pose.C));
		if (!isfinite(expectedSource.z()) || expectedSource.z() <= 0.f)
			continue;
		const Point2 sourcePixel(sourceCamera.model.TransformPointC2I(expectedSource));
		if (!DVPPointInside(sourcePixel, sourceCamera))
			continue;
		const float observedDepth(tex2D<float>(
			depthImages[view], sourcePixel.x()+0.5f, sourcePixel.y()+0.5f));
		if (!isfinite(observedDepth) || observedDepth <= 0.f)
			continue;
		const float relativeDepthError(
			fabsf(observedDepth-expectedSource.z())/max(fabsf(expectedSource.z()), 1e-6f));
		if (observedDepth < expectedSource.z()*(1.f-g_params.fDVPRelativeDepthThreshold)) {
			SetBit(candidate.occludedViews, static_cast<unsigned>(view));
			continue;
		}
		if (!isfinite(relativeDepthError) ||
			relativeDepthError > g_params.fDVPRelativeDepthThreshold)
			continue;
		const Point3 observedWorld(sourceCamera.TransformPointI2W(sourcePixel, observedDepth));
		const Point2 roundTrip(g_cameras[0].TransformPointW2I(observedWorld));
		const float reprojectionError((roundTrip-referencePixel).norm());
		if (!isfinite(reprojectionError) ||
			reprojectionError > g_params.fDVPReprojectionThreshold)
			continue;
		SetBit(candidate.supportViews, static_cast<unsigned>(view));
		++candidate.support;
		reprojectionSum += reprojectionError;
		reprojectionMax = max(reprojectionMax, reprojectionError);
		relativeDepthSum += relativeDepthError;
		relativeDepthMax = max(relativeDepthMax, relativeDepthError);
	}
	if (candidate.support < g_params.nDVPEpipolarMu) {
		candidate.valid = false;
		return candidate;
	}
	candidate.meanReprojectionError = reprojectionSum/static_cast<float>(candidate.support);
	candidate.maxReprojectionError = reprojectionMax;
	candidate.meanRelativeDepthError = relativeDepthSum/static_cast<float>(candidate.support);
	candidate.maxRelativeDepthError = relativeDepthMax;
	return candidate;
}

__device__ __forceinline__ bool DVPPreferGatedGlobalCandidate(
	const DVPGlobalCandidate& candidate, const DVPGlobalCandidate& incumbent)
{
	if (!candidate.valid)
		return false;
	if (!incumbent.valid)
		return true;
	if (candidate.support != incumbent.support)
		return candidate.support > incumbent.support;
	if (candidate.meanReprojectionError != incumbent.meanReprojectionError)
		return candidate.meanReprojectionError < incumbent.meanReprojectionError;
	if (candidate.meanRelativeDepthError != incumbent.meanRelativeDepthError)
		return candidate.meanRelativeDepthError < incumbent.meanRelativeDepthError;
	if (candidate.sourceView != incumbent.sourceView)
		return candidate.sourceView < incumbent.sourceView;
	return abs(candidate.signedOffset) < abs(incumbent.signedOffset);
}

template <bool GEOM>
__device__ DVPProposalSet DVPCollectGlobalProposal(
	const ImagePixels* depthImages, const Point2i& pixel, float depth,
	const unsigned* viewWeights, unsigned family)
{
	DVPProposalSet result;
	if constexpr (!GEOM) {
		result.reason = DVPEpipolarUnavailableReason::GEOMETRY_UNAVAILABLE;
		return result;
	}
	if (!depthImages) {
		result.reason = DVPEpipolarUnavailableReason::GEOMETRY_UNAVAILABLE;
		return result;
	}
	DVPGlobalCandidate winner;
	bool anyFinite(false), anyReprojectionPass(false), anyOccluded(false);
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!viewWeights || viewWeights[view] == 0u)
			continue;
		SetBit(result.selectedSourceViews, static_cast<unsigned>(view));
		bool directionValid(false);
		DVPGlobalCandidate candidate(DVPFindBestGlobalCandidateInView(
			depthImages, pixel, depth, view, g_params.nDVPGlobalSearchRadius,
			directionValid));
		if (directionValid)
			SetBit(result.directionSourceViews, static_cast<unsigned>(view));
		if (!candidate.valid)
			continue;
		anyFinite = true;
		if (family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0)) {
			if (!winner.valid || candidate.reprojectionError < winner.reprojectionError ||
				(candidate.reprojectionError == winner.reprojectionError &&
				candidate.sourceView < winner.sourceView))
				winner = candidate;
			continue;
		}
		if (candidate.reprojectionError <= g_params.fDVPReprojectionThreshold)
			anyReprojectionPass = true;
		candidate = DVPValidateGlobalCandidate(depthImages, pixel, viewWeights, candidate);
		anyOccluded = anyOccluded || candidate.occludedViews != 0u;
		if (DVPPreferGatedGlobalCandidate(candidate, winner))
			winner = candidate;
	}
	if (!winner.valid) {
		result.reason = !anyFinite ? DVPEpipolarUnavailableReason::NO_FINITE_GLOBAL_CANDIDATE :
			!anyReprojectionPass ? DVPEpipolarUnavailableReason::GLOBAL_REPROJECTION_GATE :
			anyOccluded ? DVPEpipolarUnavailableReason::GLOBAL_OCCLUSION_REJECTED :
			DVPEpipolarUnavailableReason::GLOBAL_SUPPORT_GATE;
		return result;
	}
	result.depths[0] = winner.depth;
	result.signedOffsets[0] = winner.signedOffset;
	result.sourceViews[0] = winner.sourceView;
	result.supports[0] = family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) ?
		1u : winner.support;
	result.supportViews[0] = family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) ?
		(1u << winner.sourceView) : winner.supportViews;
	result.occludedViews[0] = winner.occludedViews;
	result.meanReprojectionErrors[0] = family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) ?
		winner.reprojectionError : winner.meanReprojectionError;
	result.maxReprojectionErrors[0] = family == static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0) ?
		winner.reprojectionError : winner.maxReprojectionError;
	result.meanRelativeDepthErrors[0] = winner.meanRelativeDepthError;
	result.maxRelativeDepthErrors[0] = winner.maxRelativeDepthError;
	result.count = 1u;
	return result;
}

template <bool GEOM>
__device__ __forceinline__ DVPProposalSet DVPCollectProposals(
	const ImagePixels* depthImages, const Point2i& pixel, float depth,
	const unsigned* viewWeights, unsigned family)
{
	if (!DVPEpipolarFamilyEnabled(family)) {
		DVPProposalSet result;
		result.reason = DVPEpipolarUnavailableReason::FAMILY_DISABLED;
		return result;
	}
	if (DVPFamilyUsesGlobalSearch(family))
		return DVPCollectGlobalProposal<GEOM>(
			depthImages, pixel, depth, viewWeights, family);
	return DVPCollectEndpointProposals<GEOM>(
		depthImages, pixel, depth, viewWeights, family);
}

#ifdef _USE_DMAP_INSTRUMENTATION
__device__ inline double InstrumentAtomicAdd(double* address, double value)
{
#if __CUDA_ARCH__ >= 600
	return atomicAdd(address, value);
#else
	unsigned long long* addressAsULL(reinterpret_cast<unsigned long long*>(address));
	unsigned long long old(*addressAsULL);
	unsigned long long assumed;
	do {
		assumed = old;
		old = atomicCAS(addressAsULL, assumed,
			__double_as_longlong(value+__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
#endif
}

__device__ inline void InitializeDVPInstrumentUpdate(
	const DVPProposalSet& proposals,
	float incumbentDepth,
	float incumbentCost,
	const Point2i& pixel,
	int logicalIteration,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	PatchMatchDVPInstrumentUpdate& update,
	PatchMatchDVPInstrumentTrace* trace)
{
	update.incumbentDepth = incumbentDepth;
	update.incumbentCost = incumbentCost;
	update.leftIntervalMinimum = proposals.intervals.left.minimum;
	update.leftIntervalMaximum = proposals.intervals.left.maximum;
	update.rightIntervalMinimum = proposals.intervals.right.minimum;
	update.rightIntervalMaximum = proposals.intervals.right.maximum;
	update.selectedSourceViews = proposals.selectedSourceViews;
	update.directionSourceViews = proposals.directionSourceViews;
	update.leftOuterCount = proposals.endpointSamples.leftOuterCount;
	update.leftInnerCount = proposals.endpointSamples.leftInnerCount;
	update.rightInnerCount = proposals.endpointSamples.rightInnerCount;
	update.rightOuterCount = proposals.endpointSamples.rightOuterCount;
	update.family = static_cast<uint8_t>(g_params.nDVPEpipolarFamily);
	update.unavailableReason = static_cast<uint8_t>(proposals.reason);
	update.generatedCount = proposals.count;
	update.nativeDepthFallback = proposals.nativeDepthFallback ? 1u : 0u;
	update.leftIntervalValid = proposals.intervals.left.valid ? 1u : 0u;
	update.rightIntervalValid = proposals.intervals.right.valid ? 1u : 0u;
	for (unsigned proposal=0u; proposal<DVP_MAX_PROPOSALS; ++proposal) {
		update.proposalDepth[proposal] = proposals.depths[proposal];
		update.signedOffset[proposal] = proposals.signedOffsets[proposal];
		update.sourceView[proposal] = proposals.sourceViews[proposal];
		update.support[proposal] = proposals.supports[proposal];
		update.supportViews[proposal] = proposals.supportViews[proposal];
		update.occludedViews[proposal] = proposals.occludedViews[proposal];
		update.meanReprojectionError[proposal] =
			proposals.meanReprojectionErrors[proposal];
		update.maxReprojectionError[proposal] =
			proposals.maxReprojectionErrors[proposal];
		update.meanRelativeDepthError[proposal] =
			proposals.meanRelativeDepthErrors[proposal];
		update.maxRelativeDepthError[proposal] =
			proposals.maxRelativeDepthErrors[proposal];
	}
	if (!trace)
		return;
	trace->valid = 1;
	trace->imageID = instrumentParams.imageID;
	trace->scaleNumber = instrumentParams.scaleNumber;
	trace->logicalIteration = logicalIteration;
	trace->x = pixel.x();
	trace->y = pixel.y();
	trace->leftOuterViews = proposals.endpointSamples.leftOuterViews;
	trace->leftInnerViews = proposals.endpointSamples.leftInnerViews;
	trace->rightInnerViews = proposals.endpointSamples.rightInnerViews;
	trace->rightOuterViews = proposals.endpointSamples.rightOuterViews;
	for (unsigned sample=0u; sample<DVP_MAX_SOURCE_VIEWS; ++sample) {
		trace->leftOuter[sample] = proposals.endpointSamples.leftOuter[sample];
		trace->leftInner[sample] = proposals.endpointSamples.leftInner[sample];
		trace->rightInner[sample] = proposals.endpointSamples.rightInner[sample];
		trace->rightOuter[sample] = proposals.endpointSamples.rightOuter[sample];
	}
}

__device__ inline void FinalizeDVPInstrumentation(
	float finalDepth,
	int updateSource,
	int pixelIndex,
	int logicalIteration,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	PatchMatchDVPInstrumentUpdate& update,
	PatchMatchDVPInstrumentTrace* trace)
{
	const DVPProposalDecision decision(ResolveDVPProposalDecision(
		update.incumbentCost, update.candidateCost, update.generatedCount));
	update.winnerCost = decision.winnerCost;
	update.runnerUpCost = decision.runnerUpCost;
	update.winnerRunnerUpGap = decision.winnerRunnerUpGap;
	update.winnerOrdinal = decision.winnerOrdinal;
	update.accepted = decision.accepted ? 1u : 0u;
	update.finalWinner = updateSource == PM_SOURCE_DVP_EPIPOLAR ? 1u : 0u;
	update.finalUpdateSource = updateSource >= 0 &&
		updateSource < PM_DVP_INSTRUMENT_FINAL_SOURCES ?
		static_cast<uint8_t>(updateSource) : static_cast<uint8_t>(PM_SOURCE_NONE);
	update.finalDepth = finalDepth;
	update.finalDepthRetained = update.accepted &&
		update.winnerOrdinal < update.generatedCount &&
		finalDepth == update.proposalDepth[update.winnerOrdinal] ? 1u : 0u;
	update.depthDisplacement = update.incumbentDepth > 0.f && finalDepth > 0.f ?
		fabsf(finalDepth-update.incumbentDepth) : -1.f;
	if (instrumentParams.dvpUpdates && logicalIteration >= 0 &&
		logicalIteration < instrumentParams.numLogicalStates-1 &&
		instrumentParams.area > 0)
	{
		instrumentParams.dvpUpdates[
			(size_t)logicalIteration*instrumentParams.area+pixelIndex] = update;
	}
	if (trace)
		trace->update = update;
	if (!instrumentParams.dvpCounters || logicalIteration < 0 ||
		logicalIteration >= instrumentParams.numLogicalStates-1)
	{
		return;
	}
	PatchMatchDVPInstrumentCounters& counter(
		instrumentParams.dvpCounters[logicalIteration]);
	atomicAdd(&counter.attemptedPixels, 1u);
	if (update.generatedCount > 0u)
		atomicAdd(&counter.proposalAvailablePixels, 1u);
	if (update.nativeDepthFallback)
		atomicAdd(&counter.nativeDepthFallbackPixels, 1u);
	atomicAdd(&counter.proposalsGenerated, static_cast<uint32_t>(update.generatedCount));
	atomicAdd(&counter.proposalsTested, static_cast<uint32_t>(update.testedCount));
	atomicAdd(&counter.proposalsFinite, static_cast<uint32_t>(update.finiteCount));
	atomicAdd(&counter.proposalsAccepted, static_cast<uint32_t>(update.acceptedCount));
	if (update.finalWinner)
		atomicAdd(&counter.finalWinnerPixels, 1u);
	if (update.finalDepthRetained)
		atomicAdd(&counter.finalDepthRetainedPixels, 1u);
	atomicAdd(&counter.finalUpdateSource[update.finalUpdateSource], 1u);
	if (update.leftIntervalValid)
		atomicAdd(&counter.leftIntervalValidPixels, 1u);
	if (update.rightIntervalValid)
		atomicAdd(&counter.rightIntervalValidPixels, 1u);
	atomicAdd(&counter.selectedSourceViewCountSum,
		static_cast<uint32_t>(__popc(update.selectedSourceViews)));
	atomicAdd(&counter.directionSourceViewCountSum,
		static_cast<uint32_t>(__popc(update.directionSourceViews)));
	atomicAdd(&counter.endpointSupportSum[0], static_cast<uint32_t>(update.leftOuterCount));
	atomicAdd(&counter.endpointSupportSum[1], static_cast<uint32_t>(update.leftInnerCount));
	atomicAdd(&counter.endpointSupportSum[2], static_cast<uint32_t>(update.rightInnerCount));
	atomicAdd(&counter.endpointSupportSum[3], static_cast<uint32_t>(update.rightOuterCount));
	if (update.unavailableReason < PM_DVP_INSTRUMENT_UNAVAILABLE_REASONS)
		atomicAdd(&counter.unavailableReason[update.unavailableReason], 1u);
	if (update.family < PM_DVP_INSTRUMENT_FAMILIES)
		atomicAdd(&counter.family[update.family], 1u);
	for (unsigned proposal=0u; proposal<update.generatedCount; ++proposal) {
		atomicAdd(&counter.proposalSupportSum,
			static_cast<uint32_t>(update.support[proposal]));
		atomicAdd(&counter.proposalSupportSamples, 1u);
		if (update.occludedViews[proposal])
			atomicAdd(&counter.occludedCandidates, 1u);
		if (update.meanReprojectionError[proposal] >= 0.f) {
			InstrumentAtomicAdd(&counter.reprojectionErrorSum,
				update.meanReprojectionError[proposal]);
			atomicAdd(&counter.reprojectionErrorSamples, 1u);
		}
		if (update.meanRelativeDepthError[proposal] >= 0.f) {
			InstrumentAtomicAdd(&counter.relativeDepthErrorSum,
				update.meanRelativeDepthError[proposal]);
			atomicAdd(&counter.relativeDepthErrorSamples, 1u);
		}
	}
	if (DVPValidProposalCost(update.incumbentCost) &&
		DVPValidProposalCost(update.winnerCost))
	{
		InstrumentAtomicAdd(&counter.incumbentCostSum, update.incumbentCost);
		InstrumentAtomicAdd(&counter.winnerCostSum, update.winnerCost);
		InstrumentAtomicAdd(&counter.improvementSum, update.incumbentCost-update.winnerCost);
		atomicAdd(&counter.costSamples, 1u);
	}
	if (update.winnerRunnerUpGap >= 0.f) {
		InstrumentAtomicAdd(&counter.gapSum, update.winnerRunnerUpGap);
		atomicAdd(&counter.gapSamples, 1u);
	}
	if (update.depthDisplacement >= 0.f) {
		InstrumentAtomicAdd(&counter.displacementSum, update.depthDisplacement);
		atomicAdd(&counter.displacementSamples, 1u);
	}
}
#endif

#ifdef _USE_DMAP_INSTRUMENTATION
template <bool GEOM>
__device__ inline PlaneScoreComponents InstrumentScorePlaneComponents(
	const RefPatchCache& cache, const ImagePixels* images, const ImagePixels* depthImages,
	const Point2i& p, const Point4& plane, const float lowDepth, const int imgId)
{
	PlaneScoreComponents components(ScorePlaneComponents(
		cache, g_cameras[0], images[imgId+1], g_cameras[imgId+1], p, plane, lowDepth));
	// Use the production scorer for the value entering candidate comparison.
	// The richer scorer above only supplies the prior-free/prior diagnostics.
	components.photoPriorCost = ScorePlane(
		cache, g_cameras[0], images[imgId+1], g_cameras[imgId+1], p, plane, lowDepth);
	components.totalCost = components.photoPriorCost;
	if (GEOM) {
		const float geometricWeight = GeometricConsistencyWeight(
			depthImages[imgId], g_cameras[0], g_cameras[imgId+1], plane, p);
		components.geometricCost = 0.1f * geometricWeight;
		components.totalCost += components.geometricCost;
		if (geometricWeight >= 1.2f && components.badReason == PM_BAD_NONE)
			components.badReason = PM_BAD_GEOMETRIC_MISMATCH;
	}
	return components;
}

__device__ inline void AccumulateInstrumentCostComponents(
	PatchMatchInstrumentCostComponents& aggregate,
	const PlaneScoreComponents& components,
	const unsigned weight,
	const int imgId)
{
	if (!weight)
		return;
	const float w((float)weight);
	aggregate.photometricCost += w * components.photometricCost;
	aggregate.photoPriorCost += w * components.photoPriorCost;
	aggregate.depthPriorCost += w * components.depthPriorCost;
	aggregate.depthPriorWeight += w * components.depthPriorWeight;
	aggregate.geometricCost += w * components.geometricCost;
	aggregate.totalCost += w * components.totalCost;
	aggregate.sampleCount += weight;
	aggregate.depthPriorSamples += weight * components.depthPriorUsed;
	if (components.badReason != PM_BAD_NONE && components.badReason < PM_INSTRUMENT_NUM_BAD_REASONS)
		aggregate.badReason[components.badReason] += weight;
	else if (components.totalCost >= fBadCost)
		aggregate.badReason[PM_BAD_NONE] += weight;
	if (imgId < PM_INSTRUMENT_MAP_VIEWS) {
		aggregate.viewWeightSum[imgId] += w;
		if (components.totalCost >= fBadCost || components.badReason != PM_BAD_NONE)
			aggregate.viewBadCost[imgId] += w;
		aggregate.viewCostWeightedSum[imgId] += w * components.totalCost;
		aggregate.viewPhotometricCostWeightedSum[imgId] += w * components.photometricCost;
		aggregate.viewGeometricCostWeightedSum[imgId] += w * components.geometricCost;
	}
}

__device__ inline void NormalizeInstrumentCostComponents(
	PatchMatchInstrumentCostComponents& aggregate, const float denominator)
{
	if (denominator <= 0.f)
		return;
	const float invDenom(1.f / denominator);
	aggregate.photometricCost *= invDenom;
	aggregate.photoPriorCost *= invDenom;
	aggregate.depthPriorCost *= invDenom;
	aggregate.depthPriorWeight *= invDenom;
	aggregate.geometricCost *= invDenom;
	aggregate.totalCost *= invDenom;
}

template <bool GEOM>
__device__ inline PatchMatchInstrumentCostComponents AggregateCostComponents(
	const RefPatchCache& cache, const ImagePixels* images, const ImagePixels* depthImages,
	const Point2i& p, const Point4& plane, const float lowDepth,
	const unsigned* viewWeights, int numViews, float denominator)
{
	PatchMatchInstrumentCostComponents aggregate;
	if (denominator <= 0.f)
		return aggregate;
	for (int imgId = 0; imgId < numViews; ++imgId) {
		const unsigned weight(viewWeights[imgId]);
		if (!weight)
			continue;
		const PlaneScoreComponents components(InstrumentScorePlaneComponents<GEOM>(
			cache, images, depthImages, p, plane, lowDepth, imgId));
		AccumulateInstrumentCostComponents(aggregate, components, weight, imgId);
	}
	NormalizeInstrumentCostComponents(aggregate, denominator);
	return aggregate;
}

__device__ __forceinline__ float ExactCDFProbability(
	const float* cumulativeProbabilities, const int view, const int numViews)
{
	if (view < 0 || view >= numViews || !isfinite(cumulativeProbabilities[view]))
		return 0.f;
	float lower = 0.f;
	for (int i = 0; i < view; ++i) {
		const float value(cumulativeProbabilities[i]);
		if (isfinite(value))
			lower = max(lower, min(1.f, max(0.f, value)));
	}
	const float upper(min(1.f, max(0.f, cumulativeProbabilities[view])));
	return max(0.f, upper-lower);
}

__device__ __forceinline__ unsigned ExactProbabilityRank(
	const float* cumulativeProbabilities, const int view, const int numViews)
{
	const float probability(ExactCDFProbability(cumulativeProbabilities, view, numViews));
	unsigned rank = 0;
	for (int other = 0; other < numViews; ++other) {
		if (other == view)
			continue;
		const float otherProbability(ExactCDFProbability(cumulativeProbabilities, other, numViews));
		if (otherProbability > probability || (otherProbability == probability && other < view))
			++rank;
	}
	return rank;
}

__device__ __forceinline__ unsigned ExactCostRank(
	const float* viewCosts, const int view, const int numViews)
{
	const float cost(viewCosts[view]);
	unsigned rank = 0;
	for (int other = 0; other < numViews; ++other) {
		if (other == view)
			continue;
		const float otherCost(viewCosts[other]);
		if (otherCost < cost || (otherCost == cost && other < view))
			++rank;
	}
	return rank;
}

__device__ __forceinline__ float ExactSamplingScore(
	const float selectionPrior,
	const float* neighborViewCosts,
	const unsigned validNeighbors,
	const int view,
	const int neighborViewStride,
	const float costThreshold,
	unsigned& agreeCount,
	unsigned& badCount)
{
	float sumWeight = 0.f;
	agreeCount = 0;
	badCount = 0;
	for (int posId = 0; posId < 8; ++posId) {
		if (!IsBitSet(validNeighbors, posId))
			continue;
		const float cost(neighborViewCosts[posId * neighborViewStride + view]);
		if (cost < costThreshold) {
			sumWeight += __expf(Square(cost) / (-2.f * 0.3f*0.3f));
			++agreeCount;
		} else if (cost >= fBadCost) {
			++badCount;
		}
	}
	if (agreeCount > 2 && badCount < 3)
		return selectionPrior * sumWeight / (float)agreeCount;
	if (badCount < 3)
		return selectionPrior * __expf(Square(costThreshold) / (-2.f * 0.4f*0.4f));
	return 0.f;
}

template <bool GEOM>
__device__ PatchMatchInstrumentCostComponents WriteExactIterationViews(
	const RefPatchCache& cache,
	const ImagePixels* images,
	const ImagePixels* depthImages,
	const Point2i& p,
	const Point4& plane,
	const float lowDepth,
	const float* neighborViewCosts,
	const unsigned validNeighbors,
	const float* selectionPriors,
	const float* cumulativeProbabilities,
	const unsigned* viewWeights,
	const float* productionViewCosts,
	const float costThreshold,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const int pixelIndex)
{
	PatchMatchInstrumentCostComponents aggregate;
	const int numViews(g_params.nNumViews);
	const bool writeViews(
		instrumentParams.exactViews && instrumentParams.exactLogicalStateIndex >= 0 &&
		instrumentParams.exactLogicalStateIndex < instrumentParams.numLogicalStates &&
		instrumentParams.viewStride >= numViews && instrumentParams.area > 0);
	const size_t viewBase = writeViews ?
		((size_t)instrumentParams.exactLogicalStateIndex * instrumentParams.area + pixelIndex) *
			(size_t)instrumentParams.viewStride : 0;
	unsigned totalViewWeight(0u);
	for (int view=0; view<numViews; ++view)
		totalViewWeight += viewWeights[view];
	const float contributionDenominator(
		totalViewWeight > 0u ? static_cast<float>(totalViewWeight) : 1.f);
	for (int view = 0; view < numViews; ++view) {
		const unsigned weight(viewWeights[view]);
		if (!writeViews && !weight)
			continue;
		PlaneScoreComponents components(InstrumentScorePlaneComponents<GEOM>(
			cache, images, depthImages, p, plane, lowDepth, view));
		// Preserve the values used by the accepted production event; diagnostic
		// rescoring is intentionally not the source of an "exact" contribution.
		components.totalCost = productionViewCosts[view];
		components.photoPriorCost = components.totalCost - components.geometricCost;
		AccumulateInstrumentCostComponents(aggregate, components, weight, view);
		if (!writeViews)
			continue;
		unsigned agreeCount = 0;
		unsigned badCount = 0;
		const float samplingScore(ExactSamplingScore(
			selectionPriors[view], neighborViewCosts, validNeighbors, view,
			MAX_VIEWS, costThreshold, agreeCount, badCount));
		const float probability(ExactCDFProbability(cumulativeProbabilities, view, numViews));
		const bool selected(weight > 0);
		const int decision = selected ? PM_EXACT_VIEW_SELECTED_MC :
			(probability <= 0.f ? PM_EXACT_VIEW_REJECTED_ZERO_SCORE : PM_EXACT_VIEW_REJECTED_NOT_SAMPLED);
		PatchMatchInstrumentExactView& record(instrumentParams.exactViews[viewBase + (size_t)view]);
		record.photometricCost = components.photoPriorCost;
		record.geometricCost = components.geometricCost;
		record.totalCost = components.totalCost;
		record.weightedContribution = (float)weight * components.totalCost /
			contributionDenominator;
		record.selectionPrior = selectionPriors[view];
		record.samplingScore = samplingScore;
		record.samplingProbability = probability;
		record.metadata = PackExactViewMetadata(
			weight, ExactProbabilityRank(cumulativeProbabilities, view, numViews),
			agreeCount, badCount, decision, selected, selected,
			IsFiniteCandidateCost(components.totalCost), true);
	}
	NormalizeInstrumentCostComponents(aggregate, contributionDenominator);
	return aggregate;
}

template <bool GEOM>
__device__ PatchMatchInstrumentCostComponents WriteExactInitializationViews(
	const RefPatchCache& cache,
	const ImagePixels* images,
	const ImagePixels* depthImages,
	const Point2i& p,
	const Point4& plane,
	const float lowDepth,
	const float* viewCosts,
	const unsigned selectedViews,
	const int topK,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const int pixelIndex)
{
	PatchMatchInstrumentCostComponents aggregate;
	const int numViews(g_params.nNumViews);
	const bool writeViews(
		instrumentParams.exactViews && instrumentParams.exactLogicalStateIndex == 0 &&
		instrumentParams.numLogicalStates > 0 && instrumentParams.viewStride >= numViews &&
		instrumentParams.area > 0);
	const size_t viewBase = writeViews ? (size_t)pixelIndex * (size_t)instrumentParams.viewStride : 0;
	for (int view = 0; view < numViews; ++view) {
		const unsigned rank(ExactCostRank(viewCosts, view, numViews));
		const bool contributionBasis(rank < (unsigned)topK);
		if (!writeViews && !contributionBasis)
			continue;
		PlaneScoreComponents components(InstrumentScorePlaneComponents<GEOM>(
			cache, images, depthImages, p, plane, lowDepth, view));
		// Keep the exact value from the production initialization evaluation.
		components.totalCost = viewCosts[view];
		components.photoPriorCost = components.totalCost - components.geometricCost;
		AccumulateInstrumentCostComponents(aggregate, components, contributionBasis ? 1u : 0u, view);
		if (!writeViews)
			continue;
		const bool selected(IsBitSet(selectedViews, view));
		const int decision = contributionBasis ? PM_EXACT_VIEW_INIT_TOP_K :
			(selected ? PM_EXACT_VIEW_INIT_THRESHOLD_TIE : PM_EXACT_VIEW_INIT_REJECTED);
		PatchMatchInstrumentExactView& record(instrumentParams.exactViews[viewBase + (size_t)view]);
		record.photometricCost = components.photoPriorCost;
		record.geometricCost = components.geometricCost;
		record.totalCost = components.totalCost;
		record.weightedContribution = contributionBasis ? components.totalCost / (float)topK : 0.f;
		record.selectionPrior = -1.f;
		record.samplingScore = -1.f;
		record.samplingProbability = -1.f;
		record.metadata = PackExactViewMetadata(
			contributionBasis ? 1u : 0u, rank, 0u, 0u, decision, selected,
			contributionBasis, IsFiniteCandidateCost(components.totalCost), false);
	}
	NormalizeInstrumentCostComponents(aggregate, (float)topK);
	return aggregate;
}

#endif
// Per-pixel update for ACMH-style patch-match stereo on GPU; reference:
//   "Multi-View Stereo with Asymmetric Checkerboard Propagation and
//    Multi-Hypothesis Joint View Selection", Xu & Tao, 2018.
//
// Each call performs (for a single pixel):
//   1. Adaptive neighbor sampling - 8 directional patterns (4 near + 4 far);
//      pick the best plane in each direction and score it against all views
//      into costArray[posId][imgId].
//   2. Multi-hypothesis joint view selection:
//        - Build viewSelectionPriors[j] from neighbors' selectedViews bitmasks.
//        - For each view, count agreeing/disagreeing neighbor planes and form
//          samplingProbs[imgId] = prior * Gaussian-weighted local agreement.
//        - PDF2CDF normalizes; NUM_SAMPLES Monte-Carlo draws populate
//          viewWeights[imgId] (= count of times imgId was sampled).
//   3. Plane comparison - aggregate each of the 8 neighbor planes + the
//      current plane against the shared viewWeights; pick the lowest.
//   4. Plane refinement - perturb depth/normal, re-score against the same
//      viewWeights, keep if it lowers the aggregate cost.
//
// The shared viewWeights basis across (3) and (4) ensures plane hypotheses
// are evaluated on a consistent view-selection footing within this pixel.
#ifdef _USE_DMAP_INSTRUMENTATION
__device__ inline bool DVPVisibilityRestoredReason(uint8_t reason)
{
	return reason == static_cast<uint8_t>(DVPVisibilityReason::RESTORED_PAPER_2D) ||
		reason == static_cast<uint8_t>(DVPVisibilityReason::RESTORED_DEPTH_GATED);
}

__device__ inline void CaptureDVPVisibilityInstrumentation(
	const Point2i& pixel,
	int index,
	int logicalIteration,
	const uint8_t* previousWeights,
	const uint8_t* resolvedWeights,
	const uint8_t* nextWeights,
	const unsigned* activeSupportWeights,
	uint32_t candidateTestedMask,
	uint32_t candidateFiniteMask,
	const int32_t* traceMap,
	const PatchMatchInstrumentKernelParams& params)
{
	if (logicalIteration < 0 || logicalIteration >= params.numLogicalStates-1 ||
		!previousWeights || !resolvedWeights || !nextWeights ||
		!activeSupportWeights || !params.visibilityReasons)
	{
		return;
	}
	const int numViews(g_params.nNumViews);
	const size_t viewOffset((size_t)index*MAX_VIEWS);
	PatchMatchDVPVisibilityInstrumentUpdate update;
	update.mode = static_cast<uint8_t>(g_params.nDVPVisibilityMode);
	bool activeMatchesResolved(true);
	for (int view=0; view<numViews; ++view) {
		const uint8_t previous(previousWeights[viewOffset+view]);
		const uint8_t resolved(resolvedWeights[viewOffset+view]);
		const uint8_t next(nextWeights[viewOffset+view]);
		const uint8_t active(static_cast<uint8_t>(min(activeSupportWeights[view], 255u)));
		const uint8_t reason(params.visibilityReasons[viewOffset+view]);
		if (previous) SetBit(update.previousMask, view);
		if (resolved) SetBit(update.resolvedMask, view);
		if (next) SetBit(update.nextMask, view);
		if (active) SetBit(update.activeSupportMask, view);
		update.previousWeightSum = static_cast<uint16_t>(
			min(65535u, static_cast<unsigned>(update.previousWeightSum)+previous));
		update.resolvedWeightSum = static_cast<uint16_t>(
			min(65535u, static_cast<unsigned>(update.resolvedWeightSum)+resolved));
		update.nextWeightSum = static_cast<uint16_t>(
			min(65535u, static_cast<unsigned>(update.nextWeightSum)+next));
		update.activeSupportWeightSum = static_cast<uint16_t>(
			min(65535u, static_cast<unsigned>(update.activeSupportWeightSum)+active));
		update.previousVisibleCount += previous > 0u ? 1u : 0u;
		update.resolvedVisibleCount += resolved > 0u ? 1u : 0u;
		update.nextVisibleCount += next > 0u ? 1u : 0u;
		update.activeSupportCount += active > 0u ? 1u : 0u;
		if (reason < PM_DVP_VISIBILITY_INSTRUMENT_REASONS)
			++update.reasonCount[reason];
		if (DVPVisibilityRestoredReason(reason)) {
			SetBit(update.restoredMask, view);
			++update.restoredCount;
		} else if (!previous && !resolved &&
			reason != static_cast<uint8_t>(DVPVisibilityReason::MODE_DISABLED))
		{
			SetBit(update.rejectedMask, view);
			++update.rejectedCount;
		}
		activeMatchesResolved = activeMatchesResolved && active == resolved;
	}
	update.addedMask = update.nextMask & ~update.previousMask;
	update.removedMask = update.previousMask & ~update.nextMask;
	update.addedCount = static_cast<uint8_t>(__popc(update.addedMask));
	update.removedCount = static_cast<uint8_t>(__popc(update.removedMask));
	update.denominator = update.activeSupportWeightSum;
	update.denominatorDefined = update.denominator > 0u ? 1u : 0u;
	update.activeSupportMatchesResolved = activeMatchesResolved ? 1u : 0u;
	update.candidateTestedMask = candidateTestedMask;
	update.candidateFiniteMask = candidateFiniteMask;
	const DVPVisibilityStateHeader previousHeader{
		DVP_VISIBILITY_STATE_VERSION,
		static_cast<uint32_t>(g_cameras[0].size.x()),
		static_cast<uint32_t>(g_cameras[0].size.y()),
		static_cast<uint32_t>(numViews),
		static_cast<uint32_t>(logicalIteration)};
	DVPVisibilityStateHeader nextHeader(previousHeader);
	nextHeader.logicalIteration = static_cast<uint32_t>(logicalIteration+1);
	update.transitionStatus = static_cast<uint8_t>(
		ValidateDVPVisibilityTransition(previousHeader, nextHeader));

	if (params.visibilityUpdates)
		params.visibilityUpdates[(size_t)logicalIteration*params.area+index] = update;
	const int traceIndex(traceMap && params.numTracePixels > 0 ? traceMap[index] : -1);
	if (params.visibilityTraces && traceIndex >= 0 && traceIndex < params.numTracePixels) {
		PatchMatchDVPVisibilityInstrumentTrace& trace(
			params.visibilityTraces[(size_t)logicalIteration*params.numTracePixels+traceIndex]);
		trace.valid = 1;
		trace.imageID = params.imageID;
		trace.scaleNumber = params.scaleNumber;
		trace.logicalIteration = logicalIteration;
		trace.x = pixel.x();
		trace.y = pixel.y();
		trace.update = update;
		for (int view=0; view<MAX_VIEWS; ++view) {
			trace.previousWeights[view] = view < numViews ? previousWeights[viewOffset+view] : 0u;
			trace.resolvedWeights[view] = view < numViews ? resolvedWeights[viewOffset+view] : 0u;
			trace.nextWeights[view] = view < numViews ? nextWeights[viewOffset+view] : 0u;
			trace.activeSupportWeights[view] = view < numViews ?
				static_cast<uint8_t>(min(activeSupportWeights[view], 255u)) : 0u;
			trace.reasons[view] = view < numViews ? params.visibilityReasons[viewOffset+view] :
				static_cast<uint8_t>(DVPVisibilityReason::MODE_DISABLED);
		}
	}
	if (params.visibilityCounters) {
		PatchMatchDVPVisibilityInstrumentCounters& counter(
			params.visibilityCounters[logicalIteration]);
		atomicAdd(&counter.pixels, 1u);
		for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason)
			atomicAdd(&counter.reason[reason], static_cast<uint32_t>(update.reasonCount[reason]));
		atomicAdd(&counter.previousVisibleViews, static_cast<uint32_t>(update.previousVisibleCount));
		atomicAdd(&counter.resolvedVisibleViews, static_cast<uint32_t>(update.resolvedVisibleCount));
		atomicAdd(&counter.nextVisibleViews, static_cast<uint32_t>(update.nextVisibleCount));
		atomicAdd(&counter.activeSupportViews, static_cast<uint32_t>(update.activeSupportCount));
		atomicAdd(&counter.restoredViews, static_cast<uint32_t>(update.restoredCount));
		atomicAdd(&counter.rejectedViews, static_cast<uint32_t>(update.rejectedCount));
		atomicAdd(&counter.addedViews, static_cast<uint32_t>(update.addedCount));
		atomicAdd(&counter.removedViews, static_cast<uint32_t>(update.removedCount));
		atomicAdd(reinterpret_cast<unsigned long long*>(&counter.previousWeightSum),
			static_cast<unsigned long long>(update.previousWeightSum));
		atomicAdd(reinterpret_cast<unsigned long long*>(&counter.resolvedWeightSum),
			static_cast<unsigned long long>(update.resolvedWeightSum));
		atomicAdd(reinterpret_cast<unsigned long long*>(&counter.nextWeightSum),
			static_cast<unsigned long long>(update.nextWeightSum));
		atomicAdd(reinterpret_cast<unsigned long long*>(&counter.activeSupportWeightSum),
			static_cast<unsigned long long>(update.activeSupportWeightSum));
		atomicAdd(reinterpret_cast<unsigned long long*>(&counter.denominatorSum),
			static_cast<unsigned long long>(update.denominator));
		if (!update.denominatorDefined)
			atomicAdd(&counter.zeroDenominatorPixels, 1u);
		if (!update.activeSupportMatchesResolved)
			atomicAdd(&counter.supportMismatchPixels, 1u);
		if (update.addedMask || update.removedMask)
			atomicAdd(&counter.changedPixels, 1u);
		if (update.transitionStatus != static_cast<uint8_t>(DVPVisibilityTransitionStatus::VALID))
			atomicAdd(&counter.invalidTransitionPixels, 1u);
		if (update.candidateTestedMask)
			atomicAdd(&counter.candidateTestedPixels, 1u);
		if (update.candidateFiniteMask)
			atomicAdd(&counter.candidateFinitePixels, 1u);
		atomicAdd(&counter.candidateTestedCount,
			static_cast<uint32_t>(__popc(update.candidateTestedMask)));
		atomicAdd(&counter.candidateFiniteCount,
			static_cast<uint32_t>(__popc(update.candidateFiniteMask)));
	}
}

#ifdef _USE_DMAP_INSTRUMENTATION
__device__ inline void FinalizeDVPVisibleNormalInstrumentation(
	const Point2i& pixel,
	int index,
	int logicalIteration,
	const PatchMatchInstrumentKernelParams& params,
	const PatchMatchDVPVisibleNormalInstrumentUpdate& update,
	const uint8_t* propagationEvaluationReasons,
	PatchMatchDVPVisibleNormalInstrumentTrace* trace)
{
	if (logicalIteration < 0)
		return;
	if (params.visibleNormalUpdates)
		params.visibleNormalUpdates[(size_t)logicalIteration*params.area+index] = update;
	if (trace) {
		trace->valid = 1;
		trace->imageID = params.imageID;
		trace->scaleNumber = params.scaleNumber;
		trace->logicalIteration = logicalIteration;
		trace->x = pixel.x();
		trace->y = pixel.y();
		trace->update = update;
	}
	if (!params.visibleNormalCounters)
		return;
	PatchMatchDVPVisibleNormalInstrumentCounters& counter(
		params.visibleNormalCounters[logicalIteration]);
	atomicAdd(&counter.pixels, 1u);
	atomicAdd(&counter.selectedSupportViews, static_cast<uint32_t>(update.supportCount));
	if (update.currentValid) {
		atomicAdd(&counter.currentValid, 1u);
		atomicAdd(update.currentFeasible ? &counter.currentFeasible : &counter.currentRejected, 1u);
	} else {
		atomicAdd(&counter.currentInvalid, 1u);
	}
	atomicAdd(&counter.propagationCandidatesTested,
		static_cast<uint32_t>(__popc(update.propagationTestedMask)));
	atomicAdd(&counter.propagationCandidatesValid,
		static_cast<uint32_t>(__popc(update.propagationValidMask)));
	atomicAdd(&counter.propagationCandidatesFeasible,
		static_cast<uint32_t>(__popc(update.propagationFeasibleMask)));
	atomicAdd(&counter.propagationCandidatesRejected,
		static_cast<uint32_t>(__popc(update.propagationRejectedMask)));
	if (update.propagationSelected != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE) {
		atomicAdd(update.propagationSelected == update.propagationConstrainedBest ?
			&counter.propagationConstrainedSelections : &counter.propagationNativeSelections, 1u);
	}
	if (update.propagationFallback)
		atomicAdd(&counter.propagationFallbacks, 1u);
	if (update.propagationAppliedConstraint)
		atomicAdd(&counter.propagationAppliedConstraints, 1u);
	if (update.propagationAccepted)
		atomicAdd(&counter.propagationAccepted, 1u);
	atomicAdd(&counter.refinementNativeTested,
		static_cast<uint32_t>(__popc(update.refinementNativeTestedMask)));
	atomicAdd(&counter.refinementNativeValid,
		static_cast<uint32_t>(__popc(update.refinementNativeValidMask)));
	atomicAdd(&counter.refinementNativeFeasible,
		static_cast<uint32_t>(__popc(update.refinementNativeFeasibleMask)));
	atomicAdd(&counter.refinementNativeRejected,
		static_cast<uint32_t>(__popc(update.refinementNativeTestedMask &
			~update.refinementNativeFeasibleMask)));
	atomicAdd(reinterpret_cast<unsigned long long*>(&counter.refinementRetriesTested),
		static_cast<unsigned long long>(update.retriesTested[0])+
		static_cast<unsigned long long>(update.retriesTested[1]));
	atomicAdd(&counter.refinementRetrySuccess,
		static_cast<uint32_t>(__popc(update.refinementRetrySuccessMask)));
	atomicAdd(&counter.refinementExhaustion,
		static_cast<uint32_t>(__popc(update.refinementExhaustionFallbackMask)));
	atomicAdd(&counter.refinementFallback,
		static_cast<uint32_t>(__popc(update.refinementExhaustionFallbackMask)));
	atomicAdd(&counter.refinementAppliedRetry,
		static_cast<uint32_t>(__popc(update.refinementAppliedRetryMask)));
	atomicAdd(&counter.refinementAccepted,
		static_cast<uint32_t>(__popc(update.refinementAcceptedMask)));
	if (update.currentReason < PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS)
		atomicAdd(&counter.evaluationReason[update.currentReason], 1u);
	for (int proposal=0; proposal<DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS; ++proposal) {
		if (!(update.refinementNativeTestedMask & (1u << proposal)))
			continue;
		if (update.nativeReason[proposal] < PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS)
			atomicAdd(&counter.evaluationReason[update.nativeReason[proposal]], 1u);
		if (update.proposalReason[proposal] < PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS)
			atomicAdd(&counter.proposalReason[update.proposalReason[proposal]], 1u);
	}
	if (update.propagationTestedMask &&
		update.propagationReason < PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS)
	{
		atomicAdd(&counter.propagationReason[update.propagationReason], 1u);
	}
	for (int candidate=0; candidate<DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES;
		++candidate)
	{
		if (!(update.propagationTestedMask & (1u << candidate)))
			continue;
		const uint8_t reason(propagationEvaluationReasons[candidate]);
		if (reason < PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS)
			atomicAdd(&counter.evaluationReason[reason], 1u);
	}
}
#endif

template <bool GEOM, bool INSTRUMENT, bool APD, bool DVP, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__device__ __forceinline__ void ProcessPixel(
	const ImagePixels* images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews,
	const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext, APDUpdateStage apdStage,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const Point2i& p, const int iter)
#else
template <bool GEOM, bool APD, bool DVP, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__device__ __forceinline__ void ProcessPixel(
	const ImagePixels* images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates,
	unsigned* selectedViews, const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext, APDUpdateStage apdStage,
	const Point2i& p, const int iter)
#endif
{
	const int width = g_cameras[0].size.x();
	const int height = g_cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	const APDReliabilityClass apdReliabilityClass(
		APD && apdReliability ? static_cast<APDReliabilityClass>(apdReliability[idx]) :
		APDReliabilityClass::UNKNOWN);
	if constexpr (APD)
		if (!APDShouldProcess(apdReliabilityClass, apdStage))
			return;
	const bool apdActive(
		APD && apdReliability && apdAnchors && apdAnchorCounts && apdPlanesSnapshot &&
		apdSelectedViewsSnapshot &&
		apdReliabilityClass == APDReliabilityClass::UNRELIABLE &&
		apdAnchorCounts[idx] >= APD_MIN_INLIERS);
	const bool apdFullActive(
		apdActive && APDModeUsesFullMechanics(g_params.nAPDMode));
	const uint32_t* pixelAPDAnchors(apdActive ? apdAnchors+(size_t)idx*APD_MAX_ANCHORS : nullptr);
	const uint8_t pixelAPDAnchorCount(apdActive ?
		min(apdAnchorCounts[idx], static_cast<uint8_t>(APD_MAX_ANCHORS)) : 0u);
	const uint8_t* pixelVisibilityResolved(
		VISIBILITY && visibilityResolved ? visibilityResolved+(size_t)idx*MAX_VIEWS : nullptr);
#ifdef _USE_DMAP_INSTRUMENTATION
	const Point4 instrumentPlaneBefore(planes[idx]);
	const float instrumentStoredCostBefore(costs[idx]);
	const unsigned instrumentSelectedViewsBefore(selectedViews[idx]);
	uint32_t exactTestedMask = 0;
	uint32_t exactFiniteMask = 0;
	uint32_t exactAcceptedMask = 0;
	uint8_t exactTestedCount = 0;
	uint8_t exactFiniteCount = 0;
	uint8_t exactAcceptedCount = 0;
	float exactBestCost = FLT_MAX;
	float exactRunnerUpCost = FLT_MAX;
	uint8_t exactBestSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t exactRunnerUpSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t exactWinnerSlot = PM_EXACT_CANDIDATE_CURRENT;
	float exactWinningViewCosts[MAX_VIEWS];
	APDAggregateScoreComponents exactWinningAPDComponents;
	APDAggregateScoreComponents exactPropagationAPDComponents[8];
	PatchMatchAPDInstrumentTrace* apdTrace(nullptr);
	PatchMatchDVPInstrumentUpdate dvpUpdate;
	PatchMatchDVPInstrumentTrace* dvpTrace(nullptr);
	PatchMatchDVPVisibleNormalInstrumentUpdate visibleNormalUpdate;
	uint8_t visibleNormalPropagationEvaluationReasons[
		DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
	PatchMatchDVPVisibleNormalInstrumentTrace* visibleNormalTrace(nullptr);
	if constexpr (INSTRUMENT && APD) {
		const int traceIndex(instrumentTraceMap && instrumentParams.numTracePixels > 0 ?
			instrumentTraceMap[idx] : -1);
		if (instrumentParams.apdTraces && traceIndex >= 0 &&
			traceIndex < instrumentParams.numTracePixels && iter >= 0)
		{
			apdTrace = &instrumentParams.apdTraces[
				(size_t)iter*instrumentParams.numTracePixels+traceIndex];
		}
	}
	if constexpr (INSTRUMENT && DVP) {
		const int traceIndex(instrumentTraceMap && instrumentParams.numTracePixels > 0 ?
			instrumentTraceMap[idx] : -1);
		if (instrumentParams.dvpTraces && traceIndex >= 0 &&
			traceIndex < instrumentParams.numTracePixels && iter >= 0)
		{
			dvpTrace = &instrumentParams.dvpTraces[
				(size_t)iter*instrumentParams.numTracePixels+traceIndex];
		}
	}
	if constexpr (INSTRUMENT && VISIBLE_NORMAL) {
		const int traceIndex(instrumentTraceMap && instrumentParams.numTracePixels > 0 ?
			instrumentTraceMap[idx] : -1);
		if (instrumentParams.visibleNormalTraces && traceIndex >= 0 &&
			traceIndex < instrumentParams.numTracePixels && iter >= 0)
		{
			visibleNormalTrace = &instrumentParams.visibleNormalTraces[
				(size_t)iter*instrumentParams.numTracePixels+traceIndex];
		}
		visibleNormalUpdate.mode = static_cast<uint8_t>(g_params.nDVPVisibleNormalMode);
	}
#endif
	RandState* randState = &randStates[idx];
	float lowDepth = 0;
	if (g_params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// reference-patch state is invariant across views and hypotheses; cache once
	RefPatchCache refCache;
	ComputeRefPatchCache(images[0], p, refCache);

	// adaptive sampling: 0 up-near, 1 down-near, 2 left-near, 3 right-near, 4 up-far, 5 down-far, 6 left-far, 7 right-far
	static constexpr int2 dirs[8][11] = {
		{{ 0,-1},{-1,-2},{ 1,-2},{-2,-3},{ 2,-3},{-3,-4},{ 3,-4}},
		{{ 0, 1},{-1, 2},{ 1, 2},{-2, 3},{ 2, 3},{-3, 4},{ 3, 4}},
		{{-1, 0},{-2,-1},{-2, 1},{-3,-2},{-3, 2},{-4,-3},{-4, 3}},
		{{ 1, 0},{ 2,-1},{ 2, 1},{ 3,-2},{ 3, 2},{ 4,-3},{ 4, 3}},
		{{0,-3},{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
		{{0, 3},{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
		{{-3,0},{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
		{{ 3,0},{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	};
	static constexpr int numDirs[8] = {7, 7, 7, 7, 11, 11, 11, 11};
	const int neighborPositions[4] = {
		idx - width,
		idx + width,
		idx - 1,
		idx + 1,
	};
	bool valid[8] = {false, false, false, false, false, false, false, false};
#ifdef _USE_DMAP_INSTRUMENTATION
	unsigned validNeighbors = 0;
#endif
	int positions[8];
	float neighborDepths[8];
	float costArray[8][MAX_VIEWS];
	uint32_t validAnchorMask = 0u;
	uint32_t anchorSelectedViews[APD_MAX_ANCHORS] = {};

	for (int posId=0; posId<8; ++posId) {
		if constexpr (APD) {
			if (apdFullActive) {
				if (posId >= pixelAPDAnchorCount)
					continue;
				const uint32_t anchorIndex(pixelAPDAnchors[posId]);
				if (anchorIndex >= static_cast<uint32_t>(width*height))
					continue;
				const Point2i anchorPoint(
					static_cast<int>(anchorIndex%static_cast<uint32_t>(width)),
					static_cast<int>(anchorIndex/static_cast<uint32_t>(width)));
				Point4 anchorPlane(LoadPlaneLDG(&apdPlanesSnapshot[anchorIndex]));
				const float anchorDepth(InterpolatePixel(
					g_cameras[0], p, anchorPoint, anchorPlane.w(),
					anchorPlane.topLeftCorner<3,1>()));
				if (!isfinite(anchorDepth) || anchorDepth < g_params.fDepthMin ||
					anchorDepth > g_params.fDepthMax)
					continue;
				anchorPlane.w() = anchorDepth;
				valid[posId] = true;
				positions[posId] = static_cast<int>(anchorIndex);
				neighborDepths[posId] = anchorDepth;
				if constexpr (VISIBILITY) {
					anchorSelectedViews[posId] = 0u;
					const uint8_t* anchorWeights(
						visibilityResolved+(size_t)anchorIndex*MAX_VIEWS);
					for (int view=0; view<g_params.nNumViews; ++view)
						if (anchorWeights[view])
							SetBit(anchorSelectedViews[posId], view);
				} else {
					anchorSelectedViews[posId] = apdSelectedViewsSnapshot[anchorIndex];
				}
				validAnchorMask |= 1u << posId;
#ifdef _USE_DMAP_INSTRUMENTATION
				SetBit(validNeighbors, posId);
				if constexpr (INSTRUMENT) {
					if (apdTrace) {
						apdTrace->anchorCandidateValid[posId] = 1u;
						apdTrace->anchorSelectedViews[posId] = anchorSelectedViews[posId];
					}
				}
#endif
				MultiViewScorePlaneAPD<GEOM, false, VISIBILITY>(
					images, depthImages, p, anchorPlane, pixelAPDAnchors,
					pixelAPDAnchorCount, costArray[posId], nullptr, visibilityResolved);
				continue;
			}
		}
		const int2* samples = dirs[posId];
		Point2i bestNx; float bestConf(FLT_MAX);
		for (int dirId=0; dirId<numDirs[posId]; ++dirId) {
			const int2& offset = samples[dirId];
			const Point2i np(p.x()+offset.x, p.y()+offset.y);
			if (!(np.x()>=0 && np.y()>=0 && np.x()<width && np.y()<height))
				continue;
			const int nidx = Point2Idx(np, width);
			const float nconf = costs[nidx];
			if (bestConf > nconf) {
				bestNx = np;
				bestConf = nconf;
			}
		}
		if (bestConf < FLT_MAX) {
			valid[posId] = true;
#ifdef _USE_DMAP_INSTRUMENTATION
			SetBit(validNeighbors, posId);
#endif
			positions[posId] = Point2Idx(bestNx, width);
			neighborDepths[posId] = MultiViewScoreNeighborPlane<GEOM>(refCache, images, depthImages, p, bestNx, LoadPlaneLDG(&planes[positions[posId]]), lowDepth, costArray[posId]);
		}
	}

	// multi-hypothesis view selection
	float viewSelectionPriors[MAX_VIEWS] = {};
	const int nNumViews = g_params.nNumViews;
	const int viewPriorCandidates(apdFullActive ? static_cast<int>(APD_MAX_ANCHORS) : 4);
	for (int posId = 0; posId < viewPriorCandidates; ++posId) {
		if (valid[posId]) {
			const unsigned selectedView(apdFullActive ? anchorSelectedViews[posId] :
				selectedViews[neighborPositions[posId]]);
			for (int j = 0; j < nNumViews; ++j)
				viewSelectionPriors[j] += (IsBitSet(selectedView, j) ? 0.9f : 0.1f);
		}
	}
	float samplingProbs[MAX_VIEWS] = {};
	const float thCost(apdFullActive ? APDViewCostThreshold(static_cast<unsigned>(iter)) :
		0.8f*__expf(Square((float)iter)/(-2.f*4.f*4.f)));
	for (int imgId = 0; imgId < nNumViews; ++imgId) {
		if (apdFullActive) {
			const APDAnchorViewEvidence evidence(ComputeAPDAnchorViewEvidence(
				&costArray[0][0], MAX_VIEWS, anchorSelectedViews, validAnchorMask,
				pixelAPDAnchorCount, static_cast<unsigned>(imgId), thCost));
			viewSelectionPriors[imgId] = evidence.prior;
			samplingProbs[imgId] = evidence.samplingScore;
			continue;
		}
		float sumW = 0;
		unsigned count = 0;
		unsigned countBad = 0;
		for (int posId = 0; posId < 8; posId++) {
			if (valid[posId]) {
				if (costArray[posId][imgId] < thCost) {
					sumW += __expf(Square(costArray[posId][imgId]) / (-2.f * 0.3f*0.3f));
					++count;
				// Exact fBadCost marks an unavailable evaluation and remains neutral here.
				} else if (costArray[posId][imgId] > fBadCost) {
					++countBad;
				}
			}
		}
		if (count > 2 && countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * sumW / count;
		} else if (countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * __expf(Square(thCost) / (-2.f * 0.4f*0.4f));
		} else {
			samplingProbs[imgId] = 0.f;
		}
	}
	unsigned sampledViewWeights[MAX_VIEWS] = {};
	APDViewSelectionMode apdViewSelectionMode(APDViewSelectionMode::NATIVE);
	float probabilityMass(0.f);
	for (int imgId=0; imgId<nNumViews; ++imgId)
		probabilityMass += samplingProbs[imgId];
#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT && APD) {
		if (apdTrace && apdFullActive) {
			for (int view=0; view<nNumViews; ++view) {
				apdTrace->viewSelectionPriors[view] = viewSelectionPriors[view];
				apdTrace->viewSamplingScores[view] = samplingProbs[view];
				apdTrace->viewSamplingProbabilities[view] =
					probabilityMass > 0.f && isfinite(probabilityMass) ?
					samplingProbs[view]/probabilityMass : -1.f;
			}
		}
	}
#endif
	if (!apdFullActive || (probabilityMass > 0.f && isfinite(probabilityMass))) {
		PDF2CDF(samplingProbs, nNumViews);
		for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
			const float randProb(apdFullActive ?
				min(curand_uniform(randState), 1.f-FLT_EPSILON) : curand_uniform(randState));
			for (int imgId = 0; imgId < nNumViews; ++imgId) {
				if (samplingProbs[imgId] > randProb) {
					++sampledViewWeights[imgId];
					break;
				}
			}
		}
		if (apdFullActive)
			apdViewSelectionMode = APDViewSelectionMode::ANCHOR_EVIDENCE;
	} else if constexpr (APD) {
		const uint8_t* previousWeights(VISIBILITY && visibilityPrevious ?
			visibilityPrevious+(size_t)idx*MAX_VIEWS :
			(apdViewWeights ? apdViewWeights+(size_t)idx*MAX_VIEWS : nullptr));
		unsigned fallbackWeightSum(0u);
		if (previousWeights) {
			for (int view=0; view<nNumViews; ++view) {
				sampledViewWeights[view] = previousWeights[view];
				fallbackWeightSum += sampledViewWeights[view];
			}
		}
		if (fallbackWeightSum > 0u) {
			apdViewSelectionMode = APDViewSelectionMode::PREVIOUS_WEIGHTS_FALLBACK;
		} else {
			const unsigned previousMask(apdSelectedViewsSnapshot[idx]);
			for (int view=0; view<nNumViews; ++view)
				if (IsBitSet(previousMask, view)) {
					sampledViewWeights[view] = 1u;
					++fallbackWeightSum;
				}
			if (fallbackWeightSum > 0u) {
				apdViewSelectionMode = APDViewSelectionMode::SELECTED_MASK_FALLBACK;
			} else {
			sampledViewWeights[0] = 1u;
				apdViewSelectionMode = APDViewSelectionMode::FIRST_VIEW_FALLBACK;
			}
		}
	}
	if constexpr (APD) {
		if (apdViewWeights) {
			uint8_t* pixelViewWeights(apdViewWeights+(size_t)idx*MAX_VIEWS);
			for (int view=0; view<MAX_VIEWS; ++view)
				pixelViewWeights[view] = view < nNumViews ?
					static_cast<uint8_t>(sampledViewWeights[view]) : 0u;
		}
		if constexpr (VISIBILITY) {
			uint8_t* pixelNext(visibilityNext+(size_t)idx*MAX_VIEWS);
			for (int view=0; view<MAX_VIEWS; ++view)
				pixelNext[view] = view < nNumViews ?
					static_cast<uint8_t>(sampledViewWeights[view]) : 0u;
		}
	}
	unsigned viewWeights[MAX_VIEWS] = {};
	for (int view=0; view<nNumViews; ++view)
		viewWeights[view] = VISIBILITY ? pixelVisibilityResolved[view] : sampledViewWeights[view];
	#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT && APD)
		if (apdTrace)
			for (int view=0; view<nNumViews; ++view)
				apdTrace->viewWeights[view] = static_cast<uint8_t>(viewWeights[view]);
	#endif

	// propagate best neighbor plane
	Point4& plane = planes[idx];
	float& cost = costs[idx];
#ifdef _USE_DMAP_INSTRUMENTATION
	int updateSource = PM_SOURCE_NONE;
#endif
	unsigned newSelectedViews = 0;
	for (int imgId = 0; imgId < nNumViews; ++imgId)
		if (viewWeights[imgId])
			SetBit(newSelectedViews, imgId);
	float finalCosts[8];
	for (int posId = 0; posId < 8; ++posId) {
		if (!valid[posId]) {
			finalCosts[posId] = FLT_MAX;
			continue;
		}
		if constexpr (APD) {
			if (apdActive) {
				Point4 candidatePlane(apdFullActive ?
					LoadPlaneLDG(&apdPlanesSnapshot[positions[posId]]) :
					LoadPlaneLDG(&planes[positions[posId]]));
				candidatePlane.w() = neighborDepths[posId];
			#ifdef _USE_DMAP_INSTRUMENTATION
				finalCosts[posId] = ScorePatchMatchCandidate<GEOM, true, INSTRUMENT, VISIBILITY>(
					refCache, images, depthImages, p, candidatePlane, lowDepth, true,
					pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costArray[posId],
					INSTRUMENT ? &exactPropagationAPDComponents[posId] : nullptr,
					visibilityResolved);
			#else
				finalCosts[posId] = ScorePatchMatchCandidate<GEOM, true, false, VISIBILITY>(
					refCache, images, depthImages, p, candidatePlane, lowDepth, true,
					pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costArray[posId],
					nullptr, visibilityResolved);
			#endif
				continue;
			}
		}
		finalCosts[posId] = VISIBILITY ?
			AggregateAPDViewScores(viewWeights, costArray[posId], nNumViews) :
			AggregateMultiViewScores(viewWeights, costArray[posId], nNumViews);
	}
	const int nativeMinCostIdx = FindMinIndex(finalCosts, 8);
	int minCostIdx(nativeMinCostIdx);
	if constexpr (VISIBLE_NORMAL) {
		const bool evaluatePropagation(
			g_params.nDVPVisibleNormalMode == static_cast<unsigned>(DVPVisibleNormalMode::SHADOW) ||
			DVPVisibleNormalPropagationEnabled(g_params.nDVPVisibleNormalMode));
		if (evaluatePropagation) {
			uint8_t validCandidates[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
			uint8_t feasibleCandidates[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
			for (int posId=0; posId<DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES; ++posId) {
				if (!valid[posId] || !DVPVisibleNormalValidCost(finalCosts[posId]))
					continue;
				Point4 candidatePlane(apdFullActive ?
					LoadPlaneLDG(&apdPlanesSnapshot[positions[posId]]) :
					LoadPlaneLDG(&planes[positions[posId]]));
				candidatePlane.w() = neighborDepths[posId];
				#ifdef _USE_DMAP_INSTRUMENTATION
				if constexpr (INSTRUMENT) {
					uint8_t supportCount(0u);
					const DVPVisibleNormalEvaluation evaluation(
						EvaluateDVPVisibleNormalRuntime(
							p, candidatePlane.w(), newSelectedViews,
							candidatePlane.topLeftCorner<3,1>(),
							g_params.fDVPVisibleNormalDotTolerance, supportCount));
					validCandidates[posId] = evaluation.valid ? 1u : 0u;
					feasibleCandidates[posId] = evaluation.valid && evaluation.feasible ? 1u : 0u;
					visibleNormalPropagationEvaluationReasons[posId] =
						static_cast<uint8_t>(evaluation.reason);
					visibleNormalUpdate.selectedSourceViews = newSelectedViews;
					visibleNormalUpdate.supportCount = supportCount;
					visibleNormalUpdate.directionCount = evaluation.directionCount;
					visibleNormalUpdate.propagationTestedMask |= static_cast<uint8_t>(1u << posId);
					if (evaluation.valid)
						visibleNormalUpdate.propagationValidMask |= static_cast<uint8_t>(1u << posId);
					if (evaluation.valid && evaluation.feasible)
						visibleNormalUpdate.propagationFeasibleMask |= static_cast<uint8_t>(1u << posId);
					if (evaluation.valid && !evaluation.feasible)
						visibleNormalUpdate.propagationRejectedMask |= static_cast<uint8_t>(1u << posId);
					if (visibleNormalTrace) {
						const Point3 normal(candidatePlane.topLeftCorner<3,1>());
						visibleNormalTrace->propagationNormals[posId][0] = normal.x();
						visibleNormalTrace->propagationNormals[posId][1] = normal.y();
						visibleNormalTrace->propagationNormals[posId][2] = normal.z();
						visibleNormalTrace->propagationCosts[posId] = finalCosts[posId];
						visibleNormalTrace->propagationMaxDot[posId] = evaluation.maxDot;
						visibleNormalTrace->propagationReason[posId] =
							static_cast<uint8_t>(evaluation.reason);
						visibleNormalTrace->propagationRejectedDirection[posId] =
							evaluation.rejectedDirection;
					}
				}
				else
				#endif
				{
					const DVPVisibleNormalRuntimeStatus status(
						ClassifyDVPVisibleNormalRuntime(
							p, candidatePlane.w(), newSelectedViews,
							candidatePlane.topLeftCorner<3,1>(),
							g_params.fDVPVisibleNormalDotTolerance));
					validCandidates[posId] =
						status != DVPVisibleNormalRuntimeStatus::INVALID ? 1u : 0u;
					feasibleCandidates[posId] =
						status == DVPVisibleNormalRuntimeStatus::FEASIBLE ? 1u : 0u;
				}
			}
			const DVPVisibleNormalPropagationDecision decision(
				ResolveDVPVisibleNormalPropagation(finalCosts, validCandidates,
					feasibleCandidates, DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES));
			if (DVPVisibleNormalPropagationEnabled(g_params.nDVPVisibleNormalMode) &&
				decision.selected != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE)
			{
				minCostIdx = decision.selected;
			}
			#ifdef _USE_DMAP_INSTRUMENTATION
			if constexpr (INSTRUMENT) {
				visibleNormalUpdate.propagationNativeBest = decision.nativeBest;
				visibleNormalUpdate.propagationConstrainedBest = decision.constrainedBest;
				visibleNormalUpdate.propagationSelected = decision.selected;
				visibleNormalUpdate.propagationReason = static_cast<uint8_t>(decision.reason);
				visibleNormalUpdate.propagationFallback = decision.fallback ? 1u : 0u;
				visibleNormalUpdate.propagationAppliedConstraint =
					minCostIdx != nativeMinCostIdx ? 1u : 0u;
				visibleNormalUpdate.propagationNativeBestCost =
					decision.nativeBest != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE ?
						finalCosts[decision.nativeBest] : -1.f;
				visibleNormalUpdate.propagationConstrainedBestCost =
					decision.constrainedBest != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE ?
						finalCosts[decision.constrainedBest] : -1.f;
				visibleNormalUpdate.propagationSelectedCost =
					minCostIdx >= 0 && minCostIdx < 8 && DVPVisibleNormalValidCost(finalCosts[minCostIdx]) ?
						finalCosts[minCostIdx] : -1.f;
			}
			#endif
		}
	}
#ifdef _USE_DMAP_INSTRUMENTATION
	uint8_t anchorProposalCount = 0u;
	uint8_t anchorFiniteCount = 0u;
	float bestAnchorWorkingCost = -1.f;
	APDWorkingCostMinimum anchorWorkingCostMinimum;
	uint8_t acceptedAnchorSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint32_t acceptedAnchorIndex = ~uint32_t(0);
	float acceptedAnchorNativeCost = -1.f;
	if constexpr (INSTRUMENT && APD) {
		if (apdFullActive) {
			for (int slot=0; slot<static_cast<int>(APD_MAX_ANCHORS); ++slot) {
				if (!valid[slot])
					continue;
				++anchorProposalCount;
				AccumulateAPDWorkingCostMinimum(anchorWorkingCostMinimum, finalCosts[slot]);
				if (apdTrace) {
					apdTrace->anchorCandidateWorkingCosts[slot] = finalCosts[slot];
					Point4 nativePlane(LoadPlaneLDG(&apdPlanesSnapshot[positions[slot]]));
					nativePlane.w() = neighborDepths[slot];
					float nativeViewCosts[MAX_VIEWS];
					MultiViewScorePlane<GEOM>(
						refCache, images, depthImages, p, nativePlane, lowDepth, nativeViewCosts);
					apdTrace->anchorCandidateNativeCosts[slot] = AggregateAPDViewScores(
						viewWeights, nativeViewCosts, nNumViews);
				}
			}
			anchorFiniteCount = static_cast<uint8_t>(anchorWorkingCostMinimum.finiteCount);
			bestAnchorWorkingCost = anchorWorkingCostMinimum.finiteCount > 0u ?
				anchorWorkingCostMinimum.minimum : -1.f;
		}
	}
#endif
	float costVector[MAX_VIEWS];
	#ifdef _USE_DMAP_INSTRUMENTATION
	APDAggregateScoreComponents currentAPDComponents;
	cost = ScorePatchMatchCandidate<GEOM, APD, INSTRUMENT, VISIBILITY>(
		refCache, images, depthImages, p, plane, lowDepth, apdActive,
		pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
		INSTRUMENT ? &currentAPDComponents : nullptr, visibilityResolved);
	#else
	cost = ScorePatchMatchCandidate<GEOM, APD, false, VISIBILITY>(
		refCache, images, depthImages, p, plane, lowDepth, apdActive,
		pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
		nullptr, visibilityResolved);
	#endif
#ifdef _USE_DMAP_INSTRUMENTATION
	const float exactIncumbentCost(cost);
	if constexpr (INSTRUMENT) {
		if constexpr (APD)
			if (apdActive)
				exactWinningAPDComponents = currentAPDComponents;
		for (int view = 0; view < nNumViews; ++view)
			exactWinningViewCosts[view] = costVector[view];
		TrackExactCandidate(
			PM_EXACT_CANDIDATE_CURRENT, cost,
			exactTestedMask, exactFiniteMask, exactTestedCount, exactFiniteCount,
			exactBestCost, exactBestSlot, exactRunnerUpCost, exactRunnerUpSlot);
	}
	if constexpr (INSTRUMENT) {
		for (int posId = 0; posId < 8; ++posId) {
			if (!valid[posId])
				continue;
			const int slot(apdFullActive ? PM_EXACT_CANDIDATE_APD_ANCHOR_0+posId :
				PM_EXACT_CANDIDATE_PROPAGATION_0+posId);
			TrackExactCandidate(
				slot, finalCosts[posId],
				exactTestedMask, exactFiniteMask, exactTestedCount, exactFiniteCount,
				exactBestCost, exactBestSlot, exactRunnerUpCost, exactRunnerUpSlot);
		}
	}
#endif
#ifdef _USE_DMAP_INSTRUMENTATION
	const bool propagationAccepted = minCostIdx >= 0 && valid[minCostIdx] && finalCosts[minCostIdx] < cost;
	if constexpr (INSTRUMENT && VISIBLE_NORMAL)
		visibleNormalUpdate.propagationAccepted = propagationAccepted ? 1u : 0u;
	if constexpr (INSTRUMENT) {
		for (int posId = 0; posId < 8; ++posId) {
			if (!valid[posId])
				continue;
			InstrumentCandidate(
				instrumentCounters, instrumentParams, PM_CANDIDATE_PROPAGATION,
				finalCosts[posId], propagationAccepted && posId == minCostIdx);
		}
		}
#endif
#ifndef _USE_DMAP_INSTRUMENTATION
	if (finalCosts[minCostIdx] < cost) {
		ASSERT(valid[minCostIdx]);
#else
	if (propagationAccepted) {
#endif
		plane = apdFullActive ? LoadPlaneLDG(&apdPlanesSnapshot[positions[minCostIdx]]) :
			LoadPlaneLDG(&planes[positions[minCostIdx]]);
		plane.w() = neighborDepths[minCostIdx];
		cost = finalCosts[minCostIdx];
		selectedViews[idx] = newSelectedViews;
#ifdef _USE_DMAP_INSTRUMENTATION
		updateSource = apdFullActive ? PM_SOURCE_APD_ANCHOR_PROPAGATE : PM_SOURCE_PROPAGATE;
		if constexpr (INSTRUMENT) {
			if constexpr (APD) {
				if (apdFullActive) {
					acceptedAnchorSlot = static_cast<uint8_t>(minCostIdx);
					acceptedAnchorIndex = static_cast<uint32_t>(positions[minCostIdx]);
					float nativeViewCosts[MAX_VIEWS];
					MultiViewScorePlane<GEOM>(
						refCache, images, depthImages, p, plane, lowDepth, nativeViewCosts);
					acceptedAnchorNativeCost = AggregateAPDViewScores(
						viewWeights, nativeViewCosts, nNumViews);
				}
			}
			exactWinnerSlot = (uint8_t)((apdFullActive ? PM_EXACT_CANDIDATE_APD_ANCHOR_0 :
				PM_EXACT_CANDIDATE_PROPAGATION_0)+minCostIdx);
			exactAcceptedMask |= 1u << exactWinnerSlot;
			++exactAcceptedCount;
			if constexpr (APD)
				if (apdActive)
					exactWinningAPDComponents = exactPropagationAPDComponents[minCostIdx];
			for (int view = 0; view < nNumViews; ++view)
				exactWinningViewCosts[view] = costArray[minCostIdx][view];
		}
#endif
	}
	const bool fittedPlaneAvailable(
		apdFullActive && apdFittedPlanes && apdFittedPlaneValid && apdFittedPlaneValid[idx] != 0u &&
		apdFittedPlanes[idx].w() >= g_params.fDepthMin &&
		apdFittedPlanes[idx].w() <= g_params.fDepthMax);
	float fittedPlaneWorkingCost(FLT_MAX);
#ifdef _USE_DMAP_INSTRUMENTATION
	float fittedPlaneNativeCost(-1.f);
	bool fittedPlaneAcceptedEvent(false);
	APDAggregateScoreComponents fittedPlaneAPDComponents;
#endif
	if (fittedPlaneAvailable) {
		const Point4 fittedPlane(LoadPlaneLDG(&apdFittedPlanes[idx]));
		#ifdef _USE_DMAP_INSTRUMENTATION
		fittedPlaneWorkingCost = ScorePatchMatchCandidate<GEOM, true, INSTRUMENT, VISIBILITY>(
			refCache, images, depthImages, p, fittedPlane, lowDepth, true,
			pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
			INSTRUMENT ? &fittedPlaneAPDComponents : nullptr, visibilityResolved);
		const bool fittedPlaneAccepted(fittedPlaneWorkingCost < cost);
		fittedPlaneAcceptedEvent = fittedPlaneAccepted;
		if constexpr (INSTRUMENT) {
			TrackExactCandidate(
				PM_EXACT_CANDIDATE_APD_FITTED_PLANE, fittedPlaneWorkingCost,
				exactTestedMask, exactFiniteMask, exactTestedCount, exactFiniteCount,
				exactBestCost, exactBestSlot, exactRunnerUpCost, exactRunnerUpSlot);
			InstrumentCandidate(instrumentCounters, instrumentParams,
				PM_CANDIDATE_APD_FITTED_PLANE, fittedPlaneWorkingCost,
				fittedPlaneAccepted);
			float nativeViewCosts[MAX_VIEWS];
			MultiViewScorePlane<GEOM>(
				refCache, images, depthImages, p, fittedPlane, lowDepth, nativeViewCosts);
			fittedPlaneNativeCost = AggregateAPDViewScores(
				viewWeights, nativeViewCosts, nNumViews);
			if (fittedPlaneAccepted) {
				exactWinnerSlot = PM_EXACT_CANDIDATE_APD_FITTED_PLANE;
				exactAcceptedMask |= 1u << PM_EXACT_CANDIDATE_APD_FITTED_PLANE;
				++exactAcceptedCount;
				exactWinningAPDComponents = fittedPlaneAPDComponents;
				for (int view=0; view<nNumViews; ++view)
					exactWinningViewCosts[view] = costVector[view];
			}
		}
		#else
		fittedPlaneWorkingCost = ScorePatchMatchCandidate<GEOM, true, false, VISIBILITY>(
			refCache, images, depthImages, p, fittedPlane, lowDepth, true,
			pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
			nullptr, visibilityResolved);
		#endif
		if (fittedPlaneWorkingCost < cost) {
			plane = fittedPlane;
			cost = fittedPlaneWorkingCost;
			#ifdef _USE_DMAP_INSTRUMENTATION
			updateSource = PM_SOURCE_APD_FITTED_PLANE;
			#endif
		}
	}
	bool replaceNativeDepth(false);
	if constexpr (DVP) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		const float dvpIncumbentDepth(plane.w());
		const float dvpIncumbentCost(cost);
		#endif
		DVPProposalSet dvpProposals(DVPCollectProposals<GEOM>(
			depthImages, p, plane.w(), viewWeights, g_params.nDVPEpipolarFamily));
		if (DVPFamilyUsesPaperIntervals(g_params.nDVPEpipolarFamily)) {
			DVPDepthProposalRNG proposalRNG(randState);
			if (dvpProposals.intervals.left.valid) {
				const unsigned proposal(dvpProposals.count++);
				dvpProposals.depths[proposal] = DVPInterpolateInterval(
					dvpProposals.intervals.left, proposalRNG.Uniform());
				dvpProposals.signedOffsets[proposal] = -1;
				dvpProposals.supports[proposal] =
					static_cast<uint8_t>(g_params.nDVPEpipolarMu);
				dvpProposals.supportViews[proposal] =
					dvpProposals.endpointSamples.leftOuterViews &
					dvpProposals.endpointSamples.leftInnerViews;
			}
			if (dvpProposals.intervals.right.valid &&
				dvpProposals.count < DVP_MAX_PROPOSALS)
			{
				const unsigned proposal(dvpProposals.count++);
				dvpProposals.depths[proposal] = DVPInterpolateInterval(
					dvpProposals.intervals.right, proposalRNG.Uniform());
				dvpProposals.signedOffsets[proposal] = 1;
				dvpProposals.supports[proposal] =
					static_cast<uint8_t>(g_params.nDVPEpipolarMu);
				dvpProposals.supportViews[proposal] =
					dvpProposals.endpointSamples.rightInnerViews &
					dvpProposals.endpointSamples.rightOuterViews;
			}
			dvpProposals.nativeDepthFallback = dvpProposals.count == 0u;
			if (!dvpProposals.nativeDepthFallback)
				DVPDepthProposalRNG::ConsumeNativeDepthSlot(randState);
		}
		replaceNativeDepth =
			DVPFamilyUsesPaperIntervals(g_params.nDVPEpipolarFamily) &&
			dvpProposals.count > 0u;
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			InitializeDVPInstrumentUpdate(
				dvpProposals, dvpIncumbentDepth, dvpIncumbentCost, p, iter,
				instrumentParams, dvpUpdate, dvpTrace);
		#endif
		for (unsigned proposal=0u; proposal<dvpProposals.count; ++proposal) {
			const float candidateDepth(dvpProposals.depths[proposal]);
			if (!isfinite(candidateDepth) || candidateDepth < g_params.fDepthMin ||
				candidateDepth > g_params.fDepthMax)
				continue;
			Point4 candidatePlane(plane);
			candidatePlane.w() = candidateDepth;
			#ifdef _USE_DMAP_INSTRUMENTATION
			APDAggregateScoreComponents candidateAPDComponents;
			const float candidateCost(ScorePatchMatchCandidate<GEOM, APD, INSTRUMENT, VISIBILITY>(
				refCache, images, depthImages, p, candidatePlane, lowDepth, apdActive,
				pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
				INSTRUMENT ? &candidateAPDComponents : nullptr, visibilityResolved));
			const bool candidateAccepted(cost > candidateCost);
			if constexpr (INSTRUMENT) {
				++dvpUpdate.testedCount;
				dvpUpdate.testedProposalMask |= static_cast<uint8_t>(1u << proposal);
				dvpUpdate.candidateCost[proposal] =
					DVPValidProposalCost(candidateCost) ? candidateCost : -1.f;
				if (DVPValidProposalCost(candidateCost)) {
					++dvpUpdate.finiteCount;
					dvpUpdate.finiteProposalMask |= static_cast<uint8_t>(1u << proposal);
				}
				if (candidateAccepted) {
					dvpUpdate.acceptedProposalMask |= static_cast<uint8_t>(1u << proposal);
					++dvpUpdate.acceptedCount;
				}
				const int slot(
					PM_EXACT_CANDIDATE_DVP_EPIPOLAR_0+static_cast<int>(proposal));
				TrackExactCandidate(
					slot, candidateCost, exactTestedMask, exactFiniteMask,
					exactTestedCount, exactFiniteCount, exactBestCost, exactBestSlot,
					exactRunnerUpCost, exactRunnerUpSlot);
				if (candidateAccepted) {
					exactWinnerSlot = static_cast<uint8_t>(slot);
					exactAcceptedMask |= 1u << slot;
					++exactAcceptedCount;
					if constexpr (APD)
						if (apdActive)
							exactWinningAPDComponents = candidateAPDComponents;
					for (int view=0; view<nNumViews; ++view)
						exactWinningViewCosts[view] = costVector[view];
				}
			}
			#else
			const float candidateCost(ScorePatchMatchCandidate<GEOM, APD, false, VISIBILITY>(
				refCache, images, depthImages, p, candidatePlane, lowDepth, apdActive,
				pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
				nullptr, visibilityResolved));
			const bool candidateAccepted(cost > candidateCost);
			#endif
			if (candidateAccepted) {
				cost = candidateCost;
				plane = candidatePlane;
				#ifdef _USE_DMAP_INSTRUMENTATION
				updateSource = PM_SOURCE_DVP_EPIPOLAR;
				#endif
			}
		}
	}
	const float depth = plane.w();

	// refine estimate
	constexpr float perturbationDepth = 0.005f;
	constexpr float perturbationNormal = 0.01f * (float)M_PI;
	const float depthPerturbed(replaceNativeDepth ? depth :
		(g_params.bCompat23
		? GeneratePerturbedDepth23(depth, randState, perturbationDepth)
		: GeneratePerturbedDepth(depth, randState, perturbationDepth)));
	const Point3 perturbedNormal = g_params.bCompat23
		? GeneratePerturbedNormal23(g_cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal)
		: GeneratePerturbedNormal(g_cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal);
	const Point3 normalRand = GenerateRandomNormal(g_cameras[0], p, randState);
	Point3 selectedPerturbedNormal(perturbedNormal);
	Point3 selectedRandomNormal(normalRand);
	if constexpr (VISIBLE_NORMAL) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT) {
			uint8_t supportCount(0u);
			const DVPVisibleNormalEvaluation currentEvaluation(
				EvaluateDVPVisibleNormalRuntime(
					p, depth, newSelectedViews, plane.topLeftCorner<3,1>(),
					g_params.fDVPVisibleNormalDotTolerance, supportCount));
			visibleNormalUpdate.selectedSourceViews = newSelectedViews;
			visibleNormalUpdate.supportCount = supportCount;
			visibleNormalUpdate.directionCount = currentEvaluation.directionCount;
			visibleNormalUpdate.currentValid = currentEvaluation.valid ? 1u : 0u;
			visibleNormalUpdate.currentFeasible = currentEvaluation.feasible ? 1u : 0u;
			visibleNormalUpdate.currentReason = static_cast<uint8_t>(currentEvaluation.reason);
			visibleNormalUpdate.currentRejectedDirection = currentEvaluation.rejectedDirection;
			visibleNormalUpdate.currentMaxDot = currentEvaluation.maxDot;
			visibleNormalUpdate.currentMaxViolation = currentEvaluation.maxViolation;
			if (visibleNormalTrace) {
				const Point3 currentNormal(plane.topLeftCorner<3,1>());
				visibleNormalTrace->currentNormal[0] = currentNormal.x();
				visibleNormalTrace->currentNormal[1] = currentNormal.y();
				visibleNormalTrace->currentNormal[2] = currentNormal.z();
			}
		}
		#endif
		const bool evaluateRefinement(
			g_params.nDVPVisibleNormalMode == static_cast<unsigned>(DVPVisibleNormalMode::SHADOW) ||
			DVPVisibleNormalRefinementEnabled(g_params.nDVPVisibleNormalMode));
		if (evaluateRefinement) {
			// Fork after both native normals have consumed the production RNG stream.
			// Retries are deterministic but cannot change later native proposals.
			RandState retryState(*randState);
			#ifdef _USE_DMAP_INSTRUMENTATION
			if constexpr (INSTRUMENT) {
				const DVPVisibleNormalProposalDecision decisions[
					DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
					ResolveDVPVisibleNormalRuntimeProposal(
						g_cameras[0], p, depth, newSelectedViews,
						plane.topLeftCorner<3,1>(), perturbedNormal,
						&retryState, false, perturbationNormal,
						g_params.fDVPVisibleNormalDotTolerance,
						g_params.nDVPVisibleNormalAttempts),
					ResolveDVPVisibleNormalRuntimeProposal(
						g_cameras[0], p, depth, newSelectedViews,
						plane.topLeftCorner<3,1>(), normalRand,
						&retryState, true, perturbationNormal,
						g_params.fDVPVisibleNormalDotTolerance,
						g_params.nDVPVisibleNormalAttempts),
				};
				if (DVPVisibleNormalRefinementEnabled(g_params.nDVPVisibleNormalMode)) {
					selectedPerturbedNormal =
						FromDVPVisibleNormalVector(decisions[0].selected);
					selectedRandomNormal =
						FromDVPVisibleNormalVector(decisions[1].selected);
				}
				const Point3 nativeNormals[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
					perturbedNormal, normalRand};
				const Point3 selectedNormals[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
					selectedPerturbedNormal, selectedRandomNormal};
				for (int proposal=0; proposal<DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS;
					++proposal)
				{
					visibleNormalUpdate.refinementNativeTestedMask |=
						static_cast<uint8_t>(1u << proposal);
					if (decisions[proposal].nativeEvaluation.valid)
						visibleNormalUpdate.refinementNativeValidMask |=
							static_cast<uint8_t>(1u << proposal);
					if (decisions[proposal].nativeEvaluation.feasible)
						visibleNormalUpdate.refinementNativeFeasibleMask |=
							static_cast<uint8_t>(1u << proposal);
					if (decisions[proposal].reason ==
						DVPVisibleNormalProposalReason::CONSTRAINED_RETRY)
					{
						visibleNormalUpdate.refinementRetrySuccessMask |=
							static_cast<uint8_t>(1u << proposal);
						if (DVPVisibleNormalRefinementEnabled(g_params.nDVPVisibleNormalMode))
							visibleNormalUpdate.refinementAppliedRetryMask |=
								static_cast<uint8_t>(1u << proposal);
					}
					if (decisions[proposal].reason ==
						DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK)
					{
						visibleNormalUpdate.refinementExhaustionFallbackMask |=
							static_cast<uint8_t>(1u << proposal);
					}
					visibleNormalUpdate.retriesTested[proposal] = decisions[proposal].retriesTested;
					visibleNormalUpdate.selectedRetry[proposal] = decisions[proposal].selectedRetry;
					visibleNormalUpdate.nativeMaxDot[proposal] =
						decisions[proposal].nativeEvaluation.maxDot;
					visibleNormalUpdate.selectedMaxDot[proposal] =
						decisions[proposal].selectedEvaluation.maxDot;
					visibleNormalUpdate.nativeReason[proposal] =
						static_cast<uint8_t>(decisions[proposal].nativeEvaluation.reason);
					visibleNormalUpdate.proposalReason[proposal] =
						static_cast<uint8_t>(decisions[proposal].reason);
					visibleNormalUpdate.nativeRejectedDirection[proposal] =
						decisions[proposal].nativeEvaluation.rejectedDirection;
					visibleNormalUpdate.selectedRejectedDirection[proposal] =
						decisions[proposal].selectedEvaluation.rejectedDirection;
					if (visibleNormalTrace) {
						for (int component=0; component<3; ++component) {
							visibleNormalTrace->nativeNormals[proposal][component] =
								nativeNormals[proposal][component];
							visibleNormalTrace->selectedNormals[proposal][component] =
								selectedNormals[proposal][component];
						}
					}
				}
			} else
			#endif
			{
				const Point3 constrainedPerturbedNormal(
					ResolveDVPVisibleNormalRuntimeProduction(
					g_cameras[0], p, depth, newSelectedViews,
					plane.topLeftCorner<3,1>(), perturbedNormal,
					&retryState, false, perturbationNormal,
					g_params.fDVPVisibleNormalDotTolerance,
					g_params.nDVPVisibleNormalAttempts));
				const Point3 constrainedRandomNormal(
					ResolveDVPVisibleNormalRuntimeProduction(
						g_cameras[0], p, depth, newSelectedViews,
						plane.topLeftCorner<3,1>(), normalRand,
						&retryState, true, perturbationNormal,
						g_params.fDVPVisibleNormalDotTolerance,
						g_params.nDVPVisibleNormalAttempts));
				if (DVPVisibleNormalRefinementEnabled(g_params.nDVPVisibleNormalMode)) {
					selectedPerturbedNormal = constrainedPerturbedNormal;
					selectedRandomNormal = constrainedRandomNormal;
				}
			}
		}
	}
	int numValidPlanes = 3;
	Point3 surfaceNormal = Point3::Zero();
	if (!apdFullActive && valid[0] && valid[1] && valid[2] && valid[3]) {
		// estimate normal from surrounding surface
		const Point4 ndepths(
			LoadPlaneWLDG(&planes[neighborPositions[0]]),
			LoadPlaneWLDG(&planes[neighborPositions[1]]),
			LoadPlaneWLDG(&planes[neighborPositions[2]]),
			LoadPlaneWLDG(&planes[neighborPositions[3]])
		);
		surfaceNormal = ComputeDepthGradient(g_cameras[0].model, depth, p, ndepths);
		numValidPlanes = 4;
	}
	constexpr int numPlanes = 4;
	const float depths[numPlanes] = {depthPerturbed, depth, depth, depth};
	const Point3 normals[numPlanes] = {
		plane.topLeftCorner<3,1>(), selectedPerturbedNormal, selectedRandomNormal, surfaceNormal};
	for (int i = 0; i < numValidPlanes; ++i) {
		if (replaceNativeDepth && i == 0)
			continue;
		Point4 newPlane;
		newPlane.topLeftCorner<3,1>() = normals[i];
		newPlane.w() = depths[i];
		#ifdef _USE_DMAP_INSTRUMENTATION
		APDAggregateScoreComponents candidateAPDComponents;
		const float costPlane(ScorePatchMatchCandidate<GEOM, APD, INSTRUMENT, VISIBILITY>(
			refCache, images, depthImages, p, newPlane, lowDepth, apdActive,
			pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
			INSTRUMENT ? &candidateAPDComponents : nullptr, visibilityResolved));
		#else
		const float costPlane(ScorePatchMatchCandidate<GEOM, APD, false, VISIBILITY>(
			refCache, images, depthImages, p, newPlane, lowDepth, apdActive,
			pixelAPDAnchors, pixelAPDAnchorCount, viewWeights, costVector,
			nullptr, visibilityResolved));
		#endif
#ifdef _USE_DMAP_INSTRUMENTATION
		const bool candidateAccepted = cost > costPlane;
		if constexpr (INSTRUMENT) {
			const int slot(PM_EXACT_CANDIDATE_REFINE_DEPTH + i);
			TrackExactCandidate(
				slot, costPlane,
				exactTestedMask, exactFiniteMask, exactTestedCount, exactFiniteCount,
				exactBestCost, exactBestSlot, exactRunnerUpCost, exactRunnerUpSlot);
			const int candidateType(i == 2 ? PM_CANDIDATE_RANDOM_PERTURBATION : PM_CANDIDATE_REFINEMENT);
			InstrumentCandidate(instrumentCounters, instrumentParams, candidateType, costPlane, candidateAccepted);
			if (candidateAccepted) {
				if constexpr (VISIBLE_NORMAL)
					if (i == 1 || i == 2)
						visibleNormalUpdate.refinementAcceptedMask |=
							static_cast<uint8_t>(1u << (i-1));
				exactWinnerSlot = (uint8_t)slot;
				exactAcceptedMask |= 1u << slot;
				++exactAcceptedCount;
				if constexpr (APD)
					if (apdActive)
						exactWinningAPDComponents = candidateAPDComponents;
				for (int view = 0; view < nNumViews; ++view)
					exactWinningViewCosts[view] = costVector[view];
			}
		}
#endif
#ifndef _USE_DMAP_INSTRUMENTATION
		if (cost > costPlane) {
#else
		if (candidateAccepted) {
#endif
			cost = costPlane;
			plane = newPlane;
#ifdef _USE_DMAP_INSTRUMENTATION
			updateSource =
				i == 0 ? PM_SOURCE_REFINE_DEPTH :
				i == 1 ? PM_SOURCE_REFINE_NORMAL :
				i == 2 ? PM_SOURCE_REFINE_RANDOM_NORMAL :
						 PM_SOURCE_REFINE_SURFACE_NORMAL;
#endif
		}
		}
	#ifdef _USE_DMAP_INSTRUMENTATION
	const float apdWorkingWinnerCost(cost);
	#endif
	if constexpr (APD) {
		if (apdActive) {
			MultiViewScorePlane<GEOM>(
				refCache, images, depthImages, p, plane, lowDepth, costVector);
			const float nativeRescore(AggregateAPDViewScores(viewWeights, costVector, nNumViews));
			cost = ResolveAPDPersistentScore(true, cost, nativeRescore).cost;
		}
		}
	#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT && VISIBLE_NORMAL)
		FinalizeDVPVisibleNormalInstrumentation(
			p, idx, iter, instrumentParams, visibleNormalUpdate,
			visibleNormalPropagationEvaluationReasons, visibleNormalTrace);
	if constexpr (INSTRUMENT && DVP)
		FinalizeDVPInstrumentation(
			plane.w(), updateSource, idx, iter, instrumentParams, dvpUpdate, dvpTrace);
	if constexpr (INSTRUMENT && APD) {
		PatchMatchAPDInstrumentUpdate apdUpdate;
		apdUpdate.workingWinnerCost = apdWorkingWinnerCost;
		apdUpdate.nativePersistentCost = cost;
		apdUpdate.runnerUpWorkingCost = exactFiniteCount >= 2 &&
			exactRunnerUpSlot != PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE ? exactRunnerUpCost : -1.f;
		apdUpdate.winnerRunnerUpGap = apdUpdate.runnerUpWorkingCost >= 0.f ?
			max(0.f, apdUpdate.runnerUpWorkingCost-exactBestCost) : -1.f;
		apdUpdate.nativeMinusWorkingCost = cost-apdWorkingWinnerCost;
		apdUpdate.nativeStoredCostBefore = instrumentStoredCostBefore;
		apdUpdate.incumbentWorkingCost = exactIncumbentCost;
		apdUpdate.bestAnchorWorkingCost = bestAnchorWorkingCost;
		apdUpdate.acceptedAnchorNativeCost = acceptedAnchorNativeCost;
		apdUpdate.fittedPlaneWorkingCost = fittedPlaneAvailable &&
			isfinite(fittedPlaneWorkingCost) ?
			fittedPlaneWorkingCost : -1.f;
		apdUpdate.fittedPlaneNativeCost = fittedPlaneNativeCost;
		apdUpdate.candidateTestedMask = exactTestedMask;
		apdUpdate.candidateFiniteMask = exactFiniteMask;
		apdUpdate.candidateAcceptedMask = exactAcceptedMask;
		apdUpdate.acceptedAnchorIndex = acceptedAnchorIndex;
		for (int view=0; view<nNumViews; ++view) {
			if (viewWeights[view])
				SetBit(apdUpdate.workingSelectedViews, view);
			apdUpdate.selectedViewWeightSum = static_cast<uint8_t>(
				min(255u, static_cast<unsigned>(apdUpdate.selectedViewWeightSum)+viewWeights[view]));
		}
		apdUpdate.source = static_cast<uint8_t>(updateSource);
		apdUpdate.winnerSlot = exactWinnerSlot;
		apdUpdate.runnerUpSlot = apdUpdate.runnerUpWorkingCost >= 0.f ?
			exactRunnerUpSlot : PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
		apdUpdate.testedCount = exactTestedCount;
		apdUpdate.finiteCount = exactFiniteCount;
		apdUpdate.acceptedCount = exactAcceptedCount;
		apdUpdate.selectedViewCount = static_cast<uint8_t>(CountSelectedViews(
			apdUpdate.workingSelectedViews));
		apdUpdate.deformableActive = apdActive ? 1u : 0u;
		apdUpdate.viewSelectionMode = static_cast<uint8_t>(apdViewSelectionMode);
		apdUpdate.anchorEvidenceCount = apdFullActive ?
			static_cast<uint8_t>(__popc(validAnchorMask)) : 0u;
		apdUpdate.anchorProposalCount = anchorProposalCount;
		apdUpdate.anchorFiniteCount = anchorFiniteCount;
		apdUpdate.anchorAcceptedSlot = acceptedAnchorSlot;
		apdUpdate.immutableAnchorState = apdActive ? 1u : 0u;
		apdUpdate.updateStage = static_cast<uint8_t>(apdStage);
		apdUpdate.fittedPlaneAvailable = fittedPlaneAvailable ? 1u : 0u;
		apdUpdate.fittedPlaneTested = fittedPlaneAvailable ? 1u : 0u;
		apdUpdate.fittedPlaneAccepted = fittedPlaneAcceptedEvent ? 1u : 0u;
		if (apdActive) {
			apdUpdate.centerCost = exactWinningAPDComponents.centerCost;
			apdUpdate.anchorMeanCost = exactWinningAPDComponents.anchorMeanCost;
			apdUpdate.deformablePhotometricCost =
				exactWinningAPDComponents.deformablePhotometricCost;
			apdUpdate.geometricCost = exactWinningAPDComponents.geometricCost;
		}

		for (int view=0; view<nNumViews; ++view) {
			APDViewScoreComponents components;
			if (apdActive && apdTrace)
				ScorePlaneAPD<GEOM, true, VISIBILITY>(
					images, depthImages, p, plane, pixelAPDAnchors,
					pixelAPDAnchorCount, view, &components, visibilityResolved);
			if (apdTrace) {
				apdTrace->viewCenterCosts[view] = apdActive ? components.centerCost : -1.f;
				apdTrace->viewAnchorMeanCosts[view] = apdActive ? components.anchorMeanCost : -1.f;
				apdTrace->viewWorkingCosts[view] = apdActive ? components.totalCost : -1.f;
			}
		}
		if (instrumentParams.apdUpdates && iter >= 0 && iter < instrumentParams.numLogicalStates-1)
			instrumentParams.apdUpdates[(size_t)iter*instrumentParams.area+idx] = apdUpdate;
		if (apdTrace)
			apdTrace->update = apdUpdate;
		if (instrumentParams.apdCounters && iter >= 0 && iter < instrumentParams.numLogicalStates-1) {
			PatchMatchAPDInstrumentCounters& counter(instrumentParams.apdCounters[iter]);
			if (apdUpdate.updateStage < 3u)
				atomicAdd(&counter.stageUpdates[apdUpdate.updateStage], 1u);
			if (apdUpdate.fittedPlaneAvailable)
				atomicAdd(&counter.fittedPlaneAvailable, 1u);
			if (apdUpdate.fittedPlaneTested)
				atomicAdd(&counter.fittedPlaneTested, 1u);
			if (apdUpdate.fittedPlaneAvailable &&
				IsFiniteCandidateCost(apdUpdate.fittedPlaneWorkingCost))
				atomicAdd(&counter.fittedPlaneFinite, 1u);
			if (apdUpdate.fittedPlaneAccepted)
				atomicAdd(&counter.fittedPlaneAccepted, 1u);
			if (apdUpdate.source == PM_SOURCE_APD_FITTED_PLANE)
				atomicAdd(&counter.fittedPlaneFinalWinners, 1u);
			if (updateSource >= 0 && updateSource < PM_INSTRUMENT_NUM_SOURCES)
				atomicAdd(&counter.updateSource[updateSource], 1u);
			if (apdActive) {
				atomicAdd(&counter.deformableUpdates, 1u);
				InstrumentAtomicAdd(&counter.centerCostSum, apdUpdate.centerCost);
				InstrumentAtomicAdd(&counter.anchorMeanCostSum, apdUpdate.anchorMeanCost);
				InstrumentAtomicAdd(&counter.workingCostSum, apdUpdate.workingWinnerCost);
				InstrumentAtomicAdd(&counter.nativePersistentCostSum, apdUpdate.nativePersistentCost);
				if (apdUpdate.winnerRunnerUpGap >= 0.f) {
					InstrumentAtomicAdd(&counter.workingGapSum, apdUpdate.winnerRunnerUpGap);
					atomicAdd(&counter.workingGapSamples, 1u);
				}
				if (apdFullActive) {
					atomicAdd(&counter.anchorViewSelectionAttempted, 1u);
					if (apdUpdate.viewSelectionMode == static_cast<uint8_t>(APDViewSelectionMode::ANCHOR_EVIDENCE))
						atomicAdd(&counter.anchorViewSelectionUsed, 1u);
					if (apdUpdate.viewSelectionMode < PM_APD_INSTRUMENT_VIEW_SELECTION_MODES)
						atomicAdd(&counter.anchorViewSelectionMode[apdUpdate.viewSelectionMode], 1u);
				}
				atomicAdd(&counter.anchorProposalsTested,
					static_cast<uint32_t>(apdUpdate.anchorProposalCount));
				atomicAdd(&counter.anchorProposalsFinite,
					static_cast<uint32_t>(apdUpdate.anchorFiniteCount));
				atomicAdd(&counter.immutableAnchorStateUpdates,
					static_cast<uint32_t>(apdUpdate.immutableAnchorState));
				if (apdUpdate.anchorAcceptedSlot != PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE)
					atomicAdd(&counter.anchorProposalsAccepted, 1u);
				if (apdUpdate.source == PM_SOURCE_APD_ANCHOR_PROPAGATE)
					atomicAdd(&counter.anchorPropagationFinalWinners, 1u);
				if (apdUpdate.bestAnchorWorkingCost >= 0.f) {
					InstrumentAtomicAdd(&counter.bestAnchorWorkingCostSum, apdUpdate.bestAnchorWorkingCost);
					atomicAdd(&counter.bestAnchorWorkingCostSamples, 1u);
				}
				if (apdUpdate.acceptedAnchorNativeCost >= 0.f) {
					InstrumentAtomicAdd(&counter.acceptedAnchorNativeCostSum, apdUpdate.acceptedAnchorNativeCost);
					atomicAdd(&counter.acceptedAnchorNativeCostSamples, 1u);
				}
			}
		}
	}
	if constexpr (INSTRUMENT && VISIBILITY) {
		CaptureDVPVisibilityInstrumentation(
			p, idx, iter, visibilityPrevious, visibilityResolved, visibilityNext,
			viewWeights, exactTestedMask, exactFiniteMask, instrumentTraceMap,
			instrumentParams);
	}
	if constexpr (INSTRUMENT) {
		const float confidenceGap = exactFiniteCount >= 2 && exactRunnerUpSlot != PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE ?
			max(0.f, exactRunnerUpCost-exactBestCost) : -1.f;
		unsigned selectedWeightSum(0u);
		for (int view=0; view<nNumViews; ++view)
			selectedWeightSum += viewWeights[view];
		const float viewEntropy = ComputeViewEntropy(
			viewWeights, nNumViews, static_cast<float>(selectedWeightSum));
		PatchMatchInstrumentCostComponents exactComponents(WriteExactIterationViews<GEOM>(
			refCache, images, depthImages, p, plane, lowDepth, &costArray[0][0], validNeighbors,
			viewSelectionPriors, samplingProbs, viewWeights, exactWinningViewCosts,
			thCost, instrumentParams, idx));
		// The retained production value is authoritative even if a diagnostic
		// component rescore differs in its least significant bit.
		exactComponents.totalCost = cost;
		const float depthBefore(instrumentPlaneBefore.w());
		const float depthAfter(plane.w());
		const float depthAbsChange = depthBefore > 0.f && depthAfter > 0.f ? fabsf(depthAfter-depthBefore) : 0.f;
		const float depthRelChange = depthBefore > 0.f ? depthAbsChange / max(depthBefore, FLT_EPSILON) : 0.f;
		const float normalAngleChange(ComputeNormalAngleDegrees(
			instrumentPlaneBefore.topLeftCorner<3,1>(), plane.topLeftCorner<3,1>()));
		InstrumentPixel(
			instrumentCounters, instrumentTraceRecords, instrumentTraceMap, updateSources,
			instrumentParams, idx, p,
			depthBefore, depthAfter, instrumentStoredCostBefore, cost,
			exactComponents, refCache.varRef, lowDepth,
			instrumentSelectedViewsBefore, selectedViews[idx], updateSource,
			depthAbsChange, depthRelChange, normalAngleChange, viewEntropy,
			confidenceGap, finalCosts, viewWeights);
		if (instrumentParams.exactPixels && instrumentParams.exactLogicalStateIndex >= 0 &&
			instrumentParams.exactLogicalStateIndex < instrumentParams.numLogicalStates && instrumentParams.area > 0) {
			// Default construction supplies schema sentinels before available fields are filled.
			PatchMatchInstrumentExactPixel record;
			record.candidateTestedMask = exactTestedMask;
			record.candidateFiniteMask = exactFiniteMask;
			record.candidateAcceptedMask = exactAcceptedMask;
			record.selectedViewsBefore = instrumentSelectedViewsBefore;
			record.selectedViewsAfter = selectedViews[idx];
			record.storedCostBefore = instrumentStoredCostBefore;
			record.incumbentCost = exactIncumbentCost;
			record.winnerCost = cost;
			record.runnerUpCost = confidenceGap >= 0.f ? exactRunnerUpCost : -1.f;
			record.winnerRunnerUpGap = confidenceGap;
			record.source = (uint8_t)updateSource;
			record.winnerSlot = exactWinnerSlot;
			record.runnerUpSlot = confidenceGap >= 0.f ? exactRunnerUpSlot : PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
			record.testedCount = exactTestedCount;
			record.finiteCount = exactFiniteCount;
			record.acceptedCount = exactAcceptedCount;
			record.selectedCountBefore = (uint8_t)CountSelectedViews(instrumentSelectedViewsBefore);
			record.selectedCountAfter = (uint8_t)CountSelectedViews(selectedViews[idx]);
			const size_t exactIndex((size_t)instrumentParams.exactLogicalStateIndex * instrumentParams.area + idx);
			instrumentParams.exactPixels[exactIndex] = record;
		}
	}
#endif
}

// compute the score of the current plane estimate
#ifdef _USE_DMAP_INSTRUMENTATION
template <bool GEOM, bool INSTRUMENT, bool APD>
__device__ void InitializePixelScore(
	const ImagePixels *images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews,
	uint8_t* apdViewWeights,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const Point2i& p)
#else
template <bool GEOM, bool APD>
__device__ void InitializePixelScore(
	const ImagePixels *images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates,
	unsigned* selectedViews, uint8_t* apdViewWeights, const Point2i& p)
#endif
{
	const int width = g_cameras[0].size.x();
	const int height = g_cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	float lowDepth = 0;
	if (g_params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// reference-patch state is invariant across views and hypotheses; cache once
	RefPatchCache refCache;
	ComputeRefPatchCache(images[0], p, refCache);
	// initialize estimate randomly if not set
	RandState* randState = &randStates[idx];
	curand_init(1234/*threadIdx.x*/, p.y(), p.x(), randState);
	Point4& plane = planes[idx];
	float depth = plane.w();
	if (depth <= 0.f) {
		// generate random plane
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(g_cameras[0], p, randState);
		plane.w() = curand_uniform(randState) * (g_params.fDepthMax - g_params.fDepthMin) + g_params.fDepthMin;
	} else if (plane.topLeftCorner<3,1>().dot(g_cameras[0].model.ViewDirection(p)) >= 0.f) {
		// generate random normal
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(g_cameras[0], p, randState);
	}
	// compute costs
	const int nNumViews = g_params.nNumViews;
	const int nInitTopK = g_params.nInitTopK;
	float costVector[MAX_VIEWS];
	MultiViewScorePlane<GEOM>(refCache, images, depthImages, p, plane, lowDepth, costVector);
	// select best views
	float costVectorSorted[MAX_VIEWS];
	Sort(costVector, costVectorSorted, nNumViews);
	float cost = 0.f;
	for (int i = 0; i < nInitTopK; ++i)
		cost += costVectorSorted[i];
	const float costThreshold = costVectorSorted[nInitTopK - 1];
	unsigned& selectedView = selectedViews[idx];
	selectedView = 0;
	for (int imgId = 0; imgId < nNumViews; ++imgId)
		if (costVector[imgId] <= costThreshold)
			SetBit(selectedView, imgId);
	costs[idx] = cost / nInitTopK;
	if constexpr (APD) {
		if (apdViewWeights) {
			uint8_t* pixelViewWeights(apdViewWeights+(size_t)idx*MAX_VIEWS);
			for (int view=0; view<MAX_VIEWS; ++view)
				pixelViewWeights[view] = view < nNumViews &&
					APDCostRank(costVector, view, nNumViews) < static_cast<unsigned>(nInitTopK) ? 1u : 0u;
		}
	}
#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT) {
		unsigned contributionWeights[MAX_VIEWS] = {};
		for (int view = 0; view < nNumViews; ++view)
			if (ExactCostRank(costVector, view, nNumViews) < (unsigned)nInitTopK)
				contributionWeights[view] = 1u;
		PatchMatchInstrumentCostComponents exactComponents(WriteExactInitializationViews<GEOM>(
			refCache, images, depthImages, p, plane, lowDepth, costVector,
			selectedView, nInitTopK, instrumentParams, idx));
		exactComponents.totalCost = costs[idx];
		const int selectedCount = CountSelectedViews(selectedView);
		const float viewEntropy = selectedCount > 1 ? 1.f : 0.f;
		InstrumentCandidate(
			instrumentCounters, instrumentParams, PM_CANDIDATE_INIT, costs[idx], true);
		InstrumentPixel(
			instrumentCounters, instrumentTraceRecords, instrumentTraceMap, updateSources,
			instrumentParams, idx, p,
			plane.w(), plane.w(), costs[idx], costs[idx],
			exactComponents, refCache.varRef, lowDepth,
			0u, selectedView, PM_SOURCE_INIT,
			0.f, 0.f, 0.f, viewEntropy, -1.f, nullptr, contributionWeights);
		if (instrumentParams.exactPixels && instrumentParams.exactLogicalStateIndex == 0 &&
			instrumentParams.numLogicalStates > 0 && instrumentParams.area > 0) {
			PatchMatchInstrumentExactPixel record;
			record.candidateTestedMask = 1u << PM_EXACT_CANDIDATE_CURRENT;
			record.candidateFiniteMask = IsFiniteCandidateCost(costs[idx]) ?
				(1u << PM_EXACT_CANDIDATE_CURRENT) : 0u;
			record.candidateAcceptedMask = 1u << PM_EXACT_CANDIDATE_CURRENT;
			record.selectedViewsBefore = 0u;
			record.selectedViewsAfter = selectedView;
			record.storedCostBefore = -1.f;
			record.incumbentCost = costs[idx];
			record.winnerCost = costs[idx];
			record.runnerUpCost = -1.f;
			record.winnerRunnerUpGap = -1.f;
			record.source = PM_SOURCE_INIT;
			record.winnerSlot = PM_EXACT_CANDIDATE_CURRENT;
			record.runnerUpSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
			record.testedCount = 1;
			record.finiteCount = IsFiniteCandidateCost(costs[idx]) ? 1 : 0;
			record.acceptedCount = 1;
			record.selectedCountBefore = 0;
			record.selectedCountAfter = (uint8_t)selectedCount;
			instrumentParams.exactPixels[idx] = record;
		}
	}
#endif
}

// Keep the disabled production path identical to stabilized b522455b's
// ProcessPixel. APD/DVP use the experimental template above independently.
template <bool GEOM>
__device__ void ProcessPixelNative(const ImagePixels* images, const ImagePixels* depthImages, Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews, const Point2i& p, const int iter)
{
	const int width = g_cameras[0].size.x();
	const int height = g_cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	RandState* randState = &randStates[idx];
	float lowDepth = 0;
	if (g_params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// reference-patch state is invariant across views and hypotheses; cache once
	RefPatchCache refCache;
	ComputeRefPatchCache(images[0], p, refCache);

	// adaptive sampling: 0 up-near, 1 down-near, 2 left-near, 3 right-near, 4 up-far, 5 down-far, 6 left-far, 7 right-far
	static constexpr int2 dirs[8][11] = {
		{{ 0,-1},{-1,-2},{ 1,-2},{-2,-3},{ 2,-3},{-3,-4},{ 3,-4}},
		{{ 0, 1},{-1, 2},{ 1, 2},{-2, 3},{ 2, 3},{-3, 4},{ 3, 4}},
		{{-1, 0},{-2,-1},{-2, 1},{-3,-2},{-3, 2},{-4,-3},{-4, 3}},
		{{ 1, 0},{ 2,-1},{ 2, 1},{ 3,-2},{ 3, 2},{ 4,-3},{ 4, 3}},
		{{0,-3},{0,-5},{0,-7},{0,-9},{0,-11},{0,-13},{0,-15},{0,-17},{0,-19},{0,-21},{0,-23}},
		{{0, 3},{0, 5},{0, 7},{0, 9},{0, 11},{0, 13},{0, 15},{0, 17},{0, 19},{0, 21},{0, 23}},
		{{-3,0},{-5,0},{-7,0},{-9,0},{-11,0},{-13,0},{-15,0},{-17,0},{-19,0},{-21,0},{-23,0}},
		{{ 3,0},{ 5,0},{ 7,0},{ 9,0},{ 11,0},{ 13,0},{ 15,0},{ 17,0},{ 19,0},{ 21,0},{ 23,0}}
	};
	static constexpr int numDirs[8] = {7, 7, 7, 7, 11, 11, 11, 11};
	const int neighborPositions[4] = {
		idx - width,
		idx + width,
		idx - 1,
		idx + 1,
	};
	bool valid[8] = {false, false, false, false, false, false, false, false};
	int positions[8];
	float neighborDepths[8];
	float costArray[8][MAX_VIEWS];

	for (int posId=0; posId<8; ++posId) {
		const int2* samples = dirs[posId];
		Point2i bestNx; float bestConf(FLT_MAX);
		for (int dirId=0; dirId<numDirs[posId]; ++dirId) {
			const int2& offset = samples[dirId];
			const Point2i np(p.x()+offset.x, p.y()+offset.y);
			if (!(np.x()>=0 && np.y()>=0 && np.x()<width && np.y()<height))
				continue;
			const int nidx = Point2Idx(np, width);
			const float nconf = costs[nidx];
			if (bestConf > nconf) {
				bestNx = np;
				bestConf = nconf;
			}
		}
		if (bestConf < FLT_MAX) {
			valid[posId] = true;
			positions[posId] = Point2Idx(bestNx, width);
			neighborDepths[posId] = MultiViewScoreNeighborPlane<GEOM>(refCache, images, depthImages, p, bestNx, LoadPlaneLDG(&planes[positions[posId]]), lowDepth, costArray[posId]);
		}
	}

	// multi-hypothesis view selection
	float viewSelectionPriors[MAX_VIEWS] = {};
	const int nNumViews = g_params.nNumViews;
	for (int posId = 0; posId < 4; ++posId) {
		if (valid[posId]) {
			const unsigned selectedView = selectedViews[neighborPositions[posId]];
			for (int j = 0; j < nNumViews; ++j)
				viewSelectionPriors[j] += (IsBitSet(selectedView, j) ? 0.9f : 0.1f);
		}
	}
	float samplingProbs[MAX_VIEWS];
	const float thCost = 0.8f * __expf(Square((float)iter) / (-2.f * 4.f*4.f));
	for (int imgId = 0; imgId < nNumViews; ++imgId) {
		float sumW = 0;
		unsigned count = 0;
		unsigned countBad = 0;
		for (int posId = 0; posId < 8; posId++) {
			if (valid[posId]) {
				if (costArray[posId][imgId] < thCost) {
					sumW += __expf(Square(costArray[posId][imgId]) / (-2.f * 0.3f*0.3f));
					++count;
				// Exact fBadCost marks an unavailable evaluation and remains neutral here.
				} else if (costArray[posId][imgId] > fBadCost) {
					++countBad;
				}
			}
		}
		if (count > 2 && countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * sumW / count;
		} else if (countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * __expf(Square(thCost) / (-2.f * 0.4f*0.4f));
		} else {
			samplingProbs[imgId] = 0.f;
		}
	}
	PDF2CDF(samplingProbs, nNumViews);
	unsigned viewWeights[MAX_VIEWS] = {};
	for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
		const float randProb = curand_uniform(randState);
		for (int imgId = 0; imgId < nNumViews; ++imgId) {
			if (samplingProbs[imgId] > randProb) {
				++viewWeights[imgId];
				break;
			}
		}
	}

	// propagate best neighbor plane
	Point4& plane = planes[idx];
	float& cost = costs[idx];
	unsigned newSelectedViews = 0;
	for (int imgId = 0; imgId < nNumViews; ++imgId)
		if (viewWeights[imgId])
			SetBit(newSelectedViews, imgId);
	float finalCosts[8];
	for (int posId = 0; posId < 8; ++posId)
		finalCosts[posId] = valid[posId] ? AggregateMultiViewScores(viewWeights, costArray[posId], nNumViews) : FLT_MAX;
	const int minCostIdx = FindMinIndex(finalCosts, 8);
	float costVector[MAX_VIEWS];
	MultiViewScorePlane<GEOM>(refCache, images, depthImages, p, plane, lowDepth, costVector);
	cost = AggregateMultiViewScores(viewWeights, costVector, nNumViews);
	if (finalCosts[minCostIdx] < cost) {
		ASSERT(valid[minCostIdx]);
		plane = LoadPlaneLDG(&planes[positions[minCostIdx]]);
		plane.w() = neighborDepths[minCostIdx];
		cost = finalCosts[minCostIdx];
		selectedViews[idx] = newSelectedViews;
	}
	const float depth = plane.w();

	// refine estimate
	constexpr float perturbationDepth = 0.005f;
	constexpr float perturbationNormal = 0.01f * (float)M_PI;
	const float depthPerturbed = g_params.bCompat23
		? GeneratePerturbedDepth23(depth, randState, perturbationDepth)
		: GeneratePerturbedDepth(depth, randState, perturbationDepth);
	const Point3 perturbedNormal = g_params.bCompat23
		? GeneratePerturbedNormal23(g_cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal)
		: GeneratePerturbedNormal(g_cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal);
	const Point3 normalRand = GenerateRandomNormal(g_cameras[0], p, randState);
	int numValidPlanes = 3;
	Point3 surfaceNormal = Point3::Zero();
	if (valid[0] && valid[1] && valid[2] && valid[3]) {
		// estimate normal from surrounding surface
		const Point4 ndepths(
			LoadPlaneWLDG(&planes[neighborPositions[0]]),
			LoadPlaneWLDG(&planes[neighborPositions[1]]),
			LoadPlaneWLDG(&planes[neighborPositions[2]]),
			LoadPlaneWLDG(&planes[neighborPositions[3]])
		);
		surfaceNormal = ComputeDepthGradient(g_cameras[0].model, depth, p, ndepths);
		numValidPlanes = 4;
	}
	constexpr int numPlanes = 4;
	const float depths[numPlanes] = {depthPerturbed, depth, depth, depth};
	const Point3 normals[numPlanes] = {plane.topLeftCorner<3,1>(), perturbedNormal, normalRand, surfaceNormal};
	for (int i = 0; i < numValidPlanes; ++i) {
		Point4 newPlane;
		newPlane.topLeftCorner<3,1>() = normals[i];
		newPlane.w() = depths[i];
		MultiViewScorePlane<GEOM>(refCache, images, depthImages, p, newPlane, lowDepth, costVector);
		const float costPlane = AggregateMultiViewScores(viewWeights, costVector, nNumViews);
		if (cost > costPlane) {
			cost = costPlane;
			plane = newPlane;
		}
	}
}

// kernels are GEOM-templated; nvcc emits separate binaries with the
// geom-consistency loop eliminated when off; runtime params come from
// __constant__ g_params (uploaded per pyramid level by UploadParams())
template <bool GEOM>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void InitializeScore(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews)
{
	const Point2i p = GetThreadIndex2();
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	InitializePixelScore<GEOM, false, false>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p);
#else
	InitializePixelScore<GEOM, false>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, nullptr, p);
#endif
}

// traverse image in a back/red checkerboard pattern
template <bool GEOM, bool DVP = false>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void BlackPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 0 : 1);
	if constexpr (!DVP) {
		ProcessPixelNative<GEOM>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
			planes, lowDepths, costs, (RandState*)randStates, selectedViews, p, iter);
		return;
	}
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false, false, DVP>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr,
		APDUpdateStage::ALL,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM, false, DVP>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, APDUpdateStage::ALL, p, iter);
#endif
}
template <bool GEOM, bool DVP = false>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void RedPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 1 : 0);
	if constexpr (!DVP) {
		ProcessPixelNative<GEOM>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
			planes, lowDepths, costs, (RandState*)randStates, selectedViews, p, iter);
		return;
	}
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false, false, DVP>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr,
		APDUpdateStage::ALL,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM, false, DVP>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, APDUpdateStage::ALL, p, iter);
#endif
}

template <bool GEOM>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void InitializeScoreAPD(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, uint8_t* apdViewWeights)
{
	const Point2i p(GetThreadIndex2());
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	InitializePixelScore<GEOM, false, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews, apdViewWeights,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p);
#else
	InitializePixelScore<GEOM, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews, apdViewWeights, p);
#endif
}

template <bool GEOM, bool DVP = false, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__global__ PATCHMATCHCUDA_APD_LAUNCH_BOUNDS(VISIBLE_NORMAL) void BlackPixelProcessAPD(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext,
	APDUpdateStage apdStage, const int iter)
{
	Point2i p(GetThreadIndex2());
	p.y() = p.y()*2+(threadIdx.x%2 == 0 ? 0 : 1);
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage, p, iter);
#endif
}

template <bool GEOM, bool DVP = false, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__global__ PATCHMATCHCUDA_APD_LAUNCH_BOUNDS(VISIBLE_NORMAL) void RedPixelProcessAPD(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext,
	APDUpdateStage apdStage, const int iter)
{
	Point2i p(GetThreadIndex2());
	p.y() = p.y()*2+(threadIdx.x%2 == 0 ? 1 : 0);
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage, p, iter);
#endif
}

#ifdef _USE_DMAP_INSTRUMENTATION
template <bool GEOM>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void InitializeScoreAPDInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, uint8_t* apdViewWeights,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams)
{
	const Point2i p(GetThreadIndex2());
	InitializePixelScore<GEOM, true, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews, apdViewWeights,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap,
		instrumentParams, p);
}

template <bool GEOM, bool DVP = false, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void BlackPixelProcessAPDInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext, APDUpdateStage apdStage,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams,
	const int iter)
{
	Point2i p(GetThreadIndex2());
	p.y() = p.y()*2+(threadIdx.x%2 == 0 ? 0 : 1);
	ProcessPixel<GEOM, true, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap,
		instrumentParams, p, iter);
}

template <bool GEOM, bool DVP = false, bool VISIBILITY = false,
	bool VISIBLE_NORMAL = false>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void RedPixelProcessAPDInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates,
	unsigned* selectedViews, const uint8_t* apdReliability, const uint32_t* apdAnchors,
	const uint8_t* apdAnchorCounts, const Point4* apdPlanesSnapshot,
	const unsigned* apdSelectedViewsSnapshot, const Point4* apdFittedPlanes,
	const uint8_t* apdFittedPlaneValid, uint8_t* apdViewWeights,
	const uint8_t* visibilityPrevious, const uint8_t* visibilityResolved,
	uint8_t* visibilityNext, APDUpdateStage apdStage,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams,
	const int iter)
{
	Point2i p(GetThreadIndex2());
	p.y() = p.y()*2+(threadIdx.x%2 == 0 ? 1 : 0);
	ProcessPixel<GEOM, true, true, DVP, VISIBILITY, VISIBLE_NORMAL>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		apdReliability, apdAnchors, apdAnchorCounts, apdPlanesSnapshot,
		apdSelectedViewsSnapshot, apdFittedPlanes, apdFittedPlaneValid, apdViewWeights,
		visibilityPrevious, visibilityResolved, visibilityNext,
		apdStage,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap,
		instrumentParams, p, iter);
}
#endif

#ifdef _USE_DMAP_INSTRUMENTATION
__device__ inline void CaptureAPDProfileInstrumentation(
	const Point2i& p,
	const int pixelIndex,
	const unsigned logicalIteration,
	const unsigned eta,
	const float averageBaseline,
	const float currentDisparity,
	const APDProfileSummary& summary,
	const float* profile,
	const uint8_t* viewWeights,
	PatchMatchAPDInstrumentCounters* counters,
	PatchMatchAPDInstrumentState* states,
	PatchMatchAPDInstrumentTrace* traces,
	const int32_t* traceMap,
	const PatchMatchInstrumentKernelParams& params)
{
	if (logicalIteration >= static_cast<unsigned>(max(params.numLogicalStates-1, 0)) ||
		params.area <= 0)
		return;
	PatchMatchAPDInstrumentState state;
	state.averageBaseline = averageBaseline;
	state.currentDisparity = currentDisparity;
	state.globalMinimumCost = summary.globalMinimumIndex < APD_PROFILE_SIZE ?
		summary.globalMinimumCost : -1.f;
	state.separation = summary.localMinimumCount > 1u ? summary.separation : -1.f;
	state.globalMinimumOffset = static_cast<int16_t>(summary.globalMinimumOffset);
	state.reliability = static_cast<uint8_t>(summary.reliability);
	state.profileReason = static_cast<uint8_t>(summary.reason);
	state.eta = static_cast<uint8_t>(min(eta, 255u));
	state.finiteCount = static_cast<uint8_t>(min(summary.finiteCount, 255u));
	state.localMinimumCount = static_cast<uint8_t>(min(summary.localMinimumCount, 255u));
	state.globalMinimumPlateauStart = static_cast<uint8_t>(
		min(summary.globalMinimumPlateauStart, 255u));
	state.globalMinimumPlateauEnd = static_cast<uint8_t>(
		min(summary.globalMinimumPlateauEnd, 255u));
	const size_t stateIndex((size_t)logicalIteration*params.area+pixelIndex);
	if (states)
		states[stateIndex] = state;
	if (counters) {
		PatchMatchAPDInstrumentCounters& counter(counters[logicalIteration]);
		atomicAdd(&counter.classified, 1u);
		if (state.reliability < 3u)
			atomicAdd(&counter.reliability[state.reliability], 1u);
		if (state.profileReason < PM_APD_INSTRUMENT_PROFILE_REASONS)
			atomicAdd(&counter.profileReason[state.profileReason], 1u);
		if (state.globalMinimumCost >= 0.f) {
			InstrumentAtomicAdd(&counter.globalMinimumCostSum, state.globalMinimumCost);
			atomicAdd(&counter.globalMinimumCostSamples, 1u);
		}
		if (state.separation >= 0.f) {
			InstrumentAtomicAdd(&counter.separationSum, state.separation);
			atomicAdd(&counter.separationSamples, 1u);
		}
	}
	if (!traces || !traceMap || params.numTracePixels <= 0)
		return;
	const int traceIndex(traceMap[pixelIndex]);
	if (traceIndex < 0 || traceIndex >= params.numTracePixels)
		return;
	PatchMatchAPDInstrumentTrace& trace(
		traces[(size_t)logicalIteration*params.numTracePixels+traceIndex]);
	trace.valid = 1;
	trace.imageID = params.imageID;
	trace.scaleNumber = params.scaleNumber;
	trace.logicalIteration = static_cast<int32_t>(logicalIteration);
	trace.x = p.x();
	trace.y = p.y();
	trace.state = state;
	for (unsigned sample=0; sample<APD_PROFILE_SIZE; ++sample)
		trace.profile[sample] = profile ? profile[sample] : -1.f;
	for (int view=0; view<PM_INSTRUMENT_MAX_VIEWS; ++view) {
		trace.viewWeights[view] = viewWeights && view < g_params.nNumViews ? viewWeights[view] : 0u;
		trace.viewCenterCosts[view] = -1.f;
		trace.viewAnchorMeanCosts[view] = -1.f;
		trace.viewWorkingCosts[view] = -1.f;
		trace.viewSelectionPriors[view] = -1.f;
		trace.viewSamplingScores[view] = -1.f;
		trace.viewSamplingProbabilities[view] = -1.f;
	}
	for (int sector=0; sector<PM_APD_INSTRUMENT_SECTORS; ++sector)
		trace.sectorCandidates[sector] = ~uint32_t(0);
	for (int slot=0; slot<PM_APD_INSTRUMENT_ANCHORS; ++slot) {
		trace.anchors[slot] = ~uint32_t(0);
		trace.anchorResiduals[slot] = -1.f;
		trace.anchorCandidateWorkingCosts[slot] = -1.f;
		trace.anchorCandidateNativeCosts[slot] = -1.f;
		trace.anchorSelectedViews[slot] = 0u;
		trace.anchorCandidateValid[slot] = 0u;
	}
}
#endif

// Native PatchMatch retains only the final selected-view mask. Expanding it to
// uniform weights preserves the native DMAP while providing a deterministic
// compatibility input for the APD reliability profile; these are not the
// historical Monte-Carlo weights used by the official APD strong kernel.
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void SeedAPDViewWeightsFromSelectedViews(
	const unsigned* selectedViews,
	uint8_t* apdViewWeights)
{
	const Point2i p(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	const unsigned selectedMask(selectedViews[idx]);
	uint8_t* pixelViewWeights(apdViewWeights+(size_t)idx*MAX_VIEWS);
	for (int view=0; view<MAX_VIEWS; ++view)
		pixelViewWeights[view] = view < g_params.nNumViews && IsBitSet(selectedMask, view) ? 1u : 0u;
}

template <bool GEOM, bool INSTRUMENT = false>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void ClassifyAPDProfiles(
	const cudaTextureObject_t* textureImages,
	const cudaTextureObject_t* textureDepths,
	const Point4* planes,
	const float* lowDepths,
	const uint8_t* apdViewWeights,
	uint8_t* reliability,
	#ifdef _USE_DMAP_INSTRUMENTATION
	PatchMatchAPDInstrumentCounters* instrumentCounters,
	PatchMatchAPDInstrumentState* instrumentStates,
	PatchMatchAPDInstrumentTrace* instrumentTraces,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams instrumentParams,
	#endif
	const unsigned eta,
	const unsigned logicalIteration)
{
	const Point2i p(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	reliability[idx] = static_cast<uint8_t>(APDReliabilityClass::UNKNOWN);
	APDProfileSummary summary;
	const Point4 plane(LoadPlaneLDG(&planes[idx]));
	if (!(plane.w() > 0.f) || !isfinite(plane.w()) || !apdViewWeights) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			CaptureAPDProfileInstrumentation(p, idx, logicalIteration, eta, 0.f, 0.f,
				summary, nullptr, nullptr, instrumentCounters, instrumentStates,
				instrumentTraces, instrumentTraceMap, instrumentParams);
		#endif
		return;
	}
	const uint8_t* pixelViewWeights(apdViewWeights+(size_t)idx*MAX_VIEWS);
	float baselineSum(0.f);
	unsigned baselineCount(0u);
	for (int view=0; view<g_params.nNumViews; ++view) {
		if (!pixelViewWeights[view])
			continue;
		const Point3 delta(g_cameras[0].pose.C-g_cameras[view+1].pose.C);
		baselineSum += delta.norm();
		++baselineCount;
	}
	if (baselineCount == 0u) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			CaptureAPDProfileInstrumentation(p, idx, logicalIteration, eta, 0.f, 0.f,
				summary, nullptr, pixelViewWeights, instrumentCounters, instrumentStates,
				instrumentTraces, instrumentTraceMap, instrumentParams);
		#endif
		return;
	}
	const float averageBaseline(baselineSum/static_cast<float>(baselineCount));
	const float focalLength(g_cameras[0].model.f.x());
	const float currentDisparity(focalLength*averageBaseline/plane.w());
	if (!(averageBaseline > 0.f) || !(focalLength > 0.f) ||
		!(currentDisparity > 0.f) || !isfinite(currentDisparity)) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			CaptureAPDProfileInstrumentation(p, idx, logicalIteration, eta,
				averageBaseline, currentDisparity, summary, nullptr, pixelViewWeights,
				instrumentCounters, instrumentStates, instrumentTraces,
				instrumentTraceMap, instrumentParams);
		#endif
		return;
	}
	float lowDepth(0.f);
	if (g_params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	RefPatchCache refCache;
	ComputeRefPatchCache((ImagePixels)textureImages[0], p, refCache);
	float profile[APD_PROFILE_SIZE];
	for (int offset=-static_cast<int>(APD_PROFILE_RADIUS);
		offset<=static_cast<int>(APD_PROFILE_RADIUS); ++offset)
	{
		const unsigned profileIndex(static_cast<unsigned>(
			offset+static_cast<int>(APD_PROFILE_RADIUS)));
		const float disparity(currentDisparity+static_cast<float>(offset));
		if (!(disparity > 0.f)) {
			profile[profileIndex] = fBadCost;
			continue;
		}
		const float depth(focalLength*averageBaseline/disparity);
		if (!(depth >= g_params.fDepthMin && depth <= g_params.fDepthMax) || !isfinite(depth)) {
			profile[profileIndex] = fBadCost;
			continue;
		}
		Point4 hypothesis(plane);
		hypothesis.w() = depth;
		float viewCosts[MAX_VIEWS];
		MultiViewScorePlane<GEOM>(refCache,
			(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
			p, hypothesis, lowDepth, viewCosts);
		APDWeightedCostAccumulator aggregate;
		for (int view=0; view<g_params.nNumViews; ++view)
			AccumulateAPDViewCost(
				aggregate, pixelViewWeights[view] != 0u,
				static_cast<float>(pixelViewWeights[view]), viewCosts[view],
				GEOM ? APD_MAX_NCC_COST+0.4f : APD_MAX_NCC_COST);
		profile[profileIndex] = min(APD_MAX_NCC_COST, FinishAPDViewCost(aggregate));
	}
	summary = SummarizeAPDProfile(
		profile, APD_PROFILE_SIZE, eta, APDSeparationConvention::PAPER);
	reliability[idx] = static_cast<uint8_t>(summary.reliability);
	#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT)
		CaptureAPDProfileInstrumentation(p, idx, logicalIteration, eta,
			averageBaseline, currentDisparity, summary, profile, pixelViewWeights,
			instrumentCounters, instrumentStates, instrumentTraces,
			instrumentTraceMap, instrumentParams);
	#endif
}

__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void FindAPDNearestReliableRows(
	const uint8_t* reliability,
	float* rowSquaredDistances,
	int* rowNearestX,
	const int width,
	const int height)
{
	const Point2i p(GetThreadIndex2());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	float bestDistance(FLT_MAX);
	int bestX(-1);
	const int begin(max(0, p.x()-APD_NEAREST_SEARCH_RADIUS));
	const int end(min(width-1, p.x()+APD_NEAREST_SEARCH_RADIUS));
	for (int x=begin; x<=end; ++x) {
		if (reliability[Point2Idx(Point2i(x, p.y()), width)] !=
			static_cast<uint8_t>(APDReliabilityClass::RELIABLE))
			continue;
		const float distance(static_cast<float>((x-p.x())*(x-p.x())));
		if (distance < bestDistance || (distance == bestDistance && x < bestX)) {
			bestDistance = distance;
			bestX = x;
		}
	}
	rowSquaredDistances[idx] = bestDistance;
	rowNearestX[idx] = bestX;
}

__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void FindAPDNearestReliableColumns(
	const float* rowSquaredDistances,
	const int* rowNearestX,
	uint32_t* nearestReliable,
	const int width,
	const int height)
{
	const Point2i p(GetThreadIndex2());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	float bestDistance(FLT_MAX);
	uint32_t bestIndex(~uint32_t(0));
	const int begin(max(0, p.y()-APD_NEAREST_SEARCH_RADIUS));
	const int end(min(height-1, p.y()+APD_NEAREST_SEARCH_RADIUS));
	for (int y=begin; y<=end; ++y) {
		const int rowIndex(Point2Idx(Point2i(p.x(), y), width));
		const int nearestX(rowNearestX[rowIndex]);
		if (nearestX < 0)
			continue;
		const float dy(static_cast<float>(y-p.y()));
		const float distance(rowSquaredDistances[rowIndex]+dy*dy);
		const uint32_t candidate(static_cast<uint32_t>(Point2Idx(Point2i(nearestX, y), width)));
		if (distance < bestDistance || (distance == bestDistance && candidate < bestIndex)) {
			bestDistance = distance;
			bestIndex = candidate;
		}
	}
	nearestReliable[idx] = bestIndex;
}

__device__ inline APDPoint3 APDCameraPoint(const Point2i& pixel, const float depth)
{
	const Point3 point(g_cameras[0].TransformPointI2W(pixel.cast<float>(), depth));
	return APDPoint3{point.x(), point.y(), point.z()};
}

#ifdef _USE_DMAP_INSTRUMENTATION
__device__ inline void CaptureAPDAnchorInstrumentation(
	const Point2i& p,
	const int pixelIndex,
	const int width,
	const unsigned logicalIteration,
	const uint32_t nearestReliable,
	const PatchMatchAPDInstrumentAnchorReason reason,
	const unsigned candidateCount,
	const APDModelQuality& quality,
	const float threshold,
	const unsigned anchorCount,
	const uint32_t* sectorCandidates,
	const uint32_t* anchors,
	const float* anchorResiduals,
	const Point4* fittedPlanes,
	const uint8_t* fittedPlaneValid,
	PatchMatchAPDInstrumentCounters* counters,
	PatchMatchAPDInstrumentState* states,
	PatchMatchAPDInstrumentTrace* traces,
	const int32_t* traceMap,
	const PatchMatchInstrumentKernelParams& params)
{
	if (logicalIteration >= static_cast<unsigned>(max(params.numLogicalStates-1, 0)) ||
		params.area <= 0)
		return;
	const size_t stateIndex((size_t)logicalIteration*params.area+pixelIndex);
	const int traceIndex(traceMap && params.numTracePixels > 0 ? traceMap[pixelIndex] : -1);
	PatchMatchAPDInstrumentTrace* trace(
		traces && traceIndex >= 0 && traceIndex < params.numTracePixels ?
		&traces[(size_t)logicalIteration*params.numTracePixels+traceIndex] : nullptr);
	PatchMatchAPDInstrumentState state(states ? states[stateIndex] :
		(trace ? trace->state : PatchMatchAPDInstrumentState{}));
	state.nearestReliable = nearestReliable;
	if (nearestReliable < static_cast<uint32_t>(params.area)) {
		const int nearestX(static_cast<int>(nearestReliable%static_cast<uint32_t>(width)));
		const int nearestY(static_cast<int>(nearestReliable/static_cast<uint32_t>(width)));
		const float dx(static_cast<float>(nearestX-p.x()));
		const float dy(static_cast<float>(nearestY-p.y()));
		state.nearestReliableDistance = sqrtf(dx*dx+dy*dy);
	}
	state.ransacThreshold = threshold;
	state.ransacCenterResidual = quality.valid ? quality.centerResidual : -1.f;
	state.ransacMeanInlierResidual = quality.valid ? quality.meanInlierResidual : -1.f;
	state.ransacSamplePacked = quality.valid ?
		(quality.sample.first | (quality.sample.second << 8) | (quality.sample.third << 16)) :
		~uint32_t(0);
	state.candidateCount = static_cast<uint8_t>(min(candidateCount, 255u));
	state.inlierCount = static_cast<uint8_t>(min(quality.inlierCount, 255u));
	state.outlierCount = static_cast<uint8_t>(min(quality.outlierCount, 255u));
	state.anchorCount = static_cast<uint8_t>(min(anchorCount, APD_MAX_ANCHORS));
	state.anchorReason = static_cast<uint8_t>(reason);
	state.ransacValid = quality.valid ? 1u : 0u;
	state.deformableEligible = anchorCount >= APD_MIN_INLIERS ? 1u : 0u;
	state.fittedPlaneValid = fittedPlaneValid && fittedPlaneValid[pixelIndex] ? 1u : 0u;
	state.fittedPlaneDepth = state.fittedPlaneValid && fittedPlanes ?
		fittedPlanes[pixelIndex].w() : -1.f;
	if (states)
		states[stateIndex] = state;
	if (trace) {
		trace->state = state;
		for (unsigned sector=0; sector<APD_SECTOR_COUNT; ++sector)
			trace->sectorCandidates[sector] = sectorCandidates ?
				sectorCandidates[sector] : ~uint32_t(0);
		for (unsigned slot=0; slot<APD_MAX_ANCHORS; ++slot) {
			trace->anchors[slot] = anchors ? anchors[slot] : ~uint32_t(0);
			trace->anchorResiduals[slot] = anchorResiduals ? anchorResiduals[slot] : -1.f;
		}
	}
	if (counters) {
		PatchMatchAPDInstrumentCounters& counter(counters[logicalIteration]);
		if (state.anchorReason < PM_APD_INSTRUMENT_ANCHOR_REASONS)
			atomicAdd(&counter.anchorReason[state.anchorReason], 1u);
		atomicAdd(&counter.anchorCountBins[min(anchorCount, APD_MAX_ANCHORS)], 1u);
		InstrumentAtomicAdd(&counter.anchorCountSum, static_cast<double>(anchorCount));
		if (quality.valid)
			atomicAdd(&counter.ransacValid, 1u);
		if (state.deformableEligible)
			atomicAdd(&counter.deformableEligible, 1u);
	}
}
#endif

template <bool INSTRUMENT = false>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void BuildAPDAnchors(
	const Point4* planes,
	const uint8_t* reliability,
	const uint32_t* nearestReliable,
	uint32_t* anchors,
	uint8_t* anchorCounts,
	Point4* fittedPlanes,
	uint8_t* fittedPlaneValid,
	#ifdef _USE_DMAP_INSTRUMENTATION
	PatchMatchAPDInstrumentCounters* instrumentCounters,
	PatchMatchAPDInstrumentState* instrumentStates,
	PatchMatchAPDInstrumentTrace* instrumentTraces,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams instrumentParams,
	#endif
	const float ransacThreshold,
	const uint32_t stageSeed,
	const unsigned logicalIteration)
{
	const Point2i p(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	uint32_t* pixelAnchors(anchors+(size_t)idx*APD_MAX_ANCHORS);
	for (unsigned slot=0; slot<APD_MAX_ANCHORS; ++slot)
		pixelAnchors[slot] = ~uint32_t(0);
	anchorCounts[idx] = 0u;
	if (fittedPlanes)
		fittedPlanes[idx] = Point4::Zero();
	if (fittedPlaneValid)
		fittedPlaneValid[idx] = 0u;
	#ifdef _USE_DMAP_INSTRUMENTATION
	uint32_t instrumentSectorCandidates[APD_SECTOR_COUNT];
	float instrumentAnchorResiduals[APD_MAX_ANCHORS];
	if constexpr (INSTRUMENT) {
		for (unsigned sector=0; sector<APD_SECTOR_COUNT; ++sector)
			instrumentSectorCandidates[sector] = ~uint32_t(0);
		for (unsigned slot=0; slot<APD_MAX_ANCHORS; ++slot)
			instrumentAnchorResiduals[slot] = -1.f;
	}
	#endif
	if (reliability[idx] != static_cast<uint8_t>(APDReliabilityClass::UNRELIABLE)) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT) {
			const APDModelQuality quality;
			CaptureAPDAnchorInstrumentation(p, idx, width, logicalIteration,
				nearestReliable[idx], PM_APD_ANCHOR_PIXEL_NOT_UNRELIABLE, 0u, quality,
				-1.f, 0u, instrumentSectorCandidates, pixelAnchors,
				instrumentAnchorResiduals, fittedPlanes, fittedPlaneValid,
				instrumentCounters, instrumentStates,
				instrumentTraces, instrumentTraceMap, instrumentParams);
		}
		#endif
		return;
	}
	if (!(planes[idx].w() > 0.f)) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT) {
			const APDModelQuality quality;
			CaptureAPDAnchorInstrumentation(p, idx, width, logicalIteration,
				nearestReliable[idx], PM_APD_ANCHOR_INVALID_CENTER_DEPTH, 0u, quality,
				-1.f, 0u, instrumentSectorCandidates, pixelAnchors,
				instrumentAnchorResiduals, fittedPlanes, fittedPlaneValid,
				instrumentCounters, instrumentStates,
				instrumentTraces, instrumentTraceMap, instrumentParams);
		}
		#endif
		return;
	}
	uint32_t candidates[APD_SECTOR_COUNT];
	APDPoint2 candidatePixels[APD_SECTOR_COUNT];
	APDPoint3 candidatePoints[APD_SECTOR_COUNT];
	APDPoint3 candidateCameraPoints[APD_SECTOR_COUNT];
	unsigned candidateCount(0u);
	const int maximumRadius(max(width, height)*2);
	for (unsigned sector=0; sector<APD_SECTOR_COUNT; ++sector) {
		uint32_t selected(~uint32_t(0));
		for (int radius=2; radius<=maximumRadius && selected == ~uint32_t(0);
			radius=min(radius*2, radius+25))
		{
			for (unsigned attempt=0; attempt<APD_SPOKE_ATTEMPTS_PER_RADIUS; ++attempt) {
				const uint32_t randomBits(APDHash(
					static_cast<uint32_t>(idx) ^ APDHash(stageSeed) ^
					APDHash(sector*131u+attempt*17u+static_cast<unsigned>(radius))));
				const float unitOffset(static_cast<float>(randomBits&0x00FFFFFFu)/16777216.f);
				const APDSectorDirection direction(MakeAPDSectorDirection(
					sector, APD_SECTOR_COUNT, unitOffset));
				const int queryX(__float2int_rn(static_cast<float>(p.x())+direction.x*radius));
				const int queryY(__float2int_rn(static_cast<float>(p.y())+direction.y*radius));
				if (queryX < 0 || queryY < 0 || queryX >= width || queryY >= height)
					continue;
				const uint32_t candidate(nearestReliable[Point2Idx(Point2i(queryX, queryY), width)]);
				if (candidate == ~uint32_t(0) || candidate >= static_cast<uint32_t>(width*height) ||
					reliability[candidate] != static_cast<uint8_t>(APDReliabilityClass::RELIABLE) ||
					!(planes[candidate].w() > 0.f))
					continue;
				const Point2i candidatePixel(
					static_cast<int>(candidate%static_cast<uint32_t>(width)),
					static_cast<int>(candidate/static_cast<uint32_t>(width)));
				if (APDSectorForDirection(
					static_cast<float>(candidatePixel.x()-p.x()),
					static_cast<float>(candidatePixel.y()-p.y()), APD_SECTOR_COUNT) != sector)
					continue;
				bool duplicate(false);
				for (unsigned existing=0; existing<candidateCount; ++existing)
					duplicate = duplicate || candidates[existing] == candidate;
				if (duplicate)
					continue;
				selected = candidate;
				break;
			}
		}
		if (selected == ~uint32_t(0))
			continue;
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			instrumentSectorCandidates[sector] = selected;
		#endif
		const Point2i candidatePixel(
			static_cast<int>(selected%static_cast<uint32_t>(width)),
			static_cast<int>(selected/static_cast<uint32_t>(width)));
		candidates[candidateCount] = selected;
		candidatePixels[candidateCount] = APDPoint2{
			static_cast<float>(candidatePixel.x()), static_cast<float>(candidatePixel.y())};
		candidatePoints[candidateCount] = APDCameraPoint(candidatePixel, planes[selected].w());
		const Point3 cameraPoint(g_cameras[0].model.TransformPointI2C(
			candidatePixel.cast<float>(), planes[selected].w()));
		candidateCameraPoints[candidateCount] = APDPoint3{
			cameraPoint.x(), cameraPoint.y(), cameraPoint.z()};
		++candidateCount;
	}
	if (candidateCount < APD_MIN_INLIERS) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT) {
			const APDModelQuality quality;
			CaptureAPDAnchorInstrumentation(p, idx, width, logicalIteration,
				nearestReliable[idx], PM_APD_ANCHOR_INSUFFICIENT_SECTOR_CANDIDATES,
				candidateCount, quality, -1.f, 0u, instrumentSectorCandidates,
				pixelAnchors, instrumentAnchorResiduals, fittedPlanes, fittedPlaneValid,
				instrumentCounters,
				instrumentStates, instrumentTraces, instrumentTraceMap, instrumentParams);
		}
		#endif
		return;
	}
	const APDPoint2 centerPixel{static_cast<float>(p.x()), static_cast<float>(p.y())};
	const APDPoint3 centerPoint(APDCameraPoint(p, planes[idx].w()));
	const float depthRange(g_params.fDepthMax-g_params.fDepthMin);
	const float threshold(ransacThreshold);
	APDModelQuality bestQuality;
	APDPlane bestModel;
	for (unsigned trial=0; trial<APD_RANSAC_TRIALS; ++trial) {
		const APDRansacTriplet sample(MakeAPDRansacTriplet(
			static_cast<uint32_t>(idx), stageSeed, trial, candidateCount));
		if (!sample.valid || !APDTriangleContainsPoint(
			candidatePixels[sample.first], candidatePixels[sample.second],
			candidatePixels[sample.third], centerPixel))
			continue;
		APDPlane model;
		if (!FitAPDPlane(candidatePoints[sample.first], candidatePoints[sample.second],
			candidatePoints[sample.third], model))
			continue;
		unsigned inlierCount(0u);
		float residualSum(0.f);
		for (unsigned candidate=0; candidate<candidateCount; ++candidate) {
			const float residual(APDNormalizedPlaneResidual(
				model, candidatePoints[candidate], depthRange));
			if (residual < threshold) {
				++inlierCount;
				residualSum += residual;
			}
		}
		const float centerResidual(APDNormalizedPlaneResidual(model, centerPoint, depthRange));
		const APDModelQuality quality(MakeAPDModelQuality(
			candidateCount, inlierCount, true, centerResidual,
			inlierCount > 0u ? residualSum/static_cast<float>(inlierCount) : FLT_MAX, sample));
		if (PreferAPDModel(quality, bestQuality)) {
			bestQuality = quality;
			bestModel = model;
		}
	}
	if (!bestQuality.valid) {
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			CaptureAPDAnchorInstrumentation(p, idx, width, logicalIteration,
				nearestReliable[idx], PM_APD_ANCHOR_NO_VALID_RANSAC_MODEL,
				candidateCount, bestQuality, threshold, 0u, instrumentSectorCandidates,
				pixelAnchors, instrumentAnchorResiduals, fittedPlanes, fittedPlaneValid,
				instrumentCounters,
				instrumentStates, instrumentTraces, instrumentTraceMap, instrumentParams);
		#endif
		return;
	}
	bool chosen[APD_SECTOR_COUNT] = {};
	unsigned outputCount(0u);
	for (unsigned slot=0; slot<APD_MAX_ANCHORS; ++slot) {
		APDAnchorRank bestRank;
		unsigned bestCandidate(APD_SECTOR_COUNT);
		for (unsigned candidate=0; candidate<candidateCount; ++candidate) {
			if (chosen[candidate])
				continue;
			const float residual(APDNormalizedPlaneResidual(
				bestModel, candidatePoints[candidate], depthRange));
			const APDAnchorRank rank{residual < threshold, residual, candidate};
			if (PreferAPDAnchor(rank, bestRank)) {
				bestRank = rank;
				bestCandidate = candidate;
			}
		}
		if (bestCandidate >= candidateCount || !bestRank.inlier)
			break;
		chosen[bestCandidate] = true;
		pixelAnchors[outputCount++] = candidates[bestCandidate];
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			instrumentAnchorResiduals[outputCount-1u] = bestRank.normalizedPlaneResidual;
		#endif
	}
	anchorCounts[idx] = static_cast<uint8_t>(outputCount);
	if (outputCount >= APD_MIN_INLIERS && fittedPlanes && fittedPlaneValid) {
		APDPlane cameraModel;
		const APDRansacTriplet& sample(bestQuality.sample);
		if (FitAPDPlane(candidateCameraPoints[sample.first],
				candidateCameraPoints[sample.second], candidateCameraPoints[sample.third],
				cameraModel))
		{
			Point3 normal(cameraModel.x, cameraModel.y, cameraModel.z);
			if (normal.dot(g_cameras[0].model.ViewDirection(p)) >= 0.f) {
				normal = -normal;
				cameraModel.x = -cameraModel.x;
				cameraModel.y = -cameraModel.y;
				cameraModel.z = -cameraModel.z;
				cameraModel.w = -cameraModel.w;
			}
			const Point3 ray(g_cameras[0].model.TransformPointI2C(p.cast<float>(), 1.f));
			float fittedDepth(0.f);
			if (APDPlaneDepthAtRay(cameraModel,
					APDPoint3{ray.x(), ray.y(), ray.z()}, g_params.fDepthMin,
					g_params.fDepthMax, fittedDepth))
			{
				fittedPlanes[idx].topLeftCorner<3,1>() = normal;
				fittedPlanes[idx].w() = fittedDepth;
				fittedPlaneValid[idx] = 1u;
			}
		}
	}
	#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT)
		CaptureAPDAnchorInstrumentation(p, idx, width, logicalIteration,
			nearestReliable[idx], outputCount >= APD_MIN_INLIERS ?
				PM_APD_ANCHOR_READY : PM_APD_ANCHOR_INSUFFICIENT_MODEL_INLIERS,
			candidateCount, bestQuality, threshold, outputCount, instrumentSectorCandidates,
			pixelAnchors, instrumentAnchorResiduals, fittedPlanes, fittedPlaneValid,
			instrumentCounters,
			instrumentStates, instrumentTraces, instrumentTraceMap, instrumentParams);
	#endif
}

// Apply the paper's Eq. 8 only after the native APD anchor set exists. Keeping
// this as a separate active-only kernel leaves the qualified mode-0 anchor
// kernel and its launch sequence unchanged. If filtering changes the set, the
// OpenMVS fitted-plane compatibility proposal is recomputed from retained
// anchors; the retained anchor collection itself is never expanded or replaced.
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void FilterAPDAnchorsByDepthEdgeRegion(
	const Point4* planes,
	const uint16_t* regionLabels,
	uint32_t* anchors,
	uint8_t* anchorCounts,
	Point4* fittedPlanes,
	uint8_t* fittedPlaneValid,
	uint8_t* anchorCountsBefore,
	uint8_t* anchorCountsAfter,
	uint8_t* rejectedAnchorCounts,
	const unsigned mode,
	const float ransacThreshold,
	const uint32_t stageSeed)
{
	const Point2i p(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	const uint32_t area(static_cast<uint32_t>(width*height));
	uint32_t* pixelAnchors(anchors+(size_t)idx*APD_MAX_ANCHORS);
	const unsigned inputCount(min(static_cast<unsigned>(anchorCounts[idx]), APD_MAX_ANCHORS));
	const DVPDepthEdgeFilterSummary summary(FilterDVPDepthEdgeAnchors(
		mode, regionLabels[idx], regionLabels, area, pixelAnchors, inputCount,
		APD_MAX_ANCHORS, ~uint32_t(0)));
	anchorCounts[idx] = summary.outputCount;
	if (anchorCountsBefore)
		anchorCountsBefore[idx] = summary.inputCount;
	if (anchorCountsAfter)
		anchorCountsAfter[idx] = summary.outputCount;
	if (rejectedAnchorCounts)
		rejectedAnchorCounts[idx] = static_cast<uint8_t>(
			summary.inputCount-summary.outputCount);
	if (summary.outputCount == summary.inputCount)
		return;
	if (fittedPlanes)
		fittedPlanes[idx] = Point4::Zero();
	if (fittedPlaneValid)
		fittedPlaneValid[idx] = 0u;
	if (summary.outputCount < APD_MIN_INLIERS || !fittedPlanes || !fittedPlaneValid)
		return;

	APDPoint2 candidatePixels[APD_MAX_ANCHORS];
	APDPoint3 candidatePoints[APD_MAX_ANCHORS];
	APDPoint3 candidateCameraPoints[APD_MAX_ANCHORS];
	for (unsigned candidate=0u; candidate<summary.outputCount; ++candidate) {
		const uint32_t anchor(pixelAnchors[candidate]);
		const Point2i candidatePixel(
			static_cast<int>(anchor%static_cast<uint32_t>(width)),
			static_cast<int>(anchor/static_cast<uint32_t>(width)));
		candidatePixels[candidate] = APDPoint2{
			static_cast<float>(candidatePixel.x()), static_cast<float>(candidatePixel.y())};
		candidatePoints[candidate] = APDCameraPoint(candidatePixel, planes[anchor].w());
		const Point3 cameraPoint(g_cameras[0].model.TransformPointI2C(
			candidatePixel.cast<float>(), planes[anchor].w()));
		candidateCameraPoints[candidate] = APDPoint3{
			cameraPoint.x(), cameraPoint.y(), cameraPoint.z()};
	}
	const APDPoint2 centerPixel{static_cast<float>(p.x()), static_cast<float>(p.y())};
	const APDPoint3 centerPoint(APDCameraPoint(p, planes[idx].w()));
	const float depthRange(g_params.fDepthMax-g_params.fDepthMin);
	APDModelQuality bestQuality;
	for (unsigned trial=0u; trial<APD_RANSAC_TRIALS; ++trial) {
		const APDRansacTriplet sample(MakeAPDRansacTriplet(
			static_cast<uint32_t>(idx), stageSeed^0xD3E7A91Bu, trial,
			summary.outputCount));
		if (!sample.valid || !APDTriangleContainsPoint(
			candidatePixels[sample.first], candidatePixels[sample.second],
			candidatePixels[sample.third], centerPixel))
			continue;
		APDPlane model;
		if (!FitAPDPlane(candidatePoints[sample.first], candidatePoints[sample.second],
			candidatePoints[sample.third], model))
			continue;
		unsigned inlierCount(0u);
		float residualSum(0.f);
		for (unsigned candidate=0u; candidate<summary.outputCount; ++candidate) {
			const float residual(APDNormalizedPlaneResidual(
				model, candidatePoints[candidate], depthRange));
			if (residual < ransacThreshold) {
				++inlierCount;
				residualSum += residual;
			}
		}
		const float centerResidual(APDNormalizedPlaneResidual(model, centerPoint, depthRange));
		const APDModelQuality quality(MakeAPDModelQuality(
			summary.outputCount, inlierCount, true, centerResidual,
			inlierCount ? residualSum/static_cast<float>(inlierCount) : FLT_MAX, sample));
		if (PreferAPDModel(quality, bestQuality))
			bestQuality = quality;
	}
	if (!bestQuality.valid)
		return;
	APDPlane cameraModel;
	const APDRansacTriplet& sample(bestQuality.sample);
	if (!FitAPDPlane(candidateCameraPoints[sample.first],
		candidateCameraPoints[sample.second], candidateCameraPoints[sample.third], cameraModel))
		return;
	Point3 normal(cameraModel.x, cameraModel.y, cameraModel.z);
	if (normal.dot(g_cameras[0].model.ViewDirection(p)) >= 0.f) {
		normal = -normal;
		cameraModel.x = -cameraModel.x;
		cameraModel.y = -cameraModel.y;
		cameraModel.z = -cameraModel.z;
		cameraModel.w = -cameraModel.w;
	}
	const Point3 ray(g_cameras[0].model.TransformPointI2C(p.cast<float>(), 1.f));
	float fittedDepth(0.f);
	if (!APDPlaneDepthAtRay(cameraModel, APDPoint3{ray.x(), ray.y(), ray.z()},
		g_params.fDepthMin, g_params.fDepthMax, fittedDepth))
		return;
	fittedPlanes[idx].topLeftCorner<3,1>() = normal;
	fittedPlanes[idx].w() = fittedDepth;
	fittedPlaneValid[idx] = 1u;
}

template <bool GEOM, bool INSTRUMENT = false, bool VISIBILITY = false>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void FinalRefineAPD(
	const cudaTextureObject_t* textureImages,
	const cudaTextureObject_t* textureDepths,
	Point4* planes,
	const float* lowDepths,
	float* costs,
	const unsigned* selectedViews,
	const uint8_t* apdViewWeights
	#ifdef _USE_DMAP_INSTRUMENTATION
	, uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams
	#endif
)
{
	const Point2i p(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx(Point2Idx(p, width));
	Point4 plane(LoadPlaneLDG(&planes[idx]));
	if (!(plane.w() > 0.f) || !isfinite(plane.w()) || !apdViewWeights)
		return;
	const int numViews(g_params.nNumViews);
	const uint8_t* storedWeights(apdViewWeights+(size_t)idx*MAX_VIEWS);
	unsigned viewWeights[MAX_VIEWS] = {};
	unsigned weightSum(0u);
	float baselineSum(0.f);
	unsigned baselineCount(0u);
	for (int view=0; view<numViews; ++view) {
		viewWeights[view] = storedWeights[view];
		if constexpr (!VISIBILITY)
			if (!viewWeights[view] && IsBitSet(selectedViews[idx], view))
				viewWeights[view] = 1u;
		weightSum += viewWeights[view];
		if (viewWeights[view]) {
			const Point3 delta(g_cameras[0].pose.C-g_cameras[view+1].pose.C);
			baselineSum += delta.norm();
			++baselineCount;
		}
	}
	if (weightSum == 0u || baselineCount == 0u)
		return;
	const float averageBaseline(baselineSum/static_cast<float>(baselineCount));
	const float focalLength(g_cameras[0].model.f.x());
	const float disparity(focalLength*averageBaseline/plane.w());
	if (!(averageBaseline > 0.f) || !(focalLength > 0.f) || !(disparity > 0.f) ||
		!isfinite(disparity))
		return;
	float lowDepth(0.f);
	if (g_params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	RefPatchCache refCache;
	ComputeRefPatchCache((ImagePixels)textureImages[0], p, refCache);
	float incumbentViewCosts[MAX_VIEWS];
	MultiViewScorePlane<GEOM>(refCache,
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		p, plane, lowDepth, incumbentViewCosts);
	const float incumbentCost(AggregateAPDViewScores(
		viewWeights, incumbentViewCosts, numViews));
	float bestCost(incumbentCost);
	float bestDepth(plane.w());
	int bestOffset(0);
	unsigned tested(0u);
	unsigned finite(0u);
	for (int offset=-APD_FINAL_REFINEMENT_RADIUS;
		offset<=APD_FINAL_REFINEMENT_RADIUS; ++offset)
	{
		const float candidateDisparity(disparity+static_cast<float>(offset));
		if (!(candidateDisparity > 0.f))
			continue;
		const float candidateDepth(focalLength*averageBaseline/candidateDisparity);
		if (!(candidateDepth >= g_params.fDepthMin && candidateDepth <= g_params.fDepthMax) ||
			!isfinite(candidateDepth))
			continue;
		++tested;
		Point4 candidate(plane);
		candidate.w() = candidateDepth;
		float candidateViewCosts[MAX_VIEWS];
		MultiViewScorePlane<GEOM>(refCache,
			(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
			p, candidate, lowDepth, candidateViewCosts);
		const float candidateCost(AggregateAPDViewScores(
			viewWeights, candidateViewCosts, numViews));
		const bool candidateFinite(
			candidateCost >= 0.f && candidateCost <= APD_MAX_NCC_COST && isfinite(candidateCost));
		if (candidateFinite)
			++finite;
		#ifdef _USE_DMAP_INSTRUMENTATION
		if constexpr (INSTRUMENT)
			InstrumentCandidate(instrumentCounters, instrumentParams,
				PM_CANDIDATE_APD_FINAL_REFINEMENT, candidateCost, false);
		#endif
		if (candidateFinite && candidateCost < bestCost) {
			bestCost = candidateCost;
			bestDepth = candidateDepth;
			bestOffset = offset;
		}
	}
	const APDFinalRefinementDecision decision(ResolveAPDFinalRefinement(
		plane.w(), incumbentCost, bestDepth, bestCost, bestOffset));
	if (decision.accepted) {
		plane.w() = decision.depth;
		planes[idx] = plane;
		costs[idx] = decision.cost;
	}
	#ifdef _USE_DMAP_INSTRUMENTATION
	if constexpr (INSTRUMENT) {
		PatchMatchAPDInstrumentUpdate* updateRecord(nullptr);
		if (decision.accepted && instrumentCounters && instrumentParams.passIndex >= 0)
			atomicAdd(&instrumentCounters[instrumentParams.passIndex].candidateAccepted[
				PM_CANDIDATE_APD_FINAL_REFINEMENT], 1u);
		const int logicalIteration(max(g_params.nEstimationIters-1, 0));
		if (instrumentParams.apdUpdates && instrumentParams.area > 0) {
			PatchMatchAPDInstrumentUpdate& update(
				instrumentParams.apdUpdates[(size_t)logicalIteration*instrumentParams.area+idx]);
			updateRecord = &update;
			update.finalRefinementIncumbentCost = incumbentCost;
			update.finalRefinementBestCost = decision.valid ? decision.cost : -1.f;
			update.finalRefinementImprovement = decision.valid ? decision.improvement : -1.f;
			update.finalRefinementDepth = decision.valid ? decision.depth : -1.f;
			update.finalRefinementOffset = static_cast<int8_t>(decision.offset);
			update.finalRefinementTested = static_cast<uint8_t>(min(tested, 255u));
			update.finalRefinementFinite = static_cast<uint8_t>(min(finite, 255u));
			update.finalRefinementAccepted = decision.accepted ? 1u : 0u;
		}
		if (instrumentParams.apdCounters && logicalIteration < instrumentParams.numLogicalStates-1) {
			PatchMatchAPDInstrumentCounters& counter(
				instrumentParams.apdCounters[logicalIteration]);
			atomicAdd(&counter.finalRefinementPixels, 1u);
			atomicAdd(&counter.finalRefinementCandidatesTested, tested);
			atomicAdd(&counter.finalRefinementCandidatesFinite, finite);
			if (decision.accepted)
				atomicAdd(&counter.finalRefinementAccepted, 1u);
		}
		if (decision.accepted) {
			if (updateSources)
				updateSources[idx] = PM_SOURCE_APD_FINAL_REFINEMENT;
		}
		if (updateRecord && instrumentParams.apdTraces && instrumentTraceMap &&
			instrumentParams.numTracePixels > 0)
		{
			const int traceIndex(instrumentTraceMap[idx]);
			if (traceIndex >= 0 && traceIndex < instrumentParams.numTracePixels)
				instrumentParams.apdTraces[
					(size_t)logicalIteration*instrumentParams.numTracePixels+traceIndex].update =
					*updateRecord;
		}
	}
	#endif
}

#ifdef _USE_DMAP_INSTRUMENTATION
// Diagnostic kernels trade occupancy for registers so the production
// 256-thread/2-block launch bounds remain unchanged when instrumentation is off.
template <bool GEOM>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void InitializeScoreInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams)
{
	const Point2i p = GetThreadIndex2();
	InitializePixelScore<GEOM, true, false>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap, instrumentParams, p);
}

template <bool GEOM, bool DVP = false>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void BlackPixelProcessInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams,
	const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 0 : 1);
	ProcessPixel<GEOM, true, false, DVP>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr,
		APDUpdateStage::ALL,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap, instrumentParams, p, iter);
}

template <bool GEOM, bool DVP = false>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void RedPixelProcessInstrumented(
	const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths,
	Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams,
	const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 1 : 0);
	ProcessPixel<GEOM, true, false, DVP>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
		nullptr, nullptr, nullptr,
		APDUpdateStage::ALL,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap, instrumentParams, p, iter);
}

__device__ __forceinline__ float SelectedViewMean(
	const unsigned* viewWeights, const float* costs, const int numViews, const int selectedCount)
{
	if (selectedCount <= 0)
		return fBadCost;
	float value = 0.f;
	for (int view = 0; view < numViews; ++view)
		if (viewWeights[view])
			value += costs[view];
	return value / (float)selectedCount;
}

// Run after each production PatchMatch pass. Keeping diagnostics out of the
// ACMH propagation kernel avoids changing its register/local-memory behavior.
template <bool GEOM>
__global__ PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS void InstrumentPassState(
	const cudaTextureObject_t* textureImages,
	const cudaTextureObject_t* textureDepths,
	const Point4* planes,
	const float* lowDepths,
	const float* costs,
	const unsigned* selectedViews,
	const Point4* planesBefore,
	const float* costsBefore,
	const unsigned* selectedViewsBefore,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	PatchMatchInstrumentKernelParams instrumentParams)
{
	const Point2i p = GetThreadIndex2();
	const int width = g_cameras[0].size.x();
	const int height = g_cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	const Point4 after = planes[idx];
	Point4 before(after);
	float costBefore(costs[idx]);
	unsigned viewsBefore(0u);
	if (instrumentParams.passIndex > 0) {
		before = planesBefore[idx];
		costBefore = costsBefore[idx];
		viewsBefore = selectedViewsBefore[idx];
	}
	const float depthBefore = instrumentParams.passIndex == 0 ? after.w() : before.w();
	const float depthAfter = after.w();
	const float costAfter = costs[idx];
	const unsigned viewsAfter = selectedViews[idx];
	const float depthAbsChange = instrumentParams.passIndex > 0 && depthBefore > 0.f && depthAfter > 0.f ? fabsf(depthAfter-depthBefore) : 0.f;
	const float depthRelChange = depthBefore > 0.f ? depthAbsChange / max(depthBefore, FLT_EPSILON) : 0.f;
	const float normalAngleChange = instrumentParams.passIndex > 0 ? ComputeNormalAngleDegrees(before.topLeftCorner<3,1>(), after.topLeftCorner<3,1>()) : 0.f;
	const bool changed = instrumentParams.passIndex == 0 || depthAbsChange > 0.f || normalAngleChange > 0.f || viewsBefore != viewsAfter || costBefore != costAfter;
	const int source = instrumentParams.passIndex == 0 ? PM_SOURCE_INIT : (changed ? PM_SOURCE_CHANGED_UNKNOWN : PM_SOURCE_NONE);

	const int numViews = g_params.nNumViews;
	unsigned viewWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	int selectedCount = 0;
	for (int view = 0; view < numViews && view < PM_INSTRUMENT_MAX_VIEWS; ++view) {
		if (!IsBitSet(viewsAfter, view))
			continue;
		viewWeights[view] = 1u;
		++selectedCount;
	}
	const float lowDepth = g_params.bLowResProcessed ? lowDepths[idx] : 0.f;
	RefPatchCache refCache;
	ComputeRefPatchCache(((const ImagePixels*)textureImages)[0], p, refCache);
	PatchMatchInstrumentCostComponents components;
	if (selectedCount > 0)
		components = AggregateCostComponents<GEOM>(
			refCache,
			(const ImagePixels*)textureImages,
			(const ImagePixels*)textureDepths,
			p,
			after,
			lowDepth,
			viewWeights,
			numViews,
			(float)selectedCount);

	float neighborCosts[PM_INSTRUMENT_NUM_NEIGHBORS];
	for (int neighbor = 0; neighbor < PM_INSTRUMENT_NUM_NEIGHBORS; ++neighbor)
		neighborCosts[neighbor] = fBadCost;
	float best = FLT_MAX;
	float second = FLT_MAX;
	int finiteCandidates = 0;
	float costVector[MAX_VIEWS];
	MultiViewScorePlane<GEOM>(
		refCache,
		(const ImagePixels*)textureImages,
		(const ImagePixels*)textureDepths,
		p,
		after,
		lowDepth,
		costVector);
	TrackCandidateCost(SelectedViewMean(viewWeights, costVector, numViews, selectedCount), best, second, finiteCandidates);
	const int2 offsets[PM_INSTRUMENT_NUM_NEIGHBORS] = {
		{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, -1}, {-1, 1}, {1, 1}
	};
	for (int neighbor = 0; neighbor < PM_INSTRUMENT_NUM_NEIGHBORS; ++neighbor) {
		const Point2i np(p.x()+offsets[neighbor].x, p.y()+offsets[neighbor].y);
		if (np.x() < 0 || np.y() < 0 || np.x() >= width || np.y() >= height)
			continue;
		const int neighborIdx = Point2Idx(np, width);
		if (planes[neighborIdx].w() <= 0.f)
			continue;
		MultiViewScoreNeighborPlane<GEOM>(
			refCache,
			(const ImagePixels*)textureImages,
			(const ImagePixels*)textureDepths,
			p,
			np,
			LoadPlaneLDG(&planes[neighborIdx]),
			lowDepth,
			costVector);
		neighborCosts[neighbor] = SelectedViewMean(viewWeights, costVector, numViews, selectedCount);
		TrackCandidateCost(neighborCosts[neighbor], best, second, finiteCandidates);
	}
	const float confidenceGap = finiteCandidates >= 2 && isfinite(second) ? max(0.f, second-best) : -1.f;
	const float viewEntropy = selectedCount > 1 ? 1.f : 0.f;
	InstrumentPixel(
		instrumentCounters,
		instrumentTraceRecords,
		instrumentTraceMap,
		updateSources,
		instrumentParams,
		idx,
		p,
		depthBefore,
		depthAfter,
		costBefore,
		costAfter,
		components,
		refCache.varRef,
		lowDepth,
		viewsBefore,
		viewsAfter,
		source,
		depthAbsChange,
		depthRelChange,
		normalAngleChange,
		viewEntropy,
		confidenceGap,
		neighborCosts,
		viewWeights);
}

// Snapshot filtering inputs without mutating the production state. This stays
// separate from FilterPlanes so instrumentation cannot enable filtering on a
// pyramid level where production deliberately disabled it.
__global__ void CapturePreFilterState(
	const Point4* planes,
	const float* costs,
	uint8_t* updateSources,
	uint8_t* validBeforeFilter,
	uint8_t* filterRejectReasons,
	Point4* planesBeforeFilter,
	float* costsBeforeFilter,
	int width,
	int height)
{
	const Point2i p = GetThreadIndex2();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	const Point4 plane(planes[idx]);
	const float cost(costs[idx]);
	if (planesBeforeFilter)
		planesBeforeFilter[idx] = plane;
	if (costsBeforeFilter)
		costsBeforeFilter[idx] = cost;
	const bool validBefore(plane.w() > 0);
	if (validBeforeFilter)
		validBeforeFilter[idx] = validBefore ? 255 : 0;
	if (filterRejectReasons)
		filterRejectReasons[idx] = !validBefore ? 1 :
			(g_params.fThresholdKeepCost > 0.f && cost >= g_params.fThresholdKeepCost ? 2 : 0);
	if (updateSources && validBefore && g_params.fThresholdKeepCost > 0.f && cost >= g_params.fThresholdKeepCost)
		updateSources[idx] = PM_SOURCE_FILTERED;
}
#endif
// filter depth/normals
__global__ void FilterPlanes(Point4* planes, float* costs, unsigned* selectedViews, int width, int height)
{
	const Point2i p = GetThreadIndex2();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	// filter estimates if the score is not good enough
	Point4& plane = planes[idx];
	float conf = costs[idx];
	if (plane.w() <= 0 || conf >= g_params.fThresholdKeepCost) {
		conf = 0;
		plane = Point4::Zero();
		selectedViews[idx] = 0;
	}
}
/*----------------------------------------------------------------*/


// upload host cameras / params into their __constant__ symbols on cudaStream
__host__ void PatchMatch::UploadCameras()
{
	const size_t n = cameras.size();
	ASSERT(n <= MAX_VIEWS + 1);
	CUDA_CHECK(cudaMemcpyToSymbolAsync(g_cameraStorage, cameras.data(), sizeof(Camera) * n, 0, cudaMemcpyHostToDevice, cudaStream));
}
__host__ void PatchMatch::UploadParams()
{
	CUDA_CHECK(cudaMemcpyToSymbolAsync(g_params, &params, sizeof(Params), 0, cudaMemcpyHostToDevice, cudaStream));
}

__global__ void DVPContractOracleKernel(DVPCUDAOracleResult* output)
{
	if (blockIdx.x != 0u || threadIdx.x != 0u || !output)
		return;
	DVPCUDAOracleResult result;
	const Matrix3 identity(Matrix3::Identity());
	const Camera camera(
		LinearCameraModel(800.f, 810.f, 320.f, 240.f),
		Pose(identity, Point3(0.1f, -0.2f, 0.3f)), 640, 480);
	const Point2 pixel(217.25f, 139.75f);
	const float depth(4.5f);
	const Point3 world(camera.TransformPointI2W(pixel, depth));
	const Point2 roundTrip(camera.TransformPointW2I(world));
	const float roundTripDepth((camera.pose.R*(world-camera.pose.C)).z());
	result.roundTripPixelError = (roundTrip-pixel).norm();
	result.roundTripDepthError = fabsf(roundTripDepth-depth);
	if (result.roundTripPixelError <= 1e-5f)
		result.passedChecks |= 1u << 0;
	if (result.roundTripDepthError <= 1e-5f)
		result.passedChecks |= 1u << 1;

	RandState controlRandState;
	RandState dvpRandState;
	curand_init(1234u, 7u, 11u, &controlRandState);
	curand_init(1234u, 7u, 11u, &dvpRandState);
	DVPDepthProposalRNG proposalRNG(&dvpRandState);
	result.rngControlFirst = curand_uniform(&controlRandState);
	result.rngProposalFirst = proposalRNG.Uniform();
	result.rngProposalSecond = proposalRNG.Uniform();
	DVPDepthProposalRNG::ConsumeNativeDepthSlot(&dvpRandState);
	result.rngControlNext = curand_uniform(&controlRandState);
	result.rngDVPNext = curand_uniform(&dvpRandState);
	if (result.rngProposalFirst == result.rngControlFirst &&
		result.rngProposalSecond == result.rngControlNext &&
		result.rngDVPNext == result.rngControlNext)
	{
		result.passedChecks |= 1u << 6;
	}

	DVPEndpointSamples samples;
	const float leftOuter[] = {0.70f, 0.72f, 0.71f, 0.90f};
	const float leftInner[] = {0.85f, 0.84f, 0.83f, 0.80f};
	const float rightInner[] = {1.15f, 1.16f, 1.17f, 1.20f};
	const float rightOuter[] = {1.30f, 1.28f, 1.29f, 1.40f};
	for (unsigned i=0u; i<4u; ++i) {
		DVPAppendEndpoint(samples.leftOuter, samples.leftOuterCount, leftOuter[i]);
		DVPAppendEndpoint(samples.leftInner, samples.leftInnerCount, leftInner[i]);
		DVPAppendEndpoint(samples.rightInner, samples.rightInnerCount, rightInner[i]);
		DVPAppendEndpoint(samples.rightOuter, samples.rightOuterCount, rightOuter[i]);
	}
	result.paperIntervals = DVPBuildPaperIntervals(samples, 3u);
	if (result.paperIntervals.left.valid && result.paperIntervals.right.valid &&
		fabsf(result.paperIntervals.left.minimum-0.72f) <= 1e-6f &&
		fabsf(result.paperIntervals.left.maximum-0.83f) <= 1e-6f &&
		fabsf(result.paperIntervals.right.minimum-1.16f) <= 1e-6f &&
		fabsf(result.paperIntervals.right.maximum-1.30f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 2;
	}
	DVPEndpointSamples leftOnly(samples);
	leftOnly.rightInnerCount = 2u;
	leftOnly.rightOuterCount = 2u;
	const DVPIntervalSet acceptedLeft(DVPBuildPaperIntervals(leftOnly, 3u));
	if (acceptedLeft.left.valid && !acceptedLeft.right.valid &&
		acceptedLeft.reason == DVPEpipolarUnavailableReason::NONE &&
		fabsf(acceptedLeft.left.minimum-0.72f) <= 1e-6f &&
		fabsf(acceptedLeft.left.maximum-0.83f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 5;
	}

	const float candidateCosts[] = {0.40f, 0.20f};
	result.proposalDecision = ResolveDVPProposalDecision(0.50f, candidateCosts, 2u);
	if (result.proposalDecision.accepted &&
		result.proposalDecision.winnerOrdinal == 1u &&
		fabsf(result.proposalDecision.winnerCost-0.20f) <= 1e-6f &&
		fabsf(result.proposalDecision.runnerUpCost-0.40f) <= 1e-6f &&
		fabsf(result.proposalDecision.winnerRunnerUpGap-0.20f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 3;
	}
	const float unavailableCosts[] = {-1.f, 0.30f};
	const DVPProposalDecision unavailable(
		ResolveDVPProposalDecision(0.50f, unavailableCosts, 2u));
	if (unavailable.accepted && unavailable.winnerOrdinal == 1u &&
		unavailable.finiteCount == 1u &&
		fabsf(unavailable.winnerCost-0.30f) <= 1e-6f &&
		fabsf(unavailable.runnerUpCost-0.50f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 4;
	}
	*output = result;
}

__host__ bool PatchMatch::RunDVPContractOracle(DVPCUDAOracleResult& result)
{
	result = DVPCUDAOracleResult{};
	int deviceCount(0);
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
		return false;
	DVPCUDAOracleResult* deviceResult(nullptr);
	if (cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(DVPCUDAOracleResult)) !=
		cudaSuccess)
	{
		return false;
	}
	DVPContractOracleKernel<<<1, 1>>>(deviceResult);
	cudaError_t status(cudaGetLastError());
	if (status == cudaSuccess) {
		status = cudaMemcpy(&result, deviceResult, sizeof(DVPCUDAOracleResult),
			cudaMemcpyDeviceToHost);
	}
	const cudaError_t freeStatus(cudaFree(deviceResult));
	return status == cudaSuccess && freeStatus == cudaSuccess;
}

__global__ void DVPDepthEdgeContractOracleKernel(DVPDepthEdgeCUDAOracleResult* output)
{
	if (blockIdx.x != 0u || threadIdx.x != 0u || !output)
		return;
	DVPDepthEdgeCUDAOracleResult result;
	const uint16_t labels[8] = {1u, 1u, 2u, 0u, 1u, 2u, 1u, 0u};
	uint32_t mixedAnchors[8] = {1u, 2u, 3u, 4u, 5u, 6u, ~uint32_t(0), ~uint32_t(0)};
	result.mixedRegions = FilterDVPDepthEdgeAnchors(
		static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED), labels[0],
		labels, 8u, mixedAnchors, 6u, 8u, ~uint32_t(0));
	for (unsigned i=0u; i<8u; ++i)
		result.compactedAnchors[i] = mixedAnchors[i];
	if (result.mixedRegions.inputCount == 6u && result.mixedRegions.outputCount == 3u &&
		result.mixedRegions.rejectedBoundary == 1u &&
		result.mixedRegions.rejectedCrossRegion == 2u &&
		result.mixedRegions.rejectedInvalidIndex == 0u &&
		mixedAnchors[0] == 1u && mixedAnchors[1] == 4u && mixedAnchors[2] == 6u &&
		mixedAnchors[3] == ~uint32_t(0))
	{
		result.passedChecks |= 1u << 0;
	}
	uint32_t boundaryAnchors[8] = {1u, 4u, 6u, ~uint32_t(0), ~uint32_t(0),
		~uint32_t(0), ~uint32_t(0), ~uint32_t(0)};
	result.centerBoundary = FilterDVPDepthEdgeAnchors(
		static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED), 0u,
		labels, 8u, boundaryAnchors, 3u, 8u, ~uint32_t(0));
	if (result.centerBoundary.outputCount == 0u &&
		result.centerBoundary.rejectedBoundary == 3u)
		result.passedChecks |= 1u << 1;
	uint32_t disabledAnchors[8] = {1u, 2u, 3u, 4u, 5u, 6u, ~uint32_t(0), ~uint32_t(0)};
	result.disabled = FilterDVPDepthEdgeAnchors(
		static_cast<unsigned>(DVPDepthEdgeMode::DISABLED), 0u,
		nullptr, 0u, disabledAnchors, 6u, 8u, ~uint32_t(0));
	if (result.disabled.outputCount == 6u && disabledAnchors[0] == 1u &&
		disabledAnchors[5] == 6u)
		result.passedChecks |= 1u << 2;
	if (ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{}) ==
		DVPDepthEdgeConfigStatus::VALID)
		result.passedChecks |= 1u << 3;
	const DVPDepthEdgeAnchorDecision same(ResolveDVPDepthEdgeAnchorDecision(
		static_cast<unsigned>(DVPDepthEdgeMode::ERODED), 7u, 7u));
	const DVPDepthEdgeAnchorDecision cross(ResolveDVPDepthEdgeAnchorDecision(
		static_cast<unsigned>(DVPDepthEdgeMode::ERODED), 7u, 8u));
	if (same.allowed && same.reason == DVPDepthEdgeAnchorReason::SAME_REGION &&
		!cross.allowed && cross.reason == DVPDepthEdgeAnchorReason::CROSS_REGION)
		result.passedChecks |= 1u << 4;
	*output = result;
}

__host__ bool PatchMatch::RunDVPDepthEdgeContractOracle(
	DVPDepthEdgeCUDAOracleResult& result)
{
	result = DVPDepthEdgeCUDAOracleResult{};
	int deviceCount(0);
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
		return false;
	DVPDepthEdgeCUDAOracleResult* deviceResult(nullptr);
	if (cudaMalloc(reinterpret_cast<void**>(&deviceResult),
		sizeof(DVPDepthEdgeCUDAOracleResult)) != cudaSuccess)
		return false;
	// cudaMemcpy transfers padding too; initialize the complete output buffer.
	cudaError_t status(cudaMemset(deviceResult, 0, sizeof(*deviceResult)));
	if (status == cudaSuccess) {
		DVPDepthEdgeContractOracleKernel<<<1, 1>>>(deviceResult);
		status = cudaGetLastError();
	}
	if (status == cudaSuccess) {
		status = cudaMemcpy(&result, deviceResult, sizeof(DVPDepthEdgeCUDAOracleResult),
			cudaMemcpyDeviceToHost);
	}
	const cudaError_t freeStatus(cudaFree(deviceResult));
	return status == cudaSuccess && freeStatus == cudaSuccess;
}

__global__ void DVPVisibilityContractOracleKernel(DVPVisibilityCUDAOracleResult* output)
{
	if (blockIdx.x != 0 || threadIdx.x != 0)
		return;
	DVPVisibilityCUDAOracleResult result;
	const DVPVisibilityConfig depthConfig{
		static_cast<unsigned>(DVPVisibilityMode::DEPTH_GATED_RESTORE_V1),
		DVP_VISIBILITY_REPROJECTION_THRESHOLD,
		DVP_VISIBILITY_RELATIVE_DEPTH_THRESHOLD,
		true,
		true,
	};
	DVPVisibilityConfig paperConfig(depthConfig);
	paperConfig.mode = static_cast<unsigned>(DVPVisibilityMode::PAPER_2D_RESTORE_V1);
	if (ValidateDVPVisibilityConfig(depthConfig) == DVPVisibilityConfigStatus::VALID &&
		ValidateDVPVisibilityConfig(paperConfig) == DVPVisibilityConfigStatus::VALID)
	{
		result.passedChecks |= 1u << 0;
	}
	DVPVisibilityObservation occluder;
	occluder.referenceDepth = 10.f;
	occluder.expectedSourceDepth = 10.f;
	occluder.observedSourceDepth = 5.f;
	occluder.roundTripError = 0.f;
	occluder.forwardInside = true;
	occluder.backwardInside = true;
	result.retained = ResolveDVPVisibility(depthConfig, 7u, DVPVisibilityObservation{});
	if (result.retained.visible && !result.retained.restored && result.retained.weight == 7u &&
		result.retained.reason == DVPVisibilityReason::RETAINED_PREVIOUS_WEIGHT)
	{
		result.passedChecks |= 1u << 1;
	}
	result.paperOccluder = ResolveDVPVisibility(paperConfig, 0u, occluder);
	if (result.paperOccluder.visible && result.paperOccluder.restored &&
		result.paperOccluder.reason == DVPVisibilityReason::RESTORED_PAPER_2D)
	{
		result.passedChecks |= 1u << 2;
	}
	result.depthOccluder = ResolveDVPVisibility(depthConfig, 0u, occluder);
	if (!result.depthOccluder.visible && !result.depthOccluder.restored &&
		result.depthOccluder.reason == DVPVisibilityReason::NEARER_OCCLUDER_REJECTED)
	{
		result.passedChecks |= 1u << 3;
	}
	DVPVisibilityObservation agreement(occluder);
	agreement.observedSourceDepth = 10.05f;
	agreement.roundTripError = 0.25f;
	result.depthAgreement = ResolveDVPVisibility(depthConfig, 0u, agreement);
	if (result.depthAgreement.visible && result.depthAgreement.restored &&
		result.depthAgreement.reason == DVPVisibilityReason::RESTORED_DEPTH_GATED &&
		fabsf(result.depthAgreement.relativeDepthError-0.005f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 4;
	}
	DVPVisibilityObservation backwardOut(agreement);
	backwardOut.backwardInside = false;
	result.backwardOutOfBounds = ResolveDVPVisibility(depthConfig, 0u, backwardOut);
	if (!result.backwardOutOfBounds.visible &&
		result.backwardOutOfBounds.reason == DVPVisibilityReason::BACKWARD_OUT_OF_BOUNDS)
	{
		result.passedChecks |= 1u << 5;
	}
	const uint8_t weights[] = {3u, 0u, 2u};
	result.weightSummary = SummarizeDVPVisibilityWeights(weights, 3u);
	result.normalizedWeightSum = 0.f;
	for (unsigned view=0u; view<3u; ++view)
		result.normalizedWeightSum +=
			DVPVisibilityNormalizedWeight(weights[view], result.weightSummary.sum);
	if (result.weightSummary.valid && result.weightSummary.sum == 5u &&
		result.weightSummary.visibleCount == 2u &&
		fabsf(result.normalizedWeightSum-1.f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 6;
	}
	const DVPVisibilityStateHeader previous{
		DVP_VISIBILITY_STATE_VERSION, 640u, 480u, 5u, 3u};
	const DVPVisibilityStateHeader next{
		DVP_VISIBILITY_STATE_VERSION, 640u, 480u, 5u, 4u};
	result.transition = ValidateDVPVisibilityTransition(previous, next);
	if (result.transition == DVPVisibilityTransitionStatus::VALID)
		result.passedChecks |= 1u << 7;
	*output = result;
}

__host__ bool PatchMatch::RunDVPVisibilityContractOracle(
	DVPVisibilityCUDAOracleResult& result)
{
	result = DVPVisibilityCUDAOracleResult{};
	int deviceCount(0);
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
		return false;
	DVPVisibilityCUDAOracleResult* deviceResult(nullptr);
	if (cudaMalloc(reinterpret_cast<void**>(&deviceResult),
		sizeof(DVPVisibilityCUDAOracleResult)) != cudaSuccess)
	{
		return false;
	}
	// cudaMemcpy transfers padding too; initialize the complete output buffer.
	cudaError_t status(cudaMemset(deviceResult, 0, sizeof(*deviceResult)));
	if (status == cudaSuccess) {
		DVPVisibilityContractOracleKernel<<<1, 1>>>(deviceResult);
		status = cudaGetLastError();
	}
	if (status == cudaSuccess) {
		status = cudaMemcpy(&result, deviceResult, sizeof(DVPVisibilityCUDAOracleResult),
			cudaMemcpyDeviceToHost);
	}
	const cudaError_t freeStatus(cudaFree(deviceResult));
	return status == cudaSuccess && freeStatus == cudaSuccess;
}

__global__ void DVPVisibleNormalContractOracleKernel(
	DVPVisibleNormalCUDAOracleResult* output)
{
	if (blockIdx.x != 0 || threadIdx.x != 0)
		return;
	DVPVisibleNormalCUDAOracleResult result;
	DVPVisibleNormalConfig activeConfig;
	activeConfig.mode = static_cast<unsigned>(DVPVisibleNormalMode::FULL);
	activeConfig.fullAPD = true;
	if (ValidateDVPVisibleNormalConfig(DVPVisibleNormalConfig{}) ==
			DVPVisibleNormalConfigStatus::VALID &&
		!DVPVisibleNormalModeEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::DISABLED)) &&
		ValidateDVPVisibleNormalConfig(activeConfig) ==
			DVPVisibleNormalConfigStatus::VALID)
	{
		result.passedChecks |= 1u << 0;
	}

	DVPVisibleNormalRotation rotation;
	rotation.values[0] = 0.f;
	rotation.values[1] = -1.f;
	rotation.values[3] = 1.f;
	rotation.values[4] = 0.f;
	const DVPVisibleNormalVector referenceCenter{1.f, 2.f, 3.f};
	const DVPVisibleNormalVector sourceCenter{3.f, 2.f, 3.f};
	result.sourceCenter = DVPVisibleNormalSourceCenterInReference(
		rotation, referenceCenter, sourceCenter);
	const DVPVisibleNormalVector pointInReference{0.f, 0.f, 4.f};
	if (DVPVisibleNormalCameraToPointDirection(
			pointInReference, result.sourceCenter, result.sourceDirection) &&
		fabsf(result.sourceCenter.x) <= 1e-6f &&
		fabsf(result.sourceCenter.y-2.f) <= 1e-6f &&
		fabsf(result.sourceCenter.z) <= 1e-6f &&
		fabsf(result.sourceDirection.x) <= 1e-6f &&
		fabsf(result.sourceDirection.y+0.4472136f) <= 1e-6f &&
		fabsf(result.sourceDirection.z-0.8944272f) <= 1e-6f)
	{
		result.passedChecks |= 1u << 1;
	}

	const DVPVisibleNormalVector referenceOnly[] = {{0.f, 0.f, 3.f}};
	result.referenceFeasible = EvaluateDVPVisibleNormal(
		{0.f, 0.f, -2.f}, referenceOnly, 1u, 0.f);
	if (result.referenceFeasible.valid && result.referenceFeasible.feasible &&
		result.referenceFeasible.directionCount == 1u)
	{
		result.passedChecks |= 1u << 2;
	}
	const DVPVisibleNormalVector multiDirections[] = {
		{0.f, 0.f, 1.f}, result.sourceDirection};
	const DVPVisibleNormalVector feasibleNormal{0.f, 1.f, -1.f};
	result.multiViewFeasible = EvaluateDVPVisibleNormal(
		feasibleNormal, multiDirections, 2u, 0.f);
	if (result.multiViewFeasible.valid && result.multiViewFeasible.feasible &&
		result.multiViewFeasible.maxDot <= 0.f)
	{
		result.passedChecks |= 1u << 3;
	}
	const DVPVisibleNormalVector rejectedNormal{0.f, -1.f, 1.f};
	result.rejected = EvaluateDVPVisibleNormal(
		rejectedNormal, multiDirections, 2u, 0.f);
	if (result.rejected.valid && !result.rejected.feasible &&
		result.rejected.reason == DVPVisibleNormalEvaluationReason::HEMISPHERE_REJECTED &&
		result.rejected.maxViolation > 0.f)
	{
		result.passedChecks |= 1u << 4;
	}

	const DVPVisibleNormalVector contradictoryDirections[] = {
		{1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f},
		{0.f, 1.f, 0.f}, {0.f, -1.f, 0.f},
		{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f},
	};
	result.contradictory = EvaluateDVPVisibleNormal(
		{1.f, 1.f, 1.f}, contradictoryDirections, 6u, 0.f);
	if (result.contradictory.valid && !result.contradictory.feasible &&
		result.contradictory.rejectedDirection != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE)
	{
		result.passedChecks |= 1u << 5;
	}

	const DVPVisibleNormalVector retries[] = {rejectedNormal, feasibleNormal};
	result.retryDecision = ResolveDVPVisibleNormalProposal(
		rejectedNormal, retries, 2u, multiDirections, 2u, 0.f);
	if (result.retryDecision.valid && !result.retryDecision.fallback &&
		result.retryDecision.reason == DVPVisibleNormalProposalReason::CONSTRAINED_RETRY &&
		result.retryDecision.retriesTested == 2u &&
		result.retryDecision.selectedRetry == 1u &&
		result.retryDecision.selectedEvaluation.feasible)
	{
		result.passedChecks |= 1u << 6;
	}
	const DVPVisibleNormalVector rejectedRetries[] = {rejectedNormal, rejectedNormal};
	result.fallbackDecision = ResolveDVPVisibleNormalProposal(
		rejectedNormal, rejectedRetries, 2u, multiDirections, 2u, 0.f);
	if (result.fallbackDecision.valid && result.fallbackDecision.fallback &&
		result.fallbackDecision.reason ==
			DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK &&
		result.fallbackDecision.retriesTested == 2u &&
		!result.fallbackDecision.selectedEvaluation.feasible)
	{
		result.passedChecks |= 1u << 7;
	}

	const float costs[] = {0.10f, 0.20f, 0.05f};
	const uint8_t valid[] = {1u, 1u, 1u};
	const uint8_t feasible[] = {0u, 1u, 0u};
	result.propagationDecision = ResolveDVPVisibleNormalPropagation(
		costs, valid, feasible, 3u);
	if (result.propagationDecision.valid && !result.propagationDecision.fallback &&
		result.propagationDecision.nativeBest == 2u &&
		result.propagationDecision.constrainedBest == 1u &&
		result.propagationDecision.selected == 1u)
	{
		result.passedChecks |= 1u << 8;
	}
	const uint8_t noneFeasible[] = {0u, 0u, 0u};
	result.propagationFallback = ResolveDVPVisibleNormalPropagation(
		costs, valid, noneFeasible, 3u);
	if (result.propagationFallback.valid && result.propagationFallback.fallback &&
		result.propagationFallback.selected == 2u &&
		result.propagationFallback.reason ==
			DVPVisibleNormalPropagationReason::NO_FEASIBLE_NATIVE_FALLBACK)
	{
		result.passedChecks |= 1u << 9;
	}
	*output = result;
}

__host__ bool PatchMatch::RunDVPVisibleNormalContractOracle(
	DVPVisibleNormalCUDAOracleResult& result)
{
	result = DVPVisibleNormalCUDAOracleResult{};
	int deviceCount(0);
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0)
		return false;
	DVPVisibleNormalCUDAOracleResult* deviceResult(nullptr);
	if (cudaMalloc(reinterpret_cast<void**>(&deviceResult),
		sizeof(DVPVisibleNormalCUDAOracleResult)) != cudaSuccess)
	{
		return false;
	}
	// cudaMemcpy transfers padding too; initialize the complete output buffer.
	cudaError_t status(cudaMemset(deviceResult, 0, sizeof(*deviceResult)));
	if (status == cudaSuccess) {
		DVPVisibleNormalContractOracleKernel<<<1, 1>>>(deviceResult);
		status = cudaGetLastError();
	}
	if (status == cudaSuccess) {
		status = cudaMemcpy(&result, deviceResult,
			sizeof(DVPVisibleNormalCUDAOracleResult), cudaMemcpyDeviceToHost);
	}
	const cudaError_t freeStatus(cudaFree(deviceResult));
	return status == cudaSuccess && freeStatus == cudaSuccess;
}

// Resolve one immutable support tensor from accepted t-1 depth and visibility
// state. The next tensor is initialized here and written by each pixel update;
// publication is a host-side pointer swap after the full logical iteration.
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void ResolveDVPVisibilityState(
	const cudaTextureObject_t* textureDepths,
	const Point4* planes,
	const uint8_t* previousWeights,
	uint8_t* resolvedWeights,
	uint8_t* nextWeights
	#ifdef _USE_DMAP_INSTRUMENTATION
	, uint8_t* instrumentReasons
	#endif
	)
{
	const Point2i pixel(GetThreadIndex2());
	const int width(g_cameras[0].size.x());
	const int height(g_cameras[0].size.y());
	if (pixel.x() >= width || pixel.y() >= height)
		return;
	const int index(Point2Idx(pixel, width));
	const size_t offset((size_t)index*MAX_VIEWS);
	const float referenceDepth(LoadPlaneWLDG(&planes[index]));
	const DVPVisibilityConfig config{
		g_params.nDVPVisibilityMode,
		g_params.fDVPVisibilityReprojectionThreshold,
		g_params.fDVPVisibilityRelativeDepthThreshold,
		true,
		true,
	};
	const ImagePixels* depthImages((const ImagePixels*)textureDepths);
	const Point2 referencePixel(pixel.cast<float>());
	for (int view=0; view<MAX_VIEWS; ++view) {
		if (view >= g_params.nNumViews) {
			resolvedWeights[offset+view] = 0u;
			nextWeights[offset+view] = 0u;
			#ifdef _USE_DMAP_INSTRUMENTATION
			if (instrumentReasons)
				instrumentReasons[offset+view] = static_cast<uint8_t>(
					DVPVisibilityReason::MODE_DISABLED);
			#endif
			continue;
		}
		const uint8_t previousWeight(previousWeights[offset+view]);
		DVPVisibilityObservation observation;
		observation.referenceDepth = referenceDepth;
		if (previousWeight == 0u && depthImages && depthImages[view] != 0 &&
			isfinite(referenceDepth) && referenceDepth > 0.f)
		{
			const Camera& sourceCamera(g_cameras[view+1]);
			const Point3 world(g_cameras[0].TransformPointI2W(
				referencePixel, referenceDepth));
			const Point3 expectedSource(sourceCamera.pose.R*(world-sourceCamera.pose.C));
			observation.expectedSourceDepth = expectedSource.z();
			if (isfinite(expectedSource.z()) && expectedSource.z() > 0.f) {
				const Point2 sourcePixel(sourceCamera.model.TransformPointC2I(expectedSource));
				observation.forwardInside = DVPPointInside(sourcePixel, sourceCamera);
				if (observation.forwardInside) {
					observation.observedSourceDepth = tex2D<float>(
						depthImages[view], sourcePixel.x()+0.5f, sourcePixel.y()+0.5f);
					if (isfinite(observation.observedSourceDepth) &&
						observation.observedSourceDepth > 0.f)
					{
						const Point3 observedWorld(sourceCamera.TransformPointI2W(
							sourcePixel, observation.observedSourceDepth));
						const Point2 roundTrip(g_cameras[0].TransformPointW2I(observedWorld));
						observation.backwardInside = DVPPointInside(roundTrip, g_cameras[0]);
						observation.roundTripError = (roundTrip-referencePixel).norm();
					}
				}
			}
		}
		const DVPVisibilityDecision decision(
			ResolveDVPVisibility(config, previousWeight, observation));
		resolvedWeights[offset+view] = decision.weight;
		nextWeights[offset+view] = previousWeight;
		#ifdef _USE_DMAP_INSTRUMENTATION
		if (instrumentReasons)
			instrumentReasons[offset+view] = static_cast<uint8_t>(decision.reason);
		#endif
	}
}

struct APDDeviceBuffers {
	uint8_t* reliability = nullptr;
	uint8_t* transferredReliability = nullptr;
	uint8_t* viewWeights = nullptr;
	uint32_t* anchors = nullptr;
	uint8_t* anchorCounts = nullptr;
	Point4* planesSnapshot = nullptr;
	uint32_t* selectedViewsSnapshot = nullptr;
	Point4* fittedPlanes = nullptr;
	uint8_t* fittedPlaneValid = nullptr;
	float* rowSquaredDistances = nullptr;
	int* rowNearestX = nullptr;
	uint32_t* nearestReliable = nullptr;
	uint16_t* depthEdgeRegions = nullptr;
	uint8_t* depthEdgeAnchorCountsBefore = nullptr;
	uint8_t* depthEdgeAnchorCountsAfter = nullptr;
	uint8_t* depthEdgeRejectedAnchorCounts = nullptr;
	uint8_t* visibilityPrevious = nullptr;
	uint8_t* visibilityResolved = nullptr;
	uint8_t* visibilityNext = nullptr;
};

__host__ inline void AllocateAPDDeviceBuffers(
	APDDeviceBuffers& buffers,
	const size_t area,
	cudaStream_t stream)
{
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.reliability, sizeof(uint8_t)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.transferredReliability, sizeof(uint8_t)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.viewWeights, sizeof(uint8_t)*area*MAX_VIEWS, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.anchors, sizeof(uint32_t)*area*APD_MAX_ANCHORS, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.anchorCounts, sizeof(uint8_t)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.planesSnapshot, sizeof(Point4)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.selectedViewsSnapshot, sizeof(uint32_t)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.fittedPlanes, sizeof(Point4)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.fittedPlaneValid, sizeof(uint8_t)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.rowSquaredDistances, sizeof(float)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.rowNearestX, sizeof(int)*area, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.nearestReliable, sizeof(uint32_t)*area, stream));
}

__host__ inline void AllocateDVPVisibilityDeviceBuffers(
	APDDeviceBuffers& buffers,
	const size_t area,
	cudaStream_t stream)
{
	const size_t bytes(sizeof(uint8_t)*area*MAX_VIEWS);
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.visibilityPrevious, bytes, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.visibilityResolved, bytes, stream));
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.visibilityNext, bytes, stream));
}

__host__ inline void AllocateDVPDepthEdgeDeviceBuffers(
	APDDeviceBuffers& buffers,
	const PatchMatchAPDMultiscaleIO& io,
	const size_t area,
	cudaStream_t stream)
{
	ASSERT(io.depthEdgeRegions);
	CUDA_CHECK(cudaMallocAsync((void**)&buffers.depthEdgeRegions,
		sizeof(uint16_t)*area, stream));
	CUDA_CHECK(cudaMemcpyAsync(buffers.depthEdgeRegions, io.depthEdgeRegions,
		sizeof(uint16_t)*area, cudaMemcpyHostToDevice, stream));
	#ifdef _USE_DMAP_INSTRUMENTATION
	const bool captureMaps(io.outputDepthEdgeAnchorCountsBefore ||
		io.outputDepthEdgeAnchorCountsAfter || io.outputDepthEdgeRejectedAnchorCounts);
	if (captureMaps) {
		ASSERT(io.outputDepthEdgeAnchorCountsBefore &&
			io.outputDepthEdgeAnchorCountsAfter && io.outputDepthEdgeRejectedAnchorCounts);
		CUDA_CHECK(cudaMallocAsync((void**)&buffers.depthEdgeAnchorCountsBefore,
			sizeof(uint8_t)*area, stream));
		CUDA_CHECK(cudaMallocAsync((void**)&buffers.depthEdgeAnchorCountsAfter,
			sizeof(uint8_t)*area, stream));
		CUDA_CHECK(cudaMallocAsync((void**)&buffers.depthEdgeRejectedAnchorCounts,
			sizeof(uint8_t)*area, stream));
	}
	#endif
}

__host__ inline void CopyDVPDepthEdgeInstrumentation(
	const APDDeviceBuffers& buffers,
	const PatchMatchAPDMultiscaleIO& io,
	const size_t area,
	cudaStream_t stream)
{
	#ifdef _USE_DMAP_INSTRUMENTATION
	if (io.outputDepthEdgeAnchorCountsBefore) {
		ASSERT(buffers.depthEdgeAnchorCountsBefore &&
			buffers.depthEdgeAnchorCountsAfter && buffers.depthEdgeRejectedAnchorCounts);
		CUDA_CHECK(cudaMemcpyAsync(io.outputDepthEdgeAnchorCountsBefore,
			buffers.depthEdgeAnchorCountsBefore, sizeof(uint8_t)*area,
			cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(io.outputDepthEdgeAnchorCountsAfter,
			buffers.depthEdgeAnchorCountsAfter, sizeof(uint8_t)*area,
			cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(io.outputDepthEdgeRejectedAnchorCounts,
			buffers.depthEdgeRejectedAnchorCounts, sizeof(uint8_t)*area,
			cudaMemcpyDeviceToHost, stream));
	}
	#else
	(void)buffers;
	(void)io;
	(void)area;
	(void)stream;
	#endif
}

__host__ inline void ReleaseAPDDeviceBuffers(
	APDDeviceBuffers& buffers,
	cudaStream_t stream)
{
	CUDA_CHECK(cudaFreeAsync(buffers.reliability, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.transferredReliability, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.viewWeights, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.anchors, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.anchorCounts, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.planesSnapshot, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.selectedViewsSnapshot, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.fittedPlanes, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.fittedPlaneValid, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.rowSquaredDistances, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.rowNearestX, stream));
	CUDA_CHECK(cudaFreeAsync(buffers.nearestReliable, stream));
	if (buffers.depthEdgeRegions)
		CUDA_CHECK(cudaFreeAsync(buffers.depthEdgeRegions, stream));
	if (buffers.depthEdgeAnchorCountsBefore)
		CUDA_CHECK(cudaFreeAsync(buffers.depthEdgeAnchorCountsBefore, stream));
	if (buffers.depthEdgeAnchorCountsAfter)
		CUDA_CHECK(cudaFreeAsync(buffers.depthEdgeAnchorCountsAfter, stream));
	if (buffers.depthEdgeRejectedAnchorCounts)
		CUDA_CHECK(cudaFreeAsync(buffers.depthEdgeRejectedAnchorCounts, stream));
	if (buffers.visibilityPrevious)
		CUDA_CHECK(cudaFreeAsync(buffers.visibilityPrevious, stream));
	if (buffers.visibilityResolved)
		CUDA_CHECK(cudaFreeAsync(buffers.visibilityResolved, stream));
	if (buffers.visibilityNext)
		CUDA_CHECK(cudaFreeAsync(buffers.visibilityNext, stream));
	buffers = APDDeviceBuffers{};
}

#ifdef _USE_DMAP_INSTRUMENTATION
__host__ void PatchMatch::RunCUDA(float* ptrCostMap, uint32_t* ptrViewsMap,
	uint8_t* ptrUpdateSources, PatchMatchInstrumentDeviceContext* instrument,
	const PatchMatchAPDMultiscaleIO* apdMultiscaleIO)
{
	const unsigned width = cameras[0].size.x();
	const unsigned height = cameras[0].size.y();
	const bool apdRequested(APDModeEnabled(params.nAPDMode));
	const bool apdFullRequested(APDModeUsesFullMechanics(params.nAPDMode));
	const bool depthEdgeRequested(DVPDepthEdgeModeEnabled(params.nDVPDepthEdgeMode));
	ASSERT(params.nAPDMode <= static_cast<unsigned>(APDMode::DEFORMATION_ONLY));
	ASSERT(ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{
		params.nDVPDepthEdgeMode, apdRequested,
		apdMultiscaleIO && apdMultiscaleIO->depthEdgeRegions}) ==
		DVPDepthEdgeConfigStatus::VALID);
	const DVPConfig dvpConfig{
		params.nDVPEpipolarFamily,
		params.fDVPEpipolarAlpha,
		params.fDVPEpipolarBeta,
		params.nDVPEpipolarMu,
		params.nDVPGlobalSearchRadius,
		params.fDVPReprojectionThreshold,
		params.fDVPRelativeDepthThreshold,
	};
	ASSERT(ValidateDVPConfig(dvpConfig) == DVPConfigStatus::VALID);
	(void)dvpConfig;
	const bool dvpRequested(DVPEpipolarFamilyEnabled(params.nDVPEpipolarFamily));
	ASSERT(!dvpRequested || params.bGeomConsistency);
	const bool dvpEnabled(dvpRequested && params.bGeomConsistency);
	APDStageClock apdClock;
	apdClock.levelIndex = params.nAPDLevelIndex;
	apdClock.levelCount = params.nAPDLevelCount;
	apdClock.stageIndex = params.nAPDStageIndex;
	apdClock.hasTransferredState = params.bAPDTransferredState;
	apdClock.geometricConsistency = params.bGeomConsistency;
	const APDStageSchedule apdSchedule(ResolveAPDStageSchedule(apdClock));
	ASSERT(!apdRequested || apdSchedule.valid);
	const bool apdEnabled(apdRequested && !apdSchedule.conventional);
	const bool apdFullEnabled(apdEnabled && apdFullRequested);
	ASSERT(!apdEnabled || (apdMultiscaleIO && apdMultiscaleIO->transferredReliability));
	const bool visibilityRequested(DVPVisibilityModeEnabled(params.nDVPVisibilityMode));
	ASSERT(!visibilityRequested || apdFullRequested);
	const bool visibilityEnabled(
		visibilityRequested && apdFullEnabled && params.bGeomConsistency);
	if (visibilityEnabled) {
		ASSERT(ValidateDVPVisibilityConfig(DVPVisibilityConfig{
			params.nDVPVisibilityMode,
			params.fDVPVisibilityReprojectionThreshold,
			params.fDVPVisibilityRelativeDepthThreshold,
			true,
			true,
		}) == DVPVisibilityConfigStatus::VALID);
	}
	const bool visibleNormalRequested(
		DVPVisibleNormalModeEnabled(params.nDVPVisibleNormalMode));
	ASSERT(!visibleNormalRequested || apdFullRequested);
	ASSERT(ValidateDVPVisibleNormalConfig(DVPVisibleNormalConfig{
		params.nDVPVisibleNormalMode,
		params.fDVPVisibleNormalDotTolerance,
		params.nDVPVisibleNormalAttempts,
		apdFullRequested,
	}) == DVPVisibleNormalConfigStatus::VALID);
	const bool visibleNormalEnabled(visibleNormalRequested && apdFullEnabled);
	const int numPasses = 1 + params.nEstimationIters * 2;
	const bool instrumentEnabled = instrument && instrument->counters;
	const bool exactEnabled = instrumentEnabled && instrument->exact && !apdRequested;
	const bool apdInstrumentEnabled = apdEnabled && instrumentEnabled && instrument->apdCounters;
	const bool dvpInstrumentEnabled = dvpEnabled && instrumentEnabled && instrument->dvpCounters;
	const bool exactHotKernelCounters = exactEnabled || apdInstrumentEnabled || dvpInstrumentEnabled;
	const bool sampledEnabled = instrumentEnabled && instrument->sampled && instrument->traceRecords && instrument->traceMap && instrument->numTracePixels > 0;
	const bool timingEnabled = instrumentEnabled && instrument->kernelTimingsMs;
	size_t previousStackSize = 0;
	bool restoreStackSize = false;
	if (instrumentEnabled) {
		size_t stackSize = 0;
		CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
		const size_t requiredStackSize(exactHotKernelCounters ? PM_INSTRUMENT_EXACT_STACK_BYTES : 2048);
		if (stackSize < requiredStackSize) {
			previousStackSize = stackSize;
			restoreStackSize = true;
			CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, requiredStackSize));
		}
	}
	std::vector<cudaEvent_t> timingStart;
	std::vector<cudaEvent_t> timingStop;
	std::vector<cudaEvent_t> apdLateTimingStart;
	std::vector<cudaEvent_t> apdLateTimingStop;
	if (timingEnabled) {
		timingStart.resize((size_t)numPasses, nullptr);
		timingStop.resize((size_t)numPasses, nullptr);
		if (apdFullEnabled) {
			apdLateTimingStart.resize((size_t)numPasses, nullptr);
			apdLateTimingStop.resize((size_t)numPasses, nullptr);
		}
		for (int pass = 0; pass < numPasses; ++pass) {
			CUDA_CHECK(cudaEventCreate(&timingStart[(size_t)pass]));
			CUDA_CHECK(cudaEventCreate(&timingStop[(size_t)pass]));
			if (apdFullEnabled && pass > 0) {
				CUDA_CHECK(cudaEventCreate(&apdLateTimingStart[(size_t)pass]));
				CUDA_CHECK(cudaEventCreate(&apdLateTimingStop[(size_t)pass]));
			}
		}
	}

	constexpr unsigned BLOCK_W = 32;
	// BLOCK_H is selected by PATCHMATCHCUDA_LB_256_2 (build-time toggle)
	constexpr unsigned BLOCK_H = (BLOCK_W / PATCHMATCHCUDA_BLOCK_H_DIV);

	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const dim3 gridSizeFull((width + BLOCK_W - 1) / BLOCK_W, (height + BLOCK_H - 1) / BLOCK_H, 1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);
	APDDeviceBuffers apdBuffers;
	if (apdRequested) {
		AllocateAPDDeviceBuffers(apdBuffers, static_cast<size_t>(width)*height, cudaStream);
		if (visibilityEnabled)
			AllocateDVPVisibilityDeviceBuffers(
				apdBuffers, static_cast<size_t>(width)*height, cudaStream);
		if (depthEdgeRequested)
			AllocateDVPDepthEdgeDeviceBuffers(apdBuffers, *apdMultiscaleIO,
				static_cast<size_t>(width)*height, cudaStream);
		if (apdMultiscaleIO && apdMultiscaleIO->transferredReliability) {
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.transferredReliability,
				apdMultiscaleIO->transferredReliability, sizeof(uint8_t)*width*height,
				cudaMemcpyHostToDevice, cudaStream));
		} else {
			CUDA_CHECK(cudaMemsetAsync(apdBuffers.transferredReliability,
				static_cast<int>(APDReliabilityClass::UNKNOWN),
				sizeof(uint8_t)*width*height, cudaStream));
		}
	}

	// refresh constant-memory params for this pyramid level
	UploadParams();

	// dispatch templated kernels by bGeomConsistency
	#define LAUNCH_PRODUCTION_GEOM(KERNEL, GRID, ...) { \
			if (params.bGeomConsistency) \
				KERNEL<true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			else \
				KERNEL<false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
		}
	#define LAUNCH_INSTRUMENT_GEOM(KERNEL, GRID, ...) { \
			if (params.bGeomConsistency) { \
				KERNEL<true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} else { \
				KERNEL<false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} \
		}
	#define LAUNCH_PRODUCTION_PATCHMATCH_GEOM(KERNEL, GRID, ...) { \
			if (dvpEnabled) { \
				if (params.bGeomConsistency) \
					KERNEL<true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} else { \
				if (params.bGeomConsistency) \
					KERNEL<true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} \
		}
	#define LAUNCH_INSTRUMENT_PATCHMATCH_GEOM(KERNEL, GRID, ...) { \
			if (dvpEnabled) { \
				if (params.bGeomConsistency) \
					KERNEL<true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} else { \
				if (params.bGeomConsistency) \
					KERNEL<true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} \
		}
	#define LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(KERNEL, GRID, ...) { \
			if (visibilityEnabled) { \
				if (dvpEnabled) { \
					if (visibleNormalEnabled) KERNEL<true, true, true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, true, true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<true, false, true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, false, true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} else if (dvpEnabled) { \
				if (params.bGeomConsistency) { \
					if (visibleNormalEnabled) KERNEL<true, true, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, true, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<false, true, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<false, true, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} else { \
				if (params.bGeomConsistency) { \
					if (visibleNormalEnabled) KERNEL<true, false, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, false, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<false, false, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<false, false, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} \
		}
	#define LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(KERNEL, GRID, ...) \
		LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(KERNEL, GRID, __VA_ARGS__)
	#define INSTRUMENT_PARAMS(PASS, PHASE, ITER, PROXY_ONLY) PatchMatchInstrumentKernelParams{ \
			instrumentEnabled ? 1 : 0, sampledEnabled ? 1 : 0, \
			instrument ? instrument->imageID : -1, instrument ? instrument->scaleNumber : -1, \
			(PASS), (PHASE), (ITER), numPasses, \
			instrument ? instrument->numTracePixels : 0, \
			(int32_t)(width * height), \
			(PROXY_ONLY) ? ((PASS) == 0 ? 0 : ((PHASE) == 2 ? (ITER) + 1 : -1)) : -1, \
			(exactEnabled && !(PROXY_ONLY)) ? ((PASS) == 0 ? 0 : (ITER) + 1) : -1, \
			instrument && instrument->numLogicalStates > 0 ? instrument->numLogicalStates : params.nEstimationIters + 1, \
			instrument && instrument->viewStride > 0 ? instrument->viewStride : params.nNumViews, \
			((PROXY_ONLY) && exactHotKernelCounters) ? 1 : 0, \
			instrument ? instrument->improvementMaps : nullptr, \
			instrument ? instrument->passUpdateSources : nullptr, \
			instrument ? instrument->passDepthDeltas : nullptr, \
			instrument ? instrument->passDepthRelDeltas : nullptr, \
			instrument ? instrument->passNormalAngleDeltas : nullptr, \
			instrument ? instrument->passViewChurn : nullptr, \
			instrument ? instrument->logicalStoredCosts : nullptr, \
			instrument ? instrument->logicalScorePrimary : nullptr, \
			instrument ? instrument->logicalScoreSecondary : nullptr, \
			instrument ? instrument->exactLogicalScorePrimary : nullptr, \
			instrument ? instrument->exactLogicalScoreSecondary : nullptr, \
			instrument ? instrument->finalViewWeights : nullptr, \
			instrument ? instrument->finalViewCosts : nullptr, \
			instrument ? instrument->finalViewPhotometricCosts : nullptr, \
			instrument ? instrument->finalViewGeometricCosts : nullptr, \
			instrument ? instrument->finalViewEntropy : nullptr, \
			instrument ? instrument->finalLowDepth : nullptr, \
			instrument ? instrument->finalSelectedViews : nullptr, \
			instrument ? instrument->acceptedUpdateCount : nullptr, \
			instrument ? instrument->exactPixels : nullptr, \
			instrument ? instrument->exactViews : nullptr, \
			instrument ? instrument->apdCounters : nullptr, \
			instrument ? instrument->apdUpdates : nullptr, \
			instrument ? instrument->apdTraces : nullptr, \
			instrument ? instrument->dvpCounters : nullptr, \
			instrument ? instrument->dvpUpdates : nullptr, \
			instrument ? instrument->dvpTraces : nullptr, \
			instrument ? instrument->visibilityCounters : nullptr, \
			instrument ? instrument->visibilityUpdates : nullptr, \
			instrument ? instrument->visibilityTraces : nullptr, \
			instrument ? instrument->visibilityReasons : nullptr, \
			instrument ? instrument->visibleNormalCounters : nullptr, \
			instrument ? instrument->visibleNormalUpdates : nullptr, \
			instrument ? instrument->visibleNormalTraces : nullptr \
		}
	#define TIMING_START(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(timingStart[(size_t)(PASS)], cudaStream))
	#define TIMING_STOP(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(timingStop[(size_t)(PASS)], cudaStream))
	#define APD_LATE_TIMING_START(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(apdLateTimingStart[(size_t)(PASS)], cudaStream))
	#define APD_LATE_TIMING_STOP(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(apdLateTimingStop[(size_t)(PASS)], cudaStream))
	#define SNAPSHOT_STATE() { \
			if (instrumentEnabled) { \
				CUDA_CHECK(cudaMemcpyAsync(instrument->planesBeforePass, cudaDepthNormalEstimates, sizeof(Point4) * width * height, cudaMemcpyDeviceToDevice, cudaStream)); \
				CUDA_CHECK(cudaMemcpyAsync(instrument->costsBeforePass, cudaDepthNormalCosts, sizeof(float) * width * height, cudaMemcpyDeviceToDevice, cudaStream)); \
				CUDA_CHECK(cudaMemcpyAsync(instrument->selectedViewsBeforePass, cudaSelectedViews, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToDevice, cudaStream)); \
			} \
		}
	#define INSTRUMENT_STATE(PASS, PHASE, ITER) { \
			if (instrumentEnabled) { \
				if (params.bGeomConsistency) \
					InstrumentPassState<true><<<gridSizeFull, blockSize, 0, cudaStream>>>(cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews, instrument->planesBeforePass, instrument->costsBeforePass, instrument->selectedViewsBeforePass, instrument->updateSources, instrument->counters, instrument->traceRecords, instrument->traceMap, INSTRUMENT_PARAMS(PASS, PHASE, ITER, 1)); \
				else \
					InstrumentPassState<false><<<gridSizeFull, blockSize, 0, cudaStream>>>(cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews, instrument->planesBeforePass, instrument->costsBeforePass, instrument->selectedViewsBeforePass, instrument->updateSources, instrument->counters, instrument->traceRecords, instrument->traceMap, INSTRUMENT_PARAMS(PASS, PHASE, ITER, 1)); \
			} \
		}

	// Pure queueing path: stream ordering on cudaStream already chains kernels;
	// caller (EstimateDepthMap) syncs the stream once before reading results.
	// Initialization has no prior cost/view state. Snapshotting those buffers
	// here would read them before InitializeScore defines them.
	TIMING_START(0);
	if (apdInstrumentEnabled) {
		LAUNCH_INSTRUMENT_GEOM(InitializeScoreAPDInstrumented, gridSizeFull,
			cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
			cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, apdBuffers.viewWeights,
			instrument->updateSources, instrument->counters, instrument->traceRecords,
			instrument->traceMap, INSTRUMENT_PARAMS(0, 0, -1, 0));
	} else if (apdEnabled) {
		LAUNCH_PRODUCTION_GEOM(InitializeScoreAPD, gridSizeFull,
			cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
			cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, apdBuffers.viewWeights);
	} else if (exactEnabled) {
		LAUNCH_INSTRUMENT_GEOM(InitializeScoreInstrumented, gridSizeFull, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
			instrument->updateSources, instrument->counters, instrument->traceRecords,
			instrument->traceMap, INSTRUMENT_PARAMS(0, 0, -1, 0));
	} else {
		LAUNCH_PRODUCTION_GEOM(InitializeScore, gridSizeFull, cudaTextureImages, cudaTextureDepths,
			cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews);
	}
	TIMING_STOP(0);
	INSTRUMENT_STATE(0, 0, -1);
	if (visibilityEnabled) {
		CUDA_CHECK(cudaMemcpyAsync(apdBuffers.visibilityPrevious, apdBuffers.viewWeights,
			sizeof(uint8_t)*width*height*MAX_VIEWS,
			cudaMemcpyDeviceToDevice, cudaStream));
	}

	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		const int blackPass = 1 + iter * 2;
		const int redPass = blackPass + 1;
		if (apdEnabled) {
			if (visibilityEnabled)
				ResolveDVPVisibilityState<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureDepths, cudaDepthNormalEstimates,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext
					#ifdef _USE_DMAP_INSTRUMENTATION
					, apdInstrumentEnabled ? instrument->visibilityReasons : nullptr
					#endif
					);
			if (apdInstrumentEnabled) {
				if (params.bGeomConsistency)
					ClassifyAPDProfiles<true, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						apdBuffers.viewWeights, apdBuffers.reliability, instrument->apdCounters,
						instrument->apdStates, instrument->apdTraces, instrument->traceMap,
						INSTRUMENT_PARAMS(blackPass, 0, iter, 0), apdSchedule.reliabilityEta,
						static_cast<unsigned>(iter));
				else
					ClassifyAPDProfiles<false, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						apdBuffers.viewWeights, apdBuffers.reliability, instrument->apdCounters,
						instrument->apdStates, instrument->apdTraces, instrument->traceMap,
						INSTRUMENT_PARAMS(blackPass, 0, iter, 0), apdSchedule.reliabilityEta,
						static_cast<unsigned>(iter));
			} else {
				LAUNCH_PRODUCTION_GEOM(ClassifyAPDProfiles, gridSizeFull,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					apdBuffers.viewWeights, apdBuffers.reliability,
					nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
					apdSchedule.reliabilityEta, static_cast<unsigned>(iter));
			}
			const uint8_t* effectiveReliability(
				iter == 0 ? apdBuffers.transferredReliability : apdBuffers.reliability);
			APDStageClock apdIterationClock(apdClock);
			apdIterationClock.logicalIteration = static_cast<unsigned>(iter);
			const uint32_t apdStageSeed(APDStageSeed(apdIterationClock));
			if (!apdFullEnabled) {
				// Deformation-only is a causal ablation: classify and freeze anchors
				// from the incoming state, then retain the native checkerboard order,
				// propagation, view selection, and refinement mechanics.
				FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					effectiveReliability, apdBuffers.rowSquaredDistances,
					apdBuffers.rowNearestX, static_cast<int>(width), static_cast<int>(height));
				FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
					apdBuffers.nearestReliable, static_cast<int>(width), static_cast<int>(height));
				if (apdInstrumentEnabled)
					BuildAPDAnchors<true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
						apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
						apdBuffers.fittedPlaneValid, instrument->apdCounters,
						instrument->apdStates, instrument->apdTraces, instrument->traceMap,
						INSTRUMENT_PARAMS(blackPass, 0, iter, 0),
						apdSchedule.ransacNormalizedThreshold, apdStageSeed,
						static_cast<unsigned>(iter));
				else
					BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
						apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
						apdBuffers.fittedPlaneValid,
						nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
						apdSchedule.ransacNormalizedThreshold, apdStageSeed,
						static_cast<unsigned>(iter));
				if (depthEdgeRequested)
					FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
						apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
						apdBuffers.fittedPlaneValid, apdBuffers.depthEdgeAnchorCountsBefore,
						apdBuffers.depthEdgeAnchorCountsAfter,
						apdBuffers.depthEdgeRejectedAnchorCounts, params.nDVPDepthEdgeMode,
						apdSchedule.ransacNormalizedThreshold, apdStageSeed);
				CUDA_CHECK(cudaMemcpyAsync(apdBuffers.planesSnapshot, cudaDepthNormalEstimates,
					sizeof(Point4)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
				CUDA_CHECK(cudaMemcpyAsync(apdBuffers.selectedViewsSnapshot, cudaSelectedViews,
					sizeof(uint32_t)*width*height, cudaMemcpyDeviceToDevice, cudaStream));

				SNAPSHOT_STATE();
				TIMING_START(blackPass);
				if (apdInstrumentEnabled) {
					LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(BlackPixelProcessAPDInstrumented, gridSizeCheckerboard,
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
						effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
						apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
						apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
						APDUpdateStage::ALL, instrument->updateSources, instrument->counters,
						instrument->traceRecords, instrument->traceMap,
						INSTRUMENT_PARAMS(blackPass, 1, iter, 0), iter);
				} else {
					LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
						effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
						apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
						apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
						APDUpdateStage::ALL, iter);
				}
				TIMING_STOP(blackPass);
				INSTRUMENT_STATE(blackPass, 1, iter);

				SNAPSHOT_STATE();
				TIMING_START(redPass);
				if (apdInstrumentEnabled) {
					LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(RedPixelProcessAPDInstrumented, gridSizeCheckerboard,
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
						effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
						apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
						apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
						APDUpdateStage::ALL, instrument->updateSources, instrument->counters,
						instrument->traceRecords, instrument->traceMap,
						INSTRUMENT_PARAMS(redPass, 2, iter, 0), iter);
				} else {
					LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
						cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
						cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
						effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
						apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
						apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
						APDUpdateStage::ALL, iter);
				}
				TIMING_STOP(redPass);
				INSTRUMENT_STATE(redPass, 2, iter);
				continue;
			}
			// Reliable pixels update first. The later non-reliable stage consumes
			// anchors fitted from this updated state, matching the paper schedule.
			SNAPSHOT_STATE();
			TIMING_START(blackPass);
			if (apdInstrumentEnabled) {
				LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(BlackPixelProcessAPDInstrumented, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::RELIABLE, instrument->updateSources, instrument->counters,
					instrument->traceRecords, instrument->traceMap,
					INSTRUMENT_PARAMS(blackPass, 1, iter, 0), iter);
			} else {
				LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::RELIABLE, iter);
			}
			TIMING_STOP(blackPass);

			TIMING_START(redPass);
			if (apdInstrumentEnabled) {
				LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(RedPixelProcessAPDInstrumented, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::RELIABLE, instrument->updateSources, instrument->counters,
					instrument->traceRecords, instrument->traceMap,
					INSTRUMENT_PARAMS(redPass, 2, iter, 0), iter);
			} else {
				LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::RELIABLE, iter);
			}
			TIMING_STOP(redPass);

			FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				effectiveReliability, apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
				static_cast<int>(width), static_cast<int>(height));
			FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX, apdBuffers.nearestReliable,
				static_cast<int>(width), static_cast<int>(height));
			if (apdInstrumentEnabled)
				BuildAPDAnchors<true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
					apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
					apdBuffers.fittedPlaneValid, instrument->apdCounters,
					instrument->apdStates, instrument->apdTraces, instrument->traceMap,
					INSTRUMENT_PARAMS(blackPass, 0, iter, 0),
					apdSchedule.ransacNormalizedThreshold, apdStageSeed,
					static_cast<unsigned>(iter));
			else
				BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
					apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
					apdBuffers.fittedPlaneValid,
					nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
					apdSchedule.ransacNormalizedThreshold, apdStageSeed,
					static_cast<unsigned>(iter));
			if (depthEdgeRequested)
				FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
					apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
					apdBuffers.fittedPlaneValid, apdBuffers.depthEdgeAnchorCountsBefore,
					apdBuffers.depthEdgeAnchorCountsAfter,
					apdBuffers.depthEdgeRejectedAnchorCounts, params.nDVPDepthEdgeMode,
					apdSchedule.ransacNormalizedThreshold, apdStageSeed);
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.planesSnapshot, cudaDepthNormalEstimates,
				sizeof(Point4)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.selectedViewsSnapshot, cudaSelectedViews,
				sizeof(uint32_t)*width*height, cudaMemcpyDeviceToDevice, cudaStream));

			APD_LATE_TIMING_START(blackPass);
			if (apdInstrumentEnabled) {
				LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(BlackPixelProcessAPDInstrumented, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::NON_RELIABLE, instrument->updateSources,
					instrument->counters, instrument->traceRecords, instrument->traceMap,
					INSTRUMENT_PARAMS(blackPass, 1, iter, 0), iter);
			} else {
				LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::NON_RELIABLE, iter);
			}
			APD_LATE_TIMING_STOP(blackPass);
			// The lightweight Process<false> proxy compares the immutable
			// pre-iteration snapshot with the complete logical-iteration state.
			// Capturing here and again after the red stage would count reliable
			// and black-stage changes twice. Exact APD kernels already attribute
			// their own checkerboard decisions; checkerboard exposure otherwise
			// remains timing-only by contract.

			APD_LATE_TIMING_START(redPass);
			if (apdInstrumentEnabled) {
				LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM(RedPixelProcessAPDInstrumented, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::NON_RELIABLE, instrument->updateSources,
					instrument->counters, instrument->traceRecords, instrument->traceMap,
					INSTRUMENT_PARAMS(redPass, 2, iter, 0), iter);
			} else {
				LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::NON_RELIABLE, iter);
			}
			APD_LATE_TIMING_STOP(redPass);
			INSTRUMENT_STATE(redPass, 2, iter);
			if (visibilityEnabled)
				std::swap(apdBuffers.visibilityPrevious, apdBuffers.visibilityNext);
			continue;
		}

		SNAPSHOT_STATE();
		TIMING_START(blackPass);
			if (exactEnabled || dvpInstrumentEnabled) {
			LAUNCH_INSTRUMENT_PATCHMATCH_GEOM(BlackPixelProcessInstrumented, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				instrument->updateSources, instrument->counters, instrument->traceRecords,
				instrument->traceMap, INSTRUMENT_PARAMS(blackPass, 1, iter, 0), iter);
		} else {
			LAUNCH_PRODUCTION_PATCHMATCH_GEOM(BlackPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths,
				cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		}
		TIMING_STOP(blackPass);
		INSTRUMENT_STATE(blackPass, 1, iter);
		SNAPSHOT_STATE();
		TIMING_START(redPass);
			if (exactEnabled || dvpInstrumentEnabled) {
			LAUNCH_INSTRUMENT_PATCHMATCH_GEOM(RedPixelProcessInstrumented, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				instrument->updateSources, instrument->counters, instrument->traceRecords,
				instrument->traceMap, INSTRUMENT_PARAMS(redPass, 2, iter, 0), iter);
		} else {
			LAUNCH_PRODUCTION_PATCHMATCH_GEOM(RedPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths,
				cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		}
		TIMING_STOP(redPass);
		INSTRUMENT_STATE(redPass, 2, iter);
	}
	if (apdFullEnabled && params.nEstimationIters > 0) {
		const int finalIteration(params.nEstimationIters-1);
		const int finalPass(params.nEstimationIters*2);
		if (apdInstrumentEnabled) {
			const PatchMatchInstrumentKernelParams finalParams(
				INSTRUMENT_PARAMS(finalPass, 2, finalIteration, 0));
			if (visibilityEnabled)
				FinalRefineAPD<true, true, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
					cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
					apdBuffers.visibilityResolved, instrument->updateSources,
					instrument->counters, instrument->traceMap, finalParams);
			else if (params.bGeomConsistency)
				FinalRefineAPD<true, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
					cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
					apdBuffers.viewWeights, instrument->updateSources,
					instrument->counters, instrument->traceMap, finalParams);
			else
				FinalRefineAPD<false, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
					cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
					apdBuffers.viewWeights, instrument->updateSources,
					instrument->counters, instrument->traceMap, finalParams);
		} else {
			if (visibilityEnabled)
				FinalRefineAPD<true, false, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
					cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
					apdBuffers.visibilityResolved, nullptr, nullptr, nullptr,
					PatchMatchInstrumentKernelParams{});
			else LAUNCH_PRODUCTION_GEOM(FinalRefineAPD, gridSizeFull,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
				cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
				apdBuffers.viewWeights, nullptr, nullptr, nullptr,
				PatchMatchInstrumentKernelParams{});
		}
	}

	#undef TIMING_START
	#undef TIMING_STOP
	#undef APD_LATE_TIMING_START
	#undef APD_LATE_TIMING_STOP
	#undef SNAPSHOT_STATE
	#undef INSTRUMENT_STATE
	#undef LAUNCH_PRODUCTION_GEOM
	#undef LAUNCH_INSTRUMENT_GEOM
	#undef LAUNCH_PRODUCTION_PATCHMATCH_GEOM
	#undef LAUNCH_INSTRUMENT_PATCHMATCH_GEOM
	#undef LAUNCH_PRODUCTION_APD_PATCHMATCH_GEOM
	#undef LAUNCH_INSTRUMENT_APD_PATCHMATCH_GEOM
	#undef INSTRUMENT_PARAMS

	if (instrumentEnabled &&
		(instrument->validBeforeFilter || instrument->filterRejectReasons ||
		 instrument->planesBeforeFilter || instrument->costsBeforeFilter))
		CapturePreFilterState<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			cudaDepthNormalEstimates,
			cudaDepthNormalCosts,
			instrument->updateSources,
			instrument->validBeforeFilter,
			instrument->filterRejectReasons,
			instrument->planesBeforeFilter,
			instrument->costsBeforeFilter,
			width,
			height);

	if (params.fThresholdKeepCost > 0)
		FilterPlanes<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			cudaDepthNormalEstimates,
			cudaDepthNormalCosts,
			cudaSelectedViews,
			width,
			height);
	if (apdRequested) {
		ASSERT(apdMultiscaleIO && apdMultiscaleIO->outputReliability &&
			apdMultiscaleIO->outputAnchorCounts &&
			apdMultiscaleIO->outputDeformableEligible);
		APDStageClock apdOutputClock(apdClock);
		apdOutputClock.logicalIteration = static_cast<unsigned>(params.nEstimationIters);
		const uint32_t apdOutputSeed(APDStageSeed(apdOutputClock));
		if (!apdEnabled)
			SeedAPDViewWeightsFromSelectedViews<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaSelectedViews, apdBuffers.viewWeights);
		if (params.bGeomConsistency)
			ClassifyAPDProfiles<true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				apdBuffers.viewWeights, apdBuffers.reliability,
				nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
				apdSchedule.reliabilityEta,
				static_cast<unsigned>(params.nEstimationIters));
		else
			ClassifyAPDProfiles<false><<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				apdBuffers.viewWeights, apdBuffers.reliability,
				nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
				apdSchedule.reliabilityEta,
				static_cast<unsigned>(params.nEstimationIters));
		FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			apdBuffers.reliability, apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
			static_cast<int>(width), static_cast<int>(height));
		FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
			apdBuffers.nearestReliable, static_cast<int>(width), static_cast<int>(height));
		BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			cudaDepthNormalEstimates, apdBuffers.reliability, apdBuffers.nearestReliable,
			apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
			apdBuffers.fittedPlaneValid,
			nullptr, nullptr, nullptr, nullptr, PatchMatchInstrumentKernelParams{},
			apdSchedule.ransacNormalizedThreshold, apdOutputSeed,
			static_cast<unsigned>(params.nEstimationIters));
		if (depthEdgeRequested)
			FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
				apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
				apdBuffers.fittedPlaneValid, apdBuffers.depthEdgeAnchorCountsBefore,
				apdBuffers.depthEdgeAnchorCountsAfter,
				apdBuffers.depthEdgeRejectedAnchorCounts, params.nDVPDepthEdgeMode,
				apdSchedule.ransacNormalizedThreshold, apdOutputSeed);
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputReliability,
			apdBuffers.reliability, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputAnchorCounts,
			apdBuffers.anchorCounts, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputDeformableEligible,
			apdBuffers.fittedPlaneValid, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		if (depthEdgeRequested)
			CopyDVPDepthEdgeInstrumentation(apdBuffers, *apdMultiscaleIO,
				static_cast<size_t>(width)*height, cudaStream);
		ReleaseAPDDeviceBuffers(apdBuffers, cudaStream);
	}

	if (timingEnabled) {
		if (apdFullEnabled && numPasses > 1)
			CUDA_CHECK(cudaEventSynchronize(apdLateTimingStop.back()));
		else
			CUDA_CHECK(cudaEventSynchronize(timingStop.back()));
		for (int pass = 0; pass < numPasses; ++pass) {
			CUDA_CHECK(cudaEventElapsedTime(&instrument->kernelTimingsMs[pass], timingStart[(size_t)pass], timingStop[(size_t)pass]));
			if (apdFullEnabled && pass > 0) {
				float lateMilliseconds(0.f);
				CUDA_CHECK(cudaEventElapsedTime(&lateMilliseconds,
					apdLateTimingStart[(size_t)pass], apdLateTimingStop[(size_t)pass]));
				instrument->kernelTimingsMs[pass] += lateMilliseconds;
				CUDA_CHECK(cudaEventDestroy(apdLateTimingStart[(size_t)pass]));
				CUDA_CHECK(cudaEventDestroy(apdLateTimingStop[(size_t)pass]));
			}
			CUDA_CHECK(cudaEventDestroy(timingStart[(size_t)pass]));
			CUDA_CHECK(cudaEventDestroy(timingStop[(size_t)pass]));
		}
	}
	if (restoreStackSize) {
		CUDA_CHECK(cudaStreamSynchronize(cudaStream));
		CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, previousStackSize));
	}

	cudaMemcpyAsync(depthNormalEstimates, cudaDepthNormalEstimates, sizeof(Point4) * width * height, cudaMemcpyDeviceToHost, cudaStream);
	if (ptrCostMap)
		cudaMemcpyAsync(ptrCostMap, cudaDepthNormalCosts, sizeof(float) * width * height, cudaMemcpyDeviceToHost, cudaStream);
	if (ptrViewsMap)
		cudaMemcpyAsync(ptrViewsMap, cudaSelectedViews, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost, cudaStream);
	if (ptrUpdateSources && instrument && instrument->updateSources)
		cudaMemcpyAsync(ptrUpdateSources, instrument->updateSources, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost, cudaStream);
}
#else
__host__ void PatchMatch::RunCUDA(float* ptrCostMap, uint32_t* ptrViewsMap,
	const PatchMatchAPDMultiscaleIO* apdMultiscaleIO)
{
	const unsigned width = cameras[0].size.x();
	const unsigned height = cameras[0].size.y();
	const bool apdRequested(APDModeEnabled(params.nAPDMode));
	const bool apdFullRequested(APDModeUsesFullMechanics(params.nAPDMode));
	const bool depthEdgeRequested(DVPDepthEdgeModeEnabled(params.nDVPDepthEdgeMode));
	ASSERT(params.nAPDMode <= static_cast<unsigned>(APDMode::DEFORMATION_ONLY));
	ASSERT(ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{
		params.nDVPDepthEdgeMode, apdRequested,
		apdMultiscaleIO && apdMultiscaleIO->depthEdgeRegions}) ==
		DVPDepthEdgeConfigStatus::VALID);
	const DVPConfig dvpConfig{
		params.nDVPEpipolarFamily,
		params.fDVPEpipolarAlpha,
		params.fDVPEpipolarBeta,
		params.nDVPEpipolarMu,
		params.nDVPGlobalSearchRadius,
		params.fDVPReprojectionThreshold,
		params.fDVPRelativeDepthThreshold,
	};
	ASSERT(ValidateDVPConfig(dvpConfig) == DVPConfigStatus::VALID);
	(void)dvpConfig;
	const bool dvpRequested(DVPEpipolarFamilyEnabled(params.nDVPEpipolarFamily));
	ASSERT(!dvpRequested || params.bGeomConsistency);
	const bool dvpEnabled(dvpRequested && params.bGeomConsistency);
	APDStageClock apdClock;
	apdClock.levelIndex = params.nAPDLevelIndex;
	apdClock.levelCount = params.nAPDLevelCount;
	apdClock.stageIndex = params.nAPDStageIndex;
	apdClock.hasTransferredState = params.bAPDTransferredState;
	apdClock.geometricConsistency = params.bGeomConsistency;
	const APDStageSchedule apdSchedule(ResolveAPDStageSchedule(apdClock));
	ASSERT(!apdRequested || apdSchedule.valid);
	const bool apdEnabled(apdRequested && !apdSchedule.conventional);
	const bool apdFullEnabled(apdEnabled && apdFullRequested);
	ASSERT(!apdEnabled || (apdMultiscaleIO && apdMultiscaleIO->transferredReliability));
	const bool visibilityRequested(DVPVisibilityModeEnabled(params.nDVPVisibilityMode));
	ASSERT(!visibilityRequested || apdFullRequested);
	const bool visibilityEnabled(
		visibilityRequested && apdFullEnabled && params.bGeomConsistency);
	if (visibilityEnabled) {
		ASSERT(ValidateDVPVisibilityConfig(DVPVisibilityConfig{
			params.nDVPVisibilityMode,
			params.fDVPVisibilityReprojectionThreshold,
			params.fDVPVisibilityRelativeDepthThreshold,
			true,
			true,
		}) == DVPVisibilityConfigStatus::VALID);
	}
	const bool visibleNormalRequested(
		DVPVisibleNormalModeEnabled(params.nDVPVisibleNormalMode));
	ASSERT(!visibleNormalRequested || apdFullRequested);
	ASSERT(ValidateDVPVisibleNormalConfig(DVPVisibleNormalConfig{
		params.nDVPVisibleNormalMode,
		params.fDVPVisibleNormalDotTolerance,
		params.nDVPVisibleNormalAttempts,
		apdFullRequested,
	}) == DVPVisibleNormalConfigStatus::VALID);
	const bool visibleNormalEnabled(visibleNormalRequested && apdFullEnabled);

	constexpr unsigned BLOCK_W = 32;
	// BLOCK_H is selected by PATCHMATCHCUDA_LB_256_2 (build-time toggle)
	constexpr unsigned BLOCK_H = (BLOCK_W / PATCHMATCHCUDA_BLOCK_H_DIV);

	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const dim3 gridSizeFull((width + BLOCK_W - 1) / BLOCK_W, (height + BLOCK_H - 1) / BLOCK_H, 1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);
	APDDeviceBuffers apdBuffers;
	if (apdRequested) {
		AllocateAPDDeviceBuffers(apdBuffers, static_cast<size_t>(width)*height, cudaStream);
		if (visibilityEnabled)
			AllocateDVPVisibilityDeviceBuffers(
				apdBuffers, static_cast<size_t>(width)*height, cudaStream);
		if (depthEdgeRequested)
			AllocateDVPDepthEdgeDeviceBuffers(apdBuffers, *apdMultiscaleIO,
				static_cast<size_t>(width)*height, cudaStream);
		if (apdMultiscaleIO && apdMultiscaleIO->transferredReliability) {
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.transferredReliability,
				apdMultiscaleIO->transferredReliability, sizeof(uint8_t)*width*height,
				cudaMemcpyHostToDevice, cudaStream));
		} else {
			CUDA_CHECK(cudaMemsetAsync(apdBuffers.transferredReliability,
				static_cast<int>(APDReliabilityClass::UNKNOWN),
				sizeof(uint8_t)*width*height, cudaStream));
		}
	}

	// refresh constant-memory params for this pyramid level
	UploadParams();

	// dispatch templated kernels by bGeomConsistency
	#define LAUNCH_GEOM(KERNEL, GRID, ...) { \
			if (params.bGeomConsistency) \
				KERNEL<true ><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			else \
				KERNEL<false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
		}
	#define LAUNCH_PATCHMATCH_GEOM(KERNEL, GRID, ...) { \
			if (dvpEnabled) { \
				if (params.bGeomConsistency) \
					KERNEL<true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} else { \
				if (params.bGeomConsistency) \
					KERNEL<true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				else \
					KERNEL<false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			} \
		}
	#define LAUNCH_APD_PATCHMATCH_GEOM(KERNEL, GRID, ...) { \
			if (visibilityEnabled) { \
				if (dvpEnabled) { \
					if (visibleNormalEnabled) KERNEL<true, true, true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, true, true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<true, false, true, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, false, true, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} else if (dvpEnabled) { \
				if (params.bGeomConsistency) { \
					if (visibleNormalEnabled) KERNEL<true, true, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, true, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<false, true, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<false, true, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} else { \
				if (params.bGeomConsistency) { \
					if (visibleNormalEnabled) KERNEL<true, false, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<true, false, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} else { \
					if (visibleNormalEnabled) KERNEL<false, false, false, true><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
					else KERNEL<false, false, false, false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
				} \
			} \
		}

	// Pure queueing path: stream ordering on cudaStream already chains kernels;
	// caller (EstimateDepthMap) syncs the stream once before reading results.
	if (apdEnabled) {
		LAUNCH_GEOM(InitializeScoreAPD, gridSizeFull,
			cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
			cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, apdBuffers.viewWeights);
	} else {
		LAUNCH_GEOM(InitializeScore, gridSizeFull, cudaTextureImages, cudaTextureDepths,
			cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews);
	}
	if (visibilityEnabled) {
		CUDA_CHECK(cudaMemcpyAsync(apdBuffers.visibilityPrevious, apdBuffers.viewWeights,
			sizeof(uint8_t)*width*height*MAX_VIEWS,
			cudaMemcpyDeviceToDevice, cudaStream));
	}

	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		if (apdEnabled) {
			if (visibilityEnabled)
				ResolveDVPVisibilityState<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaTextureDepths, cudaDepthNormalEstimates,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext);
			LAUNCH_GEOM(ClassifyAPDProfiles, gridSizeFull,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				apdBuffers.viewWeights, apdBuffers.reliability,
				apdSchedule.reliabilityEta, static_cast<unsigned>(iter));
			const uint8_t* effectiveReliability(
				iter == 0 ? apdBuffers.transferredReliability : apdBuffers.reliability);
			APDStageClock apdIterationClock(apdClock);
			apdIterationClock.logicalIteration = static_cast<unsigned>(iter);
			const uint32_t apdStageSeed(APDStageSeed(apdIterationClock));
			if (!apdFullEnabled) {
				FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					effectiveReliability, apdBuffers.rowSquaredDistances,
					apdBuffers.rowNearestX, static_cast<int>(width), static_cast<int>(height));
				FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
					apdBuffers.nearestReliable, static_cast<int>(width), static_cast<int>(height));
				BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
					apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
					apdBuffers.fittedPlaneValid, apdSchedule.ransacNormalizedThreshold,
					apdStageSeed, static_cast<unsigned>(iter));
				if (depthEdgeRequested)
					FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
						cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
						apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
						apdBuffers.fittedPlaneValid, nullptr, nullptr, nullptr,
						params.nDVPDepthEdgeMode, apdSchedule.ransacNormalizedThreshold,
						apdStageSeed);
				CUDA_CHECK(cudaMemcpyAsync(apdBuffers.planesSnapshot, cudaDepthNormalEstimates,
					sizeof(Point4)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
				CUDA_CHECK(cudaMemcpyAsync(apdBuffers.selectedViewsSnapshot, cudaSelectedViews,
					sizeof(uint32_t)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
				LAUNCH_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::ALL, iter);
				LAUNCH_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
					cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
					cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
					effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
					apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
					apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
					APDUpdateStage::ALL, iter);
				continue;
			}
			LAUNCH_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
				apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
				apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
				APDUpdateStage::RELIABLE, iter);
			LAUNCH_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
				apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
				apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
				APDUpdateStage::RELIABLE, iter);
			FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				effectiveReliability, apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
				static_cast<int>(width), static_cast<int>(height));
			FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX, apdBuffers.nearestReliable,
				static_cast<int>(width), static_cast<int>(height));
			BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaDepthNormalEstimates, effectiveReliability, apdBuffers.nearestReliable,
				apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
				apdBuffers.fittedPlaneValid, apdSchedule.ransacNormalizedThreshold,
				apdStageSeed, static_cast<unsigned>(iter));
			if (depthEdgeRequested)
				FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
					cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
					apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
					apdBuffers.fittedPlaneValid, nullptr, nullptr, nullptr,
					params.nDVPDepthEdgeMode, apdSchedule.ransacNormalizedThreshold,
					apdStageSeed);
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.planesSnapshot, cudaDepthNormalEstimates,
				sizeof(Point4)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
			CUDA_CHECK(cudaMemcpyAsync(apdBuffers.selectedViewsSnapshot, cudaSelectedViews,
				sizeof(uint32_t)*width*height, cudaMemcpyDeviceToDevice, cudaStream));
			LAUNCH_APD_PATCHMATCH_GEOM(BlackPixelProcessAPD, gridSizeCheckerboard,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
				apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
				apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
					apdBuffers.visibilityNext,
				APDUpdateStage::NON_RELIABLE, iter);
			LAUNCH_APD_PATCHMATCH_GEOM(RedPixelProcessAPD, gridSizeCheckerboard,
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
				cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				effectiveReliability, apdBuffers.anchors, apdBuffers.anchorCounts,
				apdBuffers.planesSnapshot, apdBuffers.selectedViewsSnapshot,
				apdBuffers.fittedPlanes, apdBuffers.fittedPlaneValid, apdBuffers.viewWeights,
					apdBuffers.visibilityPrevious, apdBuffers.visibilityResolved,
				apdBuffers.visibilityNext,
				APDUpdateStage::NON_RELIABLE, iter);
			if (visibilityEnabled)
				std::swap(apdBuffers.visibilityPrevious, apdBuffers.visibilityNext);
		} else {
			LAUNCH_PATCHMATCH_GEOM(BlackPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
			LAUNCH_PATCHMATCH_GEOM(RedPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		}
	}
	if (apdFullEnabled && params.nEstimationIters > 0) {
		if (visibilityEnabled)
			FinalRefineAPD<true, false, true><<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
				cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews,
				apdBuffers.visibilityResolved);
		else LAUNCH_GEOM(FinalRefineAPD, gridSizeFull,
			cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates,
			cudaLowDepths, cudaDepthNormalCosts, cudaSelectedViews, apdBuffers.viewWeights);
	}

	if (params.fThresholdKeepCost > 0)
		FilterPlanes<<<gridSizeFull, blockSize, 0, cudaStream>>>(cudaDepthNormalEstimates, cudaDepthNormalCosts, cudaSelectedViews, width, height);
	if (apdRequested) {
		ASSERT(apdMultiscaleIO && apdMultiscaleIO->outputReliability &&
			apdMultiscaleIO->outputAnchorCounts &&
			apdMultiscaleIO->outputDeformableEligible);
		APDStageClock apdOutputClock(apdClock);
		apdOutputClock.logicalIteration = static_cast<unsigned>(params.nEstimationIters);
		const uint32_t apdOutputSeed(APDStageSeed(apdOutputClock));
		if (!apdEnabled)
			SeedAPDViewWeightsFromSelectedViews<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaSelectedViews, apdBuffers.viewWeights);
		LAUNCH_GEOM(ClassifyAPDProfiles, gridSizeFull,
			cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths,
			apdBuffers.viewWeights, apdBuffers.reliability,
			apdSchedule.reliabilityEta, static_cast<unsigned>(params.nEstimationIters));
		FindAPDNearestReliableRows<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			apdBuffers.reliability, apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
			static_cast<int>(width), static_cast<int>(height));
		FindAPDNearestReliableColumns<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			apdBuffers.rowSquaredDistances, apdBuffers.rowNearestX,
			apdBuffers.nearestReliable, static_cast<int>(width), static_cast<int>(height));
		BuildAPDAnchors<<<gridSizeFull, blockSize, 0, cudaStream>>>(
			cudaDepthNormalEstimates, apdBuffers.reliability, apdBuffers.nearestReliable,
			apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
			apdBuffers.fittedPlaneValid, apdSchedule.ransacNormalizedThreshold,
			apdOutputSeed, static_cast<unsigned>(params.nEstimationIters));
		if (depthEdgeRequested)
			FilterAPDAnchorsByDepthEdgeRegion<<<gridSizeFull, blockSize, 0, cudaStream>>>(
				cudaDepthNormalEstimates, apdBuffers.depthEdgeRegions,
				apdBuffers.anchors, apdBuffers.anchorCounts, apdBuffers.fittedPlanes,
				apdBuffers.fittedPlaneValid, nullptr, nullptr, nullptr,
				params.nDVPDepthEdgeMode, apdSchedule.ransacNormalizedThreshold,
				apdOutputSeed);
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputReliability,
			apdBuffers.reliability, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputAnchorCounts,
			apdBuffers.anchorCounts, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		CUDA_CHECK(cudaMemcpyAsync(apdMultiscaleIO->outputDeformableEligible,
			apdBuffers.fittedPlaneValid, sizeof(uint8_t)*width*height,
			cudaMemcpyDeviceToHost, cudaStream));
		ReleaseAPDDeviceBuffers(apdBuffers, cudaStream);
	}

	#undef LAUNCH_GEOM
	#undef LAUNCH_PATCHMATCH_GEOM
	#undef LAUNCH_APD_PATCHMATCH_GEOM

	cudaMemcpyAsync(depthNormalEstimates, cudaDepthNormalEstimates, sizeof(Point4) * width * height, cudaMemcpyDeviceToHost, cudaStream);
	if (ptrCostMap)
		cudaMemcpyAsync(ptrCostMap, cudaDepthNormalCosts, sizeof(float) * width * height, cudaMemcpyDeviceToHost, cudaStream);
	if (ptrViewsMap)
		cudaMemcpyAsync(ptrViewsMap, cudaSelectedViews, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost, cudaStream);
}
#endif
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS
