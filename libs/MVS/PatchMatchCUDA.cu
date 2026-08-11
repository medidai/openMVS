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
#ifdef _USE_DMAP_INSTRUMENTATION
#define PATCHMATCHCUDA_INSTRUMENT_LAUNCH_BOUNDS __launch_bounds__(256, 1)
#endif
#else
#define PATCHMATCHCUDA_BLOCK_H_DIV 2   // BLOCK_H = BLOCK_W / 2 = 16
#define PATCHMATCHCUDA_LAUNCH_BOUNDS __launch_bounds__(512, 1)
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
		if (source == PM_SOURCE_PROPAGATE)
			atomicAdd(&counter.propagationWins, 1u);
		if (source >= PM_SOURCE_REFINE_DEPTH && source <= PM_SOURCE_REFINE_SURFACE_NORMAL)
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

// randomly perturb a depth, sampling uniformly from the intersection of the
// perturbation window [(1-p)d, (1+p)d] with the valid range [fDepthMin, fDepthMax]
__device__ inline float GeneratePerturbedDepth(float depth, RandState* randState, const float perturbation)
{
	const float lo = fmaxf((1.f - perturbation) * depth, g_params.fDepthMin);
	const float hi = fminf((1.f + perturbation) * depth, g_params.fDepthMax);
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
	if (lowDepth > 0 && cache.varRef < 0.0025f) {
		const float depth(plane.w());
		const float deltaDepth(min((fabsf(lowDepth-depth) / lowDepth), 0.5f));
		constexpr float smoothSigmaDepth(-1.f / (1.f * 0.02f)); // 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
		const float factorDeltaDepth(__expf(cache.varRef * smoothSigmaDepth));
		ncc = (1.f-factorDeltaDepth)*ncc + factorDeltaDepth*deltaDepth;
	}
	return max(0.f, min(2.f, ncc));
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
		record.weightedContribution = (float)weight * components.totalCost / (float)NUM_SAMPLES;
		record.selectionPrior = selectionPriors[view];
		record.samplingScore = samplingScore;
		record.samplingProbability = probability;
		record.metadata = PackExactViewMetadata(
			weight, ExactProbabilityRank(cumulativeProbabilities, view, numViews),
			agreeCount, badCount, decision, selected, selected,
			IsFiniteCandidateCost(components.totalCost), true);
	}
	NormalizeInstrumentCostComponents(aggregate, (float)NUM_SAMPLES);
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
template <bool GEOM, bool INSTRUMENT>
__device__ void ProcessPixel(
	const ImagePixels* images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const Point2i& p, const int iter)
#else
template <bool GEOM>
__device__ void ProcessPixel(const ImagePixels* images, const ImagePixels* depthImages, Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews, const Point2i& p, const int iter)
#endif
{
	const int width = g_cameras[0].size.x();
	const int height = g_cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
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
				} else if (costArray[posId][imgId] >= fBadCost) {
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
#ifdef _USE_DMAP_INSTRUMENTATION
	int updateSource = PM_SOURCE_NONE;
#endif
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
#ifdef _USE_DMAP_INSTRUMENTATION
	const float exactIncumbentCost(cost);
	if constexpr (INSTRUMENT) {
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
			TrackExactCandidate(
				PM_EXACT_CANDIDATE_PROPAGATION_0 + posId, finalCosts[posId],
				exactTestedMask, exactFiniteMask, exactTestedCount, exactFiniteCount,
				exactBestCost, exactBestSlot, exactRunnerUpCost, exactRunnerUpSlot);
		}
	}
#endif
#ifdef _USE_DMAP_INSTRUMENTATION
	const bool propagationAccepted = minCostIdx >= 0 && valid[minCostIdx] && finalCosts[minCostIdx] < cost;
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
		plane = LoadPlaneLDG(&planes[positions[minCostIdx]]);
		plane.w() = neighborDepths[minCostIdx];
		cost = finalCosts[minCostIdx];
		selectedViews[idx] = newSelectedViews;
#ifdef _USE_DMAP_INSTRUMENTATION
		updateSource = PM_SOURCE_PROPAGATE;
		if constexpr (INSTRUMENT) {
			exactWinnerSlot = (uint8_t)(PM_EXACT_CANDIDATE_PROPAGATION_0 + minCostIdx);
			exactAcceptedMask |= 1u << exactWinnerSlot;
			++exactAcceptedCount;
			for (int view = 0; view < nNumViews; ++view)
				exactWinningViewCosts[view] = costArray[minCostIdx][view];
		}
#endif
	}
	const float depth = plane.w();

	// refine estimate
	constexpr float perturbationDepth = 0.005f;
	constexpr float perturbationNormal = 0.01f * (float)M_PI;
	const float depthPerturbed = GeneratePerturbedDepth(depth, randState, perturbationDepth);
	const Point3 perturbedNormal = GeneratePerturbedNormal(g_cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal);
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
				exactWinnerSlot = (uint8_t)slot;
				exactAcceptedMask |= 1u << slot;
				++exactAcceptedCount;
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
	if constexpr (INSTRUMENT) {
		const float confidenceGap = exactFiniteCount >= 2 && exactRunnerUpSlot != PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE ?
			max(0.f, exactRunnerUpCost-exactBestCost) : -1.f;
		const float viewEntropy = ComputeViewEntropy(viewWeights, nNumViews, (float)NUM_SAMPLES);
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
template <bool GEOM, bool INSTRUMENT>
__device__ void InitializePixelScore(
	const ImagePixels *images, const ImagePixels* depthImages,
	Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews,
	uint8_t* updateSources,
	PatchMatchInstrumentCounters* instrumentCounters,
	PatchMatchInstrumentTraceRecord* instrumentTraceRecords,
	const int32_t* instrumentTraceMap,
	const PatchMatchInstrumentKernelParams& instrumentParams,
	const Point2i& p)
#else
template <bool GEOM>
__device__ void InitializePixelScore(const ImagePixels *images, const ImagePixels* depthImages, Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews, const Point2i& p)
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

// kernels are GEOM-templated; nvcc emits separate binaries with the
// geom-consistency loop eliminated when off; runtime params come from
// __constant__ g_params (uploaded per pyramid level by UploadParams())
template <bool GEOM>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void InitializeScore(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews)
{
	const Point2i p = GetThreadIndex2();
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	InitializePixelScore<GEOM, false>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p);
#else
	InitializePixelScore<GEOM>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, p);
#endif
}

// traverse image in a back/red checkerboard pattern
template <bool GEOM>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void BlackPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 0 : 1);
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, p, iter);
#endif
}
template <bool GEOM>
__global__ PATCHMATCHCUDA_LAUNCH_BOUNDS void RedPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 1 : 0);
#ifdef _USE_DMAP_INSTRUMENTATION
	const PatchMatchInstrumentKernelParams instrumentParams;
	ProcessPixel<GEOM, false>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		nullptr, nullptr, nullptr, nullptr, instrumentParams, p, iter);
#else
	ProcessPixel<GEOM>((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, planes, lowDepths, costs, (RandState*)randStates, selectedViews, p, iter);
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
	InitializePixelScore<GEOM, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap, instrumentParams, p);
}

template <bool GEOM>
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
	ProcessPixel<GEOM, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
		updateSources, instrumentCounters, instrumentTraceRecords, instrumentTraceMap, instrumentParams, p, iter);
}

template <bool GEOM>
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
	ProcessPixel<GEOM, true>(
		(const ImagePixels*)textureImages, (const ImagePixels*)textureDepths,
		planes, lowDepths, costs, (RandState*)randStates, selectedViews,
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
	const Point4 before = planesBefore[idx];
	const Point4 after = planes[idx];
	const float depthBefore = instrumentParams.passIndex == 0 ? after.w() : before.w();
	const float depthAfter = after.w();
	const float costBefore = instrumentParams.passIndex == 0 ? costs[idx] : costsBefore[idx];
	const float costAfter = costs[idx];
	const unsigned viewsBefore = instrumentParams.passIndex == 0 ? 0u : selectedViewsBefore[idx];
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

#ifdef _USE_DMAP_INSTRUMENTATION
__host__ void PatchMatch::RunCUDA(float* ptrCostMap, uint32_t* ptrViewsMap, uint8_t* ptrUpdateSources, PatchMatchInstrumentDeviceContext* instrument)
{
	const unsigned width = cameras[0].size.x();
	const unsigned height = cameras[0].size.y();
	const int numPasses = 1 + params.nEstimationIters * 2;
	const bool instrumentEnabled = instrument && instrument->counters;
	const bool exactEnabled = instrumentEnabled && instrument->exact;
	const bool sampledEnabled = instrumentEnabled && instrument->sampled && instrument->traceRecords && instrument->traceMap && instrument->numTracePixels > 0;
	const bool timingEnabled = instrumentEnabled && instrument->kernelTimingsMs;
	size_t previousStackSize = 0;
	bool restoreStackSize = false;
	if (instrumentEnabled) {
		size_t stackSize = 0;
		CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
		const size_t requiredStackSize(exactEnabled ? PM_INSTRUMENT_EXACT_STACK_BYTES : 2048);
		if (stackSize < requiredStackSize) {
			previousStackSize = stackSize;
			restoreStackSize = true;
			CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, requiredStackSize));
		}
	}
	std::vector<cudaEvent_t> timingStart;
	std::vector<cudaEvent_t> timingStop;
	if (timingEnabled) {
		timingStart.resize((size_t)numPasses, nullptr);
		timingStop.resize((size_t)numPasses, nullptr);
		for (int pass = 0; pass < numPasses; ++pass) {
			CUDA_CHECK(cudaEventCreate(&timingStart[(size_t)pass]));
			CUDA_CHECK(cudaEventCreate(&timingStop[(size_t)pass]));
		}
	}

	constexpr unsigned BLOCK_W = 32;
	// BLOCK_H is selected by PATCHMATCHCUDA_LB_256_2 (build-time toggle)
	constexpr unsigned BLOCK_H = (BLOCK_W / PATCHMATCHCUDA_BLOCK_H_DIV);

	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const dim3 gridSizeFull((width + BLOCK_W - 1) / BLOCK_W, (height + BLOCK_H - 1) / BLOCK_H, 1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);

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
			((PROXY_ONLY) && exactEnabled) ? 1 : 0, \
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
			instrument ? instrument->exactViews : nullptr \
		}
	#define TIMING_START(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(timingStart[(size_t)(PASS)], cudaStream))
	#define TIMING_STOP(PASS) \
		if (timingEnabled) CUDA_CHECK(cudaEventRecord(timingStop[(size_t)(PASS)], cudaStream))
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
	SNAPSHOT_STATE();
	TIMING_START(0);
	if (exactEnabled) {
		LAUNCH_INSTRUMENT_GEOM(InitializeScoreInstrumented, gridSizeFull, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
			instrument->updateSources, instrument->counters, instrument->traceRecords,
			instrument->traceMap, INSTRUMENT_PARAMS(0, 0, -1, 0));
	} else {
		LAUNCH_PRODUCTION_GEOM(InitializeScore, gridSizeFull, cudaTextureImages, cudaTextureDepths,
			cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews);
	}
	TIMING_STOP(0);
	INSTRUMENT_STATE(0, 0, -1);

	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		const int blackPass = 1 + iter * 2;
		const int redPass = blackPass + 1;
		SNAPSHOT_STATE();
		TIMING_START(blackPass);
		if (exactEnabled) {
			LAUNCH_INSTRUMENT_GEOM(BlackPixelProcessInstrumented, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				instrument->updateSources, instrument->counters, instrument->traceRecords,
				instrument->traceMap, INSTRUMENT_PARAMS(blackPass, 1, iter, 0), iter);
		} else {
			LAUNCH_PRODUCTION_GEOM(BlackPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths,
				cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		}
		TIMING_STOP(blackPass);
		INSTRUMENT_STATE(blackPass, 1, iter);
		SNAPSHOT_STATE();
		TIMING_START(redPass);
		if (exactEnabled) {
			LAUNCH_INSTRUMENT_GEOM(RedPixelProcessInstrumented, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews,
				instrument->updateSources, instrument->counters, instrument->traceRecords,
				instrument->traceMap, INSTRUMENT_PARAMS(redPass, 2, iter, 0), iter);
		} else {
			LAUNCH_PRODUCTION_GEOM(RedPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths,
				cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		}
		TIMING_STOP(redPass);
		INSTRUMENT_STATE(redPass, 2, iter);
	}

	#undef TIMING_START
	#undef TIMING_STOP
	#undef SNAPSHOT_STATE
	#undef INSTRUMENT_STATE
	#undef LAUNCH_PRODUCTION_GEOM
	#undef LAUNCH_INSTRUMENT_GEOM
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

	if (timingEnabled) {
		CUDA_CHECK(cudaEventSynchronize(timingStop.back()));
		for (int pass = 0; pass < numPasses; ++pass) {
			CUDA_CHECK(cudaEventElapsedTime(&instrument->kernelTimingsMs[pass], timingStart[(size_t)pass], timingStop[(size_t)pass]));
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
__host__ void PatchMatch::RunCUDA(float* ptrCostMap, uint32_t* ptrViewsMap)
{
	const unsigned width = cameras[0].size.x();
	const unsigned height = cameras[0].size.y();

	constexpr unsigned BLOCK_W = 32;
	// BLOCK_H is selected by PATCHMATCHCUDA_LB_256_2 (build-time toggle)
	constexpr unsigned BLOCK_H = (BLOCK_W / PATCHMATCHCUDA_BLOCK_H_DIV);

	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const dim3 gridSizeFull((width + BLOCK_W - 1) / BLOCK_W, (height + BLOCK_H - 1) / BLOCK_H, 1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);

	// refresh constant-memory params for this pyramid level
	UploadParams();

	// dispatch templated kernels by bGeomConsistency
	#define LAUNCH_GEOM(KERNEL, GRID, ...) { \
			if (params.bGeomConsistency) \
				KERNEL<true ><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
			else \
				KERNEL<false><<<GRID, blockSize, 0, cudaStream>>>(__VA_ARGS__); \
		}

	// Pure queueing path: stream ordering on cudaStream already chains kernels;
	// caller (EstimateDepthMap) syncs the stream once before reading results.
	LAUNCH_GEOM(InitializeScore, gridSizeFull, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews);

	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		LAUNCH_GEOM(BlackPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
		LAUNCH_GEOM(RedPixelProcess, gridSizeCheckerboard, cudaTextureImages, cudaTextureDepths, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, iter);
	}

	#undef LAUNCH_GEOM

	if (params.fThresholdKeepCost > 0)
		FilterPlanes<<<gridSizeFull, blockSize, 0, cudaStream>>>(cudaDepthNormalEstimates, cudaDepthNormalCosts, cudaSelectedViews, width, height);

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
