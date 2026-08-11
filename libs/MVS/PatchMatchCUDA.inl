/*
* PatchMatchCUDA.inl
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

#ifndef _MVS_PATCHMATCHCUDA_INL_
#define _MVS_PATCHMATCHCUDA_INL_


// I N C L U D E S /////////////////////////////////////////////////

#include "CUDA/Camera.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

struct DepthData;

namespace CUDA {
#ifdef _USE_DMAP_INSTRUMENTATION
static constexpr int PM_INSTRUMENT_MAX_VIEWS = 32;
static constexpr int PM_INSTRUMENT_NUM_NEIGHBORS = 8;
static constexpr int PM_INSTRUMENT_NUM_SOURCES = 9;
static constexpr int PM_INSTRUMENT_NUM_BAD_REASONS = 6;
static constexpr int PM_INSTRUMENT_NUM_UPDATE_BINS = 8;
static constexpr int PM_INSTRUMENT_NUM_CANDIDATE_TYPES = 4;
static constexpr int PM_INSTRUMENT_MAP_VIEWS = 4;
static constexpr int PM_INSTRUMENT_EXACT_NUM_CANDIDATES = 13;
static constexpr uint8_t PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE = 255;
static constexpr size_t PM_INSTRUMENT_EXACT_STACK_BYTES = 4096;

enum PatchMatchInstrumentExactCandidateSlot : uint8_t {
	PM_EXACT_CANDIDATE_CURRENT = 0,
	PM_EXACT_CANDIDATE_PROPAGATION_0 = 1,
	PM_EXACT_CANDIDATE_PROPAGATION_7 = 8,
	PM_EXACT_CANDIDATE_REFINE_DEPTH = 9,
	PM_EXACT_CANDIDATE_REFINE_NORMAL = 10,
	PM_EXACT_CANDIDATE_REFINE_RANDOM_NORMAL = 11,
	PM_EXACT_CANDIDATE_REFINE_SURFACE_NORMAL = 12,
};

enum PatchMatchInstrumentExactViewDecision : uint8_t {
	PM_EXACT_VIEW_UNAVAILABLE = 0,
	PM_EXACT_VIEW_SELECTED_MC = 1,
	PM_EXACT_VIEW_REJECTED_ZERO_SCORE = 2,
	PM_EXACT_VIEW_REJECTED_NOT_SAMPLED = 3,
	PM_EXACT_VIEW_INIT_TOP_K = 4,
	PM_EXACT_VIEW_INIT_THRESHOLD_TIE = 5,
	PM_EXACT_VIEW_INIT_REJECTED = 6,
};

// PatchMatchInstrumentExactView::metadata bit layout. Rank is zero-based and
// deterministically breaks equal values by view index; 63 means unavailable.
static constexpr uint32_t PM_EXACT_VIEW_WEIGHT_SHIFT = 0;
static constexpr uint32_t PM_EXACT_VIEW_WEIGHT_MASK = 0x3Fu;
static constexpr uint32_t PM_EXACT_VIEW_RANK_SHIFT = 6;
static constexpr uint32_t PM_EXACT_VIEW_RANK_MASK = 0x3Fu;
static constexpr uint32_t PM_EXACT_VIEW_AGREE_SHIFT = 12;
static constexpr uint32_t PM_EXACT_VIEW_AGREE_MASK = 0xFu;
static constexpr uint32_t PM_EXACT_VIEW_BAD_SHIFT = 16;
static constexpr uint32_t PM_EXACT_VIEW_BAD_MASK = 0xFu;
static constexpr uint32_t PM_EXACT_VIEW_DECISION_SHIFT = 20;
static constexpr uint32_t PM_EXACT_VIEW_DECISION_MASK = 0x7u;
static constexpr uint32_t PM_EXACT_VIEW_SELECTED_BIT = 1u << 23;
static constexpr uint32_t PM_EXACT_VIEW_CONTRIBUTION_BIT = 1u << 24;
static constexpr uint32_t PM_EXACT_VIEW_FINITE_BIT = 1u << 25;
static constexpr uint32_t PM_EXACT_VIEW_PROBABILITY_BIT = 1u << 26;

enum PatchMatchInstrumentCandidateType : uint8_t {
	PM_CANDIDATE_INIT = 0,
	PM_CANDIDATE_PROPAGATION = 1,
	PM_CANDIDATE_RANDOM_PERTURBATION = 2,
	PM_CANDIDATE_REFINEMENT = 3,
};

enum PatchMatchInstrumentSource : uint8_t {
	PM_SOURCE_NONE = 0,
	PM_SOURCE_INIT = 1,
	PM_SOURCE_PROPAGATE = 2,
	PM_SOURCE_REFINE_DEPTH = 3,
	PM_SOURCE_REFINE_NORMAL = 4,
	PM_SOURCE_REFINE_RANDOM_NORMAL = 5,
	PM_SOURCE_REFINE_SURFACE_NORMAL = 6,
	PM_SOURCE_FILTERED = 7,
	PM_SOURCE_CHANGED_UNKNOWN = 8,
};

enum PatchMatchInstrumentBadReason : uint8_t {
	PM_BAD_NONE = 0,
	PM_BAD_OUT_OF_BOUNDS = 1,
	PM_BAD_INVALID_PROJECTION = 2,
	PM_BAD_LOW_REF_VARIANCE = 3,
	PM_BAD_LOW_TARGET_VARIANCE = 4,
	PM_BAD_GEOMETRIC_MISMATCH = 5,
};

struct PatchMatchInstrumentCounters {
	uint32_t processed = 0;
	uint32_t validDepth = 0;
	uint32_t invalidDepth = 0;
	uint32_t badCost = 0;
	uint32_t lowResPrior = 0;
	uint32_t lowTexture = 0;
	uint32_t propagationWins = 0;
	uint32_t refinementWins = 0;
	uint32_t accepted = 0;
	uint32_t componentSamples = 0;
	uint32_t depthPriorSamples = 0;
	uint32_t viewChurn = 0;
	uint32_t viewAddedSum = 0;
	uint32_t viewRemovedSum = 0;
	uint32_t updateMagnitudeSamples = 0;
	uint32_t candidateTested[PM_INSTRUMENT_NUM_CANDIDATE_TYPES] = {};
	uint32_t candidateFinite[PM_INSTRUMENT_NUM_CANDIDATE_TYPES] = {};
	uint32_t candidateAccepted[PM_INSTRUMENT_NUM_CANDIDATE_TYPES] = {};
	uint32_t selectedViewBins[PM_INSTRUMENT_MAX_VIEWS + 1] = {};
	uint32_t updateSource[PM_INSTRUMENT_NUM_SOURCES] = {};
	uint32_t badReason[PM_INSTRUMENT_NUM_BAD_REASONS] = {};
	uint32_t depthRelChangeBins[PM_INSTRUMENT_NUM_UPDATE_BINS] = {};
	uint32_t normalAngleBins[PM_INSTRUMENT_NUM_UPDATE_BINS] = {};
	float costBeforeSum = 0.f;
	float costSum = 0.f;
	float costSqSum = 0.f;
	float costImprovementSum = 0.f;
	float photometricCostSum = 0.f;
	float photoPriorCostSum = 0.f;
	float depthPriorCostSum = 0.f;
	float depthPriorWeightSum = 0.f;
	float geometricCostSum = 0.f;
	float viewWeightSum[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewBadCost[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewEntropySum = 0.f;
	float depthAbsChangeSum = 0.f;
	float depthRelChangeSum = 0.f;
	float normalAngleSum = 0.f;
	float viewCostWeightedSum[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewPhotometricCostWeightedSum[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewGeometricCostWeightedSum[PM_INSTRUMENT_MAX_VIEWS] = {};
};

struct PatchMatchInstrumentCostComponents {
	float photometricCost = 0.f;
	float photoPriorCost = 0.f;
	float depthPriorCost = 0.f;
	float depthPriorWeight = 0.f;
	float geometricCost = 0.f;
	float totalCost = 0.f;
	uint32_t sampleCount = 0;
	uint32_t depthPriorSamples = 0;
	uint32_t badReason[PM_INSTRUMENT_NUM_BAD_REASONS] = {};
	// Limit the hot-kernel local record to the four exported component views.
	// Full-view selected-weight counts remain available in the global counters.
	float viewWeightSum[PM_INSTRUMENT_MAP_VIEWS] = {};
	float viewBadCost[PM_INSTRUMENT_MAP_VIEWS] = {};
	float viewCostWeightedSum[PM_INSTRUMENT_MAP_VIEWS] = {};
	float viewPhotometricCostWeightedSum[PM_INSTRUMENT_MAP_VIEWS] = {};
	float viewGeometricCostWeightedSum[PM_INSTRUMENT_MAP_VIEWS] = {};
};

// Exact event summary for one ProcessPixel invocation. Iteration checkerboard
// phases write disjoint pixels into the same logical-state slice, so this
// record intentionally contains no checkerboard phase identity.
struct PatchMatchInstrumentExactPixel {
	uint32_t candidateTestedMask = 0;
	uint32_t candidateFiniteMask = 0;
	uint32_t candidateAcceptedMask = 0;
	uint32_t selectedViewsBefore = 0;
	uint32_t selectedViewsAfter = 0;
	float storedCostBefore = -1.f;
	float incumbentCost = -1.f;
	float winnerCost = -1.f;
	float runnerUpCost = -1.f;
	float winnerRunnerUpGap = -1.f;
	uint8_t source = PM_SOURCE_NONE;
	uint8_t winnerSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t runnerUpSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t testedCount = 0;
	uint8_t finiteCount = 0;
	uint8_t acceptedCount = 0;
	uint8_t selectedCountBefore = 0;
	uint8_t selectedCountAfter = 0;
};
static_assert(sizeof(PatchMatchInstrumentExactPixel) == 48, "Exact pixel record schema changed");

// Exact final-winner score and view-selection evidence for one source view.
// photometricCost is the production photo/depth-prior score. The contribution
// is weight*total/32 for PatchMatch, or deterministic top-K total/K at init.
struct PatchMatchInstrumentExactView {
	float photometricCost = -1.f;
	float geometricCost = 0.f;
	float totalCost = -1.f;
	float weightedContribution = 0.f;
	float selectionPrior = -1.f;
	float samplingScore = -1.f;
	float samplingProbability = -1.f;
	uint32_t metadata = PM_EXACT_VIEW_RANK_MASK << PM_EXACT_VIEW_RANK_SHIFT;
};
static_assert(sizeof(PatchMatchInstrumentExactView) == 32, "Exact view record schema changed");

struct PatchMatchInstrumentTraceRecord {
	int32_t valid = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t passIndex = -1;
	int32_t phase = -1;
	int32_t iteration = -1;
	int32_t x = -1;
	int32_t y = -1;
	int32_t source = PM_SOURCE_NONE;
	int32_t selectedViewCount = 0;
	uint32_t selectedViews = 0;
	float depthBefore = 0.f;
	float depthAfter = 0.f;
	float costBefore = 0.f;
	float costAfter = 0.f;
	float costImprovement = 0.f;
	float depthAbsChange = 0.f;
	float depthRelChange = 0.f;
	float normalAngleChange = 0.f;
	float viewEntropy = 0.f;
	float photometricCostAfter = 0.f;
	float photoPriorCostAfter = 0.f;
	float depthPriorCostAfter = 0.f;
	float depthPriorWeightAfter = 0.f;
	float geometricCostAfter = 0.f;
	float refVariance = 0.f;
	float lowDepth = 0.f;
	uint32_t selectedViewsBefore = 0;
	float neighborCosts[PM_INSTRUMENT_NUM_NEIGHBORS] = {};
	float viewCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewPhotometricCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewGeometricCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint32_t badReason[PM_INSTRUMENT_NUM_BAD_REASONS] = {};
	uint32_t viewWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
};

struct PatchMatchInstrumentKernelParams {
	int32_t enabled = 0;
	int32_t sampled = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t passIndex = -1;
	int32_t phase = -1;
	int32_t iteration = -1;
	int32_t numPasses = 0;
	int32_t numTracePixels = 0;
	int32_t area = 0;
	int32_t logicalStateIndex = -1;
	int32_t exactLogicalStateIndex = -1;
	int32_t numLogicalStates = 0;
	int32_t viewStride = 0;
	int32_t proxyOnly = 0;
	float* improvementMaps = nullptr;
	uint8_t* passUpdateSources = nullptr;
	float* passDepthDeltas = nullptr;
	float* passDepthRelDeltas = nullptr;
	float* passNormalAngleDeltas = nullptr;
	uint8_t* passViewChurn = nullptr;
	float* logicalStoredCosts = nullptr;
	float4* logicalScorePrimary = nullptr;
	float4* logicalScoreSecondary = nullptr;
	float4* exactLogicalScorePrimary = nullptr;
	float4* exactLogicalScoreSecondary = nullptr;
	float4* finalViewWeights = nullptr;
	float4* finalViewCosts = nullptr;
	float4* finalViewPhotometricCosts = nullptr;
	float4* finalViewGeometricCosts = nullptr;
	float* finalViewEntropy = nullptr;
	float* finalLowDepth = nullptr;
	uint32_t* finalSelectedViews = nullptr;
	uint8_t* acceptedUpdateCount = nullptr;
	PatchMatchInstrumentExactPixel* exactPixels = nullptr;
	PatchMatchInstrumentExactView* exactViews = nullptr;
};

struct PatchMatchInstrumentDeviceContext {
	PatchMatchInstrumentCounters* counters = nullptr;
	PatchMatchInstrumentTraceRecord* traceRecords = nullptr;
	const int32_t* traceMap = nullptr;
	uint8_t* updateSources = nullptr;
	uint8_t* validBeforeFilter = nullptr;
	uint8_t* filterRejectReasons = nullptr;
	float* improvementMaps = nullptr;
	uint8_t* passUpdateSources = nullptr;
	float* passDepthDeltas = nullptr;
	float* passDepthRelDeltas = nullptr;
	float* passNormalAngleDeltas = nullptr;
	uint8_t* passViewChurn = nullptr;
	float* logicalStoredCosts = nullptr;
	float4* logicalScorePrimary = nullptr;
	float4* logicalScoreSecondary = nullptr;
	float4* exactLogicalScorePrimary = nullptr;
	float4* exactLogicalScoreSecondary = nullptr;
	float4* finalViewWeights = nullptr;
	float4* finalViewCosts = nullptr;
	float4* finalViewPhotometricCosts = nullptr;
	float4* finalViewGeometricCosts = nullptr;
	float* finalViewEntropy = nullptr;
	float* finalLowDepth = nullptr;
	uint32_t* finalSelectedViews = nullptr;
	uint8_t* acceptedUpdateCount = nullptr;
	PatchMatchInstrumentExactPixel* exactPixels = nullptr;
	PatchMatchInstrumentExactView* exactViews = nullptr;
	Point4* planesBeforeFilter = nullptr;
	float* costsBeforeFilter = nullptr;
	Point4* planesBeforePass = nullptr;
	float* costsBeforePass = nullptr;
	uint32_t* selectedViewsBeforePass = nullptr;
	float* kernelTimingsMs = nullptr;
	int32_t numTracePixels = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t numLogicalStates = 0;
	int32_t viewStride = 0;
	bool sampled = false;
	bool exact = false;
};
#endif

class PatchMatch {
public:
	struct Params {
		int nNumViews = 5;
		int nEstimationIters = 3;
		float fDepthMin = 0.f;
		float fDepthMax = 100.f;
		int nInitTopK = 3;
		bool bGeomConsistency = false;
		bool bLowResProcessed = false;
		float fThresholdKeepCost = 0;
	};

public:
	PatchMatch();
	~PatchMatch();

	void Init(bool bGeomConsistency);
	void Release();

#ifdef _USE_DMAP_INSTRUMENTATION
	void EstimateDepthMap(DepthData&, int geometricIteration=-1);
#else
	void EstimateDepthMap(DepthData&);
#endif

	float4 GetPlaneHypothesis(const int index);
	float GetCost(const int index);

private:
	void ReleaseCUDA();
	void AllocatePatchMatchCUDA(const cv::Mat1f& image);
	void AllocateImageCUDA(size_t i, const cv::Mat1f& image, bool bInitImage, bool bInitDepthMap);
#ifdef _USE_DMAP_INSTRUMENTATION
	void RunCUDA(float* ptrCostMap=NULL, uint32_t* ptrViewsMap=NULL, uint8_t* ptrUpdateSources=NULL, PatchMatchInstrumentDeviceContext* instrument=NULL);
#else
	void RunCUDA(float* ptrCostMap=NULL, uint32_t* ptrViewsMap=NULL);
#endif
	void UploadCameras(); // upload host cameras into __constant__ g_cameras
	void UploadParams();  // upload host params into __constant__ g_params
	// For images large enough that the per-call driver-staging stall dominates
	// (default threshold ~1.5 MP), copy a pageable cv::Mat1f into a per-instance
	// pinned slot then enqueue a truly-async H->D DMA on cudaStream. For smaller
	// mats falls back to a direct pageable DMA (the driver's internal chunked
	// staging is cheap enough that the explicit memcpy + cudaHostAlloc overhead
	// would otherwise be a net loss).
	void StagedUploadCvMat(cudaArray_t dst, const cv::Mat1f& src,
		std::vector<float*>& slots, std::vector<size_t>& areas, size_t slotIdx);

public:
	Params params;

	std::vector<cv::Mat1f> images;
	std::vector<Camera> cameras;
	std::vector<cudaTextureObject_t> textureImages;
	std::vector<cudaTextureObject_t> textureDepths;
	Point4* depthNormalEstimates;

	std::vector<cudaArray_t> cudaImageArrays;
	std::vector<cudaArray_t> cudaDepthArrays;
	cudaTextureObject_t* cudaTextureImages;
	cudaTextureObject_t* cudaTextureDepths;
	Point4* cudaDepthNormalEstimates;
	float* cudaLowDepths;
	float* cudaDepthNormalCosts;
	curandState* cudaRandStates;
	uint32_t* cudaSelectedViews;
	// per-instance stream: scopes kernel launches and syncs to this PatchMatch
	// instead of fencing the whole device, and enables async H<->D transfers
	cudaStream_t cudaStream;
	// pinned host staging slots, indexed by view (image upload) or by neighbor
	// (depth-prior upload). Grown on demand by StagedUploadCvMat above the area
	// threshold; freed in Release(). Empty for workloads with small images.
	std::vector<float*> hostImageStaging;
	std::vector<size_t> hostImageStagingArea;
	std::vector<float*> hostDepthPriorStaging;
	std::vector<size_t> hostDepthPriorStagingArea;
};
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS

#endif // _MVS_PATCHMATCHCUDA_INL_
