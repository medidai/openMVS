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
#include "PatchMatchDVPCUDA.h"
#include "PatchMatchDVPDepthEdgeCUDA.h"
#include "PatchMatchDVPVisibilityCUDA.h"
#include "PatchMatchDVPVisibleNormalCUDA.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


// D E F I N E S ///////////////////////////////////////////////////

#ifdef _USE_DMAP_INSTRUMENTATION
// Observer-side contract for the fixed reference-patch layout. A source test
// keeps these values synchronized with the production CUDA scoring constants.
#define PATCHMATCHCUDA_PATCH_HALF_WINDOW 4
#define PATCHMATCHCUDA_PATCH_STEP 2
#endif


// S T R U C T S ///////////////////////////////////////////////////

namespace MVS {

struct DepthData;

namespace CUDA {
#ifdef _USE_DMAP_INSTRUMENTATION
static constexpr int PM_INSTRUMENT_MAX_VIEWS = 32;
static constexpr int PM_INSTRUMENT_NUM_NEIGHBORS = 8;
static constexpr unsigned PM_DMAP_INSTRUMENT_SCHEMA_VERSION = 5u;
// The legacy general/APD counter arrays retain their original 12 slots. APD
// schema v5 admits DVP in its unchanged per-pixel record layout and obtains
// the additional source count from the independent DVP counter namespace.
static constexpr int PM_INSTRUMENT_NUM_SOURCES = 12;
static constexpr unsigned PM_APD_INSTRUMENT_SCHEMA_VERSION = 5u;
static constexpr unsigned PM_DVP_INSTRUMENT_SCHEMA_VERSION = 1u;
static constexpr unsigned PM_DVP_VISIBILITY_INSTRUMENT_SCHEMA_VERSION = 1u;
static constexpr unsigned PM_DVP_VISIBLE_NORMAL_INSTRUMENT_SCHEMA_VERSION = 1u;
static constexpr int PM_DVP_VISIBILITY_INSTRUMENT_REASONS = 12;
static constexpr int PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS = 6;
static constexpr int PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS = 4;
static constexpr int PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS = 4;
static constexpr int PM_DVP_INSTRUMENT_FAMILIES = 5;
static constexpr int PM_DVP_INSTRUMENT_UNAVAILABLE_REASONS = 12;
static constexpr int PM_DVP_INSTRUMENT_FINAL_SOURCES = 13;
static constexpr int PM_INSTRUMENT_NUM_BAD_REASONS = 6;
static constexpr int PM_INSTRUMENT_NUM_UPDATE_BINS = 8;
static constexpr int PM_INSTRUMENT_NUM_CANDIDATE_TYPES = 6;
static constexpr int PM_INSTRUMENT_MAP_VIEWS = 4;
static constexpr int PM_INSTRUMENT_EXACT_NUM_CANDIDATES = 24;
static constexpr uint8_t PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE = 255;
static constexpr size_t PM_INSTRUMENT_EXACT_STACK_BYTES = 4096;
static constexpr int PM_APD_INSTRUMENT_PROFILE_SAMPLES = 61;
static constexpr int PM_APD_INSTRUMENT_SECTORS = 32;
static constexpr int PM_APD_INSTRUMENT_ANCHORS = 8;
static constexpr int PM_APD_INSTRUMENT_PROFILE_REASONS = 9;
static constexpr int PM_APD_INSTRUMENT_ANCHOR_REASONS = 7;
static constexpr int PM_APD_INSTRUMENT_VIEW_SELECTION_MODES = 5;

enum PatchMatchInstrumentExactCandidateSlot : uint8_t {
	PM_EXACT_CANDIDATE_CURRENT = 0,
	PM_EXACT_CANDIDATE_PROPAGATION_0 = 1,
	PM_EXACT_CANDIDATE_PROPAGATION_7 = 8,
	PM_EXACT_CANDIDATE_REFINE_DEPTH = 9,
	PM_EXACT_CANDIDATE_REFINE_NORMAL = 10,
	PM_EXACT_CANDIDATE_REFINE_RANDOM_NORMAL = 11,
	PM_EXACT_CANDIDATE_REFINE_SURFACE_NORMAL = 12,
	PM_EXACT_CANDIDATE_APD_ANCHOR_0 = 13,
	PM_EXACT_CANDIDATE_APD_ANCHOR_7 = 20,
	PM_EXACT_CANDIDATE_APD_FITTED_PLANE = 21,
	PM_EXACT_CANDIDATE_DVP_EPIPOLAR_0 = 22,
	PM_EXACT_CANDIDATE_DVP_EPIPOLAR_1 = 23,
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
	PM_CANDIDATE_APD_FITTED_PLANE = 4,
	PM_CANDIDATE_APD_FINAL_REFINEMENT = 5,
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
	PM_SOURCE_APD_ANCHOR_PROPAGATE = 9,
	PM_SOURCE_APD_FITTED_PLANE = 10,
	PM_SOURCE_APD_FINAL_REFINEMENT = 11,
	PM_SOURCE_DVP_EPIPOLAR = 12,
};

enum PatchMatchInstrumentBadReason : uint8_t {
	PM_BAD_NONE = 0,
	PM_BAD_OUT_OF_BOUNDS = 1,
	PM_BAD_INVALID_PROJECTION = 2,
	PM_BAD_LOW_REF_VARIANCE = 3,
	PM_BAD_LOW_TARGET_VARIANCE = 4,
	PM_BAD_GEOMETRIC_MISMATCH = 5,
};

enum PatchMatchAPDInstrumentAnchorReason : uint8_t {
	PM_APD_ANCHOR_UNKNOWN = 0,
	PM_APD_ANCHOR_PIXEL_NOT_UNRELIABLE = 1,
	PM_APD_ANCHOR_INVALID_CENTER_DEPTH = 2,
	PM_APD_ANCHOR_INSUFFICIENT_SECTOR_CANDIDATES = 3,
	PM_APD_ANCHOR_NO_VALID_RANSAC_MODEL = 4,
	PM_APD_ANCHOR_INSUFFICIENT_MODEL_INLIERS = 5,
	PM_APD_ANCHOR_READY = 6,
};

// One exact APD mechanics record per pixel and complete logical iteration.
// The initialization state has no APD classification or anchors and is
// intentionally absent from this array.
struct PatchMatchAPDInstrumentState {
	float averageBaseline = 0.f;
	float currentDisparity = 0.f;
	float globalMinimumCost = -1.f;
	float separation = -1.f;
	float nearestReliableDistance = -1.f;
	float ransacThreshold = -1.f;
	float ransacCenterResidual = -1.f;
	float ransacMeanInlierResidual = -1.f;
	float fittedPlaneDepth = -1.f;
	uint32_t nearestReliable = ~uint32_t(0);
	uint32_t ransacSamplePacked = ~uint32_t(0);
	int16_t globalMinimumOffset = 0;
	uint8_t reliability = 0;
	uint8_t profileReason = 0;
	uint8_t eta = 0;
	uint8_t finiteCount = 0;
	uint8_t localMinimumCount = 0;
	uint8_t globalMinimumPlateauStart = 0;
	uint8_t globalMinimumPlateauEnd = 0;
	uint8_t candidateCount = 0;
	uint8_t inlierCount = 0;
	uint8_t outlierCount = 0;
	uint8_t anchorCount = 0;
	uint8_t anchorReason = PM_APD_ANCHOR_UNKNOWN;
	uint8_t ransacValid = 0;
	uint8_t deformableEligible = 0;
	uint8_t fittedPlaneValid = 0;
};
static_assert(sizeof(PatchMatchAPDInstrumentState) == 64, "APD state record schema changed");

// Exact candidate-ranking and final-score decomposition for one APD pixel
// update. Working costs rank candidates; nativePersistentCost is the value
// actually retained by production after conventional winner rescoring.
struct PatchMatchAPDInstrumentUpdate {
	float workingWinnerCost = -1.f;
	float nativePersistentCost = -1.f;
	float runnerUpWorkingCost = -1.f;
	float winnerRunnerUpGap = -1.f;
	float centerCost = -1.f;
	float anchorMeanCost = -1.f;
	float deformablePhotometricCost = -1.f;
	float geometricCost = -1.f;
	float nativeMinusWorkingCost = -1.f;
	float nativeStoredCostBefore = -1.f;
	float incumbentWorkingCost = -1.f;
	float bestAnchorWorkingCost = -1.f;
	float acceptedAnchorNativeCost = -1.f;
	float fittedPlaneWorkingCost = -1.f;
	float fittedPlaneNativeCost = -1.f;
	float finalRefinementIncumbentCost = -1.f;
	float finalRefinementBestCost = -1.f;
	float finalRefinementImprovement = -1.f;
	float finalRefinementDepth = -1.f;
	uint32_t candidateTestedMask = 0;
	uint32_t candidateFiniteMask = 0;
	uint32_t candidateAcceptedMask = 0;
	uint32_t acceptedAnchorIndex = ~uint32_t(0);
	uint32_t workingSelectedViews = 0;
	uint8_t source = PM_SOURCE_NONE;
	uint8_t winnerSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t runnerUpSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t testedCount = 0;
	uint8_t finiteCount = 0;
	uint8_t acceptedCount = 0;
	uint8_t selectedViewCount = 0;
	uint8_t deformableActive = 0;
	uint8_t viewSelectionMode = 0;
	uint8_t anchorEvidenceCount = 0;
	uint8_t anchorProposalCount = 0;
	uint8_t anchorFiniteCount = 0;
	uint8_t anchorAcceptedSlot = PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE;
	uint8_t immutableAnchorState = 0;
	uint8_t selectedViewWeightSum = 0;
	int8_t finalRefinementOffset = 0;
	uint8_t updateStage = 0;
	uint8_t fittedPlaneAvailable = 0;
	uint8_t fittedPlaneTested = 0;
	uint8_t fittedPlaneAccepted = 0;
	uint8_t finalRefinementTested = 0;
	uint8_t finalRefinementFinite = 0;
	uint8_t finalRefinementAccepted = 0;
};
static_assert(sizeof(PatchMatchAPDInstrumentUpdate) == 120, "APD update record schema changed");

struct PatchMatchAPDInstrumentCounters {
	uint32_t classified = 0;
	uint32_t reliability[3] = {};
	uint32_t profileReason[PM_APD_INSTRUMENT_PROFILE_REASONS] = {};
	uint32_t anchorReason[PM_APD_INSTRUMENT_ANCHOR_REASONS] = {};
	uint32_t anchorCountBins[PM_APD_INSTRUMENT_ANCHORS + 1] = {};
	uint32_t ransacValid = 0;
	uint32_t deformableEligible = 0;
	uint32_t deformableUpdates = 0;
	uint32_t globalMinimumCostSamples = 0;
	uint32_t separationSamples = 0;
	uint32_t workingGapSamples = 0;
	uint32_t updateSource[PM_INSTRUMENT_NUM_SOURCES] = {};
	uint32_t anchorViewSelectionMode[PM_APD_INSTRUMENT_VIEW_SELECTION_MODES] = {};
	uint32_t anchorViewSelectionAttempted = 0;
	uint32_t anchorViewSelectionUsed = 0;
	uint32_t anchorProposalsTested = 0;
	uint32_t anchorProposalsFinite = 0;
	uint32_t anchorProposalsAccepted = 0;
	uint32_t anchorPropagationFinalWinners = 0;
	uint32_t immutableAnchorStateUpdates = 0;
	uint32_t bestAnchorWorkingCostSamples = 0;
	uint32_t acceptedAnchorNativeCostSamples = 0;
	uint32_t stageUpdates[3] = {};
	uint32_t fittedPlaneAvailable = 0;
	uint32_t fittedPlaneTested = 0;
	uint32_t fittedPlaneFinite = 0;
	uint32_t fittedPlaneAccepted = 0;
	uint32_t fittedPlaneFinalWinners = 0;
	uint32_t finalRefinementPixels = 0;
	uint32_t finalRefinementCandidatesTested = 0;
	uint32_t finalRefinementCandidatesFinite = 0;
	uint32_t finalRefinementAccepted = 0;
	double globalMinimumCostSum = 0.;
	double separationSum = 0.;
	double anchorCountSum = 0.;
	double centerCostSum = 0.;
	double anchorMeanCostSum = 0.;
	double workingCostSum = 0.;
	double nativePersistentCostSum = 0.;
	double workingGapSum = 0.;
	double bestAnchorWorkingCostSum = 0.;
	double acceptedAnchorNativeCostSum = 0.;
};

// Targeted deep trace. Full profiles and per-sector/per-view mechanics are
// bounded by the existing trace-pixel selection and are never allocated for
// summary-only captures.
struct PatchMatchAPDInstrumentTrace {
	int32_t valid = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t logicalIteration = -1;
	int32_t x = -1;
	int32_t y = -1;
	PatchMatchAPDInstrumentState state;
	PatchMatchAPDInstrumentUpdate update;
	float profile[PM_APD_INSTRUMENT_PROFILE_SAMPLES] = {};
	uint8_t viewWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint32_t sectorCandidates[PM_APD_INSTRUMENT_SECTORS] = {};
	uint32_t anchors[PM_APD_INSTRUMENT_ANCHORS] = {};
	float anchorResiduals[PM_APD_INSTRUMENT_ANCHORS] = {};
	float viewCenterCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewAnchorMeanCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewWorkingCosts[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewSelectionPriors[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewSamplingScores[PM_INSTRUMENT_MAX_VIEWS] = {};
	float viewSamplingProbabilities[PM_INSTRUMENT_MAX_VIEWS] = {};
	float anchorCandidateWorkingCosts[PM_APD_INSTRUMENT_ANCHORS] = {};
	float anchorCandidateNativeCosts[PM_APD_INSTRUMENT_ANCHORS] = {};
	uint32_t anchorSelectedViews[PM_APD_INSTRUMENT_ANCHORS] = {};
	uint8_t anchorCandidateValid[PM_APD_INSTRUMENT_ANCHORS] = {};
};

// Exact DVP proposal mechanics for one pixel and complete logical iteration.
// Existing v4 records remain untouched; schema v5 adds these records under a
// dedicated `dvp` namespace.
struct PatchMatchDVPInstrumentUpdate {
	float incumbentDepth = -1.f;
	float finalDepth = -1.f;
	float leftIntervalMinimum = -1.f;
	float leftIntervalMaximum = -1.f;
	float rightIntervalMinimum = -1.f;
	float rightIntervalMaximum = -1.f;
	float proposalDepth[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float candidateCost[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float incumbentCost = -1.f;
	float winnerCost = -1.f;
	float runnerUpCost = -1.f;
	float winnerRunnerUpGap = -1.f;
	float depthDisplacement = -1.f;
	float meanReprojectionError[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float maxReprojectionError[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float meanRelativeDepthError[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	float maxRelativeDepthError[DVP_MAX_PROPOSALS] = {-1.f, -1.f};
	uint32_t selectedSourceViews = 0u;
	uint32_t directionSourceViews = 0u;
	uint32_t supportViews[DVP_MAX_PROPOSALS] = {};
	uint32_t occludedViews[DVP_MAX_PROPOSALS] = {};
	int16_t signedOffset[DVP_MAX_PROPOSALS] = {};
	uint8_t leftOuterCount = 0u;
	uint8_t leftInnerCount = 0u;
	uint8_t rightInnerCount = 0u;
	uint8_t rightOuterCount = 0u;
	uint8_t sourceView[DVP_MAX_PROPOSALS] = {0xffu, 0xffu};
	uint8_t support[DVP_MAX_PROPOSALS] = {};
	uint8_t family = static_cast<uint8_t>(DVPEpipolarFamily::DISABLED);
	uint8_t unavailableReason = static_cast<uint8_t>(
		DVPEpipolarUnavailableReason::FAMILY_DISABLED);
	uint8_t generatedCount = 0u;
	uint8_t testedCount = 0u;
	uint8_t finiteCount = 0u;
	uint8_t acceptedCount = 0u;
	uint8_t testedProposalMask = 0u;
	uint8_t finiteProposalMask = 0u;
	uint8_t acceptedProposalMask = 0u;
	uint8_t winnerOrdinal = 0xffu;
	uint8_t accepted = 0u;
	uint8_t finalWinner = 0u;
	uint8_t finalUpdateSource = PM_SOURCE_NONE;
	uint8_t finalDepthRetained = 0u;
	uint8_t nativeDepthFallback = 0u;
	uint8_t leftIntervalValid = 0u;
	uint8_t rightIntervalValid = 0u;
};
static_assert(sizeof(PatchMatchDVPInstrumentUpdate) == 148,
	"DVP update record schema changed");

struct PatchMatchDVPInstrumentCounters {
	uint32_t attemptedPixels = 0u;
	uint32_t proposalAvailablePixels = 0u;
	uint32_t nativeDepthFallbackPixels = 0u;
	uint32_t proposalsGenerated = 0u;
	uint32_t proposalsTested = 0u;
	uint32_t proposalsFinite = 0u;
	uint32_t proposalsAccepted = 0u;
	uint32_t finalWinnerPixels = 0u;
	uint32_t finalDepthRetainedPixels = 0u;
	uint32_t finalUpdateSource[PM_DVP_INSTRUMENT_FINAL_SOURCES] = {};
	uint32_t leftIntervalValidPixels = 0u;
	uint32_t rightIntervalValidPixels = 0u;
	uint32_t selectedSourceViewCountSum = 0u;
	uint32_t directionSourceViewCountSum = 0u;
	uint32_t endpointSupportSum[4] = {};
	uint32_t unavailableReason[PM_DVP_INSTRUMENT_UNAVAILABLE_REASONS] = {};
	uint32_t family[PM_DVP_INSTRUMENT_FAMILIES] = {};
	uint32_t proposalSupportSum = 0u;
	uint32_t proposalSupportSamples = 0u;
	uint32_t occludedCandidates = 0u;
	uint32_t costSamples = 0u;
	uint32_t gapSamples = 0u;
	uint32_t displacementSamples = 0u;
	uint32_t reprojectionErrorSamples = 0u;
	uint32_t relativeDepthErrorSamples = 0u;
	double incumbentCostSum = 0.;
	double winnerCostSum = 0.;
	double improvementSum = 0.;
	double gapSum = 0.;
	double displacementSum = 0.;
	double reprojectionErrorSum = 0.;
	double relativeDepthErrorSum = 0.;
};
static_assert(sizeof(PatchMatchDVPInstrumentCounters) == 280,
	"DVP counter record schema changed");

// Per-view endpoint samples are stored only for configured trace pixels. The
// dense map tier retains aggregate endpoints, proposals, costs, and decisions.
struct PatchMatchDVPInstrumentTrace {
	int32_t valid = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t logicalIteration = -1;
	int32_t x = -1;
	int32_t y = -1;
	uint32_t leftOuterViews = 0u;
	uint32_t leftInnerViews = 0u;
	uint32_t rightInnerViews = 0u;
	uint32_t rightOuterViews = 0u;
	PatchMatchDVPInstrumentUpdate update;
	float leftOuter[DVP_MAX_SOURCE_VIEWS] = {};
	float leftInner[DVP_MAX_SOURCE_VIEWS] = {};
	float rightInner[DVP_MAX_SOURCE_VIEWS] = {};
	float rightOuter[DVP_MAX_SOURCE_VIEWS] = {};
};
static_assert(sizeof(PatchMatchDVPInstrumentTrace) == 700,
	"DVP trace record schema changed");

// Exact persistent-visibility mechanics for one pixel and complete logical
// iteration. Dense records retain lossless masks and per-pixel reason counts;
// selected-pixel traces retain every per-view weight and reason.
struct PatchMatchDVPVisibilityInstrumentUpdate {
	uint32_t previousMask = 0u;
	uint32_t resolvedMask = 0u;
	uint32_t nextMask = 0u;
	uint32_t activeSupportMask = 0u;
	uint32_t restoredMask = 0u;
	uint32_t rejectedMask = 0u;
	uint32_t addedMask = 0u;
	uint32_t removedMask = 0u;
	uint32_t candidateTestedMask = 0u;
	uint32_t candidateFiniteMask = 0u;
	uint16_t previousWeightSum = 0u;
	uint16_t resolvedWeightSum = 0u;
	uint16_t nextWeightSum = 0u;
	uint16_t activeSupportWeightSum = 0u;
	uint16_t denominator = 0u;
	uint8_t previousVisibleCount = 0u;
	uint8_t resolvedVisibleCount = 0u;
	uint8_t nextVisibleCount = 0u;
	uint8_t activeSupportCount = 0u;
	uint8_t restoredCount = 0u;
	uint8_t rejectedCount = 0u;
	uint8_t addedCount = 0u;
	uint8_t removedCount = 0u;
	uint8_t mode = static_cast<uint8_t>(DVPVisibilityMode::DISABLED);
	uint8_t denominatorDefined = 0u;
	uint8_t activeSupportMatchesResolved = 0u;
	uint8_t transitionStatus = static_cast<uint8_t>(
		DVPVisibilityTransitionStatus::INVALID_VERSION);
	uint8_t reasonCount[PM_DVP_VISIBILITY_INSTRUMENT_REASONS] = {};
};
static_assert(sizeof(PatchMatchDVPVisibilityInstrumentUpdate) == 76,
	"DVP visibility update record schema changed");

struct PatchMatchDVPVisibilityInstrumentCounters {
	uint32_t pixels = 0u;
	uint32_t reason[PM_DVP_VISIBILITY_INSTRUMENT_REASONS] = {};
	uint32_t previousVisibleViews = 0u;
	uint32_t resolvedVisibleViews = 0u;
	uint32_t nextVisibleViews = 0u;
	uint32_t activeSupportViews = 0u;
	uint32_t restoredViews = 0u;
	uint32_t rejectedViews = 0u;
	uint32_t addedViews = 0u;
	uint32_t removedViews = 0u;
	uint64_t previousWeightSum = 0u;
	uint64_t resolvedWeightSum = 0u;
	uint64_t nextWeightSum = 0u;
	uint64_t activeSupportWeightSum = 0u;
	uint64_t denominatorSum = 0u;
	uint32_t zeroDenominatorPixels = 0u;
	uint32_t supportMismatchPixels = 0u;
	uint32_t changedPixels = 0u;
	uint32_t invalidTransitionPixels = 0u;
	uint32_t candidateTestedPixels = 0u;
	uint32_t candidateFinitePixels = 0u;
	uint32_t candidateTestedCount = 0u;
	uint32_t candidateFiniteCount = 0u;
};
static_assert(sizeof(PatchMatchDVPVisibilityInstrumentCounters) == 160,
	"DVP visibility counter record schema changed");

struct PatchMatchDVPVisibilityInstrumentTrace {
	int32_t valid = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t logicalIteration = -1;
	int32_t x = -1;
	int32_t y = -1;
	PatchMatchDVPVisibilityInstrumentUpdate update;
	uint8_t previousWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint8_t resolvedWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint8_t nextWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint8_t activeSupportWeights[PM_INSTRUMENT_MAX_VIEWS] = {};
	uint8_t reasons[PM_INSTRUMENT_MAX_VIEWS] = {};
};
static_assert(sizeof(PatchMatchDVPVisibilityInstrumentTrace) == 260,
	"DVP visibility trace record schema changed");

// Exact visible-normal decisions for one pixel and complete logical iteration.
// Dense records retain lossless candidate masks and stage decisions; targeted
// traces retain the actual normal vectors and per-propagation-candidate tests.
struct PatchMatchDVPVisibleNormalInstrumentUpdate {
	float currentMaxDot = -FLT_MAX;
	float currentMaxViolation = -FLT_MAX;
	float propagationNativeBestCost = -1.f;
	float propagationConstrainedBestCost = -1.f;
	float propagationSelectedCost = -1.f;
	float nativeMaxDot[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {-FLT_MAX, -FLT_MAX};
	float selectedMaxDot[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {-FLT_MAX, -FLT_MAX};
	uint32_t selectedSourceViews = 0u;
	uint16_t retriesTested[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {};
	uint16_t selectedRetry[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {0xffffu, 0xffffu};
	uint8_t supportCount = 0u;
	uint8_t directionCount = 0u;
	uint8_t mode = static_cast<uint8_t>(DVPVisibleNormalMode::DISABLED);
	uint8_t currentValid = 0u;
	uint8_t currentFeasible = 0u;
	uint8_t currentReason = static_cast<uint8_t>(DVPVisibleNormalEvaluationReason::INVALID_NORMAL);
	uint8_t currentRejectedDirection = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t propagationTestedMask = 0u;
	uint8_t propagationValidMask = 0u;
	uint8_t propagationFeasibleMask = 0u;
	uint8_t propagationRejectedMask = 0u;
	uint8_t propagationNativeBest = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t propagationConstrainedBest = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t propagationSelected = DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE;
	uint8_t propagationReason = static_cast<uint8_t>(DVPVisibleNormalPropagationReason::INVALID_ARGUMENT);
	uint8_t propagationFallback = 0u;
	uint8_t propagationAppliedConstraint = 0u;
	uint8_t propagationAccepted = 0u;
	uint8_t refinementNativeTestedMask = 0u;
	uint8_t refinementNativeValidMask = 0u;
	uint8_t refinementNativeFeasibleMask = 0u;
	uint8_t refinementRetrySuccessMask = 0u;
	uint8_t refinementExhaustionFallbackMask = 0u;
	uint8_t refinementAppliedRetryMask = 0u;
	uint8_t refinementAcceptedMask = 0u;
	uint8_t nativeReason[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {};
	uint8_t proposalReason[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {};
	uint8_t nativeRejectedDirection[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
		DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE, DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE};
	uint8_t selectedRejectedDirection[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
		DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE, DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE};
	uint8_t reserved[3] = {};
};
static_assert(sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate) == 84,
	"DVP visible-normal update record schema changed");

struct PatchMatchDVPVisibleNormalInstrumentCounters {
	uint64_t refinementRetriesTested = 0u;
	uint32_t pixels = 0u;
	uint32_t selectedSupportViews = 0u;
	uint32_t currentValid = 0u;
	uint32_t currentFeasible = 0u;
	uint32_t currentRejected = 0u;
	uint32_t currentInvalid = 0u;
	uint32_t propagationCandidatesTested = 0u;
	uint32_t propagationCandidatesValid = 0u;
	uint32_t propagationCandidatesFeasible = 0u;
	uint32_t propagationCandidatesRejected = 0u;
	uint32_t propagationConstrainedSelections = 0u;
	uint32_t propagationNativeSelections = 0u;
	uint32_t propagationFallbacks = 0u;
	uint32_t propagationAppliedConstraints = 0u;
	uint32_t propagationAccepted = 0u;
	uint32_t refinementNativeTested = 0u;
	uint32_t refinementNativeValid = 0u;
	uint32_t refinementNativeFeasible = 0u;
	uint32_t refinementNativeRejected = 0u;
	uint32_t refinementRetrySuccess = 0u;
	uint32_t refinementExhaustion = 0u;
	uint32_t refinementFallback = 0u;
	uint32_t refinementAppliedRetry = 0u;
	uint32_t refinementAccepted = 0u;
	uint32_t evaluationReason[PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS] = {};
	uint32_t proposalReason[PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS] = {};
	uint32_t propagationReason[PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS] = {};
};
static_assert(sizeof(PatchMatchDVPVisibleNormalInstrumentCounters) == 160,
	"DVP visible-normal counter record schema changed");

struct PatchMatchDVPVisibleNormalInstrumentTrace {
	int32_t valid = 0;
	int32_t imageID = -1;
	int32_t scaleNumber = -1;
	int32_t logicalIteration = -1;
	int32_t x = -1;
	int32_t y = -1;
	PatchMatchDVPVisibleNormalInstrumentUpdate update;
	float currentNormal[3] = {};
	float propagationNormals[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES][3] = {};
	float propagationCosts[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
	float propagationMaxDot[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
	uint8_t propagationReason[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
	uint8_t propagationRejectedDirection[DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES] = {};
	float nativeNormals[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS][3] = {};
	float selectedNormals[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS][3] = {};
};
static_assert(sizeof(PatchMatchDVPVisibleNormalInstrumentTrace) == 344,
	"DVP visible-normal trace record schema changed");

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
	PatchMatchAPDInstrumentCounters* apdCounters = nullptr;
	PatchMatchAPDInstrumentUpdate* apdUpdates = nullptr;
	PatchMatchAPDInstrumentTrace* apdTraces = nullptr;
	PatchMatchDVPInstrumentCounters* dvpCounters = nullptr;
	PatchMatchDVPInstrumentUpdate* dvpUpdates = nullptr;
	PatchMatchDVPInstrumentTrace* dvpTraces = nullptr;
	PatchMatchDVPVisibilityInstrumentCounters* visibilityCounters = nullptr;
	PatchMatchDVPVisibilityInstrumentUpdate* visibilityUpdates = nullptr;
	PatchMatchDVPVisibilityInstrumentTrace* visibilityTraces = nullptr;
	uint8_t* visibilityReasons = nullptr;
	PatchMatchDVPVisibleNormalInstrumentCounters* visibleNormalCounters = nullptr;
	PatchMatchDVPVisibleNormalInstrumentUpdate* visibleNormalUpdates = nullptr;
	PatchMatchDVPVisibleNormalInstrumentTrace* visibleNormalTraces = nullptr;
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
	PatchMatchAPDInstrumentCounters* apdCounters = nullptr;
	PatchMatchAPDInstrumentState* apdStates = nullptr;
	PatchMatchAPDInstrumentUpdate* apdUpdates = nullptr;
	PatchMatchAPDInstrumentTrace* apdTraces = nullptr;
	PatchMatchDVPInstrumentCounters* dvpCounters = nullptr;
	PatchMatchDVPInstrumentUpdate* dvpUpdates = nullptr;
	PatchMatchDVPInstrumentTrace* dvpTraces = nullptr;
	PatchMatchDVPVisibilityInstrumentCounters* visibilityCounters = nullptr;
	PatchMatchDVPVisibilityInstrumentUpdate* visibilityUpdates = nullptr;
	PatchMatchDVPVisibilityInstrumentTrace* visibilityTraces = nullptr;
	uint8_t* visibilityReasons = nullptr;
	PatchMatchDVPVisibleNormalInstrumentCounters* visibleNormalCounters = nullptr;
	PatchMatchDVPVisibleNormalInstrumentUpdate* visibleNormalUpdates = nullptr;
	PatchMatchDVPVisibleNormalInstrumentTrace* visibleNormalTraces = nullptr;
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
	int32_t numAPDIterations = 0;
	int32_t viewStride = 0;
	bool sampled = false;
	bool exact = false;
};
#endif

struct ConfAdjustRequest; // ConfidenceCUDA.h (fused confidence recalibration)

// Host pointers are valid for one pyramid-level RunCUDA call. Input contains
// the version-checked nearest-neighbor transfer; outputs describe the exact
// post-filter state that can seed the next level/stage.
struct PatchMatchAPDMultiscaleIO {
	const uint8_t* transferredReliability = nullptr;
	// Selected cumulative C3 topology map at the current pyramid resolution.
	// Null when the depth-edge prior is disabled; in that mode RunCUDA performs
	// no prior allocation, upload, or filtering launch.
	const uint16_t* depthEdgeRegions = nullptr;
	uint8_t* outputReliability = nullptr;
	uint8_t* outputAnchorCounts = nullptr;
	uint8_t* outputDeformableEligible = nullptr;
	#ifdef _USE_DMAP_INSTRUMENTATION
	uint8_t* outputDepthEdgeAnchorCountsBefore = nullptr;
	uint8_t* outputDepthEdgeAnchorCountsAfter = nullptr;
	uint8_t* outputDepthEdgeRejectedAnchorCounts = nullptr;
	#endif
};

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
		bool bCompat23 = false;
		float fThresholdKeepCost = 0;
		unsigned nAPDMode = 0;
		unsigned nDVPEpipolarFamily = 0;
		float fDVPEpipolarAlpha = DVP_EPIPOLAR_ALPHA;
		float fDVPEpipolarBeta = DVP_EPIPOLAR_BETA;
		unsigned nDVPEpipolarMu = DVP_EPIPOLAR_MU;
		unsigned nDVPGlobalSearchRadius = DVP_GLOBAL_SEARCH_RADIUS;
		float fDVPReprojectionThreshold = DVP_GLOBAL_REPROJECTION_THRESHOLD;
		float fDVPRelativeDepthThreshold = DVP_GLOBAL_RELATIVE_DEPTH_THRESHOLD;
		unsigned nDVPDepthEdgeMode = static_cast<unsigned>(DVPDepthEdgeMode::DISABLED);
		unsigned nDVPVisibilityMode = static_cast<unsigned>(DVPVisibilityMode::DISABLED);
		float fDVPVisibilityReprojectionThreshold = DVP_VISIBILITY_REPROJECTION_THRESHOLD;
		float fDVPVisibilityRelativeDepthThreshold = DVP_VISIBILITY_RELATIVE_DEPTH_THRESHOLD;
		unsigned nDVPVisibleNormalMode = static_cast<unsigned>(DVPVisibleNormalMode::DISABLED);
		float fDVPVisibleNormalDotTolerance = DVP_VISIBLE_NORMAL_DEFAULT_DOT_TOLERANCE;
		unsigned nDVPVisibleNormalAttempts = DVP_VISIBLE_NORMAL_DEFAULT_ATTEMPTS;
		unsigned nAPDLevelIndex = 0;
		unsigned nAPDLevelCount = 1;
		unsigned nAPDStageIndex = 0;
		unsigned nAPDTransferStatus = 0;
		bool bAPDTransferredState = false;
	};

public:
	PatchMatch();
	~PatchMatch();

	void Init(bool bGeomConsistency);
	void Release();

	// pConfRequest (optional): on the last geometric-consistency iteration, run the fused GPU
	// confidence recalibration right after the estimation kernels, reusing the device-resident
	// reference buffers (see ConfidenceCUDA.h ConfAdjustRequest); its done/computeNS report back
	void EstimateDepthMap(DepthData&, int geometricIteration=-1, ConfAdjustRequest* pConfRequest=NULL);

	float4 GetPlaneHypothesis(const int index);
	float GetCost(const int index);
	// Focused device-side contract fixture used by CUDA-enabled tests. Returns
	// false when no CUDA device is available or a runtime operation fails.
	static bool RunDVPContractOracle(DVPCUDAOracleResult& result);
	static bool RunDVPDepthEdgeContractOracle(DVPDepthEdgeCUDAOracleResult& result);
	static bool RunDVPVisibilityContractOracle(DVPVisibilityCUDAOracleResult& result);
	static bool RunDVPVisibleNormalContractOracle(DVPVisibleNormalCUDAOracleResult& result);

private:
	void ReleaseCUDA();
	void AllocatePatchMatchCUDA(const cv::Mat1f& image);
	void AllocateImageCUDA(size_t i, const cv::Mat1f& image, bool bInitImage, bool bInitDepthMap);
#ifdef _USE_DMAP_INSTRUMENTATION
	void RunCUDA(float* ptrCostMap=NULL, uint32_t* ptrViewsMap=NULL, uint8_t* ptrUpdateSources=NULL,
		PatchMatchInstrumentDeviceContext* instrument=NULL,
		const PatchMatchAPDMultiscaleIO* apdMultiscaleIO=NULL);
#else
	void RunCUDA(float* ptrCostMap=NULL, uint32_t* ptrViewsMap=NULL,
		const PatchMatchAPDMultiscaleIO* apdMultiscaleIO=NULL);
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
