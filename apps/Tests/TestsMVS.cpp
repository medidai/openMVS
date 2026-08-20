/*
 * TestsMVS.cpp
 *
 * Copyright (c) 2014-2025 SEACAVE
 *
 * Author(s):
 *
 *      cDc <cdc.seacave@gmail.com>
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
 *      You are required to preserve legal notices and author attributions in
 *      that material or in the Appropriate Legal Notices displayed by works
 *      containing it.
 */

#include "../../libs/MVS.h"
#include "../../libs/MVS/PatchMatchAPDCUDA.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("TestMVS "));

namespace MVS {

bool AdaptivePatchDeformationContractTest()
{
	using namespace CUDA;
	const auto nearEqual = [](float first, float second, float epsilon = 1e-6f) {
		return std::abs(first-second) <= epsilon;
	};

	APDConfig config;
	if (ValidateAPDConfig(config) != APDConfigStatus::VALID ||
		config.mode != static_cast<unsigned>(APDMode::DISABLED) ||
		APDRansacNormalizedThreshold(0u, 4u) != APD_RANSAC_NORMALIZED_THRESHOLD_MAX ||
		APDRansacNormalizedThreshold(3u, 4u) != APD_RANSAC_NORMALIZED_THRESHOLD_MIN)
	{
		VERBOSE("ERROR: APD default-off contract failed!");
		return false;
	}

	APDStageClock coarseClock;
	coarseClock.levelIndex = 0u;
	coarseClock.levelCount = 3u;
	coarseClock.stageIndex = 0u;
	const APDStageSchedule coarseSchedule(ResolveAPDStageSchedule(coarseClock));
	APDStageClock middleClock;
	middleClock.levelIndex = 1u;
	middleClock.levelCount = 3u;
	middleClock.stageIndex = 1u;
	middleClock.hasTransferredState = true;
	const APDStageSchedule middleSchedule(ResolveAPDStageSchedule(middleClock));
	APDStageClock fineClock;
	fineClock.levelIndex = 2u;
	fineClock.levelCount = 3u;
	fineClock.stageIndex = 2u;
	fineClock.hasTransferredState = true;
	const APDStageSchedule fineSchedule(ResolveAPDStageSchedule(fineClock));
	APDStageClock geometricClock(fineClock);
	geometricClock.stageIndex = 3u;
	geometricClock.geometricConsistency = true;
	const APDStageSchedule geometricSchedule(ResolveAPDStageSchedule(geometricClock));
	APDStageClock invalidClock(fineClock);
	invalidClock.levelIndex = invalidClock.levelCount;
	if (ValidateAPDStageClock(coarseClock) != APDStageClockStatus::VALID ||
		!coarseSchedule.valid || !coarseSchedule.conventional ||
		coarseSchedule.consumesTransferredState || coarseSchedule.reliabilityEta != 6u ||
		!nearEqual(coarseSchedule.ransacNormalizedThreshold, 0.010f) ||
		!middleSchedule.valid || middleSchedule.conventional ||
		!middleSchedule.consumesTransferredState || middleSchedule.reliabilityEta != 4u ||
		!nearEqual(middleSchedule.ransacNormalizedThreshold, 0.00875f) ||
		!fineSchedule.valid || fineSchedule.conventional || fineSchedule.reliabilityEta != 2u ||
		!nearEqual(fineSchedule.ransacNormalizedThreshold, 0.0075f) ||
		!geometricSchedule.valid || geometricSchedule.conventional ||
		geometricSchedule.reliabilityEta != 2u ||
		!nearEqual(geometricSchedule.ransacNormalizedThreshold, 0.0075f) ||
		ValidateAPDStageClock(invalidClock) != APDStageClockStatus::INVALID_LEVEL_INDEX ||
		APDStageSeed(coarseClock) == APDStageSeed(middleClock) ||
		APDStageSeed(fineClock) == APDStageSeed(geometricClock))
	{
		VERBOSE("ERROR: APD explicit stage/level schedule contract failed!");
		return false;
	}

	APDMultiscaleStateHeader coarseState;
	coarseState.width = 2u;
	coarseState.height = 2u;
	coarseState.sourceLevelIndex = 0u;
	coarseState.sourceStageIndex = 0u;
	if (ValidateAPDMultiscaleTransfer(coarseState, middleClock, 4u, 4u) !=
			APDMultiscaleTransferStatus::VALID ||
		APDNearestSourceCoordinate(0u, 2u, 4u) != 0u ||
		APDNearestSourceCoordinate(1u, 2u, 4u) != 0u ||
		APDNearestSourceCoordinate(2u, 2u, 4u) != 1u ||
		APDNearestSourceCoordinate(3u, 2u, 4u) != 1u)
	{
		VERBOSE("ERROR: APD two-scale nearest-state transfer contract failed!");
		return false;
	}
	const uint8_t coarseReliability[4] = {1u, 2u, 2u, 1u};
	const uint8_t coarseAnchorCount[4] = {0u, 6u, 8u, 0u};
	const uint8_t coarseDeformable[4] = {0u, 1u, 1u, 0u};
	for (unsigned y=0u; y<4u; ++y) {
		for (unsigned x=0u; x<4u; ++x) {
			const unsigned sourceX(APDNearestSourceCoordinate(x, 2u, 4u));
			const unsigned sourceY(APDNearestSourceCoordinate(y, 2u, 4u));
			const unsigned sourceIndex(sourceY*2u+sourceX);
			const bool expectedRightHalf(x >= 2u);
			const bool expectedBottomHalf(y >= 2u);
			const unsigned expectedIndex((expectedBottomHalf ? 2u : 0u)+
				(expectedRightHalf ? 1u : 0u));
			if (sourceIndex != expectedIndex ||
				coarseReliability[sourceIndex] != coarseReliability[expectedIndex] ||
				coarseAnchorCount[sourceIndex] != coarseAnchorCount[expectedIndex] ||
				coarseDeformable[sourceIndex] != coarseDeformable[expectedIndex])
			{
				VERBOSE("ERROR: APD reliability/anchor/deformation provenance mapping failed!");
				return false;
			}
		}
	}
	APDMultiscaleStateHeader invalidState(coarseState);
	invalidState.version = APD_MULTISCALE_STATE_VERSION+1u;
	APDStageClock skippedStageClock(middleClock);
	skippedStageClock.stageIndex = 2u;
	APDMultiscaleStateHeader middleState;
	middleState.width = 4u;
	middleState.height = 4u;
	middleState.sourceLevelIndex = 1u;
	middleState.sourceStageIndex = 1u;
	APDMultiscaleStateHeader fineState;
	fineState.width = 8u;
	fineState.height = 8u;
	fineState.sourceLevelIndex = 2u;
	fineState.sourceStageIndex = 2u;
	if (ValidateAPDMultiscaleTransfer(invalidState, middleClock, 4u, 4u) !=
			APDMultiscaleTransferStatus::INVALID_VERSION ||
		ValidateAPDMultiscaleTransfer(coarseState, coarseClock, 2u, 2u) !=
			APDMultiscaleTransferStatus::INVALID_STAGE ||
		ValidateAPDMultiscaleTransfer(coarseState, fineClock, 8u, 8u) !=
			APDMultiscaleTransferStatus::INVALID_LEVEL ||
		ValidateAPDMultiscaleTransfer(coarseState, skippedStageClock, 4u, 4u) !=
			APDMultiscaleTransferStatus::INVALID_STAGE ||
		ValidateAPDMultiscaleTransfer(middleState, geometricClock, 8u, 8u) !=
			APDMultiscaleTransferStatus::INVALID_STAGE ||
		ValidateAPDMultiscaleTransfer(fineState, geometricClock, 8u, 8u) !=
			APDMultiscaleTransferStatus::VALID ||
		ValidateAPDMultiscaleTransfer(coarseState, middleClock, 1u, 4u) !=
			APDMultiscaleTransferStatus::INVALID_DESTINATION_SIZE)
	{
		VERBOSE("ERROR: APD invalid multiscale state was accepted!");
		return false;
	}
	config.mode = static_cast<unsigned>(APDMode::DEFORMABLE_COST);
	if (ValidateAPDConfig(config) != APDConfigStatus::VALID) {
		VERBOSE("ERROR: APD deformable-cost config failed validation!");
		return false;
	}
	APDConfig invalid(config);
	invalid.mode = 2u;
	if (ValidateAPDConfig(invalid) != APDConfigStatus::INVALID_MODE) {
		VERBOSE("ERROR: APD accepted an unimplemented mode!");
		return false;
	}
	for (const APDReliabilityClass reliability : {
		APDReliabilityClass::UNKNOWN,
		APDReliabilityClass::UNRELIABLE,
		APDReliabilityClass::RELIABLE})
	{
		const bool early(APDShouldProcess(reliability, APDUpdateStage::RELIABLE));
		const bool late(APDShouldProcess(reliability, APDUpdateStage::NON_RELIABLE));
		if (early == late || !APDShouldProcess(reliability, APDUpdateStage::ALL) ||
			early != (reliability == APDReliabilityClass::RELIABLE))
		{
			VERBOSE("ERROR: APD reliable-first dispatch does not cover each class exactly once!");
			return false;
		}
	}

	float profile[APD_PROFILE_SIZE];
	const auto resetProfile = [&profile](float value = 1.f) {
		for (unsigned i=0; i<APD_PROFILE_SIZE; ++i)
			profile[i] = value;
	};
	resetProfile();
	profile[APD_PROFILE_RADIUS] = 0.1f;
	APDProfileSummary summary(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	if (summary.reliability != APDReliabilityClass::RELIABLE ||
		summary.reason != APDProfileReason::RELIABLE_SINGLE_MINIMUM ||
		summary.finiteCount != APD_PROFILE_SIZE || summary.localMinimumCount != 1u ||
		summary.globalMinimumIndex != APD_PROFILE_RADIUS || summary.globalMinimumOffset != 0 ||
		!nearEqual(summary.globalMinimumCost, 0.1f))
	{
		VERBOSE("ERROR: APD single-minimum reliability classification failed!");
		return false;
	}

	profile[APD_PROFILE_RADIUS] = std::nextafter(APD_SINGLE_MINIMUM_MAX_COST, 0.f);
	const APDProfileSummary belowT2(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	profile[APD_PROFILE_RADIUS] = APD_SINGLE_MINIMUM_MAX_COST;
	const APDProfileSummary atT2(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	profile[APD_PROFILE_RADIUS] = std::nextafter(APD_SINGLE_MINIMUM_MAX_COST, 1.f);
	const APDProfileSummary aboveT2(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	if (belowT2.reliability != APDReliabilityClass::RELIABLE ||
		atT2.reliability != APDReliabilityClass::UNRELIABLE ||
		aboveT2.reliability != APDReliabilityClass::UNRELIABLE ||
		atT2.reason != APDProfileReason::UNRELIABLE_SINGLE_MINIMUM_COST_NOT_STRICTLY_BELOW_T2)
	{
		VERBOSE("ERROR: APD strict t2 boundary contract failed!");
		return false;
	}

	resetProfile();
	profile[0] = 0.1f;
	summary = SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u);
	if (summary.globalMinimumIndex != 0u || summary.localMinimumCount != 1u ||
		summary.reason != APDProfileReason::UNRELIABLE_GLOBAL_MINIMUM_OUTSIDE_ETA)
	{
		VERBOSE("ERROR: APD endpoint global-minimum contract failed!");
		return false;
	}
	resetProfile();
	profile[APD_PROFILE_RADIUS-1u] = 0.1f;
	profile[APD_PROFILE_RADIUS] = 0.1f;
	profile[APD_PROFILE_RADIUS+1u] = 0.1f;
	summary = SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u);
	if (summary.reliability != APDReliabilityClass::RELIABLE ||
		summary.localMinimumCount != 1u ||
		summary.globalMinimumIndex != APD_PROFILE_RADIUS ||
		summary.globalMinimumPlateauStart != APD_PROFILE_RADIUS-1u ||
		summary.globalMinimumPlateauEnd != APD_PROFILE_RADIUS+1u)
	{
		VERBOSE("ERROR: APD plateau-minimum contract failed!");
		return false;
	}
	resetProfile(0.1f);
	summary = SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u);
	if (summary.globalMinimumIndex != APD_PROFILE_RADIUS || summary.localMinimumCount != 0u ||
		summary.reason != APDProfileReason::UNRELIABLE_NO_LOCAL_MINIMUM)
	{
		VERBOSE("ERROR: APD flat-profile contract failed!");
		return false;
	}

	resetProfile();
	profile[APD_PROFILE_RADIUS] = 0.1f;
	profile[APD_PROFILE_RADIUS+10u] = 0.299f;
	const APDProfileSummary belowT3(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	profile[APD_PROFILE_RADIUS+10u] = 0.301f;
	const APDProfileSummary aboveT3(SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u));
	if (belowT3.reliability != APDReliabilityClass::UNRELIABLE ||
		aboveT3.reliability != APDReliabilityClass::RELIABLE ||
		ComputeAPDMinimumSeparation(0.04f, 2u) > APD_MINIMUM_SEPARATION)
	{
		VERBOSE("ERROR: APD strict t3 boundary contract failed!");
		return false;
	}
	resetProfile();
	profile[APD_PROFILE_RADIUS] = 0.1f;
	profile[APD_PROFILE_RADIUS-10u] = 0.31f;
	profile[APD_PROFILE_RADIUS+10u] = 0.31f;
	const APDProfileSummary paperSeparation(SummarizeAPDProfile(
		profile, APD_PROFILE_SIZE, 6u, APDSeparationConvention::PAPER));
	const APDProfileSummary rmsCompatibility(SummarizeAPDProfile(
		profile, APD_PROFILE_SIZE, 6u, APDSeparationConvention::RMS_COMPATIBILITY));
	if (paperSeparation.localMinimumCount != 3u ||
		paperSeparation.reliability != APDReliabilityClass::UNRELIABLE ||
		rmsCompatibility.reliability != APDReliabilityClass::RELIABLE ||
		!nearEqual(paperSeparation.separation, 0.21f/std::sqrt(2.f), 2e-6f) ||
		!nearEqual(rmsCompatibility.separation, 0.21f, 2e-6f))
	{
		VERBOSE("ERROR: APD paper separation differs from its explicit RMS compatibility mode!");
		return false;
	}
	profile[0] = std::numeric_limits<float>::infinity();
	summary = SummarizeAPDProfile(profile, APD_PROFILE_SIZE, 6u);
	if (summary.reliability != APDReliabilityClass::UNKNOWN ||
		summary.reason != APDProfileReason::UNKNOWN_NONFINITE_COST ||
		SummarizeAPDProfile(nullptr, APD_PROFILE_SIZE, 6u).reason !=
			APDProfileReason::UNKNOWN_INVALID_INPUT)
	{
		VERBOSE("ERROR: APD invalid-profile contract failed!");
		return false;
	}

	APDWeightedCostAccumulator weighted;
	APDWeightedCostAccumulator uniform;
	if (!AccumulateAPDViewCost(weighted, true, 31.f, 0.1f) ||
		!AccumulateAPDViewCost(weighted, true, 1.f, 1.1f) ||
		!AccumulateAPDViewCost(uniform, true, 1.f, 0.1f) ||
		!AccumulateAPDViewCost(uniform, true, 1.f, 1.1f) ||
		AccumulateAPDViewCost(weighted, true, 0.f, 0.2f) ||
		!nearEqual(FinishAPDViewCost(weighted), 0.13125f) ||
		!nearEqual(FinishAPDViewCost(uniform), 0.6f) ||
		weighted.contributingViews != 2u || weighted.rejectedViews != 1u)
	{
		VERBOSE("ERROR: APD exact view-weight aggregation failed!");
		return false;
	}

	bool centerOffsets[11][11] = {};
	bool anchorOffsets[11][11] = {};
	for (unsigned sample=0; sample<APD_CENTER_PATCH_SAMPLES; ++sample) {
		const APDPatchOffset offset(MakeAPDPatchOffset(APDPatchKind::CENTER, sample));
		if (!offset.valid || offset.x < -5 || offset.x > 5 || offset.y < -5 || offset.y > 5 ||
			centerOffsets[offset.y+5][offset.x+5])
		{
			VERBOSE("ERROR: APD paper center-patch footprint failed!");
			return false;
		}
		centerOffsets[offset.y+5][offset.x+5] = true;
	}
	for (unsigned sample=0; sample<APD_ANCHOR_PATCH_SAMPLES; ++sample) {
		const APDPatchOffset offset(MakeAPDPatchOffset(APDPatchKind::ANCHOR, sample));
		if (!offset.valid || offset.x < -5 || offset.x > 5 || offset.y < -5 || offset.y > 5 ||
			anchorOffsets[offset.y+5][offset.x+5])
		{
			VERBOSE("ERROR: APD paper anchor-patch footprint failed!");
			return false;
		}
		anchorOffsets[offset.y+5][offset.x+5] = true;
	}
	if (centerOffsets[5][5] || !anchorOffsets[5][5] ||
		MakeAPDPatchOffset(APDPatchKind::CENTER, APD_CENTER_PATCH_SAMPLES).valid ||
		!APDReferencePatchFits(5, 5, 11, 11) || APDReferencePatchFits(4, 5, 11, 11))
	{
		VERBOSE("ERROR: APD paper patch support boundaries failed!");
		return false;
	}

	for (const unsigned sectorCount : {7u, APD_SECTOR_COUNT}) {
		const float expectedAdjacentDot(std::cos(2.f*APD_PI/static_cast<float>(sectorCount)));
		for (unsigned sector=0; sector<sectorCount; ++sector) {
			const APDSectorDirection direction(MakeAPDSectorDirection(sector, sectorCount, 0.5f));
			const APDSectorDirection next(MakeAPDSectorDirection(
				(sector+1u)%sectorCount, sectorCount, 0.5f));
			if (!direction.valid || !next.valid ||
				!nearEqual(direction.x*direction.x+direction.y*direction.y, 1.f, 2e-6f) ||
				!nearEqual(direction.x*next.x+direction.y*next.y, expectedAdjacentDot, 2e-5f) ||
				APDSectorForDirection(direction.x, direction.y, sectorCount) != sector)
			{
				VERBOSE("ERROR: APD uniform-sector contract failed for N=%u!", sectorCount);
				return false;
			}
		}
	}

	const APDPoint2 triangleA{-5.f, -5.f};
	const APDPoint2 triangleB{5.f, -5.f};
	const APDPoint2 triangleC{0.f, 5.f};
	const APDPoint2 center{0.f, 0.f};
	if (!APDTriangleContainsPoint(triangleA, triangleB, triangleC, center) ||
		APDTriangleContainsPoint(triangleA, triangleB, triangleC, APDPoint2{10.f, 10.f}) ||
		APDTriangleContainsPoint(APDPoint2{-1.f, 0.f}, APDPoint2{0.f, 0.f},
			APDPoint2{1.f, 0.f}, center))
	{
		VERBOSE("ERROR: APD center-enclosure contract failed!");
		return false;
	}
	APDPlane plane;
	if (!FitAPDPlane(APDPoint3{0.f, 0.f, 1.f}, APDPoint3{1.f, 0.f, 1.f},
			APDPoint3{0.f, 1.f, 1.f}, plane) ||
		!nearEqual(APDNormalizedPlaneResidual(plane, APDPoint3{2.f, 2.f, 1.004f}, 1.f),
			0.004f, 2e-6f) ||
		!APDPlaneInlier(plane, APDPoint3{2.f, 2.f, 1.004f}, 1.f) ||
		APDPlaneInlier(plane, APDPoint3{2.f, 2.f, 1.006f}, 1.f))
	{
		VERBOSE("ERROR: APD plane fit/inlier contract failed!");
		return false;
	}
	float fittedDepth(0.f);
	if (!APDPlaneDepthAtRay(plane, APDPoint3{0.25f, -0.5f, 1.f}, 0.5f, 2.f,
			fittedDepth) || !nearEqual(fittedDepth, 1.f) ||
		APDPlaneDepthAtRay(APDPlane{1.f, 0.f, 0.f, -1.f},
			APDPoint3{0.f, 0.f, 1.f}, 0.5f, 2.f, fittedDepth))
	{
		VERBOSE("ERROR: APD fitted-plane depth conversion failed!");
		return false;
	}

	for (unsigned trial=0; trial<APD_RANSAC_TRIALS; ++trial) {
		const APDRansacTriplet sample(MakeAPDRansacTriplet(12345u, 7u, trial, 11u));
		const APDRansacTriplet repeated(MakeAPDRansacTriplet(12345u, 7u, trial, 11u));
		if (!sample.valid || sample.first >= sample.second || sample.second >= sample.third ||
			sample.third >= 11u || sample.first != repeated.first ||
			sample.second != repeated.second || sample.third != repeated.third)
		{
			VERBOSE("ERROR: APD deterministic RANSAC sampling failed!");
			return false;
		}
	}
	const APDRansacTriplet firstSample(MakeAPDRansacTriplet(4u, 2u, 0u, 8u));
	const APDRansacTriplet secondSample(MakeAPDRansacTriplet(4u, 2u, 1u, 8u));
	const APDModelQuality invalidLowResidual(MakeAPDModelQuality(
		8u, 5u, true, 0.0001f, 0.0001f, firstSample));
	const APDModelQuality incumbent(MakeAPDModelQuality(
		8u, 6u, true, 0.01f, 0.005f, firstSample));
	const APDModelQuality fewerOutliers(MakeAPDModelQuality(
		8u, 7u, true, 0.02f, 0.01f, secondSample));
	const APDModelQuality lowerCenterResidual(MakeAPDModelQuality(
		8u, 6u, true, 0.009f, 0.5f, secondSample));
	if (invalidLowResidual.valid || !incumbent.valid ||
		PreferAPDModel(invalidLowResidual, incumbent) ||
		!PreferAPDModel(fewerOutliers, incumbent) ||
		!PreferAPDModel(lowerCenterResidual, incumbent))
	{
		VERBOSE("ERROR: APD paper RANSAC lexicographic ranking failed!");
		return false;
	}
	const APDAnchorRank anchorIncumbent{true, 0.002f, 4u};
	const APDAnchorRank closerToPlane{true, 0.001f, 9u};
	const APDAnchorRank nonInlier{false, 0.f, 1u};
	if (!PreferAPDAnchor(closerToPlane, anchorIncumbent) ||
		PreferAPDAnchor(nonInlier, anchorIncumbent))
	{
		VERBOSE("ERROR: APD paper anchor ordering failed!");
		return false;
	}

	const float anchorViewCosts[3][2] = {
		{0.10f, 0.70f},
		{0.12f, 0.72f},
		{0.14f, 0.74f},
	};
	const uint32_t anchorSelectedViews[3] = {1u, 1u, 1u};
	const float viewThreshold(APDViewCostThreshold(0u));
	const APDAnchorViewEvidence supportedView(ComputeAPDAnchorViewEvidence(
		&anchorViewCosts[0][0], 2u, anchorSelectedViews, 0x7u, 3u, 0u, viewThreshold));
	const APDAnchorViewEvidence rejectedView(ComputeAPDAnchorViewEvidence(
		&anchorViewCosts[0][0], 2u, anchorSelectedViews, 0x7u, 3u, 1u, viewThreshold));
	const APDAnchorViewEvidence noSupport(ComputeAPDAnchorViewEvidence(
		&anchorViewCosts[0][0], 2u, anchorSelectedViews, 0u, 3u, 0u, viewThreshold));
	if (!supportedView.valid || !rejectedView.valid || noSupport.valid ||
		supportedView.supportCount != 3u || supportedView.agreeCount != 3u ||
		supportedView.badCount != 0u || !nearEqual(supportedView.prior, 2.7f) ||
		!nearEqual(rejectedView.prior, 0.3f) ||
		!(supportedView.samplingScore > rejectedView.samplingScore) ||
		!nearEqual(APDViewCostThreshold(0u), APD_VIEW_COST_THRESHOLD) ||
		!(APDViewCostThreshold(3u) < APDViewCostThreshold(0u)))
	{
		VERBOSE("ERROR: APD immutable-anchor view evidence contract failed!");
		return false;
	}
	const APDAnchorCandidateRank invalidCandidate;
	const APDAnchorCandidateRank correctAnchor{0.12f, 91u, 5u, true};
	const APDAnchorCandidateRank higherCostAnchor{0.20f, 17u, 1u, true};
	const APDAnchorCandidateRank tiedEarlierAnchor{0.12f, 33u, 3u, true};
	if (!PreferAPDAnchorCandidate(correctAnchor, invalidCandidate) ||
		PreferAPDAnchorCandidate(higherCostAnchor, correctAnchor) ||
		!PreferAPDAnchorCandidate(tiedEarlierAnchor, correctAnchor))
	{
		VERBOSE("ERROR: APD anchor-propagation candidate identity/ranking failed!");
		return false;
	}

	APDAnchorCostAccumulator anchorCosts;
	if (!AccumulateAPDAnchorCost(anchorCosts, true, 0.4f) ||
		!AccumulateAPDAnchorCost(anchorCosts, false, 0.f) ||
		anchorCosts.anchorCount != 2u || anchorCosts.validSupportCount != 1u ||
		anchorCosts.invalidSupportCount != 1u || !nearEqual(anchorCosts.costSum, 1.6f))
	{
		VERBOSE("ERROR: APD anchor support accounting failed!");
		return false;
	}
	const APDScoreDecision deformable(EvaluateAPDWorkingScore(
		true, APDReliabilityClass::UNRELIABLE, true, true, 0.2f, anchorCosts));
	const APDScoreDecision disabled(EvaluateAPDWorkingScore(
		false, APDReliabilityClass::UNRELIABLE, true, true, 0.2f, anchorCosts));
	const APDScoreDecision reliable(EvaluateAPDWorkingScore(
		true, APDReliabilityClass::RELIABLE, true, true, 0.2f, anchorCosts));
	const APDScoreDecision invalidCenter(EvaluateAPDWorkingScore(
		true, APDReliabilityClass::UNRELIABLE, false, true, 0.2f, anchorCosts));
	if (!deformable.usedDeformableCost || deformable.fallback != APDScoreFallback::NONE ||
		!nearEqual(deformable.anchorMeanCost, 0.8f) ||
		!nearEqual(deformable.workingCost, 0.65f) ||
		disabled.fallback != APDScoreFallback::MECHANISM_DISABLED ||
		reliable.fallback != APDScoreFallback::PIXEL_NOT_UNRELIABLE ||
		invalidCenter.fallback != APDScoreFallback::INVALID_CENTER_SUPPORT ||
		!nearEqual(invalidCenter.workingCost, APD_BAD_COST))
	{
		VERBOSE("ERROR: APD mechanism-local deformable score contract failed!");
		return false;
	}
	const float winnerNativeRescore(0.412345f);
	const APDPersistentScoreDecision persistent(ResolveAPDPersistentScore(
		true, 0.7f, winnerNativeRescore));
	if (!persistent.valid || !persistent.usedWinnerRescore || persistent.cost != winnerNativeRescore) {
		VERBOSE("ERROR: APD persistent winner did not retain its conventional rescore!");
		return false;
	}
	const APDFinalRefinementDecision finalAccepted(ResolveAPDFinalRefinement(
		1.f, 0.50f, 1.01f, 0.39f, 1));
	const APDFinalRefinementDecision finalAtThreshold(ResolveAPDFinalRefinement(
		1.f, 0.50f, 1.01f, 0.40f, 1));
	const APDFinalRefinementDecision finalInvalid(ResolveAPDFinalRefinement(
		1.f, 0.50f, -1.f, 0.10f, -1));
	if (!finalAccepted.valid || !finalAccepted.accepted ||
		!nearEqual(finalAccepted.improvement, 0.11f) || finalAccepted.offset != 1 ||
		!finalAtThreshold.valid || finalAtThreshold.accepted || finalInvalid.valid ||
		APD_FINAL_REFINEMENT_RADIUS != 5)
	{
		VERBOSE("ERROR: APD bounded native final-refinement contract failed!");
		return false;
	}

	VERBOSE("APD paper-mechanics contract tests passed");
	return true;
}
/*----------------------------------------------------------------*/

// test MVS stages on a small sample dataset
bool PipelineTest(bool forceCPU, bool verbose)
{
	TD_TIMER_START();
	#if defined(_USE_CUDA) || defined(_USE_METAL)
	// force CPU for testing even if a GPU backend is available
	if (forceCPU)
		SEACAVE::CUDA::desiredDeviceIDs.clear();
	#endif
	Scene scene;
	if (!scene.Load(MAKE_PATH("scene.mvs"))) {
		VERBOSE("ERROR: TestDataset failed loading the scene!");
		return false;
	}
	OPTDENSE::init();
	OPTDENSE::bRemoveDmaps = true;
	// The point/face counts and quality vary run-to-run (multi-threaded
	// densify/mesh) and differ between the CPU and GPU PatchMatch backends, so
	// these are deliberately wide plausibility windows, not tight regression
	// bounds: they bracket both backends' observed spread with margin.
	// Re-baselined 2026-08-10 for the current defaults. Note the backends now
	// differ by design, not just by numerical spread: nOptimize defaults to
	// ADJUST_CONFIDENCE_AUTO, so the confidence recalibration runs on the GPU
	// backend (fused into the last geometric-consistency iteration, nearly free)
	// and is skipped on the CPU backend (where it would cost a separate pass).
	// Also on: fusion rescue (fFusePriorWeight=3, ~+90% dense points on this
	// scene); pointWeights hold the plain [0,1] per-view confidence consumed by
	// the weighted mesh visibility (this test keeps pointWeights, unlike the
	// ReconstructMesh app whose constant-weight default discards them).
	// Measured (GPU adjust-ON / CPU adjust-OFF): recon faces 52.9k / 71.4k,
	// cleaned faces 37.0k / 49.8k, quality 50.4 / 52.2.
	if (!scene.DenseReconstruction() || scene.pointcloud.GetSize() < 50000u) {
		VERBOSE("ERROR: TestDataset failed estimating dense point-cloud (%u points)!", scene.pointcloud.GetSize());
		return false;
	}
	if (verbose)
		scene.pointcloud.Save(MAKE_PATH("scene_dense.ply"));
	if (!scene.ReconstructMesh() || !ISINSIDE(scene.mesh.faces.size(), 40000u, 100000u)) {
		VERBOSE("ERROR: TestDataset failed reconstructing the mesh (%u faces)!", scene.mesh.faces.size());
		return false;
	}
	if (verbose)
		scene.mesh.Save(MAKE_PATH("scene_dense_mesh.ply"));
	constexpr float decimate = 0.7f;
	scene.mesh.Clean(decimate);
	if (!ISINSIDE(scene.mesh.faces.size(), 28000u, 70000u)) {
		VERBOSE("ERROR: TestDataset failed cleaning the mesh (%u faces)!", scene.mesh.faces.size());
		return false;
	}
	if (verbose)
		scene.mesh.Save(MAKE_PATH("scene_dense_mesh_clean.ply"));
	#ifdef _USE_OPENMP
	TestMeshProjectionMT(scene.mesh, scene.images[1]);
	#endif
	if (!scene.TextureMesh(0, 0) || !scene.mesh.HasTexture()) {
		VERBOSE("ERROR: TestDataset failed texturing the mesh!");
		return false;
	}
	if (verbose)
		scene.mesh.Save(MAKE_PATH("scene_dense_mesh_texture.ply"));
	const float qualityScore = scene.ComputeReconstructionQuality().score();
	if (qualityScore < 43.f) {
		VERBOSE("ERROR: TestDataset reconstruction quality too low (%.1f)!", qualityScore);
		return false;
	}
	VERBOSE("All pipeline stages passed (%s)", TD_TIMER_GET_FMT().c_str());
	return true;
}
/*----------------------------------------------------------------*/

} // namespace MVS
