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
#include "../../libs/MVS/CUDA/Camera.h"
#include "../../libs/MVS/PatchMatchAPDCUDA.h"
#include "../../libs/MVS/PatchMatchDVPCUDA.h"
#include "../../libs/MVS/PatchMatchDVPDepthEdgeCUDA.h"
#include "../../libs/MVS/PatchMatchDVPVisibilityCUDA.h"
#include "../../libs/MVS/PatchMatchDVPVisibleNormalCUDA.h"
#ifdef _USE_CUDA
#include "../../libs/MVS/PatchMatchCUDA.h"
#endif

#include <chrono>
#include <filesystem>
#include <cstring>

#include <sstream>


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("TestMVS "));

namespace MVS {
// verify the exact OpenMVS 2.3 AdjustConfidenceFast formula and deferred-write contract
bool ConfidenceCompat23Test()
{
	Scene scene;
	scene.images.Resize(3);
	const Camera camera(Matrix3x3::eye(), Matrix3x3::eye(), Point3(REAL(0), REAL(0), REAL(0)));
	FOREACH(i, scene.images) {
		Image& image(scene.images[i]);
		image.ID = i;
		image.poseID = 0;
		image.width = 3;
		image.height = 1;
		image.camera = camera;
	}
	DepthMapsData depthMaps(scene);
	FOREACH(i, depthMaps.arrDepthData) {
		DepthData& depthData(depthMaps.arrDepthData[i]);
		DepthData::ViewData& view(depthData.images.AddEmpty());
		view.scale = 1.f;
		view.camera = camera;
		view.cameraDepthMap = camera;
		view.pImageData = &scene.images[i];
		depthData.size = cv::Size(3, 1);
		depthData.depthMap.create(1, 3);
		depthData.confMap.create(1, 3);
	}
	DepthData& depthDataRef(depthMaps.arrDepthData[0]);
	depthDataRef.depthMap(0,0) = 1.f;
	depthDataRef.depthMap(0,1) = 1.f;
	depthDataRef.depthMap(0,2) = 0.f;
	depthDataRef.confMap(0,0) = 0.8f;
	depthDataRef.confMap(0,1) = 0.8f;
	depthDataRef.confMap(0,2) = 0.5f;
	DepthData& depthDataN1(depthMaps.arrDepthData[1]);
	depthDataN1.depthMap(0,0) = 1.f;
	depthDataN1.depthMap(0,1) = 1.02f;
	depthDataN1.depthMap(0,2) = 1.f;
	depthDataN1.confMap(0,0) = 0.7f;
	depthDataN1.confMap(0,1) = 0.9f;
	depthDataN1.confMap(0,2) = 0.1f;
	DepthData& depthDataN2(depthMaps.arrDepthData[2]);
	depthDataN2.depthMap(0,0) = 1.f;
	depthDataN2.depthMap(0,1) = 1.03f;
	depthDataN2.depthMap(0,2) = 1.f;
	depthDataN2.confMap(0,0) = 0.6f;
	depthDataN2.confMap(0,1) = 0.8f;
	depthDataN2.confMap(0,2) = 0.1f;
	IIndexArr idxNeighbors;
	idxNeighbors.push_back(1);
	idxNeighbors.push_back(2);
	if (!depthMaps.AdjustConfidenceCompat23(depthDataRef, idxNeighbors) ||
		ABS(depthDataRef.confMapAdjusted(0,0) - 0.91f) > 1e-6f ||
		ABS(depthDataRef.confMapAdjusted(0,1) - 0.1f*0.8f/1.7f) > 1e-6f ||
		depthDataRef.confMapAdjusted(0,2) != 0.f || depthDataRef.confMap(0,0) != 0.8f) {
		VERBOSE("ERROR: OpenMVS 2.3 confidence compatibility formula failed");
		return false;
	}
	depthDataRef.bConfAdjusted = true;
	if (depthMaps.AdjustConfidenceCompat23(depthDataRef, idxNeighbors)) {
		VERBOSE("ERROR: OpenMVS 2.3 confidence compatibility double-adjust guard failed");
		return false;
	}
	return true;
}
/*----------------------------------------------------------------*/

// verify exact float32 DR round-trip and D2 auto-detection
bool DMapCompat23Test()
{
	const struct PassCase {
		unsigned nComponents;
		unsigned nTotalGeometricIters;
		int nGeometricIter;
		unsigned nExpected;
	} passCases[] = {
		{0, 4, -1, 0},
		{3, 0, -1, 0},
		{1, 1, -1, 1},
		{2, 1, 0, 0},
		{3, 4, -1, 3},
		{3, 4, 0, 3},
		{3, 4, 1, 3},
		{3, 4, 2, 3},
		{3, 4, 3, 0},
	};
	for (const PassCase& passCase: passCases) {
		if (IntermediateDMapFloatComponentsForPass(passCase.nComponents,
			passCase.nGeometricIter, passCase.nTotalGeometricIters) != passCase.nExpected) {
			VERBOSE("ERROR: intermediate DMap float-component pass selection failed");
			return false;
		}
	}

	DepthDataRaw data;
	data.header.type = HeaderDepthDataRaw::CONF_ADJUSTED;
	data.header.imageWidth = 2;
	data.header.imageHeight = 2;
	data.header.dMin = 0.5f;
	data.header.dMax = 4.f;
	data.imageFileName = "images/ref.jpg";
	data.IDs = std::vector<uint32_t>{7, 9, 11};
	for (int i=0; i<9; ++i) {
		data.K.val[i] = (i%4 == 0 ? 2.0 : 0.0);
		data.R.val[i] = (i%4 == 0 ? 1.0 : 0.0);
	}
	data.C = cv::Point3_<double>(1.0, 2.0, 3.0);

	cv::Mat depthMap(2, 2, CV_32FC1);
	cv::Mat normalMap(2, 2, CV_32FC3);
	cv::Mat confMap(2, 2, CV_32FC1);
	cv::Mat viewsMap(2, 2, CV_8UC4);
	const float depthValues[] = {0.f, 1.234567f, 2.345678f, 3.456789f};
	const float normalValues[] = {
		0.f, 0.f, 0.f,
		0.267261f, 0.534522f, -0.801784f,
		-0.436436f, 0.872872f, -0.218218f,
		0.707107f, 0.408248f, -0.577350f,
	};
	const float confValues[] = {0.f, 0.123456f, 0.654321f, 0.987654f};
	const uint8_t viewValues[] = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
	};
	std::memcpy(depthMap.ptr(), depthValues, sizeof(depthValues));
	std::memcpy(normalMap.ptr(), normalValues, sizeof(normalValues));
	std::memcpy(confMap.ptr(), confValues, sizeof(confValues));
	std::memcpy(viewsMap.ptr(), viewValues, sizeof(viewValues));

	std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
	if (!ExportDepthDataRawCompat23(stream, data, depthMap, normalMap, confMap, viewsMap)) {
		VERBOSE("ERROR: OpenMVS 2.3 DMap compatibility export failed");
		return false;
	}
	const std::string payload(stream.str());
	if (payload.size() < 2 || payload[0] != 'D' || payload[1] != 'R') {
		VERBOSE("ERROR: OpenMVS 2.3 DMap compatibility header failed");
		return false;
	}
	stream.seekg(0);
	DepthDataRaw imported;
	cv::Mat depthImported, normalImported, confImported, viewsImported;
	if (!ImportDepthDataRawAuto(stream, imported, depthImported, normalImported, confImported, viewsImported, HeaderDepthDataRaw::CONTENT_MASK) ||
		imported.imageFileName != data.imageFileName || imported.IDs != data.IDs ||
		(imported.header.type & HeaderDepthDataRaw::CONF_ADJUSTED) == 0 ||
		std::memcmp(depthMap.ptr(), depthImported.ptr(), sizeof(depthValues)) != 0 ||
		std::memcmp(normalMap.ptr(), normalImported.ptr(), sizeof(normalValues)) != 0 ||
		std::memcmp(confMap.ptr(), confImported.ptr(), sizeof(confValues)) != 0 ||
		std::memcmp(viewsMap.ptr(), viewsImported.ptr(), sizeof(viewValues)) != 0) {
		VERBOSE("ERROR: OpenMVS 2.3 DMap compatibility round-trip failed");
		return false;
	}

	std::stringstream streamD2(std::ios::in | std::ios::out | std::ios::binary);
	if (!ExportDepthDataRaw(streamD2, data, depthMap, normalMap, confMap, viewsMap)) {
		VERBOSE("ERROR: D2 regression export failed");
		return false;
	}
	const std::string payloadD2(streamD2.str());
	streamD2.seekg(0);
	DepthDataRaw importedD2;
	cv::Mat depthImportedD2, normalImportedD2, confImportedD2, viewsImportedD2;
	if (payloadD2.size() < 2 || payloadD2[0] != 'D' || payloadD2[1] != '2' ||
		!ImportDepthDataRawAuto(streamD2, importedD2, depthImportedD2, normalImportedD2, confImportedD2, viewsImportedD2, HeaderDepthDataRaw::CONTENT_MASK) ||
		(importedD2.header.type & HeaderDepthDataRaw::CONF_ADJUSTED) == 0 ||
		std::memcmp(depthMap.ptr(), depthImportedD2.ptr(), sizeof(depthValues)) == 0 ||
		std::memcmp(normalMap.ptr(), normalImportedD2.ptr(), sizeof(normalValues)) == 0 ||
		std::memcmp(confMap.ptr(), confImportedD2.ptr(), sizeof(confValues)) == 0 ||
		std::memcmp(viewsMap.ptr(), viewsImportedD2.ptr(), sizeof(viewValues)) != 0) {
		VERBOSE("ERROR: D2 auto-detection or quantization regression failed");
		return false;
	}

	for (unsigned nFloatComponents=1; nFloatComponents<=3; ++nFloatComponents) {
		std::stringstream streamHybrid(std::ios::in | std::ios::out | std::ios::binary);
		if (!ExportDepthDataRawCompat23Components(streamHybrid, data,
			depthMap, normalMap, confMap, viewsMap, nFloatComponents)) {
			VERBOSE("ERROR: intermediate component handoff export failed");
			return false;
		}
		const std::string payloadHybrid(streamHybrid.str());
		streamHybrid.seekg(0);
		DepthDataRaw importedHybrid;
		cv::Mat depthHybrid, normalHybrid, confHybrid, viewsHybrid;
		if (payloadHybrid.size() < 2 || payloadHybrid[0] != 'D' || payloadHybrid[1] != 'R' ||
			!ImportDepthDataRawAuto(streamHybrid, importedHybrid,
				depthHybrid, normalHybrid, confHybrid, viewsHybrid,
				HeaderDepthDataRaw::CONTENT_MASK) ||
			std::memcmp(depthHybrid.ptr(),
				(nFloatComponents & DMAP_FLOAT_DEPTH) ? depthMap.ptr() : depthImportedD2.ptr(),
				sizeof(depthValues)) != 0 ||
			std::memcmp(normalHybrid.ptr(),
				(nFloatComponents & DMAP_FLOAT_NORMAL) ? normalMap.ptr() : normalImportedD2.ptr(),
				sizeof(normalValues)) != 0 ||
			std::memcmp(confHybrid.ptr(), confImportedD2.ptr(), sizeof(confValues)) != 0 ||
			std::memcmp(viewsHybrid.ptr(), viewsMap.ptr(), sizeof(viewValues)) != 0) {
			VERBOSE("ERROR: intermediate component handoff contract failed for mask %u", nFloatComponents);
			return false;
		}
	}
	return true;
}
/*----------------------------------------------------------------*/

bool AdaptivePatchDeformationContractTest()
{
	using namespace CUDA;
	const auto nearEqual = [](float first, float second, float epsilon = 1e-6f) {
		return std::abs(first-second) <= epsilon;
	};

	APDConfig config;
	if (ValidateAPDConfig(config) != APDConfigStatus::VALID ||
		config.mode != static_cast<unsigned>(APDMode::DISABLED) ||
		APDModeEnabled(config.mode) || APDModeUsesFullMechanics(config.mode) ||
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
	config.mode = static_cast<unsigned>(APDMode::FULL);
	if (ValidateAPDConfig(config) != APDConfigStatus::VALID ||
		!APDModeEnabled(config.mode) || !APDModeUsesFullMechanics(config.mode)) {
		VERBOSE("ERROR: APD full-mechanics config failed validation!");
		return false;
	}
	config.mode = static_cast<unsigned>(APDMode::DEFORMATION_ONLY);
	if (ValidateAPDConfig(config) != APDConfigStatus::VALID ||
		!APDModeEnabled(config.mode) || APDModeUsesFullMechanics(config.mode)) {
		VERBOSE("ERROR: APD deformation-only config failed validation!");
		return false;
	}
	APDConfig invalid(config);
	invalid.mode = 3u;
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
	APDWorkingCostMinimum workingCostMinimum;
	volatile float workingCosts[] = {APD_BAD_COST, 0.4f, 0.6f, 0.2f};
	if (AccumulateAPDWorkingCostMinimum(workingCostMinimum, workingCosts[0]) ||
		!AccumulateAPDWorkingCostMinimum(workingCostMinimum, workingCosts[1]) ||
		!AccumulateAPDWorkingCostMinimum(workingCostMinimum, workingCosts[2]) ||
		!AccumulateAPDWorkingCostMinimum(workingCostMinimum, workingCosts[3]) ||
		workingCostMinimum.finiteCount != 3u ||
		!nearEqual(workingCostMinimum.minimum, 0.2f))
	{
		VERBOSE("ERROR: APD best finite working-cost accumulation failed!");
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

bool DepthVariationProposalContractTest()
{
	using namespace CUDA;
	const auto nearEqual = [](float first, float second, float epsilon = 1e-6f) {
		return std::abs(first-second) <= epsilon;
	};

	DVPConfig config;
	if (ValidateDVPConfig(config) != DVPConfigStatus::VALID ||
		config.family != static_cast<unsigned>(DVPEpipolarFamily::DISABLED) ||
		DVPEpipolarFamilyEnabled(config.family) ||
		config.alpha != 1.f || config.beta != 4.f || config.mu != 3u)
	{
		VERBOSE("ERROR: DVP default-off/paper-parameter contract failed!");
		return false;
	}
	for (unsigned family=static_cast<unsigned>(DVPEpipolarFamily::HISTORICAL_GLOBAL_V0);
		family<=static_cast<unsigned>(DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1); ++family)
	{
		config.family = family;
		if (ValidateDVPConfig(config) != DVPConfigStatus::VALID ||
			!DVPEpipolarFamilyEnabled(family))
		{
			VERBOSE("ERROR: DVP epipolar family %u failed validation!", family);
			return false;
		}
	}
	config.family = 5u;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_FAMILY) {
		VERBOSE("ERROR: DVP accepted an unimplemented family!");
		return false;
	}
	config = DVPConfig{};
	config.alpha = 0.f;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_OFFSETS)
		return false;
	config = DVPConfig{};
	config.beta = 0.f;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_OFFSETS)
		return false;
	config = DVPConfig{};
	config.alpha = FLT_MAX;
	config.beta = FLT_MAX;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_OFFSETS)
		return false;
	config = DVPConfig{};
	config.mu = 0u;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_SUPPORT)
		return false;
	config.mu = DVP_MAX_SOURCE_VIEWS+1u;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_SUPPORT)
		return false;
	config = DVPConfig{};
	config.searchRadius = 0u;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_SEARCH_RADIUS)
		return false;
	config.searchRadius = 1025u;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_SEARCH_RADIUS)
		return false;
	config = DVPConfig{};
	config.reprojectionThreshold = 0.f;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_REPROJECTION_THRESHOLD)
		return false;
	config = DVPConfig{};
	config.relativeDepthThreshold = -0.01f;
	if (ValidateDVPConfig(config) != DVPConfigStatus::INVALID_DEPTH_THRESHOLD) {
		VERBOSE("ERROR: DVP invalid-configuration rejection contract failed!");
		return false;
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
	const DVPIntervalSet paper(DVPBuildPaperIntervals(samples, 3u));
	if (!paper.left.valid || !paper.right.valid ||
		!nearEqual(paper.left.minimum, 0.72f) ||
		!nearEqual(paper.left.maximum, 0.83f) ||
		!nearEqual(paper.right.minimum, 1.16f) ||
		!nearEqual(paper.right.maximum, 1.30f) ||
		paper.left.maximum >= paper.right.minimum)
	{
		VERBOSE("ERROR: DVP Eq. 11 order statistics/disjoint intervals failed!");
		return false;
	}
	for (const float unit : {0.f, 0.25f, 0.5f, 1.f}) {
		const float leftDepth(DVPInterpolateInterval(paper.left, unit));
		const float rightDepth(DVPInterpolateInterval(paper.right, unit));
		if (!DVPIntervalContains(paper.left, leftDepth) ||
			!DVPIntervalContains(paper.right, rightDepth))
		{
			VERBOSE("ERROR: DVP interval-only candidate escaped its paper interval!");
			return false;
		}
	}

	DVPEndpointSamples underSupported;
	for (unsigned i=0u; i<2u; ++i) {
		DVPAppendEndpoint(underSupported.leftOuter, underSupported.leftOuterCount, leftOuter[i]);
		DVPAppendEndpoint(underSupported.leftInner, underSupported.leftInnerCount, leftInner[i]);
		DVPAppendEndpoint(underSupported.rightInner, underSupported.rightInnerCount, rightInner[i]);
		DVPAppendEndpoint(underSupported.rightOuter, underSupported.rightOuterCount, rightOuter[i]);
	}
	const DVPIntervalSet rejected(DVPBuildPaperIntervals(underSupported, 3u));
	if (rejected.left.valid || rejected.right.valid ||
		rejected.reason != DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT)
	{
		VERBOSE("ERROR: DVP accepted fewer than mu endpoint samples!");
		return false;
	}
	DVPEndpointSamples leftOnly(samples);
	leftOnly.rightInnerCount = 2u;
	leftOnly.rightOuterCount = 2u;
	const DVPIntervalSet acceptedLeft(DVPBuildPaperIntervals(leftOnly, 3u));
	if (!acceptedLeft.left.valid || acceptedLeft.right.valid ||
		acceptedLeft.reason != DVPEpipolarUnavailableReason::NONE ||
		!nearEqual(acceptedLeft.left.minimum, paper.left.minimum) ||
		!nearEqual(acceptedLeft.left.maximum, paper.left.maximum))
	{
		VERBOSE("ERROR: DVP discarded an independently supported left interval!");
		return false;
	}
	DVPEndpointSamples rightOnly(samples);
	rightOnly.leftOuterCount = 2u;
	rightOnly.leftInnerCount = 2u;
	const DVPIntervalSet acceptedRight(DVPBuildPaperIntervals(rightOnly, 3u));
	if (acceptedRight.left.valid || !acceptedRight.right.valid ||
		acceptedRight.reason != DVPEpipolarUnavailableReason::NONE ||
		!nearEqual(acceptedRight.right.minimum, paper.right.minimum) ||
		!nearEqual(acceptedRight.right.maximum, paper.right.maximum))
	{
		VERBOSE("ERROR: DVP discarded an independently supported right interval!");
		return false;
	}

	const DVPIntervalSet historical(DVPBuildHistoricalMidpointIntervals(samples, 3u));
	if (!historical.left.valid || !historical.right.valid ||
		!nearEqual(historical.left.minimum, 0.71f) ||
		!nearEqual(historical.left.maximum, 0.84f) ||
		!nearEqual(historical.right.minimum, 1.17f) ||
		!nearEqual(historical.right.maximum, 1.29f) ||
		nearEqual(historical.left.minimum, paper.left.minimum))
	{
		VERBOSE("ERROR: DVP historical endpoint policy is not isolated from Eq. 11!");
		return false;
	}

	const float candidateCosts[] = {0.40f, 0.20f};
	const DVPProposalDecision decision(ResolveDVPProposalDecision(0.50f, candidateCosts, 2u));
	const float tiedCosts[] = {0.50f, 0.50f};
	const DVPProposalDecision tied(ResolveDVPProposalDecision(0.50f, tiedCosts, 2u));
	const float unavailableCosts[] = {-1.f, 0.30f};
	const DVPProposalDecision unavailable(
		ResolveDVPProposalDecision(0.50f, unavailableCosts, 2u));
	if (!decision.accepted || decision.winnerOrdinal != 1u ||
		!nearEqual(decision.winnerCost, 0.20f) ||
		!nearEqual(decision.runnerUpCost, 0.40f) ||
		!nearEqual(decision.winnerRunnerUpGap, 0.20f) || tied.accepted ||
		!unavailable.accepted || unavailable.winnerOrdinal != 1u ||
		unavailable.finiteCount != 1u ||
		!nearEqual(unavailable.winnerCost, 0.30f) ||
		!nearEqual(unavailable.runnerUpCost, 0.50f))
	{
		VERBOSE("ERROR: DVP candidate identity/ranking contract failed!");
		return false;
	}

	// Exercise the same CUDA camera primitives used by the production DVP path
	// on the host. The device oracle repeats this fixture field-for-field.
	CUDA::Matrix3 identity(CUDA::Matrix3::Identity());
	const CUDA::Camera camera(
		CUDA::LinearCameraModel(800.f, 810.f, 320.f, 240.f),
		CUDA::Pose(identity, CUDA::Point3(0.1f, -0.2f, 0.3f)), 640, 480);
	const CUDA::Point2 pixel(217.25f, 139.75f);
	const float depth(4.5f);
	const CUDA::Point3 world(camera.TransformPointI2W(pixel, depth));
	const CUDA::Point2 roundTrip(camera.TransformPointW2I(world));
	const float roundTripDepth((camera.pose.R*(world-camera.pose.C)).z());
	if ((roundTrip-pixel).squaredNorm() > 1e-10f ||
		std::abs(roundTripDepth-depth) > 1e-5f) {
		VERBOSE("ERROR: DVP CPU project/back-project oracle failed!");
		return false;
	}

	#ifdef _USE_CUDA
	int cudaDeviceCount(0);
	const cudaError_t cudaStatus(cudaGetDeviceCount(&cudaDeviceCount));
	if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
		DVPCUDAOracleResult cudaResult;
		if (!CUDA::PatchMatch::RunDVPContractOracle(cudaResult) ||
			cudaResult.passedChecks != 0x7fu ||
			cudaResult.proposalDecision.winnerOrdinal != decision.winnerOrdinal ||
			!nearEqual(cudaResult.paperIntervals.left.minimum, paper.left.minimum) ||
			!nearEqual(cudaResult.paperIntervals.left.maximum, paper.left.maximum) ||
			!nearEqual(cudaResult.paperIntervals.right.minimum, paper.right.minimum) ||
			!nearEqual(cudaResult.paperIntervals.right.maximum, paper.right.maximum))
		{
			VERBOSE("ERROR: DVP CUDA contract oracle failed (checks=0x%02x)!",
				cudaResult.passedChecks);
			return false;
		}
		VERBOSE("DVP epipolar contract CUDA oracle passed");
	} else {
		// CUDA-built CPU-only CI remains useful; GPU qualification requires
		// the device path above and records its explicit success message.
		cudaGetLastError();
		VERBOSE("DVP CUDA contract oracle skipped: no CUDA device available");
	}
	#endif

	VERBOSE("DVP epipolar contract CPU tests passed");
	return true;
}
/*----------------------------------------------------------------*/

bool DVPDepthEdgePriorContractTest()
{
	using namespace CUDA;
	DVPDepthEdgeConfig config;
	if (ValidateDVPDepthEdgeConfig(config) != DVPDepthEdgeConfigStatus::VALID ||
		DVPDepthEdgeModeEnabled(config.mode))
	{
		VERBOSE("ERROR: DVP depth-edge default-off contract failed!");
		return false;
	}
	for (unsigned mode=static_cast<unsigned>(DVPDepthEdgeMode::ROBERTS_REGIONS);
		mode<=static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED); ++mode)
	{
		config = DVPDepthEdgeConfig{mode, true, true};
		if (ValidateDVPDepthEdgeConfig(config) != DVPDepthEdgeConfigStatus::VALID ||
			!DVPDepthEdgeModeEnabled(mode))
		{
			VERBOSE("ERROR: DVP depth-edge mode %u failed validation!", mode);
			return false;
		}
	}
	if (ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{6u, true, true}) !=
			DVPDepthEdgeConfigStatus::INVALID_MODE ||
		ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{1u, false, true}) !=
			DVPDepthEdgeConfigStatus::REQUIRES_APD ||
		ValidateDVPDepthEdgeConfig(DVPDepthEdgeConfig{1u, true, false}) !=
			DVPDepthEdgeConfigStatus::MISSING_PRIOR_DIRECTORY)
	{
		VERBOSE("ERROR: DVP depth-edge invalid-configuration contract failed!");
		return false;
	}
	if (DVP_DEPTH_EDGE_SCHEMA_VERSION != 2u ||
		ValidateDVPDepthEdgePriorGeometry(1920u, 2560u, 3019u, 4026u,
			1920u, 2560u) != DVPDepthEdgePriorGeometryStatus::VALID ||
		ValidateDVPDepthEdgePriorGeometry(1920u, 2560u, 0u, 4026u,
			1920u, 2560u) != DVPDepthEdgePriorGeometryStatus::INVALID_RAW_DIMENSIONS ||
		ValidateDVPDepthEdgePriorGeometry(1920u, 2560u, 3019u, 4026u,
			3019u, 4026u) != DVPDepthEdgePriorGeometryStatus::PROCESSING_DIMENSIONS_MISMATCH)
	{
		VERBOSE("ERROR: DVP depth-edge processing-geometry contract failed!");
		return false;
	}

	const uint16_t labels[8] = {1u, 1u, 2u, 0u, 1u, 2u, 1u, 0u};
	uint32_t anchors[8] = {1u, 2u, 3u, 4u, 5u, 6u, ~uint32_t(0), ~uint32_t(0)};
	const DVPDepthEdgeFilterSummary summary(FilterDVPDepthEdgeAnchors(
		static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED), labels[0],
		labels, 8u, anchors, 6u, 8u, ~uint32_t(0)));
	if (summary.inputCount != 6u || summary.outputCount != 3u ||
		summary.rejectedBoundary != 1u || summary.rejectedCrossRegion != 2u ||
		summary.rejectedInvalidIndex != 0u || anchors[0] != 1u || anchors[1] != 4u ||
		anchors[2] != 6u || anchors[3] != ~uint32_t(0))
	{
		VERBOSE("ERROR: DVP depth-edge stable-compaction contract failed!");
		return false;
	}
	const DVPDepthEdgeAnchorDecision boundary(ResolveDVPDepthEdgeAnchorDecision(
		static_cast<unsigned>(DVPDepthEdgeMode::ERODED), 0u, 1u));
	const DVPDepthEdgeAnchorDecision same(ResolveDVPDepthEdgeAnchorDecision(
		static_cast<unsigned>(DVPDepthEdgeMode::ERODED), 7u, 7u));
	if (boundary.allowed || boundary.reason != DVPDepthEdgeAnchorReason::CENTER_BOUNDARY ||
		!same.allowed || same.reason != DVPDepthEdgeAnchorReason::SAME_REGION)
	{
		VERBOSE("ERROR: DVP depth-edge reason attribution failed!");
		return false;
	}

	#ifdef _USE_CUDA
	namespace fs = std::filesystem;
	const fs::path labelPath(fs::temp_directory_path() /
		fs::path(String::FormatString("openmvs_dvp_labels_%lld.png",
			static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count())).c_str()));
	struct LabelMapCleanup {
		fs::path path;
		~LabelMapCleanup() { std::error_code ec; fs::remove(path, ec); }
	} labelMapCleanup{labelPath};
	Image16U expectedLabels;
	expectedLabels.create(2, 3);
	expectedLabels(0, 0) = 0u;
	expectedLabels(0, 1) = 1u;
	expectedLabels(0, 2) = 119u;
	expectedLabels(1, 0) = 163u;
	expectedLabels(1, 1) = 255u;
	expectedLabels(1, 2) = 511u;
	Image16U loadedLabels;
	if (!expectedLabels.Save(labelPath.string()) ||
		!LoadDVPDepthEdgeLabelMap(labelPath.string(), loadedLabels) ||
		loadedLabels.size() != expectedLabels.size())
	{
		VERBOSE("ERROR: DVP depth-edge 16-bit label-map load failed!");
		return false;
	}
	for (int y=0; y<expectedLabels.height(); ++y)
		for (int x=0; x<expectedLabels.width(); ++x)
			if (loadedLabels(y, x) != expectedLabels(y, x)) {
				VERBOSE("ERROR: DVP depth-edge label identity changed at (%d,%d): %u != %u!",
					x, y, static_cast<unsigned>(loadedLabels(y, x)),
					static_cast<unsigned>(expectedLabels(y, x)));
				return false;
			}

	int cudaDeviceCount(0);
	const cudaError_t cudaStatus(cudaGetDeviceCount(&cudaDeviceCount));
	if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
		DVPDepthEdgeCUDAOracleResult cudaResult;
		if (!PatchMatch::RunDVPDepthEdgeContractOracle(cudaResult) ||
			cudaResult.passedChecks != 0x1fu ||
			cudaResult.mixedRegions.outputCount != summary.outputCount ||
			cudaResult.compactedAnchors[0] != anchors[0] ||
			cudaResult.compactedAnchors[2] != anchors[2])
		{
			VERBOSE("ERROR: DVP depth-edge CUDA oracle failed (checks=0x%02x)!",
				cudaResult.passedChecks);
			return false;
		}
		VERBOSE("DVP depth-edge contract CUDA oracle passed");
	} else {
		cudaGetLastError();
		VERBOSE("DVP depth-edge CUDA oracle skipped: no CUDA device available");
	}
	#endif

	VERBOSE("DVP depth-edge topology and anchor-filter CPU tests passed");
	return true;
}
/*----------------------------------------------------------------*/

bool DVPVisibilityContractTest()
{
	using namespace CUDA;
	const auto nearEqual = [](float first, float second, float epsilon = 1e-6f) {
		return std::abs(first-second) <= epsilon;
	};

	DVPVisibilityConfig config;
	if (ValidateDVPVisibilityConfig(config) != DVPVisibilityConfigStatus::VALID ||
		DVPVisibilityModeEnabled(config.mode))
	{
		VERBOSE("ERROR: DVP visibility default-off contract failed!");
		return false;
	}
	config = DVPVisibilityConfig{
		static_cast<unsigned>(DVPVisibilityMode::PAPER_2D_RESTORE_V1),
		DVP_VISIBILITY_REPROJECTION_THRESHOLD,
		DVP_VISIBILITY_RELATIVE_DEPTH_THRESHOLD,
		true,
		true,
	};
	if (ValidateDVPVisibilityConfig(config) != DVPVisibilityConfigStatus::VALID ||
		!DVPVisibilityModeEnabled(config.mode))
	{
		VERBOSE("ERROR: DVP paper-2D visibility configuration failed!");
		return false;
	}
	config.mode = static_cast<unsigned>(DVPVisibilityMode::DEPTH_GATED_RESTORE_V1);
	if (ValidateDVPVisibilityConfig(config) != DVPVisibilityConfigStatus::VALID)
		return false;
	DVPVisibilityConfig invalid(config);
	invalid.mode = 3u;
	if (ValidateDVPVisibilityConfig(invalid) != DVPVisibilityConfigStatus::INVALID_MODE)
		return false;
	invalid = config;
	invalid.fullAPD = false;
	if (ValidateDVPVisibilityConfig(invalid) != DVPVisibilityConfigStatus::REQUIRES_FULL_APD)
		return false;
	invalid = config;
	invalid.geometricConsistency = false;
	if (ValidateDVPVisibilityConfig(invalid) !=
		DVPVisibilityConfigStatus::REQUIRES_GEOMETRIC_CONSISTENCY)
	{
		return false;
	}

	DVPVisibilityObservation sameRayOccluder;
	sameRayOccluder.referenceDepth = 10.f;
	sameRayOccluder.expectedSourceDepth = 10.f;
	sameRayOccluder.observedSourceDepth = 5.f;
	sameRayOccluder.roundTripError = 0.f;
	sameRayOccluder.forwardInside = true;
	sameRayOccluder.backwardInside = true;
	DVPVisibilityConfig paper(config);
	paper.mode = static_cast<unsigned>(DVPVisibilityMode::PAPER_2D_RESTORE_V1);
	const DVPVisibilityDecision paperOccluder(
		ResolveDVPVisibility(paper, 0u, sameRayOccluder));
	const DVPVisibilityDecision depthOccluder(
		ResolveDVPVisibility(config, 0u, sameRayOccluder));
	if (!paperOccluder.visible || !paperOccluder.restored ||
		paperOccluder.weight != DVP_VISIBILITY_RESTORED_WEIGHT ||
		paperOccluder.reason != DVPVisibilityReason::RESTORED_PAPER_2D ||
		depthOccluder.visible || depthOccluder.restored || depthOccluder.weight != 0u ||
		depthOccluder.reason != DVPVisibilityReason::NEARER_OCCLUDER_REJECTED)
	{
		VERBOSE("ERROR: DVP same-ray occluder separation failed!");
		return false;
	}

	DVPVisibilityObservation agreement(sameRayOccluder);
	agreement.observedSourceDepth = 10.05f;
	agreement.roundTripError = 0.25f;
	const DVPVisibilityDecision depthAgreement(
		ResolveDVPVisibility(config, 0u, agreement));
	if (!depthAgreement.visible || !depthAgreement.restored ||
		depthAgreement.reason != DVPVisibilityReason::RESTORED_DEPTH_GATED ||
		!nearEqual(depthAgreement.relativeDepthError, 0.005f))
	{
		VERBOSE("ERROR: DVP depth-gated restoration failed!");
		return false;
	}
	DVPVisibilityObservation fartherDisagreement(agreement);
	fartherDisagreement.observedSourceDepth = 10.2f;
	const DVPVisibilityDecision rejectedFarther(
		ResolveDVPVisibility(config, 0u, fartherDisagreement));
	if (rejectedFarther.visible ||
		rejectedFarther.reason != DVPVisibilityReason::DEPTH_DISAGREEMENT_REJECTED)
	{
		VERBOSE("ERROR: DVP depth disagreement gate failed!");
		return false;
	}
	DVPVisibilityObservation backwardOut(agreement);
	backwardOut.backwardInside = false;
	const DVPVisibilityDecision rejectedBackward(
		ResolveDVPVisibility(config, 0u, backwardOut));
	if (rejectedBackward.reason != DVPVisibilityReason::BACKWARD_OUT_OF_BOUNDS)
		return false;
	DVPVisibilityObservation forwardOut(agreement);
	forwardOut.forwardInside = false;
	const DVPVisibilityDecision rejectedForward(
		ResolveDVPVisibility(config, 0u, forwardOut));
	if (rejectedForward.reason != DVPVisibilityReason::FORWARD_OUT_OF_BOUNDS)
		return false;
	const DVPVisibilityDecision retained(ResolveDVPVisibility(config, 7u, DVPVisibilityObservation{}));
	if (!retained.visible || retained.restored || retained.weight != 7u ||
		retained.reason != DVPVisibilityReason::RETAINED_PREVIOUS_WEIGHT)
	{
		VERBOSE("ERROR: DVP immutable previous-weight retention failed!");
		return false;
	}

	const uint8_t weights[] = {3u, 0u, 2u};
	const DVPVisibilityWeightSummary summary(SummarizeDVPVisibilityWeights(weights, 3u));
	float normalizedSum(0.f);
	for (const uint8_t weight : weights)
		normalizedSum += DVPVisibilityNormalizedWeight(weight, summary.sum);
	if (!summary.valid || summary.sum != 5u || summary.visibleCount != 2u ||
		!nearEqual(normalizedSum, 1.f))
	{
		VERBOSE("ERROR: DVP visibility normalization contract failed!");
		return false;
	}
	const DVPVisibilityStateHeader previous{
		DVP_VISIBILITY_STATE_VERSION, 640u, 480u, 5u, 3u};
	const DVPVisibilityStateHeader next{
		DVP_VISIBILITY_STATE_VERSION, 640u, 480u, 5u, 4u};
	DVPVisibilityStateHeader skipped(next);
	skipped.logicalIteration = 5u;
	if (ValidateDVPVisibilityTransition(previous, next) != DVPVisibilityTransitionStatus::VALID ||
		ValidateDVPVisibilityTransition(previous, skipped) !=
			DVPVisibilityTransitionStatus::INVALID_LOGICAL_ITERATION)
	{
		VERBOSE("ERROR: DVP logical-iteration visibility transition failed!");
		return false;
	}

	#ifdef _USE_CUDA
	int cudaDeviceCount(0);
	const cudaError_t cudaStatus(cudaGetDeviceCount(&cudaDeviceCount));
	if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
		DVPVisibilityCUDAOracleResult cudaResult;
		if (!PatchMatch::RunDVPVisibilityContractOracle(cudaResult) ||
			cudaResult.passedChecks != 0xffu ||
			cudaResult.paperOccluder.reason != paperOccluder.reason ||
			cudaResult.depthOccluder.reason != depthOccluder.reason ||
			cudaResult.depthAgreement.reason != depthAgreement.reason ||
			cudaResult.weightSummary.sum != summary.sum ||
			!nearEqual(cudaResult.normalizedWeightSum, normalizedSum))
		{
			VERBOSE("ERROR: DVP visibility CUDA oracle failed (checks=0x%02x)!",
				cudaResult.passedChecks);
			return false;
		}
		VERBOSE("DVP visibility contract CUDA oracle passed");
	} else {
		cudaGetLastError();
		VERBOSE("DVP visibility CUDA contract oracle skipped: no CUDA device available");
	}
	#endif

	VERBOSE("DVP persistent visibility CPU tests passed");
	return true;
}
/*----------------------------------------------------------------*/

bool DVPVisibleNormalContractTest()
{
	using namespace CUDA;
	const auto nearEqual = [](float first, float second, float epsilon = 1e-6f) {
		return std::abs(first-second) <= epsilon;
	};
	const auto sameVector = [&](const DVPVisibleNormalVector& first,
		const DVPVisibleNormalVector& second) {
		return nearEqual(first.x, second.x) && nearEqual(first.y, second.y) &&
			nearEqual(first.z, second.z);
	};

	DVPVisibleNormalConfig config;
	if (ValidateDVPVisibleNormalConfig(config) != DVPVisibleNormalConfigStatus::VALID ||
		DVPVisibleNormalModeEnabled(config.mode) ||
		config.attempts != DVP_VISIBLE_NORMAL_DEFAULT_ATTEMPTS ||
		config.dotTolerance != DVP_VISIBLE_NORMAL_DEFAULT_DOT_TOLERANCE)
	{
		VERBOSE("ERROR: DVP visible-normal default-off contract failed!");
		return false;
	}
	for (unsigned mode=static_cast<unsigned>(DVPVisibleNormalMode::SHADOW);
		mode<=static_cast<unsigned>(DVPVisibleNormalMode::FULL); ++mode)
	{
		config = DVPVisibleNormalConfig{};
		config.mode = mode;
		config.fullAPD = true;
		if (ValidateDVPVisibleNormalConfig(config) != DVPVisibleNormalConfigStatus::VALID ||
			!DVPVisibleNormalModeEnabled(mode))
		{
			VERBOSE("ERROR: DVP visible-normal mode %u failed validation!", mode);
			return false;
		}
	}
	if (!DVPVisibleNormalRefinementEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::REFINEMENT)) ||
		DVPVisibleNormalPropagationEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::REFINEMENT)) ||
		!DVPVisibleNormalPropagationEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::PROPAGATION)) ||
		DVPVisibleNormalRefinementEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::PROPAGATION)) ||
		!DVPVisibleNormalRefinementEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::FULL)) ||
		!DVPVisibleNormalPropagationEnabled(
			static_cast<unsigned>(DVPVisibleNormalMode::FULL)))
	{
		VERBOSE("ERROR: DVP visible-normal mode stage contract failed!");
		return false;
	}
	config = DVPVisibleNormalConfig{};
	config.mode = 5u;
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::INVALID_MODE)
		return false;
	config = DVPVisibleNormalConfig{};
	config.dotTolerance = -0.01f;
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::INVALID_DOT_TOLERANCE)
		return false;
	config.dotTolerance = 1.01f;
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::INVALID_DOT_TOLERANCE)
		return false;
	config = DVPVisibleNormalConfig{};
	config.attempts = 0u;
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::INVALID_ATTEMPTS)
		return false;
	config.attempts = DVP_VISIBLE_NORMAL_MAX_ATTEMPTS+1u;
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::INVALID_ATTEMPTS)
		return false;
	config = DVPVisibleNormalConfig{};
	config.mode = static_cast<unsigned>(DVPVisibleNormalMode::SHADOW);
	if (ValidateDVPVisibleNormalConfig(config) !=
		DVPVisibleNormalConfigStatus::REQUIRES_FULL_APD)
	{
		VERBOSE("ERROR: DVP visible normals accepted a non-APD configuration!");
		return false;
	}

	DVPVisibleNormalRotation rotation;
	rotation.values[0] = 0.f;
	rotation.values[1] = -1.f;
	rotation.values[3] = 1.f;
	rotation.values[4] = 0.f;
	const DVPVisibleNormalVector referenceCenter{1.f, 2.f, 3.f};
	const DVPVisibleNormalVector sourceCenter{3.f, 2.f, 3.f};
	const DVPVisibleNormalVector sourceCenterReference(
		DVPVisibleNormalSourceCenterInReference(
			rotation, referenceCenter, sourceCenter));
	DVPVisibleNormalVector sourceDirection;
	if (!sameVector(sourceCenterReference, {0.f, 2.f, 0.f}) ||
		!DVPVisibleNormalCameraToPointDirection(
			{0.f, 0.f, 4.f}, sourceCenterReference, sourceDirection) ||
		!nearEqual(sourceDirection.x, 0.f) ||
		!nearEqual(sourceDirection.y, -0.4472136f) ||
		!nearEqual(sourceDirection.z, 0.8944272f))
	{
		VERBOSE("ERROR: DVP visible-normal reference-frame transform failed!");
		return false;
	}

	const DVPVisibleNormalVector referenceOnly[] = {{0.f, 0.f, 3.f}};
	const DVPVisibleNormalEvaluation referenceFeasible(EvaluateDVPVisibleNormal(
		{0.f, 0.f, -2.f}, referenceOnly, 1u, 0.f));
	const DVPVisibleNormalEvaluation referenceRejected(EvaluateDVPVisibleNormal(
		{0.f, 0.f, 2.f}, referenceOnly, 1u, 0.f));
	if (!referenceFeasible.valid || !referenceFeasible.feasible ||
		!referenceRejected.valid || referenceRejected.feasible ||
		referenceRejected.reason !=
			DVPVisibleNormalEvaluationReason::HEMISPHERE_REJECTED)
	{
		VERBOSE("ERROR: DVP visible-normal reference hemisphere failed!");
		return false;
	}
	const DVPVisibleNormalVector multiDirections[] = {
		{0.f, 0.f, 1.f}, sourceDirection};
	const DVPVisibleNormalVector multiDirectionsBefore[] = {
		multiDirections[0], multiDirections[1]};
	const DVPVisibleNormalVector feasibleNormal{0.f, 1.f, -1.f};
	const DVPVisibleNormalVector rejectedNormal{0.f, -1.f, 1.f};
	const DVPVisibleNormalEvaluation multiFeasible(EvaluateDVPVisibleNormal(
		feasibleNormal, multiDirections, 2u, 0.f));
	const DVPVisibleNormalEvaluation multiRejected(EvaluateDVPVisibleNormal(
		rejectedNormal, multiDirections, 2u, 0.f));
	const DVPVisibleNormalVector orthogonalDirections[] = {{1.f, 0.f, 0.f}};
	const DVPVisibleNormalEvaluation boundary(EvaluateDVPVisibleNormal(
		{0.f, 1.f, 0.f}, orthogonalDirections, 1u, 0.f));
	if (!multiFeasible.valid || !multiFeasible.feasible ||
		!multiRejected.valid || multiRejected.feasible ||
		multiRejected.maxViolation <= 0.f ||
		multiRejected.rejectedDirection == DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE ||
		!boundary.valid || !boundary.feasible || !nearEqual(boundary.maxDot, 0.f) ||
		!sameVector(multiDirections[0], multiDirectionsBefore[0]) ||
		!sameVector(multiDirections[1], multiDirectionsBefore[1]))
	{
		VERBOSE("ERROR: DVP visible-normal multi-view/tolerance contract failed!");
		return false;
	}
	const DVPVisibleNormalVector zeroDirection[] = {{0.f, 0.f, 0.f}};
	const DVPVisibleNormalEvaluation invalidNormal(EvaluateDVPVisibleNormal(
		{0.f, 0.f, 0.f}, referenceOnly, 1u, 0.f));
	const DVPVisibleNormalEvaluation invalidDirection(EvaluateDVPVisibleNormal(
		feasibleNormal, zeroDirection, 1u, 0.f));
	const DVPVisibleNormalEvaluation noDirections(EvaluateDVPVisibleNormal(
		feasibleNormal, nullptr, 0u, 0.f));
	if (invalidNormal.valid ||
		invalidNormal.reason != DVPVisibleNormalEvaluationReason::INVALID_NORMAL ||
		invalidDirection.valid ||
		invalidDirection.reason != DVPVisibleNormalEvaluationReason::INVALID_DIRECTION ||
		noDirections.valid ||
		noDirections.reason != DVPVisibleNormalEvaluationReason::INVALID_DIRECTION_COUNT)
	{
		VERBOSE("ERROR: DVP visible-normal invalid-input attribution failed!");
		return false;
	}
	const DVPVisibleNormalVector contradictoryDirections[] = {
		{1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f},
		{0.f, 1.f, 0.f}, {0.f, -1.f, 0.f},
		{0.f, 0.f, 1.f}, {0.f, 0.f, -1.f},
	};
	const DVPVisibleNormalEvaluation contradictory(EvaluateDVPVisibleNormal(
		{1.f, 1.f, 1.f}, contradictoryDirections, 6u, 0.f));
	if (!contradictory.valid || contradictory.feasible) {
		VERBOSE("ERROR: DVP visible-normal contradictory-cone fixture failed!");
		return false;
	}

	const DVPVisibleNormalVector retries[] = {rejectedNormal, feasibleNormal};
	const DVPVisibleNormalProposalDecision retryDecision(
		ResolveDVPVisibleNormalProposal(
			rejectedNormal, retries, 2u, multiDirections, 2u, 0.f));
	const DVPVisibleNormalVector rejectedRetries[] = {rejectedNormal, rejectedNormal};
	const DVPVisibleNormalProposalDecision fallbackDecision(
		ResolveDVPVisibleNormalProposal(
			rejectedNormal, rejectedRetries, 2u, multiDirections, 2u, 0.f));
	const DVPVisibleNormalProposalDecision nativeDecision(
		ResolveDVPVisibleNormalProposal(
			feasibleNormal, nullptr, 0u, multiDirections, 2u, 0.f));
	if (!retryDecision.valid || retryDecision.fallback ||
		retryDecision.reason != DVPVisibleNormalProposalReason::CONSTRAINED_RETRY ||
		retryDecision.retriesTested != 2u || retryDecision.selectedRetry != 1u ||
		!retryDecision.selectedEvaluation.feasible ||
		!fallbackDecision.valid || !fallbackDecision.fallback ||
		fallbackDecision.reason !=
			DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK ||
		fallbackDecision.retriesTested != 2u ||
		fallbackDecision.selectedEvaluation.feasible ||
		!sameVector(fallbackDecision.selected, rejectedNormal) ||
		!nativeDecision.valid || nativeDecision.fallback ||
		nativeDecision.reason != DVPVisibleNormalProposalReason::NATIVE_FEASIBLE)
	{
		VERBOSE("ERROR: DVP visible-normal retry/fallback contract failed!");
		return false;
	}

	const float costs[] = {0.10f, 0.20f, 0.05f};
	const uint8_t valid[] = {1u, 1u, 1u};
	const uint8_t feasible[] = {0u, 1u, 0u};
	const uint8_t noneFeasible[] = {0u, 0u, 0u};
	const uint8_t noneValid[] = {0u, 0u, 0u};
	const DVPVisibleNormalPropagationDecision constrained(
		ResolveDVPVisibleNormalPropagation(costs, valid, feasible, 3u));
	const DVPVisibleNormalPropagationDecision propagationFallback(
		ResolveDVPVisibleNormalPropagation(costs, valid, noneFeasible, 3u));
	const DVPVisibleNormalPropagationDecision unavailable(
		ResolveDVPVisibleNormalPropagation(costs, noneValid, noneFeasible, 3u));
	if (!constrained.valid || constrained.fallback || constrained.nativeBest != 2u ||
		constrained.constrainedBest != 1u || constrained.selected != 1u ||
		!propagationFallback.valid || !propagationFallback.fallback ||
		propagationFallback.selected != 2u ||
		propagationFallback.reason !=
			DVPVisibleNormalPropagationReason::NO_FEASIBLE_NATIVE_FALLBACK ||
		!unavailable.valid || unavailable.fallback ||
		unavailable.reason != DVPVisibleNormalPropagationReason::NO_VALID_CANDIDATE ||
		unavailable.selected != DVP_VISIBLE_NORMAL_INDEX_UNAVAILABLE)
	{
		VERBOSE("ERROR: DVP visible-normal propagation fallback contract failed!");
		return false;
	}

	#ifdef _USE_CUDA
	int cudaDeviceCount(0);
	const cudaError_t cudaStatus(cudaGetDeviceCount(&cudaDeviceCount));
	if (cudaStatus == cudaSuccess && cudaDeviceCount > 0) {
		DVPVisibleNormalCUDAOracleResult cudaResult;
		if (!PatchMatch::RunDVPVisibleNormalContractOracle(cudaResult) ||
			cudaResult.passedChecks != 0x3ffu ||
			!sameVector(cudaResult.sourceCenter, sourceCenterReference) ||
			!sameVector(cudaResult.sourceDirection, sourceDirection) ||
			cudaResult.retryDecision.selectedRetry != retryDecision.selectedRetry ||
			cudaResult.propagationDecision.selected != constrained.selected ||
			cudaResult.propagationFallback.selected != propagationFallback.selected)
		{
			VERBOSE("ERROR: DVP visible-normal CUDA contract oracle failed (checks=0x%03x)!",
				cudaResult.passedChecks);
			return false;
		}
		VERBOSE("DVP visible-normal contract CUDA oracle passed");
	} else {
		cudaGetLastError();
		VERBOSE("DVP visible-normal CUDA contract oracle skipped: no CUDA device available");
	}
	#endif

	VERBOSE("DVP visible-normal frame, hemisphere, and fallback CPU tests passed");
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
