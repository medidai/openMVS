/*
* SceneDensify.cpp
*
* Copyright (c) 2014-2015 SEACAVE
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

#include "Common.h"
#include "Scene.h"
#include "SceneDensify.h"
#include "PatchMatchCUDA.h"
#include "PatchMatchMetal.h"
#include "DMapCache.h"
#ifdef _USE_DMAP_INSTRUMENTATION
#include "../IO/json.hpp"
#endif

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif

#pragma push_macro("VERBOSE")
#undef VERBOSE
#define VERBOSE(...) LOG(lt, __VA_ARGS__)


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("ScnDense"));
#ifdef _USE_DMAP_INSTRUMENTATION
namespace {

enum CPUViewSelectionStopReason {
	CPU_VIEW_STOP_NONE = 0,
	CPU_VIEW_STOP_EXPLICIT_PAIR,
	CPU_VIEW_STOP_NUM_NEIGHBORS,
	CPU_VIEW_STOP_SCORE_THRESHOLD,
};

String TrimDMapInstrumentToken(const String& token)
{
	size_t begin(0), end(token.size());
	while (begin < end && std::isspace((unsigned char)token[begin]))
		++begin;
	while (end > begin && std::isspace((unsigned char)token[end-1]))
		--end;
	return token.substr(begin, end-begin);
}

bool DMapInstrumentImageListed(const Image& image)
{
	if (OPTDENSE::strDMapInstrumentationImageList.empty())
		return true;
	const String fileName(Util::getFileNameExt(image.name));
	const String stem(Util::getFileName(image.name));
	const String imageIDText(std::to_string(image.ID).c_str());
	size_t start(0);
	while (start <= OPTDENSE::strDMapInstrumentationImageList.size()) {
		const size_t comma(OPTDENSE::strDMapInstrumentationImageList.find(',', start));
		const size_t end(comma == String::npos ? OPTDENSE::strDMapInstrumentationImageList.size() : comma);
		const String token(TrimDMapInstrumentToken(OPTDENSE::strDMapInstrumentationImageList.substr(start, end-start)));
		if (!token.empty() &&
			(token == imageIDText || token == image.name || token == fileName || token == stem))
			return true;
		if (comma == String::npos)
			break;
		start = comma + 1;
	}
	return false;
}

bool DMapInstrumentImageEnabled(const Image& image)
{
	if (OPTDENSE::strDMapInstrumentationDir.empty())
		return false;
	if (!DMapInstrumentImageListed(image))
		return false;
	const float sampleRate(OPTDENSE::fDMapInstrumentationSampleRate);
	if (sampleRate >= 1.f)
		return true;
	if (sampleRate <= 0.f)
		return false;
	uint32_t hash(image.ID ^ (OPTDENSE::nDMapInstrumentationSampleSeed + 0x9e3779b9u));
	hash ^= hash >> 16;
	hash *= 0x7feb352du;
	hash ^= hash >> 15;
	hash *= 0x846ca68bu;
	hash ^= hash >> 16;
	return float(hash % 1000000u) / 1000000.f < sampleRate;
}

String DMapInstrumentSafeName(const String& imageName)
{
	String safe(Util::getFileName(imageName));
	if (safe.empty())
		safe = Util::getFileNameExt(imageName);
	for (char& ch: safe)
		if (!std::isalnum((unsigned char)ch) && ch != '-' && ch != '_')
			ch = '_';
	if (safe.empty())
		safe = _T("image");
	if (safe.size() > 80)
		safe.resize(80);
	return safe;
}

String DMapInstrumentFrameName(const Image& image)
{
	return String::FormatString(_T("%04u_%s"), image.ID, DMapInstrumentSafeName(image.name).c_str());
}

String DMapInstrumentFrameDir(const Image& image, int geometricIteration=-1)
{
	String root(OPTDENSE::strDMapInstrumentationDir);
	Util::ensureFolderSlash(root);
	Util::ensureFolder(root);
	if (geometricIteration >= 0) {
		root += String::FormatString(_T("geometric_iterations/iteration%02d/"), geometricIteration);
		Util::ensureFolder(root);
	}
	const String depthmapsDir(root + _T("depthmaps/"));
	const String frameDir(depthmapsDir + DMapInstrumentFrameName(image) + _T("/"));
	Util::ensureFolder(depthmapsDir);
	Util::ensureFolder(frameDir);
	return frameDir;
}

String DMapCsvEscape(const String& value)
{
	bool quote(false);
	String out;
	for (const char ch: value) {
		if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r')
			quote = true;
		if (ch == '"')
			out += _T("\"\"");
		else
			out += ch;
	}
	return quote ? _T("\"") + out + _T("\"") : out;
}

bool WriteDMapJson(const String& fileName, const nlohmann::json& data)
{
	const String temporary(fileName + _T(".tmp"));
	bool success(false);
	{
		std::ofstream fs(temporary.c_str(), std::ios::trunc);
		if (!fs)
			return false;
		fs << data.dump(2) << '\n';
		fs.flush();
		success = (bool)fs;
	}
	if (!success || !File::renameFile(temporary, fileName)) {
		File::deleteFile(temporary);
		return false;
	}
	return true;
}

const char* InitialViewDecisionName(Scene::NeighborViewCandidateObservation::InitialDecision decision)
{
	switch (decision) {
	case Scene::NeighborViewCandidateObservation::INITIAL_REFERENCE_IMAGE: return "reference_image";
	case Scene::NeighborViewCandidateObservation::INITIAL_INVALID_IMAGE: return "rejected_invalid_image";
	case Scene::NeighborViewCandidateObservation::INITIAL_INSUFFICIENT_SHARED_POINTS: return "rejected_insufficient_shared_points";
	case Scene::NeighborViewCandidateObservation::INITIAL_NO_PROJECTED_POINTS: return "rejected_no_projected_points";
	case Scene::NeighborViewCandidateObservation::INITIAL_RANKED: return "ranked";
	case Scene::NeighborViewCandidateObservation::INITIAL_PRECOMPUTED_RANKED: return "precomputed_ranked";
	default: return "not_evaluated";
	}
}

const char* FilterViewDecisionName(Scene::NeighborViewCandidateObservation::FilterDecision decision)
{
	switch (decision) {
	case Scene::NeighborViewCandidateObservation::FILTER_REJECTED_THRESHOLD: return "rejected_threshold";
	case Scene::NeighborViewCandidateObservation::FILTER_RETAINED_MINIMUM_VIEW_GUARD: return "retained_minimum_view_guard";
	case Scene::NeighborViewCandidateObservation::FILTER_RETAINED_THRESHOLD_PASS: return "retained_threshold_pass";
	case Scene::NeighborViewCandidateObservation::FILTER_REJECTED_MAX_VIEW_TRUNCATION: return "rejected_max_view_truncation";
	default: return "not_evaluated";
	}
}

const char* CPUViewStopReasonName(CPUViewSelectionStopReason reason)
{
	switch (reason) {
	case CPU_VIEW_STOP_EXPLICIT_PAIR: return "explicit_neighbor_pair";
	case CPU_VIEW_STOP_NUM_NEIGHBORS: return "requested_neighbor_limit";
	case CPU_VIEW_STOP_SCORE_THRESHOLD: return "score_threshold";
	default: return "none";
	}
}

void InitPrecomputedViewObservation(
	const Scene& scene,
	IIndex idxImage,
	Scene::NeighborViewSelectionObservation& observation)
{
	observation = Scene::NeighborViewSelectionObservation();
	observation.source = Scene::NeighborViewSelectionObservation::SOURCE_PRECOMPUTED_SCENE_NEIGHBORS;
	observation.referenceID = idxImage;
	observation.requiredMinViews = OPTDENSE::nMinViews;
	observation.requiredMinPointViews = OPTDENSE::nMinViewsTrustPoint > 1 ? OPTDENSE::nMinViewsTrustPoint : 2;
	observation.effectiveMinViews = MINF(observation.requiredMinViews, scene.nCalibratedImages-1);
	observation.effectiveMinPointViews = MINF(observation.requiredMinPointViews, scene.nCalibratedImages);
	observation.optimalAngle = D2R(OPTDENSE::fOptimAngle);
	observation.roiWeight = OPTDENSE::fWeightPointInsideROI;
	const ViewScoreArr& neighbors(scene.images[idxImage].neighbors);
	observation.rankingSucceeded = !neighbors.empty();
	observation.candidates.reserve(neighbors.size());
	FOREACH(rank, neighbors) {
		const ViewScore& neighbor(neighbors[rank]);
		Scene::NeighborViewCandidateObservation candidate;
		candidate.ID = neighbor.ID;
		candidate.imageValid = neighbor.ID < scene.images.size() && scene.images[neighbor.ID].IsValid();
		candidate.initialDecision = Scene::NeighborViewCandidateObservation::INITIAL_PRECOMPUTED_RANKED;
		candidate.sharedPoints = neighbor.points;
		candidate.avgScale = neighbor.scale;
		candidate.avgAngle = neighbor.angle;
		candidate.area = neighbor.area;
		candidate.areaFactor = MAXF(neighbor.area, 0.01f);
		candidate.score = neighbor.score;
		candidate.rawRank = rank;
		observation.candidates.emplace_back(candidate);
	}
}

nlohmann::json ViewThresholdReasons(const Scene::NeighborViewCandidateObservation& candidate)
{
	nlohmann::json reasons(nlohmann::json::array());
	if (candidate.belowMinArea)
		reasons.push_back("area_below_minimum");
	if (candidate.belowMinScale)
		reasons.push_back("scale_below_minimum");
	if (candidate.atOrAboveMaxScale)
		reasons.push_back("scale_at_or_above_maximum");
	if (candidate.scaleNotFinite)
		reasons.push_back("scale_not_finite");
	if (candidate.belowMinAngle)
		reasons.push_back("angle_below_minimum");
	if (candidate.atOrAboveMaxAngle)
		reasons.push_back("angle_at_or_above_maximum");
	if (candidate.angleNotFinite)
		reasons.push_back("angle_not_finite");
	return reasons;
}

bool WriteDMapText(const String& fileName, const std::string& text)
{
	const String temporary(fileName + _T(".tmp"));
	bool success(false);
	{
		std::ofstream fs(temporary.c_str(), std::ios::trunc);
		if (!fs)
			return false;
		fs << text;
		fs.flush();
		success = (bool)fs;
	}
	if (!success || !File::renameFile(temporary, fileName)) {
		File::deleteFile(temporary);
		return false;
	}
	return true;
}

bool DMapInstrumentWriteMaps()
{
	return OPTDENSE::bDMapInstrumentationWriteMaps ||
		OPTDENSE::strDMapInstrumentationLevel.ToLower() == _T("maps");
}

const char* DMapEstimationStageName(int geometricIteration)
{
	return geometricIteration >= 0 ? "geometric_consistency" : "photometric";
}

int DMapFinalActiveGeometricIteration(const DenseDepthMapData& data)
{
	return data.nFusionMode >= 0 && OPTDENSE::nEstimationGeometricIters > 0 ?
		(int)OPTDENSE::nEstimationGeometricIters-1 : -1;
}

uint64_t DMapSaturatingAdd(uint64_t first, uint64_t second)
{
	return first > std::numeric_limits<uint64_t>::max()-second ?
		std::numeric_limits<uint64_t>::max() : first+second;
}

uint64_t DMapSaturatingMul(uint64_t first, uint64_t second)
{
	return first && second > std::numeric_limits<uint64_t>::max()/first ?
		std::numeric_limits<uint64_t>::max() : first*second;
}

bool DMapResourceFits(uint64_t bytes, unsigned limitMB)
{
	return limitMB == 0 || bytes <= DMapSaturatingMul((uint64_t)limitMB, 1024u*1024u);
}

std::mutex g_dmapFilterResourceMutex;
std::unordered_map<std::string, uint64_t> g_dmapFilterStorageReservations;

struct DMapFilterStorageLease {
	DMapFilterStorageLease() = default;
	DMapFilterStorageLease(const DMapFilterStorageLease&) = delete;
	DMapFilterStorageLease& operator=(const DMapFilterStorageLease&) = delete;
	~DMapFilterStorageLease() { Release(); }

	void Release()
	{
		if (!active)
			return;
		std::lock_guard<std::mutex> lock(g_dmapFilterResourceMutex);
		const auto it(g_dmapFilterStorageReservations.find(key));
		if (it != g_dmapFilterStorageReservations.end()) {
			if (it->second <= bytes)
				g_dmapFilterStorageReservations.erase(it);
			else
				it->second -= bytes;
		}
		active = false;
		bytes = 0;
	}

	std::string key;
	uint64_t bytes{0};
	bool active{false};
};

struct DMapFilterResourcePlan {
	String component;
	uint64_t pixels{0};
	unsigned postprocessStageCount{0};
	bool postprocessRequested{false};
	bool confidenceRequested{false};
	bool componentRequested{false};
	bool mapsRequested{false};
	bool summaryAvailable{true};
	bool mapsAvailable{false};
	bool fatal{false};
	uint64_t postprocessSummaryHostBytes{0};
	uint64_t confidenceSummaryHostBytes{0};
	uint64_t summaryHostPeakBytes{0};
	uint64_t postprocessMapsHostPeakBytes{0};
	uint64_t confidenceMapsHostPeakBytes{0};
	uint64_t mapsHostPeakBytes{0};
	uint64_t postprocessMapStorageBytes{0};
	uint64_t confidenceMapStorageBytes{0};
	uint64_t totalMapStorageBytes{0};
	uint64_t effectiveHostBytes{0};
	uint64_t effectiveStorageBytes{0};
	bool storageQueryAttempted{false};
	bool storageQuerySucceeded{false};
	uint64_t storageAvailableBytes{0};
	uint64_t storageReservedBeforeBytes{0};
	uint64_t storageEffectiveAvailableBytes{0};
	uint64_t storageLeasedBytes{0};
	uint64_t storageReservedAfterAdmissionBytes{0};
	uint64_t storageReservedAfterReleaseBytes{0};
	bool leaseReleased{false};
	String reservationKey;
	String decision{_T("pending")};
	String reason;
};

struct DMapArtifactUsage {
	uint64_t mapCount{0};
	uint64_t declaredBytes{0};
	uint64_t fileBytes{0};
};

DMapArtifactUsage DMapArtifactUsageFromMaps(const nlohmann::json& maps)
{
	DMapArtifactUsage usage;
	if (!maps.is_array())
		return usage;
	usage.mapCount = maps.size();
	for (const nlohmann::json& map: maps) {
		usage.declaredBytes = DMapSaturatingAdd(usage.declaredBytes, map.value("declared_bytes", 0ull));
		usage.fileBytes = DMapSaturatingAdd(usage.fileBytes, map.value("file_bytes", 0ull));
	}
	return usage;
}

DMapFilterResourcePlan PlanDMapFilterResources(uint64_t pixels, const char* component)
{
	DMapFilterResourcePlan plan;
	plan.component = component;
	plan.pixels = pixels;
	const unsigned postprocessStages(
		((OPTDENSE::nOptimize & OPTDENSE::REMOVE_SPECKLES) != 0 ? 1u : 0u) +
		((OPTDENSE::nOptimize & OPTDENSE::FILL_GAPS) != 0 ? 1u : 0u));
	plan.postprocessStageCount = postprocessStages;
	plan.postprocessRequested = postprocessStages > 0;
	plan.confidenceRequested =
		(OPTDENSE::nOptimize & (OPTDENSE::ADJUST_CONFIDENCE_FAST | OPTDENSE::ADJUST_CONFIDENCE)) != 0;
	const bool isPostprocess(plan.component == _T("postprocess_filters"));
	plan.componentRequested = isPostprocess ? plan.postprocessRequested : plan.confidenceRequested;
	plan.mapsRequested = DMapInstrumentWriteMaps() && plan.componentRequested;
	const uint64_t fixedSummaryBytes(256u*1024u);
	// Each active stage snapshot retains depth, confidence, and float3 normal state.
	// Both snapshots remain live when speckle removal and gap filling are enabled.
	plan.postprocessSummaryHostBytes = plan.postprocessRequested ?
		DMapSaturatingAdd(
			DMapSaturatingMul(pixels, DMapSaturatingMul(postprocessStages, 20u)),
			fixedSummaryBytes) : fixedSummaryBytes;
	const unsigned confidenceSummaryBytesPerPixel(
		plan.confidenceRequested ?
			(((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE_FAST) != 0 &&
			  (OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE) != 0) ? 12u : 8u) : 0u);
	plan.confidenceSummaryHostBytes = plan.confidenceRequested ?
		DMapSaturatingAdd(DMapSaturatingMul(pixels, confidenceSummaryBytesPerPixel), fixedSummaryBytes) : fixedSummaryBytes;
	plan.summaryHostPeakBytes = isPostprocess ?
		plan.postprocessSummaryHostBytes : plan.confidenceSummaryHostBytes;
	// Map export additionally retains depth delta plus validity transition (5 B/pixel),
	// then confidence delta plus transition (5 B/pixel); the peak transient is 10 B/pixel.
	plan.postprocessMapsHostPeakBytes = plan.postprocessRequested ?
		DMapSaturatingAdd(
			DMapSaturatingMul(pixels,
				DMapSaturatingAdd(DMapSaturatingMul(postprocessStages, 20u), 10u)),
			fixedSummaryBytes) : fixedSummaryBytes;
	plan.confidenceMapsHostPeakBytes = plan.confidenceRequested ?
		DMapSaturatingAdd(DMapSaturatingMul(pixels, 24u), fixedSummaryBytes) : fixedSummaryBytes;
	plan.mapsHostPeakBytes = isPostprocess ?
		plan.postprocessMapsHostPeakBytes : plan.confidenceMapsHostPeakBytes;
	// Per stage: depth before/after/delta (12 B), validity transition (1 B), confidence
	// before/after/delta (12 B), confidence transition (1 B), normal before/after
	// (24 B), and normal-angle delta (4 B), for 54 B/pixel and 11 maps.
	plan.postprocessMapStorageBytes = DMapSaturatingMul(pixels, DMapSaturatingMul(postprocessStages, 54u));
	// Confidence adjustment conservatively allows input/depth-validity plus fast, full, and final output/delta/transition triplets.
	plan.confidenceMapStorageBytes = plan.confidenceRequested ? DMapSaturatingMul(pixels, 32u) : 0;
	const uint64_t postprocessPotentialMaps((uint64_t)postprocessStages*11u);
	const uint64_t confidencePotentialMaps(plan.confidenceRequested ? 11u : 0u);
	plan.postprocessMapStorageBytes = DMapSaturatingAdd(plan.postprocessMapStorageBytes,
		DMapSaturatingMul(postprocessPotentialMaps, 4096u));
	plan.confidenceMapStorageBytes = DMapSaturatingAdd(plan.confidenceMapStorageBytes,
		DMapSaturatingMul(confidencePotentialMaps, 4096u));
	plan.totalMapStorageBytes = DMapSaturatingAdd(plan.postprocessMapStorageBytes, plan.confidenceMapStorageBytes);

	const bool summaryFits(DMapResourceFits(plan.summaryHostPeakBytes, OPTDENSE::nDMapInstrumentationMaxHostMB));
	const bool mapsFitBudgets(
		DMapResourceFits(plan.mapsHostPeakBytes, OPTDENSE::nDMapInstrumentationMaxHostMB) &&
		DMapResourceFits(plan.totalMapStorageBytes, OPTDENSE::nDMapInstrumentationMaxFrameStorageMB));
	const bool errorPolicy(OPTDENSE::strDMapInstrumentationBudgetPolicy.ToLower() == _T("error"));
	plan.summaryAvailable = summaryFits;
	if (!summaryFits) {
		plan.decision = errorPolicy ? _T("error_summary_host_budget") : _T("unavailable_summary_host_budget");
		plan.reason = String::FormatString(_T("summary observer needs %llu host bytes"),
			(unsigned long long)plan.summaryHostPeakBytes);
		plan.fatal = errorPolicy;
		return plan;
	}
	if (!plan.mapsRequested) {
		plan.decision = _T("summary");
		plan.reason = _T("filter maps were not requested or no optional filter is active");
		plan.effectiveHostBytes = plan.summaryHostPeakBytes;
		return plan;
	}
	if (!mapsFitBudgets) {
		plan.decision = errorPolicy ? _T("error_map_budget") : _T("summary_budget_degraded");
		plan.reason = String::FormatString(
			_T("map observer needs %llu host bytes and %llu uncompressed storage bytes"),
			(unsigned long long)plan.mapsHostPeakBytes,
			(unsigned long long)plan.totalMapStorageBytes);
		plan.fatal = errorPolicy;
		plan.effectiveHostBytes = plan.summaryHostPeakBytes;
		return plan;
	}
	plan.mapsAvailable = true;
	plan.decision = _T("maps_pending_storage_preflight");
	plan.reason = _T("configured host and frame-storage budgets admit filter maps");
	plan.effectiveHostBytes = plan.mapsHostPeakBytes;
	plan.effectiveStorageBytes = isPostprocess ?
		plan.postprocessMapStorageBytes : plan.confidenceMapStorageBytes;
	return plan;
}

std::string DMapFilterReservationKey()
{
	std::filesystem::path keyPath(OPTDENSE::strDMapInstrumentationDir.c_str());
	std::error_code pathError;
	const std::filesystem::path canonicalPath(std::filesystem::weakly_canonical(keyPath, pathError));
	if (!pathError)
		keyPath = canonicalPath;
	else {
		pathError.clear();
		const std::filesystem::path absolutePath(std::filesystem::absolute(keyPath, pathError));
		keyPath = pathError ? keyPath.lexically_normal() : absolutePath.lexically_normal();
	}
	return keyPath.generic_string();
}

void ApplyDMapFilterStoragePreflight(
	const String& frameDir,
	DMapFilterResourcePlan& plan,
	DMapFilterStorageLease& lease)
{
	if (!plan.mapsRequested || !plan.mapsAvailable)
		return;
	plan.storageQueryAttempted = true;
	const bool errorPolicy(OPTDENSE::strDMapInstrumentationBudgetPolicy.ToLower() == _T("error"));
	std::lock_guard<std::mutex> lock(g_dmapFilterResourceMutex);
	const std::string reservationKey(DMapFilterReservationKey());
	plan.reservationKey = reservationKey.c_str();
	std::error_code spaceError;
	const std::filesystem::space_info spaceInfo(
		std::filesystem::space(std::filesystem::path(frameDir.c_str()), spaceError));
	if (spaceError) {
		plan.storageQuerySucceeded = false;
		plan.mapsAvailable = false;
		plan.effectiveHostBytes = plan.summaryAvailable ? plan.summaryHostPeakBytes : 0;
		plan.effectiveStorageBytes = 0;
		plan.decision = errorPolicy ? _T("error_storage_query") : _T("summary_storage_query_degraded");
		plan.reason = String::FormatString(_T("std::filesystem::space failed: %s"),
			spaceError.message().c_str());
		plan.fatal = errorPolicy;
		return;
	}
	plan.storageQuerySucceeded = true;
	plan.storageAvailableBytes = (uint64_t)spaceInfo.available;
	const auto existing(g_dmapFilterStorageReservations.find(reservationKey));
	plan.storageReservedBeforeBytes = existing == g_dmapFilterStorageReservations.end() ? 0 : existing->second;
	plan.storageEffectiveAvailableBytes = plan.storageAvailableBytes > plan.storageReservedBeforeBytes ?
		plan.storageAvailableBytes-plan.storageReservedBeforeBytes : 0;
	if (plan.storageEffectiveAvailableBytes < plan.effectiveStorageBytes) {
		plan.mapsAvailable = false;
		plan.effectiveHostBytes = plan.summaryAvailable ? plan.summaryHostPeakBytes : 0;
		const uint64_t requestedBytes(plan.effectiveStorageBytes);
		plan.effectiveStorageBytes = 0;
		plan.decision = errorPolicy ? _T("error_insufficient_storage") : _T("summary_storage_degraded");
		plan.reason = String::FormatString(
			_T("component needs %llu bytes, but only %llu bytes remain after %llu in-process reserved bytes"),
			(unsigned long long)requestedBytes,
			(unsigned long long)plan.storageEffectiveAvailableBytes,
			(unsigned long long)plan.storageReservedBeforeBytes);
		plan.fatal = errorPolicy;
		return;
	}
	plan.storageLeasedBytes = plan.effectiveStorageBytes;
	g_dmapFilterStorageReservations[reservationKey] =
		DMapSaturatingAdd(plan.storageReservedBeforeBytes, plan.storageLeasedBytes);
	plan.storageReservedAfterAdmissionBytes = g_dmapFilterStorageReservations[reservationKey];
	lease.key = reservationKey;
	lease.bytes = plan.storageLeasedBytes;
	lease.active = plan.storageLeasedBytes > 0;
	plan.decision = _T("maps_admitted");
	plan.reason = _T("configured budgets and concurrent-aware filesystem capacity admit component maps");
}

void ReleaseDMapFilterStorageLease(
	DMapFilterResourcePlan& plan,
	DMapFilterStorageLease& lease)
{
	lease.Release();
	plan.leaseReleased = true;
	std::lock_guard<std::mutex> lock(g_dmapFilterResourceMutex);
	const auto existing(g_dmapFilterStorageReservations.find(plan.reservationKey.c_str()));
	plan.storageReservedAfterReleaseBytes = existing == g_dmapFilterStorageReservations.end() ? 0 : existing->second;
}

std::mutex g_dmapFilterPlanMutex;

void WriteDMapFilterResourcePlan(
	const Scene& scene,
	IIndex idxImage,
	int geometricIteration,
	const DMapFilterResourcePlan& plan,
	const DMapArtifactUsage* usage=NULL)
{
	const Image& reference(scene.images[idxImage]);
	const String frameDir(DMapInstrumentFrameDir(reference, geometricIteration));
	const String fileName(frameDir + _T("filter_resource_plan.json"));
	std::lock_guard<std::mutex> lock(g_dmapFilterPlanMutex);
	nlohmann::json artifact;
	{
		std::ifstream fs(fileName.c_str());
		if (fs) {
			try { fs >> artifact; }
			catch (...) { artifact = nlohmann::json(); }
		}
	}
	if (!artifact.is_object() || artifact.value("schema_name", "") != "openmvs.dmap.filter_resource_plan" ||
		artifact.value("schema_version", 0) != 2)
	{
		artifact = {
			{"schema_name", "openmvs.dmap.filter_resource_plan"},
			{"schema_version", 2},
			{"scope", "depth-map optional postprocess and confidence adjustment"},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"reference_scene_image_index", idxImage},
			{"reference_image_id", reference.ID},
			{"reference_image_name", reference.name.c_str()},
			{"width", reference.image.width()},
			{"height", reference.image.height()},
			{"pixels", plan.pixels},
			{"optimize_flags", OPTDENSE::nOptimize},
			{"budget_policy", OPTDENSE::strDMapInstrumentationBudgetPolicy.c_str()},
			{"limits_mib", {
				{"host", OPTDENSE::nDMapInstrumentationMaxHostMB},
				{"frame_storage", OPTDENSE::nDMapInstrumentationMaxFrameStorageMB},
			}},
			{"aggregate_uncompressed_map_estimate_bytes", {
				{"postprocess_filters", plan.postprocessMapStorageBytes},
				{"confidence_adjustment", plan.confidenceMapStorageBytes},
				{"total", plan.totalMapStorageBytes},
			}},
			{"estimate_model", {
					{"postprocess_filters", {
						{"active_stage_count", plan.postprocessStageCount},
						{"maximum_maps_per_active_stage", 11},
						{"uncompressed_bytes_per_pixel_per_active_stage", 54},
						{"retained_snapshot_bytes_per_pixel_per_active_stage", 20},
						{"map_export_peak_scratch_bytes_per_pixel", 10},
						{"summary_observer_host_bytes_per_pixel", plan.postprocessStageCount*20u},
						{"conservative_maps_host_peak_bytes_per_pixel", plan.postprocessStageCount*20u+10u},
					{"per_map_reservation_overhead_bytes", 4096},
					{"normal_maps_conditional_on_normal_state", true},
					{"confidence_maps_conditional_on_confidence_state", true},
				}},
			}},
			{"components", nlohmann::json::object()},
		};
	}
	nlohmann::json component = {
		{"requested_capabilities", {
			{"summary", true},
			{"maps", plan.mapsRequested},
			{"filter_active", plan.componentRequested},
		}},
		{"effective_capabilities", {
			{"summary", plan.summaryAvailable},
			{"maps", plan.mapsAvailable},
		}},
		{"estimate_bytes", {
			{"summary_host_peak", plan.summaryHostPeakBytes},
			{"maps_host_peak", plan.mapsHostPeakBytes},
			{"component_uncompressed_map_storage", plan.component == _T("postprocess_filters") ?
				plan.postprocessMapStorageBytes : plan.confidenceMapStorageBytes},
			{"all_filter_uncompressed_map_storage", plan.totalMapStorageBytes},
		}},
		{"effective_estimate_bytes", {
			{"host", plan.effectiveHostBytes},
			{"storage", plan.effectiveStorageBytes},
		}},
		{"storage_preflight", {
			{"attempted", plan.storageQueryAttempted},
			{"succeeded", plan.storageQuerySucceeded},
			{"available_bytes", plan.storageAvailableBytes},
			{"reserved_before_bytes", plan.storageReservedBeforeBytes},
			{"effective_available_bytes", plan.storageEffectiveAvailableBytes},
			{"leased_bytes", plan.storageLeasedBytes},
			{"reserved_after_admission_bytes", plan.storageReservedAfterAdmissionBytes},
			{"lease_released", plan.leaseReleased},
			{"reserved_after_release_bytes", plan.storageReservedAfterReleaseBytes},
			{"reservation_key", plan.reservationKey.c_str()},
		}},
		{"decision", plan.decision.c_str()},
		{"reason", plan.reason.c_str()},
		{"fatal", plan.fatal},
	};
	component["actual_maps"] = usage ? nlohmann::json({
		{"map_count", usage->mapCount},
		{"declared_bytes", usage->declaredBytes},
		{"file_bytes", usage->fileBytes},
		{"estimate_covers_declared_bytes", usage->declaredBytes <=
			(plan.component == _T("postprocess_filters") ? plan.postprocessMapStorageBytes : plan.confidenceMapStorageBytes)},
	}) : nlohmann::json(nullptr);
	artifact["components"][plan.component.c_str()] = component;
	if (!WriteDMapJson(fileName, artifact))
		VERBOSE("warning: failed to write depth-map filter resource plan: %s", fileName.c_str());
}

bool PrepareDMapFilterInstrumentation(
	const Scene& scene,
	IIndex idxImage,
	int geometricIteration,
	uint64_t pixels,
	const char* component,
	DMapFilterResourcePlan& plan,
	DMapFilterStorageLease& lease)
{
	plan = PlanDMapFilterResources(pixels, component);
	const String frameDir(DMapInstrumentFrameDir(scene.images[idxImage], geometricIteration));
	ApplyDMapFilterStoragePreflight(frameDir, plan, lease);
	WriteDMapFilterResourcePlan(scene, idxImage, geometricIteration, plan);
	if (plan.fatal) {
		VERBOSE("error: depth-map %s instrumentation rejected before observer allocation: %s",
			component, plan.reason.c_str());
		exit(EXIT_FAILURE);
	}
	return plan.summaryAvailable;
}

void FinishDMapFilterInstrumentation(
	const Scene& scene,
	IIndex idxImage,
	int geometricIteration,
	DMapFilterResourcePlan& plan,
	DMapFilterStorageLease& lease,
	const DMapArtifactUsage& usage)
{
	if (plan.storageLeasedBytes > 0)
		ReleaseDMapFilterStorageLease(plan, lease);
	WriteDMapFilterResourcePlan(scene, idxImage, geometricIteration, plan, &usage);
}

uint64_t DMapFilterPixelCount(const Image& image, const DepthMap* depthMap=NULL)
{
	if (depthMap && !depthMap->empty())
		return (uint64_t)depthMap->total();
	return image.image.empty() ? 0 : (uint64_t)image.image.total();
}

struct DMapStateSummary {
	uint64_t totalPixels{0};
	uint64_t validPixels{0};
	double depthSum{0};
	bool normalAvailable{false};
	uint64_t validNormalPixels{0};
	uint64_t validDepthNormalPixels{0};
	bool confidenceAvailable{false};
	uint64_t positiveConfidencePixels{0};
	double confidenceSum{0};
};

float DMapNormalSquaredNorm(const Normal& normal)
{
	return normal.x*normal.x + normal.y*normal.y + normal.z*normal.z;
}

bool DMapNormalIsValid(const Normal& normal)
{
	const float squaredNorm(DMapNormalSquaredNorm(normal));
	return std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z) &&
		std::isfinite(squaredNorm) && squaredNorm > std::numeric_limits<float>::epsilon();
}

DMapStateSummary SummarizeDMapState(
	const DepthMap& depthMap,
	const NormalMap& normalMap,
	const ConfidenceMap& confMap)
{
	DMapStateSummary summary;
	if (depthMap.empty())
		return summary;
	summary.totalPixels = (uint64_t)depthMap.total();
	summary.normalAvailable = !normalMap.empty() && normalMap.size() == depthMap.size();
	summary.confidenceAvailable = !confMap.empty() && confMap.size() == depthMap.size();
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			const Depth depth(depthMap(r,c));
			if (depth > 0) {
				++summary.validPixels;
				summary.depthSum += depth;
			}
			if (summary.normalAvailable && DMapNormalIsValid(normalMap(r,c))) {
				++summary.validNormalPixels;
				if (depth > 0)
					++summary.validDepthNormalPixels;
			}
			if (summary.confidenceAvailable) {
				const float confidence(confMap(r,c));
				if (confidence > 0)
					++summary.positiveConfidencePixels;
				summary.confidenceSum += confidence;
			}
		}
	}
	return summary;
}

nlohmann::json DMapStateSummaryJson(const DMapStateSummary& summary)
{
	return {
		{"total_pixels", summary.totalPixels},
		{"valid_depth_pixels", summary.validPixels},
		{"valid_depth_ratio", summary.totalPixels ? double(summary.validPixels)/double(summary.totalPixels) : 0.0},
		{"mean_valid_depth", summary.validPixels ? nlohmann::json(summary.depthSum/double(summary.validPixels)) : nlohmann::json(nullptr)},
		{"normal_available", summary.normalAvailable},
		{"valid_normal_pixels", summary.normalAvailable ? nlohmann::json(summary.validNormalPixels) : nlohmann::json(nullptr)},
		{"valid_normal_ratio", summary.normalAvailable && summary.totalPixels ?
			nlohmann::json(double(summary.validNormalPixels)/double(summary.totalPixels)) : nlohmann::json(nullptr)},
		{"valid_depth_normal_pixels", summary.normalAvailable ? nlohmann::json(summary.validDepthNormalPixels) : nlohmann::json(nullptr)},
		{"confidence_available", summary.confidenceAvailable},
		{"positive_confidence_pixels", summary.confidenceAvailable ? nlohmann::json(summary.positiveConfidencePixels) : nlohmann::json(nullptr)},
		{"positive_confidence_ratio", summary.confidenceAvailable && summary.totalPixels ?
			nlohmann::json(double(summary.positiveConfidencePixels)/double(summary.totalPixels)) : nlohmann::json(nullptr)},
		{"mean_confidence", summary.confidenceAvailable && summary.totalPixels ?
			nlohmann::json(summary.confidenceSum/double(summary.totalPixels)) : nlohmann::json(nullptr)},
	};
}

struct DMapPostprocessStageStats {
	DMapStateSummary input;
	DMapStateSummary output;
	uint64_t removedPixels{0};
	uint64_t addedPixels{0};
	uint64_t retainedValidPixels{0};
	uint64_t depthChangedPixels{0};
	double depthDeltaSum{0};
	double depthAbsDeltaSum{0};
	double depthMaxAbsDelta{0};
	uint64_t comparableDepthPixels{0};
	double comparableDepthAbsDeltaSum{0};
	double comparableDepthMaxAbsDelta{0};
	bool normalDeltaAvailable{false};
	uint64_t normalChangedPixels{0};
	uint64_t normalBecameValidPixels{0};
	uint64_t normalBecameInvalidPixels{0};
	uint64_t comparableNormalPixels{0};
	double normalAngleDeltaSumDegrees{0};
	double normalAngleDeltaMaxDegrees{0};
	bool confidenceDeltaAvailable{false};
	uint64_t confidenceChangedPixels{0};
	uint64_t confidenceBecamePositivePixels{0};
	uint64_t confidenceBecameZeroPixels{0};
	double confidenceDeltaSum{0};
	double confidenceAbsDeltaSum{0};
	double confidenceMaxAbsDelta{0};
};

DMapPostprocessStageStats ComputeDMapPostprocessStageStats(
	const DepthMap& depthBefore,
	const NormalMap& normalBefore,
	const ConfidenceMap& confidenceBefore,
	const DepthMap& depthAfter,
	const NormalMap& normalAfter,
	const ConfidenceMap& confidenceAfter)
{
	ASSERT(depthBefore.size() == depthAfter.size());
	DMapPostprocessStageStats stats;
	stats.input = SummarizeDMapState(depthBefore, normalBefore, confidenceBefore);
	stats.output = SummarizeDMapState(depthAfter, normalAfter, confidenceAfter);
	stats.normalDeltaAvailable = stats.input.normalAvailable && stats.output.normalAvailable;
	stats.confidenceDeltaAvailable = stats.input.confidenceAvailable && stats.output.confidenceAvailable;
	for (int r=0; r<depthBefore.rows; ++r) {
		for (int c=0; c<depthBefore.cols; ++c) {
			const Depth inputDepth(depthBefore(r,c));
			const Depth outputDepth(depthAfter(r,c));
			const bool inputValid(inputDepth > 0);
			const bool outputValid(outputDepth > 0);
			if (inputValid && !outputValid)
				++stats.removedPixels;
			else if (!inputValid && outputValid)
				++stats.addedPixels;
			else if (inputValid)
				++stats.retainedValidPixels;
			const double depthDelta(double(outputDepth)-double(inputDepth));
			const double depthAbsDelta(ABS(depthDelta));
			if (inputDepth != outputDepth)
				++stats.depthChangedPixels;
			stats.depthDeltaSum += depthDelta;
			stats.depthAbsDeltaSum += depthAbsDelta;
			stats.depthMaxAbsDelta = MAXF(stats.depthMaxAbsDelta, depthAbsDelta);
			if (inputValid && outputValid) {
				++stats.comparableDepthPixels;
				stats.comparableDepthAbsDeltaSum += depthAbsDelta;
				stats.comparableDepthMaxAbsDelta = MAXF(stats.comparableDepthMaxAbsDelta, depthAbsDelta);
			}
			if (stats.normalDeltaAvailable) {
				const Normal& inputNormal(normalBefore(r,c));
				const Normal& outputNormal(normalAfter(r,c));
				const bool inputNormalValid(DMapNormalIsValid(inputNormal));
				const bool outputNormalValid(DMapNormalIsValid(outputNormal));
				if (inputNormal.x != outputNormal.x || inputNormal.y != outputNormal.y || inputNormal.z != outputNormal.z)
					++stats.normalChangedPixels;
				if (!inputNormalValid && outputNormalValid)
					++stats.normalBecameValidPixels;
				else if (inputNormalValid && !outputNormalValid)
					++stats.normalBecameInvalidPixels;
				if (inputNormalValid && outputNormalValid) {
					const float cosine(CLAMP(inputNormal.dot(outputNormal) /
						SQRT(DMapNormalSquaredNorm(inputNormal)*DMapNormalSquaredNorm(outputNormal)), -1.f, 1.f));
					const double angleDegrees(R2D(ACOS(cosine)));
					++stats.comparableNormalPixels;
					stats.normalAngleDeltaSumDegrees += angleDegrees;
					stats.normalAngleDeltaMaxDegrees = MAXF(stats.normalAngleDeltaMaxDegrees, angleDegrees);
				}
			}
			if (stats.confidenceDeltaAvailable) {
				const float inputConfidence(confidenceBefore(r,c));
				const float outputConfidence(confidenceAfter(r,c));
				const double confidenceDelta(double(outputConfidence)-double(inputConfidence));
				const double confidenceAbsDelta(ABS(confidenceDelta));
				if (inputConfidence != outputConfidence)
					++stats.confidenceChangedPixels;
				if (inputConfidence <= 0 && outputConfidence > 0)
					++stats.confidenceBecamePositivePixels;
				if (inputConfidence > 0 && outputConfidence <= 0)
					++stats.confidenceBecameZeroPixels;
				stats.confidenceDeltaSum += confidenceDelta;
				stats.confidenceAbsDeltaSum += confidenceAbsDelta;
				stats.confidenceMaxAbsDelta = MAXF(stats.confidenceMaxAbsDelta, confidenceAbsDelta);
			}
		}
	}
	return stats;
}

nlohmann::json DMapPostprocessStageStatsJson(const DMapPostprocessStageStats& stats)
{
	const uint64_t totalPixels(stats.input.totalPixels);
	return {
		{"input", DMapStateSummaryJson(stats.input)},
		{"output", DMapStateSummaryJson(stats.output)},
		{"removed_pixels", stats.removedPixels},
		{"added_pixels", stats.addedPixels},
		{"retained_valid_pixels", stats.retainedValidPixels},
		{"depth_changed_pixels", stats.depthChangedPixels},
		{"depth_delta_mean_all_pixels", totalPixels ? stats.depthDeltaSum/double(totalPixels) : 0.0},
		{"depth_abs_delta_mean_all_pixels", totalPixels ? stats.depthAbsDeltaSum/double(totalPixels) : 0.0},
		{"depth_abs_delta_max", stats.depthMaxAbsDelta},
		{"comparable_valid_depth_pixels", stats.comparableDepthPixels},
		{"comparable_valid_depth_abs_delta_mean", stats.comparableDepthPixels ?
			nlohmann::json(stats.comparableDepthAbsDeltaSum/double(stats.comparableDepthPixels)) : nlohmann::json(nullptr)},
		{"comparable_valid_depth_abs_delta_max", stats.comparableDepthPixels ?
			nlohmann::json(stats.comparableDepthMaxAbsDelta) : nlohmann::json(nullptr)},
		{"normal_delta_available", stats.normalDeltaAvailable},
		{"normal_changed_pixels", stats.normalDeltaAvailable ? nlohmann::json(stats.normalChangedPixels) : nlohmann::json(nullptr)},
		{"normal_became_valid_pixels", stats.normalDeltaAvailable ? nlohmann::json(stats.normalBecameValidPixels) : nlohmann::json(nullptr)},
		{"normal_became_invalid_pixels", stats.normalDeltaAvailable ? nlohmann::json(stats.normalBecameInvalidPixels) : nlohmann::json(nullptr)},
		{"comparable_normal_pixels", stats.normalDeltaAvailable ? nlohmann::json(stats.comparableNormalPixels) : nlohmann::json(nullptr)},
		{"normal_angle_delta_mean_degrees", stats.comparableNormalPixels ?
			nlohmann::json(stats.normalAngleDeltaSumDegrees/double(stats.comparableNormalPixels)) : nlohmann::json(nullptr)},
		{"normal_angle_delta_max_degrees", stats.comparableNormalPixels ?
			nlohmann::json(stats.normalAngleDeltaMaxDegrees) : nlohmann::json(nullptr)},
		{"confidence_delta_available", stats.confidenceDeltaAvailable},
		{"confidence_changed_pixels", stats.confidenceDeltaAvailable ? nlohmann::json(stats.confidenceChangedPixels) : nlohmann::json(nullptr)},
		{"confidence_became_positive_pixels", stats.confidenceDeltaAvailable ? nlohmann::json(stats.confidenceBecamePositivePixels) : nlohmann::json(nullptr)},
		{"confidence_became_zero_pixels", stats.confidenceDeltaAvailable ? nlohmann::json(stats.confidenceBecameZeroPixels) : nlohmann::json(nullptr)},
		{"confidence_delta_mean_all_pixels", stats.confidenceDeltaAvailable && totalPixels ?
			nlohmann::json(stats.confidenceDeltaSum/double(totalPixels)) : nlohmann::json(nullptr)},
		{"confidence_abs_delta_mean_all_pixels", stats.confidenceDeltaAvailable && totalPixels ?
			nlohmann::json(stats.confidenceAbsDeltaSum/double(totalPixels)) : nlohmann::json(nullptr)},
		{"confidence_abs_delta_max", stats.confidenceDeltaAvailable ? nlohmann::json(stats.confidenceMaxAbsDelta) : nlohmann::json(nullptr)},
	};
}

struct DMapStateSnapshot {
	DepthMap depth;
	NormalMap normal;
	ConfidenceMap confidence;
};

DMapStateSnapshot CaptureDMapState(const DepthData& depthData)
{
	DMapStateSnapshot snapshot;
	snapshot.depth = depthData.depthMap.clone();
	if (!depthData.normalMap.empty())
		snapshot.normal = depthData.normalMap.clone();
	if (!depthData.confMap.empty())
		snapshot.confidence = depthData.confMap.clone();
	return snapshot;
}

void AddDMapArtifactMap(
	nlohmann::json& maps,
	nlohmann::json& writeErrors,
	bool saved,
	const String& absolutePath,
	uint64_t declaredBytes,
	const String& signal,
	const String& relativePath,
	const char* dtype,
	const char* semantics,
	const char* stage)
{
	if (!saved) {
		writeErrors.push_back(signal.c_str());
		return;
	}
	std::error_code fileError;
	const uint64_t fileBytes((uint64_t)std::filesystem::file_size(
		std::filesystem::path(absolutePath.c_str()), fileError));
	if (fileError)
		writeErrors.push_back((signal + _T("_file_size")).c_str());
	maps.push_back({
		{"signal", signal.c_str()},
		{"path", relativePath.c_str()},
		{"dtype", dtype},
		{"semantics", semantics},
		{"quality", "exact"},
		{"algorithm_stage", stage},
		{"declared_bytes", declaredBytes},
		{"file_bytes", fileError ? 0 : fileBytes},
		{"file_size_available", !fileError},
	});
}

void SaveDMapPostprocessStageMaps(
	const String& frameDir,
	unsigned stageIndex,
	const char* stageName,
	const DepthMap& depthBefore,
	const NormalMap& normalBefore,
	const ConfidenceMap& confidenceBefore,
	const DepthMap& depthAfter,
	const NormalMap& normalAfter,
	const ConfidenceMap& confidenceAfter,
	nlohmann::json& maps,
	nlohmann::json& writeErrors)
{
	const String relativeDir(_T("postprocess_filters/"));
	const String mapsDir(frameDir + relativeDir);
	Util::ensureFolder(mapsDir);
	const String prefix(String::FormatString(_T("%02u_%s_"), stageIndex, stageName));
	auto saveDepth = [&](const DepthMap& map, const char* suffix, const char* semantics) {
		const String fileName(prefix + suffix + _T(".pfm"));
		const String absolutePath(mapsDir + fileName);
		AddDMapArtifactMap(maps, writeErrors, !map.empty() && map.Save(absolutePath), absolutePath,
			DMapSaturatingMul((uint64_t)map.total(), sizeof(Depth)),
			prefix + suffix, relativeDir + fileName, "float32", semantics, stageName);
	};
	saveDepth(depthBefore, "depth_before", "depth immediately before this sequential postprocess stage");
	saveDepth(depthAfter, "depth_after", "depth immediately after this sequential postprocess stage");

	DepthMap depthDelta(depthBefore.size());
	Image8U validityTransition(depthBefore.size());
	for (int r=0; r<depthBefore.rows; ++r) {
		for (int c=0; c<depthBefore.cols; ++c) {
			const Depth inputDepth(depthBefore(r,c));
			const Depth outputDepth(depthAfter(r,c));
			depthDelta(r,c) = outputDepth-inputDepth;
			const bool inputValid(inputDepth > 0), outputValid(outputDepth > 0);
			validityTransition(r,c) = inputValid ? (outputValid ? 1 : 2) : (outputValid ? 3 : 0);
		}
	}
	saveDepth(depthDelta, "depth_delta", "signed depth_after-depth_before; includes additions and removals");
	{
		const String fileName(prefix + _T("validity_transition.png"));
		const String absolutePath(mapsDir + fileName);
		AddDMapArtifactMap(maps, writeErrors, validityTransition.Save(absolutePath), absolutePath,
			(uint64_t)validityTransition.total(),
			prefix + _T("validity_transition"), relativeDir + fileName, "uint8",
			"depth validity transition codes: 0 invalid-to-invalid, 1 valid-to-valid, 2 removed, 3 added", stageName);
	}

	if (!confidenceBefore.empty() && !confidenceAfter.empty() &&
		confidenceBefore.size() == depthBefore.size() && confidenceAfter.size() == depthBefore.size())
	{
		auto saveConfidence = [&](const ConfidenceMap& map, const char* suffix, const char* semantics) {
			const String fileName(prefix + suffix + _T(".pfm"));
			const String absolutePath(mapsDir + fileName);
			AddDMapArtifactMap(maps, writeErrors, map.Save(absolutePath), absolutePath,
				DMapSaturatingMul((uint64_t)map.total(), sizeof(float)),
				prefix + suffix, relativeDir + fileName, "float32", semantics, stageName);
		};
		saveConfidence(confidenceBefore, "confidence_before", "confidence immediately before this sequential postprocess stage");
		saveConfidence(confidenceAfter, "confidence_after", "confidence immediately after this sequential postprocess stage");
		ConfidenceMap confidenceDelta(confidenceBefore.size());
		Image8U confidenceTransition(confidenceBefore.size());
		for (int r=0; r<confidenceBefore.rows; ++r) {
			for (int c=0; c<confidenceBefore.cols; ++c) {
				const float inputConfidence(confidenceBefore(r,c));
				const float outputConfidence(confidenceAfter(r,c));
				confidenceDelta(r,c) = outputConfidence-inputConfidence;
				const bool inputPositive(inputConfidence > 0), outputPositive(outputConfidence > 0);
				confidenceTransition(r,c) = inputPositive ? (outputPositive ? 1 : 2) : (outputPositive ? 3 : 0);
			}
		}
		saveConfidence(confidenceDelta, "confidence_delta", "signed confidence_after-confidence_before");
		const String fileName(prefix + _T("confidence_transition.png"));
		const String absolutePath(mapsDir + fileName);
		AddDMapArtifactMap(maps, writeErrors, confidenceTransition.Save(absolutePath), absolutePath,
			(uint64_t)confidenceTransition.total(),
			prefix + _T("confidence_transition"), relativeDir + fileName, "uint8",
			"confidence positivity transition codes: 0 zero-to-zero, 1 positive-to-positive, 2 became zero, 3 became positive", stageName);
	}

	if (!normalBefore.empty() && !normalAfter.empty() &&
		normalBefore.size() == depthBefore.size() && normalAfter.size() == depthBefore.size())
	{
		auto saveNormal = [&](const NormalMap& map, const char* suffix, const char* semantics) {
			const String fileName(prefix + suffix + _T(".pfm"));
			const String absolutePath(mapsDir + fileName);
			AddDMapArtifactMap(maps, writeErrors, map.Save(absolutePath), absolutePath,
				DMapSaturatingMul((uint64_t)map.total(), 3u*sizeof(float)),
				prefix + suffix, relativeDir + fileName, "float32x3", semantics, stageName);
		};
		saveNormal(normalBefore, "normal_before",
			"camera-space normal immediately before this sequential postprocess stage; zero encodes invalid");
		saveNormal(normalAfter, "normal_after",
			"camera-space normal immediately after this sequential postprocess stage; zero encodes invalid");
		DepthMap normalAngleDelta(normalBefore.size());
		for (int r=0; r<normalBefore.rows; ++r) {
			for (int c=0; c<normalBefore.cols; ++c) {
				const Normal& inputNormal(normalBefore(r,c));
				const Normal& outputNormal(normalAfter(r,c));
				if (DMapNormalIsValid(inputNormal) && DMapNormalIsValid(outputNormal)) {
					const float cosine(CLAMP(inputNormal.dot(outputNormal) /
						SQRT(DMapNormalSquaredNorm(inputNormal)*DMapNormalSquaredNorm(outputNormal)), -1.f, 1.f));
					normalAngleDelta(r,c) = R2D(ACOS(cosine));
				} else {
					normalAngleDelta(r,c) = std::numeric_limits<float>::quiet_NaN();
				}
			}
		}
		saveDepth(normalAngleDelta, "normal_angle_delta_degrees",
			"angular difference in degrees between valid before/after normals; NaN where either normal is invalid");
	}
}

struct DMapPostprocessStageRecord {
	unsigned index{0};
	String name;
	bool enabled{false};
	bool executed{false};
	bool success{false};
	nlohmann::json parameters;
	DMapPostprocessStageStats stats;
};

class DMapPostprocessObservation {
public:
	DMapPostprocessObservation(
		const Scene& scene,
		IIndex idxImage,
		int geometricIteration,
		const DepthData& depthData,
		const DMapFilterResourcePlan& resourcePlan)
		: reference(scene.images[idxImage]),
		  referenceSceneIndex(idxImage),
		  geometricIteration(geometricIteration),
		  frameDir(DMapInstrumentFrameDir(reference, geometricIteration)),
		  mapsRequested(resourcePlan.mapsRequested),
		  writeMaps(resourcePlan.mapsAvailable),
		  mapsUnavailableReason(resourcePlan.mapsRequested && !resourcePlan.mapsAvailable ? resourcePlan.reason : String()),
		  input(SummarizeDMapState(depthData.depthMap, depthData.normalMap, depthData.confMap)),
		  maps(nlohmann::json::array()),
		  writeErrors(nlohmann::json::array())
	{}

	void Record(
		unsigned index,
		const char* name,
		bool enabled,
		bool executed,
		bool success,
		const nlohmann::json& parameters,
		const DepthMap& depthBefore,
		const NormalMap& normalBefore,
		const ConfidenceMap& confidenceBefore,
		const DepthMap& depthAfter,
		const NormalMap& normalAfter,
		const ConfidenceMap& confidenceAfter)
	{
		DMapPostprocessStageRecord record;
		record.index = index;
		record.name = name;
		record.enabled = enabled;
		record.executed = executed;
		record.success = success;
		record.parameters = parameters;
		record.stats = ComputeDMapPostprocessStageStats(
			depthBefore, normalBefore, confidenceBefore, depthAfter, normalAfter, confidenceAfter);
		records.emplace_back(std::move(record));
		if (writeMaps && executed)
			SaveDMapPostprocessStageMaps(frameDir, index, name, depthBefore, normalBefore, confidenceBefore,
				depthAfter, normalAfter, confidenceAfter, maps, writeErrors);
	}

	DMapArtifactUsage Write(const DepthData& depthData) const
	{
		nlohmann::json stages(nlohmann::json::array());
		bool stagesSucceeded(true);
		const DMapPostprocessStageRecord* terminalExecutedStage(NULL);
		for (const DMapPostprocessStageRecord& record: records) {
			if (record.executed)
				terminalExecutedStage = &record;
		}
		for (const DMapPostprocessStageRecord& record: records) {
			if (record.executed && !record.success)
				stagesSucceeded = false;
			stages.push_back({
				{"stage_index", record.index},
				{"name", record.name.c_str()},
				{"enabled", record.enabled},
				{"executed", record.executed},
				{"success", record.executed ? nlohmann::json(record.success) : nlohmann::json(nullptr)},
				{"measurement_basis", record.executed ? "exact sequential before/after state" : "exact identity because stage was disabled"},
				{"parameters", record.parameters},
				{"metrics", DMapPostprocessStageStatsJson(record.stats)},
			});
		}
		const DMapStateSummary output(SummarizeDMapState(
			depthData.depthMap, depthData.normalMap, depthData.confMap));
		nlohmann::json pipelineOutputMapSignals(nullptr);
		if (writeMaps && terminalExecutedStage) {
			const String prefix(String::FormatString(_T("%02u_%s_"),
				terminalExecutedStage->index, terminalExecutedStage->name.c_str()));
			pipelineOutputMapSignals = {
				{"stage_index", terminalExecutedStage->index},
				{"stage_name", terminalExecutedStage->name.c_str()},
				{"depth", (prefix + _T("depth_after")).c_str()},
				{"normal", terminalExecutedStage->stats.normalDeltaAvailable ?
					nlohmann::json((prefix + _T("normal_after")).c_str()) : nlohmann::json(nullptr)},
				{"confidence", terminalExecutedStage->stats.confidenceDeltaAvailable ?
					nlohmann::json((prefix + _T("confidence_after")).c_str()) : nlohmann::json(nullptr)},
			};
		}
		nlohmann::json artifact = {
			{"schema_name", "openmvs.dmap.postprocess_filters"},
			{"schema_version", 2},
			{"algorithm_stage", "depth_map_optional_postprocess"},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"reference_scene_image_index", referenceSceneIndex},
			{"reference_image_id", reference.ID},
			{"reference_image_name", reference.name.c_str()},
			{"optimize_flags", OPTDENSE::nOptimize},
			{"pipeline_input", DMapStateSummaryJson(input)},
			{"pipeline_output", DMapStateSummaryJson(output)},
			{"pipeline_output_map_signals", pipelineOutputMapSignals},
			{"stage_order", nlohmann::json::array({"remove_speckles", "fill_gaps"})},
			{"stages", stages},
			{"resource_plan", {
				{"path", "filter_resource_plan.json"},
				{"path_base", "artifact_directory"},
				{"schema_name", "openmvs.dmap.filter_resource_plan"},
				{"schema_version", 2},
			}},
			{"maps_requested", mapsRequested},
			{"maps_enabled", writeMaps},
			{"maps_unavailable_reason", mapsUnavailableReason.empty() ? nlohmann::json(nullptr) : nlohmann::json(mapsUnavailableReason.c_str())},
			{"maps", maps},
			{"write_errors", writeErrors},
			{"complete", writeErrors.empty() && stagesSucceeded},
		};
		const String jsonFile(frameDir + _T("postprocess_filters.json"));
		if (!WriteDMapJson(jsonFile, artifact))
			VERBOSE("warning: failed to write depth-map postprocess instrumentation: %s", jsonFile.c_str());

		std::ostringstream csv;
		csv << std::setprecision(9);
		csv << "estimation_stage,geometric_iteration,reference_scene_image_index,reference_image_id,stage_index,stage_name,"
			"enabled,executed,success,total_pixels,input_valid_depth_pixels,output_valid_depth_pixels,removed_pixels,added_pixels,"
			"depth_changed_pixels,depth_abs_delta_mean_all_pixels,depth_abs_delta_max,normal_delta_available,"
			"normal_changed_pixels,normal_became_valid_pixels,normal_became_invalid_pixels,comparable_normal_pixels,"
			"normal_angle_delta_mean_degrees,normal_angle_delta_max_degrees,confidence_delta_available,"
			"confidence_changed_pixels,confidence_abs_delta_mean_all_pixels,confidence_abs_delta_max\n";
		for (const DMapPostprocessStageRecord& record: records) {
			const DMapPostprocessStageStats& stats(record.stats);
			csv << DMapEstimationStageName(geometricIteration) << ',';
			if (geometricIteration >= 0)
				csv << geometricIteration;
			csv << ',' << referenceSceneIndex << ',' << reference.ID << ',' << record.index << ',' << record.name.c_str() << ','
				<< (record.enabled ? 1 : 0) << ',' << (record.executed ? 1 : 0) << ',';
			if (record.executed)
				csv << (record.success ? 1 : 0);
			csv << ',' << stats.input.totalPixels << ',' << stats.input.validPixels << ',' << stats.output.validPixels << ','
				<< stats.removedPixels << ',' << stats.addedPixels << ',' << stats.depthChangedPixels << ','
				<< (stats.input.totalPixels ? stats.depthAbsDeltaSum/double(stats.input.totalPixels) : 0.0) << ','
				<< stats.depthMaxAbsDelta << ',' << (stats.normalDeltaAvailable ? 1 : 0) << ',';
			if (stats.normalDeltaAvailable) {
				csv << stats.normalChangedPixels << ',' << stats.normalBecameValidPixels << ','
					<< stats.normalBecameInvalidPixels << ',' << stats.comparableNormalPixels << ',';
				if (stats.comparableNormalPixels) {
					csv << stats.normalAngleDeltaSumDegrees/double(stats.comparableNormalPixels) << ','
						<< stats.normalAngleDeltaMaxDegrees;
				} else {
					csv << ',';
				}
			} else {
				csv << ",,,,,";
			}
			csv << ',' << (stats.confidenceDeltaAvailable ? 1 : 0) << ',';
			if (stats.confidenceDeltaAvailable) {
				csv << stats.confidenceChangedPixels << ','
					<< (stats.input.totalPixels ? stats.confidenceAbsDeltaSum/double(stats.input.totalPixels) : 0.0) << ','
					<< stats.confidenceMaxAbsDelta;
			} else {
				csv << ",,";
			}
			csv << '\n';
		}
		const String csvFile(frameDir + _T("postprocess_filters.csv"));
		if (!WriteDMapText(csvFile, csv.str()))
			VERBOSE("warning: failed to write depth-map postprocess CSV instrumentation: %s", csvFile.c_str());
		return DMapArtifactUsageFromMaps(maps);
	}

private:
	const Image& reference;
	IIndex referenceSceneIndex;
	int geometricIteration;
	String frameDir;
	bool mapsRequested;
	bool writeMaps;
	String mapsUnavailableReason;
	DMapStateSummary input;
	std::vector<DMapPostprocessStageRecord> records;
	nlohmann::json maps;
	nlohmann::json writeErrors;
};

void WriteUnavailableDMapPostprocessArtifacts(
	const Scene& scene,
	IIndex idxImage,
	int geometricIteration,
	const char* reason,
	const DMapFilterResourcePlan& resourcePlan)
{
	const Image& reference(scene.images[idxImage]);
	const String frameDir(DMapInstrumentFrameDir(reference, geometricIteration));
	const String jsonFile(frameDir + _T("postprocess_filters.json"));
	if (File::access(jsonFile))
		return;
	auto unavailableStage = [&](unsigned index, const char* name, bool enabled, const nlohmann::json& parameters) {
		return nlohmann::json({
			{"stage_index", index},
			{"name", name},
			{"enabled", enabled},
			{"executed", false},
			{"success", nullptr},
			{"measurement_basis", "unavailable because no in-memory final state was observed"},
			{"unavailable_reason", reason},
			{"parameters", parameters},
			{"metrics", nullptr},
		});
	};
	nlohmann::json artifact = {
		{"schema_name", "openmvs.dmap.postprocess_filters"},
		{"schema_version", 2},
		{"algorithm_stage", "depth_map_optional_postprocess"},
		{"estimation_stage", DMapEstimationStageName(geometricIteration)},
		{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
		{"reference_scene_image_index", idxImage},
		{"reference_image_id", reference.ID},
		{"reference_image_name", reference.name.c_str()},
		{"optimize_flags", OPTDENSE::nOptimize},
		{"pipeline_input", nullptr},
		{"pipeline_output", nullptr},
		{"pipeline_output_map_signals", nullptr},
		{"stage_order", nlohmann::json::array({"remove_speckles", "fill_gaps"})},
		{"stages", nlohmann::json::array({
			unavailableStage(0, "remove_speckles",
				(OPTDENSE::nOptimize & OPTDENSE::REMOVE_SPECKLES) != 0, {
				{"speckle_size", OPTDENSE::nSpeckleSize},
				{"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold*0.7f},
				{"connectivity", 4},
			}),
			unavailableStage(1, "fill_gaps",
				(OPTDENSE::nOptimize & OPTDENSE::FILL_GAPS) != 0, {
				{"maximum_gap_pixels", OPTDENSE::nIpolGapSize},
				{"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold*2.5f},
				{"passes", nlohmann::json::array({"rows", "columns"})},
			}),
		})},
		{"resource_plan", {
			{"path", "filter_resource_plan.json"},
			{"path_base", "artifact_directory"},
			{"schema_name", "openmvs.dmap.filter_resource_plan"},
			{"schema_version", 2},
		}},
		{"maps_requested", resourcePlan.mapsRequested},
		{"maps_enabled", false},
		{"maps_unavailable_reason", resourcePlan.mapsRequested ? nlohmann::json(resourcePlan.reason.c_str()) : nlohmann::json(nullptr)},
		{"maps", nlohmann::json::array()},
		{"write_errors", nlohmann::json::array()},
		{"complete", false},
		{"unavailable_signals", nlohmann::json::array({"input_state", "output_state", "per_pixel_depth_deltas", "per_pixel_normal_deltas", "per_pixel_confidence_deltas"})},
	};
	if (!WriteDMapJson(jsonFile, artifact))
		VERBOSE("warning: failed to write unavailable depth-map postprocess instrumentation: %s", jsonFile.c_str());
	std::ostringstream csv;
	csv << "estimation_stage,geometric_iteration,reference_scene_image_index,reference_image_id,stage_index,stage_name,"
		"enabled,executed,success,state_metrics_available,unavailable_reason\n";
	for (unsigned stageIndex=0; stageIndex<2; ++stageIndex) {
		csv << DMapEstimationStageName(geometricIteration) << ',';
		if (geometricIteration >= 0)
			csv << geometricIteration;
		const bool enabled(stageIndex == 0 ?
			(OPTDENSE::nOptimize & OPTDENSE::REMOVE_SPECKLES) != 0 :
			(OPTDENSE::nOptimize & OPTDENSE::FILL_GAPS) != 0);
		csv << ',' << idxImage << ',' << reference.ID << ',' << stageIndex << ','
			<< (stageIndex == 0 ? "remove_speckles" : "fill_gaps") << ','
			<< (enabled ? 1 : 0) << ",0,,0," << reason << '\n';
	}
	const String csvFile(frameDir + _T("postprocess_filters.csv"));
	if (!WriteDMapText(csvFile, csv.str()))
		VERBOSE("warning: failed to write unavailable depth-map postprocess CSV instrumentation: %s", csvFile.c_str());
}

struct DMapConfidenceDeltaStats {
	bool available{false};
	uint64_t totalPixels{0};
	uint64_t validDepthPixels{0};
	uint64_t inputPositivePixels{0};
	uint64_t outputPositivePixels{0};
	uint64_t changedPixels{0};
	uint64_t becamePositivePixels{0};
	uint64_t becameZeroPixels{0};
	double deltaSum{0};
	double absDeltaSum{0};
	double maxAbsDelta{0};
	uint64_t changedValidDepthPixels{0};
	double validDepthAbsDeltaSum{0};
	double validDepthMaxAbsDelta{0};
};

DMapConfidenceDeltaStats ComputeDMapConfidenceDeltaStats(
	const DepthMap& depthMap,
	const ConfidenceMap& input,
	const ConfidenceMap& output)
{
	DMapConfidenceDeltaStats stats;
	if (depthMap.empty() || input.empty() || output.empty() ||
		input.size() != depthMap.size() || output.size() != depthMap.size())
		return stats;
	stats.available = true;
	stats.totalPixels = (uint64_t)depthMap.total();
	for (int r=0; r<depthMap.rows; ++r) {
		for (int c=0; c<depthMap.cols; ++c) {
			const bool depthValid(depthMap(r,c) > 0);
			const float inputConfidence(input(r,c));
			const float outputConfidence(output(r,c));
			const bool inputPositive(inputConfidence > 0), outputPositive(outputConfidence > 0);
			if (depthValid)
				++stats.validDepthPixels;
			if (inputPositive)
				++stats.inputPositivePixels;
			if (outputPositive)
				++stats.outputPositivePixels;
			if (!inputPositive && outputPositive)
				++stats.becamePositivePixels;
			if (inputPositive && !outputPositive)
				++stats.becameZeroPixels;
			const double delta(double(outputConfidence)-double(inputConfidence));
			const double absDelta(ABS(delta));
			if (inputConfidence != outputConfidence) {
				++stats.changedPixels;
				if (depthValid)
					++stats.changedValidDepthPixels;
			}
			stats.deltaSum += delta;
			stats.absDeltaSum += absDelta;
			stats.maxAbsDelta = MAXF(stats.maxAbsDelta, absDelta);
			if (depthValid) {
				stats.validDepthAbsDeltaSum += absDelta;
				stats.validDepthMaxAbsDelta = MAXF(stats.validDepthMaxAbsDelta, absDelta);
			}
		}
	}
	return stats;
}

nlohmann::json DMapConfidenceDeltaStatsJson(const DMapConfidenceDeltaStats& stats)
{
	if (!stats.available)
		return nullptr;
	return {
		{"total_pixels", stats.totalPixels},
		{"valid_depth_pixels", stats.validDepthPixels},
		{"input_positive_confidence_pixels", stats.inputPositivePixels},
		{"output_positive_confidence_pixels", stats.outputPositivePixels},
		{"output_positive_confidence_ratio", stats.totalPixels ? double(stats.outputPositivePixels)/double(stats.totalPixels) : 0.0},
		{"changed_pixels", stats.changedPixels},
		{"became_positive_pixels", stats.becamePositivePixels},
		{"became_zero_pixels", stats.becameZeroPixels},
		{"delta_mean_all_pixels", stats.totalPixels ? stats.deltaSum/double(stats.totalPixels) : 0.0},
		{"abs_delta_mean_all_pixels", stats.totalPixels ? stats.absDeltaSum/double(stats.totalPixels) : 0.0},
		{"abs_delta_max", stats.maxAbsDelta},
		{"changed_valid_depth_pixels", stats.changedValidDepthPixels},
		{"abs_delta_mean_valid_depth_pixels", stats.validDepthPixels ?
			nlohmann::json(stats.validDepthAbsDeltaSum/double(stats.validDepthPixels)) : nlohmann::json(nullptr)},
		{"abs_delta_max_valid_depth_pixels", stats.validDepthPixels ?
			nlohmann::json(stats.validDepthMaxAbsDelta) : nlohmann::json(nullptr)},
	};
}

void SaveDMapConfidenceAdjustmentMaps(
	const String& frameDir,
	const DepthMap& depthMap,
	const ConfidenceMap& input,
	const ConfidenceMap* fast,
	const ConfidenceMap* full,
	const ConfidenceMap* final,
	nlohmann::json& maps,
	nlohmann::json& writeErrors)
{
	const String relativeDir(_T("confidence_adjustment/"));
	const String mapsDir(frameDir + relativeDir);
	Util::ensureFolder(mapsDir);
	const String inputPath(mapsDir + _T("confidence_input.pfm"));
	AddDMapArtifactMap(maps, writeErrors, input.Save(inputPath), inputPath,
		DMapSaturatingMul((uint64_t)input.total(), sizeof(float)),
		_T("confidence_input"), relativeDir + _T("confidence_input.pfm"), "float32",
		"production confidence before optional cross-depth-map adjustment", "confidence_adjustment");
	Image8U depthValidity(depthMap.size());
	for (int r=0; r<depthMap.rows; ++r)
		for (int c=0; c<depthMap.cols; ++c)
			depthValidity(r,c) = depthMap(r,c) > 0 ? 1 : 0;
	const String validityPath(mapsDir + _T("depth_validity.png"));
	AddDMapArtifactMap(maps, writeErrors, depthValidity.Save(validityPath), validityPath,
		(uint64_t)depthValidity.total(),
		_T("confidence_adjustment_depth_validity"), relativeDir + _T("depth_validity.png"), "uint8",
		"reference depth validity: 0 invalid, 1 valid", "confidence_adjustment");
	auto saveOutput = [&](const char* name, const ConfidenceMap* output, const char* semantics) {
		if (!output || output->empty())
			return;
		const String outputName(String::FormatString(_T("confidence_%s.pfm"), name));
		const String outputPath(mapsDir + outputName);
		AddDMapArtifactMap(maps, writeErrors, output->Save(outputPath), outputPath,
			DMapSaturatingMul((uint64_t)output->total(), sizeof(float)),
			String::FormatString(_T("confidence_%s"), name), relativeDir + outputName, "float32",
			semantics, "confidence_adjustment");
		ConfidenceMap delta(input.size());
		Image8U transition(input.size());
		for (int r=0; r<input.rows; ++r) {
			for (int c=0; c<input.cols; ++c) {
				const float inputConfidence(input(r,c));
				const float outputConfidence((*output)(r,c));
				delta(r,c) = outputConfidence-inputConfidence;
				const bool inputPositive(inputConfidence > 0), outputPositive(outputConfidence > 0);
				transition(r,c) = inputPositive ? (outputPositive ? 1 : 2) : (outputPositive ? 3 : 0);
			}
		}
		const String deltaName(String::FormatString(_T("confidence_%s_delta.pfm"), name));
		const String deltaPath(mapsDir + deltaName);
		AddDMapArtifactMap(maps, writeErrors, delta.Save(deltaPath), deltaPath,
			DMapSaturatingMul((uint64_t)delta.total(), sizeof(float)),
			String::FormatString(_T("confidence_%s_delta"), name), relativeDir + deltaName, "float32",
			"signed adjusted-confidence minus production input confidence", "confidence_adjustment");
		const String transitionName(String::FormatString(_T("confidence_%s_transition.png"), name));
		const String transitionPath(mapsDir + transitionName);
		AddDMapArtifactMap(maps, writeErrors, transition.Save(transitionPath), transitionPath,
			(uint64_t)transition.total(),
			String::FormatString(_T("confidence_%s_transition"), name), relativeDir + transitionName, "uint8",
			"confidence positivity transition: 0 zero-to-zero, 1 positive-to-positive, 2 became zero, 3 became positive",
			"confidence_adjustment");
	};
	saveOutput("fast", fast, "output of ADJUST_CONFIDENCE_FAST before final combination");
	saveOutput("full", full, "output of ADJUST_CONFIDENCE before final combination");
	saveOutput("final", final, "final production confidence after configured adjustment outputs are combined");
}

DMapArtifactUsage WriteDMapConfidenceAdjustmentArtifacts(
	const Scene& scene,
	IIndex idxImage,
	int geometricIteration,
	const IIndexArr* idxNeighbors,
	const DepthMap* depthMap,
	const ConfidenceMap* input,
	const ConfidenceMap* fast,
	const ConfidenceMap* full,
	const ConfidenceMap* final,
	const DMapFilterResourcePlan& resourcePlan,
	const char* status,
	const char* unavailableReason=NULL)
{
	const Image& reference(scene.images[idxImage]);
	const String frameDir(DMapInstrumentFrameDir(reference, geometricIteration));
	const bool fastEnabled((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE_FAST) != 0);
	const bool fullEnabled((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE) != 0);
	const bool disabledStatus(String(status) == _T("disabled"));
	const bool inputAvailable(depthMap && input && !depthMap->empty() && !input->empty() && input->size() == depthMap->size());
	const auto outputAvailable = [&](const ConfidenceMap* output) {
		return inputAvailable && output && !output->empty() && output->size() == input->size();
	};
	const bool fastOutputAvailable(outputAvailable(fast));
	const bool fullOutputAvailable(outputAvailable(full));
	const bool finalOutputAvailable(outputAvailable(final));
	const bool configuredOutputsComplete(
		(!fastEnabled || fastOutputAvailable) && (!fullEnabled || fullOutputAvailable));
	nlohmann::json neighbors(nlohmann::json::array());
	if (idxNeighbors) {
		for (IIndex neighborSceneIndex: *idxNeighbors) {
			const Image* neighbor(neighborSceneIndex < scene.images.size() ? &scene.images[neighborSceneIndex] : NULL);
			neighbors.push_back({
				{"scene_image_index", neighborSceneIndex},
				{"image_id", neighbor ? nlohmann::json(neighbor->ID) : nlohmann::json(nullptr)},
				{"image_name", neighbor ? nlohmann::json(neighbor->name.c_str()) : nlohmann::json(nullptr)},
			});
		}
	}
	auto method = [&](const char* name, bool enabled, const ConfidenceMap* output, const char* basis) {
		const bool available(inputAvailable && output && !output->empty() && output->size() == input->size());
		return nlohmann::json({
			{"name", name},
			{"enabled", enabled},
			{"executed", enabled && !disabledStatus},
			{"output_available", available},
			{"quality", available ? "exact" : "unavailable"},
			{"basis", basis},
			{"unavailable_reason", available ? nlohmann::json(nullptr) : nlohmann::json(
				!enabled ? "method disabled by nOptimize" : (unavailableReason ? unavailableReason : "output map unavailable"))},
			{"metrics", available ? DMapConfidenceDeltaStatsJson(ComputeDMapConfidenceDeltaStats(*depthMap, *input, *output)) : nlohmann::json(nullptr)},
		});
	};
	const bool finalEnabled(fastEnabled || fullEnabled);
	const char* combination(
		fast && !fast->empty() && full && !full->empty() ? "maximum_if_both_positive_else_zero" :
		(fast && !fast->empty() ? "fast_only" : (full && !full->empty() ? "full_only" : "unavailable")));
	nlohmann::json methods(nlohmann::json::array());
	methods.push_back(method("adjust_confidence_fast", fastEnabled, fast,
		"depth-similarity confidence blended with reference/neighbor photometric confidence"));
	methods.push_back(method("adjust_confidence", fullEnabled, full,
		"projected neighbor-depth confidence fusion with occlusion and free-space penalties"));
	methods.push_back(method("final_combined_confidence", finalEnabled, final, combination));
	nlohmann::json maps(nlohmann::json::array()), writeErrors(nlohmann::json::array());
	if (resourcePlan.mapsAvailable && inputAvailable)
		SaveDMapConfidenceAdjustmentMaps(frameDir, *depthMap, *input, fast, full, final, maps, writeErrors);
	nlohmann::json artifact = {
		{"schema_name", "openmvs.dmap.confidence_adjustment"},
		{"schema_version", 1},
		{"algorithm_stage", "depth_map_optional_confidence_adjustment"},
		{"estimation_stage", DMapEstimationStageName(geometricIteration)},
		{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
		{"reference_scene_image_index", idxImage},
		{"reference_image_id", reference.ID},
		{"reference_image_name", reference.name.c_str()},
		{"status", status},
		{"optimize_flags", OPTDENSE::nOptimize},
		{"neighbor_limit", 8},
		{"neighbors", neighbors},
		{"parameters", {
			{"fast_depth_similarity_threshold", 0.01},
			{"fast_similarity_weight", 0.7},
			{"fast_photometric_weight", 0.3},
			{"full_min_views", OPTDENSE::nMinViewsFilter},
			{"full_min_views_adjust", OPTDENSE::nMinViewsFilterAdjust},
			{"full_depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold*1.2f},
		}},
		{"input_available", inputAvailable},
		{"configured_outputs_complete", configuredOutputsComplete},
		{"depth_validity_unchanged", inputAvailable ? nlohmann::json(true) : nlohmann::json(nullptr)},
		{"final_combination", combination},
		{"methods", methods},
		{"resource_plan", {
			{"path", "filter_resource_plan.json"},
			{"path_base", "artifact_directory"},
			{"schema_name", "openmvs.dmap.filter_resource_plan"},
			{"schema_version", 2},
		}},
		{"maps_requested", resourcePlan.mapsRequested},
		{"maps_enabled", resourcePlan.mapsAvailable},
		{"maps_unavailable_reason", resourcePlan.mapsRequested && !resourcePlan.mapsAvailable ?
			nlohmann::json(resourcePlan.reason.c_str()) : nlohmann::json(nullptr)},
		{"maps", maps},
		{"write_errors", writeErrors},
		{"complete", writeErrors.empty() && (disabledStatus ||
			(inputAvailable && finalOutputAvailable && configuredOutputsComplete))},
	};
	if (unavailableReason)
		artifact["unavailable_reason"] = unavailableReason;
	const String jsonFile(frameDir + _T("confidence_adjustment.json"));
	if (!WriteDMapJson(jsonFile, artifact))
		VERBOSE("warning: failed to write depth-map confidence-adjustment instrumentation: %s", jsonFile.c_str());

	std::ostringstream csv;
	csv << std::setprecision(9);
	csv << "estimation_stage,geometric_iteration,reference_scene_image_index,reference_image_id,method,enabled,executed,"
		"output_available,total_pixels,valid_depth_pixels,input_positive_confidence_pixels,output_positive_confidence_pixels,"
		"changed_pixels,became_positive_pixels,became_zero_pixels,abs_delta_mean_all_pixels,abs_delta_max,unavailable_reason\n";
	auto writeMethodRow = [&](const char* name, bool enabled, const ConfidenceMap* output) {
		const DMapConfidenceDeltaStats stats(inputAvailable && output ?
			ComputeDMapConfidenceDeltaStats(*depthMap, *input, *output) : DMapConfidenceDeltaStats());
		csv << DMapEstimationStageName(geometricIteration) << ',';
		if (geometricIteration >= 0)
			csv << geometricIteration;
		csv << ',' << idxImage << ',' << reference.ID << ',' << name << ',' << (enabled ? 1 : 0) << ','
			<< (enabled && !disabledStatus ? 1 : 0) << ',' << (stats.available ? 1 : 0) << ',';
		if (stats.available) {
			csv << stats.totalPixels << ',' << stats.validDepthPixels << ',' << stats.inputPositivePixels << ','
				<< stats.outputPositivePixels << ',' << stats.changedPixels << ',' << stats.becamePositivePixels << ','
				<< stats.becameZeroPixels << ','
				<< (stats.totalPixels ? stats.absDeltaSum/double(stats.totalPixels) : 0.0) << ',' << stats.maxAbsDelta << ',';
		} else {
			csv << ",,,,,,,,,";
			csv << (!enabled ? "method disabled by nOptimize" : (unavailableReason ? unavailableReason : "output map unavailable"));
		}
		csv << '\n';
	};
	writeMethodRow("adjust_confidence_fast", fastEnabled, fast);
	writeMethodRow("adjust_confidence", fullEnabled, full);
	writeMethodRow("final_combined_confidence", finalEnabled, final);
	const String csvFile(frameDir + _T("confidence_adjustment.csv"));
	if (!WriteDMapText(csvFile, csv.str()))
		VERBOSE("warning: failed to write depth-map confidence-adjustment CSV instrumentation: %s", csvFile.c_str());
	return DMapArtifactUsageFromMaps(maps);
}

void WriteCPUViewCandidateArtifacts(
	const Scene& scene,
	IIndex idxImage,
	const Scene::NeighborViewSelectionObservation& observation)
{
	const Image& reference(scene.images[idxImage]);
	const String frameDir(DMapInstrumentFrameDir(reference));
	const bool precomputed(
		observation.source == Scene::NeighborViewSelectionObservation::SOURCE_PRECOMPUTED_SCENE_NEIGHBORS);
	nlohmann::json candidates(nlohmann::json::array());
	unsigned rankedCount(0), retainedCount(0), rejectedInitialCount(0), rejectedFilterCount(0);
	for (const Scene::NeighborViewCandidateObservation& candidate: observation.candidates) {
		const bool candidateKnown(candidate.ID < scene.images.size());
		const Image* candidateImage(candidateKnown ? &scene.images[candidate.ID] : NULL);
		nlohmann::json scoreComponents;
		scoreComponents["available"] = candidate.scoreComponentsAvailable;
		scoreComponents["quality"] = candidate.scoreComponentsAvailable ? "exact_aggregate" : "unavailable";
		scoreComponents["angle_weight_sum"] = candidate.scoreComponentsAvailable ? nlohmann::json(candidate.angleWeightSum) : nlohmann::json(nullptr);
		scoreComponents["clipped_angle_weight_sum"] = candidate.scoreComponentsAvailable ? nlohmann::json(candidate.clippedAngleWeightSum) : nlohmann::json(nullptr);
		scoreComponents["scale_weight_sum"] = candidate.scoreComponentsAvailable ? nlohmann::json(candidate.scaleWeightSum) : nlohmann::json(nullptr);
		scoreComponents["roi_weight_sum"] = candidate.scoreComponentsAvailable ? nlohmann::json(candidate.roiWeightSum) : nlohmann::json(nullptr);
		scoreComponents["score_before_area"] = candidate.scoreComponentsAvailable ? nlohmann::json(candidate.scoreBeforeArea) : nlohmann::json(nullptr);
		scoreComponents["unavailable_reason"] = candidate.scoreComponentsAvailable ? nlohmann::json(nullptr) :
			nlohmann::json(precomputed ? "precomputed ViewScore stores only the aggregate final score" : "candidate received no valid sparse-point contribution");

		nlohmann::json row = {
			{"candidate_scene_image_index", candidate.ID},
			{"candidate_image_id", candidateImage ? nlohmann::json(candidateImage->ID) : nlohmann::json(nullptr)},
			{"candidate_image_name", candidateImage ? nlohmann::json(candidateImage->name.c_str()) : nlohmann::json(nullptr)},
			{"image_valid", candidate.imageValid},
			{"initial_decision", InitialViewDecisionName(candidate.initialDecision)},
			{"shared_points", candidate.sharedPoints},
			{"projected_points", precomputed ? nlohmann::json(nullptr) : nlohmann::json(candidate.projectedPoints)},
			{"average_scale", candidate.avgScale},
			{"average_angle_radians", candidate.avgAngle},
			{"covered_area_ratio", candidate.area},
			{"area_factor", candidate.areaFactor},
			{"ranking_score", candidate.score},
			{"score_components", scoreComponents},
			{"raw_rank_zero_based", candidate.rawRank >= 0 ? nlohmann::json(candidate.rawRank) : nlohmann::json(nullptr)},
			{"filter_input_rank_zero_based", candidate.filterInputRank >= 0 ? nlohmann::json(candidate.filterInputRank) : nlohmann::json(nullptr)},
			{"filter_decision", FilterViewDecisionName(candidate.filterDecision)},
			{"filter_threshold_reasons", ViewThresholdReasons(candidate)},
			{"final_rank_zero_based", candidate.finalRank >= 0 ? nlohmann::json(candidate.finalRank) : nlohmann::json(nullptr)},
			{"accepted_after_filter", candidate.finalRank >= 0},
		};
		candidates.push_back(std::move(row));
		if (candidate.rawRank >= 0)
			++rankedCount;
		if (candidate.finalRank >= 0)
			++retainedCount;
		if (candidate.initialDecision == Scene::NeighborViewCandidateObservation::INITIAL_INVALID_IMAGE ||
			candidate.initialDecision == Scene::NeighborViewCandidateObservation::INITIAL_INSUFFICIENT_SHARED_POINTS ||
			candidate.initialDecision == Scene::NeighborViewCandidateObservation::INITIAL_NO_PROJECTED_POINTS)
			++rejectedInitialCount;
		if (candidate.filterDecision == Scene::NeighborViewCandidateObservation::FILTER_REJECTED_THRESHOLD ||
			candidate.filterDecision == Scene::NeighborViewCandidateObservation::FILTER_REJECTED_MAX_VIEW_TRUNCATION)
			++rejectedFilterCount;
	}
	nlohmann::json artifact = {
		{"schema_name", "openmvs.dmap.cpu_view_candidates"},
		{"schema_version", 1},
		{"algorithm_stage", "depth_map_neighbor_view_ranking"},
		{"estimation_stage", "photometric"},
		{"geometric_iteration", nullptr},
		{"scope", "depth-map estimation view selection only"},
		{"reference_scene_image_index", idxImage},
		{"reference_image_id", reference.ID},
		{"reference_image_name", reference.name.c_str()},
		{"candidate_source", precomputed ? "precomputed_scene_neighbors" : "computed_sparse_visibility"},
		{"ranking_succeeded", observation.rankingSucceeded},
		{"filter_succeeded", observation.filterSucceeded},
		{"parameters", {
			{"requested_min_views", observation.requiredMinViews},
			{"effective_min_views", observation.effectiveMinViews},
			{"requested_min_point_views", observation.requiredMinPointViews},
			{"effective_min_point_views", observation.effectiveMinPointViews},
			{"optimal_angle_radians", observation.optimalAngle},
			{"point_inside_roi_weight", observation.roiWeight},
			{"filter_min_area", observation.filterMinArea},
			{"filter_min_scale", observation.filterMinScale},
			{"filter_max_scale_exclusive", observation.filterMaxScale},
			{"filter_min_angle_radians", observation.filterMinAngle},
			{"filter_max_angle_radians_exclusive", observation.filterMaxAngle},
			{"filter_max_views", observation.filterMaximumViews},
			{"filter_minimum_retained_guard", observation.filterMinimumRetained},
		}},
		{"score_model", {
			{"higher_is_better", true},
			{"ranking_score", "sum(max(angle_weight,0.1)*scale_weight*roi_weight)*max(covered_area,0.01)"},
			{"angle_weight", "exp((angle-optimal_angle)^2*sigma), with asymmetric sigma around the optimum"},
			{"scale_weight", "piecewise footprint-ratio weight capped at one"},
			{"reliability_weight", nullptr},
			{"reliability_weight_status", "unavailable_at_cpu_ranking_stage; exact per-pixel reliability is evaluated inside PatchMatch"},
		}},
		{"counts", {
			{"candidate_records", observation.candidates.size()},
			{"eligible_reference_points", observation.eligibleReferencePoints},
			{"scored_reference_points", observation.scoredReferencePoints},
			{"ranked_candidates", rankedCount},
			{"filter_input_candidates", observation.filterInputCount},
			{"retained_candidates", retainedCount},
			{"initially_rejected_candidates", rejectedInitialCount},
			{"filter_rejected_candidates", rejectedFilterCount},
		}},
		{"unavailable_signals", nlohmann::json::array({"per_pixel_patchmatch_reliability_weight"})},
		{"candidates", candidates},
	};
	if (precomputed) {
		artifact["unavailable_signals"].push_back("sparse_score_component_sums");
		artifact["unavailable_signals"].push_back("projected_shared_point_count");
		artifact["unavailable_signals"].push_back("candidates_rejected_before_the_persisted_neighbor_list");
	}
	const String jsonFile(frameDir + _T("cpu_view_candidates.json"));
	if (!WriteDMapJson(jsonFile, artifact))
		VERBOSE("warning: failed to write CPU view-candidate instrumentation: %s", jsonFile.c_str());

	std::ostringstream csv;
	csv << std::setprecision(9);
	csv << "estimation_stage,geometric_iteration,reference_scene_image_index,reference_image_id,"
		"candidate_scene_image_index,candidate_image_id,candidate_image_name,"
		"initial_decision,image_valid,shared_points,projected_points,score_components_available,angle_weight_sum,"
		"clipped_angle_weight_sum,scale_weight_sum,roi_weight_sum,score_before_area,average_scale,average_angle_radians,"
		"covered_area_ratio,area_factor,ranking_score,raw_rank_zero_based,filter_input_rank_zero_based,filter_decision,"
		"filter_threshold_reasons,final_rank_zero_based,accepted_after_filter\n";
	for (const Scene::NeighborViewCandidateObservation& candidate: observation.candidates) {
		const Image* candidateImage(candidate.ID < scene.images.size() ? &scene.images[candidate.ID] : NULL);
		const nlohmann::json thresholdReasons(ViewThresholdReasons(candidate));
		csv << "photometric,," << idxImage << ',' << reference.ID << ',' << candidate.ID << ',';
		if (candidateImage)
			csv << candidateImage->ID;
		csv << ',' << DMapCsvEscape(candidateImage ? candidateImage->name : String()).c_str() << ','
			<< InitialViewDecisionName(candidate.initialDecision) << ',' << (candidate.imageValid ? 1 : 0) << ','
			<< candidate.sharedPoints << ',';
		if (!precomputed)
			csv << candidate.projectedPoints;
		csv << ',' << (candidate.scoreComponentsAvailable ? 1 : 0) << ',';
		if (candidate.scoreComponentsAvailable)
			csv << candidate.angleWeightSum << ',' << candidate.clippedAngleWeightSum << ',' << candidate.scaleWeightSum << ','
				<< candidate.roiWeightSum << ',' << candidate.scoreBeforeArea;
		else
			csv << ",,,,";
		csv << ',' << candidate.avgScale << ',' << candidate.avgAngle << ',' << candidate.area << ',' << candidate.areaFactor << ','
			<< candidate.score << ',';
		if (candidate.rawRank >= 0)
			csv << candidate.rawRank;
		csv << ',';
		if (candidate.filterInputRank >= 0)
			csv << candidate.filterInputRank;
		csv << ',' << FilterViewDecisionName(candidate.filterDecision) << ','
			<< DMapCsvEscape(thresholdReasons.dump().c_str()).c_str() << ',';
		if (candidate.finalRank >= 0)
			csv << candidate.finalRank;
		csv << ',' << (candidate.finalRank >= 0 ? 1 : 0) << '\n';
	}
	const String csvFile(frameDir + _T("cpu_view_candidates.csv"));
	if (!WriteDMapText(csvFile, csv.str()))
		VERBOSE("warning: failed to write CPU view-candidate CSV instrumentation: %s", csvFile.c_str());
}

void WriteCPUViewEstimationSelectionArtifacts(
	const Scene& scene,
	IIndex idxImage,
	const DepthData& depthData,
	IIndex explicitNeighbor,
	IIndex requestedNumNeighbors,
	bool loadImages,
	int loadDepthMaps,
	IIndex stopIndex,
	CPUViewSelectionStopReason stopReason,
	float configuredEffectiveMinScore,
	bool selectionSucceeded,
	const std::unordered_set<IIndex>* missingDepthViews,
	int geometricIteration)
{
	const Image& reference(scene.images[idxImage]);
	const bool geometricStage(geometricIteration >= 0);
	const char* estimationStage(geometricStage ? "geometric_consistency" : "photometric");
	const String frameDir(DMapInstrumentFrameDir(reference, geometricIteration));
	const String candidateContractRootRelative(
		_T("depthmaps/") + DMapInstrumentFrameName(reference) + _T("/cpu_view_candidates.json"));
	const String candidateContractArtifactRelative(geometricStage ?
		_T("../../../../") + candidateContractRootRelative : _T("cpu_view_candidates.json"));
	std::unordered_map<IIndex, int> selectedRanks;
	for (IIndex rank=1; rank<depthData.images.size(); ++rank)
		selectedRanks.emplace(depthData.images[rank].GetLocalID(scene.images), int(rank-1));
	const float bestScore(depthData.neighbors.empty() ? 0.f : depthData.neighbors.front().score);
	const bool rankedPrefixMode(explicitNeighbor == NO_ID);
	const char* admissionPolicyName(rankedPrefixMode ? "ranked_prefix_with_score_cutoff" : "explicit_neighbor");
	const char* configuredThresholdStatus(rankedPrefixMode ?
		"applied_to_patchmatch" : "not_applicable_explicit_neighbor");
	const auto selectionDecision = [&](IIndex rank, IIndex candidateID, bool isSelected) -> const char* {
		if (isSelected)
			return "selected";
		if (missingDepthViews && missingDepthViews->find(candidateID) != missingDepthViews->end())
			return "rejected_missing_depth_map";
		if (explicitNeighbor != NO_ID)
			return "rejected_not_explicit_neighbor";
		if (stopReason == CPU_VIEW_STOP_NUM_NEIGHBORS && rank == stopIndex)
			return "rejected_requested_neighbor_limit";
		if (stopReason == CPU_VIEW_STOP_NUM_NEIGHBORS && rank > stopIndex)
			return "rejected_after_ordered_cutoff";
		if (stopReason == CPU_VIEW_STOP_SCORE_THRESHOLD && rank >= stopIndex)
			return "rejected_score_threshold";
		return "rejected_unavailable_reason";
	};
	nlohmann::json candidates(nlohmann::json::array());
	FOREACH(rank, depthData.neighbors) {
		const ViewScore& candidate(depthData.neighbors[rank]);
		const auto selected(selectedRanks.find(candidate.ID));
		const bool isSelected(selected != selectedRanks.end());
		const char* decision(selectionDecision(rank, candidate.ID, isSelected));
		const nlohmann::json wouldPassConfiguredThreshold(rankedPrefixMode ?
			nlohmann::json(candidate.score >= configuredEffectiveMinScore) : nlohmann::json(nullptr));
		const Image* candidateImage(candidate.ID < scene.images.size() ? &scene.images[candidate.ID] : NULL);
		candidates.push_back({
			{"candidate_scene_image_index", candidate.ID},
			{"candidate_image_id", candidateImage ? nlohmann::json(candidateImage->ID) : nlohmann::json(nullptr)},
			{"candidate_image_name", candidateImage ? nlohmann::json(candidateImage->name.c_str()) : nlohmann::json(nullptr)},
			{"filtered_rank_zero_based", rank},
			{"ranking_score", candidate.score},
			{"score_ratio_to_best", bestScore > 0 ? nlohmann::json(candidate.score/bestScore) : nlohmann::json(nullptr)},
			{"selected", isSelected},
			{"selected_rank_zero_based", isSelected ? nlohmann::json(selected->second) : nlohmann::json(nullptr)},
			{"would_pass_configured_score_threshold", wouldPassConfiguredThreshold},
			{"decision", decision},
		});
	}
	nlohmann::json artifact = {
		{"schema_name", "openmvs.dmap.cpu_view_estimation_selection"},
		{"schema_version", 2},
		{"algorithm_stage", "depth_map_estimation_source_view_selection"},
		{"estimation_stage", estimationStage},
		{"geometric_iteration", geometricStage ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
		{"scope", "depth-map estimation view selection only"},
		{"reference_scene_image_index", idxImage},
		{"reference_image_id", reference.ID},
		{"reference_image_name", reference.name.c_str()},
		{"candidate_contract", candidateContractArtifactRelative.c_str()},
		{"candidate_contract_path_base", "artifact_directory"},
		{"candidate_contract_instrumentation_root_relative", candidateContractRootRelative.c_str()},
		{"candidate_contract_estimation_stage", "photometric"},
		{"selection_succeeded", selectionSucceeded},
		{"selection_mode", rankedPrefixMode ? "ranked_prefix" : "explicit_neighbor"},
		{"admission_policy", {
			{"name", admissionPolicyName},
			{"policy_origin", "openmvs_depth_map_estimation"},
			{"score_threshold_applied", rankedPrefixMode},
			{"configured_score_threshold_status", configuredThresholdStatus},
		}},
		{"parameters", {
			{"explicit_neighbor_rank", explicitNeighbor == NO_ID ? nlohmann::json(nullptr) : nlohmann::json(explicitNeighbor)},
			{"requested_num_neighbors", requestedNumNeighbors},
			{"view_min_score_absolute", OPTDENSE::fViewMinScore},
			{"view_min_score_ratio", OPTDENSE::fViewMinScoreRatio},
			{"best_ranking_score", bestScore},
			{"effective_min_score", rankedPrefixMode ?
				nlohmann::json(configuredEffectiveMinScore) : nlohmann::json(nullptr)},
			{"configured_effective_min_score", rankedPrefixMode ?
				nlohmann::json(configuredEffectiveMinScore) : nlohmann::json(nullptr)},
			{"load_images", loadImages},
			{"load_depth_maps_mode", loadDepthMaps},
		}},
		{"ordered_cutoff", {
			{"trigger_rank_zero_based", stopIndex < depthData.neighbors.size() ? nlohmann::json(stopIndex) : nlohmann::json(nullptr)},
			{"reason", CPUViewStopReasonName(stopReason)},
		}},
		{"selected_count", selectedRanks.size()},
		{"missing_depth_map_count", missingDepthViews ? missingDepthViews->size() : 0},
		{"candidates", candidates},
	};
	const String jsonFile(frameDir + _T("cpu_view_estimation_selection.json"));
	if (!WriteDMapJson(jsonFile, artifact))
		VERBOSE("warning: failed to write CPU estimation-view instrumentation: %s", jsonFile.c_str());

	std::ostringstream csv;
	csv << std::setprecision(9);
	csv << "estimation_stage,geometric_iteration,candidate_contract_instrumentation_root_relative,"
		"admission_policy,score_threshold_applied,configured_effective_min_score,"
		"reference_scene_image_index,reference_image_id,candidate_scene_image_index,candidate_image_id,candidate_image_name,"
		"filtered_rank_zero_based,ranking_score,score_ratio_to_best,selected,selected_rank_zero_based,"
		"would_pass_configured_score_threshold,decision\n";
	FOREACH(rank, depthData.neighbors) {
		const ViewScore& candidate(depthData.neighbors[rank]);
		const Image* candidateImage(candidate.ID < scene.images.size() ? &scene.images[candidate.ID] : NULL);
		const auto selected(selectedRanks.find(candidate.ID));
		const bool isSelected(selected != selectedRanks.end());
		const char* decision(selectionDecision(rank, candidate.ID, isSelected));
		csv << estimationStage << ',';
		if (geometricStage)
			csv << geometricIteration;
		csv << ',' << candidateContractRootRelative.c_str()
			<< ',' << admissionPolicyName << ',' << (rankedPrefixMode ? 1 : 0) << ',';
		if (rankedPrefixMode)
			csv << configuredEffectiveMinScore;
		csv << ',' << idxImage << ',' << reference.ID << ',' << candidate.ID << ',';
		if (candidateImage)
			csv << candidateImage->ID;
		csv << ',' << DMapCsvEscape(candidateImage ? candidateImage->name : String()).c_str() << ',' << rank << ',' << candidate.score << ',';
		if (bestScore > 0)
			csv << candidate.score/bestScore;
		csv << ',' << (isSelected ? 1 : 0) << ',';
		if (isSelected)
			csv << selected->second;
		csv << ',';
		if (rankedPrefixMode)
			csv << (candidate.score >= configuredEffectiveMinScore ? 1 : 0);
		csv << ',' << decision << '\n';
	}
	const String csvFile(frameDir + _T("cpu_view_estimation_selection.csv"));
	if (!WriteDMapText(csvFile, csv.str()))
		VERBOSE("warning: failed to write CPU estimation-view CSV instrumentation: %s", csvFile.c_str());
}

} // anonymous namespace
#endif

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,
	EVT_CLOSE,

	EVT_PROCESSIMAGE,

	EVT_ESTIMATEDEPTHMAP,
	EVT_OPTIMIZEDEPTHMAP,
	EVT_SAVEDEPTHMAP,

	EVT_FILTERDEPTHMAP,
	EVT_ADJUSTDEPTHMAP,
};

class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/


// convert the ZNCC score to a weight used to average the fused points
inline float Conf2Weight(float conf, Depth depth) {
	return 1.f/(MAXF(1.f-conf,0.03f)*depth*depth);
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////


DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize()),
	imageCache(_scene.images)
	#ifdef _USE_CUDA
	, pmCUDANextIdx((Thread::safe_t)-1)
	, pmCUDAEpoch(0)
	#endif // _USE_CUDA
	#ifdef _USE_METAL
	, pmMetalNextIdx((Thread::safe_t)-1)
	, pmMetalEpoch(0)
	#endif // _USE_METAL
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor

// bytes the image cache may keep for the whole depth-map estimation: the whole
// set of images when it fits in three quarters of what is free above the same
// safety margin the depth-map cache keeps, and that three quarters otherwise.
//
// Holding all of them is worth reaching for, because landing just short is the
// worst place to be: the references sweep the scene in order, so a capacity a
// few images below the set makes LRU evict exactly what the next reference is
// about to ask for. Measured on a 683-image scene, a budget 2% short of the set
// turned the per-view fetch from 0.4ms into 18ms and cost 150 decodes a pass.
//
// Below a handful of images a cache would only thrash, so disable it there and
// let the images be decoded for each use.
size_t DepthMapsData::ComputeImageCacheMemory(const IIndexArr& images) const
{
	if (images.empty())
		return 0;
	const size_t allImages(ImageCache::ComputeMemorySize(scene.images, images));
	const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
	const size_t safetyMemory(ComputeSafetyMemory(memInfo));
	if (memInfo.freePhysical <= safetyMemory)
		return 0;
	const size_t maxMemory(MINF((memInfo.freePhysical - safetyMemory) / 4 * 3, allImages));
	// require room for at least 8 average images, or the whole (small) set:
	// nothing can thrash when every image fits
	return maxMemory >= MINF(allImages, allImages / images.size() * 8) ? maxMemory : 0;
} // ComputeImageCacheMemory

#ifdef _USE_CUDA
bool DepthMapsData::AllocateCudaPool(unsigned poolSize)
{
	ASSERT(pmCUDAPool.empty());
	if (poolSize == 0)
		poolSize = 1;
	// PatchMatch's ctor triggers CUDA::initDevices() on first construction;
	// build one to probe and check whether any device was actually picked up.
	auto probe = std::make_unique<MVS::CUDA::PatchMatch>();
	if (SEACAVE::CUDA::devices.IsEmpty())
		return false;
	probe->Init(false);
	pmCUDAPool.reserve(poolSize);
	pmCUDAPool.emplace_back(std::move(probe));
	for (unsigned k = 1; k < poolSize; ++k) {
		auto pm = std::make_unique<MVS::CUDA::PatchMatch>();
		pm->Init(false);
		pmCUDAPool.emplace_back(std::move(pm));
	}
	pmCUDANextIdx = (Thread::safe_t)-1;
	return true;
}

void DepthMapsData::ReinitCudaPoolForGeom()
{
	for (auto& pm : pmCUDAPool) {
		pm->Release();
		pm->Init(true);
	}
	pmCUDANextIdx = (Thread::safe_t)-1;
	Thread::safeInc(pmCUDAEpoch);
}
#endif // _USE_CUDA

#ifdef _USE_METAL
bool DepthMapsData::AllocateMetalPool(unsigned poolSize)
{
	ASSERT(pmMetalPool.empty());
	if (poolSize == 0)
		poolSize = 1;
	auto probe = std::make_unique<MVS::METAL::PatchMatch>();
	if (!probe->IsValid())
		return false;
	probe->Init(false);
	pmMetalPool.reserve(poolSize);
	pmMetalPool.emplace_back(std::move(probe));
	for (unsigned k = 1; k < poolSize; ++k) {
		auto pm = std::make_unique<MVS::METAL::PatchMatch>();
		// the probe proved the device + pipelines build; an additional instance
		// failing is unexpected (e.g. resource exhaustion), so stop growing rather
		// than add an invalid worker that would silently produce empty depth-maps
		if (!pm->IsValid())
			break;
		pm->Init(false);
		pmMetalPool.emplace_back(std::move(pm));
	}
	pmMetalNextIdx = (Thread::safe_t)-1;
	return true;
}

void DepthMapsData::ReinitMetalPoolForGeom()
{
	for (auto& pm : pmMetalPool) {
		pm->Release();
		pm->Init(true);
	}
	pmMetalNextIdx = (Thread::safe_t)-1;
	Thread::safeInc(pmMetalEpoch);
}
#endif // _USE_METAL
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	#ifdef _USE_DMAP_INSTRUMENTATION
	const bool observeViewSelection(DMapInstrumentImageEnabled(scene.images[idxImage]));
	std::unique_ptr<Scene::NeighborViewSelectionObservation> observation;
	if (observeViewSelection)
		observation = std::make_unique<Scene::NeighborViewSelectionObservation>();
	if (scene.images[idxImage].neighbors.empty()) {
		const bool selected(observation ?
			scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews,
				OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2,
				D2R(OPTDENSE::fOptimAngle), OPTDENSE::fWeightPointInsideROI, observation.get()) :
			scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews,
				OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2,
				D2R(OPTDENSE::fOptimAngle), OPTDENSE::fWeightPointInsideROI));
		if (!selected) {
			if (observation)
				WriteCPUViewCandidateArtifacts(scene, idxImage, *observation);
			return false;
		}
	} else if (observation) {
		InitPrecomputedViewObservation(scene, idxImage, *observation);
	}
#else
	if (scene.images[idxImage].neighbors.empty() &&
		!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, D2R(OPTDENSE::fOptimAngle), OPTDENSE::fWeightPointInsideROI))
		return false;
#endif
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(D2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(D2R(OPTDENSE::fMaxAngle));
	const unsigned nMaxViews(MAXF(OPTDENSE::nMaxViews, OPTDENSE::nNumViews));
#ifdef _USE_DMAP_INSTRUMENTATION
	const bool filtered(observation ?
		Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews, observation.get()) :
		Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews));
	if (observation)
		WriteCPUViewCandidateArtifacts(scene, idxImage, *observation);
	if (!filtered) {
#else
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, nMaxViews)) {
#endif
		DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

// fetch the intensities of the given view, from the image cache when possible;
// the cached image is shared with every other view using it, so scaling it for
// this view has to write to a buffer of its own
bool DepthMapsData::FetchViewImage(DepthData::ViewData& view)
{
	Image32F imageGray;
	if (!imageCache.UseImage(view.GetLocalID(scene.images), imageGray))
		return false;
	if (!DepthData::ViewData::ScaleImage(imageGray, view.image, view.scale))
		view.image = imageGray;
	return true;
} // FetchViewImage
/*----------------------------------------------------------------*/

// select target image for the reference image (the first image in "images"),
// initialize images data, and initialize depth-map and normal-map;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// if loadImages, the image data is also setup
// if loadDepthMaps is 1, the depth-maps are loaded from disk,
// if 0, the reference depth-map is initialized from sparse point-cloud,
// and if -1, the depth-maps are not initialized
#ifdef _USE_DMAP_INSTRUMENTATION
// nGeometricIter identifies the geometric-consistency stage (-1 - photometric)
#endif
// returns false if there are no good neighbors to estimate the depth-map
#ifdef _USE_DMAP_INSTRUMENTATION
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors, bool loadImages, int loadDepthMaps, int nGeometricIter)
#else
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors, bool loadImages, int loadDepthMaps)
#endif
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());
#ifdef _USE_DMAP_INSTRUMENTATION
	const bool observeViewSelection(DMapInstrumentImageEnabled(scene.images[idxImage]));
	IIndex selectionStopIndex(depthData.neighbors.size());
	CPUViewSelectionStopReason selectionStopReason(
		idxNeighbor == NO_ID ? CPU_VIEW_STOP_NONE : CPU_VIEW_STOP_EXPLICIT_PAIR);
	float configuredEffectiveMinScore(0);
	std::unique_ptr<std::unordered_set<IIndex>> missingDepthViews;
	if (observeViewSelection && loadDepthMaps > 0)
		missingDepthViews = std::make_unique<std::unordered_set<IIndex>>();
#endif

	// set this image the first image in the array
	depthData.images.Empty();
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();

	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
		viewTrg.pImageData = &scene.images[neighbor.ID];
		viewTrg.scale = neighbor.scale;
		viewTrg.camera = viewTrg.pImageData->camera;
		if (loadImages) {
			if (!FetchViewImage(viewTrg)) {
				depthData.images.Release();
				return false;
			}
			if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());
		} else {
			if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->GetSize(), viewTrg.scale));
		}
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.ID);
	} else {
		// initialize all neighbor views too (global reconstruction is used)
		const float fMinScore(MAXF(depthData.neighbors.First().score*OPTDENSE::fViewMinScoreRatio, OPTDENSE::fViewMinScore));
		#ifdef _USE_DMAP_INSTRUMENTATION
		if (observeViewSelection)
			configuredEffectiveMinScore = fMinScore;
		#endif
		#ifdef _USE_DMAP_INSTRUMENTATION
		FOREACH(idxView, depthData.neighbors) {
			const ViewScore& neighbor(depthData.neighbors[idxView]);
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				neighbor.score < fMinScore)
			{
				if (observeViewSelection) {
					selectionStopIndex = idxView;
					selectionStopReason =
						(numNeighbors && depthData.images.GetSize() > numNeighbors) ?
						CPU_VIEW_STOP_NUM_NEIGHBORS : CPU_VIEW_STOP_SCORE_THRESHOLD;
				}
				break;
			}
		#else
		for (const ViewScore& neighbor: depthData.neighbors) {
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))
				break;
		#endif
			DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
			viewTrg.pImageData = &scene.images[neighbor.ID];
			viewTrg.scale = neighbor.scale;
			viewTrg.camera = viewTrg.pImageData->camera;
			if (loadImages) {
				if (!FetchViewImage(viewTrg)) {
					// the image cannot be decoded any more; drop this neighbor, the
					// same way a neighbor with no depth-map is dropped below
					VERBOSE("warning: skipping neighbor view %u (%s): cannot load image",
						neighbor.ID, Util::getFileNameExt(viewTrg.pImageData->name).c_str());
					depthData.images.RemoveLast();
					continue;
				}
				if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());
			} else {
				if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->GetSize(), viewTrg.scale));
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (VERBOSITY_LEVEL > 2) {
			String msg;
			for (IIndex i=1; i<depthData.images.size(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].GetID(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.size()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.size()-1);
		#endif
	}
	if (depthData.images.size() < 2) {
#ifdef _USE_DMAP_INSTRUMENTATION
		if (observeViewSelection)
			WriteCPUViewEstimationSelectionArtifacts(scene, idxImage, depthData, idxNeighbor, numNeighbors,
			loadImages, loadDepthMaps, selectionStopIndex, selectionStopReason, configuredEffectiveMinScore, false, missingDepthViews.get(), nGeometricIter);
#endif
		depthData.images.Release();
		return false;
	}

	// initialize reference image as well
	DepthData::ViewData& viewRef = depthData.images.front();
	viewRef.scale = 1;
	viewRef.pImageData = &scene.images[idxImage];
	viewRef.camera = viewRef.pImageData->camera;
	if (loadImages) {
		if (!FetchViewImage(viewRef)) {
			VERBOSE("error: cannot load image '%s'", viewRef.pImageData->name.c_str());
			depthData.images.Release();
			return false;
		}
	}
	depthData.size = viewRef.pImageData->GetSize();

	// initialize views
	for (IIndex i=1; i<depthData.images.size(); ) {
		DepthData::ViewData& view = depthData.images[i];
		if (loadDepthMaps > 0) {
			// load known depth-map
			String imageFileName;
			IIndexArr IDs;
			cv::Size imageSize;
			Depth dMin, dMax;
			NormalMap normalMap;
			ConfidenceMap confMap;
			ViewsMap viewsMap;
			if (!ImportDepthDataRaw(ComposeDepthFilePath(view.GetID(), "dmap"),
				imageFileName, IDs, imageSize, view.cameraDepthMap.K, view.cameraDepthMap.R, view.cameraDepthMap.C,
				dMin, dMax, view.depthMap, normalMap, confMap, viewsMap, HeaderDepthDataRaw::HAS_DEPTH))
			{
				// neighbor depth-maps are needed during geometric-consistency iterations;
				// some views may have failed depth estimation, so their depth-map is missing
				VERBOSE("warning: skipping neighbor view %u (%s): cannot load depth-map '%s'",
					view.GetID(), Util::getFileNameExt(view.pImageData->name).c_str(), ComposeDepthFilePath(view.GetID(), "dmap").c_str());
				#ifdef _USE_DMAP_INSTRUMENTATION
					if (missingDepthViews)
						missingDepthViews->insert(view.GetLocalID(scene.images));
				#endif
				view.depthMap.release();
				depthData.images.RemoveAtMove(i);
				continue;
			}
			ASSERT(viewRef.image.size() == view.depthMap.size());
		}
		view.Init(viewRef.camera);
		++i;
	}
	if (depthData.images.size() < 2) {
#ifdef _USE_DMAP_INSTRUMENTATION
		if (observeViewSelection)
			WriteCPUViewEstimationSelectionArtifacts(scene, idxImage, depthData, idxNeighbor, numNeighbors,
			loadImages, loadDepthMaps, selectionStopIndex, selectionStopReason, configuredEffectiveMinScore, false, missingDepthViews.get(), nGeometricIter);
#endif
		depthData.images.Release();
		return false;
 	}
#ifdef _USE_DMAP_INSTRUMENTATION
	if (observeViewSelection)
		WriteCPUViewEstimationSelectionArtifacts(scene, idxImage, depthData, idxNeighbor, numNeighbors,
		loadImages, loadDepthMaps, selectionStopIndex, selectionStopReason, configuredEffectiveMinScore, true, missingDepthViews.get(), nGeometricIter);
#endif

	// initialize depth-map and normal-map for the reference image
	if (loadDepthMaps > 0) {
		// load known depth-map and normal-map
		String imageFileName;
		IIndexArr IDs;
		cv::Size imageSize;
		Camera camera;
		ConfidenceMap confMap;
		ViewsMap viewsMap;
		if (!ImportDepthDataRaw(ComposeDepthFilePath(viewRef.GetID(), "dmap"),
				imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, depthData.dMin, depthData.dMax,
				depthData.depthMap, depthData.normalMap, confMap, viewsMap, 3))
			return false;
		ASSERT(viewRef.image.size() == depthData.depthMap.size());
		ASSERT(depthData.normalMap.empty() || viewRef.image.size() == depthData.normalMap.size());
		if (depthData.normalMap.empty()) {
			// estimate normal map
			EstimateNormalMap(viewRef.camera.K, depthData.depthMap, depthData.normalMap);
		}
	} else if (loadDepthMaps == 0) {
		// initialize depth and normal maps
		if (OPTDENSE::nMinViewsTrustPoint < 2 || depthData.points.empty()) {
			// compute depth range and initialize known depths, else random
			const Image8U::Size size(viewRef.image.size());
			depthData.depthMap.create(size); depthData.depthMap.memset(0);
			depthData.normalMap.create(size);
			if (depthData.points.empty()) {
				// all values will be initialized randomly
				depthData.dMin = 1e-1f;
				depthData.dMax = 1e+2f;
			} else {
				// initialize with the sparse point-cloud
				const int nPixelArea(2); // half windows size around a pixel to be initialize with the known depth
				depthData.dMin = FLT_MAX;
				depthData.dMax = 0;
				FOREACHPTR(pPoint, depthData.points) {
					const PointCloud::Point& X = scene.pointcloud.points[*pPoint];
					const Point3 camX(viewRef.camera.TransformPointW2C(Cast<REAL>(X)));
					const ImageRef x(ROUND2INT(viewRef.camera.TransformPointC2I(camX)));
					const float d((float)camX.z);
					const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
					const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));
					for (int y=sx.y; y<=ex.y; ++y) {
						for (int x=sx.x; x<=ex.x; ++x) {
							depthData.depthMap(y,x) = d;
							depthData.normalMap(y,x) = Normal::ZERO;
						}
					}
					if (depthData.dMin > d)
						depthData.dMin = d;
					if (depthData.dMax < d)
						depthData.dMax = d;
				}
				depthData.dMin *= 0.9f;
				depthData.dMax *= 1.1f;
			}
		} else {
			ASSERT(!depthData.points.empty());
			// compute rough estimates using the sparse point-cloud
			InitDepthMap(depthData);
		}
	}
	return true;
} // InitViews
/*----------------------------------------------------------------*/

// roughly estimate depth and normal maps by triangulating the sparse point-cloud
// and interpolating normal and depth for all pixels
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.GetView());
	TriangulatePoints2DepthMap(image.camera, image.image.size(), scene.pointcloud, depthData.points,
		depthData.depthMap, depthData.normalMap, depthData.dMin, depthData.dMax,
		OPTDENSE::bAddCorners && image.pImageData->avgDepth > 0 ? image.pImageData->avgDepth : 0.f, OPTDENSE::bInitSparse);
	depthData.dMin *= 0.9f;
	depthData.dMax *= 1.1f;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	// save rough depth map as image
	if (VERBOSITY_LEVEL > 4) {
		ExportDepthMap(ComposeDepthFilePath(image.GetID(), "init.png"), depthData.depthMap);
		ExportNormalMap(ComposeDepthFilePath(image.GetID(), "init.normal.png"), depthData.normalMap);
		ExportPointCloud(ComposeDepthFilePath(image.GetID(), "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
	}
	#endif

	DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", image.GetID(), depthData.points.size(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/


// initialize the confidence map (NCC score map) with the score of the current estimates
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch()) {
			estimator.depthMap0(x) = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			estimator.confMap0(x) = 2.f;
			continue;
		}
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		const Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));
		if (!ISINSIDE(depth, estimator.dMin, estimator.dMax)) {
			// init with random values
			depth = estimator.RandomDepth(estimator.dMinSqr, estimator.dMaxSqr);
			normal = estimator.RandomNormal(viewDir);
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values
			normal = estimator.RandomNormal(viewDir);
		}
		ASSERT(ISEQUAL(norm(normal), 1.f), "Norm = ", norm(normal));
		estimator.confMap0(x) = estimator.ScorePixel(depth, normal);
	}
	return NULL;
}
// run propagation and random refinement cycles
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);
	return NULL;
}
// remove all estimates with too big score and invert confidence map
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	MAYBEUNUSED const float fOptimAngle(D2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = estimator.depthMap0(x);
		float& conf = estimator.confMap0(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) {
			conf = 0;
			depth = 0;
			estimator.normalMap0(x) = Normal::ZERO;
		} else {
			#if 1
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			conf = conf>=1.f ? 0.f : 1.f-conf;
			#else
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.GetNth(estimator.idxScore));
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
			const float fCosAngle(estimator.scores.minCoeff());
			#else
			const float fCosAngle(estimator.idxScore ?
				std::accumulate(estimator.scores.begin(), &estimator.scores.PartialSort(estimator.idxScore), 0.f) / estimator.idxScore :
				*std::min_element(estimator.scores.cbegin(), estimator.scores.cend()));
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#else
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
			#endif
		}
	}
	return NULL;
}

DepthData DepthMapsData::ScaleDepthData(const DepthData& inputDeptData, float scale) {
	ASSERT(scale <= 1);
	if (scale == 1)
		return inputDeptData;
	DepthData rescaledDepthData(inputDeptData);
	FOREACH (idxView, rescaledDepthData.images) {
		DepthData::ViewData& viewData = rescaledDepthData.images[idxView];
		ASSERT(viewData.depthMap.empty() || viewData.image.size() == viewData.depthMap.size());
		cv::resize(viewData.image, viewData.image, cv::Size(), scale, scale, cv::INTER_AREA);
		viewData.camera = viewData.pImageData->camera;
		viewData.camera.K = viewData.camera.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		if (!viewData.depthMap.empty()) {
			cv::resize(viewData.depthMap, viewData.depthMap, viewData.image.size(), 0, 0, cv::INTER_AREA);
			viewData.cameraDepthMap = viewData.pImageData->camera;
			viewData.cameraDepthMap.K = viewData.cameraDepthMap.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		}
		viewData.Init(rescaledDepthData.images[0].camera);
	}
	if (!rescaledDepthData.depthMap.empty())
		cv::resize(rescaledDepthData.depthMap, rescaledDepthData.depthMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	if (!rescaledDepthData.normalMap.empty())
		cv::resize(rescaledDepthData.normalMap, rescaledDepthData.normalMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	return rescaledDepthData;
}

// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.
//  - nGeometricIter: current geometric-consistent estimation iteration (-1 - normal patch-match)
bool DepthMapsData::EstimateDepthMap(IIndex idxImage, int nGeometricIter)
{
	#ifdef _USE_CUDA
	if (!pmCUDAPool.empty()) {
		// claim a pool slot for this worker thread; epoch invalidates the claim
		// across phase boundaries so re-used OS threads pick a fresh slot. also
		// re-claim when a thread reused across DepthMapsData instances holds a slot
		// now out of range for a smaller pool (epochs can collide at the value 0)
		static thread_local int s_slot = -1;
		static thread_local Thread::safe_t s_epoch = (Thread::safe_t)-1;
		if (!ISINSIDE(s_slot, 0, (int)pmCUDAPool.size()) || s_epoch != pmCUDAEpoch) {
			s_slot = (int)(Thread::safeInc(pmCUDANextIdx) % (Thread::safe_t)pmCUDAPool.size());
			s_epoch = pmCUDAEpoch;
		}
		#ifdef _USE_DMAP_INSTRUMENTATION
		pmCUDAPool[s_slot]->EstimateDepthMap(arrDepthData[idxImage], nGeometricIter);
		#else
		pmCUDAPool[s_slot]->EstimateDepthMap(arrDepthData[idxImage]);
		#endif
		return true;
	}
	#endif // _USE_CUDA

	#ifdef _USE_METAL
	if (!pmMetalPool.empty()) {
		// runs both photometric (nGeometricIter < 0) and geometric-consistency passes;
		// the pool's bGeomConsistency state (Init/ReinitMetalPoolForGeom) selects the mode
		static thread_local int s_slotM = -1;
		static thread_local Thread::safe_t s_epochM = (Thread::safe_t)-1;
		// re-claim a slot when uninitialized, after a phase boundary (epoch bump), or
		// when a thread reused across DepthMapsData instances holds a slot that is now
		// out of range for a smaller pool (epochs can collide at the initial value 0)
		if (!ISINSIDE(s_slotM, 0, (int)pmMetalPool.size()) || s_epochM != pmMetalEpoch) {
			s_slotM = (int)(Thread::safeInc(pmMetalNextIdx) % (Thread::safe_t)pmMetalPool.size());
			s_epochM = pmMetalEpoch;
		}
		pmMetalPool[s_slotM]->EstimateDepthMap(arrDepthData[idxImage]);
		return true;
	}
	#endif // _USE_METAL

	TD_TIMER_STARTD();

	const unsigned nMaxThreads(scene.nMaxThreads);
	const unsigned iterBegin(nGeometricIter < 0 ? 0u : OPTDENSE::nEstimationIters+(unsigned)nGeometricIter);
	const unsigned iterEnd(nGeometricIter < 0 ? OPTDENSE::nEstimationIters : iterBegin+1);

	// init threads
	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.reserve(nMaxThreads);
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;

	// Multi-Resolution :
	DepthData& fullResDepthData(arrDepthData[idxImage]);
	const unsigned totalScaleNumber(nGeometricIter < 0 ? OPTDENSE::nSubResolutionLevels : 0u);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	DepthEstimator::WeightMap weightMap0;
	#else
	Image64F imageSum0;
	#endif
	DepthMap currentSizeResDepthMap;
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		ASSERT(depthData.images.size() > 1);
		const DepthData::ViewData& image(depthData.images.front());
		ASSERT(!image.image.empty() && !depthData.images[1].image.empty());
		const Image8U::Size size(image.image.size());
		if (scaleNumber != totalScaleNumber) {
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0, OPTDENSE::nIgnoreMaskLabel >= 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			depthData.depthMap.copyTo(currentSizeResDepthMap);
		}
		else if (totalScaleNumber > 0) {
			fullResDepthData.depthMap.release();
			fullResDepthData.normalMap.release();
			fullResDepthData.confMap.release();
		}
		depthData.confMap.create(size);

		// init integral images and index to image-ref map for the reference data
		#if DENSE_NCC == DENSE_NCC_WEIGHTED
		weightMap0.clear();
		weightMap0.resize(size.area()-(size.width+1)*DepthEstimator::nSizeHalfWindow);
		#else
		cv::integral(image.image, imageSum0, CV_64F);
		#endif
		if (prevDepthMapSize != size || OPTDENSE::nIgnoreMaskLabel >= 0) {
			BitMatrix mask;
			if (OPTDENSE::nIgnoreMaskLabel >= 0 && DepthEstimator::ImportIgnoreMask(*image.pImageData, depthData.depthMap.size(), (uint8_t)OPTDENSE::nIgnoreMaskLabel, mask))
				depthData.ApplyIgnoreMask(mask);
			DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
			#if 0 && !defined(_RELEASE)
			// show pixels to be processed
			Image8U cmask(size);
			cmask.memset(0);
			for (const DepthEstimator::MapRef& x: coords)
				cmask(x.y, x.x) = 255;
			cmask.Show("cmask");
			#endif
			prevDepthMapSize = size;
		}

		// initialize the reference confidence map (NCC score map) with the score of the current estimates
		{
			// create working threads
			idxPixel = -1;
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				estimators.emplace_back(iterBegin, depthData, idxPixel,
					#if DENSE_NCC == DENSE_NCC_WEIGHTED
					weightMap0,
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
			}
			ASSERT(estimators.size() == threads.size()+1);
			FOREACH(i, threads)
				threads[i].start(ScoreDepthMapTmp, &estimators[i]);
			ScoreDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			FOREACHPTR(pThread, threads)
				pThread->join();
			estimators.clear();
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save rough depth map as image
			if (VERBOSITY_LEVEL > 4 && nGeometricIter < 0) {
				ExportDepthMap(ComposeDepthFilePath(image.GetID(), "rough.png"), depthData.depthMap);
				ExportNormalMap(ComposeDepthFilePath(image.GetID(), "rough.normal.png"), depthData.normalMap);
				ExportPointCloud(ComposeDepthFilePath(image.GetID(), "rough.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// run propagation and random refinement cycles on the reference data
		for (unsigned iter=iterBegin; iter<iterEnd; ++iter) {
			// create working threads
			idxPixel = -1;
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				estimators.emplace_back(iter, depthData, idxPixel,
					#if DENSE_NCC == DENSE_NCC_WEIGHTED
					weightMap0,
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
			}
			ASSERT(estimators.size() == threads.size()+1);
			FOREACH(i, threads)
				threads[i].start(EstimateDepthMapTmp, &estimators[i]);
			EstimateDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			FOREACHPTR(pThread, threads)
				pThread->join();
			estimators.clear();
			#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
			// save intermediate depth map as image
			if (VERBOSITY_LEVEL > 4) {
				String path(ComposeDepthFilePath(image.GetID(), "iter")+String::ToString(iter));
				if (nGeometricIter >= 0)
					path += String::FormatString(".geo%d", nGeometricIter);
				ExportDepthMap(path+".png", depthData.depthMap);
				ExportNormalMap(path+".normal.png", depthData.normalMap);
				ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
		}
	}

	DepthData& depthData(fullResDepthData);
	// remove all estimates with too big score and invert confidence map
	{
		const float fNCCThresholdKeep(OPTDENSE::fNCCThresholdKeep);
		if (nGeometricIter < 0 && OPTDENSE::nEstimationGeometricIters)
			OPTDENSE::fNCCThresholdKeep *= 1.2f;
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.empty());
		while (estimators.size() < nMaxThreads)
			estimators.emplace_back(0, depthData, idxPixel,
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				weightMap0,
				#else
				imageSum0,
				#endif
				coords);
		ASSERT(estimators.size() == threads.size()+1);
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.back());
		// wait for the working threads to close
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.clear();
		OPTDENSE::fNCCThresholdKeep = fNCCThresholdKeep;
	}

	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.size() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		depthData.depthMap.cols, depthData.depthMap.rows, TD_TIMER_GET_FMT().c_str());
	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/


// filter out small depth segments from the given depth map
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);
	unsigned speckle_size = OPTDENSE::nSpeckleSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// allocate memory on heap for dynamic programming arrays
	TImage<bool> done_map(size, false);
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// for all pixels do
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// if the first pixel in this segment has been already processed => skip
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			seg_list[0] = ImageRef(u,v);
			seg_list_count = 1;
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment
				const ImageRef addr_curr(seg_list[seg_list_curr]);
				const Depth& depth_curr = depthMap(addr_curr);

				if (depth_curr>0) {
					// fill list with neighbor positions
					neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
					neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
					neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
					neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

					// for all neighbors do
					for (int i=0; i<4; ++i) {
						// get neighbor pixel address
						const ImageRef& addr_neighbor(neighbor[i]);
						// check if neighbor is inside image
						if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
							// check if neighbor has not been added yet
							bool& done = done_map(addr_neighbor);
							if (!done) {
								// check if the neighbor is valid and similar to the current pixel
								// (belonging to the current segment)
								const Depth& depth_neighbor = depthMap(addr_neighbor);
								if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
									// add neighbor coordinates to segment list
									seg_list[seg_list_count++] = addr_neighbor;
									// set neighbor pixel in done_map to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									done = true;
								}
							}
						}
					}
				}

				// set current pixel in seg_list to "done"
				++seg_list_curr;

				// set current pixel in done_map to "done"
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			if (seg_list_count < speckle_size) {
				// for all pixels in current segment invalidate pixels
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}

	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map
bool DepthMapsData::GapInterpolation(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// 1. Row-wise:
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter
		unsigned count = 0;

		// for each element of the row do
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation
				int u_curr(u-count);
				const int u_first(u_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v,u_first), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}

	// 2. Column-wise:
	// for each column do
	for (int u=0; u<size.x; ++u) {

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v=0; v<size.y; ++v) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v_first,u), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/


// adjust confidence-map based on the depth-map and the confidence-maps of the neighbor depth-maps
bool DepthMapsData::AdjustConfidenceFast(DepthData& depthDataRef, const IIndexArr& idxNeighbors)
{
	TD_TIMER_STARTD();

	// set confidence as the similarity of the depth values in the neighbor depth-maps
	// to the depth value from the reference depth-map
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty() && !idxNeighbors.empty());
	ASSERT(depthDataRef.confMap.size() == depthDataRef.depthMap.size());
	constexpr Depth thDepthSimilarity(0.01f);
	constexpr Depth sigmaDepthDiff(1.f / (-2.f * SQUARE(thDepthSimilarity)));
	const DepthData::ViewData& imageRef = depthDataRef.GetView();
	ConfidenceMap newConfMap(depthDataRef.depthMap.size());
	#if TD_VERBOSE != TD_VERBOSE_OFF
	unsigned nProcessed(0), nDiscarded(0);
	#endif
	for (int r=0; r<depthDataRef.depthMap.rows; ++r) {
		for (int c=0; c<depthDataRef.depthMap.cols; ++c) {
			const Depth& depthRef = depthDataRef.depthMap(r,c);
			if (depthRef <= 0) {
				newConfMap(r,c) = 0;
				continue;
			}
			const Point3 X(imageRef.camera.TransformPointI2W(Point3(c,r,depthRef)));
			const float confPhotoRef(depthDataRef.confMap(r,c));
			// check if the point's depth is similar to the depth in the neighbor depth-maps
			// and keep the smallest difference
			Depth minDiff(1.f);
			float bestConf(0), negBestConf1(0), negBestConf2(0);
			for (IIndex idxN: idxNeighbors) {
				const DepthData& depthData = arrDepthData[idxN];
				const DepthData::ViewData& image = depthData.GetView();
				const Point3 camX(image.camera.TransformPointW2C(X));
				if (camX.z <= 0)
					continue;
				const ImageRef x(ROUND2INT(image.camera.TransformPointC2I(camX)));
				if (!depthData.depthMap.isInside(x))
					continue;
				const Depth depth(depthData.depthMap(x));
				if (depth <= 0)
					continue;
				const Depth diff(DepthSimilarity((Depth)camX.z, depth));
				const float conf(depthData.confMap(x));
				if (diff > thDepthSimilarity) {
					if (negBestConf1 < conf) {
						negBestConf2 = negBestConf1;
						negBestConf1 = conf;
					} else if (negBestConf2 < conf)
						negBestConf2 = conf;
				}
				if (minDiff > diff || (minDiff == diff && bestConf < conf)) {
					minDiff = diff;
					bestConf = conf;
				}
			}
			// set confidence based on the depth difference;
			// if confidence-map is available, the final confidence is a combination of the photometric and similarity confidence
			const float confPhoto = MINF(confPhotoRef, bestConf);
			const float confSimilarity = EXP(SQUARE(minDiff) * sigmaDepthDiff);
			const float negBestConfs(negBestConf1 + negBestConf2);
			const bool bKeep(confPhoto > negBestConfs);
			newConfMap(r,c) = bKeep ? 0.3f*confPhoto + 0.7f*confSimilarity : (negBestConfs > 0.f ? 0.1f*confPhoto/negBestConfs : 0.f);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			if (confSimilarity <= 0.5f)
				++nDiscarded;
			++nProcessed;
			#endif
		}
	}
	if (!SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "adjusted.fast.cmap"), newConfMap))
		return false;

	DEBUG("Confidence-map %3u fast-adjusted using %u other images: %u/%u depths discarded (%s)",
		imageRef.GetID(), idxNeighbors.size(), nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	return true;
} // AdjustConfidenceFast
/*----------------------------------------------------------------*/

// filter confidence-map, one pixel at a time, using confidence based fusion of neighbor pixels
bool DepthMapsData::AdjustConfidence(DepthData& depthDataRef, const IIndexArr& idxNeighbors)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.size();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter, N));
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust, N));

	// project all neighbor depth-maps to this image
	const DepthData::ViewData& imageRef = depthDataRef.GetView();
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);
	ConfidenceMapArr confMaps(N);
	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		confMap.create(sizeRef);
		confMap.memset(0);
		const IIndex idxView = idxNeighbors[n];
		const DepthData& depthData = arrDepthData[idxView];
		const Camera& camera = depthData.GetView().camera;
		for (int i=0; i<depthData.depthMap.rows; ++i) {
			for (int j=0; j<depthData.depthMap.cols; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));
				const Point3 camX(cameraRef.TransformPointW2C(X));
				if (camX.z <= 0)
					continue;
				#if 0
				// set depth on the rounded image projection only
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection
				const Point2 imgX(cameraRef.TransformPointC2I(camX));
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};
				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					if (!depthMap.isInside(xRef))
						continue;
					Depth& depthRef(depthMap(xRef));
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					confMap(xRef) = depthData.confMap(x);
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (VERBOSITY_LEVEL > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.GetView().GetID(), idxView)), depthMap);
		#endif
	}

	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);
	ConfidenceMap newConfMap(sizeRef);
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	// average similar depths, and decrease confidence if depths do not agree
	// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
	for (int i=0; i<sizeRef.height; ++i) {
		for (int j=0; j<sizeRef.width; ++j) {
			const ImageRef xRef(j,i);
			const Depth depth(depthDataRef.depthMap(xRef));
			if (depth == 0) {
				newDepthMap(xRef) = 0;
				newConfMap(xRef) = 0;
				continue;
			}
			ASSERT(depth > 0);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			++nProcessed;
			#endif
			// update best depth and confidence estimate with all estimates
			float posConf(depthDataRef.confMap(xRef)), negConf(0);
			Depth avgDepth(depth*posConf);
			unsigned nPosViews(0), nNegViews(0);
			unsigned n(N);
			do {
				const Depth d(depthMaps[--n](xRef));
				if (d == 0) {
					if (nPosViews + nNegViews + n < nMinViews)
						goto DiscardDepth;
					continue;
				}
				ASSERT(d > 0);
				if (IsDepthSimilar(depth, d, thDepthDiff)) {
					// average similar depths
					const float c(confMaps[n](xRef));
					avgDepth += d*c;
					posConf += c;
					++nPosViews;
				} else {
					// penalize confidence
					if (depth > d) {
						// occlusion
						negConf += confMaps[n](xRef);
					} else {
						// free-space violation
						const DepthData& depthData = arrDepthData[idxNeighbors[n]];
						const Camera& camera = depthData.GetView().camera;
						const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
						const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
						if (depthData.confMap.isInside(x)) {
							const float c(depthData.confMap(x));
							negConf += (c > 0 ? c : confMaps[n](xRef));
						} else
							negConf += confMaps[n](xRef);
					}
					++nNegViews;
				}
			} while (n);
			ASSERT(nPosViews+nNegViews >= nMinViews);
			// if enough good views and positive confidence...
			if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
				// consider this pixel an inlier
				newConfMap(xRef) = 1.f - MINF(((negConf+0.2f) * nPosViews) / (posConf * MAXF(nNegViews,1u)), 1.f);
			} else {
				// consider this pixel an outlier
				DiscardDepth:
				newConfMap(xRef) = 0;
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nDiscarded;
				#endif
			}
		}
	}
	if (!SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "adjusted.cmap"), newConfMap))
		return false;

	DEBUG("Confidence-map %3u adjusted using %u other images: %u/%u depths discarded (%s)",
		imageRef.GetID(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	return true;
} // AdjustConfidence
/*----------------------------------------------------------------*/


// estimate normal-maps based on the depth-maps;
// loads and saves the depth-data from/to disk
void DepthMapsData::EstimateNormalMaps()
{
	#ifdef DENSE_USE_OPENMP
	bool bAbort(false);
	#pragma omp parallel for shared(bAbort)
	for (int64_t i=0; i<(int64_t)scene.images.size(); ++i) {
		#pragma omp flush (bAbort)
		if (bAbort)
			continue;
		const IIndex idxImage((IIndex)i);
	#else
	FOREACH(idxImage, scene.images) {
	#endif
		DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.IsValid())
			continue;
		const String fileName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		const bool bEmpty(depthData.IsEmpty());
		if (bEmpty && !depthData.Load(fileName)) {
			#ifdef DENSE_USE_OPENMP
			bAbort = true;
			#pragma omp flush (bAbort)
			continue;
			#else
			return;
			#endif
		}
		ASSERT(!depthData.IsEmpty());
		ASSERT(!scene.images[idxImage].neighbors.empty());
		if (depthData.normalMap.empty()) {
			EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
			if (!depthData.Save(fileName)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return;
				#endif
			}
		}
		if (bEmpty)
			depthData.Release();
	}
	#ifdef DENSE_USE_OPENMP
	if (bAbort)
		return;
	#endif
} // EstimateNormalMaps


// fuse all depth-maps by simply projecting them in a 3D point-cloud
// in the world coordinate space
void DepthMapsData::MergeDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	// estimate total number of 3D points that will be generated
	size_t nPointsEstimate(0);
	for (const DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			nPointsEstimate += (size_t)depthData.depthMap.size().area()*7/10;

	// fuse all depth-maps
	size_t nDepthMaps(0), nDepths(0);
	pointcloud.points.reserve(nPointsEstimate);
	pointcloud.pointViews.reserve(nPointsEstimate);
	if (bEstimateColor)
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.normals.reserve(nPointsEstimate);
	Util::Progress progress(_T("Merged depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	FOREACH(idxImage, arrDepthData) {
		TD_TIMER_STARTD();
		DepthData& depthData = arrDepthData[idxImage];
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		if (bEstimateNormal && depthData.normalMap.empty())
			EstimateNormalMaps();
		const DepthData::ViewData& image = depthData.GetView();
		// the colors come from the image of this depth-map only, so decode it here
		// and release it below instead of keeping every image of the scene resident
		Image& imageData = *image.pImageData;
		if (bEstimateColor && imageData.image.empty() &&
			!imageData.ReloadImageAtPreparedResolution())
			DEBUG("warning: image %u could not be decoded; its points stay uncolored", imageData.ID);
		const size_t nNumPointsPrev(pointcloud.points.size());
		for (int i=0; i<depthData.depthMap.rows; ++i) {
			for (int j=0; j<depthData.depthMap.cols; ++j) {
				// ignore invalid depth
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				// create the corresponding 3D point
				pointcloud.points.emplace_back(image.camera.TransformPointI2W(Point3(Cast<float>(x),depth)));
				pointcloud.pointViews.emplace_back().push_back(idxImage);
				if (bEstimateColor)
					pointcloud.colors.emplace_back(imageData.image.empty() ?
						PointCloud::Color(Pixel8U::BLACK) : PointCloud::Color(imageData.image(x)));
				if (bEstimateNormal)
					depthData.GetNormal(x, pointcloud.normals.emplace_back());
				++nDepths;
			}
		}
		depthData.DecRef();
		if (bEstimateColor)
			imageData.ReleaseImage();
		++nDepthMaps;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size());
		DEBUG_ULTIMATE("Depth-map for reference image %3u merged using %u depth-maps: %u new points (%s)",
			idxImage, depthData.images.size()-1, pointcloud.points.size()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(idxImage+1);
	}
	GET_LOGCONSOLE().Play();
	progress.close();

	DEBUG_EXTRA("Depth-maps merged: %u depth-maps, %u depths, %u points (%d%%) (%s)",
		nDepthMaps, nDepths, pointcloud.points.size(), ROUND2INT(100.f*pointcloud.points.size()/nDepths), TD_TIMER_GET_FMT().c_str());
} // MergeDepthMaps
/*----------------------------------------------------------------*/


// compute available memory to be used for depth-data caching
//  - numDMapsReserveFusion: maximum number of depth-maps for which to reserve memory for fusion
size_t GetAvailableMemory(const DepthDataArr& arrDepthData, const BoolArr& fusedDMaps, IIndex numDMapsReserveFusion, size_t currentCacheMemory = 0)
{
	size_t resolution(0);
	IIndex numDMaps(0);
	FOREACH(idxImage, arrDepthData) {
		const DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.IsValid())
			continue;
		if (fusedDMaps[idxImage])
			continue;
		resolution += depthData.size.area();
		if (++numDMaps >= numDMapsReserveFusion)
			break;
	}
	if (numDMaps == 0)
		return 0;
	const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
	const size_t neededPointCloudMemory(ROUND2INT<size_t>(resolution * (1/*depth*/+1/*color*/+3/*normal*/+1/*confidence*/) * 4/*bytes*/ * 0.35/*unique pixels per depth-map*/));
	const size_t freeMemory(currentCacheMemory + memInfo.freePhysical);
	const size_t safetyMemory(ComputeSafetyMemory(memInfo));
	const size_t neededMemory(neededPointCloudMemory + safetyMemory);
	const size_t minDMapsMemory(resolution / numDMaps * 8/*min dmaps in memory*/ * ((1/*depth*/ + 3/*normal*/ + 1/*confidence*/) * 4/*bytes*/ + 3/*color bytes*/));
	if (freeMemory < neededMemory) {
		DEBUG("warning: not enough memory to cache depth-maps (%luMB needed, %luMB available)", neededMemory/1024/1024, freeMemory/1024/1024);
		return MINF(currentCacheMemory, minDMapsMemory);
	}
	return freeMemory - neededMemory;
} // GetAvailableMemory

// decode the pixels of every image with a depth-map to fuse, for the steps that
// need all of them at once instead of the few a cache can hold; the images without
// one are never sampled, so decoding them would only exceed the budget the caller
// validated over exactly this set
bool LoadAllImages(ImageArr& images, const DepthDataArr& arrDepthData)
{
	ASSERT(images.size() == arrDepthData.size());
	bool bSuccess(true);
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for shared(bSuccess)
	for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
		Image& imageData = images[(IIndex)ID];
		if (!arrDepthData[(IIndex)ID].IsValid())
			continue;
	#else
	FOREACH(idxImage, images) {
		Image& imageData = images[idxImage];
		if (!arrDepthData[idxImage].IsValid())
			continue;
	#endif
		if (imageData.IsValid() && imageData.image.empty() &&
			!imageData.ReloadImageAtPreparedResolution())
			bSuccess = false;
	}
	return bSuccess;
} // LoadAllImages

// decide how the fused colors reach the image pixels, which the depth-map
// estimation no longer leaves resident:
//  - when the pixels of every image fit next to the depth-maps the cache has to
//    hold anyway, decode them all once here and leave them resident, so a scene
//    that was never memory bound does not pay a decode every time a depth-map
//    re-enters the cache (whole images are re-read, and they are the largest
//    files in play);
//  - otherwise hand the images to the cache, which loads and releases them
//    together with the depth-data, bounding what a scene with far more images
//    than fit can use.
// Returns the images for the cache to manage, or NULL once they are resident,
// taking what they occupy out of the cache budget.
ImageArr* PrepareFusionImages(const DepthDataArr& arrDepthData, ImageArr& images, size_t& cacheMemory)
{
	size_t allColors(0), resolution(0);
	IIndex numDMaps(0);
	FOREACH(idxImage, arrDepthData) {
		if (!arrDepthData[idxImage].IsValid())
			continue;
		const Image& imageData = images[idxImage];
		allColors += (size_t)imageData.GetSize().area() * sizeof(Pixel8U);
		resolution += (size_t)arrDepthData[idxImage].size.area();
		++numDMaps;
	}
	if (numDMaps == 0)
		return NULL;
	// the cache still has to hold the depth-map being fused and its neighbors
	const size_t workingSet(resolution / numDMaps *
		(MINF(OPTDENSE::nMaxViewsFuse, numDMaps) + 1) * (1/*depth*/ + 3/*normal*/ + 1/*confidence*/) * 4/*bytes*/);
	if (allColors + workingSet > cacheMemory) {
		VERBOSE("Fused colors sampled through the depth-map cache: %luMB of images do not fit in %luMB",
			allColors/1024/1024, cacheMemory/1024/1024);
		return &images;
	}
	if (!LoadAllImages(images, arrDepthData))
		VERBOSE("warning: some images could not be decoded; the points they see stay uncolored");
	cacheMemory -= allColors;
	return NULL;
} // PrepareFusionImages

// budget the memory a fusion pass may use and decide where the fused colors come
// from: images left resident (pCachedImages NULL) or managed by the depth-map cache
struct FusionCacheSetup {
	size_t cacheMemory;
	ImageArr* pCachedImages;
	FusionCacheSetup(const DepthDataArr& arrDepthData, const BoolArr& fusedDMaps, IIndex numDMapsReserveFusion, ImageArr& images, bool bEstimateColor)
		:
		cacheMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion)),
		pCachedImages(bEstimateColor ? PrepareFusionImages(arrDepthData, images, cacheMemory) : NULL)
	{
	}
};

// finds the best depth-map to fuse next that maximizes the number of neighbors already in cache
std::tuple<unsigned, unsigned, unsigned> FetchBestNextDMapIndex(const DepthDataArr& arrDepthData, const DMapCache& cacheDMaps, const BoolArr& fusedDMaps) {
	const IIndexArr cachedImages = cacheDMaps.GetCachedImageIndices(true);
	IIndex bestImageIdx = NO_ID;
	unsigned bestImageScore = 0, bestImageSize = std::numeric_limits<unsigned>::max();
	FOREACH(idxImage, arrDepthData) {
		const DepthData& depthData = arrDepthData[idxImage];
		if (!depthData.IsValid())
			continue;
		if (fusedDMaps[idxImage])
			continue;
		ASSERT(!depthData.neighbors.empty());
		IIndexArr cachedNeighbors;
		if (!cachedImages.empty()) {
			IIndexArr neighbors(0, depthData.neighbors.size());
			for (ViewScore& neighbor: depthData.neighbors)
				neighbors.push_back(neighbor.ID);
			neighbors.Sort();
			std::set_intersection(neighbors.begin(), neighbors.end(),
				cachedImages.begin(), cachedImages.end(),
				std::back_inserter(cachedNeighbors));
		}
		if (bestImageScore < cachedNeighbors.size() ||
			(bestImageScore == cachedNeighbors.size() && bestImageSize > depthData.neighbors.size())) {
			bestImageScore = cachedNeighbors.size();
			bestImageSize = depthData.neighbors.size();
			bestImageIdx = idxImage;
		}
	}
	return std::make_tuple(bestImageIdx, bestImageScore, static_cast<unsigned>(cachedImages.size()));
} // FetchBestNextDMapIndex

// fuse all valid depth-maps in the same 3D point-cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
void DepthMapsData::FuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	struct Proj {
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, arrDepthData.size()));
	const float normalError(COS(D2R(OPTDENSE::fNormalDiffThreshold)));
	const IIndex numDMapsReserveFusion(10);
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(arrDepthData.size());
	const size_t nPointsEstimate(arrDepthData.size() * 9000); //TODO: better estimate number of points
	ProjsArr projs(0, nPointsEstimate);
	pointcloud.points.reserve(nPointsEstimate);
	pointcloud.pointViews.reserve(nPointsEstimate);
	pointcloud.pointWeights.reserve(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal) {
		pointcloud.normals.reserve(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}
	Util::Progress progress(_T("Fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	const FusionCacheSetup cacheSetup(arrDepthData, fusedDMaps, numDMapsReserveFusion, scene.images, bEstimateColor);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags, cacheSetup.cacheMemory, cacheSetup.pCachedImages);
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	IIndex numDMapsFused = 0;
	for (; numDMapsFused < arrDepthData.size(); ++numDMapsFused) {
		TD_TIMER_STARTD();
		// find the best depth-map to fuse next as the one with the most neighbors already in cache
		const auto [idxImage, numImageNeighborsInCache, numImagesInCache] = FetchBestNextDMapIndex(arrDepthData, cacheDMaps, fusedDMaps);
		if (idxImage == NO_ID)
			break; // no more depth-maps to fuse (only invalid depth-maps left)
		totalNumImageNeighborsInCache += numImageNeighborsInCache;
		totalNumImagesInCache += numImagesInCache;
		// fuse depth-map
		cacheDMaps.UseImage(idxImage);
		cacheDMaps.SkipMemoryCheckIdxImage(idxImage);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		ASSERT(!depthData.IsEmpty());
		if (bEstimateNormal && depthData.normalMap.empty())
			EstimateNormalMaps();
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		IIndex numNeighbors(0);
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		for (int64_t i=0; i<(int64_t)depthData.neighbors.size(); ++i) {
			const ViewScore& neighbor = depthData.neighbors[(IIndex)i];
		#else
		for (const ViewScore& neighbor: depthData.neighbors) {
		#endif
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			if (++numNeighbors >= OPTDENSE::nMaxViewsFuse)
				#ifdef DENSE_USE_OPENMP
				continue;
				#else
				break;
				#endif
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.ID];
			if (!depthIdxs.empty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		const Image& imageData = *depthData.images.front().pImageData;
		ASSERT(&imageData-scene.images.data() == idxImage);
		ASSERT(depthData.depthMap.size() == depthData.size && imageData.GetSize() == depthData.size);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];
		if (depthIdxs.empty()) {
			depthIdxs.create(depthData.size);
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.size());
		for (int i=0; i<depthData.size.height; ++i) {
			for (int j=0; j<depthData.size.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				idxPoint = (uint32_t)pointcloud.points.size();
				PointCloud::Point& point = pointcloud.points.emplace_back();
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				PointCloud::ViewArr& views = pointcloud.pointViews.emplace_back();
				views.emplace_back(idxImage);
				PointCloud::WeightArr& weights = pointcloud.pointWeights.emplace_back();
				REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap.empty() ? 1.f : depthData.confMap(x),depth)));
				ProjArr& pointProjs = projs.emplace_back();
				pointProjs.emplace_back(Proj(x));
				const PointCloud::Normal normal(!depthData.normalMap.empty() ? Cast<Normal::Type>(imageData.camera.R.t() * Cast<REAL>(depthData.normalMap(x))) : Normal(0, 0, -1));
				ASSERT(ISEQUAL(norm(normal), 1.f, 1e-2f), "Norm = ", norm(normal));
				// check the projection in the neighbor depth-maps
				Point3 X(point*confidence);
				// the pixels are resident only when colors are fused, and an image whose
				// decode failed stays empty: its points stay uncolored
				Pixel32F C(bEstimateColor && !imageData.image.empty() ?
					Pixel32F(Cast<float>(imageData.image(x))*confidence) : Pixel32F::BLACK);
				PointCloud::Normal N(normal*confidence);
				invalidDepths.clear();
				for (const ViewScore& neighbor: depthData.neighbors) {
					const IIndex idxImageB(neighbor.ID);
					DepthData& depthDataB = arrDepthData[idxImageB];
					if (depthDataB.IsEmpty())
						continue;
					const Image& imageDataB = scene.images[idxImageB];
					const auto [pt, depthProjB] = imageDataB.camera.ProjectPointP(point);
					if (depthProjB <= 0)
						continue;
					const ImageRef xB(ROUND2INT(pt));
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					if (idxPointB != NO_ID)
						continue;
					if (IsDepthSimilar(depthProjB, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// check if normals agree
						const PointCloud::Normal normalB(!depthData.normalMap.empty() ? Cast<Normal::Type>(imageDataB.camera.R.t() * Cast<REAL>(depthDataB.normalMap(xB))) : Normal(0, 0, -1));
						ASSERT(ISEQUAL(norm(normalB), 1.f, 1e-2f), "Norm = ", norm(normalB));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap.empty() ? 1.f : depthDataB.confMap(xB),depthB));
							const IIndex idx(views.InsertSort(idxImageB));
							weights.InsertAt(idx, confidenceB);
							pointProjs.InsertAt(idx, Proj(xB));
							idxPointB = idxPoint;
							X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
							if (bEstimateColor && !imageDataB.image.empty())
								C += Cast<float>(imageDataB.image(xB))*confidenceB;
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							continue;
						}
					}
					if (depthProjB < depthB) {
						// discard depth
						invalidDepths.emplace_back(&depthB);
					}
				}
				if (views.size() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					projs.pop_back();
					pointcloud.pointWeights.pop_back();
					pointcloud.pointViews.pop_back();
					pointcloud.points.pop_back();
				} else {
					// this point is valid, store it
					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));
					if (bEstimateColor)
						pointcloud.colors.emplace_back((C*(float)nrm).cast<uint8_t>());
					if (bEstimateNormal)
						pointcloud.normals.emplace_back(normalized(N*(float)nrm));
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		fusedDMaps[idxImage] = true;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size() && pointcloud.points.size() == pointcloud.pointWeights.size() && pointcloud.points.size() == projs.size());
		DEBUG_ULTIMATE("Depth-map for reference image %3u fused using %u depth-maps: %u new points, %u/%u cached images (%s)",
			idxImage, depthData.images.size()-1, pointcloud.points.size()-nNumPointsPrev, numImageNeighborsInCache, numImagesInCache, TD_TIMER_GET_FMT().c_str());
		progress.display(numDMapsFused);
		// ensure enough memory is available for the next depth-maps chunk
		cacheDMaps.SkipMemoryCheckIdxImage();
		if (numDMapsFused % numDMapsReserveFusion == 0)
			cacheDMaps.SetMaxMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion, cacheDMaps.GetUsedMemory()));
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();
	cacheDMaps.ClearCache();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%), %.2f hits in %.2f cached (%s)",
		numDMapsFused, nDepths, pointcloud.points.size(), ROUND2INT((100.f*pointcloud.points.size())/nDepths),
		static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused,
		static_cast<double>(totalNumImagesInCache) / numDMapsFused, TD_TIMER_GET_FMT().c_str());

	if (bEstimateNormal && !pointcloud.points.empty() && pointcloud.normals.empty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		TD_TIMER_STARTD();
		pointcloud.normals.resize(pointcloud.points.size());
		const int64_t nPoints((int64_t)pointcloud.points.size());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.empty());
			IIndex idxView(0);
			float bestWeight = weights.front();
			for (IIndex idx=1; idx<weights.size(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
} // FuseDepthMaps


// fuse all valid depth-maps in the same 3D point-cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
void DepthMapsData::DenseFuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool _bEstimateNormal)
{
	TD_TIMER_STARTD();

	typedef SEACAVE::BitMatrix UseMask;
	typedef CLISTDEFIDX(UseMask,IIndex) UseMaskArr;

	// fuse all depth-maps, processing the best connected images first
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, arrDepthData.size()));
	const float normalError(COS(D2R(OPTDENSE::fNormalDiffThreshold)));
	const float minConfidence(1.f - OPTDENSE::fNCCThresholdKeep);
	const float maxReprojErrorSq(SQUARE(OPTDENSE::fDepthReprojectionErrorThreshold));
	const IIndex numDMapsReserveFusion(10);
	const bool bEstimateNormal(true); // always estimate normals as they are needed for the fusion
	size_t nDepths(0);
	UseMaskArr arrUseMask(arrDepthData.size());
	const size_t nPointsEstimate(arrDepthData.size() * 9000); //TODO: better estimate number of points
	pointcloud.points.reserve(nPointsEstimate);
	pointcloud.pointViews.reserve(nPointsEstimate);
	pointcloud.pointWeights.reserve(nPointsEstimate);
	unsigned depthDataLoadFlags(HeaderDepthDataRaw::HAS_DEPTH | HeaderDepthDataRaw::HAS_CONF);
	if (bEstimateColor)
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal) {
		pointcloud.normals.reserve(nPointsEstimate);
		depthDataLoadFlags |= HeaderDepthDataRaw::HAS_NORMAL;
	}
	Util::Progress progress(_T("Dense fused depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	BoolArr fusedDMaps(arrDepthData.size());
	fusedDMaps.Memset(0);
	const FusionCacheSetup cacheSetup(arrDepthData, fusedDMaps, numDMapsReserveFusion, scene.images, bEstimateColor);
	DMapCache cacheDMaps(arrDepthData, depthDataLoadFlags, cacheSetup.cacheMemory, cacheSetup.pCachedImages);
	unsigned totalNumImageNeighborsInCache = 0, totalNumImagesInCache = 0;
	BoolArr neighbors(arrDepthData.size());
	PointCloud::Point refPoint;
	PointCloud::Normal refNormal;
	CLISTDEF0IDX(float, unsigned) fusedPoints[3];
	PointCloud::ViewArr fusedViews;
	FloatArr fusedWeights;
	Point3d fusedNormal;
	Pixel32F fusedColor;
	const auto FusePoint = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth) -> void {
		const auto lambda = [&](IIndex ID, const ImageRef& x, unsigned fuseDepth, const auto& FusePointImpl) -> void {
			const DepthData& depthData = arrDepthData[ID];
			if (!Image8U::isInside(x, depthData.size))
				return;
			// ignore pixel if not estimated
			ASSERT(depthData.depthMap.size() == depthData.size);
			const Depth depth = depthData.depthMap(x);
			if (depth <= Depth(0))
				return;
			ASSERT(ISINSIDE(depth, depthData.dMin * 0.95f, depthData.dMax * 1.05f));
			// ignore pixel if already fused
			UseMask& useMask = arrUseMask[ID];
			if (useMask(x))
				return;
			// ignore pixel if not confident
			const float conf(depthData.confMap.empty() ? 1.f : depthData.confMap(x));
			if (conf < minConfidence)
				return;
			const DepthData::ViewData& image = depthData.GetView();
			// if the fusion depth is greater than zero, the initial reference pixel
			// has already been added and we need to check for consistency
			PointCloud::Normal normal;
			if (fuseDepth > 0) {
				// project reference point into current view
				const auto [pt, depthProj] = image.camera.ProjectPointP(refPoint);
				// check if depth agrees with current depth
				ASSERT(depthProj > Depth(0) || !IsDepthSimilar(depth, depthProj, OPTDENSE::fDepthDiffThreshold));
				if (!IsDepthSimilar(depth, depthProj, OPTDENSE::fDepthDiffThreshold))
					return;
				// check reprojection error of the reference point in the current view
				const Point2f diff(pt - Cast<float>(x));
				if (normSq(diff) > maxReprojErrorSq)
					return;
				// check if normals agree
				normal = image.camera.R.t() * Cast<REAL>(depthData.normalMap(x));
				ASSERT(ISEQUAL(norm(normal), 1.f, 1e-2f), "Norm = ", norm(normal));
				if (refNormal.dot(normal) < normalError)
					return;
			} else {
				normal = image.camera.R.t() * Cast<REAL>(depthData.normalMap(x));
				ASSERT(ISEQUAL(norm(normal), 1.f, 1e-2f), "Norm = ", norm(normal));
			}
			// set the current pixel as visited
			useMask.set(x);
			// compute 3D location of the current depth
			const PointCloud::Point X(image.camera.TransformPointI2W(Point3(REAL(x.x), REAL(x.y), REAL(depth))));
			// accumulate statistics for fused point
			{
				fusedPoints[0].push_back(X(0));
				fusedPoints[1].push_back(X(1));
				fusedPoints[2].push_back(X(2));
				const float weight(Conf2Weight(conf, depth));
				const auto it(fusedViews.InsertSortUnique(ID));
				if (it.second)
					fusedWeights[it.first] += weight;
				else
					fusedWeights.InsertAt(it.first, weight);
				if (bEstimateNormal)
					fusedNormal += Cast<double>(normal);
				if (bEstimateColor && !image.pImageData->image.empty())
					fusedColor += Cast<float>(image.pImageData->image(x));
			}
			// remember the first pixel as the reference.
			if (fuseDepth == 0) {
				refPoint = X;
				refNormal = normal;
			}
			// do not traverse the graph infinitely in one branch and
			// limit the maximum number of pixels fused in one point
			// to avoid stack overflow
			if (++fuseDepth >= OPTDENSE::nMaxFuseDepth || fusedPoints[0].size() >= OPTDENSE::nMaxPointsFuse)
				return;
			// traverse the neighbors graph by projecting the point into other views
			for (const ViewScore& neighbor : image.pImageData->neighbors) {
				const IIndex nextID(neighbor.ID);
				ASSERT(nextID != ID);
				if (!neighbors[nextID])
					continue;
				const DepthData& nextDepthData = arrDepthData[nextID];
				const ImageRef nextx(ROUND2INT(std::get<0>(nextDepthData.GetCamera().ProjectPointP(X))));
				FusePointImpl(nextID, nextx, fuseDepth, FusePointImpl);
			}
		};
		lambda(ID, x, fuseDepth, lambda);
	};
	// loop over each depth-map
	IIndex numDMapsFused = 0;
	while (true) {
		TD_TIMER_STARTD();
		// find the best depth-map to fuse next as the one with the most neighbors already in cache
		const auto [idxImage, numImageNeighborsInCache, numImagesInCache] = FetchBestNextDMapIndex(arrDepthData, cacheDMaps, fusedDMaps);
		if (idxImage == NO_ID)
			break; // no more depth-maps to fuse (only invalid depth-maps left)
		totalNumImageNeighborsInCache += numImageNeighborsInCache;
		totalNumImagesInCache += numImagesInCache;
		++numDMapsFused;
		// fuse depth-map
		cacheDMaps.UseImage(idxImage);
		cacheDMaps.SkipMemoryCheckIdxImage(idxImage);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		ASSERT(!depthData.IsEmpty());
		if (bEstimateNormal && depthData.normalMap.empty())
			EstimateNormalMaps();
		// make sure all neighbors are cached
		neighbors.Memset(0);
		neighbors[idxImage] = true;
		IIndex numNeighbors(0);
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for
		for (int64_t i=0; i<(int64_t)depthData.neighbors.size(); ++i) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const ViewScore& neighbor = depthData.neighbors[(IIndex)i];
		#else
		for (const ViewScore& neighbor: depthData.neighbors) {
		#endif
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (!depthDataB.IsValid())
				continue;
			cacheDMaps.UseImage(neighbor.ID);
			if (depthDataB.IsEmpty())
				continue;
			neighbors[neighbor.ID] = true;
			UseMask& useMask = arrUseMask[neighbor.ID];
			if (!useMask.empty())
				continue;
			useMask.create(depthDataB.depthMap.size());
			useMask.memset(0);
			if (++numNeighbors >= OPTDENSE::nMaxViewsFuse) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				#else
				break;
				#endif
			}
		}
		ASSERT(!depthData.IsEmpty());
		MAYBEUNUSED const Image& imageData = *depthData.images.front().pImageData;
		ASSERT(&imageData-scene.images.data() == idxImage);
		ASSERT(depthData.depthMap.size() == depthData.size && imageData.GetSize() == depthData.size);
		UseMask& useMask = arrUseMask[idxImage];
		if (useMask.empty()) {
			useMask.create(depthData.size);
			useMask.memset(0);
		}
		// try to fuse each depth estimate
		const size_t nNumPointsPrev(pointcloud.points.size());
		for (int i=0; i<depthData.size.height; ++i) {
			for (int j=0; j<depthData.size.width; ++j) {
				FusePoint(idxImage, ImageRef(j,i), 0);
				if (fusedPoints[0].size() >= OPTDENSE::nMinPixelsFuse && fusedViews.size() >= nMinViewsFuse) {
					// create the corresponding 3D point
					pointcloud.points.emplace_back(
						fusedPoints[0].GetMedian(),
						fusedPoints[1].GetMedian(),
						fusedPoints[2].GetMedian()
					);
					ASSERT(fusedViews.size() == fusedWeights.size());
					PointCloud::WeightArr& weights = pointcloud.pointWeights.AddEmpty();
					for (float weight: fusedWeights)
						weights.push_back(weight);
					pointcloud.pointViews.emplace_back(fusedViews);
					if (bEstimateNormal)
						pointcloud.normals.emplace_back(normalized(fusedNormal));
					if (bEstimateColor)
						pointcloud.colors.emplace_back((fusedColor/static_cast<float>(fusedPoints[0].size())).cast<uint8_t>());
				}
				if (!fusedViews.empty()) {
					nDepths += fusedViews.size();
					fusedPoints[0].clear();
					fusedPoints[1].clear();
					fusedPoints[2].clear();
					fusedViews.clear();
					fusedWeights.clear();
					fusedNormal = Point3d::ZERO;
					fusedColor = Pixel32F::BLACK;
				}
			}
		}
		fusedDMaps[idxImage] = true;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size() && pointcloud.points.size() == pointcloud.pointWeights.size());
		DEBUG_ULTIMATE("Depth-map for reference image %3u fused using %u depth-maps: %u new points, %u/%u cached images (%s)",
			idxImage, depthData.images.size() - 1, pointcloud.points.size() - nNumPointsPrev, numImageNeighborsInCache, numImagesInCache, TD_TIMER_GET_FMT().c_str());
		progress.display(numDMapsFused);
		// ensure enough memory is available for the next depth-maps chunk
		cacheDMaps.SkipMemoryCheckIdxImage();
		if (numDMapsFused % numDMapsReserveFusion == 0)
			cacheDMaps.SetMaxMemory(GetAvailableMemory(arrDepthData, fusedDMaps, numDMapsReserveFusion, cacheDMaps.GetUsedMemory()));
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrUseMask.Release();
	cacheDMaps.ClearCache();
	if (!_bEstimateNormal)
		pointcloud.normals.Release();

	DEBUG_EXTRA("Depth-maps dense fused and filtered: %u depth-maps, %u depths, %u points (%d%%), %.2f hits in %.2f cached (%s)",
		numDMapsFused, nDepths, pointcloud.points.size(), ROUND2INT((100.f*pointcloud.points.size())/nDepths),
		static_cast<double>(totalNumImageNeighborsInCache) / numDMapsFused,
		static_cast<double>(totalNumImagesInCache) / numDMapsFused, TD_TIMER_GET_FMT().c_str());
} // DenseFuseDepthMaps
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

DenseDepthMapData::DenseDepthMapData(Scene& _scene, int _nFusionMode, float _fSampleMeshNeighbors) :
	scene(_scene), depthMaps(_scene), idxImage(0), sem(1), nEstimationGeometricIter(-1),
	nFusionMode(_nFusionMode), fSampleMeshNeighbors(_fSampleMeshNeighbors), nClosing(0), nDenseWorkers(2u)
{
	if (nFusionMode < 0) {
		STEREO::SemiGlobalMatcher::CreateThreads(scene.nMaxThreads);
		if (nFusionMode == -1)
			OPTDENSE::nOptimize = 0;
	}
}
DenseDepthMapData::~DenseDepthMapData()
{
	if (nFusionMode < 0)
		STEREO::SemiGlobalMatcher::DestroyThreads();
}

void DenseDepthMapData::SignalCompleteDepthmapFilter()
{
	ASSERT(idxImage > 0);
	if (Thread::safeDec(idxImage) == 0)
		sem.Signal((unsigned)images.GetSize()*2);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);

bool Scene::DenseReconstruction(int nFusionMode, bool bCrop2ROI, float fBorderROI, float fSampleMeshNeighbors)
{
	DenseDepthMapData data(*this, nFusionMode, fSampleMeshNeighbors);

	// estimate depth-maps
	if (!ComputeDepthMaps(data))
		return false;
	if (ABS(nFusionMode) == 1)
		return true;


	// fuse all depth-maps
	pointcloud.Release();
	switch (OPTDENSE::nFuseFilter) {
	case OPTDENSE::FUSE_NOFILTER:
		// merge depth-maps
		data.depthMaps.MergeDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
		break;
	case OPTDENSE::FUSE_FILTER:
		// fuse depth-maps
		data.depthMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
		break;
	case OPTDENSE::FUSE_DENSEFILTER:
		// dense fuse depth-maps
		data.depthMaps.DenseFuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	}
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2) {
		// print number of points with 3+ views
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		FOREACHPTR(pViews, pointcloud.pointViews) {
			switch (pViews->GetSize())
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif

	if (!pointcloud.IsEmpty()) {
		if (bCrop2ROI && IsBounded()) {
			TD_TIMER_START();
			const size_t numPoints = pointcloud.GetSize();
			const OBB3f ROI(fBorderROI == 0 ? obb : (fBorderROI > 0 ? OBB3f(obb).EnlargePercent(fBorderROI) : OBB3f(obb).Enlarge(-fBorderROI)));
			pointcloud.RemovePointsOutside(ROI);
			VERBOSE("Point-cloud trimmed to ROI: %u points removed (%s)",
				numPoints-pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
		}
		if (pointcloud.colors.IsEmpty() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (pointcloud.normals.IsEmpty() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}

	if (OPTDENSE::bRemoveDmaps) {
		// delete all depth-map files
		FOREACH(i, images) {
			const DepthData& depthData = data.depthMaps.arrDepthData[i];
			if (!depthData.IsValid())
				continue;
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		}
	}
	return true;
} // DenseReconstruction
/*----------------------------------------------------------------*/

// number of depth-map estimation workers that fit in the memory left after the
// images were loaded; every worker keeps a whole DepthData alive while its
// depth-map is estimated, so the requested pool size is only an upper bound
//
// a worker accounts for two initialized DepthData, not one: each of them queues
// the next image before estimating its own, so the depth-data being prepared and
// the one being estimated are alive at the same time. Per DepthData:
//  - the gray image of the reference and of each of its neighbors
//  - the neighbor depth-maps, loaded during the geometric-consistency passes
//  - the reference depth, normal, confidence and views maps
// and, once per worker, the backend staging buffers mirroring the images, the
// neighbor depth-maps and the packed depth+normal estimates
unsigned DenseWorkerPoolSize(unsigned requested, unsigned maxWorkers,
	const ImageArr& sceneImages, const IIndexArr& images, const DepthDataArr& arrDepthData,
	size_t reservedMemory)
{
	ASSERT(requested > 0 && maxWorkers > 0);
	size_t area(0);
	IIndex numViews(0);
	for (IIndex idxImage: images) {
		const DepthData& depthData = arrDepthData[idxImage];
		if (depthData.neighbors.IsEmpty())
			continue;
		area = MAXF(area, (size_t)sceneImages[idxImage].GetSize().area());
		// nNumViews is 0 when all the neighbor views are to be used
		numViews = MAXF(numViews, (OPTDENSE::nNumViews ?
			MINF(depthData.neighbors.GetSize(), OPTDENSE::nNumViews) :
			depthData.neighbors.GetSize()) + 1);
	}
	if (area == 0 || numViews < 2)
		return MINF(requested, maxWorkers);
	const size_t sizeEstimate(4*sizeof(float)); // packed depth+normal, as the backends stage it
	const size_t hostImages(numViews * area * sizeof(float));
	const size_t hostDepthMaps((numViews - 1) * area * sizeof(float));
	const size_t hostRefMaps(area * (sizeof(Depth) + sizeof(Normal) + sizeof(float) + sizeof(ViewsID)));
	const size_t hostStaging(area * sizeEstimate + hostImages + hostDepthMaps);
	const size_t hostWorker(2 * (hostImages + hostDepthMaps + hostRefMaps) + hostStaging);
	// what is free now, minus what the image cache may still fill with the images
	// it has not decoded yet, is what the workers may share; leave the same
	// safety margin the depth-map cache uses, both for the fusion that follows
	// and for the file cache absorbing the depth-map traffic the estimation generates
	const Util::MemoryInfo memInfo(Util::GetMemoryInfo());
	const size_t safetyMemory(ComputeSafetyMemory(memInfo) + reservedMemory);
	const size_t freeMemory(memInfo.freePhysical > safetyMemory ? memInfo.freePhysical - safetyMemory : 0);
	unsigned poolSize(MINF(requested, maxWorkers));
	const unsigned hostWorkers(MAXF((unsigned)(freeMemory / hostWorker), 1u));
	if (hostWorkers < poolSize) {
		VERBOSE("Depth-map estimation limited to %u workers (%u requested): %.1fGB free, %.1fGB needed per worker",
			hostWorkers, poolSize, (double)freeMemory/(1024*1024*1024), (double)hostWorker/(1024*1024*1024));
		poolSize = hostWorkers;
	}
	#ifdef _USE_CUDA
	// same for the device: each worker owns the image and depth textures, the
	// depth+normal estimates, their costs, the selected views and the RNG states
	size_t freeDevice(0), totalDevice(0);
	if (cudaMemGetInfo(&freeDevice, &totalDevice) == cudaSuccess) {
		const size_t deviceWorker(area * (
			numViews * sizeof(float)/*image arrays*/ +
			(numViews - 1) * sizeof(float)/*depth arrays*/ +
			sizeEstimate/*estimates*/ + sizeof(float)/*costs*/ +
			sizeof(unsigned)/*selected views*/ + sizeof(curandState)));
		const unsigned deviceWorkers(MAXF((unsigned)(freeDevice * 4 / 5 / deviceWorker), 1u));
		if (deviceWorkers < poolSize) {
			VERBOSE("Depth-map estimation limited to %u workers (%u requested): %.1fGB free on device, %.1fGB needed per worker",
				deviceWorkers, poolSize, (double)freeDevice/(1024*1024*1024), (double)deviceWorker/(1024*1024*1024));
			poolSize = deviceWorkers;
		}
	}
	#endif // _USE_CUDA
	return poolSize;
} // DenseWorkerPoolSize
/*----------------------------------------------------------------*/

// order the images so that the depth-maps estimated one after the other are
// computed from as many common views as possible
//
// Estimating a depth-map reads the intensities of its reference image and of each of
// its neighbor views, which the image cache keeps for as long as its budget allows,
// so how many images have to be decoded again is decided entirely by the order the
// references come in. The order they are stored in carries no such property: even a
// sequential capture pairs an image with views far from it in the file order (an
// orbit closing on itself, a flight passing over the same ground again), and an
// unordered collection has no meaningful order at all. Walk the view graph greedily
// instead, always taking next the image sharing the most views with the one just
// taken, so an image decoded once serves as many consecutive depth-maps as it can
// before it ages out of the cache.
//
// Scenes whose images all fit in the cache are unaffected, nothing being ever
// ejected, and the estimation result does not depend on the order: a depth-map is
// computed from the images and, in the geometric passes, from the depth-maps the
// previous pass wrote, never from a depth-map of the pass it belongs to.
void SortImagesByViewLocality(const DepthDataArr& arrDepthData, IIndexArr& images)
{
	const IIndex numImages(images.size());
	if (numImages < 3)
		return;
	// the views each depth-map is estimated from: the image itself and the neighbors
	// InitViews keeps of it
	CLISTDEF2IDX(IIndexArr,IIndex) views(numImages);
	CLISTDEF2IDX(IIndexArr,IIndex) usedBy(arrDepthData.size());
	FOREACH(i, images) {
		const IIndex idxImage(images[i]);
		const ViewScoreArr& neighbors = arrDepthData[idxImage].neighbors;
		ASSERT(!neighbors.empty());
		IIndexArr& viewsImage = views[i];
		viewsImage.push_back(idxImage);
		usedBy[idxImage].push_back(i);
		const float fMinScore(MAXF(neighbors.First().score*OPTDENSE::fViewMinScoreRatio, OPTDENSE::fViewMinScore));
		for (const ViewScore& neighbor: neighbors) {
			if ((OPTDENSE::nNumViews && viewsImage.size() > OPTDENSE::nNumViews) ||
				neighbor.score < fMinScore)
				break;
			viewsImage.push_back(neighbor.ID);
			usedBy[neighbor.ID].push_back(i);
		}
	}
	// walk the graph, scoring the images left by the number of views they share with
	// the one just taken
	BoolArr scheduled(numImages);
	scheduled.Memset(0);
	IIndexArr scores(numImages);
	scores.Memset(0);
	IIndexArr order(0, numImages), touched;
	IIndex idxNext(0), idxFirstLeft(0);
	for (IIndex n=0; n<numImages; ++n) {
		scheduled[idxNext] = true;
		order.push_back(images[idxNext]);
		if (n+1 == numImages)
			break;
		touched.Empty();
		for (IIndex idxView: views[idxNext]) {
			for (IIndex i: usedBy[idxView]) {
				if (scheduled[i])
					continue;
				if (scores[i]++ == 0)
					touched.push_back(i);
			}
		}
		IIndex idxBest(NO_ID), bestScore(0);
		for (IIndex i: touched) {
			if (bestScore < scores[i] || (bestScore == scores[i] && idxBest > i)) {
				bestScore = scores[i];
				idxBest = i;
			}
			scores[i] = 0;
		}
		if (idxBest == NO_ID) {
			// every image sharing a view with this one is estimated already;
			// continue with the first image left, starting the walk over in whatever
			// part of the scene it belongs to
			while (scheduled[idxFirstLeft])
				++idxFirstLeft;
			idxBest = idxFirstLeft;
		}
		idxNext = idxBest;
	}
	ASSERT(order.size() == numImages);
	images = std::move(order);
} // SortImagesByViewLocality
/*----------------------------------------------------------------*/

// do first half of dense reconstruction: depth map computation
// results are saved to "data"
bool Scene::ComputeDepthMaps(DenseDepthMapData& data)
{
	// compute point-cloud from the existing mesh
	if (!mesh.IsEmpty() && !ImagesHaveNeighbors()) {
		SampleMeshWithVisibility(static_cast<REAL>(data.fSampleMeshNeighbors));
		mesh.Release();
	}

	// if no geometry available, estimate neighbor views based on image pairs baseline
	if (IsEmpty() && !ImagesHaveNeighbors()) {
		VERBOSE("warning: empty point-cloud, rough neighbor views selection based on image pairs baseline");
		EstimateNeighborViewsPointCloud();
	}

	{
	// maps global view indices to our list of views to be processed
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());
		imagesMap.Resize(images.GetSize());
		imagesMap.MemsetValue(NO_ID);
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {
		#endif
			// skip invalid, uncalibrated or discarded images
			Image& imageData = images[idxImage];
			if (!imageData.IsValid())
				continue;
			// reload image at the appropriate resolution; the depth-map estimation
			// reads the images through the image cache, which decodes them on demand,
			// so only their resolution is resolved here and the pixels are left for
			// later -- the SGM fusion modes instead work directly on the color images
			unsigned nResolutionLevel(OPTDENSE::nResolutionLevel);
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(nResolutionLevel, OPTDENSE::nMinResolution, OPTDENSE::nMaxResolution));
			if (!imageData.ReloadImage(nMaxResolution, data.nFusionMode < 0)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			imageData.UpdateCamera(platforms);
			// print image camera
			DEBUG_ULTIMATE("K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort) {
			VERBOSE("error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		#endif
		// collect the images to be processed; the loop above cannot do it as the
		// order it completes its iterations in is arbitrary, while the estimation
		// walks this list in order
		FOREACH(idxImage, images) {
			if (!images[idxImage].IsValid())
				continue;
			imagesMap[idxImage] = data.images.GetSize();
			data.images.Insert(idxImage);
		}
		if (data.images.IsEmpty()) {
			VERBOSE("error: preparing images for dense reconstruction failed (no valid image)");
			return false;
		}
		VERBOSE("Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// select images to be used for dense reconstruction
	{
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (OPTDENSE::fWeightPointInsideROI > 0 && IsBounded()) {
			VERBOSE("Select neighbor views by weighting inside ROI points with %.2f", OPTDENSE::fWeightPointInsideROI);
		}
		#endif
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		IIndexArr invalidIDs;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
		#else
		FOREACH(idx, data.images) {
		#endif
			const IIndex idxImage(data.images[idx]);
			ASSERT(imagesMap[idxImage] != NO_ID);
			DepthData& depthData(data.depthMaps.arrDepthData[idxImage]);
			if (!data.depthMaps.SelectViews(depthData)) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				invalidIDs.InsertSort(idx);
			}
		}
		RFOREACH(i, invalidIDs) {
			const IIndex idx(invalidIDs[i]);
			imagesMap[data.images.Last()] = idx;
			imagesMap[data.images[idx]] = NO_ID;
			data.images.RemoveAt(idx);
		}
		ASSERT(!data.images.IsEmpty());
		VERBOSE("Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	}

	// estimate the depth-maps in the order reusing the decoded images best
	SortImagesByViewLocality(data.depthMaps.arrDepthData, data.images);

	// size the cache decoding the images on demand and fill it with the images the
	// estimation starts on; the SGM fusion modes keep working on the color images
	// loaded above and never ask it for anything
	if (data.nFusionMode >= 0) {
		TD_TIMER_START();
		ImageCache& imageCache = data.depthMaps.imageCache;
		imageCache.Reset(data.depthMaps.ComputeImageCacheMemory(data.images));
		if (!imageCache.Prefetch(data.images)) {
			VERBOSE("error: preparing images for dense reconstruction failed (errors decoding images)");
			return false;
		}
		const size_t allImagesMemory(ImageCache::ComputeMemorySize(images, data.images));
		if (imageCache.GetMaxMemory() == 0)
			VERBOSE("warning: not enough memory to cache the images (%luMB needed); each use decodes them again",
				allImagesMemory/1024/1024);
		else
			VERBOSE("Image cache filled with %u of %u images: %luMB budget, %luMB to hold them all (%s)",
				imageCache.GetNumImageReads(), data.images.GetSize(),
				imageCache.GetMaxMemory()/1024/1024, allImagesMemory/1024/1024, TD_TIMER_GET_FMT().c_str());
	}

	#if defined(_USE_CUDA) || defined(_USE_METAL)
	// One PatchMatch instance per worker thread; host-side prep (image upload,
	// depth-prior packing, result unpack) parallelizes across the worker pool while
	// the backend serializes the kernel launches as needed (CUDA via the cudaEvent_t
	// chain). The GPU backend (CUDA on Windows/Linux, Metal on Apple) is selected by
	// the shared --gpu-device param (-1 GPU, -2/cpu/empty CPU).
	if (!SEACAVE::CUDA::isCpuRequested(SEACAVE::CUDA::desiredDeviceIDs) && data.nFusionMode >= 0) {
		const unsigned poolSize = (nMaxThreads > 1)
			? DenseWorkerPoolSize(MAXF(OPTDENSE::nPatchMatchCUDAInstances, 1u), nMaxThreads, images, data.images, data.depthMaps.arrDepthData, data.depthMaps.imageCache.GetFreeMemory())
			: 1u;
		#ifdef _USE_CUDA
		const bool bAllocatedPool = data.depthMaps.AllocateCudaPool(poolSize);
		#else
		const bool bAllocatedPool = SEACAVE::METAL::isRuntimeAvailable() && data.depthMaps.AllocateMetalPool(poolSize);
		if (!bAllocatedPool)
			VERBOSE("WARNING: Metal runtime health check failed; using CPU depth-map estimation");
		#endif
		if (bAllocatedPool) {
			// raise the in-flight semaphore so all pool workers can run
			// EstimateDepthMap concurrently
			data.sem.Clear(poolSize);
			data.nDenseWorkers = poolSize;
			#ifdef _USE_CUDA
			VERBOSE("Using CUDA compute backend for depth-map estimation (%u workers)", poolSize);
			#else
			VERBOSE("Using Metal compute backend for depth-map estimation (%u workers)", poolSize);
			#endif
		}
	}
	#endif // _USE_CUDA || _USE_METAL

	// initialize the queue of images to be processed
	const int nOptimize(OPTDENSE::nOptimize);
	if (OPTDENSE::nEstimationGeometricIters && data.nFusionMode >= 0)
		OPTDENSE::nOptimize = 0;
	data.idxImage = 0;
	data.nClosing = 0;
	ASSERT(data.events.IsEmpty());
	data.events.AddEvent(new EVTProcessImage(0));
	// start working threads
	data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
	GET_LOGCONSOLE().Pause();
	if (nMaxThreads > 1) {
		// data.nDenseWorkers is set to the CUDA pool size (or kept at the
		// constructor default of 2 for the CPU path) before we get here.
		cList<SEACAVE::Thread> threads(data.nDenseWorkers);
		FOREACHPTR(pThread, threads)
			pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
		FOREACHPTR(pThread, threads)
			pThread->join();
	} else {
		// single-thread execution
		DenseReconstructionEstimate((void*)&data);
	}
	GET_LOGCONSOLE().Play();
	// the balanced shutdown leaves the queue empty on success; anything left is a
	// genuine worker failure (e.g. a propagated EVTFail)
	if (!data.events.IsEmpty())
		return false;
	data.progress.Release();

	if (data.nFusionMode >= 0) {
		#ifdef _USE_CUDA
		if (!data.depthMaps.pmCUDAPool.empty() && OPTDENSE::nEstimationGeometricIters)
			data.depthMaps.ReinitCudaPoolForGeom();
		#endif // _USE_CUDA
		#ifdef _USE_METAL
		if (!data.depthMaps.pmMetalPool.empty() && OPTDENSE::nEstimationGeometricIters)
			data.depthMaps.ReinitMetalPoolForGeom();
		#endif // _USE_METAL
		while (++data.nEstimationGeometricIter < (int)OPTDENSE::nEstimationGeometricIters) {
			// initialize the queue of images to be geometric processed
			if (data.nEstimationGeometricIter+1 == (int)OPTDENSE::nEstimationGeometricIters)
				OPTDENSE::nOptimize = nOptimize;
			data.idxImage = 0;
			data.nClosing = 0;
			ASSERT(data.events.IsEmpty());
			data.events.AddEvent(new EVTProcessImage(0));
			// start working threads
			data.progress = new Util::Progress("Geometric-consistent estimated depth-maps", data.images.GetSize());
			GET_LOGCONSOLE().Pause();
			if (nMaxThreads > 1) {
				// same worker count as the depth-map phase
				cList<SEACAVE::Thread> threads(data.nDenseWorkers);
				FOREACHPTR(pThread, threads)
					pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
				FOREACHPTR(pThread, threads)
					pThread->join();
			} else {
				// single-thread execution
				DenseReconstructionEstimate((void*)&data);
			}
			GET_LOGCONSOLE().Play();
			if (!data.events.IsEmpty())
				return false;
			data.progress.Release();
			// replace raw depth-maps with the geometric-consistent ones
			for (IIndex idx: data.images) {
				const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
				if (!depthData.IsValid())
					continue;
				const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
				File::deleteFile(rawName);
				File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);
			}
		}
		data.nEstimationGeometricIter = -1;
	}
	// nothing reads the images any more, so give the memory they occupy back to
	// the depth-map caches of the filtering and the fusion that follow
	data.depthMaps.imageCache.Reset(0);

#ifdef _USE_DMAP_INSTRUMENTATION
	const int finalGeometricIteration(DMapFinalActiveGeometricIteration(data));
	const bool confidenceAdjustmentEnabled(
		(OPTDENSE::nOptimize & (OPTDENSE::ADJUST_CONFIDENCE | OPTDENSE::ADJUST_CONFIDENCE_FAST)) != 0);
	if (!OPTDENSE::strDMapInstrumentationDir.empty() &&
		(OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) == 0)
	{
		for (IIndex idxImage: data.images) {
			if (!DMapInstrumentImageEnabled(images[idxImage]))
				continue;
			DMapFilterResourcePlan resourcePlan;
			DMapFilterStorageLease storageLease;
			const bool summaryAvailable(PrepareDMapFilterInstrumentation(*this, idxImage,
				finalGeometricIteration, DMapFilterPixelCount(images[idxImage]), "postprocess_filters",
				resourcePlan, storageLease));
			WriteUnavailableDMapPostprocessArtifacts(*this, idxImage, finalGeometricIteration,
				summaryAvailable ? "final depth-map state was not resident (for example, a cached DMAP was reused)" :
				resourcePlan.reason.c_str(), resourcePlan);
			FinishDMapFilterInstrumentation(*this, idxImage, finalGeometricIteration,
				resourcePlan, storageLease, DMapArtifactUsage());
		}
	}
	if (!confidenceAdjustmentEnabled && !OPTDENSE::strDMapInstrumentationDir.empty()) {
		for (IIndex idxImage: data.images) {
			if (!DMapInstrumentImageEnabled(images[idxImage]))
				continue;
			DMapFilterResourcePlan resourcePlan;
			DMapFilterStorageLease storageLease;
			const bool summaryAvailable(PrepareDMapFilterInstrumentation(*this, idxImage,
				finalGeometricIteration, DMapFilterPixelCount(images[idxImage]), "confidence_adjustment",
				resourcePlan, storageLease));
			const DMapArtifactUsage usage(WriteDMapConfidenceAdjustmentArtifacts(*this, idxImage,
				finalGeometricIteration, NULL, NULL, NULL, NULL, NULL, NULL, resourcePlan,
				summaryAvailable ? "disabled" : "resource_unavailable",
				summaryAvailable ? NULL : resourcePlan.reason.c_str()));
			FinishDMapFilterInstrumentation(*this, idxImage, finalGeometricIteration,
				resourcePlan, storageLease, usage);
		}
	}
	if (confidenceAdjustmentEnabled) {
#else
	if ((OPTDENSE::nOptimize & (OPTDENSE::ADJUST_CONFIDENCE | OPTDENSE::ADJUST_CONFIDENCE_FAST)) != 0) {
#endif
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}
	return true;
} // ComputeDepthMaps
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point-cloud
void Scene::DenseReconstructionEstimate(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {
		case EVT_PROCESSIMAGE: {
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt);
			if (evtImage.idxImage >= data.images.size()) {
				if (nMaxThreads > 1) {
					// Work is exhausted. More than one worker can reach this branch
					// (each pulls a distinct safeInc'd index past the end), so don't
					// let every one of them broadcast: the first worker here (latch
					// 0->1) enqueues exactly one EVT_CLOSE per worker -- itself
					// included -- and every worker, including the ones in this branch,
					// then exits by consuming exactly one. The counts stay balanced,
					// so no orphaned EVT_CLOSE is left behind and a non-empty queue
					// after the join remains a reliable failure signal.
					if (Thread::safeInc(data.nClosing) == 1)
						for (unsigned k = 0; k < data.nDenseWorkers; ++k)
							data.events.AddEvent(new EVTClose);
					break; // loop back to consume our own EVT_CLOSE
				}
				return;
			}
			// select views to reconstruct the depth-map for this image
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			// cached .dmap is only reusable if it was written at the current
			// image resolution; a .dmap from a previous run at a different
			// resolution-level would propagate its (stale) size through
			// InitViews and violate the image/depthMap size invariant that
			// PatchMatch::EstimateDepthMap relies on. Peek the header (flags=0
			// reads only the metadata) and treat a size mismatch as if the
			// cache were missing so the image gets re-estimated cleanly.
			const auto isCachedDmapUsable = [&](IIndex idxImg) {
				const String path(ComposeDepthFilePath(data.scene.images[idxImg].ID, "dmap"));
				if (!File::access(path))
					return false;
				String storedImageFileName;
				IIndexArr storedIDs;
				cv::Size storedImageSize;
				KMatrix K; RMatrix R; CMatrix C;
				Depth dMin, dMax;
				DepthMap _d; NormalMap _n; ConfidenceMap _c; ViewsMap _v;
				if (!ImportDepthDataRaw(path, storedImageFileName, storedIDs, storedImageSize,
						K, R, C, dMin, dMax, _d, _n, _c, _v, 0))
					return false;
				return data.scene.images[idxImg].GetSize() == storedImageSize;
			};
			const bool depthmapComputed(data.nFusionMode < 0 || (data.nFusionMode >= 0 && data.nEstimationGeometricIter < 0 && isCachedDmapUsable(idx)));
			// initialize images pair: reference image and the best neighbor view
			ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
			#ifdef _USE_DMAP_INSTRUMENTATION
			if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews, !depthmapComputed, depthmapComputed ? -1 : (data.nEstimationGeometricIter >= 0 ? 1 : 0), data.nEstimationGeometricIter)) {
			#else
			if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews, !depthmapComputed, depthmapComputed ? -1 : (data.nEstimationGeometricIter >= 0 ? 1 : 0))) {
			#endif
				// process next image
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			// try to load already compute depth-map for this image
			if (depthmapComputed && data.nFusionMode >= 0) {
				if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
					if (!depthData.Load(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"))) {
						VERBOSE("error: invalid depth-map '%s'", ComposeDepthFilePath(depthData.GetView().GetID(), "dmap").c_str());
						exit(EXIT_FAILURE);
					}
					// optimize depth-map
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				}
				// process next image
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			} else {
				// estimate depth-map
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_ESTIMATEDEPTHMAP: {
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			data.sem.Wait();
			if (data.nFusionMode >= 0) {
				// extract depth-map using Patch-Match algorithm
				data.depthMaps.EstimateDepthMap(data.images[evtImage.idxImage], data.nEstimationGeometricIter);
			} else {
				// extract disparity-maps using SGM algorithm
				if (data.nFusionMode == -1) {
					data.sgm.Match(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews);
				} else {
					// fuse existing disparity-maps
					const IIndex idx(data.images[evtImage.idxImage]);
					DepthData& depthData(data.depthMaps.arrDepthData[idx]);
					data.sgm.Fuse(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews, 2, depthData.depthMap, depthData.confMap);
					if (OPTDENSE::nEstimateNormals == 2)
						EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
					depthData.dMin = ZEROTOLERANCE<float>(); depthData.dMax = FLT_MAX;
				}
			}
			data.sem.Signal();
			if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
				// optimize depth-map
				data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
			} else {
				// save depth-map
				data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			}
			break; }

		case EVT_OPTIMIZEDEPTHMAP: {
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
#ifdef _USE_DMAP_INSTRUMENTATION
			// Record every logical stage so disabled intermediate filters remain
			// explicit and the final filter transition is isolated unambiguously.
			const bool observePostprocess(DMapInstrumentImageEnabled(images[idx]));
			DMapFilterResourcePlan postprocessResourcePlan;
			DMapFilterStorageLease postprocessStorageLease;
			std::unique_ptr<DMapPostprocessObservation> postprocessObservation;
			if (observePostprocess && PrepareDMapFilterInstrumentation(*this, idx, data.nEstimationGeometricIter, DMapFilterPixelCount(images[idx], &depthData.depthMap), "postprocess_filters", postprocessResourcePlan, postprocessStorageLease)) {
				postprocessObservation = std::make_unique<DMapPostprocessObservation>(
				    *this, idx, data.nEstimationGeometricIter, depthData, postprocessResourcePlan);
			}
#endif
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (VERBOSITY_LEVEL > 3)
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "raw.png"), depthData.depthMap);
			#endif
			// apply filters
#ifdef _USE_DMAP_INSTRUMENTATION
			const bool removeSpecklesEnabled((OPTDENSE::nOptimize & OPTDENSE::REMOVE_SPECKLES) != 0);
			DMapStateSnapshot beforeRemoveSpeckles;
			if (postprocessObservation && removeSpecklesEnabled)
				beforeRemoveSpeckles = CaptureDMapState(depthData);
			bool removeSpecklesSuccess(false);
			if (removeSpecklesEnabled) {
				TD_TIMER_START();
				removeSpecklesSuccess = data.depthMaps.RemoveSmallSegments(depthData);
				if (removeSpecklesSuccess) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			if (postprocessObservation) {
				const DepthMap& depthBefore(removeSpecklesEnabled ? beforeRemoveSpeckles.depth : depthData.depthMap);
				const NormalMap& normalBefore(removeSpecklesEnabled ? beforeRemoveSpeckles.normal : depthData.normalMap);
				const ConfidenceMap& confidenceBefore(removeSpecklesEnabled ? beforeRemoveSpeckles.confidence : depthData.confMap);
				postprocessObservation->Record(0, "remove_speckles", removeSpecklesEnabled, removeSpecklesEnabled,
				                               removeSpecklesSuccess, {
				                                                          {"speckle_size", OPTDENSE::nSpeckleSize},
				                                                          {"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold * 0.7f},
				                                                          {"connectivity", 4},
				                                                      },
				                               depthBefore, normalBefore, confidenceBefore, depthData.depthMap, depthData.normalMap, depthData.confMap);
			}

			const bool fillGapsEnabled((OPTDENSE::nOptimize & OPTDENSE::FILL_GAPS) != 0);
			DMapStateSnapshot beforeFillGaps;
			if (postprocessObservation && fillGapsEnabled)
				beforeFillGaps = CaptureDMapState(depthData);
			bool fillGapsSuccess(false);
			if (fillGapsEnabled) {
				TD_TIMER_START();
				fillGapsSuccess = data.depthMaps.GapInterpolation(depthData);
				if (fillGapsSuccess) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			if (postprocessObservation) {
				const DepthMap& depthBefore(fillGapsEnabled ? beforeFillGaps.depth : depthData.depthMap);
				const NormalMap& normalBefore(fillGapsEnabled ? beforeFillGaps.normal : depthData.normalMap);
				const ConfidenceMap& confidenceBefore(fillGapsEnabled ? beforeFillGaps.confidence : depthData.confMap);
				postprocessObservation->Record(1, "fill_gaps", fillGapsEnabled, fillGapsEnabled,
				                               fillGapsSuccess, {
				                                                    {"maximum_gap_pixels", OPTDENSE::nIpolGapSize},
				                                                    {"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold * 2.5f},
				                                                    {"passes", nlohmann::json::array({"rows", "columns"})},
				                                                },
				                               depthBefore, normalBefore, confidenceBefore, depthData.depthMap, depthData.normalMap, depthData.confMap);
			}
			if (observePostprocess) {
				DMapArtifactUsage usage;
				if (postprocessObservation)
					usage = postprocessObservation->Write(depthData);
				else
					WriteUnavailableDMapPostprocessArtifacts(*this, idx, data.nEstimationGeometricIter,
					                                         postprocessResourcePlan.reason.c_str(), postprocessResourcePlan);
				FinishDMapFilterInstrumentation(*this, idx, data.nEstimationGeometricIter,
				                                postprocessResourcePlan, postprocessStorageLease, usage);
			}
#else
			if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) {
				TD_TIMER_START();
				if (data.depthMaps.RemoveSmallSegments(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) {
				TD_TIMER_START();
				if (data.depthMaps.GapInterpolation(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
#endif
			// save depth-map
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break; }

		case EVT_SAVEDEPTHMAP: {
			TD_TIMER_STARTD();
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
#ifdef _USE_DMAP_INSTRUMENTATION
			if ((OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) == 0 &&
				DMapInstrumentImageEnabled(images[idx]))
			{
				DMapFilterResourcePlan resourcePlan;
				DMapFilterStorageLease storageLease;
				DMapArtifactUsage usage;
				if (PrepareDMapFilterInstrumentation(*this, idx, data.nEstimationGeometricIter,
					DMapFilterPixelCount(images[idx], &depthData.depthMap), "postprocess_filters",
					resourcePlan, storageLease))
				{
					DMapPostprocessObservation observation(*this, idx, data.nEstimationGeometricIter,
						depthData, resourcePlan);
					observation.Record(0, "remove_speckles", false, false, false, {
						{"speckle_size", OPTDENSE::nSpeckleSize},
						{"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold*0.7f},
						{"connectivity", 4},
					}, depthData.depthMap, depthData.normalMap, depthData.confMap,
					depthData.depthMap, depthData.normalMap, depthData.confMap);
					observation.Record(1, "fill_gaps", false, false, false, {
						{"maximum_gap_pixels", OPTDENSE::nIpolGapSize},
						{"depth_similarity_threshold", OPTDENSE::fDepthDiffThreshold*2.5f},
						{"passes", nlohmann::json::array({"rows", "columns"})},
					}, depthData.depthMap, depthData.normalMap, depthData.confMap,
					depthData.depthMap, depthData.normalMap, depthData.confMap);
					usage = observation.Write(depthData);
				} else {
					WriteUnavailableDMapPostprocessArtifacts(*this, idx, data.nEstimationGeometricIter,
						resourcePlan.reason.c_str(), resourcePlan);
				}
				FinishDMapFilterInstrumentation(*this, idx, data.nEstimationGeometricIter,
					resourcePlan, storageLease, usage);
			}
#endif
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (VERBOSITY_LEVEL > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.png"), depthData.confMap);
				// the exported cloud is colored from the pixels, which the estimation
				// itself does not keep resident any more; decode them into a copy, as
				// other estimation workers read the shared image concurrently
				Image imageData(*depthData.images.First().pImageData);
				if (imageData.image.empty())
					imageData.ReloadImageAtPreparedResolution();
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "ply"), imageData, depthData.depthMap, depthData.normalMap);
				if (VERBOSITY_LEVEL > 4) {
					ExportNormalMap(ComposeDepthFilePath(depthData.GetView().GetID(), "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.pfm"));
				}
			}
			#endif
			// capture identifiers before Release wipes depthData state
			const IIndex viewID = depthData.GetView().GetID();
			const int dmRows = depthData.depthMap.rows;
			const int dmCols = depthData.depthMap.cols;
			// save compute depth-map for this image
			if (!depthData.depthMap.empty()) {
				if (!depthData.Save(ComposeDepthFilePath(viewID, data.nEstimationGeometricIter < 0 ? "dmap" : "geo.dmap")))
					exit(EXIT_FAILURE);
			}
			depthData.ReleaseImages();
			depthData.Release();
			data.progress->operator++();
			// per-image save timing (gated at -v 2 so the default-verbose run
			// is not dragged down by per-image logging when pool-size grows)
			DEBUG_ULTIMATE("Depth-map %3u saved: %dx%d (%s)", viewID,
				dmCols, dmRows, TD_TIMER_GET_FMT().c_str());
			break; }

		case EVT_CLOSE: {
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/

void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: {
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded
			depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			for (const ViewScore& neighbor: depthData.neighbors) {
				DepthData& depthDataPair = data.depthMaps.arrDepthData[neighbor.ID];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(depthDataPair.GetView().GetID(), "dmap")) == 0) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.push_back(neighbor.ID);
				if (idxNeighbors.size() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image
#ifdef _USE_DMAP_INSTRUMENTATION
			const bool adjustFastEnabled((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE_FAST) != 0);
			const bool adjustFullEnabled((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE) != 0);
			const bool adjustFastSucceeded(adjustFastEnabled && data.depthMaps.AdjustConfidenceFast(depthData, idxNeighbors));
			const bool adjustFullSucceeded(adjustFullEnabled && data.depthMaps.AdjustConfidence(depthData, idxNeighbors));
			const bool observeConfidenceAdjustment(DMapInstrumentImageEnabled(images[idx]));
			if (adjustFastSucceeded | adjustFullSucceeded) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			} else if (observeConfidenceAdjustment) {
				const int geometricIteration(DMapFinalActiveGeometricIteration(data));
				DMapFilterResourcePlan resourcePlan;
				DMapFilterStorageLease storageLease;
				const bool summaryAvailable(PrepareDMapFilterInstrumentation(*this, idx, geometricIteration,
				                                                             DMapFilterPixelCount(images[idx], &depthData.depthMap), "confidence_adjustment",
				                                                             resourcePlan, storageLease));
				const DMapArtifactUsage usage(WriteDMapConfidenceAdjustmentArtifacts(*this, idx,
				                                                                     geometricIteration, summaryAvailable ? &idxNeighbors : NULL,
				                                                                     summaryAvailable ? &depthData.depthMap : NULL,
				                                                                     summaryAvailable ? &depthData.confMap : NULL, NULL, NULL, NULL, resourcePlan,
				                                                                     summaryAvailable ? "failed" : "resource_unavailable",
				                                                                     summaryAvailable ? "configured confidence-adjustment method did not produce a readable output" : resourcePlan.reason.c_str()));
				FinishDMapFilterInstrumentation(*this, idx, geometricIteration,
				                                resourcePlan, storageLease, usage);
			}
#else
			if (((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE_FAST) != 0 && data.depthMaps.AdjustConfidenceFast(depthData, idxNeighbors)) |
				((OPTDENSE::nOptimize & OPTDENSE::ADJUST_CONFIDENCE) != 0 && data.depthMaps.AdjustConfidence(depthData, idxNeighbors))) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
#endif
			// unload referenced depth-maps
			for (IIndex idxNeighbor: idxNeighbors) {
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxNeighbor];
				depthDataPair.DecRef();
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();
			break; }

		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();
			// load filtered maps
			ConfidenceMap confMapFast, confMap;
#ifdef _USE_DMAP_INSTRUMENTATION
			const bool referenceLoaded(
			    depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) != 0);
			bool fastLoaded(false), fullLoaded(false);
			if (referenceLoaded) {
				fastLoaded = LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.fast.cmap"), confMapFast);
				fullLoaded = LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.cmap"), confMap);
			}
			const bool observeConfidenceAdjustment(DMapInstrumentImageEnabled(images[idx]));
			const int geometricIteration(DMapFinalActiveGeometricIteration(data));
			DMapFilterResourcePlan confidenceResourcePlan;
			DMapFilterStorageLease confidenceStorageLease;
			bool confidenceSummaryAvailable(false);
			IIndexArr observedNeighbors;
			if (observeConfidenceAdjustment) {
				confidenceSummaryAvailable = PrepareDMapFilterInstrumentation(*this, idx,
				                                                              geometricIteration, DMapFilterPixelCount(images[idx], referenceLoaded ? &depthData.depthMap : NULL), "confidence_adjustment",
				                                                              confidenceResourcePlan, confidenceStorageLease);
				if (confidenceSummaryAvailable) {
					observedNeighbors.Reserve(8);
					for (const ViewScore& neighbor : depthData.neighbors) {
						if (neighbor.ID >= data.depthMaps.arrDepthData.size() || !data.depthMaps.arrDepthData[neighbor.ID].IsValid())
							continue;
						observedNeighbors.push_back(neighbor.ID);
						if (observedNeighbors.size() == 8)
							break;
					}
				}
			}
			if (!referenceLoaded || !(fastLoaded | fullLoaded)) {
				if (observeConfidenceAdjustment) {
					const DMapArtifactUsage usage(WriteDMapConfidenceAdjustmentArtifacts(*this, idx,
					                                                                     geometricIteration,
					                                                                     confidenceSummaryAvailable ? &observedNeighbors : NULL,
					                                                                     confidenceSummaryAvailable && referenceLoaded ? &depthData.depthMap : NULL,
					                                                                     confidenceSummaryAvailable && referenceLoaded ? &depthData.confMap : NULL,
					                                                                     confidenceSummaryAvailable && fastLoaded ? &confMapFast : NULL,
					                                                                     confidenceSummaryAvailable && fullLoaded ? &confMap : NULL, NULL,
					                                                                     confidenceResourcePlan,
					                                                                     confidenceSummaryAvailable ? "failed" : "resource_unavailable",
					                                                                     confidenceSummaryAvailable ? "failed to reload the reference DMAP or adjusted confidence output" : confidenceResourcePlan.reason.c_str()));
					FinishDMapFilterInstrumentation(*this, idx, geometricIteration,
					                                confidenceResourcePlan, confidenceStorageLease, usage);
				}
#else
			if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0 ||
				!(LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.fast.cmap"), confMapFast) |
				  LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.cmap"), confMap)))
			{
#endif
				// signal error and terminate
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
#ifdef _USE_DMAP_INSTRUMENTATION
			ConfidenceMap confidenceInput, confidenceFastOutput, confidenceFullOutput;
			if (confidenceSummaryAvailable) {
				confidenceInput = depthData.confMap.clone();
				if (!confMapFast.empty())
					confidenceFastOutput = confMapFast.clone();
				if (!confMap.empty())
					confidenceFullOutput = confMap.clone();
			}
#endif
			if (!confMapFast.empty())
				File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.fast.cmap").c_str());
			if (!confMap.empty())
				File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "adjusted.cmap").c_str());
			if (confMapFast.empty()) {
				depthData.confMap = std::move(confMap);
			} else if (confMap.empty()) {
				depthData.confMap = std::move(confMapFast);
			} else {
				// set confidence-map as the best confidence if both confMap and congMapFast are set
				for (int r = 0; r<depthData.confMap.rows; ++r) {
					for (int c = 0; c<depthData.confMap.cols; ++c) {
						const float conf = confMap(r,c);
						const float confFast = confMapFast(r,c);
						depthData.confMap(r,c) = conf > 0 && confFast > 0 ? MAXF(conf, confFast) : 0.f;
					}
				}
				confMapFast.release();
				confMap.release();
			}
#ifdef _USE_DMAP_INSTRUMENTATION
			if (observeConfidenceAdjustment) {
				const DMapArtifactUsage usage(WriteDMapConfidenceAdjustmentArtifacts(*this, idx,
				                                                                     geometricIteration,
				                                                                     confidenceSummaryAvailable ? &observedNeighbors : NULL,
				                                                                     confidenceSummaryAvailable ? &depthData.depthMap : NULL,
				                                                                     confidenceSummaryAvailable ? &confidenceInput : NULL,
				                                                                     confidenceSummaryAvailable && !confidenceFastOutput.empty() ? &confidenceFastOutput : NULL,
				                                                                     confidenceSummaryAvailable && !confidenceFullOutput.empty() ? &confidenceFullOutput : NULL,
				                                                                     confidenceSummaryAvailable ? &depthData.confMap : NULL,
				                                                                     confidenceResourcePlan,
				                                                                     confidenceSummaryAvailable ? "completed" : "resource_unavailable",
				                                                                     confidenceSummaryAvailable ? NULL : confidenceResourcePlan.reason.c_str()));
				FinishDMapFilterInstrumentation(*this, idx, geometricIteration,
				                                confidenceResourcePlan, confidenceStorageLease, usage);
			}
#endif
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (VERBOSITY_LEVEL > 2) {
				DepthMap depthMap(depthData.depthMap.clone());
				NormalMap normalMap(depthData.normalMap.clone());
				FilterDepthMap(depthMap, normalMap, depthData.confMap);
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.png"), depthMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.ply"), *depthData.images.First().pImageData, depthMap, normalMap);
			}
			#endif
			// save filtered depth-map for this image
			if (!depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")))
				exit(EXIT_FAILURE);
			depthData.DecRef();
			data.progress->operator++();
			break; }

		case EVT_FAIL: {
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter
/*----------------------------------------------------------------*/

// filter point-cloud based on camera-point visibility intersections
void Scene::PointCloudFilter(int thRemove)
{
	TD_TIMER_STARTD();

	typedef TOctree<PointCloud::PointArr,PointCloud::Point::Type,3,uint32_t> Octree;
	struct Collector {
		typedef Octree::IDX_TYPE IDX;
		typedef PointCloud::Point::Type Real;
		typedef TCone<Real,3> Cone;
		typedef TSphere<Real,3> Sphere;
		typedef TConeIntersect<Real,3> ConeIntersect;

		Cone cone;
		const ConeIntersect coneIntersect;
		const PointCloud& pointcloud;
		IntArr& visibility;
		PointCloud::Index idxPoint;
		Real distance;
		int weight;
		#ifdef DENSE_USE_OPENMP
		uint8_t pcs[sizeof(CriticalSection)];
		#endif

		Collector(const Cone::RAY& ray, Real angle, const PointCloud& _pointcloud, IntArr& _visibility)
			: cone(ray, angle), coneIntersect(cone), pointcloud(_pointcloud), visibility(_visibility)
		#ifdef DENSE_USE_OPENMP
		{ new(pcs) CriticalSection; }
		~Collector() { reinterpret_cast<CriticalSection*>(pcs)->~CriticalSection(); }
		inline CriticalSection& GetCS() { return *reinterpret_cast<CriticalSection*>(pcs); }
		#else
		{}
		#endif
		inline void Init(PointCloud::Index _idxPoint, const PointCloud::Point& X, int _weight) {
			const Real thMaxDepth(1.02f);
			idxPoint =_idxPoint;
			const PointCloud::Point::EVec D((PointCloud::Point::EVec&)X-cone.ray.m_pOrig);
			distance = D.norm();
			cone.ray.m_vDir = D/distance;
			cone.maxHeight = MaxDepthDifference(distance, thMaxDepth);
			weight = _weight;
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
			return coneIntersect(Sphere(center, radius*Real(SQRT_3)));
		}
		inline void operator() (const IDX* idices, IDX size) {
			const Real thSimilar(0.01f);
			Real dist;
			FOREACHRAWPTR(pIdx, idices, size) {
				const PointCloud::Index idx(*pIdx);
				if (coneIntersect.Classify(pointcloud.points[idx], dist) == VISIBLE && !IsDepthSimilar(distance, dist, thSimilar)) {
					if (dist > distance)
						visibility[idx] += pointcloud.pointViews[idx].size();
					else
						visibility[idx] -= weight;
				}
			}
		}
	};
	typedef CLISTDEF2(Collector) Collectors;

	// create octree to speed-up search
	Octree octree(pointcloud.points, [](Octree::IDX_TYPE size, Octree::Type /*radius*/) {
		return size > 128;
	});
	IntArr visibility(pointcloud.GetSize()); visibility.Memset(0);
	Collectors collectors; collectors.reserve(images.size());
	FOREACH(idxView, images) {
		const Image& image = images[idxView];
		const Ray3f ray(Cast<float>(image.camera.C), Cast<float>(image.camera.Direction()));
		const float angle(float(image.ComputeFOV(0)/image.width));
		collectors.emplace_back(ray, angle, pointcloud, visibility);
	}

	// run all camera-point visibility intersections
	Util::Progress progress(_T("Point visibility checks"), pointcloud.GetSize());
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for //schedule(dynamic)
	for (int64_t i=0; i<(int64_t)pointcloud.GetSize(); ++i) {
		const PointCloud::Index idxPoint((PointCloud::Index)i);
	#else
	FOREACH(idxPoint, pointcloud.points) {
	#endif
		const PointCloud::Point& X = pointcloud.points[idxPoint];
		const PointCloud::ViewArr& views = pointcloud.pointViews[idxPoint];
		for (PointCloud::View idxView: views) {
			Collector& collector = collectors[idxView];
			#ifdef DENSE_USE_OPENMP
			Lock l(collector.GetCS());
			#endif
			collector.Init(idxPoint, X, (int)views.size());
			octree.Collect(collector, collector);
		}
		++progress;
	}
	progress.close();

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2) {
		// print visibility stats
		UnsignedArr counts(0, 64);
		for (int views: visibility) {
			if (views > 0)
				continue;
			while (counts.size() <= IDX(-views))
				counts.push_back(0);
			++counts[-views];
		}
		String msg;
		msg.reserve(64*counts.size());
		FOREACH(c, counts)
			if (counts[c])
				msg += String::FormatString("\n\t% 3u - % 9u", c, counts[c]);
		VERBOSE("Visibility lengths (%u points):%s", pointcloud.GetSize(), msg.c_str());
		// save outlier points
		PointCloud pc;
		RFOREACH(idxPoint, pointcloud.points) {
			if (visibility[idxPoint] <= thRemove) {
				pc.points.push_back(pointcloud.points[idxPoint]);
				pc.colors.push_back(pointcloud.colors[idxPoint]);
			}
		}
		pc.Save(MAKE_PATH("scene_dense_outliers.ply"));
	}
	#endif

	// filter points
	const size_t numInitPoints(pointcloud.GetSize());
	RFOREACH(idxPoint, pointcloud.points) {
		if (visibility[idxPoint] <= thRemove)
			pointcloud.RemovePoint(idxPoint);
	}

	DEBUG_EXTRA("Point-cloud filtered: %u/%u points (%d%%) (%s)", pointcloud.points.size(), numInitPoints, ROUND2INT((100.f*pointcloud.points.GetSize())/numInitPoints), TD_TIMER_GET_FMT().c_str());
} // PointCloudFilter
/*----------------------------------------------------------------*/

#pragma pop_macro("VERBOSE")
