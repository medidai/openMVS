/*
* PatchMatchCUDA.cpp
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

#include "Common.h"
#include "PatchMatchCUDA.h"
#include "PatchMatchAPDCUDA.h"
#include "PatchMatchDVPCUDA.h"
#include "PatchMatchDVPDepthEdgeCUDA.h"
#include "DepthMap.h"
#include "ConfidenceCUDA.h"
#include "../IO/json.hpp"

#include <filesystem>
#include <fstream>
#ifdef _USE_DMAP_INSTRUMENTATION
#include <unordered_map>
#include <unordered_set>
#endif

#ifdef _USE_CUDA


// D E F I N E S ///////////////////////////////////////////////////

#pragma push_macro("VERBOSE")
#undef VERBOSE
#define VERBOSE(...) LOG(lt, __VA_ARGS__)


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("PtchMtch"));

namespace MVS {

namespace CUDA {

bool LoadDVPDepthEdgeLabelMap(const String& fileName, Image16U& labels)
{
	labels.release();
	const cv::Mat decoded(cv::imread(fileName.c_str(), cv::IMREAD_UNCHANGED));
	if (decoded.empty() || decoded.type() != CV_16UC1)
		return false;
	decoded.copyTo(labels);
	return !labels.empty();
}
/*----------------------------------------------------------------*/

// Kernel-launch serializer for the CUDA backend. The kernels read cameras/params
// from module-global __constant__ memory (g_cameras / g_params, see
// PatchMatchCUDA.cu), which is shared by every PatchMatch instance on the
// device. Concurrent overlap would race the __constant__ writes against an
// in-flight kernel's reads.
//
// Strategy: a global cudaEvent_t chains worker N+1's kernels behind worker N's
// kernels on the GPU side. A tiny host mutex covers only the queueing sequence
// {wait-event, upload-cameras, queue-kernels, record-event} so the host releases
// after queueing (~1ms) rather than after kernel execution (~80-400ms). Each
// worker then cudaStreamSynchronize's its own stream outside the mutex before
// reading results into its per-instance pinned buffer.
namespace {
std::mutex g_patchMatchCudaMutex;
cudaEvent_t g_constMemReady = nullptr;
std::once_flag g_constMemEventInit;

struct DVPDepthEdgePrior {
	Image16U labels;
	String manifestPath;
	String labelMapPath;
	String stageName;
	String labelMapSHA256;
	unsigned schemaVersion = 0u;
	unsigned rawSourceWidth = 0u;
	unsigned rawSourceHeight = 0u;
	unsigned processingWidth = 0u;
	unsigned processingHeight = 0u;
};

const char* DVPDepthEdgeStageName(unsigned mode)
{
	switch (static_cast<DVPDepthEdgeMode>(mode)) {
	case DVPDepthEdgeMode::ROBERTS_REGIONS: return "roberts_regions";
	case DVPDepthEdgeMode::DAV2_PLANARIZED: return "dav2_planarized";
	case DVPDepthEdgeMode::ERODED: return "eroded";
	case DVPDepthEdgeMode::DILATED: return "dilated";
	case DVPDepthEdgeMode::PIXEL_REASSIGNED: return "pixel_reassigned";
	default: return "disabled";
	}
}

bool LoadDVPDepthEdgePrior(const String& root, unsigned depthID,
	const String& sourceImageName, const Image8U::Size& preparedSize, unsigned mode,
	DVPDepthEdgePrior& prior, String& error)
{
	prior = DVPDepthEdgePrior{};
	error.clear();
	if (!DVPDepthEdgeModeEnabled(mode))
		return true;
	try {
		const std::filesystem::path frameDir(
			std::filesystem::path(root.c_str()) / "frames" /
			String::FormatString("depth%04u", depthID).c_str());
		const std::filesystem::path manifestPath(frameDir / "manifest.json");
		std::ifstream stream(manifestPath);
		if (!stream) {
			error = String::FormatString("missing manifest '%s'", manifestPath.c_str());
			return false;
		}
		const nlohmann::json manifest(nlohmann::json::parse(stream, nullptr, false));
		if (manifest.is_discarded() || !manifest.is_object()) {
			error = String::FormatString("invalid JSON manifest '%s'", manifestPath.c_str());
			return false;
		}
		if (manifest.value("schema", std::string()) != "openmvs_dvp_depth_edge_prior" ||
			manifest.value("schema_version", 0u) != DVP_DEPTH_EDGE_SCHEMA_VERSION ||
			manifest.value("complete", false) != true ||
			manifest.value("depth_id", ~0u) != depthID)
		{
			error = String::FormatString("schema/identity mismatch in '%s'", manifestPath.c_str());
			return false;
		}
		const nlohmann::json source(manifest.value("source_image", nlohmann::json::object()));
		const nlohmann::json processing(manifest.value("processing_image", nlohmann::json::object()));
		const std::string expectedName(source.value("name", std::string()));
		const std::string actualName(std::filesystem::path(sourceImageName.c_str()).filename().string());
		const int rawWidth(source.value("width", -1));
		const int rawHeight(source.value("height", -1));
		const int processingWidth(processing.value("width", -1));
		const int processingHeight(processing.value("height", -1));
		if (expectedName.empty() || expectedName != actualName ||
			source.value("sha256", std::string()).size() != 64u ||
			processing.value("resize_interpolation", std::string()) != "cv::INTER_AREA" ||
			processing.value("geometry_contract", std::string()) !=
				"OpenMVS_Image_RecomputeMaxResolution_then_ReloadImage" ||
			ValidateDVPDepthEdgePriorGeometry(
				static_cast<uint32_t>(preparedSize.width),
				static_cast<uint32_t>(preparedSize.height),
				static_cast<uint32_t>(MAXF(rawWidth, 0)),
				static_cast<uint32_t>(MAXF(rawHeight, 0)),
				static_cast<uint32_t>(MAXF(processingWidth, 0)),
				static_cast<uint32_t>(MAXF(processingHeight, 0))) !=
				DVPDepthEdgePriorGeometryStatus::VALID)
		{
			error = String::FormatString(
				"source/prepared image contract mismatch (raw %dx%d, prior %dx%d, OpenMVS %dx%d) in '%s'",
				rawWidth, rawHeight, processingWidth, processingHeight,
				preparedSize.width, preparedSize.height, manifestPath.c_str());
			return false;
		}
		const char* stageName(DVPDepthEdgeStageName(mode));
		const nlohmann::json stages(manifest.value("stages", nlohmann::json::object()));
		const auto stageIt(stages.find(stageName));
		if (stageIt == stages.end() || !stageIt->is_object()) {
			error = String::FormatString("stage '%s' unavailable in '%s'", stageName, manifestPath.c_str());
			return false;
		}
		const nlohmann::json stage(*stageIt);
		const std::string relativeMap(stage.value("label_map", std::string()));
		const std::string mapSHA256(stage.value("sha256", std::string()));
		if (relativeMap.empty() || mapSHA256.size() != 64u) {
			error = String::FormatString("stage '%s' lacks a pinned label map", stageName);
			return false;
		}
		const std::filesystem::path mapPath(frameDir / relativeMap);
		if (!LoadDVPDepthEdgeLabelMap(mapPath.string(), prior.labels)) {
			error = String::FormatString("unable to load 16-bit label map '%s'", mapPath.c_str());
			return false;
		}
		if (prior.labels.size() != preparedSize) {
			error = String::FormatString(
				"label-map dimensions %dx%d do not match prepared image %dx%d in '%s'",
				prior.labels.width(), prior.labels.height(), preparedSize.width, preparedSize.height,
				mapPath.c_str());
			prior.labels.release();
			return false;
		}
		const nlohmann::json stats(stage.value("stats", nlohmann::json::object()));
		const uint64_t area(static_cast<uint64_t>(preparedSize.area()));
		const uint64_t expectedBoundaryPixels(stats.value("boundary_pixels", ~uint64_t(0)));
		const uint64_t expectedCoveredPixels(stats.value("covered_pixels", ~uint64_t(0)));
		if (expectedBoundaryPixels > area || expectedCoveredPixels != area-expectedBoundaryPixels) {
			error = String::FormatString("invalid stage pixel statistics in '%s'", manifestPath.c_str());
			prior.labels.release();
			return false;
		}
		uint64_t boundaryPixels(0u);
		for (int y=0; y<prior.labels.height(); ++y)
			for (int x=0; x<prior.labels.width(); ++x)
				boundaryPixels += prior.labels(y, x) == DVP_DEPTH_EDGE_BOUNDARY_LABEL;
		if (boundaryPixels != expectedBoundaryPixels) {
			error = String::FormatString(
				"decoded label-map statistics mismatch (expected %llu boundary/%llu covered, got %llu/%llu) in '%s'",
				static_cast<unsigned long long>(expectedBoundaryPixels),
				static_cast<unsigned long long>(expectedCoveredPixels),
				static_cast<unsigned long long>(boundaryPixels),
				static_cast<unsigned long long>(area-boundaryPixels), mapPath.c_str());
			prior.labels.release();
			return false;
		}
		prior.manifestPath = manifestPath.string();
		prior.labelMapPath = mapPath.string();
		prior.stageName = stageName;
		prior.labelMapSHA256 = mapSHA256;
		prior.schemaVersion = DVP_DEPTH_EDGE_SCHEMA_VERSION;
		prior.rawSourceWidth = static_cast<unsigned>(rawWidth);
		prior.rawSourceHeight = static_cast<unsigned>(rawHeight);
		prior.processingWidth = static_cast<unsigned>(processingWidth);
		prior.processingHeight = static_cast<unsigned>(processingHeight);
		return true;
	} catch (const std::exception& exception) {
		error = String::FormatString("depth-edge prior validation failed: %s", exception.what());
		prior = DVPDepthEdgePrior{};
		return false;
	}
}

#ifdef _USE_DMAP_INSTRUMENTATION
struct InstrumentTracePixel {
	int imageID = -1;
	int x = -1;
	int y = -1;
	String label;
};

static constexpr size_t PM_INSTRUMENT_MAX_TRACE_PIXELS_PER_LEVEL = 4096;
static constexpr size_t PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES_PER_LEVEL = 1024u * 1024u;
static constexpr size_t PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES = 4096;
static constexpr size_t PM_INSTRUMENT_MAX_TRACE_CONFIG_PIXELS = 65536;
static constexpr size_t PM_INSTRUMENT_MAX_TRACE_CONFIG_LABEL_BYTES = 16u * 1024u * 1024u;

struct InstrumentTraceSelection {
	size_t configIndex = 0;
	int x = -1;
	int y = -1;
};

struct InstrumentTraceSelectionResult {
	std::vector<InstrumentTraceSelection> pixels;
	uint64_t labelBytes = 0;
};

struct InstrumentConfig {
	bool loaded = false;
	std::vector<InstrumentTracePixel> tracePixels;
};

constexpr uint64_t DMAP_EXACT_BASE_STATE_BYTES_PER_PIXEL =
	8u * sizeof(float) +
	7u * sizeof(float) +
	3u * 3u * sizeof(uint8_t) +
	2u * 4u * sizeof(uint8_t);
static_assert(DMAP_EXACT_BASE_STATE_BYTES_PER_PIXEL == 77u,
	"exact map resource accounting must match the exported per-state payload");
constexpr uint64_t DMAP_EXACT_BASE_STATE_ARTIFACTS = 20u;
constexpr uint64_t DMAP_EXACT_VIEW_ARTIFACTS = 5u;
constexpr uint64_t DMAP_APD_MAP_STORAGE_BYTES_PER_PIXEL =
	23u * sizeof(float) + 22u * sizeof(uint8_t);
constexpr uint64_t DMAP_APD_MAP_ARTIFACTS_PER_ITERATION = 45u;
constexpr uint64_t DMAP_APD_MULTISCALE_MAP_STORAGE_BYTES_PER_PIXEL = 6u * sizeof(uint8_t);
constexpr uint64_t DMAP_APD_MULTISCALE_MAP_ARTIFACTS = 6u;
constexpr uint64_t DMAP_DVP_MAP_STORAGE_BYTES_PER_PIXEL = 149u;
constexpr uint64_t DMAP_DVP_MAP_ARTIFACTS_PER_ITERATION = 56u;
constexpr uint64_t DMAP_DVP_VISIBILITY_MAP_STORAGE_BYTES_PER_PIXEL = 74u;
constexpr uint64_t DMAP_DVP_VISIBILITY_MAP_ARTIFACTS_PER_ITERATION = 39u;
constexpr uint64_t DMAP_DVP_VISIBLE_NORMAL_MAP_STORAGE_BYTES_PER_PIXEL =
	sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate);
constexpr uint64_t DMAP_DVP_VISIBLE_NORMAL_MAP_ARTIFACTS_PER_ITERATION = 47u;
static_assert(DMAP_APD_MAP_STORAGE_BYTES_PER_PIXEL == 114u,
	"APD map resource accounting must match the exported per-iteration payload");
static_assert(DMAP_APD_MULTISCALE_MAP_STORAGE_BYTES_PER_PIXEL == 6u,
	"APD multiscale map accounting must match three transferred and three output byte maps");
static_assert(DMAP_DVP_MAP_STORAGE_BYTES_PER_PIXEL == 149u,
	"DVP map resource accounting must match the exported per-iteration payload");
static_assert(DMAP_DVP_VISIBILITY_MAP_STORAGE_BYTES_PER_PIXEL == 74u,
	"DVP visibility map accounting must match the exported per-iteration payload");
static_assert(DMAP_DVP_VISIBLE_NORMAL_MAP_STORAGE_BYTES_PER_PIXEL == 84u,
	"DVP visible-normal map accounting must match the dense update record");
constexpr uint64_t DMAP_INSTRUMENT_FIXED_HOST_BYTES = 256u * 1024u;
constexpr uint64_t DMAP_SUMMARY_HOST_SCRATCH_BYTES_PER_PIXEL = sizeof(float);
constexpr uint64_t DMAP_MAP_EXPORT_HOST_SCRATCH_BYTES_PER_PIXEL =
	4u * sizeof(float) + sizeof(uint8_t) + sizeof(float);
constexpr uint64_t DMAP_LEGACY_LOGICAL_STATE_STORAGE_BYTES_PER_PIXEL = 61u;
constexpr uint64_t DMAP_LEGACY_TERMINAL_BYTES_PER_PIXEL =
	4u * sizeof(float4) + 2u * sizeof(float) + sizeof(uint32_t) + sizeof(uint8_t) +
	sizeof(Point4) + sizeof(float);
static_assert(DMAP_MAP_EXPORT_HOST_SCRATCH_BYTES_PER_PIXEL == 21u,
	"map export peak accounting must match retained event maps and scalar-map scratch");
static_assert(DMAP_LEGACY_LOGICAL_STATE_STORAGE_BYTES_PER_PIXEL == 61u,
	"logical-state storage accounting must include cost_improvement_exact");
static_assert(DMAP_LEGACY_TERMINAL_BYTES_PER_PIXEL == 97u,
	"terminal map resource accounting must match retained final diagnostics");

struct InstrumentExtendedMaps {
	int numLogicalStates = 0;
	int numViews = 0;
	int numTracePixels = 0;
	bool summaryAvailable = true;
	bool traceRequested = false;
	bool traceAvailable = false;
	bool compatibilityMapsRequested = false;
	bool mapsRequested = false;
	bool mapsAvailable = false;
	bool exactRequested = false;
	bool exactAvailable = false;
	bool prefilterRequested = false;
	bool prefilterAvailable = false;
	bool apdRequested = false;
	bool apdAvailable = false;
	bool apdMapsAvailable = false;
	bool apdStageActive = false;
	bool dvpRequested = false;
	bool dvpAvailable = false;
	bool dvpMapsAvailable = false;
	bool visibilityRequested = false;
	bool visibilityAvailable = false;
	bool visibilityMapsAvailable = false;
	bool visibleNormalRequested = false;
	bool visibleNormalAvailable = false;
	bool visibleNormalMapsAvailable = false;
	String resourceDecision = _T("not_requested");
	String exactUnavailableReason;
	String prefilterUnavailableReason;
	uint64_t estimatedDeviceBytes = 0;
	uint64_t estimatedHostBytes = 0;
	uint64_t estimatedStorageBytes = 0;
	uint64_t frameStorageCommittedBeforeBytes = 0;
	uint64_t frameStoragePriorityReserveBytes = 0;
	uint64_t frameStorageBudgetBytes = 0;
	uint64_t summaryDeviceBytes = 0;
	uint64_t traceDeviceBytes = 0;
	uint64_t legacyMapDeviceBytes = 0;
	uint64_t exactDeviceBytes = 0;
	uint64_t prefilterDeviceBytes = 0;
	uint64_t summaryHostBytes = 0;
	uint64_t traceHostBytes = 0;
	uint64_t legacyMapHostBytes = 0;
	uint64_t exactHostBytes = 0;
	uint64_t prefilterHostBytes = 0;
	uint64_t summaryStorageBytes = 0;
	uint64_t traceStorageBytes = 0;
	uint64_t legacyMapStorageBytes = 0;
	uint64_t exactStorageBytes = 0;
	uint64_t prefilterStorageBytes = 0;
	uint64_t apdSummaryDeviceBytes = 0;
	uint64_t apdSummaryHostBytes = 0;
	uint64_t apdTraceDeviceBytes = 0;
	uint64_t apdTraceHostBytes = 0;
	uint64_t apdTraceStorageBytes = 0;
	uint64_t apdMapDeviceBytes = 0;
	uint64_t apdMapHostBytes = 0;
	uint64_t apdMapStorageBytes = 0;
	uint64_t dvpSummaryDeviceBytes = 0;
	uint64_t dvpSummaryHostBytes = 0;
	uint64_t dvpTraceDeviceBytes = 0;
	uint64_t dvpTraceHostBytes = 0;
	uint64_t dvpTraceStorageBytes = 0;
	uint64_t dvpMapDeviceBytes = 0;
	uint64_t dvpMapHostBytes = 0;
	uint64_t dvpMapStorageBytes = 0;
	uint64_t visibilitySummaryDeviceBytes = 0;
	uint64_t visibilitySummaryHostBytes = 0;
	uint64_t visibilityTraceDeviceBytes = 0;
	uint64_t visibilityTraceHostBytes = 0;
	uint64_t visibilityTraceStorageBytes = 0;
	uint64_t visibilityMapDeviceBytes = 0;
	uint64_t visibilityMapHostBytes = 0;
	uint64_t visibilityMapStorageBytes = 0;
	uint64_t visibleNormalSummaryDeviceBytes = 0;
	uint64_t visibleNormalSummaryHostBytes = 0;
	uint64_t visibleNormalTraceDeviceBytes = 0;
	uint64_t visibleNormalTraceHostBytes = 0;
	uint64_t visibleNormalTraceStorageBytes = 0;
	uint64_t visibleNormalMapDeviceBytes = 0;
	uint64_t visibleNormalMapHostBytes = 0;
	uint64_t visibleNormalMapStorageBytes = 0;
	bool storagePreflightAttempted = false;
	bool storagePreflightSucceeded = false;
	uint64_t storageAvailableBytes = 0;
	uint64_t storageReservedBeforeBytes = 0;
	uint64_t storageEffectiveAvailableBytes = 0;
	uint64_t storageRequestedBytes = 0;
	uint64_t storageReservationBytes = 0;
	uint64_t storagePriorityReservationBytes = 0;
	bool storagePriorityReservationConsumed = false;
	String storagePreflightDecision = _T("not_requested");
	String storagePreflightReason;
	String storageReservationKey;
	String traceUnavailableReason;
	std::vector<float> passDepthDeltas;
	std::vector<float> passDepthRelDeltas;
	std::vector<float> passNormalAngleDeltas;
	std::vector<uint8_t> passViewChurn;
	std::vector<float> logicalStoredCosts;
	std::vector<float4> logicalScorePrimary;
	std::vector<float4> logicalScoreSecondary;
	std::vector<float4> exactLogicalScorePrimary;
	std::vector<float4> exactLogicalScoreSecondary;
	std::vector<PatchMatchInstrumentExactPixel> exactPixels;
	std::vector<PatchMatchInstrumentExactView> exactViews;
	std::vector<float4> finalViewWeights;
	std::vector<float4> finalViewCosts;
	std::vector<float4> finalViewPhotometricCosts;
	std::vector<float4> finalViewGeometricCosts;
	std::vector<float> finalViewEntropy;
	std::vector<float> finalLowDepth;
	std::vector<uint32_t> finalSelectedViews;
	std::vector<uint8_t> acceptedUpdateCount;
	std::vector<Point4> planesBeforeFilter;
	std::vector<float> costsBeforeFilter;
	std::vector<PatchMatchAPDInstrumentCounters> apdCounters;
	std::vector<PatchMatchAPDInstrumentState> apdStates;
	std::vector<PatchMatchAPDInstrumentUpdate> apdUpdates;
	std::vector<PatchMatchAPDInstrumentTrace> apdTraces;
	std::vector<PatchMatchDVPInstrumentCounters> dvpCounters;
	std::vector<PatchMatchDVPInstrumentUpdate> dvpUpdates;
	std::vector<PatchMatchDVPInstrumentTrace> dvpTraces;
	std::vector<PatchMatchDVPVisibilityInstrumentCounters> visibilityCounters;
	std::vector<PatchMatchDVPVisibilityInstrumentUpdate> visibilityUpdates;
	std::vector<PatchMatchDVPVisibilityInstrumentTrace> visibilityTraces;
	std::vector<PatchMatchDVPVisibleNormalInstrumentCounters> visibleNormalCounters;
	std::vector<PatchMatchDVPVisibleNormalInstrumentUpdate> visibleNormalUpdates;
	std::vector<PatchMatchDVPVisibleNormalInstrumentTrace> visibleNormalTraces;
	nlohmann::json apdMultiscale = nlohmann::json::object();
	std::vector<uint8_t> apdTransferredReliability;
	std::vector<uint8_t> apdTransferredAnchorCounts;
	std::vector<uint8_t> apdTransferredDeformableEligible;
	std::vector<uint8_t> apdOutputReliability;
	std::vector<uint8_t> apdOutputAnchorCounts;
	std::vector<uint8_t> apdOutputDeformableEligible;
};

void SetAlgorithmInstrumentAvailability(
	InstrumentExtendedMaps& plan, bool summaryAvailable, bool mapsAvailable)
{
	plan.apdAvailable = plan.apdRequested && summaryAvailable;
	plan.apdMapsAvailable = plan.apdRequested && summaryAvailable && mapsAvailable;
	plan.dvpAvailable = plan.dvpRequested && summaryAvailable;
	plan.dvpMapsAvailable = plan.dvpRequested && summaryAvailable && mapsAvailable;
	plan.visibilityAvailable = plan.visibilityRequested && summaryAvailable;
	plan.visibilityMapsAvailable =
		plan.visibilityRequested && summaryAvailable && mapsAvailable;
	plan.visibleNormalAvailable = plan.visibleNormalRequested && summaryAvailable;
	plan.visibleNormalMapsAvailable =
		plan.visibleNormalRequested && summaryAvailable && mapsAvailable;
}

struct InstrumentSidecarWriteError {
	String artifact;
	int pyramidLevel = -1;
};

void RecordInstrumentSidecarWriteError(
	std::vector<InstrumentSidecarWriteError>& errors,
	const char* artifact,
	int pyramidLevel)
{
	for (const InstrumentSidecarWriteError& error : errors) {
		if (error.artifact == artifact && error.pyramidLevel == pyramidLevel)
			return;
	}
	errors.push_back({artifact, pyramidLevel});
	if (pyramidLevel >= 0) {
		VERBOSE("warning: depth-map instrumentation failed to write requested sidecar '%s' at pyramid level %d",
			artifact, pyramidLevel);
	} else {
		VERBOSE("warning: depth-map instrumentation failed to write requested sidecar '%s'", artifact);
	}
}

nlohmann::json InstrumentSidecarWriteErrorsJson(
	const std::vector<InstrumentSidecarWriteError>& errors)
{
	nlohmann::json result(nlohmann::json::array());
	for (const InstrumentSidecarWriteError& error : errors) {
		result.push_back({
			{"artifact", error.artifact.c_str()},
			{"pyramid_level", error.pyramidLevel >= 0 ?
				nlohmann::json(error.pyramidLevel) : nlohmann::json(nullptr)}
		});
	}
	return result;
}

std::mutex g_instrumentMutex;
InstrumentConfig g_instrumentConfig;
std::set<std::string> g_dmapMetadataRoots;
std::unordered_map<std::string, uint64_t> g_instrumentStorageReservations;

struct InstrumentStorageReservation {
	InstrumentStorageReservation() = default;
	InstrumentStorageReservation(const InstrumentStorageReservation&) = delete;
	InstrumentStorageReservation& operator=(const InstrumentStorageReservation&) = delete;
	~InstrumentStorageReservation() { Release(); }

	void Release()
	{
		if (!active)
			return;
		std::lock_guard<std::mutex> lock(g_instrumentMutex);
		ASSERT(!key.empty());
		const auto it(g_instrumentStorageReservations.find(key.c_str()));
		if (it != g_instrumentStorageReservations.end()) {
			if (it->second <= bytes)
				g_instrumentStorageReservations.erase(it);
			else
				it->second -= bytes;
		}
		active = false;
		bytes = 0;
		key.clear();
	}

	String key;
	uint64_t bytes = 0;
	bool active = false;
};

enum DMapCandidateSourceCode {
	DMAP_SOURCE_UNKNOWN = 0,
	DMAP_SOURCE_INIT_RANDOM = 1,
	DMAP_SOURCE_INIT_SPARSE_OR_EXISTING = 2,
	DMAP_SOURCE_SPATIAL_PROPAGATION = 3,
	DMAP_SOURCE_VIEW_PROPAGATION = 4,
	DMAP_SOURCE_RANDOM_PERTURBATION = 5,
	DMAP_SOURCE_REFINEMENT = 6,
	DMAP_SOURCE_PRIOR_OR_GUIDANCE = 7,
	DMAP_SOURCE_OTHER = 8
};

enum DMapRejectionReasonCode {
	DMAP_REJECTION_NONE_SURVIVED = 0,
	DMAP_REJECTION_UNKNOWN_REJECTED = 1,
	DMAP_REJECTION_LOW_SCORE = 2,
	DMAP_REJECTION_INSUFFICIENT_VIEW_SUPPORT = 3,
	DMAP_REJECTION_GEOMETRIC_INCONSISTENCY = 4,
	DMAP_REJECTION_NORMAL_INCONSISTENCY = 5,
	DMAP_REJECTION_DEPTH_RANGE = 6,
	DMAP_REJECTION_OCCLUSION = 7,
	DMAP_REJECTION_MASKED = 8,
	DMAP_REJECTION_SMALL_COMPONENT = 9,
	DMAP_REJECTION_OTHER = 10
};

const char* InstrumentPhaseName(int phase)
{
	switch (phase) {
	case 0: return "init";
	case 1: return "black";
	case 2: return "red";
	default: return "unknown";
	}
}

const char* InstrumentSourceName(int source)
{
	switch (source) {
	case PM_SOURCE_NONE: return "none";
	case PM_SOURCE_INIT: return "init";
	case PM_SOURCE_PROPAGATE: return "propagate";
	case PM_SOURCE_REFINE_DEPTH: return "refine_depth";
	case PM_SOURCE_REFINE_NORMAL: return "refine_normal";
	case PM_SOURCE_REFINE_RANDOM_NORMAL: return "refine_random_normal";
	case PM_SOURCE_REFINE_SURFACE_NORMAL: return "refine_surface_normal";
	case PM_SOURCE_FILTERED: return "filtered";
	case PM_SOURCE_CHANGED_UNKNOWN: return "changed_unknown";
	case PM_SOURCE_APD_ANCHOR_PROPAGATE: return "apd_anchor_propagate";
	case PM_SOURCE_APD_FITTED_PLANE: return "apd_fitted_plane";
	case PM_SOURCE_APD_FINAL_REFINEMENT: return "apd_final_refinement";
	case PM_SOURCE_DVP_EPIPOLAR: return "dvp_epipolar";
	default: return "unknown";
	}
}

nlohmann::json APDCandidateAccountingJson(bool dvpActive)
{
	return {
		{"schema_name", "openmvs.dmap.apd_candidate_accounting"},
		{"schema_version", 1},
		{"candidate_slots", PM_INSTRUMENT_EXACT_NUM_CANDIDATES},
		{"dvp_candidate_slots", {PM_EXACT_CANDIDATE_DVP_EPIPOLAR_0,
			PM_EXACT_CANDIDATE_DVP_EPIPOLAR_1}},
		{"dvp_active", dvpActive},
		{"update_source_codes", PM_DVP_INSTRUMENT_FINAL_SOURCES},
		{"scope", "complete_active_process_pixel_working_objective"}
	};
}

const char* DVPFamilyName(unsigned family)
{
	switch (static_cast<DVPEpipolarFamily>(family)) {
	case DVPEpipolarFamily::DISABLED: return "disabled";
	case DVPEpipolarFamily::HISTORICAL_GLOBAL_V0: return "historical_global_v0";
	case DVPEpipolarFamily::GLOBAL_SEARCH_GATED_V1: return "global_search_gated_v1";
	case DVPEpipolarFamily::HISTORICAL_MIDPOINT_V1: return "historical_midpoint_v1";
	case DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1: return "dvp_eq11_interval_v1";
	default: return "unknown";
	}
}

const char* DVPUnavailableReasonName(unsigned reason)
{
	switch (static_cast<DVPEpipolarUnavailableReason>(reason)) {
	case DVPEpipolarUnavailableReason::NONE: return "none";
	case DVPEpipolarUnavailableReason::FAMILY_DISABLED: return "family_disabled";
	case DVPEpipolarUnavailableReason::GEOMETRY_UNAVAILABLE: return "geometry_unavailable";
	case DVPEpipolarUnavailableReason::INVALID_REFERENCE_DEPTH: return "invalid_reference_depth";
	case DVPEpipolarUnavailableReason::NO_SELECTED_SOURCE_VIEW: return "no_selected_source_view";
	case DVPEpipolarUnavailableReason::INVALID_EPIPOLAR_DIRECTION: return "invalid_epipolar_direction";
	case DVPEpipolarUnavailableReason::INSUFFICIENT_ENDPOINT_SUPPORT: return "insufficient_endpoint_support";
	case DVPEpipolarUnavailableReason::INVALID_INTERVAL_ORDER: return "invalid_interval_order";
	case DVPEpipolarUnavailableReason::NO_FINITE_GLOBAL_CANDIDATE: return "no_finite_global_candidate";
	case DVPEpipolarUnavailableReason::GLOBAL_REPROJECTION_GATE: return "global_reprojection_gate";
	case DVPEpipolarUnavailableReason::GLOBAL_SUPPORT_GATE: return "global_support_gate";
	case DVPEpipolarUnavailableReason::GLOBAL_OCCLUSION_REJECTED: return "global_occlusion_rejected";
	default: return "unknown";
	}
}

bool DVPFamilyPromotionEligible(unsigned family)
{
	return family == static_cast<unsigned>(DVPEpipolarFamily::GLOBAL_SEARCH_GATED_V1) ||
		family == static_cast<unsigned>(DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1);
}

const char* DVPClaimLabel(unsigned family)
{
	switch (static_cast<DVPEpipolarFamily>(family)) {
	case DVPEpipolarFamily::DISABLED:
		return "disabled";
	case DVPEpipolarFamily::HISTORICAL_GLOBAL_V0:
	case DVPEpipolarFamily::HISTORICAL_MIDPOINT_V1:
		return "compatibility_behavior";
	case DVPEpipolarFamily::GLOBAL_SEARCH_GATED_V1:
		return "openmvs_experimental_mechanism";
	case DVPEpipolarFamily::DVP_EQ11_INTERVAL_V1:
		return "paper_mechanics_complete_openmvs";
	}
	return "invalid";
}

const char* InstrumentBadReasonName(int reason)
{
	switch (reason) {
	case PM_BAD_NONE: return "unclassified";
	case PM_BAD_OUT_OF_BOUNDS: return "out_of_bounds";
	case PM_BAD_INVALID_PROJECTION: return "invalid_projection";
	case PM_BAD_LOW_REF_VARIANCE: return "low_ref_variance";
	case PM_BAD_LOW_TARGET_VARIANCE: return "low_target_variance";
	case PM_BAD_GEOMETRIC_MISMATCH: return "geometric_mismatch";
	default: return "unknown";
	}
}

const char* InstrumentCandidateTypeName(int type)
{
	switch (type) {
	case PM_CANDIDATE_INIT: return "init";
	case PM_CANDIDATE_PROPAGATION: return "propagation";
	case PM_CANDIDATE_RANDOM_PERTURBATION: return "random_perturbation";
	case PM_CANDIDATE_REFINEMENT: return "refinement";
	case PM_CANDIDATE_APD_FITTED_PLANE: return "apd_fitted_plane";
	case PM_CANDIDATE_APD_FINAL_REFINEMENT: return "apd_final_refinement";
	default: return "unknown";
	}
}

const char* APDProfileReasonName(int reason)
{
	switch (static_cast<APDProfileReason>(reason)) {
	case APDProfileReason::UNKNOWN_INVALID_INPUT: return "unknown_invalid_input";
	case APDProfileReason::UNKNOWN_NONFINITE_COST: return "unknown_nonfinite_cost";
	case APDProfileReason::UNRELIABLE_NO_LOCAL_MINIMUM: return "unreliable_no_local_minimum";
	case APDProfileReason::UNRELIABLE_GLOBAL_MINIMUM_OUTSIDE_ETA: return "unreliable_global_minimum_outside_eta";
	case APDProfileReason::UNRELIABLE_GLOBAL_MINIMUM_COST_TOO_HIGH: return "unreliable_global_minimum_cost_too_high";
	case APDProfileReason::UNRELIABLE_SINGLE_MINIMUM_COST_NOT_STRICTLY_BELOW_T2: return "unreliable_single_minimum_cost_not_strictly_below_t2";
	case APDProfileReason::UNRELIABLE_MULTI_MINIMUM_SEPARATION_NOT_ABOVE_T3: return "unreliable_multi_minimum_separation_not_above_t3";
	case APDProfileReason::RELIABLE_SINGLE_MINIMUM: return "reliable_single_minimum";
	case APDProfileReason::RELIABLE_SEPARATED_MINIMA: return "reliable_separated_minima";
	default: return "unknown";
	}
}

const char* APDAnchorReasonName(int reason)
{
	switch (reason) {
	case PM_APD_ANCHOR_UNKNOWN: return "unknown";
	case PM_APD_ANCHOR_PIXEL_NOT_UNRELIABLE: return "pixel_not_unreliable";
	case PM_APD_ANCHOR_INVALID_CENTER_DEPTH: return "invalid_center_depth";
	case PM_APD_ANCHOR_INSUFFICIENT_SECTOR_CANDIDATES: return "insufficient_sector_candidates";
	case PM_APD_ANCHOR_NO_VALID_RANSAC_MODEL: return "no_valid_ransac_model";
	case PM_APD_ANCHOR_INSUFFICIENT_MODEL_INLIERS: return "insufficient_model_inliers";
	case PM_APD_ANCHOR_READY: return "ready";
	default: return "unknown";
	}
}

const char* APDViewSelectionModeName(int mode)
{
	switch (static_cast<APDViewSelectionMode>(mode)) {
	case APDViewSelectionMode::NATIVE: return "native";
	case APDViewSelectionMode::ANCHOR_EVIDENCE: return "anchor_evidence";
	case APDViewSelectionMode::PREVIOUS_WEIGHTS_FALLBACK: return "previous_weights_fallback";
	case APDViewSelectionMode::SELECTED_MASK_FALLBACK: return "selected_mask_fallback";
	case APDViewSelectionMode::FIRST_VIEW_FALLBACK: return "first_view_fallback";
	default: return "unknown";
	}
}

const char* APDStageClockStatusName(APDStageClockStatus status)
{
	switch (status) {
	case APDStageClockStatus::VALID: return "valid";
	case APDStageClockStatus::INVALID_LEVEL_COUNT: return "invalid_level_count";
	case APDStageClockStatus::INVALID_LEVEL_INDEX: return "invalid_level_index";
	case APDStageClockStatus::INVALID_STAGE_INDEX: return "invalid_stage_index";
	default: return "unknown";
	}
}

const char* APDMultiscaleTransferStatusName(APDMultiscaleTransferStatus status)
{
	switch (status) {
	case APDMultiscaleTransferStatus::VALID: return "valid";
	case APDMultiscaleTransferStatus::UNAVAILABLE_NO_SOURCE: return "unavailable_no_source";
	case APDMultiscaleTransferStatus::INVALID_VERSION: return "invalid_version";
	case APDMultiscaleTransferStatus::INVALID_SOURCE_SIZE: return "invalid_source_size";
	case APDMultiscaleTransferStatus::INVALID_DESTINATION_SIZE: return "invalid_destination_size";
	case APDMultiscaleTransferStatus::INVALID_LEVEL: return "invalid_level";
	case APDMultiscaleTransferStatus::INVALID_STAGE: return "invalid_stage";
	default: return "unknown";
	}
}

nlohmann::json APDMultiscaleStageJson(const PatchMatch::Params& params)
{
	APDStageClock clock;
	clock.levelIndex = params.nAPDLevelIndex;
	clock.levelCount = params.nAPDLevelCount;
	clock.stageIndex = params.nAPDStageIndex;
	clock.hasTransferredState = params.bAPDTransferredState;
	clock.geometricConsistency = params.bGeomConsistency;
	const APDStageClockStatus clockStatus(ValidateAPDStageClock(clock));
	const APDStageSchedule schedule(ResolveAPDStageSchedule(clock));
	const APDMultiscaleTransferStatus transferStatus(
		static_cast<APDMultiscaleTransferStatus>(params.nAPDTransferStatus));
	return {
		{"schema_name", "openmvs.dmap.apd_multiscale_stage"},
		{"schema_version", 1},
		{"state_schema_version", APD_MULTISCALE_STATE_VERSION},
		{"clock", {
			{"level_index", clock.levelIndex},
			{"level_count", clock.levelCount},
			{"stage_index", clock.stageIndex},
			{"geometric_consistency", clock.geometricConsistency},
			{"status", APDStageClockStatusName(clockStatus)}
		}},
		{"transfer", {
			{"status_code", params.nAPDTransferStatus},
			{"status", APDMultiscaleTransferStatusName(transferStatus)},
			{"available", clock.hasTransferredState},
			{"resize", clock.hasTransferredState ? "opencv_inter_nearest" : "not_applied"}
		}},
		{"schedule", {
			{"valid", schedule.valid},
			{"policy", schedule.conventional ? "conventional_native" : "adaptive_patch_deformation"},
			{"consumes_transferred_reliability_at_iteration_zero", schedule.consumesTransferredState},
			{"reliability_eta", schedule.reliabilityEta},
			{"ransac_normalized_threshold", schedule.ransacNormalizedThreshold},
			{"checkerboard_exposure", "timings_only"}
		}}
	};
}

nlohmann::json APDReliabilityMapStats(const Image8U& image)
{
	uint64_t counts[4] = {};
	for (int i=0; i<image.area(); ++i) {
		const unsigned value(image[i]);
		++counts[value <= static_cast<unsigned>(APDReliabilityClass::RELIABLE) ? value : 3u];
	}
	const uint64_t total(static_cast<uint64_t>(image.area()));
	return {
		{"available", !image.empty()},
		{"width", image.cols},
		{"height", image.rows},
		{"num_pixels", total},
		{"unknown", counts[static_cast<unsigned>(APDReliabilityClass::UNKNOWN)]},
		{"unreliable", counts[static_cast<unsigned>(APDReliabilityClass::UNRELIABLE)]},
		{"reliable", counts[static_cast<unsigned>(APDReliabilityClass::RELIABLE)]},
		{"invalid_code", counts[3]},
		{"reliable_ratio", total ? nlohmann::json((double)counts[2]/(double)total) : nlohmann::json(nullptr)}
	};
}

nlohmann::json APDByteMapStats(const Image8U& image, bool booleanMap)
{
	uint64_t sum(0u), nonzero(0u);
	unsigned maximum(0u);
	for (int i=0; i<image.area(); ++i) {
		const unsigned value(image[i]);
		sum += value;
		nonzero += value != 0u;
		maximum = MAXF(maximum, value);
	}
	const uint64_t total(static_cast<uint64_t>(image.area()));
	return {
		{"available", !image.empty()},
		{"width", image.cols},
		{"height", image.rows},
		{"num_pixels", total},
		{"nonzero", nonzero},
		{"nonzero_ratio", total ? nlohmann::json((double)nonzero/(double)total) : nlohmann::json(nullptr)},
		{"mean", total ? nlohmann::json((double)sum/(double)total) : nlohmann::json(nullptr)},
		{"maximum", maximum},
		{"expected_domain", booleanMap ? "0_or_1" : "nonnegative_uint8"}
	};
}

nlohmann::json APDMultiscaleObservabilityJson(
	const PatchMatch::Params& params,
	const APDMultiscaleStateHeader* sourceHeader,
	const Image8U& transferredReliability,
	const Image8U& transferredAnchorCounts,
	const Image8U& transferredDeformableEligible,
	const Image8U& outputReliability,
	const Image8U& outputAnchorCounts,
	const Image8U& outputDeformableEligible)
{
	nlohmann::json result(APDMultiscaleStageJson(params));
	const bool transferred(params.bAPDTransferredState);
	result["source_state"] = sourceHeader ? nlohmann::json({
		{"available", true},
		{"version", sourceHeader->version},
		{"width", sourceHeader->width},
		{"height", sourceHeader->height},
		{"level_index", sourceHeader->sourceLevelIndex},
		{"stage_index", sourceHeader->sourceStageIndex}
	}) : nlohmann::json({{"available", false}});
	result["input_state"] = {
		{"available", transferred},
		{"reliability", APDReliabilityMapStats(transferredReliability)},
		{"anchor_count_provenance", APDByteMapStats(transferredAnchorCounts, false)},
		{"deformable_eligible_provenance", APDByteMapStats(transferredDeformableEligible, true)},
		{"iteration_zero_consumes", "reliability"},
		{"provenance_maps_directly_consumed", false},
		{"selected_views_source", params.bLowResProcessed ?
			"native_openmvs_nearest_resized_low_resolution_views_map" : "current_stage_initialization"}
	};
	result["output_state"] = {
		{"available", !outputReliability.empty()},
		{"reliability", APDReliabilityMapStats(outputReliability)},
		{"anchor_count_provenance", APDByteMapStats(outputAnchorCounts, false)},
		{"deformable_eligible_provenance", APDByteMapStats(outputDeformableEligible, true)},
		{"classifier_view_weight_source", transferred ?
			"final_apd_iteration_weights" : "uniform_expansion_of_native_selected_view_mask"},
		{"classifier_view_weight_quality", transferred ? "exact_runtime_state" : "compatibility_derived"}
	};
	result["availability"] = {
		{"historical_coarsest_monte_carlo_view_weights", false},
		{"historical_coarsest_monte_carlo_view_weights_reason",
			"native OpenMVS exposes only the final selected-view mask; uniform compatibility weights preserve native coarsest DMAP parity"},
		{"runtime_reliability_transfer", transferred},
		{"runtime_output_state", !outputReliability.empty()}
	};
	return result;
}

nlohmann::json APDContractJson(unsigned mode)
{
	const bool enabled(APDModeEnabled(mode));
	const bool full(APDModeUsesFullMechanics(mode));
	const bool deformationOnly(
		mode == static_cast<unsigned>(APDMode::DEFORMATION_ONLY));
	nlohmann::json activeMechanics(nlohmann::json::array());
	nlohmann::json omittedMechanics(nlohmann::json::array());
	if (enabled) {
		activeMechanics = {
			"reliability_profiles", "reliable_anchor_ransac", "deformable_patch_cost",
			"native_persistent_winner_rescore", "multiscale_reliability_transfer"
		};
	}
	if (full) {
		activeMechanics.push_back("anchor_driven_view_selection");
		activeMechanics.push_back("anchor_plane_propagation");
		activeMechanics.push_back("reliable_first_schedule");
		activeMechanics.push_back("fitted_plane_proposal");
		activeMechanics.push_back("terminal_native_refinement");
	} else if (deformationOnly) {
		omittedMechanics = {
			"anchor_driven_view_selection", "anchor_plane_propagation",
			"reliable_first_schedule", "fitted_plane_proposal",
			"terminal_native_refinement"
		};
	}
	return {
		{"mode", mode},
		{"mode_name", full ? "deformable_cost" : deformationOnly ? "deformation_only" : "disabled"},
		{"mode_role", full ? "full" : deformationOnly ? "deformation_only" : "control"},
		{"enabled", enabled},
		{"full_paper_mechanics_enabled", full},
		{"implementation_label", full ? "paper_mechanics_complete_openmvs" :
			deformationOnly ? "deformation_only_openmvs" : "disabled"},
		{"target_label", "paper_mechanics_complete_openmvs"},
		{"implemented_through", enabled ? "APD14-C5" : "none"},
		{"active_mechanics", activeMechanics},
		{"exact_author_code_equivalence_claimed", false},
		{"observability_semantics", {
			{"finite_candidate_compatibility_fields",
				"A legacy field name meaning IEEE-finite and below the OpenMVS fBadCost sentinel; it is a support-usable count, not an IEEE-finite-only count."},
			{"ieee_finite_only_count_available", false},
			{"evaluated_fitted_plane_score",
				"Schema v5 retains every IEEE-finite evaluated fitted-plane working score; values at or above fBadCost are evaluated but not support-usable."},
			{"accepted_candidate_may_be_non_usable", true},
			{"candidate_accounting",
				"Schema v5 covers the complete active 24-slot ProcessPixel objective; DVP slots 22 and 23 are also described in the independent DVP namespace."},
			{"working_cost_domain", enabled ? "APD deformable cost for supported unreliable pixels" : "native OpenMVS cost"},
			{"persistent_cost_domain", "native OpenMVS cost"}
		}},
		{"pins", {
			{"official_repository_commit", "5b2b31c02941cfa28808b4e37f06a87e8dc0f2d9"},
			{"colleague_donor_commit", "c6f0f2c0fe2e31ec68c54cac681cc3b1665d2f7b"},
			{"paper", "CVPR 2023 APD paper and supplementary material"}
		}},
		{"profile", {
			{"radius", APD_PROFILE_RADIUS}, {"samples", APD_PROFILE_SIZE},
			{"maximum_cost_t1", APD_PROFILE_MAX_COST},
			{"single_minimum_t2", APD_SINGLE_MINIMUM_MAX_COST},
			{"minimum_separation_t3", APD_MINIMUM_SEPARATION},
			{"separation_convention", "sqrt(sum_squared_minimum_cost_differences)/(number_of_minima-1)"},
			{"eta_schedule", {6, 4, 2}}
		}},
		{"anchors", {
			{"angular_sectors", APD_SECTOR_COUNT}, {"maximum_anchors", APD_MAX_ANCHORS},
			{"ransac_trials", APD_RANSAC_TRIALS}, {"minimum_inliers", APD_MIN_INLIERS},
			{"normalized_threshold_max", APD_RANSAC_NORMALIZED_THRESHOLD_MAX},
			{"normalized_threshold_min", APD_RANSAC_NORMALIZED_THRESHOLD_MIN}
		}},
		{"anchor_update", {
			{"enabled", full},
			{"state_semantics", "reliable black/red updates precede one immutable plane and selected-view snapshot shared by both non-reliable checkerboard phases"},
			{"view_evidence", "accepted reliable-anchor hypotheses and their snapshot selected-view masks"},
			{"propagation", "interpolated immutable reliable-anchor planes replace native adaptive-neighbor proposals for APD weak pixels"},
			{"maximum_candidates", APD_MAX_ANCHORS},
			{"view_minimum_agreement", APD_VIEW_MIN_AGREEMENT},
			{"view_maximum_bad", APD_VIEW_MAX_BAD},
			{"fallback_order", {"previous_iteration_weights", "snapshot_selected_mask", "first_source_view"}}
		}},
		{"schedule", {
			{"enabled", full},
			{"classification_state", "incoming complete logical-iteration state"},
			{"ordered_stages", {"reliable_black", "reliable_red", "anchor_and_fitted_plane_generation", "non_reliable_black", "non_reliable_red"}},
			{"unknown_class_policy", "processed exactly once in the non-reliable stage with native fallback"}
		}},
		{"fitted_plane", {
			{"enabled", full},
			{"source", "deterministic RANSAC model selected from reliable anchor candidates after reliable-first updates"},
			{"candidate_position", "before random depth and normal refinement in each non-reliable pixel update"},
			{"working_score", "deformable APD cost"},
			{"persistent_score", "conventional native OpenMVS winner rescore"}
		}},
		{"final_refinement", {
			{"enabled", full},
			{"domain", "conventional native OpenMVS cost"},
			{"disparity_radius", APD_FINAL_REFINEMENT_RADIUS},
			{"minimum_strict_improvement", APD_FINAL_REFINEMENT_MIN_IMPROVEMENT},
			{"timing", "after the final complete logical iteration and before production filtering"}
		}},
		{"multiscale", {
			{"enabled", enabled},
			{"state_schema_version", APD_MULTISCALE_STATE_VERSION},
			{"coarsest_schedule", "conventional_native"},
			{"active_schedule", "adaptive_patch_deformation_after_valid_state_transfer"},
			{"transfer_interpolation", "opencv_inter_nearest"},
			{"iteration_zero_consumes", "transferred_reliability"},
			{"transferred_provenance_not_directly_consumed", {"anchor_count", "deformable_eligible"}},
			{"stage_clock", "monotonic across photometric pyramid levels and geometric-consistency iterations"},
			{"checkerboard_exposure", "timings_only"}
		}},
		{"patch", {
			{"enabled", enabled},
			{"radius", APD_PATCH_RADIUS},
			{"center_increment", APD_CENTER_PATCH_INCREMENT},
			{"center_samples", APD_CENTER_PATCH_SAMPLES},
			{"anchor_increment", APD_ANCHOR_PATCH_INCREMENT},
			{"anchor_samples", APD_ANCHOR_PATCH_SAMPLES},
			{"center_weight", APD_CENTER_WEIGHT}, {"anchor_weight", APD_ANCHOR_WEIGHT}
		}},
		{"compatibility_behavior", {
			{"nearest_reliable_window", "symmetric integer [-50,+50] window (101x101) for the paper's stated 100x100 neighborhood"},
			{"persistent_score", "APD working scores remain local; retained winners are conventionally rescored into the native OpenMVS confidence domain"},
			{"view_threshold_schedule", "OpenMVS native 0.8*exp(-iteration^2/(2*4^2)) schedule with APD anchor evidence"},
			{"coarsest_reliability_view_weights", "uniform expansion of the exact native final selected-view mask because historical native Monte Carlo weights are unavailable"}
		}},
		{"required_mechanics_not_active", omittedMechanics},
		{"required_mechanics_not_yet_implemented", nlohmann::json::array()}
	};
}

nlohmann::json APDObservabilityJson(
	unsigned mode,
	const InstrumentExtendedMaps* maps)
{
	nlohmann::json result(APDContractJson(mode));
	const bool deformationOnly(
		mode == static_cast<unsigned>(APDMode::DEFORMATION_ONLY));
	const bool requested(maps && maps->apdRequested);
	const bool stageActive(maps && maps->apdStageActive);
	const bool available(maps && maps->apdAvailable && stageActive);
	result["schema_name"] = "openmvs.dmap.apd_observability";
	result["schema_version"] = PM_APD_INSTRUMENT_SCHEMA_VERSION;
	result["requested"] = requested;
	result["stage_active"] = stageActive;
	result["summary_available"] = available;
	result["maps_available"] = maps && maps->apdMapsAvailable && stageActive;
	result["targeted_trace_available"] = available && maps->traceAvailable && !maps->apdTraces.empty();
	result["state_record_bytes"] = sizeof(PatchMatchAPDInstrumentState);
	result["update_record_bytes"] = sizeof(PatchMatchAPDInstrumentUpdate);
	result["trace_record_bytes"] = sizeof(PatchMatchAPDInstrumentTrace);
	result["candidate_accounting"] = APDCandidateAccountingJson(
		available && maps->dvpRequested);
	const bool visibilityAwareComponents(
		maps && maps->visibilityRequested && maps->apdStageActive);
	result["component_contract"] = {
		{"anchor_component_field", "anchor_mean_cost"},
		{"anchor_component_basis", visibilityAwareComponents ?
			"visibility_aware_effective_anchor_leg" : "active_anchor_mean"},
		{"deformable_photometric_identity",
			"0.25*center_cost + 0.75*anchor_component"},
		{"visibility_fallback", visibilityAwareComponents ?
			"center_cost_substitutes_for_anchor_component_when_per_view_visible_anchor_consensus_is_invalid" :
			"not_active"}
	};
	result["measurement_basis"] = available ?
		"APD hot-kernel and same-stream mechanics records" :
		requested && !stageActive ? "unavailable_conventional_native_stage" : "unavailable";
	result["iterations"] = nlohmann::json::array();
	if (!available)
		return result;
	auto optionalMean = [](double sum, uint32_t count) {
		return count > 0u ? nlohmann::json(sum/(double)count) : nlohmann::json(nullptr);
	};
	auto ratio = [](uint32_t numerator, uint32_t denominator) {
		return denominator > 0u ? nlohmann::json((double)numerator/(double)denominator) : nlohmann::json(nullptr);
	};
	for (size_t iteration=0; iteration<maps->apdCounters.size(); ++iteration) {
		const PatchMatchAPDInstrumentCounters& counter(maps->apdCounters[iteration]);
		nlohmann::json profileReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_APD_INSTRUMENT_PROFILE_REASONS; ++reason)
			profileReasons[APDProfileReasonName(reason)] = counter.profileReason[reason];
		nlohmann::json anchorReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_APD_INSTRUMENT_ANCHOR_REASONS; ++reason)
			anchorReasons[APDAnchorReasonName(reason)] = counter.anchorReason[reason];
		nlohmann::json anchorCounts(nlohmann::json::object());
		for (int count=0; count<=PM_APD_INSTRUMENT_ANCHORS; ++count)
			anchorCounts[std::to_string(count)] = counter.anchorCountBins[count];
		nlohmann::json updateSources(nlohmann::json::object());
		for (int source=0; source<PM_INSTRUMENT_NUM_SOURCES; ++source)
			updateSources[InstrumentSourceName(source)] = counter.updateSource[source];
		updateSources[InstrumentSourceName(PM_SOURCE_DVP_EPIPOLAR)] =
			maps->dvpAvailable && iteration < maps->dvpCounters.size() ?
			maps->dvpCounters[iteration].finalUpdateSource[PM_SOURCE_DVP_EPIPOLAR] : 0u;
		nlohmann::json viewSelectionModes(nlohmann::json::object());
		for (int mode=0; mode<PM_APD_INSTRUMENT_VIEW_SELECTION_MODES; ++mode)
			viewSelectionModes[APDViewSelectionModeName(mode)] = counter.anchorViewSelectionMode[mode];
		result["iterations"].push_back({
			{"logical_iteration", iteration}, {"classified_pixels", counter.classified},
			{"reliability", {
				{"unknown", counter.reliability[0]}, {"unreliable", counter.reliability[1]},
				{"reliable", counter.reliability[2]},
				{"unreliable_ratio", ratio(counter.reliability[1], counter.classified)},
				{"reliable_ratio", ratio(counter.reliability[2], counter.classified)}
			}},
			{"profile_reason_counts", profileReasons},
			{"global_minimum_cost", {
				{"samples", counter.globalMinimumCostSamples},
				{"mean", optionalMean(counter.globalMinimumCostSum, counter.globalMinimumCostSamples)}
			}},
			{"profile_separation", {
				{"samples", counter.separationSamples},
				{"mean", optionalMean(counter.separationSum, counter.separationSamples)}
			}},
			{"anchor_model", {
				{"ransac_valid_pixels", counter.ransacValid},
				{"deformable_eligible_pixels", counter.deformableEligible},
				{"deformable_eligible_ratio", ratio(counter.deformableEligible, counter.classified)},
				{"anchor_count_mean", optionalMean(counter.anchorCountSum, counter.classified)},
				{"reason_counts", anchorReasons}, {"anchor_count_histogram", anchorCounts}
			}},
			{"updates", {
				{"deformable_updates", counter.deformableUpdates},
				{"deformable_update_ratio", ratio(counter.deformableUpdates, counter.classified)},
				{"source_counts", updateSources},
				{"working_gap_samples", counter.workingGapSamples},
				{"working_gap_mean", optionalMean(counter.workingGapSum, counter.workingGapSamples)},
				{"center_cost_mean", optionalMean(counter.centerCostSum, counter.deformableUpdates)},
				{"anchor_mean_cost_mean", optionalMean(counter.anchorMeanCostSum, counter.deformableUpdates)},
				{"working_cost_mean", optionalMean(counter.workingCostSum, counter.deformableUpdates)},
				{"native_persistent_cost_mean", optionalMean(counter.nativePersistentCostSum, counter.deformableUpdates)},
				{"immutable_anchor_state_updates", counter.immutableAnchorStateUpdates},
				{"stage_counts", {
					{"all_compatibility", counter.stageUpdates[static_cast<unsigned>(APDUpdateStage::ALL)]},
					{"all_deformation_only", deformationOnly ?
						counter.stageUpdates[static_cast<unsigned>(APDUpdateStage::ALL)] : 0u},
					{"reliable_first", counter.stageUpdates[static_cast<unsigned>(APDUpdateStage::RELIABLE)]},
					{"non_reliable_second", counter.stageUpdates[static_cast<unsigned>(APDUpdateStage::NON_RELIABLE)]}
				}}
			}},
			{"anchor_view_selection", {
				{"attempted_pixels", counter.anchorViewSelectionAttempted},
				{"anchor_evidence_pixels", counter.anchorViewSelectionUsed},
				{"anchor_evidence_ratio", ratio(counter.anchorViewSelectionUsed, counter.anchorViewSelectionAttempted)},
				{"mode_counts", viewSelectionModes}
			}},
			{"anchor_propagation", {
				{"tested_candidates", counter.anchorProposalsTested},
				{"finite_candidates", counter.anchorProposalsFinite},
				{"accepted_events", counter.anchorProposalsAccepted},
				{"final_winner_pixels", counter.anchorPropagationFinalWinners},
				{"best_working_cost_samples", counter.bestAnchorWorkingCostSamples},
				{"best_working_cost_mean", optionalMean(counter.bestAnchorWorkingCostSum, counter.bestAnchorWorkingCostSamples)},
				{"accepted_native_cost_samples", counter.acceptedAnchorNativeCostSamples},
				{"accepted_native_cost_mean", optionalMean(counter.acceptedAnchorNativeCostSum, counter.acceptedAnchorNativeCostSamples)}
			}},
			{"fitted_plane", {
				{"available_pixels", counter.fittedPlaneAvailable},
				{"tested_pixels", counter.fittedPlaneTested},
				{"finite_pixels", counter.fittedPlaneFinite},
				{"accepted_events", counter.fittedPlaneAccepted},
				{"final_winner_pixels", counter.fittedPlaneFinalWinners}
			}},
			{"final_native_refinement", {
				{"eligible_pixels", counter.finalRefinementPixels},
				{"tested_candidates", counter.finalRefinementCandidatesTested},
				{"finite_candidates", counter.finalRefinementCandidatesFinite},
				{"accepted_pixels", counter.finalRefinementAccepted},
				{"accepted_ratio", ratio(counter.finalRefinementAccepted, counter.finalRefinementPixels)}
			}}
		});
	}
	return result;
}

nlohmann::json DVPContractJson(const PatchMatch::Params& params)
{
	const unsigned family(params.nDVPEpipolarFamily);
	const bool enabled(DVPEpipolarFamilyEnabled(family));
	return {
		{"enabled", enabled},
		{"family", DVPFamilyName(family)},
		{"promotion_eligible", DVPFamilyPromotionEligible(family)},
		{"claim_label", DVPClaimLabel(family)},
		{"author_code_exact_equivalence", false},
		{"parameters", {
			{"alpha", params.fDVPEpipolarAlpha},
			{"beta", params.fDVPEpipolarBeta},
			{"mu", params.nDVPEpipolarMu},
			{"global_search_radius", params.nDVPGlobalSearchRadius},
			{"global_reprojection_threshold_px", params.fDVPReprojectionThreshold},
			{"global_relative_depth_threshold", params.fDVPRelativeDepthThreshold}
		}},
		{"families", {
			{"historical_global_v0", {
				{"purpose", "bounded donor-compatible single-view global epipolar search"},
				{"promotion_eligible", false}
			}},
			{"global_search_gated_v1", {
				{"purpose", "global epipolar search with roundtrip, support, depth, and occlusion gates"},
				{"promotion_eligible", true}
			}},
			{"historical_midpoint_v1", {
				{"purpose", "historical endpoint midpoint candidates plus native depth perturbation"},
				{"promotion_eligible", false}
			}},
			{"dvp_eq11_interval_v1", {
				{"purpose", "paper Eq. 11 robust disjoint depth intervals replacing native depth perturbation when available"},
				{"promotion_eligible", true}
			}}
		}},
		{"mechanics", {
			{"candidate_order", "DVP proposals are scored after propagation/APD fitted-plane proposals and before native depth/normal refinement"},
			{"paper_order_statistics", "mu-th smallest outer and mu-th largest inner endpoint depths; each interval side independently requires at least mu samples in its outer and inner groups"},
			{"interval_sampling", "one sample per valid disjoint left/right interval from a fork of the per-pixel RNG stream"},
			{"rng_schedule", "when interval proposals replace native depth perturbation, the persistent stream advances by exactly one native-depth draw regardless of whether one or two intervals are valid"},
			{"native_depth_fallback", "native depth perturbation runs only when the Eq. 11 family has no valid interval proposal"},
			{"persistent_score", "APD working scores remain local and any retained APD winner is conventionally rescored"},
			{"logical_iteration", "complete black plus red checkerboard update; phases are exposed only in timings"}
		}},
		{"provenance", {
			{"paper", "DVP-MVS paper contract pinned by Plan 15 C0"},
			{"official_repository_commit", "5b2b31c02941cfa28808b4e37f06a87e8dc0f2d9"},
			{"colleague_donor_commit", "c6f0f2c0fe2e31ec68c54cac681cc3b1665d2f7b"},
			{"implementation_basis", "independent OpenMVS implementation validated by CPU and CUDA oracles"}
		}},
		{"scope_exclusions", {"depth_edge_prior", "persistent_visibility", "visible_normal_constraint", "dvp_plus_plus"}}
	};
}

const char* DVPVisibilityModeName(unsigned mode)
{
	switch (static_cast<DVPVisibilityMode>(mode)) {
	case DVPVisibilityMode::DISABLED: return "disabled";
	case DVPVisibilityMode::PAPER_2D_RESTORE_V1: return "paper_2d_restore_v1";
	case DVPVisibilityMode::DEPTH_GATED_RESTORE_V1: return "depth_gated_restore_v1";
	}
	return "invalid";
}

const char* DVPVisibilityReasonName(unsigned reason)
{
	switch (static_cast<DVPVisibilityReason>(reason)) {
	case DVPVisibilityReason::MODE_DISABLED: return "mode_disabled";
	case DVPVisibilityReason::RETAINED_PREVIOUS_WEIGHT: return "retained_previous_weight";
	case DVPVisibilityReason::INVALID_REFERENCE_DEPTH: return "invalid_reference_depth";
	case DVPVisibilityReason::FORWARD_OUT_OF_BOUNDS: return "forward_out_of_bounds";
	case DVPVisibilityReason::INVALID_EXPECTED_SOURCE_DEPTH: return "invalid_expected_source_depth";
	case DVPVisibilityReason::INVALID_OBSERVED_SOURCE_DEPTH: return "invalid_observed_source_depth";
	case DVPVisibilityReason::BACKWARD_OUT_OF_BOUNDS: return "backward_out_of_bounds";
	case DVPVisibilityReason::ROUND_TRIP_REJECTED: return "round_trip_rejected";
	case DVPVisibilityReason::NEARER_OCCLUDER_REJECTED: return "nearer_occluder_rejected";
	case DVPVisibilityReason::DEPTH_DISAGREEMENT_REJECTED: return "depth_disagreement_rejected";
	case DVPVisibilityReason::RESTORED_PAPER_2D: return "restored_paper_2d";
	case DVPVisibilityReason::RESTORED_DEPTH_GATED: return "restored_depth_gated";
	}
	return "invalid";
}

const char* DVPVisibilityTransitionStatusName(unsigned status)
{
	switch (static_cast<DVPVisibilityTransitionStatus>(status)) {
	case DVPVisibilityTransitionStatus::VALID: return "valid";
	case DVPVisibilityTransitionStatus::INVALID_VERSION: return "invalid_version";
	case DVPVisibilityTransitionStatus::INVALID_GEOMETRY: return "invalid_geometry";
	case DVPVisibilityTransitionStatus::INVALID_VIEW_COUNT: return "invalid_view_count";
	case DVPVisibilityTransitionStatus::INVALID_LOGICAL_ITERATION: return "invalid_logical_iteration";
	}
	return "invalid";
}

nlohmann::json DVPVisibilityContractJson(const PatchMatch::Params& params)
{
	return {
		{"schema_name", "openmvs.dmap.dvp_visibility_contract"},
		{"schema_version", DVP_VISIBILITY_STATE_VERSION},
		{"enabled", DVPVisibilityModeEnabled(params.nDVPVisibilityMode)},
		{"mode", DVPVisibilityModeName(params.nDVPVisibilityMode)},
		{"parameters", {
			{"reprojection_threshold_px", params.fDVPVisibilityReprojectionThreshold},
			{"relative_depth_threshold", params.fDVPVisibilityRelativeDepthThreshold},
			{"restored_weight", DVP_VISIBILITY_RESTORED_WEIGHT}
		}},
		{"state_timing", "accepted t-1 state resolves immutable support; current selection publishes once after the complete logical iteration"},
		{"paper_contract_label", "paper_mechanics_complete_openmvs"},
		{"author_code_exact_equivalence", false},
		{"compatibility_behavior", "weight one, explicit logical-iteration publication, and optional depth/occlusion gating are OpenMVS integration choices"}
	};
}

nlohmann::json DVPVisibilityObservabilityJson(
	const PatchMatch::Params& params,
	const InstrumentExtendedMaps* maps)
{
	nlohmann::json result(DVPVisibilityContractJson(params));
	const bool requested(maps && maps->visibilityRequested);
	const bool stageActive(maps && maps->apdStageActive && params.bGeomConsistency);
	const bool available(maps && maps->visibilityAvailable && stageActive);
	result["schema_name"] = "openmvs.dmap.dvp_visibility_observability";
	result["schema_version"] = PM_DVP_VISIBILITY_INSTRUMENT_SCHEMA_VERSION;
	result["num_views"] = maps ? maps->numViews : 0;
	result["requested"] = requested;
	result["stage_active"] = stageActive;
	result["summary_available"] = available;
	result["maps_available"] = maps && maps->visibilityMapsAvailable && stageActive;
	result["targeted_trace_available"] = available && maps->traceAvailable &&
		!maps->visibilityTraces.empty();
	result["counter_record_bytes"] = sizeof(PatchMatchDVPVisibilityInstrumentCounters);
	result["update_record_bytes"] = sizeof(PatchMatchDVPVisibilityInstrumentUpdate);
	result["trace_record_bytes"] = sizeof(PatchMatchDVPVisibilityInstrumentTrace);
	result["aggregate_weight_sum_precision"] = "uint64 device atomic";
	result["measurement_quality"] = available ? "exact" : "unavailable";
	result["measurement_basis"] = available ?
		"same-stream Process<true> logical-iteration visibility transition and active candidate support" :
		requested && !stageActive ? "unavailable_conventional_native_stage" : "unavailable";
	result["observer_specialization_risk"] =
		"Process<true> is separately compiled and can perturb register pressure or scheduling; production quality uses Process<false>";
	result["reason_enum"] = nlohmann::json::object();
	for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason)
		result["reason_enum"][std::to_string(reason)] = DVPVisibilityReasonName((unsigned)reason);
	result["iterations"] = nlohmann::json::array();
	if (!available)
		return result;
	auto ratio = [](uint64_t numerator, uint64_t denominator) {
		return denominator > 0u ? nlohmann::json((double)numerator/(double)denominator) :
			nlohmann::json(nullptr);
	};
	auto mean = [](uint64_t sum, uint64_t count) {
		return count > 0u ? nlohmann::json((double)sum/(double)count) :
			nlohmann::json(nullptr);
	};
	for (size_t iteration=0; iteration<maps->visibilityCounters.size(); ++iteration) {
		const PatchMatchDVPVisibilityInstrumentCounters& counter(
			maps->visibilityCounters[iteration]);
		nlohmann::json reasons(nlohmann::json::object());
		for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason)
			reasons[DVPVisibilityReasonName((unsigned)reason)] = counter.reason[reason];
		const uint64_t resolvedDecisions(
			static_cast<uint64_t>(counter.restoredViews)+counter.rejectedViews);
		result["iterations"].push_back({
			{"logical_iteration", iteration},
			{"pixels", counter.pixels},
			{"mode", DVPVisibilityModeName(params.nDVPVisibilityMode)},
			{"views", {
				{"previous_visible", counter.previousVisibleViews},
				{"resolved_visible", counter.resolvedVisibleViews},
				{"next_visible", counter.nextVisibleViews},
				{"active_support", counter.activeSupportViews},
				{"previous_visible_mean_per_pixel", mean(counter.previousVisibleViews, counter.pixels)},
				{"resolved_visible_mean_per_pixel", mean(counter.resolvedVisibleViews, counter.pixels)},
				{"next_visible_mean_per_pixel", mean(counter.nextVisibleViews, counter.pixels)},
				{"active_support_mean_per_pixel", mean(counter.activeSupportViews, counter.pixels)},
				{"restored", counter.restoredViews},
				{"rejected", counter.rejectedViews},
				{"restored_ratio_of_zero_weight_decisions", ratio(counter.restoredViews, resolvedDecisions)},
				{"added", counter.addedViews},
				{"removed", counter.removedViews}
			}},
			{"weights", {
				{"previous_sum", counter.previousWeightSum},
				{"resolved_sum", counter.resolvedWeightSum},
				{"next_sum", counter.nextWeightSum},
				{"active_support_sum", counter.activeSupportWeightSum},
				{"denominator_sum", counter.denominatorSum},
				{"previous_mean_per_pixel", mean(counter.previousWeightSum, counter.pixels)},
				{"resolved_mean_per_pixel", mean(counter.resolvedWeightSum, counter.pixels)},
				{"next_mean_per_pixel", mean(counter.nextWeightSum, counter.pixels)},
				{"active_support_mean_per_pixel", mean(counter.activeSupportWeightSum, counter.pixels)},
				{"denominator_mean_per_pixel", mean(counter.denominatorSum, counter.pixels)},
				{"zero_denominator_pixels", counter.zeroDenominatorPixels},
				{"zero_denominator_ratio", ratio(counter.zeroDenominatorPixels, counter.pixels)}
			}},
			{"transitions", {
				{"changed_pixels", counter.changedPixels},
				{"changed_ratio", ratio(counter.changedPixels, counter.pixels)},
				{"support_mismatch_pixels", counter.supportMismatchPixels},
				{"support_mismatch_ratio", ratio(counter.supportMismatchPixels, counter.pixels)},
				{"invalid_transition_pixels", counter.invalidTransitionPixels},
				{"invalid_transition_ratio", ratio(counter.invalidTransitionPixels, counter.pixels)}
			}},
			{"candidates", {
				{"tested_pixels", counter.candidateTestedPixels},
				{"finite_pixels", counter.candidateFinitePixels},
				{"tested_count", counter.candidateTestedCount},
				{"finite_count", counter.candidateFiniteCount},
				{"finite_ratio", ratio(counter.candidateFiniteCount, counter.candidateTestedCount)}
			}},
			{"reason_counts", reasons}
		});
	}
	return result;
}

const char* DVPVisibleNormalModeName(unsigned mode)
{
	switch (static_cast<DVPVisibleNormalMode>(mode)) {
	case DVPVisibleNormalMode::DISABLED: return "disabled";
	case DVPVisibleNormalMode::SHADOW: return "shadow";
	case DVPVisibleNormalMode::REFINEMENT: return "refinement";
	case DVPVisibleNormalMode::PROPAGATION: return "propagation";
	case DVPVisibleNormalMode::FULL: return "full";
	}
	return "invalid";
}

const char* DVPVisibleNormalEvaluationReasonName(unsigned reason)
{
	switch (static_cast<DVPVisibleNormalEvaluationReason>(reason)) {
	case DVPVisibleNormalEvaluationReason::VALID: return "valid";
	case DVPVisibleNormalEvaluationReason::INVALID_NORMAL: return "invalid_normal";
	case DVPVisibleNormalEvaluationReason::INVALID_DIRECTION_COUNT: return "invalid_direction_count";
	case DVPVisibleNormalEvaluationReason::INVALID_DIRECTION: return "invalid_direction";
	case DVPVisibleNormalEvaluationReason::INVALID_DOT_TOLERANCE: return "invalid_dot_tolerance";
	case DVPVisibleNormalEvaluationReason::HEMISPHERE_REJECTED: return "hemisphere_rejected";
	}
	return "invalid";
}

const char* DVPVisibleNormalProposalReasonName(unsigned reason)
{
	switch (static_cast<DVPVisibleNormalProposalReason>(reason)) {
	case DVPVisibleNormalProposalReason::INVALID_ARGUMENT: return "invalid_argument_native_retained";
	case DVPVisibleNormalProposalReason::NATIVE_FEASIBLE: return "native_feasible";
	case DVPVisibleNormalProposalReason::CONSTRAINED_RETRY: return "constrained_retry";
	case DVPVisibleNormalProposalReason::RETRY_EXHAUSTED_NATIVE_FALLBACK:
		return "retry_exhausted_native_fallback";
	}
	return "invalid";
}

const char* DVPVisibleNormalPropagationReasonName(unsigned reason)
{
	switch (static_cast<DVPVisibleNormalPropagationReason>(reason)) {
	case DVPVisibleNormalPropagationReason::INVALID_ARGUMENT: return "invalid_argument";
	case DVPVisibleNormalPropagationReason::NO_VALID_CANDIDATE: return "no_valid_candidate";
	case DVPVisibleNormalPropagationReason::CONSTRAINED_FEASIBLE: return "constrained_feasible";
	case DVPVisibleNormalPropagationReason::NO_FEASIBLE_NATIVE_FALLBACK:
		return "no_feasible_native_fallback";
	}
	return "invalid";
}

nlohmann::json DVPVisibleNormalContractJson(const PatchMatch::Params& params)
{
	return {
		{"schema_name", "openmvs.dmap.dvp_visible_normal_contract"},
		{"schema_version", 1},
		{"enabled", DVPVisibleNormalModeEnabled(params.nDVPVisibleNormalMode)},
		{"mode", DVPVisibleNormalModeName(params.nDVPVisibleNormalMode)},
		{"claim_label", "dvp_visible_normal_selected_support_compatibility_v1"},
		{"paper_fidelity", "paper_partial_until_visibility_context_qualifies"},
		{"parameters", {
			{"dot_tolerance", params.fDVPVisibleNormalDotTolerance},
			{"max_attempts_per_stochastic_proposal", params.nDVPVisibleNormalAttempts},
			{"selected_source_view_limit", DVP_VISIBLE_NORMAL_MAX_SOURCE_VIEWS}
		}},
		{"coordinate_contract", {
			{"normal_frame", "reference_camera"},
			{"direction_frame", "reference_camera"},
			{"reference_direction", "normalize(reference_camera_point)"},
			{"source_center", "R_ref*(C_source-C_ref)"},
			{"source_direction", "normalize(reference_camera_point-source_center_reference)"},
			{"feasible_test", "dot(normal,direction)<=dot_tolerance for reference and every selected source view"}
		}},
		{"support_contract", "current ProcessPixel selected source-view mask only; no persistent P04 visibility and no source depth-map sampling"},
		{"rng_contract", "native depth, perturbed-normal, and random-normal draws advance the production RNG exactly once; bounded retries consume only a copied post-native state"},
		{"fallback_contract", {
			{"propagation", "cheapest feasible candidate, otherwise native cheapest candidate"},
			{"refinement", "first feasible copied-state retry, otherwise native proposal"},
			{"invalid_geometry", "native proposal retained with explicit invalid reason"}
		}},
		{"refinement_scope", {
			{"constrained", {"perturbed_normal", "random_normal"}},
			{"unchanged", {"depth_only", "surface_gradient_normal"}},
			{"reason", "surface-gradient replacement is not specified by the pinned paper contract; it remains a named compatibility exclusion"}
		}},
		{"persistent_winner_conventionally_rescored", true},
		{"author_code_exact_equivalence", false},
		{"provenance", {
			{"official_repository_commit", "38660f3567b5989f74ee718382b2302c6768398e"},
			{"colleague_donor_commit", "c6f0f2c0fe2e31ec68c54cac681cc3b1665d2f7b"},
			{"implementation_basis", "independent frame-correct OpenMVS implementation validated by CPU and live-CUDA oracles"}
		}}
	};
}

nlohmann::json DVPVisibleNormalObservabilityJson(
	const PatchMatch::Params& params,
	const InstrumentExtendedMaps* maps)
{
	nlohmann::json result(DVPVisibleNormalContractJson(params));
	const bool requested(maps && maps->visibleNormalRequested);
	const bool stageActive(maps && maps->apdStageActive);
	const bool available(maps && maps->visibleNormalAvailable && stageActive);
	result["schema_name"] = "openmvs.dmap.dvp_visible_normal_observability";
	result["schema_version"] = PM_DVP_VISIBLE_NORMAL_INSTRUMENT_SCHEMA_VERSION;
	result["requested"] = requested;
	result["stage_active"] = stageActive;
	result["summary_available"] = available;
	result["maps_available"] = maps && maps->visibleNormalMapsAvailable && stageActive;
	result["targeted_trace_available"] = available && maps->traceAvailable &&
		!maps->visibleNormalTraces.empty();
	result["counter_record_bytes"] = sizeof(PatchMatchDVPVisibleNormalInstrumentCounters);
	result["update_record_bytes"] = sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate);
	result["trace_record_bytes"] = sizeof(PatchMatchDVPVisibleNormalInstrumentTrace);
	result["measurement_quality"] = available ? "exact" : "unavailable";
	result["measurement_basis"] = available ?
		"same_stream_Process_true_logical_iteration_proposal_decisions" :
		requested && !stageActive ? "unavailable_conventional_native_stage" : "unavailable";
	result["observer_specialization_risk"] =
		"Process<true> is separately compiled and can perturb register pressure or scheduling; production quality uses Process<false>";
	result["evaluation_reason_enum"] = nlohmann::json::object();
	for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS; ++reason)
		result["evaluation_reason_enum"][std::to_string(reason)] =
			DVPVisibleNormalEvaluationReasonName((unsigned)reason);
	result["proposal_reason_enum"] = nlohmann::json::object();
	for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS; ++reason)
		result["proposal_reason_enum"][std::to_string(reason)] =
			DVPVisibleNormalProposalReasonName((unsigned)reason);
	result["propagation_reason_enum"] = nlohmann::json::object();
	for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS; ++reason)
		result["propagation_reason_enum"][std::to_string(reason)] =
			DVPVisibleNormalPropagationReasonName((unsigned)reason);
	result["iterations"] = nlohmann::json::array();
	if (!available)
		return result;
	auto ratio = [](uint64_t numerator, uint64_t denominator) {
		return denominator ? nlohmann::json((double)numerator/(double)denominator) :
			nlohmann::json(nullptr);
	};
	for (size_t iteration=0; iteration<maps->visibleNormalCounters.size(); ++iteration) {
		const PatchMatchDVPVisibleNormalInstrumentCounters& c(
			maps->visibleNormalCounters[iteration]);
		nlohmann::json evaluationReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS; ++reason)
			evaluationReasons[DVPVisibleNormalEvaluationReasonName((unsigned)reason)] =
				c.evaluationReason[reason];
		nlohmann::json proposalReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS; ++reason)
			proposalReasons[DVPVisibleNormalProposalReasonName((unsigned)reason)] =
				c.proposalReason[reason];
		nlohmann::json propagationReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS; ++reason)
			propagationReasons[DVPVisibleNormalPropagationReasonName((unsigned)reason)] =
				c.propagationReason[reason];
		result["iterations"].push_back({
			{"logical_iteration", iteration},
			{"mode", DVPVisibleNormalModeName(params.nDVPVisibleNormalMode)},
			{"pixels", c.pixels},
			{"selected_support_views", c.selectedSupportViews},
			{"selected_support_mean_per_pixel", c.pixels ?
				nlohmann::json((double)c.selectedSupportViews/c.pixels) : nlohmann::json(nullptr)},
			{"current", {
				{"valid", c.currentValid}, {"feasible", c.currentFeasible},
				{"rejected", c.currentRejected}, {"invalid", c.currentInvalid},
				{"feasible_ratio", ratio(c.currentFeasible, c.currentValid)}
			}},
			{"propagation", {
				{"tested", c.propagationCandidatesTested},
				{"valid", c.propagationCandidatesValid},
				{"feasible", c.propagationCandidatesFeasible},
				{"rejected", c.propagationCandidatesRejected},
				{"feasible_ratio", ratio(c.propagationCandidatesFeasible,
					c.propagationCandidatesValid)},
				{"constrained_selections", c.propagationConstrainedSelections},
				{"native_selections", c.propagationNativeSelections},
				{"fallbacks", c.propagationFallbacks},
				{"applied_constraints", c.propagationAppliedConstraints},
				{"accepted", c.propagationAccepted},
				{"reason_counts", propagationReasons}
			}},
			{"refinement", {
				{"native_tested", c.refinementNativeTested},
				{"native_valid", c.refinementNativeValid},
				{"native_feasible", c.refinementNativeFeasible},
				{"native_rejected", c.refinementNativeRejected},
				{"native_feasible_ratio", ratio(c.refinementNativeFeasible,
					c.refinementNativeValid)},
				{"retries_tested", c.refinementRetriesTested},
				{"retry_success", c.refinementRetrySuccess},
				{"exhaustion", c.refinementExhaustion},
				{"fallback", c.refinementFallback},
				{"applied_retry", c.refinementAppliedRetry},
				{"accepted", c.refinementAccepted},
				{"proposal_reason_counts", proposalReasons}
			}},
			{"evaluation_reason_counts", evaluationReasons}
		});
	}
	return result;
}

nlohmann::json DVPObservabilityJson(
	const PatchMatch::Params& params,
	const InstrumentExtendedMaps* maps)
{
	nlohmann::json result(DVPContractJson(params));
	const bool requested(maps && maps->dvpRequested);
	const bool available(maps && maps->dvpAvailable);
	result["schema_name"] = "openmvs.dmap.dvp_observability";
	result["schema_version"] = PM_DVP_INSTRUMENT_SCHEMA_VERSION;
	result["requested"] = requested;
	result["summary_available"] = available;
	result["maps_available"] = maps && maps->dvpMapsAvailable;
	result["targeted_trace_available"] = available && maps->traceAvailable && !maps->dvpTraces.empty();
	result["counter_record_bytes"] = sizeof(PatchMatchDVPInstrumentCounters);
	result["update_record_bytes"] = sizeof(PatchMatchDVPInstrumentUpdate);
	result["trace_record_bytes"] = sizeof(PatchMatchDVPInstrumentTrace);
	result["aggregate_sum_precision"] = "float64 device atomic; counts are uint32 device atomic";
	result["measurement_basis"] = available ?
		"exact active Process<true> candidate generation, scoring, acceptance, and retained outcome" : "unavailable";
	result["observer_specialization_risk"] =
		"Process<true> is a separately compiled observer kernel and can perturb register pressure or scheduling; production quality uses Process<false>";
	result["iterations"] = nlohmann::json::array();
	if (!available)
		return result;
	auto optionalMean = [](double sum, uint32_t count) {
		return count > 0u ? nlohmann::json(sum/(double)count) : nlohmann::json(nullptr);
	};
	auto ratio = [](uint32_t numerator, uint32_t denominator) {
		return denominator > 0u ? nlohmann::json((double)numerator/(double)denominator) : nlohmann::json(nullptr);
	};
	for (size_t iteration=0; iteration<maps->dvpCounters.size(); ++iteration) {
		const PatchMatchDVPInstrumentCounters& counter(maps->dvpCounters[iteration]);
		nlohmann::json unavailableReasons(nlohmann::json::object());
		for (int reason=0; reason<PM_DVP_INSTRUMENT_UNAVAILABLE_REASONS; ++reason)
			unavailableReasons[DVPUnavailableReasonName((unsigned)reason)] = counter.unavailableReason[reason];
		nlohmann::json finalSources(nlohmann::json::object());
		for (int source=0; source<PM_DVP_INSTRUMENT_FINAL_SOURCES; ++source)
			finalSources[InstrumentSourceName(source)] = counter.finalUpdateSource[source];
		result["iterations"].push_back({
			{"logical_iteration", iteration},
			{"attempted_pixels", counter.attemptedPixels},
			{"proposal_available_pixels", counter.proposalAvailablePixels},
			{"proposal_available_ratio", ratio(counter.proposalAvailablePixels, counter.attemptedPixels)},
			{"native_depth_fallback_pixels", counter.nativeDepthFallbackPixels},
			{"native_depth_fallback_ratio", ratio(counter.nativeDepthFallbackPixels, counter.attemptedPixels)},
			{"proposals", {
				{"generated", counter.proposalsGenerated},
				{"tested", counter.proposalsTested},
				{"finite", counter.proposalsFinite},
				{"accepted_events", counter.proposalsAccepted},
				{"final_winner_pixels", counter.finalWinnerPixels},
				{"final_winner_ratio", ratio(counter.finalWinnerPixels, counter.attemptedPixels)},
				{"final_depth_retained_pixels", counter.finalDepthRetainedPixels},
				{"final_depth_retained_ratio", ratio(counter.finalDepthRetainedPixels, counter.attemptedPixels)},
				{"final_update_source_counts", finalSources},
				{"support_mean", optionalMean((float)counter.proposalSupportSum, counter.proposalSupportSamples)},
				{"occluded_candidates", counter.occludedCandidates}
			}},
			{"intervals", {
				{"left_valid_pixels", counter.leftIntervalValidPixels},
				{"right_valid_pixels", counter.rightIntervalValidPixels},
				{"left_valid_ratio", ratio(counter.leftIntervalValidPixels, counter.attemptedPixels)},
				{"right_valid_ratio", ratio(counter.rightIntervalValidPixels, counter.attemptedPixels)},
				{"endpoint_support_mean", {
					{"left_outer", optionalMean((float)counter.endpointSupportSum[0], counter.attemptedPixels)},
					{"left_inner", optionalMean((float)counter.endpointSupportSum[1], counter.attemptedPixels)},
					{"right_inner", optionalMean((float)counter.endpointSupportSum[2], counter.attemptedPixels)},
					{"right_outer", optionalMean((float)counter.endpointSupportSum[3], counter.attemptedPixels)}
				}}
			}},
			{"views", {
				{"selected_source_count_mean", optionalMean((float)counter.selectedSourceViewCountSum, counter.attemptedPixels)},
				{"direction_source_count_mean", optionalMean((float)counter.directionSourceViewCountSum, counter.attemptedPixels)}
			}},
			{"costs", {
				{"samples", counter.costSamples},
				{"incumbent_mean", optionalMean(counter.incumbentCostSum, counter.costSamples)},
				{"winner_mean", optionalMean(counter.winnerCostSum, counter.costSamples)},
				{"improvement_mean", optionalMean(counter.improvementSum, counter.costSamples)},
				{"winner_runner_up_gap_mean", optionalMean(counter.gapSum, counter.gapSamples)}
			}},
			{"geometry", {
				{"depth_displacement_mean", optionalMean(counter.displacementSum, counter.displacementSamples)},
				{"reprojection_error_mean_px", optionalMean(counter.reprojectionErrorSum, counter.reprojectionErrorSamples)},
				{"relative_depth_error_mean", optionalMean(counter.relativeDepthErrorSum, counter.relativeDepthErrorSamples)}
			}},
			{"unavailable_reason_counts", unavailableReasons}
		});
	}
	return result;
}

const char* ExactCandidateAcceptedSemantics(bool initialization)
{
	return initialization ?
		"production stored-assignment count; initialization records the production assignment even when no candidate cost is usable/finite, so accepted can exceed finite" :
		"sequential incumbent-improvement count; every accepted candidate improved the incumbent at that point, and a later candidate can replace an earlier accepted candidate";
}

const char* DMapCandidateSourceName(int source)
{
	switch (source) {
	case DMAP_SOURCE_UNKNOWN: return "UNKNOWN";
	case DMAP_SOURCE_INIT_RANDOM: return "INIT_RANDOM";
	case DMAP_SOURCE_INIT_SPARSE_OR_EXISTING: return "INIT_SPARSE_OR_EXISTING";
	case DMAP_SOURCE_SPATIAL_PROPAGATION: return "SPATIAL_PROPAGATION";
	case DMAP_SOURCE_VIEW_PROPAGATION: return "VIEW_PROPAGATION";
	case DMAP_SOURCE_RANDOM_PERTURBATION: return "RANDOM_PERTURBATION";
	case DMAP_SOURCE_REFINEMENT: return "REFINEMENT";
	case DMAP_SOURCE_PRIOR_OR_GUIDANCE: return "PRIOR_OR_GUIDANCE";
	case DMAP_SOURCE_OTHER: return "OTHER";
	default: return "UNKNOWN";
	}
}

const char* DMapRejectionReasonName(int reason)
{
	switch (reason) {
	case DMAP_REJECTION_NONE_SURVIVED: return "NONE_SURVIVED";
	case DMAP_REJECTION_UNKNOWN_REJECTED: return "UNKNOWN_REJECTED";
	case DMAP_REJECTION_LOW_SCORE: return "LOW_SCORE";
	case DMAP_REJECTION_INSUFFICIENT_VIEW_SUPPORT: return "INSUFFICIENT_VIEW_SUPPORT";
	case DMAP_REJECTION_GEOMETRIC_INCONSISTENCY: return "GEOMETRIC_INCONSISTENCY";
	case DMAP_REJECTION_NORMAL_INCONSISTENCY: return "NORMAL_INCONSISTENCY";
	case DMAP_REJECTION_DEPTH_RANGE: return "DEPTH_RANGE";
	case DMAP_REJECTION_OCCLUSION: return "OCCLUSION";
	case DMAP_REJECTION_MASKED: return "MASKED";
	case DMAP_REJECTION_SMALL_COMPONENT: return "SMALL_COMPONENT";
	case DMAP_REJECTION_OTHER: return "OTHER";
	default: return "UNKNOWN_REJECTED";
	}
}

int DMapCandidateSourceFromPatchMatch(uint8_t source)
{
	switch (source) {
	case PM_SOURCE_INIT: return DMAP_SOURCE_INIT_RANDOM;
	case PM_SOURCE_PROPAGATE: return DMAP_SOURCE_SPATIAL_PROPAGATION;
	case PM_SOURCE_APD_ANCHOR_PROPAGATE: return DMAP_SOURCE_SPATIAL_PROPAGATION;
	case PM_SOURCE_APD_FITTED_PLANE:
	case PM_SOURCE_APD_FINAL_REFINEMENT:
	case PM_SOURCE_DVP_EPIPOLAR:
		return DMAP_SOURCE_REFINEMENT;
	case PM_SOURCE_REFINE_RANDOM_NORMAL: return DMAP_SOURCE_RANDOM_PERTURBATION;
	case PM_SOURCE_REFINE_DEPTH:
	case PM_SOURCE_REFINE_NORMAL:
	case PM_SOURCE_REFINE_SURFACE_NORMAL:
		return DMAP_SOURCE_REFINEMENT;
	case PM_SOURCE_FILTERED:
	case PM_SOURCE_NONE:
		return DMAP_SOURCE_UNKNOWN;
	default:
		return DMAP_SOURCE_OTHER;
	}
}

void EnsureInstrumentConfigLoaded()
{
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	if (g_instrumentConfig.loaded)
		return;
	g_instrumentConfig.loaded = true;
	g_instrumentConfig.tracePixels.clear();
	if (OPTDENSE::strPatchMatchInstrumentConfig.empty())
		return;
	std::ifstream fs(MAKE_PATH_SAFE(OPTDENSE::strPatchMatchInstrumentConfig));
	if (!fs) {
		VERBOSE("warning: cannot open CUDA PatchMatch instrumentation config '%s'", OPTDENSE::strPatchMatchInstrumentConfig.c_str());
		return;
	}
	nlohmann::json data = nlohmann::json::parse(fs, nullptr, false);
	if (data.is_discarded()) {
		VERBOSE("warning: cannot parse CUDA PatchMatch instrumentation config '%s'", OPTDENSE::strPatchMatchInstrumentConfig.c_str());
		return;
	}
	const auto it = data.find("trace_pixels");
	if (it == data.end() || !it->is_array())
		return;
	if (it->size() > PM_INSTRUMENT_MAX_TRACE_CONFIG_PIXELS) {
		VERBOSE("error: CUDA PatchMatch instrumentation config contains %llu trace pixels; maximum is %llu",
			(unsigned long long)it->size(),
			(unsigned long long)PM_INSTRUMENT_MAX_TRACE_CONFIG_PIXELS);
		std::exit(EXIT_FAILURE);
	}
	size_t totalLabelBytes(0);
	for (const nlohmann::json& px : *it) {
		if (!px.is_object())
			continue;
		InstrumentTracePixel tracePixel;
		tracePixel.imageID = px.value("image_id", -1);
		tracePixel.x = px.value("x", -1);
		tracePixel.y = px.value("y", -1);
		const std::string label = px.value("label", std::string());
		if (label.size() > PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES) {
			VERBOSE("error: CUDA PatchMatch trace label contains %llu bytes; maximum is %llu",
				(unsigned long long)label.size(),
				(unsigned long long)PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES);
			std::exit(EXIT_FAILURE);
		}
		totalLabelBytes += label.size();
		if (totalLabelBytes > PM_INSTRUMENT_MAX_TRACE_CONFIG_LABEL_BYTES) {
			VERBOSE("error: CUDA PatchMatch trace labels exceed the %llu-byte configuration limit",
				(unsigned long long)PM_INSTRUMENT_MAX_TRACE_CONFIG_LABEL_BYTES);
			std::exit(EXIT_FAILURE);
		}
		tracePixel.label = label.c_str();
		if (tracePixel.imageID >= 0 && tracePixel.x >= 0 && tracePixel.y >= 0)
			g_instrumentConfig.tracePixels.emplace_back(std::move(tracePixel));
	}
}

const char* DMapEstimationStageName(int geometricIteration)
{
	return geometricIteration >= 0 ? "geometric_consistency" : "photometric";
}

String InstrumentRunRoot(int geometricIteration=-1)
{
	String root(OPTDENSE::strPatchMatchInstrumentOutput.empty() ? _T("pm_instrumentation") : OPTDENSE::strPatchMatchInstrumentOutput);
	Util::ensureFolderSlash(root);
	Util::ensureFolder(root);
	if (geometricIteration >= 0) {
		root += _T("geometric_iterations/");
		Util::ensureFolder(root);
		root += String::FormatString(_T("iteration%02d/"), geometricIteration);
		Util::ensureFolder(root);
	}
	return root;
}

String InstrumentOutputRoot(int geometricIteration=-1)
{
	const String root(InstrumentRunRoot(geometricIteration));
	const String dir(root + _T("instrumentation/"));
	Util::ensureFolder(dir);
	Util::ensureFolder(dir + _T("costs/"));
	Util::ensureFolder(dir + _T("improvements/"));
	Util::ensureFolder(dir + _T("maps/"));
	return dir;
}

String TrimToken(const String& token)
{
	size_t begin = 0;
	size_t end = token.size();
	while (begin < end && std::isspace((unsigned char)token[begin]))
		++begin;
	while (end > begin && std::isspace((unsigned char)token[end-1]))
		--end;
	return token.substr(begin, end-begin);
}

bool InstrumentImageListed(int imageID, const String& imageName)
{
	if (OPTDENSE::strDMapInstrumentationImageList.empty())
		return true;
	const String fileName(Util::getFileNameExt(imageName));
	const String stem(Util::getFileName(imageName));
	const String imageIDText(std::to_string(imageID).c_str());
	size_t start = 0;
	while (start <= OPTDENSE::strDMapInstrumentationImageList.size()) {
		const size_t comma(OPTDENSE::strDMapInstrumentationImageList.find(',', start));
		const size_t end(comma == String::npos ? OPTDENSE::strDMapInstrumentationImageList.size() : comma);
		const String token(TrimToken(OPTDENSE::strDMapInstrumentationImageList.substr(start, end-start)));
		if (!token.empty() && (token == imageIDText || token == imageName || token == fileName || token == stem))
			return true;
		if (comma == String::npos)
			break;
		start = comma + 1;
	}
	return false;
}

uint32_t InstrumentImageSampleHash(int imageID)
{
	uint32_t hash(uint32_t(imageID) ^ (OPTDENSE::nDMapInstrumentationSampleSeed + 0x9e3779b9u));
	hash ^= hash >> 16;
	hash *= 0x7feb352du;
	hash ^= hash >> 15;
	hash *= 0x846ca68bu;
	hash ^= hash >> 16;
	return hash;
}

float InstrumentImageSampleValue(int imageID)
{
	return float(InstrumentImageSampleHash(imageID) % 1000000u) / 1000000.f;
}

bool InstrumentImageSampled(int imageID)
{
	const float sampleRate(OPTDENSE::fDMapInstrumentationSampleRate);
	return sampleRate >= 1.f || (sampleRate > 0.f && InstrumentImageSampleValue(imageID) < sampleRate);
}

bool InstrumentImageEnabled(int imageID, const String& imageName)
{
	if (OPTDENSE::nPatchMatchInstrumentLevel == 0)
		return false;
	if (!OPTDENSE::strDMapInstrumentationDir.empty() && !InstrumentImageListed(imageID, imageName))
		return false;
	if (!OPTDENSE::strDMapInstrumentationDir.empty() && !InstrumentImageSampled(imageID))
		return false;
	return true;
}

bool InstrumentWriteMaps()
{
	return OPTDENSE::bDMapInstrumentationWriteMaps || OPTDENSE::strDMapInstrumentationLevel.ToLower() == _T("maps");
}

bool InstrumentPrefilterRequested()
{
	return OPTDENSE::strDMapInstrumentationLevel.ToLower() == _T("prefilter");
}

uint64_t SaturatingAdd(uint64_t first, uint64_t second)
{
	return first > std::numeric_limits<uint64_t>::max() - second ?
		std::numeric_limits<uint64_t>::max() : first + second;
}

uint64_t SaturatingMul(uint64_t first, uint64_t second)
{
	return first && second > std::numeric_limits<uint64_t>::max() / first ?
		std::numeric_limits<uint64_t>::max() : first * second;
}

bool InstrumentResourceFits(uint64_t bytes, unsigned limitMB)
{
	return limitMB == 0 || bytes <= SaturatingMul((uint64_t)limitMB, 1024u * 1024u);
}

InstrumentExtendedMaps PlanInstrumentResources(
	int area,
	int numPasses,
	int numLogicalStates,
	int numViews,
	int numTracePixels,
	uint64_t traceLabelBytes,
	uint64_t frameStorageCommittedBeforeBytes,
	uint64_t frameStoragePriorityReserveBytes,
	bool compatibilityMapsRequested,
	bool mapsRequested,
	bool exactRequested,
	bool prefilterRequested,
	bool depthEdgeMapsRequested,
	bool apdRequested,
	bool dvpRequested,
	bool visibilityRequested,
	bool visibleNormalRequested)
{
	InstrumentExtendedMaps plan;
	plan.numLogicalStates = numLogicalStates;
	plan.numViews = numViews;
	plan.numTracePixels = numTracePixels;
	plan.traceRequested = numTracePixels > 0;
	plan.compatibilityMapsRequested = compatibilityMapsRequested;
	plan.mapsRequested = mapsRequested;
	plan.exactRequested = exactRequested;
	plan.prefilterRequested = prefilterRequested;
	plan.apdRequested = apdRequested;
	plan.dvpRequested = dvpRequested;
	plan.visibilityRequested = visibilityRequested;
	plan.visibleNormalRequested = visibleNormalRequested;
	plan.frameStorageCommittedBeforeBytes = frameStorageCommittedBeforeBytes;
	plan.frameStoragePriorityReserveBytes = frameStoragePriorityReserveBytes;
	plan.storagePreflightDecision = _T("pending");
	const uint64_t pixels((uint64_t)MAXF(area, 0));
	const uint64_t passes((uint64_t)MAXF(numPasses, 0));
	const uint64_t states((uint64_t)MAXF(numLogicalStates, 0));
	const uint64_t views((uint64_t)MAXF(numViews, 0));
	const uint64_t traces((uint64_t)MAXF(numTracePixels, 0));
	const uint64_t apdStates((uint64_t)MAXF(numLogicalStates-1, 0));
	const uint64_t dvpStates((uint64_t)MAXF(numLogicalStates-1, 0));
	const uint64_t visibilityStates((uint64_t)MAXF(numLogicalStates-1, 0));
	const uint64_t visibleNormalStates((uint64_t)MAXF(numLogicalStates-1, 0));
	const uint64_t traceRecords(SaturatingMul(traces, passes));
	const uint64_t traceMapBytes(traces > 0 ? SaturatingMul(pixels, sizeof(int32_t)) : 0u);
	plan.traceDeviceBytes = SaturatingAdd(traceMapBytes,
		SaturatingMul(traceRecords, sizeof(PatchMatchInstrumentTraceRecord)));
	plan.traceHostBytes = SaturatingAdd(
		plan.traceDeviceBytes,
		SaturatingAdd(
			SaturatingMul(traces,
				SaturatingAdd(sizeof(InstrumentTracePixel), sizeof(InstrumentTraceSelection))),
			traceLabelBytes));
	plan.traceStorageBytes = traces > 0 ? SaturatingAdd(
		64u * 1024u,
		SaturatingAdd(
			SaturatingMul(traceRecords, 16u * 1024u),
			SaturatingMul(SaturatingMul(traceLabelBytes, passes), 6u))) : 0u;
	if (apdRequested) {
		plan.apdSummaryDeviceBytes = SaturatingMul(
			apdStates, sizeof(PatchMatchAPDInstrumentCounters));
		plan.apdSummaryHostBytes = plan.apdSummaryDeviceBytes;
		const uint64_t apdTraceRecords(SaturatingMul(traces, apdStates));
		plan.apdTraceDeviceBytes = SaturatingMul(
			apdTraceRecords, sizeof(PatchMatchAPDInstrumentTrace));
		plan.apdTraceHostBytes = plan.apdTraceDeviceBytes;
		plan.apdTraceStorageBytes = apdTraceRecords > 0 ? SaturatingAdd(
			64u * 1024u, SaturatingMul(apdTraceRecords, 64u * 1024u)) : 0u;
		plan.apdMapDeviceBytes = mapsRequested ? SaturatingMul(
			SaturatingMul(pixels, apdStates),
			sizeof(PatchMatchAPDInstrumentState)+sizeof(PatchMatchAPDInstrumentUpdate)) : 0u;
		plan.apdMapHostBytes = mapsRequested ? SaturatingAdd(
			plan.apdMapDeviceBytes,
			SaturatingMul(pixels, DMAP_APD_MULTISCALE_MAP_STORAGE_BYTES_PER_PIXEL)) : 0u;
		plan.apdMapStorageBytes = mapsRequested ? SaturatingAdd(
			SaturatingMul(SaturatingMul(pixels, apdStates),
				DMAP_APD_MAP_STORAGE_BYTES_PER_PIXEL),
			SaturatingAdd(
				SaturatingMul(SaturatingMul(apdStates,
					DMAP_APD_MAP_ARTIFACTS_PER_ITERATION), 4096u),
				SaturatingAdd(
					64u * 1024u,
					SaturatingAdd(
						SaturatingMul(pixels,
							DMAP_APD_MULTISCALE_MAP_STORAGE_BYTES_PER_PIXEL),
						SaturatingMul(DMAP_APD_MULTISCALE_MAP_ARTIFACTS, 4096u))))) : 0u;
	}
	if (dvpRequested) {
		plan.dvpSummaryDeviceBytes = SaturatingMul(
			dvpStates, sizeof(PatchMatchDVPInstrumentCounters));
		plan.dvpSummaryHostBytes = plan.dvpSummaryDeviceBytes;
		const uint64_t dvpTraceRecords(SaturatingMul(traces, dvpStates));
		plan.dvpTraceDeviceBytes = SaturatingMul(
			dvpTraceRecords, sizeof(PatchMatchDVPInstrumentTrace));
		plan.dvpTraceHostBytes = plan.dvpTraceDeviceBytes;
		plan.dvpTraceStorageBytes = dvpTraceRecords > 0 ? SaturatingAdd(
			64u * 1024u, SaturatingMul(dvpTraceRecords, 16u * 1024u)) : 0u;
		plan.dvpMapDeviceBytes = mapsRequested ? SaturatingMul(
			SaturatingMul(pixels, dvpStates), sizeof(PatchMatchDVPInstrumentUpdate)) : 0u;
		plan.dvpMapHostBytes = plan.dvpMapDeviceBytes;
		plan.dvpMapStorageBytes = mapsRequested ? SaturatingAdd(
			SaturatingMul(SaturatingMul(pixels, dvpStates),
				DMAP_DVP_MAP_STORAGE_BYTES_PER_PIXEL),
			SaturatingAdd(
				SaturatingMul(SaturatingMul(dvpStates,
					DMAP_DVP_MAP_ARTIFACTS_PER_ITERATION), 4096u),
				64u * 1024u)) : 0u;
	}
	if (visibilityRequested) {
		plan.visibilitySummaryDeviceBytes = SaturatingAdd(
			SaturatingMul(visibilityStates,
				sizeof(PatchMatchDVPVisibilityInstrumentCounters)),
			SaturatingMul(pixels, PM_INSTRUMENT_MAX_VIEWS*sizeof(uint8_t)));
		plan.visibilitySummaryHostBytes = SaturatingMul(
			visibilityStates, sizeof(PatchMatchDVPVisibilityInstrumentCounters));
		const uint64_t visibilityTraceRecords(SaturatingMul(traces, visibilityStates));
		plan.visibilityTraceDeviceBytes = SaturatingMul(
			visibilityTraceRecords, sizeof(PatchMatchDVPVisibilityInstrumentTrace));
		plan.visibilityTraceHostBytes = plan.visibilityTraceDeviceBytes;
		plan.visibilityTraceStorageBytes = visibilityTraceRecords > 0 ? SaturatingAdd(
			64u * 1024u, SaturatingMul(visibilityTraceRecords, 32u * 1024u)) : 0u;
		plan.visibilityMapDeviceBytes = mapsRequested ? SaturatingMul(
			SaturatingMul(pixels, visibilityStates),
			sizeof(PatchMatchDVPVisibilityInstrumentUpdate)) : 0u;
		plan.visibilityMapHostBytes = plan.visibilityMapDeviceBytes;
		plan.visibilityMapStorageBytes = mapsRequested ? SaturatingAdd(
			SaturatingMul(SaturatingMul(pixels, visibilityStates),
				DMAP_DVP_VISIBILITY_MAP_STORAGE_BYTES_PER_PIXEL),
			SaturatingAdd(
				SaturatingMul(SaturatingMul(visibilityStates,
					DMAP_DVP_VISIBILITY_MAP_ARTIFACTS_PER_ITERATION), 4096u),
				64u * 1024u)) : 0u;
	}
	if (visibleNormalRequested) {
		plan.visibleNormalSummaryDeviceBytes = SaturatingMul(
			visibleNormalStates, sizeof(PatchMatchDVPVisibleNormalInstrumentCounters));
		plan.visibleNormalSummaryHostBytes = plan.visibleNormalSummaryDeviceBytes;
		const uint64_t visibleNormalTraceRecords(
			SaturatingMul(traces, visibleNormalStates));
		plan.visibleNormalTraceDeviceBytes = SaturatingMul(
			visibleNormalTraceRecords, sizeof(PatchMatchDVPVisibleNormalInstrumentTrace));
		plan.visibleNormalTraceHostBytes = plan.visibleNormalTraceDeviceBytes;
		plan.visibleNormalTraceStorageBytes = visibleNormalTraceRecords > 0 ?
			SaturatingAdd(64u*1024u,
				SaturatingMul(visibleNormalTraceRecords, 48u*1024u)) : 0u;
		plan.visibleNormalMapDeviceBytes = mapsRequested ? SaturatingMul(
			SaturatingMul(pixels, visibleNormalStates),
			sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)) : 0u;
		plan.visibleNormalMapHostBytes = plan.visibleNormalMapDeviceBytes;
		plan.visibleNormalMapStorageBytes = mapsRequested ? SaturatingAdd(
			SaturatingMul(SaturatingMul(pixels, visibleNormalStates),
				DMAP_DVP_VISIBLE_NORMAL_MAP_STORAGE_BYTES_PER_PIXEL),
			SaturatingAdd(
				SaturatingMul(SaturatingMul(visibleNormalStates,
					DMAP_DVP_VISIBLE_NORMAL_MAP_ARTIFACTS_PER_ITERATION), 4096u),
				64u*1024u)) : 0u;
	}
	plan.summaryStorageBytes = SaturatingAdd(
		256u * 1024u,
		SaturatingMul(SaturatingAdd(passes, states), 8u * 1024u));
	if (apdRequested)
		plan.summaryStorageBytes = SaturatingAdd(
			plan.summaryStorageBytes,
				SaturatingAdd(64u * 1024u, SaturatingMul(apdStates, 8u * 1024u)));
	if (dvpRequested)
		plan.summaryStorageBytes = SaturatingAdd(
			plan.summaryStorageBytes,
			SaturatingAdd(64u * 1024u, SaturatingMul(dvpStates, 8u * 1024u)));
	if (visibilityRequested)
		plan.summaryStorageBytes = SaturatingAdd(
			plan.summaryStorageBytes,
			SaturatingAdd(64u * 1024u, SaturatingMul(visibilityStates, 8u * 1024u)));
	if (visibleNormalRequested)
		plan.summaryStorageBytes = SaturatingAdd(
			plan.summaryStorageBytes,
			SaturatingAdd(64u*1024u, SaturatingMul(visibleNormalStates, 8u*1024u)));
	if (compatibilityMapsRequested) {
		plan.summaryStorageBytes = SaturatingAdd(
			plan.summaryStorageBytes,
			SaturatingAdd(SaturatingMul(pixels, 6u), 3u * 4096u));
	}
	// Device summary buffers: counters, update source, and before-pass state.
	plan.summaryDeviceBytes = SaturatingAdd(
		SaturatingMul(sizeof(PatchMatchInstrumentCounters), passes),
		SaturatingMul(pixels, sizeof(uint8_t) + sizeof(Point4) + sizeof(float) + sizeof(uint32_t)));
	// Host summary peak retains pass and logical-iteration counters, the update map,
	// the production cost clone, timings, CostStats scratch, and fixed JSON/vector overhead.
	plan.summaryHostBytes = SaturatingAdd(
		DMAP_INSTRUMENT_FIXED_HOST_BYTES,
		SaturatingAdd(
			SaturatingMul(sizeof(PatchMatchInstrumentCounters), SaturatingAdd(passes, states)),
			SaturatingAdd(
				SaturatingMul(pixels,
					sizeof(uint8_t) + sizeof(float) + DMAP_SUMMARY_HOST_SCRATCH_BYTES_PER_PIXEL),
				SaturatingMul(passes, sizeof(float)))));
	// Legacy map capture retains pass snapshots, logical proxy rescoring, and final diagnostics.
	plan.legacyMapDeviceBytes = SaturatingMul(pixels,
		2u + passes * 18u + states * 36u + DMAP_LEGACY_TERMINAL_BYTES_PER_PIXEL);
	// Summary already budgets 4 B/pixel of transient scratch. Extended map export
	// peaks at 21 B/pixel, so the additive map tier carries the 17 B/pixel delta.
	plan.legacyMapHostBytes = SaturatingAdd(
		plan.legacyMapDeviceBytes,
		SaturatingMul(pixels,
			DMAP_MAP_EXPORT_HOST_SCRATCH_BYTES_PER_PIXEL -
			DMAP_SUMMARY_HOST_SCRATCH_BYTES_PER_PIXEL));
	// Conservative uncompressed estimate including compatibility aliases and legacy previews.
	plan.legacyMapStorageBytes = SaturatingMul(pixels,
		159u + passes * 6u + states * DMAP_LEGACY_LOGICAL_STATE_STORAGE_BYTES_PER_PIXEL);
	// Exact hot-kernel state: two float4 component records, one event record, and all source views.
	plan.exactDeviceBytes = SaturatingMul(pixels,
		SaturatingMul(states, 80u + SaturatingMul(views, sizeof(PatchMatchInstrumentExactView))));
	plan.exactHostBytes = plan.exactDeviceBytes;
	// Exact export uses scalar component/event maps plus grouped view-state maps.
	const uint64_t exactPayloadBytes(SaturatingMul(pixels,
		SaturatingMul(states, DMAP_EXACT_BASE_STATE_BYTES_PER_PIXEL + SaturatingMul(views, 34u))));
	const uint64_t exactArtifactCount(SaturatingMul(states,
		DMAP_EXACT_BASE_STATE_ARTIFACTS + SaturatingMul(views, DMAP_EXACT_VIEW_ARTIFACTS)));
	const uint64_t exactFileOverhead(states > 0 ? SaturatingAdd(
		SaturatingMul(exactArtifactCount, 4096u), 64u * 1024u) : 0u);
	plan.exactStorageBytes = SaturatingAdd(exactPayloadBytes, exactFileOverhead);
	// The bounded pre-filter profile retains one terminal float4 plane snapshot
	// on device and host, but publishes only its scalar depth channel.
	plan.prefilterDeviceBytes = prefilterRequested ?
		SaturatingMul(pixels, sizeof(Point4)) : 0u;
	// Summary scratch already covers the temporary scalar depth export buffer.
	plan.prefilterHostBytes = plan.prefilterDeviceBytes;
	plan.prefilterStorageBytes = prefilterRequested ?
		SaturatingAdd(SaturatingMul(pixels, sizeof(float)), 64u * 1024u) : 0u;
	if (depthEdgeMapsRequested) {
		const uint64_t depthEdgeMapDeviceBytes(SaturatingMul(pixels, 3u));
		const uint64_t depthEdgeMapHostBytes(SaturatingMul(pixels, 3u));
		const uint64_t depthEdgeMapStorageBytes(SaturatingMul(pixels, 5u));
		plan.dvpMapDeviceBytes = SaturatingAdd(
			plan.dvpMapDeviceBytes, depthEdgeMapDeviceBytes);
		plan.dvpMapHostBytes = SaturatingAdd(
			plan.dvpMapHostBytes, depthEdgeMapHostBytes);
		plan.dvpMapStorageBytes = SaturatingAdd(
			plan.dvpMapStorageBytes, depthEdgeMapStorageBytes);
		// Coarse depth-edge maps are compatibility evidence and remain available
		// with the summary tier. Fine-level maps follow the admitted maps tier.
		if (compatibilityMapsRequested) {
			plan.summaryDeviceBytes = SaturatingAdd(
				plan.summaryDeviceBytes, depthEdgeMapDeviceBytes);
			plan.summaryHostBytes = SaturatingAdd(
				plan.summaryHostBytes, depthEdgeMapHostBytes);
			plan.summaryStorageBytes = SaturatingAdd(
				plan.summaryStorageBytes, depthEdgeMapStorageBytes);
		}
	}
	plan.summaryDeviceBytes = SaturatingAdd(
		plan.summaryDeviceBytes, plan.apdSummaryDeviceBytes);
	plan.summaryHostBytes = SaturatingAdd(
		plan.summaryHostBytes, plan.apdSummaryHostBytes);
	plan.traceDeviceBytes = SaturatingAdd(
		plan.traceDeviceBytes, plan.apdTraceDeviceBytes);
	plan.traceHostBytes = SaturatingAdd(
		plan.traceHostBytes, plan.apdTraceHostBytes);
	plan.traceStorageBytes = SaturatingAdd(
		plan.traceStorageBytes, plan.apdTraceStorageBytes);
	plan.legacyMapDeviceBytes = SaturatingAdd(
		plan.legacyMapDeviceBytes, plan.apdMapDeviceBytes);
	plan.legacyMapHostBytes = SaturatingAdd(
		plan.legacyMapHostBytes, plan.apdMapHostBytes);
	plan.legacyMapStorageBytes = SaturatingAdd(
		plan.legacyMapStorageBytes, plan.apdMapStorageBytes);
	plan.summaryDeviceBytes = SaturatingAdd(
		plan.summaryDeviceBytes, plan.dvpSummaryDeviceBytes);
	plan.summaryHostBytes = SaturatingAdd(
		plan.summaryHostBytes, plan.dvpSummaryHostBytes);
	plan.traceDeviceBytes = SaturatingAdd(
		plan.traceDeviceBytes, plan.dvpTraceDeviceBytes);
	plan.traceHostBytes = SaturatingAdd(
		plan.traceHostBytes, plan.dvpTraceHostBytes);
	plan.traceStorageBytes = SaturatingAdd(
		plan.traceStorageBytes, plan.dvpTraceStorageBytes);
	plan.legacyMapDeviceBytes = SaturatingAdd(
		plan.legacyMapDeviceBytes, plan.dvpMapDeviceBytes);
	plan.legacyMapHostBytes = SaturatingAdd(
		plan.legacyMapHostBytes, plan.dvpMapHostBytes);
	plan.legacyMapStorageBytes = SaturatingAdd(
		plan.legacyMapStorageBytes, plan.dvpMapStorageBytes);
	plan.summaryDeviceBytes = SaturatingAdd(
		plan.summaryDeviceBytes, plan.visibilitySummaryDeviceBytes);
	plan.summaryHostBytes = SaturatingAdd(
		plan.summaryHostBytes, plan.visibilitySummaryHostBytes);
	plan.traceDeviceBytes = SaturatingAdd(
		plan.traceDeviceBytes, plan.visibilityTraceDeviceBytes);
	plan.traceHostBytes = SaturatingAdd(
		plan.traceHostBytes, plan.visibilityTraceHostBytes);
	plan.traceStorageBytes = SaturatingAdd(
		plan.traceStorageBytes, plan.visibilityTraceStorageBytes);
	plan.legacyMapDeviceBytes = SaturatingAdd(
		plan.legacyMapDeviceBytes, plan.visibilityMapDeviceBytes);
	plan.legacyMapHostBytes = SaturatingAdd(
		plan.legacyMapHostBytes, plan.visibilityMapHostBytes);
	plan.legacyMapStorageBytes = SaturatingAdd(
		plan.legacyMapStorageBytes, plan.visibilityMapStorageBytes);
	plan.summaryDeviceBytes = SaturatingAdd(
		plan.summaryDeviceBytes, plan.visibleNormalSummaryDeviceBytes);
	plan.summaryHostBytes = SaturatingAdd(
		plan.summaryHostBytes, plan.visibleNormalSummaryHostBytes);
	plan.traceDeviceBytes = SaturatingAdd(
		plan.traceDeviceBytes, plan.visibleNormalTraceDeviceBytes);
	plan.traceHostBytes = SaturatingAdd(
		plan.traceHostBytes, plan.visibleNormalTraceHostBytes);
	plan.traceStorageBytes = SaturatingAdd(
		plan.traceStorageBytes, plan.visibleNormalTraceStorageBytes);
	plan.legacyMapDeviceBytes = SaturatingAdd(
		plan.legacyMapDeviceBytes, plan.visibleNormalMapDeviceBytes);
	plan.legacyMapHostBytes = SaturatingAdd(
		plan.legacyMapHostBytes, plan.visibleNormalMapHostBytes);
	plan.legacyMapStorageBytes = SaturatingAdd(
		plan.legacyMapStorageBytes, plan.visibleNormalMapStorageBytes);

	auto total = [](uint64_t first, uint64_t second, uint64_t third = 0, uint64_t fourth = 0) {
		return SaturatingAdd(SaturatingAdd(first, second), SaturatingAdd(third, fourth));
	};
	auto fits = [&](uint64_t device, uint64_t host, uint64_t currentStorage) {
		return InstrumentResourceFits(device, OPTDENSE::nDMapInstrumentationMaxDeviceMB) &&
			InstrumentResourceFits(host, OPTDENSE::nDMapInstrumentationMaxHostMB) &&
			InstrumentResourceFits(
				SaturatingAdd(SaturatingAdd(frameStorageCommittedBeforeBytes, currentStorage),
					frameStoragePriorityReserveBytes),
				OPTDENSE::nDMapInstrumentationMaxFrameStorageMB);
	};
	auto admit = [&](bool maps, bool exact, bool trace, bool prefilter,
		const String& decision, uint64_t device, uint64_t host, uint64_t currentStorage) {
		plan.summaryAvailable = true;
		plan.mapsAvailable = maps;
		plan.exactAvailable = exact;
		plan.traceAvailable = trace && plan.traceRequested;
		plan.prefilterAvailable = prefilter && plan.prefilterRequested;
		SetAlgorithmInstrumentAvailability(plan, true, maps);
		plan.resourceDecision = decision;
		plan.estimatedDeviceBytes = device;
		plan.estimatedHostBytes = host;
		plan.estimatedStorageBytes = currentStorage;
		plan.frameStorageBudgetBytes = SaturatingAdd(frameStorageCommittedBeforeBytes, currentStorage);
	};
	auto tier = [&](uint64_t mapDevice, uint64_t mapHost, uint64_t mapStorage, bool trace) {
		return std::array<uint64_t, 3>{
			total(plan.summaryDeviceBytes, trace ? plan.traceDeviceBytes : 0u, mapDevice),
			total(plan.summaryHostBytes, trace ? plan.traceHostBytes : 0u, mapHost),
			total(plan.summaryStorageBytes, trace ? plan.traceStorageBytes : 0u, mapStorage)
		};
	};
	plan.summaryAvailable = false;
	const bool withTrace(plan.traceRequested);
	if (mapsRequested && exactRequested) {
		for (int attempt = 0; attempt < (withTrace ? 2 : 1); ++attempt) {
			const bool trace(withTrace && attempt == 0);
			const auto values(tier(
				SaturatingAdd(plan.legacyMapDeviceBytes, plan.exactDeviceBytes),
				SaturatingAdd(plan.legacyMapHostBytes, plan.exactHostBytes),
				SaturatingAdd(plan.legacyMapStorageBytes, plan.exactStorageBytes), trace));
			if (fits(values[0], values[1], values[2])) {
				admit(true, true, trace, false,
					trace || !withTrace ? _T("exact_maps") : _T("exact_maps_trace_budget_degraded"),
					values[0], values[1], values[2]);
				if (withTrace && !trace)
					plan.traceUnavailableReason = _T("targeted trace capture exceeds a configured device, host, or cumulative frame-storage budget");
				return plan;
			}
		}
	}
	if (mapsRequested) {
		for (int attempt = 0; attempt < (withTrace ? 2 : 1); ++attempt) {
			const bool trace(withTrace && attempt == 0);
			const auto values(tier(plan.legacyMapDeviceBytes, plan.legacyMapHostBytes,
				plan.legacyMapStorageBytes, trace));
			if (fits(values[0], values[1], values[2])) {
				admit(true, false, trace, false,
					exactRequested ? _T("legacy_maps_budget_degraded") : _T("legacy_maps"),
					values[0], values[1], values[2]);
				plan.exactUnavailableReason = exactRequested ?
					_T("exact capture exceeds a configured device, host, or cumulative frame-storage budget") :
					_T("exact capture was not requested");
				if (withTrace && !trace)
					plan.traceUnavailableReason = _T("targeted trace capture exceeds a configured device, host, or cumulative frame-storage budget");
				return plan;
			}
		}
	}
	if (prefilterRequested) {
		for (int attempt = 0; attempt < (withTrace ? 2 : 1); ++attempt) {
			const bool trace(withTrace && attempt == 0);
			const auto values(tier(plan.prefilterDeviceBytes, plan.prefilterHostBytes,
				plan.prefilterStorageBytes, trace));
			if (fits(values[0], values[1], values[2])) {
				admit(false, false, trace, true,
					trace || !withTrace ? _T("prefilter") : _T("prefilter_trace_budget_degraded"),
					values[0], values[1], values[2]);
				plan.exactUnavailableReason = _T("exact map capture was not requested");
				if (withTrace && !trace)
					plan.traceUnavailableReason = _T("targeted trace capture exceeds a configured device, host, or cumulative frame-storage budget");
				return plan;
			}
		}
	}
	for (int attempt = 0; attempt < (withTrace ? 2 : 1); ++attempt) {
		const bool trace(withTrace && attempt == 0);
		const auto values(tier(0u, 0u, 0u, trace));
		if (fits(values[0], values[1], values[2])) {
			admit(false, false, trace, false,
				mapsRequested ? _T("summary_budget_degraded") :
					prefilterRequested ? _T("summary_prefilter_budget_degraded") :
					(trace || !withTrace ? _T("summary") : _T("summary_trace_budget_degraded")),
				values[0], values[1], values[2]);
			plan.exactUnavailableReason = mapsRequested ?
				_T("map capture exceeds a configured device, host, or cumulative frame-storage budget") :
				_T("exact capture was not requested");
			if (prefilterRequested)
				plan.prefilterUnavailableReason = _T("pre-filter snapshot exceeds a configured device, host, or cumulative frame-storage budget");
			if (withTrace && !trace)
				plan.traceUnavailableReason = _T("targeted trace capture exceeds a configured device, host, or cumulative frame-storage budget");
			return plan;
		}
	}
	plan.resourceDecision = _T("disabled_budget");
	plan.storagePreflightDecision = _T("not_attempted_configured_budget_rejected");
	plan.storagePreflightReason = _T("no capture tier fits the configured device, host, cumulative frame-storage, and full-resolution-priority budgets");
	plan.exactUnavailableReason = mapsRequested ?
		_T("map capture exceeds a configured device, host, or cumulative frame-storage budget") :
		_T("exact capture was not requested");
	if (prefilterRequested)
		plan.prefilterUnavailableReason = _T("pre-filter snapshot exceeds a configured device, host, or cumulative frame-storage budget");
	if (withTrace)
		plan.traceUnavailableReason = _T("targeted trace capture exceeds a configured device, host, or cumulative frame-storage budget");
	plan.frameStorageBudgetBytes = frameStorageCommittedBeforeBytes;
	return plan;
}

void ApplyInstrumentStoragePreflight(
	const String& root,
	InstrumentExtendedMaps& plan,
	InstrumentStorageReservation& reservation,
	InstrumentStorageReservation& framePriorityReservation,
	bool consumeFramePriorityReservation)
{
	if (plan.estimatedStorageBytes == 0) {
		plan.storagePreflightDecision = _T("not_requested");
		return;
	}

	plan.storagePreflightAttempted = true;
	plan.storageRequestedBytes = plan.estimatedStorageBytes;
	const bool failPolicy(OPTDENSE::strDMapInstrumentationBudgetPolicy.ToLower() == _T("error"));
	std::lock_guard<std::mutex> lock(g_instrumentMutex);

	String baseRoot(OPTDENSE::strPatchMatchInstrumentOutput.empty() ?
		_T("pm_instrumentation") : OPTDENSE::strPatchMatchInstrumentOutput);
	std::filesystem::path keyPath(baseRoot.c_str());
	std::error_code keyError;
	const std::filesystem::path canonicalPath(std::filesystem::weakly_canonical(keyPath, keyError));
	if (!keyError)
		keyPath = canonicalPath;
	else {
		keyError.clear();
		const std::filesystem::path absolutePath(std::filesystem::absolute(keyPath, keyError));
		if (!keyError)
			keyPath = absolutePath.lexically_normal();
		else
			keyPath = keyPath.lexically_normal();
	}
	const std::string reservationKey(keyPath.generic_string());
	plan.storageReservationKey = reservationKey.c_str();

	std::error_code spaceError;
	const std::filesystem::space_info spaceInfo(std::filesystem::space(
		std::filesystem::path(root.c_str()), spaceError));
	if (spaceError) {
		plan.storagePreflightSucceeded = false;
		plan.storagePreflightDecision = failPolicy ? _T("query_error_fail") : _T("query_error_instrumentation_disabled");
		plan.storagePreflightReason = String::FormatString(
			_T("std::filesystem::space failed: %s"), spaceError.message().c_str());
		plan.resourceDecision = failPolicy ? _T("storage_preflight_error") : _T("disabled_storage_query_error");
		plan.exactUnavailableReason = plan.storagePreflightReason;
		plan.summaryAvailable = false;
		plan.traceAvailable = false;
		plan.mapsAvailable = false;
		plan.exactAvailable = false;
		plan.prefilterAvailable = false;
		SetAlgorithmInstrumentAvailability(plan, false, false);
		if (plan.traceRequested)
			plan.traceUnavailableReason = plan.storagePreflightReason;
		if (plan.prefilterRequested)
			plan.prefilterUnavailableReason = plan.storagePreflightReason;
		plan.estimatedDeviceBytes = 0;
		plan.estimatedHostBytes = 0;
		plan.estimatedStorageBytes = 0;
		plan.frameStorageBudgetBytes = plan.frameStorageCommittedBeforeBytes;
		return;
	}

	plan.storagePreflightSucceeded = true;
	plan.storageAvailableBytes = (uint64_t)spaceInfo.available;
	const auto existing(g_instrumentStorageReservations.find(reservationKey));
	const uint64_t totalReservedBefore(
		existing == g_instrumentStorageReservations.end() ? 0 : existing->second);
	const uint64_t ownedPriorityReservedBefore(
		framePriorityReservation.active && framePriorityReservation.key == plan.storageReservationKey ?
			framePriorityReservation.bytes : 0u);
	ASSERT(totalReservedBefore >= ownedPriorityReservedBefore);
	plan.storagePriorityReservationBytes = ownedPriorityReservedBefore;
	plan.storageReservedBeforeBytes = totalReservedBefore - ownedPriorityReservedBefore;
	plan.storageEffectiveAvailableBytes = plan.storageAvailableBytes > plan.storageReservedBeforeBytes ?
		plan.storageAvailableBytes - plan.storageReservedBeforeBytes : 0;

	auto releaseLocked = [&](InstrumentStorageReservation& held) {
		if (!held.active)
			return;
		const auto it(g_instrumentStorageReservations.find(held.key.c_str()));
		ASSERT(it != g_instrumentStorageReservations.end());
		ASSERT(it->second >= held.bytes);
		if (it->second == held.bytes)
			g_instrumentStorageReservations.erase(it);
		else
			it->second -= held.bytes;
		held.active = false;
		held.bytes = 0;
		held.key.clear();
	};
	auto reserve = [&](uint64_t bytes) {
		if (consumeFramePriorityReservation && framePriorityReservation.active) {
			plan.storagePriorityReservationConsumed = true;
			releaseLocked(framePriorityReservation);
		}
		if (!consumeFramePriorityReservation &&
			plan.frameStoragePriorityReserveBytes > 0 &&
			!framePriorityReservation.active)
		{
			g_instrumentStorageReservations[reservationKey] = SaturatingAdd(
				g_instrumentStorageReservations[reservationKey],
				plan.frameStoragePriorityReserveBytes);
			framePriorityReservation.key = plan.storageReservationKey;
			framePriorityReservation.bytes = plan.frameStoragePriorityReserveBytes;
			framePriorityReservation.active = true;
			plan.storagePriorityReservationBytes = plan.frameStoragePriorityReserveBytes;
		}
		g_instrumentStorageReservations[reservationKey] = SaturatingAdd(
			g_instrumentStorageReservations[reservationKey], bytes);
		plan.storageReservationBytes = bytes;
		reservation.key = plan.storageReservationKey;
		reservation.bytes = bytes;
		reservation.active = true;
	};
	const uint64_t requestedWithPriorityReserve(SaturatingAdd(
		plan.storageRequestedBytes, plan.frameStoragePriorityReserveBytes));
	if (plan.storageEffectiveAvailableBytes >= requestedWithPriorityReserve) {
		reserve(plan.storageRequestedBytes);
		plan.storagePreflightDecision = plan.exactAvailable ? _T("reserved_exact_maps") :
			(plan.mapsAvailable ? _T("reserved_legacy_maps") :
				(plan.prefilterAvailable ? _T("reserved_prefilter") :
					(plan.traceAvailable ? _T("reserved_summary_and_targeted_traces") : _T("reserved_summary"))));
		plan.storagePreflightReason = _T("requested capture fits concurrent-aware available storage");
		return;
	}

	const String insufficientReason(String::FormatString(
		_T("requested capture needs %llu bytes plus %llu bytes reserved for higher-priority full-resolution evidence, but only %llu bytes remain after %llu in-process reserved bytes"),
		(unsigned long long)plan.storageRequestedBytes,
		(unsigned long long)plan.frameStoragePriorityReserveBytes,
		(unsigned long long)plan.storageEffectiveAvailableBytes,
		(unsigned long long)plan.storageReservedBeforeBytes));
	struct StorageCandidate {
		bool maps;
		bool exact;
		bool trace;
		bool prefilter;
		const char* resourceDecision;
		const char* preflightDecision;
		uint64_t device;
		uint64_t host;
		uint64_t storage;
	};
	std::vector<StorageCandidate> candidates;
	auto appendCandidate = [&](bool maps, bool exact, bool trace, bool prefilter,
		const char* resourceDecision, const char* preflightDecision,
		uint64_t mapDevice, uint64_t mapHost, uint64_t mapStorage) {
		candidates.push_back({maps, exact, trace, prefilter, resourceDecision, preflightDecision,
			SaturatingAdd(SaturatingAdd(plan.summaryDeviceBytes,
				trace ? plan.traceDeviceBytes : 0u), mapDevice),
			SaturatingAdd(SaturatingAdd(plan.summaryHostBytes,
				trace ? plan.traceHostBytes : 0u), mapHost),
			SaturatingAdd(SaturatingAdd(plan.summaryStorageBytes,
				trace ? plan.traceStorageBytes : 0u), mapStorage)});
	};
	const bool withTrace(plan.traceRequested);
	if (plan.mapsRequested && plan.exactRequested) {
		const uint64_t device(SaturatingAdd(plan.legacyMapDeviceBytes, plan.exactDeviceBytes));
		const uint64_t host(SaturatingAdd(plan.legacyMapHostBytes, plan.exactHostBytes));
		const uint64_t storage(SaturatingAdd(plan.legacyMapStorageBytes, plan.exactStorageBytes));
		if (withTrace)
			appendCandidate(true, true, true, false,
				"exact_maps_storage_preflight_degraded", "reserved_exact_maps_with_trace_after_degradation",
				device, host, storage);
		appendCandidate(true, true, false, false,
			"exact_maps_trace_storage_preflight_degraded", "reserved_exact_maps_without_trace_after_degradation",
			device, host, storage);
	}
	if (plan.mapsRequested) {
		if (withTrace)
			appendCandidate(true, false, true, false,
				"legacy_maps_storage_preflight_degraded", "reserved_legacy_maps_with_trace_after_degradation",
				plan.legacyMapDeviceBytes, plan.legacyMapHostBytes, plan.legacyMapStorageBytes);
		appendCandidate(true, false, false, false,
			"legacy_maps_trace_storage_preflight_degraded", "reserved_legacy_maps_without_trace_after_degradation",
			plan.legacyMapDeviceBytes, plan.legacyMapHostBytes, plan.legacyMapStorageBytes);
	}
	if (plan.prefilterRequested) {
		if (withTrace)
			appendCandidate(false, false, true, true,
				"prefilter_storage_preflight_degraded", "reserved_prefilter_with_trace_after_degradation",
				plan.prefilterDeviceBytes, plan.prefilterHostBytes, plan.prefilterStorageBytes);
		appendCandidate(false, false, false, true,
			"prefilter_trace_storage_preflight_degraded", "reserved_prefilter_without_trace_after_degradation",
			plan.prefilterDeviceBytes, plan.prefilterHostBytes, plan.prefilterStorageBytes);
	}
	if (withTrace)
		appendCandidate(false, false, true, false,
			"summary_storage_preflight_degraded", "reserved_summary_with_trace_after_degradation",
			0u, 0u, 0u);
	appendCandidate(false, false, false, false,
		"summary_trace_storage_preflight_degraded", "reserved_summary_without_trace_after_degradation",
		0u, 0u, 0u);
	if (!failPolicy) {
		for (const StorageCandidate& candidate : candidates) {
			const uint64_t cumulativeStorage(SaturatingAdd(
				SaturatingAdd(plan.frameStorageCommittedBeforeBytes, candidate.storage),
				plan.frameStoragePriorityReserveBytes));
			const uint64_t physicalStorage(SaturatingAdd(
				candidate.storage, plan.frameStoragePriorityReserveBytes));
			if (!InstrumentResourceFits(candidate.device, OPTDENSE::nDMapInstrumentationMaxDeviceMB) ||
				!InstrumentResourceFits(candidate.host, OPTDENSE::nDMapInstrumentationMaxHostMB) ||
				!InstrumentResourceFits(cumulativeStorage, OPTDENSE::nDMapInstrumentationMaxFrameStorageMB) ||
				plan.storageEffectiveAvailableBytes < physicalStorage)
				continue;
			plan.summaryAvailable = true;
			plan.mapsAvailable = candidate.maps;
			plan.exactAvailable = candidate.exact;
			plan.traceAvailable = candidate.trace;
			plan.prefilterAvailable = candidate.prefilter;
			SetAlgorithmInstrumentAvailability(plan, true, candidate.maps);
			plan.resourceDecision = candidate.resourceDecision;
			if (plan.exactRequested && !candidate.exact)
				plan.exactUnavailableReason = insufficientReason;
			if (plan.traceRequested && !candidate.trace)
				plan.traceUnavailableReason = insufficientReason;
			if (plan.prefilterRequested && !candidate.prefilter)
				plan.prefilterUnavailableReason = insufficientReason;
			plan.estimatedDeviceBytes = candidate.device;
			plan.estimatedHostBytes = candidate.host;
			plan.estimatedStorageBytes = candidate.storage;
			plan.frameStorageBudgetBytes = SaturatingAdd(
				plan.frameStorageCommittedBeforeBytes, candidate.storage);
			reserve(candidate.storage);
			plan.storagePreflightDecision = candidate.preflightDecision;
			plan.storagePreflightReason = insufficientReason;
			return;
		}
	}
	plan.summaryAvailable = false;
	plan.traceAvailable = false;
	plan.mapsAvailable = false;
	plan.exactAvailable = false;
	plan.prefilterAvailable = false;
	SetAlgorithmInstrumentAvailability(plan, false, false);
	plan.resourceDecision = failPolicy ? _T("storage_preflight_error") : _T("disabled_storage_preflight_degraded");
	if (plan.exactRequested)
		plan.exactUnavailableReason = insufficientReason;
	if (plan.traceRequested)
		plan.traceUnavailableReason = insufficientReason;
	if (plan.prefilterRequested)
		plan.prefilterUnavailableReason = insufficientReason;
	plan.estimatedDeviceBytes = 0;
	plan.estimatedHostBytes = 0;
	plan.estimatedStorageBytes = 0;
	plan.frameStorageBudgetBytes = plan.frameStorageCommittedBeforeBytes;
	plan.storagePreflightDecision = failPolicy ? _T("insufficient_storage_fail") : _T("insufficient_summary_storage_disabled");
	plan.storagePreflightReason = insufficientReason;
}

String DMapSafeName(const String& imageName)
{
	String safe(Util::getFileName(imageName));
	if (safe.empty())
		safe = Util::getFileNameExt(imageName);
	for (char& ch : safe) {
		if (!std::isalnum((unsigned char)ch) && ch != '-' && ch != '_')
			ch = '_';
	}
	if (safe.empty())
		safe = _T("image");
	if (safe.size() > 80)
		safe.resize(80);
	return safe;
}

String DMapDepthMapDir(const String& root, int imageID, const String& imageName)
{
	String dir(root + _T("depthmaps/"));
	Util::ensureFolder(dir);
	dir += String::FormatString(_T("%04d_%s/"), imageID, DMapSafeName(imageName).c_str());
	Util::ensureFolder(dir);
	return dir;
}

String CsvEscape(const String& value)
{
	bool quote(false);
	String out;
	for (const char ch : value) {
		if (ch == '"' || ch == ',' || ch == '\n' || ch == '\r')
			quote = true;
		if (ch == '"')
			out += _T("\"\"");
		else
			out += ch;
	}
	if (!quote)
		return out;
	return _T("\"") + out + _T("\"");
}

bool AppendDMapInstrumentationSelection(
	const String& root,
	int imageID,
	const String& imageName,
	bool geometricConsistency,
	int geometricIteration)
{
	const bool listed(InstrumentImageListed(imageID, imageName));
	const float sampleValue(InstrumentImageSampleValue(imageID));
	const bool sampled(OPTDENSE::fDMapInstrumentationSampleRate >= 1.f ||
		(OPTDENSE::fDMapInstrumentationSampleRate > 0.f && sampleValue < OPTDENSE::fDMapInstrumentationSampleRate));
	const bool selected(listed && sampled);
	const char* reason(selected ? "selected" : !listed ? "not_in_image_list" : "sampled_out");
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String fileName(root + _T("frame_selection.csv"));
	const bool writeHeader(!File::access(fileName));
	std::ofstream fs(fileName.c_str(), std::ios::app);
	if (!fs)
		return false;
	if (writeHeader)
		fs << "image_id,image_name,geometric_consistency,selected,listed,sampled,sample_value,sample_rate,sample_seed,reason,estimation_stage,geometric_iteration\n";
	fs << imageID << ',' << CsvEscape(imageName) << ',' << (geometricConsistency ? 1 : 0) << ','
	   << (selected ? 1 : 0) << ',' << (listed ? 1 : 0) << ',' << (sampled ? 1 : 0) << ',' << sampleValue << ','
	   << OPTDENSE::fDMapInstrumentationSampleRate << ',' << OPTDENSE::nDMapInstrumentationSampleSeed << ','
	   << reason << ',' << DMapEstimationStageName(geometricIteration) << ',';
	if (geometricIteration >= 0)
		fs << geometricIteration;
	fs << '\n';
	fs.flush();
	return (bool)fs;
}

bool AppendDMapInstrumentationResourcePlan(
	const String& root,
	int imageID,
	const String& imageName,
	bool geometricConsistency,
	int geometricIteration,
	int pyramidLevel,
	int width,
	int height,
	const InstrumentExtendedMaps& plan)
{
	nlohmann::json record = {
		{"schema_name", "openmvs.dmap.resource_plan"},
		{"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION},
		{"image_id", imageID},
		{"image_name", imageName.c_str()},
		{"geometric_consistency", geometricConsistency},
		{"estimation_stage", DMapEstimationStageName(geometricIteration)},
		{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
		{"pyramid_level", pyramidLevel},
		{"width", width},
		{"height", height},
		{"num_logical_states", plan.numLogicalStates},
		{"num_views", plan.numViews},
		{"num_trace_pixels", plan.numTracePixels},
		{"trace_requested", plan.traceRequested},
		{"trace_available", plan.traceAvailable},
		{"trace_unavailable_reason", plan.traceUnavailableReason.c_str()},
		{"compatibility_maps_requested", plan.compatibilityMapsRequested},
		{"compatibility_map_contract", {
			{"update_source_map_expected", plan.compatibilityMapsRequested},
			{"cost_map_expected", false},
			{"cost_map_unavailable_reason", plan.compatibilityMapsRequested ?
				"production confidence maps are retained at pyramid level 0 only" : "not_requested"}
		}},
		{"maps_requested", plan.mapsRequested},
		{"maps_available", plan.mapsAvailable},
		{"exact_requested", plan.exactRequested},
		{"exact_available", plan.exactAvailable},
		{"prefilter_requested", plan.prefilterRequested},
		{"prefilter_available", plan.prefilterAvailable},
		{"prefilter_unavailable_reason", plan.prefilterUnavailableReason.c_str()},
		{"apd_requested", plan.apdRequested},
		{"apd_summary_available", plan.apdAvailable},
		{"apd_maps_available", plan.apdMapsAvailable},
		{"dvp_requested", plan.dvpRequested},
		{"dvp_summary_available", plan.dvpAvailable},
		{"dvp_maps_available", plan.dvpMapsAvailable},
		{"dvp_visibility_requested", plan.visibilityRequested},
		{"dvp_visibility_stage_active", plan.apdStageActive},
		{"dvp_visibility_summary_available", plan.visibilityAvailable && plan.apdStageActive},
		{"dvp_visibility_maps_available", plan.visibilityMapsAvailable && plan.apdStageActive},
		{"dvp_visible_normal_requested", plan.visibleNormalRequested},
		{"dvp_visible_normal_stage_active", plan.apdStageActive},
		{"dvp_visible_normal_summary_available", plan.visibleNormalAvailable && plan.apdStageActive},
		{"dvp_visible_normal_maps_available", plan.visibleNormalMapsAvailable && plan.apdStageActive},
		{"summary_available", plan.summaryAvailable},
		{"decision", plan.resourceDecision.c_str()},
		{"exact_unavailable_reason", plan.exactUnavailableReason.c_str()},
		{"limits_mib", {
			{"device", OPTDENSE::nDMapInstrumentationMaxDeviceMB},
			{"host", OPTDENSE::nDMapInstrumentationMaxHostMB},
			{"frame_storage", OPTDENSE::nDMapInstrumentationMaxFrameStorageMB}
		}},
		{"effective_estimate_bytes", {
			{"device", plan.estimatedDeviceBytes},
			{"host", plan.estimatedHostBytes},
			{"frame_storage", plan.frameStorageBudgetBytes},
			{"current_pyramid_storage", plan.estimatedStorageBytes},
			{"frame_storage_committed_before", plan.frameStorageCommittedBeforeBytes},
			{"full_resolution_priority_reserve", plan.frameStoragePriorityReserveBytes}
		}},
		{"component_estimate_bytes", {
			{"additive_within_pyramid_level", true},
			{"summary_device", plan.summaryDeviceBytes},
			{"trace_device", plan.traceDeviceBytes},
			{"legacy_maps_device", plan.legacyMapDeviceBytes},
			{"exact_device", plan.exactDeviceBytes},
			{"prefilter_device", plan.prefilterDeviceBytes},
			{"summary_host", plan.summaryHostBytes},
			{"trace_host", plan.traceHostBytes},
			{"legacy_maps_host", plan.legacyMapHostBytes},
			{"exact_host", plan.exactHostBytes},
			{"prefilter_host", plan.prefilterHostBytes},
			{"summary_storage", plan.summaryStorageBytes},
			{"trace_storage", plan.traceStorageBytes},
			{"legacy_maps_storage", plan.legacyMapStorageBytes},
			{"exact_storage", plan.exactStorageBytes},
			{"prefilter_storage", plan.prefilterStorageBytes},
			{"apd_summary_device", plan.apdSummaryDeviceBytes},
			{"apd_summary_host", plan.apdSummaryHostBytes},
			{"apd_trace_device", plan.apdTraceDeviceBytes},
			{"apd_trace_host", plan.apdTraceHostBytes},
			{"apd_trace_storage", plan.apdTraceStorageBytes},
			{"apd_maps_device", plan.apdMapDeviceBytes},
			{"apd_maps_host", plan.apdMapHostBytes},
			{"apd_maps_storage", plan.apdMapStorageBytes},
			{"dvp_summary_device", plan.dvpSummaryDeviceBytes},
			{"dvp_summary_host", plan.dvpSummaryHostBytes},
			{"dvp_trace_device", plan.dvpTraceDeviceBytes},
			{"dvp_trace_host", plan.dvpTraceHostBytes},
			{"dvp_trace_storage", plan.dvpTraceStorageBytes},
			{"dvp_maps_device", plan.dvpMapDeviceBytes},
			{"dvp_maps_host", plan.dvpMapHostBytes},
			{"dvp_maps_storage", plan.dvpMapStorageBytes},
			{"dvp_visibility_summary_device", plan.visibilitySummaryDeviceBytes},
			{"dvp_visibility_summary_host", plan.visibilitySummaryHostBytes},
			{"dvp_visibility_trace_device", plan.visibilityTraceDeviceBytes},
			{"dvp_visibility_trace_host", plan.visibilityTraceHostBytes},
			{"dvp_visibility_trace_storage", plan.visibilityTraceStorageBytes},
			{"dvp_visibility_maps_device", plan.visibilityMapDeviceBytes},
			{"dvp_visibility_maps_host", plan.visibilityMapHostBytes},
			{"dvp_visibility_maps_storage", plan.visibilityMapStorageBytes},
			{"dvp_visible_normal_summary_device", plan.visibleNormalSummaryDeviceBytes},
			{"dvp_visible_normal_summary_host", plan.visibleNormalSummaryHostBytes},
			{"dvp_visible_normal_trace_device", plan.visibleNormalTraceDeviceBytes},
			{"dvp_visible_normal_trace_host", plan.visibleNormalTraceHostBytes},
			{"dvp_visible_normal_trace_storage", plan.visibleNormalTraceStorageBytes},
			{"dvp_visible_normal_maps_device", plan.visibleNormalMapDeviceBytes},
			{"dvp_visible_normal_maps_host", plan.visibleNormalMapHostBytes},
			{"dvp_visible_normal_maps_storage", plan.visibleNormalMapStorageBytes}
		}},
		{"peak_model", {
			{"fixed_host_bytes", DMAP_INSTRUMENT_FIXED_HOST_BYTES},
			{"summary_scratch_bytes_per_pixel", DMAP_SUMMARY_HOST_SCRATCH_BYTES_PER_PIXEL},
			{"map_export_scratch_bytes_per_pixel", DMAP_MAP_EXPORT_HOST_SCRATCH_BYTES_PER_PIXEL},
			{"legacy_terminal_bytes_per_pixel", DMAP_LEGACY_TERMINAL_BYTES_PER_PIXEL}
		}},
		{"trace_limits", {
			{"pixels_per_pyramid_level", PM_INSTRUMENT_MAX_TRACE_PIXELS_PER_LEVEL},
			{"label_bytes_per_pyramid_level", PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES_PER_LEVEL},
			{"label_bytes_per_entry", PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES},
			{"config_pixels", PM_INSTRUMENT_MAX_TRACE_CONFIG_PIXELS},
			{"config_label_bytes", PM_INSTRUMENT_MAX_TRACE_CONFIG_LABEL_BYTES}
		}},
		{"exact_record_bytes", {
			{"pixel", sizeof(PatchMatchInstrumentExactPixel)},
			{"view", sizeof(PatchMatchInstrumentExactView)},
			{"trace", sizeof(PatchMatchInstrumentTraceRecord)},
			{"apd_state", sizeof(PatchMatchAPDInstrumentState)},
			{"apd_update", sizeof(PatchMatchAPDInstrumentUpdate)},
			{"apd_trace", sizeof(PatchMatchAPDInstrumentTrace)},
			{"dvp_counter", sizeof(PatchMatchDVPInstrumentCounters)},
			{"dvp_update", sizeof(PatchMatchDVPInstrumentUpdate)},
			{"dvp_trace", sizeof(PatchMatchDVPInstrumentTrace)},
			{"dvp_visibility_counter", sizeof(PatchMatchDVPVisibilityInstrumentCounters)},
			{"dvp_visibility_update", sizeof(PatchMatchDVPVisibilityInstrumentUpdate)},
			{"dvp_visibility_trace", sizeof(PatchMatchDVPVisibilityInstrumentTrace)},
			{"dvp_visible_normal_counter", sizeof(PatchMatchDVPVisibleNormalInstrumentCounters)},
			{"dvp_visible_normal_update", sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)},
			{"dvp_visible_normal_trace", sizeof(PatchMatchDVPVisibleNormalInstrumentTrace)}
		}},
		{"storage_preflight", {
			{"attempted", plan.storagePreflightAttempted},
			{"succeeded", plan.storagePreflightSucceeded},
			{"available_bytes", plan.storageAvailableBytes},
			{"reserved_before_bytes", plan.storageReservedBeforeBytes},
			{"effective_available_bytes", plan.storageEffectiveAvailableBytes},
			{"requested_bytes", plan.storageRequestedBytes},
			{"requested_plus_priority_reserve_bytes", SaturatingAdd(plan.storageRequestedBytes, plan.frameStoragePriorityReserveBytes)},
			{"reservation_bytes", plan.storageReservationBytes},
			{"frame_priority_reservation_bytes", plan.storagePriorityReservationBytes},
			{"frame_priority_reservation_consumed", plan.storagePriorityReservationConsumed},
			{"reservation_key", plan.storageReservationKey.c_str()},
			{"decision", plan.storagePreflightDecision.c_str()},
			{"reason", plan.storagePreflightReason.c_str()}
		}}
	};
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	std::ofstream fs((root + _T("resource_plans.jsonl")).c_str(), std::ios::app);
	if (!fs)
		return false;
	fs << record.dump() << '\n';
	fs.flush();
	return (bool)fs;
}

bool WriteJsonFile(const String& fileName, const nlohmann::json& data)
{
	const String temporary(fileName + _T(".tmp"));
	bool writeSucceeded(false);
	{
		std::ofstream fs(temporary.c_str(), std::ios::trunc);
		if (!fs)
			return false;
		fs << data.dump(2) << '\n';
		fs.flush();
		writeSucceeded = (bool)fs;
	}
	if (!writeSucceeded) {
		File::deleteFile(temporary);
		return false;
	}
	if (!File::renameFile(temporary, fileName)) {
		File::deleteFile(temporary);
		return false;
	}
	return true;
}

bool RemoveDMapCompletionMarkers(const String& depthMapDir)
{
	File::deleteFile(depthMapDir + _T("capture_complete.json"));
	File::deleteFile(depthMapDir + _T("capture_complete.json.tmp"));
	File::deleteFile(depthMapDir + _T("summary_complete.json"));
	File::deleteFile(depthMapDir + _T("summary_complete.json.tmp"));
	File::deleteFile(depthMapDir + _T("prefilter_capture_complete.json"));
	File::deleteFile(depthMapDir + _T("prefilter_capture_complete.json.tmp"));
	return !File::access(depthMapDir + _T("capture_complete.json")) &&
		!File::access(depthMapDir + _T("capture_complete.json.tmp")) &&
		!File::access(depthMapDir + _T("summary_complete.json")) &&
		!File::access(depthMapDir + _T("summary_complete.json.tmp")) &&
		!File::access(depthMapDir + _T("prefilter_capture_complete.json")) &&
		!File::access(depthMapDir + _T("prefilter_capture_complete.json.tmp"));
}

InstrumentTraceSelectionResult SelectInstrumentTracePixels(
	int imageID,
	float scale,
	const cv::Size& size)
{
	EnsureInstrumentConfigLoaded();
	InstrumentTraceSelectionResult selected;
	if (OPTDENSE::nPatchMatchInstrumentLevel < 2 || size.area() <= 0)
		return selected;
	std::unordered_set<int> selectedIndices;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	for (size_t configIndex = 0; configIndex < g_instrumentConfig.tracePixels.size(); ++configIndex) {
		const InstrumentTracePixel& pixel(g_instrumentConfig.tracePixels[configIndex]);
		if (pixel.imageID != imageID)
			continue;
		const int x(ROUND2INT((float)pixel.x * scale));
		const int y(ROUND2INT((float)pixel.y * scale));
		if (x < 0 || y < 0 || x >= size.width || y >= size.height)
			continue;
		const int idx(y * size.width + x);
		if (!selectedIndices.insert(idx).second)
			continue;
		if (selected.pixels.size() >= PM_INSTRUMENT_MAX_TRACE_PIXELS_PER_LEVEL) {
			VERBOSE("error: CUDA PatchMatch trace selection for image %d exceeds %llu pixels at one pyramid level",
				imageID, (unsigned long long)PM_INSTRUMENT_MAX_TRACE_PIXELS_PER_LEVEL);
			std::exit(EXIT_FAILURE);
		}
		selected.labelBytes = SaturatingAdd(selected.labelBytes, pixel.label.size());
		if (selected.labelBytes > PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES_PER_LEVEL) {
			VERBOSE("error: CUDA PatchMatch trace labels for image %d exceed %llu bytes at one pyramid level",
				imageID, (unsigned long long)PM_INSTRUMENT_MAX_TRACE_LABEL_BYTES_PER_LEVEL);
			std::exit(EXIT_FAILURE);
		}
		selected.pixels.push_back({configIndex, x, y});
	}
	return selected;
}

std::vector<InstrumentTracePixel> MaterializeInstrumentTracePixels(
	int imageID,
	const InstrumentTraceSelectionResult& selected)
{
	std::vector<InstrumentTracePixel> pixels;
	pixels.reserve(selected.pixels.size());
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	for (const InstrumentTraceSelection& selection : selected.pixels) {
		ASSERT(selection.configIndex < g_instrumentConfig.tracePixels.size());
		InstrumentTracePixel pixel(g_instrumentConfig.tracePixels[selection.configIndex]);
		ASSERT(pixel.imageID == imageID);
		pixel.x = selection.x;
		pixel.y = selection.y;
		pixels.emplace_back(std::move(pixel));
	}
	return pixels;
}

std::vector<int32_t> BuildInstrumentTraceMap(
	const cv::Size& size,
	const std::vector<InstrumentTracePixel>& selected)
{
	if (selected.empty())
		return {};
	std::vector<int32_t> traceMap((size_t)size.area(), -1);
	for (size_t traceIndex = 0; traceIndex < selected.size(); ++traceIndex) {
		const InstrumentTracePixel& pixel(selected[traceIndex]);
		ASSERT(pixel.x >= 0 && pixel.y >= 0 && pixel.x < size.width && pixel.y < size.height);
		traceMap[(size_t)pixel.y * (size_t)size.width + (size_t)pixel.x] = (int32_t)traceIndex;
	}
	return traceMap;
}

PatchMatchInstrumentCounters SumInstrumentCounters(
	const PatchMatchInstrumentCounters& first,
	const PatchMatchInstrumentCounters& second)
{
	PatchMatchInstrumentCounters sum;
#define SUM_INSTRUMENT_FIELD(FIELD) sum.FIELD = first.FIELD + second.FIELD
	SUM_INSTRUMENT_FIELD(processed);
	SUM_INSTRUMENT_FIELD(validDepth);
	SUM_INSTRUMENT_FIELD(invalidDepth);
	SUM_INSTRUMENT_FIELD(badCost);
	SUM_INSTRUMENT_FIELD(lowResPrior);
	SUM_INSTRUMENT_FIELD(lowTexture);
	SUM_INSTRUMENT_FIELD(propagationWins);
	SUM_INSTRUMENT_FIELD(refinementWins);
	SUM_INSTRUMENT_FIELD(accepted);
	SUM_INSTRUMENT_FIELD(componentSamples);
	SUM_INSTRUMENT_FIELD(depthPriorSamples);
	SUM_INSTRUMENT_FIELD(viewChurn);
	SUM_INSTRUMENT_FIELD(viewAddedSum);
	SUM_INSTRUMENT_FIELD(viewRemovedSum);
	SUM_INSTRUMENT_FIELD(updateMagnitudeSamples);
	SUM_INSTRUMENT_FIELD(costBeforeSum);
	SUM_INSTRUMENT_FIELD(costSum);
	SUM_INSTRUMENT_FIELD(costSqSum);
	SUM_INSTRUMENT_FIELD(costImprovementSum);
	SUM_INSTRUMENT_FIELD(photometricCostSum);
	SUM_INSTRUMENT_FIELD(photoPriorCostSum);
	SUM_INSTRUMENT_FIELD(depthPriorCostSum);
	SUM_INSTRUMENT_FIELD(depthPriorWeightSum);
	SUM_INSTRUMENT_FIELD(geometricCostSum);
	SUM_INSTRUMENT_FIELD(viewEntropySum);
	SUM_INSTRUMENT_FIELD(depthAbsChangeSum);
	SUM_INSTRUMENT_FIELD(depthRelChangeSum);
	SUM_INSTRUMENT_FIELD(normalAngleSum);
#undef SUM_INSTRUMENT_FIELD
	for (int type = 0; type < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++type) {
		sum.candidateTested[type] = first.candidateTested[type] + second.candidateTested[type];
		sum.candidateFinite[type] = first.candidateFinite[type] + second.candidateFinite[type];
		sum.candidateAccepted[type] = first.candidateAccepted[type] + second.candidateAccepted[type];
	}
	for (int bin = 0; bin <= PM_INSTRUMENT_MAX_VIEWS; ++bin)
		sum.selectedViewBins[bin] = first.selectedViewBins[bin] + second.selectedViewBins[bin];
	for (int source = 0; source < PM_INSTRUMENT_NUM_SOURCES; ++source)
		sum.updateSource[source] = first.updateSource[source] + second.updateSource[source];
	for (int reason = 0; reason < PM_INSTRUMENT_NUM_BAD_REASONS; ++reason)
		sum.badReason[reason] = first.badReason[reason] + second.badReason[reason];
	for (int bin = 0; bin < PM_INSTRUMENT_NUM_UPDATE_BINS; ++bin) {
		sum.depthRelChangeBins[bin] = first.depthRelChangeBins[bin] + second.depthRelChangeBins[bin];
		sum.normalAngleBins[bin] = first.normalAngleBins[bin] + second.normalAngleBins[bin];
	}
	for (int view = 0; view < PM_INSTRUMENT_MAX_VIEWS; ++view) {
		sum.viewWeightSum[view] = first.viewWeightSum[view] + second.viewWeightSum[view];
		sum.viewBadCost[view] = first.viewBadCost[view] + second.viewBadCost[view];
		sum.viewCostWeightedSum[view] = first.viewCostWeightedSum[view] + second.viewCostWeightedSum[view];
		sum.viewPhotometricCostWeightedSum[view] = first.viewPhotometricCostWeightedSum[view] + second.viewPhotometricCostWeightedSum[view];
		sum.viewGeometricCostWeightedSum[view] = first.viewGeometricCostWeightedSum[view] + second.viewGeometricCostWeightedSum[view];
	}
	return sum;
}

std::vector<PatchMatchInstrumentCounters> AggregateInstrumentIterations(
	const std::vector<PatchMatchInstrumentCounters>& passes,
	bool exactHotKernelCounters)
{
	if (passes.empty())
		return {};
	std::vector<PatchMatchInstrumentCounters> iterations;
	iterations.reserve(1 + (passes.size()-1)/2);
	iterations.push_back(passes.front());
	for (size_t firstPass = 1; firstPass+1 < passes.size(); firstPass += 2) {
		const PatchMatchInstrumentCounters& first(passes[firstPass]);
		const PatchMatchInstrumentCounters& second(passes[firstPass+1]);
		PatchMatchInstrumentCounters logical(exactHotKernelCounters ? SumInstrumentCounters(first, second) : second);
		if (exactHotKernelCounters) {
			iterations.emplace_back(logical);
			continue;
		}
		logical.propagationWins = first.propagationWins + second.propagationWins;
		logical.refinementWins = first.refinementWins + second.refinementWins;
		logical.accepted = first.accepted + second.accepted;
		logical.viewChurn = first.viewChurn + second.viewChurn;
		logical.viewAddedSum = first.viewAddedSum + second.viewAddedSum;
		logical.viewRemovedSum = first.viewRemovedSum + second.viewRemovedSum;
		logical.updateMagnitudeSamples = first.updateMagnitudeSamples + second.updateMagnitudeSamples;
		logical.costBeforeSum = first.costBeforeSum;
		logical.costImprovementSum = first.costImprovementSum + second.costImprovementSum;
		logical.depthAbsChangeSum = first.depthAbsChangeSum + second.depthAbsChangeSum;
		logical.depthRelChangeSum = first.depthRelChangeSum + second.depthRelChangeSum;
		logical.normalAngleSum = first.normalAngleSum + second.normalAngleSum;
		for (int type = 0; type < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++type) {
			logical.candidateTested[type] = first.candidateTested[type] + second.candidateTested[type];
			logical.candidateFinite[type] = first.candidateFinite[type] + second.candidateFinite[type];
			logical.candidateAccepted[type] = first.candidateAccepted[type] + second.candidateAccepted[type];
		}
		for (int source = 0; source < PM_INSTRUMENT_NUM_SOURCES; ++source)
			logical.updateSource[source] = first.updateSource[source] + second.updateSource[source];
		logical.updateSource[PM_SOURCE_NONE] = logical.processed > logical.accepted ? logical.processed-logical.accepted : 0;
		for (int bin = 0; bin < PM_INSTRUMENT_NUM_UPDATE_BINS; ++bin) {
			logical.depthRelChangeBins[bin] = first.depthRelChangeBins[bin] + second.depthRelChangeBins[bin];
			logical.normalAngleBins[bin] = first.normalAngleBins[bin] + second.normalAngleBins[bin];
		}
		iterations.emplace_back(logical);
	}
	return iterations;
}

bool AppendInstrumentCounters(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const std::vector<PatchMatchInstrumentCounters>& counters)
{
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String fileName(dir + _T("counters.csv"));
	const bool writeHeader(!File::access(fileName));
	std::ofstream fs(fileName.c_str(), std::ios::app);
	if (!fs)
		return false;
	if (writeHeader) {
		fs << "image_id,scale_number,width,height,pass_index,phase,iteration,processed,valid_depth,invalid_depth,bad_cost,low_res_prior,low_texture,propagation_wins,refinement_wins,accepted,component_samples,depth_prior_samples,view_churn,view_added_sum,view_removed_sum,update_magnitude_samples,cost_before_sum,cost_sum,cost_sq_sum,cost_improvement_sum,photometric_cost_sum,photo_prior_cost_sum,depth_prior_cost_sum,depth_prior_weight_sum,geometric_cost_sum,view_entropy_sum,depth_abs_change_sum,depth_rel_change_sum,normal_angle_sum";
		for (int i = 0; i <= PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",selected_views_" << i;
		for (int i = 0; i < PM_INSTRUMENT_NUM_SOURCES; ++i)
			fs << ",source_" << InstrumentSourceName(i);
		for (int i = 0; i < PM_INSTRUMENT_NUM_BAD_REASONS; ++i)
			fs << ",bad_reason_" << InstrumentBadReasonName(i);
		for (int i = 0; i < PM_INSTRUMENT_NUM_UPDATE_BINS; ++i)
			fs << ",depth_rel_change_bin_" << i;
		for (int i = 0; i < PM_INSTRUMENT_NUM_UPDATE_BINS; ++i)
			fs << ",normal_angle_bin_" << i;
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ",candidate_tested_" << InstrumentCandidateTypeName(i);
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ",candidate_finite_" << InstrumentCandidateTypeName(i);
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ",candidate_accepted_" << InstrumentCandidateTypeName(i);
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",view_weight_" << i;
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",view_bad_cost_" << i;
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",view_cost_weighted_sum_" << i;
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",view_photometric_cost_weighted_sum_" << i;
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ",view_geometric_cost_weighted_sum_" << i;
		fs << '\n';
	}
	for (size_t pass = 0; pass < counters.size(); ++pass) {
		const int iter(pass == 0 ? -1 : (int)pass-1);
		const PatchMatchInstrumentCounters& c = counters[pass];
		fs << imageID << ',' << scaleNumber << ',' << size.width << ',' << size.height << ','
		   << pass << ',' << (pass == 0 ? "initialization" : "iteration") << ',' << iter << ','
		   << c.processed << ',' << c.validDepth << ',' << c.invalidDepth << ','
		   << c.badCost << ',' << c.lowResPrior << ',' << c.lowTexture << ','
		   << c.propagationWins << ',' << c.refinementWins << ','
		   << c.accepted << ',' << c.componentSamples << ',' << c.depthPriorSamples << ','
		   << c.viewChurn << ',' << c.viewAddedSum << ',' << c.viewRemovedSum << ',' << c.updateMagnitudeSamples << ','
		   << c.costBeforeSum << ',' << c.costSum << ',' << c.costSqSum << ',' << c.costImprovementSum << ','
		   << c.photometricCostSum << ',' << c.photoPriorCostSum << ','
		   << c.depthPriorCostSum << ',' << c.depthPriorWeightSum << ',' << c.geometricCostSum << ','
		   << c.viewEntropySum << ',' << c.depthAbsChangeSum << ',' << c.depthRelChangeSum << ',' << c.normalAngleSum;
		for (int i = 0; i <= PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.selectedViewBins[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_SOURCES; ++i)
			fs << ',' << c.updateSource[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_BAD_REASONS; ++i)
			fs << ',' << c.badReason[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_UPDATE_BINS; ++i)
			fs << ',' << c.depthRelChangeBins[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_UPDATE_BINS; ++i)
			fs << ',' << c.normalAngleBins[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ',' << c.candidateTested[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ',' << c.candidateFinite[i];
		for (int i = 0; i < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++i)
			fs << ',' << c.candidateAccepted[i];
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.viewWeightSum[i];
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.viewBadCost[i];
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.viewCostWeightedSum[i];
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.viewPhotometricCostWeightedSum[i];
		for (int i = 0; i < PM_INSTRUMENT_MAX_VIEWS; ++i)
			fs << ',' << c.viewGeometricCostWeightedSum[i];
		fs << '\n';
	}
	fs.flush();
	return (bool)fs;
}

bool AppendInstrumentTraces(
	const String& dir,
	int imageID,
	int scaleNumber,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const std::vector<PatchMatchInstrumentTraceRecord>& records,
	int numPasses,
	bool exactHotKernelRecords)
{
	if (tracePixels.empty() && records.empty())
		return true;
	if (numPasses <= 0 || (numPasses & 1) == 0 || tracePixels.empty() ||
		records.size() != tracePixels.size() * (size_t)numPasses)
	{
		return false;
	}
	for (size_t traceIdx = 0; traceIdx < tracePixels.size(); ++traceIdx) {
		const size_t base(traceIdx * (size_t)numPasses);
		if (!records[base].valid)
			return false;
		for (int firstPass = 1; firstPass+1 < numPasses; firstPass += 2) {
			const bool firstValid(records[base + (size_t)firstPass].valid != 0);
			const bool secondValid(records[base + (size_t)firstPass + 1u].valid != 0);
			if (exactHotKernelRecords ? !(firstValid || secondValid) : !(firstValid && secondValid))
				return false;
		}
	}
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	std::ofstream fs((dir + _T("traces.jsonl")).c_str(), std::ios::app);
	if (!fs)
		return false;
	size_t writtenRows(0);
	auto writeRecord = [&](size_t traceIdx, const PatchMatchInstrumentTraceRecord& before, const PatchMatchInstrumentTraceRecord& after, int iteration, const char* stage, const std::vector<int>& rawPassIndices, bool singleRecord = false) {
		if (!before.valid || !after.valid)
			return false;
		const int source(after.source != PM_SOURCE_NONE ? after.source : before.source);
		const float depthAbsChange(
			iteration >= 0 && before.depthBefore > 0.f && after.depthAfter > 0.f ?
			ABS(after.depthAfter-before.depthBefore) : 0.f);
		const float depthRelChange(before.depthBefore > 0.f ? depthAbsChange / MAXF(before.depthBefore, FLT_EPSILON) : 0.f);
		nlohmann::json line;
		line["image_id"] = imageID;
		line["scale_number"] = scaleNumber;
		line["trace_index"] = traceIdx;
		line["label"] = tracePixels[traceIdx].label.c_str();
		line["x"] = after.x;
		line["y"] = after.y;
		line["pass_index"] = iteration < 0 ? 0 : iteration + 1;
		line["phase"] = stage;
		line["iteration"] = iteration;
		line["logical_iteration"] = iteration;
		line["stage"] = iteration < 0 ? "initialization" : std::string("iteration ") + std::to_string(iteration+1);
		line["measurement_basis"] = exactHotKernelRecords ? "exact_hot_kernel" : "post_pass_proxy";
		line["source_quality"] = exactHotKernelRecords ? "exact" : "proxy";
		line["raw_pass_indices"] = rawPassIndices;
		line["source"] = InstrumentSourceName(source);
		line["selected_view_count"] = after.selectedViewCount;
		line["selected_views_mask"] = after.selectedViews;
		line["selected_views_before_mask"] = before.selectedViewsBefore;
		line["depth_before"] = before.depthBefore;
		line["depth_after"] = after.depthAfter;
		line["cost_before"] = before.costBefore;
		line["cost_after"] = after.costAfter;
		line["cost_improvement"] = before.costBefore > after.costAfter ? before.costBefore-after.costAfter : 0.f;
		line["depth_abs_change"] = depthAbsChange;
		line["depth_rel_change"] = depthRelChange;
		line["normal_angle_change"] = singleRecord ? after.normalAngleChange : before.normalAngleChange + (iteration >= 0 ? after.normalAngleChange : 0.f);
		line["view_entropy"] = after.viewEntropy;
		line["photometric_cost_after"] = after.photometricCostAfter;
		line["photo_prior_cost_after"] = after.photoPriorCostAfter;
		line["depth_prior_cost_after"] = after.depthPriorCostAfter;
		line["depth_prior_weight_after"] = after.depthPriorWeightAfter;
		line["geometric_cost_after"] = after.geometricCostAfter;
		line["ref_variance"] = after.refVariance;
		line["low_depth"] = after.lowDepth;
		line["neighbor_costs"] = std::vector<float>(after.neighborCosts, after.neighborCosts + PM_INSTRUMENT_NUM_NEIGHBORS);
		line["view_costs"] = std::vector<float>(after.viewCosts, after.viewCosts + PM_INSTRUMENT_MAX_VIEWS);
		line["view_photometric_costs"] = std::vector<float>(after.viewPhotometricCosts, after.viewPhotometricCosts + PM_INSTRUMENT_MAX_VIEWS);
		line["view_geometric_costs"] = std::vector<float>(after.viewGeometricCosts, after.viewGeometricCosts + PM_INSTRUMENT_MAX_VIEWS);
		line["bad_reasons"] = std::vector<uint32_t>(after.badReason, after.badReason + PM_INSTRUMENT_NUM_BAD_REASONS);
		line["view_weights"] = std::vector<uint32_t>(after.viewWeights, after.viewWeights + PM_INSTRUMENT_MAX_VIEWS);
		fs << line.dump() << '\n';
		if (!fs)
			return false;
		++writtenRows;
		return true;
	};
	for (size_t traceIdx = 0; traceIdx < tracePixels.size(); ++traceIdx) {
		const size_t base(traceIdx * (size_t)numPasses);
		const PatchMatchInstrumentTraceRecord& init(records[base]);
		if (!writeRecord(traceIdx, init, init, -1, "initialization", {0}))
			return false;
		for (int iteration = 0; 2 + iteration * 2 < numPasses; ++iteration) {
			const int firstPass(1 + iteration * 2);
			const int secondPass(firstPass + 1);
			const PatchMatchInstrumentTraceRecord& first(records[base + (size_t)firstPass]);
			const PatchMatchInstrumentTraceRecord& second(records[base + (size_t)secondPass]);
			if (exactHotKernelRecords) {
				const PatchMatchInstrumentTraceRecord& record(second.valid ? second : first);
				if (!writeRecord(traceIdx, record, record, iteration, "iteration", {firstPass, secondPass}, true))
					return false;
			} else {
				if (!writeRecord(traceIdx, first, second, iteration, "iteration", {firstPass, secondPass}))
					return false;
			}
		}
	}
	fs.flush();
	const size_t expectedRows(tracePixels.size() * (size_t)(1 + (numPasses-1) / 2));
	return (bool)fs && writtenRows == expectedRows;
}

bool AppendAPDMultiscaleStage(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const nlohmann::json& multiscale)
{
	if (!multiscale.is_object() || multiscale.empty())
		return false;
	nlohmann::json record(multiscale);
	record["schema_name"] = "openmvs.dmap.apd_multiscale_stage_record";
	record["schema_version"] = 1;
	record["image_id"] = imageID;
	record["pyramid_level"] = scaleNumber;
	record["width"] = size.width;
	record["height"] = size.height;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	std::ofstream stream((dir + _T("apd_multiscale_stages.jsonl")).c_str(), std::ios::app);
	if (!stream)
		return false;
	stream << record.dump() << '\n';
	return (bool)stream;
}

const char* DVPDepthEdgeAnchorReasonName(DVPDepthEdgeAnchorReason reason)
{
	switch (reason) {
	case DVPDepthEdgeAnchorReason::SAME_REGION: return "same_region";
	case DVPDepthEdgeAnchorReason::MODE_DISABLED: return "mode_disabled";
	case DVPDepthEdgeAnchorReason::CENTER_BOUNDARY: return "center_boundary";
	case DVPDepthEdgeAnchorReason::ANCHOR_BOUNDARY: return "anchor_boundary";
	case DVPDepthEdgeAnchorReason::CROSS_REGION: return "cross_region";
	case DVPDepthEdgeAnchorReason::INVALID_ANCHOR_INDEX: return "invalid_anchor_index";
	default: return "unknown";
	}
}

bool SaveDVPDepthEdgeInstrumentation(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const PatchMatch::Params& params,
	const DVPDepthEdgePrior& prior,
	const Image16U& regions,
	const Image8U& anchorCountsBefore,
	const Image8U& anchorCountsAfter,
	const Image8U& rejectedAnchorCounts,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const std::vector<PatchMatchAPDInstrumentTrace>& apdTraces,
	int numIterations)
{
	if (!DVPDepthEdgeModeEnabled(params.nDVPDepthEdgeMode) || regions.empty() ||
		anchorCountsBefore.empty() || anchorCountsAfter.empty() ||
		rejectedAnchorCounts.empty() || regions.size() != size ||
		anchorCountsBefore.size() != size || anchorCountsAfter.size() != size ||
		rejectedAnchorCounts.size() != size)
	{
		return false;
	}
	const String prefix(String::FormatString(
		_T("maps/depth%04d_scale%02d_dvp_depth_edge_"), imageID, scaleNumber));
	const String regionsName(prefix + _T("regions.png"));
	const String beforeName(prefix + _T("anchor_count_before.png"));
	const String afterName(prefix + _T("anchor_count_after.png"));
	const String rejectedName(prefix + _T("anchor_count_rejected.png"));
	if (!regions.Save(dir+regionsName) || !anchorCountsBefore.Save(dir+beforeName) ||
		!anchorCountsAfter.Save(dir+afterName) || !rejectedAnchorCounts.Save(dir+rejectedName))
	{
		return false;
	}
	uint64_t boundaryPixels(0u), pixelsWithAnchorsBefore(0u), pixelsWithAnchorsAfter(0u);
	uint64_t rejectedAnchors(0u), anchorsBefore(0u), anchorsAfter(0u);
	uint64_t beforeHistogram[APD_MAX_ANCHORS+1u] = {};
	uint64_t afterHistogram[APD_MAX_ANCHORS+1u] = {};
	for (int y=0; y<size.height; ++y) {
		for (int x=0; x<size.width; ++x) {
			boundaryPixels += regions(y, x) == DVP_DEPTH_EDGE_BOUNDARY_LABEL;
			const unsigned before(min(static_cast<unsigned>(anchorCountsBefore(y, x)), APD_MAX_ANCHORS));
			const unsigned after(min(static_cast<unsigned>(anchorCountsAfter(y, x)), APD_MAX_ANCHORS));
			anchorsBefore += before;
			anchorsAfter += after;
			rejectedAnchors += rejectedAnchorCounts(y, x);
			pixelsWithAnchorsBefore += before > 0u;
			pixelsWithAnchorsAfter += after > 0u;
			++beforeHistogram[before];
			++afterHistogram[after];
		}
	}
	nlohmann::json beforeBins(nlohmann::json::array());
	nlohmann::json afterBins(nlohmann::json::array());
	for (unsigned count=0u; count<=APD_MAX_ANCHORS; ++count) {
		beforeBins.push_back(beforeHistogram[count]);
		afterBins.push_back(afterHistogram[count]);
	}
	nlohmann::json record = {
		{"schema_name", "openmvs.dmap.dvp_depth_edge_stage"},
		{"schema_version", 1},
		{"measurement_quality", "exact"},
		{"measurement_basis", "active_filter_kernel_final_anchor_rebuild"},
		{"image_id", imageID},
		{"scale_number", scaleNumber},
		{"width", size.width},
		{"height", size.height},
		{"mode", params.nDVPDepthEdgeMode},
		{"stage", DVPDepthEdgeStageName(params.nDVPDepthEdgeMode)},
		{"paper_contract_label", "paper_mechanics_complete_openmvs"},
		{"author_code_exact_equivalence", false},
		{"direct_monocular_depth_use", false},
			{"prior", {
				{"manifest", prior.manifestPath.c_str()},
				{"label_map", prior.labelMapPath.c_str()},
				{"label_map_sha256", prior.labelMapSHA256.c_str()},
				{"schema_version", prior.schemaVersion},
				{"raw_source_width", prior.rawSourceWidth},
				{"raw_source_height", prior.rawSourceHeight},
				{"processing_width", prior.processingWidth},
				{"processing_height", prior.processingHeight}
			}},
		{"maps", {
			{"regions", regionsName.c_str()},
			{"anchor_count_before", beforeName.c_str()},
			{"anchor_count_after", afterName.c_str()},
			{"anchor_count_rejected", rejectedName.c_str()}
		}},
		{"summary", {
			{"boundary_pixels", boundaryPixels},
			{"boundary_ratio", size.area() ? (double)boundaryPixels/(double)size.area() : 0.0},
			{"pixels_with_anchors_before", pixelsWithAnchorsBefore},
			{"pixels_with_anchors_after", pixelsWithAnchorsAfter},
			{"anchors_before", anchorsBefore},
			{"anchors_after", anchorsAfter},
			{"anchors_rejected", rejectedAnchors},
			{"anchor_retention_ratio", anchorsBefore ?
				nlohmann::json((double)anchorsAfter/(double)anchorsBefore) : nlohmann::json(nullptr)},
			{"anchor_count_before_histogram", beforeBins},
			{"anchor_count_after_histogram", afterBins}
		}},
		{"unavailable_signals", nlohmann::json::array({
			{{"name", "full_frame_rejection_reason_map"}, {"reason", "v1 stores exact aggregate rejected counts; reason attribution is selected-pixel only"}},
			{{"name", "per_iteration_full_frame_anchor_maps"}, {"reason", "v1 retains full-frame maps for the final anchor rebuild only"}},
			{{"name", "post_filter_fitted_plane_trace"}, {"reason", "v1 does not retain the active-only fitted-plane recomputation payload"}}
		})}
	};
	{
		std::lock_guard<std::mutex> lock(g_instrumentMutex);
		std::ofstream stream((dir + _T("dvp_depth_edge_stages.jsonl")).c_str(), std::ios::app);
		if (!stream)
			return false;
		stream << record.dump() << '\n';
		if (!stream)
			return false;
	}
	if (tracePixels.empty() && apdTraces.empty())
		return true;
	if (numIterations <= 0 || tracePixels.empty() ||
		apdTraces.size() != tracePixels.size()*static_cast<size_t>(numIterations))
	{
		return false;
	}
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	std::ofstream traces((dir + _T("dvp_depth_edge_traces.jsonl")).c_str(), std::ios::app);
	if (!traces)
		return false;
	for (int iteration=0; iteration<numIterations; ++iteration) {
		for (size_t traceIndex=0; traceIndex<tracePixels.size(); ++traceIndex) {
			const PatchMatchAPDInstrumentTrace& apd(
				apdTraces[static_cast<size_t>(iteration)*tracePixels.size()+traceIndex]);
			if (!apd.valid || apd.x < 0 || apd.y < 0 || apd.x >= size.width || apd.y >= size.height)
				return false;
			const uint16_t centerRegion(regions(apd.y, apd.x));
			nlohmann::json anchors(nlohmann::json::array());
			unsigned retained(0u);
			for (unsigned slot=0u; slot<apd.state.anchorCount && slot<APD_MAX_ANCHORS; ++slot) {
				const uint32_t anchorIndex(apd.anchors[slot]);
				DVPDepthEdgeAnchorDecision decision;
				if (anchorIndex >= static_cast<uint32_t>(size.area())) {
					decision.allowed = false;
					decision.centerRegion = centerRegion;
					decision.reason = DVPDepthEdgeAnchorReason::INVALID_ANCHOR_INDEX;
				} else {
					decision = ResolveDVPDepthEdgeAnchorDecision(params.nDVPDepthEdgeMode,
						centerRegion, regions(static_cast<int>(anchorIndex/(uint32_t)size.width),
							static_cast<int>(anchorIndex%(uint32_t)size.width)));
				}
				retained += decision.allowed;
				anchors.push_back({
					{"slot", slot},
					{"index", anchorIndex < static_cast<uint32_t>(size.area()) ?
						nlohmann::json(anchorIndex) : nlohmann::json(nullptr)},
					{"x", anchorIndex < static_cast<uint32_t>(size.area()) ?
						nlohmann::json(anchorIndex%(uint32_t)size.width) : nlohmann::json(nullptr)},
					{"y", anchorIndex < static_cast<uint32_t>(size.area()) ?
						nlohmann::json(anchorIndex/(uint32_t)size.width) : nlohmann::json(nullptr)},
					{"region", decision.anchorRegion},
					{"retained", decision.allowed},
					{"reason", DVPDepthEdgeAnchorReasonName(decision.reason)}
				});
			}
			traces << nlohmann::json({
				{"schema_name", "openmvs.dmap.dvp_depth_edge_trace"},
				{"schema_version", 1},
				{"measurement_quality", "exact"},
				{"measurement_basis", "exact_pre_filter_APD_anchor_identity_and_active_region_map"},
				{"image_id", imageID},
				{"scale_number", scaleNumber},
				{"logical_iteration", iteration},
				{"trace_index", traceIndex},
				{"label", tracePixels[traceIndex].label.c_str()},
				{"x", apd.x},
				{"y", apd.y},
				{"center_region", centerRegion},
				{"center_is_boundary", centerRegion == DVP_DEPTH_EDGE_BOUNDARY_LABEL},
				{"anchors_before", apd.state.anchorCount},
				{"anchors_after", retained},
				{"anchors", anchors}
			}).dump() << '\n';
		}
	}
	return (bool)traces;
}

bool AppendAPDInstrumentation(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	int numViews,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const InstrumentExtendedMaps& maps)
{
	if (!maps.apdRequested || !maps.apdStageActive)
		return true;
	const size_t numIterations(maps.apdCounters.size());
	if (!maps.apdAvailable || numIterations == 0)
		return false;
	const bool visibilityAwareComponents(
		maps.visibilityRequested && maps.apdStageActive);
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String csvPath(dir + _T("apd_iteration.csv"));
	const bool writeHeader(!File::access(csvPath));
	std::ofstream csv(csvPath.c_str(), std::ios::app);
	if (!csv)
		return false;
	if (writeHeader) {
		csv << "image_id,scale_number,width,height,logical_iteration,classified,reliability_unknown,reliability_unreliable,reliability_reliable,ransac_valid,deformable_eligible,deformable_updates,global_minimum_cost_samples,global_minimum_cost_mean,separation_samples,separation_mean,anchor_count_mean,working_gap_samples,working_gap_mean,center_cost_mean,anchor_mean_cost_mean,working_cost_mean,native_persistent_cost_mean,anchor_view_selection_attempted,anchor_view_selection_used,anchor_proposals_tested,anchor_proposals_finite,anchor_proposals_accepted,anchor_propagation_final_winners,immutable_anchor_state_updates,best_anchor_working_cost_samples,best_anchor_working_cost_mean,accepted_anchor_native_cost_samples,accepted_anchor_native_cost_mean,stage_all_compatibility,stage_all_deformation_only,stage_reliable_first,stage_non_reliable_second,fitted_plane_available,fitted_plane_tested,fitted_plane_finite,fitted_plane_accepted,fitted_plane_final_winners,final_refinement_pixels,final_refinement_candidates_tested,final_refinement_candidates_finite,final_refinement_accepted";
		for (int mode=0; mode<PM_APD_INSTRUMENT_VIEW_SELECTION_MODES; ++mode)
			csv << ",view_selection_mode_" << APDViewSelectionModeName(mode);
		for (int reason=0; reason<PM_APD_INSTRUMENT_PROFILE_REASONS; ++reason)
			csv << ",profile_reason_" << APDProfileReasonName(reason);
		for (int reason=0; reason<PM_APD_INSTRUMENT_ANCHOR_REASONS; ++reason)
			csv << ",anchor_reason_" << APDAnchorReasonName(reason);
		for (int count=0; count<=PM_APD_INSTRUMENT_ANCHORS; ++count)
			csv << ",anchor_count_" << count;
		for (int source=0; source<PM_INSTRUMENT_NUM_SOURCES; ++source)
			csv << ",source_" << InstrumentSourceName(source);
		csv << '\n';
	}
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		const PatchMatchAPDInstrumentCounters& c(maps.apdCounters[iteration]);
		auto mean = [](double sum, uint32_t count) { return count > 0u ? sum/(double)count : -1.; };
		csv << imageID << ',' << scaleNumber << ',' << size.width << ',' << size.height << ','
			<< iteration << ',' << c.classified << ',' << c.reliability[0] << ','
			<< c.reliability[1] << ',' << c.reliability[2] << ',' << c.ransacValid << ','
			<< c.deformableEligible << ',' << c.deformableUpdates << ','
			<< c.globalMinimumCostSamples << ',' << mean(c.globalMinimumCostSum, c.globalMinimumCostSamples) << ','
			<< c.separationSamples << ',' << mean(c.separationSum, c.separationSamples) << ','
			<< mean(c.anchorCountSum, c.classified) << ','
			<< c.workingGapSamples << ',' << mean(c.workingGapSum, c.workingGapSamples) << ','
			<< mean(c.centerCostSum, c.deformableUpdates) << ','
			<< mean(c.anchorMeanCostSum, c.deformableUpdates) << ','
			<< mean(c.workingCostSum, c.deformableUpdates) << ','
			<< mean(c.nativePersistentCostSum, c.deformableUpdates) << ','
			<< c.anchorViewSelectionAttempted << ',' << c.anchorViewSelectionUsed << ','
			<< c.anchorProposalsTested << ',' << c.anchorProposalsFinite << ','
			<< c.anchorProposalsAccepted << ',' << c.anchorPropagationFinalWinners << ','
			<< c.immutableAnchorStateUpdates << ',' << c.bestAnchorWorkingCostSamples << ','
			<< mean(c.bestAnchorWorkingCostSum, c.bestAnchorWorkingCostSamples) << ','
			<< c.acceptedAnchorNativeCostSamples << ','
			<< mean(c.acceptedAnchorNativeCostSum, c.acceptedAnchorNativeCostSamples) << ','
			<< c.stageUpdates[0] << ','
			<< (OPTDENSE::nPatchMatchCUDAAPD == static_cast<unsigned>(APDMode::DEFORMATION_ONLY) ?
				c.stageUpdates[0] : 0u) << ','
			<< c.stageUpdates[1] << ',' << c.stageUpdates[2] << ','
			<< c.fittedPlaneAvailable << ',' << c.fittedPlaneTested << ','
			<< c.fittedPlaneFinite << ',' << c.fittedPlaneAccepted << ','
			<< c.fittedPlaneFinalWinners << ',' << c.finalRefinementPixels << ','
			<< c.finalRefinementCandidatesTested << ','
			<< c.finalRefinementCandidatesFinite << ',' << c.finalRefinementAccepted;
		for (uint32_t value : c.anchorViewSelectionMode)
			csv << ',' << value;
		for (uint32_t value : c.profileReason)
			csv << ',' << value;
		for (uint32_t value : c.anchorReason)
			csv << ',' << value;
		for (uint32_t value : c.anchorCountBins)
			csv << ',' << value;
		for (uint32_t value : c.updateSource)
			csv << ',' << value;
		csv << '\n';
	}
	csv.flush();
	if (!csv)
		return false;
	if (tracePixels.empty() && maps.apdTraces.empty())
		return true;
	if (tracePixels.empty() || maps.apdTraces.size() != tracePixels.size()*numIterations)
		return false;
	std::ofstream traces((dir + _T("apd_traces.jsonl")).c_str(), std::ios::app);
	if (!traces)
		return false;
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		for (size_t traceIndex=0; traceIndex<tracePixels.size(); ++traceIndex) {
			const PatchMatchAPDInstrumentTrace& record(
				maps.apdTraces[iteration*tracePixels.size()+traceIndex]);
			if (!record.valid)
				return false;
			const PatchMatchAPDInstrumentState& state(record.state);
			const PatchMatchAPDInstrumentUpdate& update(record.update);
			nlohmann::json line = {
				{"schema_name", "openmvs.dmap.apd_trace"},
				{"schema_version", 3},
				{"measurement_quality", "proxy"},
				{"measurement_basis", "exact_hot_kernel_state_and_update_with_post_winner_per_view_replay"},
				{"component_contract", {
					{"anchor_component_field", "anchor_mean_cost"},
					{"anchor_component_basis", visibilityAwareComponents ?
						"visibility_aware_effective_anchor_leg" : "active_anchor_mean"},
					{"deformable_photometric_identity",
						"0.25*center_cost + 0.75*anchor_component"},
					{"visibility_fallback", visibilityAwareComponents ?
						"center_cost_substitutes_for_anchor_component_when_per_view_visible_anchor_consensus_is_invalid" :
						"not_active"}
				}},
				{"field_quality", {
					{"paper_profile", "exact"},
					{"anchor_model", "exact"},
					{"update", "exact"},
					{"anchor_view_selection", "exact_immutable_production_evidence"},
					{"anchor_candidate_working_costs", "exact_production_values"},
					{"anchor_candidate_native_costs", "exact_same_kernel_diagnostic_rescore_not_used_for_ranking"},
					{"views", "diagnostic_replay_proxy"}
				}},
				{"image_id", imageID},
				{"scale_number", scaleNumber},
				{"logical_iteration", iteration},
				{"trace_index", traceIndex},
				{"label", tracePixels[traceIndex].label.c_str()},
				{"x", record.x},
				{"y", record.y},
				{"paper_profile", {
					{"delta", APD_PROFILE_RADIUS},
					{"eta", state.eta},
					{"costs", std::vector<float>(record.profile, record.profile+PM_APD_INSTRUMENT_PROFILE_SAMPLES)},
					{"reliability", state.reliability},
					{"reason", APDProfileReasonName(state.profileReason)},
					{"global_minimum_offset", state.globalMinimumOffset},
					{"global_minimum_cost", state.globalMinimumCost},
					{"local_minimum_count", state.localMinimumCount},
					{"separation", state.separation},
					{"plateau_start", state.globalMinimumPlateauStart},
					{"plateau_end", state.globalMinimumPlateauEnd}
				}},
				{"anchor_model", {
					{"nearest_reliable_index", state.nearestReliable < (uint32_t)size.area() ? nlohmann::json(state.nearestReliable) : nlohmann::json(nullptr)},
					{"nearest_reliable_distance", state.nearestReliableDistance},
					{"candidate_count", state.candidateCount},
					{"ransac_valid", state.ransacValid != 0},
					{"ransac_threshold", state.ransacThreshold},
					{"inlier_count", state.inlierCount},
					{"outlier_count", state.outlierCount},
					{"center_residual", state.ransacCenterResidual},
					{"mean_inlier_residual", state.ransacMeanInlierResidual},
					{"anchor_count", state.anchorCount},
					{"reason", APDAnchorReasonName(state.anchorReason)},
					{"deformable_eligible", state.deformableEligible != 0},
					{"fitted_plane_valid", state.fittedPlaneValid != 0},
					{"fitted_plane_depth", state.fittedPlaneDepth}
				}},
				{"update", {
					{"deformable_active", update.deformableActive != 0},
					{"source", InstrumentSourceName(update.source)},
					{"working_winner_cost", update.workingWinnerCost},
					{"native_persistent_cost", update.nativePersistentCost},
					{"native_minus_working_cost", update.nativeMinusWorkingCost},
					{"incumbent_working_cost", update.incumbentWorkingCost},
					{"runner_up_working_cost", update.runnerUpWorkingCost},
					{"winner_runner_up_gap", update.winnerRunnerUpGap},
					{"center_cost", update.centerCost},
					{"anchor_mean_cost", update.anchorMeanCost},
					{"deformable_photometric_cost", update.deformablePhotometricCost},
					{"geometric_cost", update.geometricCost},
					{"best_anchor_working_cost", update.bestAnchorWorkingCost},
					{"accepted_anchor_native_cost", update.acceptedAnchorNativeCost},
					{"accepted_anchor_index", update.acceptedAnchorIndex < (uint32_t)size.area() ? nlohmann::json(update.acceptedAnchorIndex) : nlohmann::json(nullptr)},
					{"anchor_accepted_slot", update.anchorAcceptedSlot != PM_INSTRUMENT_EXACT_SLOT_UNAVAILABLE ? nlohmann::json(update.anchorAcceptedSlot) : nlohmann::json(nullptr)},
					{"view_selection_mode", APDViewSelectionModeName(update.viewSelectionMode)},
					{"anchor_evidence_count", update.anchorEvidenceCount},
					{"anchor_proposal_count", update.anchorProposalCount},
					{"anchor_finite_count", update.anchorFiniteCount},
					{"immutable_anchor_state", update.immutableAnchorState != 0},
					{"update_stage", update.updateStage},
					{"fitted_plane_available", update.fittedPlaneAvailable != 0},
					{"fitted_plane_tested", update.fittedPlaneTested != 0},
					{"fitted_plane_accepted", update.fittedPlaneAccepted != 0},
					{"fitted_plane_working_cost", update.fittedPlaneWorkingCost},
					{"fitted_plane_native_cost", update.fittedPlaneNativeCost},
					{"final_refinement", {
						{"incumbent_cost", update.finalRefinementIncumbentCost},
						{"best_cost", update.finalRefinementBestCost},
						{"improvement", update.finalRefinementImprovement},
						{"depth", update.finalRefinementDepth},
						{"offset", update.finalRefinementOffset},
						{"tested_count", update.finalRefinementTested},
						{"finite_count", update.finalRefinementFinite},
						{"accepted", update.finalRefinementAccepted != 0}
					}},
					{"candidate_tested_mask", update.candidateTestedMask},
					{"candidate_finite_mask", update.candidateFiniteMask},
					{"candidate_accepted_mask", update.candidateAcceptedMask},
					{"winner_slot", update.winnerSlot},
					{"runner_up_slot", update.runnerUpSlot},
					{"tested_count", update.testedCount},
					{"finite_count", update.finiteCount},
					{"accepted_count", update.acceptedCount},
					{"selected_view_count", update.selectedViewCount},
					{"working_selected_views", update.workingSelectedViews},
					{"selected_view_weight_sum", update.selectedViewWeightSum}
				}}
			};
			nlohmann::json sectorCandidates(nlohmann::json::array());
			for (int sector=0; sector<PM_APD_INSTRUMENT_SECTORS; ++sector) {
				const uint32_t index(record.sectorCandidates[sector]);
				sectorCandidates.push_back(index < (uint32_t)size.area() ? nlohmann::json({
					{"sector", sector}, {"index", index}, {"x", index%(uint32_t)size.width},
					{"y", index/(uint32_t)size.width}}) : nlohmann::json(nullptr));
			}
			line["anchor_model"]["sector_candidates"] = std::move(sectorCandidates);
			nlohmann::json anchors(nlohmann::json::array());
			for (int slot=0; slot<PM_APD_INSTRUMENT_ANCHORS; ++slot) {
				const uint32_t index(record.anchors[slot]);
				anchors.push_back(index < (uint32_t)size.area() ? nlohmann::json({
					{"slot", slot}, {"index", index}, {"x", index%(uint32_t)size.width},
					{"y", index/(uint32_t)size.width}, {"normalized_plane_residual", record.anchorResiduals[slot]},
					{"candidate_valid", record.anchorCandidateValid[slot] != 0},
					{"snapshot_selected_views", record.anchorSelectedViews[slot]},
					{"candidate_working_cost", record.anchorCandidateWorkingCosts[slot]},
					{"candidate_native_rescore", record.anchorCandidateNativeCosts[slot]}}) :
					nlohmann::json(nullptr));
			}
			line["anchor_model"]["anchors"] = std::move(anchors);
			nlohmann::json views(nlohmann::json::array());
			for (int view=0; view<numViews && view<PM_INSTRUMENT_MAX_VIEWS; ++view) {
				views.push_back({
					{"view_index", view},
					{"reliability_weight", record.viewWeights[view]},
					{"selection_prior", record.viewSelectionPriors[view]},
					{"sampling_score", record.viewSamplingScores[view]},
					{"sampling_probability", record.viewSamplingProbabilities[view]},
					{"center_cost", record.viewCenterCosts[view]},
					{"anchor_mean_cost", record.viewAnchorMeanCosts[view]},
					{"working_cost", record.viewWorkingCosts[view]}
				});
			}
			line["views"] = std::move(views);
			traces << line.dump() << '\n';
			if (!traces)
				return false;
		}
	}
	traces.flush();
	return (bool)traces;
}

bool AppendDVPInstrumentation(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const PatchMatch::Params& params,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const InstrumentExtendedMaps& maps)
{
	if (!maps.dvpRequested)
		return true;
	const size_t numIterations(maps.dvpCounters.size());
	if (!maps.dvpAvailable || numIterations == 0)
		return false;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String csvPath(dir + _T("dvp_iteration.csv"));
	const bool writeHeader(!File::access(csvPath));
	std::ofstream csv(csvPath.c_str(), std::ios::app);
	if (!csv)
		return false;
	if (writeHeader) {
		csv << "image_id,scale_number,width,height,logical_iteration,family,promotion_eligible,attempted_pixels,proposal_available_pixels,native_depth_fallback_pixels,proposals_generated,proposals_tested,proposals_finite,proposals_accepted,final_winner_pixels,final_depth_retained_pixels";
		for (int source=0; source<PM_DVP_INSTRUMENT_FINAL_SOURCES; ++source)
			csv << ",final_source_" << InstrumentSourceName(source);
		csv << ",left_interval_valid_pixels,right_interval_valid_pixels,selected_source_view_count_mean,direction_source_view_count_mean,left_outer_support_mean,left_inner_support_mean,right_inner_support_mean,right_outer_support_mean,proposal_support_mean,occluded_candidates,cost_samples,incumbent_cost_mean,winner_cost_mean,improvement_mean,gap_samples,winner_runner_up_gap_mean,displacement_samples,depth_displacement_mean,reprojection_error_samples,reprojection_error_mean,relative_depth_error_samples,relative_depth_error_mean";
		for (int reason=0; reason<PM_DVP_INSTRUMENT_UNAVAILABLE_REASONS; ++reason)
			csv << ",unavailable_" << DVPUnavailableReasonName((unsigned)reason);
		csv << '\n';
	}
	auto mean = [](double sum, uint32_t count) { return count > 0u ? sum/(double)count : -1.; };
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		const PatchMatchDVPInstrumentCounters& c(maps.dvpCounters[iteration]);
		csv << imageID << ',' << scaleNumber << ',' << size.width << ',' << size.height << ','
			<< iteration << ',' << DVPFamilyName(params.nDVPEpipolarFamily) << ','
			<< (DVPFamilyPromotionEligible(params.nDVPEpipolarFamily) ? 1 : 0) << ','
			<< c.attemptedPixels << ',' << c.proposalAvailablePixels << ','
			<< c.nativeDepthFallbackPixels << ',' << c.proposalsGenerated << ','
			<< c.proposalsTested << ',' << c.proposalsFinite << ',' << c.proposalsAccepted << ','
			<< c.finalWinnerPixels << ',' << c.finalDepthRetainedPixels;
		for (uint32_t value : c.finalUpdateSource)
			csv << ',' << value;
		csv << ',' << c.leftIntervalValidPixels << ','
			<< c.rightIntervalValidPixels << ','
			<< mean((float)c.selectedSourceViewCountSum, c.attemptedPixels) << ','
			<< mean((float)c.directionSourceViewCountSum, c.attemptedPixels) << ','
			<< mean((float)c.endpointSupportSum[0], c.attemptedPixels) << ','
			<< mean((float)c.endpointSupportSum[1], c.attemptedPixels) << ','
			<< mean((float)c.endpointSupportSum[2], c.attemptedPixels) << ','
			<< mean((float)c.endpointSupportSum[3], c.attemptedPixels) << ','
			<< mean((float)c.proposalSupportSum, c.proposalSupportSamples) << ','
			<< c.occludedCandidates << ',' << c.costSamples << ','
			<< mean(c.incumbentCostSum, c.costSamples) << ','
			<< mean(c.winnerCostSum, c.costSamples) << ','
			<< mean(c.improvementSum, c.costSamples) << ','
			<< c.gapSamples << ',' << mean(c.gapSum, c.gapSamples) << ','
			<< c.displacementSamples << ',' << mean(c.displacementSum, c.displacementSamples) << ','
			<< c.reprojectionErrorSamples << ','
			<< mean(c.reprojectionErrorSum, c.reprojectionErrorSamples) << ','
			<< c.relativeDepthErrorSamples << ','
			<< mean(c.relativeDepthErrorSum, c.relativeDepthErrorSamples);
		for (uint32_t value : c.unavailableReason)
			csv << ',' << value;
		csv << '\n';
	}
	csv.flush();
	if (!csv)
		return false;
	if (tracePixels.empty() && maps.dvpTraces.empty())
		return true;
	if (tracePixels.empty() || maps.dvpTraces.size() != tracePixels.size()*numIterations)
		return false;
	std::ofstream traces((dir + _T("dvp_traces.jsonl")).c_str(), std::ios::app);
	if (!traces)
		return false;
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		for (size_t traceIndex=0; traceIndex<tracePixels.size(); ++traceIndex) {
			const PatchMatchDVPInstrumentTrace& record(
				maps.dvpTraces[iteration*tracePixels.size()+traceIndex]);
			if (!record.valid)
				return false;
			const PatchMatchDVPInstrumentUpdate& update(record.update);
			nlohmann::json proposals(nlohmann::json::array());
			for (unsigned proposal=0; proposal<update.generatedCount; ++proposal) {
				proposals.push_back({
					{"ordinal", proposal},
					{"tested", (update.testedProposalMask & (1u << proposal)) != 0u},
					{"finite", (update.finiteProposalMask & (1u << proposal)) != 0u},
					{"sequentially_accepted", (update.acceptedProposalMask & (1u << proposal)) != 0u},
					{"depth", update.proposalDepth[proposal]},
					{"signed_offset", update.signedOffset[proposal]},
					{"source_view", update.sourceView[proposal] == 0xffu ?
						nlohmann::json(nullptr) : nlohmann::json(update.sourceView[proposal])},
					{"support", update.support[proposal]},
					{"support_views", update.supportViews[proposal]},
					{"occluded_views", update.occludedViews[proposal]},
					{"candidate_cost", update.candidateCost[proposal]},
					{"mean_reprojection_error_px", update.meanReprojectionError[proposal]},
					{"max_reprojection_error_px", update.maxReprojectionError[proposal]},
					{"mean_relative_depth_error", update.meanRelativeDepthError[proposal]},
					{"max_relative_depth_error", update.maxRelativeDepthError[proposal]}
				});
			}
			auto endpointRecords = [](const float* depths, unsigned count,
				uint32_t views, float offset) {
				nlohmann::json records(nlohmann::json::array());
				unsigned sample(0u);
				for (unsigned view=0u; view<DVP_MAX_SOURCE_VIEWS && sample<count; ++view) {
					if ((views & (1u << view)) == 0u)
						continue;
					records.push_back({
						{"view", view},
						{"offset_px", offset},
						{"back_projected_depth", depths[sample++]}
					});
				}
				return records;
			};
			const nlohmann::json leftOuterRecords(endpointRecords(
				record.leftOuter, update.leftOuterCount, record.leftOuterViews,
				-(params.fDVPEpipolarAlpha+params.fDVPEpipolarBeta)));
			const nlohmann::json leftInnerRecords(endpointRecords(
				record.leftInner, update.leftInnerCount, record.leftInnerViews,
				-params.fDVPEpipolarAlpha));
			const nlohmann::json rightInnerRecords(endpointRecords(
				record.rightInner, update.rightInnerCount, record.rightInnerViews,
				params.fDVPEpipolarAlpha));
			const nlohmann::json rightOuterRecords(endpointRecords(
				record.rightOuter, update.rightOuterCount, record.rightOuterViews,
				params.fDVPEpipolarAlpha+params.fDVPEpipolarBeta));
			if (leftOuterRecords.size() != update.leftOuterCount ||
				leftInnerRecords.size() != update.leftInnerCount ||
				rightInnerRecords.size() != update.rightInnerCount ||
				rightOuterRecords.size() != update.rightOuterCount)
			{
				return false;
			}
			nlohmann::json line = {
				{"schema_name", "openmvs.dmap.dvp_trace"},
				{"schema_version", PM_DVP_INSTRUMENT_SCHEMA_VERSION},
				{"measurement_quality", "exact"},
				{"measurement_basis", "active_process_pixel_candidate_path"},
				{"image_id", imageID},
				{"scale_number", scaleNumber},
				{"logical_iteration", iteration},
				{"trace_index", traceIndex},
				{"label", tracePixels[traceIndex].label.c_str()},
				{"x", record.x},
				{"y", record.y},
				{"family", DVPFamilyName(update.family)},
				{"promotion_eligible", DVPFamilyPromotionEligible(update.family)},
				{"paper_contract_label", DVPClaimLabel(update.family)},
				{"author_code_exact_equivalence", false},
				{"unavailable_reason", DVPUnavailableReasonName(update.unavailableReason)},
				{"selected_source_views", update.selectedSourceViews},
				{"direction_source_views", update.directionSourceViews},
				{"endpoint_samples", {
					{"left_outer", std::vector<float>(record.leftOuter, record.leftOuter+update.leftOuterCount)},
					{"left_inner", std::vector<float>(record.leftInner, record.leftInner+update.leftInnerCount)},
					{"right_inner", std::vector<float>(record.rightInner, record.rightInner+update.rightInnerCount)},
					{"right_outer", std::vector<float>(record.rightOuter, record.rightOuter+update.rightOuterCount)}
				}},
				{"endpoint_sample_views", {
					{"left_outer", record.leftOuterViews},
					{"left_inner", record.leftInnerViews},
					{"right_inner", record.rightInnerViews},
					{"right_outer", record.rightOuterViews}
				}},
				{"endpoint_records", {
					{"left_outer", leftOuterRecords},
					{"left_inner", leftInnerRecords},
					{"right_inner", rightInnerRecords},
					{"right_outer", rightOuterRecords}
				}},
				{"intervals", {
					{"left", {{"valid", update.leftIntervalValid != 0}, {"minimum", update.leftIntervalMinimum}, {"maximum", update.leftIntervalMaximum}}},
					{"right", {{"valid", update.rightIntervalValid != 0}, {"minimum", update.rightIntervalMinimum}, {"maximum", update.rightIntervalMaximum}}}
				}},
				{"proposals", proposals},
				{"decision", {
					{"incumbent_depth", update.incumbentDepth},
					{"final_depth", update.finalDepth},
					{"incumbent_cost", update.incumbentCost},
					{"winner_cost", update.winnerCost},
					{"runner_up_cost", update.runnerUpCost},
					{"winner_runner_up_gap", update.winnerRunnerUpGap},
					{"winner_ordinal", update.winnerOrdinal == 0xffu ?
						nlohmann::json(nullptr) : nlohmann::json(update.winnerOrdinal)},
					{"dvp_accepted", update.accepted != 0},
					{"accepted_proposal_mask", update.acceptedProposalMask},
					{"dvp_final_winner", update.finalWinner != 0},
					{"dvp_final_hypothesis_winner", update.finalWinner != 0},
					{"final_update_source", InstrumentSourceName(update.finalUpdateSource)},
					{"final_update_source_code", update.finalUpdateSource},
					{"dvp_depth_retained", update.finalDepthRetained != 0},
					{"native_depth_fallback", update.nativeDepthFallback != 0},
					{"depth_displacement", update.depthDisplacement},
					{"generated_count", update.generatedCount},
					{"tested_count", update.testedCount},
					{"finite_count", update.finiteCount},
					{"sequential_acceptance_count", update.acceptedCount}
				}}
			};
			traces << line.dump() << '\n';
			if (!traces)
				return false;
		}
	}
	traces.flush();
	return (bool)traces;
}

bool AppendDVPVisibilityInstrumentation(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const PatchMatch::Params& params,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const InstrumentExtendedMaps& maps)
{
	if (!maps.visibilityRequested)
		return true;
	if (!maps.apdStageActive)
		return true;
	const size_t numIterations(maps.visibilityCounters.size());
	if (!maps.visibilityAvailable || numIterations == 0)
		return false;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String csvPath(dir + _T("dvp_visibility_iteration.csv"));
	const bool writeHeader(!File::access(csvPath));
	std::ofstream csv(csvPath.c_str(), std::ios::app);
	if (!csv)
		return false;
	if (writeHeader) {
		csv << "image_id,scale_number,width,height,logical_iteration,mode,pixels,previous_visible_views,resolved_visible_views,next_visible_views,active_support_views,restored_views,rejected_views,added_views,removed_views,previous_weight_sum,resolved_weight_sum,next_weight_sum,active_support_weight_sum,denominator_sum,zero_denominator_pixels,support_mismatch_pixels,changed_pixels,invalid_transition_pixels,candidate_tested_pixels,candidate_finite_pixels,candidate_tested_count,candidate_finite_count";
		for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason)
			csv << ",reason_" << DVPVisibilityReasonName((unsigned)reason);
		csv << '\n';
	}
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		const PatchMatchDVPVisibilityInstrumentCounters& c(
			maps.visibilityCounters[iteration]);
		csv << imageID << ',' << scaleNumber << ',' << size.width << ',' << size.height << ','
			<< iteration << ',' << DVPVisibilityModeName(params.nDVPVisibilityMode) << ','
			<< c.pixels << ',' << c.previousVisibleViews << ',' << c.resolvedVisibleViews << ','
			<< c.nextVisibleViews << ',' << c.activeSupportViews << ',' << c.restoredViews << ','
			<< c.rejectedViews << ',' << c.addedViews << ',' << c.removedViews << ','
			<< c.previousWeightSum << ',' << c.resolvedWeightSum << ',' << c.nextWeightSum << ','
			<< c.activeSupportWeightSum << ',' << c.denominatorSum << ','
			<< c.zeroDenominatorPixels << ',' << c.supportMismatchPixels << ','
			<< c.changedPixels << ',' << c.invalidTransitionPixels << ','
			<< c.candidateTestedPixels << ',' << c.candidateFinitePixels << ','
			<< c.candidateTestedCount << ',' << c.candidateFiniteCount;
		for (uint32_t value : c.reason)
			csv << ',' << value;
		csv << '\n';
	}
	csv.flush();
	if (!csv)
		return false;
	if (tracePixels.empty() && maps.visibilityTraces.empty())
		return true;
	if (tracePixels.empty() ||
		maps.visibilityTraces.size() != tracePixels.size()*numIterations)
	{
		return false;
	}
	std::ofstream traces((dir + _T("dvp_visibility_traces.jsonl")).c_str(), std::ios::app);
	if (!traces)
		return false;
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		for (size_t traceIndex=0; traceIndex<tracePixels.size(); ++traceIndex) {
			const PatchMatchDVPVisibilityInstrumentTrace& record(
				maps.visibilityTraces[iteration*tracePixels.size()+traceIndex]);
			if (!record.valid || record.imageID != imageID ||
				record.scaleNumber != scaleNumber ||
				record.logicalIteration != static_cast<int32_t>(iteration))
			{
				return false;
			}
			const PatchMatchDVPVisibilityInstrumentUpdate& update(record.update);
			nlohmann::json views(nlohmann::json::array());
			for (int view=0; view<maps.numViews; ++view) {
				const uint32_t bit(1u << view);
				views.push_back({
					{"view_index", view},
					{"previous_weight", record.previousWeights[view]},
					{"resolved_weight", record.resolvedWeights[view]},
					{"next_weight", record.nextWeights[view]},
					{"active_support_weight", record.activeSupportWeights[view]},
					{"normalized_active_support_weight", update.denominator > 0u ?
						nlohmann::json((double)record.activeSupportWeights[view]/update.denominator) :
						nlohmann::json(nullptr)},
					{"reason_code", record.reasons[view]},
					{"reason", DVPVisibilityReasonName(record.reasons[view])},
					{"restored", (update.restoredMask & bit) != 0u},
					{"rejected", (update.rejectedMask & bit) != 0u},
					{"added_next", (update.addedMask & bit) != 0u},
					{"removed_next", (update.removedMask & bit) != 0u}
				});
			}
			nlohmann::json reasonCounts(nlohmann::json::object());
			for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason)
				reasonCounts[DVPVisibilityReasonName((unsigned)reason)] = update.reasonCount[reason];
			nlohmann::json line = {
				{"schema_name", "openmvs.dmap.dvp_visibility_trace"},
				{"schema_version", PM_DVP_VISIBILITY_INSTRUMENT_SCHEMA_VERSION},
				{"measurement_quality", "exact"},
				{"measurement_basis", "same_stream_logical_iteration_visibility_transition"},
				{"image_id", imageID},
				{"scale_number", scaleNumber},
				{"logical_iteration", iteration},
				{"trace_index", traceIndex},
				{"label", tracePixels[traceIndex].label.c_str()},
				{"x", record.x},
				{"y", record.y},
				{"mode", DVPVisibilityModeName(update.mode)},
				{"state", {
					{"previous_mask", update.previousMask},
					{"resolved_mask", update.resolvedMask},
					{"next_mask", update.nextMask},
					{"active_support_mask", update.activeSupportMask},
					{"restored_mask", update.restoredMask},
					{"rejected_mask", update.rejectedMask},
					{"added_mask", update.addedMask},
					{"removed_mask", update.removedMask},
					{"transition_status_code", update.transitionStatus},
					{"transition_status", DVPVisibilityTransitionStatusName(update.transitionStatus)},
					{"active_support_matches_resolved", update.activeSupportMatchesResolved != 0u}
				}},
				{"weights", {
					{"previous_sum", update.previousWeightSum},
					{"resolved_sum", update.resolvedWeightSum},
					{"next_sum", update.nextWeightSum},
					{"active_support_sum", update.activeSupportWeightSum},
					{"denominator", update.denominator},
					{"denominator_defined", update.denominatorDefined != 0u}
				}},
				{"candidates", {
					{"tested_mask", update.candidateTestedMask},
					{"finite_mask", update.candidateFiniteMask}
				}},
				{"reason_counts", reasonCounts},
				{"views", views}
			};
			traces << line.dump() << '\n';
			if (!traces)
				return false;
		}
	}
	traces.flush();
	return (bool)traces;
}

bool AppendDVPVisibleNormalInstrumentation(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	const PatchMatch::Params& params,
	const std::vector<InstrumentTracePixel>& tracePixels,
	const InstrumentExtendedMaps& maps)
{
	if (!maps.visibleNormalRequested || !maps.apdStageActive)
		return true;
	const size_t numIterations(maps.visibleNormalCounters.size());
	if (!maps.visibleNormalAvailable || numIterations == 0)
		return false;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String csvPath(dir + _T("dvp_visible_normal_iteration.csv"));
	const bool writeHeader(!File::access(csvPath));
	std::ofstream csv(csvPath.c_str(), std::ios::app);
	if (!csv)
		return false;
	if (writeHeader) {
		csv << "image_id,scale_number,width,height,logical_iteration,mode,pixels,selected_support_views,current_valid,current_feasible,current_rejected,current_invalid,propagation_tested,propagation_valid,propagation_feasible,propagation_rejected,propagation_constrained_selections,propagation_native_selections,propagation_fallbacks,propagation_applied_constraints,propagation_accepted,refinement_native_tested,refinement_native_valid,refinement_native_feasible,refinement_native_rejected,refinement_retries_tested,refinement_retry_success,refinement_exhaustion,refinement_fallback,refinement_applied_retry,refinement_accepted";
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_EVALUATION_REASONS; ++reason)
			csv << ",evaluation_reason_" << DVPVisibleNormalEvaluationReasonName((unsigned)reason);
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPOSAL_REASONS; ++reason)
			csv << ",proposal_reason_" << DVPVisibleNormalProposalReasonName((unsigned)reason);
		for (int reason=0; reason<PM_DVP_VISIBLE_NORMAL_PROPAGATION_REASONS; ++reason)
			csv << ",propagation_reason_" << DVPVisibleNormalPropagationReasonName((unsigned)reason);
		csv << '\n';
	}
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		const PatchMatchDVPVisibleNormalInstrumentCounters& c(
			maps.visibleNormalCounters[iteration]);
		csv << imageID << ',' << scaleNumber << ',' << size.width << ',' << size.height << ','
			<< iteration << ',' << DVPVisibleNormalModeName(params.nDVPVisibleNormalMode) << ','
			<< c.pixels << ',' << c.selectedSupportViews << ','
			<< c.currentValid << ',' << c.currentFeasible << ',' << c.currentRejected << ','
			<< c.currentInvalid << ',' << c.propagationCandidatesTested << ','
			<< c.propagationCandidatesValid << ',' << c.propagationCandidatesFeasible << ','
			<< c.propagationCandidatesRejected << ',' << c.propagationConstrainedSelections << ','
			<< c.propagationNativeSelections << ',' << c.propagationFallbacks << ','
			<< c.propagationAppliedConstraints << ',' << c.propagationAccepted << ','
			<< c.refinementNativeTested << ',' << c.refinementNativeValid << ','
			<< c.refinementNativeFeasible << ',' << c.refinementNativeRejected << ','
			<< c.refinementRetriesTested << ',' << c.refinementRetrySuccess << ','
			<< c.refinementExhaustion << ',' << c.refinementFallback << ','
			<< c.refinementAppliedRetry << ',' << c.refinementAccepted;
		for (uint32_t value : c.evaluationReason)
			csv << ',' << value;
		for (uint32_t value : c.proposalReason)
			csv << ',' << value;
		for (uint32_t value : c.propagationReason)
			csv << ',' << value;
		csv << '\n';
	}
	csv.flush();
	if (!csv)
		return false;
	if (tracePixels.empty() && maps.visibleNormalTraces.empty())
		return true;
	if (tracePixels.empty() ||
		maps.visibleNormalTraces.size() != tracePixels.size()*numIterations)
	{
		return false;
	}
	std::ofstream traces((dir + _T("dvp_visible_normal_traces.jsonl")).c_str(),
		std::ios::app);
	if (!traces)
		return false;
	for (size_t iteration=0; iteration<numIterations; ++iteration) {
		for (size_t traceIndex=0; traceIndex<tracePixels.size(); ++traceIndex) {
			const PatchMatchDVPVisibleNormalInstrumentTrace& record(
				maps.visibleNormalTraces[iteration*tracePixels.size()+traceIndex]);
			if (!record.valid || record.imageID != imageID ||
				record.scaleNumber != scaleNumber ||
				record.logicalIteration != static_cast<int32_t>(iteration))
			{
				return false;
			}
			const PatchMatchDVPVisibleNormalInstrumentUpdate& update(record.update);
			nlohmann::json propagation(nlohmann::json::array());
			for (int candidate=0;
				candidate<DVP_VISIBLE_NORMAL_MAX_PROPAGATION_CANDIDATES; ++candidate)
			{
				if (!(update.propagationTestedMask & (1u << candidate)))
					continue;
				propagation.push_back({
					{"candidate", candidate},
					{"normal_reference", {
						record.propagationNormals[candidate][0],
						record.propagationNormals[candidate][1],
						record.propagationNormals[candidate][2]}},
					{"cost", record.propagationCosts[candidate]},
					{"max_dot", record.propagationMaxDot[candidate]},
					{"reason_code", record.propagationReason[candidate]},
					{"reason", DVPVisibleNormalEvaluationReasonName(
						record.propagationReason[candidate])},
					{"rejected_direction", record.propagationRejectedDirection[candidate]},
					{"feasible", (update.propagationFeasibleMask & (1u << candidate)) != 0u}
				});
			}
			nlohmann::json refinement(nlohmann::json::array());
			static const char* proposalNames[DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS] = {
				"perturbed_normal", "random_normal"};
			for (int proposal=0; proposal<DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS; ++proposal) {
				if (!(update.refinementNativeTestedMask & (1u << proposal)))
					continue;
				refinement.push_back({
					{"proposal", proposalNames[proposal]},
					{"native_normal_reference", {
						record.nativeNormals[proposal][0], record.nativeNormals[proposal][1],
						record.nativeNormals[proposal][2]}},
					{"selected_normal_reference", {
						record.selectedNormals[proposal][0], record.selectedNormals[proposal][1],
						record.selectedNormals[proposal][2]}},
					{"native_max_dot", update.nativeMaxDot[proposal]},
					{"selected_max_dot", update.selectedMaxDot[proposal]},
					{"native_reason_code", update.nativeReason[proposal]},
					{"native_reason", DVPVisibleNormalEvaluationReasonName(
						update.nativeReason[proposal])},
					{"proposal_reason_code", update.proposalReason[proposal]},
					{"proposal_reason", DVPVisibleNormalProposalReasonName(
						update.proposalReason[proposal])},
					{"retries_tested", update.retriesTested[proposal]},
					{"selected_retry", update.selectedRetry[proposal] == 0xffffu ?
						nlohmann::json(nullptr) : nlohmann::json(update.selectedRetry[proposal])},
					{"retry_applied", (update.refinementAppliedRetryMask &
						(1u << proposal)) != 0u},
					{"accepted", (update.refinementAcceptedMask & (1u << proposal)) != 0u}
				});
			}
			nlohmann::json line = {
				{"schema_name", "openmvs.dmap.dvp_visible_normal_trace"},
				{"schema_version", PM_DVP_VISIBLE_NORMAL_INSTRUMENT_SCHEMA_VERSION},
				{"measurement_quality", "exact"},
				{"measurement_basis", "same_stream_Process_true_logical_iteration_proposals"},
				{"image_id", imageID}, {"scale_number", scaleNumber},
				{"logical_iteration", iteration}, {"trace_index", traceIndex},
				{"label", tracePixels[traceIndex].label.c_str()},
				{"x", record.x}, {"y", record.y},
				{"mode", DVPVisibleNormalModeName(update.mode)},
				{"selected_source_views", update.selectedSourceViews},
				{"support_count", update.supportCount},
				{"direction_count", update.directionCount},
				{"current", {
					{"normal_reference", {record.currentNormal[0], record.currentNormal[1],
						record.currentNormal[2]}},
					{"valid", update.currentValid != 0u},
					{"feasible", update.currentFeasible != 0u},
					{"max_dot", update.currentMaxDot},
					{"max_violation", update.currentMaxViolation},
					{"reason_code", update.currentReason},
					{"reason", DVPVisibleNormalEvaluationReasonName(update.currentReason)},
					{"rejected_direction", update.currentRejectedDirection}
				}},
				{"propagation_decision", {
					{"native_best", update.propagationNativeBest},
					{"constrained_best", update.propagationConstrainedBest},
					{"selected", update.propagationSelected},
					{"reason_code", update.propagationReason},
					{"reason", DVPVisibleNormalPropagationReasonName(update.propagationReason)},
					{"fallback", update.propagationFallback != 0u},
					{"applied_constraint", update.propagationAppliedConstraint != 0u},
					{"accepted", update.propagationAccepted != 0u}
				}},
				{"propagation_candidates", propagation},
				{"refinement_proposals", refinement}
			};
			traces << line.dump() << '\n';
			if (!traces)
				return false;
		}
	}
	traces.flush();
	return (bool)traces;
}

bool SaveInstrumentUpdateMap(const String& dir, int imageID, int scaleNumber, const cv::Size& size, const std::vector<uint8_t>& updateSources)
{
	if (size.area() <= 0 || updateSources.size() < (size_t)size.area())
		return false;
	Image8U image(size);
	for (int i = 0; i < image.area(); ++i)
		image[i] = (uint8_t)MINF(255, updateSources[(size_t)i] * 32);
	return image.Save(dir + String::FormatString(_T("maps/depth%04u_scale%02d_update_source.png"), imageID, scaleNumber));
}

bool SaveInstrumentCostMap(const String& dir, int imageID, int scaleNumber, const ConfidenceMap& costMap)
{
	if (costMap.empty())
		return false;
	const String baseName(String::FormatString(_T("depth%04u_scale%02d_cost"), imageID, scaleNumber));
	const bool costSaved(costMap.Save(dir + _T("costs/") + baseName + _T(".pfm")));

	// CUDA PatchMatch uses 1.2 as the bad-cost sentinel. Keep high/bad costs
	// bright in the visual map so holes and weak photometric regions stand out.
	static constexpr float kBadCost = 1.2f;
	Image8U image(costMap.size());
	for (int i = 0; i < image.area(); ++i) {
		const float cost = costMap[i];
		image[i] = ISFINITE(cost) ?
			(uint8_t)CLAMP(cost * 255.f / kBadCost, 0.f, 255.f) :
			(uint8_t)255;
	}
	const bool previewSaved(image.Save(dir + _T("maps/") + baseName + _T(".png")));
	return costSaved && previewSaved;
}

bool SaveInstrumentImprovementMaps(
	const String& dir,
	int imageID,
	int scaleNumber,
	const cv::Size& size,
	int numPasses,
	const std::vector<float>& improvementMaps,
	const std::vector<uint8_t>& passUpdateSources)
{
	const size_t area((size_t)size.area());
	if (area == 0 || numPasses <= 0 ||
		improvementMaps.size() < area * (size_t)numPasses ||
		passUpdateSources.size() < area * (size_t)numPasses)
	{
		return false;
	}
	bool saved(true);
	for (int pass = 0; pass < numPasses; ++pass) {
		const String baseName(String::FormatString(_T("depth%04u_scale%02d_pass%02d_improvement"), imageID, scaleNumber, pass));
		ConfidenceMap improvementMap(size);
		float maxImprovement = 0.f;
		const size_t offset((size_t)pass * area);
		for (size_t i = 0; i < area; ++i) {
			const float value(improvementMaps[offset + i]);
			improvementMap[(int)i] = value;
			if (ISFINITE(value))
				maxImprovement = MAXF(maxImprovement, value);
		}
		saved = improvementMap.Save(dir + _T("improvements/") + baseName + _T(".pfm")) && saved;

		Image8U image(size);
		const float invScale(maxImprovement > 0.f ? 255.f / maxImprovement : 0.f);
		for (int i = 0; i < image.area(); ++i) {
			const float value(improvementMap[i]);
			image[i] = ISFINITE(value) ? (uint8_t)CLAMP(value * invScale, 0.f, 255.f) : (uint8_t)0;
		}
		saved = image.Save(dir + _T("maps/") + baseName + _T(".png")) && saved;
	}
	for (int pass = 0; pass < numPasses; ++pass) {
		Image8U image(size);
		const size_t offset((size_t)pass * area);
		for (int i = 0; i < image.area(); ++i)
			image[i] = (uint8_t)MINF(255, passUpdateSources[offset + (size_t)i] * 32);
		saved = image.Save(dir + String::FormatString(_T("maps/depth%04u_scale%02d_pass%02d_update_source.png"), imageID, scaleNumber, pass)) && saved;
	}
	return saved;
}

bool SaveInstrumentScalarMap(
	const String& fileName,
	const cv::Size& size,
	const std::vector<float>& values,
	size_t offset = 0)
{
	if (values.size() < offset + (size_t)size.area())
		return false;
	ConfidenceMap image(size);
	for (int i = 0; i < image.area(); ++i)
		image[i] = values[offset + (size_t)i];
	return image.Save(fileName);
}

bool SaveInstrumentFloat4Channel(
	const String& fileName,
	const cv::Size& size,
	const std::vector<float4>& values,
	int channel,
	size_t offset = 0)
{
	if (values.size() < offset + (size_t)size.area() || channel < 0 || channel > 3)
		return false;
	ConfidenceMap image(size);
	for (int i = 0; i < image.area(); ++i) {
		const float4& value(values[offset + (size_t)i]);
		image[i] = channel == 0 ? value.x : channel == 1 ? value.y : channel == 2 ? value.z : value.w;
	}
	return image.Save(fileName);
}

bool DMapRegularFileSize(const std::filesystem::path& path, uint64_t& bytes)
{
	std::error_code statusError;
	const std::filesystem::file_status status(
		std::filesystem::symlink_status(path, statusError));
	if (statusError || !std::filesystem::is_regular_file(status))
		return false;
	std::error_code sizeError;
	const uintmax_t size(std::filesystem::file_size(path, sizeError));
	if (sizeError || size == 0 || size > std::numeric_limits<uint64_t>::max())
		return false;
	bytes = (uint64_t)size;
	return true;
}

bool AddDMapManifestEntry(
	nlohmann::json& manifest,
	const String& depthMapDir,
	const char* signal,
	const String& relativePath,
	const char* dtype,
	const char* role,
	const char* semantics,
	const char* measurementQuality,
	const char* measurementBasis,
	const nlohmann::json& extra = nlohmann::json::object())
{
	const std::filesystem::path relative(relativePath.c_str());
	bool safeRelative(!relative.empty() && !relative.is_absolute());
	for (const std::filesystem::path& component : relative) {
		if (component == "..") {
			safeRelative = false;
			break;
		}
	}
	const std::filesystem::path artifactPath(
		std::filesystem::path(depthMapDir.c_str()) / relative);
	uint64_t fileSize(0);
	if (!safeRelative || !DMapRegularFileSize(artifactPath, fileSize))
	{
		manifest["write_errors"].push_back(std::string(signal) + "_file_unavailable_or_empty");
		return false;
	}
	nlohmann::json entry = {
		{"signal", signal},
		{"path", relativePath.c_str()},
		{"dtype", dtype},
		{"role", role},
		{"semantics", semantics},
		{"measurement_quality", measurementQuality},
		{"measurement_basis", measurementBasis},
		{"bytes", fileSize}
	};
	for (auto it = extra.begin(); it != extra.end(); ++it)
		entry[it.key()] = it.value();
	manifest["maps"].push_back(std::move(entry));
	return true;
}

float InstrumentPercentile(std::vector<float>& values, float percentile)
{
	if (values.empty())
		return 0.f;
	const size_t index((size_t)CLAMP(percentile * (float)(values.size()-1), 0.f, (float)(values.size()-1)));
	std::nth_element(values.begin(), values.begin()+index, values.end());
	return values[index];
}

void WriteExactObservabilityTables(
	const String& depthMapDir,
	const DepthData& depthData,
	const InstrumentExtendedMaps& maps,
	size_t area,
	nlohmann::json& manifest)
{
	if (!maps.exactAvailable || area == 0 || maps.numLogicalStates <= 0 || maps.numViews <= 0)
		return;
	const String iterationPath(depthMapDir + _T("exact_iteration.csv"));
	const String viewPath(depthMapDir + _T("exact_view_summary.csv"));
	std::ofstream iterationCSV(iterationPath.c_str(), std::ios::trunc);
	std::ofstream viewCSV(viewPath.c_str(), std::ios::trunc);
	if (!iterationCSV || !viewCSV) {
		manifest["write_errors"].push_back("exact_observability_tables");
		return;
	}
	iterationCSV << "logical_iteration,stage,pixels,tested_candidates,finite_candidates,accepted_candidates,gap_available_pixels,gap_mean,gap_p50,gap_p90,view_churn_pixels";
	for (int source = 0; source < PM_INSTRUMENT_NUM_SOURCES; ++source)
		iterationCSV << ",source_" << InstrumentSourceName(source);
	for (int slot = 0; slot < PM_INSTRUMENT_EXACT_NUM_CANDIDATES; ++slot)
		iterationCSV << ",winner_slot_" << slot;
	iterationCSV << '\n';
	viewCSV << "logical_iteration,stage,source_view_index,source_image_id,source_image_name,pixels,selected_pixels,finite_cost_pixels,probability_available_pixels,weight_mean,probability_mean,weighted_contribution_mean,photometric_cost_mean,geometric_cost_mean,total_cost_mean";
	for (int decision = PM_EXACT_VIEW_UNAVAILABLE; decision <= PM_EXACT_VIEW_INIT_REJECTED; ++decision)
		viewCSV << ",decision_" << decision;
	viewCSV << '\n';

	nlohmann::json summary = {
		{"schema_name", "openmvs.dmap.exact_observability"},
		{"schema_version", 2},
		{"num_logical_states", maps.numLogicalStates},
		{"num_views", maps.numViews},
		{"candidate_accepted_semantics", {
			{"initialization", ExactCandidateAcceptedSemantics(true)},
			{"iteration", ExactCandidateAcceptedSemantics(false)},
			{"cross_stage_comparison", "do not infer accepted<=finite during initialization; iterative accepted counts are sequential events, not final winners"}
		}},
		{"states", nlohmann::json::array()}
	};
	for (int stateIndex = 0; stateIndex < maps.numLogicalStates; ++stateIndex) {
		const int logicalIteration(stateIndex == 0 ? -1 : stateIndex-1);
		const char* stage(stateIndex == 0 ? "initialization" : "iteration");
		const size_t pixelOffset((size_t)stateIndex * area);
		uint64_t tested(0), finite(0), accepted(0), churnPixels(0);
		uint64_t sourceCounts[PM_INSTRUMENT_NUM_SOURCES] = {};
		uint64_t winnerCounts[PM_INSTRUMENT_EXACT_NUM_CANDIDATES] = {};
		uint64_t acceptedSlotCounts[PM_INSTRUMENT_EXACT_NUM_CANDIDATES] = {};
		std::vector<float> gaps;
		gaps.reserve(area);
		for (size_t pixel = 0; pixel < area; ++pixel) {
			const PatchMatchInstrumentExactPixel& record(maps.exactPixels[pixelOffset+pixel]);
			tested += record.testedCount;
			finite += record.finiteCount;
			accepted += record.acceptedCount;
			churnPixels += record.selectedViewsBefore != record.selectedViewsAfter;
			if (record.source < PM_INSTRUMENT_NUM_SOURCES)
				++sourceCounts[record.source];
			if (record.winnerSlot < PM_INSTRUMENT_EXACT_NUM_CANDIDATES)
				++winnerCounts[record.winnerSlot];
			for (int slot = 0; slot < PM_INSTRUMENT_EXACT_NUM_CANDIDATES; ++slot)
				acceptedSlotCounts[slot] += (record.candidateAcceptedMask >> slot) & 1u;
			if (ISFINITE(record.winnerRunnerUpGap) && record.winnerRunnerUpGap >= 0.f)
				gaps.push_back(record.winnerRunnerUpGap);
		}
		double gapSum(0.0);
		for (float gap : gaps)
			gapSum += gap;
		std::vector<float> gapCopy(gaps);
		const float gapP50(InstrumentPercentile(gapCopy, 0.5f));
		gapCopy = gaps;
		const float gapP90(InstrumentPercentile(gapCopy, 0.9f));
		iterationCSV << logicalIteration << ',' << stage << ',' << area << ',' << tested << ',' << finite << ',' << accepted << ','
			<< gaps.size() << ',' << (gaps.empty() ? 0.0 : gapSum/(double)gaps.size()) << ',' << gapP50 << ',' << gapP90 << ',' << churnPixels;
		for (uint64_t count : sourceCounts)
			iterationCSV << ',' << count;
		for (uint64_t count : winnerCounts)
			iterationCSV << ',' << count;
		iterationCSV << '\n';
			nlohmann::json state = {
				{"logical_iteration", logicalIteration}, {"stage", stage}, {"pixels", area},
				{"candidate_tested", tested}, {"candidate_finite", finite}, {"candidate_accepted", accepted},
				{"candidate_accepted_semantics", ExactCandidateAcceptedSemantics(stateIndex == 0)},
			{"gap_available_pixels", gaps.size()}, {"gap_mean", gaps.empty() ? 0.0 : gapSum/(double)gaps.size()},
			{"gap_p50", gapP50}, {"gap_p90", gapP90}, {"view_churn_pixels", churnPixels},
			{"source_counts", std::vector<uint64_t>(sourceCounts, sourceCounts+PM_INSTRUMENT_NUM_SOURCES)},
			{"winner_slot_counts", std::vector<uint64_t>(winnerCounts, winnerCounts+PM_INSTRUMENT_EXACT_NUM_CANDIDATES)},
			{"accepted_slot_counts", std::vector<uint64_t>(acceptedSlotCounts, acceptedSlotCounts+PM_INSTRUMENT_EXACT_NUM_CANDIDATES)},
			{"views", nlohmann::json::array()}
		};
		for (int view = 0; view < maps.numViews; ++view) {
			const DepthData::ViewData& sourceView(depthData.images[(IIndex)view+1]);
			uint64_t selectedPixels(0), finitePixels(0), probabilityPixels(0), decisionCounts[PM_EXACT_VIEW_INIT_REJECTED+1] = {};
			double weightSum(0.0), probabilitySum(0.0), contributionSum(0.0), photoSum(0.0), geometricSum(0.0), totalSum(0.0);
			for (size_t pixel = 0; pixel < area; ++pixel) {
				const size_t index((pixelOffset+pixel) * (size_t)maps.numViews + (size_t)view);
				const PatchMatchInstrumentExactView& record(maps.exactViews[index]);
				const uint32_t metadata(record.metadata);
				const unsigned weight((metadata >> PM_EXACT_VIEW_WEIGHT_SHIFT) & PM_EXACT_VIEW_WEIGHT_MASK);
				const unsigned decision((metadata >> PM_EXACT_VIEW_DECISION_SHIFT) & PM_EXACT_VIEW_DECISION_MASK);
				weightSum += weight;
				selectedPixels += (metadata & PM_EXACT_VIEW_SELECTED_BIT) != 0;
				finitePixels += (metadata & PM_EXACT_VIEW_FINITE_BIT) != 0;
				if (decision <= PM_EXACT_VIEW_INIT_REJECTED)
					++decisionCounts[decision];
				if (metadata & PM_EXACT_VIEW_PROBABILITY_BIT) {
					++probabilityPixels;
					probabilitySum += record.samplingProbability;
				}
				contributionSum += record.weightedContribution;
				if (metadata & PM_EXACT_VIEW_FINITE_BIT) {
					photoSum += record.photometricCost;
					geometricSum += record.geometricCost;
					totalSum += record.totalCost;
				}
			}
			const double invPixels(area ? 1.0/(double)area : 0.0);
			const double invFinite(finitePixels ? 1.0/(double)finitePixels : 0.0);
			const double probabilityMean(probabilityPixels ? probabilitySum/(double)probabilityPixels : 0.0);
			viewCSV << logicalIteration << ',' << stage << ',' << view << ',' << sourceView.GetID() << ','
				<< CsvEscape(sourceView.pImageData ? sourceView.pImageData->name : String()) << ',' << area << ',' << selectedPixels << ','
				<< finitePixels << ',' << probabilityPixels << ',' << weightSum*invPixels << ',' << probabilityMean << ','
				<< contributionSum*invPixels << ',' << photoSum*invFinite << ',' << geometricSum*invFinite << ',' << totalSum*invFinite;
			for (uint64_t count : decisionCounts)
				viewCSV << ',' << count;
			viewCSV << '\n';
			state["views"].push_back({
				{"source_view_index", view}, {"source_image_id", sourceView.GetID()},
				{"source_image_name", sourceView.pImageData ? sourceView.pImageData->name.c_str() : ""},
				{"selected_pixels", selectedPixels}, {"finite_cost_pixels", finitePixels},
				{"probability_available_pixels", probabilityPixels}, {"weight_mean", weightSum*invPixels},
				{"probability_mean", probabilityMean}, {"weighted_contribution_mean", contributionSum*invPixels},
				{"photometric_cost_mean", photoSum*invFinite}, {"geometric_cost_mean", geometricSum*invFinite},
				{"total_cost_mean", totalSum*invFinite},
				{"decision_counts", std::vector<uint64_t>(decisionCounts, decisionCounts+PM_EXACT_VIEW_INIT_REJECTED+1)}
			});
		}
		summary["states"].push_back(std::move(state));
	}
	iterationCSV.flush();
	viewCSV.flush();
	const bool iterationOK((bool)iterationCSV);
	const bool viewOK((bool)viewCSV);
	iterationCSV.close();
	viewCSV.close();
	const String summaryPath(depthMapDir + _T("exact_observability.json"));
	const bool summaryOK(WriteJsonFile(summaryPath, summary));
	const size_f_t iterationBytes(File::getSize(iterationPath));
	const size_f_t viewBytes(File::getSize(viewPath));
	const size_f_t summaryBytes(File::getSize(summaryPath));
	if (!iterationOK || !viewOK || !summaryOK ||
		iterationBytes == SIZE_NA || iterationBytes == 0 ||
		viewBytes == SIZE_NA || viewBytes == 0 ||
		summaryBytes == SIZE_NA || summaryBytes == 0)
	{
		manifest["write_errors"].push_back("exact_observability_tables");
		return;
	}
	manifest["tables"] = nlohmann::json::array({
		{{"schema_name", "openmvs.dmap.exact_iteration"}, {"schema_version", 2}, {"path", "exact_iteration.csv"}, {"bytes", (uint64_t)iterationBytes},
			{"candidate_accepted_semantics", {{"initialization", ExactCandidateAcceptedSemantics(true)}, {"iteration", ExactCandidateAcceptedSemantics(false)}}}},
		{{"schema_name", "openmvs.dmap.exact_view_summary"}, {"schema_version", 1}, {"path", "exact_view_summary.csv"}, {"bytes", (uint64_t)viewBytes}},
		{{"schema_name", "openmvs.dmap.exact_observability"}, {"schema_version", 2}, {"path", "exact_observability.json"}, {"bytes", (uint64_t)summaryBytes}}
	});
}

void SaveDMapExtendedMaps(
	const String& depthMapDir,
	const DepthData& depthData,
	int numPasses,
	const std::vector<float>& passCostImprovements,
	const InstrumentExtendedMaps& maps,
	nlohmann::json& manifest)
{
	const cv::Size size(depthData.depthMap.size());
	const size_t area((size_t)size.area());
	if (area == 0)
		return;
	const String mapsDir(depthMapDir + _T("maps/"));
	const String logicalDir(depthMapDir + _T("logical_states/"));
	Util::ensureFolder(mapsDir);
	Util::ensureFolder(logicalDir);
	auto addMap = [&](const char* signal, const String& relativePath, const char* dtype, const char* role,
		const char* semantics, const char* quality, const char* basis, const nlohmann::json& extra = nlohmann::json::object()) {
		AddDMapManifestEntry(manifest, depthMapDir, signal, relativePath, dtype, role, semantics, quality, basis, extra);
	};
	auto saveScalar = [&](const char* signal, const char* fileName, const std::vector<float>& values, const char* semantics,
		const char* quality, const char* basis, size_t offset = 0) {
		if (SaveInstrumentScalarMap(mapsDir + fileName, size, values, offset))
			addMap(signal, String(_T("maps/")) + fileName, "float32", "final_state", semantics, quality, basis);
		else
			manifest["write_errors"].push_back(signal);
	};
	auto saveFloat4 = [&](const char* signal, const char* fileName, const std::vector<float4>& values, int channel,
		const char* semantics, const char* quality, const char* basis, size_t offset = 0) {
		if (SaveInstrumentFloat4Channel(mapsDir + fileName, size, values, channel, offset))
			addMap(signal, String(_T("maps/")) + fileName, "float32", "final_state", semantics, quality, basis);
		else
			manifest["write_errors"].push_back(signal);
	};
	if (maps.apdRequested && maps.apdMultiscale.is_object() && !maps.apdMultiscale.empty()) {
		manifest["apd_multiscale"] = maps.apdMultiscale;
		const String relativeMultiscaleDir(_T("maps/apd_multiscale/"));
		const String multiscaleDir(depthMapDir + relativeMultiscaleDir);
		Util::ensureFolder(multiscaleDir);
		const bool transferred(
			maps.apdMultiscale.value("input_state", nlohmann::json::object()).value("available", false));
		const bool conventional(
			maps.apdMultiscale.value("schedule", nlohmann::json::object()).value("policy", "") ==
			"conventional_native");
		auto saveByteMap = [&](const char* signal, const char* fileName,
			const std::vector<uint8_t>& values, bool required, const char* role,
			const char* semantics, const char* quality, const char* basis) {
			if (values.size() < area) {
				if (required)
					manifest["write_errors"].push_back(signal);
				return;
			}
			Image8U image(size);
			for (int i=0; i<image.area(); ++i)
				image[i] = values[(size_t)i];
			const String relativePath(relativeMultiscaleDir + fileName);
			if (image.Save(depthMapDir + relativePath))
				addMap(signal, relativePath, "uint8", role, semantics, quality, basis,
					{{"lossless", true}, {"state_schema_version", APD_MULTISCALE_STATE_VERSION}});
			else
				manifest["write_errors"].push_back(signal);
		};
		saveByteMap("apd_transferred_reliability", "transferred_reliability.png",
			maps.apdTransferredReliability, transferred, "multiscale_input",
			"nearest-neighbor resized reliability class consumed at logical iteration zero",
			"exact", "runtime_host_transfer_state");
		saveByteMap("apd_transferred_anchor_count", "transferred_anchor_count.png",
			maps.apdTransferredAnchorCounts, transferred, "multiscale_input_provenance",
			"nearest-neighbor resized prior-stage anchor count retained for provenance; not directly consumed",
			"exact", "runtime_host_transfer_state");
		saveByteMap("apd_transferred_deformable_eligible", "transferred_deformable_eligible.png",
			maps.apdTransferredDeformableEligible, transferred, "multiscale_input_provenance",
			"nearest-neighbor resized prior-stage deformable eligibility retained for provenance; not directly consumed",
			"exact", "runtime_host_transfer_state");
		const char* outputQuality(conventional ? "derived_exact" : "exact");
		const char* outputBasis(conventional ?
			"native_selected_view_mask_uniform_compatibility_classifier" :
			"apd_runtime_profile_and_anchor_state");
		saveByteMap("apd_output_reliability", "output_reliability.png",
			maps.apdOutputReliability, true, "multiscale_output",
			"post-filter reliability class published to the next APD stage",
			outputQuality, outputBasis);
		saveByteMap("apd_output_anchor_count", "output_anchor_count.png",
			maps.apdOutputAnchorCounts, true, "multiscale_output_provenance",
			"post-filter reliable-anchor count published as next-stage provenance",
			outputQuality, outputBasis);
		saveByteMap("apd_output_deformable_eligible", "output_deformable_eligible.png",
			maps.apdOutputDeformableEligible, true, "multiscale_output_provenance",
			"post-filter fitted-plane eligibility published as next-stage provenance",
			outputQuality, outputBasis);
	}

	if (maps.planesBeforeFilter.size() >= area) {
		DepthMap depth(size);
		NormalMap normal(size);
		for (int i = 0; i < depth.area(); ++i) {
			const Point4& plane(maps.planesBeforeFilter[(size_t)i]);
			depth[i] = plane.w();
			normal[i] = plane.topLeftCorner<3,1>();
		}
		if (depth.Save(mapsDir + _T("depth_final_before_filter.pfm")))
			addMap("depth_final_before_filter", _T("maps/depth_final_before_filter.pfm"), "float32", "final_state", "exact production depth immediately before filtering", "exact", "production_pre_filter_snapshot");
		else
			manifest["write_errors"].push_back("depth_final_before_filter");
		if (normal.Save(mapsDir + _T("normal_final_before_filter.pfm")))
			addMap("normal_final_before_filter", _T("maps/normal_final_before_filter.pfm"), "float32x3", "final_state", "estimated normal immediately before filtering", "exact", "production_pre_filter_snapshot");
		else
			manifest["write_errors"].push_back("normal_final_before_filter");
	} else {
		manifest["write_errors"].push_back("depth_final_before_filter");
		manifest["write_errors"].push_back("normal_final_before_filter");
	}
	const size_t finalLogicalOffset(maps.numLogicalStates > 0 ? (size_t)(maps.numLogicalStates-1) * area : 0);
	saveScalar("cost_final_before_filter", "cost_final_before_filter.pfm", maps.costsBeforeFilter, "aggregate PatchMatch cost immediately before filtering; lower is better", "exact", "production_pre_filter_snapshot");
	saveFloat4("cost_photometric", "cost_photometric.pfm", maps.logicalScorePrimary, 0, "legacy alias: equal-selected-view post-pass raw photometric rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("cost_photo_prior", "cost_photo_prior.pfm", maps.logicalScorePrimary, 1, "legacy alias: equal-selected-view post-pass prior-blended photometric rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("cost_geometric", "cost_geometric.pfm", maps.logicalScorePrimary, 2, "legacy alias: equal-selected-view post-pass geometric rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("cost_total_components", "cost_total_components.pfm", maps.logicalScorePrimary, 3, "legacy alias: total equal-selected-view post-pass rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("cost_depth_prior", "cost_depth_prior.pfm", maps.logicalScoreSecondary, 0, "legacy alias: low-resolution depth-prior disagreement used by the proxy rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("depth_prior_weight", "depth_prior_weight.pfm", maps.logicalScoreSecondary, 1, "legacy alias: low-texture depth-prior blend weight used by the proxy rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("confidence_gap", "confidence_gap.pfm", maps.logicalScoreSecondary, 2, "legacy alias: local-neighbor rescore gap; not the production winner-versus-runner-up gap", "proxy", "equal_selected_view_binary_post_pass_rescore", finalLogicalOffset);
	saveFloat4("reference_variance", "reference_variance.pfm", maps.logicalScoreSecondary, 3, "weighted reference-patch variance", "exact", "post_pass_reference_patch", finalLogicalOffset);
	saveScalar("view_entropy", "view_entropy.pfm", maps.finalViewEntropy, "legacy selected-view membership entropy proxy; binary membership cannot recover production reliability entropy", "proxy", "selected_view_membership");
	saveScalar("low_depth_prior", "low_depth_prior.pfm", maps.finalLowDepth, "low-resolution depth prior supplied to PatchMatch", "exact", "production_input");

	for (int view = 0; view < PM_INSTRUMENT_MAP_VIEWS; ++view) {
		const String suffix(String::FormatString(_T("_%u.pfm"), view));
		const String weightSignal(String::FormatString(_T("view_weight_%u"), view));
		if (SaveInstrumentFloat4Channel(mapsDir + _T("view_weight") + suffix, size, maps.finalViewWeights, view))
			addMap(weightSignal.c_str(), String(_T("maps/view_weight")) + suffix, "float32", "final_state", "final selected-view membership indicator; not the production Monte Carlo reliability weight", "proxy", "selected_view_membership");
		else
			manifest["write_errors"].push_back(weightSignal.c_str());
		const String costSignal(String::FormatString(_T("view_cost_%u"), view));
		if (SaveInstrumentFloat4Channel(mapsDir + _T("view_cost") + suffix, size, maps.finalViewCosts, view))
			addMap(costSignal.c_str(), String(_T("maps/view_cost")) + suffix, "float32", "final_state", "final per-view equal-selected-view total rescore", "proxy", "equal_selected_view_binary_post_pass_rescore");
		else
			manifest["write_errors"].push_back(costSignal.c_str());
		const String photometricSignal(String::FormatString(_T("view_photometric_cost_%u"), view));
		if (SaveInstrumentFloat4Channel(mapsDir + _T("view_photometric_cost") + suffix, size, maps.finalViewPhotometricCosts, view))
			addMap(photometricSignal.c_str(), String(_T("maps/view_photometric_cost")) + suffix, "float32", "final_state", "final per-view photometric rescore", "proxy", "equal_selected_view_binary_post_pass_rescore");
		else
			manifest["write_errors"].push_back(photometricSignal.c_str());
		const String geometricSignal(String::FormatString(_T("view_geometric_cost_%u"), view));
		if (SaveInstrumentFloat4Channel(mapsDir + _T("view_geometric_cost") + suffix, size, maps.finalViewGeometricCosts, view))
			addMap(geometricSignal.c_str(), String(_T("maps/view_geometric_cost")) + suffix, "float32", "final_state", "final per-view geometric rescore", "proxy", "equal_selected_view_binary_post_pass_rescore");
		else
			manifest["write_errors"].push_back(geometricSignal.c_str());
	}

	if (maps.finalSelectedViews.size() >= area) {
		Image8U selectedCount(size);
		for (int i = 0; i < selectedCount.area(); ++i) {
			uint32_t value(maps.finalSelectedViews[(size_t)i]);
			uint8_t count(0);
			while (value) {
				count += (uint8_t)(value & 1u);
				value >>= 1;
			}
			selectedCount[i] = count;
		}
		if (selectedCount.Save(mapsDir + _T("selected_view_count.png")))
			addMap("selected_view_count", _T("maps/selected_view_count.png"), "uint8", "final_state", "number of selected source views in the final state", "exact", "production_selected_view_mask");
		else
			manifest["write_errors"].push_back("selected_view_count");
	} else
		manifest["write_errors"].push_back("selected_view_count");
	if (maps.acceptedUpdateCount.size() >= area) {
		Image8U updateCount(size);
		for (int i = 0; i < updateCount.area(); ++i)
			updateCount[i] = maps.acceptedUpdateCount[(size_t)i];
		if (updateCount.Save(mapsDir + _T("accepted_update_count.png")))
			addMap("accepted_update_count", _T("maps/accepted_update_count.png"), "uint8", "final_state", "number of diagnostic pass snapshots in which this pixel changed; exact source attribution unavailable", "proxy", "post_pass_change_detection");
		else
			manifest["write_errors"].push_back("accepted_update_count");
	} else
		manifest["write_errors"].push_back("accepted_update_count");

	const bool logicalMapsComplete(
		maps.numLogicalStates > 0 &&
		maps.logicalStoredCosts.size() >= area * (size_t)maps.numLogicalStates &&
		maps.logicalScorePrimary.size() >= area * (size_t)maps.numLogicalStates &&
		maps.logicalScoreSecondary.size() >= area * (size_t)maps.numLogicalStates);
	if (logicalMapsComplete) {
		for (int stateIndex = 0; stateIndex < maps.numLogicalStates; ++stateIndex) {
			const int logicalIteration(stateIndex == 0 ? -1 : stateIndex-1);
			const String stateName(stateIndex == 0 ? _T("state00_initialization") :
				String::FormatString(_T("state%02d_iteration%02d"), stateIndex, stateIndex));
			const String relativeStateDir(String(_T("logical_states/")) + stateName + _T("/"));
			const String stateDir(depthMapDir + relativeStateDir);
			Util::ensureFolder(stateDir);
			const size_t offset((size_t)stateIndex * area);
			const nlohmann::json stateMetadata = {
				{"logical_iteration", logicalIteration},
				{"stage", stateIndex == 0 ? "initialization" : "iteration"},
				{"stage_index", stateIndex},
				{"candidate_accepted_semantics", ExactCandidateAcceptedSemantics(stateIndex == 0)},
				{"uncompressed_bytes", (uint64_t)(area * sizeof(float))}
			};
			auto saveLogicalScalar = [&](const char* signal, const char* fileName, const std::vector<float>& values,
				const char* semantics, const char* quality, const char* basis, const nlohmann::json& extra = nlohmann::json::object()) {
				nlohmann::json metadata(stateMetadata);
				for (auto it = extra.begin(); it != extra.end(); ++it)
					metadata[it.key()] = it.value();
				if (SaveInstrumentScalarMap(stateDir + fileName, size, values, offset))
					addMap(signal, relativeStateDir + fileName, "float32", "logical_state", semantics, quality, basis, metadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			auto saveLogicalFloat4 = [&](const char* signal, const char* fileName, const std::vector<float4>& values, int channel,
				const char* semantics, const char* quality, const char* basis, const nlohmann::json& extra = nlohmann::json::object()) {
				nlohmann::json metadata(stateMetadata);
				for (auto it = extra.begin(); it != extra.end(); ++it)
					metadata[it.key()] = it.value();
				if (SaveInstrumentFloat4Channel(stateDir + fileName, size, values, channel, offset))
					addMap(signal, relativeStateDir + fileName, "float32", "logical_state", semantics, quality, basis, metadata);
				else
					manifest["write_errors"].push_back(signal);
			};

			saveLogicalScalar("cost_stored", "cost_stored.pfm", maps.logicalStoredCosts,
				"exact aggregate cost retained by production PatchMatch at this logical state; lower is better",
				"exact", "production_cost_snapshot", {{"valid_domain", "finite cost"}, {"lower_is_better", true}});
			std::vector<float> confidence(area);
			std::vector<float> residual(area);
			for (size_t i = 0; i < area; ++i) {
				const float stored(maps.logicalStoredCosts[offset+i]);
				const float total(maps.logicalScorePrimary[offset+i].w);
				confidence[i] = stored >= 1.f ? 0.f : 1.f-stored;
				residual[i] = stored-total;
			}
			nlohmann::json confidenceMetadata(stateMetadata);
			confidenceMetadata["valid_min"] = 0.0;
			confidenceMetadata["valid_max"] = 1.0;
			if (SaveInstrumentScalarMap(stateDir + _T("confidence_stored.pfm"), size, confidence))
				addMap("confidence_stored", relativeStateDir + _T("confidence_stored.pfm"), "float32", "logical_state",
					"production confidence derived exactly as max(1-cost_stored, 0)", "derived_exact", "max(1-cost,0)",
					confidenceMetadata);
			else
				manifest["write_errors"].push_back("confidence_stored");
			const nlohmann::json proxyMetadata = {
				{"proxy_target", "production_cost_component_decomposition"},
				{"limitations", "Production uses 32-sample Monte Carlo reliability counts; this rescore gives every selected view unit weight."}
			};
			saveLogicalFloat4("cost_photo_raw_equal_selected_rescore_proxy", "cost_photo_raw_equal_selected_rescore_proxy.pfm", maps.logicalScorePrimary, 0,
				"raw photometric component recomputed for the retained plane with equal selected-view weights", "proxy", "equal_selected_view_binary_post_pass_rescore", proxyMetadata);
			saveLogicalFloat4("cost_photo_prior_equal_selected_rescore_proxy", "cost_photo_prior_equal_selected_rescore_proxy.pfm", maps.logicalScorePrimary, 1,
				"prior-blended photometric component recomputed with equal selected-view weights", "proxy", "equal_selected_view_binary_post_pass_rescore", proxyMetadata);
			saveLogicalFloat4("cost_geometric_equal_selected_rescore_proxy", "cost_geometric_equal_selected_rescore_proxy.pfm", maps.logicalScorePrimary, 2,
				"geometric component recomputed with equal selected-view weights", "proxy", "equal_selected_view_binary_post_pass_rescore", proxyMetadata);
			saveLogicalFloat4("cost_total_equal_selected_rescore_proxy", "cost_total_equal_selected_rescore_proxy.pfm", maps.logicalScorePrimary, 3,
				"total retained-plane score reconstructed from equal-selected-view components", "proxy", "equal_selected_view_binary_post_pass_rescore", proxyMetadata);
			if (SaveInstrumentScalarMap(stateDir + _T("cost_stored_minus_rescore.pfm"), size, residual))
				addMap("cost_stored_minus_rescore", relativeStateDir + _T("cost_stored_minus_rescore.pfm"), "float32", "logical_state",
					"exact arithmetic difference cost_stored - cost_total_equal_selected_rescore_proxy", "derived_exact",
					"production_snapshot_minus_equal_selected_view_rescore", stateMetadata);
			else
				manifest["write_errors"].push_back("cost_stored_minus_rescore");
			saveLogicalFloat4("depth_prior_disagreement_equal_selected_rescore_proxy", "depth_prior_disagreement_equal_selected_rescore_proxy.pfm", maps.logicalScoreSecondary, 0,
				"raw low-resolution depth-prior disagreement in the proxy rescore", "proxy", "equal_selected_view_binary_post_pass_rescore", proxyMetadata);
			nlohmann::json priorWeightMetadata(proxyMetadata);
			priorWeightMetadata["valid_min"] = 0.0;
			priorWeightMetadata["valid_max"] = 1.0;
			saveLogicalFloat4("depth_prior_weight_equal_selected_rescore_proxy", "depth_prior_weight_equal_selected_rescore_proxy.pfm", maps.logicalScoreSecondary, 1,
				"low-texture depth-prior blend weight in the proxy rescore", "proxy", "equal_selected_view_binary_post_pass_rescore",
				priorWeightMetadata);
			saveLogicalFloat4("gap_local_neighbor_equal_selected_rescore_proxy", "gap_local_neighbor_equal_selected_rescore_proxy.pfm", maps.logicalScoreSecondary, 2,
				"second-best minus best equal-selected-view score across the retained plane and eight immediate neighbor planes; -1 means unavailable",
				"proxy", "equal_selected_view_binary_post_pass_rescore",
				{{"proxy_target", "exact_same_pass_winner_runner_up_gap"}, {"limitations", "Candidate set and reliability weights differ from the production update."}, {"unavailable_value", -1.0}});
			saveLogicalFloat4("reference_variance_equal_selected_rescore_proxy", "reference_variance_equal_selected_rescore_proxy.pfm", maps.logicalScoreSecondary, 3,
				"weighted reference-patch variance recomputed by the post-pass diagnostic kernel", "proxy", "equal_selected_view_binary_post_pass_rescore",
				{{"proxy_target", "production_reference_patch_variance"}, {"limitations", "Captured in the post-pass diagnostic kernel rather than the production candidate kernel."}});
		}
	} else
		manifest["write_errors"].push_back("logical_state_capture_incomplete");

	const size_t exactStateArea(area * (size_t)maps.numLogicalStates);
	const bool exactMapsComplete(
		maps.exactAvailable && maps.numLogicalStates > 0 && maps.numViews > 0 &&
		maps.exactLogicalScorePrimary.size() >= exactStateArea &&
		maps.exactLogicalScoreSecondary.size() >= exactStateArea &&
		maps.exactPixels.size() >= exactStateArea &&
		maps.exactViews.size() >= exactStateArea * (size_t)maps.numViews);
	if (maps.exactAvailable && !exactMapsComplete)
		manifest["write_errors"].push_back("exact_capture_incomplete");
	if (exactMapsComplete) {
		for (int stateIndex = 0; stateIndex < maps.numLogicalStates; ++stateIndex) {
			const int logicalIteration(stateIndex == 0 ? -1 : stateIndex-1);
			const String stateName(stateIndex == 0 ? _T("state00_initialization") :
				String::FormatString(_T("state%02d_iteration%02d"), stateIndex, stateIndex));
			const String relativeStateDir(String(_T("logical_states/")) + stateName + _T("/"));
			const String stateDir(depthMapDir + relativeStateDir);
			Util::ensureFolder(stateDir);
			const size_t offset((size_t)stateIndex * area);
			const nlohmann::json stateMetadata = {
				{"logical_iteration", logicalIteration},
				{"stage", stateIndex == 0 ? "initialization" : "iteration"},
				{"stage_index", stateIndex},
				{"uncompressed_bytes", (uint64_t)(area * sizeof(float))}
			};
			auto saveExactScore = [&](const char* signal, const char* fileName,
				const std::vector<float4>& values, int channel, const char* semantics,
				const nlohmann::json& extra = nlohmann::json::object()) {
				nlohmann::json metadata(stateMetadata);
				for (auto it = extra.begin(); it != extra.end(); ++it)
					metadata[it.key()] = it.value();
				if (SaveInstrumentFloat4Channel(stateDir + fileName, size, values, channel, offset))
					addMap(signal, relativeStateDir + fileName, "float32", "logical_state", semantics,
						"exact", "production_hot_kernel_contribution_basis", metadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			saveExactScore("cost_photo_raw_production_exact", "cost_photo_raw_production_exact.pfm",
				maps.exactLogicalScorePrimary, 0,
				"prior-free photometric cost of the retained winner on the exact production contribution basis");
			saveExactScore("cost_photo_prior_production_exact", "cost_photo_prior_production_exact.pfm",
				maps.exactLogicalScorePrimary, 1,
				"photometric cost after the production low-texture depth-prior blend on the exact contribution basis");
			saveExactScore("cost_geometric_production_exact", "cost_geometric_production_exact.pfm",
				maps.exactLogicalScorePrimary, 2,
				"production-weighted geometric consistency contribution of the retained winner");
			saveExactScore("cost_total_production_exact", "cost_total_production_exact.pfm",
				maps.exactLogicalScorePrimary, 3,
				"total retained-winner cost on the production contribution basis",
				{{"lower_is_better", true}});
			saveExactScore("depth_prior_disagreement_production_exact", "depth_prior_disagreement_production_exact.pfm",
				maps.exactLogicalScoreSecondary, 0,
				"low-resolution depth-prior disagreement evaluated in the production candidate kernel");
			saveExactScore("depth_prior_weight_production_exact", "depth_prior_weight_production_exact.pfm",
				maps.exactLogicalScoreSecondary, 1,
				"low-texture depth-prior blend weight evaluated in the production candidate kernel",
				{{"valid_min", 0.0}, {"valid_max", 1.0}});
			saveExactScore("gap_winner_runner_up_exact", "gap_winner_runner_up_exact.pfm",
				maps.exactLogicalScoreSecondary, 2,
				"second-lowest minus lowest finite candidate cost in the production ProcessPixel invocation; -1 means unavailable",
				{{"unavailable_value", -1.0}, {"gap_scope", "one complete logical update invocation per pixel"}});
			saveExactScore("reference_variance_production_exact", "reference_variance_production_exact.pfm",
				maps.exactLogicalScoreSecondary, 3,
				"weighted reference-patch variance evaluated in the production candidate kernel");

			auto saveExactPixelScalar = [&](const char* signal, const char* fileName, auto value,
				const char* semantics, const char* dtype = "float32", const nlohmann::json& extra = nlohmann::json::object()) {
				ConfidenceMap image(size);
				for (int i = 0; i < image.area(); ++i)
					image[i] = value(maps.exactPixels[offset + (size_t)i]);
				nlohmann::json metadata(stateMetadata);
				for (auto it = extra.begin(); it != extra.end(); ++it)
					metadata[it.key()] = it.value();
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, dtype, "logical_event", semantics,
						"exact", "production_hot_kernel_candidate_record", metadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			saveExactPixelScalar("candidate_stored_cost_before_exact", "candidate_stored_cost_before_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return value.storedCostBefore; },
				"stored production cost on entry to the logical update; -1 for initialization",
				"float32", {{"unavailable_value", -1.0}});
			saveExactPixelScalar("candidate_incumbent_cost_exact", "candidate_incumbent_cost_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return value.incumbentCost; },
				"current-plane cost after production view selection and before propagation/refinement comparisons");
			saveExactPixelScalar("candidate_winner_cost_exact", "candidate_winner_cost_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return value.winnerCost; },
				"cost of the retained winner after all candidates in the logical update");
			saveExactPixelScalar("candidate_runner_up_cost_exact", "candidate_runner_up_cost_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return value.runnerUpCost; },
				"second-lowest finite candidate cost; -1 when fewer than two finite candidates exist",
				"float32", {{"unavailable_value", -1.0}});
			saveExactPixelScalar("candidate_tested_mask_exact", "candidate_tested_mask_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return (float)value.candidateTestedMask; },
				"bit mask of candidate slots tested by the production logical update",
				"float32", {{"integer_encoding", "exact_uint13_in_float32"}});
			saveExactPixelScalar("candidate_finite_mask_exact", "candidate_finite_mask_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return (float)value.candidateFiniteMask; },
				"bit mask of candidate slots with finite production costs",
				"float32", {{"integer_encoding", "exact_uint13_in_float32"}});
			saveExactPixelScalar("candidate_accepted_mask_exact", "candidate_accepted_mask_exact.pfm",
				[](const PatchMatchInstrumentExactPixel& value) { return (float)value.candidateAcceptedMask; },
				stateIndex == 0 ?
					"bit mask of the production initialization stored assignment; this is not constrained by the finite-candidate count" :
					"bit mask of candidates accepted while the production incumbent evolved sequentially",
				"float32", {{"integer_encoding", "exact_uint13_in_float32"}});

			auto saveExactPixelRGB = [&](const char* signal, const char* fileName, auto red, auto green, auto blue,
				const char* semantics, const nlohmann::json& channels) {
				Image8U3 image(size);
				for (int i = 0; i < image.area(); ++i) {
					const PatchMatchInstrumentExactPixel& value(maps.exactPixels[offset + (size_t)i]);
					Pixel8U& pixel(image[i]);
					pixel.r = red(value);
					pixel.g = green(value);
					pixel.b = blue(value);
				}
				nlohmann::json metadata(stateMetadata);
				metadata["uncompressed_bytes"] = (uint64_t)(area * 3u);
				metadata["channels"] = channels;
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, "uint8x3", "logical_event", semantics,
						"exact", "production_hot_kernel_candidate_record", metadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			saveExactPixelRGB("candidate_counts_exact", "candidate_counts_exact.png",
				[](const PatchMatchInstrumentExactPixel& value) { return value.testedCount; },
				[](const PatchMatchInstrumentExactPixel& value) { return value.finiteCount; },
				[](const PatchMatchInstrumentExactPixel& value) { return value.acceptedCount; },
				stateIndex == 0 ?
					"RGB channels store tested, finite, and production stored-assignment counts for initialization" :
					"RGB channels store tested, finite, and sequential incumbent-improvement counts",
				{{"R", "tested_count"}, {"G", "finite_count"}, {"B", "accepted_count"}});
			saveExactPixelRGB("candidate_identity_exact", "candidate_identity_exact.png",
				[](const PatchMatchInstrumentExactPixel& value) { return value.winnerSlot; },
				[](const PatchMatchInstrumentExactPixel& value) { return value.runnerUpSlot; },
				[](const PatchMatchInstrumentExactPixel& value) { return value.source; },
				"RGB channels store winner slot, runner-up slot, and final update source enum",
				{{"R", "winner_slot"}, {"G", "runner_up_slot"}, {"B", "update_source"}});
			saveExactPixelRGB("selected_view_counts_exact", "selected_view_counts_exact.png",
				[](const PatchMatchInstrumentExactPixel& value) { return value.selectedCountBefore; },
				[](const PatchMatchInstrumentExactPixel& value) { return value.selectedCountAfter; },
				[](const PatchMatchInstrumentExactPixel& value) { return (uint8_t)__builtin_popcount(value.selectedViewsBefore ^ value.selectedViewsAfter); },
				"RGB channels store selected-view counts before/after and exact symmetric-difference count",
				{{"R", "selected_count_before"}, {"G", "selected_count_after"}, {"B", "view_churn"}});

			auto saveExactMaskRGBA = [&](const char* signal, const char* fileName, auto value, const char* semantics) {
				Image8U4 image(size);
				for (int i = 0; i < image.area(); ++i) {
					const uint32_t mask(value(maps.exactPixels[offset + (size_t)i]));
					Color8U& pixel(image[i]);
					pixel.r = (uint8_t)(mask & 0xFFu);
					pixel.g = (uint8_t)((mask >> 8) & 0xFFu);
					pixel.b = (uint8_t)((mask >> 16) & 0xFFu);
					pixel.a = (uint8_t)((mask >> 24) & 0xFFu);
				}
				nlohmann::json metadata(stateMetadata);
				metadata["uncompressed_bytes"] = (uint64_t)(area * 4u);
				metadata["encoding"] = "uint32 little-endian bytes in RGBA channels";
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, "uint8x4", "logical_event", semantics,
						"exact", "production_selected_view_mask", metadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			saveExactMaskRGBA("selected_views_before_mask_exact", "selected_views_before_mask_exact.png",
				[](const PatchMatchInstrumentExactPixel& value) { return value.selectedViewsBefore; },
				"lossless 32-bit selected-source-view mask on entry to the logical update");
			saveExactMaskRGBA("selected_views_after_mask_exact", "selected_views_after_mask_exact.png",
				[](const PatchMatchInstrumentExactPixel& value) { return value.selectedViewsAfter; },
				"lossless 32-bit selected-source-view mask after the logical update");

			for (int view = 0; view < maps.numViews; ++view) {
				const DepthData::ViewData& sourceView(depthData.images[(IIndex)view+1]);
				const int sourceID((int)sourceView.GetID());
				const String suffix(String::FormatString(_T("_view%02d_id%04d"), view, sourceID));
				const size_t viewOffset(offset * (size_t)maps.numViews + (size_t)view);
				nlohmann::json viewMetadata(stateMetadata);
				viewMetadata["source_view_index"] = view;
				viewMetadata["source_image_id"] = sourceID;
				viewMetadata["source_image_name"] = sourceView.pImageData ? sourceView.pImageData->name.c_str() : "";
				viewMetadata["contribution_basis"] = stateIndex == 0 ? "deterministic_initialization_top_k" : "production_32_draw_monte_carlo";
				auto exactViewAt = [&](size_t pixelIndex) -> const PatchMatchInstrumentExactView& {
					return maps.exactViews[viewOffset + pixelIndex * (size_t)maps.numViews];
				};
				auto saveViewFloat3 = [&](const char* signal, const String& fileName, auto first, auto second, auto third,
					const char* semantics, const nlohmann::json& channels) {
					Image32F3 image(size);
					for (int i = 0; i < image.area(); ++i) {
						const PatchMatchInstrumentExactView& value(exactViewAt((size_t)i));
						Pixel32F& pixel(image[i]);
						pixel.b = first(value);
						pixel.g = second(value);
						pixel.r = third(value);
					}
					nlohmann::json metadata(viewMetadata);
					metadata["uncompressed_bytes"] = (uint64_t)(area * sizeof(float) * 3u);
					metadata["channels_memory_order"] = channels;
					if (image.Save(stateDir + fileName))
						addMap(signal, relativeStateDir + fileName, "float32x3", "logical_view_state", semantics,
							"exact", "production_hot_kernel_view_record", metadata);
					else
						manifest["write_errors"].push_back(signal);
				};
				saveViewFloat3("view_cost_components_exact", _T("view_cost_components_exact") + suffix + _T(".pfm"),
					[](const PatchMatchInstrumentExactView& value) { return value.photometricCost; },
					[](const PatchMatchInstrumentExactView& value) { return value.geometricCost; },
					[](const PatchMatchInstrumentExactView& value) { return value.totalCost; },
					"per-view retained-winner production cost components",
					{{"0", "photo_after_prior"}, {"1", "geometric"}, {"2", "total"}});
				saveViewFloat3("view_selection_metrics_exact", _T("view_selection_metrics_exact") + suffix + _T(".pfm"),
					[](const PatchMatchInstrumentExactView& value) { return value.selectionPrior; },
					[](const PatchMatchInstrumentExactView& value) { return value.samplingScore; },
					[](const PatchMatchInstrumentExactView& value) { return value.samplingProbability; },
					"per-view production selection prior, unnormalized sampling score, and normalized probability; -1 marks unavailable initialization fields",
					{{"0", "selection_prior"}, {"1", "sampling_score"}, {"2", "sampling_probability"}});
				ConfidenceMap contribution(size);
				for (int i = 0; i < contribution.area(); ++i)
					contribution[i] = exactViewAt((size_t)i).weightedContribution;
				if (contribution.Save(stateDir + _T("view_weighted_contribution_exact") + suffix + _T(".pfm")))
					addMap("view_weighted_contribution_exact", relativeStateDir + _T("view_weighted_contribution_exact") + suffix + _T(".pfm"),
						"float32", "logical_view_state", "this source view's exact additive contribution to the retained total cost",
						"exact", "production_hot_kernel_view_record", viewMetadata);
				else
					manifest["write_errors"].push_back("view_weighted_contribution_exact");

				auto metadataField = [](uint32_t metadata, uint32_t shift, uint32_t mask) {
					return (uint8_t)((metadata >> shift) & mask);
				};
				Image8U3 selectionState(size), agreementState(size);
				for (int i = 0; i < selectionState.area(); ++i) {
					const uint32_t metadata(exactViewAt((size_t)i).metadata);
					Pixel8U& selection(selectionState[i]);
					selection.r = metadataField(metadata, PM_EXACT_VIEW_WEIGHT_SHIFT, PM_EXACT_VIEW_WEIGHT_MASK);
					selection.g = metadataField(metadata, PM_EXACT_VIEW_RANK_SHIFT, PM_EXACT_VIEW_RANK_MASK);
					selection.b = metadataField(metadata, PM_EXACT_VIEW_DECISION_SHIFT, PM_EXACT_VIEW_DECISION_MASK);
					Pixel8U& agreement(agreementState[i]);
					agreement.r = metadataField(metadata, PM_EXACT_VIEW_AGREE_SHIFT, PM_EXACT_VIEW_AGREE_MASK);
					agreement.g = metadataField(metadata, PM_EXACT_VIEW_BAD_SHIFT, PM_EXACT_VIEW_BAD_MASK);
					agreement.b =
						((metadata & PM_EXACT_VIEW_SELECTED_BIT) ? 1u : 0u) |
						((metadata & PM_EXACT_VIEW_CONTRIBUTION_BIT) ? 2u : 0u) |
						((metadata & PM_EXACT_VIEW_FINITE_BIT) ? 4u : 0u) |
						((metadata & PM_EXACT_VIEW_PROBABILITY_BIT) ? 8u : 0u);
				}
				nlohmann::json selectionMetadata(viewMetadata);
				selectionMetadata["uncompressed_bytes"] = (uint64_t)(area * 3u);
				selectionMetadata["channels"] = {{"R", "monte_carlo_weight_0_to_32"}, {"G", "rank_0_based_63_unavailable"}, {"B", "decision_enum"}};
				if (selectionState.Save(stateDir + _T("view_selection_state_exact") + suffix + _T(".png")))
					addMap("view_selection_state_exact", relativeStateDir + _T("view_selection_state_exact") + suffix + _T(".png"),
						"uint8x3", "logical_view_state", "lossless per-view production weight, deterministic rank, and selection decision",
						"exact", "production_hot_kernel_view_record", selectionMetadata);
				else
					manifest["write_errors"].push_back("view_selection_state_exact");
				nlohmann::json agreementMetadata(viewMetadata);
				agreementMetadata["uncompressed_bytes"] = (uint64_t)(area * 3u);
				agreementMetadata["channels"] = {{"R", "agreeing_neighbor_count"}, {"G", "bad_neighbor_count"}, {"B", "flags"}};
				agreementMetadata["flags"] = {{"bit0", "selected"}, {"bit1", "contribution_basis_available"}, {"bit2", "finite_cost"}, {"bit3", "probability_available"}};
				if (agreementState.Save(stateDir + _T("view_agreement_state_exact") + suffix + _T(".png")))
					addMap("view_agreement_state_exact", relativeStateDir + _T("view_agreement_state_exact") + suffix + _T(".png"),
						"uint8x3", "logical_view_state", "lossless per-view neighbor agreement counts and availability/selection flags",
						"exact", "production_hot_kernel_view_record", agreementMetadata);
				else
					manifest["write_errors"].push_back("view_agreement_state_exact");
			}
		}
	}
	if (exactMapsComplete)
		WriteExactObservabilityTables(depthMapDir, depthData, maps, area, manifest);

	const size_t apdIterationCount(maps.numLogicalStates > 0 ?
		(size_t)(maps.numLogicalStates-1) : 0u);
	const bool apdMapsComplete(
		maps.apdMapsAvailable && apdIterationCount > 0 &&
		maps.apdStates.size() >= area*apdIterationCount &&
		maps.apdUpdates.size() >= area*apdIterationCount);
	if (apdMapsComplete) {
		const bool visibilityAwareComponents(
			maps.visibilityRequested && maps.apdStageActive);
		const char* anchorComponentSemantics(visibilityAwareComponents ?
			"visibility-aware effective anchor leg for the winning candidate: each view uses its active-anchor mean when visible anchor consensus is valid, otherwise its center cost; lower is better" :
			"mean photometric component over the active APD anchors for the winning candidate; lower is better");
		const char* deformableComponentSemantics(visibilityAwareComponents ?
			"visibility-aware APD photometric objective 0.25*center + 0.75*effective anchor leg; lower is better" :
			"paper-weighted APD photometric objective 0.25*center + 0.75*mean anchors; lower is better");
		const String apdDir(depthMapDir + _T("apd_states/"));
		Util::ensureFolder(apdDir);
		for (size_t iteration=0; iteration<apdIterationCount; ++iteration) {
			const String stateName(String::FormatString(_T("iteration%02u"), (unsigned)(iteration+1)));
			const String relativeStateDir(String(_T("apd_states/")) + stateName + _T("/"));
			const String stateDir(depthMapDir + relativeStateDir);
			Util::ensureFolder(stateDir);
			const size_t offset(iteration*area);
			const nlohmann::json metadata = {
				{"logical_iteration", iteration},
				{"stage", "iteration"},
				{"stage_index", iteration+1},
				{"apd_schema_name", "openmvs.dmap.apd_pixel_mechanics"},
				{"apd_schema_version", PM_APD_INSTRUMENT_SCHEMA_VERSION},
				{"uncompressed_float_bytes", (uint64_t)(area*sizeof(float))},
				{"uncompressed_byte_bytes", (uint64_t)area}
			};
			auto saveStateFloat = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
				std::vector<float> values(area);
				for (size_t i=0; i<area; ++i)
					values[i] = extractor(maps.apdStates[offset+i]);
				nlohmann::json fieldMetadata(metadata);
				for (auto it=extra.begin(); it!=extra.end(); ++it)
					fieldMetadata[it.key()] = it.value();
				if (SaveInstrumentScalarMap(stateDir + fileName, size, values))
					addMap(signal, relativeStateDir + fileName, "float32", "apd_logical_iteration_state",
						semantics, "exact", "apd_same_stream_mechanics_record", fieldMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			auto saveUpdateFloat = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
				std::vector<float> values(area);
				for (size_t i=0; i<area; ++i)
					values[i] = extractor(maps.apdUpdates[offset+i]);
				nlohmann::json fieldMetadata(metadata);
				for (auto it=extra.begin(); it!=extra.end(); ++it)
					fieldMetadata[it.key()] = it.value();
				if (SaveInstrumentScalarMap(stateDir + fileName, size, values))
					addMap(signal, relativeStateDir + fileName, "float32", "apd_logical_iteration_update",
						semantics, "exact", "apd_process_pixel_candidate_record", fieldMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			auto saveStateByte = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
				Image8U image(size);
				for (int i=0; i<image.area(); ++i)
					image[i] = extractor(maps.apdStates[offset+(size_t)i]);
				nlohmann::json fieldMetadata(metadata);
				for (auto it=extra.begin(); it!=extra.end(); ++it)
					fieldMetadata[it.key()] = it.value();
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, "uint8", "apd_logical_iteration_state",
						semantics, "exact", "apd_same_stream_mechanics_record", fieldMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			auto saveUpdateByte = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
				Image8U image(size);
				for (int i=0; i<image.area(); ++i)
					image[i] = extractor(maps.apdUpdates[offset+(size_t)i]);
				nlohmann::json fieldMetadata(metadata);
				for (auto it=extra.begin(); it!=extra.end(); ++it)
					fieldMetadata[it.key()] = it.value();
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, "uint8", "apd_logical_iteration_update",
						semantics, "exact", "apd_process_pixel_candidate_record", fieldMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			auto saveUpdateMaskRGBA = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics) {
				Image8U4 image(size);
				for (int i=0; i<image.area(); ++i) {
					const uint32_t mask(extractor(maps.apdUpdates[offset+(size_t)i]));
					Color8U& pixel(image[i]);
					pixel.r = (uint8_t)(mask & 0xFFu);
					pixel.g = (uint8_t)((mask >> 8) & 0xFFu);
					pixel.b = (uint8_t)((mask >> 16) & 0xFFu);
					pixel.a = (uint8_t)((mask >> 24) & 0xFFu);
				}
				nlohmann::json fieldMetadata(metadata);
				fieldMetadata["uncompressed_bytes"] = (uint64_t)(area * 4u);
				fieldMetadata["encoding"] = "uint32 little-endian bytes in RGBA channels";
				if (image.Save(stateDir + fileName))
					addMap(signal, relativeStateDir + fileName, "uint8x4", "apd_logical_iteration_update",
						semantics, "exact", "apd_process_pixel_candidate_record", fieldMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};

			saveStateFloat("apd_average_baseline", "average_baseline.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.averageBaseline; },
				"average source-view baseline used to convert the paper disparity profile to depth");
			saveStateFloat("apd_current_disparity", "current_disparity.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.currentDisparity; },
				"center disparity at the start of APD reliability classification");
			saveStateFloat("apd_global_minimum_offset", "global_minimum_offset.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return (float)value.globalMinimumOffset; },
				"signed integer offset of the global profile minimum in the 61-sample disparity profile",
				{{"valid_min", -30}, {"valid_max", 30}, {"integer_encoded_exactly", true}});
			saveStateFloat("apd_global_minimum_cost", "global_minimum_cost.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.globalMinimumCost; },
				"minimum finite photometric cost in the APD disparity profile; lower is better; -1 means unavailable");
			saveStateFloat("apd_profile_separation", "profile_separation.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.separation; },
				"paper reliability separation sqrt(sum of squared local-minimum cost deviations)/(local minimum count minus one); larger is more discriminative; -1 means unavailable");
			saveStateFloat("apd_nearest_reliable_distance", "nearest_reliable_distance.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.nearestReliableDistance; },
				"Euclidean reference-pyramid pixel distance to the selected nearest reliable pixel; -1 means unavailable");
			saveStateFloat("apd_ransac_threshold", "ransac_threshold.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.ransacThreshold; },
				"normalized plane residual threshold selected by APD RANSAC; -1 means no valid model");
			saveStateFloat("apd_ransac_center_residual", "ransac_center_residual.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.ransacCenterResidual; },
				"normalized fitted-plane residual at the center pixel; lower is better; -1 means unavailable");
			saveStateFloat("apd_ransac_mean_inlier_residual", "ransac_mean_inlier_residual.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.ransacMeanInlierResidual; },
				"mean normalized fitted-plane residual over RANSAC inliers; lower is better; -1 means unavailable");
			saveStateFloat("apd_fitted_plane_depth", "fitted_plane_depth.pfm",
				[](const PatchMatchAPDInstrumentState& value) { return value.fittedPlaneDepth; },
				"native reference-camera depth of the RANSAC fitted-plane candidate generated after reliable-first updates; -1 means unavailable");

			saveStateByte("apd_reliability_class", "reliability_class.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.reliability; },
				"current paper-profile reliability classification; exact dispatch basis except at multiscale logical iteration zero, which consumes apd_transferred_reliability",
				{{"enum", {{"0", "unknown"}, {"1", "unreliable"}, {"2", "reliable"}}}});
			saveStateByte("apd_profile_reason", "profile_reason.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.profileReason; },
				"exact terminal reason from APD profile classification");
			saveStateByte("apd_profile_eta", "profile_eta.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.eta; },
				"iteration-dependent APD minimum-separation offset eta in profile samples");
			saveStateByte("apd_profile_finite_count", "profile_finite_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.finiteCount; },
				"number of finite values in the 61-sample APD disparity profile");
			saveStateByte("apd_profile_local_minimum_count", "profile_local_minimum_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.localMinimumCount; },
				"number of strict local minima detected in the APD disparity profile");
			saveStateByte("apd_profile_plateau_start", "profile_plateau_start.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.globalMinimumPlateauStart; },
				"inclusive sample index at the start of the global-minimum plateau");
			saveStateByte("apd_profile_plateau_end", "profile_plateau_end.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.globalMinimumPlateauEnd; },
				"inclusive sample index at the end of the global-minimum plateau");
			saveStateByte("apd_sector_candidate_count", "sector_candidate_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.candidateCount; },
				"number of populated angular-sector candidates supplied to APD RANSAC");
			saveStateByte("apd_ransac_inlier_count", "ransac_inlier_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.inlierCount; },
				"number of candidates classified as fitted-plane inliers");
			saveStateByte("apd_ransac_outlier_count", "ransac_outlier_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.outlierCount; },
				"number of candidates classified as fitted-plane outliers");
			saveStateByte("apd_anchor_count", "anchor_count.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.anchorCount; },
				"number of APD anchor samples retained after plane fitting", {{"valid_min", 0}, {"valid_max", PM_APD_INSTRUMENT_ANCHORS}});
			saveStateByte("apd_anchor_reason", "anchor_reason.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.anchorReason; },
				"exact terminal reason from APD anchor construction");
			saveStateByte("apd_ransac_valid", "ransac_valid.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.ransacValid; },
				"one when APD RANSAC produced a valid plane model", {{"binary", true}});
			saveStateByte("apd_deformable_eligible", "deformable_eligible.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.deformableEligible; },
				"one when the pixel has a complete anchor model and may use deformable cost", {{"binary", true}});
			saveStateByte("apd_fitted_plane_valid", "fitted_plane_valid.png",
				[](const PatchMatchAPDInstrumentState& value) { return value.fittedPlaneValid; },
				"one when a bounded native fitted-plane candidate was available before the non-reliable update", {{"binary", true}});

			saveUpdateFloat("apd_working_winner_cost", "working_winner_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.workingWinnerCost; },
				"APD working objective used to rank the winning candidate; lower is better; not persisted as OpenMVS confidence");
			saveUpdateFloat("apd_native_persistent_cost", "native_persistent_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.nativePersistentCost; },
				"conventional OpenMVS score recomputed for and persisted with the APD working winner before the separately recorded terminal native refinement; lower is better");
			saveUpdateFloat("apd_runner_up_working_cost", "runner_up_working_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.runnerUpWorkingCost; },
				"second-best finite APD working objective in the unconstrained evaluated-candidate ordering for the complete logical iteration; lower is better; -1 means unavailable");
			saveUpdateFloat("apd_winner_runner_up_gap", "winner_runner_up_gap.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.winnerRunnerUpGap; },
				"exact unconstrained evaluated-candidate runner-up cost minus evaluated-candidate best cost; larger separates the top two evaluated candidates; this is not a selected-winner gap when an optional constraint rejects the raw best; -1 means unavailable",
				{{"gap_basis", "unconstrained_evaluated_candidate_set_best_vs_runner_up"},
				 {"selected_winner_relation", "exact_unless_optional_candidate_constraint_rejects_raw_best"}});
			saveUpdateFloat("apd_center_cost", "center_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.centerCost; },
				"APD center-patch photometric component for the winning candidate; lower is better");
			saveUpdateFloat("apd_anchor_mean_cost", "anchor_mean_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.anchorMeanCost; },
				anchorComponentSemantics,
				{{"component_basis", visibilityAwareComponents ?
					"visibility_aware_effective_anchor_leg" : "active_anchor_mean"}});
			saveUpdateFloat("apd_deformable_photometric_cost", "deformable_photometric_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.deformablePhotometricCost; },
				deformableComponentSemantics,
				{{"component_identity", "0.25*center_cost + 0.75*anchor_component"}});
			saveUpdateFloat("apd_geometric_cost", "geometric_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.geometricCost; },
				"geometric-consistency component added to the APD working objective; zero when geometric consistency is disabled");
			saveUpdateFloat("apd_native_minus_working_cost", "native_minus_working_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.nativeMinusWorkingCost; },
				"native persistent cost minus APD working cost for the selected winner; this is objective drift, not an accuracy error");
			saveUpdateFloat("apd_native_stored_cost_before", "native_stored_cost_before.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.nativeStoredCostBefore; },
				"conventional OpenMVS cost stored before the complete APD logical iteration");
			saveUpdateFloat("apd_incumbent_working_cost", "incumbent_working_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.incumbentWorkingCost; },
				"APD working objective of the incoming candidate before propagation and refinement");
			saveUpdateFloat("apd_best_anchor_working_cost", "best_anchor_working_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.bestAnchorWorkingCost; },
				"lowest finite working cost among immutable anchor-plane propagation candidates; -1 means unavailable");
			saveUpdateFloat("apd_accepted_anchor_native_cost", "accepted_anchor_native_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.acceptedAnchorNativeCost; },
				"conventional same-kernel rescore of the accepted anchor proposal before later refinement; -1 means no anchor was accepted");
			saveUpdateFloat("apd_fitted_plane_working_cost", "fitted_plane_working_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.fittedPlaneWorkingCost; },
				"evaluated deformable working score of the fitted-plane candidate; values at or above fBadCost are non-usable and -1 means unavailable");
			saveUpdateFloat("apd_fitted_plane_native_cost", "fitted_plane_native_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.fittedPlaneNativeCost; },
				"conventional native score of the fitted-plane candidate under the same selected-view weights; -1 means unavailable");
			saveUpdateFloat("apd_final_refinement_incumbent_cost", "final_refinement_incumbent_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementIncumbentCost; },
				"conventional native cost before the final bounded disparity refinement; -1 means unavailable");
			saveUpdateFloat("apd_final_refinement_best_cost", "final_refinement_best_cost.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementBestCost; },
				"best conventional native cost in the final bounded disparity search; -1 means unavailable");
			saveUpdateFloat("apd_final_refinement_improvement", "final_refinement_improvement.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementImprovement; },
				"native incumbent cost minus the best final-refinement cost; acceptance requires a strict improvement above 0.10");
			saveUpdateFloat("apd_final_refinement_depth", "final_refinement_depth.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementDepth; },
				"best depth from the final bounded disparity search, whether or not it passed the acceptance threshold; -1 means unavailable");
			saveUpdateFloat("apd_accepted_anchor_index", "accepted_anchor_index.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) {
					return value.acceptedAnchorIndex == ~uint32_t(0) ? -1.f : (float)value.acceptedAnchorIndex;
				},
				"exact flattened pixel index of the accepted immutable anchor; -1 means no anchor was accepted",
				{{"encoding", "uint32_pixel_index_stored_exactly_in_float32_for_admitted_frame_size"}});
			auto saveUpdateMask = [&](const char* signal, const char* fileName, auto extractor,
				const char* semantics) {
				saveUpdateFloat(signal, fileName,
					[&](const PatchMatchAPDInstrumentUpdate& value) { return (float)extractor(value); },
					semantics, {{"encoding", "uint32_bit_mask_stored_exactly_in_float32"},
						{"candidate_slots", PM_INSTRUMENT_EXACT_NUM_CANDIDATES}});
			};
			saveUpdateMask("apd_candidate_tested_mask", "candidate_tested_mask.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.candidateTestedMask; },
				"bit mask of active candidate slots evaluated in this logical iteration; schema v5 includes DVP slots 22 and 23");
			saveUpdateMask("apd_candidate_finite_mask", "candidate_finite_mask.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.candidateFiniteMask; },
				"bit mask of active candidate slots that produced support-usable working scores; schema v5 includes DVP slots 22 and 23");
			saveUpdateMask("apd_candidate_accepted_mask", "candidate_accepted_mask.pfm",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.candidateAcceptedMask; },
				"bit mask of sequential active working-objective improvements during this logical iteration; schema v5 includes DVP slots 22 and 23");
			saveUpdateByte("apd_update_source", "update_source.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.source; },
				"exact candidate family that produced the final active working winner, including DVP when enabled");
			saveUpdateByte("apd_winner_slot", "winner_slot.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.winnerSlot; },
				"exact candidate slot of the APD working winner; 255 means unavailable");
			saveUpdateByte("apd_runner_up_slot", "runner_up_slot.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.runnerUpSlot; },
				"exact candidate slot of the APD working runner-up; 255 means unavailable");
			saveUpdateByte("apd_candidate_tested_count", "candidate_tested_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.testedCount; },
				"number of APD candidates tested in the complete logical iteration");
			saveUpdateByte("apd_candidate_finite_count", "candidate_finite_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finiteCount; },
				"number of tested APD candidates with finite working scores");
			saveUpdateByte("apd_candidate_accepted_count", "candidate_accepted_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.acceptedCount; },
				"number of sequential APD working-score improvements in the complete logical iteration");
			saveUpdateByte("apd_selected_view_count", "selected_view_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.selectedViewCount; },
				"number of source views with nonzero working weight in this logical iteration");
			saveUpdateMaskRGBA("apd_working_selected_views_mask", "working_selected_views_mask.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.workingSelectedViews; },
				"lossless 32-bit mask of source views with nonzero APD working weight in this logical iteration");
			saveUpdateByte("apd_selected_view_weight_sum", "selected_view_weight_sum.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.selectedViewWeightSum; },
				"sum of integer Monte Carlo or fallback weights used by the APD working objective");
			saveUpdateByte("apd_deformable_active", "deformable_active.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.deformableActive; },
				"one when deformable APD scoring was active for the winning candidate", {{"binary", true}});
			saveUpdateByte("apd_view_selection_mode", "view_selection_mode.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.viewSelectionMode; },
				"exact APD view-selection evidence or fallback mode");
			saveUpdateByte("apd_anchor_evidence_count", "anchor_evidence_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.anchorEvidenceCount; },
				"number of immutable anchors contributing valid candidate/view evidence");
			saveUpdateByte("apd_anchor_proposal_count", "anchor_proposal_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.anchorProposalCount; },
				"number of immutable anchor-plane candidates tested");
			saveUpdateByte("apd_anchor_finite_count", "anchor_finite_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.anchorFiniteCount; },
				"number of immutable anchor-plane candidates with finite usable working costs");
			saveUpdateByte("apd_anchor_accepted_slot", "anchor_accepted_slot.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.anchorAcceptedSlot; },
				"zero-based immutable anchor slot accepted before later refinements; 255 means unavailable");
			saveUpdateByte("apd_immutable_anchor_state", "immutable_anchor_state.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.immutableAnchorState; },
				"one when both non-reliable checkerboard phases consumed the logical iteration's immutable anchor plane/view snapshot",
				{{"binary", true}});
			saveUpdateByte("apd_update_stage", "update_stage.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.updateStage; },
				"exact reliable-first dispatch stage", {{"enum", {{"1", "reliable_first"}, {"2", "non_reliable_second"}}}});
			saveUpdateByte("apd_fitted_plane_available", "fitted_plane_available.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.fittedPlaneAvailable; },
				"one when a fitted-plane candidate was available to this update", {{"binary", true}});
			saveUpdateByte("apd_fitted_plane_tested", "fitted_plane_tested.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.fittedPlaneTested; },
				"one when the fitted-plane candidate was scored", {{"binary", true}});
			saveUpdateByte("apd_fitted_plane_accepted", "fitted_plane_accepted.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.fittedPlaneAccepted; },
				"one when the fitted plane improved the sequential working incumbent before later refinement", {{"binary", true}});
			saveUpdateByte("apd_final_refinement_offset", "final_refinement_offset.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return static_cast<uint8_t>(value.finalRefinementOffset+APD_FINAL_REFINEMENT_RADIUS); },
				"signed final-refinement disparity offset encoded as offset+5", {{"encoded_offset", APD_FINAL_REFINEMENT_RADIUS}});
			saveUpdateByte("apd_final_refinement_tested_count", "final_refinement_tested_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementTested; },
				"number of in-range conventional candidates tested by final refinement");
			saveUpdateByte("apd_final_refinement_finite_count", "final_refinement_finite_count.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementFinite; },
				"number of finite conventional candidates in final refinement");
			saveUpdateByte("apd_final_refinement_accepted", "final_refinement_accepted.png",
				[](const PatchMatchAPDInstrumentUpdate& value) { return value.finalRefinementAccepted; },
				"one when final native refinement passed the strict 0.10 improvement threshold", {{"binary", true}});
		}
		} else if (maps.apdRequested && maps.apdStageActive && maps.mapsRequested) {
			manifest["write_errors"].push_back(maps.apdMapsAvailable ?
				"apd_map_capture_incomplete" : "apd_map_capture_not_admitted");
		}

		const size_t dvpIterationCount(maps.numLogicalStates > 0 ?
			(size_t)(maps.numLogicalStates-1) : 0u);
		const bool dvpMapsComplete(
			maps.dvpMapsAvailable && dvpIterationCount > 0 &&
			maps.dvpUpdates.size() >= area*dvpIterationCount);
		if (dvpMapsComplete) {
			const String dvpDir(depthMapDir + _T("dvp_states/"));
			Util::ensureFolder(dvpDir);
			for (size_t iteration=0; iteration<dvpIterationCount; ++iteration) {
				const String stateName(String::FormatString(_T("iteration%02u"), (unsigned)(iteration+1)));
				const String relativeStateDir(String(_T("dvp_states/")) + stateName + _T("/"));
				const String stateDir(depthMapDir + relativeStateDir);
				Util::ensureFolder(stateDir);
				const size_t offset(iteration*area);
				const nlohmann::json metadata = {
					{"logical_iteration", iteration},
					{"stage", "iteration"},
					{"stage_index", iteration+1},
					{"dvp_schema_name", "openmvs.dmap.dvp_pixel_mechanics"},
					{"dvp_schema_version", PM_DVP_INSTRUMENT_SCHEMA_VERSION},
					{"measurement_quality", "exact"},
					{"measurement_basis", "active_process_pixel_candidate_path"}
				};
				auto saveFloat = [&](const String& signal, const String& fileName, auto extractor,
					const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
					std::vector<float> values(area);
					for (size_t i=0; i<area; ++i)
						values[i] = extractor(maps.dvpUpdates[offset+i]);
					nlohmann::json fieldMetadata(metadata);
					for (auto it=extra.begin(); it!=extra.end(); ++it)
						fieldMetadata[it.key()] = it.value();
					if (SaveInstrumentScalarMap(stateDir + fileName, size, values))
						addMap(signal.c_str(), relativeStateDir + fileName, "float32",
							"dvp_logical_iteration_update", semantics, "exact",
							"dvp_process_pixel_candidate_record", fieldMetadata);
					else
						manifest["write_errors"].push_back(signal.c_str());
				};
				auto saveByte = [&](const String& signal, const String& fileName, auto extractor,
					const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
					Image8U image(size);
					for (int i=0; i<image.area(); ++i)
						image[i] = extractor(maps.dvpUpdates[offset+(size_t)i]);
					nlohmann::json fieldMetadata(metadata);
					for (auto it=extra.begin(); it!=extra.end(); ++it)
						fieldMetadata[it.key()] = it.value();
					if (image.Save(stateDir + fileName))
						addMap(signal.c_str(), relativeStateDir + fileName, "uint8",
							"dvp_logical_iteration_update", semantics, "exact",
							"dvp_process_pixel_candidate_record", fieldMetadata);
					else
						manifest["write_errors"].push_back(signal.c_str());
				};
				auto saveMask = [&](const String& signal, const String& fileName, auto extractor,
					const char* semantics) {
					Image8U4 image(size);
					for (int i=0; i<image.area(); ++i) {
						const uint32_t mask(extractor(maps.dvpUpdates[offset+(size_t)i]));
						Color8U& pixel(image[i]);
						pixel.r = (uint8_t)(mask & 0xFFu);
						pixel.g = (uint8_t)((mask >> 8) & 0xFFu);
						pixel.b = (uint8_t)((mask >> 16) & 0xFFu);
						pixel.a = (uint8_t)((mask >> 24) & 0xFFu);
					}
					nlohmann::json fieldMetadata(metadata);
					fieldMetadata["encoding"] = "uint32 little-endian bytes in RGBA channels";
					if (image.Save(stateDir + fileName))
						addMap(signal.c_str(), relativeStateDir + fileName, "uint8x4",
							"dvp_logical_iteration_update", semantics, "exact",
							"dvp_process_pixel_candidate_record", fieldMetadata);
					else
						manifest["write_errors"].push_back(signal.c_str());
				};

				saveFloat(_T("dvp_incumbent_depth"), _T("incumbent_depth.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.incumbentDepth; },
					"depth entering the DVP proposal stage after propagation and fitted-plane candidates");
				saveFloat(_T("dvp_final_depth"), _T("final_depth.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.finalDepth; },
					"depth retained after DVP and native depth/normal refinement");
				saveFloat(_T("dvp_left_interval_minimum"), _T("left_interval_minimum.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.leftIntervalMinimum; },
					"minimum of the independently retained left epipolar depth interval; -1 means unavailable");
				saveFloat(_T("dvp_left_interval_maximum"), _T("left_interval_maximum.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.leftIntervalMaximum; },
					"maximum of the independently retained left epipolar depth interval; -1 means unavailable");
				saveFloat(_T("dvp_right_interval_minimum"), _T("right_interval_minimum.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.rightIntervalMinimum; },
					"minimum of the independently retained right epipolar depth interval; -1 means unavailable");
				saveFloat(_T("dvp_right_interval_maximum"), _T("right_interval_maximum.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.rightIntervalMaximum; },
					"maximum of the independently retained right epipolar depth interval; -1 means unavailable");
				saveFloat(_T("dvp_incumbent_cost"), _T("incumbent_cost.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.incumbentCost; },
					"active working cost entering the DVP proposal stage");
				saveFloat(_T("dvp_winner_cost"), _T("winner_cost.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.winnerCost; },
					"lowest valid cost among the DVP incumbent and tested DVP proposals");
				saveFloat(_T("dvp_runner_up_cost"), _T("runner_up_cost.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.runnerUpCost; },
					"second-lowest valid cost among the DVP incumbent and proposals; -1 means unavailable");
				saveFloat(_T("dvp_winner_runner_up_gap"), _T("winner_runner_up_gap.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.winnerRunnerUpGap; },
					"runner-up cost minus winner cost in the active DVP working domain; -1 means unavailable");
				saveFloat(_T("dvp_depth_displacement"), _T("depth_displacement.pfm"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.depthDisplacement; },
					"absolute difference between DVP-stage incumbent depth and final retained depth");

				for (unsigned proposal=0; proposal<DVP_MAX_PROPOSALS; ++proposal) {
					const String suffix(String::FormatString(_T("_%u"), proposal));
					saveFloat(_T("dvp_proposal_depth") + suffix, _T("proposal_depth") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.proposalDepth[proposal]; },
						"exact generated proposal depth; -1 means unavailable");
					saveFloat(_T("dvp_candidate_cost") + suffix, _T("candidate_cost") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.candidateCost[proposal]; },
						"exact active candidate cost; -1 means not tested or non-finite");
					saveFloat(_T("dvp_mean_reprojection_error") + suffix, _T("mean_reprojection_error") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.meanReprojectionError[proposal]; },
						"mean gated roundtrip reprojection error in source pixels; -1 means unavailable");
					saveFloat(_T("dvp_max_reprojection_error") + suffix, _T("max_reprojection_error") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.maxReprojectionError[proposal]; },
						"maximum gated roundtrip reprojection error in source pixels; -1 means unavailable");
					saveFloat(_T("dvp_mean_relative_depth_error") + suffix, _T("mean_relative_depth_error") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.meanRelativeDepthError[proposal]; },
						"mean gated relative source-depth disagreement; -1 means unavailable");
					saveFloat(_T("dvp_max_relative_depth_error") + suffix, _T("max_relative_depth_error") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.maxRelativeDepthError[proposal]; },
						"maximum gated relative source-depth disagreement; -1 means unavailable");
					saveFloat(_T("dvp_signed_offset") + suffix, _T("signed_offset") + suffix + _T(".pfm"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return (float)v.signedOffset[proposal]; },
						"signed source epipolar offset for global/midpoint proposals or -1/+1 interval-side identity");
					saveByte(_T("dvp_source_view") + suffix, _T("source_view") + suffix + _T(".png"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.sourceView[proposal]; },
						"source view that generated a one-view proposal; 255 means multi-view or unavailable");
					saveByte(_T("dvp_support") + suffix, _T("support") + suffix + _T(".png"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.support[proposal]; },
						"number of supporting views for this proposal");
					saveMask(_T("dvp_support_views") + suffix, _T("support_views") + suffix + _T(".png"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.supportViews[proposal]; },
						"lossless 32-bit mask of views supporting this proposal");
					saveMask(_T("dvp_occluded_views") + suffix, _T("occluded_views") + suffix + _T(".png"),
						[proposal](const PatchMatchDVPInstrumentUpdate& v) { return v.occludedViews[proposal]; },
						"lossless 32-bit mask of views rejected as occluded during gated validation");
				}

				saveMask(_T("dvp_selected_source_views"), _T("selected_source_views.png"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.selectedSourceViews; },
					"lossless 32-bit selected-view mask entering DVP proposal generation");
				saveMask(_T("dvp_direction_source_views"), _T("direction_source_views.png"),
					[](const PatchMatchDVPInstrumentUpdate& v) { return v.directionSourceViews; },
					"lossless 32-bit mask of views with valid epipolar directions");
				saveByte(_T("dvp_left_outer_count"), _T("left_outer_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.leftOuterCount; }, "number of valid left-outer endpoint depths");
				saveByte(_T("dvp_left_inner_count"), _T("left_inner_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.leftInnerCount; }, "number of valid left-inner endpoint depths");
				saveByte(_T("dvp_right_inner_count"), _T("right_inner_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.rightInnerCount; }, "number of valid right-inner endpoint depths");
				saveByte(_T("dvp_right_outer_count"), _T("right_outer_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.rightOuterCount; }, "number of valid right-outer endpoint depths");
				saveByte(_T("dvp_family"), _T("family.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.family; }, "configured DVP proposal-family enum");
				saveByte(_T("dvp_unavailable_reason"), _T("unavailable_reason.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.unavailableReason; }, "terminal reason no DVP proposal was available");
				saveByte(_T("dvp_generated_count"), _T("generated_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.generatedCount; }, "number of generated DVP proposals");
				saveByte(_T("dvp_tested_count"), _T("tested_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.testedCount; }, "number of in-range DVP proposals scored");
				saveByte(_T("dvp_finite_count"), _T("finite_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.finiteCount; }, "number of DVP proposals with finite nonnegative active cost");
				saveByte(_T("dvp_accepted_count"), _T("accepted_count.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.acceptedCount; }, "number of sequential DVP incumbent improvements");
				saveByte(_T("dvp_tested_proposal_mask"), _T("tested_proposal_mask.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.testedProposalMask; }, "two-bit mask of generated DVP proposals scored by the active objective");
				saveByte(_T("dvp_finite_proposal_mask"), _T("finite_proposal_mask.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.finiteProposalMask; }, "two-bit mask of tested DVP proposals with finite nonnegative active costs");
				saveByte(_T("dvp_accepted_proposal_mask"), _T("accepted_proposal_mask.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.acceptedProposalMask; }, "two-bit mask of DVP proposals that sequentially improved the active incumbent");
				saveByte(_T("dvp_winner_ordinal"), _T("winner_ordinal.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.winnerOrdinal; }, "winning DVP proposal ordinal; 255 means incumbent");
				saveByte(_T("dvp_accepted"), _T("accepted.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.accepted; }, "one when a DVP proposal beat the DVP-stage incumbent", {{"binary", true}});
				saveByte(_T("dvp_final_winner"), _T("final_winner.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.finalWinner; }, "one when the complete DVP plane hypothesis remained the final winner after native refinements", {{"binary", true}});
				saveByte(_T("dvp_final_update_source"), _T("final_update_source.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.finalUpdateSource; }, "exact source of the final complete plane hypothesis after DVP and native refinements");
				saveByte(_T("dvp_final_depth_retained"), _T("final_depth_retained.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.finalDepthRetained; }, "one when the final depth exactly equals the winning DVP proposal depth even if a later normal-only candidate replaced the plane hypothesis", {{"binary", true}});
				saveByte(_T("dvp_native_depth_fallback"), _T("native_depth_fallback.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.nativeDepthFallback; }, "one when Eq. 11 intervals were unavailable and native depth perturbation ran", {{"binary", true}});
				saveByte(_T("dvp_left_interval_valid"), _T("left_interval_valid.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.leftIntervalValid; }, "one when the left Eq. 11 interval is valid", {{"binary", true}});
				saveByte(_T("dvp_right_interval_valid"), _T("right_interval_valid.png"), [](const PatchMatchDVPInstrumentUpdate& v) { return v.rightIntervalValid; }, "one when the right Eq. 11 interval is valid", {{"binary", true}});
			}
		} else if (maps.dvpRequested && maps.mapsRequested) {
			manifest["write_errors"].push_back(maps.dvpMapsAvailable ?
				"dvp_map_capture_incomplete" : "dvp_map_capture_not_admitted");
		}

			const size_t visibilityIterationCount(maps.numLogicalStates > 0 ?
				(size_t)(maps.numLogicalStates-1) : 0u);
			const bool visibilityMapsComplete(
				maps.visibilityMapsAvailable && maps.apdStageActive &&
				visibilityIterationCount > 0 &&
				maps.visibilityUpdates.size() >= area*visibilityIterationCount);
			if (visibilityMapsComplete) {
				const String visibilityDir(depthMapDir + _T("dvp_visibility_states/"));
				Util::ensureFolder(visibilityDir);
				for (size_t iteration=0; iteration<visibilityIterationCount; ++iteration) {
					const String stateName(String::FormatString(
						_T("iteration%02u"), (unsigned)(iteration+1)));
					const String relativeStateDir(
						String(_T("dvp_visibility_states/")) + stateName + _T("/"));
					const String stateDir(depthMapDir + relativeStateDir);
					Util::ensureFolder(stateDir);
					const size_t offset(iteration*area);
					const nlohmann::json metadata = {
						{"logical_iteration", iteration},
						{"stage", "iteration"},
						{"stage_index", iteration+1},
						{"dvp_visibility_schema_name", "openmvs.dmap.dvp_visibility_pixel_mechanics"},
						{"dvp_visibility_schema_version", PM_DVP_VISIBILITY_INSTRUMENT_SCHEMA_VERSION},
						{"measurement_quality", "exact"},
						{"measurement_basis", "same_stream_logical_iteration_visibility_transition"}
					};
					auto saveMask = [&](const String& signal, const String& fileName, auto extractor,
						const char* semantics) {
						Image8U4 image(size);
						for (int i=0; i<image.area(); ++i) {
							const uint32_t mask(extractor(
								maps.visibilityUpdates[offset+(size_t)i]));
							Color8U& pixel(image[i]);
							pixel.r = (uint8_t)(mask & 0xFFu);
							pixel.g = (uint8_t)((mask >> 8) & 0xFFu);
							pixel.b = (uint8_t)((mask >> 16) & 0xFFu);
							pixel.a = (uint8_t)((mask >> 24) & 0xFFu);
						}
						nlohmann::json fieldMetadata(metadata);
						fieldMetadata["encoding"] = "uint32 little-endian bytes in RGBA channels";
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint8x4",
								"dvp_visibility_logical_iteration_state", semantics, "exact",
								"dvp_visibility_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};
					auto saveWord = [&](const String& signal, const String& fileName, auto extractor,
						const char* semantics) {
						Image16U image(size);
						for (int i=0; i<image.area(); ++i)
							image[i] = extractor(maps.visibilityUpdates[offset+(size_t)i]);
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint16",
								"dvp_visibility_logical_iteration_state", semantics, "exact",
								"dvp_visibility_process_pixel_record", metadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};
					auto saveByte = [&](const String& signal, const String& fileName, auto extractor,
						const char* semantics, const nlohmann::json& extra = nlohmann::json::object()) {
						Image8U image(size);
						for (int i=0; i<image.area(); ++i)
							image[i] = extractor(maps.visibilityUpdates[offset+(size_t)i]);
						nlohmann::json fieldMetadata(metadata);
						for (auto it=extra.begin(); it!=extra.end(); ++it)
							fieldMetadata[it.key()] = it.value();
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint8",
								"dvp_visibility_logical_iteration_state", semantics, "exact",
								"dvp_visibility_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};

					saveMask(_T("dvp_visibility_previous_mask"), _T("previous_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.previousMask; },
						"lossless per-view support mask published by logical iteration t-1");
					saveMask(_T("dvp_visibility_resolved_mask"), _T("resolved_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.resolvedMask; },
						"lossless immutable support mask after visibility restoration and rejection");
					saveMask(_T("dvp_visibility_next_mask"), _T("next_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.nextMask; },
						"lossless view-selection mask published for the next logical iteration");
					saveMask(_T("dvp_visibility_active_support_mask"), _T("active_support_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.activeSupportMask; },
						"lossless support mask consumed by every candidate in this logical iteration");
					saveMask(_T("dvp_visibility_restored_mask"), _T("restored_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.restoredMask; },
						"views restored from zero previous weight by the active visibility rule");
					saveMask(_T("dvp_visibility_rejected_mask"), _T("rejected_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.rejectedMask; },
						"zero-weight views rejected by an exact visibility reason");
					saveMask(_T("dvp_visibility_added_mask"), _T("added_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.addedMask; },
						"views selected in next state but absent from previous state");
					saveMask(_T("dvp_visibility_removed_mask"), _T("removed_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.removedMask; },
						"views removed from next state relative to previous state");
					saveMask(_T("dvp_visibility_candidate_tested_mask"), _T("candidate_tested_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.candidateTestedMask; },
						"lossless mask of candidate slots tested against the immutable support set");
					saveMask(_T("dvp_visibility_candidate_finite_mask"), _T("candidate_finite_mask.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.candidateFiniteMask; },
						"lossless mask of tested candidate slots with finite active cost");

					saveWord(_T("dvp_visibility_previous_weight_sum"), _T("previous_weight_sum.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.previousWeightSum; },
						"sum of previous integer per-view reliability weights");
					saveWord(_T("dvp_visibility_resolved_weight_sum"), _T("resolved_weight_sum.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.resolvedWeightSum; },
						"sum of exact resolved integer per-view reliability weights");
					saveWord(_T("dvp_visibility_next_weight_sum"), _T("next_weight_sum.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.nextWeightSum; },
						"sum of integer per-view reliability weights published for the next iteration");
					saveWord(_T("dvp_visibility_active_support_weight_sum"), _T("active_support_weight_sum.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.activeSupportWeightSum; },
						"sum of exact integer reliability weights consumed by candidate scoring");
					saveWord(_T("dvp_visibility_denominator"), _T("denominator.png"),
						[](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.denominator; },
						"exact normalization denominator used by all candidates at the pixel");

					saveByte(_T("dvp_visibility_previous_visible_count"), _T("previous_visible_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.previousVisibleCount; }, "number of nonzero previous per-view weights");
					saveByte(_T("dvp_visibility_resolved_visible_count"), _T("resolved_visible_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.resolvedVisibleCount; }, "number of nonzero resolved per-view weights");
					saveByte(_T("dvp_visibility_next_visible_count"), _T("next_visible_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.nextVisibleCount; }, "number of nonzero next per-view weights");
					saveByte(_T("dvp_visibility_active_support_count"), _T("active_support_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.activeSupportCount; }, "number of views contributing to active candidate scoring");
					saveByte(_T("dvp_visibility_restored_count"), _T("restored_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.restoredCount; }, "number of views restored by the active visibility rule");
					saveByte(_T("dvp_visibility_rejected_count"), _T("rejected_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.rejectedCount; }, "number of zero-weight views rejected by visibility resolution");
					saveByte(_T("dvp_visibility_added_count"), _T("added_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.addedCount; }, "number of views added to next state relative to previous state");
					saveByte(_T("dvp_visibility_removed_count"), _T("removed_count.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.removedCount; }, "number of views removed from next state relative to previous state");
					saveByte(_T("dvp_visibility_mode"), _T("mode.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.mode; }, "active DVP visibility mode enum");
					saveByte(_T("dvp_visibility_denominator_defined"), _T("denominator_defined.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.denominatorDefined; }, "one when the candidate-support normalization denominator is nonzero", {{"binary", true}});
					saveByte(_T("dvp_visibility_support_matches_resolved"), _T("support_matches_resolved.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.activeSupportMatchesResolved; }, "one when every active candidate-support weight equals the immutable resolved weight", {{"binary", true}});
					saveByte(_T("dvp_visibility_transition_status"), _T("transition_status.png"), [](const PatchMatchDVPVisibilityInstrumentUpdate& v) { return v.transitionStatus; }, "versioned logical-state transition validation status enum");
					for (int reason=0; reason<PM_DVP_VISIBILITY_INSTRUMENT_REASONS; ++reason) {
						const String reasonName(DVPVisibilityReasonName((unsigned)reason));
						saveByte(_T("dvp_visibility_reason_") + reasonName,
							reasonName + _T("_count.png"),
							[reason](const PatchMatchDVPVisibilityInstrumentUpdate& v) {
								return v.reasonCount[reason];
							},
							"number of source views assigned this exact visibility resolution reason");
					}
				}
			} else if (maps.visibilityRequested && maps.apdStageActive && maps.mapsRequested) {
				manifest["write_errors"].push_back(maps.visibilityMapsAvailable ?
					"dvp_visibility_map_capture_incomplete" :
					"dvp_visibility_map_capture_not_admitted");
			}

			const size_t visibleNormalIterationCount(maps.numLogicalStates > 0 ?
				(size_t)(maps.numLogicalStates-1) : 0u);
			const bool visibleNormalMapsComplete(
				maps.visibleNormalMapsAvailable && maps.apdStageActive &&
				visibleNormalIterationCount > 0 &&
				maps.visibleNormalUpdates.size() >= area*visibleNormalIterationCount);
			if (visibleNormalMapsComplete) {
				const String visibleNormalDir(depthMapDir + _T("dvp_visible_normal_states/"));
				Util::ensureFolder(visibleNormalDir);
				for (size_t iteration=0; iteration<visibleNormalIterationCount; ++iteration) {
					const String stateName(String::FormatString(
						_T("iteration%02u"), (unsigned)(iteration+1)));
					const String relativeStateDir(
						String(_T("dvp_visible_normal_states/")) + stateName + _T("/"));
					const String stateDir(depthMapDir + relativeStateDir);
					Util::ensureFolder(stateDir);
					const size_t offset(iteration*area);
					const nlohmann::json metadata = {
						{"logical_iteration", iteration},
						{"stage", "iteration"},
						{"stage_index", iteration+1},
						{"dvp_visible_normal_schema_name",
							"openmvs.dmap.dvp_visible_normal_pixel_mechanics"},
						{"dvp_visible_normal_schema_version",
							PM_DVP_VISIBLE_NORMAL_INSTRUMENT_SCHEMA_VERSION},
						{"measurement_quality", "exact"},
						{"measurement_basis", "active_process_pixel_visible_normal_decision"}
					};
					auto saveFloat = [&](const String& signal, const String& fileName,
						auto extractor, const char* semantics,
						const nlohmann::json& extra = nlohmann::json::object()) {
						std::vector<float> values(area);
						for (size_t i=0; i<area; ++i)
							values[i] = extractor(maps.visibleNormalUpdates[offset+i]);
						nlohmann::json fieldMetadata(metadata);
						for (auto it=extra.begin(); it!=extra.end(); ++it)
							fieldMetadata[it.key()] = it.value();
						if (SaveInstrumentScalarMap(stateDir + fileName, size, values))
							addMap(signal.c_str(), relativeStateDir + fileName, "float32",
								"dvp_visible_normal_logical_iteration_update", semantics, "exact",
								"dvp_visible_normal_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};
					auto saveByte = [&](const String& signal, const String& fileName,
						auto extractor, const char* semantics,
						const nlohmann::json& extra = nlohmann::json::object()) {
						Image8U image(size);
						for (int i=0; i<image.area(); ++i)
							image[i] = extractor(maps.visibleNormalUpdates[offset+(size_t)i]);
						nlohmann::json fieldMetadata(metadata);
						for (auto it=extra.begin(); it!=extra.end(); ++it)
							fieldMetadata[it.key()] = it.value();
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint8",
								"dvp_visible_normal_logical_iteration_update", semantics, "exact",
								"dvp_visible_normal_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};
					auto saveWord = [&](const String& signal, const String& fileName,
						auto extractor, const char* semantics,
						const nlohmann::json& extra = nlohmann::json::object()) {
						Image16U image(size);
						for (int i=0; i<image.area(); ++i)
							image[i] = extractor(maps.visibleNormalUpdates[offset+(size_t)i]);
						nlohmann::json fieldMetadata(metadata);
						for (auto it=extra.begin(); it!=extra.end(); ++it)
							fieldMetadata[it.key()] = it.value();
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint16",
								"dvp_visible_normal_logical_iteration_update", semantics, "exact",
								"dvp_visible_normal_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};
					auto saveMask = [&](const String& signal, const String& fileName,
						auto extractor, const char* semantics) {
						Image8U4 image(size);
						for (int i=0; i<image.area(); ++i) {
							const uint32_t mask(extractor(
								maps.visibleNormalUpdates[offset+(size_t)i]));
							Color8U& pixel(image[i]);
							pixel.r = (uint8_t)(mask & 0xFFu);
							pixel.g = (uint8_t)((mask >> 8) & 0xFFu);
							pixel.b = (uint8_t)((mask >> 16) & 0xFFu);
							pixel.a = (uint8_t)((mask >> 24) & 0xFFu);
						}
						nlohmann::json fieldMetadata(metadata);
						fieldMetadata["encoding"] =
							"uint32 little-endian bytes in RGBA channels";
						if (image.Save(stateDir + fileName))
							addMap(signal.c_str(), relativeStateDir + fileName, "uint8x4",
								"dvp_visible_normal_logical_iteration_update", semantics, "exact",
								"dvp_visible_normal_process_pixel_record", fieldMetadata);
						else
							manifest["write_errors"].push_back(signal.c_str());
					};

					saveFloat(_T("dvp_visible_normal_current_max_dot"),
						_T("current_max_dot.pfm"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.currentMaxDot;
						}, "maximum current-normal dot product over selected camera rays");
					saveFloat(_T("dvp_visible_normal_current_max_violation"),
						_T("current_max_violation.pfm"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.currentMaxViolation;
						}, "positive hemisphere-constraint violation of the current normal");
					saveFloat(_T("dvp_visible_normal_propagation_native_best_cost"),
						_T("propagation_native_best_cost.pfm"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.propagationNativeBestCost;
						}, "lowest finite native propagation-candidate cost; -1 means unavailable");
					saveFloat(_T("dvp_visible_normal_propagation_constrained_best_cost"),
						_T("propagation_constrained_best_cost.pfm"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.propagationConstrainedBestCost;
						}, "lowest finite feasible propagation-candidate cost; -1 means unavailable");
					saveFloat(_T("dvp_visible_normal_propagation_selected_cost"),
						_T("propagation_selected_cost.pfm"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.propagationSelectedCost;
						}, "cost of the propagation candidate selected by the configured mode");
					for (unsigned proposal=0;
						 proposal<DVP_VISIBLE_NORMAL_REFINEMENT_PROPOSALS; ++proposal) {
						const String suffix(String::FormatString(_T("_%u"), proposal));
						saveFloat(_T("dvp_visible_normal_native_max_dot") + suffix,
							_T("native_max_dot") + suffix + _T(".pfm"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.nativeMaxDot[proposal];
							}, "maximum selected-ray dot product for the native stochastic proposal");
						saveFloat(_T("dvp_visible_normal_selected_max_dot") + suffix,
							_T("selected_max_dot") + suffix + _T(".pfm"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.selectedMaxDot[proposal];
							}, "maximum selected-ray dot product for the proposal actually scored");
						saveWord(_T("dvp_visible_normal_retries_tested") + suffix,
							_T("retries_tested") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.retriesTested[proposal];
							}, "number of local-RNG constrained retry candidates tested");
						saveWord(_T("dvp_visible_normal_selected_retry") + suffix,
							_T("selected_retry") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.selectedRetry[proposal];
							}, "zero-based accepted retry; 65535 means native or exhausted fallback",
							{{"unavailable", 65535}});
						saveByte(_T("dvp_visible_normal_native_reason") + suffix,
							_T("native_reason") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.nativeReason[proposal];
							}, "exact native stochastic-proposal feasibility reason");
						saveByte(_T("dvp_visible_normal_proposal_reason") + suffix,
							_T("proposal_reason") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.proposalReason[proposal];
							}, "exact bounded-retry selection or fallback reason");
						saveByte(_T("dvp_visible_normal_native_rejected_direction") + suffix,
							_T("native_rejected_direction") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.nativeRejectedDirection[proposal];
							}, "first selected-ray index rejecting the native proposal; 255 means none");
						saveByte(_T("dvp_visible_normal_selected_rejected_direction") + suffix,
							_T("selected_rejected_direction") + suffix + _T(".png"),
							[proposal](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
								return v.selectedRejectedDirection[proposal];
							}, "first selected-ray index rejecting the scored proposal; 255 means none");
					}

					saveMask(_T("dvp_visible_normal_selected_source_views"),
						_T("selected_source_views.png"),
						[](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) {
							return v.selectedSourceViews;
						}, "lossless selected-source-view mask used to build visibility directions");
					saveByte(_T("dvp_visible_normal_support_count"), _T("support_count.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.supportCount; }, "number of selected source cameras contributing valid directions");
					saveByte(_T("dvp_visible_normal_direction_count"), _T("direction_count.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.directionCount; }, "number of visibility directions including the reference camera ray");
					saveByte(_T("dvp_visible_normal_mode"), _T("mode.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.mode; }, "configured visible-normal mode enum");
					saveByte(_T("dvp_visible_normal_current_valid"), _T("current_valid.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.currentValid; }, "one when current normal and direction inputs are valid", {{"binary", true}});
					saveByte(_T("dvp_visible_normal_current_feasible"), _T("current_feasible.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.currentFeasible; }, "one when the current normal satisfies every selected-view hemisphere constraint", {{"binary", true}});
					saveByte(_T("dvp_visible_normal_current_reason"), _T("current_reason.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.currentReason; }, "exact current-normal feasibility reason");
					saveByte(_T("dvp_visible_normal_current_rejected_direction"), _T("current_rejected_direction.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.currentRejectedDirection; }, "first selected-ray index rejecting the current normal; 255 means none");
					saveByte(_T("dvp_visible_normal_propagation_tested_mask"), _T("propagation_tested_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationTestedMask; }, "lossless mask of native propagation candidates tested");
					saveByte(_T("dvp_visible_normal_propagation_valid_mask"), _T("propagation_valid_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationValidMask; }, "lossless mask of propagation candidates with valid feasibility inputs");
					saveByte(_T("dvp_visible_normal_propagation_feasible_mask"), _T("propagation_feasible_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationFeasibleMask; }, "lossless mask of propagation candidates satisfying every direction");
					saveByte(_T("dvp_visible_normal_propagation_rejected_mask"), _T("propagation_rejected_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationRejectedMask; }, "lossless mask of valid propagation candidates rejected by a hemisphere constraint");
					saveByte(_T("dvp_visible_normal_propagation_native_best"), _T("propagation_native_best.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationNativeBest; }, "native lowest-cost propagation-candidate index; 255 means unavailable");
					saveByte(_T("dvp_visible_normal_propagation_constrained_best"), _T("propagation_constrained_best.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationConstrainedBest; }, "lowest-cost feasible propagation-candidate index; 255 means unavailable");
					saveByte(_T("dvp_visible_normal_propagation_selected"), _T("propagation_selected.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationSelected; }, "propagation-candidate index selected by the configured mode; 255 means unavailable");
					saveByte(_T("dvp_visible_normal_propagation_reason"), _T("propagation_reason.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationReason; }, "exact constrained-selection or native-fallback reason");
					saveByte(_T("dvp_visible_normal_propagation_fallback"), _T("propagation_fallback.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationFallback; }, "one when no feasible propagation candidate existed and native best was retained", {{"binary", true}});
					saveByte(_T("dvp_visible_normal_propagation_applied_constraint"), _T("propagation_applied_constraint.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationAppliedConstraint; }, "one when active mode selected a feasible propagation candidate instead of native best", {{"binary", true}});
					saveByte(_T("dvp_visible_normal_propagation_accepted"), _T("propagation_accepted.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.propagationAccepted; }, "one when the selected propagation candidate improved the active incumbent", {{"binary", true}});
					saveByte(_T("dvp_visible_normal_refinement_native_tested_mask"), _T("refinement_native_tested_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementNativeTestedMask; }, "two-bit mask of native stochastic normal proposals generated and tested");
					saveByte(_T("dvp_visible_normal_refinement_native_valid_mask"), _T("refinement_native_valid_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementNativeValidMask; }, "two-bit mask of native stochastic proposals with valid feasibility inputs");
					saveByte(_T("dvp_visible_normal_refinement_native_feasible_mask"), _T("refinement_native_feasible_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementNativeFeasibleMask; }, "two-bit mask of native stochastic proposals satisfying every direction");
					saveByte(_T("dvp_visible_normal_refinement_retry_success_mask"), _T("refinement_retry_success_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementRetrySuccessMask; }, "two-bit mask of stochastic proposals that found a bounded feasible retry");
					saveByte(_T("dvp_visible_normal_refinement_exhaustion_fallback_mask"), _T("refinement_exhaustion_fallback_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementExhaustionFallbackMask; }, "two-bit mask of proposals retaining native fallback after retry exhaustion");
					saveByte(_T("dvp_visible_normal_refinement_applied_retry_mask"), _T("refinement_applied_retry_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementAppliedRetryMask; }, "two-bit mask of proposals for which active mode scored a constrained retry");
					saveByte(_T("dvp_visible_normal_refinement_accepted_mask"), _T("refinement_accepted_mask.png"), [](const PatchMatchDVPVisibleNormalInstrumentUpdate& v) { return v.refinementAcceptedMask; }, "two-bit mask of scored stochastic proposals that improved the incumbent");
				}
			} else if (maps.visibleNormalRequested && maps.apdStageActive && maps.mapsRequested) {
				manifest["write_errors"].push_back(maps.visibleNormalMapsAvailable ?
					"dvp_visible_normal_map_capture_incomplete" :
					"dvp_visible_normal_map_capture_not_admitted");
			}

			const bool eventMapsComplete(
		maps.numLogicalStates > 0 && numPasses == 1 + (maps.numLogicalStates-1)*2 &&
		passCostImprovements.size() >= area * (size_t)numPasses &&
		maps.passDepthDeltas.size() >= area * (size_t)numPasses &&
		maps.passDepthRelDeltas.size() >= area * (size_t)numPasses &&
		maps.passNormalAngleDeltas.size() >= area * (size_t)numPasses &&
		maps.passViewChurn.size() >= area * (size_t)numPasses);
	if (eventMapsComplete) {
		std::vector<float> costImprovement(area), depthDelta(area), depthRelativeDelta(area), normalAngleDelta(area);
		for (int stateIndex = 0; stateIndex < maps.numLogicalStates; ++stateIndex) {
			const int firstPass(stateIndex == 0 ? 0 : 1 + (stateIndex-1)*2);
			const int secondPass(stateIndex == 0 ? -1 : firstPass+1);
			Image8U viewChurn(size);
			for (size_t i = 0; i < area; ++i) {
				costImprovement[i] = stateIndex == 0 ? 0.f : passCostImprovements[(size_t)firstPass*area+i];
				depthDelta[i] = maps.passDepthDeltas[(size_t)firstPass*area+i];
				depthRelativeDelta[i] = maps.passDepthRelDeltas[(size_t)firstPass*area+i];
				normalAngleDelta[i] = maps.passNormalAngleDeltas[(size_t)firstPass*area+i];
				unsigned churn(maps.passViewChurn[(size_t)firstPass*area+i]);
				if (secondPass >= 0) {
					costImprovement[i] += passCostImprovements[(size_t)secondPass*area+i];
					depthDelta[i] += maps.passDepthDeltas[(size_t)secondPass*area+i];
					depthRelativeDelta[i] += maps.passDepthRelDeltas[(size_t)secondPass*area+i];
					normalAngleDelta[i] += maps.passNormalAngleDeltas[(size_t)secondPass*area+i];
					churn += maps.passViewChurn[(size_t)secondPass*area+i];
				}
				viewChurn[(int)i] = (uint8_t)MINF(churn, 255u);
			}
			const int logicalIteration(stateIndex == 0 ? -1 : stateIndex-1);
			const String stateName(stateIndex == 0 ? _T("state00_initialization") : String::FormatString(_T("state%02d_iteration%02d"), stateIndex, stateIndex));
			const String relativeStateDir(String(_T("logical_states/")) + stateName + _T("/"));
			const String stateDir(depthMapDir + relativeStateDir);
			Util::ensureFolder(stateDir);
			const nlohmann::json metadata = {
				{"logical_iteration", logicalIteration}, {"stage", stateIndex == 0 ? "initialization" : "iteration"},
				{"stage_index", stateIndex}, {"uncompressed_bytes", (uint64_t)(area*sizeof(float))}
			};
			auto saveEvent = [&](const char* signal, const char* fileName, const std::vector<float>& values,
				const char* semantics, const char* quality, const char* basis,
				const nlohmann::json& extra = nlohmann::json::object()) {
				nlohmann::json eventMetadata(metadata);
				for (auto it = extra.begin(); it != extra.end(); ++it)
					eventMetadata[it.key()] = it.value();
				if (SaveInstrumentScalarMap(stateDir + fileName, size, values))
					addMap(signal, relativeStateDir + fileName, "float32", "logical_event", semantics, quality, basis, eventMetadata);
				else
					manifest["write_errors"].push_back(signal);
			};
			saveEvent("cost_improvement_exact", "cost_improvement_exact.pfm", costImprovement,
				stateIndex == 0 ?
					"nonnegative production cost reduction; defined as zero for initialization" :
					"nonnegative production cost reduction summed across the complete logical iteration",
				"derived_exact", "sum_of_disjoint_checkerboard_production_cost_reductions",
				{{"valid_min", 0.0}, {"aggregation", stateIndex == 0 ? "defined_zero" : "sum_of_disjoint_checkerboard_passes"},
				 {"checkerboard_identity_exposed", false},
				 {"limitations", "nonnegative stored-cost reduction; cost increases are zero and view selection can change the cost basis"}});
			saveEvent("depth_delta", "depth_delta.pfm", depthDelta,
				"signed depth change across the complete logical iteration", "exact", "post_pass_production_state_difference");
			saveEvent("depth_relative_delta", "depth_relative_delta.pfm", depthRelativeDelta,
				"signed depth change divided by incoming depth across the complete logical iteration", "exact", "post_pass_production_state_difference");
			saveEvent("normal_angle_delta", "normal_angle_delta.pfm", normalAngleDelta,
				"normal angular change in degrees across the complete logical iteration", "exact", "post_pass_production_state_difference");
			if (viewChurn.Save(stateDir + _T("view_churn.png")))
				addMap("view_churn", relativeStateDir + _T("view_churn.png"), "uint8", "logical_event", "number of selected views added or removed across the complete logical iteration", "exact", "post_pass_production_state_difference", metadata);
			else
				manifest["write_errors"].push_back("view_churn");
		}
	} else
		manifest["write_errors"].push_back("logical_event_capture_incomplete");
}

bool AppendInstrumentTimings(
	const String& dir,
	int imageID,
	int scaleNumber,
	const std::vector<float>& kernelTimingsMs)
{
	if (kernelTimingsMs.empty())
		return false;
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String fileName(dir + _T("timings.csv"));
	const bool writeHeader(!File::access(fileName));
	std::ofstream fs(fileName.c_str(), std::ios::app);
	if (!fs)
		return false;
	if (writeHeader)
		fs << "image_id,scale_number,pass_index,phase,iteration,kernel_ms\n";
	for (size_t pass = 0; pass < kernelTimingsMs.size(); ++pass) {
		const int phase(pass == 0 ? 0 : ((pass & 1) ? 1 : 2));
		const int iter(pass == 0 ? -1 : (int)((pass - 1) / 2));
		fs << imageID << ',' << scaleNumber << ',' << pass << ','
		   << InstrumentPhaseName(phase) << ',' << iter << ','
		   << kernelTimingsMs[pass] << '\n';
	}
	fs.flush();
	return (bool)fs;
}

nlohmann::json ConfiguredPatchMatchCUDAParametersJson()
{
	nlohmann::json sampleOffsets(nlohmann::json::array());
	for (int y = -PATCHMATCHCUDA_PATCH_HALF_WINDOW;
		 y <= PATCHMATCHCUDA_PATCH_HALF_WINDOW;
		 y += PATCHMATCHCUDA_PATCH_STEP) {
		for (int x = -PATCHMATCHCUDA_PATCH_HALF_WINDOW;
			 x <= PATCHMATCHCUDA_PATCH_HALF_WINDOW;
			 x += PATCHMATCHCUDA_PATCH_STEP) {
			sampleOffsets.push_back({x, y});
		}
	}
	nlohmann::json parameters = {
		{"geometric_weight", 0.1f},
		{"refine_depth_ratio", 0.005f},
		{"refine_normal_radians", 0.01f * (float)M_PI},
		{"low_texture_variance_max", 0.0025f},
		{"low_texture_decay_scale", 0.02f},
		{"view_samples", 32},
		{"init_top_k_configured", 3},
		{"reference_patch_layout", {
			{"schema_name", "openmvs.dmap.reference_patch_layout"},
			{"schema_version", 1},
			{"kind", "fixed_cartesian_grid"},
			{"coordinate_domain", "reference_pyramid_pixels"},
			{"sample_position", "integer_offset_from_pixel_center"},
			{"texel_center_offset", 0.5f},
			{"texture_address_mode_configured", "wrap"},
			{"texture_address_mode_effective", "clamp"},
			{"texture_address_mode_effective_basis", "cuda_runtime_unnormalized_wrap_is_clamped"},
			{"texture_coordinates_normalized", false},
			{"texture_filter_mode", "linear"},
			{"half_window_pixels", PATCHMATCHCUDA_PATCH_HALF_WINDOW},
			{"step_pixels", PATCHMATCHCUDA_PATCH_STEP},
			{"sample_count", sampleOffsets.size()},
			{"sample_offsets_pixels", sampleOffsets},
			{"layout_provenance", "observer_contract_source_checked_against_cuda_scoring_constants"},
			{"sample_locations_captured_by_kernel", false},
			{"sample_values_captured_by_kernel", false},
			{"source_view_footprints_captured_by_kernel", false}
		}}
	};
	parameters["adaptive_patch_deformation"] = APDContractJson(OPTDENSE::nPatchMatchCUDAAPD);
	PatchMatch::Params dvpParams;
	dvpParams.nDVPEpipolarFamily = OPTDENSE::nPatchMatchCUDADVPEpipolarFamily;
	dvpParams.fDVPEpipolarAlpha = OPTDENSE::fPatchMatchCUDADVPEpipolarAlpha;
	dvpParams.fDVPEpipolarBeta = OPTDENSE::fPatchMatchCUDADVPEpipolarBeta;
	dvpParams.nDVPEpipolarMu = OPTDENSE::nPatchMatchCUDADVPEpipolarMu;
	dvpParams.nDVPGlobalSearchRadius = OPTDENSE::nPatchMatchCUDADVPGlobalSearchRadius;
	dvpParams.fDVPReprojectionThreshold = OPTDENSE::fPatchMatchCUDADVPReprojectionThreshold;
	dvpParams.fDVPRelativeDepthThreshold = OPTDENSE::fPatchMatchCUDADVPRelativeDepthThreshold;
	parameters["dvp_epipolar"] = DVPContractJson(dvpParams);
	parameters["dvp_depth_edge"] = {
		{"schema_name", "openmvs.dmap.dvp_depth_edge_contract"},
		{"schema_version", 1},
		{"enabled", DVPDepthEdgeModeEnabled(OPTDENSE::nPatchMatchCUDADVPDepthEdgeMode)},
		{"mode", OPTDENSE::nPatchMatchCUDADVPDepthEdgeMode},
		{"stage", DVPDepthEdgeStageName(OPTDENSE::nPatchMatchCUDADVPDepthEdgeMode)},
		{"prior_directory_provided", !OPTDENSE::strPatchMatchCUDADVPDepthEdgePriorDir.empty()},
		{"direct_monocular_depth_use", false},
		{"paper_contract_label", "paper_mechanics_complete_openmvs"},
		{"author_code_exact_equivalence", false}
	};
	dvpParams.nDVPVisibilityMode = OPTDENSE::nPatchMatchCUDADVPVisibilityMode;
	dvpParams.fDVPVisibilityReprojectionThreshold =
		OPTDENSE::fPatchMatchCUDADVPVisibilityReprojectionThreshold;
	dvpParams.fDVPVisibilityRelativeDepthThreshold =
		OPTDENSE::fPatchMatchCUDADVPVisibilityRelativeDepthThreshold;
	parameters["dvp_visibility"] = DVPVisibilityContractJson(dvpParams);
	dvpParams.nDVPVisibleNormalMode = OPTDENSE::nPatchMatchCUDADVPVisibleNormalMode;
	dvpParams.fDVPVisibleNormalDotTolerance =
		OPTDENSE::fPatchMatchCUDADVPVisibleNormalDotTolerance;
	dvpParams.nDVPVisibleNormalAttempts = OPTDENSE::nPatchMatchCUDADVPVisibleNormalAttempts;
	parameters["dvp_visible_normal"] = DVPVisibleNormalContractJson(dvpParams);
	return parameters;
}

nlohmann::json EffectivePatchMatchCUDAParametersJson(const PatchMatch::Params& params)
{
	nlohmann::json metadata(ConfiguredPatchMatchCUDAParametersJson());
	metadata["source_view_count"] = params.nNumViews;
	metadata["estimation_iterations"] = params.nEstimationIters;
	metadata["geometric_consistency"] = params.bGeomConsistency;
	metadata["low_resolution_prior_available"] = params.bLowResProcessed;
	metadata["init_top_k_effective"] = params.nInitTopK;
	metadata["keep_cost_threshold"] = params.fThresholdKeepCost;
	metadata["adaptive_patch_deformation"] = APDContractJson(params.nAPDMode);
	metadata["adaptive_patch_deformation"]["multiscale_stage"] =
		APDMultiscaleStageJson(params);
	metadata["dvp_epipolar"] = DVPContractJson(params);
	metadata["dvp_depth_edge"] = {
		{"schema_name", "openmvs.dmap.dvp_depth_edge_contract"},
		{"schema_version", 1},
		{"enabled", DVPDepthEdgeModeEnabled(params.nDVPDepthEdgeMode)},
		{"mode", params.nDVPDepthEdgeMode},
		{"stage", DVPDepthEdgeStageName(params.nDVPDepthEdgeMode)},
		{"direct_monocular_depth_use", false},
		{"paper_contract_label", "paper_mechanics_complete_openmvs"},
		{"author_code_exact_equivalence", false}
	};
	metadata["dvp_visibility"] = DVPVisibilityContractJson(params);
	metadata["dvp_visible_normal"] = DVPVisibleNormalContractJson(params);
	return metadata;
}

bool WriteDMapRunMetadata(const String& root, int geometricIteration)
{
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	if (g_dmapMetadataRoots.find(root.c_str()) != g_dmapMetadataRoots.end())
		return true;
	Util::ensureFolder(root);
	Util::ensureFolder(root + _T("depthmaps/"));
	nlohmann::json metadata;
	metadata["schema_name"] = "openmvs.dmap.run";
	metadata["schema_version"] = PM_DMAP_INSTRUMENT_SCHEMA_VERSION;
	metadata["backend"] = "cuda";
	metadata["estimation_stage"] = DMapEstimationStageName(geometricIteration);
	metadata["geometric_iteration"] = geometricIteration >= 0 ?
		nlohmann::json(geometricIteration) : nlohmann::json(nullptr);
	metadata["geometric_consistency"] = geometricIteration >= 0;
	metadata["scope"] = "depth-map estimation and filtering only";
	metadata["fusion_instrumented"] = false;
	metadata["mesh_instrumented"] = false;
	metadata["cuda_patchmatch_parameters"] = ConfiguredPatchMatchCUDAParametersJson();
	const String effectiveLevel(OPTDENSE::strDMapInstrumentationLevel.ToLower());
	const bool writeMaps(InstrumentWriteMaps());
	const bool prefilterRequested(InstrumentPrefilterRequested());
	metadata["instrumentation"] = {
		{"enabled", true},
		{"level", OPTDENSE::strDMapInstrumentationLevel.c_str()},
		{"effective_level", effectiveLevel.c_str()},
		{"sample_rate", OPTDENSE::fDMapInstrumentationSampleRate},
		{"sample_seed", OPTDENSE::nDMapInstrumentationSampleSeed},
		{"sampling_hash", "32-bit integer avalanche over image_id XOR seed; thresholded at one-million buckets"},
		{"image_list", OPTDENSE::strDMapInstrumentationImageList.c_str()},
		{"write_maps", writeMaps},
		{"prefilter_requested", prefilterRequested},
		{"budget_policy", OPTDENSE::strDMapInstrumentationBudgetPolicy.c_str()},
		{"max_device_mib", OPTDENSE::nDMapInstrumentationMaxDeviceMB},
		{"max_host_mib", OPTDENSE::nDMapInstrumentationMaxHostMB},
		{"max_frame_storage_mib", OPTDENSE::nDMapInstrumentationMaxFrameStorageMB}
	};
	metadata["instrumentation"]["capabilities"] = {
		{"summary_json_csv", true},
		{"iteration_counters", true},
		{"kernel_timings", true},
		{"targeted_traces", OPTDENSE::nPatchMatchInstrumentLevel >= 2},
		{"bounded_prefilter_snapshot", prefilterRequested},
		{"full_resolution_maps", writeMaps},
		{"post_pass_component_rescore", true},
		{"logical_state_cost_evolution", writeMaps},
		{"per_iteration_state_deltas", writeMaps},
		{"exact_candidate_runner_up_gap", writeMaps && !OPTDENSE::strDMapInstrumentationDir.empty()},
		{"exact_per_view_reliability", writeMaps && !OPTDENSE::strDMapInstrumentationDir.empty()},
		{"exact_update_attribution", writeMaps && !OPTDENSE::strDMapInstrumentationDir.empty()},
		{"exact_capability_subject_to_per_frame_budget", true},
		{"cpu_neighbor_candidate_ranking", !OPTDENSE::strDMapInstrumentationDir.empty()},
		{"reference_patch_layout_contract", true},
		{"reference_patch_sample_locations", false},
		{"reference_patch_sample_values", false},
		{"source_view_patch_footprints", false}
	};
	metadata["instrumentation"]["capabilities"]["apd_exact_iteration_summary"] =
		APDModeEnabled(OPTDENSE::nPatchMatchCUDAAPD);
	metadata["instrumentation"]["capabilities"]["apd_exact_pixel_maps"] =
		writeMaps && APDModeEnabled(OPTDENSE::nPatchMatchCUDAAPD);
	metadata["instrumentation"]["capabilities"]["apd_targeted_profile_anchor_and_view_trace"] =
		OPTDENSE::nPatchMatchInstrumentLevel >= 2 &&
		APDModeEnabled(OPTDENSE::nPatchMatchCUDAAPD);
	const bool dvpEnabled(DVPEpipolarFamilyEnabled(OPTDENSE::nPatchMatchCUDADVPEpipolarFamily));
	metadata["instrumentation"]["capabilities"]["dvp_exact_iteration_summary"] = dvpEnabled;
	metadata["instrumentation"]["capabilities"]["dvp_exact_pixel_maps"] = writeMaps && dvpEnabled;
	metadata["instrumentation"]["capabilities"]["dvp_targeted_endpoint_and_candidate_trace"] =
		OPTDENSE::nPatchMatchInstrumentLevel >= 2 && dvpEnabled;
	metadata["reporting_contract"] = {
		{"mechanics_granularity", "logical PatchMatch iteration"},
		{"timing_granularity", "checkerboard phase"},
		{"map_granularity", "logical PatchMatch iteration"},
		{"aggregation", "initialization remains separate; complementary checkerboard events are combined before map export; checkerboard identity is exposed only for timings"}
	};
	metadata["storage_preflight_contract"] = {
		{"enabled_for", "every selected pyramid level, including summary-only and targeted-trace capture"},
		{"filesystem_queries_when_disabled_or_unselected", false},
		{"concurrency_model", "filesystem free bytes minus active in-process reservations under the instrumentation mutex"},
		{"cumulative_frame_model", "each level records storage committed by earlier levels and is checked against the frame cap"},
		{"full_resolution_priority", "coarse levels hold a long-lived reservation for the selected full-resolution tier"},
		{"degrade_order", "exact maps, legacy maps, bounded pre-filter snapshot, targeted traces, summary"},
		{"reservation_lifetime", "preflight through artifact publication"},
		{"records", "resource_plans.jsonl storage_preflight"}
	};
	metadata["publication_contract"] = {
		{"map_capture_marker", "depthmaps/<frame>/capture_complete.json"},
		{"prefilter_capture_marker", "depthmaps/<frame>/prefilter_capture_complete.json"},
		{"prefilter_capture_marker_condition", "one exact production pre-filter depth snapshot, complete prefilter_manifest.json, and frame/filter/view-support/scene summaries successfully published"},
		{"map_capture_marker_condition", "atomic map manifest complete and frame/scene summaries successfully published"},
		{"required_map_file_condition", "every manifest map is a non-symlink regular file with a statable nonzero size"},
		{"summary_marker", "depthmaps/<frame>/summary_complete.json"},
		{"summary_marker_maps_complete", false},
		{"stale_markers_removed_before_preflight", true},
		{"optional_filter_sidecars_bound_to_core_marker", false},
		{"optional_filter_sidecar_boundary", "postprocess_filters.json, confidence_adjustment.json, and filter_resource_plan.json are emitted after core PatchMatch publication and require independent validation"}
	};
	metadata["level_contract"] = {
		{"summary", {
			{"outputs", "JSON/CSV summaries, logical-iteration counters, and phase-level kernel timings"},
			{"overhead", "non-APD captures use device snapshots and one post-pass diagnostic rescore; APD exact aggregates use instrumented hot-kernel counters; no full-resolution map downloads"}
		}},
		{"prefilter", {
			{"outputs", "summary outputs plus one exact full-resolution depth snapshot immediately before production filtering"},
			{"overhead", "Process<false>, one terminal float4 device/host snapshot, one scalar PFM, and no logical/pass map retention"}
		}},
		{"debug", {
			{"outputs", "summary outputs plus configured targeted pixel traces"},
			{"overhead", "summary overhead plus trace buffers for configured pixels"}
		}},
		{"maps", {
			{"outputs", "debug outputs plus retained proxy maps, exact production cost/candidate/view maps, and logical-iteration report maps"},
			{"overhead", "Process<true> hot-kernel records, full device/host buffers, downloads, and artifact I/O; subject to per-frame budgets"}
		}}
	};
	metadata["disabled_contract"] = {
		{"default", true},
		{"activation", "non-empty --dmap-instrumentation-dir"},
		{"instrumentation_output", false},
		{"instrumentation_allocations", false},
		{"diagnostic_snapshots_or_kernels", false},
		{"instrumentation_device_state_changes", false}
	};
	metadata["measurement_model"] = "prefilter and lightweight non-mechanics summary use Process<false>; exact APD and DVP aggregate/deep capture uses Process<true> active-path records; checkerboard phases remain timings-only";
	metadata["candidate_accounting_mode"] = "exact in admitted generic maps captures; APD and DVP have independent exact active-objective aggregate/map/trace namespaces; terminal APD native refinement remains separately labeled";
	metadata["candidate_accepted_semantics"] = {
		{"initialization", ExactCandidateAcceptedSemantics(true)},
		{"iteration", ExactCandidateAcceptedSemantics(false)},
		{"cross_stage_comparison", "initialization accepted can exceed finite; iterative accepted counts are sequential improvement events rather than final winners"}
	};
	metadata["confidence_gap_mode"] = "exact ProcessPixel winner-versus-runner-up in admitted generic maps; exact APD working-domain and DVP incumbent/proposal-domain gaps in their independent captures; retained local-neighbor proxy otherwise";
	metadata["cuda_exact_kernel_resources"] = {
		{"measurement_status", "unavailable_without_build_bound_receipt"},
		{"measurement_values", nullptr},
		{"measurement_requirement", "publish register, stack, spill, and occupancy values only from a receipt bound to this executable and CUDA build"},
		{"device_budget_scope", "explicit observer-owned buffers only; excludes CUDA runtime stack-pool growth and is not a physical-free-memory guarantee"},
		{"allocation_failure_behavior", "CUDA_CHECK terminates the capture on an actual allocation failure; partial evidence is not published as complete"},
		{"production_resource_authority", "separate instrumentation-OFF DensifyPointCloud binary"},
		{"observer_process_false_resource_equivalent", false},
		{"qualification_rule", "summary and prefilter require output parity, timing qualification, and independent resource inspection; null observer pointers do not make the compiled observer specialization resource-identical"},
		{"requested_exact_stack_limit_bytes", PM_INSTRUMENT_EXACT_STACK_BYTES},
		{"stack_limit_behavior", "raised and restored only around exact enabled launches"}
	};
	metadata["exact_candidate_slots"] = {
		{"0", "current_or_initialization"}, {"1-8", "directional_propagation"},
		{"9", "depth_refinement"}, {"10", "normal_refinement"},
		{"11", "random_normal_refinement"}, {"12", "surface_normal_refinement"},
		{"13-20", "apd_immutable_anchor_propagation"}, {"21", "apd_fitted_plane"},
		{"22-23", "dvp_epipolar_proposals"},
		{"255", "unavailable"}
	};
	metadata["exact_view_decisions"] = {
		{"0", "unavailable"}, {"1", "selected_monte_carlo"}, {"2", "rejected_zero_score"},
		{"3", "rejected_not_sampled"}, {"4", "initialization_top_k"},
		{"5", "initialization_threshold_tie"}, {"6", "initialization_rejected"}
	};
	metadata["apd_view_selection_modes"] = nlohmann::json::object();
	for (int mode=0; mode<PM_APD_INSTRUMENT_VIEW_SELECTION_MODES; ++mode)
		metadata["apd_view_selection_modes"][std::to_string(mode)] =
			APDViewSelectionModeName(mode);
	metadata["candidate_source_enum"] = nlohmann::json::array();
	for (int i = DMAP_SOURCE_UNKNOWN; i <= DMAP_SOURCE_OTHER; ++i)
		metadata["candidate_source_enum"].push_back({{"code", i}, {"name", DMapCandidateSourceName(i)}});
	metadata["filtering_rejection_reason_enum"] = nlohmann::json::array();
	for (int i = DMAP_REJECTION_NONE_SURVIVED; i <= DMAP_REJECTION_OTHER; ++i)
		metadata["filtering_rejection_reason_enum"].push_back({{"code", i}, {"name", DMapRejectionReasonName(i)}});
	metadata["region_schema_reserved_for"] = {
		"plane_interiors",
		"line_bands",
		"textureless_regions",
		"high_texture_regions",
		"window_reflective_regions"
	};
	metadata["limitations"] = {
		"CPU PatchMatch path is not instrumented in this first pass; CUDA builds and CUDA depth estimation are covered.",
		"Dense point-cloud fusion, mesh reconstruction, mesh refinement, and texturing are intentionally not instrumented.",
		"View-propagation and sparse/existing initialization are not separately distinguished by the current CUDA source enum.",
		"iteration.csv median/p90/p95 cost fields are blank because per-iteration per-pixel score distributions are not retained online.",
		"Exact hot-kernel maps are available only for public maps-level captures admitted by the configured per-frame resource budgets.",
		"Enabled exact maps instantiate Process<true>, which materially increases registers/stack and can change occupancy or scheduling; bit-exact parity and sanitizer gates are mandatory.",
		"Summary/debug and budget-degraded map captures retain the post-pass proxy diagnostics and explicitly report exact signals unavailable."
	};
	if (!WriteJsonFile(root + _T("run_metadata.json"), metadata))
		return false;
	g_dmapMetadataRoots.emplace(root.c_str());
	return true;
}

float Ratio(uint64_t numerator, uint64_t denominator)
{
	return denominator ? float(double(numerator) / double(denominator)) : 0.f;
}

bool WriteDMapIterationCSV(
	const String& depthMapDir,
	int imageID,
	const String& imageName,
	int scaleNumber,
	const cv::Size& size,
	const std::vector<PatchMatchInstrumentCounters>& counters)
{
	const String fileName(depthMapDir + _T("iteration.csv"));
	const bool writeHeader(!File::access(fileName));
	std::ofstream fs(fileName.c_str(), std::ios::app);
	if (!fs)
		return false;
	if (writeHeader) {
		fs << "image_id,image_name,scale_level,iteration,num_pixels,valid_ratio,changed_ratio,candidates_tested,candidates_accepted,acceptance_rate,median_cost,p90_cost,p95_cost,mean_cost_delta,median_cost_delta,mean_abs_depth_delta,p90_abs_depth_delta,mean_normal_delta_deg,p90_normal_delta_deg,accepted_from_init,accepted_from_spatial_propagation,accepted_from_view_propagation,accepted_from_random_perturbation,accepted_from_refinement,accepted_from_prior_or_guidance,mean_cost,phase,pass_index\n";
	}
	const uint64_t numPixels((uint64_t)size.area());
	for (size_t pass = 0; pass < counters.size(); ++pass) {
		const int iter(pass == 0 ? -1 : (int)pass-1);
		const PatchMatchInstrumentCounters& c(counters[pass]);
		const uint32_t acceptedFromRefinement(
			c.updateSource[PM_SOURCE_REFINE_DEPTH] +
			c.updateSource[PM_SOURCE_REFINE_NORMAL] +
			c.updateSource[PM_SOURCE_REFINE_SURFACE_NORMAL]);
		const float meanCost(c.processed ? c.costSum / float(c.processed) : 0.f);
		const float meanCostDelta(c.processed ? c.costImprovementSum / float(c.processed) : 0.f);
		const float meanAbsDepthDelta(c.updateMagnitudeSamples ? c.depthAbsChangeSum / float(c.updateMagnitudeSamples) : 0.f);
		const float meanNormalDelta(c.updateMagnitudeSamples ? c.normalAngleSum / float(c.updateMagnitudeSamples) : 0.f);
		uint64_t candidatesTested(0), candidatesAccepted(0);
		for (int type = 0; type < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++type) {
			candidatesTested += c.candidateTested[type];
			candidatesAccepted += c.candidateAccepted[type];
		}
		fs << imageID << ',' << CsvEscape(imageName) << ',' << scaleNumber << ',' << iter << ','
		   << numPixels << ','
		   << Ratio(c.validDepth, numPixels) << ','
		   << Ratio(c.accepted, numPixels) << ','
		   << candidatesTested << ','
		   << candidatesAccepted << ','
		   << Ratio(candidatesAccepted, candidatesTested) << ','
		   << ",,," // median/p90/p95 cost unavailable
		   << meanCostDelta << ','
		   << ',' // median_cost_delta unavailable
		   << meanAbsDepthDelta << ','
		   << ',' // p90_abs_depth_delta unavailable
		   << meanNormalDelta << ','
		   << ',' // p90_normal_delta_deg unavailable
		   << c.updateSource[PM_SOURCE_INIT] << ','
		   << c.updateSource[PM_SOURCE_PROPAGATE] << ','
		   << ',' // view propagation unavailable
		   << c.updateSource[PM_SOURCE_REFINE_RANDOM_NORMAL] << ','
		   << acceptedFromRefinement << ','
		   << ',' // prior/guidance unavailable
		   << meanCost << ','
		   << (pass == 0 ? "initialization" : "iteration") << ','
		   << pass << '\n';
	}
	fs.flush();
	return (bool)fs;
}

std::vector<uint64_t> ComputeSupportHistogram(const DepthData& depthData)
{
	std::vector<uint64_t> histogram(5, 0);
	if (depthData.viewsMap.empty())
		return histogram;
	for (int i = 0; i < depthData.viewsMap.area(); ++i) {
		if (!depthData.depthMap.empty() && depthData.depthMap[i] <= 0) {
			++histogram[0];
			continue;
		}
		const ViewsID& views(depthData.viewsMap[i]);
		unsigned count(0);
		for (int j = 0; j < 4; ++j)
			if (views[j] != 255)
				++count;
		histogram[MINF((unsigned)histogram.size()-1, count)]++;
	}
	return histogram;
}

bool WriteDMapViewSupportCSV(const String& depthMapDir, const std::vector<uint64_t>& histogram)
{
	std::ofstream fs((depthMapDir + _T("view_support.csv")).c_str());
	if (!fs)
		return false;
	fs << "supporting_view_count,pixels\n";
	for (size_t i = 0; i < histogram.size(); ++i)
		fs << i << ',' << histogram[i] << '\n';
	fs.flush();
	return (bool)fs;
}

std::vector<float> CollectFiniteCosts(const ConfidenceMap& costMap, const DepthMap& depthMap)
{
	std::vector<float> costs;
	if (costMap.empty())
		return costs;
	costs.reserve((size_t)costMap.area());
	for (int i = 0; i < costMap.area(); ++i) {
		if (!depthMap.empty() && depthMap[i] <= 0)
			continue;
		const float cost(costMap[i]);
		if (ISFINITE(cost))
			costs.push_back(cost);
	}
	std::sort(costs.begin(), costs.end());
	return costs;
}

nlohmann::json CostStatsJson(const ConfidenceMap& costMap, const DepthMap& depthMap)
{
	nlohmann::json stats;
	const std::vector<float> costs(CollectFiniteCosts(costMap, depthMap));
	if (costs.empty()) {
		stats["count"] = 0;
		stats["mean"] = nullptr;
		stats["median"] = nullptr;
		stats["p90"] = nullptr;
		stats["p95"] = nullptr;
		return stats;
	}
	double sum(0.0);
	for (float cost : costs)
		sum += cost;
	auto percentile = [&costs](double p) {
		const size_t idx((size_t)CLAMP(std::round(p * double(costs.size()-1)), 0.0, double(costs.size()-1)));
		return costs[idx];
	};
	stats["count"] = costs.size();
	stats["mean"] = sum / double(costs.size());
	stats["median"] = percentile(0.5);
	stats["p90"] = percentile(0.9);
	stats["p95"] = percentile(0.95);
	stats["min"] = costs.front();
	stats["max"] = costs.back();
	return stats;
}

uint64_t CountValidDepth(const DepthMap& depthMap)
{
	uint64_t count(0);
	for (int i = 0; i < depthMap.area(); ++i)
		if (depthMap[i] > 0)
			++count;
	return count;
}

bool SaveDMapByteImage(const String& fileName, const cv::Size& size, const std::vector<uint8_t>& values, int maxValue)
{
	if (values.empty() || values.size() < (size_t)size.area())
		return false;
	Image8U image(size);
	const float scale(maxValue > 0 ? 255.f / float(maxValue) : 1.f);
	for (int i = 0; i < image.area(); ++i)
		image[i] = (uint8_t)CLAMP(float(values[(size_t)i]) * scale, 0.f, 255.f);
	return image.Save(fileName);
}

bool SaveDMapCostImage(const String& fileName, const ConfidenceMap& costMap)
{
	if (costMap.empty())
		return false;
	static constexpr float kBadCost = 1.2f;
	Image8U image(costMap.size());
	for (int i = 0; i < image.area(); ++i) {
		const float cost(costMap[i]);
		image[i] = ISFINITE(cost) ?
			(uint8_t)CLAMP(cost * 255.f / kBadCost, 0.f, 255.f) :
			(uint8_t)255;
	}
	return image.Save(fileName);
}

bool SaveDMapValidAfterImage(const String& fileName, const DepthMap& depthMap)
{
	if (depthMap.empty())
		return false;
	Image8U image(depthMap.size());
	for (int i = 0; i < image.area(); ++i)
		image[i] = depthMap[i] > 0 ? 255 : 0;
	return image.Save(fileName);
}

bool SaveDMapSupportImage(const String& fileName, const DepthData& depthData)
{
	if (depthData.viewsMap.empty())
		return false;
	Image8U image(depthData.viewsMap.size());
	for (int i = 0; i < image.area(); ++i) {
		if (!depthData.depthMap.empty() && depthData.depthMap[i] <= 0) {
			image[i] = 0;
			continue;
		}
		const ViewsID& views(depthData.viewsMap[i]);
		unsigned count(0);
		for (int j = 0; j < 4; ++j)
			if (views[j] != 255)
				++count;
		image[i] = (uint8_t)(count * 255u / 4u);
	}
	return image.Save(fileName);
}

bool SaveDMapCandidateSourceImage(const String& fileName, const cv::Size& size, const std::vector<uint8_t>& updateSources)
{
	if (updateSources.empty() || updateSources.size() < (size_t)size.area())
		return false;
	Image8U image(size);
	for (int i = 0; i < image.area(); ++i)
		image[i] = (uint8_t)(DMapCandidateSourceFromPatchMatch(updateSources[(size_t)i]) * 28);
	return image.Save(fileName);
}

bool SaveDMapLastChangedImage(
	const String& fileName,
	const cv::Size& size,
	int numPasses,
	const std::vector<uint8_t>& passUpdateSources)
{
	const size_t area((size_t)size.area());
	if (area == 0 || numPasses <= 0 || passUpdateSources.size() < area * (size_t)numPasses)
		return false;
	Image8U image(size);
	for (size_t i = 0; i < area; ++i) {
		int lastPass(-1);
		for (int pass = 0; pass < numPasses; ++pass) {
			const uint8_t source(passUpdateSources[(size_t)pass * area + i]);
			if (source != PM_SOURCE_NONE && source != PM_SOURCE_FILTERED)
				lastPass = pass;
		}
		image[(int)i] = lastPass < 0 ? 0 : (uint8_t)CLAMP((lastPass + 1) * 255 / numPasses, 0, 255);
	}
	return image.Save(fileName);
}

nlohmann::json CandidateAcceptanceJson(const std::vector<PatchMatchInstrumentCounters>& counters)
{
	uint64_t accepted[DMAP_SOURCE_OTHER + 1] = {};
	uint64_t tested[DMAP_SOURCE_OTHER + 1] = {};
	uint64_t finite[DMAP_SOURCE_OTHER + 1] = {};
	static constexpr int candidateToDMap[PM_INSTRUMENT_NUM_CANDIDATE_TYPES] = {
		DMAP_SOURCE_INIT_RANDOM,
		DMAP_SOURCE_SPATIAL_PROPAGATION,
		DMAP_SOURCE_RANDOM_PERTURBATION,
		DMAP_SOURCE_REFINEMENT,
	};
	for (const PatchMatchInstrumentCounters& c : counters) {
		for (int type = 0; type < PM_INSTRUMENT_NUM_CANDIDATE_TYPES; ++type) {
			const int code(candidateToDMap[type]);
			tested[code] += c.candidateTested[type];
			finite[code] += c.candidateFinite[type];
			accepted[code] += c.candidateAccepted[type];
		}
	}
	nlohmann::json rows(nlohmann::json::array());
	for (int code = DMAP_SOURCE_UNKNOWN; code <= DMAP_SOURCE_OTHER; ++code) {
		if (accepted[code] == 0 && tested[code] == 0)
			continue;
		nlohmann::json row;
		row["candidate_type"] = DMapCandidateSourceName(code);
		row["tested_count"] = tested[code];
		row["finite_count"] = finite[code];
		row["bad_count"] = tested[code] - MINF(tested[code], finite[code]);
		row["accepted_count"] = accepted[code];
		row["acceptance_rate"] = tested[code] ? nlohmann::json(double(accepted[code]) / double(tested[code])) : nlohmann::json(nullptr);
		row["finite_rate"] = tested[code] ? nlohmann::json(double(finite[code]) / double(tested[code])) : nlohmann::json(nullptr);
		row["counts_exact"] = true;
		row["median_cost_improvement"] = nullptr;
		row["median_abs_depth_change"] = nullptr;
		row["median_normal_change_deg"] = nullptr;
		rows.push_back(row);
	}
	return rows;
}

nlohmann::json FilteringJson(
	const DepthData& depthData,
	const std::vector<uint8_t>& validBeforeFilter,
	const std::vector<uint8_t>& filterRejectReasons,
	const std::vector<PatchMatchInstrumentCounters>& counters,
	uint64_t validAfterKeepCost,
	uint64_t rejectedByIgnoreMask,
	bool ignoreMaskRequested,
	int ignoreMaskLabel,
	bool ignoreMaskLoaded)
{
	const uint64_t total((uint64_t)depthData.depthMap.area());
	const uint64_t validAfter(CountValidDepth(depthData.depthMap));
	uint64_t validBefore(0);
	if (!validBeforeFilter.empty()) {
		for (uint8_t value : validBeforeFilter)
			if (value)
				++validBefore;
	} else if (!counters.empty()) {
		validBefore = counters.back().validDepth;
	} else {
		validBefore = validAfter;
	}
	validBefore = MINF(validBefore, total);
	validAfterKeepCost = MINF(validAfterKeepCost, total);
	validAfterKeepCost = MAXF(validAfterKeepCost, validAfter);
	const bool ignoreMaskCountAvailable(!ignoreMaskRequested || ignoreMaskLoaded);
	rejectedByIgnoreMask = MINF(rejectedByIgnoreMask, validAfterKeepCost-validAfter);
	const uint64_t invalidBefore(total - validBefore);
	const uint64_t rejected(validBefore > validAfter ? validBefore - validAfter : 0);
	const uint64_t rejectedByKeepCost(validBefore > validAfterKeepCost ? validBefore - validAfterKeepCost : 0);
	uint64_t reasons[DMAP_REJECTION_OTHER + 1] = {};
	if (!filterRejectReasons.empty()) {
		for (size_t i = 0; i < filterRejectReasons.size() && i < (size_t)depthData.depthMap.area(); ++i) {
			const bool wasValidBefore(validBeforeFilter.empty() ? filterRejectReasons[i] != DMAP_REJECTION_UNKNOWN_REJECTED : validBeforeFilter[i] != 0);
			const bool validAfterPixel(depthData.depthMap[(int)i] > 0);
			if (!wasValidBefore || validAfterPixel)
				continue;
			const int reason(CLAMP((int)filterRejectReasons[i], (int)DMAP_REJECTION_UNKNOWN_REJECTED, (int)DMAP_REJECTION_OTHER));
			reasons[reason]++;
		}
	} else {
		reasons[DMAP_REJECTION_LOW_SCORE] = rejectedByKeepCost;
		reasons[DMAP_REJECTION_MASKED] = rejectedByIgnoreMask;
	}
	nlohmann::json reasonJson;
	reasonJson["low_score"] = reasons[DMAP_REJECTION_LOW_SCORE];
	reasonJson["insufficient_view_support"] = reasons[DMAP_REJECTION_INSUFFICIENT_VIEW_SUPPORT];
	reasonJson["geometric_inconsistency"] = reasons[DMAP_REJECTION_GEOMETRIC_INCONSISTENCY];
	reasonJson["normal_inconsistency"] = reasons[DMAP_REJECTION_NORMAL_INCONSISTENCY];
	reasonJson["depth_range"] = reasons[DMAP_REJECTION_DEPTH_RANGE];
	reasonJson["occlusion"] = reasons[DMAP_REJECTION_OCCLUSION];
	reasonJson["masked"] = ignoreMaskCountAvailable ?
		nlohmann::json(reasons[DMAP_REJECTION_MASKED]) : nlohmann::json(nullptr);
	reasonJson["small_component"] = reasons[DMAP_REJECTION_SMALL_COMPONENT];
	reasonJson["unknown"] = reasons[DMAP_REJECTION_UNKNOWN_REJECTED];
	const String ignoreMaskPath(ignoreMaskRequested && depthData.GetView().pImageData ?
		depthData.GetView().pImageData->GetMaskFileName() : String());
	const String ignoreMaskUnavailableReason(ignoreMaskRequested && !ignoreMaskLoaded ?
		String::FormatString(_T("requested ignore mask label %d could not be loaded from '%s'"),
			ignoreMaskLabel, ignoreMaskPath.c_str()) : String());
	return {
		{"num_pixels_total", total},
		{"num_valid_before_filter", validBefore},
		{"num_valid_after_keep_cost_filter", validAfterKeepCost},
		{"num_valid_after_filter", validAfter},
		{"num_invalid_before_filter", invalidBefore},
		{"num_rejected_by_keep_cost_filter", rejectedByKeepCost},
		{"num_rejected_by_ignore_mask", ignoreMaskCountAvailable ?
			nlohmann::json(rejectedByIgnoreMask) : nlohmann::json(nullptr)},
		{"num_rejected_by_filter", rejected},
		{"valid_ratio_before_filter", Ratio(validBefore, total)},
		{"valid_ratio_after_keep_cost_filter", Ratio(validAfterKeepCost, total)},
		{"valid_ratio_after_filter", Ratio(validAfter, total)},
		{"ignore_mask", {
			{"requested", ignoreMaskRequested},
			{"label", ignoreMaskRequested ? nlohmann::json(ignoreMaskLabel) : nlohmann::json(nullptr)},
			{"load_attempted", ignoreMaskRequested},
			{"loaded", ignoreMaskRequested ? nlohmann::json(ignoreMaskLoaded) : nlohmann::json(nullptr)},
			{"status", !ignoreMaskRequested ? "not_requested" : ignoreMaskLoaded ? "loaded" : "unavailable"},
			{"source_path", ignoreMaskRequested ? nlohmann::json(ignoreMaskPath.c_str()) : nlohmann::json(nullptr)},
			{"rejection_count_available", ignoreMaskCountAvailable},
			{"unavailable_reason", ignoreMaskUnavailableReason.c_str()}
		}},
		{"rejection_reasons", reasonJson}
	};
}

bool WriteDMapSceneSummary(const String& root, const nlohmann::json& summary, int geometricIteration)
{
	std::lock_guard<std::mutex> lock(g_instrumentMutex);
	const String fileName(root + _T("scene_summary.json"));
	nlohmann::json scene;
	if (File::access(fileName)) {
		std::ifstream fs(fileName.c_str());
		scene = nlohmann::json::parse(fs, nullptr, false);
		if (scene.is_discarded() || !scene.is_object())
			scene = nlohmann::json::object();
	}
	if (!scene.is_object())
		scene = nlohmann::json::object();
	scene["schema_name"] = "openmvs.dmap.scene_summary";
	scene["schema_version"] = PM_DMAP_INSTRUMENT_SCHEMA_VERSION;
	scene["backend"] = "cuda";
	scene["estimation_stage"] = DMapEstimationStageName(geometricIteration);
	scene["geometric_iteration"] = geometricIteration >= 0 ?
		nlohmann::json(geometricIteration) : nlohmann::json(nullptr);
	scene["geometric_consistency"] = geometricIteration >= 0;
	scene["depthmaps_by_image"] = scene.value("depthmaps_by_image", nlohmann::json::object());
	const std::string imageID(std::to_string(summary.value("image_id", -1)));
	scene["depthmaps_by_image"][imageID] = summary;
	scene["num_depthmaps"] = scene["depthmaps_by_image"].size();
	uint64_t totalPixels(0), validBefore(0), validAfterKeepCost(0), validAfter(0);
	uint64_t rejectedByKeepCost(0), rejectedByIgnoreMask(0), rejected(0);
	uint64_t ignoreMaskRequestedFrames(0), ignoreMaskLoadedFrames(0), ignoreMaskUnavailableFrames(0);
	bool ignoreMaskCountsAvailable(true);
	for (auto it = scene["depthmaps_by_image"].begin(); it != scene["depthmaps_by_image"].end(); ++it) {
		const nlohmann::json& item(it.value());
		totalPixels += item.value("num_pixels_total", 0ull);
		validBefore += item.value("num_valid_before_filter", 0ull);
		validAfterKeepCost += item.value("num_valid_after_keep_cost_filter", item.value("num_valid_after_filter", 0ull));
		validAfter += item.value("num_valid_after_filter", 0ull);
		rejectedByKeepCost += item.value("num_rejected_by_keep_cost_filter", item.value("num_rejected_by_filter", 0ull));
		const nlohmann::json ignoreMask(item.value("ignore_mask", nlohmann::json::object()));
		const bool ignoreMaskRequested(ignoreMask.value("requested", false));
		const auto ignoreMaskLoadedIt(ignoreMask.find("loaded"));
		const bool ignoreMaskLoaded(
			ignoreMaskLoadedIt != ignoreMask.end() &&
			ignoreMaskLoadedIt->is_boolean() &&
			ignoreMaskLoadedIt->get<bool>());
		ignoreMaskRequestedFrames += ignoreMaskRequested;
		ignoreMaskLoadedFrames += ignoreMaskRequested && ignoreMaskLoaded;
		ignoreMaskUnavailableFrames += ignoreMaskRequested && !ignoreMaskLoaded;
		const auto rejectedMaskIt(item.find("num_rejected_by_ignore_mask"));
		if (rejectedMaskIt != item.end() && rejectedMaskIt->is_number_unsigned())
			rejectedByIgnoreMask += rejectedMaskIt->get<uint64_t>();
		else if (ignoreMaskRequested || (rejectedMaskIt != item.end() && rejectedMaskIt->is_null()))
			ignoreMaskCountsAvailable = false;
		rejected += item.value("num_rejected_by_filter", 0ull);
	}
	scene["aggregate"] = {
		{"num_pixels_total", totalPixels},
		{"num_valid_before_filter", validBefore},
		{"num_valid_after_keep_cost_filter", validAfterKeepCost},
		{"num_valid_after_filter", validAfter},
		{"num_rejected_by_keep_cost_filter", rejectedByKeepCost},
		{"num_rejected_by_ignore_mask", ignoreMaskCountsAvailable ?
			nlohmann::json(rejectedByIgnoreMask) : nlohmann::json(nullptr)},
		{"num_rejected_by_filter", rejected},
		{"valid_ratio_before_filter", Ratio(validBefore, totalPixels)},
		{"valid_ratio_after_keep_cost_filter", Ratio(validAfterKeepCost, totalPixels)},
		{"valid_ratio_after_filter", Ratio(validAfter, totalPixels)},
		{"rejected_ratio", Ratio(rejected, totalPixels)},
		{"ignore_mask", {
			{"requested_frames", ignoreMaskRequestedFrames},
			{"loaded_frames", ignoreMaskLoadedFrames},
			{"unavailable_frames", ignoreMaskUnavailableFrames},
			{"rejection_counts_available", ignoreMaskCountsAvailable}
		}}
	};
	return WriteJsonFile(fileName, scene);
}

void WriteDMapArtifacts(
	const String& root,
	const DepthData& depthData,
	const PatchMatch::Params& params,
	int scaleNumber,
	int numPasses,
	const std::vector<PatchMatchInstrumentCounters>& counters,
	const ConfidenceMap& costMap,
	const std::vector<uint8_t>& updateSources,
	const std::vector<uint8_t>& validBeforeFilter,
	const std::vector<uint8_t>& filterRejectReasons,
	const std::vector<float>& passCostImprovements,
	const std::vector<uint8_t>& passUpdateSources,
	const std::vector<InstrumentSidecarWriteError>& sidecarWriteErrors,
	bool writeMaps,
	const InstrumentExtendedMaps* extendedMaps,
	int geometricIteration,
	uint64_t validAfterKeepCost,
	uint64_t rejectedByIgnoreMask,
	bool ignoreMaskRequested,
	int ignoreMaskLabel,
	bool ignoreMaskLoaded)
{
	if (scaleNumber != 0)
		return;
	const int imageID((int)depthData.GetView().GetID());
	const String imageName(depthData.GetView().pImageData->name);
	const String depthMapDir(DMapDepthMapDir(root, imageID, imageName));
	const String mapsDir(depthMapDir + _T("maps/"));
	std::vector<InstrumentSidecarWriteError> publicationWriteErrors(sidecarWriteErrors);
	const bool prefilterRequested(extendedMaps && extendedMaps->prefilterRequested);
	const bool prefilterAvailable(extendedMaps && extendedMaps->prefilterAvailable);
	if (writeMaps || prefilterRequested)
		Util::ensureFolder(mapsDir);
	const std::vector<uint64_t> supportHistogram(ComputeSupportHistogram(depthData));
	const bool viewSupportWritten(WriteDMapViewSupportCSV(depthMapDir, supportHistogram));
	if (!viewSupportWritten)
		RecordInstrumentSidecarWriteError(publicationWriteErrors, "view_support.csv", scaleNumber);
	const nlohmann::json filtering(FilteringJson(
		depthData, validBeforeFilter, filterRejectReasons, counters,
		validAfterKeepCost, rejectedByIgnoreMask,
		ignoreMaskRequested, ignoreMaskLabel, ignoreMaskLoaded));
	const bool filteringWritten(WriteJsonFile(depthMapDir + _T("filtering.json"), filtering));
	if (!filteringWritten)
		RecordInstrumentSidecarWriteError(publicationWriteErrors, "filtering.json", scaleNumber);

	nlohmann::json selectedViews(nlohmann::json::array());
	for (IIndex i = 1; i < depthData.images.size(); ++i) {
		const DepthData::ViewData& view(depthData.images[i]);
		selectedViews.push_back({
			{"id", view.GetID()},
			{"name", view.pImageData ? view.pImageData->name.c_str() : ""}
		});
	}
	nlohmann::json missingMaps(nlohmann::json::array());
	const bool exactAvailable(extendedMaps && extendedMaps->exactAvailable);
	const bool apdEnabled(APDModeEnabled(params.nAPDMode));
	const bool apdFullMode(APDModeUsesFullMechanics(params.nAPDMode));
	const bool apdDeformationOnly(
		params.nAPDMode == static_cast<unsigned>(APDMode::DEFORMATION_ONLY));
	const bool apdAvailable(extendedMaps && extendedMaps->apdAvailable &&
		extendedMaps->apdStageActive);
	const bool apdMapsAvailable(extendedMaps && extendedMaps->apdMapsAvailable &&
		extendedMaps->apdStageActive);
	const bool dvpEnabled(DVPEpipolarFamilyEnabled(params.nDVPEpipolarFamily));
	const bool dvpAvailable(extendedMaps && extendedMaps->dvpAvailable);
	const bool dvpMapsAvailable(extendedMaps && extendedMaps->dvpMapsAvailable);
	const bool visibilityEnabled(DVPVisibilityModeEnabled(params.nDVPVisibilityMode));
	const bool visibilityAvailable(extendedMaps && extendedMaps->visibilityAvailable &&
		extendedMaps->apdStageActive);
	const bool visibilityMapsAvailable(extendedMaps && extendedMaps->visibilityMapsAvailable &&
		extendedMaps->apdStageActive);
	const bool visibleNormalEnabled(
		DVPVisibleNormalModeEnabled(params.nDVPVisibleNormalMode));
	const bool visibleNormalAvailable(extendedMaps && extendedMaps->visibleNormalAvailable &&
		extendedMaps->apdStageActive);
	const bool visibleNormalMapsAvailable(
		extendedMaps && extendedMaps->visibleNormalMapsAvailable &&
		extendedMaps->apdStageActive);
	missingMaps.push_back("depth_initial");
	if (!writeMaps) {
		if (!prefilterAvailable)
			missingMaps.push_back("depth_final_before_filter");
		missingMaps.push_back("cost_photometric");
		missingMaps.push_back("cost_geometric");
		missingMaps.push_back("confidence_gap");
		missingMaps.push_back(prefilterRequested ?
			"all per-pixel maps except the bounded pre-filter depth snapshot" :
			"all per-pixel maps because --dmap-instrumentation-write-maps=0 and level is not maps");
	} else {
		if (validBeforeFilter.empty())
			missingMaps.push_back("valid_before_filter");
		if (filterRejectReasons.empty())
			missingMaps.push_back("rejection_reason");
		if (passUpdateSources.empty())
			missingMaps.push_back("last_changed_iter");
		if (!exactAvailable)
			missingMaps.push_back(extendedMaps && !extendedMaps->exactUnavailableReason.empty() ?
					extendedMaps->exactUnavailableReason.c_str() : "exact hot-kernel maps unavailable");
		if (dvpEnabled && !dvpMapsAvailable)
			missingMaps.push_back("DVP exact full-frame pixel mechanics unavailable");
		if (visibilityEnabled && !visibilityMapsAvailable)
			missingMaps.push_back("DVP visibility exact full-frame state mechanics unavailable");
		if (visibleNormalEnabled && !visibleNormalMapsAvailable)
			missingMaps.push_back("DVP visible-normal exact full-frame decision mechanics unavailable");
	}
	nlohmann::json summary;
	summary["schema_name"] = "openmvs.dmap.frame_summary";
	summary["schema_version"] = PM_DMAP_INSTRUMENT_SCHEMA_VERSION;
	summary["estimation_stage"] = DMapEstimationStageName(geometricIteration);
	summary["geometric_iteration"] = geometricIteration >= 0 ?
		nlohmann::json(geometricIteration) : nlohmann::json(nullptr);
	summary["geometric_consistency"] = geometricIteration >= 0;
	summary["image_id"] = imageID;
	summary["image_name"] = imageName.c_str();
	summary["safe_image_name"] = DMapSafeName(imageName).c_str();
	summary["scale_level"] = scaleNumber;
	summary["width"] = depthData.depthMap.cols;
	summary["height"] = depthData.depthMap.rows;
	summary["cuda_patchmatch_parameters"] = EffectivePatchMatchCUDAParametersJson(params);
	summary["num_pixels_total"] = filtering["num_pixels_total"];
	summary["num_valid_before_filter"] = filtering["num_valid_before_filter"];
	summary["num_valid_after_keep_cost_filter"] = filtering["num_valid_after_keep_cost_filter"];
	summary["num_valid_after_filter"] = filtering["num_valid_after_filter"];
	summary["num_invalid_before_filter"] = filtering["num_invalid_before_filter"];
	summary["num_rejected_by_keep_cost_filter"] = filtering["num_rejected_by_keep_cost_filter"];
	summary["num_rejected_by_ignore_mask"] = filtering["num_rejected_by_ignore_mask"];
	summary["num_rejected_by_filter"] = filtering["num_rejected_by_filter"];
	summary["ignore_mask"] = filtering["ignore_mask"];
	summary["valid_ratio_before_filter"] = filtering["valid_ratio_before_filter"];
	summary["valid_ratio_after_keep_cost_filter"] = filtering["valid_ratio_after_keep_cost_filter"];
	summary["valid_ratio_after_filter"] = filtering["valid_ratio_after_filter"];
	summary["rejected_by_filter_ratio"] = Ratio(filtering["num_rejected_by_filter"].get<uint64_t>(), filtering["num_pixels_total"].get<uint64_t>());
	summary["final_cost"] = CostStatsJson(costMap, depthData.depthMap);
	summary["selected_source_views"] = selectedViews;
	summary["supporting_view_histogram"] = supportHistogram;
	auto updateSummaryObserverReceipt = [&]() {
		summary["observer_sidecars"] = {
			{"complete", publicationWriteErrors.empty()},
			{"write_error_count", publicationWriteErrors.size()},
			{"write_errors", InstrumentSidecarWriteErrorsJson(publicationWriteErrors)}
		};
	};
	updateSummaryObserverReceipt();
	summary["prefilter_capture"] = {
		{"requested", prefilterRequested},
		{"available", prefilterAvailable},
		{"process_specialization", prefilterRequested ? "Process<false>" : "not_requested"},
		{"artifact", prefilterAvailable ? nlohmann::json("maps/depth_final_before_filter.pfm") : nlohmann::json(nullptr)},
		{"unavailable_reason", extendedMaps ? extendedMaps->prefilterUnavailableReason.c_str() : "resource plan unavailable"}
	};
	summary["apd_observability"] = APDObservabilityJson(params.nAPDMode, extendedMaps);
	summary["dvp_observability"] = DVPObservabilityJson(params, extendedMaps);
	summary["dvp_visibility_observability"] =
		DVPVisibilityObservabilityJson(params, extendedMaps);
	summary["dvp_visible_normal_observability"] =
		DVPVisibleNormalObservabilityJson(params, extendedMaps);
	summary["apd_multiscale"] = extendedMaps && extendedMaps->apdMultiscale.is_object() &&
		!extendedMaps->apdMultiscale.empty() ?
		extendedMaps->apdMultiscale : APDMultiscaleStageJson(params);
	summary["candidate_acceptance"] = CandidateAcceptanceJson(counters);
	summary["candidate_accounting_mode"] = apdAvailable && dvpAvailable ?
		(apdMapsAvailable && dvpMapsAvailable ? "exact_apd_and_dvp_active_objectives_full_frame" : "exact_apd_and_dvp_active_objectives_aggregate") :
		apdAvailable ? (apdMapsAvailable ? "exact_apd_working_objective_full_frame" : "exact_apd_working_objective_aggregate") :
		dvpAvailable ? (dvpMapsAvailable ? "exact_dvp_active_objective_full_frame" : "exact_dvp_active_objective_aggregate") :
		(exactAvailable ? "exact_production_hot_kernel_full_frame" : "unavailable_post_pass_snapshot");
	summary["confidence_gap_mode"] = apdAvailable && dvpAvailable ?
		(apdMapsAvailable && dvpMapsAvailable ? "exact_apd_and_dvp_stage_specific_full_frame" : "exact_apd_and_dvp_stage_specific_aggregate") :
		apdAvailable ? (apdMapsAvailable ? "exact_apd_working_winner_runner_up_full_frame" : "exact_apd_working_winner_runner_up_aggregate") :
		dvpAvailable ? (dvpMapsAvailable ? "exact_dvp_incumbent_proposal_runner_up_full_frame" : "exact_dvp_incumbent_proposal_runner_up_aggregate") :
		(exactAvailable ? "exact_process_pixel_winner_runner_up_full_frame" : "post_pass_current_plus_eight_neighbors");
	summary["unavailable_signals"] = nlohmann::json::array();
	if (!exactAvailable && !apdAvailable && !dvpAvailable) {
		summary["unavailable_signals"].push_back("candidate_family_tested_finite_accepted");
		summary["unavailable_signals"].push_back("exact_propagation_vs_refinement_acceptance");
		summary["unavailable_signals"].push_back("exact_same_pass_runner_up_gap");
		summary["unavailable_signals"].push_back("exact_per_view_reliability_and_contributions");
	}
	if (apdEnabled && !apdAvailable)
		summary["unavailable_signals"].push_back("apd_exact_iteration_mechanics");
	if (apdEnabled && !apdMapsAvailable)
		summary["unavailable_signals"].push_back("apd_exact_full_frame_pixel_mechanics");
	if (apdEnabled) {
		summary["unavailable_signals"].push_back("apd_historical_coarsest_monte_carlo_view_weights");
		summary["unavailable_signals"].push_back("apd_multiscale_resume_state_across_process_restart");
	}
	if (dvpEnabled && !dvpAvailable)
		summary["unavailable_signals"].push_back("dvp_exact_iteration_mechanics");
	if (dvpEnabled && !dvpMapsAvailable)
		summary["unavailable_signals"].push_back("dvp_exact_full_frame_pixel_mechanics");
	if (dvpEnabled && (!extendedMaps || !extendedMaps->traceAvailable || extendedMaps->dvpTraces.empty()))
		summary["unavailable_signals"].push_back("dvp_targeted_per_view_endpoint_samples");
	if (visibilityEnabled && !visibilityAvailable)
		summary["unavailable_signals"].push_back("dvp_visibility_exact_iteration_mechanics");
	if (visibilityEnabled && !visibilityMapsAvailable)
		summary["unavailable_signals"].push_back("dvp_visibility_exact_full_frame_state_mechanics");
	if (visibilityEnabled && (!extendedMaps || !extendedMaps->traceAvailable ||
		extendedMaps->visibilityTraces.empty()))
	{
		summary["unavailable_signals"].push_back("dvp_visibility_targeted_per_view_weights_and_reasons");
	}
	if (visibleNormalEnabled && !visibleNormalAvailable)
		summary["unavailable_signals"].push_back("dvp_visible_normal_exact_iteration_mechanics");
	if (visibleNormalEnabled && !visibleNormalMapsAvailable)
		summary["unavailable_signals"].push_back("dvp_visible_normal_exact_full_frame_decision_mechanics");
	if (visibleNormalEnabled && (!extendedMaps || !extendedMaps->traceAvailable ||
		extendedMaps->visibleNormalTraces.empty()))
	{
		summary["unavailable_signals"].push_back(
			"dvp_visible_normal_targeted_actual_normal_vectors");
	}
	if (ignoreMaskRequested && !ignoreMaskLoaded)
		summary["unavailable_signals"].push_back("ignore_mask_rejection_count");
	if (extendedMaps) {
			summary["resource_plan"] = {
				{"decision", extendedMaps->resourceDecision.c_str()},
				{"summary_available", extendedMaps->summaryAvailable},
				{"trace_requested", extendedMaps->traceRequested},
				{"trace_available", extendedMaps->traceAvailable},
				{"trace_unavailable_reason", extendedMaps->traceUnavailableReason.c_str()},
			{"maps_requested", extendedMaps->mapsRequested},
			{"maps_available", extendedMaps->mapsAvailable},
			{"exact_requested", extendedMaps->exactRequested},
			{"exact_available", extendedMaps->exactAvailable},
			{"prefilter_requested", extendedMaps->prefilterRequested},
			{"prefilter_available", extendedMaps->prefilterAvailable},
				{"apd_requested", extendedMaps->apdRequested},
				{"apd_stage_active", extendedMaps->apdStageActive},
				{"apd_summary_available", extendedMaps->apdAvailable && extendedMaps->apdStageActive},
				{"apd_maps_available", extendedMaps->apdMapsAvailable && extendedMaps->apdStageActive},
					{"apd_multiscale_maps_available", extendedMaps->apdRequested && extendedMaps->mapsAvailable},
					{"dvp_requested", extendedMaps->dvpRequested},
					{"dvp_summary_available", extendedMaps->dvpAvailable},
					{"dvp_maps_available", extendedMaps->dvpMapsAvailable},
					{"dvp_visibility_requested", extendedMaps->visibilityRequested},
					{"dvp_visibility_summary_available", visibilityAvailable},
					{"dvp_visibility_maps_available", visibilityMapsAvailable},
					{"dvp_visible_normal_requested", extendedMaps->visibleNormalRequested},
					{"dvp_visible_normal_summary_available", visibleNormalAvailable},
					{"dvp_visible_normal_maps_available", visibleNormalMapsAvailable},
			{"prefilter_unavailable_reason", extendedMaps->prefilterUnavailableReason.c_str()},
			{"exact_unavailable_reason", extendedMaps->exactUnavailableReason.c_str()},
			{"estimated_device_bytes", extendedMaps->estimatedDeviceBytes},
			{"estimated_host_bytes", extendedMaps->estimatedHostBytes},
				{"estimated_current_pyramid_storage_bytes", extendedMaps->estimatedStorageBytes},
				{"estimated_cumulative_frame_storage_bytes", extendedMaps->frameStorageBudgetBytes},
				{"frame_storage_committed_before_bytes", extendedMaps->frameStorageCommittedBeforeBytes},
				{"full_resolution_priority_reserve_bytes", extendedMaps->frameStoragePriorityReserveBytes},
				{"storage_preflight", {
				{"attempted", extendedMaps->storagePreflightAttempted},
				{"succeeded", extendedMaps->storagePreflightSucceeded},
				{"available_bytes", extendedMaps->storageAvailableBytes},
				{"reserved_before_bytes", extendedMaps->storageReservedBeforeBytes},
				{"effective_available_bytes", extendedMaps->storageEffectiveAvailableBytes},
					{"requested_bytes", extendedMaps->storageRequestedBytes},
					{"requested_plus_priority_reserve_bytes", SaturatingAdd(
						extendedMaps->storageRequestedBytes,
						extendedMaps->frameStoragePriorityReserveBytes)},
					{"reservation_bytes", extendedMaps->storageReservationBytes},
					{"frame_priority_reservation_bytes", extendedMaps->storagePriorityReservationBytes},
					{"frame_priority_reservation_consumed", extendedMaps->storagePriorityReservationConsumed},
				{"decision", extendedMaps->storagePreflightDecision.c_str()},
				{"reason", extendedMaps->storagePreflightReason.c_str()}
			}}
		};
	}
	summary["missing_maps"] = missingMaps;
	summary["region_stats"] = nlohmann::json::object();
	summary["region_stats"]["global"] = {
		{"valid_ratio_before_filter", summary["valid_ratio_before_filter"]},
		{"valid_ratio_after_keep_cost_filter", summary["valid_ratio_after_keep_cost_filter"]},
		{"valid_ratio_after_filter", summary["valid_ratio_after_filter"]},
		{"rejected_by_filter_ratio", summary["rejected_by_filter_ratio"]}
	};
	summary["completion_marker"] = writeMaps ? nlohmann::json({
		{"schema_name", "openmvs.dmap.capture_complete"},
		{"schema_version", 1},
		{"path", "capture_complete.json"},
		{"maps_complete", false},
		{"eligible", false},
		{"state", "pending_map_publication"}
	}) : prefilterRequested ? nlohmann::json({
		{"schema_name", "openmvs.dmap.prefilter_capture_complete"},
		{"schema_version", 1},
		{"path", "prefilter_capture_complete.json"},
		{"maps_complete", false},
		{"prefilter_complete", false},
		{"eligible", false},
		{"state", "pending_prefilter_publication"}
	}) : nlohmann::json({
		{"schema_name", "openmvs.dmap.summary_complete"},
		{"schema_version", 1},
		{"path", "summary_complete.json"},
		{"maps_complete", false},
		{"eligible", false},
		{"state", "pending_summary_publication"}
	});
	const nlohmann::json sidecarWriteErrorsJson(
		InstrumentSidecarWriteErrorsJson(publicationWriteErrors));

	bool mapManifestComplete(false);
	bool mapManifestWritten(false);
	bool prefilterManifestComplete(false);
	bool prefilterManifestWritten(false);
	if (writeMaps) {
		nlohmann::json manifest = {
			{"schema_name", "openmvs.dmap.map_manifest"},
			{"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"geometric_consistency", geometricIteration >= 0},
			{"width", depthData.depthMap.cols},
			{"height", depthData.depthMap.rows},
			{"num_passes", numPasses},
			{"num_iterations", extendedMaps ? std::max(0, extendedMaps->numLogicalStates-1) : 0},
			{"num_logical_states", extendedMaps ? extendedMaps->numLogicalStates : 0},
			{"map_granularity", "logical_iteration"},
			{"measurement_model", "production hot-kernel exact observability plus retained state snapshots and explicitly classified post-pass proxies"},
			{"completion_marker", {
				{"schema_name", "openmvs.dmap.capture_complete"},
				{"schema_version", 1},
				{"path", "capture_complete.json"},
				{"atomic_publish", true},
				{"written_only_after_complete_manifest", true}
			}},
			{"exact_capture", {
				{"requested", extendedMaps && extendedMaps->exactRequested},
				{"available", exactAvailable},
				{"unavailable_reason", extendedMaps ? extendedMaps->exactUnavailableReason.c_str() : "resource plan unavailable"},
				{"num_views", extendedMaps ? extendedMaps->numViews : 0},
				{"candidate_slots", PM_INSTRUMENT_EXACT_NUM_CANDIDATES},
				{"record_pixel_bytes", sizeof(PatchMatchInstrumentExactPixel)},
				{"record_view_bytes", sizeof(PatchMatchInstrumentExactView)}
			}},
			{"apd_capture", {
				{"schema_name", "openmvs.dmap.apd_pixel_mechanics"},
				{"schema_version", PM_APD_INSTRUMENT_SCHEMA_VERSION},
				{"candidate_accounting", APDCandidateAccountingJson(
					extendedMaps && extendedMaps->apdRequested && extendedMaps->dvpRequested)},
				{"mode", params.nAPDMode},
				{"mode_role", apdFullMode ? "full" :
					apdDeformationOnly ? "deformation_only" : "control"},
				{"requested", extendedMaps && extendedMaps->apdRequested},
				{"stage_active", extendedMaps && extendedMaps->apdStageActive},
				{"summary_available", apdAvailable},
				{"maps_available", apdMapsAvailable},
				{"num_iterations", extendedMaps ? std::max(0, extendedMaps->numLogicalStates-1) : 0},
				{"state_record_bytes", sizeof(PatchMatchAPDInstrumentState)},
				{"update_record_bytes", sizeof(PatchMatchAPDInstrumentUpdate)},
				{"working_score_persisted", false},
				{"persistent_winner_conventionally_rescored", true},
				{"anchor_state", apdFullMode ?
					"immutable_for_non_reliable_stage_after_reliable_first_updates" :
					apdDeformationOnly ?
					"immutable_pre_iteration_support_for_native_checkerboard" :
					"unavailable_disabled"},
				{"anchor_candidate_slots", apdFullMode ?
					nlohmann::json({PM_EXACT_CANDIDATE_APD_ANCHOR_0, PM_EXACT_CANDIDATE_APD_ANCHOR_7}) :
					nlohmann::json(nullptr)},
				{"native_propagation_candidate_slots", apdDeformationOnly ?
					nlohmann::json({PM_EXACT_CANDIDATE_PROPAGATION_0,
						PM_EXACT_CANDIDATE_PROPAGATION_0+7}) : nlohmann::json(nullptr)},
				{"fitted_plane_candidate_slot", apdFullMode ?
					nlohmann::json(PM_EXACT_CANDIDATE_APD_FITTED_PLANE) : nlohmann::json(nullptr)},
					{"final_refinement_candidate_accounting", apdFullMode ?
						"separate_native_domain_fields_not_working_candidate_masks" : "not_active"}
				}},
				{"dvp_capture", {
					{"schema_name", "openmvs.dmap.dvp_pixel_mechanics"},
					{"schema_version", PM_DVP_INSTRUMENT_SCHEMA_VERSION},
					{"family", DVPFamilyName(params.nDVPEpipolarFamily)},
					{"requested", extendedMaps && extendedMaps->dvpRequested},
					{"summary_available", dvpAvailable},
					{"maps_available", dvpMapsAvailable},
					{"targeted_trace_available", extendedMaps && extendedMaps->traceAvailable &&
						!extendedMaps->dvpTraces.empty()},
					{"promotion_eligible", DVPFamilyPromotionEligible(params.nDVPEpipolarFamily)},
					{"claim_label", DVPClaimLabel(params.nDVPEpipolarFamily)},
					{"author_code_exact_equivalence", false},
					{"num_iterations", extendedMaps ? std::max(0, extendedMaps->numLogicalStates-1) : 0},
					{"update_record_bytes", sizeof(PatchMatchDVPInstrumentUpdate)},
					{"counter_record_bytes", sizeof(PatchMatchDVPInstrumentCounters)},
					{"aggregate_sum_precision", "float64 device atomic; counts are uint32 device atomic"},
					{"proposal_candidate_slots", {PM_EXACT_CANDIDATE_DVP_EPIPOLAR_0,
						PM_EXACT_CANDIDATE_DVP_EPIPOLAR_1}},
					{"disjoint_intervals_preserved", true},
					{"endpoint_samples_full_frame", false},
					{"endpoint_samples_targeted_trace", true}
				}},
				{"dvp_visibility_capture", {
					{"schema_name", "openmvs.dmap.dvp_visibility_pixel_mechanics"},
					{"schema_version", PM_DVP_VISIBILITY_INSTRUMENT_SCHEMA_VERSION},
					{"mode", DVPVisibilityModeName(params.nDVPVisibilityMode)},
					{"requested", extendedMaps && extendedMaps->visibilityRequested},
					{"stage_active", extendedMaps && extendedMaps->apdStageActive},
					{"summary_available", visibilityAvailable},
					{"maps_available", visibilityMapsAvailable},
					{"targeted_trace_available", extendedMaps && extendedMaps->traceAvailable &&
						!extendedMaps->visibilityTraces.empty()},
					{"num_iterations", extendedMaps ?
						std::max(0, extendedMaps->numLogicalStates-1) : 0},
					{"num_views", extendedMaps ? extendedMaps->numViews : 0},
					{"counter_record_bytes", sizeof(PatchMatchDVPVisibilityInstrumentCounters)},
					{"update_record_bytes", sizeof(PatchMatchDVPVisibilityInstrumentUpdate)},
					{"trace_record_bytes", sizeof(PatchMatchDVPVisibilityInstrumentTrace)},
					{"aggregate_weight_sum_precision", "uint64 device atomic"},
					{"immutable_candidate_support", true},
					{"per_view_weights_full_frame", false},
					{"per_view_weights_targeted_trace", true},
					{"per_view_reason_full_frame", "per-pixel reason counts and lossless masks"},
					{"per_view_reason_targeted_trace", true}
				}},
				{"dvp_visible_normal_capture", {
					{"schema_name", "openmvs.dmap.dvp_visible_normal_pixel_mechanics"},
					{"schema_version", PM_DVP_VISIBLE_NORMAL_INSTRUMENT_SCHEMA_VERSION},
					{"mode", DVPVisibleNormalModeName(params.nDVPVisibleNormalMode)},
					{"requested", extendedMaps && extendedMaps->visibleNormalRequested},
					{"stage_active", extendedMaps && extendedMaps->apdStageActive},
					{"summary_available", visibleNormalAvailable},
					{"maps_available", visibleNormalMapsAvailable},
					{"targeted_trace_available", extendedMaps && extendedMaps->traceAvailable &&
						!extendedMaps->visibleNormalTraces.empty()},
					{"num_iterations", extendedMaps ?
						std::max(0, extendedMaps->numLogicalStates-1) : 0},
					{"counter_record_bytes", sizeof(PatchMatchDVPVisibleNormalInstrumentCounters)},
					{"update_record_bytes", sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)},
					{"trace_record_bytes", sizeof(PatchMatchDVPVisibleNormalInstrumentTrace)},
					{"main_rng_schedule", "native_unchanged"},
					{"retry_rng", "bounded_local_copy"},
					{"propagation_fallback", "native_best"},
					{"refinement_fallback", "native_proposal"},
					{"actual_normal_vectors_full_frame", false},
					{"actual_normal_vectors_targeted_trace", true}
				}},
				{"observer_sidecars", {
				{"complete", publicationWriteErrors.empty()},
				{"write_error_count", publicationWriteErrors.size()},
				{"write_errors", sidecarWriteErrorsJson}
			}},
			{"maps", nlohmann::json::array()},
			{"write_errors", nlohmann::json::array()}
		};
		auto recordMap = [&](bool saved, const char* signal, const char* fileName, const char* dtype,
			const char* role, const char* semantics, const char* quality, const char* basis,
			const nlohmann::json& extra = nlohmann::json::object()) {
			if (!saved) {
				manifest["write_errors"].push_back(signal);
				return;
			}
			AddDMapManifestEntry(manifest, depthMapDir, signal, String(_T("maps/")) + fileName,
				dtype, role, semantics, quality, basis, extra);
		};
		if (!costMap.empty()) {
			recordMap(costMap.Save(mapsDir + _T("cost_final.pfm")), "cost_final", "cost_final.pfm", "float32", "final_state",
				"aggregate production cost after filtering; keep-cost rejects are zero and ignore-mask rejects use cost 1 so max(1-cost,0) matches production confidence", "exact", "production_post_filter_state");
				recordMap(SaveDMapCostImage(mapsDir + _T("cost_final.png"), costMap), "cost_final_preview", "cost_final.png", "uint8", "preview",
					"display encoding of cost_final", "derived_exact", "cost_final_preview_encoding", {{"encoding", "clamp(cost*255/1.2,0,255)"}});
			} else {
			manifest["write_errors"].push_back("cost_final");
			manifest["write_errors"].push_back("cost_final_preview");
		}
		recordMap(!depthData.depthMap.empty() && depthData.depthMap.Save(mapsDir + _T("depth_final_after_filter.pfm")),
			"depth_final_after_filter", "depth_final_after_filter.pfm", "float32", "final_state", "production depth after filtering", "exact", "production_post_filter_state");
		recordMap(!depthData.normalMap.empty() && depthData.normalMap.Save(mapsDir + _T("normal_final.pfm")),
			"normal_final", "normal_final.pfm", "float32x3", "final_state", "production normal after filtering", "exact", "production_post_filter_state");
		recordMap(SaveDMapByteImage(mapsDir + _T("valid_before_filter.png"), depthData.depthMap.size(), validBeforeFilter, 255),
			"valid_before_filter", "valid_before_filter.png", "uint8", "data", "validity immediately before filtering", "exact", "production_pre_filter_snapshot");
		recordMap(SaveDMapValidAfterImage(mapsDir + _T("valid_after_filter.png"), depthData.depthMap),
			"valid_after_filter", "valid_after_filter.png", "uint8", "data", "validity after filtering", "derived_exact", "depth_final_after_filter>0");
		recordMap(SaveDMapByteImage(mapsDir + _T("rejection_reason.png"), depthData.depthMap.size(), filterRejectReasons, DMAP_REJECTION_OTHER),
			"rejection_reason", "rejection_reason.png", "uint8", "preview", "display encoding of rejection reason codes", "derived_exact", "rejection_reason_preview_encoding", {{"encoding", "reason_code*255/10"}});
		recordMap(SaveDMapCandidateSourceImage(mapsDir + _T("candidate_source.png"), depthData.depthMap.size(), updateSources),
			"candidate_source", "candidate_source.png", "uint8", "preview", "display encoding of the last detected update source; exact source family is unavailable", "proxy", "post_pass_change_detection", {{"encoding", "source_code*28"}});
		recordMap(SaveDMapSupportImage(mapsDir + _T("num_supporting_views.png"), depthData),
			"num_supporting_views", "num_supporting_views.png", "uint8", "preview", "display encoding of final support count", "derived_exact", "selected_view_count_preview_encoding", {{"encoding", "support_count*255/4"}});
		recordMap(SaveDMapLastChangedImage(mapsDir + _T("last_changed_iter.png"), depthData.depthMap.size(), numPasses, passUpdateSources),
			"last_changed_iter", "last_changed_iter.png", "uint8", "preview", "display encoding of the last logical iteration in which a change was detected", "proxy", "post_pass_change_detection", {{"encoding", "legacy normalized pass index; report converts to logical iteration"}});
			if (extendedMaps)
				SaveDMapExtendedMaps(depthMapDir, depthData, numPasses, passCostImprovements, *extendedMaps, manifest);
		const size_t expectedMapCount(manifest["maps"].size() + manifest["write_errors"].size());
		manifest["expected_map_count"] = expectedMapCount;
		manifest["written_map_count"] = manifest["maps"].size();
		mapManifestComplete = manifest["write_errors"].empty() && manifest["maps"].size() == expectedMapCount;
		manifest["complete"] = mapManifestComplete;
		mapManifestWritten = WriteJsonFile(depthMapDir + _T("map_manifest.json"), manifest);
		if (!mapManifestWritten)
			VERBOSE("error: failed to write depth-map instrumentation manifest: %s", (depthMapDir + _T("map_manifest.json")).c_str());
	}
	if (prefilterRequested) {
		nlohmann::json manifest = {
			{"schema_name", "openmvs.dmap.prefilter_manifest"},
			{"schema_version", 1},
			{"capture_kind", "prefilter"},
			{"image_id", imageID},
			{"image_name", imageName.c_str()},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"geometric_consistency", geometricIteration >= 0},
			{"pyramid_level", scaleNumber},
			{"width", depthData.depthMap.cols},
			{"height", depthData.depthMap.rows},
			{"process_specialization", "Process<false>"},
			{"measurement_model", "non-mutating terminal production state captured immediately before keep-cost filtering"},
			{"maps", nlohmann::json::array()},
			{"write_errors", nlohmann::json::array()}
		};
		const size_t area((size_t)depthData.depthMap.area());
		if (prefilterAvailable && extendedMaps && extendedMaps->planesBeforeFilter.size() >= area && area > 0) {
			DepthMap depth(depthData.depthMap.size());
			for (int i = 0; i < depth.area(); ++i)
				depth[i] = extendedMaps->planesBeforeFilter[(size_t)i].w();
			const String prefilterMapPath(mapsDir + _T("depth_final_before_filter.pfm"));
			if (depth.Save(prefilterMapPath)) {
				const size_f_t mapBytes(File::getSize(prefilterMapPath));
				if (mapBytes == SIZE_NA || mapBytes == 0) {
					manifest["write_errors"].push_back("depth_final_before_filter_size_unavailable");
				} else {
					AddDMapManifestEntry(
						manifest,
						depthMapDir,
						"depth_final_before_filter",
						_T("maps/depth_final_before_filter.pfm"),
						"float32",
						"final_state",
						"exact production depth immediately before filtering",
						"exact",
						"production_pre_filter_snapshot");
				}
			} else
				manifest["write_errors"].push_back("depth_final_before_filter");
		} else {
			manifest["write_errors"].push_back(
				prefilterAvailable ? "pre_filter_state_incomplete" : "pre_filter_capture_not_admitted");
		}
		manifest["expected_map_count"] = 1;
		manifest["written_map_count"] = manifest["maps"].size();
		prefilterManifestComplete = manifest["write_errors"].empty() && manifest["maps"].size() == 1;
		manifest["complete"] = prefilterManifestComplete;
		prefilterManifestWritten = WriteJsonFile(
			depthMapDir + _T("prefilter_manifest.json"), manifest);
		if (!prefilterManifestWritten)
			VERBOSE("error: failed to write pre-filter instrumentation manifest: %s", (depthMapDir + _T("prefilter_manifest.json")).c_str());
	}
	uint64_t mapManifestBytes(0);
	const bool mapManifestBound(mapManifestWritten && DMapRegularFileSize(
		std::filesystem::path((depthMapDir + _T("map_manifest.json")).c_str()),
		mapManifestBytes));
	uint64_t prefilterManifestBytes(0);
	const bool prefilterManifestBound(prefilterManifestWritten && DMapRegularFileSize(
		std::filesystem::path((depthMapDir + _T("prefilter_manifest.json")).c_str()),
		prefilterManifestBytes));
	const bool mapEvidenceComplete(mapManifestComplete && mapManifestBound);
	const bool prefilterEvidenceComplete(prefilterManifestComplete && prefilterManifestBound);
	if (writeMaps) {
		if (mapEvidenceComplete && publicationWriteErrors.empty()) {
			summary["completion_marker"] = {
				{"schema_name", "openmvs.dmap.capture_complete"},
				{"schema_version", 1},
				{"path", "capture_complete.json"},
				{"maps_complete", true},
				{"eligible", true}
			};
		} else {
			summary["completion_marker"]["maps_complete"] = mapEvidenceComplete;
			summary["completion_marker"]["eligible"] = false;
			summary["completion_marker"]["state"] = mapEvidenceComplete ?
				"observer_sidecar_publication_incomplete" : "map_publication_incomplete";
		}
	} else if (prefilterRequested) {
		if (prefilterEvidenceComplete && publicationWriteErrors.empty()) {
			summary["completion_marker"] = {
				{"schema_name", "openmvs.dmap.prefilter_capture_complete"},
				{"schema_version", 1},
				{"path", "prefilter_capture_complete.json"},
				{"maps_complete", true},
				{"prefilter_complete", true},
				{"eligible", true}
			};
		} else {
			summary["completion_marker"]["maps_complete"] = prefilterEvidenceComplete;
			summary["completion_marker"]["prefilter_complete"] = prefilterEvidenceComplete;
			summary["completion_marker"]["eligible"] = false;
			summary["completion_marker"]["state"] = prefilterEvidenceComplete ?
				"observer_sidecar_publication_incomplete" : "prefilter_publication_incomplete";
		}
	} else {
		if (publicationWriteErrors.empty()) {
			summary["completion_marker"] = {
				{"schema_name", "openmvs.dmap.summary_complete"},
				{"schema_version", 1},
				{"path", "summary_complete.json"},
				{"maps_complete", false},
				{"eligible", true}
			};
		} else {
			summary["completion_marker"]["eligible"] = false;
			summary["completion_marker"]["state"] = "observer_sidecar_publication_incomplete";
		}
	}
	bool sceneSummaryWritten(WriteDMapSceneSummary(root, summary, geometricIteration));
	if (!sceneSummaryWritten) {
		RecordInstrumentSidecarWriteError(publicationWriteErrors, "scene_summary.json", -1);
		updateSummaryObserverReceipt();
		summary["completion_marker"]["eligible"] = false;
		summary["completion_marker"]["state"] = "scene_summary_publication_failed";
	}
	const bool summaryWritten(WriteJsonFile(depthMapDir + _T("summary.json"), summary));
	uint64_t summaryBytes(0);
	const bool summaryBound(summaryWritten && DMapRegularFileSize(
		std::filesystem::path((depthMapDir + _T("summary.json")).c_str()), summaryBytes));
	if (!summaryBound) {
		RecordInstrumentSidecarWriteError(publicationWriteErrors, "summary.json", -1);
		updateSummaryObserverReceipt();
		summary["completion_marker"]["eligible"] = false;
		summary["completion_marker"]["state"] = "summary_publication_failed";
		// The scene summary may already have been written; rewrite it so it cannot
		// retain a stale eligible=true receipt when the frame summary failed.
		sceneSummaryWritten = WriteDMapSceneSummary(root, summary, geometricIteration) &&
			sceneSummaryWritten;
	}
	const bool sidecarsComplete(publicationWriteErrors.empty());
	const bool summariesComplete(summaryBound && filteringWritten && viewSupportWritten && sceneSummaryWritten && sidecarsComplete);
	if (writeMaps && mapEvidenceComplete && summariesComplete) {
		const nlohmann::json marker = {
			{"schema_name", "openmvs.dmap.capture_complete"},
			{"schema_version", 1},
			{"capture_kind", "maps"},
			{"image_id", imageID},
			{"image_name", imageName.c_str()},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"maps_complete", true},
			{"observer_sidecars_complete", true},
			{"map_manifest", {{"path", "map_manifest.json"}, {"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION}, {"bytes", mapManifestBytes}, {"complete", true}}},
			{"summary", {{"path", "summary.json"}, {"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION}, {"bytes", summaryBytes}}}
		};
		if (!WriteJsonFile(depthMapDir + _T("capture_complete.json"), marker))
			VERBOSE("error: failed to publish depth-map instrumentation completion marker: %s", (depthMapDir + _T("capture_complete.json")).c_str());
	} else if (prefilterRequested && prefilterEvidenceComplete && summariesComplete) {
			const nlohmann::json marker = {
				{"schema_name", "openmvs.dmap.prefilter_capture_complete"},
				{"schema_version", 1},
				{"capture_kind", "prefilter"},
				{"image_id", imageID},
				{"image_name", imageName.c_str()},
				{"estimation_stage", DMapEstimationStageName(geometricIteration)},
				{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
				{"eligible", true},
				{"prefilter_complete", true},
				{"maps_complete", true},
				{"observer_sidecars_complete", true},
				{"manifest", {
					{"path", "prefilter_manifest.json"},
					{"schema_version", 1},
					{"bytes", prefilterManifestBytes},
					{"complete", true}
				}},
				{"summary", {
					{"path", "summary.json"},
					{"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION},
					{"bytes", summaryBytes}
				}}
			};
			if (!WriteJsonFile(depthMapDir + _T("prefilter_capture_complete.json"), marker))
				VERBOSE("error: failed to publish pre-filter instrumentation completion marker: %s", (depthMapDir + _T("prefilter_capture_complete.json")).c_str());
	} else if (!writeMaps && !prefilterRequested && summariesComplete) {
		const nlohmann::json marker = {
			{"schema_name", "openmvs.dmap.summary_complete"},
			{"schema_version", 1},
			{"capture_kind", "summary_only"},
			{"image_id", imageID},
			{"image_name", imageName.c_str()},
			{"estimation_stage", DMapEstimationStageName(geometricIteration)},
			{"geometric_iteration", geometricIteration >= 0 ? nlohmann::json(geometricIteration) : nlohmann::json(nullptr)},
			{"summary_complete", true},
			{"maps_requested", extendedMaps && extendedMaps->mapsRequested},
			{"maps_complete", false},
			{"observer_sidecars_complete", true},
			{"summary", {{"path", "summary.json"}, {"schema_version", PM_DMAP_INSTRUMENT_SCHEMA_VERSION}, {"bytes", summaryBytes}}}
		};
		if (!WriteJsonFile(depthMapDir + _T("summary_complete.json"), marker))
			VERBOSE("error: failed to publish depth-map summary completion marker: %s", (depthMapDir + _T("summary_complete.json")).c_str());
	}
}
#endif
} // anonymous namespace

PatchMatch::PatchMatch()
	: cudaStream(0)
{
	// initialize CUDA device if needed
	if (SEACAVE::CUDA::devices.IsEmpty())
		SEACAVE::CUDA::initDevices(SEACAVE::CUDA::desiredDeviceIDs);
	CUDA_CHECK(cudaStreamCreate(&cudaStream));
}

PatchMatch::~PatchMatch()
{
	Release();
	if (cudaStream)
		cudaStreamDestroy(cudaStream);
}

void PatchMatch::Release()
{
	if (images.empty())
		return;

	FOREACH(i, cudaImageArrays) {
		cudaDestroyTextureObject(textureImages[i]);
		cudaFreeArray(cudaImageArrays[i]);
	}
	cudaImageArrays.clear();

	if (params.bGeomConsistency) {
		FOREACH(i, cudaDepthArrays) {
			cudaDestroyTextureObject(textureDepths[i]);
			cudaFreeArray(cudaDepthArrays[i]);
		}
		cudaDepthArrays.clear();
	}

	images.clear();
	cameras.clear();

	for (float*& p : hostImageStaging) if (p) cudaFreeHost(p);
	hostImageStaging.clear();
	hostImageStagingArea.clear();
	for (float*& p : hostDepthPriorStaging) if (p) cudaFreeHost(p);
	hostDepthPriorStaging.clear();
	hostDepthPriorStagingArea.clear();

	ReleaseCUDA();
}

// pinned staging wins on large images (driver-internal staging stall scales with
// area) but loses on small ones (cudaHostAlloc + explicit memcpy overhead is fixed)
void PatchMatch::StagedUploadCvMat(cudaArray_t dst, const cv::Mat1f& src,
	std::vector<float*>& slots, std::vector<size_t>& areas, size_t slotIdx)
{
	ASSERT(src.type() == CV_32FC1);
	const size_t area = (size_t)src.rows * (size_t)src.cols;
	const size_t rowBytes = (size_t)src.cols * sizeof(float);
	constexpr size_t kPinnedStagingThresholdArea = 1500000;
	if (area < kPinnedStagingThresholdArea) {
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(dst, 0, 0, src.ptr<float>(), src.step[0],
			rowBytes, src.rows, cudaMemcpyHostToDevice, cudaStream));
		return;
	}
	if (slots.size() <= slotIdx) slots.resize(slotIdx + 1, nullptr);
	if (areas.size() <= slotIdx) areas.resize(slotIdx + 1, 0);
	if (slots[slotIdx] == nullptr || areas[slotIdx] < area) {
		if (slots[slotIdx]) CUDA_CHECK(cudaFreeHost(slots[slotIdx]));
		CUDA_CHECK(cudaHostAlloc((void**)&slots[slotIdx], area * sizeof(float), cudaHostAllocDefault));
		areas[slotIdx] = area;
	}
	float* dstPinned = slots[slotIdx];
	if (src.isContinuous() && src.step[0] == rowBytes) {
		memcpy(dstPinned, src.ptr<float>(), area * sizeof(float));
	} else {
		for (int r = 0; r < src.rows; ++r)
			memcpy(dstPinned + (size_t)r * src.cols, src.ptr<float>(r), rowBytes);
	}
	CUDA_CHECK(cudaMemcpy2DToArrayAsync(dst, 0, 0, dstPinned, rowBytes,
		rowBytes, src.rows, cudaMemcpyHostToDevice, cudaStream));
}

void PatchMatch::ReleaseCUDA()
{
	cudaFree(cudaTextureImages);
	cudaFree(cudaDepthNormalEstimates);
	cudaFree(cudaDepthNormalCosts);
	cudaFree(cudaRandStates);
	cudaFree(cudaSelectedViews);
	if (params.bGeomConsistency)
		cudaFree(cudaTextureDepths);
	if (depthNormalEstimates) {
		cudaFreeHost(depthNormalEstimates);
		depthNormalEstimates = NULL;
	}
}

void PatchMatch::Init(bool bGeomConsistency)
{
	params.bCompat23 = OPTDENSE::bPatchMatchCUDACompat23;
	if (bGeomConsistency) {
		params.bGeomConsistency = true;
		params.nEstimationIters = 1;
	} else {
		params.bGeomConsistency = false;
		params.nEstimationIters = OPTDENSE::nEstimationIters;
	}
}

void PatchMatch::AllocatePatchMatchCUDA(const cv::Mat1f& image)
{
	const size_t num_images = images.size();
	CUDA_CHECK(cudaMalloc((void**)&cudaTextureImages, sizeof(cudaTextureObject_t) * num_images));
	if (params.bGeomConsistency)
		CUDA_CHECK(cudaMalloc((void**)&cudaTextureDepths, sizeof(cudaTextureObject_t) * (num_images-1)));

	const size_t size = image.size().area();
	// pin estimates buffer so the H<->D copies on cudaStream run as true DMA-async without driver staging
	CUDA_CHECK(cudaHostAlloc((void**)&depthNormalEstimates, sizeof(Point4) * size, cudaHostAllocDefault));
	CUDA_CHECK(cudaMalloc((void**)&cudaDepthNormalEstimates, sizeof(Point4) * size));

	CUDA_CHECK(cudaMalloc((void**)&cudaDepthNormalCosts, sizeof(float) * size));
	CUDA_CHECK(cudaMalloc((void**)&cudaSelectedViews, sizeof(unsigned) * size));
	CUDA_CHECK(cudaMalloc((void**)&cudaRandStates, sizeof(curandState) * size));
}

void PatchMatch::AllocateImageCUDA(size_t i, const cv::Mat1f& image, bool bInitImage, bool bInitDepthMap)
{
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	if (bInitImage) {
		CUDA_CHECK(cudaMallocArray(&cudaImageArrays[i], &channelDesc, image.cols, image.rows));

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cudaImageArrays[i];

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode  = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA_CHECK(cudaCreateTextureObject(&textureImages[i], &resDesc, &texDesc, NULL));
	}

	if (params.bGeomConsistency && i > 0) {
		if (!bInitDepthMap) {
			textureDepths[i-1] = 0;
			cudaDepthArrays[i-1] = NULL;
			return;
		}

		CUDA_CHECK(cudaMallocArray(&cudaDepthArrays[i-1], &channelDesc, image.cols, image.rows));

		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cudaDepthArrays[i-1];

		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode  = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		CUDA_CHECK(cudaCreateTextureObject(&textureDepths[i-1], &resDesc, &texDesc, NULL));
	}
}

void PatchMatch::EstimateDepthMap(DepthData& depthData, int geometricIteration, ConfAdjustRequest* pConfRequest)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.size() > 1);

	// multi-resolution
	DepthData& fullResDepthData(depthData);
	const unsigned totalScaleNumber(params.bGeomConsistency ? 0u : OPTDENSE::nSubResolutionLevels);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	ViewsMap lowResViewsMap;
	IIndex prevNumImages = (IIndex)images.size();
	const IIndex numImages = depthData.images.size();
	params.nNumViews = (int)numImages-1;
	params.nInitTopK = MINF(params.nInitTopK, params.nNumViews);
	params.fDepthMin = depthData.dMin;
	params.fDepthMax = depthData.dMax;
	params.nAPDMode = OPTDENSE::nPatchMatchCUDAAPD;
	params.nDVPEpipolarFamily = OPTDENSE::nPatchMatchCUDADVPEpipolarFamily;
	params.fDVPEpipolarAlpha = OPTDENSE::fPatchMatchCUDADVPEpipolarAlpha;
	params.fDVPEpipolarBeta = OPTDENSE::fPatchMatchCUDADVPEpipolarBeta;
	params.nDVPEpipolarMu = OPTDENSE::nPatchMatchCUDADVPEpipolarMu;
	params.nDVPGlobalSearchRadius = OPTDENSE::nPatchMatchCUDADVPGlobalSearchRadius;
	params.fDVPReprojectionThreshold = OPTDENSE::fPatchMatchCUDADVPReprojectionThreshold;
	params.fDVPRelativeDepthThreshold = OPTDENSE::fPatchMatchCUDADVPRelativeDepthThreshold;
	params.nDVPDepthEdgeMode = OPTDENSE::nPatchMatchCUDADVPDepthEdgeMode;
	params.nDVPVisibilityMode = OPTDENSE::nPatchMatchCUDADVPVisibilityMode;
	params.fDVPVisibilityReprojectionThreshold =
		OPTDENSE::fPatchMatchCUDADVPVisibilityReprojectionThreshold;
	params.fDVPVisibilityRelativeDepthThreshold =
		OPTDENSE::fPatchMatchCUDADVPVisibilityRelativeDepthThreshold;
	params.nDVPVisibleNormalMode = OPTDENSE::nPatchMatchCUDADVPVisibleNormalMode;
	params.fDVPVisibleNormalDotTolerance =
		OPTDENSE::fPatchMatchCUDADVPVisibleNormalDotTolerance;
	params.nDVPVisibleNormalAttempts = OPTDENSE::nPatchMatchCUDADVPVisibleNormalAttempts;
	const bool apdRequested(APDModeEnabled(params.nAPDMode));
	const bool depthEdgeRequested(DVPDepthEdgeModeEnabled(params.nDVPDepthEdgeMode));
	const DVPDepthEdgeConfig depthEdgeConfig{
		params.nDVPDepthEdgeMode, apdRequested,
		!OPTDENSE::strPatchMatchCUDADVPDepthEdgePriorDir.empty()};
	const DVPDepthEdgeConfigStatus depthEdgeConfigStatus(ValidateDVPDepthEdgeConfig(depthEdgeConfig));
	if (depthEdgeConfigStatus != DVPDepthEdgeConfigStatus::VALID) {
		VERBOSE("error: invalid DVP depth-edge configuration (status %u)",
			static_cast<unsigned>(depthEdgeConfigStatus));
		ABORT(invalid DVP depth-edge configuration);
	}
	DVPDepthEdgePrior depthEdgePrior;
	if (depthEdgeRequested) {
		String priorError;
		const Image8U::Size sourceSize(depthData.GetView().pImageData->GetSize());
		if (!LoadDVPDepthEdgePrior(OPTDENSE::strPatchMatchCUDADVPDepthEdgePriorDir,
			depthData.GetView().GetID(), depthData.GetView().pImageData->name,
			sourceSize, params.nDVPDepthEdgeMode, depthEdgePrior, priorError))
		{
			VERBOSE("error: DVP depth-edge prior rejected for depth%04u: %s",
				depthData.GetView().GetID(), priorError.c_str());
			ABORT(invalid DVP depth-edge prior);
		}
	}
	APDMultiscaleDepthState apdCarryState;
	if (apdRequested && params.bGeomConsistency && fullResDepthData.apdMultiscaleState.IsValid())
		apdCarryState = fullResDepthData.apdMultiscaleState;
	// The shared DepthData slot is only a cross-event mailbox. This invocation
	// owns the state until it either publishes the next stage or releases it.
	fullResDepthData.apdMultiscaleState.Release();
	if (prevNumImages < numImages) {
		images.resize(numImages);
		cameras.resize(numImages);
		cudaImageArrays.resize(numImages);
		textureImages.resize(numImages);
	}
	if (params.bGeomConsistency && cudaDepthArrays.size() < (size_t)params.nNumViews) {
		cudaDepthArrays.resize(params.nNumViews);
		textureDepths.resize(params.nNumViews);
	}
	const int maxPixelViews(MINF(params.nNumViews, 4));
	#ifdef _USE_DMAP_INSTRUMENTATION
	std::vector<InstrumentSidecarWriteError> instrumentSidecarWriteErrors;
	uint64_t instrumentFrameStorageCommittedBytes(0);
	uint64_t instrumentFullResolutionPriorityReserveBytes(0);
	InstrumentStorageReservation instrumentFramePriorityReservation;
	const int instrumentImageID((int)depthData.GetView().GetID());
	const String& instrumentImageName(depthData.GetView().pImageData->name);
	const bool instrumentSelected(InstrumentImageEnabled(instrumentImageID, instrumentImageName));
	const bool instrumentMapsRequested(instrumentSelected && InstrumentWriteMaps());
	const bool instrumentPrefilterRequested(instrumentSelected && InstrumentPrefilterRequested());
		const bool instrumentAPDRequested(
			APDModeEnabled(params.nAPDMode) &&
			(!instrumentPrefilterRequested || instrumentMapsRequested));
	const bool instrumentDVPRequested(
		DVPEpipolarFamilyEnabled(params.nDVPEpipolarFamily) && params.bGeomConsistency &&
		(!instrumentPrefilterRequested || instrumentMapsRequested));
	const bool instrumentVisibilityRequested(
		DVPVisibilityModeEnabled(params.nDVPVisibilityMode) &&
		APDModeUsesFullMechanics(params.nAPDMode) && params.bGeomConsistency &&
		(!instrumentPrefilterRequested || instrumentMapsRequested));
	const bool instrumentVisibleNormalRequested(
		DVPVisibleNormalModeEnabled(params.nDVPVisibleNormalMode) &&
		APDModeUsesFullMechanics(params.nAPDMode) &&
		(!instrumentPrefilterRequested || instrumentMapsRequested));
	const bool instrumentExactRequested(
		instrumentMapsRequested && !instrumentAPDRequested &&
		!OPTDENSE::strDMapInstrumentationDir.empty());
	if (instrumentSelected && !OPTDENSE::strDMapInstrumentationDir.empty()) {
		const String instrumentRoot(InstrumentRunRoot(geometricIteration));
		const String instrumentDepthMapDir(DMapDepthMapDir(
			instrumentRoot, instrumentImageID, instrumentImageName));
		if (!RemoveDMapCompletionMarkers(instrumentDepthMapDir))
			RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "stale_completion_markers", -1);
	}
	if (instrumentSelected && totalScaleNumber > 0) {
		const int numInstrumentPasses(1 + params.nEstimationIters * 2);
		const int numLogicalStates(1 + params.nEstimationIters);
		const Image8U::Size fullResolutionSize(fullResDepthData.images.front().image.size());
		const InstrumentTraceSelectionResult fullResolutionTraces(
			SelectInstrumentTracePixels(instrumentImageID, 1.f, fullResolutionSize));
		const InstrumentExtendedMaps fullResolutionPriorityPlan(PlanInstrumentResources(
			fullResolutionSize.area(),
			numInstrumentPasses,
			numLogicalStates,
			params.nNumViews,
			(int)fullResolutionTraces.pixels.size(),
			fullResolutionTraces.labelBytes,
			0u,
			0u,
			false,
			instrumentMapsRequested,
			instrumentExactRequested,
			instrumentPrefilterRequested,
			depthEdgeRequested && instrumentMapsRequested,
			instrumentAPDRequested,
			instrumentDVPRequested,
			instrumentVisibilityRequested,
			instrumentVisibleNormalRequested));
		instrumentFullResolutionPriorityReserveBytes =
			fullResolutionPriorityPlan.estimatedStorageBytes;
	}
	#endif
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		const float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(DepthMapsData::ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		const Image8U::Size size(depthData.images.front().image.size());
		const unsigned apdLevelCount(OPTDENSE::nSubResolutionLevels+1u);
		const unsigned apdLevelIndex(params.bGeomConsistency ?
			OPTDENSE::nSubResolutionLevels : totalScaleNumber-scaleNumber);
		const unsigned apdStageIndex(params.bGeomConsistency ?
			OPTDENSE::nSubResolutionLevels+1u+
				static_cast<unsigned>(geometricIteration >= 0 ? geometricIteration : 0) :
			apdLevelIndex);
		APDStageClock apdClock;
		apdClock.levelIndex = apdLevelIndex;
		apdClock.levelCount = apdLevelCount;
		apdClock.stageIndex = apdStageIndex;
		apdClock.geometricConsistency = params.bGeomConsistency;
		APDMultiscaleTransferStatus apdTransferStatus(
			APDMultiscaleTransferStatus::UNAVAILABLE_NO_SOURCE);
		#ifdef _USE_DMAP_INSTRUMENTATION
		const bool instrumentAPDSourceStateAvailable(apdRequested && apdCarryState.IsValid());
		const APDMultiscaleStateHeader instrumentAPDSourceStateHeader(apdCarryState.header);
		#endif
		Image8U apdTransferredReliability;
		Image8U apdTransferredAnchorCounts;
		Image8U apdTransferredDeformableEligible;
		if (apdRequested && apdCarryState.IsValid()) {
			apdTransferStatus = ValidateAPDMultiscaleTransfer(
				apdCarryState.header, apdClock, (unsigned)size.width, (unsigned)size.height);
			if (apdTransferStatus == APDMultiscaleTransferStatus::VALID) {
				cv::resize(apdCarryState.reliabilityMap, apdTransferredReliability,
					size, 0, 0, cv::INTER_NEAREST);
				cv::resize(apdCarryState.anchorCountMap, apdTransferredAnchorCounts,
					size, 0, 0, cv::INTER_NEAREST);
				cv::resize(apdCarryState.deformableEligibleMap,
					apdTransferredDeformableEligible, size, 0, 0, cv::INTER_NEAREST);
				apdClock.hasTransferredState = true;
			}
		}
		params.nAPDLevelIndex = apdLevelIndex;
		params.nAPDLevelCount = apdLevelCount;
		params.nAPDStageIndex = apdStageIndex;
		params.nAPDTransferStatus = static_cast<unsigned>(apdTransferStatus);
		params.bAPDTransferredState = apdClock.hasTransferredState;
		Image8U apdOutputReliability;
		Image8U apdOutputAnchorCounts;
		Image8U apdOutputDeformableEligible;
		Image16U depthEdgeRegions;
		#ifdef _USE_DMAP_INSTRUMENTATION
		Image8U depthEdgeAnchorCountsBefore;
		Image8U depthEdgeAnchorCountsAfter;
		Image8U depthEdgeRejectedAnchorCounts;
		#endif
		PatchMatchAPDMultiscaleIO apdMultiscaleIO;
		if (apdRequested) {
			apdOutputReliability.create(size);
			apdOutputAnchorCounts.create(size);
			apdOutputDeformableEligible.create(size);
			ASSERT(apdOutputReliability.isContinuous() &&
				apdOutputAnchorCounts.isContinuous() &&
				apdOutputDeformableEligible.isContinuous());
			apdMultiscaleIO.transferredReliability = apdClock.hasTransferredState ?
				apdTransferredReliability.ptr<uint8_t>() : nullptr;
			apdMultiscaleIO.outputReliability = apdOutputReliability.ptr<uint8_t>();
			apdMultiscaleIO.outputAnchorCounts = apdOutputAnchorCounts.ptr<uint8_t>();
			apdMultiscaleIO.outputDeformableEligible =
				apdOutputDeformableEligible.ptr<uint8_t>();
			if (depthEdgeRequested) {
				cv::resize(depthEdgePrior.labels, depthEdgeRegions, size, 0, 0,
					cv::INTER_NEAREST);
				ASSERT(depthEdgeRegions.isContinuous());
				apdMultiscaleIO.depthEdgeRegions = depthEdgeRegions.ptr<uint16_t>();
			}
		}
		params.bLowResProcessed = false;
		if (scaleNumber != totalScaleNumber) {
			// all resolutions, but the smallest one, if multi-resolution is enabled
			params.bLowResProcessed = true;
			// The compatibility path intentionally restores 2.3's interpolated
			// coarse depth prior; discrete normals and view IDs remain nearest-neighbor.
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0,
				params.bCompat23 ? cv::INTER_LINEAR : cv::INTER_NEAREST);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			cv::resize(lowResViewsMap, depthData.viewsMap, size, 0, 0, cv::INTER_NEAREST);
			CUDA_CHECK(cudaMallocAsync((void**)&cudaLowDepths, sizeof(float) * size.area(), cudaStream));
		} else {
			if (totalScaleNumber > 0) {
				// smallest resolution, when multi-resolution is enabled
				fullResDepthData.depthMap.release();
				fullResDepthData.normalMap.release();
				fullResDepthData.confMap.release();
				fullResDepthData.viewsMap.release();
			}
			// smallest resolution if multi-resolution is enabled; highest otherwise
			if (depthData.viewsMap.empty())
				depthData.viewsMap.create(size);
		}
		if (scaleNumber == 0) {
			// highest resolution
			if (depthData.confMap.empty())
				depthData.confMap.create(size);
		}

		// set keep threshold to:
		params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep;
		if (totalScaleNumber) {
			// multi-resolution enabled
			if (scaleNumber > 0 && scaleNumber != totalScaleNumber) {
				// all sub-resolutions, but the smallest and highest
				params.fThresholdKeepCost = 0.f; // disable filtering
			} else if (scaleNumber == totalScaleNumber || (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters)) {
				// smallest sub-resolution OR highest resolution and geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		} else {
			// multi-resolution disabled
			if (!params.bGeomConsistency && OPTDENSE::nEstimationGeometricIters) {
				// geometric consistency is not running but enabled
				params.fThresholdKeepCost = OPTDENSE::fNCCThresholdKeep*1.2f;
			}
		}

		for (IIndex i = 0; i < numImages; ++i) {
			const DepthData::ViewData& view = depthData.images[i];
			const Image32F image = view.image;
			const Camera camera(
				Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.K.val).cast<float>(),
				Eigen::Map<const SEACAVE::Matrix3x3::EMat>(view.camera.R.val).cast<float>(),
				Eigen::Map<const SEACAVE::Point3::EVec>(view.camera.C.ptr()).cast<float>(),
				image.cols, image.rows);
			// store camera and image
			if (i == 0 && (prevNumImages < numImages || images[0].size() != image.size())) {
				// allocate/reallocate PatchMatch CUDA memory
				if (prevNumImages > 0)
					ReleaseCUDA();
				AllocatePatchMatchCUDA(image);
			}
			if (i >= prevNumImages) {
				// allocate image CUDA memory
				AllocateImageCUDA(i, image, true, !view.depthMap.empty());
			} else
			if (images[i].size() != image.size()) {
				// reallocate image CUDA memory
				cudaDestroyTextureObject(textureImages[i]);
				cudaFreeArray(cudaImageArrays[i]);
				if (params.bGeomConsistency && i > 0) {
					cudaDestroyTextureObject(textureDepths[i-1]);
					cudaFreeArray(cudaDepthArrays[i-1]);
				}
				AllocateImageCUDA(i, image, true, !view.depthMap.empty());
			} else
			if (params.bGeomConsistency && i > 0 && (view.depthMap.empty() != (cudaDepthArrays[i-1] == NULL))) {
				// reallocate depth CUDA memory
				if (cudaDepthArrays[i-1]) {
					cudaDestroyTextureObject(textureDepths[i-1]);
					cudaFreeArray(cudaDepthArrays[i-1]);
				}
				AllocateImageCUDA(i, image, false, !view.depthMap.empty());
			}
			// large images stage through per-instance pinned slot for a truly-async
			// H->D DMA on cudaStream; small images fall through to direct pageable
			// DMA inside StagedUploadCvMat (driver-internal staging is cheaper)
			StagedUploadCvMat(cudaImageArrays[i], image, hostImageStaging, hostImageStagingArea, (size_t)i);
			if (params.bGeomConsistency && i > 0 && !view.depthMap.empty()) {
				// set previously computed depth-map
				DepthMap depthMap(view.depthMap);
				if (depthMap.size() != image.size())
					cv::resize(depthMap, depthMap, image.size(), 0, 0, cv::INTER_LINEAR);
				StagedUploadCvMat(cudaDepthArrays[i-1], depthMap, hostDepthPriorStaging, hostDepthPriorStagingArea, (size_t)(i-1));
			}
			images[i] = std::move(image);
			cameras[i] = std::move(camera);
		}
		if (params.bGeomConsistency && cudaDepthArrays.size() > numImages - 1) {
			for (IIndex i = numImages; i < prevNumImages; ++i) {
				// free image CUDA memory
				cudaDestroyTextureObject(textureDepths[i-1]);
				cudaFreeArray(cudaDepthArrays[i-1]);
			}
			cudaDepthArrays.resize(params.nNumViews);
			textureDepths.resize(params.nNumViews);
		}
		if (prevNumImages > numImages) {
			for (IIndex i = numImages; i < prevNumImages; ++i) {
				// free image CUDA memory
				cudaDestroyTextureObject(textureImages[i]);
				cudaFreeArray(cudaImageArrays[i]);
			}
			images.resize(numImages);
			cameras.resize(numImages);
			cudaImageArrays.resize(numImages);
			textureImages.resize(numImages);
		}
		prevNumImages = numImages;

		// setup CUDA memory (queued on cudaStream)
		CUDA_CHECK(cudaMemcpyAsync(cudaTextureImages, textureImages.data(), sizeof(cudaTextureObject_t) * numImages, cudaMemcpyHostToDevice, cudaStream));
		if (params.bGeomConsistency) {
			// set previously computed depth-maps
			ASSERT(depthData.depthMap.size() == depthData.GetView().image.size());
			CUDA_CHECK(cudaMemcpyAsync(cudaTextureDepths, textureDepths.data(), sizeof(cudaTextureObject_t) * params.nNumViews, cudaMemcpyHostToDevice, cudaStream));
		}

		// load depth-map and normal-map into CUDA memory
		for (int r = 0; r < depthData.depthMap.rows; ++r) {
			const int baseIndex = r * depthData.depthMap.cols;
			for (int c = 0; c < depthData.depthMap.cols; ++c) {
				const Normal& n = depthData.normalMap(r, c);
				const int index = baseIndex + c;
				Point4& depthNormal = depthNormalEstimates[index];
				depthNormal.topLeftCorner<3, 1>() = Eigen::Map<const Normal::EVec>(n.ptr());
				depthNormal.w() = depthData.depthMap(r, c);
			}
		}
		// pinned host buffer => DMA-async on cudaStream
		CUDA_CHECK(cudaMemcpyAsync(cudaDepthNormalEstimates, depthNormalEstimates, sizeof(Point4) * depthData.depthMap.size().area(), cudaMemcpyHostToDevice, cudaStream));

		// load low resolution depth-map into CUDA memory
		if (params.bLowResProcessed) {
			ASSERT(depthData.depthMap.isContinuous());
			CUDA_CHECK(cudaMemcpyAsync(cudaLowDepths, depthData.depthMap.ptr<float>(), sizeof(float) * depthData.depthMap.size().area(), cudaMemcpyHostToDevice, cudaStream));
		}

#ifdef _USE_DMAP_INSTRUMENTATION
			String instrumentRoot;
			if (scaleNumber == 0 && !OPTDENSE::strDMapInstrumentationDir.empty()) {
				instrumentRoot = InstrumentRunRoot(geometricIteration);
				if (!WriteDMapRunMetadata(instrumentRoot, geometricIteration))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "run_metadata.json", -1);
				if (!AppendDMapInstrumentationSelection(
					instrumentRoot, instrumentImageID, instrumentImageName,
					params.bGeomConsistency, geometricIteration))
				{
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "frame_selection.csv", 0);
				}
			}
			const int numInstrumentPasses(1 + params.nEstimationIters * 2);
			const int instrumentArea(size.area());
			const int numLogicalStates(1 + params.nEstimationIters);
			InstrumentTraceSelectionResult instrumentTraceSelection;
			if (instrumentSelected && OPTDENSE::nPatchMatchInstrumentLevel >= 2)
				instrumentTraceSelection = SelectInstrumentTracePixels(
					instrumentImageID, scale, size);
			InstrumentExtendedMaps instrumentExtendedMaps;
			if (instrumentSelected) {
				instrumentExtendedMaps = PlanInstrumentResources(
					instrumentArea,
					numInstrumentPasses,
					numLogicalStates,
					params.nNumViews,
					(int)instrumentTraceSelection.pixels.size(),
					instrumentTraceSelection.labelBytes,
					instrumentFrameStorageCommittedBytes,
					scaleNumber > 0 ? instrumentFullResolutionPriorityReserveBytes : 0u,
					instrumentMapsRequested && scaleNumber != 0,
					scaleNumber == 0 && instrumentMapsRequested,
					scaleNumber == 0 && instrumentExactRequested,
					scaleNumber == 0 && instrumentPrefilterRequested,
					depthEdgeRequested && instrumentMapsRequested,
					instrumentAPDRequested,
					instrumentDVPRequested,
						instrumentVisibilityRequested,
						instrumentVisibleNormalRequested);
				instrumentExtendedMaps.apdStageActive = params.bAPDTransferredState;
				if (instrumentAPDRequested && instrumentMapsRequested)
					instrumentExtendedMaps.exactUnavailableReason =
						_T("generic exact candidate/view maps are unavailable for APD; use the APD-specific exact mechanics capture");
			}
			InstrumentStorageReservation instrumentStorageReservation;
			if (instrumentSelected && instrumentExtendedMaps.estimatedStorageBytes > 0) {
				if (instrumentRoot.empty())
					instrumentRoot = InstrumentRunRoot(geometricIteration);
				ApplyInstrumentStoragePreflight(
					instrumentRoot, instrumentExtendedMaps, instrumentStorageReservation,
					instrumentFramePriorityReservation, scaleNumber == 0);
			}
			if (instrumentSelected && instrumentExtendedMaps.summaryAvailable)
				instrumentFrameStorageCommittedBytes = instrumentExtendedMaps.frameStorageBudgetBytes;
			if (instrumentSelected) {
				if (instrumentRoot.empty())
					instrumentRoot = InstrumentRunRoot(geometricIteration);
				if (!AppendDMapInstrumentationResourcePlan(
					instrumentRoot,
					instrumentImageID,
					instrumentImageName,
					params.bGeomConsistency,
					geometricIteration,
					(int)scaleNumber,
					size.width,
					size.height,
					instrumentExtendedMaps))
				{
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "resource_plans.jsonl", (int)scaleNumber);
				}
				const bool requestedEvidenceUnavailable(
					!instrumentExtendedMaps.summaryAvailable ||
					(instrumentExtendedMaps.traceRequested && !instrumentExtendedMaps.traceAvailable) ||
					(scaleNumber == 0 && instrumentMapsRequested && !instrumentExtendedMaps.mapsAvailable) ||
					(scaleNumber == 0 && instrumentExactRequested && !instrumentExtendedMaps.exactAvailable) ||
					(scaleNumber == 0 && instrumentPrefilterRequested && !instrumentExtendedMaps.prefilterAvailable));
				if (requestedEvidenceUnavailable) {
					const bool fail(OPTDENSE::strDMapInstrumentationBudgetPolicy.ToLower() == _T("error"));
					const String& unavailableReason(!instrumentExtendedMaps.storagePreflightReason.empty() ?
						instrumentExtendedMaps.storagePreflightReason :
						!instrumentExtendedMaps.traceUnavailableReason.empty() ?
							instrumentExtendedMaps.traceUnavailableReason :
							!instrumentExtendedMaps.prefilterUnavailableReason.empty() ?
								instrumentExtendedMaps.prefilterUnavailableReason : instrumentExtendedMaps.exactUnavailableReason);
				VERBOSE("%s: depth-map instrumentation resource plan for image %d is '%s' (%s)",
					fail ? "error" : "warning", instrumentImageID,
					instrumentExtendedMaps.resourceDecision.c_str(),
					unavailableReason.c_str());
				if (fail)
					exit(EXIT_FAILURE);
			}
		}
		const bool instrumentEnabled(instrumentSelected && instrumentExtendedMaps.summaryAvailable);
		const bool writeInstrumentMaps(instrumentEnabled && instrumentMapsRequested &&
			(scaleNumber != 0 || instrumentExtendedMaps.mapsAvailable));
		if (depthEdgeRequested && writeInstrumentMaps) {
			depthEdgeAnchorCountsBefore.create(size);
			depthEdgeAnchorCountsAfter.create(size);
			depthEdgeRejectedAnchorCounts.create(size);
			ASSERT(depthEdgeAnchorCountsBefore.isContinuous() &&
				depthEdgeAnchorCountsAfter.isContinuous() &&
				depthEdgeRejectedAnchorCounts.isContinuous());
			apdMultiscaleIO.outputDepthEdgeAnchorCountsBefore =
				depthEdgeAnchorCountsBefore.ptr<uint8_t>();
			apdMultiscaleIO.outputDepthEdgeAnchorCountsAfter =
				depthEdgeAnchorCountsAfter.ptr<uint8_t>();
			apdMultiscaleIO.outputDepthEdgeRejectedAnchorCounts =
				depthEdgeRejectedAnchorCounts.ptr<uint8_t>();
		}
		String instrumentDir;
		std::vector<PatchMatchInstrumentCounters> instrumentCounters;
		std::vector<PatchMatchInstrumentCounters> instrumentIterationCounters;
			std::vector<PatchMatchInstrumentTraceRecord> instrumentTraceRecords;
			std::vector<InstrumentTracePixel> activeTracePixels;
			std::vector<int32_t> instrumentTraceMap;
			if (instrumentExtendedMaps.traceAvailable) {
				activeTracePixels = MaterializeInstrumentTracePixels(
					instrumentImageID, instrumentTraceSelection);
				instrumentTraceMap = BuildInstrumentTraceMap(size, activeTracePixels);
			}
			std::vector<InstrumentTraceSelection>().swap(instrumentTraceSelection.pixels);
		std::vector<uint8_t> instrumentUpdateSources;
		std::vector<uint8_t> instrumentValidBeforeFilter;
		std::vector<uint8_t> instrumentFilterRejectReasons;
		std::vector<float> instrumentImprovementMaps;
		std::vector<uint8_t> instrumentPassUpdateSources;
		std::vector<float> instrumentKernelTimingsMs;
		PatchMatchInstrumentCounters* cudaInstrumentCounters(nullptr);
		PatchMatchInstrumentTraceRecord* cudaInstrumentTraceRecords(nullptr);
		int32_t* cudaInstrumentTraceMap(nullptr);
		uint8_t* cudaInstrumentUpdateSources(nullptr);
		uint8_t* cudaInstrumentValidBeforeFilter(nullptr);
		uint8_t* cudaInstrumentFilterRejectReasons(nullptr);
		float* cudaInstrumentImprovementMaps(nullptr);
		uint8_t* cudaInstrumentPassUpdateSources(nullptr);
		float* cudaInstrumentPassDepthDeltas(nullptr);
		float* cudaInstrumentPassDepthRelDeltas(nullptr);
		float* cudaInstrumentPassNormalAngleDeltas(nullptr);
		uint8_t* cudaInstrumentPassViewChurn(nullptr);
		float* cudaInstrumentLogicalStoredCosts(nullptr);
		float4* cudaInstrumentLogicalScorePrimary(nullptr);
		float4* cudaInstrumentLogicalScoreSecondary(nullptr);
		float4* cudaInstrumentExactLogicalScorePrimary(nullptr);
		float4* cudaInstrumentExactLogicalScoreSecondary(nullptr);
			PatchMatchInstrumentExactPixel* cudaInstrumentExactPixels(nullptr);
			PatchMatchInstrumentExactView* cudaInstrumentExactViews(nullptr);
			PatchMatchAPDInstrumentCounters* cudaInstrumentAPDCounters(nullptr);
			PatchMatchAPDInstrumentState* cudaInstrumentAPDStates(nullptr);
				PatchMatchAPDInstrumentUpdate* cudaInstrumentAPDUpdates(nullptr);
				PatchMatchAPDInstrumentTrace* cudaInstrumentAPDTraces(nullptr);
					PatchMatchDVPInstrumentCounters* cudaInstrumentDVPCounters(nullptr);
					PatchMatchDVPInstrumentUpdate* cudaInstrumentDVPUpdates(nullptr);
					PatchMatchDVPInstrumentTrace* cudaInstrumentDVPTraces(nullptr);
						PatchMatchDVPVisibilityInstrumentCounters* cudaInstrumentVisibilityCounters(nullptr);
						PatchMatchDVPVisibilityInstrumentUpdate* cudaInstrumentVisibilityUpdates(nullptr);
						PatchMatchDVPVisibilityInstrumentTrace* cudaInstrumentVisibilityTraces(nullptr);
						uint8_t* cudaInstrumentVisibilityReasons(nullptr);
						PatchMatchDVPVisibleNormalInstrumentCounters* cudaInstrumentVisibleNormalCounters(nullptr);
						PatchMatchDVPVisibleNormalInstrumentUpdate* cudaInstrumentVisibleNormalUpdates(nullptr);
						PatchMatchDVPVisibleNormalInstrumentTrace* cudaInstrumentVisibleNormalTraces(nullptr);
		float4* cudaInstrumentFinalViewWeights(nullptr);
		float4* cudaInstrumentFinalViewCosts(nullptr);
		float4* cudaInstrumentFinalViewPhotometricCosts(nullptr);
		float4* cudaInstrumentFinalViewGeometricCosts(nullptr);
		float* cudaInstrumentFinalViewEntropy(nullptr);
		float* cudaInstrumentFinalLowDepth(nullptr);
		uint32_t* cudaInstrumentFinalSelectedViews(nullptr);
		uint8_t* cudaInstrumentAcceptedUpdateCount(nullptr);
		Point4* cudaInstrumentPlanesBeforeFilter(nullptr);
		float* cudaInstrumentCostsBeforeFilter(nullptr);
		Point4* cudaInstrumentPlanesBeforePass(nullptr);
		float* cudaInstrumentCostsBeforePass(nullptr);
		uint32_t* cudaInstrumentSelectedViewsBeforePass(nullptr);
		PatchMatchInstrumentDeviceContext instrumentContext;
		if (instrumentEnabled) {
				if (instrumentRoot.empty())
					instrumentRoot = InstrumentRunRoot(geometricIteration);
				instrumentDir = InstrumentOutputRoot(geometricIteration);
				if (!WriteDMapRunMetadata(instrumentRoot, geometricIteration))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "run_metadata.json", -1);
			instrumentCounters.resize(numInstrumentPasses);
			CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentCounters, sizeof(PatchMatchInstrumentCounters) * instrumentCounters.size()));
			CUDA_CHECK(cudaMemsetAsync(cudaInstrumentCounters, 0, sizeof(PatchMatchInstrumentCounters) * instrumentCounters.size(), cudaStream));
			instrumentUpdateSources.resize((size_t)instrumentArea, 0);
			CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentUpdateSources, sizeof(uint8_t) * (size_t)instrumentArea));
			CUDA_CHECK(cudaMemsetAsync(cudaInstrumentUpdateSources, 0, sizeof(uint8_t) * (size_t)instrumentArea, cudaStream));
			CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPlanesBeforePass, sizeof(Point4) * (size_t)instrumentArea));
			CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentCostsBeforePass, sizeof(float) * (size_t)instrumentArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentSelectedViewsBeforePass, sizeof(uint32_t) * (size_t)instrumentArea));
					if (instrumentExtendedMaps.apdAvailable &&
					instrumentExtendedMaps.apdStageActive && params.nEstimationIters > 0) {
					const size_t apdIterationCount((size_t)params.nEstimationIters);
					instrumentExtendedMaps.apdCounters.resize(apdIterationCount);
					CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentAPDCounters,
						sizeof(PatchMatchAPDInstrumentCounters) * apdIterationCount));
					CUDA_CHECK(cudaMemsetAsync(cudaInstrumentAPDCounters, 0,
						sizeof(PatchMatchAPDInstrumentCounters) * apdIterationCount, cudaStream));
					if (instrumentExtendedMaps.apdMapsAvailable) {
						const size_t apdMapArea((size_t)instrumentArea * apdIterationCount);
						instrumentExtendedMaps.apdStates.resize(apdMapArea);
						instrumentExtendedMaps.apdUpdates.resize(apdMapArea);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentAPDStates,
							sizeof(PatchMatchAPDInstrumentState) * apdMapArea));
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentAPDUpdates,
							sizeof(PatchMatchAPDInstrumentUpdate) * apdMapArea));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentAPDStates, 0,
							sizeof(PatchMatchAPDInstrumentState) * apdMapArea, cudaStream));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentAPDUpdates, 0,
							sizeof(PatchMatchAPDInstrumentUpdate) * apdMapArea, cudaStream));
					}
						if (instrumentExtendedMaps.traceAvailable && !activeTracePixels.empty()) {
						const size_t apdTraceCount(activeTracePixels.size() * apdIterationCount);
						instrumentExtendedMaps.apdTraces.resize(apdTraceCount);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentAPDTraces,
							sizeof(PatchMatchAPDInstrumentTrace) * apdTraceCount));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentAPDTraces, 0,
							sizeof(PatchMatchAPDInstrumentTrace) * apdTraceCount, cudaStream));
						}
					}
					if (instrumentExtendedMaps.dvpAvailable && params.nEstimationIters > 0) {
						const size_t dvpIterationCount((size_t)params.nEstimationIters);
						instrumentExtendedMaps.dvpCounters.resize(dvpIterationCount);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentDVPCounters,
							sizeof(PatchMatchDVPInstrumentCounters) * dvpIterationCount));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentDVPCounters, 0,
							sizeof(PatchMatchDVPInstrumentCounters) * dvpIterationCount, cudaStream));
						if (instrumentExtendedMaps.dvpMapsAvailable) {
							const size_t dvpMapArea((size_t)instrumentArea * dvpIterationCount);
							instrumentExtendedMaps.dvpUpdates.resize(dvpMapArea);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentDVPUpdates,
								sizeof(PatchMatchDVPInstrumentUpdate) * dvpMapArea));
							CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentDVPUpdates,
								instrumentExtendedMaps.dvpUpdates.data(),
								sizeof(PatchMatchDVPInstrumentUpdate) * dvpMapArea,
								cudaMemcpyHostToDevice, cudaStream));
						}
						if (instrumentExtendedMaps.traceAvailable && !activeTracePixels.empty()) {
							const size_t dvpTraceCount(activeTracePixels.size() * dvpIterationCount);
							instrumentExtendedMaps.dvpTraces.resize(dvpTraceCount);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentDVPTraces,
								sizeof(PatchMatchDVPInstrumentTrace) * dvpTraceCount));
							CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentDVPTraces,
								instrumentExtendedMaps.dvpTraces.data(),
								sizeof(PatchMatchDVPInstrumentTrace) * dvpTraceCount,
								cudaMemcpyHostToDevice, cudaStream));
						}
					}
					if (instrumentExtendedMaps.visibilityAvailable &&
						instrumentExtendedMaps.apdStageActive && params.nEstimationIters > 0)
					{
						const size_t visibilityIterationCount((size_t)params.nEstimationIters);
						instrumentExtendedMaps.visibilityCounters.resize(visibilityIterationCount);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibilityCounters,
							sizeof(PatchMatchDVPVisibilityInstrumentCounters) * visibilityIterationCount));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibilityCounters, 0,
							sizeof(PatchMatchDVPVisibilityInstrumentCounters) * visibilityIterationCount,
							cudaStream));
						const size_t visibilityReasonCount(
							(size_t)instrumentArea * PM_INSTRUMENT_MAX_VIEWS);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibilityReasons,
							sizeof(uint8_t) * visibilityReasonCount));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibilityReasons,
							static_cast<int>(DVPVisibilityReason::MODE_DISABLED),
							sizeof(uint8_t) * visibilityReasonCount, cudaStream));
						if (instrumentExtendedMaps.visibilityMapsAvailable) {
							const size_t visibilityMapArea(
								(size_t)instrumentArea * visibilityIterationCount);
							instrumentExtendedMaps.visibilityUpdates.resize(visibilityMapArea);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibilityUpdates,
								sizeof(PatchMatchDVPVisibilityInstrumentUpdate) * visibilityMapArea));
							CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibilityUpdates, 0,
								sizeof(PatchMatchDVPVisibilityInstrumentUpdate) * visibilityMapArea,
								cudaStream));
						}
						if (instrumentExtendedMaps.traceAvailable && !activeTracePixels.empty()) {
							const size_t visibilityTraceCount(
								activeTracePixels.size() * visibilityIterationCount);
							instrumentExtendedMaps.visibilityTraces.resize(visibilityTraceCount);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibilityTraces,
								sizeof(PatchMatchDVPVisibilityInstrumentTrace) * visibilityTraceCount));
							CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibilityTraces, 0,
								sizeof(PatchMatchDVPVisibilityInstrumentTrace) * visibilityTraceCount,
								cudaStream));
						}
					}
					if (instrumentExtendedMaps.visibleNormalAvailable &&
						instrumentExtendedMaps.apdStageActive && params.nEstimationIters > 0)
					{
						const size_t visibleNormalIterationCount((size_t)params.nEstimationIters);
						instrumentExtendedMaps.visibleNormalCounters.resize(
							visibleNormalIterationCount);
						CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibleNormalCounters,
							sizeof(PatchMatchDVPVisibleNormalInstrumentCounters)*
								visibleNormalIterationCount));
						CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibleNormalCounters, 0,
							sizeof(PatchMatchDVPVisibleNormalInstrumentCounters)*
								visibleNormalIterationCount, cudaStream));
						if (instrumentExtendedMaps.visibleNormalMapsAvailable) {
							const size_t visibleNormalMapArea(
								(size_t)instrumentArea*visibleNormalIterationCount);
							instrumentExtendedMaps.visibleNormalUpdates.resize(visibleNormalMapArea);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibleNormalUpdates,
								sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)*visibleNormalMapArea));
							CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibleNormalUpdates, 0,
								sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)*visibleNormalMapArea,
								cudaStream));
						}
						if (instrumentExtendedMaps.traceAvailable && !activeTracePixels.empty()) {
							const size_t visibleNormalTraceCount(
								activeTracePixels.size()*visibleNormalIterationCount);
							instrumentExtendedMaps.visibleNormalTraces.resize(visibleNormalTraceCount);
							CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentVisibleNormalTraces,
								sizeof(PatchMatchDVPVisibleNormalInstrumentTrace)*visibleNormalTraceCount));
							CUDA_CHECK(cudaMemsetAsync(cudaInstrumentVisibleNormalTraces, 0,
								sizeof(PatchMatchDVPVisibleNormalInstrumentTrace)*visibleNormalTraceCount,
								cudaStream));
						}
					}
				if (writeInstrumentMaps && scaleNumber == 0) {
				instrumentValidBeforeFilter.resize((size_t)instrumentArea, 0);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentValidBeforeFilter, sizeof(uint8_t) * instrumentValidBeforeFilter.size()));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentValidBeforeFilter, 0, sizeof(uint8_t) * instrumentValidBeforeFilter.size(), cudaStream));
				instrumentFilterRejectReasons.resize((size_t)instrumentArea, 0);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFilterRejectReasons, sizeof(uint8_t) * instrumentFilterRejectReasons.size()));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFilterRejectReasons, 0, sizeof(uint8_t) * instrumentFilterRejectReasons.size(), cudaStream));
				instrumentImprovementMaps.resize((size_t)instrumentArea * (size_t)numInstrumentPasses, 0.f);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentImprovementMaps, sizeof(float) * instrumentImprovementMaps.size()));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentImprovementMaps, 0, sizeof(float) * instrumentImprovementMaps.size(), cudaStream));
				instrumentPassUpdateSources.resize((size_t)instrumentArea * (size_t)numInstrumentPasses, 0);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPassUpdateSources, sizeof(uint8_t) * instrumentPassUpdateSources.size()));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPassUpdateSources, 0, sizeof(uint8_t) * instrumentPassUpdateSources.size(), cudaStream));
				const size_t passArea((size_t)instrumentArea * (size_t)numInstrumentPasses);
				instrumentExtendedMaps.passDepthDeltas.resize(passArea, 0.f);
				instrumentExtendedMaps.passDepthRelDeltas.resize(passArea, 0.f);
				instrumentExtendedMaps.passNormalAngleDeltas.resize(passArea, 0.f);
				instrumentExtendedMaps.passViewChurn.resize(passArea, 0);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPassDepthDeltas, sizeof(float) * passArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPassDepthRelDeltas, sizeof(float) * passArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPassNormalAngleDeltas, sizeof(float) * passArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPassViewChurn, sizeof(uint8_t) * passArea));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPassDepthDeltas, 0, sizeof(float) * passArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPassDepthRelDeltas, 0, sizeof(float) * passArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPassNormalAngleDeltas, 0, sizeof(float) * passArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPassViewChurn, 0, sizeof(uint8_t) * passArea, cudaStream));

				const size_t mapArea((size_t)instrumentArea);
				const size_t logicalMapArea(mapArea * (size_t)instrumentExtendedMaps.numLogicalStates);
				instrumentExtendedMaps.logicalStoredCosts.resize(logicalMapArea, 0.f);
				instrumentExtendedMaps.logicalScorePrimary.resize(logicalMapArea);
				instrumentExtendedMaps.logicalScoreSecondary.resize(logicalMapArea);
				if (instrumentExtendedMaps.exactAvailable) {
					instrumentExtendedMaps.exactLogicalScorePrimary.resize(logicalMapArea);
					instrumentExtendedMaps.exactLogicalScoreSecondary.resize(logicalMapArea);
					instrumentExtendedMaps.exactPixels.resize(logicalMapArea);
					instrumentExtendedMaps.exactViews.resize(logicalMapArea * (size_t)instrumentExtendedMaps.numViews);
					const float4 unavailableScore(make_float4(-1.f, -1.f, -1.f, -1.f));
					std::fill(instrumentExtendedMaps.exactLogicalScorePrimary.begin(), instrumentExtendedMaps.exactLogicalScorePrimary.end(), unavailableScore);
					std::fill(instrumentExtendedMaps.exactLogicalScoreSecondary.begin(), instrumentExtendedMaps.exactLogicalScoreSecondary.end(), unavailableScore);
					for (PatchMatchInstrumentExactView& view : instrumentExtendedMaps.exactViews)
						view.metadata = PM_EXACT_VIEW_RANK_MASK << PM_EXACT_VIEW_RANK_SHIFT;
				}
				instrumentExtendedMaps.finalViewWeights.resize(mapArea);
				instrumentExtendedMaps.finalViewCosts.resize(mapArea);
				instrumentExtendedMaps.finalViewPhotometricCosts.resize(mapArea);
				instrumentExtendedMaps.finalViewGeometricCosts.resize(mapArea);
				instrumentExtendedMaps.finalViewEntropy.resize(mapArea, 0.f);
				instrumentExtendedMaps.finalLowDepth.resize(mapArea, 0.f);
				instrumentExtendedMaps.finalSelectedViews.resize(mapArea, 0);
				instrumentExtendedMaps.acceptedUpdateCount.resize(mapArea, 0);
				instrumentExtendedMaps.planesBeforeFilter.resize(mapArea);
				instrumentExtendedMaps.costsBeforeFilter.resize(mapArea, 0.f);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentLogicalStoredCosts, sizeof(float) * logicalMapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentLogicalScorePrimary, sizeof(float4) * logicalMapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentLogicalScoreSecondary, sizeof(float4) * logicalMapArea));
				if (instrumentExtendedMaps.exactAvailable) {
					CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentExactLogicalScorePrimary, sizeof(float4) * logicalMapArea));
					CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentExactLogicalScoreSecondary, sizeof(float4) * logicalMapArea));
					CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentExactPixels, sizeof(PatchMatchInstrumentExactPixel) * logicalMapArea));
					CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentExactViews, sizeof(PatchMatchInstrumentExactView) * instrumentExtendedMaps.exactViews.size()));
				}
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalViewWeights, sizeof(float4) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalViewCosts, sizeof(float4) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalViewPhotometricCosts, sizeof(float4) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalViewGeometricCosts, sizeof(float4) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalViewEntropy, sizeof(float) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalLowDepth, sizeof(float) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentFinalSelectedViews, sizeof(uint32_t) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentAcceptedUpdateCount, sizeof(uint8_t) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPlanesBeforeFilter, sizeof(Point4) * mapArea));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentCostsBeforeFilter, sizeof(float) * mapArea));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentLogicalStoredCosts, 0, sizeof(float) * logicalMapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentLogicalScorePrimary, 0, sizeof(float4) * logicalMapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentLogicalScoreSecondary, 0, sizeof(float4) * logicalMapArea, cudaStream));
				if (instrumentExtendedMaps.exactAvailable) {
					CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentExactLogicalScorePrimary, instrumentExtendedMaps.exactLogicalScorePrimary.data(), sizeof(float4) * logicalMapArea, cudaMemcpyHostToDevice, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentExactLogicalScoreSecondary, instrumentExtendedMaps.exactLogicalScoreSecondary.data(), sizeof(float4) * logicalMapArea, cudaMemcpyHostToDevice, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentExactPixels, instrumentExtendedMaps.exactPixels.data(), sizeof(PatchMatchInstrumentExactPixel) * logicalMapArea, cudaMemcpyHostToDevice, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentExactViews, instrumentExtendedMaps.exactViews.data(), sizeof(PatchMatchInstrumentExactView) * instrumentExtendedMaps.exactViews.size(), cudaMemcpyHostToDevice, cudaStream));
				}
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalViewWeights, 0, sizeof(float4) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalViewCosts, 0, sizeof(float4) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalViewPhotometricCosts, 0, sizeof(float4) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalViewGeometricCosts, 0, sizeof(float4) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalViewEntropy, 0, sizeof(float) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalLowDepth, 0, sizeof(float) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentFinalSelectedViews, 0, sizeof(uint32_t) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentAcceptedUpdateCount, 0, sizeof(uint8_t) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPlanesBeforeFilter, 0, sizeof(Point4) * mapArea, cudaStream));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentCostsBeforeFilter, 0, sizeof(float) * mapArea, cudaStream));
			}
			if (instrumentExtendedMaps.prefilterAvailable && instrumentExtendedMaps.planesBeforeFilter.empty()) {
				const size_t mapArea((size_t)instrumentArea);
				instrumentExtendedMaps.planesBeforeFilter.resize(mapArea);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentPlanesBeforeFilter, sizeof(Point4) * mapArea));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentPlanesBeforeFilter, 0, sizeof(Point4) * mapArea, cudaStream));
			}
			instrumentKernelTimingsMs.resize((size_t)numInstrumentPasses, 0.f);
			if (!activeTracePixels.empty()) {
				ASSERT(instrumentTraceMap.size() == (size_t)instrumentArea);
				instrumentTraceRecords.resize(activeTracePixels.size() * (size_t)numInstrumentPasses);
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentTraceMap, sizeof(int32_t) * instrumentTraceMap.size()));
				CUDA_CHECK(cudaMemcpyAsync(cudaInstrumentTraceMap, instrumentTraceMap.data(), sizeof(int32_t) * instrumentTraceMap.size(), cudaMemcpyHostToDevice, cudaStream));
				CUDA_CHECK(cudaMalloc((void**)&cudaInstrumentTraceRecords, sizeof(PatchMatchInstrumentTraceRecord) * instrumentTraceRecords.size()));
				CUDA_CHECK(cudaMemsetAsync(cudaInstrumentTraceRecords, 0, sizeof(PatchMatchInstrumentTraceRecord) * instrumentTraceRecords.size(), cudaStream));
			}
			instrumentContext.counters = cudaInstrumentCounters;
			instrumentContext.traceRecords = cudaInstrumentTraceRecords;
			instrumentContext.traceMap = cudaInstrumentTraceMap;
			instrumentContext.updateSources = cudaInstrumentUpdateSources;
			instrumentContext.validBeforeFilter = cudaInstrumentValidBeforeFilter;
			instrumentContext.filterRejectReasons = cudaInstrumentFilterRejectReasons;
			instrumentContext.improvementMaps = cudaInstrumentImprovementMaps;
			instrumentContext.passUpdateSources = cudaInstrumentPassUpdateSources;
			instrumentContext.passDepthDeltas = cudaInstrumentPassDepthDeltas;
			instrumentContext.passDepthRelDeltas = cudaInstrumentPassDepthRelDeltas;
			instrumentContext.passNormalAngleDeltas = cudaInstrumentPassNormalAngleDeltas;
			instrumentContext.passViewChurn = cudaInstrumentPassViewChurn;
			instrumentContext.logicalStoredCosts = cudaInstrumentLogicalStoredCosts;
			instrumentContext.logicalScorePrimary = cudaInstrumentLogicalScorePrimary;
			instrumentContext.logicalScoreSecondary = cudaInstrumentLogicalScoreSecondary;
			instrumentContext.exactLogicalScorePrimary = cudaInstrumentExactLogicalScorePrimary;
			instrumentContext.exactLogicalScoreSecondary = cudaInstrumentExactLogicalScoreSecondary;
				instrumentContext.exactPixels = cudaInstrumentExactPixels;
				instrumentContext.exactViews = cudaInstrumentExactViews;
				instrumentContext.apdCounters = cudaInstrumentAPDCounters;
				instrumentContext.apdStates = cudaInstrumentAPDStates;
				instrumentContext.apdUpdates = cudaInstrumentAPDUpdates;
					instrumentContext.apdTraces = cudaInstrumentAPDTraces;
					instrumentContext.dvpCounters = cudaInstrumentDVPCounters;
					instrumentContext.dvpUpdates = cudaInstrumentDVPUpdates;
					instrumentContext.dvpTraces = cudaInstrumentDVPTraces;
					instrumentContext.visibilityCounters = cudaInstrumentVisibilityCounters;
					instrumentContext.visibilityUpdates = cudaInstrumentVisibilityUpdates;
					instrumentContext.visibilityTraces = cudaInstrumentVisibilityTraces;
					instrumentContext.visibilityReasons = cudaInstrumentVisibilityReasons;
					instrumentContext.visibleNormalCounters = cudaInstrumentVisibleNormalCounters;
					instrumentContext.visibleNormalUpdates = cudaInstrumentVisibleNormalUpdates;
					instrumentContext.visibleNormalTraces = cudaInstrumentVisibleNormalTraces;
			instrumentContext.finalViewWeights = cudaInstrumentFinalViewWeights;
			instrumentContext.finalViewCosts = cudaInstrumentFinalViewCosts;
			instrumentContext.finalViewPhotometricCosts = cudaInstrumentFinalViewPhotometricCosts;
			instrumentContext.finalViewGeometricCosts = cudaInstrumentFinalViewGeometricCosts;
			instrumentContext.finalViewEntropy = cudaInstrumentFinalViewEntropy;
			instrumentContext.finalLowDepth = cudaInstrumentFinalLowDepth;
			instrumentContext.finalSelectedViews = cudaInstrumentFinalSelectedViews;
			instrumentContext.acceptedUpdateCount = cudaInstrumentAcceptedUpdateCount;
			instrumentContext.planesBeforeFilter = cudaInstrumentPlanesBeforeFilter;
			instrumentContext.costsBeforeFilter = cudaInstrumentCostsBeforeFilter;
			instrumentContext.planesBeforePass = cudaInstrumentPlanesBeforePass;
			instrumentContext.costsBeforePass = cudaInstrumentCostsBeforePass;
			instrumentContext.selectedViewsBeforePass = cudaInstrumentSelectedViewsBeforePass;
			instrumentContext.kernelTimingsMs = instrumentKernelTimingsMs.data();
			instrumentContext.numTracePixels = (int32_t)activeTracePixels.size();
			instrumentContext.imageID = (int32_t)depthData.GetView().GetID();
			instrumentContext.scaleNumber = (int32_t)scaleNumber;
				instrumentContext.numLogicalStates = instrumentExtendedMaps.numLogicalStates;
				instrumentContext.numAPDIterations = params.nEstimationIters;
			instrumentContext.viewStride = instrumentExtendedMaps.numViews;
			instrumentContext.sampled = (OPTDENSE::nPatchMatchInstrumentLevel >= 2);
			instrumentContext.exact = instrumentExtendedMaps.exactAvailable;
		}
#endif
		// run CUDA patch-match: GPU-side event chains successive workers'
		// kernel sequences so the next worker's __constant__ writes wait for
		// the previous worker's kernels to finish reading them. Host mutex
		// covers only the queueing window, not kernel execution.
		ASSERT(!depthData.viewsMap.empty());
		std::call_once(g_constMemEventInit, []() {
			CUDA_CHECK(cudaEventCreateWithFlags(&g_constMemReady, cudaEventDisableTiming));
		});
		{
			std::lock_guard<std::mutex> queueLock(g_patchMatchCudaMutex);
			CUDA_CHECK(cudaStreamWaitEvent(cudaStream, g_constMemReady, 0));
			UploadCameras();
#ifdef _USE_DMAP_INSTRUMENTATION
			RunCUDA(depthData.confMap.getData(), (uint32_t*)depthData.viewsMap.getData(),
				instrumentEnabled ? instrumentUpdateSources.data() : nullptr,
				instrumentEnabled ? &instrumentContext : nullptr,
				apdRequested ? &apdMultiscaleIO : nullptr);
				if (instrumentEnabled) {
					CUDA_CHECK(cudaMemcpyAsync(instrumentCounters.data(), cudaInstrumentCounters, sizeof(PatchMatchInstrumentCounters) * instrumentCounters.size(), cudaMemcpyDeviceToHost, cudaStream));
					if (!instrumentExtendedMaps.apdCounters.empty())
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.apdCounters.data(), cudaInstrumentAPDCounters,
							sizeof(PatchMatchAPDInstrumentCounters) * instrumentExtendedMaps.apdCounters.size(),
							cudaMemcpyDeviceToHost, cudaStream));
					if (!instrumentExtendedMaps.apdStates.empty())
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.apdStates.data(), cudaInstrumentAPDStates,
							sizeof(PatchMatchAPDInstrumentState) * instrumentExtendedMaps.apdStates.size(),
							cudaMemcpyDeviceToHost, cudaStream));
					if (!instrumentExtendedMaps.apdUpdates.empty())
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.apdUpdates.data(), cudaInstrumentAPDUpdates,
							sizeof(PatchMatchAPDInstrumentUpdate) * instrumentExtendedMaps.apdUpdates.size(),
							cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.apdTraces.empty())
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.apdTraces.data(), cudaInstrumentAPDTraces,
							sizeof(PatchMatchAPDInstrumentTrace) * instrumentExtendedMaps.apdTraces.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.dvpCounters.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.dvpCounters.data(), cudaInstrumentDVPCounters,
								sizeof(PatchMatchDVPInstrumentCounters) * instrumentExtendedMaps.dvpCounters.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.dvpUpdates.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.dvpUpdates.data(), cudaInstrumentDVPUpdates,
								sizeof(PatchMatchDVPInstrumentUpdate) * instrumentExtendedMaps.dvpUpdates.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.dvpTraces.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.dvpTraces.data(), cudaInstrumentDVPTraces,
								sizeof(PatchMatchDVPInstrumentTrace) * instrumentExtendedMaps.dvpTraces.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibilityCounters.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibilityCounters.data(),
								cudaInstrumentVisibilityCounters,
								sizeof(PatchMatchDVPVisibilityInstrumentCounters) *
									instrumentExtendedMaps.visibilityCounters.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibilityUpdates.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibilityUpdates.data(),
								cudaInstrumentVisibilityUpdates,
								sizeof(PatchMatchDVPVisibilityInstrumentUpdate) *
									instrumentExtendedMaps.visibilityUpdates.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibilityTraces.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibilityTraces.data(),
								cudaInstrumentVisibilityTraces,
								sizeof(PatchMatchDVPVisibilityInstrumentTrace) *
									instrumentExtendedMaps.visibilityTraces.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibleNormalCounters.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibleNormalCounters.data(),
								cudaInstrumentVisibleNormalCounters,
								sizeof(PatchMatchDVPVisibleNormalInstrumentCounters)*
									instrumentExtendedMaps.visibleNormalCounters.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibleNormalUpdates.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibleNormalUpdates.data(),
								cudaInstrumentVisibleNormalUpdates,
								sizeof(PatchMatchDVPVisibleNormalInstrumentUpdate)*
									instrumentExtendedMaps.visibleNormalUpdates.size(),
								cudaMemcpyDeviceToHost, cudaStream));
						if (!instrumentExtendedMaps.visibleNormalTraces.empty())
							CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.visibleNormalTraces.data(),
								cudaInstrumentVisibleNormalTraces,
								sizeof(PatchMatchDVPVisibleNormalInstrumentTrace)*
									instrumentExtendedMaps.visibleNormalTraces.size(),
								cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentValidBeforeFilter.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentValidBeforeFilter.data(), cudaInstrumentValidBeforeFilter, sizeof(uint8_t) * instrumentValidBeforeFilter.size(), cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentFilterRejectReasons.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentFilterRejectReasons.data(), cudaInstrumentFilterRejectReasons, sizeof(uint8_t) * instrumentFilterRejectReasons.size(), cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentImprovementMaps.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentImprovementMaps.data(), cudaInstrumentImprovementMaps, sizeof(float) * instrumentImprovementMaps.size(), cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentPassUpdateSources.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentPassUpdateSources.data(), cudaInstrumentPassUpdateSources, sizeof(uint8_t) * instrumentPassUpdateSources.size(), cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentExtendedMaps.passDepthDeltas.empty()) {
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.passDepthDeltas.data(), cudaInstrumentPassDepthDeltas, sizeof(float) * instrumentExtendedMaps.passDepthDeltas.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.passDepthRelDeltas.data(), cudaInstrumentPassDepthRelDeltas, sizeof(float) * instrumentExtendedMaps.passDepthRelDeltas.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.passNormalAngleDeltas.data(), cudaInstrumentPassNormalAngleDeltas, sizeof(float) * instrumentExtendedMaps.passNormalAngleDeltas.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.passViewChurn.data(), cudaInstrumentPassViewChurn, sizeof(uint8_t) * instrumentExtendedMaps.passViewChurn.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.logicalStoredCosts.data(), cudaInstrumentLogicalStoredCosts, sizeof(float) * instrumentExtendedMaps.logicalStoredCosts.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.logicalScorePrimary.data(), cudaInstrumentLogicalScorePrimary, sizeof(float4) * instrumentExtendedMaps.logicalScorePrimary.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.logicalScoreSecondary.data(), cudaInstrumentLogicalScoreSecondary, sizeof(float4) * instrumentExtendedMaps.logicalScoreSecondary.size(), cudaMemcpyDeviceToHost, cudaStream));
					if (instrumentExtendedMaps.exactAvailable) {
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.exactLogicalScorePrimary.data(), cudaInstrumentExactLogicalScorePrimary, sizeof(float4) * instrumentExtendedMaps.exactLogicalScorePrimary.size(), cudaMemcpyDeviceToHost, cudaStream));
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.exactLogicalScoreSecondary.data(), cudaInstrumentExactLogicalScoreSecondary, sizeof(float4) * instrumentExtendedMaps.exactLogicalScoreSecondary.size(), cudaMemcpyDeviceToHost, cudaStream));
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.exactPixels.data(), cudaInstrumentExactPixels, sizeof(PatchMatchInstrumentExactPixel) * instrumentExtendedMaps.exactPixels.size(), cudaMemcpyDeviceToHost, cudaStream));
						CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.exactViews.data(), cudaInstrumentExactViews, sizeof(PatchMatchInstrumentExactView) * instrumentExtendedMaps.exactViews.size(), cudaMemcpyDeviceToHost, cudaStream));
					}
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalViewWeights.data(), cudaInstrumentFinalViewWeights, sizeof(float4) * instrumentExtendedMaps.finalViewWeights.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalViewCosts.data(), cudaInstrumentFinalViewCosts, sizeof(float4) * instrumentExtendedMaps.finalViewCosts.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalViewPhotometricCosts.data(), cudaInstrumentFinalViewPhotometricCosts, sizeof(float4) * instrumentExtendedMaps.finalViewPhotometricCosts.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalViewGeometricCosts.data(), cudaInstrumentFinalViewGeometricCosts, sizeof(float4) * instrumentExtendedMaps.finalViewGeometricCosts.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalViewEntropy.data(), cudaInstrumentFinalViewEntropy, sizeof(float) * instrumentExtendedMaps.finalViewEntropy.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalLowDepth.data(), cudaInstrumentFinalLowDepth, sizeof(float) * instrumentExtendedMaps.finalLowDepth.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.finalSelectedViews.data(), cudaInstrumentFinalSelectedViews, sizeof(uint32_t) * instrumentExtendedMaps.finalSelectedViews.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.acceptedUpdateCount.data(), cudaInstrumentAcceptedUpdateCount, sizeof(uint8_t) * instrumentExtendedMaps.acceptedUpdateCount.size(), cudaMemcpyDeviceToHost, cudaStream));
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.costsBeforeFilter.data(), cudaInstrumentCostsBeforeFilter, sizeof(float) * instrumentExtendedMaps.costsBeforeFilter.size(), cudaMemcpyDeviceToHost, cudaStream));
				}
				if (!instrumentExtendedMaps.planesBeforeFilter.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentExtendedMaps.planesBeforeFilter.data(), cudaInstrumentPlanesBeforeFilter, sizeof(Point4) * instrumentExtendedMaps.planesBeforeFilter.size(), cudaMemcpyDeviceToHost, cudaStream));
				if (!instrumentTraceRecords.empty())
					CUDA_CHECK(cudaMemcpyAsync(instrumentTraceRecords.data(), cudaInstrumentTraceRecords, sizeof(PatchMatchInstrumentTraceRecord) * instrumentTraceRecords.size(), cudaMemcpyDeviceToHost, cudaStream));
			}
#else
			RunCUDA(depthData.confMap.getData(), (uint32_t*)depthData.viewsMap.getData(),
				apdRequested ? &apdMultiscaleIO : nullptr);
#endif
			CUDA_CHECK(cudaEventRecord(g_constMemReady, cudaStream));
		}
		// wait for our own kernels + D2H copies to finish before the unpack loop
		// reads from the pinned host buffer
			CUDA_CHECK(cudaStreamSynchronize(cudaStream));
			CUDA_CHECK(cudaGetLastError());
			#ifdef _USE_DMAP_INSTRUMENTATION
			if (instrumentEnabled && depthEdgeRequested && writeInstrumentMaps &&
				!SaveDVPDepthEdgeInstrumentation(
					instrumentDir, instrumentImageID, (int)scaleNumber, size, params,
					depthEdgePrior, depthEdgeRegions, depthEdgeAnchorCountsBefore,
					depthEdgeAnchorCountsAfter, depthEdgeRejectedAnchorCounts,
					activeTracePixels, instrumentExtendedMaps.apdTraces,
					params.nEstimationIters))
			{
				RecordInstrumentSidecarWriteError(
					instrumentSidecarWriteErrors, "dvp_depth_edge_stages.jsonl", (int)scaleNumber);
			}
			if (instrumentEnabled && apdRequested) {
			instrumentExtendedMaps.apdMultiscale = APDMultiscaleObservabilityJson(
				params,
				instrumentAPDSourceStateAvailable ? &instrumentAPDSourceStateHeader : nullptr,
				apdTransferredReliability,
				apdTransferredAnchorCounts,
				apdTransferredDeformableEligible,
				apdOutputReliability,
				apdOutputAnchorCounts,
				apdOutputDeformableEligible);
			if (instrumentExtendedMaps.apdMapsAvailable) {
				auto copyByteMap = [](const Image8U& source, std::vector<uint8_t>& destination) {
					if (source.empty()) {
						destination.clear();
						return;
					}
					ASSERT(source.isContinuous());
					const uint8_t* begin(source.ptr<uint8_t>());
					destination.assign(begin, begin+(size_t)source.area());
				};
				copyByteMap(apdTransferredReliability,
					instrumentExtendedMaps.apdTransferredReliability);
				copyByteMap(apdTransferredAnchorCounts,
					instrumentExtendedMaps.apdTransferredAnchorCounts);
				copyByteMap(apdTransferredDeformableEligible,
					instrumentExtendedMaps.apdTransferredDeformableEligible);
				copyByteMap(apdOutputReliability,
					instrumentExtendedMaps.apdOutputReliability);
				copyByteMap(apdOutputAnchorCounts,
					instrumentExtendedMaps.apdOutputAnchorCounts);
				copyByteMap(apdOutputDeformableEligible,
					instrumentExtendedMaps.apdOutputDeformableEligible);
			}
			if (!AppendAPDMultiscaleStage(
				instrumentDir, instrumentImageID, (int)scaleNumber, size,
				instrumentExtendedMaps.apdMultiscale))
			{
				RecordInstrumentSidecarWriteError(
					instrumentSidecarWriteErrors, "apd_multiscale_stages.jsonl", (int)scaleNumber);
			}
		}
		#endif
		if (apdRequested) {
			apdCarryState.Release();
			apdCarryState.header.version = APD_MULTISCALE_STATE_VERSION;
			apdCarryState.header.width = (uint32_t)size.width;
			apdCarryState.header.height = (uint32_t)size.height;
			apdCarryState.header.sourceLevelIndex = apdLevelIndex;
			apdCarryState.header.sourceStageIndex = apdStageIndex;
			apdCarryState.reliabilityMap = apdOutputReliability;
			apdCarryState.anchorCountMap = apdOutputAnchorCounts;
			apdCarryState.deformableEligibleMap = apdOutputDeformableEligible;
			ASSERT(apdCarryState.IsValid());
		}
		if (params.bLowResProcessed)
			CUDA_CHECK(cudaFreeAsync(cudaLowDepths, cudaStream));
#ifdef _USE_DMAP_INSTRUMENTATION
		ConfidenceMap instrumentCostMap;
		bool writeDMapArtifacts(false);
		if (instrumentEnabled) {
				if (!depthData.confMap.empty())
					instrumentCostMap = depthData.confMap.clone();
				instrumentIterationCounters = AggregateInstrumentIterations(
						instrumentCounters,
						instrumentExtendedMaps.exactAvailable ||
							(instrumentExtendedMaps.apdAvailable && instrumentExtendedMaps.apdStageActive) ||
							instrumentExtendedMaps.dvpAvailable);
				if (!AppendInstrumentCounters(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size, instrumentIterationCounters))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "counters.csv", (int)scaleNumber);
				// APD instrumented kernels also emit one exact record per logical
				// iteration (from the pixel's active checkerboard phase), even though
				// the generic exact full-frame map bundle is deliberately unavailable.
				const bool exactHotKernelTraceRecords(
						instrumentExtendedMaps.exactAvailable ||
							(instrumentExtendedMaps.apdAvailable && instrumentExtendedMaps.apdStageActive) ||
							instrumentExtendedMaps.dvpAvailable);
				if (!AppendInstrumentTraces(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, activeTracePixels, instrumentTraceRecords, numInstrumentPasses, exactHotKernelTraceRecords))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "traces.jsonl", (int)scaleNumber);
				if (!AppendAPDInstrumentation(
					instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size,
					params.nNumViews, activeTracePixels, instrumentExtendedMaps))
				{
					RecordInstrumentSidecarWriteError(
							instrumentSidecarWriteErrors, "apd_iteration.csv_or_apd_traces.jsonl", (int)scaleNumber);
					}
					if (!AppendDVPInstrumentation(
						instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size,
						params, activeTracePixels, instrumentExtendedMaps))
					{
						RecordInstrumentSidecarWriteError(
							instrumentSidecarWriteErrors, "dvp_iteration.csv_or_dvp_traces.jsonl", (int)scaleNumber);
					}
					if (!AppendDVPVisibilityInstrumentation(
						instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size,
						params, activeTracePixels, instrumentExtendedMaps))
					{
						RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors,
							"dvp_visibility_iteration.csv_or_dvp_visibility_traces.jsonl",
							(int)scaleNumber);
					}
					if (!AppendDVPVisibleNormalInstrumentation(
						instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size,
						params, activeTracePixels, instrumentExtendedMaps))
					{
						RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors,
							"dvp_visible_normal_iteration.csv_or_dvp_visible_normal_traces.jsonl",
							(int)scaleNumber);
					}
				if (!AppendInstrumentTimings(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, instrumentKernelTimingsMs))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "timings.csv", (int)scaleNumber);
				if (!WriteDMapIterationCSV(DMapDepthMapDir(instrumentRoot, instrumentImageID, instrumentImageName), instrumentImageID, instrumentImageName, (int)scaleNumber, size, instrumentIterationCounters))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "iteration.csv", (int)scaleNumber);
				if (writeInstrumentMaps && scaleNumber == 0) {
					if (!SaveInstrumentImprovementMaps(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size, numInstrumentPasses, instrumentImprovementMaps, instrumentPassUpdateSources))
						RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "legacy_improvement_and_pass_update_maps", (int)scaleNumber);
				}
			CUDA_CHECK(cudaFree(cudaInstrumentCounters));
			CUDA_CHECK(cudaFree(cudaInstrumentUpdateSources));
			CUDA_CHECK(cudaFree(cudaInstrumentPlanesBeforePass));
			CUDA_CHECK(cudaFree(cudaInstrumentCostsBeforePass));
			CUDA_CHECK(cudaFree(cudaInstrumentSelectedViewsBeforePass));
			if (cudaInstrumentValidBeforeFilter)
				CUDA_CHECK(cudaFree(cudaInstrumentValidBeforeFilter));
			if (cudaInstrumentFilterRejectReasons)
				CUDA_CHECK(cudaFree(cudaInstrumentFilterRejectReasons));
			if (cudaInstrumentImprovementMaps)
				CUDA_CHECK(cudaFree(cudaInstrumentImprovementMaps));
			if (cudaInstrumentPassUpdateSources)
				CUDA_CHECK(cudaFree(cudaInstrumentPassUpdateSources));
			if (cudaInstrumentPassDepthDeltas)
				CUDA_CHECK(cudaFree(cudaInstrumentPassDepthDeltas));
			if (cudaInstrumentPassDepthRelDeltas)
				CUDA_CHECK(cudaFree(cudaInstrumentPassDepthRelDeltas));
			if (cudaInstrumentPassNormalAngleDeltas)
				CUDA_CHECK(cudaFree(cudaInstrumentPassNormalAngleDeltas));
			if (cudaInstrumentPassViewChurn)
				CUDA_CHECK(cudaFree(cudaInstrumentPassViewChurn));
			if (cudaInstrumentLogicalStoredCosts)
				CUDA_CHECK(cudaFree(cudaInstrumentLogicalStoredCosts));
			if (cudaInstrumentLogicalScorePrimary)
				CUDA_CHECK(cudaFree(cudaInstrumentLogicalScorePrimary));
			if (cudaInstrumentLogicalScoreSecondary)
				CUDA_CHECK(cudaFree(cudaInstrumentLogicalScoreSecondary));
			if (cudaInstrumentExactLogicalScorePrimary)
				CUDA_CHECK(cudaFree(cudaInstrumentExactLogicalScorePrimary));
			if (cudaInstrumentExactLogicalScoreSecondary)
				CUDA_CHECK(cudaFree(cudaInstrumentExactLogicalScoreSecondary));
			if (cudaInstrumentExactPixels)
				CUDA_CHECK(cudaFree(cudaInstrumentExactPixels));
				if (cudaInstrumentExactViews)
					CUDA_CHECK(cudaFree(cudaInstrumentExactViews));
				if (cudaInstrumentAPDCounters)
					CUDA_CHECK(cudaFree(cudaInstrumentAPDCounters));
				if (cudaInstrumentAPDStates)
					CUDA_CHECK(cudaFree(cudaInstrumentAPDStates));
				if (cudaInstrumentAPDUpdates)
					CUDA_CHECK(cudaFree(cudaInstrumentAPDUpdates));
					if (cudaInstrumentAPDTraces)
						CUDA_CHECK(cudaFree(cudaInstrumentAPDTraces));
					if (cudaInstrumentDVPCounters)
						CUDA_CHECK(cudaFree(cudaInstrumentDVPCounters));
					if (cudaInstrumentDVPUpdates)
						CUDA_CHECK(cudaFree(cudaInstrumentDVPUpdates));
					if (cudaInstrumentDVPTraces)
						CUDA_CHECK(cudaFree(cudaInstrumentDVPTraces));
					if (cudaInstrumentVisibilityCounters)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibilityCounters));
					if (cudaInstrumentVisibilityUpdates)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibilityUpdates));
					if (cudaInstrumentVisibilityTraces)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibilityTraces));
					if (cudaInstrumentVisibilityReasons)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibilityReasons));
					if (cudaInstrumentVisibleNormalCounters)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibleNormalCounters));
					if (cudaInstrumentVisibleNormalUpdates)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibleNormalUpdates));
					if (cudaInstrumentVisibleNormalTraces)
						CUDA_CHECK(cudaFree(cudaInstrumentVisibleNormalTraces));
			if (cudaInstrumentFinalViewWeights)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalViewWeights));
			if (cudaInstrumentFinalViewCosts)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalViewCosts));
			if (cudaInstrumentFinalViewPhotometricCosts)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalViewPhotometricCosts));
			if (cudaInstrumentFinalViewGeometricCosts)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalViewGeometricCosts));
			if (cudaInstrumentFinalViewEntropy)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalViewEntropy));
			if (cudaInstrumentFinalLowDepth)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalLowDepth));
			if (cudaInstrumentFinalSelectedViews)
				CUDA_CHECK(cudaFree(cudaInstrumentFinalSelectedViews));
			if (cudaInstrumentAcceptedUpdateCount)
				CUDA_CHECK(cudaFree(cudaInstrumentAcceptedUpdateCount));
			if (cudaInstrumentPlanesBeforeFilter)
				CUDA_CHECK(cudaFree(cudaInstrumentPlanesBeforeFilter));
			if (cudaInstrumentCostsBeforeFilter)
				CUDA_CHECK(cudaFree(cudaInstrumentCostsBeforeFilter));
			if (cudaInstrumentTraceMap)
				CUDA_CHECK(cudaFree(cudaInstrumentTraceMap));
			if (cudaInstrumentTraceRecords)
				CUDA_CHECK(cudaFree(cudaInstrumentTraceRecords));
			writeDMapArtifacts = true;
		}
#endif

		// resident-buffer reuse: recalibrate the confidence as a true extension of the last
		// geometric-consistency iteration, reading the final reference depth+normal
		// (cudaDepthNormalEstimates) and raw NCC cost (cudaDepthNormalCosts) still resident on the
		// device from the kernels above; only the neighbors' raw previous-iteration
		// depth/conf/normal snapshots (host, loaded by InitViews -- the raw-neighbor-conf
		// invariant) are uploaded. The adjusted confidence is downloaded straight into
		// depthData.confMap, overwriting the raw cost RunCUDA already downloaded there (that D2H
		// is kept: it is the conversion input the unpack loop below needs if this launch fails);
		// only the cost->conf conversion is skipped. The kernels use no __constant__ state, so no
		// serialization with other workers' PatchMatch kernels is needed beyond this instance's
		// own stream order.
		// On any CUDA error done stays false, the conversion below runs as usual and the caller
		// falls back to the epilogue (re-upload) path.
		bool bFusedConfDone(false);
		if (pConfRequest && scaleNumber == 0 && params.bGeomConsistency &&
			!depthData.confMap.empty() && depthData.confMap.isContinuous() &&
			depthData.confMap.size() == depthData.depthMap.size()) {
			// neighbor-depth texture reuse: the geometric-consistency pass already holds every
			// neighbor's raw previous-iteration depth in cudaDepthArrays/textureDepths, so point
			// the launcher at the resident texture instead of re-uploading the same map -- but only
			// when the upload above did NOT resize it (view.depthMap.size() == image.size()),
			// otherwise the texture holds an INTER_LINEAR-resized copy while the host pointer (and
			// the CPU path) sample the native map; mismatched neighbors keep the linear upload.
			for (ConfNeighborHost& n : pConfRequest->neighbors) {
				n.texDepth = 0;
				if (n.srcImage >= 1 && (size_t)n.srcImage < depthData.images.size() &&
					(size_t)(n.srcImage-1) < textureDepths.size() && textureDepths[n.srcImage-1] != 0 &&
					depthData.images[n.srcImage].depthMap.size() == images[n.srcImage].size() &&
					n.width == images[n.srcImage].cols && n.height == images[n.srcImage].rows)
					n.texDepth = (unsigned long long)textureDepths[n.srcImage-1];
			}
			const std::chrono::steady_clock::time_point t0(std::chrono::steady_clock::now());
			bFusedConfDone = RunConfidenceFusedCUDA(size.width, size.height,
				cudaDepthNormalEstimates, cudaDepthNormalCosts,
				pConfRequest->k00, pConfRequest->k11, pConfRequest->k02, pConfRequest->k12,
				pConfRequest->neighbors.data(), (int)pConfRequest->neighbors.size(),
				pConfRequest->params,
				(void*)cudaStream, depthData.confMap.ptr<float>());
			pConfRequest->computeNS += std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - t0).count();
			pConfRequest->done = bFusedConfDone;
			#ifdef _USE_DMAP_INSTRUMENTATION
			// RunConfidenceFusedCUDA consumes the resident production cost map directly.
			// Preserve its exact cost-to-confidence input for the later sidecar before the
			// adjusted result overwrites depthData.confMap. This observer-only snapshot is
			// covered by the confidence producer's existing two-map host budget.
			if (bFusedConfDone && instrumentEnabled && !instrumentCostMap.empty()) {
				depthData.confMapBeforeAdjustmentInstrument = instrumentCostMap.clone();
				for (float& confidence: depthData.confMapBeforeAdjustmentInstrument)
					confidence = confidence >= 1.f ? 0.f : 1.f-confidence;
			}
			#endif
		}

		// load depth-map, normal-map and confidence-map from CUDA memory
		for (int r = 0; r < depthData.depthMap.rows; ++r) {
			for (int c = 0; c < depthData.depthMap.cols; ++c) {
				const int index = r * depthData.depthMap.cols + c;
				const Point4& depthNormal = depthNormalEstimates[index];
				const Depth depth = depthNormal.w();
				ASSERT(ISFINITE(depth));
				depthData.depthMap(r, c) = depth;
				depthData.normalMap(r, c) = depthNormal.topLeftCorner<3, 1>();
				if (scaleNumber == 0) {
					if (!bFusedConfDone) {
						// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is
						// best (skipped when the fused recalibration above already replaced confMap
						// with the adjusted confidence, which is no longer a cost)
						ASSERT(!depthData.confMap.empty());
						float& conf = depthData.confMap(r, c);
						conf = conf >= 1.f ? 0.f : 1.f - conf;
					}
					// map pixel views from bit-mask to index
					ASSERT(!depthData.viewsMap.empty());
					ViewsID& views = depthData.viewsMap(r, c);
					if (depth > 0) {
						const uint32_t bitviews(*reinterpret_cast<const uint32_t*>(views.val));
						int j = 0;
						for (int i = 0; i < 32; ++i) {
							if (bitviews & (1 << i)) {
								views[j] = i;
								if (++j == maxPixelViews)
									break;
							}
						}
						while (j < 4)
							views[j++] = 255;
					} else
						views = ViewsID(255, 255, 255, 255);
				}
			}
		}
		#ifdef _USE_DMAP_INSTRUMENTATION
			uint64_t validAfterKeepCost(0), rejectedByIgnoreMask(0);
			const bool ignoreMaskRequested(scaleNumber == 0 && OPTDENSE::nIgnoreMaskLabel >= 0);
			bool ignoreMaskLoaded(false);
			if (scaleNumber == 0) {
				if (writeDMapArtifacts)
					validAfterKeepCost = CountValidDepth(depthData.depthMap);
				// Apply the production ignore mask before publishing final-state
				// artifacts so map-to-DMAP parity and rejection accounting remain exact.
				if (ignoreMaskRequested) {
					const DepthData::ViewData& view(depthData.GetView());
					BitMatrix mask;
					ignoreMaskLoaded = DepthEstimator::ImportIgnoreMask(
						*view.pImageData, depthData.depthMap.size(),
						(uint8_t)OPTDENSE::nIgnoreMaskLabel, mask);
					if (ignoreMaskLoaded) {
						if (writeDMapArtifacts) {
							const size_t area((size_t)depthData.depthMap.area());
							for (int r=0; r<depthData.depthMap.rows; ++r) {
								for (int c=0; c<depthData.depthMap.cols; ++c) {
									if (mask.isSet(r,c))
										continue;
									const size_t index((size_t)r * depthData.depthMap.cols + (size_t)c);
									const bool rejected(depthData.depthMap(r,c) > 0);
									if (rejected) {
										++rejectedByIgnoreMask;
										if (instrumentFilterRejectReasons.size() >= area)
											instrumentFilterRejectReasons[index] = DMAP_REJECTION_MASKED;
										if (instrumentUpdateSources.size() >= area)
											instrumentUpdateSources[index] = PM_SOURCE_FILTERED;
									}
									if (!instrumentCostMap.empty())
										instrumentCostMap[(int)index] = 1.f;
								}
							}
						}
						depthData.ApplyIgnoreMask(mask);
					}
				}
			}
			if (instrumentEnabled && writeInstrumentMaps) {
				if (!SaveInstrumentUpdateMap(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, size, instrumentUpdateSources))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "legacy_update_source_map", (int)scaleNumber);
				if (scaleNumber == 0 &&
					!SaveInstrumentCostMap(instrumentDir, (int)depthData.GetView().GetID(), (int)scaleNumber, instrumentCostMap))
					RecordInstrumentSidecarWriteError(instrumentSidecarWriteErrors, "legacy_cost_maps", (int)scaleNumber);
			}
			if (writeDMapArtifacts) {
				WriteDMapArtifacts(
					instrumentRoot,
					depthData,
					params,
					(int)scaleNumber,
					numInstrumentPasses,
					instrumentIterationCounters,
					instrumentCostMap,
					instrumentUpdateSources,
					instrumentValidBeforeFilter,
					instrumentFilterRejectReasons,
					instrumentImprovementMaps,
					instrumentPassUpdateSources,
					instrumentSidecarWriteErrors,
					writeInstrumentMaps,
					scaleNumber == 0 ? &instrumentExtendedMaps : nullptr,
					geometricIteration,
					validAfterKeepCost,
					rejectedByIgnoreMask,
					ignoreMaskRequested,
					OPTDENSE::nIgnoreMaskLabel,
					ignoreMaskLoaded);
			}
		#endif

		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
			lowResViewsMap = depthData.viewsMap;
		}
	}
	const bool apdHasNextStage(
		apdRequested &&
		(geometricIteration < 0 ? OPTDENSE::nEstimationGeometricIters > 0u :
			geometricIteration+1 < (int)OPTDENSE::nEstimationGeometricIters));
	if (apdHasNextStage && apdCarryState.IsValid())
		fullResDepthData.apdMultiscaleState = apdCarryState;
	else
		fullResDepthData.apdMultiscaleState.Release();

	#ifndef _USE_DMAP_INSTRUMENTATION
	// apply ignore mask
	if (OPTDENSE::nIgnoreMaskLabel >= 0) {
		const DepthData::ViewData& view = depthData.GetView();
		BitMatrix mask;
		if (DepthEstimator::ImportIgnoreMask(*view.pImageData, depthData.depthMap.size(), (uint8_t)OPTDENSE::nIgnoreMaskLabel, mask))
			depthData.ApplyIgnoreMask(mask);
	}
	#endif

	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.GetSize() > 2 ?
		String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
		String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		images.front().cols, images.front().rows, TD_TIMER_GET_FMT().c_str());
}
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS

#pragma pop_macro("VERBOSE")

#endif // _USE_CUDA
