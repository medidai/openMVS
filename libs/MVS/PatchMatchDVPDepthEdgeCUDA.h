/*
 * PatchMatchDVPDepthEdgeCUDA.h
 *
 * Shared host/device contract for the DVP-MVS depth-edge topology prior.
 * The monocular prediction is used only to construct offline region labels;
 * it never initializes or overwrites an OpenMVS depth hypothesis.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE __host__ __device__
#else
#define PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE
#endif

namespace MVS {
namespace CUDA {

static constexpr unsigned DVP_DEPTH_EDGE_SCHEMA_VERSION = 2u;
static constexpr uint16_t DVP_DEPTH_EDGE_BOUNDARY_LABEL = 0u;

enum class DVPDepthEdgeMode : uint8_t {
	DISABLED = 0,
	ROBERTS_REGIONS = 1,
	DAV2_PLANARIZED = 2,
	ERODED = 3,
	DILATED = 4,
	PIXEL_REASSIGNED = 5,
};

PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE constexpr bool DVPDepthEdgeModeEnabled(
	unsigned mode)
{
	return mode >= static_cast<unsigned>(DVPDepthEdgeMode::ROBERTS_REGIONS) &&
		mode <= static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED);
}

enum class DVPDepthEdgeConfigStatus : uint8_t {
	VALID = 0,
	INVALID_MODE,
	REQUIRES_APD,
	MISSING_PRIOR_DIRECTORY,
};

struct DVPDepthEdgeConfig {
	unsigned mode = static_cast<unsigned>(DVPDepthEdgeMode::DISABLED);
	bool apdEnabled = false;
	bool priorDirectoryProvided = false;
};

enum class DVPDepthEdgePriorGeometryStatus : uint8_t {
	VALID = 0,
	INVALID_RAW_DIMENSIONS,
	INVALID_PROCESSING_DIMENSIONS,
	PROCESSING_DIMENSIONS_MISMATCH,
};

PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE constexpr DVPDepthEdgePriorGeometryStatus
ValidateDVPDepthEdgePriorGeometry(uint32_t expectedWidth, uint32_t expectedHeight,
	uint32_t rawWidth, uint32_t rawHeight, uint32_t processingWidth,
	uint32_t processingHeight)
{
	if (rawWidth == 0u || rawHeight == 0u)
		return DVPDepthEdgePriorGeometryStatus::INVALID_RAW_DIMENSIONS;
	if (processingWidth == 0u || processingHeight == 0u)
		return DVPDepthEdgePriorGeometryStatus::INVALID_PROCESSING_DIMENSIONS;
	if (processingWidth != expectedWidth || processingHeight != expectedHeight)
		return DVPDepthEdgePriorGeometryStatus::PROCESSING_DIMENSIONS_MISMATCH;
	return DVPDepthEdgePriorGeometryStatus::VALID;
}

PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE constexpr DVPDepthEdgeConfigStatus
ValidateDVPDepthEdgeConfig(const DVPDepthEdgeConfig& config)
{
	if (config.mode > static_cast<unsigned>(DVPDepthEdgeMode::PIXEL_REASSIGNED))
		return DVPDepthEdgeConfigStatus::INVALID_MODE;
	if (!DVPDepthEdgeModeEnabled(config.mode))
		return DVPDepthEdgeConfigStatus::VALID;
	if (!config.apdEnabled)
		return DVPDepthEdgeConfigStatus::REQUIRES_APD;
	if (!config.priorDirectoryProvided)
		return DVPDepthEdgeConfigStatus::MISSING_PRIOR_DIRECTORY;
	return DVPDepthEdgeConfigStatus::VALID;
}

enum class DVPDepthEdgeAnchorReason : uint8_t {
	SAME_REGION = 0,
	MODE_DISABLED,
	CENTER_BOUNDARY,
	ANCHOR_BOUNDARY,
	CROSS_REGION,
	INVALID_ANCHOR_INDEX,
};

struct DVPDepthEdgeAnchorDecision {
	uint16_t centerRegion = DVP_DEPTH_EDGE_BOUNDARY_LABEL;
	uint16_t anchorRegion = DVP_DEPTH_EDGE_BOUNDARY_LABEL;
	DVPDepthEdgeAnchorReason reason = DVPDepthEdgeAnchorReason::MODE_DISABLED;
	bool allowed = true;
};

PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE constexpr DVPDepthEdgeAnchorDecision
ResolveDVPDepthEdgeAnchorDecision(unsigned mode, uint16_t centerRegion,
	uint16_t anchorRegion)
{
	DVPDepthEdgeAnchorDecision result;
	result.centerRegion = centerRegion;
	result.anchorRegion = anchorRegion;
	if (!DVPDepthEdgeModeEnabled(mode))
		return result;
	result.allowed = false;
	if (centerRegion == DVP_DEPTH_EDGE_BOUNDARY_LABEL) {
		result.reason = DVPDepthEdgeAnchorReason::CENTER_BOUNDARY;
		return result;
	}
	if (anchorRegion == DVP_DEPTH_EDGE_BOUNDARY_LABEL) {
		result.reason = DVPDepthEdgeAnchorReason::ANCHOR_BOUNDARY;
		return result;
	}
	if (centerRegion != anchorRegion) {
		result.reason = DVPDepthEdgeAnchorReason::CROSS_REGION;
		return result;
	}
	result.reason = DVPDepthEdgeAnchorReason::SAME_REGION;
	result.allowed = true;
	return result;
}

struct DVPDepthEdgeFilterSummary {
	uint8_t inputCount = 0u;
	uint8_t outputCount = 0u;
	uint8_t rejectedBoundary = 0u;
	uint8_t rejectedCrossRegion = 0u;
	uint8_t rejectedInvalidIndex = 0u;
};

// Compact an anchor list in place. Invalid slots use invalidIndex, matching the
// APD anchor buffer contract. This helper is deliberately free of image or
// model state so host and device fixtures exercise the exact production rule.
PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE inline DVPDepthEdgeFilterSummary
FilterDVPDepthEdgeAnchors(unsigned mode, uint16_t centerRegion,
	const uint16_t* regionLabels, uint32_t regionCount, uint32_t* anchors,
	unsigned anchorCount, unsigned maximumAnchors, uint32_t invalidIndex)
{
	DVPDepthEdgeFilterSummary result;
	result.inputCount = static_cast<uint8_t>(
		anchorCount < maximumAnchors ? anchorCount : maximumAnchors);
	if (!DVPDepthEdgeModeEnabled(mode)) {
		result.outputCount = result.inputCount;
		return result;
	}
	unsigned outputCount(0u);
	for (unsigned slot=0u; slot<result.inputCount; ++slot) {
		const uint32_t anchor(anchors ? anchors[slot] : invalidIndex);
		if (!regionLabels || anchor == invalidIndex || anchor >= regionCount) {
			++result.rejectedInvalidIndex;
			continue;
		}
		const DVPDepthEdgeAnchorDecision decision(
			ResolveDVPDepthEdgeAnchorDecision(mode, centerRegion, regionLabels[anchor]));
		if (!decision.allowed) {
			if (decision.reason == DVPDepthEdgeAnchorReason::CROSS_REGION)
				++result.rejectedCrossRegion;
			else
				++result.rejectedBoundary;
			continue;
		}
		anchors[outputCount++] = anchor;
	}
	if (anchors) {
		for (unsigned slot=outputCount; slot<maximumAnchors; ++slot)
			anchors[slot] = invalidIndex;
	}
	result.outputCount = static_cast<uint8_t>(outputCount);
	return result;
}

struct DVPDepthEdgeCUDAOracleResult {
	DVPDepthEdgeFilterSummary mixedRegions;
	DVPDepthEdgeFilterSummary centerBoundary;
	DVPDepthEdgeFilterSummary disabled;
	uint32_t compactedAnchors[8] = {};
	uint32_t passedChecks = 0u;
};

} // namespace CUDA
} // namespace MVS

#undef PATCHMATCH_DVP_DEPTH_EDGE_HOST_DEVICE
