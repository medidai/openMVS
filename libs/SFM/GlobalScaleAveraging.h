/*
 * GlobalScaleAveraging.h
 *
 * Copyright (c) 2014-2025 SEACAVE
 */

#ifndef _SFM_GLOBAL_SCALE_AVERAGING_H_
#define _SFM_GLOBAL_SCALE_AVERAGING_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Camera.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SFM {

/**
 * @brief Pairwise scale ratio between two sub-scenes or images
 */
struct SFM_API ScalePair
{
	uint32_t idxA;      // First index (sub-scene or image)
	uint32_t idxB;      // Second index (sub-scene or image)
	REAL scaleRatio;    // scale_B / scale_A
	float weight;       // confidence weight (e.g., number of correspondences)

	ScalePair() : idxA(NO_ID), idxB(NO_ID), scaleRatio(REAL(1)), weight(0.f) {}
	ScalePair(uint32_t a, uint32_t b, REAL ratio, float w)
		: idxA(a), idxB(b), scaleRatio(ratio), weight(w) {}
};

/**
 * @brief Global scale estimation from pairwise scale ratios
 *
 * Estimates global scales by solving a weighted least-squares system:
 *   log(s_j) - log(s_i) ≈ log(scale_ratio_ij)
 *
 * This is solved via SVD decomposition with exact gauge enforcement by
 * eliminating one fixed-scale variable (scale = 1.0).
 *
 * Generalized from StarInitializer::EstimateGlobalScale() to work with
 * arbitrary indices (sub-scenes, images, etc.) instead of image pairs.
 */
class SFM_API GlobalScaleEstimator
{
public:
	/**
	 * @brief Estimate global scales from pairwise ratios
	 * @param pairwiseScales Vector of pairwise scale ratios
	 * @param numIndices Total number of indices (scenes/images)
	 * @param fixedIdx Optional fixed index to set exact scale=1 (NO_ID = auto-select best-connected index)
	 * @param outScales Output vector of global scales (indexed by scene/image ID)
	 * @return true if estimation successful
	 */
	bool EstimateScales(
		const std::vector<ScalePair>& pairwiseScales,
		const uint32_t numIndices,
		const uint32_t fixedIdx,
		std::vector<REAL>& outScales);

	bool EstimateScales(
		const std::vector<ScalePair>& pairwiseScales,
		const uint32_t numIndices,
		std::vector<REAL>& outScales) {
		return EstimateScales(pairwiseScales, numIndices, NO_ID, outScales);
	}
};
/*----------------------------------------------------------------*/

} // namespace SFM

#endif // _SFM_GLOBAL_SCALE_AVERAGING_H_
