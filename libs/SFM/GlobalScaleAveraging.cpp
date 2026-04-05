/*
 * GlobalScaleAveraging.cpp
 *
 * Copyright (c) 2014-2025 SEACAVE
 */

#include "Common.h"
#include "GlobalScaleAveraging.h"

using namespace SFM;


// D E F I N E S ///////////////////////////////////////////////////

#pragma push_macro("VERBOSE")
#undef VERBOSE
#define VERBOSE(...) LOG(lt, __VA_ARGS__)


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("GlbSclAg"));

namespace {

bool SelectFixedNode(
	const std::vector<ScalePair>& pairwiseScales,
	const uint32_t fixedIdx,
	const std::unordered_map<uint32_t, int>& idxToVar,
	uint32_t& fixedNode)
{
	fixedNode = fixedIdx;
	if (fixedNode == NO_ID) {
		std::unordered_map<uint32_t, float> connectionWeights;
		for (const ScalePair& scalePair : pairwiseScales) {
			connectionWeights[scalePair.idxA] += scalePair.weight;
			connectionWeights[scalePair.idxB] += scalePair.weight;
		}
		float maxWeight = 0.f;
		for (const auto& [idx, weight] : connectionWeights) {
			if (weight > maxWeight) {
				maxWeight = weight;
				fixedNode = idx;
			}
		}
	}
	if (fixedNode == NO_ID || idxToVar.find(fixedNode) == idxToVar.end()) {
		VERBOSE("warning: invalid fixed index for scale estimation");
		return false;
	}
	return true;
}

} // namespace


bool GlobalScaleEstimator::EstimateScales(
	const std::vector<ScalePair>& pairwiseScales,
	const uint32_t numIndices,
	const uint32_t fixedIdx,
	std::vector<REAL>& outScales)
{
	if (pairwiseScales.empty() || numIndices == 0) {
		VERBOSE("warning: no pairwise scales provided");
		return false;
	}

	// Map index to variable position in the linear system
	std::unordered_map<uint32_t, int> idxToVar;
	for (const ScalePair& scalePair : pairwiseScales) {
		ASSERT(scalePair.scaleRatio > REAL(0));
		ASSERT(ISFINITE(scalePair.scaleRatio));
		ASSERT(scalePair.weight > 0.f);
		ASSERT(ISFINITE(scalePair.weight));
		const auto [itA, insertedA] = idxToVar.emplace(scalePair.idxA, (int)idxToVar.size());
		(void)itA;
		(void)insertedA;
		const auto [itB, insertedB] = idxToVar.emplace(scalePair.idxB, (int)idxToVar.size());
		(void)itB;
		(void)insertedB;
	}

	const int N = (int)idxToVar.size(); // number of variables (scales)
	const int M = (int)pairwiseScales.size(); // number of equations
	if (N < 2) {
		VERBOSE("warning: insufficient indices for scale estimation");
		return false;
	}

	uint32_t fixedNode;
	if (!SelectFixedNode(pairwiseScales, fixedIdx, idxToVar, fixedNode))
		return false;

	// Eliminate the fixed variable for exact gauge enforcement.
	std::unordered_map<uint32_t, int> idxToTmpVar;
	idxToTmpVar.reserve((size_t)N - 1);
	for (const auto& [idx, varIdx] : idxToVar) {
		(void)varIdx;
		if (idx == fixedNode)
			continue;
		idxToTmpVar.emplace(idx, (int)idxToTmpVar.size());
	}
	const int Ntmp = (int)idxToTmpVar.size();
	if (Ntmp == 0) {
		outScales.resize(numIndices, REAL(1));
		return true;
	}

	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(M, Ntmp);
	Eigen::VectorXd b = Eigen::VectorXd::Zero(M);
	for (int i = 0; i < M; ++i) {
		const ScalePair& scalePair = pairwiseScales[i];
		const double weight = (double)scalePair.weight;
		const double logRatio = LOGN((double)scalePair.scaleRatio);
		auto itA = idxToTmpVar.find(scalePair.idxA);
		auto itB = idxToTmpVar.find(scalePair.idxB);
		if (itA != idxToTmpVar.end())
			A(i, itA->second) = -weight;
		if (itB != idxToTmpVar.end())
			A(i, itB->second) = weight;
		b(i) = weight * logRatio;
	}

	const Eigen::VectorXd x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

	outScales.resize(numIndices, REAL(1));
	for (const auto& [idx, tmpVarIdx] : idxToTmpVar) {
		ASSERT(idx < numIndices);
		outScales[idx] = EXP(x(tmpVarIdx));
	}
	if (fixedNode < numIndices)
		outScales[fixedNode] = REAL(1);

	DEBUG("Global scale averaging completed: %u indices, %u pairs (exact gauge, fixed %u)",
		N, M, fixedNode);
	return true;
}
/*----------------------------------------------------------------*/
