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

// static max supported views
#define MAX_VIEWS 32

// samples used to perform views selection
#define NUM_SAMPLES 32

// patch window radius
#define nSizeHalfWindow 4

// patch stepping
#define nSizeStep 2


namespace MVS {

namespace CUDA {

#define ImagePixels cudaTextureObject_t
#define RandState curandState

// set/check a bit
__device__ constexpr void SetBit(unsigned& input, unsigned i) {
	input |= (1u << i);
}
__device__ constexpr int IsBitSet(unsigned input, unsigned i) {
	return (input >> i) & 1u;
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
/*----------------------------------------------------------------*/


// generate a random normal
__device__ inline Point3 GenerateRandomNormal(const CUDA::Camera& camera, const Point2i& p, RandState* randState)
{
	float q1, q2, s;
	do {
		q1 = 2.f * curand_uniform(randState) - 1.f;
		q2 = 2.f * curand_uniform(randState) - 1.f;
		s = q1 * q1 + q2 * q2;
	} while (s >= 1.f);
	const float sq = sqrt(1.f - s);
	Point3 normal(
		2.f * q1 * sq,
		2.f * q2 * sq,
		1.f - 2.f * s);

	const Point3 viewDirection = camera.model.ViewDirection(p);
	if (normal.dot(viewDirection) > 0.f)
		normal = -normal;
	return normal.normalized();
}

// randomly perturb a normal
__device__ inline Point3 GeneratePerturbedNormal(const CUDA::Camera& camera, const Point2i& p, const Point3& normal, RandState* randState, const float perturbation)
{
	const Point3 viewDirection = camera.model.ViewDirection(p);

	const float a1 = (curand_uniform(randState) - 0.5f) * perturbation;
	const float a2 = (curand_uniform(randState) - 0.5f) * perturbation;
	const float a3 = (curand_uniform(randState) - 0.5f) * perturbation;

	const float sinA1 = sin(a1);
	const float sinA2 = sin(a2);
	const float sinA3 = sin(a3);
	const float cosA1 = cos(a1);
	const float cosA2 = cos(a2);
	const float cosA3 = cos(a3);

	Matrix3 perturb; perturb <<
		cosA2 * cosA3,
		cosA3 * sinA1 * sinA2 - cosA1 * sinA3,
		sinA1 * sinA3 + cosA1 * cosA3 * sinA2,
		cosA2 * sinA3,
		cosA1 * cosA3 + sinA1 * sinA2 * sinA3,
		cosA1 * sinA2 * sinA3 - cosA3 * sinA1,
		-sinA2,
		cosA2 * sinA1,
		cosA1 * cosA2;

	Point3 normalPerturbed = perturb * normal.topLeftCorner<3,1>();
	if (normalPerturbed.dot(viewDirection) >= 0.f)
		return normal;
	return normalPerturbed.normalized();
}

// randomly perturb a normal
__device__ inline float GeneratePerturbedDepth(float depth, RandState* randState, const float perturbation, const PatchMatch::Params& params)
{
	const float depthMinPerturbed = (1.f - perturbation) * depth;
	const float depthMaxPerturbed = (1.f + perturbation) * depth;
	float depthPerturbed;
	do {
		depthPerturbed = curand_uniform(randState) * (depthMaxPerturbed - depthMinPerturbed) + depthMinPerturbed;
	} while (depthPerturbed < params.fDepthMin && depthPerturbed > params.fDepthMax);
	return depthPerturbed;
}

// interpolate given pixel's estimate to the current position
__device__ inline float InterpolatePixel(const CUDA::Camera& camera, const Point2i& p, const Point2i& np, float depth, const Point3& normal, const PatchMatch::Params& params)
{
	float depthNew;
	if (p.x() == np.x()) {
		const float nx1 = (p.y() - camera.model.p.y()) / camera.model.f.y();
		const float denom = normal.z() + nx1 * normal.y();
		if (abs(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.y() - camera.model.p.y()) / camera.model.f.y();
		const float nom = depth * (normal.z() + x1 * normal.y());
		depthNew = nom / denom;
	} else if (p.y() == np.y()) {
		const float nx1 = (p.x() - camera.model.p.x()) / camera.model.f.x();
		const float denom = normal.z() + nx1 * normal.x();
		if (abs(denom) < FLT_EPSILON)
			return depth;
		const float x1 = (np.x() - camera.model.p.x()) / camera.model.f.x();
		const float nom = depth * (normal.z() + x1 * normal.x());
		depthNew = nom / denom;
	} else {
		const float planeD = normal.dot(camera.model.TransformPointI2C(np.cast<float>(), depth));
		depthNew = planeD / normal.dot(camera.model.TransformPointI2C(p.cast<float>()));
	}
	return (depthNew >= params.fDepthMin && depthNew <= params.fDepthMax) ? depthNew : depth;
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
	const Point3 t = (refCamera.pose.C - trgCamera.pose.C) / (normal.dot(X));
	const Matrix3 H = trgCamera.pose.R * (refCamera.pose.R.transpose() + t*normal.transpose());
	return trgCamera.model.K() * H * refCamera.model.K().inverse();
}

// weight a neighbor texel based on color similarity and distance to the center texel
__device__ inline float ComputeBilateralWeight(int xDist, int yDist, float pix, float centerPix)
{
	constexpr float sigmaSpatial = -1.f / (2.f * (nSizeHalfWindow-1)*(nSizeHalfWindow-1));
	constexpr float sigmaColor = -1.f / (2.f * 25.f/255.f*25.f/255.f);
	const float spatialDistSq = float(xDist * xDist + yDist * yDist);
	const float colorDistSq = Square(pix - centerPix);
	return exp(spatialDistSq * sigmaSpatial + colorDistSq * sigmaColor);
}

// patch texture variance to textureless-factor mapping:
// 0.12: patch texture variance below 0.02 (0.12^2) is considered texture-less
constexpr float smoothSigmaDepth = -1.f / (1.f * 0.02f);

// compute the textureless gate factor for the reference patch: ~1 for textureless
// patches (where NCC is unreliable and segment guidance should kick in), ~0 for
// textured ones; uses the same window, bilateral weights and variance formulation
// as ScorePlane
__device__ inline float ComputeTexturelessFactor(const ImagePixels refImage, const Point2i& p)
{
	float sumRef = 0.f;
	float sumRefRef = 0.f;
	float bilateralWeightSum = 0.f;
	const float refCenterPix = tex2D<float>(refImage, p.x() + 0.5f, p.y() + 0.5f);

	for (int i = -nSizeHalfWindow; i <= nSizeHalfWindow; i += nSizeStep) {
		for (int j = -nSizeHalfWindow; j <= nSizeHalfWindow; j += nSizeStep) {
			const float refPix = tex2D<float>(refImage, p.x() + j + 0.5f, p.y() + i + 0.5f);
			const float weight = ComputeBilateralWeight(j, i, refPix, refCenterPix);
			sumRef += weight * refPix;
			sumRefRef += weight * refPix * refPix;
			bilateralWeightSum += weight;
		}
	}
	const float varRef = sumRefRef * bilateralWeightSum - sumRef * sumRef;
	return exp(varRef * smoothSigmaDepth);
}

// segment-guidance tuning (used by segment-gated propagation and per-segment plane fit)
constexpr float thTextureless = 0.5f;        // textureless-factor above which NCC is unreliable
constexpr int   SEGMENT_PATCH_DILATION = 2;  // patch dilation for textureless segmented pixels
constexpr float thSegmentFitCost = 0.3f;     // max aggregated cost to anchor a segment plane fit
constexpr int   minSegmentFitCount = 32;     // min confident+textured pixels to accept a fit

// smallest-eigenvalue eigenvector of a symmetric 3x3 matrix (analytic; Smith 1961).
// Used to fit a plane normal to the 3D positions of a segment's confident pixels: the
// normal is the eigenvector of the point-scatter covariance with the least variance.
// Returns a zero vector if the fit is degenerate.
__device__ inline Point3 SmallestEigenvector3x3(float a00, float a11, float a22, float a01, float a02, float a12)
{
	const float p1 = a01*a01 + a02*a02 + a12*a12;
	if (p1 < 1e-18f) {
		// already diagonal: axis of the smallest diagonal entry
		if (a00 <= a11 && a00 <= a22) return Point3(1.f, 0.f, 0.f);
		if (a11 <= a22)               return Point3(0.f, 1.f, 0.f);
		return Point3(0.f, 0.f, 1.f);
	}
	const float q = (a00 + a11 + a22) / 3.f;
	const float b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
	const float p2 = b00*b00 + b11*b11 + b22*b22 + 2.f*p1;
	const float p = sqrt(p2 / 6.f);
	if (p < 1e-18f)
		return Point3(0.f, 0.f, 0.f);
	const float invp = 1.f / p;
	const float d00 = b00*invp, d11 = b11*invp, d22 = b22*invp;
	const float d01 = a01*invp, d02 = a02*invp, d12 = a12*invp;
	float r = 0.5f * (d00*(d11*d22 - d12*d12)
	                - d01*(d01*d22 - d12*d02)
	                + d02*(d01*d12 - d11*d02));
	r = fminf(fmaxf(r, -1.f), 1.f);
	const float phi = acosf(r) / 3.f;
	// smallest eigenvalue of the symmetric matrix
	const float eig = q + 2.f * p * cosf(phi + 2.f * (float)M_PI / 3.f);
	// eigenvector = null space of (A - eig*I): cross product of the two most independent rows
	const Point3 r0(a00 - eig, a01, a02);
	const Point3 r1(a01, a11 - eig, a12);
	const Point3 r2(a02, a12, a22 - eig);
	Point3 v = r0.cross(r1); float best = v.squaredNorm();
	const Point3 va = r0.cross(r2); const float na = va.squaredNorm();
	if (na > best) { v = va; best = na; }
	const Point3 vb = r1.cross(r2); const float nb = vb.squaredNorm();
	if (nb > best) { v = vb; best = nb; }
	if (best < 1e-18f)
		return Point3(0.f, 0.f, 0.f);
	return v / sqrt(best);
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
	const float dist = diff.norm();
	return min(maxDist, sqrt(dist*(dist+2.f)));
}

// compute photometric score using weighted ZNCC
__device__ float ScorePlane(const ImagePixels refImage, const CUDA::Camera& refCamera, const ImagePixels trgImage, const CUDA::Camera& trgCamera, const Point2i& p, const Point4& plane, const float lowDepth, const PatchMatch::Params& params, const int dilation = 1)
{
	constexpr float maxCost = 1.2f;

	// patch dilation keeps the same number of taps but spreads them over a wider area
	// (radius/step scaled by 'dilation') so textureless patches can reach nearby texture;
	// the bilateral spatial weight is evaluated at the un-dilated tap index to preserve
	// the original weighting profile, and the tap count is unchanged so cost scale matches.
	const int halfWindow = nSizeHalfWindow * dilation;
	const int stepWindow = nSizeStep * dilation;

	Matrix3 H = ComputeHomography(refCamera, trgCamera, p.cast<float>(), plane);
	const Point2 pt = (H * p.cast<float>().homogeneous()).hnormalized();
	if (pt.x() >= trgCamera.size.x() || pt.x() < 0.f || pt.y() >= trgCamera.size.y() || pt.y() < 0.f)
		return maxCost;
	Point3 X = H * Point2(p.x()-halfWindow, p.y()-halfWindow).homogeneous();
	Point3 baseX(X);
	H *= float(stepWindow);

	float sumRef = 0.f;
	float sumRefRef = 0.f;
	float sumTrg = 0.f;
	float sumTrgTrg = 0.f;
	float sumRefTrg = 0.f;
	float bilateralWeightSum = 0.f;
	const float refCenterPix = tex2D<float>(refImage, p.x() + 0.5f, p.y() + 0.5f);
	for (int i = -halfWindow; i <= halfWindow; i += stepWindow) {
		for (int j = -halfWindow; j <= halfWindow; j += stepWindow) {
			const Point2i refPt = Point2i(p.x() + j, p.y() + i);
			const Point2 trgPt = X.hnormalized();
			const float refPix = tex2D<float>(refImage, refPt.x() + 0.5f, refPt.y() + 0.5f);
			const float trgPix = tex2D<float>(trgImage, trgPt.x() + 0.5f, trgPt.y() + 0.5f);
			const float weight = ComputeBilateralWeight(j / dilation, i / dilation, refPix, refCenterPix);
			const float weightRefPix = weight * refPix;
			const float weightTrgPix = weight * trgPix;
			sumRef += weightRefPix;
			sumTrg += weightTrgPix;
			sumRefRef += weightRefPix * refPix;
			sumTrgTrg += weightTrgPix * trgPix;
			sumRefTrg += weightRefPix * trgPix;
			bilateralWeightSum += weight;
			X += H.col(0);
		}
		baseX += H.col(1);
		X = baseX;
	}

	const float varRef = sumRefRef * bilateralWeightSum - sumRef * sumRef;
	if (lowDepth <= 0 && varRef < 1e-8f)
		return maxCost;
	const float varTrg = sumTrgTrg * bilateralWeightSum - sumTrg * sumTrg;
	const float varRefTrg = varRef * varTrg;
	if (varRefTrg < 1e-16f)
		return maxCost;
	const float covarTrgRef = sumRefTrg * bilateralWeightSum - sumRef * sumTrg;
	float ncc = 1.f - covarTrgRef / sqrt(varRefTrg);

	// apply depth prior weight based on patch textureless
	if (lowDepth > 0) {
		const float depth(plane.w());
		const float deltaDepth(MIN((abs(lowDepth-depth) / lowDepth), 0.5f));
		const float factorDeltaDepth(exp(varRef * smoothSigmaDepth));
		ncc = (1.f-factorDeltaDepth)*ncc + factorDeltaDepth*deltaDepth;
	}
	return max(0.f, min(2.f, ncc));
}

// compute photometric score for all neighbor images
__device__ inline void MultiViewScorePlane(const ImagePixels *images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const Point2i& p, const Point4& plane, const float lowDepth, float* costVector, const PatchMatch::Params& params, const int dilation = 1)
{
	for (int imgId = 1; imgId <= params.nNumViews; ++imgId)
		costVector[imgId-1] = ScorePlane(images[0], cameras[0], images[imgId], cameras[imgId], p, plane, lowDepth, params, dilation);
	if (params.bGeomConsistency)
		for (int imgId = 0; imgId < params.nNumViews; ++imgId)
			costVector[imgId] += 0.1f * GeometricConsistencyWeight(depthImages[imgId], cameras[0], cameras[imgId+1], plane, p);
}
// same as above, but interpolate the plane to current pixel position
__device__ inline float MultiViewScoreNeighborPlane(const ImagePixels* images, const ImagePixels* depthImages, const CUDA::Camera* cameras, const Point2i& p, const Point2i& np, Point4 plane, const float lowDepth, float* costVector, const PatchMatch::Params& params, const int dilation = 1)
{
	plane.w() = InterpolatePixel(cameras[0], p, np, plane.w(), plane.topLeftCorner<3,1>(), params);
	MultiViewScorePlane(images, depthImages, cameras, p, plane, lowDepth, costVector, params, dilation);
	return plane.w();
}

// aggregate photometric score from all images
__device__ inline float AggregateMultiViewScores(const unsigned* viewWeights, const float* costVector, int numViews)
{
	float cost = 0;
	for (int imgId = 0; imgId < numViews; ++imgId)
		if (viewWeights[imgId])
			cost += viewWeights[imgId] * costVector[imgId];
	return cost / NUM_SAMPLES;
}

// propagate and refine the plane estimate for the current pixel employing the asymmetric approach described in:
// "Multi-View Stereo with Asymmetric Checkerboard Propagation and Multi-Hypothesis Joint View Selection", 2018
__device__ void ProcessPixel(const ImagePixels* images, const ImagePixels* depthImages, const CUDA::Camera* cameras, Point4* planes, const float* lowDepths, const uint16_t* priorSegments, const Point4* segmentPlanes, float* costs, RandState* randStates, unsigned* selectedViews, const Point2i& p, const PatchMatch::Params& params, const int iter)
{
	const int width = cameras[0].size.x();
	const int height = cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	RandState* randState = &randStates[idx];
	float lowDepth = 0;
	if (params.bLowResProcessed)
		lowDepth = lowDepths[idx];

	// segment-gated propagation: on a textureless pixel that belongs to a MoGe planar
	// segment, restrict neighbor-hypothesis selection to neighbors sharing the same
	// segment id. This lets a confident hypothesis at a textured segment border spread
	// across the blank interior of the same plane instead of leaking hypotheses across
	// segment boundaries (e.g. from a neighboring wall). Pixels without a segment
	// (id 0) or with enough texture behave exactly as on develop.
	const uint16_t segment = (priorSegments != NULL) ? priorSegments[idx] : (uint16_t)0;
	const bool segmentGated = (segment != 0) && (ComputeTexturelessFactor(images[0], p) > thTextureless);
	// dilated patch scoring for textureless segmented pixels: applied uniformly to every
	// score in this pixel (propagation, self-cost, all refine candidates) so comparisons
	// and the stored cost stay on the same footing.
	const int dilation = segmentGated ? SEGMENT_PATCH_DILATION : 1;
	// globally-fitted plane for this pixel's segment (normal + offset c); zero = no fit yet
	Point4 segmentPlane = Point4::Zero();
	bool segmentPlaneValid = false;
	if (segmentGated && segmentPlanes != NULL) {
		segmentPlane = segmentPlanes[segment];
		segmentPlaneValid = segmentPlane.topLeftCorner<3,1>().squaredNorm() > 0.5f;
	}

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
			// on textureless segmented pixels, only accept hypotheses from the same segment
			if (segmentGated && priorSegments[nidx] != segment) {
				continue;
			}
			const float nconf = costs[nidx];
			if (bestConf > nconf) {
				bestNx = np;
				bestConf = nconf;
			}
		}
		if (bestConf < FLT_MAX) {
			valid[posId] = true;
			positions[posId] = Point2Idx(bestNx, width);
			neighborDepths[posId] = MultiViewScoreNeighborPlane(images, depthImages, cameras, p, bestNx, planes[positions[posId]], lowDepth, costArray[posId], params, dilation);
		}
	}

	// multi-hypothesis view selection
	float viewSelectionPriors[MAX_VIEWS] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	for (int posId = 0; posId < 4; ++posId) {
		if (valid[posId]) {
			const unsigned selectedView = selectedViews[neighborPositions[posId]];
			for (int j = 0; j < params.nNumViews; ++j)
				viewSelectionPriors[j] += (IsBitSet(selectedView, j) ? 0.9f : 0.1f);
		}
	}
	float samplingProbs[MAX_VIEWS];
	constexpr float thCostBad = 1.2f;
	const float thCost = 0.8f * exp(Square((float)iter) / (-2.f * 4.f*4.f));
	for (int imgId = 0; imgId < params.nNumViews; ++imgId) {
		float sumW = 0;
		unsigned count = 0;
		unsigned countBad = 0;
		for (int posId = 0; posId < 8; posId++) {
			if (valid[posId]) {
				if (costArray[posId][imgId] < thCost) {
					sumW += exp(Square(costArray[posId][imgId]) / (-2.f * 0.3f*0.3f));
					++count;
				} else if (costArray[posId][imgId] > thCostBad) {
					++countBad;
				}
			}
		}
		if (count > 2 && countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * sumW / count;
		} else if (countBad < 3) {
			samplingProbs[imgId] = viewSelectionPriors[imgId] * exp(Square(thCost) / (-2.f * 0.4f*0.4f));
		} else {
			samplingProbs[imgId] = 0.f;
		}
	}
	PDF2CDF(samplingProbs, params.nNumViews);
	unsigned viewWeights[MAX_VIEWS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	for (int sample = 0; sample < NUM_SAMPLES; ++sample) {
		const float randProb = curand_uniform(randState);
		for (int imgId = 0; imgId < params.nNumViews; ++imgId) {
			if (samplingProbs[imgId] > randProb) {
				++viewWeights[imgId];
				break;
			}
		}
	}

	// propagate best neighbor plane
	Point4& plane = planes[idx];
	float& cost = costs[idx];
	unsigned newSelectedViews = 0;
	for (int imgId = 0; imgId < params.nNumViews; ++imgId)
		if (viewWeights[imgId])
			SetBit(newSelectedViews, imgId);
	float finalCosts[8];
	for (int posId = 0; posId < 8; ++posId) {
		// costArray[posId] is uninitialized local memory for invalid directions (image edges). Scoring
		// them would feed garbage into the argmin below and make results non-deterministic
		finalCosts[posId] = valid[posId] ? AggregateMultiViewScores(viewWeights, costArray[posId], params.nNumViews) : FLT_MAX;
	}
	const int minCostIdx = FindMinIndex(finalCosts, 8);
	float costVector[MAX_VIEWS];
	MultiViewScorePlane(images, depthImages, cameras, p, plane, lowDepth, costVector, params, dilation);
	cost = AggregateMultiViewScores(viewWeights, costVector, params.nNumViews);
	if (finalCosts[minCostIdx] < cost) {
		ASSERT(valid[minCostIdx]);
		plane = planes[positions[minCostIdx]];
		plane.w() = neighborDepths[minCostIdx];
		cost = finalCosts[minCostIdx];
		selectedViews[idx] = newSelectedViews;
	}
	const float depth = plane.w();

	// refine estimate
	constexpr float perturbationDepth = 0.005f;
	constexpr float perturbationNormal = 0.01f * (float)M_PI;
	const float depthPerturbed = GeneratePerturbedDepth(depth, randState, perturbationDepth, params);
	const Point3 perturbedNormal = GeneratePerturbedNormal(cameras[0], p, plane.topLeftCorner<3,1>(), randState, perturbationNormal);
	const Point3 normalRand = GenerateRandomNormal(cameras[0], p, randState);
	int numValidPlanes = 3;
	Point3 surfaceNormal;
	if (valid[0] && valid[1] && valid[2] && valid[3]) {
		// estimate normal from surrounding surface
		const Point4 ndepths(
			planes[neighborPositions[0]].w(),
			planes[neighborPositions[1]].w(),
			planes[neighborPositions[2]].w(),
			planes[neighborPositions[3]].w()
		);
		surfaceNormal = ComputeDepthGradient(cameras[0].model, depth, p, ndepths);
		numValidPlanes = 4;
	}
	constexpr int numPlanes = 4;
	const float depths[numPlanes] = {depthPerturbed, depth, depth, depth};
	const Point3 normals[numPlanes] = {plane.topLeftCorner<3,1>(), perturbedNormal, normalRand, surfaceNormal};
	for (int i = 0; i < numValidPlanes; ++i) {
		Point4 newPlane;
		newPlane.topLeftCorner<3,1>() = normals[i];
		newPlane.w() = depths[i];
		MultiViewScorePlane(images, depthImages, cameras, p, newPlane, lowDepth, costVector, params, dilation);
		const float costPlane = AggregateMultiViewScores(viewWeights, costVector, params.nNumViews);
		if (cost > costPlane) {
			cost = costPlane;
			plane = newPlane;
		}
	}

	// segment-plane hypotheses (textureless segmented pixels only): reuse the globally
	// fitted plane normal (a reliable orientation) with three depth candidates, hedging
	// against an inaccurate fitted offset:
	//   1) depth where this pixel's ray meets the fitted plane  -> snaps onto the plane
	//   2) current estimated depth                              -> re-orients a good depth
	//   3) a random depth in [fDepthMin, fDepthMax]             -> searches depth on-plane
	if (segmentPlaneValid) {
		const Point3 segNormal = segmentPlane.topLeftCorner<3,1>();
		const float segOffset = segmentPlane.w();
		float depthRayPlane = plane.w();
		const float denom = segNormal.dot(cameras[0].model.TransformPointI2C(p.cast<float>()));
		if (abs(denom) > FLT_EPSILON) {
			const float d = segOffset / denom;
			if (d >= params.fDepthMin && d <= params.fDepthMax)
				depthRayPlane = d;
		}
		const float depthRand = curand_uniform(randState) * (params.fDepthMax - params.fDepthMin) + params.fDepthMin;
		const float segDepths[3] = { depthRayPlane, plane.w(), depthRand };
		for (int i = 0; i < 3; ++i) {
			Point4 newPlane;
			newPlane.topLeftCorner<3,1>() = segNormal;
			newPlane.w() = segDepths[i];
			MultiViewScorePlane(images, depthImages, cameras, p, newPlane, lowDepth, costVector, params, dilation);
			const float costPlane = AggregateMultiViewScores(viewWeights, costVector, params.nNumViews);
			if (cost > costPlane) {
				cost = costPlane;
				plane = newPlane;
			}
		}
	}
}

// compute the score of the current plane estimate
__device__ void InitializePixelScore(const ImagePixels *images, const ImagePixels* depthImages, const CUDA::Camera* cameras, Point4* planes, const float* lowDepths, float* costs, RandState* randStates, unsigned* selectedViews, const Point2i& p, const PatchMatch::Params params)
{
	const int width = cameras[0].size.x();
	const int height = cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	float lowDepth = 0;
	if (params.bLowResProcessed)
		lowDepth = lowDepths[idx];
	// initialize estimate randomly if not set
	RandState* randState = &randStates[idx];
	curand_init(1234/*threadIdx.x*/, p.y(), p.x(), randState);
	Point4& plane = planes[idx];
	float depth = plane.w();
	if (depth <= 0.f) {
		// generate random plane
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(cameras[0], p, randState);
		plane.w() = curand_uniform(randState) * (params.fDepthMax - params.fDepthMin) + params.fDepthMin;
	} else if (plane.topLeftCorner<3,1>().dot(cameras[0].model.ViewDirection(p)) >= 0.f) {
		// generate random normal
		plane.topLeftCorner<3,1>() = GenerateRandomNormal(cameras[0], p, randState);
	}
	// compute costs
	float costVector[MAX_VIEWS];
	MultiViewScorePlane(images, depthImages, cameras, p, plane, lowDepth, costVector, params);
	// select best views
	float costVectorSorted[MAX_VIEWS];
	Sort(costVector, costVectorSorted, params.nNumViews);
	float cost = 0.f;
	for (int i = 0; i < params.nInitTopK; ++i)
		cost += costVectorSorted[i];
	const float costThreshold = costVectorSorted[params.nInitTopK - 1];
	unsigned& selectedView = selectedViews[idx];
	selectedView = 0;
	for (int imgId = 0; imgId < params.nNumViews; ++imgId)
		if (costVector[imgId] <= costThreshold)
			SetBit(selectedView, imgId);
	costs[idx] = cost / params.nInitTopK;
}
__global__ void InitializeScore(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, const CUDA::Camera* cameras, Point4* planes, const float* lowDepths, float* costs, curandState* randStates, unsigned* selectedViews, const PatchMatch::Params params)
{
	const Point2i p = GetThreadIndex2();
	InitializePixelScore((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, planes, lowDepths, costs, (RandState*)randStates, selectedViews, p, params);
}

// traverse image in a back/red checkerboard pattern
__global__ void BlackPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, const CUDA::Camera* cameras, Point4* planes, const float* lowDepths, const uint16_t* priorSegments, const Point4* segmentPlanes, float* costs, curandState* randStates, unsigned* selectedViews, const PatchMatch::Params params, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 0 : 1);
	ProcessPixel((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, planes, lowDepths, priorSegments, segmentPlanes, costs, (RandState*)randStates, selectedViews, p, params, iter);
}
__global__ void RedPixelProcess(const cudaTextureObject_t* textureImages, const cudaTextureObject_t* textureDepths, const CUDA::Camera* cameras, Point4* planes, const float* lowDepths, const uint16_t* priorSegments, const Point4* segmentPlanes, float* costs, curandState* randStates, unsigned* selectedViews, const PatchMatch::Params params, const int iter)
{
	Point2i p = GetThreadIndex2();
	p.y() = p.y() * 2 + (threadIdx.x % 2 == 0 ? 1 : 0);
	ProcessPixel((const ImagePixels*)textureImages, (const ImagePixels*)textureDepths, cameras, planes, lowDepths, priorSegments, segmentPlanes, costs, (RandState*)randStates, selectedViews, p, params, iter);
}

// accumulate per-segment plane-fit scatter statistics from confident, textured pixels.
// Only textured (reliable NCC) low-cost pixels contribute, so the fit is anchored by
// trustworthy 3D points rather than by textureless pixels that may still be wandering.
__global__ void AccumulateSegmentPlanes(const cudaTextureObject_t* textureImages, const CUDA::Camera* cameras, const Point4* planes, const float* costs, const uint16_t* priorSegments, float* segmentAccum, const PatchMatch::Params params)
{
	const Point2i p = GetThreadIndex2();
	const int width = cameras[0].size.x();
	const int height = cameras[0].size.y();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	const uint16_t s = priorSegments[idx];
	if (s == 0 || (int)s >= params.nSegments)
		return;
	const float depth = planes[idx].w();
	if (depth <= 0.f || costs[idx] >= thSegmentFitCost)
		return;
	const ImagePixels refImage = ((const ImagePixels*)textureImages)[0];
	if (ComputeTexturelessFactor(refImage, p) > thTextureless)
		return; // require texture: only reliable anchors feed the fit
	const Point3 X = cameras[0].model.TransformPointI2C(p.cast<float>(), depth);
	float* acc = segmentAccum + (int)s * PatchMatch::SEGMENT_ACCUM_STRIDE;
	atomicAdd(acc + 0, X.x());
	atomicAdd(acc + 1, X.y());
	atomicAdd(acc + 2, X.z());
	atomicAdd(acc + 3, X.x() * X.x());
	atomicAdd(acc + 4, X.y() * X.y());
	atomicAdd(acc + 5, X.z() * X.z());
	atomicAdd(acc + 6, X.x() * X.y());
	atomicAdd(acc + 7, X.x() * X.z());
	atomicAdd(acc + 8, X.y() * X.z());
	atomicAdd(acc + 9, 1.f);
}

// finalize one plane per segment from the accumulated scatter statistics: the normal is
// the least-variance eigenvector of the point covariance and the offset c = n . centroid.
// Segments with too few anchors are left invalid (zero) so they provide no guidance.
__global__ void FinalizeSegmentPlanes(const float* segmentAccum, Point4* segmentPlanes, const PatchMatch::Params params)
{
	const int s = blockIdx.x * blockDim.x + threadIdx.x;
	if (s >= params.nSegments)
		return;
	segmentPlanes[s] = Point4::Zero();
	if (s == 0)
		return; // id 0 = unassigned
	const float* acc = segmentAccum + s * PatchMatch::SEGMENT_ACCUM_STRIDE;
	const float count = acc[9];
	if (count < (float)minSegmentFitCount)
		return;
	const float inv = 1.f / count;
	const Point3 mean(acc[0]*inv, acc[1]*inv, acc[2]*inv);
	const float c00 = acc[3]*inv - mean.x()*mean.x();
	const float c11 = acc[4]*inv - mean.y()*mean.y();
	const float c22 = acc[5]*inv - mean.z()*mean.z();
	const float c01 = acc[6]*inv - mean.x()*mean.y();
	const float c02 = acc[7]*inv - mean.x()*mean.z();
	const float c12 = acc[8]*inv - mean.y()*mean.z();
	Point3 normal = SmallestEigenvector3x3(c00, c11, c22, c01, c02, c12);
	if (normal.squaredNorm() < 0.5f)
		return; // degenerate fit
	// orient the normal towards the camera (scene points are in front, z>0)
	if (normal.dot(mean) > 0.f)
		normal = -normal;
	Point4 plane;
	plane.topLeftCorner<3,1>() = normal;
	plane.w() = normal.dot(mean); // plane offset c: n . X = c for any X on the plane
	segmentPlanes[s] = plane;
}

// filter depth/normals
__global__ void FilterPlanes(Point4* planes, float* costs, unsigned* selectedViews, int width, int height, const PatchMatch::Params params)
{
	const Point2i p = GetThreadIndex2();
	if (p.x() >= width || p.y() >= height)
		return;
	const int idx = Point2Idx(p, width);
	// filter estimates if the score is not good enough
	Point4& plane = planes[idx];
	float conf = costs[idx];
	if (plane.w() <= 0 || conf >= params.fThresholdKeepCost) {
		conf = 0;
		plane = Point4::Zero();
		selectedViews[idx] = 0;
	}
}
/*----------------------------------------------------------------*/


__host__ void PatchMatch::RunCUDA(float* ptrCostMap, uint32_t* ptrViewsMap)
{
	const unsigned width = cameras[0].size.x();
	const unsigned height = cameras[0].size.y();

	constexpr unsigned BLOCK_W = 32;
	constexpr unsigned BLOCK_H = (BLOCK_W / 2);

	const dim3 blockSize(BLOCK_W, BLOCK_H, 1);
	const dim3 gridSizeFull((width + BLOCK_H - 1) / BLOCK_H, (height + BLOCK_H - 1) / BLOCK_H, 1);
	const dim3 gridSizeCheckerboard((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);

	InitializeScore<<<gridSizeFull, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaDepthNormalEstimates, cudaLowDepths, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, params);
	cudaDeviceSynchronize();

	const dim3 blockSeg(256, 1, 1);
	const dim3 gridSeg((params.nSegments + blockSeg.x - 1) / blockSeg.x, 1, 1);
	for (int iter = 0; iter < params.nEstimationIters; ++iter) {
		// refresh per-segment plane fits from the current confident+textured estimates so
		// textureless pixels get an up-to-date global plane hypothesis this iteration.
		// Skip only when estimates are still random: the first iter of the coarsest
		// photometric level. Upper multi-res levels and the geometric pass start seeded.
		const bool haveSeededEstimates = (iter > 0) || params.bLowResProcessed || params.bGeomConsistency;
		if (params.bUseSegments && params.nSegments > 0 && haveSeededEstimates) {
			cudaMemset(cudaSegmentAccum, 0, sizeof(float) * (size_t)params.nSegments * PatchMatch::SEGMENT_ACCUM_STRIDE);
			AccumulateSegmentPlanes<<<gridSizeFull, blockSize>>>(cudaTextureImages, cudaCameras, cudaDepthNormalEstimates, cudaDepthNormalCosts, cudaPriorSegments, cudaSegmentAccum, params);
			cudaDeviceSynchronize();
			FinalizeSegmentPlanes<<<gridSeg, blockSeg>>>(cudaSegmentAccum, cudaSegmentPlanes, params);
			cudaDeviceSynchronize();
		}
		BlackPixelProcess<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaDepthNormalEstimates, cudaLowDepths, cudaPriorSegments, cudaSegmentPlanes, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, params, iter);
		cudaDeviceSynchronize();
		RedPixelProcess<<<gridSizeCheckerboard, blockSize>>>(cudaTextureImages, cudaTextureDepths, cudaCameras, cudaDepthNormalEstimates, cudaLowDepths, cudaPriorSegments, cudaSegmentPlanes, cudaDepthNormalCosts, cudaRandStates, cudaSelectedViews, params, iter);
		cudaDeviceSynchronize();
	}

	if (params.fThresholdKeepCost > 0)
		FilterPlanes<<<gridSizeFull, blockSize>>>(cudaDepthNormalEstimates, cudaDepthNormalCosts, cudaSelectedViews, width, height, params);

	cudaMemcpy(depthNormalEstimates, cudaDepthNormalEstimates, sizeof(Point4) * width * height, cudaMemcpyDeviceToHost);
	if (ptrCostMap)
		cudaMemcpy(ptrCostMap, cudaDepthNormalCosts, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
	if (ptrViewsMap)
		cudaMemcpy(ptrViewsMap, cudaSelectedViews, sizeof(uint32_t) * width * height, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
}
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS
