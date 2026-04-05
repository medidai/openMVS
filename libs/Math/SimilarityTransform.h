/*
* SimilarityTransform.h
*
* Copyright (c) 2014-2022 SEACAVE
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

#ifndef _SEACAVE_SIMILARITY_TRANSFORM_H_
#define _SEACAVE_SIMILARITY_TRANSFORM_H_


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SEACAVE {

// 7-DOF similarity transform (scale, rotation, translation)
struct MATH_API Transform
{
	RMatrix R;  // rotation matrix (3x3)
	Point3 t;   // translation
	REAL scale; // uniform scale

	Transform() : R(RMatrix::IDENTITY), t(Point3::ZERO), scale(REAL(1)) {}

	// Generate a random transform with random rotation, translation, and scale
	static Transform Random(std::mt19937& rng, REAL maxRotationAngleDeg = 15, REAL maxTranslation = 1.0, REAL minScalePercent = 0.3);

	// Invert this transform
	Transform Invert() const;

	// Compose with another transform (T_res = this * T_other)
	Transform operator*(const Transform& other) const;

	// Transform a point
	Point3 operator*(const Point3& p) const;

	// Convert to/from Eigen transform
	Eigen::Affine3d ToEigen() const;
	Transform& FromEigen(const Eigen::Affine3d& T);
};
/*----------------------------------------------------------------*/

// compute the similarity transform that best aligns the given two sets of corresponding 3D points
MATH_API Matrix4x4 SimilarityTransform(const Point3Arr& points, const Point3Arr& pointsRef);

// decompose similarity transform into rotation, translation and scale
MATH_API void DecomposeSimilarityTransform(const Matrix4x4& transform, Matrix3x3& R, Point3& t, REAL& s);

// Estimate similarity transform from 3D point correspondences
MATH_API Transform EstimateSimilarityTransform(const Point3Arr& srcPoints, const Point3Arr& dstPoints);
/*----------------------------------------------------------------*/

// estimate the rotation that best maps srcRots to dstRots (dstR = srcR * alignR) in a
// robust way against outliers; uses a consensus seed followed by IRLS refinement on SO(3)
MATH_API bool EstimateRotationAlignment(
	const Matrix3x3Arr& srcRots, const Matrix3x3Arr& dstRots,
	Matrix3x3& alignR,
	REAL inlierThresholdDeg = 10, unsigned maxRefineIters = 15);
/*----------------------------------------------------------------*/

// assembly/decomposition of projection matrix: P=KR[I|-C]
MATH_API void DecomposeProjectionMatrix(const PMatrix& P, KMatrix& K, RMatrix& R, CMatrix& C);
MATH_API void DecomposeProjectionMatrix(const PMatrix& P, RMatrix& R, CMatrix& C);
MATH_API void AssembleProjectionMatrix(const KMatrix& K, const RMatrix& R, const CMatrix& C, PMatrix& P);
MATH_API void AssembleProjectionMatrix(const RMatrix& R, const CMatrix& C, PMatrix& P);
/*----------------------------------------------------------------*/

} // namespace SEACAVE

#endif // _SEACAVE_SIMILARITY_TRANSFORM_H_
