/*
* SceneRefineCUDA.cu
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

#include "SceneRefineCUDA.inl"

#include <cuda_fp16.h>


namespace MVS {

namespace CUDA {


// D E V I C E   H E L P E R S ////////////////////////////////////////

// Read half-float from surface, return float
__device__ inline float readSurfHalf(cudaSurfaceObject_t surf, int x, int y) {
	unsigned short h;
	surf2Dread(&h, surf, x * (int)sizeof(unsigned short), y);
	return __half2float(*reinterpret_cast<const __half*>(&h));
}

// Write float as half-float to surface
__device__ inline void writeSurfHalf(cudaSurfaceObject_t surf, int x, int y, float val) {
	const __half h = __float2half(val);
	surf2Dwrite(*reinterpret_cast<const unsigned short*>(&h), surf, x * (int)sizeof(unsigned short), y);
}

// Atomic min for float using CAS loop (valid for positive floats)
__device__ inline bool atomicMinFloat(float* addr, float value) {
	unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
	unsigned int old = *addr_as_uint;
	unsigned int assumed;
	do {
		if (__uint_as_float(old) <= value)
			return false;
		assumed = old;
		old = atomicCAS(addr_as_uint, assumed, __float_as_uint(value));
	} while (assumed != old);
	return true;
}

// Atomic add a Point3 to a device array
__device__ inline void atomicAddPoint3(Point3* addr, const Point3& val) {
	atomicAdd(&addr->x(), val.x());
	atomicAdd(&addr->y(), val.y());
	atomicAdd(&addr->z(), val.z());
}
/*----------------------------------------------------------------*/


// K E R N E L S ////////////////////////////////////////////////////

// 1. ProjectMesh — 1D, 1 thread per visible face
__global__ void kernelProjectMesh(
	const Point3* __restrict__ vertices,
	const Point3u* __restrict__ faces,
	const uint32_t* __restrict__ faceIDs,
	float* __restrict__ depthMap,
	uint32_t* __restrict__ faceMap,
	uint16_t* __restrict__ baryMap,
	Camera camera,
	uint32_t numFacesView)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numFacesView) return;

	const uint32_t faceID = faceIDs[tid];
	const Point3u& face = faces[faceID];

	// Load and project 3 vertices
	const Point3 Xc0 = camera.pose.TransformPointW2C(vertices[face.x()]);
	const Point3 Xc1 = camera.pose.TransformPointW2C(vertices[face.y()]);
	const Point3 Xc2 = camera.pose.TransformPointW2C(vertices[face.z()]);

	const Point2 p0 = camera.model.TransformPointC2I(Xc0);
	const Point2 p1 = camera.model.TransformPointC2I(Xc1);
	const Point2 p2 = camera.model.TransformPointC2I(Xc2);

	// Front-face check via determinant
	const Point2 e10 = p1 - p0, e20 = p2 - p0;
	const float det = e10.x() * e20.y() - e20.x() * e10.y();
	const float invDet = 1.f / det;
	if (invDet <= 0.f) return;

	// Bounding box with ±0.5 padding, clamped to 5-pixel border
	const int border = 5;
	const int ixMin = max(__float2int_ru(fminf(fminf(p0.x(), p1.x()), p2.x()) - 0.5f), border);
	const int ixMax = min(__float2int_rd(fmaxf(fmaxf(p0.x(), p1.x()), p2.x()) + 0.5f), camera.size.x() - border);
	const int iyMin = max(__float2int_ru(fminf(fminf(p0.y(), p1.y()), p2.y()) - 0.5f), border);
	const int iyMax = min(__float2int_rd(fmaxf(fmaxf(p0.y(), p1.y()), p2.y()) + 0.5f), camera.size.y() - border);
	if (ixMin > ixMax || iyMin > iyMax) return;

	const int width = camera.size.x();
	for (int iy = iyMin; iy <= iyMax; ++iy) {
		for (int ix = ixMin; ix <= ixMax; ++ix) {
			const Point2 d((float)ix - p0.x(), (float)iy - p0.y());

			// Barycentric coordinates
			const float b1 = (d.x() * e20.y() - e20.x() * d.y()) * invDet;
			const float b2 = (e10.x() * d.y() - d.x() * e10.y()) * invDet;
			const float b0 = 1.f - b1 - b2;
			if (b0 < 0.f || b1 < 0.f || b2 < 0.f) continue;

			const float depth = b0 * Xc0.z() + b1 * Xc1.z() + b2 * Xc2.z();

			const int pixIdx = iy * width + ix;
			if (atomicMinFloat(&depthMap[pixIdx], depth)) {
				faceMap[pixIdx] = faceID;
				baryMap[pixIdx * 3 + 0] = __half_as_ushort(__float2half(b0));
				baryMap[pixIdx * 3 + 1] = __half_as_ushort(__float2half(b1));
				baryMap[pixIdx * 3 + 2] = __half_as_ushort(__float2half(b2));
			}
		}
	}
}


// 2. CrossCheckProjection — 2D
__global__ void kernelCrossCheckProjection(
	float* __restrict__ depthMap,
	uint32_t* __restrict__ faceMap,
	int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (__float_as_uint(depthMap[pixIdx]) == 0x7F7FFFFFu || faceMap[pixIdx] == (uint32_t)-1) {
		depthMap[pixIdx] = 0.f;
		faceMap[pixIdx] = (uint32_t)-1;
	}
}


// 3. ImageMeshWarp — 2D, texture + 2 surfaces
__global__ void kernelImageMeshWarp(
	const float* __restrict__ depthMapA,
	const float* __restrict__ depthMapB,
	uint8_t* __restrict__ mask,
	Camera camA,
	Camera camB,
	cudaTextureObject_t texImageB,
	cudaSurfaceObject_t surfImageA,
	cudaSurfaceObject_t surfImageProj)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= camA.size.x() || y >= camA.size.y()) return;

	const int pixIdx = y * camA.size.x() + x;
	unsigned short convergePix = 0;
	uint8_t convergeMask = 0;

	surf2Dread(&convergePix, surfImageA, x * (int)sizeof(unsigned short), y);

	const float depthA = depthMapA[pixIdx];
	if (depthA > 0.f) {
		const Point3 X_world = camA.TransformPointI2W(Point2((float)x, (float)y), depthA);
		const Point3 Xc_B = camB.pose.TransformPointW2C(X_world);
		const float pz = Xc_B.z();

		if (pz > 0.f) {
			const Point2 projB = camB.model.TransformPointC2I(Xc_B);
			const float xB = projB.x(), yB = projB.y();
			const float borderMin = 10.f;
			const float borderMaxX = (float)(camB.size.x() - 10);
			const float borderMaxY = (float)(camB.size.y() - 10);

			if (xB > borderMin && xB < borderMaxX && yB > borderMin && yB < borderMaxY) {
				const int ixB = __float2int_rz(xB);
				const int iyB = __float2int_rz(yB);
				const int widthB = camB.size.x();
				const int idxB = iyB * widthB + ixB;
				const float tol = 0.01f * pz;

				bool consistent = false;
				if (fabsf(depthMapB[idxB] - pz) < tol) consistent = true;
				else if (fabsf(depthMapB[idxB + 1] - pz) < tol) consistent = true;
				else if (fabsf(depthMapB[idxB + widthB] - pz) < tol) consistent = true;
				else if (fabsf(depthMapB[idxB + widthB + 1] - pz) < tol) consistent = true;

				if (consistent) {
					const float texVal = tex2D<float>(texImageB, xB, yB);
					const __half h = __float2half(texVal);
					convergePix = *reinterpret_cast<const unsigned short*>(&h);
					convergeMask = 1;
				}
			}
		}
	}

	surf2Dwrite(convergePix, surfImageProj, x * (int)sizeof(unsigned short), y);
	mask[pixIdx] = convergeMask;
}


// 4. ComputeImageMean — 2D
__global__ void kernelComputeImageMean(
	const uint8_t* __restrict__ mask,
	float* __restrict__ imageMean,
	cudaSurfaceObject_t surfImage,
	int width, int height, int halfSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (x < halfSize || y < halfSize || x >= width - halfSize || y >= height - halfSize || mask[pixIdx] != 1) {
		imageMean[pixIdx] = 0.f;
		return;
	}

	const float windowArea = (float)(2 * halfSize + 1) * (float)(2 * halfSize + 1);
	float sum = 0.f;
	for (int dy = -halfSize; dy <= halfSize; ++dy)
		for (int dx = -halfSize; dx <= halfSize; ++dx)
			sum += readSurfHalf(surfImage, x + dx, y + dy);
	imageMean[pixIdx] = sum / windowArea;
}


// 5. ComputeImageVar — 2D
__global__ void kernelComputeImageVar(
	const float* __restrict__ imageMean,
	const uint8_t* __restrict__ mask,
	float* __restrict__ imageVar,
	cudaSurfaceObject_t surfImage,
	int width, int height, int halfSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (x < halfSize || y < halfSize || x >= width - halfSize || y >= height - halfSize || mask[pixIdx] != 1) {
		imageVar[pixIdx] = 0.f;
		return;
	}

	const float windowArea = (float)(2 * halfSize + 1) * (float)(2 * halfSize + 1);
	const float mean = imageMean[pixIdx];
	float sum = 0.f;
	for (int dy = -halfSize; dy <= halfSize; ++dy) {
		for (int dx = -halfSize; dx <= halfSize; ++dx) {
			const float diff = readSurfHalf(surfImage, x + dx, y + dy) - mean;
			sum += diff * diff;
		}
	}
	imageVar[pixIdx] = fmaxf(sum / windowArea, 1e-4f);
}


// 6. ComputeImageCov — 2D
__global__ void kernelComputeImageCov(
	const float* __restrict__ imageMeanA,
	const float* __restrict__ imageMeanB,
	const uint8_t* __restrict__ mask,
	float* __restrict__ imageCov,
	cudaSurfaceObject_t surfImageA,
	cudaSurfaceObject_t surfImageProj,
	int width, int height, int halfSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (x < halfSize || y < halfSize || x >= width - halfSize || y >= height - halfSize || mask[pixIdx] != 1) {
		imageCov[pixIdx] = 0.f;
		return;
	}

	const float windowArea = (float)(2 * halfSize + 1) * (float)(2 * halfSize + 1);
	const float meanA = imageMeanA[pixIdx], meanB = imageMeanB[pixIdx];
	float sum = 0.f;
	for (int dy = -halfSize; dy <= halfSize; ++dy)
		for (int dx = -halfSize; dx <= halfSize; ++dx)
			sum += (readSurfHalf(surfImageA, x+dx, y+dy) - meanA) * (readSurfHalf(surfImageProj, x+dx, y+dy) - meanB);
	imageCov[pixIdx] = sum / windowArea;
}


// 7. ComputeImageZNCC — 2D
__global__ void kernelComputeImageZNCC(
	const float* __restrict__ imageCov,
	const float* __restrict__ imageVarA,
	const float* __restrict__ imageVarB,
	const uint8_t* __restrict__ mask,
	float* __restrict__ imageZNCC,
	int width, int height, int halfSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (x < halfSize || y < halfSize || x >= width - halfSize || y >= height - halfSize || mask[pixIdx] != 1) {
		imageZNCC[pixIdx] = 0.f;
		return;
	}
	imageZNCC[pixIdx] = imageCov[pixIdx] / sqrtf(imageVarA[pixIdx] * imageVarB[pixIdx]);
}


// 8. ComputeImageDZNCC — 2D
__global__ void kernelComputeImageDZNCC(
	const float* __restrict__ meanA,
	const float* __restrict__ meanB,
	const float* __restrict__ varA,
	const float* __restrict__ varB,
	const float* __restrict__ zncc,
	const uint8_t* __restrict__ mask,
	float* __restrict__ dzncc,
	cudaSurfaceObject_t surfImageA,
	cudaSurfaceObject_t surfImageProj,
	int width, int height, int halfSize)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (x < halfSize || y < halfSize || x >= width - halfSize || y >= height - halfSize || mask[pixIdx] != 1) {
		dzncc[pixIdx] = 0.f;
		return;
	}

	float sumInvSqrtVarProd = 0.f, sumZnccOverVar = 0.f, sumMeanTerm = 0.f, count = 0.f;
	for (int dy = -halfSize; dy <= halfSize; ++dy) {
		const int ny = y + dy;
		if (ny < halfSize || ny >= height - halfSize) continue;
		for (int dx = -halfSize; dx <= halfSize; ++dx) {
			const int nx = x + dx;
			if (nx < halfSize || nx >= width - halfSize) continue;
			const int nIdx = ny * width + nx;
			if (mask[nIdx] != 1) continue;
			const float sqrtVarProd = sqrtf(varA[nIdx] * varB[nIdx]);
			if (sqrtVarProd == 0.f) continue;
			const float invSqrtVarProd = 1.f / sqrtVarProd;
			const float znccOverVar = zncc[nIdx] / varB[nIdx];
			sumInvSqrtVarProd += invSqrtVarProd;
			sumZnccOverVar += znccOverVar;
			sumMeanTerm += meanA[nIdx] * invSqrtVarProd - meanB[nIdx] * znccOverVar;
			count += 1.f;
		}
	}
	if (count == 0.f) { dzncc[pixIdx] = 0.f; return; }

	const float pixA = readSurfHalf(surfImageA, x, y);
	const float pixB = readSurfHalf(surfImageProj, x, y);
	const float gradient = (-pixA * sumInvSqrtVarProd + pixB * sumZnccOverVar + sumMeanTerm) / count;

	const float minVar = fminf(varA[pixIdx], varB[pixIdx]);
	dzncc[pixIdx] = gradient * minVar / (minVar + 1.5e-3f);
}


// 9. ComputePhotometricGradient — 2D, texture + atomicAdd
__global__ void kernelComputePhotometricGradient(
	const Point3u* __restrict__ faces,
	const Point3* __restrict__ normals,
	const float* __restrict__ depthMap,
	const uint32_t* __restrict__ faceMap,
	const uint16_t* __restrict__ baryMap,
	const float* __restrict__ dznccMap,
	const uint8_t* __restrict__ mask,
	Point3* __restrict__ photoGrad,
	float* __restrict__ photoGradPixels,
	Camera camA,
	Camera camB,
	cudaTextureObject_t texImageB,
	float regScale,
	int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	const int pixIdx = y * width + x;
	if (mask[pixIdx] != 1) return;

	const float depth = depthMap[pixIdx];
	const uint32_t faceID = faceMap[pixIdx];
	const float bary0 = __half2float(*reinterpret_cast<const __half*>(&baryMap[pixIdx * 3 + 0]));
	const float bary1 = __half2float(*reinterpret_cast<const __half*>(&baryMap[pixIdx * 3 + 1]));
	const float bary2 = __half2float(*reinterpret_cast<const __half*>(&baryMap[pixIdx * 3 + 2]));

	const Point3u& face = faces[faceID];
	const Point3& normal = normals[faceID];

	// View direction in world space (normalized)
	const Point3 camRay = camA.model.TransformPointI2C(Point2((float)x, (float)y));
	const Point3 worldDir = camA.pose.R.transpose() * camRay;
	const Point3 viewDir = worldDir.normalized();

	const float viewDotNormal = viewDir.dot(normal);
	if (viewDotNormal > -0.1f) return;

	// Back-project to 3D and forward-project to camera B
	const Point3 X_world = camA.TransformPointI2W(Point2((float)x, (float)y), depth);
	const Point3 Xc_B = camB.pose.TransformPointW2C(X_world);
	const float pz = Xc_B.z();

	Point2 projB;
	if (pz > 0.f)
		projB = camB.model.TransformPointC2I(Xc_B);
	else
		projB = Point2(-1.f, -1.f);

	// Jacobian d(u,v)/d(X_world): KR = K * R
	const Matrix3 KR = camB.model.K() * camB.pose.R;
	const Point3 p = camB.model.K() * Xc_B; // raw projection before perspective divide
	const float pz2 = pz * pz;

	// du/dX = (KR.row(0)*pz - KR.row(2)*px) / pz², same for dv/dX
	const Point3 dudX = (KR.row(0).transpose() * pz - KR.row(2).transpose() * p.x()) / pz2;
	const Point3 dvdX = (KR.row(1).transpose() * pz - KR.row(2).transpose() * p.y()) / pz2;

	// Image derivatives at projected point
	const float pixC = tex2D<float>(texImageB, projB.x(), projB.y());
	const float dx = tex2D<float>(texImageB, projB.x() + 1.f, projB.y()) - pixC;
	const float dy = tex2D<float>(texImageB, projB.x(), projB.y() + 1.f) - pixC;

	// 3D gradient = dzncc * J^T * [dx, dy]
	const float dz = dznccMap[pixIdx];
	const Point3 grad = dz * (dx * dudX + dy * dvdX);

	// Project gradient along view direction, scale by 1/dot(viewDir, normal)
	const float projMag = grad.dot(viewDir) / viewDotNormal;

	// Distribute to 3 vertices weighted by bary coords × regScale × normal
	atomicAddPoint3(&photoGrad[face.x()], (regScale * bary0 * projMag) * normal);
	atomicAddPoint3(&photoGrad[face.y()], (regScale * bary1 * projMag) * normal);
	atomicAddPoint3(&photoGrad[face.z()], (regScale * bary2 * projMag) * normal);

	atomicAdd(&photoGradPixels[face.x()], 1.f);
	atomicAdd(&photoGradPixels[face.y()], 1.f);
	atomicAdd(&photoGradPixels[face.z()], 1.f);
}


// 10. UpdatePhotoGradNorm — 1D
__global__ void kernelUpdatePhotoGradNorm(
	float* __restrict__ photoGradNorm,
	const float* __restrict__ photoGradPixels,
	uint32_t numVertices)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numVertices) return;
	if (photoGradPixels[tid] > 0.f)
		photoGradNorm[tid] += 1.f;
}


// 11. ComputeSmoothnessGradient — 1D
__global__ void kernelComputeSmoothnessGradient(
	const Point3* __restrict__ vertices,
	const uint32_t* __restrict__ vertVertices,
	const uint32_t* __restrict__ vertSizes,
	const uint32_t* __restrict__ vertPointers,
	Point3* __restrict__ smoothGrad,
	uint32_t numVertices,
	uint8_t mode)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numVertices) return;

	const uint32_t numNeighbors = vertSizes[tid];
	if (numNeighbors == 0) {
		smoothGrad[tid] = Point3::Zero();
		return;
	}

	const uint32_t ptr = vertPointers[tid];
	const float invN = 1.f / (float)numNeighbors;

	// Both modes: vertex - (1/N)*sum(neighbors)
	Point3 result = vertices[tid];
	float totalWeight = 1.f;
	for (uint32_t i = 0; i < numNeighbors; ++i) {
		const uint32_t ni = vertVertices[ptr + i];
		result -= vertices[ni] * invN;
		if (mode != 0) {
			// Valence-weighted: accumulate 1/(Ni*N) where Ni = valence of neighbor
			totalWeight += invN / (float)vertSizes[ni];
		}
	}
	if (mode != 0)
		result /= totalWeight;
	smoothGrad[tid] = result;
}


// 12. CombineGradients — 1D
__global__ void kernelCombineGradients(
	Point3* __restrict__ photoGrad,
	const float* __restrict__ photoGradNorm,
	const Point3* __restrict__ smoothGrad,
	uint32_t numVertices,
	float smoothWeight)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numVertices) return;

	const float norm = photoGradNorm[tid];
	if (norm > 0.f)
		photoGrad[tid] = photoGrad[tid] / norm + smoothWeight * smoothGrad[tid];
	else
		photoGrad[tid] = smoothWeight * smoothGrad[tid];
}


// 13. CombineAllGradients — 1D
__global__ void kernelCombineAllGradients(
	Point3* __restrict__ photoGrad,
	const float* __restrict__ photoGradNorm,
	const Point3* __restrict__ smoothGrad1,
	const Point3* __restrict__ smoothGrad2,
	uint32_t numVertices,
	float rigidity,
	float elasticity)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numVertices) return;

	const float norm = photoGradNorm[tid];
	if (norm > 0.f)
		photoGrad[tid] = photoGrad[tid] / norm + rigidity * smoothGrad1[tid] + elasticity * smoothGrad2[tid];
	else
		photoGrad[tid] = rigidity * smoothGrad1[tid] + elasticity * smoothGrad2[tid];
}


// 14. ComputeFaceNormal — 1D, 1 thread per face
__global__ void kernelComputeFaceNormal(
	const Point3* __restrict__ vertices,
	const Point3u* __restrict__ faces,
	Point3* __restrict__ normals,
	uint32_t numFaces)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= (int)numFaces) return;
	const Point3u& face = faces[tid];
	const Point3 v0 = vertices[face.x()];
	const Point3 v1 = vertices[face.y()];
	const Point3 v2 = vertices[face.z()];
	const Point3 e1 = v1 - v0;
	const Point3 e2 = v2 - v0;
	const Point3 n = e1.cross(e2);
	normals[tid] = n.normalized();
}
/*----------------------------------------------------------------*/


// H O S T   L A U N C H E R S ////////////////////////////////////////

void LaunchProjectMesh(
	const Point3* vertices, const Point3u* faces, const uint32_t* faceIDs,
	float* depthMap, uint32_t* faceMap, uint16_t* baryMap,
	const Camera& camera, uint32_t numFacesView)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numFacesView + blockSize - 1) / blockSize;
	kernelProjectMesh<<<numBlocks, blockSize>>>(
		vertices, faces, faceIDs, depthMap, faceMap, baryMap, camera, numFacesView);
}

void LaunchCrossCheckProjection(float* depthMap, uint32_t* faceMap, int width, int height)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelCrossCheckProjection<<<grid, block>>>(depthMap, faceMap, width, height);
}

void LaunchImageMeshWarp(
	const float* depthMapA, const float* depthMapB, uint8_t* mask,
	const Camera& camA, const Camera& camB,
	cudaTextureObject_t texImageB, cudaSurfaceObject_t surfImageA, cudaSurfaceObject_t surfImageProj)
{
	const dim3 block(16, 16);
	const dim3 grid((camA.size.x() + block.x - 1) / block.x, (camA.size.y() + block.y - 1) / block.y);
	kernelImageMeshWarp<<<grid, block>>>(depthMapA, depthMapB, mask, camA, camB, texImageB, surfImageA, surfImageProj);
}

void LaunchComputeImageMean(const uint8_t* mask, float* imageMean, cudaSurfaceObject_t surfImage, int width, int height, int halfSize)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputeImageMean<<<grid, block>>>(mask, imageMean, surfImage, width, height, halfSize);
}

void LaunchComputeImageVar(const float* imageMean, const uint8_t* mask, float* imageVar, cudaSurfaceObject_t surfImage, int width, int height, int halfSize)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputeImageVar<<<grid, block>>>(imageMean, mask, imageVar, surfImage, width, height, halfSize);
}

void LaunchComputeImageCov(
	const float* imageMeanA, const float* imageMeanB, const uint8_t* mask, float* imageCov,
	cudaSurfaceObject_t surfImageA, cudaSurfaceObject_t surfImageProj, int width, int height, int halfSize)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputeImageCov<<<grid, block>>>(imageMeanA, imageMeanB, mask, imageCov, surfImageA, surfImageProj, width, height, halfSize);
}

void LaunchComputeImageZNCC(const float* imageCov, const float* imageVarA, const float* imageVarB, const uint8_t* mask, float* imageZNCC, int width, int height, int halfSize)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputeImageZNCC<<<grid, block>>>(imageCov, imageVarA, imageVarB, mask, imageZNCC, width, height, halfSize);
}

void LaunchComputeImageDZNCC(
	const float* meanA, const float* meanB, const float* varA, const float* varB, const float* zncc,
	const uint8_t* mask, float* dzncc, cudaSurfaceObject_t surfImageA, cudaSurfaceObject_t surfImageProj, int width, int height, int halfSize)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputeImageDZNCC<<<grid, block>>>(meanA, meanB, varA, varB, zncc, mask, dzncc, surfImageA, surfImageProj, width, height, halfSize);
}

void LaunchComputePhotometricGradient(
	const Point3u* faces, const Point3* normals,
	const float* depthMap, const uint32_t* faceMap, const uint16_t* baryMap,
	const float* dzncc, const uint8_t* mask,
	Point3* photoGrad, float* photoGradPixels,
	const Camera& camA, const Camera& camB,
	cudaTextureObject_t texImageB, float regScale, int width, int height)
{
	const dim3 block(16, 16);
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	kernelComputePhotometricGradient<<<grid, block>>>(
		faces, normals, depthMap, faceMap, baryMap, dzncc, mask,
		photoGrad, photoGradPixels, camA, camB, texImageB, regScale, width, height);
}

void LaunchUpdatePhotoGradNorm(float* photoGradNorm, const float* photoGradPixels, uint32_t numVertices)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numVertices + blockSize - 1) / blockSize;
	kernelUpdatePhotoGradNorm<<<numBlocks, blockSize>>>(photoGradNorm, photoGradPixels, numVertices);
}

void LaunchComputeSmoothnessGradient(
	const Point3* vertices, const uint32_t* vertVertices, const uint32_t* vertSizes, const uint32_t* vertPointers,
	Point3* smoothGrad, uint32_t numVertices, uint8_t mode)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numVertices + blockSize - 1) / blockSize;
	kernelComputeSmoothnessGradient<<<numBlocks, blockSize>>>(vertices, vertVertices, vertSizes, vertPointers, smoothGrad, numVertices, mode);
}

void LaunchCombineGradients(Point3* photoGrad, const float* photoGradNorm, const Point3* smoothGrad, uint32_t numVertices, float smoothWeight)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numVertices + blockSize - 1) / blockSize;
	kernelCombineGradients<<<numBlocks, blockSize>>>(photoGrad, photoGradNorm, smoothGrad, numVertices, smoothWeight);
}

void LaunchCombineAllGradients(
	Point3* photoGrad, const float* photoGradNorm, const Point3* smoothGrad1, const Point3* smoothGrad2,
	uint32_t numVertices, float rigidity, float elasticity)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numVertices + blockSize - 1) / blockSize;
	kernelCombineAllGradients<<<numBlocks, blockSize>>>(photoGrad, photoGradNorm, smoothGrad1, smoothGrad2, numVertices, rigidity, elasticity);
}

void LaunchComputeFaceNormal(
	const Point3* vertices, const Point3u* faces, Point3* normals, uint32_t numFaces)
{
	const int blockSize = 256;
	const int numBlocks = ((int)numFaces + blockSize - 1) / blockSize;
	kernelComputeFaceNormal<<<numBlocks, blockSize>>>(vertices, faces, normals, numFaces);
}
/*----------------------------------------------------------------*/

} // namespace CUDA

} // namespace MVS
