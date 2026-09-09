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
