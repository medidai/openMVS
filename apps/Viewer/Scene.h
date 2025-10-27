/*
 * Scene.h
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

#pragma once

#include <cstdint>
#include "Window.h"

namespace VIEWER {

class Scene {
public:
	typedef MVS::PointCloud::Octree OctreePoints;
	typedef MVS::Mesh::Octree OctreeMesh;

public:
	struct DensifyWorkflowOptions {
		unsigned resolutionLevel{1};
		unsigned maxResolution{2560};
		unsigned minResolution{640};
		unsigned subResolutionLevels{2};
		#ifdef _USE_CUDA
		unsigned numViews{8};
		#else
		unsigned numViews{5};
		#endif
		unsigned minViews{3};
		unsigned minViewsTrust{2};
		unsigned minViewsFuse{2};
		#ifdef _USE_CUDA
		unsigned estimationIters{4};
		#else
		unsigned estimationIters{3};
		#endif
		unsigned geometricIters{2};
		unsigned fuseFilter{2};
		bool estimateColors{true};
		bool estimateNormals{true};
		bool removeDepthMaps{false};
		bool postprocess{false};
		int fusionMode{0};
		float fDepthReprojectionErrorThreshold{1.2f};
		bool cropToROI{true};
		float borderROI{0.f};
		float sampleMeshNeighbors{0.f};
	};

	struct ReconstructMeshWorkflowOptions {
		float minPointDistance{1.5f};
		bool useFreeSpaceSupport{false};
		bool useOnlyROI{false};
		bool constantWeight{true};
		float thicknessFactor{1.f};
		float qualityFactor{1.f};
		float decimateMesh{1.f};
		unsigned targetFaceNum{0};
		float removeSpurious{20.f};
		bool removeSpikes{true};
		unsigned closeHoles{30};
		unsigned smoothSteps{2};
		float edgeLength{0.f};
		bool cropToROI{true};
	};

	struct RefineMeshWorkflowOptions {
		unsigned resolutionLevel{0};
		unsigned minResolution{640};
		unsigned maxViews{8};
		float decimateMesh{0.f};
		unsigned closeHoles{30};
		unsigned ensureEdgeSize{1};
		unsigned maxFaceArea{32};
		unsigned scales{2};
		float scaleStep{0.5f};
		unsigned alternatePair{0};
		float regularityWeight{0.2f};
		float rigidityElasticityRatio{0.9f};
		float gradientStep{45.05f};
		float planarVertexRatio{0.f};
		unsigned reduceMemory{1};
	};

	struct TextureMeshWorkflowOptions {
		float decimateMesh{1.f};
		unsigned closeHoles{30};
		unsigned resolutionLevel{0};
		unsigned minResolution{640};
		unsigned minCommonCameras{0};
		float outlierThreshold{6e-2f};
		float ratioDataSmoothness{0.1f};
		bool globalSeamLeveling{true};
		bool localSeamLeveling{true};
		unsigned textureSizeMultiple{0};
		unsigned rectPackingHeuristic{3};
		uint32_t emptyColor{0x00FF7F27};
		float sharpnessWeight{0.5f};
		int ignoreMaskLabel{-1};
		int maxTextureSize{8192};
	};

public:
	ARCHIVE_TYPE nArchiveType;
	String name;

	String sceneName;
	String geometryName;
	bool geometryMesh;
	bool estimateSfMNormals;
	bool estimateSfMPatches;
	MVS::Scene scene;
	Window window;
	ImageArr images; // scene photos (only valid)

	DensifyWorkflowOptions densifyOptions;
	ReconstructMeshWorkflowOptions reconstructOptions;
	RefineMeshWorkflowOptions refineOptions;
	TextureMeshWorkflowOptions textureOptions;

	OctreePoints octPoints;
	OctreeMesh octMesh;

	// multi-threading
	static SEACAVE::EventQueue events; // internal events queue (processed by the working threads)
	static SEACAVE::Thread thread; // worker thread

public:
	explicit Scene(ARCHIVE_TYPE _nArchiveType = ARCHIVE_MVS);
	~Scene();

	bool Initialize(const cv::Size& size, const String& windowName, 
				   const String& fileName = String(), const String& geometryFileName = String());
	void Run();

	void Reset();
	void Release();

	inline bool IsValid() const { return window.IsValid(); }
	inline bool IsOpen() const { return IsValid() && !scene.IsEmpty(); }
	inline bool IsOctreeValid() const { return !octPoints.IsEmpty() || !octMesh.IsEmpty(); }

	// Scene management
	bool Open(const String& fileName, String geometryFileName = {});
	bool Save(const String& fileName = String(), bool bRescaleImages = false);
	bool Export(const String& fileName, const String& exportType = String(), bool bViews = true) const;

	// Workflows
	bool RunDensifyWorkflow(const DensifyWorkflowOptions& options, bool bUpdateGeometry=true);
	bool RunReconstructMeshWorkflow(const ReconstructMeshWorkflowOptions& options, bool bUpdateGeometry=true);
	bool RunRefineMeshWorkflow(const RefineMeshWorkflowOptions& options, bool bUpdateGeometry=true);
	bool RunTextureMeshWorkflow(const TextureMeshWorkflowOptions& options, bool bUpdateGeometry=true);

	// Geometry operations
	void RemoveSelectedGeometry();
	void SetROIFromSelection(bool aabb = false);
	MVS::Scene CropToPoints(const MVS::PointCloud::IndexArr& selectedPointIndices, unsigned minPoints = 20) const;

	// Getters
	DensifyWorkflowOptions& GetDensifyWorkflowOptions() { return densifyOptions; }
	const DensifyWorkflowOptions& GetDensifyWorkflowOptions() const { return densifyOptions; }
	ReconstructMeshWorkflowOptions& GetReconstructMeshWorkflowOptions() { return reconstructOptions; }
	const ReconstructMeshWorkflowOptions& GetReconstructMeshWorkflowOptions() const { return reconstructOptions; }
	RefineMeshWorkflowOptions& GetRefineMeshWorkflowOptions() { return refineOptions; }
	const RefineMeshWorkflowOptions& GetRefineMeshWorkflowOptions() const { return refineOptions; }
	TextureMeshWorkflowOptions& GetTextureMeshWorkflowOptions() { return textureOptions; }
	const TextureMeshWorkflowOptions& GetTextureMeshWorkflowOptions() const { return textureOptions; }
	const MVS::Scene& GetScene() const { return scene; }
	MVS::Scene& GetScene() { return scene; }
	const ImageArr& GetImages() const { return images; } 
	ImageArr& GetImages() { return images; }
	Window& GetWindow() { return window; }
	MVS::IIndex ImageIdxMVS2Viewer(MVS::IIndex idx) const;

	// Event handlers
	void OnCenterScene(const Point3f& center);
	void OnCastRay(const Ray3d&, int button, int action, int mods);
	void OnSetCameraViewMode(MVS::IIndex camID);
	void OnSelectPointsByCamera(bool highlightCameraVisiblePoints);

private:
	void CropToBounds();
	void TogleSceneBox();

	// Internal geometry operation
	void UpdateGeometryAfterModification();

	static void* ThreadWorker(void*);
};

} // namespace VIEWER
