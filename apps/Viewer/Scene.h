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

#include "Window.h"

namespace VIEWER {

class Scene {
public:
	typedef MVS::PointCloud::Octree OctreePoints;
	typedef MVS::Mesh::Octree OctreeMesh;

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
	bool Export(const String& fileName, const String& exportType = String()) const;

	// Geometry operations
	void RemoveSelectedGeometry();
	void SetROIFromSelection(bool aabb = false);
	MVS::Scene CropToPoints(const MVS::PointCloud::IndexArr& selectedPointIndices, unsigned minPoints = 20) const;

	// Getters
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
