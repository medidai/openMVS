/*
 * Scene.cpp
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

#include "Common.h"
#include "Scene.h"

using namespace VIEWER;


// D E F I N E S ///////////////////////////////////////////////////

#define IMAGE_MAX_RESOLUTION 1024


// S T R U C T S ///////////////////////////////////////////////////

enum EVENT_TYPE {
	EVT_JOB = 0,
	EVT_CLOSE,
};

class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};
class EVTLoadImage : public Event
{
public:
	Scene* pScene;
	MVS::IIndex idx;
	unsigned nMaxResolution;
	bool Run(void*) {
		Image& image = pScene->images[idx];
		ASSERT(image.idx != NO_ID);
		MVS::Image& imageData = pScene->scene.images[image.idx];
		ASSERT(imageData.IsValid());
		if (imageData.image.empty() && !imageData.ReloadImage(nMaxResolution))
			return false;
		imageData.UpdateCamera(pScene->scene.platforms);
		image.AssignImage(imageData.image);
		imageData.ReleaseImage();
		glfwPostEmptyEvent();
		return true;
	}
	EVTLoadImage(Scene* _pScene, MVS::IIndex _idx, unsigned _nMaxResolution=0)
		: Event(EVT_JOB), pScene(_pScene), idx(_idx), nMaxResolution(_nMaxResolution) {}
};
class EVTComputeOctree : public Event
{
public:
	Scene* pScene;
	bool Run(void*) {
		MVS::Scene& scene = pScene->scene;
		if (!scene.mesh.IsEmpty()) {
			Scene::OctreeMesh octMesh(scene.mesh.vertices, [](Scene::OctreeMesh::IDX_TYPE size, Scene::OctreeMesh::Type /*radius*/) {
				return size > 256;
			});
			scene.mesh.ListIncidentFaces();
			pScene->octMesh.Swap(octMesh);
		}
		if (!scene.pointcloud.IsEmpty()) {
			Scene::OctreePoints octPoints(scene.pointcloud.points, [](Scene::OctreePoints::IDX_TYPE size, Scene::OctreePoints::Type /*radius*/) {
				return size > 512;
			});
			pScene->octPoints.Swap(octPoints);
		}
		return true;
	}
	EVTComputeOctree(Scene* _pScene)
		: Event(EVT_JOB), pScene(_pScene) {}
};

void* Scene::ThreadWorker(void*) {
	while (true) {
		CAutoPtr<Event> evt(events.GetEvent());
		switch (evt->GetID()) {
		case EVT_JOB:
			evt->Run();
			break;
		case EVT_CLOSE:
			return NULL;
		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
	return NULL;
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////

SEACAVE::EventQueue Scene::events;
SEACAVE::Thread Scene::thread;

Scene::Scene(ARCHIVE_TYPE _nArchiveType)
	: nArchiveType(_nArchiveType)
	, geometryMesh(false)
	, estimateSfMNormals(false)
	, estimateSfMPatches(false)
{
}

Scene::~Scene() {
	Release();
}

void Scene::Reset()
{
	octPoints.Release();
	octMesh.Release();
	window.Reset();
	images.Release();
	scene.Release();
	sceneName.clear();
	geometryName.clear();
}
void Scene::Release()
{
	if (window.IsValid())
		window.SetVisible(false);
	if (!thread.isRunning()) {
		events.AddEvent(new EVTClose());
		thread.join();
	}
	Reset();
	window.Release();
	glfwTerminate();
}

bool Scene::Initialize(const cv::Size& size, const String& windowName, const String& fileName, const String& geometryFileName) {
	// initialize window
	if (!window.Initialize(size, windowName, *this)) {
		DEBUG("error: Failed to initialize window");
		return false;
	}
	VERBOSE("OpenGL: %s %s", glGetString(GL_RENDERER), glGetString(GL_VERSION));
	name = windowName;

	// init working thread
	thread.start(ThreadWorker);

	// open scene or init empty scene
	if (!fileName.empty())
		Open(fileName, geometryFileName);
	else
		window.SetVisible(true);
	return true;
}

void Scene::Run() {
	window.Run();
}

bool Scene::Open(const String& fileName, String geometryFileName) {
	ASSERT(!fileName.empty());
	window.SetVisible(false);
	DEBUG_EXTRA("Loading: '%s'", Util::getFileNameExt(fileName).c_str());
	Reset();
	sceneName = fileName;

	// load the scene
	WORKING_FOLDER = Util::getFilePath(fileName);
	INIT_WORKING_FOLDER;
	const MVS::Scene::SCENE_TYPE sceneType(scene.Load(fileName, true));
	if (sceneType == MVS::Scene::SCENE_NA) {
		DEBUG("error: can not open scene '%s'", fileName.c_str());
		window.SetVisible(true);
		return false;
	}
	if (geometryFileName.empty() && sceneType == MVS::Scene::SCENE_INTERFACE)
		geometryFileName = Util::getFileFullName(fileName) + _T(".ply");
	if (!geometryFileName.empty()) {
		// try to load given mesh
		MVS::Mesh mesh;
		MVS::PointCloud pointcloud;
		if (mesh.Load(geometryFileName)) {
			scene.mesh.Swap(mesh);
			geometryName = geometryFileName;
			geometryMesh = true;
		} else
		// try to load as a point-cloud
		if (pointcloud.Load(geometryFileName)) {
			scene.pointcloud.Swap(pointcloud);
			geometryName = geometryFileName;
			geometryMesh = false;
		}
	}
	if (!scene.pointcloud.IsEmpty()) {
		scene.pointcloud.PrintStatistics(scene.images.data(), &scene.obb);
		if (estimateSfMNormals && scene.EstimatePointCloudNormals())
			if (estimateSfMPatches && scene.mesh.IsEmpty())
				scene.EstimateSparseSurface();
	}

	// create octree structure used to accelerate selection functionality
	if (!scene.IsEmpty())
		events.AddEvent(new EVTComputeOctree(this));

	// init scene
	AABB3f bounds(true);
	Point3f sceneCenter(0, 0, 0);
	if (scene.IsBounded()) {
		bounds = scene.obb.GetAABB();
		sceneCenter = bounds.GetCenter();
	} else {
		if (!scene.pointcloud.IsEmpty()) {
			bounds = scene.pointcloud.GetAABB(MINF(3u,scene.nCalibratedImages));
			if (bounds.IsEmpty())
				bounds = scene.pointcloud.GetAABB();
			sceneCenter = scene.pointcloud.GetCenter();
		}
		if (!scene.mesh.IsEmpty()) {
			scene.mesh.ComputeNormalFaces();
			bounds.Insert(scene.mesh.GetAABB());
			sceneCenter = scene.mesh.GetCenter();
		}
	}

	// init images
	AABB3f imageBounds(true);
	images.Reserve(scene.images.size());
	FOREACH(idxImage, scene.images) {
		const MVS::Image& imageData = scene.images[idxImage];
		if (!imageData.IsValid())
			continue;
		images.emplace_back(idxImage);
		imageBounds.InsertFull(Cast<float>(imageData.camera.C));
	}
	if (bounds.IsEmpty() && !imageBounds.IsEmpty()) {
		// if no geometry is present, use image bounds
		imageBounds.Enlarge(0.5);
		bounds = imageBounds;
		sceneCenter = imageBounds.GetCenter();
	}

	// fit camera to scene
	if (!bounds.IsEmpty()) {
		const Point3f sceneSize = bounds.GetSize().cast<float>();
		window.SetSceneBounds(sceneCenter, sceneSize);
	}

	// Set images size for camera view mode
	window.GetCamera().SetMaxCamID(images.size());

	// Set up camera view mode callback
	window.GetCamera().SetCameraViewModeCallback([this](MVS::IIndex camID) {
		OnSetCameraViewMode(camID);
	});

	// set window title
	window.SetTitle(String::FormatString((name + _T(": %s")).c_str(), Util::getFileName(fileName).c_str()));

	// upload render data
	window.UploadRenderData();

	window.SetVisible(true);
	return true;
}

bool Scene::Save(const String& _fileName, bool bRescaleImages) {
	if (!IsOpen())
		return false;
	REAL imageScale = 0;
	if (bRescaleImages) {
		window.SetVisible(false);
		VERBOSE("Enter image resolution scale: ");
		String strScale;
		std::cin >> strScale;
		window.SetVisible(true);
		imageScale = strScale.From<REAL>(0);
	}
	const String fileName(!_fileName.empty() ? _fileName : Util::insertBeforeFileExt(sceneName, _T("_new")));
	MVS::Mesh mesh;
	if (!scene.mesh.IsEmpty() && !geometryName.empty() && geometryMesh)
		mesh.Swap(scene.mesh);
	MVS::PointCloud pointcloud;
	if (!scene.pointcloud.IsEmpty() && !geometryName.empty() && !geometryMesh)
		pointcloud.Swap(scene.pointcloud);
	if (imageScale > 0 && imageScale < 1) {
		// scale and save images
		const String folderName(Util::getFilePath(MAKE_PATH_FULL(WORKING_FOLDER_FULL, fileName)) + String::FormatString("images%d" PATH_SEPARATOR_STR, ROUND2INT(imageScale*100)));
		if (!scene.ScaleImages(0, imageScale, folderName)) {
			DEBUG("error: can not scale scene images to '%s'", folderName.c_str());
			return false;
		}
	}
	if (!scene.Save(fileName, nArchiveType)) {
		DEBUG("error: can not save scene to '%s'", fileName.c_str());
		return false;
	}
	if (!mesh.IsEmpty())
		scene.mesh.Swap(mesh);
	if (!pointcloud.IsEmpty())
		scene.pointcloud.Swap(pointcloud);
	sceneName = fileName;
	return true;
}

bool Scene::Export(const String& _fileName, const String& exportType) const {
	if (!IsOpen())
		return false;
	ASSERT(!sceneName.IsEmpty());
	String lastFileName;
	const String fileName(!_fileName.empty() ? _fileName : sceneName);
	const String baseFileName(Util::getFileFullName(fileName));
	const bool bPoints(scene.pointcloud.Save(lastFileName=(baseFileName+_T("_pointcloud.ply")), nArchiveType==ARCHIVE_MVS));
	const bool bMesh(scene.mesh.Save(lastFileName=(baseFileName+_T("_mesh")+(!exportType.empty()?exportType.c_str():(Util::getFileExt(fileName)==_T(".obj")?_T(".obj"):_T(".ply")))), cList<String>(), true));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2 && (bPoints || bMesh))
		scene.ExportCamerasMLP(Util::getFileFullName(lastFileName)+_T(".mlp"), lastFileName);
	#endif
	AABB3f aabb(true);
	if (scene.IsBounded()) {
		std::ofstream fs(baseFileName+_T("_roi.txt"));
		if (fs)
			fs << scene.obb;
		aabb = scene.obb.GetAABB();
	} else
	if (!scene.pointcloud.IsEmpty()) {
		aabb = scene.pointcloud.GetAABB();
	} else
	if (!scene.mesh.IsEmpty()) {
		aabb = scene.mesh.GetAABB();
	}
	if (!aabb.IsEmpty()) {
		std::ofstream fs(baseFileName+_T("_roi_box.txt"));
		if (fs)
			fs << aabb;
	}
	return bPoints || bMesh;
}

MVS::IIndex Scene::ImageIdxMVS2Viewer(MVS::IIndex idx) const {
	// Convert MVS image index to viewer index
	// The list of images in the viewer is a subset of the MVS images,
	// more exactly only the valid images are stored in the viewer.
	// So we can use a small trick to search fast the index in the viewer:
	// start from the MVS index and search backwards
	MVS::IIndex i = MINF(idx+1, images.size());
	while (i-- > 0)
		if (images[i].idx == idx)
			return i;
	return NO_ID;
}

void Scene::CropToBounds()
{
	if (!IsOpen())
		return;
	if (!scene.IsBounded())
		return;
	scene.pointcloud.RemovePointsOutside(scene.obb);
	scene.mesh.RemoveFacesOutside(scene.obb);
	AABB3f bounds = scene.obb.GetAABB();
	window.SetSceneBounds(bounds.GetCenter(), bounds.GetSize());
}

void Scene::TogleSceneBox()
{
	if (!IsOpen())
		return;
	const auto EnlargeAABB = [](AABB3f aabb) {
		return aabb.Enlarge(aabb.GetSize().maxCoeff()*0.03f);
	};
	if (scene.IsBounded())
		scene.obb = OBB3f(true);
	else if (!scene.mesh.IsEmpty())
		scene.obb.Set(EnlargeAABB(scene.mesh.GetAABB()));
	else if (!scene.pointcloud.IsEmpty())
		scene.obb.Set(EnlargeAABB(scene.pointcloud.GetAABB(window.minViews)));
	window.GetRenderer().UploadBounds(scene);
}

void Scene::OnCenterScene(const Point3f& center) {
	if (!IsOpen())
		return;
	if (window.GetControlMode() != Window::CONTROL_ARCBALL)
		return; // Only allow centering in Arcball mode

	// Calculate direction from current target to new center
	const Eigen::Vector3d currentPos = window.GetCamera().GetPosition();
	const Eigen::Vector3d currentTarget = window.GetCamera().GetTarget();

	// Calculate current distance from camera to target
	const double currentDistance = (currentPos - currentTarget).norm();

	// Zoom in by reducing the distance by 25%
	const double zoomFactor = 0.75;
	const double newDistance = currentDistance * zoomFactor;

	// Calculate direction from new target to current camera position
	const Eigen::Vector3d newTarget = Cast<double>(center);
	Eigen::Vector3d direction = (currentPos - newTarget).normalized();

	// If the direction is too small (camera very close to target), use a default direction
	if (direction.norm() < 0.001)
		direction = Eigen::Vector3d(0, 0, 1); // Default to looking along Z axis

	// Calculate new camera position: newTarget + direction * newDistance
	const Eigen::Vector3d newPosition = newTarget + direction * newDistance;

	// Use ArcballControls animation instead of Camera animation
	window.GetArcballControls().animateTo(newPosition, newTarget, /*duration (s)*/ 0.5);
}

void Scene::OnCastRay(const Ray3d& ray, int button, int action, int mods) {
	if (!IsOpen() || !IsOctreeValid())
		return;
	const double timeClick(0.2);
	const double timeDblClick(0.3);
	const double now(glfwGetTime());

	switch (action) {
	case GLFW_PRESS: {
		// remember when the click action started
		window.selectionTimeClick = now;
		break; }
	case GLFW_RELEASE: {
		if (now-window.selectionTimeClick > timeClick) {
			// this is a long click, ignore it
			break;
		}
		if (window.selectionType != Window::SEL_NA && now-window.selectionTime < timeDblClick) {
			// this is a double click, center scene at the selected element
			if (window.selectionType == Window::SEL_CAMERA)
				window.GetCamera().SetCameraViewMode(window.selectionIdx);
			else {
				window.GetCamera().DisableCameraViewMode();
				OnCenterScene(window.selectionPoints[3]);
			}
			window.selectionTime = now;
			break;
		}
		const Window::SELECTION prevSelectionType = window.selectionType;
		window.selectionType = Window::SEL_NA;
		REAL minDist = REAL(FLT_MAX);
		IDX newSelectionIdx = NO_IDX;
		Point3f newSelectionPoints[4];
		if (window.showMesh && !octMesh.IsEmpty()) {
			// find ray intersection with the mesh
			const MVS::IntersectRayMesh intRay(octMesh, ray, scene.mesh);
			if (intRay.pick.IsValid()) {
				window.selectionType = Window::SEL_TRIANGLE;
				minDist = intRay.pick.dist;
				newSelectionIdx = intRay.pick.idx;
				const MVS::Mesh::Face& face = scene.mesh.faces[(MVS::Mesh::FIndex)newSelectionIdx];
				newSelectionPoints[0] = scene.mesh.vertices[face[0]];
				newSelectionPoints[1] = scene.mesh.vertices[face[1]];
				newSelectionPoints[2] = scene.mesh.vertices[face[2]];
				newSelectionPoints[3] = ray.GetPoint(minDist).cast<float>();
			}
		}
		if (window.showPointCloud && !octPoints.IsEmpty()) {
			// find ray intersection with the points
			const MVS::IIndex minViews(CLAMP(window.minViews, 1u, scene.images.size()));
			const MVS::IntersectRayPoints intRay(octPoints, ray, scene.pointcloud, minViews);
			if (intRay.pick.IsValid() && intRay.pick.dist < minDist) {
				window.selectionType = Window::SEL_POINT;
				minDist = intRay.pick.dist;
				newSelectionIdx = intRay.pick.idx;
				newSelectionPoints[0] = newSelectionPoints[3] = scene.pointcloud.points[newSelectionIdx];
			}
		}
		// check for camera intersection
		const TCone<REAL, 3> cone(ray, D2R(REAL(0.5)));
		const TConeIntersect<REAL, 3> coneIntersect(cone);
		FOREACH(idx, images) {
			const Image& image = images[idx];
			const MVS::Image& imageData = scene.images[image.idx];
			ASSERT(imageData.IsValid());
			REAL dist;
			if (coneIntersect.Classify(imageData.camera.C, dist) == VISIBLE && dist < minDist) {
				window.selectionType = Window::SEL_CAMERA;
				minDist = dist;
				newSelectionIdx = idx;
				newSelectionPoints[0] = newSelectionPoints[3] = imageData.camera.C;
			}
		}
		// check if we have a new selection
		if (window.selectionType != Window::SEL_NA) {
			if (window.selectionType == Window::SEL_CAMERA && (mods & GLFW_MOD_ALT)) {
				// If alt is pressed, set view camera mode
				window.selectionType = prevSelectionType; // Restore previous selection type
				window.GetCamera().SetCameraViewMode(newSelectionIdx);
			} else if (window.selectionType == Window::SEL_CAMERA && (mods & GLFW_MOD_CONTROL)) {
				// If control is pressed, select neighbor camera
				window.selectedNeighborCamera = newSelectionIdx;
			} else {
				// Normal selection
				window.selectionIdx = newSelectionIdx;
				window.selectedNeighborCamera = NO_ID;
				window.selectionPoints[0] = newSelectionPoints[0];
				window.selectionPoints[1] = newSelectionPoints[1];
				window.selectionPoints[2] = newSelectionPoints[2];
				window.selectionPoints[3] = newSelectionPoints[3];
				window.selectionTime = now;
			}
			switch (window.selectionType) {
			case Window::SEL_TRIANGLE: {
				DEBUG("Face selected:\n\tindex: %u\n\tvertex 1: %u (%g, %g, %g)\n\tvertex 2: %u (%g, %g, %g)\n\tvertex 3: %u (%g, %g, %g)",
					newSelectionIdx,
					scene.mesh.faces[newSelectionIdx][0], newSelectionPoints[0].x, newSelectionPoints[0].y, newSelectionPoints[0].z,
					scene.mesh.faces[newSelectionIdx][1], newSelectionPoints[1].x, newSelectionPoints[1].y, newSelectionPoints[1].z,
					scene.mesh.faces[newSelectionIdx][2], newSelectionPoints[2].x, newSelectionPoints[2].y, newSelectionPoints[2].z
				);
				break; }
			case Window::SEL_POINT: {
				DEBUG("Point selected:\n\tindex: %u (%g, %g, %g)%s",
					newSelectionIdx,
					newSelectionPoints[0].x, newSelectionPoints[0].y, newSelectionPoints[0].z,
					[&]() {
						if (scene.pointcloud.pointViews.empty())
							return String();
						const MVS::PointCloud::ViewArr& views = scene.pointcloud.pointViews[newSelectionIdx];
						ASSERT(!views.empty());
						String strViews(String::FormatString("\n\tviews: %u", views.size()));
						FOREACH(v, views) {
							const MVS::PointCloud::View idxImage = views[v];
							if (scene.images.empty()) {
								strViews += String::FormatString("\n\t\tview %u (no image data)", idxImage);
								continue;
							}
							const MVS::Image& imageData = scene.images[idxImage];
							const Point2 x(imageData.camera.TransformPointW2I(Cast<REAL>(window.selectionPoints[0])));
							const float conf = scene.pointcloud.pointWeights.empty() ? 0.f : scene.pointcloud.pointWeights[newSelectionIdx][v];
							strViews += String::FormatString("\n\t\t%s (%.2f %.2f pixel, %.2f conf)", Util::getFileNameExt(imageData.name).c_str(), x.x, x.y, conf);
						}
						return strViews;
					}().c_str()
				);
				break; }
			case Window::SEL_CAMERA: {
				if (!(mods & (GLFW_MOD_ALT | GLFW_MOD_CONTROL)))
					window.GetCamera().DisableCameraViewMode();
				const Image& image = images[newSelectionIdx];
				const MVS::Image& imageData = scene.images[image.idx];
				const MVS::Camera& camera = imageData.camera;
				Point3 eulerAngles;
				camera.R.GetRotationAnglesZYX(eulerAngles.x, eulerAngles.y, eulerAngles.z);
				DEBUG("Camera selected:\n\tindex: %u (ID: %u)\n\tname: %s (mask %s)\n\timage size: %ux%u"
					"\n\tintrinsics: fx %.2f, fy %.2f, cx %.2f, cy %.2f"
					"\n\tposition: %g, %g, %g\n\trotation (deg): %.2f, %.2f, %.2f"
					"\n\taverage depth: %.2g\n\tneighbors: %u",
					image.idx, imageData.ID, Util::getFileNameExt(imageData.name).c_str(),
					imageData.maskName.empty() ? "none" : Util::getFileNameExt(imageData.maskName).c_str(),
					imageData.width, imageData.height,
					camera.K(0, 0), camera.K(1, 1), camera.K(0, 2), camera.K(1, 2),
					camera.C.x, camera.C.y, camera.C.z,
					R2D(eulerAngles.x), R2D(eulerAngles.y), R2D(eulerAngles.z),
					imageData.avgDepth, imageData.neighbors.size()
				);
				break; }
			}
		}
		if (window.selectionType != Window::SEL_NA || prevSelectionType != Window::SEL_NA) {
			window.GetRenderer().UploadSelection(window);
			window.RequestRedraw();
		}
		break; }
	}
}

void Scene::OnSetCameraViewMode(MVS::IIndex camID) {
	if (!IsOpen() || camID >= images.size())
		return;

	// Save current camera state if entering camera view mode for the first time
	if (!window.GetCamera().IsCameraViewMode())
		window.GetCamera().SaveCurrentState();
	window.GetCamera().SetCurrentCamID(camID);

	// Get the Image from images and then access the MVS::Image via its index
	Image& image = images[camID];
	const MVS::Image& imageData = scene.images[image.idx];

	// Load the image if not already loaded
	if (!image.IsValid() && !image.IsImageLoading()) {
		// Load image asynchronously
		image.SetImageLoading();
		events.AddEvent(new EVTLoadImage(this, camID, IMAGE_MAX_RESOLUTION));
	}

	// Update camera with the scene data and viewport
	window.GetCamera().SetCameraFromSceneData(imageData);
}

void Scene::OnSelectPointsByCamera(bool highlightCameraVisiblePoints) {
	if (!scene.pointcloud.IsValid() || scene.images.empty())
		return;
	SelectionController& selectionController = window.GetSelectionController();
	// Prefer explicit selection of a camera, otherwise use camera-view-mode currentCamID
	MVS::IIndex camViewerIdx = NO_ID;
	if (window.selectionType == Window::SEL_CAMERA && window.selectionIdx != NO_ID)
		camViewerIdx = window.selectionIdx;
	else if (window.GetCamera().IsCameraViewMode())
		camViewerIdx = window.GetCamera().GetCurrentCamID();
	if (!highlightCameraVisiblePoints || camViewerIdx == NO_ID) {
		// Turn off: clear selection highlighting produced by this toggle
		selectionController.clearSelection();
		window.GetRenderer().UploadSelection(window);
		window.RequestRedraw();
		return;
	}
	// Highlight points visible in the current camera
	if (selectionController.getCurrentCameraIdxForHighlight() != camViewerIdx) {
		// Update current camera, recompute
		selectionController.setCurrentCameraIdxForHighlight(camViewerIdx);
		// Map viewer camera index to MVS image index
		const Image& img = images[camViewerIdx];
		// Build list of point indices visible in this image via pointViews
		MVS::PointCloud::IndexArr indices(0, 1024);
		FOREACH(p, scene.pointcloud.points) {
			const MVS::PointCloud::ViewArr& views = scene.pointcloud.pointViews[p];
			for (const auto v : views)
				if (v == img.idx) {
					indices.emplace_back(p);
					break;
				}
		}
		// Apply selection to highlight
		selectionController.setSelectedPoints(indices, scene.pointcloud.points.size());
		// Upload selection-related rendering state
		window.GetRenderer().UploadSelection(window);
		window.RequestRedraw();
	}
}
/*----------------------------------------------------------------*/

// Remove selected geometry (points and faces)
void Scene::RemoveSelectedGeometry() {
	if (!window.GetSelectionController().hasSelection())
		return;

	bool bDirtyScene = false;
	SelectionController& selectionController = window.GetSelectionController();

	// Classify geometry based on current selection
	if (!scene.pointcloud.IsEmpty()) {
		// Get selected point indices
		MVS::PointCloud::IndexArr selectedIndices = selectionController.getSelectedPointIndices();
		if (!selectedIndices.empty()) {
			// Remove selected points
			bDirtyScene = true;
			scene.pointcloud.RemovePoints(selectedIndices);
			VERBOSE("Removed %zu selected points", selectedIndices.size());
		}
	}

	if (!scene.mesh.IsEmpty()) {
		// Get selected face indices for removal
		MVS::Mesh::FaceIdxArr selectedIndices = selectionController.getSelectedFaceIndices();
		if (!selectedIndices.empty()) {
			// Remove selected faces
			bDirtyScene = true;
			scene.mesh.RemoveFaces(selectedIndices);
			VERBOSE("Removed %zu selected faces", selectedIndices.size());
		}
	}

	// If any geometry was modified, update the scene
	if (bDirtyScene)
		UpdateGeometryAfterModification();

	// Request a redraw
	window.RequestRedraw();
}

// Update geometry after modification (rebuild octrees, update rendering, etc.)
void Scene::UpdateGeometryAfterModification() {
	// Release and rebuild octrees
	octPoints.Release();
	octMesh.Release();
	if (!scene.IsEmpty())
		events.AddEvent(new EVTComputeOctree(this));

	// Update rendering data
	window.UploadRenderData();

	// Clear the selection since geometry has changed
	window.GetSelectionController().clearSelection();
}

// Set the ROI (region of interest) based on the current selection
//  - aabb: if true, use axis-aligned bounding box; if false, use oriented bounding box
void Scene::SetROIFromSelection(bool aabb) {
	if (!IsOpen())
		return;

	SelectionController& selectionController = window.GetSelectionController();
	if (!selectionController.hasSelection())
		return;

	// Collect all selected points for OBB fitting directly as Eigen vectors
	std::vector<OBB3f::POINT> selectedPoints;

	// Add selected point cloud points
	if (!scene.pointcloud.IsEmpty()) {
		MVS::PointCloud::IndexArr selectedIndices = selectionController.getSelectedPointIndices();
		selectedPoints.reserve(selectedPoints.size() + selectedIndices.size());
		for (MVS::PointCloud::Index idx : selectedIndices) {
			if (idx < scene.pointcloud.points.size()) {
				const Point3f& pt = scene.pointcloud.points[idx];
				selectedPoints.emplace_back(pt.x, pt.y, pt.z);
			}
		}
	}

	// Add vertices of selected mesh faces
	if (!scene.mesh.IsEmpty()) {
		MVS::Mesh::FaceIdxArr selectedIndices = selectionController.getSelectedFaceIndices();
		// Reserve space for up to 3 vertices per face (may have duplicates)
		selectedPoints.reserve(selectedPoints.size() + selectedIndices.size() * 3);
		for (uint32_t idx : selectedIndices) {
			if (idx < scene.mesh.faces.size()) {
				const MVS::Mesh::Face& face = scene.mesh.faces[idx];
				// Include all vertices of the selected face
				for (int j = 0; j < 3; ++j) {
					if (face[j] < scene.mesh.vertices.size()) {
						const Point3f& pt = scene.mesh.vertices[face[j]];
						selectedPoints.emplace_back(pt.x, pt.y, pt.z);
					}
				}
			}
		}
	}
	// Check if we found any selected geometry
	if (selectedPoints.empty())
		return;

	// If AABB is requested, compute it directly
	if (aabb) {
		// Compute the axis-aligned bounding box from selected points
		AABB3f aabbBounds;
		aabbBounds.Set(selectedPoints.data(), selectedPoints.size());
		// Set the OBB to the computed AABB
		scene.obb.Set(aabbBounds);
	} else {
		// If OBB is requested,
		// Use OBB3f's built-in fitting functionality to compute the optimal oriented bounding box
		scene.obb.Set(selectedPoints.data(), selectedPoints.size(), 32);
	}
	// Add a small margin by enlarging the OBB
	const float margin = scene.obb.GetSize().maxCoeff() * 0.03f; // 3% margin
	scene.obb.Enlarge(margin);

	// Update bounds rendering data
	window.GetRenderer().UploadBounds(scene);

	// Request a redraw
	window.RequestRedraw();
}

// Crop scene to only images that see at least minPoints of the selected points
MVS::Scene Scene::CropToPoints(const MVS::PointCloud::IndexArr& selectedPointIndices, unsigned minPoints) const {
	if (!scene.IsValid() || !scene.pointcloud.IsValid())
		return MVS::Scene(); // Return empty scene

	// Count how many selected points each image sees
	std::unordered_map<MVS::IIndex, unsigned> imageCounts;
	for (MVS::PointCloud::Index pointIdx : selectedPointIndices) {
		const MVS::PointCloud::ViewArr& views = scene.pointcloud.pointViews[pointIdx];
		for (MVS::PointCloud::View imageIdx : views)
			imageCounts[imageIdx]++;
	}

	// Select images that see at least minPoints selected points
	MVS::IIndexArr selectedImageIndices;
	for (const auto& pair : imageCounts)
		if (pair.second >= minPoints)
			selectedImageIndices.emplace_back(pair.first);

	// Create sub-scene with selected images
	if (selectedImageIndices.size() < 2) {
		DEBUG("error: no images see %u or more points from %u selected", minPoints, scene.pointcloud.GetSize());
		return MVS::Scene(); // Return empty scene
	}
	if (selectedImageIndices.size() == scene.images.size()) {
		VERBOSE("Cropping scene: all %u images see at least %u points from %u selected; nothing to do", 
			selectedImageIndices.size(), minPoints, scene.pointcloud.GetSize());
		return MVS::Scene(); // If all images are selected, return empty scene
	}
	VERBOSE("Cropping scene: found %u images that see at least %u points from %u selected", 
		selectedImageIndices.size(), minPoints, scene.pointcloud.GetSize());
	return scene.SubScene(selectedImageIndices);
}
/*----------------------------------------------------------------*/
