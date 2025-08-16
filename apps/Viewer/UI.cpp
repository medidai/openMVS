/*
 * UI.cpp
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
#include "UI.h"
#include "Scene.h"
#include "Camera.h"
#include "Window.h"
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <portable-file-dialogs.h>
#include <unordered_map>

using namespace VIEWER;

constexpr float PAD = 10.f;

UI::UI()
	: showSceneInfo(false)
	, showCameraControls(false)
	, showSelectionControls(false)
	, showRenderSettings(false)
	, showPerformanceOverlay(true)
	, showViewportOverlay(true)
	, showSelectionOverlay(true)
	, showAboutDialog(false)
	, showHelpDialog(false)
	, showExportDialog(false)
	, showCameraInfoDialog(false)
	, showSelectionDialog(false)
	, showMainMenu(false)
	, menuWasVisible(false)
	, menuTriggerHeight(50.f)
	, lastMenuInteraction(0.0)
	, menuFadeOutDelay(2.f)
	, deltaTime(0.0)
	, frameCount(0)
	, fps(0.f)
{
}

UI::~UI() {
	Release();
}

bool UI::Initialize(Window& window, const String& glslVersion) {
	// Setup Dear ImGui context
	#ifndef _RELEASE
	IMGUI_CHECKVERSION();
	#endif
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	iniPath = Util::getApplicationFolder() + "Viewer.ini";
	io.IniFilename = iniPath.c_str();

	// Try to enable docking (available in ImGui 1.80+)
	#if defined(ImGuiConfigFlags_DockingEnable)
	try {
		// Check if the flag exists by testing if it's defined
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
		VERBOSE("Docking enabled");
	} catch (...) {
		VERBOSE("Docking feature not available");
	}
	#endif

	// Try to enable multi-viewport (available in ImGui 1.80+)
	#if defined(ImGuiConfigFlags_ViewportsEnable)
	try {
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
		VERBOSE("Multi-viewport enabled");
	} catch (...) {
		VERBOSE("Multi-viewport feature not available");
	}
	#endif

	// Setup Dear ImGui style
	SetupStyle();
	// Setup custom settings handler
	SetupCustomSettings(window);

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window.GetGLFWWindow(), true);
	ImGui_ImplOpenGL3_Init(glslVersion);

	return true;
}

void UI::Release() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void UI::NewFrame(Window& window) {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	// Handle global keyboard shortcuts
	HandleGlobalKeys(window);

	// Update menu visibility based on mouse position and usage
	UpdateMenuVisibility();
}

void UI::Render() {
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	// Update and render additional Platform Windows (if multi-viewport is enabled)
	#ifdef IMGUI_HAS_VIEWPORT
	ImGuiIO& io = ImGui::GetIO();
	if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
		GLFWwindow* backup_current_context = glfwGetCurrentContext();
		ImGui::UpdatePlatformWindows();
		ImGui::RenderPlatformWindowsDefault();
		glfwMakeContextCurrent(backup_current_context);
	}
	#endif
}

void UI::ShowMainMenuBar(Window& window) {
	Scene& scene = window.GetScene();

	// Handle dialogs even when menu is hidden
	if (showAboutDialog)
		ShowAboutDialog();
	if (showHelpDialog)
		ShowHelpDialog();
	if (showExportDialog)
		ShowExportDialog(scene);
	if (showCameraInfoDialog)
		ShowCameraInfoDialog(window);
	if (showSelectionDialog)
		ShowSelectionDialog(window);

	// Only show menu bar if it should be visible
	if (!showMainMenu) {
		return;
	}

	if (ImGui::BeginMainMenuBar()) {
		// Update last interaction time when menu bar is actively being used
		if (ImGui::IsWindowHovered() || ImGui::IsAnyItemActive() || ImGui::IsAnyItemFocused())
			lastMenuInteraction = glfwGetTime();

		if (ImGui::BeginMenu("File")) {
			lastMenuInteraction = glfwGetTime(); // Update interaction time when menu is open
			#ifdef __APPLE__
			if (ImGui::MenuItem("Open Scene...", "Cmd+O")) {
			#else
			if (ImGui::MenuItem("Open Scene...", "Ctrl+O")) {
			#endif
				// Open file dialog and load scene if file selected
				window.SetVisible(false);
				String filename, geometryFilename;
				if (ShowOpenFileDialog(filename, geometryFilename))
					scene.Open(filename, geometryFilename);
				window.SetVisible(true);
			}
			#ifdef __APPLE__
			if (ImGui::MenuItem("Save Scene", "Cmd+S", false, scene.IsOpen())) {
			#else
			if (ImGui::MenuItem("Save Scene", "Ctrl+S", false, scene.IsOpen())) {
			#endif
				// Save scene to current file
				scene.Save();
			}
			#ifdef __APPLE__
			if (ImGui::MenuItem("Save Scene As...", "Cmd+Shift+S", false, scene.IsOpen())) {
			#else
			if (ImGui::MenuItem("Save Scene As...", "Ctrl+Shift+S", false, scene.IsOpen())) {
			#endif
				// Always prompt for save location
				window.SetVisible(false);
				String filename;
				if (ShowSaveFileDialog(filename))
					scene.Save(filename);
				window.SetVisible(true);
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Export...", nullptr, false, scene.IsOpen())) {
				// Show export dialog with export format options
				showExportDialog = true;
			}
			ImGui::Separator();
			#ifdef __APPLE__
			if (ImGui::MenuItem("Exit", "Cmd+Q")) {
			#else
			if (ImGui::MenuItem("Exit", "Alt+F4")) {
			#endif
				glfwSetWindowShouldClose(window.GetGLFWWindow(), GLFW_TRUE);
			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("View")) {
			lastMenuInteraction = glfwGetTime(); // Update interaction time when menu is open
			ImGui::MenuItem("Scene Info", nullptr, &showSceneInfo);
			ImGui::MenuItem("Camera Info", nullptr, &showCameraInfoDialog);
			ImGui::MenuItem("Camera Controls", nullptr, &showCameraControls);
			ImGui::MenuItem("Selection Controls", nullptr, &showSelectionControls);
			ImGui::MenuItem("Render Settings", nullptr, &showRenderSettings);
			ImGui::MenuItem("Performance Overlay", nullptr, &showPerformanceOverlay);
			ImGui::MenuItem("Viewport Overlay", nullptr, &showViewportOverlay);
			ImGui::MenuItem("Selection Overlay", nullptr, &showSelectionOverlay);
			ImGui::Separator();
			ImGui::MenuItem("Show Point Cloud", "P", &window.showPointCloud);
			ImGui::MenuItem("Show Mesh", "M", &window.showMesh);
			ImGui::MenuItem("Show Cameras", "C", &window.showCameras);
			if (window.showMesh) {
				ImGui::MenuItem("Wireframe", "W", &window.showMeshWireframe);
				ImGui::MenuItem("Textured", "T", &window.showMeshTextured);
			}
			ImGui::Separator();
			if (ImGui::MenuItem("Reset Camera", "R"))
				window.ResetView();
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu("Help")) {
			lastMenuInteraction = glfwGetTime(); // Update interaction time when menu is open
			if (ImGui::MenuItem("Help", "F1"))
				showHelpDialog = true;
			ImGui::Separator();
			if (ImGui::MenuItem("About"))
				showAboutDialog = true;
			ImGui::EndMenu();
		}

		ImGui::EndMainMenuBar();
	}
}

void UI::ShowSceneInfo(const Window& window) {
	if (!showSceneInfo) return;
	const MVS::Scene& scene = window.GetScene().GetScene();

	ImGui::SetNextWindowPos(ImVec2(10, 110), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(240, 410), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Scene Info", &showSceneInfo)) {
		ImGui::Text("Scene Statistics");
		ImGui::Separator();
		ImGui::Text("Images: %u valid (%u total)", scene.nCalibratedImages, scene.images.size());
		ImGui::Text("Platforms: %u", scene.platforms.size());
		ImGui::Text("OBB: %s", scene.obb.IsValid() ? "valid" : "NA");
		// Show full obb
		if (scene.obb.IsValid() && ImGui::CollapsingHeader("Oriented Bounding-Box")) {
			ImGui::Text("  rot1: [%.6f  %.6f  %.6f]", scene.obb.m_rot(0, 0), scene.obb.m_rot(0, 1), scene.obb.m_rot(0, 2));
			ImGui::Text("  rot2: [%.6f  %.6f  %.6f]", scene.obb.m_rot(1, 0), scene.obb.m_rot(1, 1), scene.obb.m_rot(1, 2));
			ImGui::Text("  rot3: [%.6f  %.6f  %.6f]", scene.obb.m_rot(2, 0), scene.obb.m_rot(2, 1), scene.obb.m_rot(2, 2));
			ImGui::Text("  pos : [%.6f  %.6f  %.6f]", scene.obb.m_pos.x(), scene.obb.m_pos.y(), scene.obb.m_pos.z());
			ImGui::Text("  ext : [%.6f  %.6f  %.6f]", scene.obb.m_ext.x(), scene.obb.m_ext.y(), scene.obb.m_ext.z());
		}
		ImGui::Text("Transform: %s", scene.HasTransform() ? "valid" : "NA");
		// Show full transform
		if (scene.HasTransform() && ImGui::CollapsingHeader("Transform")) {
			ImGui::Text("  [%.6f  %.6f  %.6f  %.6f]", scene.transform(0, 0), scene.transform(0, 1), scene.transform(0, 2), scene.transform(0, 3));
			ImGui::Text("  [%.6f  %.6f  %.6f  %.6f]", scene.transform(1, 0), scene.transform(1, 1), scene.transform(1, 2), scene.transform(1, 3));
			ImGui::Text("  [%.6f  %.6f  %.6f  %.6f]", scene.transform(2, 0), scene.transform(2, 1), scene.transform(2, 2), scene.transform(2, 3));
			ImGui::Text("  [%.6f  %.6f  %.6f  %.6f]", scene.transform(3, 0), scene.transform(3, 1), scene.transform(3, 2), scene.transform(3, 3));
		}

		if (!scene.pointcloud.IsEmpty()) {
			ImGui::Separator();
			ImGui::Text("Point Cloud Statistics");
			ImGui::Separator();
			ImGui::Text("Points: %zu", scene.pointcloud.points.size());
			ImGui::Text("Point Views: %zu", scene.pointcloud.pointViews.size());
			ImGui::Text("Point Weights: %zu", scene.pointcloud.pointWeights.size());
			ImGui::Text("Colors: %zu", scene.pointcloud.colors.size());
			ImGui::Text("Normals: %zu", scene.pointcloud.normals.size());
			AABB3f bounds = scene.pointcloud.GetAABB();
			ImGui::Text("Bounds:");
			ImGui::Text("  Min: (%.3f, %.3f, %.3f)", bounds.ptMin.x(), bounds.ptMin.y(), bounds.ptMin.z());
			ImGui::Text("  Max: (%.3f, %.3f, %.3f)", bounds.ptMax.x(), bounds.ptMax.y(), bounds.ptMax.z());
			Point3f size = bounds.GetSize();
			ImGui::Text("  Size: (%.3f, %.3f, %.3f)", size.x, size.y, size.z);
		}

		if (!scene.mesh.IsEmpty()) {
			ImGui::Separator();
			ImGui::Text("Mesh Statistics");
			ImGui::Separator();
			ImGui::Text("Vertices: %u", scene.mesh.vertices.size());
			ImGui::Text("Faces: %u", scene.mesh.faces.size());
			ImGui::Text("Textures: %u", scene.mesh.texturesDiffuse.size());
			// Show mesh bounds if available
			AABB3f meshBounds = scene.mesh.GetAABB();
			ImGui::Text("Mesh Bounds:");
			ImGui::Text("  Min: (%.3f, %.3f, %.3f)", meshBounds.ptMin.x(), meshBounds.ptMin.y(), meshBounds.ptMin.z());
			ImGui::Text("  Max: (%.3f, %.3f, %.3f)", meshBounds.ptMax.x(), meshBounds.ptMax.y(), meshBounds.ptMax.z());
			Point3f meshSize = meshBounds.GetSize();
			ImGui::Text("  Size: (%.3f, %.3f, %.3f)", meshSize.x, meshSize.y, meshSize.z);
		}

		// Estimate SfM normals and mesh patches if valid SfM scene
		ImGui::Separator();
		if (ImGui::Checkbox("Estimate SfM Normals", &window.GetScene().estimateSfMNormals))
			window.RequestRedraw();
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Toggle SfM normals estimation; need to reopen the scene");
		if (ImGui::Checkbox("Estimate SfM Patches", &window.GetScene().estimateSfMPatches))
			window.RequestRedraw();
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Toggle SfM patches estimation; need to reopen the scene");
	}
	ImGui::End();
}

void UI::ShowCameraControls(Window& window) {
	if (!showCameraControls) return;

	ImGui::SetNextWindowPos(ImVec2(1044, 100), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(224, 296), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Camera Controls", &showCameraControls)) {
		Camera& camera = window.GetCamera();

		// Navigation mode
		const char* navModes[] = { "Arcball", "First Person", "Selection" };
		int currentMode = (int)window.GetControlMode();
		if (ImGui::Combo("Navigation Mode", &currentMode, navModes, IM_ARRAYSIZE(navModes))) {
			window.SetControlMode((Window::ControlMode)currentMode);
			// Auto-open selection controls when switching to selection mode
			if (currentMode == Window::CONTROL_SELECTION)
				showSelectionControls = true;
		}

		// Projection mode
		bool ortho = camera.IsOrthographic();
		if (ImGui::Checkbox("Orthographic", &ortho))
			camera.SetOrthographic(ortho);
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Toggle orthographic/perspective projection mode");

		// FOV slider
		float fov = (float)camera.GetFOV();
		if (ImGui::SliderFloat("FOV", &fov, 1.f, 179.f, "%.1f°"))
			camera.SetFOV(fov);
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Field of View (FOV) angle");

		// Camera rendering checkbox
		if (ImGui::Checkbox("Show Cameras", &window.showCameras))
			window.RequestRedraw();
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Toggle camera frustum display (C key)");

		// Arcball sensitivity controls (only show when in arcball mode)
		if (window.GetControlMode() == Window::CONTROL_ARCBALL) {
			ImGui::Separator();
			ImGui::Text("Arcball Sensitivity");
			ArcballControls& arcballControls = window.GetArcballControls();
			// General Sensitivity input
			float sensitivity = (float)arcballControls.getSensitivity();
			if (ImGui::InputFloat("Sensitivity", &sensitivity, 0.1f, 1.f, "%.3f"))
				arcballControls.setSensitivity(MAXF(0.001f, sensitivity));
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Overall sensitivity multiplier");
			// Rotation Sensitivity slider
			float rotationSensitivity = (float)arcballControls.getRotationSensitivity();
			if (ImGui::SliderFloat("Rotation", &rotationSensitivity, 0.1f, 5.f, "%.2f"))
				arcballControls.setRotationSensitivity(rotationSensitivity);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Rotation sensitivity");
			// Zoom Sensitivity slider
			float zoomSensitivity = (float)arcballControls.getZoomSensitivity();
			if (ImGui::SliderFloat("Zoom", &zoomSensitivity, 0.1f, 10.f, "%.2f"))
				arcballControls.setZoomSensitivity(zoomSensitivity);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Zoom/scroll sensitivity");
			// Pan Sensitivity slider
			float panSensitivity = (float)arcballControls.getPanSensitivity();
			if (ImGui::SliderFloat("Pan", &panSensitivity, 0.1f, 5.f, "%.2f"))
				arcballControls.setPanSensitivity(panSensitivity);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Pan/translate sensitivity");
		}

		// First person sensitivity controls (only show when in first person mode)
		if (window.GetControlMode() == Window::CONTROL_FIRST_PERSON) {
			ImGui::Separator();
			ImGui::Text("First Person Sensitivity");
			FirstPersonControls& firstPersonControls = window.GetFirstPersonControls();
			// Movement Speed input
			float movementSpeed = (float)firstPersonControls.getMovementSpeed();
			if (ImGui::InputFloat("Speed", &movementSpeed, 0.1f, 1.f, "%.3f"))
				firstPersonControls.setMovementSpeed(MAXF(0.001f, movementSpeed));
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Movement speed multiplier");
			// Rotation Sensitivity slider
			float mouseSensitivity = (float)firstPersonControls.getMouseSensitivity();
			if (ImGui::SliderFloat("Sensitivity", &mouseSensitivity, 0.1f, 5.f, "%.2f"))
				firstPersonControls.setMouseSensitivity(mouseSensitivity);
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Mouse sensitivity");
		}

		// Camera view mode info
		if (camera.IsCameraViewMode()) {
			ImGui::Separator();
			ImGui::Text("Camera View Mode");
			ImGui::Text("Current Camera: %d", (int)camera.GetCurrentCamID());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Use Left/Right arrows to switch cameras");
			// Show restore button if we have a saved state
			ImGui::SameLine();
			if (ImGui::SmallButton("Exit"))
				camera.DisableCameraViewMode();
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Exit camera view mode and restore previous position");
		} else {
			// Show save current state button when not in camera view mode
			ImGui::Separator();
			ImGui::Text("Camera State:");
			ImGui::SameLine();
			if (ImGui::SmallButton("Save"))
				camera.SaveCurrentState();
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Save current camera position and view direction");
			// Show status if state is saved
			if (camera.HasSavedState()) {
				ImGui::SameLine();
				if (ImGui::SmallButton("Restore"))
					camera.RestoreSavedState();
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Restore previous camera position and view direction");
			}
		}

		// Camera info
		ImGui::Separator();
		Eigen::Vector3d pos = camera.GetPosition();
		ImGui::Text("Position: %.4g, %.4g, %.4g", pos.x(), pos.y(), pos.z());
		Eigen::Vector3d target = camera.GetTarget();
		ImGui::Text("Target: %.4g, %.4g, %.4g", target.x(), target.y(), target.z());

		// Highlight points visible by the current/selected camera
		ImGui::Separator();
		bool highlightCameraVisiblePoints(window.GetSelectionController().getCurrentCameraIdxForHighlight() != NO_ID);
		if (ImGui::Checkbox("Highlight points seen by camera", &highlightCameraVisiblePoints))
			window.GetScene().OnSelectPointsByCamera(highlightCameraVisiblePoints);
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Select and highlight all points observed by the active camera");
		// Keep highlight in sync if camera selection changes while toggle is on
		if (highlightCameraVisiblePoints)
			window.GetScene().OnSelectPointsByCamera(true);

		// Reset button
		ImGui::Separator();
		if (ImGui::Button("Reset Camera"))
			window.ResetView();
	}
	ImGui::End();
}

void UI::ShowSelectionControls(Window& window) {
	// Auto-open selection controls when in selection mode
	showSelectionControls = (window.GetControlMode() == Window::CONTROL_SELECTION);
	if (!showSelectionControls) return;

	ImGui::SetNextWindowPos(ImVec2(990, 210), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(280, 320), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Selection Controls", &showSelectionControls)) {
		// Only show controls if we have a selection controller
		if (window.GetControlMode() != Window::CONTROL_SELECTION) {
			ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.f), "Selection mode not active");
			ImGui::Text("Switch to Selection mode in Camera Controls");
			ImGui::Text("or press Tab to cycle navigation modes");
			ImGui::End();
			return;
		}

		SelectionController& selectionController = window.GetSelectionController();

		// Selection tool selection
		ImGui::Text("Selection Tools");
		ImGui::Separator();
		const char* selectionModes[] = { "Box", "Lasso", "Circle" };
		int selMode = (int)selectionController.getSelectionMode();
		if (ImGui::Combo("Tool", &selMode, selectionModes, IM_ARRAYSIZE(selectionModes)))
			selectionController.setSelectionMode((SelectionController::SelectionMode)selMode);

		// Quick tool shortcuts
		ImGui::Text("Shortcuts: B = Box, L = Lasso, C = Circle");

		// Selection statistics
		ImGui::Separator();
		ImGui::Text("Selection Statistics");
		if (selectionController.hasSelection()) {
			ImGui::Text("Selected: %zu points, %zu faces", 
				selectionController.getSelectedPointCount(),
				selectionController.getSelectedFaceCount());
		} else {
			ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.f), "No selection");
		}

		// Selection operations
		ImGui::Separator();
		ImGui::Text("Selection Operations");
		if (ImGui::Button("Clear Selection", ImVec2(-1, 0)))
			selectionController.clearSelection();

		if (selectionController.hasSelection()) {
			if (ImGui::Button("Invert Selection", ImVec2(-1, 0)))
				selectionController.invertSelection();

			// Geometry operations
			ImGui::Separator();
			ImGui::Text("Geometry Operations");
			if (ImGui::Button("Remove Selected", ImVec2(-1, 0)))
				ImGui::OpenPopup("Confirm Remove Selected");

			// ROI selection
			bool aabb = selectionController.isROIfromSelectionMode();
			if (ImGui::Checkbox("AABBox", &aabb))
				selectionController.setROIfromSelectionMode(aabb);
			ImGui::SameLine();
			if (ImGui::Button("Set ROI to Selection", ImVec2(-1, 0)))
				selectionController.runROICallback();

			static int minPoints = 150;
			if (selectionController.getSelectedPointCount() >= 3) {
				// Add parameter input for minimum views
				ImGui::InputInt("Min Points", &minPoints, 1, 10);
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Minimum number of selected points an image must see to be included");

				if (ImGui::Button("Crop Scene to Selection", ImVec2(-1, 0)))
					ImGui::OpenPopup("Crop Scene to Selection");
				if (ImGui::IsItemHovered())
					ImGui::SetTooltip("Create a new scene containing only images that see the selected points");
			}

			// Crop Scene to Selection popup
			if (ImGui::BeginPopupModal("Crop Scene to Selection", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
				ImGui::Text("Create a new scene with images that see");
				ImGui::Text("at least %d selected points?", minPoints);
				ImGui::Separator();
				if (ImGui::Button("Crop Scene", ImVec2(120, 0))) {
					// Use the new Scene::CropToPoints function
					MVS::PointCloud::IndexArr selectedPointIndices = selectionController.getSelectedPointIndices();
					// Call the new CropToPoints function
					const Scene& scene = window.GetScene();
					MVS::Scene croppedScene = scene.CropToPoints(selectedPointIndices, static_cast<unsigned>(minPoints));
					// Check if we got a valid cropped scene
					if (!croppedScene.IsEmpty()) {
						// Show save dialog and export the new scene
						window.SetVisible(false);
						String filename;
						if (ShowSaveFileDialog(filename)) {
							// Ensure .mvs extension
							if (Util::getFileExt(filename).empty())
								filename += ".mvs";
							// Save the cropped scene directly
							if (!croppedScene.Save(filename, scene.nArchiveType))
								DEBUG("error: failed to save cropped scene to '%s'", filename.c_str());
						}
						window.SetVisible(true);
						ImGui::CloseCurrentPopup();
					} else {
						ImGui::TextColored(ImVec4(1.f, 0.6f, 0.6f, 1.f), 
							"No images see %d or more selected points!", minPoints);
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0)))
					ImGui::CloseCurrentPopup();
				ImGui::EndPopup();
			}

			// Confirmation popups
			if (ImGui::BeginPopupModal("Confirm Remove Selected", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
				ImGui::Text("Remove %zu selected points/faces?", 
					selectionController.getSelectedPointCount() + selectionController.getSelectedFaceCount());
				ImGui::TextColored(ImVec4(1.f, 0.6f, 0.6f, 1.f), "This operation cannot be undone!");
				ImGui::Separator();
				if (ImGui::Button("Remove", ImVec2(120, 0))) {
					selectionController.runDeleteCallback();
					ImGui::CloseCurrentPopup();
				}
				ImGui::SameLine();
				if (ImGui::Button("Cancel", ImVec2(120, 0))) {
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndPopup();
			}
		}

		// Controls help
		ImGui::Separator();
		ImGui::Text("Controls");
		ImGui::Text("• G: Exit selection mode");
		ImGui::Text("• B/L/C: Switch selection tools");
		ImGui::Text("• Drag to select geometry");
		ImGui::Text("• Hold Shift: Add to selection");
		ImGui::Text("• Hold Ctrl: Remove from selection");
		ImGui::Text("• I: Invert selection");
		ImGui::Text("• R: Reset selection");
		ImGui::Text("• O: Set ROI from selection");
		ImGui::Text("• Delete: Delete selected elements");
	}
	ImGui::End();
}

void UI::ShowRenderSettings(Window& window) {
	if (!showRenderSettings) return;

	ImGui::SetNextWindowPos(ImVec2(10, 120), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(270, 320), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Render Settings", &showRenderSettings)) {
		ShowRenderingControls(window);
		ShowPointCloudControls(window);
		ShowMeshControls(window);
	}
	ImGui::End();
}

void UI::ShowPerformanceOverlay(Window& window) {
	if (!showPerformanceOverlay) return;

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | 
								   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | 
								   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

	const ImGuiViewport* viewport = ImGui::GetMainViewport();
	ImVec2 work_pos = viewport->WorkPos;
	ImVec2 work_size = viewport->WorkSize;
	ImVec2 window_pos, window_pos_pivot;

	window_pos.x = work_pos.x + work_size.x - PAD;
	window_pos.y = work_pos.y + PAD;
	window_pos_pivot.x = 1.f;
	window_pos_pivot.y = 0.f;

	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.35f);

	if (ImGui::Begin("Performance", &showPerformanceOverlay, window_flags)) {
		if (window.renderOnlyOnChange) {
			ImGui::Text("Frame Time: %.3f ms", deltaTime);
		} else {
			ImGui::Text("FPS: %.1f", fps);
			ImGui::Text("Frame Time: %.3f ms", 1000.f / fps);
		}
		ImGui::Separator();
		if (ImGui::IsMousePosValid())
			ImGui::Text("Mouse: %.f, %.f", ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
		else
			ImGui::Text("Mouse: <invalid>");
		if (window.GetControlMode() == Window::CONTROL_ARCBALL) {
			const auto& target = window.GetCamera().GetTarget();
			ImGui::Text("Target: %.4g, %.4g, %.4g", target.x(), target.y(), target.z());
		}
	}
	ImGui::End();
}

void UI::ShowViewportOverlay(const Window& window) {
	if (!showViewportOverlay) return;

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | 
								   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | 
								   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

	const ImGuiViewport* vp = ImGui::GetMainViewport();
	ImVec2 work_pos = vp->WorkPos;
	ImVec2 window_pos;

	window_pos.x = work_pos.x + PAD;
	window_pos.y = work_pos.y + PAD;

	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.35f);

	if (ImGui::Begin("Viewport Info", &showViewportOverlay, window_flags)) {
		const Camera& camera = window.GetCamera();
		ImGui::Text("Viewport: %dx%d", camera.GetSize().width, camera.GetSize().height);
		ImGui::Text("FOV: %.1f°", camera.GetFOV());
		ImGui::Text("Mode: %s", camera.IsOrthographic() ? "Orthographic" : "Perspective");
		// Navigation mode display
		const char* modeText(window.GetControlMode() == Window::CONTROL_ARCBALL ? "Arcball" : window.GetControlMode() == Window::CONTROL_FIRST_PERSON ? "First Person" : "Selection");
		ImGui::Text("Navigation: %s", modeText);
	}
	ImGui::End();
}

void UI::ShowAboutDialog() {
	if (!showAboutDialog)
		return;

	ImGui::OpenPopup("About");
	if (ImGui::BeginPopupModal("About", &showAboutDialog, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("OpenMVS Viewer " OpenMVS_VERSION);
		ImGui::Text("Author: SEACAVE");
		ImGui::Separator();
		ImGui::Text("Built with ImGui %s and", ImGui::GetVersion());
		ImGui::Text("OpenGL %s", glGetString(GL_VERSION));
		ImGui::Separator();
		if (ImGui::Button("Close")) {
			showAboutDialog = false;
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
}

void UI::ShowHelpDialog() {
	if (!showHelpDialog)
		return;

	ImGui::OpenPopup("Help");
	if (ImGui::BeginPopupModal("Help", &showHelpDialog, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("OpenMVS Viewer - Help & Controls");
		ImGui::Separator();

		// Detect macOS for platform-specific shortcuts
		#ifdef __APPLE__
		const bool isMacOS = true;
		const char* cmdKey = "Cmd";
		#else
		const bool isMacOS = false;
		const char* cmdKey = "Ctrl";
		#endif

		// File Operations
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "File Operations:");
		if (isMacOS) {
			ImGui::Text("  Cmd+O         Open Scene");
			ImGui::Text("  Cmd+S         Save Scene");
			ImGui::Text("  Cmd+Shift+S   Save Scene As");
			ImGui::Text("  Cmd+Q         Exit");
		} else {
			ImGui::Text("  Ctrl+O        Open Scene");
			ImGui::Text("  Ctrl+S        Save Scene");
			ImGui::Text("  Ctrl+Shift+S  Save Scene As");
			ImGui::Text("  Alt+F4        Exit");
		}
		ImGui::Separator();

		// Camera Navigation
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Camera Navigation:");
		ImGui::Text("  Tab           Switch navigation mode (Arcball/First Person)");
		ImGui::Text("  R             Reset camera");
		ImGui::Text("  F1            Show this help");
		ImGui::Text("  F11           Toggle fullscreen");
		ImGui::Separator();

		// Display Controls
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Display Controls:");
		ImGui::Text("  P             Toggle point cloud display");
		ImGui::Text("  M             Toggle mesh display");
		ImGui::Text("  C             Toggle camera frustum display");
		ImGui::Text("  W             Toggle wireframe mesh rendering");
		ImGui::Text("  T             Toggle textured mesh rendering");
		ImGui::Separator();

		// Arcball Controls
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Arcball Mode:");
		if (isMacOS) {
			ImGui::Text("  Left click + drag   Rotate camera around target");
			ImGui::Text("  Right click + drag  Pan camera");
			ImGui::Text("  Two-finger drag     Pan camera (trackpad)");
			ImGui::Text("  Scroll/pinch        Zoom in/out");
			ImGui::Text("  Double-click        Focus on clicked point");
		} else {
			ImGui::Text("  Left click + drag   Rotate camera around target");
			ImGui::Text("  Right click + drag  Pan camera");
			ImGui::Text("  Middle click + drag Pan camera");
			ImGui::Text("  Scroll wheel        Zoom in/out");
			ImGui::Text("  Double-click        Focus on clicked point");
		}
		ImGui::Separator();

		// First Person Controls
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "First Person Mode:");
		ImGui::Text("  Mouse movement      Look around");
		ImGui::Text("  W, A, S, D          Move forward/left/backward/right");
		ImGui::Text("  Q, E                Move down/up");
		ImGui::Text("  Scroll wheel        Adjust movement speed");
		if (isMacOS) {
			ImGui::Text("  Shift (hold)        Move faster");
			ImGui::Text("  Cmd (hold)          Move slower");
		} else {
			ImGui::Text("  Shift (hold)        Move faster");
			ImGui::Text("  Ctrl (hold)         Move slower");
		}
		ImGui::Separator();

		// Camera View Mode
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Camera View Mode:");
		ImGui::Text("  Left/Right arrows   Switch between cameras");
		ImGui::Text("  Escape              Exit camera view mode");
		ImGui::Text("  Any camera movement Exit camera view mode");
		ImGui::Separator();

		// Selection & Interaction
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Selection & Interaction:");
		ImGui::Text("  Single click        Select point/face/camera");
		ImGui::Text("  Double-click        Focus on selection");
		ImGui::Text("                      (or enter camera view for cameras)");
		ImGui::Separator();

		// Selection Tools
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Selection Tools:");
		ImGui::Text("  B                   Box selection mode");
		ImGui::Text("  L                   Lasso selection mode");
		ImGui::Text("  C                   Circle selection mode");
		ImGui::Text("  Left click + drag   Create selection area");
		if (isMacOS) {
			ImGui::Text("  Shift + drag        Add to selection");
			ImGui::Text("  Cmd + drag          Subtract from selection");
		} else {
			ImGui::Text("  Shift + drag        Add to selection");
			ImGui::Text("  Ctrl + drag         Subtract from selection");
		}
		ImGui::Text("  I                   Invert selection");
		ImGui::Text("  O                   Set ROI from selection");
		ImGui::Text("  Delete              Delete selected elements");
		ImGui::Text("  Escape              Clear selection");
		ImGui::Separator();

		// UI Controls
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "UI Controls:");
		ImGui::Text("  Mouse at top        Show/hide menu bar");
		ImGui::Text("  Escape              Close dialogs/windows");
		ImGui::Text("                      Clear focus/hide menu");
		ImGui::Separator();

		// File Formats
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Supported Formats:");
		ImGui::Text("  Scene files:        .mvs, .dmap, .ply");
		ImGui::Text("  Geometry files:     .ply, .obj");
		ImGui::Text("  Export formats:     .ply, .obj");
		ImGui::Separator();

		// Tips
		ImGui::TextColored(ImVec4(1.f, 0.9f, 0.6f, 1.f), "Tips:");
		ImGui::Text("  • Use the View menu to toggle overlays and panels");
		ImGui::Text("  • Selection info appears in bottom-left corner");
		ImGui::Text("  • Viewport info appears in top-left corner");
		ImGui::Text("  • Performance stats appear in top-right corner");
		ImGui::Text("  • Double-click selections to focus/navigate to them");
		ImGui::Text("  • Selection tools work on both point clouds and meshes");
		ImGui::Text("  • Use modifier keys to combine multiple selections");
		if (isMacOS) {
			ImGui::Text("  • Use trackpad gestures for smooth navigation");
			ImGui::Text("  • Three-finger drag works as middle-click");
		}

		ImGui::Separator();
		if (ImGui::Button("Close", ImVec2(120, 0))) {
			showHelpDialog = false;
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}
}

void UI::ShowExportDialog(Scene& scene) {
	if (!showExportDialog) return;

	// Show as a regular window instead of modal popup
	ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Export Scene", &showExportDialog, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("Export scene geometry to various formats");
		ImGui::Separator();

		static int exportFormat = 0;
		const char* formatOptions[] = { "PLY Point Cloud", "PLY Mesh", "OBJ Mesh", "GLTF Mesh" };
		ImGui::Combo("Export Format", &exportFormat, formatOptions, IM_ARRAYSIZE(formatOptions));

		ImGui::Separator();

		// Show what will be exported based on format and scene content
		const MVS::Scene& mvs_scene = scene.GetScene();
		bool hasPointCloud = !mvs_scene.pointcloud.IsEmpty();
		bool hasMesh = !mvs_scene.mesh.IsEmpty();

		switch (exportFormat) {
		case 0: // PLY Point Cloud
			if (hasPointCloud) {
				ImGui::Text("✓ Point cloud: %zu points", mvs_scene.pointcloud.points.size());
				if (!mvs_scene.pointcloud.colors.empty())
					ImGui::Text("✓ Point colors available");
				if (!mvs_scene.pointcloud.normals.empty())
					ImGui::Text("✓ Point normals available");
			} else {
				ImGui::TextColored(ImVec4(1.f, 0.6f, 0.6f, 1.f), "⚠ No point cloud data to export");
			}
			break;
		case 1: // PLY Mesh
		case 2: // OBJ Mesh
		case 3: // GLTF Mesh
			if (hasMesh) {
				ImGui::Text("✓ Mesh: %u vertices, %u faces", mvs_scene.mesh.vertices.size(), mvs_scene.mesh.faces.size());
				if (!mvs_scene.mesh.faceTexcoords.empty() && !mvs_scene.mesh.texturesDiffuse.empty())
					ImGui::Text("✓ Texture coordinates and textures available");
				if (!mvs_scene.mesh.vertexNormals.empty())
					ImGui::Text("✓ Vertex normals available");
			} else {
				ImGui::TextColored(ImVec4(1.f, 0.6f, 0.6f, 1.f), "⚠ No mesh data to export");
			}
			break;
		}

		ImGui::Separator();

		bool canExport = (exportFormat == 0 && hasPointCloud) || ((exportFormat == 1 || exportFormat == 2 || exportFormat == 3) && hasMesh);

		if (ImGui::Button("Export...", ImVec2(120, 0)) && canExport) {
			String filename;
			if (ShowSaveFileDialog(filename)) {
				// Determine export type based on format selection
				String exportType;
				switch (exportFormat) {
				case 0: exportType = ".ply"; break;
				case 1: exportType = ".ply"; break;
				case 2: exportType = ".obj"; break;
				case 3: exportType = ".glb"; break;
				}

				// Ensure the filename has the correct extension
				String baseFileName = Util::getFileFullName(filename);
				String finalFileName = baseFileName + exportType;
				scene.Export(finalFileName, exportType);
			}
			showExportDialog = false;
		}

		if (!canExport) {
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.f), "(Export disabled - no compatible data)");
		}

		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
			showExportDialog = false;
	}
	ImGui::End();
}

void UI::ShowCameraInfoDialog(Window& window) {
	if (!showCameraInfoDialog) return;

	// Show as a regular window instead of modal popup
	ImGui::SetNextWindowPos(ImVec2(880, 100), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(390, 612), ImGuiCond_FirstUseEver);
	if (ImGui::Begin("Camera Information", &showCameraInfoDialog)) {
		const Scene& scene = window.GetScene();
		const ImageArr& images = scene.GetImages();
		const MVS::Scene& mvs_scene = scene.GetScene();

		// Check if we have a selected camera
		if (window.selectionType == Window::SEL_CAMERA && window.selectionIdx < images.size()) {
			const Image& image = images[window.selectionIdx];
			ASSERT(image.idx < mvs_scene.images.size());
			const MVS::Image& imageData = mvs_scene.images[image.idx];
			const MVS::Camera& camera = imageData.camera;
			Point3 eulerAngles;
			camera.R.GetRotationAnglesZYX(eulerAngles.x, eulerAngles.y, eulerAngles.z);

			// Basic image information
			ImGui::Text("Index: %u (ID: %u)", image.idx, imageData.ID);
			ImGui::Text("Name: %s", Util::getFileNameExt(imageData.name).c_str());
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Full Path: %s", imageData.name.c_str());
			if (!imageData.maskName.empty()) {
				ImGui::Text("Mask: %s", Util::getFileNameExt(imageData.maskName).c_str());
				ImGui::Text("Mask Path: %s", imageData.maskName.c_str());
			} else {
				ImGui::Text("Mask: None");
			}

			ImGui::Separator();

			// Image dimensions and properties
			ImGui::Text("Image Properties");
			ImGui::Text("  Size: %ux%u pixels", imageData.width, imageData.height);
			ImGui::Text("  Scale: %.3f", imageData.scale);
			ImGui::Text("  Average Depth: %.3g", imageData.avgDepth);

			// Additional image statistics if available
			if (ImGui::CollapsingHeader("Image Additional Information")) {
				// Check if image is loaded
				if (!imageData.image.empty()) {
					ImGui::Text("  Image Status: Loaded (%dx%dx%d)", 
						imageData.image.cols, imageData.image.rows, imageData.image.channels());
				} else {
					ImGui::Text("  Image Status: Not loaded");
				}
				// Show if image is calibrated
				ASSERT(imageData.platformID != NO_ID);
				ImGui::Text("  Platform ID: %u", imageData.platformID);
				ImGui::Text("  Camera ID: %u (from %u)", imageData.cameraID, mvs_scene.platforms[imageData.cameraID].cameras.size());
				ImGui::Text("  Pose ID: %u", imageData.poseID);
			}

			ImGui::Separator();

			// Camera intrinsics
			ImGui::Text("Camera Intrinsics");
			ImGui::Text("  Focal Length: fx=%.2f, fy=%.2f", camera.K(0, 0), camera.K(1, 1));
			ImGui::Text("  Principal Point: cx=%.2f, cy=%.2f", camera.K(0, 2), camera.K(1, 2));

			// Show full intrinsic matrix
			if (ImGui::CollapsingHeader("Camera Additional Information")) {
				ImGui::Text("  FOV: x=%.2f, y=%.2f", R2D(imageData.ComputeFOV(0)), R2D(imageData.ComputeFOV(1)));
				ImGui::Text("  Intrinsic Matrix K:");
				ImGui::Text("    [%.2f  %.2f  %.2f]", camera.K(0, 0), camera.K(0, 1), camera.K(0, 2));
				ImGui::Text("    [%.2f  %.2f  %.2f]", camera.K(1, 0), camera.K(1, 1), camera.K(1, 2));
				ImGui::Text("    [%.2f  %.2f  %.2f]", camera.K(2, 0), camera.K(2, 1), camera.K(2, 2));
			}

			ImGui::Separator();

			// Camera extrinsics
			ImGui::Text("Camera Extrinsics");
			ImGui::Text("  Position: (%.6f, %.6f, %.6f)", camera.C.x, camera.C.y, camera.C.z);
			ImGui::Text("  Rotation (Euler XYZ): %.3f°, %.3f°, %.3f°", 
				R2D(eulerAngles.x), R2D(eulerAngles.y), R2D(eulerAngles.z));

			// Show full rotation matrix
			if (ImGui::CollapsingHeader("Rotation Matrix R")) {
				ImGui::Text("  [%.6f  %.6f  %.6f]", camera.R(0, 0), camera.R(0, 1), camera.R(0, 2));
				ImGui::Text("  [%.6f  %.6f  %.6f]", camera.R(1, 0), camera.R(1, 1), camera.R(1, 2));
				ImGui::Text("  [%.6f  %.6f  %.6f]", camera.R(2, 0), camera.R(2, 1), camera.R(2, 2));
			}

			ImGui::Separator();

			// Neighbors information
			ImGui::Text("Neighbor Images: %u", imageData.neighbors.size());
			ImGui::Text("Selected Neighbor Index: %s", window.selectedNeighborCamera == NO_ID ? "NA" : std::to_string(window.selectedNeighborCamera).c_str());
			ImGui::Text("Selected Neighbor Angle: %s", 
				window.selectedNeighborCamera == NO_ID ? "NA" : String::FormatString("%.2f", R2D(ACOS(ComputeAngle(
					mvs_scene.images[images[window.selectionIdx].idx].camera.Direction().ptr(),
					mvs_scene.images[images[window.selectedNeighborCamera].idx].camera.Direction().ptr())))).c_str());
			if (!imageData.neighbors.empty()) {
				// Create a scrollable region for the neighbors list
				ImGui::BeginChild("NeighborsScrollRegion", ImVec2(0, 220), true, ImGuiWindowFlags_HorizontalScrollbar);
				// Table headers
				if (ImGui::BeginTable("NeighborsTable", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_HighlightHoveredColumn)) {
					ImGui::TableSetupColumn("Index/ID", ImGuiTableColumnFlags_WidthFixed, 45.f);
					ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 50.f);
					ImGui::TableSetupColumn("Angle", ImGuiTableColumnFlags_WidthFixed, 33.f);
					ImGui::TableSetupColumn("Area", ImGuiTableColumnFlags_WidthFixed, 24.f);
					ImGui::TableSetupColumn("Points", ImGuiTableColumnFlags_WidthFixed, 39.f);
					ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
					ImGui::TableHeadersRow();
					// Show neighbors in table format
					for (const auto& neighbor : imageData.neighbors) {
						const MVS::Image& neighborImage = mvs_scene.images[neighbor.ID];
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						// Highlight and make the entire row clickable by using Selectable
						const bool isSelected(window.selectedNeighborCamera == neighbor.ID);
						String rowLabel = String::FormatString("%u/%u##neighbor_%u", neighbor.ID, neighborImage.ID, neighbor.ID);
						bool rowClicked = ImGui::Selectable(rowLabel.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap);
						// Handle row click
						if (rowClicked) {
							// Deselect if already selected, or select it otherwise
							window.selectedNeighborCamera = (window.selectedNeighborCamera == neighbor.ID) ? NO_ID : scene.ImageIdxMVS2Viewer(neighbor.ID);
							window.GetRenderer().UploadSelection(window);
							window.RequestRedraw();
						}
						// Handle double-click to focus on the neighbor camera
						if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
							// Select and focus on the neighbor camera
							window.selectionType = Window::SEL_CAMERA;
							window.selectionIdx = scene.ImageIdxMVS2Viewer(neighbor.ID);
							window.selectedNeighborCamera = NO_ID;
							window.GetCamera().SetCameraViewMode(window.selectionIdx);
							window.GetRenderer().UploadSelection(window);
							ImGui::SetWindowFocus(nullptr); // Defocus dialog window
							window.RequestRedraw();
						}
						// Fill in the other columns with data
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("%.2f", neighbor.score);
						ImGui::TableSetColumnIndex(2);
						ImGui::Text("%.2f", FR2D(neighbor.angle));
						ImGui::TableSetColumnIndex(3);
						ImGui::Text("%d", ROUND2INT(neighbor.area*100));
						ImGui::TableSetColumnIndex(4);
						ImGui::Text("%u", neighbor.points);
						ImGui::TableSetColumnIndex(5);
						ImGui::Text("%s", Util::getFileNameExt(neighborImage.name).c_str());
					}
					ImGui::EndTable();
				}
				ImGui::EndChild();
			}
		} else {
			// No camera selected - clear neighbor selection as well
			if (window.selectedNeighborCamera != NO_ID) {
				window.selectedNeighborCamera = NO_ID;
				window.RequestRedraw();
			}
			ImGui::Text("No camera/image selected");
			ImGui::Separator();
			ImGui::Text("Select a camera by clicking on it in the 3D view");
			ImGui::Text("or double-clicking to enter camera view mode.");
			ImGui::Spacing();
			ImGui::Text("Select a camera in 3D while pressing Ctrl in order");
			ImGui::Text("to select a neighbor camera, or select it in the");
			ImGui::Text("neighbors list.");
			ImGui::Separator();
			ImGui::Text("Total cameras in scene: %u", (unsigned)mvs_scene.images.size());
		}
	}
	ImGui::End();
}

void UI::ShowSelectionDialog(Window& window) {
	if (!showSelectionDialog) return;

	// Input buffers for the dialog
	static char selectionInputBuffer[256] = "";
	static int selectionType = 0; // 0 Point, 1 Face, 2 Camera by Index, 3 Camera by Name

	// Set dialog properties
	ImGui::OpenPopup("Selection Dialog");
	if (ImGui::BeginPopupModal("Selection Dialog", &showSelectionDialog, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("Select an element by index or name:");
		ImGui::Separator();

		// Selection type radio buttons
		ImGui::RadioButton("Point by Index", &selectionType, 0);
		ImGui::SameLine();
		ImGui::RadioButton("Face by Index", &selectionType, 1);
		ImGui::RadioButton("Camera by Index", &selectionType, 2);
		ImGui::SameLine();
		ImGui::RadioButton("Camera by Name", &selectionType, 3);

		ImGui::Separator();

		// Input fields based on selection type
		IDX selectionIdx = NO_IDX;
		const Scene& scene = window.GetScene();
		const MVS::Scene& mvs_scene = scene.GetScene();
		ImGui::InputText("##selectionInput", selectionInputBuffer, sizeof(selectionInputBuffer),
			selectionType < 3 ? ImGuiInputTextFlags_CharsDecimal : ImGuiInputTextFlags_None);
		if (strlen(selectionInputBuffer) > 0) {
			switch (selectionType) {
			case 0: { // Point by Index
				const int pointIndex = atoi(selectionInputBuffer);
				if (pointIndex >= 0 && pointIndex < (int)mvs_scene.pointcloud.points.size())
					selectionIdx = pointIndex;
				else
					ImGui::TextColored(ImVec4(1, 0, 0, 1), "Invalid point index! Range: 0-%lu", mvs_scene.pointcloud.points.size() - 1);
				} break;

			case 1: { // Face by Index
				const int faceIndex = atoi(selectionInputBuffer);
				if (faceIndex >= 0 && faceIndex < (int)mvs_scene.mesh.faces.size())
					selectionIdx = faceIndex;
				else
					ImGui::TextColored(ImVec4(1, 0, 0, 1), "Invalid face index! Range: 0-%u", mvs_scene.mesh.faces.size() - 1);
				} break;

			case 2: { // Camera by Index
				const int cameraIndex = atoi(selectionInputBuffer);
				if (cameraIndex >= 0 && cameraIndex < (int)mvs_scene.images.size())
					selectionIdx = cameraIndex;
				else
					ImGui::TextColored(ImVec4(1, 0, 0, 1), "Invalid camera index! Range: 0-%u", mvs_scene.images.size() - 1);
				} break;

			case 3: { // Camera by Name
				int cameraIndex = -1;
				const ImageArr& images = scene.GetImages();
				FOREACH(i, images) {
					if (images[i].idx < mvs_scene.images.size()) {
						const MVS::Image& imageData = mvs_scene.images[images[i].idx];
						String fileName = Util::getFileNameExt(imageData.name);
						if (fileName.find(selectionInputBuffer) != String::npos) {
							cameraIndex = i;
							break;
						}
					}
				}
				if (cameraIndex != -1)
					selectionIdx = cameraIndex;
				else
					ImGui::TextColored(ImVec4(1, 0, 0, 1), "Camera name not found!");
				} break;
			}
		}

		ImGui::Separator();

		// Buttons
		if (ImGui::Button("Select", ImVec2(120, 0)) && selectionIdx != NO_IDX) {
			// Perform the selection based on the type
			switch (selectionType) {
			case 0: { // Point by Index
				window.selectionType = Window::SEL_POINT;
				window.selectionIdx = selectionIdx;
				window.selectionPoints[0] = mvs_scene.pointcloud.points[selectionIdx];
			} break;

			case 1: { // Face by Index
				window.selectionType = Window::SEL_TRIANGLE;
				window.selectionIdx = selectionIdx;
				const MVS::Mesh::Face& face = mvs_scene.mesh.faces[selectionIdx];
				window.selectionPoints[0] = mvs_scene.mesh.vertices[face[0]];
				window.selectionPoints[1] = mvs_scene.mesh.vertices[face[1]];
				window.selectionPoints[2] = mvs_scene.mesh.vertices[face[2]];
			} break;

			case 2:   // Camera by Index
			case 3: { // Camera by Name
				window.selectionType = Window::SEL_CAMERA;
				window.selectionIdx = selectionIdx;
				const MVS::Image& imageData = mvs_scene.images[scene.GetImages()[selectionIdx].idx];
				window.selectionPoints[0] = imageData.camera.C;
			} break;
			}
			window.selectionTime = glfwGetTime();

			// Update renderer and request redraw
			window.GetRenderer().UploadSelection(window);
			window.RequestRedraw();
			
			// Close dialog
			showSelectionDialog = false;
			ImGui::CloseCurrentPopup();
		}

		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0))) {
			showSelectionDialog = false;
			ImGui::CloseCurrentPopup();
		}

		ImGui::EndPopup();
	}
}

void UI::UpdateFrameStats(double frameDeltaTime) {
	constexpr float updateInterval = 0.5f; // Update every 500ms
	++frameCount;
	deltaTime += frameDeltaTime;
	if (deltaTime >= updateInterval) {
		fps = static_cast<double>(frameCount) / deltaTime;
		deltaTime = 0;
		frameCount = 0;
	}
}

void UI::SetupStyle() {
	ImGuiStyle& style = ImGui::GetStyle();

	// Color scheme
	ImVec4* colors = style.Colors;
	colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.1f, 0.9f);
	colors[ImGuiCol_MenuBarBg] = ImVec4(0.2f, 0.2f, 0.2f, 1.f);
	colors[ImGuiCol_Header] = ImVec4(0.3f, 0.3f, 0.3f, 1.f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.4f, 0.4f, 0.4f, 1.f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.5f, 0.5f, 0.5f, 1.f);

	// Spacing
	style.WindowPadding = ImVec2(8, 8);
	style.ItemSpacing = ImVec2(6, 4);
	style.ItemInnerSpacing = ImVec2(4, 4);
	style.WindowRounding = 5.f;
	style.FrameRounding = 3.f;
}

void UI::ShowRenderingControls(Window& window) {
	ImGui::Text("Rendering");
	ImGui::Separator();

	// Background color
	if (ImGui::ColorEdit3("Background", window.clearColor.data()))
		window.RequestRedraw();

	// Render-only-on-change optimization
	ImGui::Checkbox("Render Only on Change", &window.renderOnlyOnChange);
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("Optimize performance by rendering only when scene changes\n"
						  "Reduces CPU/GPU usage for static scenes");

	// Image overlay opacity (only show when in camera view mode)
	if (window.GetCamera().IsCameraViewMode()) {
		ImGui::Separator();
		ImGui::Text("Image Overlay");
		if (ImGui::SliderFloat("Opacity", &window.imageOverlayOpacity, 0.f, 1.f, "%.2f"))
			window.RequestRedraw();
		ImGui::Text("Camera ID: %d", (int)window.GetCamera().GetCurrentCamID());
	}

	// Arcball gizmo controls (only show when in arcball mode)
	if (window.GetControlMode() == Window::CONTROL_ARCBALL) {
		ImGui::Separator();
		ImGui::Text("Arcball Gizmos");
		bool enableGizmos = window.GetArcballControls().getEnableGizmos();
		if (ImGui::Checkbox("Show Gizmos", &enableGizmos)) {
			window.GetArcballControls().setEnableGizmos(enableGizmos);
			window.RequestRedraw();
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Show arcball gizmos (replaces coordinate axes)");
		if (enableGizmos) {
			// Gizmo center control (indented under main gizmo control)
			ImGui::SameLine();
			bool enableGizmosCenter = window.GetArcballControls().getEnableGizmosCenter();
			if (ImGui::Checkbox("Show Center", &enableGizmosCenter)) {
				window.GetArcballControls().setEnableGizmosCenter(enableGizmosCenter);
				window.RequestRedraw();
			}
			if (ImGui::IsItemHovered())
				ImGui::SetTooltip("Show small axes at the center of the trackball");
		}
	}
}

void UI::ShowPointCloudControls(Window& window) {
	ImGui::Text("Point Cloud");
	ImGui::Separator();

	if (ImGui::Checkbox("Show Point Cloud", &window.showPointCloud))
		window.RequestRedraw();
	if (window.showPointCloud) {
		ImGui::Indent();
		if (ImGui::SliderFloat("Point Size", &window.pointSize, 1.f, 10.f))
			window.RequestRedraw();
		// Check if normals are available
		const MVS::Scene& scene = window.GetScene().GetScene();
		if (!scene.pointcloud.normals.empty()) {
			if (ImGui::Checkbox("Show Normals", &window.showPointCloudNormals))
				window.RequestRedraw();
			if (window.showPointCloudNormals) {
				ImGui::Indent();
				if (ImGui::SliderFloat("Normal Length", &window.pointNormalLength, 0.001f, 0.1f, "%.3f")) {
					// Re-upload point cloud with new normal length
					window.GetRenderer().UploadPointCloud(scene.pointcloud, window.pointNormalLength);
					window.RequestRedraw();
				}
				ImGui::Unindent();
			}
		} else {
			// Disable the checkbox if no normals are available
			bool disabled = false;
			ImGui::BeginDisabled();
			ImGui::Checkbox("Show Normals (NA)", &disabled);
			ImGui::EndDisabled();
		}
		ImGui::Unindent();
	}
}

void UI::ShowMeshControls(Window& window) {
	ImGui::Text("Mesh");
	ImGui::Separator();

	if (ImGui::Checkbox("Show Mesh", &window.showMesh))
		window.RequestRedraw();
	if (window.showMesh) {
		ImGui::Indent();
		if (ImGui::Checkbox("Wireframe", &window.showMeshWireframe))
			window.RequestRedraw();
		if (ImGui::Checkbox("Textured", &window.showMeshTextured))
			window.RequestRedraw();

		// Sub-mesh controls
		if (!window.meshSubMeshVisible.empty()) {
			ImGui::Separator();
			ImGui::Text("Sub-meshes (%zu total)", window.meshSubMeshVisible.size());

			// All/None buttons for convenience
			ImGui::SameLine();
			if (ImGui::SmallButton("All")) {
				std::fill(window.meshSubMeshVisible.begin(), window.meshSubMeshVisible.end(), true);
				window.RequestRedraw();
			}
			ImGui::SameLine();
			if (ImGui::SmallButton("None")) {
				std::fill(window.meshSubMeshVisible.begin(), window.meshSubMeshVisible.end(), false);
				window.RequestRedraw();
			}

			// Individual sub-mesh checkboxes
			FOREACH(i, window.meshSubMeshVisible) {
				// Create a meaningful label for each sub-mesh
				String label = String::FormatString("Sub-mesh %zu", i);
				// Add texture info if available
				bool isVisible = window.meshSubMeshVisible[i];
				if (ImGui::Checkbox(label, &isVisible)) {
					window.meshSubMeshVisible[i] = isVisible;
					window.RequestRedraw();
				}
			}
		}
		ImGui::Unindent();
	}
}

void UI::ShowSelectionOverlay(const Window& window) {
	if (!showSelectionOverlay) return;

	// Only show if there's a valid selection
	if (window.selectionType == Window::SEL_NA) return;

	ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | 
								   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | 
								   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;

	const ImGuiViewport* vp = ImGui::GetMainViewport();
	ImVec2 work_pos = vp->WorkPos;
	ImVec2 work_size = vp->WorkSize;

	// Position in bottom left corner
	ImVec2 window_pos;
	window_pos.x = work_pos.x + PAD;
	window_pos.y = work_pos.y + work_size.y - PAD;

	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(0.f, 1.f));
	ImGui::SetNextWindowBgAlpha(0.35f);

	if (ImGui::Begin("Selection Info", &showSelectionOverlay, window_flags)) {
		// Check for double-click on the selection overlay to open selection dialog
		if (ImGui::IsWindowHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			showSelectionDialog = true;
		const Scene& scene = window.GetScene();
		switch (window.selectionType) {
		case Window::SEL_TRIANGLE: {
			const MVS::Scene& mvs_scene = scene.GetScene();
			ImGui::Text("Face selected:");
			ImGui::Text("  index: %zu", window.selectionIdx);
			if (!mvs_scene.mesh.IsEmpty() && window.selectionIdx < mvs_scene.mesh.faces.size()) {
				const MVS::Mesh::Face& face = mvs_scene.mesh.faces[window.selectionIdx];
				ImGui::Text("  vertex 1: %u (%.3f, %.3f, %.3f)", face[0], 
					window.selectionPoints[0].x, window.selectionPoints[0].y, window.selectionPoints[0].z);
				ImGui::Text("  vertex 2: %u (%.3f, %.3f, %.3f)", face[1], 
					window.selectionPoints[1].x, window.selectionPoints[1].y, window.selectionPoints[1].z);
				ImGui::Text("  vertex 3: %u (%.3f, %.3f, %.3f)", face[2], 
					window.selectionPoints[2].x, window.selectionPoints[2].y, window.selectionPoints[2].z);
			}
			break; }
		case Window::SEL_POINT: {
			const MVS::Scene& mvs_scene = scene.GetScene();
			ImGui::Text("Point selected:");
			ImGui::Text("  index: %zu (%.3f, %.3f, %.3f)", 
				window.selectionIdx,
				window.selectionPoints[0].x, window.selectionPoints[0].y, window.selectionPoints[0].z);

			// Show view information if available
			if (!mvs_scene.pointcloud.pointViews.empty() && window.selectionIdx < mvs_scene.pointcloud.pointViews.size()) {
				const MVS::PointCloud::ViewArr& views = mvs_scene.pointcloud.pointViews[window.selectionIdx];
				if (!views.empty()) {
					ImGui::Text("  views: %u", (unsigned)views.size());
					// Show first few views to avoid overwhelming the display
					int maxViewsToShow = MINF(8, (int)views.size());
					for (int v = 0; v < maxViewsToShow; ++v) {
						const MVS::PointCloud::View idxImage = views[v];
						if (idxImage < mvs_scene.images.size()) {
							const MVS::Image& imageData = mvs_scene.images[idxImage];
							const Point2 x(imageData.camera.TransformPointW2I(Cast<REAL>(window.selectionPoints[0])));
							const float conf = mvs_scene.pointcloud.pointWeights.empty() ? 0.f : 
								mvs_scene.pointcloud.pointWeights[window.selectionIdx][v];

							String fileName = Util::getFileNameExt(imageData.name);
							ImGui::Text("    %s (%.1f %.1f px, %.2f conf)", 
								fileName.c_str(), x.x, x.y, conf);
						}
					}
					if ((int)views.size() > maxViewsToShow)
						ImGui::Text("    ... and %d more", (int)views.size() - maxViewsToShow);
				}
			}
			break; }
		case Window::SEL_CAMERA: {
			const ImageArr& images = scene.GetImages();
			const MVS::Scene& mvs_scene = scene.GetScene();
			if (window.selectionIdx < images.size()) {
				const Image& image = images[window.selectionIdx];
				if (image.idx < mvs_scene.images.size()) {
					const MVS::Image& imageData = mvs_scene.images[image.idx];
					const MVS::Camera& camera = imageData.camera;
					Point3 eulerAngles;
					camera.R.GetRotationAnglesZYX(eulerAngles.x, eulerAngles.y, eulerAngles.z);

					ImGui::Text("Camera selected:");
					ImGui::Text("  index: %u (ID: %u)", image.idx, imageData.ID);
					ImGui::Text("  name: %s", Util::getFileNameExt(imageData.name).c_str());
					if (!imageData.maskName.empty()) {
						ImGui::Text("  mask: %s", Util::getFileNameExt(imageData.maskName).c_str());
					}
					ImGui::Text("  image size: %ux%u", imageData.width, imageData.height);
					ImGui::Text("  intrinsics: fx %.1f, fy %.1f", camera.K(0, 0), camera.K(1, 1));
					ImGui::Text("             cx %.1f, cy %.1f", camera.K(0, 2), camera.K(1, 2));
					ImGui::Text("  position: %.3f, %.3f, %.3f", camera.C.x, camera.C.y, camera.C.z);
					ImGui::Text("  rotation: %.1f°, %.1f°, %.1f°", 
						R2D(eulerAngles.x), R2D(eulerAngles.y), R2D(eulerAngles.z));
					ImGui::Text("  avg depth: %.2g", imageData.avgDepth);
					ImGui::Text("  neighbors: %u", (unsigned)imageData.neighbors.size());
				}
			}
			break; }
		}
		if (window.GetCamera().IsCameraViewMode()) {
			// Show camera view information if in camera view mode
			const MVS::Scene& mvs_scene = scene.GetScene();
			const Image& image = scene.GetImages()[window.GetCamera().GetCurrentCamID()];
			ASSERT(image.idx < mvs_scene.images.size());
			const MVS::Image& imageData = mvs_scene.images[image.idx];
			ImGui::Separator();
			ImGui::Text("Camera View Mode:");
			ImGui::Text("  index: %u (ID: %u)", image.idx, imageData.ID);
			ImGui::Text("  Image: %s", Util::getFileNameExt(imageData.name).c_str());
		}
	}
	ImGui::End();
}

void UI::UpdateMenuVisibility() {
	bool mouseNearMenu = IsMouseNearMenuArea();
	bool menuInUse = IsMenuInUse();
	double currentTime = glfwGetTime();

	// Show menu if mouse is near top or menu is in use
	if (mouseNearMenu || menuInUse) {
		showMainMenu = true;
		lastMenuInteraction = currentTime;
	}
	// Hide menu if enough time has passed since last interaction
	else if (showMainMenu && (currentTime - lastMenuInteraction) > menuFadeOutDelay) {
		showMainMenu = false;
	}

	// Track if menu visibility changed
	menuWasVisible = showMainMenu;
}

bool UI::IsMouseNearMenuArea() const {
	ImGuiIO& io = ImGui::GetIO();

	// Check if mouse position is valid and near the top of the screen
	if (io.MousePos.x < 0 || io.MousePos.y < 0) return false;

	return io.MousePos.y <= menuTriggerHeight;
}

bool UI::IsMenuInUse() const {
	// Menu is in use if menu-related dialogs are open
	if (showAboutDialog || showHelpDialog || showExportDialog)
		return true;

	// Check if any menu-related popup is open (but not all popups)
	if (ImGui::IsPopupOpen("About", ImGuiPopupFlags_None) ||
		ImGui::IsPopupOpen("Help", ImGuiPopupFlags_None))
		return true;

	// Check if the main menu bar is currently active or being interacted with
	if (showMainMenu) {
		// Check if any menu item is active, focused, or hovered
		if (ImGui::IsAnyItemActive() || ImGui::IsAnyItemFocused() || ImGui::IsAnyItemHovered())
			return true;

		// Check if any menu popup is open (File, View, Help menus)
		if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopup))
			return true;
	}

	return false;
}

bool UI::WantCaptureMouse() const {
	return ImGui::GetIO().WantCaptureMouse;
}

bool UI::WantCaptureKeyboard() const {
	return ImGui::GetIO().WantCaptureKeyboard;
}

void UI::HandleGlobalKeys(Window& window) {
	// Handle UI-specific escape behavior for closing dialogs and defocusing
	if (ImGui::IsKeyReleased(ImGuiKey_Escape)) {
		// If camera in view mode, exit camera view
		if (window.GetCamera().IsCameraViewMode()) {
			window.GetCamera().DisableCameraViewMode();
			return;
		}
		// If any dialog is open, close it
		if (showAboutDialog) {
			showAboutDialog = false;
			return;
		}
		if (showHelpDialog) {
			showHelpDialog = false;
			return;
		}
		if (showExportDialog) {
			showExportDialog = false;
			return;
		}
		if (showSceneInfo) {
			showSceneInfo = false;
			return;
		}
		if (showCameraInfoDialog) {
			showCameraInfoDialog = false;
			return;
		}
		if (showCameraControls) {
			showCameraControls = false;
			return;
		}
		if (showSelectionControls) {
			showSelectionControls = false;
			return;
		}
		if (showSelectionDialog) {
			showSelectionDialog = false;
			return;
		}
		if (showRenderSettings) {
			showRenderSettings = false;
			return;
		}

		// If any popup is open, close it
		if (ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopup)) {
			ImGui::CloseCurrentPopup();
			return;
		}

		// Clear focus from any focused window or item
		ImGui::SetWindowFocus(nullptr);
		ImGui::ClearActiveID();

		// Also hide the main menu if it's visible
		if (showMainMenu)
			showMainMenu = false;
	}
}

void* SettingsReadOpen(ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) {
	if (strcmp(name, "Window") == 0)
		return handler->UserData;
	return nullptr;
}

void SettingsReadLine(ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
	Window& window = *static_cast<Window*>(handler->UserData);

	float x, y, z, w;
	int intVal;

	if (sscanf(line, "RenderOnlyOnChange=%d", &intVal) == 1) {
		window.renderOnlyOnChange = (intVal != 0);
	}
	else if (sscanf(line, "ClearColor=%f,%f,%f,%f", &x, &y, &z, &w) == 4) {
		window.clearColor = Eigen::Vector4f(x, y, z, w);
	}
	else if (sscanf(line, "PointSize=%f", &x) == 1) {
		window.pointSize = x;
	}
	else if (sscanf(line, "EstimateSfMNormals=%d", &intVal) == 1) {
		window.GetScene().estimateSfMNormals = (intVal != 0);
	}
	else if (sscanf(line, "EstimateSfMPatches=%d", &intVal) == 1) {
		window.GetScene().estimateSfMPatches = (intVal != 0);
	}
	else if (sscanf(line, "ShowCameras=%d", &intVal) == 1) {
		window.showCameras = (intVal != 0);
	}
	else if (sscanf(line, "ShowMeshWireframe=%d", &intVal) == 1) {
		window.showMeshWireframe = (intVal != 0);
	}
	else if (sscanf(line, "ShowMeshTextured=%d", &intVal) == 1) {
		window.showMeshTextured = (intVal != 0);
	}
	else if (sscanf(line, "ImageOverlayOpacity=%f", &x) == 1) {
		window.imageOverlayOpacity = x;
	}
	else if (sscanf(line, "ArcballRenderGizmos=%d", &intVal) == 1) {
		window.GetArcballControls().setEnableGizmos(intVal != 0);
	}
	else if (sscanf(line, "ArcballRenderGizmosCenter=%d", &intVal) == 1) {
		window.GetArcballControls().setEnableGizmosCenter(intVal != 0);
	}
	else if (sscanf(line, "ArcballRotationSensitivity=%f", &x) == 1) {
		window.GetArcballControls().setRotationSensitivity(x);
	}
	else if (sscanf(line, "ArcballZoomSensitivity=%f", &x) == 1) {
		window.GetArcballControls().setZoomSensitivity(x);
	}
	else if (sscanf(line, "ArcballPanSensitivity=%f", &x) == 1) {
		window.GetArcballControls().setPanSensitivity(x);
	}
}

void SettingsWriteAll(ImGuiContext*, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
	Window& window = *static_cast<Window*>(handler->UserData);
	buf->appendf("[%s][Window]\n", handler->TypeName);
	buf->appendf("RenderOnlyOnChange=%d\n", window.renderOnlyOnChange ? 1 : 0);
	buf->appendf("ClearColor=%f,%f,%f,%f\n", 
		window.clearColor[0], window.clearColor[1], 
		window.clearColor[2], window.clearColor[3]);
	buf->appendf("PointSize=%f\n", window.pointSize);
	buf->appendf("EstimateSfMNormals=%d\n", 
		window.GetScene().estimateSfMNormals ? 1 : 0);
	buf->appendf("EstimateSfMPatches=%d\n",
		window.GetScene().estimateSfMPatches ? 1 : 0);
	buf->appendf("ShowCameras=%d\n", window.showCameras ? 1 : 0);
	buf->appendf("ShowMeshWireframe=%d\n", window.showMeshWireframe ? 1 : 0);
	buf->appendf("ShowMeshTextured=%d\n", window.showMeshTextured ? 1 : 0);
	buf->appendf("ImageOverlayOpacity=%f\n", window.imageOverlayOpacity);
	buf->appendf("ArcballRenderGizmos=%d\n", 
		window.GetArcballControls().getEnableGizmos() ? 1 : 0);
	buf->appendf("ArcballRenderGizmosCenter=%d\n", 
		window.GetArcballControls().getEnableGizmosCenter() ? 1 : 0);
	buf->appendf("ArcballRotationSensitivity=%f\n", 
		window.GetArcballControls().getRotationSensitivity());
	buf->appendf("ArcballZoomSensitivity=%f\n", 
		window.GetArcballControls().getZoomSensitivity());
	buf->appendf("ArcballPanSensitivity=%f\n", 
		window.GetArcballControls().getPanSensitivity());
}

// Custom settings implementation
bool UI::ShowOpenFileDialog(String& filename, String& geometryFilename) {
	// Use portable-file-dialogs for cross-platform file dialog
	try {
		auto dialog = pfd::open_file(
			"Open Scene File",                          // title
			WORKING_FOLDER_FULL,                        // initial path (absolute)
			{
				"OpenMVS Scene Files", "*.mvs",
				"Mesh / Point Cloud Files", "*.ply", 
				"Mesh Files", "*.obj",
				"Mesh Files", "*.glb",
				"Depth Map Files", "*.dmap",
				"All Files", "*"
			},                                          // filters
			pfd::opt::multiselect                       // options
		);

		// Get the result
		auto result = dialog.result();
		if (!result.empty()) {
			filename = result[0];
			if (result.size() > 1)
				geometryFilename = result[1];
			else
				geometryFilename.clear();
			return true;
		}
	} catch (const std::exception& e) {
		DEBUG("File dialog error: %s", e.what());
	}
	return false;
}

bool UI::ShowSaveFileDialog(String& filename) {
	// Use portable-file-dialogs for cross-platform save dialog
	try {
		auto dialog = pfd::save_file(
			"Save Scene File",                          // title
			WORKING_FOLDER_FULL,                        // initial directory (like open dialog)
			{
				"OpenMVS Scene Files", "*.mvs",
				"Mesh / Point Cloud Files", "*.ply",
				"Mesh Files", "*.obj",
				"Mesh Files", "*.glb",
				"All Files", "*"
			},                                          // filters
			pfd::opt::none                              // options
		);

		// Get the result - save_file returns a string directly, not a vector
		auto result = dialog.result();
		if (!result.empty()) {
			filename = result;
			return true;
		}
	} catch (const std::exception& e) {
		DEBUG("File dialog error: %s", e.what());
	}
	return false;
}

void UI::SetupCustomSettings(Window& window) {
	// Register custom settings handler
	ImGuiContext& ctx = *ImGui::GetCurrentContext();
	ImGuiSettingsHandler handler;
	handler.TypeName = "ViewerSettings";
	handler.TypeHash = ImHashStr("ViewerSettings");
	handler.ReadOpenFn = SettingsReadOpen;
	handler.ReadLineFn = SettingsReadLine;
	handler.WriteAllFn = SettingsWriteAll;
	handler.UserData = &window; // Pass window pointer as user data
	ctx.SettingsHandlers.push_back(handler);
}
/*----------------------------------------------------------------*/
