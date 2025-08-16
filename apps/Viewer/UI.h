/*
 * UI.h
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

#include "Camera.h"
#include "Image.h"

namespace VIEWER {

// Forward declarations
class Scene;
class Window;

class UI {
private:
	String iniPath;

	bool showSceneInfo;
	bool showCameraControls;
	bool showSelectionControls;
	bool showRenderSettings;
	bool showPerformanceOverlay;
	bool showViewportOverlay;
	bool showSelectionOverlay;
	bool showAboutDialog;
	bool showHelpDialog;
	bool showExportDialog;
	bool showCameraInfoDialog;
	bool showSelectionDialog;

	// Auto-hiding menu state
	bool showMainMenu;
	bool menuWasVisible;
	float menuTriggerHeight;
	double lastMenuInteraction;
	float menuFadeOutDelay;

	// Statistics
	double deltaTime;
	uint32_t frameCount;
	float fps;

public:
	UI();
	~UI();

	bool Initialize(Window& window, const String& glslVersion = "#version 330");
	void Release();

	void NewFrame(Window& window);
	void Render();

	// Main UI panels
	void ShowMainMenuBar(Window& window);
	void ShowSceneInfo(const Window& window);
	void ShowCameraControls(Window& window);
	void ShowSelectionControls(Window& window);
	void ShowRenderSettings(Window& window);
	void ShowPerformanceOverlay(Window& window);
	void ShowViewportOverlay(const Window& window);
	void ShowSelectionOverlay(const Window& window);
	void ToggleHelpDialog() { showHelpDialog = !showHelpDialog; }

	// Dialogs
	void ShowAboutDialog();
	void ShowHelpDialog();
	void ShowExportDialog(Scene& scene);
	void ShowCameraInfoDialog(Window& window);
	void ShowSelectionDialog(Window& window);
	static bool ShowOpenFileDialog(String& filename, String& geometryFilename);
	static bool ShowSaveFileDialog(String& filename);

	// Input handling
	bool WantCaptureMouse() const;
	bool WantCaptureKeyboard() const;
	void HandleGlobalKeys(Window& window);

	void UpdateFrameStats(double deltaTime);

private:
	void SetupStyle();
	void SetupCustomSettings(Window& window);
	void ShowRenderingControls(Window& window);
	void ShowPointCloudControls(Window& window);
	void ShowMeshControls(Window& window);
	void ShowSelectionInfo(const Window& window);

	// Auto-hiding menu helpers
	void UpdateMenuVisibility();
	bool IsMouseNearMenuArea() const;
	bool IsMenuInUse() const;

	String FormatFileSize(size_t bytes);
	String FormatDuration(double seconds);
};
/*----------------------------------------------------------------*/

} // namespace VIEWER
