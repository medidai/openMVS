/*
 * BoundingBoxEdit.h
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

namespace VIEWER {

//  BoxHandleInteraction - pure math for picking/dragging OBB handles
//
// Self-contained: inputs are an OBB3f, a Ray3d, and for drags the start +
// current rays. Outputs are either a picked handle descriptor or a new OBB3f.
// No rendering, no ImGui, no Window dependency - reusable anywhere.
//
// Corner indexing (matches OBB3f::GetCorners):
//   bit 0 = X half-extent sign (0=minus, 1=plus)
//   bit 1 = Y half-extent sign
//   bit 2 = Z half-extent sign
// So corner 0 = (---), corner 7 = (+++), opposite corner = i XOR 7.
//
// Face indexing 0..5 maps to (axis, sign):
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z   (face = 2*axis + (sign<0 ? 1 : 0)).
namespace BoxHandleInteraction {

enum HandleKind {
	HANDLE_NONE,
	HANDLE_CORNER,   // 8 corners
	HANDLE_FACE,     // 6 face centers
	HANDLE_ROT_X,    // rotation ring around local X axis
	HANDLE_ROT_Y,
	HANDLE_ROT_Z
};

struct Pick {
	HandleKind kind = HANDLE_NONE;
	int index = -1;          // corner / face index; unused for rotation rings
	double distance = 0.0;   // world-space t along the ray at the hit
	bool valid() const { return kind != HANDLE_NONE; }
};

// Pick the nearest OBB handle hit by 'ray'.
// Corner/face handles are spheres of radius 'handleRadiusWorld'; rings are
// thick circles with half-thickness 'ringThicknessWorld' (defaults to handleRadiusWorld).
Pick PickHandle(const OBB3f& obb, const Ray3d& ray,
                double handleRadiusWorld, double ringThicknessWorld = 0.0);

// Drag a corner while keeping the opposite corner fixed. Resizes m_ext and
// shifts m_pos; m_rot is preserved. cornerIdx must be in [0, 7].
OBB3f DragCorner(const OBB3f& original, int cornerIdx, const Ray3d& ray);

// Drag a face along its outward local-axis direction. The opposite face stays
// fixed; perpendicular motion is discarded. m_rot preserved. faceIdx in [0, 5].
OBB3f DragFace(const OBB3f& original, int faceIdx, const Ray3d& ray);

// Rotate the OBB around its own local axis 'axisIdx' (0..2). Angle is the
// signed sweep between the start/current ray hits on the ring plane through
// m_pos. m_pos and m_ext unchanged.
OBB3f DragRotation(const OBB3f& original, int axisIdx,
                   const Ray3d& startRay, const Ray3d& currentRay);

// Geometry helpers shared by the controller and the renderer.
void GetCornerWorldPositions(const OBB3f& obb, Eigen::Vector3f out[8]);
void GetFaceCenterWorldPositions(const OBB3f& obb, Eigen::Vector3f out[6]);

// World-space direction of the k-th local axis of 'obb' (unit vector).
// Derived from m_rot.row(k) because m_rot is stored as world->local.
Eigen::Vector3f GetLocalAxisWorldDir(const OBB3f& obb, int axisIdx);

// Padded radius for rotation rings so they visibly sit outside the faces.
float GetRotationRingRadius(const OBB3f& obb);

} // namespace BoxHandleInteraction
/*----------------------------------------------------------------*/


//  BoxRotationWidget - ImGui widgets for editing rotations/OBBs
//
// Each function displays its UI and returns true when the underlying value
// changed this frame. Rotations are exchanged as 3x3 float matrices; Euler
// angles are XYZ intrinsic in degrees (R = Rx * Ry * Rz when applied to a
// column vector).
//
// Gimbal-lock mitigation: EditEulerDeg caches its last Euler state in ImGui
// per-widget storage, so repeated drags near pitch = +-90 degrees do not
// cause the angles to drift. The cache is invalidated only when the
// externally supplied matrix differs from the one the widget last produced.
namespace BoxRotationWidget {

// Edit a 3x3 rotation through XYZ intrinsic Euler sliders in degrees.
bool EditEulerDeg(const char* label, Eigen::Matrix3f& rot);

// EditEulerDeg plus a collapsible read-only 3x3 matrix view.
bool EditMatrix(const char* label, Eigen::Matrix3f& rot);

// Edit an OBB's center, half-extents and rotation inline.
// dragSpeed = 0 derives a speed proportional to the current extents.
bool EditOBB(const char* label, OBB3f& obb, float dragSpeed = 0.0f);

} // namespace BoxRotationWidget
/*----------------------------------------------------------------*/


//  BoundingBoxEditController - mouse/keyboard state machine
//
// Turns viewport input into OBB edits via BoxHandleInteraction. Mirrors
// SelectionController in shape: owned by Window as unique_ptr, receives
// dispatched HandleMouse*/HandleKeyboard events when the current control
// mode is CONTROL_BBOX_EDIT.
//
// State machine:
//   STATE_IDLE        -> no hover, no drag
//   STATE_HOVER       -> a handle is under the cursor, waiting for click
//   STATE_DRAGGING    -> left button held, dragging the active handle
//
// Drag lifecycle:
//   * mouse-down over a handle: snapshot = working, hover fixed, state -> DRAGGING.
//   * mouse-move while dragging: recompute working via BoxHandleInteraction and
//     fire changeCallback(working) live so the viewport updates each frame.
//   * mouse-up: state -> HOVER (commit already happened live).
//   * Esc during drag: revert() restores snapshot via changeCallback.
//
// The change callback is expected to push the new OBB through
// Scene::SetBoundingBox so GPU buffers and redraws stay centralized.
class BoundingBoxEditController {
public:
	enum State {
		STATE_IDLE,
		STATE_HOVER,
		STATE_DRAGGING
	};

	explicit BoundingBoxEditController(Camera& camera);

	// OBB state
	void setOBB(const OBB3f& obb);
	const OBB3f& getOBB() const { return working; }

	void commit();    // fire callback with current working
	void revert();    // restore snapshot (used by Esc)

	State getState() const { return state; }
	bool isDragging() const { return state == STATE_DRAGGING; }

	// Hover query for renderer highlighting
	int getHoverCornerIdx() const;
	int getHoverFaceIdx() const;
	int getHoverAxisIdx() const;

	// Input dispatch (mirrors SelectionController API)
	void handleMouseMove(const Eigen::Vector2d& normalizedPos);
	void handleMouseButton(int button, int action, const Eigen::Vector2d& normalizedPos, int mods);
	void handleKeyboard(int key, int action, int mods);
	void update(double deltaTime);

	// Live callback fired every time 'working' is mutated (drag frames too).
	using ChangeCallback = std::function<void(const OBB3f&)>;
	void setChangeCallback(ChangeCallback cb) { changeCallback = std::move(cb); }

private:
	Camera& camera;
	OBB3f working;
	OBB3f snapshot;
	BoxHandleInteraction::Pick hover;
	Ray3d dragStartRay;
	State state;
	ChangeCallback changeCallback;

	Ray3d buildRay(const Eigen::Vector2d& normalizedPos) const;
	double computePickRadius() const;
	void fireChange() const;
};
/*----------------------------------------------------------------*/

} // namespace VIEWER
