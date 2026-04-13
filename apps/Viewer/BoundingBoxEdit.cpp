/*
 * BoundingBoxEdit.cpp
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
#include "BoundingBoxEdit.h"

#include <imgui_internal.h>

namespace VIEWER {

// ===================================================================
//  BoxHandleInteraction implementation
// ===================================================================
namespace BoxHandleInteraction {

namespace {

// Intersect a ray with a sphere. Returns smallest positive hit distance, or -1 on miss.
double RaySphereHit(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                    const Eigen::Vector3d& center, double radius)
{
	const Eigen::Vector3d oc = ro - center;
	const double a = rd.dot(rd);
	const double b = 2.0 * oc.dot(rd);
	const double c = oc.dot(oc) - radius * radius;
	const double disc = b * b - 4.0 * a * c;
	if (disc < 0.0)
		return -1.0;
	const double sqrtDisc = std::sqrt(disc);
	const double t1 = (-b - sqrtDisc) / (2.0 * a);
	if (t1 > 1e-6) return t1;
	const double t2 = (-b + sqrtDisc) / (2.0 * a);
	if (t2 > 1e-6) return t2;
	return -1.0;
}

// Closest-to-ring distance: intersect the ray with the ring plane and measure
// how far the hit is from the ring radius. Returns the absolute distance
// (>= 0) on hit, -1 on miss or if outside the allowed thickness.
double RayRingDistance(const Eigen::Vector3d& ro, const Eigen::Vector3d& rd,
                       const Eigen::Vector3d& center, const Eigen::Vector3d& normal,
                       double radius, double thickness, double& outT)
{
	const double denom = rd.dot(normal);
	if (std::abs(denom) < 1e-9)
		return -1.0;
	const double t = (center - ro).dot(normal) / denom;
	if (t < 1e-6)
		return -1.0;
	const Eigen::Vector3d hit = ro + t * rd;
	const double planarDist = (hit - center).norm();
	const double diff = std::abs(planarDist - radius);
	if (diff > thickness)
		return -1.0;
	outT = t;
	return diff;
}

// Intersect a ray with a plane through 'planePoint' with normal 'planeNormal'.
bool IntersectRayPlane(const Ray3d& ray,
                       const Eigen::Vector3d& planePoint,
                       const Eigen::Vector3d& planeNormal,
                       Eigen::Vector3d& hit)
{
	const Eigen::Vector3d ro = ray.m_pOrig;
	const Eigen::Vector3d rd = ray.m_vDir.normalized();
	const double denom = rd.dot(planeNormal);
	if (std::abs(denom) < 1e-9)
		return false;
	const double t = (planePoint - ro).dot(planeNormal) / denom;
	if (t < 1e-6)
		return false;
	hit = ro + rd * t;
	return true;
}

} // anonymous namespace


void GetCornerWorldPositions(const OBB3f& obb, Eigen::Vector3f out[8]) {
	obb.GetCorners(out);
}

void GetFaceCenterWorldPositions(const OBB3f& obb, Eigen::Vector3f out[6]) {
	Eigen::Vector3f corners[8];
	obb.GetCorners(corners);
	for (int axis = 0; axis < 3; ++axis) {
		const int bit = 1 << axis;
		Eigen::Vector3f sumPlus  = Eigen::Vector3f::Zero();
		Eigen::Vector3f sumMinus = Eigen::Vector3f::Zero();
		for (int i = 0; i < 8; ++i) {
			if (i & bit) sumPlus  += corners[i];
			else         sumMinus += corners[i];
		}
		out[axis * 2 + 0] = sumPlus  * 0.25f; // +axis face
		out[axis * 2 + 1] = sumMinus * 0.25f; // -axis face
	}
}

Eigen::Vector3f GetLocalAxisWorldDir(const OBB3f& obb, int axisIdx) {
	// m_rot is world->local, so m_rot.row(k) is the k-th local axis in world coords.
	const Eigen::Matrix3f rot = obb.m_rot;
	return rot.row(axisIdx).normalized();
}

float GetRotationRingRadius(const OBB3f& obb) {
	const float extMax = obb.m_ext.maxCoeff();
	return extMax > 0.0f ? extMax * 1.15f : 1.0f;
}


Pick PickHandle(const OBB3f& obb, const Ray3d& ray,
                double handleRadiusWorld, double ringThicknessWorld)
{
	Pick best;
	best.distance = DBL_MAX;

	if (!obb.IsValid())
		return Pick{};

	const Eigen::Vector3d ro = ray.m_pOrig;
	Eigen::Vector3d rd = ray.m_vDir;
	const double rdNorm = rd.norm();
	if (rdNorm < 1e-9)
		return Pick{};
	rd /= rdNorm;

	// Corner spheres
	Eigen::Vector3f corners[8];
	obb.GetCorners(corners);
	for (int i = 0; i < 8; ++i) {
		const Eigen::Vector3d c = corners[i].cast<double>();
		const double t = RaySphereHit(ro, rd, c, handleRadiusWorld);
		if (t > 0.0 && t < best.distance) {
			best.kind = HANDLE_CORNER;
			best.index = i;
			best.distance = t;
		}
	}

	// Face-center spheres (slightly smaller so corners win ties)
	Eigen::Vector3f faceCenters[6];
	GetFaceCenterWorldPositions(obb, faceCenters);
	const double faceRadius = handleRadiusWorld * 0.75;
	for (int i = 0; i < 6; ++i) {
		const Eigen::Vector3d c = faceCenters[i].cast<double>();
		const double t = RaySphereHit(ro, rd, c, faceRadius);
		if (t > 0.0 && t < best.distance) {
			best.kind = HANDLE_FACE;
			best.index = i;
			best.distance = t;
		}
	}

	// Rotation rings, one per local axis
	const Eigen::Vector3d center = obb.m_pos.cast<double>();
	const double ringRadius = GetRotationRingRadius(obb);
	const double thickness = ringThicknessWorld > 0.0 ? ringThicknessWorld : handleRadiusWorld;
	for (int axis = 0; axis < 3; ++axis) {
		const Eigen::Vector3f axisDirF = GetLocalAxisWorldDir(obb, axis);
		const Eigen::Vector3d axisDir = axisDirF.cast<double>();
		double t;
		const double ringDist = RayRingDistance(ro, rd, center, axisDir, ringRadius, thickness, t);
		if (ringDist >= 0.0 && t < best.distance) {
			best.kind = axis == 0 ? HANDLE_ROT_X : (axis == 1 ? HANDLE_ROT_Y : HANDLE_ROT_Z);
			best.index = axis;
			best.distance = t;
		}
	}

	if (best.kind == HANDLE_NONE)
		return Pick{};
	return best;
}


OBB3f DragCorner(const OBB3f& original, int cornerIdx, const Ray3d& ray) {
	if (cornerIdx < 0 || cornerIdx > 7)
		return original;

	Eigen::Vector3f corners[8];
	original.GetCorners(corners);
	const Eigen::Vector3f anchor = corners[cornerIdx ^ 7];
	const Eigen::Vector3f originalCorner = corners[cornerIdx];

	// Drag plane at the original corner, facing the camera.
	const Eigen::Vector3d planePoint = originalCorner.cast<double>();
	Eigen::Vector3d planeNormal = ray.m_pOrig - planePoint;
	const double nNorm = planeNormal.norm();
	if (nNorm < 1e-9)
		return original;
	planeNormal /= nNorm;

	Eigen::Vector3d hit;
	if (!IntersectRayPlane(ray, planePoint, planeNormal, hit))
		return original;
	const Eigen::Vector3f newCornerWorld = hit.cast<float>();

	// New center = midpoint of anchor and new corner.
	const Eigen::Vector3f newCenter = (anchor + newCornerWorld) * 0.5f;

	// Decompose the world diagonal into the OBB's local frame:
	// m_rot * worldVec = localVec  (world->local).
	const Eigen::Matrix3f rotMat = original.m_rot;
	const Eigen::Vector3f worldDiag = newCornerWorld - newCenter;
	const Eigen::Vector3f localDiag = rotMat * worldDiag;
	Eigen::Vector3f newExt = localDiag.cwiseAbs();

	// Clamp so the box never collapses to degenerate extents.
	const float minExt = std::max(original.m_ext.maxCoeff() * 1e-4f, 1e-6f);
	newExt = newExt.cwiseMax(minExt);

	OBB3f result = original;
	result.m_pos = newCenter;
	result.m_ext = newExt;
	return result;
}


OBB3f DragFace(const OBB3f& original, int faceIdx, const Ray3d& ray) {
	if (faceIdx < 0 || faceIdx > 5)
		return original;
	const int axis = faceIdx / 2;
	const bool isPlus = (faceIdx % 2) == 0;

	Eigen::Vector3f faceCenters[6];
	GetFaceCenterWorldPositions(original, faceCenters);
	const Eigen::Vector3f faceCenter = faceCenters[faceIdx];
	const Eigen::Vector3f oppositeFaceCenter = faceCenters[axis * 2 + (isPlus ? 1 : 0)];

	// Drag plane at faceCenter facing the camera.
	const Eigen::Vector3d planePoint = faceCenter.cast<double>();
	Eigen::Vector3d planeNormal = ray.m_pOrig - planePoint;
	const double nNorm = planeNormal.norm();
	if (nNorm < 1e-9)
		return original;
	planeNormal /= nNorm;

	Eigen::Vector3d hit;
	if (!IntersectRayPlane(ray, planePoint, planeNormal, hit))
		return original;
	const Eigen::Vector3f newFaceCenterWorld = hit.cast<float>();

	// Project displacement along the axis (sideways components discarded).
	const Eigen::Vector3f fromOppositeToFace = faceCenter - oppositeFaceCenter;
	const float origLength = fromOppositeToFace.norm();
	if (origLength < 1e-9f)
		return original;
	const Eigen::Vector3f axisDir = fromOppositeToFace / origLength;

	const float newLength = (newFaceCenterWorld - oppositeFaceCenter).dot(axisDir);
	const float minLen = std::max(origLength * 2e-4f, 1e-6f);
	const float clampedLength = std::max(newLength, minLen);

	const float newHalfExtent = clampedLength * 0.5f;
	const Eigen::Vector3f newCenter = oppositeFaceCenter + axisDir * (clampedLength * 0.5f);

	OBB3f result = original;
	result.m_pos = newCenter;
	result.m_ext[axis] = newHalfExtent;
	return result;
}


OBB3f DragRotation(const OBB3f& original, int axisIdx,
                   const Ray3d& startRay, const Ray3d& currentRay)
{
	if (axisIdx < 0 || axisIdx > 2)
		return original;

	const Eigen::Vector3f axisDirF = GetLocalAxisWorldDir(original, axisIdx);
	const Eigen::Vector3d axisDir = axisDirF.cast<double>();
	const Eigen::Vector3d center = original.m_pos.cast<double>();

	Eigen::Vector3d startHit, currentHit;
	if (!IntersectRayPlane(startRay, center, axisDir, startHit))
		return original;
	if (!IntersectRayPlane(currentRay, center, axisDir, currentHit))
		return original;

	const Eigen::Vector3d v0 = startHit - center;
	const Eigen::Vector3d v1 = currentHit - center;
	const double n0 = v0.norm();
	const double n1 = v1.norm();
	if (n0 < 1e-9 || n1 < 1e-9)
		return original;
	const Eigen::Vector3d u0 = v0 / n0;
	const Eigen::Vector3d u1 = v1 / n1;

	// Signed angle around axisDir: sinA = (u0 x u1) . axis, cosA = u0 . u1.
	const Eigen::Vector3d crossV = u0.cross(u1);
	const double sinA = crossV.dot(axisDir);
	const double cosA = u0.dot(u1);
	const double angle = std::atan2(sinA, cosA);

	// m_rot is world->local, so for a world-frame rotation R:
	// new_m_rot = old_m_rot * R^T.
	const Eigen::Matrix3f R = Eigen::AngleAxisf(static_cast<float>(angle),
	                                            axisDirF.normalized()).toRotationMatrix();
	const Eigen::Matrix3f oldRot = original.m_rot;
	const Eigen::Matrix3f newRot = oldRot * R.transpose();

	OBB3f result = original;
	result.m_rot = newRot;
	return result;
}

} // namespace BoxHandleInteraction


// ===================================================================
//  BoxRotationWidget implementation
// ===================================================================
namespace BoxRotationWidget {

namespace {

// Local pi constant - Common.h may or may not define M_PI depending on platform.
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kRadToDeg = 180.0f / kPi;

// ImGui per-widget storage offsets for the Euler cache.
constexpr ImGuiID OFFSET_EULER_X = 0;
constexpr ImGuiID OFFSET_EULER_Y = 1;
constexpr ImGuiID OFFSET_EULER_Z = 2;
constexpr ImGuiID OFFSET_VALID  = 3;

// Build XYZ intrinsic rotation R = Rx * Ry * Rz from Euler angles (degrees).
Eigen::Matrix3f EulerDegToMatrix(float degX, float degY, float degZ) {
	const Eigen::AngleAxisf rx(degX * kDegToRad, Eigen::Vector3f::UnitX());
	const Eigen::AngleAxisf ry(degY * kDegToRad, Eigen::Vector3f::UnitY());
	const Eigen::AngleAxisf rz(degZ * kDegToRad, Eigen::Vector3f::UnitZ());
	return (rx * ry * rz).toRotationMatrix();
}

// Decompose XYZ intrinsic Euler from a rotation matrix (returns degrees).
Eigen::Vector3f MatrixToEulerDeg(const Eigen::Matrix3f& rot) {
	const float sy = rot(0, 2);
	float a, b, c;
	if (std::abs(sy) < 0.9999f) {
		b = std::asin(sy);
		a = std::atan2(-rot(1, 2), rot(2, 2));
		c = std::atan2(-rot(0, 1), rot(0, 0));
	} else {
		b = (sy > 0.0f) ? (kPi * 0.5f) : (-kPi * 0.5f);
		a = std::atan2(rot(2, 1), rot(1, 1));
		c = 0.0f;
	}
	return Eigen::Vector3f(a * kRadToDeg, b * kRadToDeg, c * kRadToDeg);
}

bool RotClose(const Eigen::Matrix3f& a, const Eigen::Matrix3f& b, float eps = 1e-4f) {
	return (a - b).squaredNorm() < eps * eps;
}

} // anonymous namespace


bool EditEulerDeg(const char* label, Eigen::Matrix3f& rot) {
	ImGuiStorage* storage = ImGui::GetStateStorage();
	const ImGuiID baseId = ImGui::GetID(label);

	float eulerDeg[3] = {
		storage->GetFloat(baseId + OFFSET_EULER_X, 0.0f),
		storage->GetFloat(baseId + OFFSET_EULER_Y, 0.0f),
		storage->GetFloat(baseId + OFFSET_EULER_Z, 0.0f),
	};
	const bool hasCache = storage->GetBool(baseId + OFFSET_VALID, false);

	bool needDecompose = !hasCache;
	if (hasCache) {
		const Eigen::Matrix3f reconstructed = EulerDegToMatrix(eulerDeg[0], eulerDeg[1], eulerDeg[2]);
		if (!RotClose(reconstructed, rot))
			needDecompose = true;
	}
	if (needDecompose) {
		const Eigen::Vector3f fresh = MatrixToEulerDeg(rot);
		eulerDeg[0] = fresh[0];
		eulerDeg[1] = fresh[1];
		eulerDeg[2] = fresh[2];
	}

	const bool changed = ImGui::DragFloat3(label, eulerDeg, 0.5f, -360.0f, 360.0f, "%.2f deg");

	storage->SetFloat(baseId + OFFSET_EULER_X, eulerDeg[0]);
	storage->SetFloat(baseId + OFFSET_EULER_Y, eulerDeg[1]);
	storage->SetFloat(baseId + OFFSET_EULER_Z, eulerDeg[2]);
	storage->SetBool(baseId + OFFSET_VALID, true);

	if (changed) {
		rot = EulerDegToMatrix(eulerDeg[0], eulerDeg[1], eulerDeg[2]);
		return true;
	}
	return false;
}


bool EditMatrix(const char* label, Eigen::Matrix3f& rot) {
	ImGui::PushID(label);
	const bool changed = EditEulerDeg(label, rot);
	if (ImGui::CollapsingHeader("Matrix (read-only)")) {
		ImGui::Text("[ %8.4f  %8.4f  %8.4f ]", rot(0, 0), rot(0, 1), rot(0, 2));
		ImGui::Text("[ %8.4f  %8.4f  %8.4f ]", rot(1, 0), rot(1, 1), rot(1, 2));
		ImGui::Text("[ %8.4f  %8.4f  %8.4f ]", rot(2, 0), rot(2, 1), rot(2, 2));
	}
	ImGui::PopID();
	return changed;
}


bool EditOBB(const char* label, OBB3f& obb, float dragSpeed) {
	ImGui::PushID(label);
	bool changed = false;

	float speed = dragSpeed;
	if (speed <= 0.0f) {
		const float extMax = obb.m_ext.maxCoeff();
		speed = extMax > 0.0f ? extMax * 0.01f : 0.01f;
	}

	// OBB3f::POINT is Eigen::Vector3f (column-major), data() is safe.
	if (ImGui::DragFloat3("Center", obb.m_pos.data(), speed, 0.0f, 0.0f, "%.4f"))
		changed = true;

	if (ImGui::DragFloat3("Half-Extents", obb.m_ext.data(), speed, 0.0f, FLT_MAX, "%.4f"))
		changed = true;

	// OBB3f::MATRIX is row-major; copy into a column-major Matrix3f at the
	// API boundary (Eigen handles the storage-order conversion).
	Eigen::Matrix3f rot = obb.m_rot;
	if (EditEulerDeg("Rotation XYZ (deg)", rot)) {
		obb.m_rot = rot;
		changed = true;
	}

	ImGui::PopID();
	return changed;
}

} // namespace BoxRotationWidget


// ===================================================================
//  BoundingBoxEditController implementation
// ===================================================================

BoundingBoxEditController::BoundingBoxEditController(Camera& cam)
	: camera(cam)
	, working(true)       // zero-extent until setOBB()
	, snapshot(true)
	, state(STATE_IDLE)
{
	hover.kind = BoxHandleInteraction::HANDLE_NONE;
	hover.index = -1;
	hover.distance = 0.0;
}

void BoundingBoxEditController::setOBB(const OBB3f& obb) {
	working = obb;
	snapshot = obb;
	state = STATE_IDLE;
	hover.kind = BoxHandleInteraction::HANDLE_NONE;
	hover.index = -1;
}

void BoundingBoxEditController::commit() {
	snapshot = working;
	fireChange();
}

void BoundingBoxEditController::revert() {
	working = snapshot;
	state = STATE_IDLE;
	hover.kind = BoxHandleInteraction::HANDLE_NONE;
	hover.index = -1;
	fireChange();
}

int BoundingBoxEditController::getHoverCornerIdx() const {
	return hover.kind == BoxHandleInteraction::HANDLE_CORNER ? hover.index : -1;
}

int BoundingBoxEditController::getHoverFaceIdx() const {
	return hover.kind == BoxHandleInteraction::HANDLE_FACE ? hover.index : -1;
}

int BoundingBoxEditController::getHoverAxisIdx() const {
	if (hover.kind == BoxHandleInteraction::HANDLE_ROT_X) return 0;
	if (hover.kind == BoxHandleInteraction::HANDLE_ROT_Y) return 1;
	if (hover.kind == BoxHandleInteraction::HANDLE_ROT_Z) return 2;
	return -1;
}

Ray3d BoundingBoxEditController::buildRay(const Eigen::Vector2d& normalizedPos) const {
	return camera.GetPickingRay(normalizedPos);
}

double BoundingBoxEditController::computePickRadius() const {
	// Scale by camera-to-OBB distance so the click target feels constant in
	// screen space. ~1.8% of distance maps to ~14 pixels at 60deg FOV / 800 px.
	const Eigen::Vector3d eye = camera.GetPosition();
	const Eigen::Vector3d center = working.m_pos.cast<double>();
	const double dist = (eye - center).norm();
	return std::max(dist * 0.018, 1e-4);
}

void BoundingBoxEditController::fireChange() const {
	if (changeCallback)
		changeCallback(working);
}

void BoundingBoxEditController::handleMouseMove(const Eigen::Vector2d& normalizedPos) {
	if (!working.IsValid())
		return;

	const Ray3d ray = buildRay(normalizedPos);

	if (state == STATE_DRAGGING) {
		OBB3f next = working;
		switch (hover.kind) {
		case BoxHandleInteraction::HANDLE_CORNER:
			next = BoxHandleInteraction::DragCorner(snapshot, hover.index, ray);
			break;
		case BoxHandleInteraction::HANDLE_FACE:
			next = BoxHandleInteraction::DragFace(snapshot, hover.index, ray);
			break;
		case BoxHandleInteraction::HANDLE_ROT_X:
		case BoxHandleInteraction::HANDLE_ROT_Y:
		case BoxHandleInteraction::HANDLE_ROT_Z: {
			const int axis = hover.kind == BoxHandleInteraction::HANDLE_ROT_X ? 0
			               : hover.kind == BoxHandleInteraction::HANDLE_ROT_Y ? 1 : 2;
			next = BoxHandleInteraction::DragRotation(snapshot, axis, dragStartRay, ray);
			break;
		}
		default:
			return;
		}
		working = next;
		fireChange();
		return;
	}

	// Not dragging: update hover state.
	const BoxHandleInteraction::Pick newHover =
		BoxHandleInteraction::PickHandle(working, ray, computePickRadius());
	if (newHover.valid()) {
		hover = newHover;
		state = STATE_HOVER;
	} else {
		hover.kind = BoxHandleInteraction::HANDLE_NONE;
		hover.index = -1;
		state = STATE_IDLE;
	}
}

void BoundingBoxEditController::handleMouseButton(int button, int action,
                                                  const Eigen::Vector2d& normalizedPos,
                                                  int /*mods*/)
{
	if (button != GLFW_MOUSE_BUTTON_LEFT)
		return;
	if (!working.IsValid())
		return;

	if (action == GLFW_PRESS) {
		const Ray3d ray = buildRay(normalizedPos);
		// Re-pick at press time in case hover was stale.
		const BoxHandleInteraction::Pick pickNow =
			BoxHandleInteraction::PickHandle(working, ray, computePickRadius());
		if (!pickNow.valid())
			return;
		hover = pickNow;
		snapshot = working;
		dragStartRay = ray;
		state = STATE_DRAGGING;
	} else if (action == GLFW_RELEASE) {
		if (state == STATE_DRAGGING) {
			snapshot = working;
			state = STATE_HOVER;
		}
	}
}

void BoundingBoxEditController::handleKeyboard(int key, int action, int /*mods*/) {
	if (action != GLFW_PRESS)
		return;
	if (key == GLFW_KEY_ESCAPE) {
		if (state == STATE_DRAGGING)
			revert();
	}
}

void BoundingBoxEditController::update(double /*deltaTime*/) {
	// No continuous updates: all logic is event-driven.
}

} // namespace VIEWER
