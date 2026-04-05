////////////////////////////////////////////////////////////////////
// RelativePoseRefine.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _SFM_RELATIVE_POSE_REFINE_H_
#define _SFM_RELATIVE_POSE_REFINE_H_


// I N C L U D E S /////////////////////////////////////////////////


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SFM {

class SFM_API PinholeCamera;
class SFM_API Pose3D;
struct SFM_API DMatch;

class SFM_API RelativePoseRefine {
public:
	// Configuration for two-view calibration refinement
	struct Config {
		unsigned maxMatches{2000};   // maximum number of matches to use
		double robustThreshold{1.5}; // Huber loss threshold (pixels)
		int maxIterations{50};	     // maximum solver iterations
		bool refineFocalLength{true};// refine focal length
		bool verbose{false};         // print Ceres summary
	};

	// Result statistics
	struct Result {
		bool success{false};
		double initialCost{0.0};
		double finalCost{0.0};
	};

	// Refine shared pinhole intrinsics and relative pose (R, C) from two keyframes.
	// Parameterization inside Ceres:
	//   intr[5] = { f, k1, k2, cx, cy } with cx,cy constant via SubsetManifold
	//   pose[7] = { qw, qx, qy, qz, Cx, Cy, Cz } where quaternion maps world -> camera2; camera1 is identity at origin.
	static bool RefineTwoViewCalibration(
		const std::vector<cv::KeyPoint>& keypoints1,
		const std::vector<cv::KeyPoint>& keypoints2,
		const std::vector<DMatch>& matches,
		PinholeCamera& camera,
		Pose3D& relativePose,
		const Config& config,
		Result* result = nullptr);
};
/*----------------------------------------------------------------*/

} // namespace SFM

#endif // _SFM_RELATIVE_POSE_REFINE_H_
