////////////////////////////////////////////////////////////////////
// MatchGeometric.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "MatchGeometric.h"
#include "Image.h"
#include "ImagePair.h"
// PoseLib for robust relative/fundamental estimation
#include <PoseLib/poselib.h>

using namespace SFM;


bool SFM::MatchFeaturesGeometric(
	PairsMatcher& pairsMatcher,
	const Image& img1,
	const Image& img2,
	const std::vector<Point2f>& trackedPoints1,
	const std::vector<Point2f>& trackedPoints2,
	const std::vector<uchar>& trackStatus,
	ImagePair& pair,
	float epipolarThreshold)
{
	// Sanity check: keypoints1 correspond to trackedPoints1 by index
	ASSERT(img1.keypoints.size() == trackedPoints1.size());
	ASSERT(trackedPoints1.size() == trackedPoints2.size());
	ASSERT(trackStatus.size() == trackedPoints1.size());

	pair.Reset();

	// Step 1: Estimate relative pose / F from tracked points
	// Initialize pair with tracked points as initial matches
	for (size_t i = 0; i < trackStatus.size(); ++i)
		if (trackStatus[i])
			pair.matches.emplace_back((uint32_t)i, (uint32_t)i);
	if (pair.matches.size() < pairsMatcher.GetConfig().minMatches) {
		DEBUG("MatchFeaturesGeometric: insufficient tracked points (%zu) for F-matrix estimation", pair.matches.size());
		// Fallback to descriptor-only matching
		pairsMatcher.MatchFeatures(img1.descriptors, img2.descriptors, pair.matches);
		return false;
	}
	{
		// Make copies to avoid modifying original images
		Image img1Copy(img1.ID, img1.fileName, reinterpret_cast<const Pose3D&>(img1), img1.cameraID, img1.pCamera);
		Image img2Copy(img2.ID, img2.fileName, reinterpret_cast<const Pose3D&>(img2), img2.cameraID, img2.pCamera);
		img1Copy.keypoints = ConvertToKeypoints(trackedPoints1);
		img2Copy.keypoints = ConvertToKeypoints(trackedPoints2);
		// Use GeometricFilter to estimate geometry from tracked points
		if (!pairsMatcher.GeometricFilter(img1Copy, img2Copy, pair)) {
			DEBUG("MatchFeaturesGeometric: GeometricFilter failed, falling back to descriptor-only matching");
			pair.matches.clear();
			pairsMatcher.MatchFeatures(img1.descriptors, img2.descriptors, pair.matches);
			return false;
		}
	}
	if (pair.GetNumFilteredInliers() < pairsMatcher.GetConfig().minMatches) {
		DEBUG("MatchFeaturesGeometric: PoseLib estimation failed, falling back to descriptor-only matching");
		pair.ResetMatches();
		pairsMatcher.MatchFeatures(img1.descriptors, img2.descriptors, pair.matches);
		return false;
	}
	pair.ResetMatches();

	// Step 2: Match descriptors with epipolar and ratio constraints
	// For each feature in image1, find matches satisfying both geometric and descriptor constraints.
	// We further restrict the search to a spatial neighborhood around the tracked point in image2
	// (trackedPoints2[i]) to avoid scanning the entire epipolar line.
	// Choose a reasonable spatial radius: at least a few pixels, scaled from epipolarThreshold
	const float spatialThreshold = MAXF(10.f, epipolarThreshold * 6.f);

	// Build a 2D octree over keypoints2 for fast spatial neighbor queries around trackedPoints2.
	typedef CLISTDEF0(Point2f::EVec) Point2fs;
	Point2fs kpts2(img2.keypoints.size());
	FOREACH(i, img2.keypoints) {
		const cv::KeyPoint& keypoint = img2.keypoints[i];
		kpts2[i] = Point2f::EVec(keypoint.pt.x, keypoint.pt.y);
	}
	typedef TOctree<Point2fs, float, 2> Octree2f;
	Octree2f octree(kpts2, [](Octree2f::IDX_TYPE n, Octree2f::Type r) { return n > 16 && r > 8.f; });

	// For each keypoint in image1, find candidates in image2
	const Matrix3x3f F = pair.F.value();
	const float matchRatio = pairsMatcher.GetConfig().matchRatio;
	const int normType = pairsMatcher.GetConfig().descriptorsAreBinary ? cv::NORM_HAMMING : cv::NORM_L2;
	FOREACH(i, img1.keypoints) {
		const Point2f& pt1 = img1.keypoints[i].pt;

		// Compute epipolar line in image2: L = F * pt1
		const Point3f line = F * pt1.homogeneous();
		const float normFactor = SQRT(line.x*line.x + line.y*line.y);
		if (normFactor < FZERO_TOLERANCE)
			continue;

		// Find candidate matches near the epipolar line AND (if tracked) close to expectedPt2
		std::vector<cv::DMatch> candidates;
		if (trackStatus[i]) {
			const Point2f& expectedPt2 = trackedPoints2[i];
			Octree2f::IDXARR_TYPE neighbors;
			octree.Collect(neighbors, expectedPt2, spatialThreshold);
			if (neighbors.empty())
				goto BruteForceFallback;
			for (const Octree2f::IDX_TYPE idx : neighbors) {
				const cv::Point2f& pt2 = img2.keypoints[idx].pt;
				const float distance = ABS(line.x * pt2.x + line.y * pt2.y + line.z) / normFactor;
				if (distance < epipolarThreshold)
					candidates.emplace_back((int)i, (int)idx, 0.f);
			}
		} else {
			BruteForceFallback:
			// fallback: scan all keypoints2 and use only epipolar constraint
			for (size_t j = 0; j < img2.keypoints.size(); ++j) {
				const cv::Point2f& pt2 = img2.keypoints[j].pt;
				const float distance = ABS(line.x * pt2.x + line.y * pt2.y + line.z) / normFactor;
				if (distance < epipolarThreshold)
					candidates.emplace_back((int)i, (int)j, 0.f);
			}
		}
		if (candidates.empty())
			continue;

		if (candidates.size() == 1) {
			// Only one candidate, accept it
			pair.matches.push_back(candidates[0]);
		} else {
			// Compute descriptor distances for candidates
			cv::Mat desc1 = img1.descriptors.row((int)i);
			for (auto& candidate : candidates) {
				cv::Mat desc2 = img2.descriptors.row(candidate.trainIdx);
				// Compute Hamming distance for binary descriptors, L2 otherwise
				candidate.distance = (float)cv::norm(desc1, desc2, normType);
			}
			// Sort candidates by descriptor distance
			std::sort(candidates.begin(), candidates.end(),
				[](const cv::DMatch& a, const cv::DMatch& b) {
					return a.distance < b.distance;
				});
			// Apply ratio test on candidates
			if (candidates[0].distance < matchRatio * candidates[1].distance)
				pair.matches.push_back(candidates[0]);
		}
	}
	if (pair.matches.size() < pairsMatcher.GetConfig().minMatches) {
		pair.InvalidateMatches();
		return false;
	}
	if (pairsMatcher.GetConfig().IsMatchesFilterOn()) {
		// Further filter matches based on triangulation angle, reprojection error, epipole proximity
		const unsigned numFilteredInliers = pair.FilterMatches(img1, img2, pairsMatcher.GetConfig().minTriangulationAngle, pairsMatcher.GetConfig().reprojThreshold, pairsMatcher.GetConfig().epipoleFilterThreshold);
		if (numFilteredInliers < pairsMatcher.GetConfig().minMatches) {
			pair.InvalidateMatches();
			return false;
		}
	}
	return true;
}
/*----------------------------------------------------------------*/
