////////////////////////////////////////////////////////////////////
// InterfaceMVS.h
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#ifndef _SFM_INTERFACEMVS_H_
#define _SFM_INTERFACEMVS_H_


// I N C L U D E S /////////////////////////////////////////////////

#include "Camera.h"


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SFM {

// forward declarations to avoid circular includes
class SFM_API Scene;


// Depth-map undistortion using depth-aware interpolation
bool UndistortDMAP(const String& depthMapFile,
	const cv::Mat& map1, const cv::Mat& map2, const KMatrix& imageUndistortedK);

// Batch undistort depth-maps at their native resolution, matching image undistortion.
// alpha should match the alpha used for image undistortion.
bool UndistortDepthMaps(const Scene& scene,
	const CLISTDEF2(String)& depthMapFiles,
	float alpha=0.6f,
	std::unordered_map<const Camera*, KMatrix>* undistortedIntrinsics=NULL);


// Depth-map import from MVS format (similar to MVS::ImportDepthDataRaw)
bool ImportDepthDataRaw(const String& fileName, String& imageFileName,
	IIndexArr& IDs, cv::Size& imageSize, cv::Size& depthSize,
	KMatrix& K, RMatrix& R, CMatrix& C,
	float& dMin, float& dMax,
	Image32F& depthMap, Image32F3& normalMap, Image32F& confMap, Image8U4& viewsMap, unsigned flags=15/*all*/);

// Depth-map export to MVS format (similar to MVS::ExportDepthDataRaw)
bool ExportDepthDataRaw(const String& fileName, const String& imageFileName,
	const IIndexArr& IDs, const cv::Size& imageSize,
	const KMatrix& K, const RMatrix& R, const Point3& C,
	float dMin, float dMax,
	const Image32F& depthMap, const Image32F& confMap, const Image8U4& viewsMap);


/**
 * @brief Import an MVS::Interface (.mvs) project file into the SfM scene
 * Populates cameras, images, poses, tracks and optional colors from the
 * serialized interface. Existing scene data is released prior to import.
 * @param fileName input .mvs file path
 * @param scene output SfM scene to populate
 * @param loadColors whether to import per-point colors when present
 * @return true on success
 */
bool ImportMVS(const String& fileName, Scene& scene, bool loadColors=true);

/**
 * @brief Export current SfM scene to an MVS::Interface (.mvs) project file.
 * All images are exported; images without pose keep poseID=NO_ID.
 * Intrinsics are always written in full pixel space (no normalization).
 * Optionally undistorts pinhole images (ignoring distortion thereafter).
 * @param fileName output .mvs file path
 * @param scene input SfM scene to export
 * @param undistortImageDir optional directory to store undistorted images (created if missing);
 *        if empty, do not perform undistortion and export undistorted image files
 * @param undistortAlpha alpha parameter for undistortion (0=zoomed in, 1=all pixels retained)
 * @param onlyInlierTracks export only tracks with numInliers>1 when true (otherwise >=2 observations)
 * @param includeColors export per-point color if available
 * @return true on success
 */
bool ExportMVS(const String& fileName, const Scene& scene,
	String undistortImageDir={},
	float undistortAlpha=0.6f,
	bool onlyInlierTracks=true,
	bool includeColors=true);
/*----------------------------------------------------------------*/

} // namespace SFM

#endif // _SFM_INTERFACEMVS_H_
