////////////////////////////////////////////////////////////////////
// InterfaceMVS.cpp
//
// Copyright 2007 cDc@seacave
// Distributed under the Boost Software License, Version 1.0
// (See http://www.boost.org/LICENSE_1_0.txt)

#include "Common.h"
#include "InterfaceMVS.h"
#include "Scene.h"

// Import/Export scene and depth-maps to MVS and DMAP Interface format respectively
#ifndef _USE_OPENCV
#define _USE_OPENCV
#endif
#include "../MVS/Interface.h"

using namespace SFM;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define INTERFACEMVS_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

bool SFM::UndistortDMAP(const String& depthMapFile,
	const cv::Mat& map1, const cv::Mat& map2, const KMatrix& imageUndistortedK)
{
	// Import depth-map
	String imageFileName;
	IIndexArr IDs;
	cv::Size imageSize, depthSize;
	KMatrix K;
	RMatrix R;
	CMatrix C;
	float dMin, dMax;
	Image32F depthMap, confMap;
	Image32F3 normalMap;
	Image8U4 viewsMap;
	if (!ImportDepthDataRaw(depthMapFile, imageFileName, IDs, imageSize, depthSize, K, R, C, dMin, dMax,
			depthMap, normalMap, confMap, viewsMap)) {
		DEBUG("warning: failed to import depth-map from '%s'", depthMapFile.c_str());
		return false;
	}
	// Undistort depth-map using depth-aware interpolation at its native resolution
	Image32F undistortedDepthMap(depthMap.size(), 0.f);
	if (!normalMap.empty()) {
		DEBUG("warning: undistortion of normal-maps not implemented yet, skipping");
	}
	// For each pixel in the undistorted depth-map, find the corresponding pixel in the distorted depth-map
	for (int y = 0; y < undistortedDepthMap.rows; ++y) {
		for (int x = 0; x < undistortedDepthMap.cols; ++x) {
			// Get the corresponding distorted pixel coordinates from the remap maps
			// Convert from fixed-point to float (map1: integer part, map2: fractional part)
			const Point2f srcPt = CoordinateRemap2Float(cv::Point2i(x, y), map1, map2);
			// Check if source point is within bounds
			if (!depthMap.isInsideWithBorder(srcPt, 1))
				continue;
			// Sample depth using depth-aware interpolation
			// Only interpolate between pixels with similar depths
			const Point2i centerPt = ROUND2INT(srcPt);
			const float centerDepth = depthMap(centerPt);
			if (centerDepth <= 0.f)
				continue;
			const auto Sample = [centerDepth](const float& depth) {
				return IsDepthSimilar(centerDepth, depth);
			};
			// Sample with a functor that checks depth similarity
			float sampledDepth;
			if (!depthMap.sampleSafe(sampledDepth, srcPt, Sample))
				continue;
			undistortedDepthMap(y, x) = sampledDepth;
		}
	}
	// Undistort confidence and views-map if available (using nearest neighbor interpolation)
	Image32F undistortedConfMap;
	if (!confMap.empty())
		cv::remap(confMap, undistortedConfMap, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	Image8U4 undistortedViewsMap;
	if (!viewsMap.empty())
		cv::remap(viewsMap, undistortedViewsMap, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	// Export undistorted depth-map to a temporary file first
	const String tempDepthMapFile = depthMapFile + ".tmp";
	if (!ExportDepthDataRaw(tempDepthMapFile, imageFileName, IDs, imageSize,
			imageUndistortedK, R, C, dMin, dMax,
			undistortedDepthMap, undistortedConfMap, undistortedViewsMap)) {
		DEBUG("warning: failed to export undistorted depth-map to '%s'", tempDepthMapFile.c_str());
		return false;
	}
	// Rename temporary file to the original file
	if (!File::renameFile(tempDepthMapFile, depthMapFile)) {
		DEBUG("warning: failed to rename '%s' to '%s'", tempDepthMapFile.c_str(), depthMapFile.c_str());
		// Try to remove the temporary file
		File::deleteFile(tempDepthMapFile);
	}
	return true;
}

bool SFM::UndistortDepthMaps(const Scene& scene,
	const CLISTDEF2(String)& depthMapFiles,
	float alpha,
	std::unordered_map<const Camera*, KMatrix>* undistortedIntrinsics)
{
	ASSERT(!depthMapFiles.empty());
	struct UndistortData
	{
		cv::Mat map1;
		cv::Mat map2;
		KMatrix newK;
	};

	TD_TIMER_STARTD();
	std::unordered_map<const Camera*, UndistortData> undistortMaps;
	undistortMaps.reserve(scene.cameras.size());

	// For each image/depth-map, compute maps at the depth resolution using K scaled to that resolution
	#ifdef INTERFACEMVS_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (int_t _i = 0; _i < (int_t)scene.images.size(); ++_i) {
		const IIndex i = static_cast<IIndex>(_i);
		const String& dfile = depthMapFiles[i];
		if (dfile.empty())
			continue;
		const Image& img = scene.images[i];
		if (!img.IsValid())
			continue;
		const Camera* cam = img.pCamera;
		if (cam->GetType() != CameraType::PINHOLE)
			continue; // only pinhole supported
		const PinholeCamera* pc = static_cast<const PinholeCamera*>(cam);
		if (!pc->HasDistortion())
			continue; // nothing to undistort for depth-map
		// Check if undistortion maps already computed for this camera
		UndistortData& data = [&]() -> UndistortData& {
			UndistortData* pData = nullptr;
			#ifdef INTERFACEMVS_USE_OPENMP
			#pragma omp critical
			#endif
			{
				pData = &undistortMaps[pc];
				if (pData->map1.empty() || pData->map2.empty()) {
					// Read only header to obtain image size and depth-map resolution
					String imageFileName;
					IIndexArr IDs;
					cv::Size imageSize, depthSize;
					KMatrix Ki; RMatrix R; CMatrix C;
					float dMin=0, dMax=0; // unused here
					Image32F depthMapStub, confStub; Image32F3 normalStub; Image8U4 viewsStub;
					// flags=0 → header + meta only
					if (!ImportDepthDataRaw(dfile, imageFileName, IDs, imageSize, depthSize, Ki, R, C, dMin, dMax,
							depthMapStub, normalStub, confStub, viewsStub, 0)) {
						DEBUG("warning: failed to read depth header '%s'", dfile.c_str());
						exit(EXIT_FAILURE);
					}
					// Build K at depth resolution by scaling the image K
					const KMatrix Kdepth = ScaleK(pc->GetK(), imageSize, depthSize);
					const cv::Mat distCoeffs = pc->GetDistortionCoeffs();
					pData->newK = cv::getOptimalNewCameraMatrix(Kdepth, distCoeffs, depthSize, alpha);
					cv::initUndistortRectifyMap(Kdepth, distCoeffs, cv::noArray(), pData->newK, depthSize, CV_16SC2, pData->map1, pData->map2);
					if (undistortedIntrinsics)
						undistortedIntrinsics->emplace(pc, pData->newK);
				}
			}
			return *pData;
		}();
		// Undistort DMAP with depth-aware interpolation and write back
		if (!UndistortDMAP(dfile, data.map1, data.map2, img.GetK())) {
			DEBUG("warning: undistort depth failed for '%s'", dfile.c_str());
			continue;
		}
		DEBUG_EXTRA("Depth-map undistorted for image %d", i);
	}
	DEBUG("Depth-maps undistorted successfully in %s", TD_TIMER_GET_FMT().c_str());
	return true;
}
/*----------------------------------------------------------------*/


bool SFM::ImportDepthDataRaw(const String& fileName, String& imageFileName,
	IIndexArr& IDs, cv::Size& imageSize, cv::Size& depthSize,
	KMatrix& K, RMatrix& R, CMatrix& C,
	float& dMin, float& dMax,
	Image32F& depthMap, Image32F3& normalMap, Image32F& confMap, Image8U4& viewsMap, unsigned flags)
{
	std::unique_ptr<FILE, decltype(&fclose)> f(fopen(fileName, "rb"), &fclose);
	if (!f) {
		DEBUG("error: opening file '%s' for reading depth-data", fileName.c_str());
		return false;
	}

	// read header
	MVS::HeaderDepthDataRaw header;
	if (fread(&header, sizeof(MVS::HeaderDepthDataRaw), 1, f.get()) != 1 ||
		header.name != MVS::HeaderDepthDataRaw::HeaderDepthDataRawName() ||
		(header.type & MVS::HeaderDepthDataRaw::HAS_DEPTH) == 0 ||
		header.depthWidth <= 0 || header.depthHeight <= 0 ||
		header.imageWidth < header.depthWidth || header.imageHeight < header.depthHeight)
	{
		DEBUG("error: invalid depth-data file '%s'", fileName.c_str());
		return false;
	}

	// read image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	uint16_t nFileNameSize;
	fread(&nFileNameSize, sizeof(uint16_t), 1, f.get());
	imageFileName.resize(nFileNameSize);
	fread(imageFileName.data(), sizeof(char), nFileNameSize, f.get());

	// read neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	uint32_t nIDs;
	fread(&nIDs, sizeof(IIndex), 1, f.get());
	ASSERT(nIDs > 0 && nIDs < 256);
	IDs.resize(nIDs);
	fread(IDs.data(), sizeof(IIndex), nIDs, f.get());

	// read pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fread(K.val, sizeof(REAL), 9, f.get());
	fread(R.val, sizeof(REAL), 9, f.get());
	fread(C.ptr(), sizeof(REAL), 3, f.get());

	// parse sizes
	dMin = header.dMin;
	dMax = header.dMax;
	imageSize.width = header.imageWidth;
	imageSize.height = header.imageHeight;
	depthSize.width = header.depthWidth;
	depthSize.height = header.depthHeight;
	if (flags == 0)
		return ferror(f.get()) == 0; // only header + meta requested

	// read depth-map
	if ((flags & MVS::HeaderDepthDataRaw::HAS_DEPTH) != 0) {
		depthMap.create(header.depthHeight, header.depthWidth);
		if (fread(depthMap.getData(), sizeof(float), depthMap.area(), f.get()) != static_cast<size_t>(depthMap.area())) {
			DEBUG("error: reading depth-data from file '%s'", fileName.c_str());
			return false;
		}
	} else {
		fseek(f.get(), sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
	}

	// read normal-map
	if ((header.type & MVS::HeaderDepthDataRaw::HAS_NORMAL) != 0) {
		if ((flags & MVS::HeaderDepthDataRaw::HAS_NORMAL) != 0) {
			normalMap.create(header.depthHeight, header.depthWidth);
			fread(normalMap.getData(), sizeof(float)*3, normalMap.area(), f.get());
		} else {
			fseek(f.get(), sizeof(float)*3*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read confidence-map
	if ((header.type & MVS::HeaderDepthDataRaw::HAS_CONF) != 0) {
		if ((flags & MVS::HeaderDepthDataRaw::HAS_CONF) != 0) {
			confMap.create(header.depthHeight, header.depthWidth);
			fread(confMap.getData(), sizeof(float), confMap.area(), f.get());
		} else {
			fseek(f.get(), sizeof(float)*header.depthWidth*header.depthHeight, SEEK_CUR);
		}
	}

	// read visibility-map
	if ((header.type & MVS::HeaderDepthDataRaw::HAS_VIEWS) != 0) {
		if ((flags & MVS::HeaderDepthDataRaw::HAS_VIEWS) != 0) {
			viewsMap.create(header.depthHeight, header.depthWidth);
			fread(viewsMap.getData(), sizeof(uint8_t)*4, viewsMap.area(), f.get());
		}
	}

	return ferror(f.get()) == 0;
} // ImportDepthDataRaw

bool SFM::ExportDepthDataRaw(const String& fileName, const String& imageFileName,
	const IIndexArr& IDs, const cv::Size& imageSize,
	const KMatrix& K, const RMatrix& R, const Point3& C,
	float dMin, float dMax,
	const Image32F& depthMap, const Image32F& confMap, const Image8U4& viewsMap)
{
	ASSERT(!IDs.empty() && IDs.size() < 256);
	ASSERT(!depthMap.empty());
	ASSERT(confMap.empty() || depthMap.size() == confMap.size());
	ASSERT(viewsMap.empty() || depthMap.size() == viewsMap.size());
	ASSERT(depthMap.width() <= (int)imageSize.width && depthMap.height() <= (int)imageSize.height);

	std::unique_ptr<FILE, decltype(&fclose)> f(fopen(fileName, "wb"), &fclose);
	if (!f) {
		DEBUG("error: opening file '%s' for writing depth-data", fileName.c_str());
		return false;
	}

	// Write header
	MVS::HeaderDepthDataRaw header;
	header.name = MVS::HeaderDepthDataRaw::HeaderDepthDataRawName();
	header.type = MVS::HeaderDepthDataRaw::HAS_DEPTH;
	header.imageWidth = (uint32_t)imageSize.width;
	header.imageHeight = (uint32_t)imageSize.height;
	header.depthWidth = (uint32_t)depthMap.cols;
	header.depthHeight = (uint32_t)depthMap.rows;
	header.dMin = dMin;
	header.dMax = dMax;
	header.padding = 0;
	if (!confMap.empty())
		header.type |= MVS::HeaderDepthDataRaw::HAS_CONF;
	if (!viewsMap.empty())
		header.type |= MVS::HeaderDepthDataRaw::HAS_VIEWS;
	fwrite(&header, sizeof(MVS::HeaderDepthDataRaw), 1, f.get());

	// Write image file name
	STATIC_ASSERT(sizeof(String::value_type) == sizeof(char));
	const String FileName(MAKE_PATH_REL(Util::getFullPath(Util::getFilePath(fileName)), Util::getFullPath(imageFileName)));
	const uint16_t nFileNameSize((uint16_t)FileName.length());
	fwrite(&nFileNameSize, sizeof(uint16_t), 1, f.get());
	fwrite(FileName.c_str(), sizeof(char), nFileNameSize, f.get());

	// Write neighbor IDs
	STATIC_ASSERT(sizeof(uint32_t) == sizeof(IIndex));
	const uint32_t nIDs(IDs.size());
	fwrite(&nIDs, sizeof(IIndex), 1, f.get());
	fwrite(IDs.data(), sizeof(IIndex), nIDs, f.get());

	// Write pose
	STATIC_ASSERT(sizeof(double) == sizeof(REAL));
	fwrite(K.val, sizeof(REAL), 9, f.get());
	fwrite(R.val, sizeof(REAL), 9, f.get());
	fwrite(C.ptr(), sizeof(REAL), 3, f.get());

	// Write depth-map
	if (fwrite(depthMap.getData(), sizeof(float), depthMap.area(), f.get()) != static_cast<size_t>(depthMap.area())) {
		DEBUG("error: writing depth-data to file '%s'", fileName.c_str());
		return false;
	}

	// Write confidence-map
	if (!confMap.empty())
		fwrite(confMap.getData(), sizeof(float), confMap.area(), f.get());

	// Write views-map
	if (!viewsMap.empty())
		fwrite(viewsMap.getData(), sizeof(uint8_t)*4, viewsMap.area(), f.get());

	return ferror(f.get()) == 0;
} // ExportDepthDataRaw
/*----------------------------------------------------------------*/


bool SFM::ImportMVS(const String& fileName, Scene& scene, bool loadColors)
{
	using MVSInterface = MVS::Interface;
	using MVSPlatform = MVS::Interface::Platform;

	TD_TIMER_STARTD();
	scene.Release();

	MVSInterface iface;
	if (!MVS::ARCHIVE::SerializeLoad(iface, fileName.c_str())) {
		VERBOSE("error: failed to load '%s'", fileName.c_str());
		return false;
	}

	const String basePath = MAKE_PATH_FULL(WORKING_FOLDER_FULL, Util::getFilePath(fileName));
	std::unordered_map<PairIdx::PairIndex, IIndex> camToID;
	camToID.reserve(iface.platforms.size());
	const auto DeduceSize = [&](uint32_t pid, uint32_t cid)->cv::Size {
		for (const auto& img : iface.images) {
			if (img.platformID != pid || img.cameraID != cid)
				continue;
			IMAGEPTR pImage(CImage::Create(basePath + img.name, CImage::READ));
			const bool valid = (pImage != NULL && pImage->ReadHeader());
			pImage.Release();
			if (valid)
				return cv::Size(pImage->GetWidth(), pImage->GetHeight());
		}
		return cv::Size();
	};

	for (uint32_t pid = 0; pid < iface.platforms.size(); ++pid) {
		const MVSPlatform& platform = iface.platforms[pid];
		for (uint32_t cid = 0; cid < platform.cameras.size(); ++cid) {
			const auto& cam = platform.cameras[cid];
			cv::Size size((int)cam.width, (int)cam.height);
			if (size.width == 0 || size.height == 0)
				size = DeduceSize(pid, cid);
			if (size.width <= 0 || size.height <= 0) {
				VERBOSE("error: missing resolution for platform %u camera %u", pid, cid);
				continue;
			}
			const auto fullK = platform.GetFullK(cid, (uint32_t)size.width, (uint32_t)size.height);
			PinholeCamera* pc = new PinholeCamera(size);
			pc->SetK(fullK);
			pc->trustIntrinsics = true;
			pc->SetName(cam.name.empty() ? platform.name : cam.name);
			pc->metadata.model = cam.bandName;
			const IIndex camID = scene.cameras.size();
			scene.cameras.emplace_back(pc);
			camToID.emplace(PairIdx(pid, cid).idx, camID);
		}
	}

	scene.images.reserve(iface.images.size());
	uint32_t posed = 0;
	for (size_t i = 0; i < iface.images.size(); ++i) {
		const auto& inImg = iface.images[i];
		const auto key = PairIdx(inImg.platformID, inImg.cameraID).idx;
		auto itCam = camToID.find(key);
		if (itCam == camToID.end()) {
			VERBOSE("error: skipping image %zu (platform %u camera %u not found)", i, inImg.platformID, inImg.cameraID);
			continue;
		}
		const IIndex camID = itCam->second;
		const String imgPath = basePath + inImg.name;
		Image& img = scene.images.emplace_back(
		    (inImg.ID != MVS::NO_ID) ? inImg.ID : scene.images.size(),
			imgPath);
		img.cameraID = camID;
		img.pCamera = scene.cameras[camID];
		if (inImg.IsValid()) {
			const auto pose = iface.platforms[inImg.platformID].GetPose(inImg.cameraID, inImg.poseID);
			img.R = pose.R;
			img.C = pose.C;
			posed++;
		}
	}

	scene.tracks.reserve(iface.vertices.size());
	if (loadColors)
		scene.colors.reserve(iface.vertices.size());
	for (size_t v = 0; v < iface.vertices.size(); ++v) {
		const auto& vx = iface.vertices[v];
		Track track(vx.X);
		track.observations.reserve(vx.views.size());
		for (const auto& vw : vx.views) {
			if (vw.imageID >= scene.images.size())
				continue;
			track.observations.emplace_back(vw.imageID, NO_ID);
		}
		track.numInliers = (uint8_t)std::min<size_t>(track.observations.size(), (size_t)std::numeric_limits<uint8_t>::max());
		if (!track.IsValid())
			continue;
		scene.tracks.emplace_back(std::move(track));
		if (loadColors) {
			Pixel8U col = Pixel8U::BLACK;
			if (v < iface.verticesColor.size()) {
				const TPoint3<uint8_t> c = iface.verticesColor[v].c;
				col = Pixel8U(c);
			}
			scene.colors.emplace_back(col);
		}
	}

	scene.transform = iface.transform;
	if (iface.obb.IsValid()) {
		OBB3::MATRIX rot = Matrix3x3(iface.obb.rot);
		OBB3::POINT ptMin = Point3(iface.obb.ptMin);
		OBB3::POINT ptMax = Point3(iface.obb.ptMax);
		scene.obb.Set(rot, ptMin, ptMax);
	}

	scene.status.nCalibratedImages = posed;
	scene.status.nTracks = (uint32_t)std::count_if(scene.tracks.begin(), scene.tracks.end(),
		[](const Track& t) { return t.IsInlier(); });
	scene.status.nState = Scene::Status::STATE::EMPTY;
	if (scene.status.nTracks > 0)
		scene.status.nState.set(Scene::Status::STATE::MATCHED);
	if (posed > 0)
		scene.status.nState.set(Scene::Status::STATE::CALIBRATED);

	DEBUG("Scene imported from MVS: %u platforms, %u images (%u posed), %u tracks%s from '%s' (%s)",
		(unsigned)iface.platforms.size(), scene.images.size(), posed, (unsigned)scene.tracks.size(),
		(loadColors && !scene.colors.empty()) ? " with colors" : "",
		fileName.c_str(), TD_TIMER_GET_FMT().c_str());
	return true;
} // ImportMVS

bool SFM::ExportMVS(const String& fileName, const Scene& scene,
	String undistortImageDir,
	float undistortAlpha,
	bool onlyInlierTracks,
	bool includeColors)
{
	using MVSInterface = MVS::Interface;
	using MVSPlatform = MVS::Interface::Platform;
	using MVSCamera = MVS::Interface::Platform::Camera;
	using MVSVertex = MVS::Interface::Vertex;
	using Mat33d = MVS::Interface::Mat33d;
	using Pos3d = MVS::Interface::Pos3d;

	if (scene.images.empty()) {
		VERBOSE("error: scene to be exported is empty");
		return false;
	}
	TD_TIMER_STARTD();
	MVSInterface iface;

	// Map each SFM camera pointer to platform index
	std::unordered_map<const Camera*, uint32_t> camToPlatform;
	camToPlatform.reserve(scene.cameras.size());

	const bool undistort = !undistortImageDir.empty();
	std::unordered_map<const Camera*, KMatrix> undistortK;
	CLISTDEF2(String) undistortedPaths;
	if (undistort)
		scene.UndistortImages(undistortImageDir, ".jpg", undistortAlpha, &undistortedPaths, &undistortK);

	// Create platforms (one camera per platform)
	unsigned numDistortedCams = 0;
	for (CameraPtr const cam : scene.cameras) {
		ASSERT(camToPlatform.count(cam) == 0);
		const uint32_t platformID = (uint32_t)iface.platforms.size();
		camToPlatform[cam] = platformID;
		MVSPlatform& platform = iface.platforms.emplace_back();
		platform.name = cam->metadata.name;
		MVSCamera& outCam = platform.cameras.emplace_back();

		// Build intrinsic matrix
		outCam.K = cam->GetK();
		if (cam->HasDistortion()) {
			if (undistort)
				outCam.K = undistortK.at(cam);
			++numDistortedCams;
		}
		outCam.width = (uint32_t)cam->GetWidth();
		outCam.height = (uint32_t)cam->GetHeight();
		// Camera extrinsics relative to platform: identity (platform pose holds absolute)
		outCam.R = Mat33d::eye();
		outCam.C = Pos3d(0, 0, 0);
		// Find first image using this camera to deduce image rotation
		for (const Image& img : scene.images) {
			if (img.pCamera != cam)
				continue;
			if (!img.IsRotated())
				break;
			// Image was rotated at load; adjust intrinsic to match the original orientation
			const cv::Size size = img.RevertRotation(&outCam.K);
			outCam.width = (uint32_t)size.width;
			outCam.height = (uint32_t)size.height;
			break;
		}
	}

	// Collect pose indices per image (stored in platform.poses vector);
	// platform poses are filled as we iterate images.
	const String basePath = MAKE_PATH_FULL(WORKING_FOLDER_FULL, Util::getFilePath(fileName));
	iface.images.reserve(scene.images.size());
	FOREACH(i, scene.images) {
		const Image& img = scene.images[i];
		const Camera* cam = img.pCamera;
		ASSERT(cam != NULL);
		const uint32_t platformID = camToPlatform[cam];
		MVSPlatform& platform = iface.platforms[platformID];
		uint32_t poseID = MVS::NO_ID;
		if (img.IsValid()) {
			// Pose3D stores R (world->camera) and C (camera center world coordinates)
			poseID = (uint32_t)platform.poses.size();
			auto& pose = platform.poses.emplace_back(img.R, img.C);
			img.RevertRotation(NULL, &pose.R); // revert any image rotation
		}
		// Build Interface::Image record
		// Use undistorted image path if available, otherwise use original
		MVS::Interface::Image outImg;
		outImg.name = undistort ? undistortedPaths[i] : img.fileName;
		outImg.name = MAKE_PATH_REL(basePath, outImg.name);
		outImg.platformID = platformID;
		outImg.cameraID = 0; // single camera per platform
		outImg.poseID = poseID; // may be NO_ID
		outImg.ID = img.ID; // preserve global ID
		iface.images.emplace_back(std::move(outImg));
	}

	// Export points (tracks)
	iface.vertices.reserve(scene.tracks.size());
	if (includeColors)
		includeColors = !scene.colors.empty(); // only if we have track colors
	if (includeColors)
		iface.verticesColor.reserve(scene.tracks.size());
	FOREACH(i, scene.tracks) {
		const Track& track = scene.tracks[i];
		if (!track.IsValid())
			continue;
		if (onlyInlierTracks && !track.IsInlier())
			continue;
		MVSVertex v;
		v.X = Cast<float>(track.position);
		v.views.reserve(track.GetNumInliers());
		for (const Observation& obs : track) {
			MVSVertex::View view;
			view.imageID = obs.imageID;
			view.confidence = 0.f;
			v.views.push_back(view);
		}
		ASSERT(v.views.size() >= 2);
		iface.vertices.emplace_back(std::move(v));
		if (includeColors) {
			MVS::Interface::Color col; // BGR order
			col.c = scene.colors[i];
			iface.verticesColor.push_back(col);
		}
	}

	// Serialize
	if (!MVS::ARCHIVE::SerializeSave(iface, fileName.c_str())) {
		VERBOSE("error: failed serialization for '%s'", fileName.c_str());
		return false;
	}
	DEBUG("Scene exported as MVS: %u platforms, %u images (%u valid), %u points%s to '%s' (%s)",
		(unsigned)iface.platforms.size(), (unsigned)iface.images.size(), scene.status.nCalibratedImages, (unsigned)iface.vertices.size(), undistort ? " (undistorted)" : "",
		fileName.c_str(), TD_TIMER_GET_FMT().c_str());
	if (numDistortedCams > 0 && !undistort)
		VERBOSE("warning: %u cameras had distortion; export ignores distortion (consider enabling undistort)", numDistortedCams);
	return true;
} // ExportMVS
/*----------------------------------------------------------------*/
