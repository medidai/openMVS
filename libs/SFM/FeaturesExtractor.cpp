/*
 * FeaturesExtractor.cpp
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
 */

#include "Common.h"
#include "FeaturesExtractor.h"
#include "Scene.h"
#include <opencv2/features2d.hpp>

#ifdef _USE_SIFTGPU
#include <glad/glad.h>
#include <siftgpu/SiftGPU.h>
#endif

using namespace SFM;


// D E F I N E S ///////////////////////////////////////////////////

#pragma push_macro("VERBOSE")
#undef VERBOSE
#define VERBOSE(...) LOG(lt, __VA_ARGS__)


// S T R U C T S ///////////////////////////////////////////////////

DEFINE_LOG_NAME(lt, _T("FeatExtr"));


#ifdef _USE_SIFTGPU
/**
 * @brief Coordinates SiftGPU feature extraction using thread pool
 *
 * Implements producer-consumer pattern:
 * - Main thread: Executes GPU operations (SiftGPU::RunSIFT)
 * - Worker threads: Pre-load images and post-process features in parallel
 */
class SiftGPUFeatureCoordinator
{
public:
	SiftGPUFeatureCoordinator(FeaturesExtractor& _extractor)
		: extractor(_extractor) {}

	// Initialize SiftGPU context (returns false on failure)
	bool Initialize() {
		gpu = std::make_unique<SiftGPU>();

		constexpr unsigned maxImageSize = 5120;
		constexpr int firstOctave = -1;
		constexpr int octaveResolution = 3;
		constexpr unsigned maxNumOrientations = 2;
		constexpr float peakThreshold = 0.005f;
		constexpr float edgeThreshold = 20.f;
		constexpr unsigned maxNumFeatures = 0; // 0 = no limit
		constexpr bool upright = false;
		const bool darknessAdaptivity = extractor.GetConfig().useCUDA ? false : true;
		int gpuIndices[1] = { -1 };

		std::vector<std::string> args;
		args.push_back("./sift_gpu");
		#ifndef _RELEASE
		args.push_back("-v"); args.push_back("1");
		#else
		args.push_back("-v"); args.push_back("0");
		#endif
		#ifdef _USE_CUDA
		if (extractor.GetConfig().useCUDA && gpuIndices[0] < 0)
			gpuIndices[0] = 0;
		if (gpuIndices[0] >= 0) {
			args.push_back("-cuda"); args.push_back(std::to_string(gpuIndices[0]));
		}
		#endif
		if (darknessAdaptivity) {
			if (gpuIndices[0] >= 0)
				DEBUG("warning: darkness adaptivity only available for GLSL SiftGPU.");
			args.push_back("-da");
		}
		const int octaveFactor = 1 << -MINF(0, firstOctave);
		args.push_back("-maxd"); args.push_back(std::to_string(maxImageSize * octaveFactor));
		args.push_back("-t"); args.push_back(std::to_string(peakThreshold));
		args.push_back("-e"); args.push_back(std::to_string(edgeThreshold));
		if (maxNumFeatures > 0) {
			args.push_back("-tc2"); args.push_back(std::to_string(maxNumFeatures));
		}
		args.push_back("-fo"); args.push_back(std::to_string(firstOctave));
		args.push_back("-d"); args.push_back(std::to_string(octaveResolution));
		if (upright) {
			args.push_back("-ofix");
			args.push_back("-mo"); args.push_back("1");
		} else {
			args.push_back("-mo"); args.push_back(std::to_string(maxNumOrientations));
		}

		std::vector<const char*> argv;
		for (const auto& a : args) argv.push_back(a.c_str());
		gpu->ParseParam(argv.size(), argv.data());

		if (gpu->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
			VERBOSE("error: SiftGPU not fully supported");
			return false;
		}
		const int maxNumFeaturesPerImage = extractor.GetConfig().GetMaxNumFeatures();
		const int maxNumFeaturesPerImageGPU = gpu->GetMaxNumFeatures();
		if (maxNumFeaturesPerImageGPU < maxNumFeaturesPerImage) {
			constexpr LPCTSTR warningMessage = "warning: SiftGPU only supports a maximum of %d features per image"
				#ifdef _USE_CUDA
				", consider using CUDA to avoid this limitation"
				#endif
				"; the max number of features will be capped from %d to %d";
			VERBOSE(warningMessage, maxNumFeaturesPerImageGPU, maxNumFeaturesPerImage, maxNumFeaturesPerImageGPU);
			extractor.GetConfig().SetMaxNumFeatures(maxNumFeaturesPerImageGPU);
		}
		DEBUG_EXTRA("SiftGPU initialized: %s mode (%d max-features-per-image)",
			gpu->GetLanguage() == SiftGPU::SIFTGPULANG_CUDA ? "CUDA" : (gpu->GetLanguage() == SiftGPU::SIFTGPULANG_OPENCL ? "OpenCL" : "GLSL"), maxNumFeaturesPerImageGPU);
		return true;
	}

	// Process all images using GPU + async post-processing
	size_t ProcessImages(Util::Progress& progress) {
		std::atomic<size_t> numFeatures(0);
		FOREACH(i, extractor.GetScene().images) {
			++progress;
			Image& img = extractor.GetScene().images[i];
			// Skip if already processed
			if (img.HasFeatures() && img.HasDescriptors())
				continue;
			// Pre-load next image (async task)
			if (i + 1 < extractor.GetScene().images.size()) {
				Image& imgNext = extractor.GetScene().images[i+1];
				extractor.GetScene().threadPool.detach_task([&imgNext]() {
					if (!imgNext.HasPixels())
						imgNext.LoadPixels(true);
				});
			}
			// Ensure current image is loaded
			if (!img.HasPixels() && !img.LoadPixels(true)) {
				VERBOSE("error: no pixels loaded for image %u", img.ID);
				continue;
			}
			// GPU processing (main thread, blocking)
			if (!gpu->RunSIFT(img.pixels.cols, img.pixels.rows, img.pixels.data, GL_LUMINANCE, GL_UNSIGNED_BYTE)) {
				VERBOSE("error: SiftGPU failed on image %u", img.ID);
				continue;
			}
			// Download GPU results
			const int num = gpu->GetFeatureNum();
			CLISTDEF0IDX(SiftGPU::SiftKeypoint,uint32_t) keys(num);
			FloatArr descs(num * 128);
			gpu->GetFeatureVector(keys.data(), descs.data());
			// Submit post-processing task (captures GPU results by value)
			extractor.GetScene().threadPool.detach_task([this, &img, keys = std::move(keys), descs = std::move(descs), &numFeatures]() {
				// Grid-based feature filtering
				const int cellWidth = img.pixels.cols / 3;
				const int cellHeight = img.pixels.rows / 3;
				std::vector<Unsigned32Arr> grid(9);
				FOREACH(k, keys) {
					const int cx = (int)keys[k].x / cellWidth;
					const int cy = (int)keys[k].y / cellHeight;
					if (cx >= 0 && cx < 3 && cy >= 0 && cy < 3)
						grid[cy * 3 + cx].push_back(k);
				}
				// Select best features from each cell
				Unsigned32Arr selected;
				selected.reserve(MINF(keys.size(), (uint32_t)extractor.GetConfig().maxFeaturesPerCell * 9));
				for (int c = 0; c < 9; ++c) {
					Unsigned32Arr& cells = grid[c];
					if (cells.size() > (uint32_t)extractor.GetConfig().maxFeaturesPerCell) {
						std::partial_sort(cells.begin(), cells.begin() + extractor.GetConfig().maxFeaturesPerCell, cells.end(),
							[&keys](int a, int b) { return keys[a].s > keys[b].s; });
						cells.resize(extractor.GetConfig().maxFeaturesPerCell);
					}
					selected.JoinRemove(cells);
				}
				// Store keypoints and descriptors
				img.keypoints.resize(selected.size());
				img.descriptors.create((int)selected.size(), 128, CV_8U);
				FOREACH(k, selected) {
					const uint32_t idx = selected[k];
					const SiftGPU::SiftKeypoint& sk = keys[idx];
					img.keypoints[k] = cv::KeyPoint(sk.x, sk.y, sk.s, sk.o, 0.1f);
					// RootSIFT conversion
					cv::Mat siftRow(1, 128, CV_32F, const_cast<float*>(&descs[idx * 128]));
					FeaturesExtractor::ConvertToRootSIFT(siftRow).copyTo(img.descriptors.row((int)k));
				}
				numFeatures.fetch_add(img.keypoints.size(), std::memory_order_relaxed);
				if (extractor.GetConfig().releaseImagePixels)
					img.ReleasePixels();
				DEBUG_ULTIMATE("Extracted features for image % 4u: % 6u features using %s (%.2f%s focal-length)",
					img.ID, img.keypoints.size(), FeatureTypeToString(extractor.GetConfig().detectorType).c_str(), img.pCamera->GetFocalLength(), img.TrustIntrinsics() ? "" : "*");
				if (!extractor.GetConfig().exportOpenMVGDir.empty())
					FeaturesExtractor::ExportFeaturesOpenMVG(extractor.GetConfig().exportOpenMVGDir, img);
			});
		}
		// Wait for all post-processing tasks to complete
		extractor.GetScene().threadPool.wait();
		return numFeatures.load(std::memory_order_relaxed);
	}

private:
	FeaturesExtractor& extractor;
	std::unique_ptr<SiftGPU> gpu;
};
#endif // _USE_SIFTGPU


FeaturesExtractor::FeaturesExtractor(Scene& _scene, const FeatureExtractionConfig& _config)
	: scene(_scene), config(_config)
{
}

FeaturesExtractor::~FeaturesExtractor() = default;


size_t FeaturesExtractor::Extract()
{
	// Per-thread feature extraction for efficient parallel processing
	cv::setNumThreads(1); // temporary turn off multi-threading for OpenCV functions
	Util::Progress progress(_T("Extract features from images"), scene.images.size());
	GET_LOGCONSOLE().Pause();
	size_t numFeatures = 0;
	#ifdef _USE_SIFTGPU
	if (config.detectorType == FeatureType::SIFTGPU) {
		// Use coordinator for producer-consumer pattern with thread pool
		SiftGPUFeatureCoordinator coordinator(*this);
		if (!coordinator.Initialize()) {
			GET_LOGCONSOLE().Play();
			progress.close();
			return 0;
		}
		numFeatures = coordinator.ProcessImages(progress);
	} else
	#endif // _USE_SIFTGPU
	{
		// CPU multi-threaded feature extraction with per-thread detectors

		// Create per-thread detectors using thread-local storage
		std::unordered_map<std::thread::id, cv::Ptr<cv::Feature2D>> detectors;
		std::atomic<size_t> atomicNumFeatures{0};

		scene.threadPool.detach_loop(IIndex(0), scene.images.size(), [&](IIndex i) {
			Image& img = scene.images[i];
			if (img.HasFeatures() && (img.HasDescriptors() || scene.status.nFeaturesType != FeatureType::NONE)) {
				++progress;
				return;
			}
			if (ExtractImage(img, detectors[std::this_thread::get_id()]))
				atomicNumFeatures.fetch_add(img.keypoints.size(), std::memory_order_relaxed);
			++progress;
		});

		scene.threadPool.wait();
		numFeatures = atomicNumFeatures.load(std::memory_order_relaxed);
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	cv::setNumThreads(scene.nMaxThreads); // restore OpenCV threading
	return numFeatures;
}

bool FeaturesExtractor::ExtractImage(Image& image, cv::Ptr<cv::Feature2D>& detector)
{
	if (!config.importOpenMVGDir.empty() && ImportFeaturesOpenMVG(config.importOpenMVGDir, image)) {
		image.ReleasePixels(); // free pixel memory after feature extraction
		DEBUG_ULTIMATE("Imported features for image % 4u: % 6u features (%.2f focal-length)",
			image.ID, image.keypoints.size(), image.pCamera->GetFocalLength());
		return !image.keypoints.empty();
	}

	if (!image.HasPixels() && !image.LoadPixels(true)) {
		VERBOSE("FeaturesExtractor::ExtractImage: no pixels loaded for image %u", image.ID);
		return false;
	}

	// Clear existing features
	image.keypoints.clear();
	image.descriptors.release();

	// Create the feature detector based on type
	if (!detector) {
		switch (config.detectorType) {
		case FeatureType::AKAZE:
			detector = cv::AKAZE::create();
			break;
		case FeatureType::ORB:
			detector = cv::ORB::create((int)config.maxFeaturesPerCell);
			break;
		case FeatureType::SIFT:
			detector = cv::SIFT::create();
			break;
		default:
			VERBOSE("FeaturesExtractor::ExtractImage: unknown detector type '%s'", FeatureTypeToString(config.detectorType).c_str());
			return false;
		}
	}

	// Divide image into 3x3 grid with overlapping borders
	const int cellWidth = image.pixels.cols / 3;
	const int cellHeight = image.pixels.rows / 3;
	const int borderSize = MINF(64, MINF(cellWidth, cellHeight) / 2); // overlap border size

	// Extract features from each cell
	std::vector<cv::Mat> vecDescriptors;
	image.keypoints.reserve(config.minFeaturesPerCell * 9);
	vecDescriptors.reserve(config.minFeaturesPerCell * 9);
	for (int row = 0; row < 3; ++row) {
		for (int col = 0; col < 3; ++col) {
			// Define cell region with overlapping borders
			const cv::Rect rcCell(
				col * cellWidth,
				row * cellHeight,
				col == 2 ? image.pixels.cols - col * cellWidth : cellWidth,
				row == 2 ? image.pixels.rows - row * cellHeight : cellHeight);
			// Extend cell bounds with border (clamped to image bounds)
			const cv::Rect rcCellExtended(
				col == 0 ? 0 : rcCell.x - borderSize,
				row == 0 ? 0 : rcCell.y - borderSize,
				col == 2 ? image.pixels.cols - rcCell.x + borderSize : rcCell.width + borderSize * 2,
				row == 2 ? image.pixels.rows - rcCell.y + borderSize : rcCell.height + borderSize * 2);
			// Initialize image for this cell as a ROI in the full image
			cv::Mat cellImage = image.pixels(rcCellExtended);

			// Extract features in this cell with iterative sensitivity adjustment;
			// OpenCV feature detectors (SIFT/ORB/AKAZE) report cv::KeyPoint::pt
			// already in pixel coordinates consistent with “integer = pixel center” convention
			std::vector<cv::KeyPoint> cellKeypoints;
			cv::Mat cellDescriptors;
			detector->detectAndCompute(cellImage, cv::noArray(), cellKeypoints, cellDescriptors);

			// Retry up to 5 times with progressively more sensitive settings if needed
			for (int retry = 0; retry < 5 && cellKeypoints.size() < (size_t)config.minFeaturesPerCell; ++retry) {
				cellKeypoints.clear();
				cellDescriptors.release();

				switch (config.detectorType) {
				case FeatureType::AKAZE: {
					cv::Ptr<cv::AKAZE> akaze = detector.dynamicCast<cv::AKAZE>();
					const double thresholds[] = {0.0005, 0.0001, 0.00005, 0.00001, 0.000001};
					akaze->setThreshold(thresholds[retry]);
					akaze->detectAndCompute(cellImage, cv::noArray(), cellKeypoints, cellDescriptors);
					akaze->setThreshold(0.001); // Reset to default
				} break;
				case FeatureType::ORB: {
					cv::Ptr<cv::ORB> orb = detector.dynamicCast<cv::ORB>();
					const int fastThresholds[] = {15, 10, 7, 5, 3};
					orb->setFastThreshold(fastThresholds[retry]);
					orb->detectAndCompute(cellImage, cv::noArray(), cellKeypoints, cellDescriptors);
					orb->setFastThreshold(20); // Reset to default
				} break;
				case FeatureType::SIFT: {
					cv::Ptr<cv::SIFT> sift = detector.dynamicCast<cv::SIFT>();
					const double contrastThresholds[] = {0.03, 0.02, 0.015, 0.01, 0.005};
					sift->setContrastThreshold(contrastThresholds[retry]);
					sift->detectAndCompute(cellImage, cv::noArray(), cellKeypoints, cellDescriptors);
					sift->setContrastThreshold(0.04); // Reset to default
				} break;
				}
			}

			// Build indices for features within the core cell (without border)
			// Also adjust keypoint coordinates to global image coordinates
			std::vector<int> selectedIndices;
			selectedIndices.reserve(cellKeypoints.size());
			for (size_t i = 0; i < cellKeypoints.size(); ++i) {
				cv::KeyPoint& kp = cellKeypoints[i];
				// Adjust to global coordinates
				kp.pt.x += rcCellExtended.x;
				kp.pt.y += rcCellExtended.y;
				// Check if within core cell
				if (rcCell.contains(kp.pt))
					selectedIndices.push_back(i);
			}

			// Limit features per cell if needed
			if (selectedIndices.size() > (size_t)config.maxFeaturesPerCell) {
				// Sort indices by keypoint response * size (descending)
				std::sort(selectedIndices.begin(), selectedIndices.end(),
					[&cellKeypoints](int a, int b) {
						return Image::ComputeKeypointWeight(cellKeypoints[a]) > Image::ComputeKeypointWeight(cellKeypoints[b]);
					});
				// Keep only the best
				selectedIndices.resize(config.maxFeaturesPerCell);
			}

			// Copy selected keypoints and descriptors to output arrays (only once)
			ASSERT(image.keypoints.size() == vecDescriptors.size());
			const size_t offset = image.keypoints.size();
			for (int idx : selectedIndices) {
				image.keypoints.push_back(cellKeypoints[idx]);
				if (!cellDescriptors.empty()) {
					if (config.detectorType == FeatureType::SIFT)
						vecDescriptors.push_back(ConvertToRootSIFT(cellDescriptors.row(idx)));
					else
						vecDescriptors.push_back(cellDescriptors.row(idx));
				}
			}

			// Normalize keypoint responses to linear scale
			// - SIFT: already linear (DoG value)
			// - AKAZE: quadratic (det(Hessian)) -> apply sqrt()
			// - ORB: quadratic by default (HARRIS_SCORE) -> apply sqrt()
			// This ensures responses scale linearly with image contrast for proper weighting
			auto NormalizeResponses = [](std::vector<cv::KeyPoint>& kps, size_t offset, FeatureType type) {
				if (type == FeatureType::AKAZE || type == FeatureType::ORB)
					for (size_t i = offset; i < kps.size(); ++i)
						kps[i].response = SQRT(MAXF(0.f, kps[i].response));
			};
			// Normalize responses after detection (whether initial or retry)
			NormalizeResponses(image.keypoints, offset, config.detectorType);
		}
	}
	cv::vconcat(vecDescriptors, image.descriptors);
	if (config.releaseImagePixels)
		image.ReleasePixels(); // free pixel memory after feature extraction

	DEBUG_ULTIMATE("Extracted features for image % 4u: % 6u features using %s (%.2f%s focal-length)",
	    image.ID, image.keypoints.size(), FeatureTypeToString(config.detectorType).c_str(), image.pCamera->GetFocalLength(), image.TrustIntrinsics() ? "" : "*");

	if (!config.exportOpenMVGDir.empty())
		ExportFeaturesOpenMVG(config.exportOpenMVGDir, image);
	return !image.keypoints.empty();
}

cv::Mat FeaturesExtractor::ConvertToRootSIFT(const cv::Mat& siftDesc)
{
	// RootSIFT: L1-normalize each descriptor, then sqrt, then quantize to uint8_t [0-255]
	// Input: CV_32F SIFT descriptors (each row is usually a 128-dim descriptor)
	// Output: CV_8U RootSIFT descriptors
	ASSERT(siftDesc.type() == CV_32F);
	cv::Mat rootsiftDesc(siftDesc.rows, siftDesc.cols, CV_8U);
	// Process each descriptor row individually
	for (int i = 0; i < siftDesc.rows; ++i) {
		cv::Mat normalized;
		// L1-normalize
		cv::normalize(siftDesc.row(i), normalized, 1.0, 0.0, cv::NORM_L1);
		// Square root
		cv::sqrt(normalized, normalized);
		// Scale to [0, 255] and quantize to uint8_t;
		// even though RootSIFT values are in [0,1] after sqrt normalization,
		// most are below 0.4, so for better precision they are quantized by scaling by 512
		normalized.convertTo(rootsiftDesc.row(i), CV_8U, 512.0);
	}
	return rootsiftDesc;
}
/*----------------------------------------------------------------*/


bool FeaturesExtractor::ExportFeaturesOpenMVG(const String& outputDir, const Image& image)
{
	// Require keypoints; descriptors may be empty (exported count will be zero)
	if (image.keypoints.empty())
		return false;

	const String basePath = outputDir + PATH_SEPARATOR_STR + Util::getFileName(image.fileName);
	const String featPath = basePath + ".feat";
	const String descPath = basePath + ".desc";

	// Export keypoints (x y scale orientation)
	{
		std::ofstream file(featPath, std::ios::trunc);
		if (!file.is_open()) {
			VERBOSE("error: failed to open feature file: %s", featPath.c_str());
			return false;
		}
		for (const auto& kp : image.keypoints)
			file << kp.pt.x << ' ' << kp.pt.y << ' ' << kp.size << ' ' << kp.angle << '\n';
	}

	// Export descriptors as binary (size_t count + raw bytes)
	{
		std::ofstream file(descPath, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!file.is_open()) {
			VERBOSE("error: failed to open descriptor file: %s", descPath.c_str());
			return false;
		}
		const size_t numDesc = (image.descriptors.type() == CV_8U) ? static_cast<size_t>(image.descriptors.rows) : 0;
		file.write(reinterpret_cast<const char*>(&numDesc), sizeof(size_t));
		if (numDesc > 0) {
			ASSERT(image.descriptors.cols > 0);
			const size_t rowBytes = static_cast<size_t>(image.descriptors.cols) * image.descriptors.elemSize();
			for (int r = 0; r < image.descriptors.rows; ++r)
				file.write(reinterpret_cast<const char*>(image.descriptors.ptr(r)), rowBytes);
		}
	}

	DEBUG_ULTIMATE("Image % 4u exported %zu OpenMVG features: %s, %s", image.ID, image.keypoints.size(), featPath.c_str(), descPath.c_str());
	return true;
}

bool FeaturesExtractor::ImportFeaturesOpenMVG(const String& inputDir, Image& image)
{
	const String basePath = inputDir + PATH_SEPARATOR_STR + Util::getFileName(image.fileName);
	const String featPath = basePath + ".feat";
	const String descPath = basePath + ".desc";

	image.keypoints.clear();
	image.descriptors.release();

	// Load keypoints (x y scale orientation)
	std::ifstream featFile(featPath);
	if (!featFile.is_open()) {
		VERBOSE("error: failed to open feature file: %s", featPath.c_str());
		return false;
	}
	double x = 0.0, y = 0.0, size = 0.0, angle = 0.0;
	while (featFile >> x >> y >> size >> angle)
		image.keypoints.emplace_back((float)x, (float)y, (float)size, (float)angle, 0.01f);
	if (image.keypoints.empty()) {
		VERBOSE("error: no keypoints read from: %s", featPath.c_str());
		return false;
	}

	// Load descriptors if available (expects the same binary layout as ExportFeaturesOpenMVG)
	std::ifstream descFile(descPath, std::ios::binary);
	if (descFile.is_open()) {
		descFile.seekg(0, std::ios::end);
		const std::streamoff fileSize = descFile.tellg();
		if (fileSize < (std::streamoff)sizeof(size_t)) {
			VERBOSE("error: descriptor file too small: %s", descPath.c_str());
			image.descriptors.release();
			return false;
		}
		descFile.seekg(0, std::ios::beg);
		size_t numDesc = 0;
		descFile.read(reinterpret_cast<char*>(&numDesc), sizeof(size_t));
		const std::streamoff dataBytes = fileSize - (std::streamoff)sizeof(size_t);
		if (numDesc == 0 || dataBytes == 0) {
			image.descriptors.release();
			DEBUG_LEVEL(3, "Image % 4u imported %zu OpenMVG features (no descriptors): %s",
				image.ID, image.keypoints.size(), featPath.c_str());
			return true;
		}
		const size_t rowBytes = static_cast<size_t>(dataBytes) / numDesc;
		if (rowBytes == 0 || rowBytes * numDesc != static_cast<size_t>(dataBytes)) {
			VERBOSE("error: descriptor file size mismatch: %s", descPath.c_str());
			image.descriptors.release();
			return false;
		}
		image.descriptors.create((int)numDesc, (int)rowBytes, CV_8U);
		for (size_t r = 0; r < numDesc; ++r) {
			descFile.read(reinterpret_cast<char*>(image.descriptors.ptr((int)r)), rowBytes);
			if (!descFile) {
				VERBOSE("error: failed to read descriptor row %zu from: %s", r, descPath.c_str());
				image.descriptors.release();
				return false;
			}
		}
		if (image.keypoints.size() != numDesc)
			VERBOSE("error: descriptor/keypoint count mismatch: %zu descriptors vs %zu keypoints", numDesc, image.keypoints.size());
		DEBUG_LEVEL(3, "Image % 4u imported %zu OpenMVG features and descriptors: %s, %s",
			image.ID, image.keypoints.size(), featPath.c_str(), descPath.c_str());
		return true;
	}

	DEBUG_LEVEL(3, "Image % 4u imported %zu OpenMVG features (descriptors missing): %s",
		image.ID, image.keypoints.size(), featPath.c_str());
	return true;
}
/*----------------------------------------------------------------*/

#pragma pop_macro("VERBOSE")
