/*
 * VocabularyTree.h
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

#ifndef _SFM_VOCABULARYTREE_H_
#define _SFM_VOCABULARYTREE_H_

// I N C L U D E S /////////////////////////////////////////////////

#include "Scene.h"
#include "Image.h"


// D E F I N E S ///////////////////////////////////////////////////

namespace SFM {

// S T R U C T S ///////////////////////////////////////////////////

/**
 * @brief Image retrieval using hierarchical vocabulary tree (bag-of-words)
 * 
 * Efficient image matching by finding similar images using bag-of-words representation.
 * Avoids exhaustive O(N²) matching by querying top-K similar images via inverted index.
 * 
 * Implementation uses hierarchical K-means clustering to build a vocabulary tree,
 * TF-IDF scoring with burstiness reduction (sqrt(TF) weighting), and optional
 * soft assignment and query expansion for improved accuracy.
 * 
 * Key Features:
 * - Consistent descriptor sampling across training, indexing, and querying
 * - Optional soft assignment (k-best leaves) for +10-15% mAP improvement
 * - Optional query expansion for +10-20% recall improvement
 * - Support for both binary (Hamming) and float (L2) descriptors
 * 
 * References:
 * - Philbin et al. (2008): Soft assignment for improved recall
 * - Jégou et al. (2009): Burstiness weighting with sqrt(TF)
 * - Chum et al. (2007): Query expansion for retrieval
 */
class SFM_API VocabularyTree
{
public:
	VocabularyTree();
	~VocabularyTree();

	// Configuration for building/querying the vocabulary
	struct Config {
		// Descriptor type selector:
		//  - true: binary descriptors (AKAZE, ORB) using Hamming distance
		//  - false: quantized float descriptors (RootSIFT-like) using L2 distance
		// Both are stored as CV_8U bytes; this flag determines distance metric and clustering.
		bool descriptorsAreBinary = false;

		// Tree branching factor (children per node).
		// Typical: 8-10 for balanced tree. Higher K = flatter tree, faster query but larger vocabulary.
		// With K=10 and L=6, vocabulary has ~10^6 visual words.
		int K = 10;

		// Maximum tree depth (number of levels from root to leaves).
		// Typical: 5-6 for million-word vocabularies. Deeper = more words but slower quantization.
		// Total visual words ≈ K^L (e.g., 10^6 for K=10, L=6).
		int L = 6;

		// Maximum iterations for K-means clustering at each tree level.
		// Higher values improve centroid quality but increase training time.
		// Typical: 5-15 iterations; convergence usually happens within 10 iterations.
		unsigned maxKMeansIters = 10;

		// Random seed for reproducible vocabulary training.
		// Ensures K-means initialization produces same tree structure across runs.
		unsigned randomSeed = 42;

		// Descriptor sampling limit per image during training and indexing.
		//  - 0: use all descriptors (highest quality, slower for images with many features)
		//  - >0: select top N descriptors via grid-based spatial sampling (`SelectTopKeypoints`)
		// Recommendation: 1000-2000 for large datasets to balance speed and coverage.
		// Applied consistently across Build(), buildDatabase(), and Query().
		unsigned maxDescriptorsPerImage = 2000;

		// Soft assignment parameter: number of best-matching visual words per descriptor.
		//  - 0: hard assignment (each descriptor - 1 word, fast)
		//  - 3-5: soft assignment (each descriptor - k words with Gaussian weights)
		// Soft assignment improves recall (+10-15% mAP) at ~5× query cost.
		// Uses beam search to find k-best leaves with distance-based weighting.
		int softAssignmentK = 3;

		// Query expansion: number of top retrieval results to use for re-querying.
		//  - 0: disabled (standard single-pass query)
		//  - 5-10: average BoW vectors of top results with query, then re-query
		// Improves recall (+10-20%) by leveraging relevant images' features.
		// Weighted by rank: result i contributes with weight 1.0/(i+2).
		unsigned queryExpansionImages = 5;
	};

	/**
	 * @brief Build vocabulary from scene descriptors with explicit configuration
	 * @param scene Scene containing images with extracted features
	 * @param cfg   Vocabulary configuration (descriptor kind, K/L, etc.)
	 * @param vocabFile Optional pre-trained vocabulary file to load topology from
	 * @return true if vocabulary built successfully
	 */
	bool Build(const Scene& scene, const Config& cfg, const String& vocabFile = String());

	/**
	 * @brief Query similar images for a given image (thread-safe)
	 * @param image Query image with extracted features
	 * @param maxResults Maximum number of similar images to return
	 * @param minScore Minimum similarity score threshold (0.0-1.0)
	 * @return Vector of (imageID, score) pairs sorted by score (descending)
	 * 
	 * Computes bag-of-words vector and queries database using TF-IDF scoring.
	 */
	std::vector<std::pair<uint32_t, float>> Query(
		const Image& image,
		unsigned maxResults = 50,
		float minScore = 0.0f) const;

	/**
	 * @brief Save vocabulary to file
	 * @param path File path to save vocabulary
	 * @return true if saved successfully
	 */
	bool Save(const String& path) const;

	/**
	 * @brief Load vocabulary topology from file (no scene index)
	 * @param path File path to load vocabulary
	 * @return true if loaded successfully
	 *
	 * Loads only the reusable tree (topology + centroids + IDF). To use for
	 * retrieval in a given scene, call `Index(scene)` after `Load`.
	 */
	bool Load(const String& path);

	/**
	 * @brief Build scene-specific inverted index for a loaded/trained tree
	 * @param scene Scene containing images with descriptors
	 * @return true if indexing succeeded
	 */
	bool Index(const Scene& scene);

	/**
	 * @brief Check if vocabulary is ready
	 * @return true if vocabulary has been built or loaded
	 */
	bool IsValid() const { return pImpl != nullptr; }

	/**
	 * @brief Release vocabulary and free memory
	 */
	void Release();

	/**
	 * @brief Get top descriptors for an image (cached)
	 * @param image Image to extract descriptors from
	 * @return Matrix of descriptors (rows x dim)
	 */
	const cv::Mat& GetTopDescriptors(const Image& image) const;

	/**
	 * @brief Clear the descriptor cache
	 */
	void ClearDescriptorsCache();

	/**
	 * @brief Get the configured max descriptors per image
	 */
	unsigned GetMaxDescriptors() const;

private:
	// Forward declaration for PIMPL pattern (hide OpenCV implementation)
	struct Impl;
	Impl* pImpl;
};
/*----------------------------------------------------------------*/

} // namespace SFM

#endif // _SFM_VOCABULARYTREE_H_
