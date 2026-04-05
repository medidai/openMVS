/*
 * TestsSFM.h
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


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

namespace SFM {

// VocabularyTree save/load roundtrip test
bool VocabularyTreeTest();

// Test Bundle-Adjustment PinholeReprojectionErrorAnalytic Jacobians against AutoDiff
bool BAPinholeReprojectionJacobianTest();

// Small SFM smoke test: build tiny scene and run BundleAdjustment::Adjust
bool PipelineTest();

// Triplet star-initialization test: 3-view scene with tracks + StarInitializer + BA
bool TripletStarInitTest();

// Two-view geometry test: PairsMatcher and ImagePair matrix operations
bool TwoViewTest();

// Reconstruction test: Import images, extract features, match pairs, build tracks, and initialize
bool ReconstructTest();

// Test function for rotation estimation
bool RotationEstimatorTest();

// Test function for global scale estimation
bool ScaleEstimatorTest();

// Test function for global translation estimation
bool TranslationEstimatorTest();

// Pairs weighting test
bool PairsWeightingTest();

// PairsMatcher sequential mode test
bool PairMatcherTest();

// Pre-matching optimization test
bool PreMatchTest();

// View graph calibrator test: Refine focal length using view graph optimization
bool ViewGraphCalibratorTest();

// Phase 1: Scene Clustering tests
bool SceneClusterSingleClusterTest();
bool SceneClusterSizeConstraintsTest();
bool SceneClusterDisconnectedComponentsTest();
bool SceneClusterMemoryProtocolTest();
bool SceneClusterIDRemappingTest();
bool SceneClusterSmallClusterRescueTest();

// Phase 3: Global Alignment tests
bool GlobalAlignmentBuildGlobalToLocalMapTest();
bool GlobalAlignmentRotationAveragingExtendedTest();
bool GlobalAlignmentScaleAveragingExtendedTest();
bool GlobalAlignmentScaleAveragingFallbackTest();
bool GlobalAlignmentTranslationAveragingExtendedTest();
bool GlobalAlignmentMergeSingleSceneTest();
bool GlobalAlignmentTrackMergeDuplicateImageGuardTest();
bool GlobalAlignmentTrackMerge3DProximityGuardTest();

// End-to-end hierarchical SFM tests
bool HierarchicalSFMSplitMergeRoundtripTest();
bool HierarchicalSFMWithRandomTransformTest();
/*----------------------------------------------------------------*/

} // namespace SFM
