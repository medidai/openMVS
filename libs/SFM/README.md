# SFM Library

The SFM (Structure from Motion) library provides complete photogrammetric reconstruction from images or video. It takes a collection of images, finds feature correspondences, estimates camera poses, and produces a sparse 3D point cloud -- the input that the MVS library needs for dense reconstruction.

## What You Need to Know First

### SFM vs MVS: two halves of a pipeline

The SFM library operates **before** the MVS library in the photogrammetry pipeline. SFM answers: "Where was each camera, and where are the sparse 3D points?" MVS then answers: "What does the dense surface look like?"

```
Images → [SFM: poses + sparse points] → [MVS: dense mesh + texture]
```

### SFM::Scene vs MVS::Scene

These are **different classes in different namespaces**. `SFM::Scene` stores feature descriptors, image pairs, and tracks. `MVS::Scene` stores depth maps, meshes, and textures. At the handoff point, `SFM::Scene` is converted to `MVS::Scene` via the export functions in `InterfaceMVS.h`.

### Three reconstruction strategies

The library supports three approaches, each suited to different scenarios:

1. **Incremental** (`Scene::Reconstruct`): Registers images one at a time. Most robust, handles difficult cases, but O(N) bundle adjustments.
2. **Hierarchical** (`Scene::ReconstructHierarchical`): Splits into clusters, reconstructs each independently, then merges. Best for large datasets (1000+ images).
3. **Global** (`Scene::ReconstructGlobal`): Solves all rotations and translations simultaneously. Fastest when it works, but less robust to outliers.

## Architecture

### Camera System (`Camera.h`)

The camera model is **polymorphic** -- an abstract `Camera` base class with two implementations:

**PinholeCamera** (most common):
- Intrinsics: focal lengths (fx, fy), principal point (cx, cy)
- Brown-Conrady distortion: radial (k1-k6) and tangential (p1, p2)
- `useAdditionalDistortion` flag enables k4-k6 (off by default)
- `trustIntrinsics` flag indicates calibration reliability (affects matching strategy)

**SphericalCamera** (360 imagery):
- Equirectangular projection
- No distortion parameters

Cameras can be **shared** between images (same physical camera). During bundle adjustment, shared cameras are optimized once and the result applies to all images using that camera.

### Pose Convention (`Pose.h`)

```cpp
class Pose3D {
    RMatrix R;   // 3x3 rotation: world → camera coordinates
    CMatrix C;   // 3D camera center in world coordinates
};
```

This follows the OpenMVS convention: `P = KR[I|-C]`. The camera "looks down" the Z axis. Operators allow composition (`A * B`) and relative pose computation (`A / B`).

### Image and Features (`Image.h`)

Each Image stores:
- **Keypoints**: Detected feature locations (`cv::KeyPoint` array)
- **Descriptors**: Feature vectors (`cv::Mat`, either `CV_8U` binary or `CV_32F` float)
- **Metadata**: EXIF data (focal length, GPS, timestamp, sensor size)
- **View**: Camera model reference + pose

Features are extracted with a **3x3 spatial grid** to ensure even distribution across the image. Each cell targets up to 3000 features, giving ~27k features per image.

### Tracks (`Track.h`)

A track is a 3D point observed in multiple images:

```cpp
class Track {
    Point3 position;              // 3D world coordinates
    ObservationArr observations;  // [(imageID, featureID), ...]
    uint32_t numInliers;          // First N observations are inliers
};
```

Tracks are built using **union-find** (disjoint sets): if feature A in image 1 matches feature B in image 2, and feature B matches feature C in image 3, then A, B, C all belong to the same track.

### Image Pairs (`ImagePair.h`)

Stores the geometric relationship between two images:

```cpp
class ImagePair {
    MatchArr matches;              // Inlier feature correspondences
    std::optional<Matrix3> F, E, H; // Estimated geometry matrices
    Pose3D relativePose;           // Relative camera pose
    float weightSpatial;           // How well features cover the image
    float weightConnectivity;      // Importance in the view graph
    float weightTriplet;           // 3-view consistency score
};
```

The **composite weight** (`spatial × connectivity × triplet`) ranks pairs by reliability. Triplet weight is the strongest quality signal -- it measures consistency across three-view loops.

## The Reconstruction Pipeline

Both reconstruction workflows share a common front-end that extracts features, matches images, and builds tracks. They diverge after that: the **hierarchical** workflow clusters the scene and uses incremental reconstruction per cluster, while the **global** workflow solves all poses simultaneously.

```
Input: Images (or video keyframes)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    COMMON FRONT-END                      │
│                                                          │
│  1. Feature Extraction (AKAZE/ORB/SIFT)                 │
│  2. Feature Matching (Vocabulary/Exhaustive/Sequential)  │
│  3. Geometric Verification (RANSAC: E/F/H matrices)      │
│  4. View Graph Calibration (focal length estimation)     │
│  5. Track Building (union-find on matches)               │
│  6. Track & Image Filtering (outliers, weak views)       │
│                                                          │
└──────────────────────┬──────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
┌─────────────────────┐  ┌──────────────────────┐
│    HIERARCHICAL     │  │       GLOBAL         │
│    (robust, slower) │  │    (fast, simpler)   │
│                     │  │                      │
│  Scene Clustering   │  │  Rotation Averaging  │
│        │            │  │       │              │
│        ▼            │  │       ▼              │
│  Per-cluster:       │  │  Global Positioning  │
│   Star Init         │  │  (translations +     │
│   Resection + BA    │  │   points, rotations  │
│        │            │  │   held fixed)        │
│        ▼            │  │       │              │
│  Global Alignment   │  │       ▼              │
│  (5-stage merge)    │  │  Optional final BA   │
│        │            │  │                      │
│        ▼            │  │                      │
│  Final BA           │  │                      │
└────────┬────────────┘  └──────────┬───────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
         Export to MVS::Scene
         (dense reconstruction)
```

---

### Common Front-End

These steps are shared by both workflows.

#### 1. Feature Extraction (`FeaturesExtractor.h`)

Supported detectors:
- **AKAZE** (default): Fast binary descriptors, good for most cases
- **ORB**: Lighter weight, binary descriptors
- **SIFT**: Highest quality, float descriptors (slower)
- **SiftGPU**: CUDA-accelerated SIFT (optional)

The 3x3 grid extraction ensures features aren't concentrated in textured areas while ignoring featureless regions. Each cell targets up to 3000 features, giving ~27k features per image.

#### 2. Feature Matching (`PairsMatcher.h`, `MatchGeometric.h`)

Three matching strategies:
- **VOCABULARY** (recommended): Build a visual vocabulary tree, retrieve top-K similar images per query. O(N log N) instead of O(N²).
- **EXHAUSTIVE**: Match all pairs. Only practical for small datasets (<100 images).
- **SEQUENTIAL**: Match consecutive frames only. For ordered video sequences.

The matching pipeline:
1. **Descriptor matching**: FLANN (LSH for binary, KDTree for float) or brute-force
2. **Lowe's ratio test**: Keep match only if best/second-best distance ratio < 0.8
3. **Cross-check** (optional): Both images must agree on the match
4. **Geometric verification**: RANSAC to estimate E (calibrated) or F (uncalibrated) matrix
5. **Cheirality check**: Points must be in front of both cameras

#### 3. View Graph Calibration (`ViewGraphCalibrator.h`)

If camera intrinsics aren't fully trusted (no EXIF or imprecise calibration), this stage estimates focal lengths globally across all image pairs using the Fetzer method.

#### 4. Track Building & Filtering (`Track.h`)

Union-find merges matched features across all image pairs into tracks. Filtering removes:
- Tracks with too few observations
- Images with spatially clustered tracks (likely degenerate geometry)
- Images with small triangulation angles

---

### Hierarchical Workflow (`Scene::ReconstructHierarchical`)

The hierarchical workflow is the **recommended default**. It splits the scene into manageable clusters, reconstructs each independently using incremental SFM, and then stitches them together with global alignment. When the dataset is small enough to fit in a single cluster, it degrades gracefully to a pure incremental reconstruction -- so it works well for any scene size.

```
Input: Tracks + Image Pairs (from common front-end)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Scene Clustering                                      │
│    SceneCluster.h/cpp                                    │
│    Aggregative clustering on covisibility graph          │
│    Partition into clusters of ≤200 images                │
│    (if scene ≤ maxViewsPerCluster → 1 cluster = pure    │
│     incremental reconstruction, no alignment needed)     │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Per-Cluster Incremental Reconstruction                │
│    (each cluster runs independently, can be parallelized)│
│                                                          │
│    ┌───────────────────────────────────────────────┐     │
│    │ a. Star Initialization (StarInitializer.h)    │     │
│    │    Select most-connected reference view       │     │
│    │    Register multiple views simultaneously     │     │
│    │    Triangulate initial points                 │     │
│    │    Estimate global scale from median depths   │     │
│    └───────────────────┬───────────────────────────┘     │
│                        ▼                                 │
│    ┌───────────────────────────────────────────────┐     │
│    │ b. Incremental Resection (Resection.h)        │     │
│    │    For each unregistered image:               │     │
│    │      Find 2D-3D correspondences with tracks   │     │
│    │      Solve PnP + RANSAC (PoseLib)             │     │
│    │      Triangulate new points                   │     │
│    │      Local BA on nearby cameras               │     │
│    │      Periodic global BA to correct drift      │     │
│    └───────────────────┬───────────────────────────┘     │
│                        ▼                                 │
│    ┌───────────────────────────────────────────────┐     │
│    │ c. Bundle Adjustment (BundleAdjustment.h)     │     │
│    │    Final global BA with Ceres Solver           │     │
│    │    Refine: poses + points + intrinsics         │     │
│    │    Optional GPS position constraints           │     │
│    └───────────────────────────────────────────────┘     │
│                                                          │
└─────────────────────────────────────────────────────────┘
  │
  │  (skip if only 1 cluster)
  ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Global Alignment -- 5-stage merge                     │
│    GlobalAlignment.h/cpp                                 │
│                                                          │
│    a. Estimate relative poses between sub-scene pairs    │
│       (PoseLib generalized absolute pose from            │
│        cross-cluster 2D-3D correspondences)              │
│                        ▼                                 │
│    b. Rotation averaging (GlobalRotationAveraging.h)     │
│       MST init → L1-ADMM → IRLS on SO(3)                │
│                        ▼                                 │
│    c. Scale averaging (GlobalScaleAveraging.h)           │
│       Log-space least-squares on pairwise scale ratios   │
│                        ▼                                 │
│    d. Translation averaging (GlobalTranslationAveraging) │
│       Linear system solve with gauge constraint          │
│                        ▼                                 │
│    e. Merge sub-scenes into reference scene              │
│       Apply similarity transforms to each cluster        │
│       Average shared camera intrinsics                   │
│       Merge tracks via union-find + 3D proximity guards  │
│                                                          │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Final Bundle Adjustment (optional)                    │
│    Global BA on the merged scene                         │
│    Optional GPS alignment (SimilarityTransform.h)        │
└─────────────────────────────────────────────────────────┘
  │
  ▼
Output: Calibrated poses + sparse point cloud
```

**Key implementation details**:

- **Scene Clustering** (`SceneCluster.h`): Builds a covisibility graph (nodes = images, edges = inlier match counts), then uses bottom-up aggregative clustering. It merges the highest-weight edge at each step until all clusters have ≤ `maxViewsPerCluster` images (default 200). A refinement pass merges small clusters, moves boundary images for better modularity, and splits disconnected components.

- **Data protocol**: Keypoints and descriptors are **moved** (not copied) from the global scene into sub-scenes to save memory. Cross-cluster image pairs remain in the global scene for use during alignment. After merge, data is moved back.

- **Star Initialization** (`StarInitializer.h`): Instead of the classic two-view initialization (sensitive to baseline selection), OpenMVS uses a star configuration: the most-connected image becomes the reference, and multiple views are registered simultaneously. This averages over multiple baselines for a more stable initial estimate.

- **Bundle Adjustment** (`BundleAdjustment.h`): Uses Ceres Solver. **Local BA** optimizes a window of cameras + their points with fixed intrinsics (fast, used during resection). **Global BA** optimizes everything including intrinsics (slower, used at end). GPS constraints can be added when EXIF GPS data is available.

---

### Global Workflow (`Scene::ReconstructGlobal`)

The global workflow bypasses incremental reconstruction entirely. Instead of registering images one by one, it solves for all camera rotations and translations simultaneously using averaging algorithms. This is fundamentally different from the incremental approach.

```
Input: Tracks + Image Pairs (from common front-end)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 1. Compute Relative Poses                                │
│    Extract relative rotations and translation directions │
│    from E/F matrices in all verified image pairs         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Global Rotation Averaging                             │
│    GlobalRotationAveraging.h/cpp                         │
│                                                          │
│    Input: pairwise relative rotations R_ij               │
│    Solve: find global R_i for each image such that       │
│           R_ij ≈ R_j × R_i^T for all pairs              │
│                                                          │
│    Algorithm:                                            │
│      a. MST initialization (propagate from root)         │
│      b. L1 minimization (tangent space, angle-axis)      │
│      c. IRLS refinement (Geman-McClure robust loss)      │
│                                                          │
│    Output: global rotations for all images               │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Global Positioning                                    │
│    GlobalPositioning.h/cpp                               │
│                                                          │
│    Rotations are now FIXED.                              │
│    Solve for translations + 3D points simultaneously     │
│    using point-to-camera reprojection constraints        │
│                                                          │
│    Uses Ceres Solver with optional GPU acceleration      │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Optional Bundle Adjustment                            │
│    Refine all poses + points + intrinsics jointly         │
│    Optional GPS alignment                                │
└─────────────────────────────────────────────────────────┘
  │
  ▼
Output: Calibrated poses + sparse point cloud
```

**Key implementation details**:

- **Rotation averaging** operates on the SO(3) manifold. It parameterizes rotations in tangent space (angle-axis vectors) and solves a linear system. The IRLS refinement uses robust loss functions (Geman-McClure or Half-Norm) to downweight inconsistent pairs that are likely wrong matches.

- **Global positioning** treats rotations as fixed and solves only for translations and 3D point positions. This makes the problem linear (or nearly so), which is what gives the global approach its speed advantage. The downside is that any errors in the rotation averaging stage are baked in and cannot be corrected.

- **No incremental registration**: Images are not added one at a time. All poses are estimated in one shot. This means there's no opportunity for the system to detect and reject problematic images during reconstruction.

---

### Choosing Between Workflows

| Aspect | Hierarchical | Global |
|--------|-------------|--------|
| **Speed** | Slower -- runs full incremental SFM per cluster, plus alignment | Faster -- solves rotations and translations in closed form |
| **Robustness** | More robust -- incremental registration can detect and reject bad images; BA continuously corrects drift | Less robust -- relies on pairwise relative poses being correct; a few bad pairs can corrupt the entire solution |
| **Difficult scenes** | Handles well -- star initialization and incremental resection adapt to varying baselines, repeated structures, and challenging geometry | Can fail -- rotation averaging may not converge if many pairs have wrong relative poses (e.g., repetitive textures, symmetric structures) |
| **Completeness** | Higher -- incremental approach tries hard to register every image, running BA after each addition | Lower -- images that don't fit the global solution are simply lost; no mechanism to retry or adjust |
| **Scalability** | Excellent -- clustering + parallel reconstruction handles 1000+ images; memory bounded by cluster size | Good for medium scenes -- the global solve is O(N) but can struggle numerically with very large systems |
| **Small scenes** | Works well -- with 1 cluster, degrades to pure incremental reconstruction | Works well -- fastest option for well-connected small datasets |
| **Drift** | Controlled -- periodic global BA during resection prevents drift accumulation | No drift -- all poses solved simultaneously (but errors are global rather than local) |

**Practical guidance**:

- **Start with hierarchical** (`Scene::ReconstructHierarchical`). It's the safer default. For small scenes (< 200 images) it automatically runs as a single incremental reconstruction with no clustering overhead.
- **Try global** (`Scene::ReconstructGlobal`) when you need speed and your dataset is well-connected with reliable matches (e.g., drone surveys with good overlap, indoor scans with distinctive features). If the global result has missing or misaligned cameras, fall back to hierarchical.
- **Global is not "better hierarchical"**. The two approaches have fundamentally different failure modes. Hierarchical fails gracefully (some images may not register, but the rest are correct). Global can fail catastrophically (a few bad rotations corrupt the entire solution).

## External Format Integration

| Module | Format | Direction |
|--------|--------|-----------|
| `ImportCOLMAP.h` | COLMAP binary | Import cameras, poses, tracks |
| `ImportROMA2.h` | ROMA2 .npz | Import robust matches + depth |
| `InterfaceMVS.h` | OpenMVS .mvs | Export to MVS pipeline |
| Scene::ExportPLY | PLY | Export sparse point cloud |

## Usage Examples

### Keyframe Extraction from Video

```cpp
KeyframeConfig config;
config.detectorType = "AKAZE";
config.overlapThreshold = 0.8f;     // 80% overlap between keyframes

Scene scene;
KeyframeExtractor::ExtractFromVideo("video.mp4", config, scene);
```

### Full Reconstruction

```cpp
Scene scene;
// ... load images, extract features ...

// Match using vocabulary tree
VocabularyTree vocab;
vocab.Build(scene);
PairsMatcher::MatchScene(scene, matchConfig);

// Build tracks and reconstruct
BuildTracks(scene);
StarInitializer().Initialize(scene);
// ... incremental resection + BA ...

// Export to MVS format
ExportToMVS(scene, mvsScene);
```

## Performance Considerations

| Strategy | Complexity | Best for |
|----------|-----------|----------|
| Vocabulary matching | O(N log N) | Default for most datasets |
| Exhaustive matching | O(N²) | Small datasets (<100 images) |
| Sequential matching | O(N) | Ordered video frames |
| Scene clustering | Enables parallel reconstruction | 200+ images |
| Local BA | O(window size) | During incremental registration |
| Global BA | O(all cameras + points) | Final refinement |

**Memory**: Lazy image loading (`LoadPixels()` / `ReleasePixels()`) and feature data movement (not copy) during clustering keep memory usage bounded.

## File Organization

```
libs/SFM/
├── Common.h/cpp                        # Library init
│
│ # Core data structures
├── Camera.h/cpp                        # Pinhole + Spherical camera models
├── Pose.h/cpp                          # 3D pose (R, C)
├── View.h/cpp                          # Pose + Camera reference
├── Image.h/cpp                         # Features, descriptors, metadata
├── ImagePair.h/cpp                     # Pairwise matches and geometry
├── Track.h/cpp                         # 3D tracks + union-find builder
├── Scene.h/cpp                         # Central container
│
│ # Feature pipeline
├── FeaturesExtractor.h/cpp             # AKAZE/ORB/SIFT extraction
├── VocabularyTree.h/cpp                # Visual vocabulary for retrieval
├── PairsMatcher.h/cpp                  # Matching strategies
├── MatchGeometric.h/cpp                # RANSAC geometric verification
├── PairsWeighting.h/cpp                # Composite pair quality scores
│
│ # Reconstruction
├── StarInitializer.h/cpp               # Star-config initialization
├── Resection.h/cpp                     # Incremental PnP registration
├── Triangulation.h/cpp                 # Multi-view triangulation
├── BundleAdjustment.h/cpp              # Ceres-based optimization
├── BundleAdjustmentCostFunctions.h     # Reprojection error residuals
├── ViewGraphCalibrator.h/cpp           # Global focal estimation
├── RelativePoseRefine.h/cpp            # Two-view calibration refinement
│
│ # Hierarchical / Global reconstruction
├── SceneCluster.h/cpp                  # Aggregative scene clustering
├── GlobalAlignment.h/cpp               # 5-stage sub-scene merging
├── GlobalRotationAveraging.h/cpp       # SO(3) rotation averaging
├── GlobalScaleAveraging.h/cpp          # Log-space scale averaging
├── GlobalTranslationAveraging.h/cpp    # Linear translation solving
├── GlobalPositioning.h/cpp             # Translation refinement
├── SimilarityTransform.h/cpp           # 7-DOF transform + GPS alignment
│
│ # Video support
├── KeyframeExtractor.h/cpp             # Keyframe selection from video
│
│ # External format support
├── ImportCOLMAP.h/cpp                  # COLMAP import
├── ImportROMA2.h/cpp                   # ROMA2 match import
└── InterfaceMVS.h/cpp                  # MVS format export
```

## Conventions

- **Namespace**: `SFM` (separate from `MVS`)
- **Coordinate system**: Right-handed, X right, Y down, Z forward
- **Pose**: `P = KR[I|-C]`, R is world-to-camera, C is camera center in world
- **Pixel origin**: Integer coordinates at pixel center, (-0.5, -0.5) at top-left corner
- **Thread pool**: `Scene::threadPool` (`BS::light_thread_pool`) for parallel algorithms

## Dependencies

- **Common, Math, IO** (required): OpenMVS internal libraries
- **Ceres Solver** (required): Bundle adjustment
- **PoseLib** (required): Pose estimation (E/F/H matrices, PnP)
- **TinyEXIF** (required): EXIF metadata parsing
- **TinyNPY** (required): NumPy .npz file I/O (ROMA2 support)
- **OpenCV** (inherited): Feature detection, matching, image I/O
- **Eigen3** (inherited): Linear algebra
- **Boost** (inherited): Serialization
- **SiftGPU** (optional): GPU-accelerated SIFT
