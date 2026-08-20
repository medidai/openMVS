# Annotation consistency metrics

Optional 2D line and plane annotations provide end metrics for reconstructed
depth. They measure whether depth samples inside the annotated structure form a
consistent 3D line or plane. They are not absolute depth ground truth and cannot
detect a perfectly planar surface at the wrong global depth by themselves.

## Generic sidecar

Set `annotation_sidecar` on a scene. The JSON object uses this contract:

```json
{
  "schema_name": "openmvs.dmap.annotation_sidecar",
  "schema_version": 1,
  "scene_id": "scene-a",
  "image_mapping": {
    "frame-a": "images/0001.jpg"
  },
  "frames": [
    {
      "id": "frame-a",
      "imageResolution": {"width": 1920, "height": 1080}
    }
  ],
  "annotations": {
    "controlEdges": [
      {
        "id": "edge-a",
        "chunks": [
          {
            "id": "edge-a-0",
            "frameId": "frame-a",
            "start": {"x": 100.0, "y": 200.0},
            "end": {"x": 500.0, "y": 200.0}
          }
        ]
      }
    ],
    "controlPlanes": [
      {
        "id": "plane-a",
        "chunks": [
          {
            "id": "plane-a-0",
            "frameId": "frame-a",
            "points": [
              {"x": 700.0, "y": 300.0},
              {"x": 1100.0, "y": 300.0},
              {"x": 1100.0, "y": 700.0},
              {"x": 700.0, "y": 700.0}
            ]
          }
        ]
      }
    ]
  },
  "camera_final": {"width": 1920, "height": 1080}
}
```

For `evaluation.annotation_space: final`, coordinates are expressed in the
`camera_final` image grid and rescaled to the DMAP grid. For distorted source
coordinates, provide `camera_distorted` with a supported camera model and
parameters, and set the annotation space accordingly.

Every `frameId` must occur in `frames` and `image_mapping`. Mapped image names
are matched by basename against the DMAP image name. Sidecar `scene_id` must
equal the experiment scene ID. Malformed or unmatched structures are reported,
not silently dropped.

## Evaluation

For a plane polygon, valid depth pixels inside the polygon are unprojected to
3D with the DMAP camera. Deterministic RANSAC estimates a plane, followed by a
fit on its inliers. A line uses valid pixels in a configurable ribbon around the
annotated segment and fits a 3D line in the same manner.

Run multiple distance thresholds. A single threshold can hide whether a change
slightly tightens noise or merely moves points across one cutoff.

## Metrics

| Metric | Definition | Direction |
|---|---|---|
| valid coverage | valid depth pixels divided by annotated mask pixels | higher |
| spatial coverage | occupied valid grid cells or line bins divided by annotated cells/bins | higher |
| inlier fraction at T | valid samples with orthogonal residual at most `T`, divided by valid samples | higher |
| effective inlier coverage at T | valid coverage multiplied by inlier fraction at `T` | higher |
| residual median | median orthogonal distance of all valid samples to the fitted model | lower |
| residual P95 | 95th percentile orthogonal distance of all valid samples | lower |
| residual RMSE | root mean square orthogonal distance of all valid samples | lower |
| threshold AUC | normalized area under the inlier-fraction curve across configured thresholds | higher |

`T` is a 3D metric distance in scene units, normally metres. Coverage is not
defined relative to an error threshold unless the metric name explicitly says
`inlier` or `effective inlier`. Plain valid coverage only asks whether a usable
depth exists in the annotated region.

## Accuracy-first interpretation

Compare residual P95, threshold AUC, and tight-threshold inlier fraction before
coverage. Then require coverage to remain acceptable. A candidate that gains
coverage by adding noisy or cross-surface depth is not an accuracy improvement.

Inspect per-structure distributions and model stability. A new RANSAC model can
fit a different physical surface and still report a low self-residual. The
report therefore compares fitted normal/direction, position, line extent, and a
fixed baseline model evaluated on candidate points where available.

## Limitations

- Metrics describe local geometric self-consistency, not absolute accuracy.
- A broad annotation that spans multiple surfaces violates the model.
- Line ribbons can collect background or foreground samples near occlusions.
- Sparse valid depth can achieve low residual with poor spatial coverage.
- Camera or image mapping errors invalidate the result.
- Synthetic demo annotations are interface examples, not benchmark labels.
