# Experimental APD/DVP depth estimation

## Scope

This branch retains the tested Adaptive Patch Deformation (APD) implementation
and gated global epipolar proposals inspired by DVP-MVS, on the stabilized
OpenMVS 2.4 compatibility base. All experimental modes are disabled by default.
It is not a complete or unmodified reproduction of either paper.

APD selects reliable support anchors and adapts local plane scoring. The
retained DVP component adds geometrically gated epipolar proposals during
geometric-consistency iterations. Historical depth-edge, persistent-visibility,
visible-normal and other proposal variants remain available for research, but
are not part of the recommended experimental configuration.

## Build

Use the normal OpenMVS dependencies and build procedure with CUDA enabled.
Keep `OpenMVS_DMAP_INSTRUMENTATION=OFF` for production-path comparisons. Enabling
that separate build option creates the diagnostic `DensifyPointCloudDMapObserve`
target; diagnostic captures must not be substituted for production results.

For focused CPU contract checks, build with `OpenMVS_BUILD_SFM=OFF` and
`OpenMVS_ENABLE_TESTS=ON`, then run:

```sh
ctest --test-dir build-release -R '^MVSConfidenceCompat23Test$' --output-on-failure
python3 -m unittest scripts.python.test_apd_dvp_release_defaults
```

The native contract test also checks APD/DVP numerical helpers. A CPU-only
build cannot validate CUDA execution or reconstruction quality.

## Opt-in configuration

Use a fresh output directory and a calibrated local input scene. Preserve SfM
inputs and use the same resolution, neighbours, iterations, postprocessing and
compatibility settings in both comparison arms.

Add these options to the normal `DensifyPointCloud` command to enable the
retained candidate:

```text
--patch-match-cuda-apd 1
--patch-match-cuda-dvp-epipolar-family 2
--patch-match-cuda-dvp-epipolar-alpha 1
--patch-match-cuda-dvp-epipolar-beta 4
--patch-match-cuda-dvp-epipolar-mu 3
--patch-match-cuda-dvp-global-search-radius 160
--patch-match-cuda-dvp-reprojection-threshold 2
--patch-match-cuda-dvp-relative-depth-threshold 0.01
--patch-match-cuda-dvp-depth-edge-mode 0
--patch-match-cuda-dvp-visibility-mode 0
--patch-match-cuda-dvp-visible-normal-mode 0
```

For the native control, set APD and the epipolar family to `0` and leave the
other mechanism modes at `0`. Omitting all experimental options also keeps
them disabled. No experiment is launched automatically.

The development comparison additionally used `--patch-match-cuda-compat-23 1`,
`--roi-compat-23 1`, and `--dmap-intermediate-float-components 2` in **both**
arms. These compatibility options are explicit experiment settings, not changes
to production defaults. Do not compare against an arm with different filtering
or image support and attribute the difference solely to APD/DVP.

## Evidence and limitations

- The retained integration passed focused CPU/CUDA contracts, sanitizer checks,
  and exact default-off parity on a small four-view fixture before publication.
  Fixture parity does not establish full-scene parity.
- A matched con-ds development comparison passed its frozen consistency checks.
  It covers one annotated frame with 11 edges, with most aggregate improvement
  concentrated in one edge. This is fitted geometric consistency, not physical
  ground-truth accuracy or proof of broad dataset superiority.
- The candidate took 8.30 times native runtime and slightly reduced full-scene
  valid-depth coverage in that comparison. Keep it opt-in.
- Eval-v4 qualification was not completed and was explicitly deferred for this
  source publication. Do not describe the two-dataset release gate as passed.
- The comparison measures APD plus retained DVP against native, not the isolated
  contribution of DVP. Private inputs, measurements, generated evidence and
  binaries are intentionally not distributed in this source branch.
- Source ancestry is the tested stabilization tip, not the newest develop.
  Integrating subsequent upstream changes requires separate validation.
- Publication preserves the base's container-memory-limit handling, which was
  absent from the development source snapshot. APD/DVP numerical code is
  unchanged; the published build has not received a new full-scene GPU run.

## Rollback

Disable the experimental modes or use the stabilized parent revision. Do not
reuse depth-map caches across changed configurations when comparing quality.
