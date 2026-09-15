# DVP epipolar proposals

This page documents the depth-variation proposal (DVP) families implemented in
CUDA PatchMatch and the observability needed to debug them. The feature is
disabled by default and runs only during geometric-consistency estimation. It
does not instrument or modify fusion, meshing, refinement, or texturing.

## Fidelity boundary

The `dvp_eq11_interval_v1` family is an independent OpenMVS adaptation of the
DVP-MVS depth-aggregation mechanism. Its scoped claim is
`paper_mechanics_complete_openmvs`, not exact author-code equivalence. The
contract is derived from DVP-MVS Eq. 11 and its supplementary explanation, with
the paper parameters `alpha=1`, `beta=4`, and `mu=3` as defaults.

The implementation intentionally uses OpenMVS cameras, selected source views,
active cost objective, candidate ordering, and native normal refinement. The
following later DVP-MVS mechanisms are outside this component and must not be
inferred from the label:

- depth-edge aligned priors;
- persistent or restored visibility;
- visible-normal constraints;
- DVP-MVS++ priors and scoring changes.

The official DVP-MVS repository and a historical donor implementation are
provenance references, not executable correctness oracles. Their exact commit
identities are emitted in every DVP capture. CPU and CUDA contract fixtures are
the executable arithmetic oracles.

## Families

| ID | Family | Mechanism | Promotion status |
|---:|---|---|---|
| `0` | `disabled` | Native OpenMVS depth refinement | default |
| `1` | `historical_global_v0` | Bounded single-view global epipolar search compatible with the historical experiment | diagnostic only |
| `2` | `global_search_gated_v1` | Global search with round-trip, relative-depth, occlusion, and multi-view support gates | eligible for evaluation |
| `3` | `historical_midpoint_v1` | Historical endpoint ordering and midpoint proposals while retaining native depth perturbation | diagnostic only |
| `4` | `dvp_eq11_interval_v1` | Correct Eq. 11 endpoint statistics and interval sampling that replaces native depth perturbation when available | eligible for evaluation |

Families 1 and 3 preserve historical behavior only to decompose earlier
results. Never present them as paper-faithful DVP.

## Eq. 11 mechanics

For each pixel and selected source view, the active reference depth is projected
to the source image. A local epipolar direction is estimated using nearby depths
on the same reference ray. Source depth is sampled at four locations:

```text
left outer   = -(alpha + beta)
left inner   = -alpha
right inner  = +alpha
right outer  = +(alpha + beta)
```

Each valid source sample is back-projected and converted to reference-camera
depth. For order `mu`, the two possible aggregate intervals are:

```text
left  = [mu-th smallest(left outer), mu-th largest(left inner)]
right = [mu-th largest(right inner), mu-th smallest(right outer)]
```

Each side is admitted independently. A side requires at least `mu` samples in
both its outer and inner endpoint groups and a strictly increasing numeric
interval. A valid left interval remains usable when the right side is
under-supported or invalid, and vice versa.

One sample is drawn from each valid interval using a fork of the per-pixel RNG
stream. The persistent stream advances by exactly the one draw that native depth
perturbation would have consumed, regardless of whether one or two intervals are
valid. Later normal and random-normal proposals therefore remain aligned with
the control RNG schedule. The resulting depths retain the incumbent normal and
are scored in deterministic left-then-right order using the active OpenMVS
objective. Strictly lower cost replaces the incumbent. When at least one Eq. 11
interval exists, these interval candidates replace native fixed depth
perturbation for that pixel; native normal, random-normal, and surface-normal
candidates still run. When neither interval exists, native depth perturbation is
the explicit fallback.

`support=mu` on an interval proposal means the order-statistic support contract.
The exported endpoint-view mask is the intersection of the two endpoint groups
and may contain fewer than `mu` bits because the qualifying order statistics do
not have to come from the same source views.

## Runtime configuration

The common DVP options are available on the CUDA densifier:

```text
--patch-match-cuda-dvp-epipolar-family 0..4
--patch-match-cuda-dvp-epipolar-alpha FLOAT
--patch-match-cuda-dvp-epipolar-beta FLOAT
--patch-match-cuda-dvp-epipolar-mu UINT
--patch-match-cuda-dvp-global-search-radius UINT
--patch-match-cuda-dvp-reprojection-threshold FLOAT
--patch-match-cuda-dvp-relative-depth-threshold FLOAT
```

Family `0` is the default. An enabled family requires
`--geometric-iters` greater than zero. Invalid family, offset, support, search,
or threshold values fail during CLI initialization.

A production endpoint run for the Eq. 11 arm is typically:

```bash
build-dmap-production/bin/DensifyPointCloud \
  --working-folder /path/to/variant/work \
  --input-file /path/to/variant/work/scene.mvs \
  --output-file /path/to/variant/dense.mvs \
  --geometric-iters 1 \
  --patch-match-cuda-dvp-epipolar-family 4 \
  --patch-match-cuda-dvp-epipolar-alpha 1 \
  --patch-match-cuda-dvp-epipolar-beta 4 \
  --patch-match-cuda-dvp-epipolar-mu 3 \
  <all frozen baseline arguments>
```

Run a separate family-0 endpoint with every other argument and input identity
matched. The production executable is the quality and timing authority.

## Observability tiers

DVP observability uses global frame-summary schema v5, nested DVP schema v1,
and nested APD schema v5 for combined APD+DVP captures. APD v5 preserves the
v4 state/update binary record sizes but admits DVP candidate slots `22/23` and
update source `12` in complete active-objective accounting. DVP-specific
interval, support, and proposal evidence remains in its independent namespace.

| Tier | Retained DVP evidence | Interpretation |
|---|---|---|
| `summary` | exact per-logical-iteration counters and double-precision aggregate sums | diagnostic `Process<true>` mechanics, not endpoint quality |
| `maps` | 56 full-frame DVP mechanics maps per logical iteration | exact active-path proposal, interval, cost, source, support, and outcome state |
| targeted trace | per-view endpoint depths and identities plus proposal decisions for configured pixels | exact selected-pixel drilldown within the diagnostic specialization; not production-causal |

Summary and maps report complete logical iterations. Checkerboard phases remain
timing-only.

All `exact` labels in these tiers describe the executed `Process<true>`
diagnostic path. `OBS-001` remains open: the separately compiled diagnostic
specialization can produce a different terminal DMAP from production
`Process<false>`. Treat its counters, maps, and traces as internally exact
mechanics evidence, but do not use them to assert the production winner or as a
quality endpoint. Production `Process<false>` runs remain the quality and timing
authority.

The principal fields are:

- proposal availability and native-fallback counts;
- selected and valid-direction source-view masks;
- four endpoint support counts;
- left/right interval bounds and validity;
- proposal depth, side, source view, support, support-view mask, and occlusion
  mask;
- incumbent, proposal, winner, runner-up, gap, and improvement costs;
- sequential acceptance, final update source, final DVP winner, and exact final
  depth retention;
- prioritized unavailable reason;
- mean/max gated-global reprojection and relative-depth errors.

An improvement mean in the summary is averaged over every attempted pixel,
including zero improvement where no proposal wins. Use maps or traces for
proposal-conditioned analysis.

Per-iteration event counts and support-count sums use unsigned 32-bit device
atomics. A single logical iteration must therefore remain below 134,217,728
attempted pixels for the worst-case 32-view support sums to be representable.
The default `max-resolution=2560` is well below this boundary; unusually large
single-frame captures must be rejected or partitioned instead of interpreting a
wrapped aggregate. Float-valued sums use double-precision device atomics.

## Capture and validate

Use the observer binary only for mechanics:

```bash
build-dmap-observer/bin/DensifyPointCloudDMapObserve \
  --working-folder /path/to/deep/work \
  --input-file /path/to/deep/work/scene.mvs \
  --output-file /path/to/deep/dense.mvs \
  --geometric-iters 1 \
  --patch-match-cuda-dvp-epipolar-family 4 \
  --dmap-instrumentation-dir /path/to/deep/instrumentation \
  --dmap-instrumentation-level maps \
  --dmap-instrumentation-image-list 17 \
  --dmap-instrumentation-write-maps 1 \
  --dmap-instrumentation-budget-policy error \
  <all frozen baseline arguments>
```

Add `--dmap-instrumentation-config /path/to/trace_config.json` for targeted
endpoint records. Validate every selected frame:

```bash
python3 scripts/python/validate_dmap_instrumentation.py \
  --frame-dir /path/to/instrumentation/stage/depthmaps/0017_frame \
  --output /path/to/validation.json
```

Success requires `valid: true`, `dvp_validation.available: true`, zero domain
errors, counter/map closure, the expected 56 maps per logical iteration, and a
valid targeted trace when one was requested.

## Report workflow

1. Compare family 0 and the candidate at the production endpoint. Rank accuracy
   and tail noise before coverage.
2. Use summary evidence to compare proposal availability, fallback, acceptance,
   retention, unavailable reasons, support, and runtime by logical iteration.
3. Sort endpoint regressions and improvements, then select representative
   frames for matched maps capture.
4. In the HTML investigation interface, select the same stage, pyramid level,
   and logical iteration for both arms and use the DVP mechanics preset.
5. Synchronize RGB, interval-validity, proposal-depth, candidate-cost,
   acceptance, final-source, and retained-depth maps.
6. Open targeted traces at accepted, rejected, fallback, and stable-control
   pixels. Inspect endpoint identities before interpreting the interval.
7. Return to production annotation metrics to decide whether the mechanism
   improves the goal. Diagnostic maps alone cannot make a quality claim.

Useful failure patterns include:

- high fallback with low endpoint counts: insufficient geometric/source-depth
  evidence;
- high interval validity but low finite proposals: interval or depth-range
  mismatch;
- finite proposals with low acceptance: no active-cost advantage;
- accepted proposals with low final retention: later normal/depth refinement
  replaces the DVP hypothesis;
- high final depth retention but poor endpoint accuracy: the proposal is causal
  but geometrically wrong;
- gated-global occlusion/support rejection: inspect the per-view masks and
  error thresholds before changing the cost function.

## Qualification

For any DVP producer or schema change, run:

```bash
build-dmap-production/bin/Tests --help
build-dmap-observer/bin/Tests --help
python3 -m pytest -q scripts/python
node --test scripts/dmap_report_ui/test_capture_profile_coverage.js
```

Then perform fresh CUDA qualification on the exact binaries:

1. family-0 production output versus the frozen oracle;
2. repeated family-0 and enabled-family DMAP hashes;
3. observer all-off parity against its same-compile-form oracle;
4. `initcheck` and `memcheck` for both promotion-eligible families in
   production and observer builds;
5. sanitized versus unsanitized DMAP hash equality;
6. one real maps/trace capture, schema validation, report generation, and
   browser inspection;
7. disabled lifecycle checks for observer artifacts, diagnostic kernels,
   optional symbols, allocations, and device-state changes.

The CUDA oracle validates project/back-project arithmetic, Eq. 11 statistics,
one-sided support, interval sampling bounds, the control-equivalent RNG schedule,
and candidate ranking on device.
Compute Sanitizer and real captures remain mandatory because the oracle alone
does not exercise full-frame memory ownership or production scheduling.
