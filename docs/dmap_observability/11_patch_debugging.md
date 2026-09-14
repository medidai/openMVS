# Patch debugging: current capability and required extension

This note defines what the public depth-map observability tooling can establish
about PatchMatch patches today, what can be reconstructed as a bounded offline
diagnostic, and what must be implemented before the report can display literal
CUDA patch samples. It applies only to depth-map estimation and filtering.

## Current capability

The current deep and trace profiles explain the effects of a patch or
photometric-scoring change. For matched baseline and candidate runs, the report
can synchronize the following available evidence by frame, pyramid level,
complete logical iteration, source view, and pixel:

- reference RGB context and exact reference-patch variance;
- raw photometric, prior-adjusted photometric, geometric, and total costs;
- exact per-view costs, weights, decisions, and weighted contributions;
- finite, accepted, and winning candidate state plus winner/runner-up gap;
- update attribution, selected-view masks, view churn, depth change, and normal
  change;
- final production annotation residuals and coverage when annotations are
  configured.

New captures also write the independently versioned fixed observer-layout
contract `openmvs.dmap.reference_patch_layout` into `run_metadata.json` and the
frame summary. The report combines that exact configuration with the exact
per-level `width` and `height` from the CUDA resource plan. The resulting
reference-grid positions are labeled `derived_exact`: they are deterministic
geometry derived from captured contracts, not locations emitted sample by
sample by the CUDA kernel. A focused source test keeps the observer layout
constants aligned with the production CUDA scoring constants.

Public v1 retains the complete exact cost-component and per-view map family at
pyramid level 0. Coarse levels expose counters, timings, resource contracts, and
selected compatibility evidence; unavailable cost maps remain explicit, and
coarse trace evidence must retain its exact or proxy quality label.

The targeted trace additionally records retained-state cost and geometry
transitions, winning update source, selected views, per-view aggregate costs and
weights, neighbor costs, and bad-score reason counts across initialization and
complete logical iterations. Checkerboard phases remain timing-only.

Open the interactive report, select matched deep runs, and use **Guide -> Debug
a patch or photometric-scoring change -> Set up patch inspection**. Keep **Same
logical iteration**, **Shared** display scale, and **Derived patch grid** enabled
while comparing runs. Click any synchronized map. The reference RGB row shows
the full-frame positions and **Reference patch layout** opens a baseline/variant
loupe. Blue squares identify baseline positions, red circles identify variant
positions, cyan identifies the selected pixel, and yellow rings identify samples
clamped to an image edge. OpenMVS requests wrap addressing, but CUDA makes clamp
effective for its unnormalized texture coordinates. Trace at least one low-texture
interior, boundary, improvement, regression, and stable control pixel.

For an existing experiment, a reproducible targeted trace has this form:

```bash
tools/dmap_observability.sh drilldown \
  --config /path/to/experiment.yaml \
  --scene SCENE_ID \
  --frame IMAGE_ID \
  --pixel X,Y \
  --variant CANDIDATE_LABEL \
  --execute \
  --refresh-report \
  --report-dir /path/to/reports
```

Public-v1 trace targeting selects the pixels presented in the report but still
runs and retains the selected frame's full deep `Process<true>` payload. Budget
it as a deep frame capture, not as a compact pixel-only execution.

## Reading the current evidence

Use the evidence chain rather than interpreting a cost map in isolation:

1. A reference-variance change between implementations indicates changed
   reference support or weighting.
2. A raw photometric-cost change with stable reference variance points to target
   projection, sampling, or cost-function behavior.
3. Stable raw cost with changed prior-adjusted cost isolates textureless or
   depth-prior behavior.
4. A change confined to one source view identifies a view-specific projection
   or appearance problem.
5. Lost finite candidates indicate projection, boundary, or numerical failures.
6. Candidate gaps and accepted masks show whether score changes alter ordering.
7. Depth/normal changes and production annotation residuals determine whether
   the scoring change improves geometry.

For the current fixed patch, the reference-side sample layout is known from the
CUDA implementation:

```text
x offsets = {-4, -2, 0, 2, 4}
y offsets = {-4, -2, 0, 2, 4}
samples   = 25 Cartesian-product locations
```

At pyramid level 0 this spans a 9 by 9 reference-image region. The locations are
derived exactly in that level's coordinate system from the declared layout and
grid extent. Coarser levels use their own recorded resource-plan dimensions;
the UI does not infer dimensions by a power-of-two shift. Older captures that
lack the layout contract show it as unavailable rather than assuming 25 samples.

The loupe background is a report-owned RGB thumbnail. It is qualitative scene
context, not the CUDA float grayscale pyramid texture. The markers do not
contain sampled intensities, bilateral weights, target projections, residuals,
or score contributions. A fixed reference grid is not evidence of the warped
source footprint and is invalid as a representation of a deformable patch.

## Evidence that is not captured

The current trace schema does not retain:

- retained plane normal/depth as a complete plane record at every trace state;
- per-sample locations emitted by the production or diagnostic CUDA kernel;
- projected source coordinates or source-view footprints;
- sampled reference and source intensities;
- bilateral weights, sample-domain status, or actual texture-address result;
- weighted covariance/variance terms or per-sample score contributions;
- per-state homographies or source-image crops.

Consequently, the report can identify where, when, and in which source view a
patch-related score changes. It cannot prove which samples crossed another
surface or reconstruct the CUDA ZNCC score sample by sample. Generic registry
signals such as patch deformation, valid-sample fraction, and patch score delta
remain explicitly unavailable until an extension producer supplies them.

## Bounded offline workaround

For the final retained state only, a separate diagnostic can reconstruct the
fixed footprint from the frozen scene, terminal DMAP depth/normal, cameras, and
source images. Such a tool can display reference/source crops and homography-
projected sample locations without changing the CUDA estimator.

This evidence must be labeled `derived` or `reconstructed`, not captured. It
does not recover intermediate logical states, the exact texture fetches, or the
sample values used by the production scoring call. It is useful for selecting
trace pixels and visually checking final-state boundary crossings, but it is not
a substitute for the extension below.

## Required literal patch-trace extension

The recommended implementation is a trace-only retained-winner capture. It must
never be a full-frame per-sample map. Capture initialization and the state after
each complete logical iteration; do not expose black/red phases except in
timings.

### CUDA capture

Add a compact diagnostic capture for explicitly selected trace pixels. For each
retained pixel state, record:

- patch layout identifier and version, radius, step, and sample count;
- trace identity, pyramid dimensions, retained plane, and selected-view mask;
- each reference offset, coordinate, intensity, bilateral weight, and weighted
  intensity;
- for every runtime source view, its source image identity, selection state,
  weight, bad reason, homography, aggregate score terms, and recorded cost;
- each raw projected coordinate, sampled intensity, coordinate-domain status,
  and weighted covariance/variance terms required to reconstruct ZNCC.

A separate kernel after initialization and after the final checkerboard phase of
each iteration minimizes changes to the PatchMatch hot kernel. Its measurement
basis should be labeled `retained_winner_diagnostic_rescore`, with an explicit
comparison against the retained production per-view cost. Capturing the values
directly from every production candidate evaluation is a separate, substantially
more invasive tier because it changes the hottest scoring path and multiplies
storage by every rejected hypothesis.

The capture must reuse the scoring implementation's patch definition and
sampling semantics. CUDA image textures currently use linear filtering and a
configured address mode, so the artifact must retain both the raw projected
coordinate and the actual sampling/address semantics. An `in_bounds` flag alone
does not describe the value returned by the texture unit.

### Schema and resource contract

Introduce an independently versioned `openmvs.dmap.patch_trace` contract rather
than expanding each aggregate trace row with an unbounded nested payload. A
small JSON index plus bounded compact numeric arrays avoids excessive JSONL
size. The resource plan must account for device, host, and storage bytes before
allocation and obey the existing `error` or `degrade` budget policy.

The schema must support variable sample counts and named layouts from its first
version so adaptive and deformable patches do not require reinterpreting the
fixed 25-sample contract. Older captures must report patch samples as
unavailable, never as an empty or zero-valued patch.

Full-frame capture is not acceptable. Even a compact 32-byte sample record,
25 samples, eight views, six logical states, and a 640 by 480 frame would exceed
10 GiB before report previews. Selected-pixel capture keeps typical records in
the kilobyte range and remains compatible with trace-row and storage limits.

### Report interface

Extend the current derived reference-layout loupe into a literal Patch Inspector
inside the completed targeted-trace panel with:

- baseline and candidate reference/source crops shown side by side;
- logical-state and source-view controls linked to the report's shared state;
- exact sample-point and footprint overlays, not only a bounding rectangle;
- coloring by weight, intensity difference, or normalized score contribution;
- sample-domain and bad-score annotations;
- recorded versus reconstructed ZNCC and tolerance status.

Source previews must be report-owned, deduplicated by source-image identity,
coordinate-aligned with the captured pyramid, and included in the report
inventory and closure.

### Validation and release criteria

The extension is complete only when tests cover:

- exact trace/state/view/sample cardinality and identity;
- a synthetic identity warp with zero photometric disagreement;
- projected coordinates and texture-address behavior at image boundaries;
- host reconstruction of the recorded weighted ZNCC within a declared float32
  tolerance;
- malformed, truncated, duplicate, oversized, and non-finite payload rejection;
- storage-budget `error` and `degrade` behavior;
- absence of patch allocation, kernels, and output when tracing is disabled;
- unchanged production endpoint DMAPs and existing parity qualification;
- all pyramid levels and photometric/geometric estimation stages;
- nonblank reference/source crops, overlays, selectors, and score validation in
  the Firefox UI test.

A narrow retained-winner prototype is approximately two to three engineer days.
Production-quality schema, budgets, validators, portable report integration,
UI, documentation, CUDA parity, and browser validation are approximately seven
to ten engineer days. Capturing every tested candidate and rejected proposal is
a separate two-to-three-week investigation.
