# Report guide

The report is designed for investigation, not just scorekeeping. Begin with
production quality, identify where behavior changed, then descend into
diagnostic mechanics.

## Canonical outputs

- `01_development_report.md`: human-readable canonical report.
- `report_model.json`: structured source for investigation clients.
- CSV and Parquet tables: frame, iteration, annotation, timing, availability,
  and comparison records.
- `02_investigation.html`: interactive client over the validated model.
- `visualizations/`: deterministic plots, overlays, heatmaps, and panels.
- `report_tree_closure.json`: exact SHA-256/size/mode inventory of the final
  report tree, excluding only itself and the mutable Markdown validation sidecar.

Validate Markdown and structured references before opening the UI:

```bash
tools/dmap_observability.sh validate \
  --report /path/to/reports/01_development_report.md
```

## Confirm capture coverage first

The status area and **Capture profile coverage** matrix answer a provenance
question before any metric is interpreted: which of `endpoint`, `summary`,
`prefilter`, `deep`, and `trace` was requested and validated for each configured
run, repeat, and scene. The column header reports complete versus expected
capture units and identifies the process specialization and quality authority.
The cells preserve four producer states:

- **Complete:** the profile-specific completion and evidence contract validated.
- **Failed:** capture was attempted but did not complete or validate; read the
  reason before interpreting downstream omissions.
- **Unavailable:** the producer explicitly could not supply the requested
  evidence, with a reason.
- **Not requested:** the experiment deliberately omitted the profile.

`Not recorded` is a UI compatibility state, not a producer result. It means an
older report lacks the coverage contract, or a requested matrix unit is absent.
Do not infer that a profile completed because another profile contains a map
with the same signal name or payload hash.

For a new experiment, each complete capture unit also reports a verified
artifact closure and links `capture_artifact_closure.json`. A
`legacy-unverified` unit can be inspected for compatibility but does not prove
that its retained files still equal the post-capture tree.

Expand **Selected-frame evidence** below the matrix after choosing a scene and
frame. Its profile cells link separately to frame evidence and capture-level
manifests or completion records. In particular, a Process<false> `prefilter`
map and a Process<true> `deep` copy remain separate evidence even when their
bytes happen to match.

## First pass: did quality improve?

1. Select the production baseline and one variant.
2. Confirm endpoint and light-observer parity status.
3. Inspect aggregate annotation residuals first; lower noise is the primary
   objective.
4. Inspect effective inlier coverage and valid depth second.
5. Check repeat distributions and failed or unavailable scenes.
6. Sort by regression magnitude, not scene name.

Do not use deep or trace terminal output to rank quality.

## Second pass: where did it change?

Navigate from aggregate rows into per-scene analysis, then expand one frame's
diagnostics. Use synchronized baseline/variant maps and a shared scale for:

- final and prefilter depth;
- total and component costs;
- confidence and winner-runner-up gap;
- selected/supporting view counts and view churn;
- candidate/update source;
- texture, patch, and multiscale signals when registered and available;
- filter transitions and rejection reasons.

Reference RGB thumbnails establish whether a signal lies on a surface interior,
edge, occlusion, reflection, or low-texture region. Delta maps should use a
signed diverging scale centered at zero.

## Third pass: when did it change?

Select `Same logical iteration` when comparing runs. Inspect initialization and
each complete PatchMatch iteration:

- cost component distributions and closure;
- accepted update rate and improvement magnitude;
- propagation, random, depth, normal, prior, and view-set attribution;
- winning-versus-runner-up gaps;
- selected-view set and per-view contribution changes.

Black/red phases are useful only in the timing view. They must not appear as two
independent algorithm iterations.

`cost_improvement_exact` is a derived-exact, nonnegative reduction map for one
complete logical iteration. It sums the disjoint checkerboard reductions,
maps cost increases to zero, and can change basis when view selection changes.
Use it beside stored-cost, view-set, and component maps; it is not a signed
objective delta by itself.

## Fourth pass: why did it change?

Choose pixels from representative clusters rather than a single hand-picked
outlier. Include:

- an improvement;
- a regression;
- a stable control;
- a boundary or occlusion;
- a low-texture interior when relevant.

Create a paired drilldown request and rerun baseline plus variant:

```bash
tools/dmap_observability.sh drilldown \
  --config /path/to/experiment.yaml \
  --scene scene-a \
  --frame 0 \
  --pixel 120,80 \
  --pixel 180,96 \
  --variant candidate \
  --execute
```

Treat each trace as a separate capture. The UI must show its capture identity and
must not imply cross-capture pixel parity without an explicit parity record.
Public-v1 trace targeting narrows the rows investigated, not capture storage:
the selected frame also carries the full exact deep-map payload.

## Share a validated report

Create a sanitized, content-addressed review archive rather than sharing a
mutable experiment directory:

```bash
tools/dmap_observability.sh package \
  --report-dir /path/to/reports \
  --output /tmp/01_dmap_review.tar.zst
```

The wrapper validates the archive after creation and rejects host-specific
references, credential-bearing URLs, private keys, and common high-confidence
token formats without echoing a detected credential. It also verifies the source
report closure, creates a new closure after portable sanitization, and then
builds the archive-level SHA-256 inventory. Share the archive and its checksum
together. Raw capture maps are not included unless an explicit bounded
raw-artifact manifest requests them.

## Report size policy

The report generator has a hardcoded 768 MiB budget for selected uncompressed
browser pixel-payload source data, but public v1 has no strict aggregate cap for
generated previews or the complete report tree. Content, resolution, selected
signals, and frame count can move total size substantially.

Prefer summary/prefilter across the suite, then deep-capture selected frames.
Use package size limits when creating a review archive, inspect the storage
estimate and final report closure, and do not claim that the pixel budget alone
bounds total report size.

## Reading unavailable data

An unavailable panel is evidence about capture scope, resource admission, or
schema compatibility. Read its reason. Common valid reasons include a profile
not requesting the signal, an image not selected by deterministic sampling, or
a resource budget deliberately degrading an optional map.

Malformed, partial, non-finite, shape-incompatible, or unindexed data is not
validly unavailable; it is a validation failure.

When **Capture profile coverage** is marked unavailable because the report is
legacy, regenerate the report before making a profile-completeness or
cross-profile provenance claim. Existing quality and mechanics panels remain
readable, but map presence alone cannot establish which capture produced them.

## Decision checklist

A change is ready for broader evaluation only when:

- production endpoint quality improves or remains within the declared tolerance;
- accuracy is not traded for unexplained coverage;
- repeat behavior is stable;
- the report explains the mechanism with matched deep evidence;
- runtime and memory changes are acceptable;
- failures and unavailable evidence cannot bias the aggregate;
- the next experiment follows directly from observed evidence.
