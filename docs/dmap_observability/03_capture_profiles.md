# Capture profiles

Choose the least expensive profile that can answer the question. Broad sweeps
should remain production-authoritative; deep capture is a follow-up for selected
frames or pixels.

| Profile | Executable | CUDA path | Retained evidence | Quality authority |
|---|---|---|---|---|
| `endpoint` | production | `Process<false>` | terminal production DMAP and run metadata | yes |
| `summary` | observer | `Process<false>` normally; APD exact aggregates use `Process<true>` | counters, availability, metadata, logical summaries, timings | non-APD only after endpoint parity; APD summary is diagnostic-only |
| `prefilter` | observer | `Process<false>` | summary plus bounded production-path state before filtering | after endpoint parity |
| `deep` | observer | `Process<true>` | full-frame mechanics maps for initialization and logical iterations | no |
| `trace` | observer | `Process<true>` | selected-pixel or bounded-ROI trajectories plus full selected-frame deep maps | no |

## Endpoint

Use for every baseline and variant. It runs the binary compiled without
observability and establishes final depth-map quality and runtime. Observer
arguments are removed from its command line.

## Summary

Use across scenes and repeats. It provides storage-light frame, iteration,
availability, and timing evidence without full-frame diagnostic maps. Pair it
with the endpoint and require parity before using an ordinary non-APD summary
for quality gates. When APD exact aggregate mechanics are requested, summary
uses `Process<true>` so the hot kernel can emit exact APD counters. The report
marks that cohort diagnostic-only; its DMAP must not be substituted for the
production endpoint.

The observer `Process<false>` specialization is not assumed to have the same
stack usage or occupancy as the separate production binary; parity establishes
output equivalence, while timing and kernel-resource comparisons remain
separate qualification gates.

## Prefilter

Use when a final regression may originate in filtering. It retains a bounded
production-path snapshot before filtering so the report can distinguish
estimator changes from keep-cost, speckle, consistency, or confidence rejection.
Pair it with the endpoint and validate parity.

## Deep

Use on a small image allowlist after summary evidence identifies an interesting
frame. It captures cost components, exact candidate accounting, winning and
runner-up gaps, view decisions and contributions, propagation/update sources,
texture and patch signals, multiscale transfers, and filtering maps when those
signals are available.

Deep/maps capture always requests exact `Process<true>` observability. Exact
per-view records include selection prior, sampling score, and probability-health
fields; there is no separate runtime toggle for exactness or probability health.

Deep capture uses the diagnostic specialization. Compare its internal maps
within matched deep runs; do not compare its terminal DMAP to the production
endpoint as if it were a quality result.

## Trace

Use when a map identifies a representative or surprising pixel. A trace records
the retained state, winning update source, cost components, and view state across
logical iterations and relevant stages. It does not retain the full cost history
of every rejected candidate; the accompanying exact maps retain candidate masks,
winner/runner-up values, and aggregate candidate accounting. Select pixels using
coordinates from the captured map grid, not the source RGB resolution. A trace
is a separate rerun and must carry its parent experiment, run, scene, frame,
target, binary, and configuration identities.

The target is logically bounded, but public-v1 storage is not pixel-bounded:
the rerun requests maps mode for the selected frame so that the trace comes from
the exact `Process<true>` path. It allocates and writes the full-frame deep
payload, then presents only the requested trace rows. Compact selected-pixel
exact buffers are explicitly unavailable in this version.

## Sampling and limits

Observer selection is deterministic through image allowlists, sample rates, and
sample seeds. Set `instrumentation.sample_rate` in the experiment YAML; keep the
seed and an optional `--dmap-instrumentation-image-list` in
`default_densify_args`. Prefer an explicit image list for deep work. Set all
three budgets:

- additional CUDA device memory per frame;
- retained host memory per frame;
- estimated uncompressed map storage per frame.

Use `budget_policy: error` for controlled experiments where missing evidence
would invalidate the hypothesis. Use `degrade` only when the report can remain
useful with explicitly unavailable optional signals.

Within PatchMatch, the frame-storage limit is cumulative across pyramid levels.
Each level has a storage preflight record, and coarse levels reserve the admitted
full-resolution tier before writing their compatibility update-source maps. The
later postprocess/confidence sidecar producer enforces the same configured limit
independently; public v1 does not claim that the option is a single combined cap
across both producers. The production confidence map is retained only at pyramid
level 0, so every coarse resource-plan row declares its compatibility cost map
unavailable by design. Trace admission also includes the dense per-frame
trace-index map, trace records, labels, and JSONL allowance.

## Recommended progression

```text
endpoint + summary across all scenes
  -> endpoint + prefilter on filtering suspects
  -> deep on selected frames
  -> trace on selected pixels or a bounded ROI
```

Do not run deep capture across a full corpus by default. It increases runtime,
storage, and diagnostic specialization risk without improving the authority of
the final quality metrics.
