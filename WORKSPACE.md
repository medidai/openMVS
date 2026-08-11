# Depth-map observability workspace

This source tree contains the shareable OpenMVS CUDA depth-map observability
tooling. Start with [`AGENTS.md`](AGENTS.md), then read the human
[quickstart](docs/dmap_observability/01_quickstart.md), the
[agent guide](docs/dmap_observability/agent_guide.md), and the machine-readable
[capability index](docs/dmap_observability/capabilities.json).

## Scope

- Instrument depth-map estimation, source-view preparation, depth-map
  postprocessing, and filtering only.
- Do not add observability to point-cloud fusion, mesh reconstruction, mesh
  refinement, or texturing.
- Keep this repository source-only. Do not add experiment captures, reports,
  annotations, visualizations, benchmark results, or machine-specific paths.

## Production contracts

- `OpenMVS_DMAP_INSTRUMENTATION` defaults to `OFF`.
- The production and observer executables use separate Release build trees.
- An `OFF` build contains no observer CLI, state, diagnostic kernels,
  allocations, output, or instrumentation-related device-state changes.
- An observer executable remains inactive when
  `--dmap-instrumentation-dir` is empty.
- Production DMAP output is the quality authority. Summary and prefilter
  evidence need same-host endpoint parity before they support quality claims.
- The observer and OFF binaries are not assumed to be kernel-resource
  equivalent even on `Process<false>`; the OFF binary is the performance/resource
  authority, and summary/prefilter need separate parity and timing qualification.
- Deep and exact trace captures use diagnostic `Process<true>` kernels. Public
  v1 exact traces select pixels for investigation but allocate and retain the
  selected frame's full deep payload.
- Missing signals remain explicitly unavailable with a reason; unavailable is
  never replaced by zero.
- Report mechanics by complete logical PatchMatch iteration. Expose black/red
  checkerboard phases only for timings.
- Core PatchMatch completion markers do not attest optional postprocess or
  confidence-adjustment sidecars written later; validate those independently.
- Experiment-lock schema v4 freezes the exact bounded input file set and
  revalidates live, frozen, and profile workspaces around every capture.
- New captures and reports require exact SHA-256/size/mode artifact closures.
  Preserve older missing-closure evidence only as `legacy-unverified`; never
  regenerate a closure merely to conceal unexplained drift.
- Release qualification must exercise endpoint, summary, prefilter, deep, and
  trace captures and confirm report navigation from aggregate metrics through
  scene, frame, logical iteration, and exact retained-pixel evidence.
- Treat capture profiles and pyramid levels as separate axes. Public v1 exposes
  per-level counters, timings, resource plans, and selected maps, while the full
  cost-component family is retained at pyramid level 0 only.
- Release qualification must also capture at least one coarse pyramid level and
  verify that every retained level is selectable in the report. Level 0 must
  expose the exact map family; coarse levels must expose their counters,
  timings, resource contract, compatibility update-source map, and explicit
  unavailable state for cost maps that production does not retain.
- Treat the configured frame-storage limit as a per-producer bound in public
  v1: PatchMatch enforces it cumulatively across pyramid levels, while later
  postprocess/confidence sidecars enforce it through a separate ledger. Do not
  describe it as one combined whole-frame or whole-report cap.

## Development workflow

1. State a concrete hypothesis and the algorithm mechanism it changes.
2. Freeze the same input scene for baseline and variant.
3. Start with endpoint plus summary or prefilter capture.
4. Validate evidence and parity before comparing quality.
5. Use aggregate metrics to select scenes and frames for deep capture.
6. Use exact drilldowns for representative regressions, improvements, and
   stable controls.
7. Generate and validate the canonical Markdown, structured model, and HTML
   report.
8. Analyze the result, record limitations, and define the next bounded
   experiment.

Use [`tools/dmap_observability.sh`](tools/dmap_observability.sh) as the stable
human-facing CLI. Put every generated experiment and report in `/tmp` or another
explicit directory outside this source tree. Portable colleague bundles must be
created with the wrapper's `package` command so staging, sanitization, and
validation cannot be bypassed.

Do not commit unless explicitly requested. Keep changes minimal, tested, and
limited to the stated observability scope.
