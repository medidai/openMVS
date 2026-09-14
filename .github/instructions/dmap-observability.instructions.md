---
description: "Use when changing CUDA depth-map instrumentation, experiment capture, report generation, schemas, validation, or investigation UI."
name: "OpenMVS Depth-map Observability"
---
# Depth-map observability workflow

## Scope

Work only in CUDA depth-map estimation, view preparation, depth-map
postprocessing, and filtering. Fusion, point-cloud fusion, mesh reconstruction,
mesh refinement, and texturing are out of scope.

## Contracts

1. `OpenMVS_DMAP_INSTRUMENTATION` remains `OFF` by default.
2. An `OFF` build must not compile observer CLI options, observer state,
   diagnostic kernels, map exporters, or observer allocations.
3. In an observer build, an empty `--dmap-instrumentation-dir` performs no
   capture work and emits no artifacts.
4. Production endpoint output must remain bit-exact. Validate the same-host
   endpoint against disabled, summary, and prefilter observer captures before
   treating those profiles as quality evidence.
5. `Process<true>` deep and trace captures can change kernel resource pressure
   and output. They are diagnostic-only unless a separate parity qualification
   proves otherwise. A public-v1 exact trace still allocates and exports deep
   maps for the selected frame; the pixel or ROI target bounds investigation,
   not the hot-kernel buffer footprint.
6. Never infer that an unavailable signal is zero. Emit explicit availability,
   reason, measurement quality, basis, proxy target, and limitations.
7. Aggregate black/red checkerboard passes into one logical iteration for
   mechanics. Keep phase-level rows only in timing tables.
8. Capture files and reports are immutable after completion. A changed config,
   binary, input, schema, or source identity requires a new experiment ID.

## Change workflow

1. Write the hypothesis and identify one algorithmic mechanism.
2. Locate the producer, descriptor, schema, validator, report model, and UI
   presentation using `docs/dmap_observability/capabilities.json`.
3. Add the smallest signal that resolves the hypothesis. Prefer existing
   buffers and registries where that is exact and does not change production
   behavior.
4. Define units, domain, direction, capture profile, exact/proxy status,
   unavailable representation, storage, and lifetime.
5. Add malformed, missing, non-finite, shape, completion, and version tests.
6. Build both production and observer boundaries.
7. Verify disabled behavior and endpoint parity before using quality metrics.
8. Run a small endpoint plus summary/prefilter comparison. Select regressions,
   then rerun only bounded frames or pixels at deep/trace level.
9. Generate and validate Markdown, structured model, and HTML. Inspect the UI
   from aggregate metrics through frame maps to pixel traces.
10. Keep all generated evidence in `/tmp` or a user-selected external output
    directory. Do not add generated evidence to Git.

## Required checks

```bash
tools/dmap_observability.sh doctor
tools/dmap_observability.sh build
python3 -m unittest discover -s scripts/python -p 'test_*.py'
bash -n tools/dmap_observability.sh
```

For a captured report:

```bash
tools/dmap_observability.sh validate --report /path/to/01_development_report.md
```

Also run the smallest relevant C++/CUDA build or test first, then the broader
suite appropriate to the changed ownership boundary. Do not claim CUDA parity
without a fresh CUDA run on the target host.

## Review focus

- Does disabled production behavior remain absent rather than merely unused?
- Can partial writes, stale artifacts, or schema drift be mistaken as complete?
- Is every quality claim backed by `Process<false>` endpoint parity?
- Are logical iteration and checkerboard phase semantics unambiguous?
- Are storage and device/host allocation limits checked before allocation?
- Can the report navigate from aggregate change to scene, frame, map, and pixel?
- Are new mechanism signals discoverable without UI-specific hard-coding?
- Are paths, hostnames, dataset identifiers, annotations, and generated results
  absent from source and documentation?
