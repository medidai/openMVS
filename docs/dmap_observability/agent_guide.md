# Agent guide

This is the minimum context for an automated coding agent working on depth-map
observability. Read [`capabilities.json`](capabilities.json) first, then the
mechanism-specific human documentation.

## Mission and scope

The platform makes CUDA depth-map estimation and filtering changes measurable
from aggregate production quality through scene, frame, map, logical iteration,
and pixel mechanics. Do not add instrumentation to fusion, mesh reconstruction,
mesh refinement, or texturing.

## Invariants

1. Build-time instrumentation defaults `OFF`.
2. The production binary contains no observer CLI, state, allocations, output,
   diagnostic kernels, or device-state changes.
3. An observer binary with an empty instrumentation directory performs no
   capture and emits no instrumentation artifacts.
4. Production DMAPs remain bit-exact when validating disabled, summary, and
   prefilter paths on the same host and build lineage.
5. Deep and trace `Process<true>` evidence is diagnostic-only unless separately
   qualified. Never use it to rank final quality by default. Public-v1 trace
   selection narrows analysis rows but still uses a full-frame deep allocation
   for the selected frame; compact exact pixel-only buffers are unavailable.
6. Unavailable is distinct from zero and invalid. Preserve reason codes.
7. Mechanics use initialization and complete logical iterations. Checkerboard
   phases appear only in timing evidence.
8. Existing complete captures and reports are immutable.
9. The current experiment-lock schema is v4. Capture closure is mandatory for
   locks v3 and later; v4 additionally binds runtime-boundary receipts and the
   current input-snapshot rules. Report-policy schema v3 requires report-tree
   closure and binds capture-profile coverage plus capture-intent,
   capture-closure, and trace-control content identities; schema v2 covered only
   the report tree. Never regenerate a closure merely to make unexplained drift
   pass.
10. Generated evidence remains outside Git.

## Start here

```bash
cat docs/dmap_observability/capabilities.json
tools/dmap_observability.sh doctor
git status --short
```

For a branch based on a ref other than `origin/develop`, pass
`--public-base REF` or set `DMAP_PUBLIC_BASE`. Doctor runs the public-tree guard
against that exact ref when the checker is present.

Locate relevant code with the ownership paths in `capabilities.json`. Read the
current producer and validator before proposing a change. Search for the signal
ID in the CUDA producer, component registry, validator, report builder, model,
UI, and tests.

## Standard task loop

1. State the hypothesis and the fastest bounded experiment that can falsify it.
2. Define the signal semantics and whether it is exact or a proxy.
3. Estimate memory, storage, and runtime before adding a buffer.
4. Implement the smallest producer/registry/schema/report change.
5. Add focused valid and invalid fixtures.
6. Build and test the narrow ownership boundary.
7. Qualify disabled behavior and `Process<false>` parity on CUDA.
8. Capture endpoint/light evidence, select frames, then capture deep/trace.
9. Generate, validate, and visually inspect the report.
10. Record conclusions and the next experiment. Do not commit unless asked.

## Signal checklist

Every new signal needs:

- a stable ID and component registry descriptor;
- units, domain, preferred direction, profile, and measurement kind;
- state/event timing and pyramid/estimation stage;
- exact basis or proxy target and limitations;
- unavailable representation and reasons;
- device/host/storage resource accounting;
- atomic writer and completion behavior;
- schema and relationship validation;
- structured report inclusion and generic visual presentation;
- malformed/missing/non-finite/shape/version tests;
- human documentation and a reproducible capture command.

## Quality reasoning

Accuracy is primary; coverage is secondary. For line/plane annotations, inspect
residual P95, threshold AUC, and tight-threshold inlier fraction before valid or
effective coverage. Check fitted-model stability so a candidate cannot look
clean merely by fitting a different surface.

Separate final quality from causal explanation:

```text
endpoint / parity-qualified summary or prefilter -> quality conclusion
matched deep maps and traces                      -> mechanism conclusion
```

Do not combine the two authority classes into one unlabeled metric.

## Report expectations

A complete development report includes:

- executive summary and explicit recommendation;
- algorithm/config delta and objective mechanics;
- precise metric definitions;
- aggregate and per-scene quality;
- repeat distributions, runtime, and resource use;
- rich synchronized cost/view/update/patch/multiscale/filter maps when each
  signal is registered, captured, and validated as available;
- annotation overlays and residual distributions when available;
- regression sorting and aggregate-to-frame links;
- targeted trace navigation;
- all failed and unavailable evidence;
- exact reproduction commands and artifact identities.

Before reasoning from a report, inspect top-level
`capture_profile_coverage`. Treat `(configured_run, repeat, scene_id,
capture_profile)` as the capture-unit identity, and use its frame records for
frame-specific provenance. A requested profile must be complete, failed, or
explicitly unavailable; absence is not success. Never substitute deep evidence
for prefilter evidence based on a shared signal name or identical bytes, and do
not feed profile-evidence units into quality aggregates.

Require `artifact_closure.status == "verified"` for new capture evidence and a
valid `report_tree_closure.json` before reuse or packaging. A
`legacy-unverified` record may explain an older artifact but must not be silently
upgraded. Input and closure hashing is intentionally linear in retained bytes;
report its cost or failure rather than bypassing it.

Markdown and structured data are canonical. HTML must not recompute metrics or
hide evidence absent from the model.

## Forbidden shortcuts

- Do not pass null observer pointers through an otherwise changed production
  kernel and call it zero overhead.
- Do not add observer options to `DensifyPointCloud`.
- Do not zero-fill missing values.
- Do not call a post-pass attribution an exact winning candidate.
- Do not expose black/red phases as separate mechanics iterations.
- Do not silently drop failed scenes, malformed annotations, or partial maps.
- Do not run full-corpus deep capture before summary evidence selects targets.
- Do not publish generated reports, private paths, hostnames, datasets, or
  annotations with the code.

## Validation commands

```bash
bash -n tools/dmap_observability.sh
python3 -m json.tool docs/dmap_observability/capabilities.json >/dev/null
python3 -m unittest discover -s scripts/python -p 'test_*.py'
node --test scripts/dmap_report_ui/test_capture_profile_coverage.js
tools/dmap_observability.sh build
tools/dmap_observability.sh doctor
```

Run `tools/check_dmap_observability_public_tree.py` when present. A fresh CUDA
capture is required for runtime parity claims; compilation or synthetic fixture
tests alone are insufficient.
