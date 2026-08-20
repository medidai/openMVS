# Troubleshooting

## `doctor` reports missing observer components

Build both boundaries:

```bash
tools/dmap_observability.sh build
tools/dmap_observability.sh doctor
```

If a required Python import fails, install the observability requirements into
the interpreter selected by `PYTHON` and rerun doctor. Missing `pyarrow`,
`plotly`, `jinja2`, `zarr`, or `numcodecs` is a nonfatal warning: the canonical
report remains available, but the corresponding columnar, plotting, or Zarr
extras are disabled. Do not switch interpreters between capture and report
without recording the resolved environment.

## CMake cannot resolve TinyNPY

The top-level OpenMVS configure traverses SFM even when the requested build
target is only `DensifyPointCloud`. TinyNPY packages in use expose either
`TinyNPY::TinyNPY` or `TinyNPYstatic`; this branch accepts both imported target
names so the production/observer build command remains portable. That CMake-only
compatibility does not add SFM instrumentation or change the observability
scope. An API compilation error inside `ImportROMA2.cpp` is a separate upstream
TinyNPY version mismatch and does not invalidate focused depth-map target builds.

## Observer options are unknown

You are probably running the production binary. Observer CLI options exist only
in `DensifyPointCloudDMapObserve`, built with
`OpenMVS_DMAP_INSTRUMENTATION=ON`. This separation is intentional.

## No instrumentation files appear

Confirm all of the following:

- the observer executable was used;
- `--dmap-instrumentation-dir` is nonempty;
- the requested image is in the image allowlist;
- deterministic sampling selected the frame;
- the profile requested the signal;
- the resource plan admitted it;
- the run did not stop before atomic completion.

The selection and availability ledgers should explain a deliberate absence.
An unexplained empty directory is not a completed capture.

## Capture refuses an existing directory

Experiment output is immutable. If config, executable, input, or completion
identity changed, assign a new `experiment_id`. Do not delete or edit a valid
prior run to make a new experiment fit its directory.

## Capture rejects dynamic-loader overrides

Attested captures require empty `LD_LIBRARY_PATH`, `LD_PRELOAD`, and `LD_AUDIT`.
These variables can bypass or interpose ELF RUNPATH and execute code that is not
bound by the experiment lock. Start capture from a clean shell or unset them
explicitly:

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT \
  tools/dmap_observability.sh capture \
  --config /path/to/experiment.yaml
```

Do not bypass this check. Rebuild with the required runtime search path or add
actual-loader resolution attestation in a future schema.

## Input snapshot hashing fails

The framework hashes every regular input copied into the frozen/profile
workspace. Stop processes that are writing the scene, remove symlinks or special
files from the retained input set, or raise `input_snapshot.max_files` /
`input_snapshot.max_bytes` deliberately. An intentional byte or path-set change
requires a new `experiment_id`. Large inputs take time because every retained
byte is read once per snapshot validation.

## Artifact closure validation fails

Do not repair a manifest by hand. A `changed`, `missing`, or `unexpected` path
means the completed capture/report tree no longer matches its post-close
inventory, even when file size stayed constant. Preserve the tree for diagnosis
and generate a new capture or numbered report. `legacy-unverified` is readable
compatibility state, not proof of integrity. Closure hashing adds sequential I/O
at capture/report close; it does not add CUDA work or disabled-mode overhead.

## Storage admission fails

Reduce deep image selection, use summary/prefilter first, or raise a budget only
after checking available space. Public-v1 exact pixel/ROI traces run maps mode
for the selected frame and therefore require the same full-frame deep budget.
`--allow-over-budget` is an explicit acknowledgement, not a default workflow.

Report generation separately selects at most 768 MiB of uncompressed numeric
source data for browser pixel payloads, but preview generation and total report
size have no strict aggregate preflight in public v1. Limit deep capture to
selected frames and enforce package size limits before sharing.

## Deep output differs from endpoint

This is an expected risk of `Process<true>`. Deep and trace captures are
diagnostic-only. Use endpoint and parity-qualified `Process<false>` captures for
quality. Compare mechanics between matched diagnostic runs, and keep the
specialization label visible.

## Maps show a different number of iterations

Check whether the view is showing checkerboard timing phases. Mechanics should
show initialization plus complete logical iterations. For `N` iterations, the
timing table has one initialization row and `2*N` phase rows.

## Annotation section is unavailable

Check the scene's `annotation_sidecar`, sidecar schema/version, matching
`scene_id`, frame mapping, DMAP image basename, camera dimensions/model, and
annotation coordinate space. Missing mappings and structures remain explicit.
The rest of the report can still be valid.

## Annotation coverage is high but residuals are worse

Coverage only indicates valid reconstructed samples. Inspect inlier fractions,
threshold AUC, residual P95, overlays, and competing fitted models. Added depth
may be noisy or may belong to a second surface inside the annotation.

## Report validation finds broken links

Generate the report and assets as one staged output. Do not move only the
Markdown or HTML. Rerun:

```bash
tools/dmap_observability.sh report \
  --config /path/to/experiment.yaml \
  --report-dir /path/to/reports \
  --rebuild-report
```

The rebuild path validates staging before promoting it and preserves the prior
report under a separate backup identity.

## Browser UI is blank

Do not open the HTML through `file://`. Serve its report directory:

```bash
tools/dmap_observability.sh serve --report-dir /path/to/reports
```

Check the browser console, `report_model.json`, model validation sidecar, and
report inventory. A missing model is a report-generation failure, not a UI
fallback condition.

## Package rejects host paths

The portable packager fails closed on host-specific references,
credential-bearing URLs, private keys, and common high-confidence token
formats. Keep capture paths and credentials out of report-owned evidence and
let the packager stage and rewrite legitimate path references. Validation names
the detected credential class without printing the credential. Do not bypass
the check for a bundle intended for colleagues.

## A signal is absent from the UI

Confirm it is registered, appears in the map/table catalog, validates, and is
included in the structured report model. The generic component presentation
should expose registered signals even without a custom panel. See the
[extension guide](09_extension_guide.md).
