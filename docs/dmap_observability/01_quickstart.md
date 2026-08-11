# Depth-map observability quickstart

This toolkit captures and explains CUDA depth-map estimation and filtering. It
does not instrument fusion, mesh reconstruction, mesh refinement, or texturing.
Generated captures and reports belong in an external output directory, not in
the repository.

## Prerequisites

- A CUDA-capable OpenMVS build environment.
- CMake, a build tool such as Ninja, and Pandoc for the canonical HTML companion.
- `nvcc` on `PATH` and either `VCPKG_ROOT` pointing at a vcpkg checkout or
  equivalent system dependencies. OpenMVS also accepts an explicit vcpkg
  toolchain through `build --cmake-arg`.
- Python 3.11 or newer with the dependencies declared by the observability
  requirements files.
- Enough disk for the selected capture profile. Deep maps are deliberately
  bounded but can still be large.

Run the environment check first:

```bash
export VCPKG_ROOT=/path/to/vcpkg
export PATH=/path/to/cuda/bin:$PATH
tools/dmap_observability.sh doctor
```

Install the recommended report and array-store environment when `doctor` reports
missing Python modules:

```bash
python3 -m venv .venv
.venv/bin/pip install \
  -r scripts/python/requirements-depth-benchmark.txt \
  -r scripts/python/requirements-dmap-array-store.txt
PYTHON="$PWD/.venv/bin/python" tools/dmap_observability.sh doctor
```

Core capture, validation, canonical Markdown/model/HTML reporting, and portable
packaging dependencies are required. `doctor` reports missing `pyarrow`,
`plotly`, `jinja2`, `zarr`, and `numcodecs` as nonfatal optional warnings:
CSV remains the structured-table fallback, while optional columnar, plotting,
and Zarr archival capabilities are unavailable until their dependencies are
installed.

Optional archival conversion is exposed through the stable wrapper:

```bash
tools/dmap_observability.sh array-store \
  --config /path/to/experiment.yaml \
  --run baseline candidate \
  --scene scene-a
```

This creates immutable Zarr v3 derivatives for selected validated frame maps;
it is not required to generate or inspect the report.

The command reports missing build products as `not built` with the next build
command; that alone is not an environment failure. Missing build tools, core
Python dependencies, publication checks, or an explicitly selected publication
base cause a nonzero exit. Missing optional Python modules only emit warnings.
Doctor does not modify the repository.

## Build both boundaries

```bash
tools/dmap_observability.sh build
```

This creates separate Release build trees:

```text
build-dmap-production/bin/DensifyPointCloud
build-dmap-observer/bin/DensifyPointCloudDMapObserve
```

The production tree configures `OpenMVS_DMAP_INSTRUMENTATION=OFF`; the observer
tree configures it `ON`. Do not use the observer executable as a substitute for
the production endpoint in a baseline.

## Run the ephemeral demo

The repository contains a small public MVS test scene. The default demo creates
its configuration, synthetic line/plane annotations, endpoint, summary,
prefilter, and deep captures, tables, maps, plots, and reports under `/tmp`:

```bash
tools/dmap_observability.sh demo \
  --output /tmp/openmvs-dmap-observability-demo
```

The output directory must not already contain data. This prevents a demo from
silently mixing evidence from different binaries or configurations.

For a release or end-to-end qualification, include an exact paired trace and
require all five capture profiles to validate:

```bash
tools/dmap_observability.sh demo \
  --all-profiles \
  --multiscale \
  --geometric-iters 1 \
  --output /tmp/openmvs-dmap-observability-all-profiles
```

`--all-profiles` means the five capture profiles `endpoint`, `summary`,
`prefilter`, `deep`, and `trace`; it does not refer to CUDA or PatchMatch
pyramid levels. The older `--all-levels` spelling remains a backward-compatible
alias. `--multiscale` separately requires pyramid levels 0 and 1, verifies that
both are selectable in the report model, and requires the coarse compatibility
update-source proxy. Coarse cost-component maps remain explicitly unavailable.
`--geometric-iters 1` adds a geometric-consistency stage and requires its exact
trace rows in the generated report. Together, these options exercise all five
capture profiles, pyramid levels 0 and 1, and both photometric and geometric
estimation stages.
The trace profile is another full-frame `Process<true>` rerun for the baseline
and candidate, even though the report focuses it on one pixel. Budget this mode
for more than twice the deep-capture work and storage of the default demo.

After it completes:

```bash
tools/dmap_observability.sh serve \
  --report-dir /tmp/openmvs-dmap-observability-demo/experiments/01_ephemeral_demo/reports
```

Open the URL printed by the command. Stop the server with `Ctrl-C`. The demo is
illustrative; its synthetic annotations are not benchmark ground truth and its
results must not be checked into Git.

## Start an experiment on your scene

Copy the template outside the repository and replace every
`REPLACE_WITH_...` value:

```bash
mkdir -p /tmp/my-dmap-experiment
cp examples/dmap_observability/experiment.template.yaml \
  /tmp/my-dmap-experiment/experiment.yaml
```

Resolve inputs, bind their identities, and estimate storage:

```bash
tools/dmap_observability.sh init \
  --config /tmp/my-dmap-experiment/experiment.yaml
```

Start with production-authoritative evidence:

```bash
tools/dmap_observability.sh capture \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --profile endpoint summary prefilter
```

Generate and validate the canonical report:

```bash
tools/dmap_observability.sh report \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --report-dir /tmp/my-dmap-experiment/reports

tools/dmap_observability.sh validate \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --report-dir /tmp/my-dmap-experiment/reports
```

Use aggregate metrics to choose a scene and frame. Only then capture expensive
mechanics:

```bash
tools/dmap_observability.sh capture \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --profile deep

tools/dmap_observability.sh drilldown \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --scene scene-a \
  --frame 0 \
  --pixel 120,80 \
  --variant candidate \
  --execute
```

Regenerate the report after a completed drilldown. Deep and trace results
explain mechanics; endpoint/summary/prefilter parity remains the quality
authority. The public-v1 trace path runs exact `Process<true>` capture for the
selected frame, so budget it as one full-frame deep capture even when only a few
pixels or a bounded ROI are selected for the trace table.

```bash
tools/dmap_observability.sh report \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --report-dir /tmp/my-dmap-experiment/reports \
  --rebuild-report
```

Report policy v3 binds capture intents, profile coverage, trace controls, and
capture-closure identities. A direct reuse attempt therefore fails closed when
new evidence appears; `--rebuild-report` stages, validates, and atomically
promotes the refreshed report while preserving the previous one.

## Qualify one CUDA diagnostic capture

Run Compute Sanitizer on one selected deep frame before treating new device
buffers or kernels as development-ready. Preserve the same production arguments
used by the matched experiment:

```bash
/usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck \
  --error-exitcode 99 \
  /path/to/build-dmap-observer/bin/DensifyPointCloudDMapObserve \
  --working-folder /path/to/frozen-scene/work \
  --input-file /path/to/frozen-scene/work/scene.mvs \
  --output-file /path/to/sanitizer-run/deep.mvs \
  --fusion-mode 1 \
  --dmap-instrumentation-dir /path/to/sanitizer-run/instrumentation \
  --dmap-instrumentation-level maps \
  --dmap-instrumentation-sample-rate 1 \
  --dmap-instrumentation-image-list 17 \
  --dmap-instrumentation-write-maps 1 \
  <the matched run's remaining DensifyPointCloud arguments>
```

For a trace qualification, add
`--dmap-instrumentation-config /path/to/trace_config.json` while keeping the
image list to the single targeted frame. Success requires exit status zero and
the sanitizer footer `ERROR SUMMARY: 0 errors`. Save the complete command and
log with the external experiment evidence.

Memcheck complements schema/resource validation and fresh CUDA execution. It
does **not** replace the separate production-`OFF`, observer-disabled, endpoint,
summary, and prefilter bit-exact parity checks.

## Experiment loop

1. State one hypothesis and one isolated code or configuration change.
2. Freeze the same input `.mvs` scene for baseline and variant.
3. Capture endpoint plus summary or prefilter for both.
4. Validate output parity and compare accuracy before coverage.
5. Sort regressions and improvements in the report.
6. Select representative frames, pixels, or ROIs for deep/trace reruns.
7. Inspect mechanism maps and iteration evolution.
8. Accept, reject, or refine the hypothesis; use a new experiment ID for the
   next immutable run.

Continue with [capture profiles](03_capture_profiles.md), [report
navigation](05_report_guide.md), and [debugging playbooks](06_debugging_playbooks.md).
