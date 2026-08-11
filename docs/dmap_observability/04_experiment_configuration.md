# Experiment configuration

Experiment YAML is schema-versioned and immutable after a capture begins. Start
from [`examples/dmap_observability/experiment.template.yaml`](../../examples/dmap_observability/experiment.template.yaml),
store the filled copy outside the repository, and assign a new `experiment_id`
for every changed hypothesis, binary, input, or algorithm configuration.

## Required intent

Before running, write down:

- the hypothesis;
- the one isolated baseline-to-variant change;
- the frozen input scenes;
- the primary accuracy metrics and secondary coverage metrics;
- expected failure modes;
- capture profiles and sampling policy;
- device, host, storage, and wall-time limits;
- the decision rule for the next experiment.

## Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | Experiment configuration contract version. |
| `experiment_id` | Stable output identity; never reuse for changed inputs. |
| `hypothesis` | Testable mechanism and expected measurable effect. |
| `output_root` | External parent directory for generated evidence. |
| `densify_bin` | Production `DensifyPointCloud` executable. |
| `densify_observe_bin` | Separate observer executable. |
| `capture_profiles` | Default ordered profile list. |
| `suite` | Explicit scene IDs; avoid hidden dataset discovery in shareable configs. |
| `scenes` | Scene label, source working folder, and frozen `.mvs` path. |
| `input_snapshot` | Optional file/byte safety limits for exact staged-input hashing. |
| `instrumentation` | sampling, expected shape, and resource/storage policy. |
| `evaluation` | deterministic annotation fitting and reporting parameters. |
| `default_densify_args` | Arguments shared by every run. |
| `runs` | Named baseline and variant definitions with repeats. |

Paths may be relative to the config file. Prefer relative executable paths in a
shared repository and explicit external paths for input and output data. The
checked-in template uses `REPLACE_WITH_...` placeholders and cannot accidentally
run before they are replaced.

## Freeze SfM

Algorithm experiments in this framework start from the same `.mvs` scene and
image set. Do not rerun feature extraction, matching, or bundle adjustment
between baseline and variant. The experiment lock records input and executable
identities so a changed upstream scene cannot be silently compared.

The default input snapshot limits can be reduced for a bounded deployment:

```yaml
input_snapshot:
  max_files: 250000
  max_bytes: 1099511627776
```

Preparation hashes the exact regular files that will be copied into each
profile workspace. The same snapshot is checked before every profile and after
execution. Hashing cost is linear in retained input bytes and is intentionally
paid at orchestration boundaries, not inside CUDA. A changed file, added or
removed path, symlink, special file, concurrent write, or exceeded bound is a
hard error; use a new `experiment_id` after an intentional input change.

## Baseline and variants

Use one run with `role: baseline` and descriptive variant labels. Keep all
unrelated arguments in `default_densify_args`; put only the changed arguments in
each run. Use repeated runs when nondeterminism or timing variance matters.

Example structure:

```yaml
runs:
  - label: baseline
    role: baseline
    repeats: 2
    densify_args:
      - --iters
      - "3"
  - label: candidate
    role: variant
    repeats: 2
    densify_args:
      - --iters
      - "4"
```

This example illustrates structure, not a recommended parameter change.

Set `instrumentation.sample_rate` to a finite value in `(0, 1]` for
deterministic per-reference-image observer sampling. The default is `1.0`.
`--dmap-instrumentation-sample-rate` remains orchestration-owned and must not be
duplicated in `default_densify_args`; use the YAML field so the resolved config,
capture intent, and report provenance agree.

Targeted trace requests are admitted against both a pixel limit and a
conservative report-row limit:

```yaml
instrumentation:
  max_trace_pixels_per_request: 4096
  max_trace_rows_per_request: 4096
```

The row bound multiplies requested pixels by the logical states, photometric
pyramid levels, geometric-consistency stages, and selected baseline/variant
runs. The calculation applies the selected scene's `argument_overrides` and
rebinds the immutable request to the current run definitions before every
execution. `max_trace_rows_per_request` may lower the public report limit but
cannot raise it. Oversized requests fail before CUDA work with guidance to
reduce the ROI, run count, iterations, or pyramid/stage count. Coarse-level
coordinate collisions can make the actual capture smaller; admission
deliberately uses the safe upper bound.

Targeted drill-downs reject program-options `--config-file` inputs because an
external file can silently change `--iters`, `--sub-resolution-levels`, or
`--geometric-iters` after admission. Execution also fails before CUDA if the
observer's implicit `DensifyPointCloudDMapObserve.cfg` exists in the effective
working directory, then binds the command to an immutable empty
`generated/Densify.drilldown.cfg` so a default file cannot appear between
admission and launch. Express program options in `default_densify_args`, run
`densify_args`, or scene `argument_overrides` so they are captured by the
immutable request and experiment lock. This restriction does not apply to
`--dense-config-file`, whose densifier parameters do not define trace-row
topology.

For a broader benchmark, list every frozen scene explicitly and keep the same
run labels and repeats across them. `capture` traverses the configured
run/repeat/scene/profile matrix, reuses only validated completed captures, and
stops on invalid nonempty output. This provides resumability without silently
changing the comparison cohort.

## Prepare and inspect

```bash
tools/dmap_observability.sh init --config /path/to/experiment.yaml
```

Inspect the generated resolved config, immutable lock, selected scene list, and
storage estimate before launching CUDA work. Preparation must fail when inputs
or executables are missing, the schema is unsupported, or estimated storage
exceeds the configured limit.

## Annotation inputs

Annotations are optional. When supplied, they belong in an external dataset
sidecar and are bound into the experiment identity. Missing annotations remain
visible as unavailable evidence; they do not block mechanics reporting. See
[annotation metrics](07_annotation_metrics.md).

## Reproducibility record

Retain outside Git:

```text
experiment root/
  00_experiment_lock.json
  00_resolved_experiment.yaml
  00_storage_estimate.json
  frozen_inputs/
  runs/
  drilldowns/
  reports/
```

The report records commands, configuration, source and executable identities,
the staged-input snapshot, scene list, seeds, failures, unavailable signals,
artifact paths, and capture/report closure status. Do not manually edit
completed output.
