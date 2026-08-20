# Depth-map observability example

`experiment.template.yaml` is a portable starting point. It contains no scene,
annotation, result, hostname, or machine-specific path. Copy it to an external
experiment directory and replace every `REPLACE_WITH_...` token.

```bash
mkdir -p /tmp/my-dmap-experiment
cp examples/dmap_observability/experiment.template.yaml \
  /tmp/my-dmap-experiment/experiment.yaml
```

At minimum, set:

- a unique `experiment_id`;
- an external `output_root`;
- production and observer executable paths;
- the frozen scene working folder and `.mvs` file;
- the scene ID and label;
- the isolated baseline/variant difference;
- realistic image/frame counts and storage limits.

An annotation sidecar is optional. Remove the `annotation_sidecar` field when
none is available; the report will retain explicit unavailable annotation
evidence. Its generic JSON contract is documented in
[`07_annotation_metrics.md`](../../docs/dmap_observability/07_annotation_metrics.md).

Prepare without running CUDA:

```bash
tools/dmap_observability.sh init \
  --config /tmp/my-dmap-experiment/experiment.yaml
```

Review the resolved config and storage estimate. Then capture production and
light evidence:

```bash
tools/dmap_observability.sh capture \
  --config /tmp/my-dmap-experiment/experiment.yaml \
  --profile endpoint summary prefilter
```

After report inspection selects a frame, add a separate deep capture or
drilldown. Do not turn the template into a checked-in results directory.

For a fully generated example that uses only the public OpenMVS test scene and
writes under `/tmp`, run:

```bash
tools/dmap_observability.sh demo \
  --output /tmp/openmvs-dmap-observability-demo
```

Add `--all-profiles` to run a paired exact pixel trace and assert that endpoint,
summary, prefilter, deep, and trace all complete in the report. This refers to
the five capture profiles, not CUDA/PatchMatch pyramid levels. The legacy
`--all-levels` spelling remains an alias. This mode adds two full-frame
diagnostic reruns and is intended for end-to-end qualification.
