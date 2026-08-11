# Schema reference

All retained evidence is versioned and validated before reporting. This page
describes the public contracts and their relationships; producer code remains
the definitive field-level source for the current schema version.

## Core records

| Contract | Purpose | Typical encoding |
|---|---|---|
| experiment config | requested scenes, runs, profiles, limits, and evaluation | YAML |
| experiment lock | immutable config, source, executable, input, and annotation identities | JSON |
| staged input snapshot | exact bounded file set copied into capture workspaces | JSON |
| run metadata | capture profile, build boundary, CLI, and completion identity | JSON |
| selection ledger | deterministic frame selection and exclusion reasons | CSV/JSONL |
| resource plan | requested/admitted buffers and device/host/storage bytes | JSONL |
| frame summary | final state statistics, availability, and endpoint relationship | JSON |
| pass timing | CUDA checkerboard phase timings | CSV |
| logical iteration | initialization and complete-iteration metrics | CSV/Parquet |
| map manifest | shape, dtype, role, semantics, state, source view, and file identity | JSON |
| trace request | immutable parent experiment, runs, frame, pixels/ROI, and limits | JSON |
| trace rows | selected-pixel retained-state, winning-source, cost-component, and view trajectories | JSONL |
| annotation sidecar | optional image mapping and line/plane structures | JSON |
| capture profile coverage | requested and validated profile evidence by run, repeat, scene, and frame | report-model JSON |
| capture artifact closure | exact post-close capture path/hash/size/mode inventory | JSON |
| report model | canonical structured comparison and UI input | JSON |
| report inventory | semantic report artifacts grouped by mechanism | JSON |
| report tree closure | exact final report path/hash/size/mode inventory | JSON |

## Input snapshot

Experiment-lock schema v4 embeds one
`openmvs.dmap.staged_input_snapshot` schema-v1 object per scene. It contains
`excluded_patterns`, `max_files`, `max_bytes`, `file_count`, `total_bytes`,
`snapshot_sha256`, and sorted `files` rows. Each row contains exactly `path`,
`bytes`, and `sha256`. Paths are relative to the scene working folder (or the
staged external MVS filename) and cannot traverse the root.

Default limits are 250,000 files and 1 TiB. The exclusions are generated-output
patterns: `depth*.dmap`, `*.log`, `*_dense.mvs`, `depth_maps`, and
`dmap_instrumentation`. Producers reject symlinks, non-regular files, mutation
during hashing, and any mismatch between the live, frozen, or profile-workspace
snapshot. Hashing therefore has linear read cost and can fail before CUDA work
when input identity is not stable.

## Artifact closures

`openmvs.dmap.capture_artifact_closure` schema v1 contains:

```json
{
  "schema_name": "openmvs.dmap.capture_artifact_closure",
  "schema_version": 1,
  "root": ".",
  "hash_algorithm": "sha256",
  "capture_profile": "deep",
  "exclusions": [
    {
      "path": "capture_artifact_closure.json",
      "scope": "exact",
      "reason": "closure manifest cannot include its own digest"
    },
    {
      "path": "work/",
      "scope": "prefix",
      "reason": "runtime staging and duplicate process outputs; terminal DMAPs, logs, and instrumentation are retained at the capture root"
    }
  ],
  "files": [],
  "file_count": 0,
  "total_bytes": 0,
  "files_sha256": "64-lowercase-hex-characters"
}
```

Every sorted `files` entry contains exactly `path`, `sha256`, `bytes`, and
`mode`. The real exclusion array is fixed by schema and names the self manifest
and the `work/` prefix with reasons. The manifest is written only after process
termination, terminal DMAP copying, and endpoint/instrumentation publication.
Experiment-lock schema v3 and later make it mandatory; the current lock schema
is v4. A missing closure under an older lock is `legacy-unverified`; a present
malformed or mismatched closure is always invalid.

Experiment-lock schema v4 also contains `runtime_boundaries.production` and
`runtime_boundaries.observer`. `repro.json` for every governed subprocess carries
an `openmvs.dmap.runtime_boundary_receipt` schema-v1 object with the expected,
pre-run, and post-run identities plus a self digest. Missing or drifting receipts
make the capture ineligible. Sibling runtime symlinks and nonempty
`LD_LIBRARY_PATH`, `LD_PRELOAD`, or `LD_AUDIT` fail closed.

Each real `run` invocation writes a content-addressed
`openmvs.dmap.capture_intent` schema-v1 record before launching the first capture.
It binds the experiment lock, activation source, requested profiles, and full
run/repeat/scene/profile matrix. Dry runs do not write intent. Coverage loads all
valid intents so a CLI-only profile remains requested and explicitly unavailable
even when launch fails before a capture directory exists.

`openmvs.dmap.report_tree_closure` schema v1 has the same aggregate and file-row
fields without `capture_profile`. Its fixed exclusions name the self manifest
and `01_development_report.validation.json`, which validators may regenerate.
Report-policy schema v3 declares both closure schema versions and binds a
deterministic capture-profile coverage digest plus capture-intent, trace-control,
and capture-closure identities. New deep or trace evidence cannot silently
reuse a stale report. Complete legacy evidence without a verified closure
remains readable, but report reuse is disabled and requires a staged rebuild.
Exact path-set, hash, byte-size, and mode equality is required by report
validation and reuse. Portable staging regenerates the manifest after path
sanitization, omitted raw artifacts, portable finalizer rewriting, and staging
metadata are complete.

Closure manifests provide deterministic integrity and same-size tamper
detection, not signer identity or hostile-author authenticity.

## Capture profile coverage

Report model schema v3 may include the additive top-level
`capture_profile_coverage` object. Its contract is independently versioned:

```json
{
  "schema_name": "openmvs.dmap.capture_profile_coverage",
  "schema_version": 1,
  "requested_profiles": ["endpoint", "summary", "prefilter", "deep", "trace"],
  "capture_intents": [],
  "profiles": [],
  "units": []
}
```

Each `profiles` row contains `profile`, `requested`, `status`,
`expected_units`, `complete_units`, `failed_units`, `unavailable_units`,
`process_specialization`, and `quality_authority`. These rows drive the compact
five-profile summary; they do not create additional quality candidates.

Each `units` row is uniquely identified by `configured_run`, `repeat`,
`scene_id`, and `capture_profile`. It contains `status`, `reason`, capture-level
`evidence_links`, an optional additive `artifact_closure` status record, and
zero or more frame records. Closure status is `verified`, `legacy-unverified`, `invalid`, or
`unavailable`, with the requirement flag and reason. A frame record contains
`image_id`, `status`, and its own `evidence_links`. Evidence links use a human
label and a portable report-relative path.

The coverage contract reports `complete`, `failed`, `unavailable`, and
`not_requested` explicitly. A requested profile cannot disappear from the
model. A complete `prefilter` unit must point to the actual prefilter manifest
and Process<false> evidence; a deep manifest or a byte-identical deep map cannot
substitute for it. Capture units supply provenance only and must not duplicate
frames or enter quality, annotation, performance, gate, or Pareto aggregates.

Readers remain backward-compatible when the top-level field is absent, but
must display coverage as not recorded. They must not reconstruct profile
identity from filenames, signal names, or artifact hashes.

## Map manifest semantics

Every map record identifies:

- `signal` and optional component ID;
- scene, run, repeat, image, estimation stage, and pyramid level;
- initialization or logical iteration;
- shape, dtype, encoding, and channels;
- measurement quality and basis;
- exact value, proxy target and limitation, or unavailable reason;
- source view when the signal is view-specific;
- relative file path, byte size, and content identity.

A readable file without a valid manifest entry is not report evidence. A
manifest entry whose file is partial, shape-incompatible, non-finite outside its
declared domain, or identity-mismatched is invalid.

## Availability states

Use three distinct states:

1. **Available:** validated value with complete semantics.
2. **Unavailable:** optional signal was not requested, not selected, not
   admitted, or not applicable, with a stable reason.
3. **Invalid:** expected evidence exists but violates schema, identity,
   completion, shape, domain, or relationship constraints.

Only the first two states can enter a valid report. Zero is a measurement, not
an availability marker. Float maps use an explicit unavailable representation
and companion availability metadata; categorical maps reserve documented enum
values.

## Exact and proxy measurements

An exact record names its computation basis. A proxy additionally names the
target it approximates and its limitation. Labels such as `post-pass source`
must not be shortened to `winning candidate` if the exact winner was not
captured.

The report does not upgrade a proxy based on correlation or visual similarity.
Schema transitions that improve a proxy to exact require a new schema version
or a backward-compatible measurement-quality field with validator coverage.

## Logical states

State maps are indexed as initialization plus one state after each complete
PatchMatch iteration. Event maps describe candidates tested and decisions made
during the transition to a state. Checkerboard phase indices are permitted in
timing rows only.

For configured `N` PatchMatch iterations, a complete timing topology contains
one initialization pass and `2*N` alternating checkerboard passes. A missing,
duplicate, reordered, or non-finite timing row invalidates timing completeness.

## Completion

Writers publish manifests and completion markers only after data files are
closed and atomically renamed within the target filesystem. Public v1 does not
claim power-loss durability because writers do not `fsync` every file and parent
directory. Validators reject nonempty incomplete directories. Recovery uses a
new experiment or capture identity rather than treating leftovers as a resumable
valid run.

A required map counts as published only when its relative path stays inside the
frame directory and resolves directly to a non-symlink regular file with a
statable, nonzero size. `summary.json` starts with a pending completion state;
its `maps_complete` and `eligible` fields are finalized only after the relevant
manifest has been written and rebound by size. Completion markers include the
bound manifest and summary sizes.

`capture_complete.json`, `prefilter_capture_complete.json`, and
`summary_complete.json` attest the core CUDA PatchMatch frame publication.
Optional `postprocess_filters.json`, `confidence_adjustment.json`, and
`filter_resource_plan.json` are later SceneDensify sidecars. They have their own
resource-plan and schema validation and are not transitively attested by the
earlier core marker in public v1.

Resource-plan schema v4 reports the current pyramid estimate, storage committed
by earlier levels, and the full-resolution priority reserve separately. Its
component estimates are additive within one pyramid level and include trace
buffers, retained map state, export scratch, and fixed host overhead.

## Terminal depth-map codecs

Terminal validation recognizes both the legacy `DR` depth-map encoding and the
current `D2` encoding. Comparisons between retained float maps and `D2` outputs
use explicit bounds for half-precision depth, octahedral normal encoding, and
quantized confidence; the validator reports those bounds with the result.
Endpoint comparisons between two files using the same codec remain exact.

## Version policy

- Readers reject unknown major contracts by default.
- Additive optional fields require explicit defaults and tests.
- Removed or reinterpreted fields require a version increment and migration or
  a clear unsupported result.
- Reports state unavailable signals from older captures explicitly.
- Reports without `capture_profile_coverage` remain readable, but cannot support
  profile-completeness or cross-profile provenance claims.
- Captures without a closure are accepted only for pre-v3 experiment locks and
  are explicitly `legacy-unverified`; reports without a closure are accepted
  only for pre-v2 report policies.
- Schema validators are part of the public interface and must remain usable
  without private datasets.

See the machine-readable [capability index](capabilities.json) for ownership
paths and [extension guide](09_extension_guide.md) for adding a signal.
