# Architecture

## Boundary

Depth-map observability has two compile-time products:

```text
OpenMVS_DMAP_INSTRUMENTATION=OFF
  -> DensifyPointCloud
  -> production CUDA path only

OpenMVS_DMAP_INSTRUMENTATION=ON
  -> DensifyPointCloudDMapObserve
  -> optional runtime capture surfaces
```

The option defaults to `OFF`. In the production build, observer CLI options,
state, allocations, diagnostic kernels, and exporters are absent. In the
observer build, runtime capture remains off until
`--dmap-instrumentation-dir` is nonempty.

The publication guard compares the disabled source projection with its Git base.
It recognizes one explicit cross-build compatibility transform in
`PatchMatchCUDA.cu`: newer CUDA toolchains require the Eigen-backed camera
constant to use aligned byte storage. That change contains no observer behavior
and applies to both products; every other unguarded depth-map source difference
fails disabled-source parity.

## Data flow

```text
frozen .mvs scene and images
  -> production endpoint capture
  -> observer summary/prefilter capture
  -> parity and schema validation
  -> aggregate quality comparison
  -> selected deep frame or trace rerun
  -> versioned structured report model
  -> canonical Markdown
  -> interactive HTML investigation client
```

The Markdown report and structured JSON/CSV/Parquet model are canonical. HTML
is a client over those validated artifacts, not an independent computation.

## Capture producers

- `apps/DensifyPointCloud/` owns the observer command-line boundary.
- `libs/MVS/PatchMatchCUDA.*` owns CUDA PatchMatch mechanics and exact maps.
- `libs/MVS/SceneDensify.cpp` owns frame selection, host-side capture,
  resource planning, postprocessing, and filtering observations.
- `scripts/python/dmap_dev.py` owns immutable experiment orchestration,
  comparison, report generation, and report validation.
- `scripts/python/dmap_observability/component_registry.py` describes signals
  independently of a particular experiment.
- `scripts/dmap_report_ui/` owns the static investigation interface.

## Evidence authority

`Process<false>` is the production CUDA specialization. Endpoint and prefilter
evidence may be used for quality ranking only after same-host output parity
succeeds. Ordinary non-APD summary capture also uses `Process<false>` and has
the same parity requirement. APD exact aggregate summary capture instead uses
`Process<true>` and is always diagnostic-only; the production endpoint remains
the quality authority.

The label `Process<false>` does not imply that the observer and production
binaries are resource-equivalent. The observer translation unit contains
additional compiled branches and parameters even when the launch passes null
observer pointers. Register, stack, spill, and occupancy measurements vary by
compiler, architecture, and source revision, so runtime metadata reports them
as unavailable unless a build-bound resource receipt exists. Treat the OFF
binary as the production resource and performance authority. Non-APD summary
and prefilter require output parity and their own timing/resource qualification.

Deep maps and admitted traces use `Process<true>` to expose candidate, cost,
view, and update mechanics. Compiling that specialization can change register
or stack pressure, and its terminal output is not assumed to match production.
It is diagnostic-only unless an explicit qualification proves otherwise.

The device-memory preflight budgets explicit observer-owned buffers. It does
not include CUDA runtime stack-pool growth caused by raising the per-thread
stack limit, and it does not guarantee that the GPU currently has enough free
physical memory. An actual CUDA allocation failure terminates the capture via
`CUDA_CHECK`; incomplete evidence is never published as a complete capture.

In public v1, pixel and ROI trace requests select the frame and the trace rows,
but exact admission still allocates and exports that frame's complete deep-map
mechanics payload. Compact exact allocation for only selected pixels is not
implemented. Storage and device/host budgets therefore use full-frame deep
accounting for an exact trace rerun.

The report carries this distinction on each relevant record. A diagnostic map
may explain why a change behaves differently, but cannot by itself establish
that production accuracy improved.

## Iteration model

CUDA PatchMatch executes checkerboard phases. The mechanics model combines the
black and red phases into a complete logical iteration:

```text
initialization
logical iteration 0 = black pass 0 + red pass 0
logical iteration 1 = black pass 1 + red pass 1
...
```

Cost evolution, update attribution, view churn, and improvement maps use this
logical model. Phase rows remain available only in timing diagnostics.

## Artifact lifecycle

Every experiment binds the source config, input scene, executable identities,
and selected runs. Capture output is written beneath an experiment root with
atomic completion markers. Existing valid output is reused; incomplete or
identity-incompatible output fails closed instead of being overwritten.

Experiment-lock schema v4 also binds the exact file set copied from each scene
working folder. The `openmvs.dmap.staged_input_snapshot` record stores sorted
relative paths, byte counts, SHA-256 identities, and one aggregate digest. It is
recomputed before every capture profile and checked again for the frozen input
and profile workspace after the process exits. The default bound is 250,000
files and 1 TiB. Symlinks, special files, concurrent mutation, limit overflow,
or any digest/path-set change stop the run. Generated depth maps, logs, dense
outputs, and instrumentation directories are excluded because they are outputs,
not frozen inputs.

After the process, logs, terminal DMAP copies, endpoint metadata, and observer
sidecars are closed, orchestration writes
`capture_artifact_closure.json`. It inventories every retained regular file by
relative path, SHA-256, byte count, and POSIX mode. Only the manifest itself and
the duplicate runtime `work/` tree are explicitly excluded; terminal DMAPs and
the complete instrumentation tree remain covered. Capture closure is mandatory
for experiment-lock schema v3 and later; the current lock schema is v4. Older
captures remain readable only as `legacy-unverified`.

Schema v4 also binds a runtime boundary for each executable role: the regular
executable plus regular sibling `lib*.so*` build products. Every capture writes
a self-digested receipt proving that boundary matched immediately before and
after the subprocess. `LD_LIBRARY_PATH`, `LD_PRELOAD`, and `LD_AUDIT` must be
empty so loader overrides cannot bypass or interpose the locked RUNPATH
resolution. System, CUDA, and package-manager libraries remain
recorded in the environment manifest rather than copied into the boundary.
Pre/post checks close ordinary concurrent rebuild and relink drift; they are not
a defense against a malicious transient replace-and-restore between checks.

Report-policy schema v3 requires `report_tree_closure.json`, written after all
report assets and campaign/finalizer bindings. It also binds capture-profile
coverage plus content identities for capture intents, capture closures, and
trace request controls, so newly admitted deep or trace evidence cannot reuse a
stale report. Schema v2 enforced only the exact report tree. The closure's own
file and the mutable Markdown validator sidecar are the only exclusions. Report
validation, reuse, portable staging, and packaging all enforce the closure.
Portable staging rewrites a sanitized tree and then creates a new closure for
that exact tree.

Closure creation deliberately reads every retained byte. This adds close-time
I/O proportional to capture/report size, but no CUDA allocation, kernel,
device-state change, or instrumentation-OFF runtime work. A closure is a drift
detector, not an authenticity signature; externally published evidence still
needs a trusted checksum/signature channel.

Per-pyramid PatchMatch resource planning happens before instrumentation allocation. It
accounts retained buffers, targeted trace records and the full trace-index map,
export scratch, fixed host overhead, and estimated sidecar/map storage. Storage
is cumulative across the PatchMatch pyramid for a frame. Coarse levels hold a
long-lived reservation for the selected full-resolution tier so early
compatibility maps cannot consume space needed by higher-value level-0 evidence.
The later postprocess/confidence producer has a separate planner and reservation
ledger using the same configured limit; public v1 therefore treats the option as
a per-producer bound rather than a combined whole-frame cap. Device, host, and
storage budgets either degrade optional signals with explicit reasons or stop
capture, depending on policy. Optional absence is never encoded as a numeric
zero.

The core PatchMatch completion marker binds the map manifest and frame/scene
summaries. Optional speckle/gap postprocess and confidence-adjustment sidecars
are produced later by `SceneDensify`; public v1 validates those artifacts
independently and does not claim that the earlier core marker attests them.

## Extensibility

Signals are registered by mechanism, quantity, units, domain, preferred
direction, minimum profile, measurement kind, color map, and exact/proxy
semantics. The report can render registered signals generically, so a new cost
component or patch metric does not require a bespoke dashboard to remain
visible. Purpose-built UI panels may be added after the generic path works.

See [schema reference](08_schema_reference.md) and [extension
guide](09_extension_guide.md).
