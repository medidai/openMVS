# Extension guide

Use this workflow to add a cost component, candidate source, view metric,
propagation event, texture score, patch statistic, multiscale transfer, filter
transition, or annotation metric.

## 1. Define the question

Write one sentence that distinguishes the possible outcomes. For example: does
a new texture reliability term reject ambiguous low-texture updates while
retaining strongly supported updates? Avoid adding a buffer without a concrete
decision it enables.

Define:

- signal ID and label;
- mechanism and quantity;
- scalar, enum, vector, map, table, trace, or image representation;
- units, numeric domain, preferred direction, and color map;
- minimum capture profile;
- state versus event semantics;
- exact computation basis, or proxy target and limitations;
- unavailable conditions;
- expected device, host, and storage cost.

## 2. Register the signal

Add a descriptor in
`scripts/python/dmap_observability/component_registry.py`. Use a stable,
mechanism-specific ID. Unknown but registered signals must remain discoverable
through the generic report path even before a custom panel exists.

Add registry tests for duplicate IDs, invalid mechanisms, profiles, domains,
directions, and round-trip serialization.

## 3. Capture at the narrowest ownership point

Instrument the code that computes the value. Do not reconstruct an exact
component later from rounded or aggregated state. Reuse an existing value only
when its domain, rejection rules, precision, and lifetime exactly match the
signal definition.

Guard all observer code with the build option. Runtime allocations and copies
must be conditional on an active output directory and an admitted capture plan.
Do not alter production synchronization, random-number consumption, launch
topology, memory layout, or device state.

## 4. Account resources before allocation

Extend the resource plan with bytes per pixel, fixed overhead, number of states,
views, and files. Test integer overflow and each device, host, and storage
budget. Under `degrade`, record a stable unavailable reason. Under `error`, stop
before allocation or partial output.

## 5. Extend the schema and validator

Add manifest semantics, shape/channel rules, dtype, domain, exact/proxy fields,
and relationships to other signals. Validate:

- supported version;
- required identity keys;
- completion marker order;
- file size and content identity;
- shape and dtype;
- finite/domain constraints;
- logical state and timing topology;
- source-view bounds;
- unavailable-reason consistency;
- required cross-signal closure where applicable.

Test valid, missing, partial, malformed, non-finite, wrong-shape, wrong-version,
duplicate, and stale-file fixtures.

## 6. Add report presentation

First add the signal to structured model tables and generic map/table rendering.
Then add a focused plot only when it answers a recurring mechanism question.
Use shared baseline/variant scales, deterministic sorting, captions that state
what to notice, and aggregate-to-frame navigation.

For cost components, include distributions, per-iteration evolution, deltas,
and objective closure. For categorical sources or decisions, include counts and
transition matrices. For signed changes, center the color scale at zero.

## 7. Add agent and human documentation

Update `capabilities.json` when ownership or a public mechanism changes. Add the
signal to the relevant debugging playbook and schema notes. Document the exact
command needed to capture and inspect it.

## 8. Qualify incrementally

1. Unit-test registry, schema, report, and malformed evidence.
2. Compile the smallest affected C++/CUDA target.
3. Build production `OFF` and observer `ON` boundaries.
4. Prove the production binary has no observer CLI or symbols.
5. Run observer-disabled parity and confirm no artifacts.
6. Run summary/prefilter parity for quality-authoritative additions.
7. Run a bounded deep/trace capture and validate every retained signal.
8. Run the one-frame Compute Sanitizer memcheck qualification documented in the
   [quickstart](01_quickstart.md#qualify-one-cuda-diagnostic-capture); require
   `ERROR SUMMARY: 0 errors`.
9. Generate and inspect Markdown plus HTML on desktop and mobile layouts.

Do not call an extension production-ready until disabled behavior, parity,
resource limits, schema failure behavior, and report visibility have all been
validated on fresh CUDA execution.
