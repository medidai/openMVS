---
description: "Use when running tests, fixing test failures, or validating C++ changes in OpenMVS. Covers make/ build-tree usage, target selection, and regression checks."
name: "OpenMVS Testing Workflow"
---
# OpenMVS Testing Workflow

- Prefer the existing `make/` CMake build tree for test workflows.
- Build before running tests when code changed: configure/build from `make/` using incremental builds.
- Prefer focused CTest targets first, then broaden only if needed.
- Current primary CTest targets are `CommonUnitTests`, `SFMPipelineTest`, and `MVSPipelineTest`.
- Suggested target mapping:
  - `libs/Common`, utility/infrastructure changes: `CommonUnitTests`
  - `libs/SFM`, SfM pipeline logic, or structure initialization: `SFMPipelineTest`
  - `libs/MVS` and dense/mesh pipeline changes: `MVSPipelineTest`
- Run all relevant targets for cross-cutting changes.
- When reporting failures, include failing test name, key assertion/error line, and likely regression cause.
- After a fix, rerun the affected target(s) and report pass/fail status clearly.
