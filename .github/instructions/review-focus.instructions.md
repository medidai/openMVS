---
description: "Use when reviewing OpenMVS pull requests or code changes. Prioritize behavioral regressions, numerical robustness, performance risks, and missing test coverage for MVS/SFM pipelines."
name: "OpenMVS Review Focus"
---
# OpenMVS Review Focus

- Report findings first, ordered by severity, with exact file references.
- Prioritize correctness risks in reconstruction logic:
  - camera geometry and transforms
  - depth, normal, and confidence estimation
  - triangulation and fusion behavior
  - mesh reconstruction and refinement
- Check numerical robustness for threshold logic, normalization, angle/depth constraints, and float comparisons.
- Review concurrency-sensitive code (`OpenMP`, thread pools, shared state) for races and non-determinism.
- Review memory/resource handling for leaks, lifetime errors, and invalidated references.
- Verify logging and assertions follow project conventions (`ASSERT`, `DEBUG*`, `VERBOSE`) and are not noisy in hot paths.
- Flag repeated `O(n)` container lookups inside hot loops (for example repeated linear scans, `find`+`insert` double lookups, or nested search patterns) and suggest single-pass or indexed alternatives.
- Require test coverage guidance with each review:
  - suggest or run `CommonUnitTests`, `SFMPipelineTest`, `MVSPipelineTest` based on touched areas
  - call out missing targeted tests when behavior changes are not validated
- Flag performance regressions in dense pipeline hotspots and recommend timing checks for expensive paths.
- Keep summaries brief and separate from findings.
