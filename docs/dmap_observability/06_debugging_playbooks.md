# Debugging playbooks

Each playbook follows the same feedback loop: state a mechanism hypothesis,
capture production-authoritative evidence, locate a frame-level effect, capture
bounded diagnostics, explain the effect, and choose the next experiment.

## Cost-function change

**Hypothesis:** the new component reduces residual noise without creating
ambiguous minima or removing valid support.

1. Capture endpoint, summary, and prefilter for baseline and candidate.
2. Compare final annotation residuals, valid coverage, total cost distribution,
   and runtime.
3. Select frames with the largest residual or coverage deltas.
4. Capture deep maps for the same image IDs and logical iterations.
5. Inspect raw photometric cost, prior cost, geometric cost, total cost, closure
   residual, improvement maps, and winner-runner-up gap.
6. Check whether component deltas spatially align with final depth changes and
   annotated structures.

An improvement should lower production residuals and increase separation
between the winner and plausible alternatives in the intended regions. A lower
diagnostic cost alone is not evidence of better depth. Unexpected closure
residuals indicate an instrumentation or objective-accounting defect before an
algorithm conclusion.

The `cost_improvement_exact` map is positive-only: a cost increase is zero, not
a negative value. A view-set change can also change the basis of stored cost.
Interpret it with synchronized stored-cost and selected-view maps rather than
as a signed objective delta.

## Propagation change

**Hypothesis:** useful neighbor hypotheses reach uncertain pixels earlier or
more reliably without spreading across discontinuities.

Inspect logical-iteration maps for candidate source, accepted propagation,
accepted improvement, depth/normal change, selected-view-set change, and final
gap. Compare convergence speed and the spatial origin of accepted hypotheses.

Trace stable interiors, improved holes, regressions near boundaries, and a
control pixel. A desirable change increases productive propagation and reduces
late churn. If propagation gains concentrate across depth edges with worsening
line/plane residuals, the support or acceptance rule is too permissive.

## Patch or deformable-support change

**Hypothesis:** the patch gathers coherent evidence in low-texture interiors
without mixing surfaces at edges or occlusions.

Inspect reference RGB, reference variance or texture score, requested patch
mode, activation, valid sample fraction, fallback reason, raw photometric cost,
gap, and depth error proxies. Overlay the actual sampled support when available;
a bounding square is not sufficient evidence for a deformable footprint.

Stratify metrics by texture band and distance to a depth or annotation edge.
Improvement limited to interiors with stable boundary residuals supports the
hypothesis. Better coverage paired with worse residuals, lower valid sample
fraction, or frequent cross-surface support argues for a more selective shape
or stronger reliability gate.

## View-selection change

**Hypothesis:** ranking retains geometrically useful source views and rejects
unreliable or redundant ones.

Inspect the complete candidate ranking, score components, reliability weight,
selection/rejection reason, before/after masks, selected count, entropy,
per-view photometric/geometric costs, weighted contributions, and view-set
churn. Compare the same logical iteration and source-view identity across runs.

A successful change improves production residuals with coherent supporting
views and no unexplained collapse in probability mass. Watch for a single last
view absorbing invalid probability, selection changes that contribute no cost,
or lower entropy caused by accidental candidate loss.

## Textureless handling

**Hypothesis:** the reliability rule prevents noise-driven updates while still
allowing strong priors or multi-view evidence to fill textureless regions.

Define texture bands before inspecting the outcome. Compare reference variance
or the registered texture metric with eligibility, required gain, proposed
gain, accepted/rejected source, prior weight, gap, and final coverage. Report
interior low-texture regions separately from boundaries.

An effective rule reduces depth noise and update churn in the low-texture band
without simply invalidating it. If residuals improve only because coverage
collapses, it is a rejection policy, not a reconstruction improvement. If high
texture changes unexpectedly, audit gating and unavailable-value handling.

## Multiscale change

**Hypothesis:** coarse levels stabilize broad geometry and fine levels recover
detail without inheriting coarse errors.

Public v1 exposes per-pyramid-level iteration counters, timings, resource plans,
and a coarse-level `candidate_source` compatibility preview. The preview is a
proxy based on post-pass change detection: it shows where terminal state changed
but does not identify the exact winning candidate. Level 0 additionally exposes
`low_depth_prior` and the exact terminal depth/cost map family. Start by comparing
those signals across identical runs and inspect whether the fine-level prior
agrees with the terminal result. Coarse cost-component maps are explicitly
unavailable; do not infer cost evolution or a transfer mechanism from the
compatibility preview or level-0 prior alone.

Entry depth/cost, transferred depth, nearest-depth alternative, transfer delta,
fallback status, hierarchy proposal, improvement margin, and exact terminal
update attribution are extension signals, not a guaranteed public-v1 map family.
Compare them by scale and logical state only after the experiment has registered,
captured, and validated those signals as available.

Run a scale ablation with identical endpoint inputs. A useful hierarchy lowers
large-area noise at coarse levels and preserves or improves edges at fine
levels. Fine-level correction that repeatedly reverses coarse transfer suggests
the transfer or coarse objective is poorly matched. A stable but biased coarse
surface requires geometry evidence, not more iterations.

## Filtering change

**Hypothesis:** the filter removes inconsistent estimates rather than masking an
estimator regression.

Use endpoint plus prefilter. Compare validity before filtering, after each
filter stage, and at the endpoint. Inspect rejection reason, keep-cost decision,
support and confidence transitions, speckle transitions, and depth-consistency
maps. Evaluate annotations both before and after filtering when the capture
supports both stages.

Improved post-filter residuals with unchanged prefilter state isolates a filter
benefit. Worse prefilter geometry hidden by more rejection is an estimator
regression. Better coverage with worse residuals usually means thresholds were
relaxed beyond the evidence quality.

## PatchMatch iteration or convergence change

**Hypothesis:** added iterations continue to make reliable improvements rather
than churn among near-tied candidates.

Plot total/component costs, accepted-update rate, depth/normal change,
winner-runner-up gap, view churn, and runtime by complete logical iteration.
Compare terminal production metrics across iteration counts.

Stop increasing iterations when accepted gains approach zero, candidate gaps
remain ambiguous, or production accuracy no longer improves. More iterations
cannot repair a systematically wrong cost, patch, or view set.

## Minimum comparison set

For every mechanism include:

- at least one baseline and one isolated variant;
- production endpoint evidence;
- same-scene and same-frame pairing;
- accuracy and coverage metrics;
- runtime and resource evidence;
- improvement, regression, stable, boundary, and unavailable cases;
- exact/proxy labels in every diagnostic conclusion.
