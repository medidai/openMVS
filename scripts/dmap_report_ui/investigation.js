(function () {
  "use strict";

  const model = JSON.parse(document.getElementById("dmap-report-model").textContent);
  const byId = (id) => document.getElementById(id);
  const controls = {
    baseline: byId("baseline-select"), baselineRepeat: byId("baseline-repeat-select"),
    variant: byId("variant-select"), variantRepeat: byId("variant-repeat-select"),
    scene: byId("scene-select"), frame: byId("frame-select"), iteration: byId("iteration-select"),
    captureStage: byId("capture-stage-select"),
    pyramidLevel: byId("pyramid-level-select"),
    alignment: byId("alignment-select"),
    mapPreset: byId("map-preset-select"),
    mechanism: byId("mechanism-select"), component: byId("component-select"),
    sourceView: byId("source-view-select"), channel: byId("channel-select"),
    annotationStage: byId("annotation-stage-select"), annotationKind: byId("annotation-kind-select"),
  };
  const state = {
    baseline: "", baselineRepeat: 0, variant: "", variantRepeat: 0,
    scene: "", frame: "", iteration: -1, pyramidLevel: "unspecified", scale: "shared", signals: [],
    sourceView: "auto", channel: 0,
    captureStage: "photometric", alignment: "final", mapPreset: "overview",
    mechanism: "all", component: "all",
    annotationStage: "post_filter", annotationKind: "all",
    x: null, y: null, sort: "regression_score", sortDirection: -1, regressionFilter: "",
  };
  const CAPTURE_PROFILE_ORDER = ["endpoint", "summary", "prefilter", "deep", "trace"];
  const CAPTURE_PROFILE_SCHEMA = "openmvs.dmap.capture_profile_coverage";
  const COLUMN_HELP = {
    "Rank": "Accuracy-first ordinal rank. Candidates are ordered by noise class, worst normalized loss, median normalized loss, then coverage tie-breakers. A high screening rank does not replace confirmation on every scene.",
    "Candidate": "Compared candidate. In the candidate ledger this is an experiment run; in CPU view tables it is a candidate source image.",
    "Noise": "Accuracy classification from paired primary line metrics. A scene/metric loss above +1 tolerance is a regression and below -1 is an improvement; both gives mixed. Losing a baseline-successful fit forces inconclusive.",
    "Availability": "Number of baseline-successful annotation structures that remain jointly evaluable. Any lost baseline fit is reported explicitly to prevent apparent gains caused by deleting difficult geometry.",
    "Scenes": "Number of scenes jointly evaluated for this candidate. One-scene rows are screening evidence; they are less confirmed than two-scene rows.",
    "Worst loss / tolerance": "Dimensionless worst primary scene/metric loss. For lower-is-better metrics: (candidate - baseline) / effective tolerance. For higher-is-better metrics: (baseline - candidate) / effective tolerance. Structure losses are averaged within each scene/metric; +1 is materially worse and -1 materially better.",
    "Effective coverage": "Mean candidate-minus-baseline change in annotation coverage multiplied by the fixed-model 20 mm inlier fraction. Reported in pp, meaning percentage points: 40% to 43% is +3 pp. Higher is better.",
    "Spatial coverage": "Mean candidate-minus-baseline change in the fraction of annotation spatial bins containing valid reconstructed samples. Reported in percentage points (pp). Higher is better.",
    "Estimator validity": "Mean candidate-minus-baseline change in valid pixels at the estimator capture stage divided by all frame pixels. Reported in percentage points (pp). This is not the terminal production DMAP endpoint.",
    "Terminal endpoint validity": "Mean candidate-minus-baseline change in valid depth pixels read from the terminal production DMAP endpoint. Reported in percentage points (pp). Higher is better.",
    "Strict accuracy": "Whether the candidate satisfied every registered strict accuracy rule in the external production-quality authority. Unavailable is distinct from a failed rule.",
    "Mechanics coverage": "Whether this candidate has matching diagnostic mechanics capture. Quality-only candidates remain valid external quality rows but have no mechanics maps in this report.",
    "Fit availability": "Paired successful structures versus baseline-successful structures. Lost baseline fits mark availability-biased evidence and cannot be hidden by averaging only surviving structures.",
    "Median loss / tolerance": "Median direction-corrected primary accuracy loss divided by the registered tolerance. Positive is worse; +1 reaches the material-regression boundary.",
    "Residual P95 delta": "Candidate-minus-baseline change in the 95th percentile annotation residual, in millimetres. Lower is better.",
    "Threshold AUC delta": "Candidate-minus-baseline change in annotation threshold AUC, in percentage points. Higher is better.",
    "5 mm inlier delta": "Candidate-minus-baseline change in the 5 mm annotation inlier fraction, in percentage points. Higher is better.",
    "Estimator validity delta": "Candidate-minus-baseline change in estimator-stage valid-depth coverage, in percentage points. It must not be confused with the terminal endpoint metric.",
    "Terminal endpoint validity delta": "Candidate-minus-baseline change in valid-depth coverage from the terminal production DMAP endpoint, in percentage points.",
    "Runtime delta": "Candidate-minus-baseline production endpoint monotonic full-process wall-time change in percent, paired at scene level after repeat aggregation.",
    "Coverage": "Declared mechanics-capture availability for the candidate under the diagnostic authority.",
    "Note": "Authority-provided qualification or availability note for this candidate.",
    "Authority": "Evidence authority that binds this artifact: diagnostic mechanics or external production quality.",
    "Artifact role": "Stable semantic role of the content-attested source artifact.",
    "Bytes": "Exact source artifact file size bound by the report policy.",
    "Cardinality": "Optional source record counts used to make omissions and scope changes visible.",
    "File SHA-256": "SHA-256 of the complete source artifact bytes.",
    "Semantic digest": "Optional source-defined digest of normalized semantic content, independent of formatting or file layout.",
    "Runtime": "Authoritative candidate-versus-baseline runtime from production endpoint monotonic full-process wall time, aggregated by scene and repeat. Negative is faster. Observer CUDA kernel timings are separate diagnostic evidence and do not drive runtime gates.",
    "Inspect": "Select this candidate or evidence row and navigate to its scene/frame for detailed inspection.",
    "Status": "Directional classification using the tolerance registered for this metric, or the execution/validation state for mechanics tables. Read the surrounding table because status semantics are table-specific.",
    "Metric": "Registered measured quantity and evidence level. Direction, tolerance, and units are defined by the metric registry.",
    "Regression score": "Direction-corrected raw delta. Lower-is-better: candidate - baseline. Higher-is-better: baseline - candidate. Production endpoint performance deltas are relative to abs(baseline). Positive is worse, but units vary by metric, so scores from different metrics must not be compared.",
    "Delta": "Candidate minus baseline before preferred-direction correction. Production endpoint performance rows divide this difference by abs(baseline); other rows retain the metric's native units. Use the metric direction to decide whether the sign is good or bad.",
    "Scene": "Stable benchmark scene identifier for this evidence row.",
    "Frame": "Scene image identifier used to navigate to frame-level mechanics and annotation evidence.",
    "Run": "Experiment run label that produced the row.",
    "Signal": "Versioned instrumentation signal or measured quantity represented by the row.",
    "Stage": "Logical estimation, filtering, confidence, or evaluation stage at which the value was recorded.",
    "Quality": "Measurement provenance class, such as exact production value, exact derived value, proxy, or unavailable. Exact and proxy evidence are not interchangeable.",
    "Value": "Decoded value at the selected shared pixel. Units and channel interpretation come from the registered signal metadata.",
    "Provenance": "How and where the value was measured, derived, or declared unavailable.",
    "Pixel": "Integer depth-map coordinate traced in both compared runs.",
    "State": "Complete logical PatchMatch initialization or iteration represented by the trace row.",
    "Update source": "Accepted update attribution for the traced logical state. The adjacent quality badge is exact only for hot-kernel rows backed by valid schema-v4 exact-map completion evidence; legacy post-pass rows remain proxy.",
    "Cost": "Recorded objective transition before and after the traced update. Lower production cost is preferred internally but does not alone prove better geometry.",
    "Depth": "Depth transition in metres at the traced pixel, including absolute change.",
    "Normal change": "Angle in degrees between the before and after surface normals at the traced pixel.",
    "Views / masks": "Selected-view count and before-to-after view bitmasks for the traced update.",
    "Payload": "Expandable per-view cost, weight, neighborhood, bad-candidate, and texture context retained by the trace.",
    "Structure": "Stable annotated line or plane identity evaluated in reconstructed 3D.",
    "Fit status": "Baseline and variant deterministic RANSAC availability. A missing candidate fit is retained as an availability loss rather than silently omitted.",
    "Effective @20 mm": "Valid annotation coverage multiplied by the inlier fraction from the single model fitted at 20 mm. Higher is better and missing depth reduces the score.",
    "Threshold AUC": "Area under the fixed-model inlier-fraction curve across the registered 5, 10, 20, and 50 mm thresholds. Higher is better.",
    "Residual P95": "95th percentile 3D distance from all valid reconstructed annotation samples to the fitted line or plane, reported in millimetres. Lower is better.",
    "Evidence": "Expand the structure to load threshold profiles, overlays, residual histograms, and provenance.",
    "Baseline median": "Median finite baseline pixel value for the selected signal, frame, and logical state.",
    "Variant median": "Median finite variant pixel value for the selected signal, frame, and logical state.",
    "Baseline P90": "90th percentile finite baseline pixel value for the selected signal, frame, and logical state.",
    "Variant P90": "90th percentile finite variant pixel value for the selected signal, frame, and logical state.",
    "Texture region": "Low, mid, or high within-frame finite-value third of the registered texture signal. These are relative strata, not universal texture thresholds.",
    "Pixels": "Number of finite pixels contributing to this aggregate row.",
    "Baseline mean": "Arithmetic mean of finite baseline values in the selected cohort.",
    "Variant mean": "Arithmetic mean of finite variant values in the selected cohort.",
    "Tested": "Number of candidate hypotheses evaluated during this complete logical state.",
    "Finite": "Number of tested candidates whose recorded objective was finite and usable.",
    "Accepted / stored": "Sequential accepted updates for iterations, or stored assignments for initialization. Initialization is not a propagation-win count.",
    "Meaning": "Interpretation of the count semantics for initialization versus sequential logical iterations.",
    "Gap P50": "Median exact winning-versus-runner-up candidate cost gap. Larger positive gaps indicate more decisive recorded winners.",
    "Gap P90": "90th percentile exact winning-versus-runner-up candidate cost gap.",
    "Source counts": "Accepted or stored candidates grouped by registered update source, such as propagation, perturbation, depth refinement, normal refinement, prior guidance, or view-set change.",
    "Eligible pixels": "Pixels processed once in the complete logical iteration with a valid coarse prior, reference variance below the configured threshold, positive minimum gain, and a nonzero gate mask.",
    "Propagation accepted": "Legacy-improving propagation proposals that also exceeded the ambiguity-scaled required gain.",
    "Propagation rejected": "Legacy-improving propagation proposals suppressed only by the low-texture hysteresis margin.",
    "Propagation rejection": "Propagation rejected divided by propagation accepted plus rejected. It is a proposal rate, not a fraction of all pixels.",
    "Refinement accepted": "Legacy-improving gate-controlled refinement proposals that also exceeded the required gain. Multiple sequential refinements can occur at one pixel.",
    "Refinement rejected": "Legacy-improving gate-controlled refinement proposals suppressed only by hysteresis. Multiple sequential refinements can occur at one pixel.",
    "Refinement rejection": "Refinement rejected divided by refinement accepted plus rejected. It is a proposal rate and can count multiple proposals per pixel.",
    "Mean required gain": "Sum of ambiguity-scaled required gains divided by eligible pixels for this complete logical iteration, in cost units.",
    "Mean best proposed gain": "Sum of each eligible pixel's largest positive legacy gain among gate-controlled proposals divided by eligible pixels, in cost units. Pixels with no improving proposal contribute zero.",
    "Low-texture pixels": "Finite exact-cost pixels whose reference variance is below this run's explicitly configured low-texture threshold. Coarse-prior eligibility is not inferred in the pre-change census.",
    "Accepted gain pixels": "Low-texture pixels where exact candidate winner cost is lower than exact iteration-entry incumbent cost.",
    "Gain P50": "Median positive incumbent-minus-winner cost gain in the configured low-texture region.",
    "Below 0.00025": "Fraction of positive low-texture retained-winner gains strictly below 0.00025 cost units.",
    "Below 0.0005": "Fraction of positive low-texture retained-winner gains strictly below 0.0005 cost units.",
    "Below 0.001": "Fraction of positive low-texture retained-winner gains strictly below 0.001 cost units.",
    "View": "Zero-based source-view slot within the PatchMatch frame configuration.",
    "Image": "Scene image identifier assigned to the source-view slot.",
    "Selected": "Fraction of evaluated pixels for which this source view was selected.",
    "Weight": "Mean recorded reliability weight for this source view over evaluated pixels.",
    "Probability": "Mean recorded source-view selection probability over evaluated pixels.",
    "Contribution": "Mean reliability-weighted cost contribution from this source view.",
    "Total cost": "Mean recorded total candidate cost associated with this source-view summary.",
    "Raw rank": "Zero-based CPU candidate rank before threshold filtering and maximum-view truncation. Lower ranks first.",
    "Score": "Recorded CPU source-view ranking score assembled from sparse-visibility angle, scale, ROI, and area terms. Higher scores rank earlier.",
    "Initial": "CPU candidate-ranking decision before filtering.",
    "Filter": "Threshold and maximum-view filtering decision applied to the CPU candidate.",
    "Final rank": "Zero-based rank after CPU candidate filtering; unavailable for rejected candidates.",
    "Accepted": "Whether the CPU source-view candidate survived filtering.",
	"Filtered rank": "Zero-based source-view rank entering estimation selection after candidate filtering.",
	"Ratio": "Candidate ranking score divided by the best filtered score for the same reference frame.",
	"Admission policy": "Host-side policy that admits geometrically retained ranked views into PatchMatch.",
	"Score cutoff applied": "Whether View Min Score or View Min Score Ratio rejected PatchMatch source views.",
	"Configured cutoff": "The effective max(absolute score threshold, best score times relative threshold) that would have applied under upstream PatchMatch admission.",
	"Cutoff status": "Whether the configured threshold was applied, reported only as a counterfactual, or not applicable.",
	"Would pass configured cutoff": "Counterfactual result of the configured score threshold. It is diagnostic only when the admission policy bypasses score rejection.",
	"Selected rank": "Zero-based rank among source views selected for depth estimation; unavailable when rejected.",
    "Decision": "Recorded admission, degradation, rejection, or selection decision for this component.",
    "Sequence": "Zero-based execution order of the postprocess stage.",
    "Enabled": "Whether configuration enabled this stage.",
    "Executed": "Whether the enabled stage actually ran and produced an observation.",
    "Valid in": "Valid-depth pixel count entering the stage.",
    "Valid out": "Valid-depth pixel count leaving the stage.",
    "Removed": "Pixels valid on input and invalid on output.",
    "Added": "Pixels invalid on input and valid on output.",
    "Depth changed": "Pixels valid before and after whose stored depth value changed.",
    "Mean |depth delta|": "Mean absolute depth change over all frame pixels, in metres; unchanged and invalid pixels contribute according to the stage contract.",
    "Unavailable reason": "Explicit reason the requested signal or observation could not be produced.",
    "Method": "Confidence-adjustment algorithm or observation method.",
    "Output": "Whether an adjusted confidence output was available.",
    "Positive in": "Number of pixels with positive confidence entering adjustment.",
    "Positive out": "Number of pixels with positive confidence after adjustment.",
    "Changed": "Number of pixels whose confidence value changed.",
    "Mean |delta|": "Mean absolute confidence difference over all frame pixels.",
    "Combination": "Recorded rule used to combine adjusted confidence with the final stored output.",
    "Component": "Instrumented CUDA, filtering, or storage component covered by the resource plan.",
    "Maps requested": "Number or boolean declaration of map outputs requested before resource admission.",
    "Maps admitted": "Number or boolean declaration of map outputs admitted after device, host, storage, and filesystem checks.",
    "Device MiB": "Estimated additional CUDA device allocation after admission, in mebibytes (2^20 bytes).",
    "Host MiB": "Estimated retained host allocation after admission, in mebibytes (2^20 bytes).",
    "Storage MiB": "Estimated uncompressed frame output after admission, in mebibytes (2^20 bytes).",
    "Preflight": "Whether filesystem capacity and configured resource limits passed before allocation or output.",
    "Lease released": "Whether the in-process resource reservation was released after completion or failure.",
    "Actual maps": "Number of map artifacts actually written and validated.",
    "Valid": "Validator result for the resource plan or produced artifact set.",
    "Reason": "Recorded explanation for the resource decision or validation result.",
    "Ignore mask": "Ignore-mask request/load status for the selected frame and run.",
    "Requested": "Whether configuration requested this input or output.",
    "Loaded": "Whether the requested input was successfully loaded and applied.",
    "Mask-rejected pixels": "Pixels rejected by the loaded ignore mask. Unavailable is distinct from a measured zero.",
  };
  const FALLBACK_GUIDE = {
    title: "Investigation guide",
    introduction: "Use one comparison state for the selectors, mechanics tables, maps, and pixel inspector. Every recipe below updates that shared state and records it in the URL.",
    quick_start: {
      title: "Baseline versus variant",
      summary: "Start with a matched frame and inspect the result before narrowing the comparison to a mechanism or logical iteration.",
      steps: [
        "Choose the baseline and variant runs, including repeats, then select a scene, frame, capture stage, and pyramid level.",
        "Use Final state per run for the end result. Use Same logical iteration when both runs expose the same iteration and you need a like-for-like algorithm state.",
        "Keep Shared map scale for visual comparison. Click the same location in any map to read synchronized numeric values in Pixel inspection.",
        "Treat quality and availability labels as part of the evidence. Exact, derived, proxy, and unavailable signals are not interchangeable.",
      ],
      action_label: "Load comparison overview",
    },
    recipes: [
      {
        key: "cost", title: "Inspect a cost change",
        summary: "Localize which recorded objective component moved and whether the winning decision was decisive.",
        steps: [
          "Compare the same logical iteration first, then switch to final-state alignment to include convergence differences.",
          "Read the exact production total and available components beside stored cost and winner-versus-runner-up gap.",
          "Use a shared crosshair to compare exact pixel payloads where the spatial maps diverge.",
        ],
        look_for: ["A total-cost change that is spatially colocated with one component.", "Changed winner gaps or acceptance counts near the same region."],
        cautions: ["A lower internal cost does not by itself establish better geometry.", "Do not interpret proxy rescoring as the production objective."],
        action_label: "Load cost evidence",
      },
      {
        key: "propagation", title: "Inspect a propagation change",
        summary: "Separate changes in propagation attribution from changes in refinement or candidate acceptance.",
        steps: [
          "Use Same logical iteration so source counts describe comparable passes.",
          "Read Candidate update attribution for propagate and refinement source counts, tested candidates, accepted candidates, and winner gap.",
          "Localize candidate identity, acceptance, depth change, and normal change with synchronized maps.",
        ],
        look_for: ["Propagation counts moving without a matching acceptance change.", "Accepted updates clustering where depth or normal deltas also move."],
        cautions: ["Initialization rows are stored assignments, not sequential propagation wins.", "A candidate-source map marked proxy cannot replace exact attribution counts."],
        action_label: "Load propagation evidence",
      },
      {
        key: "view_selection", title: "Inspect a view-selection change",
        summary: "Check probability health before interpreting selected views, weights, and contribution.",
        steps: [
          "Select the pyramid level and same logical iteration where the rule executes.",
          "Inspect probability mass, positive-view count, health status, unassigned draws, and predicted legacy collapse.",
          "Select individual source views to compare exact decisions, weights, costs, and weighted contribution.",
        ],
        look_for: ["Degenerate probability mass aligned with last-view collapse.", "Improved probability health without concentration on an unreliable source view."],
        cautions: ["Pre-CDF health explains mechanics but does not prove final geometric improvement.", "Unavailable exact maps must not be reconstructed from summary counters."],
        action_label: "Load view-selection evidence",
      },
      {
        key: "patch", title: "Inspect a patch or scoring change",
        summary: "Trace a patch-related code change through texture evidence, photometric scoring, finite candidates, and view contribution.",
        steps: [
          "Start at the same logical iteration and select a source view when per-view exact maps are available.",
          "Compare reference variance, exact photometric cost, finite and accepted candidate masks, and per-view contribution.",
          "Cross-check the affected pixels against the reference image before attributing the change to texture or patch support.",
        ],
        look_for: ["Score changes concentrated in low-variance or boundary regions.", "Finite-candidate or selected-view changes that coincide with the score change."],
        cautions: ["The report does not reconstruct raw patch crops; these signals are downstream evidence.", "Keep exact and proxy cost maps separate when drawing conclusions."],
        action_label: "Load patch-scoring evidence",
      },
      {
        key: "multiscale", title: "Inspect a multiscale change",
        summary: "Find the pyramid level where two runs first diverge, then connect that change to the final endpoint.",
        steps: [
          "Select the same pyramid level in both runs and compare initialization, complete logical iterations, and final state separately.",
          "Inspect the same pixels with Shared display scale and compare capture stages separately.",
          "Use any transfer or hierarchy signals declared by the component registry; their absence must remain explicit.",
          "Use same-iteration comparisons only when logical iterations have the same meaning in both runs.",
        ],
        look_for: ["Changes at thin structures, depth discontinuities, weak texture, or low support.", "Cost changes that agree with depth, normal, support, or validity changes."],
        cautions: ["Pyramid level selects algorithm state; Shared and Local control only preview color normalization.", "An Unspecified level means the capture did not retain level identity and cannot support level attribution."],
        action_label: "Load multiscale end effects",
      },
    ],
  };
  const GUIDE_RECIPE_ACTIONS = {
    overview: {
      alignment: "final", mapPreset: "overview", sourceView: "auto", target: "maps-heading",
    },
    cost: {
      alignment: "same", target: "mechanics-heading",
      preferredSignals: [
        "reference_rgb", "cost_improvement_exact", "cost_total_production_exact", "cost_photo_raw_production_exact",
        "cost_photo_prior_production_exact", "cost_geometric_production_exact", "cost_stored",
        "cost_stored_minus_rescore", "gap_winner_runner_up_exact",
        "gap_raw_best_runner_up_exact", "candidate_retained_minus_raw_best_exact",
      ],
      mechanisms: ["cost", "candidate_update"], keywords: ["cost", "gap"],
    },
    propagation: {
      alignment: "same", target: "mechanics-heading",
      preferredSignals: [
        "reference_rgb", "cost_improvement_exact", "candidate_identity_exact", "candidate_accepted_mask_exact",
        "candidate_counts_exact", "gap_winner_runner_up_exact",
        "candidate_raw_suppression_identity_exact", "gap_raw_best_runner_up_exact",
        "candidate_retained_minus_raw_best_exact", "depth_delta",
        "normal_angle_delta", "candidate_source",
      ],
      mechanisms: ["candidate_update", "update"], keywords: ["candidate", "delta", "update"],
    },
    view_selection: {
      alignment: "same", target: "maps-heading", sourceView: "auto",
      preferredSignals: [
        "reference_rgb", "view_probability_mass", "view_probability_positive_count",
        "view_probability_health_status", "view_probability_unassigned_draw_count",
        "view_probability_legacy_last_view_collapse", "view_selection_state_exact",
        "view_weighted_contribution_exact",
      ],
      mechanisms: ["view_selection"], keywords: ["view", "probability", "weight", "contribution"],
    },
    patch: {
      alignment: "same", sourceView: "first", target: "maps-heading",
      preferredSignals: [
        "reference_rgb", "reference_variance_production_exact", "cost_photo_raw_production_exact",
        "cost_photo_prior_production_exact", "view_cost_components_exact",
        "candidate_finite_mask_exact", "candidate_accepted_mask_exact", "gap_winner_runner_up_exact",
      ],
      mechanisms: ["cost", "view_selection", "candidate_update"],
      keywords: ["photo", "variance", "finite", "accepted", "view_cost"],
    },
    multiscale: {
      alignment: "final", sourceView: "auto", target: "maps-heading",
      preferredSignals: [
        "reference_rgb", "cost_total_production_exact", "gap_winner_runner_up_exact",
        "depth_delta", "normal_angle_delta", "view_churn", "depth_final_after_filter",
      ],
      mechanisms: ["multiscale"],
      keywords: ["scale", "pyramid", "transfer", "hierarchy"],
    },
  };
  let renderedTiles = [];
  let deltaJobs = [];
  let guideOpener = null;
  let columnTooltip = null;
  let activeColumnHelp = null;
  const guideRecipeActions = new Map();
  const guideRecipeExactKeys = new Set(Object.keys(GUIDE_RECIPE_ACTIONS));
  const decodedPayloads = new Map();

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
    })[character]);
  }

  function format(value, digits = 5) {
    if (value == null || Number.isNaN(Number(value))) return "n/a";
    const number = Number(value);
    if (Math.abs(number) >= 10000 || (Math.abs(number) > 0 && Math.abs(number) < 0.0001)) return number.toExponential(3);
    return number.toFixed(digits).replace(/\.?0+$/, "");
  }

  function finiteNumber(value) {
    if (value == null || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function formatPercent(value, digits = 1) {
    const number = finiteNumber(value);
    return number == null ? "n/a" : `${(number * 100).toFixed(digits)}%`;
  }

  function formatPercentagePoints(value, digits = 1) {
    const number = finiteNumber(value);
    if (number == null) return "n/a";
    const percentagePoints = number * 100;
    return `${percentagePoints > 0 ? "+" : ""}${percentagePoints.toFixed(digits)} pp`;
  }

  function formatMillimetres(value, digits = 1) {
    const number = finiteNumber(value);
    return number == null ? "n/a" : `${(number * 1000).toFixed(digits)} mm`;
  }

  function formatMillimetreDelta(value, digits = 1) {
    const number = finiteNumber(value);
    if (number == null) return "n/a";
    const millimetres = number * 1000;
    return `${millimetres > 0 ? "+" : ""}${millimetres.toFixed(digits)} mm`;
  }

  function ensureColumnTooltip() {
    if (columnTooltip) return columnTooltip;
    columnTooltip = document.createElement("div");
    columnTooltip.id = "column-tooltip";
    columnTooltip.className = "column-tooltip";
    columnTooltip.setAttribute("role", "tooltip");
    columnTooltip.hidden = true;
    document.body.append(columnTooltip);
    return columnTooltip;
  }

  function hideColumnTooltip() {
    if (!columnTooltip) return;
    columnTooltip.hidden = true;
    activeColumnHelp = null;
  }

  function positionColumnTooltip(target) {
    const tooltip = ensureColumnTooltip();
    const targetRect = target.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const margin = 8;
    const gap = 7;
    const centered = targetRect.left + targetRect.width / 2 - tooltipRect.width / 2;
    const left = Math.max(margin, Math.min(centered, window.innerWidth - tooltipRect.width - margin));
    let top = targetRect.bottom + gap;
    if (top + tooltipRect.height > window.innerHeight - margin)
      top = Math.max(margin, targetRect.top - tooltipRect.height - gap);
    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;
  }

  function showColumnTooltip(target) {
    const tooltip = ensureColumnTooltip();
    tooltip.textContent = target.dataset.tooltip || "";
    tooltip.hidden = false;
    activeColumnHelp = target;
    positionColumnTooltip(target);
  }

  function installColumnTooltips() {
    ensureColumnTooltip();
    const headers = document.querySelectorAll("th, .annotation-structure-head > span");
    headers.forEach((header) => {
      if (header.querySelector(":scope > .column-help")) return;
      const sortButton = header.querySelector("button[data-sort]");
      const label = String(sortButton?.textContent || header.textContent || "Column").trim();
      const defined = Object.prototype.hasOwnProperty.call(COLUMN_HELP, label);
      const help = COLUMN_HELP[label] || `Values reported in the ${label} column. Consult the surrounding table heading and report model for the registered formula, units, and preferred direction.`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "column-help";
      button.textContent = "?";
      button.dataset.tooltip = help;
      button.setAttribute("aria-label", `${label}: ${help}`);
      button.setAttribute("aria-describedby", "column-tooltip");
      header.dataset.columnHelp = defined ? "defined" : "fallback";
      header.append(button);
      button.addEventListener("mouseenter", () => showColumnTooltip(button));
      button.addEventListener("mouseleave", () => {
        if (document.activeElement !== button) hideColumnTooltip();
      });
      button.addEventListener("focus", () => showColumnTooltip(button));
      button.addEventListener("blur", hideColumnTooltip);
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        showColumnTooltip(button);
      });
    });
  }

  function option(value, label, selected) {
    return `<option value="${escapeHtml(value)}"${selected ? " selected" : ""}>${escapeHtml(label)}</option>`;
  }

  function asArray(value) {
    if (value == null) return [];
    return Array.isArray(value) ? value : [value];
  }

  function normalizeGuideRecipeKey(value) {
    return String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  }

  function guideRecipeKey(value) {
    const key = normalizeGuideRecipeKey(value);
    if (guideRecipeExactKeys.has(key)) return key;
    if (key.includes("propagat")) return "propagation";
    if (key.includes("patch") || key.includes("scor")) return "patch";
    if (key.includes("multi") || key.includes("pyramid") || key.includes("resolution")) return "multiscale";
    if (key.includes("cost") || key.includes("objective")) return "cost";
    if (key.includes("overview") || key.includes("baseline") || key.includes("quick")) return "overview";
    return key;
  }

  function guideItemParts(item) {
    if (item == null) return { heading: "", body: "" };
    if (typeof item !== "object") return { heading: "", body: String(item) };
    const heading = item.title ?? item.label ?? item.control ?? "";
    const body = item.text ?? item.description ?? item.instruction ?? item.action ??
      item.selection ?? item.value ?? item.expected ?? item.why ?? "";
    if (body !== "") return { heading: String(heading), body: String(body) };
    const fallback = Object.entries(item)
      .filter(([key, value]) => !["key", "title", "label", "control"].includes(key) && ["string", "number", "boolean"].includes(typeof value))
      .map(([, value]) => String(value)).join(" / ");
    return { heading: String(heading), body: fallback };
  }

  function guideItemsHtml(items, ordered = false) {
    const rows = asArray(items).map(guideItemParts).filter((item) => item.heading || item.body);
    if (!rows.length) return "";
    const tag = ordered ? "ol" : "ul";
    const className = ordered ? "guide-steps" : "";
    return `<${tag}${className ? ` class="${className}"` : ""}>${rows.map((item) =>
      `<li>${item.heading ? `<strong>${escapeHtml(item.heading)}</strong>` : ""}${escapeHtml(item.body)}</li>`
    ).join("")}</${tag}>`;
  }

  function guideSummary(section) {
    return section?.summary ?? section?.description ?? section?.question ?? section?.purpose ?? "";
  }

  function guideSectionHtml(section, index, quickStart = false) {
    const rawKey = section?.key ?? section?.recipe_key ?? section?.id ?? (quickStart ? "overview" : `recipe_${index}`);
    const actionKey = quickStart ? "overview" : guideRecipeKey(rawKey);
    const sectionId = quickStart ? "guide-quick-start" : `guide-recipe-${String(rawKey).replace(/[^a-zA-Z0-9_-]+/g, "-")}-${index}`;
    const title = section?.title || (quickStart ? "Baseline versus variant" : `Debugging recipe ${index + 1}`);
    const steps = section?.steps ?? section?.workflow ?? section?.actions ?? [];
    const observations = section?.look_for ?? section?.evidence ?? section?.observations ?? section?.interpretation ?? [];
    const cautions = section?.cautions ?? section?.limitations ?? section?.guardrails ?? [];
    const action = section?.action && typeof section.action === "object" ? section.action : {};
    if (Object.keys(action).length) guideRecipeActions.set(actionKey, action);
    const actionLabel = section?.action_label ?? action.label ?? (quickStart ? "Load comparison overview" : `Load ${title.toLowerCase()}`);
    const canApply = Object.prototype.hasOwnProperty.call(GUIDE_RECIPE_ACTIONS, actionKey)
      || Object.keys(action).length > 0;
    const current = quickStart ? `<div class="guide-current"><strong>Current comparison</strong><span id="guide-current-comparison"></span><strong>Current frame</strong><span id="guide-current-frame"></span></div>` : "";
    const reading = observations.length || cautions.length ? `<div class="guide-reading">
      ${observations.length ? `<div><h4>Look for</h4>${guideItemsHtml(observations)}</div>` : ""}
      ${cautions.length ? `<div class="guide-caution"><h4>Interpret carefully</h4>${guideItemsHtml(cautions)}</div>` : ""}
    </div>` : "";
    const actionButton = canApply ? `<button class="guide-action" type="button" data-guide-recipe="${escapeHtml(actionKey)}" data-guide-label="${escapeHtml(title)}">${escapeHtml(actionLabel)}</button>
      <p class="guide-action-note">Updates the controls, mechanics, maps, pixel selection, and URL as one comparison state.</p>` : "";
    return {
      id: sectionId,
      title,
      html: `<section id="${escapeHtml(sectionId)}" class="guide-section">
        ${quickStart ? `<p class="eyebrow">WORKED EXAMPLE</p>` : ""}
        <h3>${escapeHtml(title)}</h3>
        ${guideSummary(section) ? `<p class="guide-summary">${escapeHtml(guideSummary(section))}</p>` : ""}
        ${current}${guideItemsHtml(steps, true)}${reading}${actionButton}
      </section>`,
    };
  }

  function guideRecipes(guide) {
    const value = guide.recipes ?? guide.debugging_recipes ?? guide.debug_recipes ?? FALLBACK_GUIDE.recipes;
    if (Array.isArray(value)) return value;
    if (value && typeof value === "object") return Object.entries(value).map(([key, recipe]) => ({ key, ...(recipe || {}) }));
    return FALLBACK_GUIDE.recipes;
  }

  function renderGuide() {
    const guide = model.investigation_guide || FALLBACK_GUIDE;
    guideRecipeActions.clear();
    guideRecipeExactKeys.clear();
    Object.keys(GUIDE_RECIPE_ACTIONS).forEach((key) => guideRecipeExactKeys.add(key));
    const quickStart = guide.quick_start ?? guide.worked_example ?? guide.example_workflow ?? FALLBACK_GUIDE.quick_start;
    const introduction = guide.introduction ?? guide.intro ?? guide.summary ?? FALLBACK_GUIDE.introduction;
    const recipes = guideRecipes(guide);
    recipes.forEach((recipe, index) => {
      const rawKey = recipe?.key ?? recipe?.recipe_key ?? recipe?.id ?? `recipe_${index}`;
      guideRecipeExactKeys.add(normalizeGuideRecipeKey(rawKey));
    });
    const sections = [guideSectionHtml(quickStart, 0, true), ...recipes.map((recipe, index) => guideSectionHtml(recipe, index, false))];
    byId("guide-title").textContent = guide.title || FALLBACK_GUIDE.title;
    byId("guide-navigation").innerHTML = sections.map((section) =>
      `<button type="button" data-guide-target="${escapeHtml(section.id)}">${escapeHtml(section.title)}</button>`
    ).join("");
    byId("guide-content").innerHTML = `<p id="guide-introduction" class="guide-introduction">${escapeHtml(introduction)}</p>
      ${sections.map((section) => section.html).join("")}
      <p class="guide-footer">Continue with <a href="01_development_report.md">the canonical Markdown report</a> or <a href="report_model.json">the structured report model</a>.</p>`;
    byId("guide-navigation").querySelectorAll("[data-guide-target]").forEach((button) => button.addEventListener("click", () => {
      byId(button.dataset.guideTarget)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    byId("guide-content").querySelectorAll("[data-guide-recipe]").forEach((button) => button.addEventListener("click", () => {
      applyGuideRecipe(button.dataset.guideRecipe, button.dataset.guideLabel);
    }));
    renderGuideContext();
  }

  function renderGuideContext() {
    const comparison = byId("guide-current-comparison");
    const currentFrame = byId("guide-current-frame");
    if (comparison) comparison.textContent = `${state.baseline || "n/a"} / repeat ${state.baselineRepeat} versus ${state.variant || "n/a"} / repeat ${state.variantRepeat}`;
    if (currentFrame) {
      const selectedScene = scene();
      const selectedFrame = frame();
      currentFrame.textContent = selectedFrame ? `${selectedScene?.label || state.scene} / image ${selectedFrame.image_id} / ${state.captureStage} / ${pyramidLevelLabel(String(state.pyramidLevel))}` : "No frame selected";
    }
  }

  function availableGuideSignals(action) {
    const catalog = model.signals || [];
    const available = catalog.filter((signal) => signal.name === "reference_rgb" || Number(signal.available_artifacts) > 0);
    const byName = new Map(available.map((signal) => [signal.name, signal]));
    const selected = [];
    const add = (name) => {
      if (!selected.includes(name) && (name === "reference_rgb" || byName.has(name))) selected.push(name);
    };
    add("reference_rgb");
    (action.preferredSignals || []).forEach(add);
    const mechanisms = new Set(action.mechanisms || []);
    const keywords = action.keywords || [];
    available
      .filter((signal) => mechanisms.has(signal.mechanism) && (!keywords.length || keywords.some((word) => `${signal.name} ${signal.label}`.includes(word))))
      .sort((left, right) => {
        const exactLeft = (left.measurement_qualities || []).some((quality) => quality === "exact" || quality === "derived_exact") ? 0 : 1;
        const exactRight = (right.measurement_qualities || []).some((quality) => quality === "exact" || quality === "derived_exact") ? 0 : 1;
        return exactLeft - exactRight || Number(Boolean(right.default)) - Number(Boolean(left.default)) || left.name.localeCompare(right.name);
      })
      .forEach((signal) => add(signal.name));
    return selected.slice(0, 8);
  }

  function configuredGuideSignals(signals) {
    const declared = new Set((model.signals || []).map((signal) => signal.name));
    return [...new Set(asArray(signals).map(String))].filter((name) => name === "reference_rgb" || declared.has(name));
  }

  function commonLogicalIteration() {
    const currentFrame = frame();
    if (!currentFrame) return null;
    const stageKey = (map) => map.estimation_stage === "geometric_consistency" ? `geometric_consistency:${map.geometric_iteration}` : "photometric";
    const iterationRows = (currentFrame.run_frames || []).flatMap((row) => row.iterations || []);
    const candidates = [...(currentFrame.maps || []), ...iterationRows];
    const iterationsFor = (run, repeat) => new Set(candidates
      .filter((map) => map.run === run && Number(map.repeat) === Number(repeat) && stageKey(map) === state.captureStage && pyramidLevelMatches(map) && map.logical_iteration != null)
      .map((map) => Number(map.logical_iteration)));
    const baseline = iterationsFor(state.baseline, state.baselineRepeat);
    const variant = iterationsFor(state.variant, state.variantRepeat);
    const common = [...baseline].filter((iteration) => variant.has(iteration)).sort((left, right) => left - right);
    if (!common.length) return null;
    if (Number(state.iteration) >= 0 && common.includes(Number(state.iteration))) return Number(state.iteration);
    return common.find((iteration) => iteration >= 0) ?? common[0];
  }

  function firstSourceView() {
    const currentFrame = frame();
    if (!currentFrame) return "auto";
    const views = (currentFrame.maps || [])
      .filter((map) => map.source_view_index != null && [state.baseline, state.variant].includes(map.run) && pyramidLevelMatches(map))
      .map((map) => Number(map.source_view_index)).filter(Number.isFinite).sort((left, right) => left - right);
    return views.length ? String(views[0]) : "auto";
  }

  function announce(message) {
    byId("ui-announcer").textContent = "";
    window.setTimeout(() => { byId("ui-announcer").textContent = message; }, 20);
  }

  function applyGuideRecipe(key, label) {
    const recipeKey = guideRecipeKey(key);
    const fallbackAction = GUIDE_RECIPE_ACTIONS[recipeKey];
    const modelAction = guideRecipeActions.get(recipeKey);
    if (!fallbackAction && !modelAction) return;
    const action = { ...(fallbackAction || {}), ...(modelAction || {}) };
    state.alignment = action.alignment || state.alignment;
    if (state.alignment === "same") {
      const common = commonLogicalIteration();
      if (common == null) state.alignment = "final"; else state.iteration = common;
    }
    state.scale = "shared";
    state.sourceView = action.sourceView === "first" ? firstSourceView() : (action.sourceView || "auto");
    state.channel = 0;
    state.x = null;
    state.y = null;
    const mapPreset = action.map_preset || action.mapPreset;
    const configuredSignals = configuredGuideSignals(action.signals);
    if (mapPreset) {
      state.mapPreset = mapPreset;
    }
    if (configuredSignals.length) {
      state.mapPreset = "custom";
      state.signals = configuredSignals;
    } else if (mapPreset) {
      applyMapPreset();
    } else {
      state.mapPreset = "custom";
      state.signals = availableGuideSignals(action);
    }
    renderAll();
    closeGuide();
    window.setTimeout(() => {
      const target = byId(action.target)?.closest(".section-band") || byId(action.target);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 40);
    announce(`${label || "Guide recipe"} loaded with ${state.signals.length} visible signals and ${state.alignment === "same" ? "same-iteration" : "final-state"} alignment.`);
  }

  function openGuide() {
    const dialog = byId("guide-dialog");
    guideOpener = document.activeElement;
    renderGuideContext();
    document.body.classList.add("guide-open");
    if (typeof dialog.showModal === "function") dialog.showModal(); else dialog.setAttribute("open", "");
    byId("guide-close").focus();
  }

  function closeGuide() {
    const dialog = byId("guide-dialog");
    if (!dialog.open) return;
    if (typeof dialog.close === "function") dialog.close(); else dialog.removeAttribute("open");
  }

  function runsFor(label) {
    return model.runs.filter((run) => run.label === label).sort((a, b) => a.repeat - b.repeat);
  }

  function runRecord(label, repeat = null) {
    const candidates = runsFor(label);
    if (repeat == null) return candidates[0];
    return candidates.find((run) => Number(run.repeat) === Number(repeat)) || candidates[0];
  }

  function runSelectorLabel(label) {
    return runRecord(label)?.diagnostic_only
      ? `${label} (diagnostic mechanics only)`
      : label;
  }

  function selectedDiagnosticRuns() {
    return [
      runRecord(state.baseline, state.baselineRepeat),
      runRecord(state.variant, state.variantRepeat),
    ].filter((run) => run?.diagnostic_only);
  }

  function scene() { return model.scenes.find((item) => item.id === state.scene) || model.scenes[0]; }
  function frame() { const current = scene(); return current && current.frames.find((item) => item.id === state.frame) || (current && current.frames[0]); }

  function pyramidLevelOf(row) {
    for (const key of ["pyramid_level", "scale_level", "scale_number"]) {
      const value = row?.[key];
      if (value == null || value === "") continue;
      const level = Number(value);
      if (Number.isInteger(level) && level >= 0) return String(level);
    }
    return "unspecified";
  }

  function pyramidLevelLabel(value) {
    if (value === "unspecified") return "Unspecified";
    return Number(value) === 0 ? "0 / full res" : `${value} / coarser`;
  }

  function pyramidLevelMatches(row) {
    return pyramidLevelOf(row) === String(state.pyramidLevel);
  }

  function mechanicsPyramidLevelMatches(row) {
    const level = pyramidLevelOf(row);
    return level === "unspecified" || level === String(state.pyramidLevel);
  }

  function initializeState() {
    const qualityRuns = model.runs.filter((run) => !run.diagnostic_only);
    const defaultRuns = qualityRuns.length ? qualityRuns : model.runs;
    const baselineRuns = defaultRuns.filter((run) => run.role === "baseline");
    const variantRuns = defaultRuns.filter((run) => run.role !== "baseline");
    state.baseline = (baselineRuns[0] || defaultRuns[0] || {}).label || "";
    state.baselineRepeat = (baselineRuns[0] || defaultRuns[0] || {}).repeat || 0;
    state.variant = (variantRuns[0] || defaultRuns.find((run) => run.label !== state.baseline) || baselineRuns[0] || {}).label || "";
    state.variantRepeat = (variantRuns[0] || defaultRuns.find((run) => run.label === state.variant) || {}).repeat || 0;
    state.scene = (model.scenes[0] || {}).id || "";
    state.frame = ((model.scenes[0] || {}).frames || [])[0]?.id || "";
    const iterations = frame()?.logical_iterations || [];
    state.iteration = iterations.includes(-1) ? -1 : (iterations[0] ?? -1);
    const availableNames = new Set(model.signals.filter((signal) => signal.available_artifacts > 0 || signal.name === "reference_rgb").map((signal) => signal.name));
    state.signals = model.signals.filter((signal) => signal.default && availableNames.has(signal.name)).slice(0, 8).map((signal) => signal.name);
    parseHash();
  }

  function parseHash() {
    const params = new URLSearchParams(location.hash.replace(/^#/, ""));
    ["baseline", "variant", "scene", "frame", "pyramidLevel", "scale", "sourceView", "captureStage", "alignment", "mapPreset", "mechanism", "component", "annotationStage", "annotationKind"].forEach((key) => {
      if (params.has(key)) state[key] = params.get(key);
    });
    ["baselineRepeat", "variantRepeat", "iteration", "channel"].forEach((key) => {
      if (params.has(key)) state[key] = Number(params.get(key));
    });
    if (params.has("signals")) state.signals = params.get("signals").split(",").filter(Boolean);
    if (params.has("x") && params.has("y")) { state.x = Number(params.get("x")); state.y = Number(params.get("y")); }
  }

  function writeHash() {
    const params = new URLSearchParams();
    ["baseline", "baselineRepeat", "variant", "variantRepeat", "scene", "frame", "captureStage", "pyramidLevel", "alignment", "mapPreset", "mechanism", "component", "iteration", "scale", "sourceView", "channel", "annotationStage", "annotationKind"].forEach((key) => params.set(key, state[key]));
    params.set("signals", state.signals.join(","));
    if (state.x != null && state.y != null) { params.set("x", state.x.toFixed(6)); params.set("y", state.y.toFixed(6)); }
    history.replaceState(null, "", `#${params.toString()}`);
  }

  function captureCoverageContract() {
    const coverage = model.capture_profile_coverage;
    return coverage && typeof coverage === "object" && !Array.isArray(coverage) ? coverage : null;
  }

  function captureCoverageSupported(coverage) {
    return coverage?.schema_name === CAPTURE_PROFILE_SCHEMA && Number(coverage?.schema_version) === 1;
  }

  function captureProfileNames(coverage) {
    const discovered = [
      ...asArray(coverage?.requested_profiles),
      ...asArray(coverage?.profiles).map((profile) => profile?.profile),
      ...asArray(coverage?.units).map((unit) => unit?.capture_profile),
    ].filter(Boolean).map(String);
    return [...CAPTURE_PROFILE_ORDER, ...[...new Set(discovered)].filter((profile) => !CAPTURE_PROFILE_ORDER.includes(profile)).sort()];
  }

  function captureStatus(value, requested = true) {
    const status = String(value || "").toLowerCase().replaceAll("-", "_");
    if (["complete", "failed", "unavailable", "not_requested"].includes(status)) return status;
    return requested ? "not_recorded" : "not_requested";
  }

  function captureStatusLabel(status) {
    const labels = {
      complete: "Complete", failed: "Failed", unavailable: "Unavailable",
      not_requested: "Not requested", not_recorded: "Not recorded",
    };
    return labels[status] || String(status || "Not recorded").replaceAll("_", " ");
  }

  function captureStatusClass(status) {
    return {
      complete: "complete", failed: "failed", unavailable: "unavailable",
      not_requested: "not-requested", not_recorded: "not-recorded",
    }[status] || "not-recorded";
  }

  function captureProfileRecord(coverage, profile) {
    return asArray(coverage?.profiles).find((row) => row?.profile === profile) || null;
  }

  function captureProfileRequested(coverage, profile) {
    const record = captureProfileRecord(coverage, profile);
    if (record && typeof record.requested === "boolean") return record.requested;
    return asArray(coverage?.requested_profiles).includes(profile);
  }

  function safeEvidenceHref(path) {
    const value = String(path || "").trim();
    if (!value || value.startsWith("/") || value.startsWith("//") || /^[a-z][a-z0-9+.-]*:/i.test(value) || value.includes("\\")) return "";
    let decoded = value;
    try { decoded = decodeURIComponent(value); } catch (_error) { return ""; }
    if (decoded.split(/[/?#]/).includes("..")) return "";
    return value;
  }

  function evidenceReference(path, label, className = "") {
    const value = String(path || "").trim();
    const href = safeEvidenceHref(value);
    if (!href) return `<span class="${escapeHtml(className)} blocked" title="External evidence path (not served by this closed report): ${escapeHtml(value)}">${escapeHtml(label)} <small>(external evidence)</small></span>`;
    return `<a class="${escapeHtml(className)}" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="${escapeHtml(value)}">${escapeHtml(label)}</a>`;
  }

  function captureEvidenceLinks(links, scope) {
    const seen = new Set();
    const rendered = asArray(links).flatMap((link) => {
      if (!link || typeof link !== "object") return [];
      const path = String(link.path || "").trim();
      const label = String(link.label || path.split("/").filter(Boolean).pop() || "evidence");
      const key = `${label}\u0000${path}`;
      if (!path || seen.has(key)) return [];
      seen.add(key);
      const prefix = scope ? `<span class="capture-evidence-scope">${escapeHtml(scope)}</span>` : "";
      const href = safeEvidenceHref(path);
      if (!href) return [`<span class="capture-evidence-link blocked" title="External evidence path (not served by this closed report): ${escapeHtml(path)}">${prefix}${escapeHtml(label)} <small>(external evidence)</small></span>`];
      return [`<a class="capture-evidence-link" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="${escapeHtml(path)}">${prefix}${escapeHtml(label)}</a>`];
    });
    return rendered.join("");
  }

  function coverageIdentity(unit) {
    return `${String(unit?.configured_run || "")}\u0000${Number(unit?.repeat || 0)}\u0000${String(unit?.scene_id || "")}`;
  }

  function captureCoverageGroups(coverage) {
    const groups = new Map();
    asArray(coverage?.units).forEach((unit) => {
      if (!unit || typeof unit !== "object") return;
      const key = coverageIdentity(unit);
      if (!groups.has(key)) groups.set(key, {
        configured_run: String(unit.configured_run || "unavailable"),
        repeat: Number(unit.repeat || 0),
        scene_id: String(unit.scene_id || "unavailable"),
        profiles: new Map(),
      });
      const profile = String(unit.capture_profile || "unavailable");
      if (!groups.get(key).profiles.has(profile)) groups.get(key).profiles.set(profile, unit);
    });
    return [...groups.values()].sort((left, right) => (
      left.configured_run.localeCompare(right.configured_run) || left.repeat - right.repeat || left.scene_id.localeCompare(right.scene_id)
    ));
  }

  function captureProfileHeader(coverage, profile) {
    const record = captureProfileRecord(coverage, profile);
    const requested = captureProfileRequested(coverage, profile);
    const status = captureStatus(record?.status, requested);
    const expected = Number(record?.expected_units || 0);
    const complete = Number(record?.complete_units || 0);
    const failed = Number(record?.failed_units || 0);
    const unavailable = Number(record?.unavailable_units || 0);
    const process = String(record?.process_specialization || "not declared").replaceAll("_", " ");
    const authority = String(record?.quality_authority || "not declared").replaceAll("_", " ");
    const count = requested ? `${complete}/${expected}` : "off";
    const incomplete = [failed ? `${failed} failed` : "", unavailable ? `${unavailable} unavailable` : ""].filter(Boolean).join(" / ");
    return `<div class="capture-profile-column">
      <span><code>${escapeHtml(profile)}</code><span class="capture-count ${captureStatusClass(status)}">${escapeHtml(count)}</span></span>
      <small>${escapeHtml(process)} / ${escapeHtml(authority)}${incomplete ? ` / ${escapeHtml(incomplete)}` : ""}</small>
    </div>`;
  }

  function captureUnitCell(unit, requested) {
    const status = unit ? captureStatus(unit.status, requested) : captureStatus(null, requested);
    const frames = asArray(unit?.frames);
    const completeFrames = frames.filter((row) => captureStatus(row?.status, true) === "complete").length;
    const evidenceCount = asArray(unit?.evidence_links).length;
    const details = [];
    if (frames.length) details.push(`${completeFrames}/${frames.length} frames complete`);
    if (evidenceCount) details.push(`${evidenceCount} evidence link${evidenceCount === 1 ? "" : "s"}`);
    const reason = String(unit?.reason || (unit ? "" : requested ? "Requested profile has no unit record." : "Profile was not requested."));
    return `<td class="capture-unit ${captureStatusClass(status)}">
      <span class="capture-state ${captureStatusClass(status)}"${reason ? ` title="${escapeHtml(reason)}"` : ""}>${escapeHtml(captureStatusLabel(status))}</span>
      ${details.length ? `<small>${escapeHtml(details.join(" / "))}</small>` : ""}
      ${reason && status !== "complete" ? `<small class="capture-reason">${escapeHtml(reason)}</small>` : ""}
    </td>`;
  }

  function configuredRunCandidates(label, repeat) {
    const record = runRecord(label, repeat) || {};
    return new Set([
      record.configured_run, record.source_run, record.config_run, label,
      String(label || "").replace(/\s+\[[^\]]+\]\s*$/, ""),
    ].filter(Boolean).map(String));
  }

  function sameImageId(left, right) {
    const leftNumber = finiteNumber(left);
    const rightNumber = finiteNumber(right);
    return leftNumber != null && rightNumber != null ? leftNumber === rightNumber : String(left) === String(right);
  }

  function captureFrameCell(unit, requested, imageId) {
    if (!unit) return captureUnitCell(null, requested);
    const frameEvidence = asArray(unit.frames).find((row) => sameImageId(row?.image_id, imageId));
    const status = frameEvidence
      ? captureStatus(frameEvidence.status, requested)
      : captureStatus(unit.status, requested) === "complete" ? "not_recorded" : captureStatus(unit.status, requested);
    const frameLinks = captureEvidenceLinks(frameEvidence?.evidence_links, "frame");
    const unitLinks = captureEvidenceLinks(unit.evidence_links, "capture");
    const reason = frameEvidence
      ? String(unit.reason || "")
      : captureStatus(unit.status, requested) === "complete"
        ? "This capture has no evidence record for the selected frame. It may not have been selected by this profile."
        : String(unit.reason || "No frame evidence was recorded.");
    return `<td class="capture-unit capture-frame-unit ${captureStatusClass(status)}">
      <span class="capture-state ${captureStatusClass(status)}"${reason ? ` title="${escapeHtml(reason)}"` : ""}>${escapeHtml(captureStatusLabel(status))}</span>
      ${reason && status !== "complete" ? `<small class="capture-reason">${escapeHtml(reason)}</small>` : ""}
      ${frameLinks || unitLinks ? `<div class="capture-evidence-links">${frameLinks}${unitLinks}</div>` : '<small class="capture-no-links">No evidence links</small>'}
    </td>`;
  }

  function captureCoverageStatusPill() {
    const coverage = captureCoverageContract();
    if (!coverage) return '<span class="status-pill warning">Capture profile coverage not recorded (legacy model)</span>';
    if (!captureCoverageSupported(coverage)) return '<span class="status-pill error">Unsupported capture profile coverage contract</span>';
    const requested = captureProfileNames(coverage).filter((profile) => captureProfileRequested(coverage, profile));
    const records = requested.map((profile) => captureProfileRecord(coverage, profile));
    const complete = records.filter((record) => captureStatus(record?.status, true) === "complete").length;
    const failed = records.filter((record) => captureStatus(record?.status, true) === "failed").length;
    const unavailable = records.filter((record) => captureStatus(record?.status, true) === "unavailable").length;
    const missing = records.length - complete - failed - unavailable;
    const cssClass = failed || missing ? "error" : unavailable ? "warning" : "good";
    const suffix = [failed ? `${failed} failed` : "", unavailable ? `${unavailable} unavailable` : "", missing ? `${missing} not recorded` : ""].filter(Boolean).join(" / ");
    return `<span class="status-pill ${cssClass}">Capture profiles ${complete}/${requested.length} complete${suffix ? ` / ${suffix}` : ""}</span>`;
  }

  function renderCaptureProfileCoverage() {
    const section = byId("capture-profile-coverage-section");
    const summary = byId("capture-profile-coverage-summary");
    const content = byId("capture-profile-coverage-content");
    const coverage = captureCoverageContract();
    if (!coverage) {
      section.dataset.state = "legacy";
      summary.textContent = "Not recorded by this report model";
      content.innerHTML = '<div class="capture-coverage-notice"><strong>Capture profile coverage is unavailable.</strong><span>This report predates the profile-evidence contract. Do not infer capture completeness or provenance from map presence.</span></div>';
      return;
    }
    if (!captureCoverageSupported(coverage)) {
      section.dataset.state = "unsupported";
      summary.textContent = `${coverage.schema_name || "unknown schema"} v${coverage.schema_version ?? "unknown"}`;
      content.innerHTML = '<div class="capture-coverage-notice error"><strong>Unsupported capture profile coverage contract.</strong><span>Use a compatible report generator; this client will not guess at unknown profile semantics.</span></div>';
      return;
    }
    section.dataset.state = "available";
    const profiles = captureProfileNames(coverage);
    const groups = captureCoverageGroups(coverage);
    const requested = profiles.filter((profile) => captureProfileRequested(coverage, profile));
    const expectedUnits = asArray(coverage.profiles).filter((record) => record?.requested).reduce((total, record) => total + Number(record.expected_units || 0), 0);
    const completeUnits = asArray(coverage.profiles).filter((record) => record?.requested).reduce((total, record) => total + Number(record.complete_units || 0), 0);
    summary.textContent = `${requested.length}/${profiles.length} profiles requested / ${completeUnits}/${expectedUnits} expected units complete`;
    const matrixRows = groups.map((group) => `<tr>
      <td><strong>${escapeHtml(group.configured_run)}</strong><small>repeat ${group.repeat}</small></td>
      <td>${escapeHtml(group.scene_id)}</td>
      ${profiles.map((profile) => captureUnitCell(group.profiles.get(profile), captureProfileRequested(coverage, profile))).join("")}
    </tr>`).join("");
    const selectedFrame = frame();
    const selectedNames = [
      configuredRunCandidates(state.baseline, state.baselineRepeat),
      configuredRunCandidates(state.variant, state.variantRepeat),
    ];
    let frameGroups = groups.filter((group) => group.scene_id === state.scene && selectedNames.some((names) => names.has(group.configured_run)));
    if (!frameGroups.length) frameGroups = groups.filter((group) => group.scene_id === state.scene);
    const frameRows = selectedFrame ? frameGroups.map((group) => `<tr>
      <td><strong>${escapeHtml(group.configured_run)}</strong><small>repeat ${group.repeat}</small></td>
      ${profiles.map((profile) => captureFrameCell(group.profiles.get(profile), captureProfileRequested(coverage, profile), selectedFrame.image_id)).join("")}
    </tr>`).join("") : "";
    content.innerHTML = `
      <div class="table-wrap capture-profile-matrix-wrap"><table class="capture-profile-matrix">
        <thead><tr><th>Configured run / repeat</th><th>Scene</th>${profiles.map((profile) => `<th>${captureProfileHeader(coverage, profile)}</th>`).join("")}</tr></thead>
        <tbody>${matrixRows || `<tr><td colspan="${profiles.length + 2}">No capture units were recorded.</td></tr>`}</tbody>
      </table></div>
      <details class="capture-frame-evidence" open>
        <summary>Selected-frame evidence <span>${selectedFrame ? `${escapeHtml(state.scene)} / image ${escapeHtml(selectedFrame.image_id)}` : "no frame selected"}</span></summary>
        ${selectedFrame ? `<div class="table-wrap"><table class="capture-profile-frame-table">
          <thead><tr><th>Configured run / repeat</th>${profiles.map((profile) => `<th><code>${escapeHtml(profile)}</code></th>`).join("")}</tr></thead>
          <tbody>${frameRows || `<tr><td colspan="${profiles.length + 1}">No capture units match the selected scene and comparison.</td></tr>`}</tbody>
        </table></div>` : '<p class="empty">No frame is selected.</p>'}
      </details>`;
  }

  function renderStatus() {
    const summary = model.map_catalog_summary;
    const pixel = summary.pixel_data;
    const mechanics = model.mechanics?.availability || {};
    const filterStatus = mechanics.postprocess_filters
      ? ["good", "executed"]
      : mechanics.postprocess_filter_contract ? ["warning", "contract available (disabled)"] : ["error", "unavailable"];
    const confidenceStatus = mechanics.confidence_adjustment
      ? ["good", "executed"]
      : mechanics.confidence_adjustment_contract ? ["warning", "contract available (disabled)"] : ["error", "unavailable"];
    const failures = model.aggregates.gates.filter((row) => row.status === "fail").length;
    const capture = model.capture_validation || {};
    const endpointChecked = Number(capture.endpoint_dmap_sets_checked || 0);
    const endpointExact = Number(capture.endpoint_dmap_sets_bit_exact || 0);
    const endpointClass = endpointChecked && endpointChecked === endpointExact ? "good" : "error";
    const summaryChecked = Number(capture.maps_summary_frames_checked || 0);
    const specializationDivergence = Number(capture.diagnostic_process_specialization_divergence_frames || 0);
    const qualificationAlert = capture.production_qualification_status === "failed_allowed_diagnostic_only"
      ? `<span class="status-pill error">NOT PRODUCTION-PARITY QUALIFIED: Process&lt;true&gt; divergence on ${specializationDivergence} frame(s); deep maps are mechanics-only</span>`
      : "";
    const diagnosticSelections = selectedDiagnosticRuns();
    const diagnosticSelectionAlert = diagnosticSelections.length
      ? `<span class="status-pill warning">Selected diagnostic-only cohort: mechanics/maps available; quality rankings and annotation comparisons excluded</span>`
      : "";
    const unavailableClass = summary.unavailable ? "warning" : "good";
    byId("contract-alerts").innerHTML = [
      qualificationAlert,
      diagnosticSelectionAlert,
      `<span class="status-pill good">Schema ${escapeHtml(model.schema_version)} validated model</span>`,
      `<span class="status-pill ${unavailableClass}">${summary.available} maps available / ${summary.unavailable} unavailable</span>`,
      `<span class="status-pill ${failures ? "error" : "good"}">${failures} failed aggregate gates</span>`,
      `<span class="status-pill ${pixel.omitted_artifacts ? "warning" : "good"}">Exact pixel payloads ${pixel.selected_artifacts}/${pixel.eligible_artifacts} / ${((pixel.encoded_pixel_output_bytes || 0) / 1048576).toFixed(1)} MiB</span>`,
      `<span class="status-pill ${mechanics.exact_hot_kernel ? "good" : "warning"}">Exact hot-kernel tables ${mechanics.exact_hot_kernel ? "available" : "unavailable"}</span>`,
      `<span class="status-pill ${mechanics.cpu_view_ranking ? "good" : "warning"}">CPU view ranking ${mechanics.cpu_view_ranking ? "available" : "unavailable"}</span>`,
      `<span class="status-pill ${filterStatus[0]}">Sequential filters ${filterStatus[1]}</span>`,
      `<span class="status-pill ${confidenceStatus[0]}">Confidence adjustment ${confidenceStatus[1]}</span>`,
      `<span class="status-pill ${mechanics.resource_plans_valid ? "good" : "warning"}">Resource plans ${mechanics.resource_plans_valid ? "valid" : "incomplete"}</span>`,
      `<span class="status-pill ${endpointClass}">Production parity ${endpointExact}/${endpointChecked} DMAP sets, ${Number(capture.endpoint_dmaps_shared || 0)} files</span>`,
      `<span class="status-pill ${specializationDivergence ? "error" : summaryChecked ? "good" : "warning"}">Deep/summary parity ${summaryChecked} checked${specializationDivergence ? `, ${specializationDivergence} divergent` : ""}</span>`,
      captureCoverageStatusPill(),
      `<span class="status-pill">Logical iterations only; checkerboards limited to timing</span>`,
    ].filter(Boolean).join("");
  }

  function renderEvidenceContext() {
    const section = byId("evidence-context-section");
    const context = model.evidence_context;
    if (!context) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    const subject = context.subject;
    const mechanics = context.mechanics_authority;
    const quality = context.quality_authority;
    const statusClass = (value) => escapeHtml(String(value || "unavailable").replaceAll("_", "-"));
    const statusLabel = (value) => escapeHtml(String(value || "unavailable").replaceAll("_", " "));
    const badge = (value, label = null) => `<span class="evidence-status ${statusClass(value)}">${escapeHtml(label == null ? String(value || "unavailable").replaceAll("_", " ") : label)}</span>`;
    const scalar = (value, digits = 3) => {
      if (value == null) return "unavailable";
      if (typeof value === "boolean") return value ? "yes" : "no";
      const number = finiteNumber(value);
      if (number == null) return escapeHtml(value);
      if (typeof value === "number" && Number.isInteger(number)) return number.toLocaleString();
      return format(number, digits);
    };
    const signed = (value, unit, digits = 2) => {
      const number = finiteNumber(value);
      if (number == null) return "unavailable";
      return `${number > 0 ? "+" : ""}${number.toFixed(digits)} ${escapeHtml(unit)}`;
    };
    const mechanicsMetrics = mechanics.summary_metrics.map((metric) => `<div class="evidence-metric">
      <span>${escapeHtml(metric.label)} ${badge(metric.status)}</span>
      <strong>${scalar(metric.value)}${metric.value == null ? "" : ` ${escapeHtml(metric.unit)}`}</strong>
      <small>${escapeHtml(metric.description)}</small>
    </div>`).join("");
    const coverageRows = mechanics.candidate_coverage.map((row) => `<tr>
      <td>${escapeHtml(row.candidate)}</td>
      <td>${badge(row.coverage)}</td>
      <td>${escapeHtml(row.note || "-")}</td>
    </tr>`).join("");
    const qualityRows = [...quality.candidates]
      .sort((left, right) => left.accuracy_rank - right.accuracy_rank || left.candidate.localeCompare(right.candidate))
      .map((row) => {
        const strict = row.strict_accuracy_pass === true ? badge("pass")
          : row.strict_accuracy_pass === false ? badge("fail") : badge("unavailable");
        const availability = row.availability_biased
          ? `${badge("warning", "biased")} ${Number(row.lost_baseline_fit_count)} baseline fit(s) lost`
          : `${Number(row.paired_successful_structures)}/${Number(row.baseline_successful_structures)} paired`;
        return `<tr>
          <td>${Number(row.accuracy_rank)}</td>
          <td>${escapeHtml(row.candidate)}</td>
          <td>${strict}</td>
          <td>${badge(row.mechanics_coverage)}</td>
          <td>${Number(row.scene_count)}</td>
          <td>${availability}</td>
          <td>${scalar(row.median_normalized_noise_loss)}</td>
          <td>${scalar(row.worst_normalized_noise_loss)}</td>
          <td>${signed(row.residual_p95_delta_mm, "mm")}</td>
          <td>${signed(row.threshold_auc_delta_pp, "pp")}</td>
          <td>${signed(row.inlier_5mm_delta_pp, "pp")}</td>
          <td>${signed(row.effective_coverage_delta_pp, "pp")}</td>
          <td>${signed(row.spatial_coverage_delta_pp, "pp")}</td>
          <td>${signed(row.estimator_validity_delta_pp, "pp")}</td>
          <td>${signed(row.endpoint_validity_delta_pp, "pp")}</td>
          <td>${signed(row.runtime_delta_percent, "%")}</td>
        </tr>`;
      }).join("");
    const provenanceRows = [
      ...mechanics.source_artifacts.map((source) => ["mechanics", source]),
      ...quality.source_artifacts.map((source) => ["quality", source]),
    ].map(([authority, source]) => {
      const cardinality = Object.entries(source.cardinality || {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, value]) => `${escapeHtml(key)}=${Number(value)}`).join(", ") || "-";
      return `<tr>
        <td>${escapeHtml(authority)}</td><td>${escapeHtml(source.role)}</td>
        <td>${Number(source.bytes).toLocaleString()}</td><td>${cardinality}</td>
        <td><code>${escapeHtml(source.sha256)}</code></td>
        <td><code>${escapeHtml(source.content_digest || "-")}</code></td>
      </tr>`;
    }).join("");

    byId("evidence-context-heading").textContent = subject.title;
    byId("evidence-context-id").textContent = `${subject.experiment_id} / ${context.context_sha256}`;
    byId("evidence-context-summary").textContent = subject.summary;
    byId("evidence-context-content").innerHTML = `
      <div class="evidence-authority-grid">
        <section class="evidence-authority" aria-labelledby="mechanics-authority-heading">
          <header class="evidence-authority-header">
            <div><span class="evidence-authority-label">Diagnostic mechanics authority</span><h3 id="mechanics-authority-heading">${escapeHtml(mechanics.headline)}</h3></div>
            ${badge(mechanics.verdict)}
          </header>
          <p class="evidence-authority-meta"><code>${escapeHtml(mechanics.experiment_id)}</code> / <code>${escapeHtml(mechanics.process_specialization)}</code> / not quality eligible</p>
          <p class="evidence-authority-scope">${escapeHtml(mechanics.scope)}</p>
          <div class="evidence-metric-grid">${mechanicsMetrics}</div>
          <div class="table-wrap">
            <table class="evidence-coverage-table"><thead><tr><th>Candidate</th><th>Coverage</th><th>Note</th></tr></thead><tbody>${coverageRows}</tbody></table>
          </div>
        </section>
        <section class="evidence-authority" aria-labelledby="quality-authority-heading">
          <header class="evidence-authority-header">
            <div><span class="evidence-authority-label">External production quality authority</span><h3 id="quality-authority-heading">${escapeHtml(quality.headline)}</h3></div>
            ${badge("pass", quality.status.replaceAll("_", " "))}
          </header>
          <p class="evidence-authority-meta"><code>${escapeHtml(quality.experiment_id)}</code> / <code>${escapeHtml(quality.process_specialization)}</code> / quality eligible / manual promotion ${quality.manual_promotion_required ? "required" : "not required"}</p>
          <p class="evidence-authority-scope">${escapeHtml(quality.scope)}</p>
          <div class="table-wrap">
            <table class="evidence-quality-table"><thead><tr>
              <th>Rank</th><th>Candidate</th><th>Strict accuracy</th><th>Mechanics coverage</th><th>Scenes</th><th>Fit availability</th>
              <th>Median loss / tolerance</th><th>Worst loss / tolerance</th><th>Residual P95 delta</th><th>Threshold AUC delta</th>
              <th>5 mm inlier delta</th><th>Effective coverage</th><th>Spatial coverage</th><th>Estimator validity delta</th>
              <th>Terminal endpoint validity delta</th><th>Runtime delta</th>
            </tr></thead><tbody>${qualityRows}</tbody></table>
          </div>
        </section>
      </div>
      <div class="evidence-separation-note"><strong>Interpretation boundary:</strong> mechanics evidence explains behavior but does not establish quality. External quality rows are content-attested summaries and are not copied into this capture's annotations, comparisons, gates, regressions, or candidate ledger. Automatic promotion is disabled.</div>
      <details class="evidence-provenance-ui"><summary>Bound authority provenance</summary><div class="table-wrap">
        <table class="evidence-provenance-table"><thead><tr><th>Authority</th><th>Artifact role</th><th>Bytes</th><th>Cardinality</th><th>File SHA-256</th><th>Semantic digest</th></tr></thead><tbody>${provenanceRows}</tbody></table>
      </div></details>`;
  }

  function renderAccuracyLedger() {
    const rows = model.aggregates.accuracy_ledger || [];
    const section = byId("accuracy-ledger-section");
    if (!rows.length) {
      section.hidden = true;
      return;
    }
    section.hidden = false;
    byId("accuracy-ledger-table").querySelector("tbody").innerHTML = rows.map((row) => {
      const availability = row.availability_biased
        ? `${Number(row.lost_baseline_fit_count || 0)} baseline fits lost`
        : `${Number(row.paired_successful_structures || 0)} paired`;
      const coverageNote = row.coverage_advisory ? " advisory loss" : "";
      return `<tr>
        <td>${Number(row.accuracy_rank || 0)}</td>
        <td>${escapeHtml(row.candidate)}</td>
        <td><span class="status ${escapeHtml(row.noise_class)}">${escapeHtml(row.noise_class)}</span></td>
        <td>${escapeHtml(availability)}</td>
        <td>${Number(row.scene_count || 0)}</td>
        <td>${format(row.worst_normalized_noise_loss, 2)}</td>
        <td>${formatPercentagePoints(row.effective_coverage_delta)}${coverageNote}</td>
        <td>${formatPercentagePoints(row.spatial_coverage_delta)}</td>
        <td>${formatPercentagePoints(row.valid_coverage_delta)}</td>
        <td>${formatPercentagePoints(row.endpoint_valid_depth_coverage_delta)}</td>
        <td>${formatPercent(row.runtime_relative_delta)}</td>
        <td><button class="inspect-button accuracy-inspect" type="button" data-candidate="${escapeHtml(row.candidate)}">Inspect</button></td>
      </tr>`;
    }).join("");
    section.querySelectorAll("button.accuracy-inspect").forEach((button) => button.addEventListener("click", () => {
      state.variant = button.dataset.candidate;
      state.variantRepeat = repeatsForRun(state.variant)[0] || 0;
      renderAll();
      byId("maps-heading").scrollIntoView({ behavior: "smooth" });
    }));
  }

  function renderMechanismImpact() {
    const rows = model.aggregates.mechanism_impact || [];
    if (!rows.length) {
      byId("mechanism-impact").innerHTML = '<p class="empty">No registered mechanism evidence.</p>';
      return;
    }
    byId("mechanism-impact").innerHTML = rows.map((row) => {
      const total = Number(row.available_artifacts || 0) + Number(row.unavailable_artifacts || 0);
      const coverage = total ? Number(row.available_artifacts || 0) / total : 0;
      const status = coverage >= 0.95 ? "good" : coverage > 0 ? "warning" : "error";
      return `<button class="mechanism-card" type="button" data-mechanism="${escapeHtml(row.mechanism)}"><span class="mechanism-title">${escapeHtml(String(row.mechanism).replaceAll("_", " "))}</span><strong>${row.available_artifacts}/${total}</strong><span class="coverage-track"><i style="width:${Math.max(0, Math.min(100, coverage * 100)).toFixed(1)}%"></i></span><small>${row.signals.length} signals / ${row.frames} frames / ${row.exact_artifacts} exact / ${row.proxy_artifacts} proxy</small><span class="status-dot ${status}" aria-label="${status}"></span></button>`;
    }).join("");
    byId("mechanism-impact").querySelectorAll("button[data-mechanism]").forEach((button) => button.addEventListener("click", () => {
      state.mechanism = button.dataset.mechanism;
      state.component = "all";
      state.mapPreset = "custom";
      const matching = model.signals.filter((signal) => signalMatchesFilters(signal) && (signal.name === "reference_rgb" || signal.available_artifacts > 0));
      state.signals = ["reference_rgb", ...matching.map((signal) => signal.name).filter((name) => name !== "reference_rgb")];
      renderAll();
      byId("maps-heading").scrollIntoView({ behavior: "smooth" });
    }));
  }

  function shellQuote(value) {
    return `'${String(value).replaceAll("'", `'"'"'`)}'`;
  }

  function selectedPixel() {
    if (state.x == null || state.y == null) return null;
    const numeric = renderedTiles.find(({ artifact }) => artifact.pixel_data?.width && artifact.pixel_data?.height);
    const width = Number(numeric?.artifact.pixel_data.width || 0);
    const height = Number(numeric?.artifact.pixel_data.height || 0);
    if (!width || !height) return null;
    return {
      x: Math.min(width - 1, Math.max(0, Math.floor(state.x * width))),
      y: Math.min(height - 1, Math.max(0, Math.floor(state.y * height))),
    };
  }

  function drilldownCommand(pixel) {
    const currentFrame = frame();
    const configReference = model.experiment.source_config || "<CONFIG>";
    let configPath = configReference;
    try {
      if (location.protocol === "file:") configPath = decodeURIComponent(new URL(configReference, location.href).pathname);
    } catch (_error) { /* Keep the portable config hint. */ }
    const parts = [
      "tools/dmap_observability.sh", "drilldown",
      "--config", shellQuote(configPath),
      "--scene", shellQuote(state.scene),
      "--frame", String(currentFrame?.image_id ?? "<IMAGE_ID>"),
      "--variant", shellQuote(state.variant),
    ];
    if (pixel) parts.push("--pixel", `${pixel.x},${pixel.y}`);
    parts.push("--execute", "--refresh-report");
    if (location.protocol === "file:") {
      const reportPath = decodeURIComponent(location.pathname).replace(/\/02_investigation\.html$/, "");
      parts.push("--report-dir", shellQuote(reportPath));
    }
    return parts.join(" ");
  }

  function exportDrilldownRequest(profile) {
    const currentFrame = frame();
    if (!currentFrame) return;
    const pixel = profile === "trace" ? selectedPixel() : null;
    if (profile === "trace" && !pixel) return;
    const command = drilldownCommand(pixel);
    const lines = [
      "schema_name: openmvs.dmap.drilldown_suggestion",
      "schema_version: 1",
      `capture_profile: ${profile}`,
      `scene_id: ${JSON.stringify(state.scene)}`,
      `image_id: ${Number(currentFrame.image_id)}`,
      `pyramid_level: ${state.pyramidLevel === "unspecified" ? "null" : Number(state.pyramidLevel)}`,
      `baseline: ${JSON.stringify(state.baseline)}`,
      `variant: ${JSON.stringify(state.variant)}`,
    ];
    if (pixel) lines.push(`pixels:`, `  - {x: ${pixel.x}, y: ${pixel.y}}`);
    else lines.push("pixels: []");
    lines.push(`report_state: ${JSON.stringify(location.hash)}`, `command: ${JSON.stringify(command)}`, "");
    const blob = new Blob([lines.join("\n")], { type: "application/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `drilldown_${state.scene}_${currentFrame.image_id}_${profile}.yaml`.replace(/[^a-zA-Z0-9_.-]/g, "_");
    anchor.click();
    URL.revokeObjectURL(url);
    byId("ui-announcer").textContent = `${profile} drill-down request exported`;
  }

  function renderDrilldownStatus() {
    const currentFrame = frame();
    const rows = (model.drilldowns?.entries || []).filter((row) => (
      row.scene_id === state.scene && Number(row.image_id) === Number(currentFrame?.image_id)
    ));
    byId("drilldown-status").innerHTML = rows.map((row) => {
      const label = `${escapeHtml(row.capture_profile)} / ${escapeHtml(row.status)} / ${Number(row.trace_pixel_count || 0)} pixels / ${escapeHtml(String(row.request_sha256 || "invalid").slice(0, 12))}`;
      const target = row.executions || row.request;
      if (!target) return `<span class="drilldown-entry ${escapeHtml(row.status)}">${label}</span>`;
      const title = row.executions ? "Open trace execution manifest" : "Open immutable drill-down request";
      const href = safeEvidenceHref(target);
      if (!href) return `<span class="drilldown-entry ${escapeHtml(row.status)} blocked" title="External evidence path (not served by this closed report): ${escapeHtml(target)}">${label} / external evidence</span>`;
      return `<a class="drilldown-entry ${escapeHtml(row.status)}" href="${escapeHtml(href)}" target="_blank" rel="noopener" title="${title}">${label} / open</a>`;
    }).join("");
    byId("drilldown-detail").innerHTML = rows.map((row, index) => renderCompletedTrace(row, rows.length === 1 || index === 0)).join("");
    byId("drilldown-detail").querySelectorAll(".trace-pixel-jump").forEach((button) => button.addEventListener("click", () => {
      const numeric = currentFrame?.maps?.find((artifact) => artifact.pixel_data?.width && artifact.pixel_data?.height);
      const width = Number(numeric?.pixel_data?.width || 0);
      const height = Number(numeric?.pixel_data?.height || 0);
      if (!width || !height) return;
      state.x = (Number(button.dataset.x) + 0.5) / width;
      state.y = (Number(button.dataset.y) + 0.5) / height;
      updateCrosshairs(); renderPixelTable(); renderDrilldownStatus(); writeHash();
      byId("pixel-heading").scrollIntoView({ behavior: "smooth", block: "start" });
    }));
    byId("trace-pixel-request").disabled = selectedPixel() == null;
    installColumnTooltips();
  }

  function traceTransition(before, after, unit = "") {
    const left = finiteNumber(before), right = finiteNumber(after);
    if (left == null && right == null) return "n/a";
    return `${left == null ? "n/a" : format(left, 6)} → ${right == null ? "n/a" : format(right, 6)}${unit}`;
  }

  function traceMask(value) {
    const number = finiteNumber(value);
    return number == null ? "n/a" : `0x${(number >>> 0).toString(16)}`;
  }

  function tracePayload(row) {
    const arrays = row.arrays || {};
    const costs = arrays.view_costs || [];
    const photo = arrays.view_photometric_costs || [];
    const geometric = arrays.view_geometric_costs || [];
    const weights = arrays.view_weights || [];
    const views = costs.map((cost, index) => ({ cost, index, photo: photo[index], geometric: geometric[index], weight: weights[index] }))
      .filter((item) => finiteNumber(item.cost) !== 0 || finiteNumber(item.weight) !== 0)
      .map((item) => `v${item.index}: total=${format(item.cost, 6)}, photo=${format(item.photo, 6)}, geometric=${format(item.geometric, 6)}, weight=${format(item.weight, 4)}`);
    const neighbors = (arrays.neighbor_costs || []).map((value, index) => `n${index}=${format(value, 6)}`);
    const badReasons = (arrays.bad_reasons || []).map((value, index) => ({ value: Number(value || 0), index }))
      .filter((item) => item.value > 0).map((item) => `reason${item.index}=${item.value}`);
    const lines = [
      views.length ? `Views: ${views.join("; ")}` : "Views: no nonzero contribution recorded",
      neighbors.length ? `Neighbors: ${neighbors.join(", ")}` : "Neighbors: unavailable",
      `Bad candidates: ${badReasons.join(", ") || "none"}`,
      `Reference variance: ${format(row.cost?.reference_variance, 6)} / view entropy: ${format(row.view?.entropy, 6)}`,
    ];
    return `<details class="trace-payload"><summary>View and neighborhood payload</summary><pre>${escapeHtml(lines.join("\n"))}</pre></details>`;
  }

  function renderCompletedTrace(entry, open) {
    const trace = entry.trace_data;
    if (entry.capture_profile !== "trace" || entry.status !== "complete") return "";
    if (!trace?.available) {
      return `<div class="trace-unavailable"><strong>Completed trace data unavailable.</strong> ${escapeHtml(trace?.unavailable_reason || "No normalized trace rows were indexed.")}</div>`;
    }
    const runOrder = new Map([[state.baseline, 0], [state.variant, 1]]);
    const traceStageKey = (row) => row.estimation_stage === "geometric_consistency"
      ? `geometric_consistency:${row.geometric_iteration}` : "photometric";
    const rows = trace.rows.filter((row) => (
      traceStageKey(row) === state.captureStage && mechanicsPyramidLevelMatches(row)
    )).sort((left, right) => (
      Number(left.y) - Number(right.y) || Number(left.x) - Number(right.x) ||
      (runOrder.get(left.run) ?? 2) - (runOrder.get(right.run) ?? 2) ||
      Number(left.logical_iteration) - Number(right.logical_iteration)
    ));
    const body = rows.map((row) => {
      const iteration = Number(row.logical_iteration) < 0 ? "Initialization" : `Iteration ${Number(row.display_iteration)}`;
      const views = `${row.view?.selected_count ?? "n/a"} / ${traceMask(row.view?.selected_before_mask)} → ${traceMask(row.view?.selected_mask)}`;
      const sourceQuality = row.source_quality || "proxy";
      const measurementBasis = row.measurement_basis || "legacy_unclassified_trace";
      const identity = row.request_identity || {};
      const aliases = (identity.alias_coordinates || []).map((coordinate, index) => {
        const requestIndex = (identity.alias_request_indices || [])[index];
        return `#${requestIndex ?? "?"} (${Number(coordinate.x)}, ${Number(coordinate.y)})`;
      });
      const requestLabel = aliases.length
        ? `${aliases.join(", ")} → slot ${Number(row.trace_index)} (${Number(row.x)}, ${Number(row.y)})`
        : `slot ${Number(row.trace_index)} (${Number(row.x)}, ${Number(row.y)})`;
      return `<tr>
        <td>${escapeHtml(row.run)}</td>
        <td><button class="trace-pixel-jump" type="button" data-x="${Number(row.x)}" data-y="${Number(row.y)}" title="Inspect scaled trace coordinate in synchronized maps">${escapeHtml(requestLabel)}</button></td>
        <td>${escapeHtml(iteration)}</td>
        <td>${escapeHtml(row.source)} <span class="quality ${escapeHtml(sourceQuality)}" title="${escapeHtml(measurementBasis)}">${escapeHtml(sourceQuality)}</span><br><small>${escapeHtml(measurementBasis)}</small></td>
        <td>${escapeHtml(traceTransition(row.cost?.before, row.cost?.after))}<br><small>improvement ${escapeHtml(format(row.cost?.improvement, 6))}</small></td>
        <td>${escapeHtml(traceTransition(row.depth?.before, row.depth?.after, " m"))}<br><small>|change| ${escapeHtml(format(row.depth?.absolute_change, 6))} m</small></td>
        <td>${escapeHtml(format(row.normal?.angle_change_degrees, 4))}°</td>
        <td>${escapeHtml(views)}</td>
        <td>${tracePayload(row)}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="9">No trace rows were retained for ${escapeHtml(pyramidLevelLabel(String(state.pyramidLevel)))}.</td></tr>`;
    const selectedSources = (trace.sources || []).filter((source) => (
      source.available && source.source_path && traceStageKey(source) === state.captureStage
    ));
    const sourceLinks = selectedSources.map((source) => (
      evidenceReference(source.source_path, `${source.run} ${source.stage_key} traces.jsonl`, "trace-source-link")
    )).join(" / ");
    const executions = entry.execution_metadata?.executions || [];
    const durations = executions.map((execution) => {
      const seconds = execution.elapsed_seconds ?? execution.duration_seconds ?? execution.wall_time_seconds;
      return `${execution.run}: ${seconds == null ? "duration unavailable" : `${format(seconds, 2)} s`} / ${execution.validation || "validation unavailable"}`;
    }).join("; ");
    const qualityCounts = rows.reduce((counts, row) => {
      const quality = row.source_quality === "exact" ? "exact" : "proxy";
      counts[quality] += 1;
      return counts;
    }, { exact: 0, proxy: 0 });
    return `<details class="completed-trace"${open ? " open" : ""}>
      <summary><span>Targeted pixel trace</span><strong>${rows.length}/${trace.row_count} rows at ${escapeHtml(state.captureStage)} / ${escapeHtml(pyramidLevelLabel(String(state.pyramidLevel)))} / ${selectedSources.length} sources</strong></summary>
      <div class="completed-trace-body">
        <p>New trace requests use a full-frame Process&lt;true&gt; maps rerun with <code>write_maps=1</code>; public v1 has no compact exact-trace path. Source attribution is classified per row: exact requires hot-kernel provenance backed by valid schema-v4 exact-map completion evidence, while imported legacy or post-pass rows remain proxy. This table contains ${Number(qualityCounts.exact || 0)} exact and ${Number(qualityCounts.proxy || 0)} proxy source rows.</p>
        <p class="trace-executions">${escapeHtml(durations)}</p>
        <div class="trace-links">${sourceLinks}${entry.executions ? ` / ${evidenceReference(entry.executions, "execution manifest", "trace-execution-link")}` : ""}</div>
        <div class="table-wrap"><table class="trace-table"><thead><tr><th>Run</th><th>Pixel</th><th>State</th><th>Update source</th><th>Cost</th><th>Depth</th><th>Normal change</th><th>Views / masks</th><th>Payload</th></tr></thead><tbody>${body}</tbody></table></div>
      </div>
    </details>`;
  }

  function renderSelectors() {
    const labels = [...new Set(model.runs.map((run) => run.label))];
    controls.baseline.innerHTML = labels.map((label) => option(label, runSelectorLabel(label), label === state.baseline)).join("");
    controls.variant.innerHTML = labels.map((label) => option(label, runSelectorLabel(label), label === state.variant)).join("");
    const baselineRuns = runsFor(state.baseline); if (!baselineRuns.some((run) => run.repeat === state.baselineRepeat)) state.baselineRepeat = baselineRuns[0]?.repeat || 0;
    const variantRuns = runsFor(state.variant); if (!variantRuns.some((run) => run.repeat === state.variantRepeat)) state.variantRepeat = variantRuns[0]?.repeat || 0;
    controls.baselineRepeat.innerHTML = baselineRuns.map((run) => option(run.repeat, `repeat ${run.repeat}`, run.repeat === state.baselineRepeat)).join("");
    controls.variantRepeat.innerHTML = variantRuns.map((run) => option(run.repeat, `repeat ${run.repeat}`, run.repeat === state.variantRepeat)).join("");
    if (!model.scenes.some((item) => item.id === state.scene)) state.scene = model.scenes[0]?.id || "";
    controls.scene.innerHTML = model.scenes.map((item) => option(item.id, `${item.label} (${item.frame_count})`, item.id === state.scene)).join("");
    const sceneFrames = scene()?.frames || []; if (!sceneFrames.some((item) => item.id === state.frame)) state.frame = sceneFrames[0]?.id || "";
    controls.frame.innerHTML = sceneFrames.map((item) => option(item.id, `${item.image_id}: ${item.label}`, item.id === state.frame)).join("");
    const annotationRows = frame()?.annotations || [];
    const annotationStages = [...new Set(annotationRows.map((row) => row.stage).filter(Boolean))].sort((left, right) => {
      const priority = { post_filter: 0, pre_filter: 1 };
      return (priority[left] ?? 10) - (priority[right] ?? 10) || String(left).localeCompare(String(right));
    });
    if (!annotationStages.includes(state.annotationStage)) state.annotationStage = annotationStages[0] || "post_filter";
    controls.annotationStage.innerHTML = annotationStages.map((value) => option(
      value, value === "post_filter" ? "Post-filter" : value === "pre_filter" ? "Pre-filter" : String(value).replaceAll("_", " "),
      value === state.annotationStage,
    )).join("");
    controls.annotationStage.disabled = annotationStages.length < 2;
    const annotationKinds = [...new Set(annotationRows.map((row) => row.annotation_kind).filter(Boolean))].sort();
    if (state.annotationKind !== "all" && !annotationKinds.includes(state.annotationKind)) state.annotationKind = "all";
    controls.annotationKind.innerHTML = [option("all", "All geometry", state.annotationKind === "all"), ...annotationKinds.map((value) => option(
      value, value === "plane" ? "Planes" : value === "edge" ? "Lines / edges" : value,
      value === state.annotationKind,
    ))].join("");
    controls.annotationKind.disabled = annotationKinds.length < 2;
    const frameRows = frame()?.run_frames || [];
    const algorithmRows = [
      ...(frame()?.maps || []),
      ...frameRows.flatMap((row) => row.iterations || []),
    ];
    const stageRows = [...new Map(algorithmRows.map((map) => {
      const key = map.estimation_stage === "geometric_consistency" ? `geometric_consistency:${map.geometric_iteration}` : "photometric";
      return [key, { key, label: key === "photometric" ? "Photometric" : `Geometric ${map.geometric_iteration}` }];
    })).values()];
    if (!stageRows.some((item) => item.key === state.captureStage)) state.captureStage = stageRows[0]?.key || "photometric";
    controls.captureStage.innerHTML = stageRows.map((item) => option(item.key, item.label, item.key === state.captureStage)).join("");
    const stageMaps = (frame()?.maps || []).filter((map) => (map.estimation_stage === "geometric_consistency" ? `geometric_consistency:${map.geometric_iteration}` : "photometric") === state.captureStage);
    const stageIterations = frameRows
      .flatMap((row) => row.iterations || [])
      .filter((row) => (row.estimation_stage === "geometric_consistency" ? `geometric_consistency:${row.geometric_iteration}` : "photometric") === state.captureStage);
    const pyramidLevels = [...new Set([...stageMaps, ...stageIterations].map(pyramidLevelOf))].sort((left, right) => {
      if (left === "unspecified") return 1;
      if (right === "unspecified") return -1;
      return Number(left) - Number(right);
    });
    if (!pyramidLevels.includes(String(state.pyramidLevel))) {
      state.pyramidLevel = pyramidLevels.includes("0") ? "0" : (pyramidLevels[0] || "unspecified");
    }
    controls.pyramidLevel.innerHTML = (pyramidLevels.length ? pyramidLevels : ["unspecified"]).map((value) => option(
      value, pyramidLevelLabel(value), String(value) === String(state.pyramidLevel),
    )).join("");
    controls.pyramidLevel.disabled = pyramidLevels.length < 2;
    const levelMaps = stageMaps.filter(pyramidLevelMatches);
    const levelRows = [...levelMaps, ...stageIterations.filter(pyramidLevelMatches)];
    const iterations = [...new Set(levelRows.map((map) => map.logical_iteration).filter((value) => value != null))].sort((a, b) => a - b); if (!iterations.includes(state.iteration)) state.iteration = iterations[0] ?? -1;
    controls.alignment.innerHTML = [
      ["final", "Final state per run"], ["same", "Same logical iteration"],
    ].map(([value, label]) => option(value, label, value === state.alignment)).join("");
    controls.iteration.innerHTML = iterations.map((value) => option(value, value < 0 ? "Initialization" : `Iteration ${value + 1}`, value === state.iteration)).join("");
    controls.iteration.disabled = state.alignment === "final";
    controls.mapPreset.innerHTML = [
      ["overview", "Overview"], ["cost", "Cost and candidate mechanics"],
      ["view", "View selection"], ["filtering", "Sequential filtering"], ["custom", "Custom"],
    ].map(([value, label]) => option(value, label, value === state.mapPreset)).join("");
    const mechanisms = [...new Set(model.signals.map((signal) => signal.mechanism).filter(Boolean))].sort();
    if (state.mechanism !== "all" && !mechanisms.includes(state.mechanism)) state.mechanism = "all";
    controls.mechanism.innerHTML = [
      option("all", "All mechanisms", state.mechanism === "all"),
      ...mechanisms.map((value) => option(value, value.replaceAll("_", " "), value === state.mechanism)),
    ].join("");
    const componentSignals = model.signals.filter((signal) => (
      state.mechanism === "all" || signal.mechanism === state.mechanism
    ));
    const components = [...new Map(componentSignals.filter((signal) => signal.component_id).map((signal) => [
      String(signal.component_id), signal,
    ])).values()].sort((left, right) => String(left.component_id).localeCompare(String(right.component_id)));
    if (state.component !== "all" && !components.some((signal) => String(signal.component_id) === state.component)) state.component = "all";
    controls.component.innerHTML = [
      option("all", "All components", state.component === "all"),
      ...components.map((signal) => option(
        signal.component_id,
        `${String(signal.component_id).replaceAll("_", " ")} / ${String(signal.mechanism).replaceAll("_", " ")}`,
        String(signal.component_id) === state.component,
      )),
    ].join("");
    const sourceViews = [...new Map(levelMaps.filter((map) => map.source_view_index != null).map((map) => [Number(map.source_view_index), map])).values()].sort((a, b) => Number(a.source_view_index) - Number(b.source_view_index));
    const sourceValues = sourceViews.map((map) => String(map.source_view_index));
    if (state.sourceView !== "auto" && !sourceValues.includes(String(state.sourceView))) state.sourceView = "auto";
    controls.sourceView.innerHTML = option("auto", "Auto / non-view", state.sourceView === "auto") + sourceViews.map((map) => option(map.source_view_index, `view ${map.source_view_index} / image ${map.source_image_id ?? "n/a"}`, String(map.source_view_index) === String(state.sourceView))).join("");
    const channelRows = levelMaps.flatMap((map) => map.preview?.channels || []);
    const channels = [...new Map(channelRows.map((channel) => [Number(channel.index), channel])).values()].sort((a, b) => a.index - b.index);
    if (channels.length && !channels.some((item) => Number(item.index) === Number(state.channel))) state.channel = Number(channels[0].index);
    controls.channel.innerHTML = channels.length ? channels.map((item) => option(item.index, `${item.index}: ${item.label}`, Number(item.index) === Number(state.channel))).join("") : option(0, "scalar / composite", true);
    document.querySelector(`input[name=scale][value="${state.scale}"]`).checked = true;
    const alignmentLabel = state.alignment === "final" ? "final state per run" : (state.iteration < 0 ? "initialization" : `iteration ${state.iteration + 1}`);
    byId("frame-context").textContent = frame() ? `${scene().label} / image ${frame().image_id} / ${state.captureStage} / ${pyramidLevelLabel(String(state.pyramidLevel))} / ${alignmentLabel}` : "No frame selected";
  }

  function signalMatchesFilters(signal) {
    if (!signal) return false;
    if (signal.name === "reference_rgb") return true;
    if (state.mechanism !== "all" && signal.mechanism !== state.mechanism) return false;
    if (state.component !== "all" && String(signal.component_id || "") !== state.component) return false;
    return true;
  }

  function renderSignalPicker() {
    byId("signal-picker").innerHTML = model.signals.filter(signalMatchesFilters).map((signal) => {
      const checked = state.signals.includes(signal.name);
      const quality = signal.measurement_qualities.join(", ") || "unavailable";
      const component = signal.component_id ? ` / ${signal.component_id}` : "";
      return `<label><input type="checkbox" value="${escapeHtml(signal.name)}" ${checked ? "checked" : ""}><span>${escapeHtml(signal.label)}<small>${escapeHtml(component)}</small></span><span class="quality-mini">${escapeHtml(quality)}</span></label>`;
    }).join("");
    byId("signal-picker").querySelectorAll("input").forEach((input) => input.addEventListener("change", () => {
      state.signals = [...byId("signal-picker").querySelectorAll("input:checked")].map((item) => item.value);
      state.mapPreset = "custom";
      controls.mapPreset.value = "custom";
      renderMaps(); writeHash();
    }));
  }

  function applyMapPreset() {
    if (state.mapPreset === "custom") return;
    const available = model.signals.filter((signal) => (
      (signal.name === "reference_rgb" || signal.available_artifacts > 0) && signalMatchesFilters(signal)
    ));
    if (state.mapPreset === "overview") {
      state.signals = available.filter((signal) => signal.default).slice(0, 8).map((signal) => signal.name);
      return;
    }
    const mechanisms = state.mapPreset === "cost" ? new Set(["cost", "candidate_update"]) :
      state.mapPreset === "view" ? new Set(["view_selection"]) : new Set(["filtering"]);
    const selected = available.filter((signal) => mechanisms.has(signal.mechanism));
    if (state.mapPreset === "filtering") {
      selected.sort((left, right) => {
        const rank = (name) => name.includes("transition") ? 0 : (name.includes("delta") ? 1 : (name.includes("before") || name.includes("input") ? 2 : 3));
        return rank(left.name) - rank(right.name) || left.name.localeCompare(right.name);
      });
    }
    state.signals = ["reference_rgb", ...selected.map((signal) => signal.name).filter((name) => name !== "reference_rgb")];
  }

  function pickAlignedArtifact(preferred, alignment, iteration) {
    if (alignment === "final") {
      const stateMaps = preferred.filter((map) => map.logical_iteration != null);
      const finalIteration = stateMaps.length ? Math.max(...stateMaps.map((map) => Number(map.logical_iteration))) : null;
      const finalMaps = finalIteration == null ? [] : stateMaps.filter((map) => Number(map.logical_iteration) === finalIteration);
      return finalMaps.find((map) => map.available)
        || preferred.find((map) => map.logical_iteration == null && map.available)
        || finalMaps[0]
        || preferred.find((map) => map.logical_iteration == null)
        || null;
    }
    const logical = preferred.filter((map) => map.logical_iteration != null && Number(map.logical_iteration) === Number(iteration));
    return logical.find((map) => map.available) || logical[0] || null;
  }

  function selectedArtifact(run, repeat, signal) {
    if (signal === "reference_rgb") return null;
    const candidates = (frame()?.maps || []).filter((map) => map.run === run && Number(map.repeat) === Number(repeat) && map.signal === signal);
    const stageCandidates = candidates.filter((map) => (map.estimation_stage === "geometric_consistency" ? `geometric_consistency:${map.geometric_iteration}` : "photometric") === state.captureStage);
    const levelCandidates = stageCandidates.filter(pyramidLevelMatches);
    const hasPerViewArtifacts = levelCandidates.some((map) => map.source_view_index != null);
    const preferred = state.sourceView === "auto" || !hasPerViewArtifacts
      ? levelCandidates
      : levelCandidates.filter((map) => Number(map.source_view_index) === Number(state.sourceView));
    return pickAlignedArtifact(preferred, state.alignment, state.iteration);
  }

  function referenceTile(runLabel) {
    const reference = frame()?.reference || { available: false, unavailable_reason: "reference image unavailable" };
    return {
      id: `reference-${runLabel}`, signal: "reference_rgb", available: reference.available,
      preview: { local: reference.path, shared: reference.path }, measurement_quality: "source",
      stage: "input", measurement_basis: "reference image", unavailable_reason: reference.unavailable_reason,
      pixel_data: { available: false, reason: "RGB display values only" },
    };
  }

  function coarseCostUnavailability(runLabel, repeat, signal) {
    const signalMetadata = model.signals.find((item) => item.name === signal) || {};
    const costLike = signalMetadata.mechanism === "cost"
      || /cost|confidence|gap/i.test(String(signal));
    if (!costLike || state.pyramidLevel === "unspecified" || Number(state.pyramidLevel) <= 0) return null;
    const current = frame();
    const contract = (model.mechanics?.coarse_compatibility_map_availability || []).find((row) => {
      const key = row.estimation_stage === "geometric_consistency"
        ? `geometric_consistency:${row.geometric_iteration}` : "photometric";
      return row.run === runLabel && Number(row.repeat) === Number(repeat)
        && row.scene_id === state.scene && Number(row.image_id) === Number(current?.image_id)
        && key === state.captureStage && Number(row.pyramid_level) === Number(state.pyramidLevel)
        && row.cost_map_expected === false;
    });
    if (!contract) return null;
    return {
      available: false,
      unavailable_reason: contract.cost_map_unavailable_reason,
      measurement_quality: "unavailable",
      measurement_basis: contract.measurement_basis,
      pyramid_level: Number(contract.pyramid_level),
      estimation_stage: contract.estimation_stage,
      geometric_iteration: contract.geometric_iteration,
    };
  }

  function artifactTile(artifact, runLabel, repeat, signal) {
    if (!artifact) artifact = coarseCostUnavailability(runLabel, repeat, signal);
    if (!artifact) artifact = {
      available: false,
      unavailable_reason: "signal not declared for this run/frame/stage/pyramid level",
      measurement_quality: "unavailable",
      pyramid_level: state.pyramidLevel === "unspecified" ? null : Number(state.pyramidLevel),
    };
    const quality = artifact.measurement_quality || (artifact.available ? "source" : "unavailable");
    const stage = artifact.logical_iteration == null ? (artifact.algorithm_stage || artifact.stage || artifact.role || "final") : (artifact.logical_iteration < 0 ? "initialization" : `iteration ${Number(artifact.logical_iteration) + 1}`);
    const pyramidLabel = signal === "reference_rgb"
      ? "reference input"
      : artifact.pyramid_level == null && artifact.scale_level == null && artifact.scale_number == null
        ? "pyramid level unspecified"
        : `pyramid ${pyramidLevelLabel(pyramidLevelOf(artifact)).toLowerCase()}`;
    const channelPreview = (artifact.preview?.channels || []).find((item) => Number(item.index) === Number(state.channel));
    const preview = channelPreview?.[state.scale] || artifact.preview?.[state.scale];
    const scale = channelPreview?.[`${state.scale}_scale`] || artifact.preview?.[`${state.scale}_scale`];
    const source = artifact.source_path ? evidenceReference(artifact.source_path, "source", "map-source-link") : "no source link";
    const limitation = artifact.limitations ? ` / ${escapeHtml(artifact.limitations)}` : "";
    const semantics = artifact.semantics ? ` / ${escapeHtml(artifact.semantics)}` : "";
    const categoryLegend = Object.entries(artifact.category_legend || {})
      .map(([code, label]) => `${code} = ${label}`).join(" / ");
    if (!artifact.available || !preview) {
      return `<article class="map-tile" data-run="${escapeHtml(runLabel)}"><div class="map-stage"><span>${escapeHtml(stage)} / ${escapeHtml(pyramidLabel)}</span><span class="quality unavailable">unavailable</span></div><div class="unavailable-tile">${escapeHtml(artifact.unavailable_reason || "preview unavailable")}</div><p class="provenance">${escapeHtml(artifact.measurement_basis || "No signal provenance")}${semantics}${limitation}</p></article>`;
    }
    const scaleLabel = artifact.preview?.scale_mode === "categorical_registered_codes"
      ? "registered categories"
      : scale ? `${format(scale.low, 4)} to ${format(scale.high, 4)}` : state.scale;
    renderedTiles.push({ artifact, runLabel, signal });
    const viewLabel = artifact.source_view_index == null ? "" : ` / view ${artifact.source_view_index} (image ${artifact.source_image_id ?? "n/a"})`;
    const channelLabel = channelPreview ? ` / ${channelPreview.label}` : "";
    return `<article class="map-tile" id="${escapeHtml(artifact.deep_link_id || artifact.id)}" data-artifact="${escapeHtml(artifact.id)}"><div class="map-stage"><span>${escapeHtml(stage)} / ${escapeHtml(pyramidLabel)}${escapeHtml(viewLabel)}${escapeHtml(channelLabel)} / ${escapeHtml(scaleLabel)}</span><span class="quality ${escapeHtml(quality)}">${escapeHtml(quality)}</span></div><div class="map-image-wrap"><img class="map-image" src="${escapeHtml(preview)}" alt="${escapeHtml(signal)} for ${escapeHtml(runLabel)}"><i class="crosshair-x"></i><i class="crosshair-y"></i></div>${categoryLegend ? `<p class="category-legend">${escapeHtml(categoryLegend)}</p>` : ""}<p class="provenance">${escapeHtml(artifact.measurement_basis || "unspecified basis")}${semantics}${limitation} / ${source}</p></article>`;
  }

  function deltaTile(baseline, variant, signal) {
    const id = `delta-${String(baseline?.id || "missing")}-${String(variant?.id || "missing")}`.replace(/[^a-zA-Z0-9_-]/g, "-");
    if (signal === "reference_rgb") {
      return '<article class="map-tile"><div class="map-stage"><span>variant - baseline</span><span class="quality unavailable">not defined</span></div><div class="unavailable-tile">RGB subtraction is intentionally not presented as algorithm evidence.</div><p class="provenance">Use the shared reference image for spatial context.</p></article>';
    }
    if (baseline?.measurement_kind === "enum" || variant?.measurement_kind === "enum") {
      return '<article class="map-tile"><div class="map-stage"><span>variant - baseline</span><span class="quality unavailable">not defined</span></div><div class="unavailable-tile">Arithmetic subtraction is not meaningful for categorical status codes.</div><p class="provenance">Compare the registered category labels in the baseline and variant tiles.</p></article>';
    }
    const baselinePixels = baseline?.pixel_data;
    const variantPixels = variant?.pixel_data;
    if (!baseline?.available || !variant?.available || !baselinePixels?.available || !variantPixels?.available) {
      const reason = baselinePixels?.reason || variantPixels?.reason || baseline?.unavailable_reason || variant?.unavailable_reason || "paired numeric payload unavailable";
      return `<article class="map-tile"><div class="map-stage"><span>variant - baseline</span><span class="quality unavailable">unavailable</span></div><div class="unavailable-tile">${escapeHtml(reason)}</div><p class="provenance">The report keeps this missing paired delta explicit.</p></article>`;
    }
    if (baselinePixels.width !== variantPixels.width || baselinePixels.height !== variantPixels.height || baselinePixels.channels !== variantPixels.channels) {
      return '<article class="map-tile"><div class="map-stage"><span>variant - baseline</span><span class="quality unavailable">domain mismatch</span></div><div class="unavailable-tile">Baseline and variant numeric domains differ.</div><p class="provenance">A delta is not computed across mismatched shapes.</p></article>';
    }
    const quality = [baseline.measurement_quality, variant.measurement_quality].every((value) => value === "exact" || value === "derived_exact") ? "derived_exact" : "proxy";
    const job = { id, baseline, variant, signal, quality };
    deltaJobs.push(job);
    return `<article class="map-tile" id="${escapeHtml(id)}"><div class="map-stage"><span>variant - baseline / <span class="delta-range">loading scale</span></span><span class="quality ${escapeHtml(quality)}">${escapeHtml(quality)}</span></div><div class="map-image-wrap"><canvas class="map-image delta-canvas" id="canvas-${escapeHtml(id)}" aria-label="${escapeHtml(signal)} variant minus baseline"></canvas><i class="crosshair-x"></i><i class="crosshair-y"></i></div><p class="provenance">Browser-computed float32 difference from paired numeric payloads.</p></article>`;
  }

  function divergentColor(value, magnitude) {
    const normalized = Math.max(-1, Math.min(1, value / Math.max(magnitude, 1e-12)));
    if (normalized < 0) {
      const amount = normalized + 1;
      return [Math.round(45 + 210 * amount), Math.round(93 + 162 * amount), 210, 255];
    }
    return [220, Math.round(255 - 195 * normalized), Math.round(255 - 200 * normalized), 255];
  }

  async function renderDeltaCanvas(job) {
    const canvas = byId(`canvas-${job.id}`);
    if (!canvas) return;
    try {
      const [baselineValues, variantValues] = await Promise.all([
        decodePayload(job.baseline.pixel_data), decodePayload(job.variant.pixel_data),
      ]);
      const metadata = job.baseline.pixel_data;
      const channel = Math.min(Number(state.channel) || 0, Number(metadata.channels) - 1);
      const count = Number(metadata.width) * Number(metadata.height);
      const sample = [];
      const stride = Math.max(1, Math.floor(count / 65536));
      for (let pixel = 0; pixel < count; pixel += stride) {
        const index = pixel * metadata.channels + channel;
        const delta = variantValues[index] - baselineValues[index];
        if (Number.isFinite(delta)) sample.push(Math.abs(delta));
      }
      sample.sort((left, right) => left - right);
      const magnitude = sample.length ? Math.max(sample[Math.min(sample.length - 1, Math.floor(sample.length * 0.99))], 1e-12) : 1;
      const maxDimension = 480;
      const scale = Math.min(1, maxDimension / Math.max(metadata.width, metadata.height));
      const width = Math.max(1, Math.round(metadata.width * scale));
      const height = Math.max(1, Math.round(metadata.height * scale));
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      const image = context.createImageData(width, height);
      for (let y = 0; y < height; y += 1) {
        const sourceY = Math.min(metadata.height - 1, Math.floor(y / scale));
        for (let x = 0; x < width; x += 1) {
          const sourceX = Math.min(metadata.width - 1, Math.floor(x / scale));
          const source = (sourceY * metadata.width + sourceX) * metadata.channels + channel;
          const destination = (y * width + x) * 4;
          const delta = variantValues[source] - baselineValues[source];
          const color = Number.isFinite(delta) ? divergentColor(delta, magnitude) : [0, 0, 0, 0];
          image.data.set(color, destination);
        }
      }
      context.putImageData(image, 0, 0);
      const label = document.querySelector(`#${CSS.escape(job.id)} .delta-range`);
      if (label) label.textContent = `${format(-magnitude, 4)} to ${format(magnitude, 4)}`;
    } catch (error) {
      const tile = byId(job.id);
      if (tile) tile.querySelector(".map-image-wrap").innerHTML = `<div class="unavailable-tile">${escapeHtml(error.message)}</div>`;
    }
  }

  function renderMaps() {
    renderedTiles = [];
    deltaJobs = [];
    const columns = [
      { label: state.baseline, repeat: state.baselineRepeat, role: "Baseline" },
      { label: state.variant, repeat: state.variantRepeat, role: "Variant" },
    ];
    let html = `<div class="grid-header">Signal</div>${columns.map((column) => `<div class="grid-header">${escapeHtml(column.role)}: ${escapeHtml(column.label)} / repeat ${column.repeat}</div>`).join("")}<div class="grid-header">Delta: variant - baseline</div>`;
    state.signals.filter((signal) => signalMatchesFilters(model.signals.find((item) => item.name === signal))).forEach((signal) => {
      const metadata = model.signals.find((item) => item.name === signal) || { label: signal, measurement_qualities: [] };
      html += `<div class="signal-label" title="${escapeHtml(metadata.description || "")}"><strong>${escapeHtml(metadata.label)}</strong><small>${escapeHtml(metadata.mechanism || "state")} / ${escapeHtml(metadata.quantity || "value")} / ${escapeHtml(metadata.measurement_qualities.join(" / "))}</small></div>`;
      const baseline = signal === "reference_rgb" ? referenceTile(columns[0].label)
        : selectedArtifact(columns[0].label, columns[0].repeat, signal)
          || coarseCostUnavailability(columns[0].label, columns[0].repeat, signal);
      const variant = signal === "reference_rgb" ? referenceTile(columns[1].label)
        : selectedArtifact(columns[1].label, columns[1].repeat, signal)
          || coarseCostUnavailability(columns[1].label, columns[1].repeat, signal);
      html += artifactTile(baseline, columns[0].label, columns[0].repeat, signal);
      html += artifactTile(variant, columns[1].label, columns[1].repeat, signal);
      html += deltaTile(baseline, variant, signal);
    });
    byId("map-grid").innerHTML = html;
    byId("map-grid").querySelectorAll(".map-image").forEach((image) => image.addEventListener("click", imageClicked));
    deltaJobs.forEach((job) => renderDeltaCanvas(job));
    updateCrosshairs();
    renderPixelTable();
  }

  function imageClicked(event) {
    const rectangle = event.currentTarget.getBoundingClientRect();
    state.x = Math.min(1, Math.max(0, (event.clientX - rectangle.left) / rectangle.width));
    state.y = Math.min(1, Math.max(0, (event.clientY - rectangle.top) / rectangle.height));
    updateCrosshairs(); renderPixelTable(); renderDrilldownStatus(); writeHash();
  }

  function updateCrosshairs() {
    document.querySelectorAll(".map-image-wrap").forEach((wrap) => {
      wrap.classList.toggle("has-crosshair", state.x != null && state.y != null);
      if (state.x == null || state.y == null) return;
      wrap.querySelector(".crosshair-x").style.left = `${state.x * 100}%`;
      wrap.querySelector(".crosshair-y").style.top = `${state.y * 100}%`;
    });
  }

  function loadPixelScript(pixelData) {
    window.__DMAP_PIXEL_PAYLOADS = window.__DMAP_PIXEL_PAYLOADS || {};
    if (window.__DMAP_PIXEL_PAYLOADS[pixelData.payload_id]) return Promise.resolve(window.__DMAP_PIXEL_PAYLOADS[pixelData.payload_id]);
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = pixelData.script_path;
      script.onload = () => {
        script.remove();
        const payload = window.__DMAP_PIXEL_PAYLOADS[pixelData.payload_id];
        if (payload) resolve(payload); else reject(new Error("numeric payload script did not register its data"));
      };
      script.onerror = () => reject(new Error("numeric payload script could not be loaded"));
      document.head.appendChild(script);
    });
  }

  function decodePayload(pixelData) {
    if (decodedPayloads.has(pixelData.payload_id)) return decodedPayloads.get(pixelData.payload_id);
    const promise = loadPixelScript(pixelData).then(async (payload) => {
      if (typeof DecompressionStream === "undefined") throw new Error("this browser lacks gzip DecompressionStream support");
      const binary = atob(payload.data);
      const compressed = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) compressed[index] = binary.charCodeAt(index);
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
      const buffer = await new Response(stream).arrayBuffer();
      return new Float32Array(buffer);
    });
    decodedPayloads.set(pixelData.payload_id, promise);
    return promise;
  }

  async function decodePixel(pixelData, normalizedX, normalizedY) {
    if (!pixelData?.available) throw new Error(pixelData?.reason || "exact numeric payload unavailable");
    const values = await decodePayload(pixelData);
    const x = Math.min(pixelData.width - 1, Math.max(0, Math.floor(normalizedX * pixelData.width)));
    const y = Math.min(pixelData.height - 1, Math.max(0, Math.floor(normalizedY * pixelData.height)));
    const selected = [];
    const offset = (y * pixelData.width + x) * pixelData.channels;
    for (let channel = 0; channel < pixelData.channels; channel += 1) {
      selected.push(values[offset + channel]);
    }
    return { x, y, values: selected };
  }

  function candidateSlotLabel(value) {
    const slot = Math.round(Number(value));
    if (slot === 255) return "unavailable";
    if (slot === 0) return "current";
    if (slot >= 1 && slot <= 8) return `propagation_${slot - 1}`;
    return ({ 9: "refine_depth", 10: "refine_normal", 11: "refine_random_normal", 12: "refine_surface_normal" })[slot] || "unknown";
  }

  function updateSourceLabel(value) {
    const source = Math.round(Number(value));
    return ({ 0: "none", 1: "init", 2: "propagate", 3: "refine_depth", 4: "refine_normal", 5: "refine_random_normal", 6: "refine_surface_normal", 7: "filtered", 8: "changed_unknown" })[source] || "unknown";
  }

  function formatPixelValues(artifact, values) {
    if (artifact.signal === "candidate_identity_exact" && values.length >= 3) {
      const winner = Math.round(values[0]), runner = Math.round(values[1]), source = Math.round(values[2]);
      return `winner_slot=${winner} (${candidateSlotLabel(winner)}), runner_up_slot=${runner} (${candidateSlotLabel(runner)}), update_source=${source} (${updateSourceLabel(source)})`;
    }
    if (artifact.signal === "candidate_counts_exact" && values.length >= 3) {
      return `tested_count=${Math.round(values[0])}, finite_count=${Math.round(values[1])}, accepted_count=${Math.round(values[2])}`;
    }
    if (["selected_views_before_mask_exact", "selected_views_after_mask_exact"].includes(artifact.signal) && values.length >= 4) {
      const mask = (Math.round(values[0]) | (Math.round(values[1]) << 8) | (Math.round(values[2]) << 16) | (Math.round(values[3]) << 24)) >>> 0;
      return `mask=0x${mask.toString(16)} (${mask.toString(2).padStart(32, "0")})`;
    }
    const categories = artifact.category_legend || {};
    if (artifact.measurement_kind === "enum" && values.length === 1) {
      const code = Math.round(Number(values[0]));
      return `code=${code} (${categories[String(code)] || "unregistered category"})`;
    }
    const labels = artifact.pixel_data?.channel_labels || [];
    return values.map((value, index) => labels[index] ? `${labels[index]}=${format(value, 7)}` : format(value, 7)).join(", ");
  }

  function renderPixelTable() {
    const body = byId("pixel-table").querySelector("tbody");
    if (state.x == null || state.y == null) { byId("pixel-coordinate").textContent = "Select a point in any map."; body.innerHTML = ""; return; }
    byId("pixel-coordinate").textContent = `normalized x=${state.x.toFixed(4)}, y=${state.y.toFixed(4)}`;
    body.innerHTML = [
      ...renderedTiles.map(({ artifact, runLabel, signal }) => `<tr id="pixel-${escapeHtml(artifact.id)}"><td>${escapeHtml(runLabel)}</td><td>${escapeHtml(signal)}</td><td>${artifact.logical_iteration == null ? escapeHtml(artifact.stage || "final") : (artifact.logical_iteration < 0 ? "initialization" : `iteration ${Number(artifact.logical_iteration) + 1}`)}</td><td><span class="quality ${escapeHtml(artifact.measurement_quality || "source")}">${escapeHtml(artifact.measurement_quality || "source")}</span></td><td class="pixel-value">loading</td><td>${escapeHtml(artifact.measurement_basis || "unspecified")}</td></tr>`),
      ...deltaJobs.map((job) => `<tr id="pixel-${escapeHtml(job.id)}"><td>Delta</td><td>${escapeHtml(job.signal)}</td><td>variant - baseline</td><td><span class="quality ${escapeHtml(job.quality)}">${escapeHtml(job.quality)}</span></td><td class="pixel-value">loading</td><td>paired float32 numeric payloads</td></tr>`),
    ].join("");
    renderedTiles.forEach(async ({ artifact }) => {
      const cell = document.querySelector(`#pixel-${CSS.escape(artifact.id)} .pixel-value`); if (!cell) return;
      try {
        const decoded = await decodePixel(artifact.pixel_data, state.x, state.y);
        cell.textContent = `(${decoded.x}, ${decoded.y}) = ${formatPixelValues(artifact, decoded.values)}`;
      } catch (error) { cell.textContent = `unavailable: ${error.message}`; }
    });
    deltaJobs.forEach(async (job) => {
      const cell = document.querySelector(`#pixel-${CSS.escape(job.id)} .pixel-value`); if (!cell) return;
      try {
        const [baseline, variant] = await Promise.all([
          decodePixel(job.baseline.pixel_data, state.x, state.y),
          decodePixel(job.variant.pixel_data, state.x, state.y),
        ]);
        const values = variant.values.map((value, index) => value - baseline.values[index]);
        cell.textContent = `(${variant.x}, ${variant.y}) = ${values.map((value) => format(value, 7)).join(", ")}`;
      } catch (error) { cell.textContent = `unavailable: ${error.message}`; }
    });
  }

  function renderRegressions() {
    const query = state.regressionFilter.toLowerCase();
    const direction = state.sortDirection;
    const rows = model.aggregates.regressions.filter((row) => row.candidate === state.variant && [row.metric, row.scene_id, row.status, row.level].join(" ").toLowerCase().includes(query));
    rows.sort((left, right) => {
      const a = left[state.sort], b = right[state.sort];
      if (typeof a === "number" && typeof b === "number") return direction * (a - b);
      return direction * String(a).localeCompare(String(b));
    });
    byId("regression-table").querySelector("tbody").innerHTML = rows.map((row) => {
      const sceneLevel = row.navigation_level === "scene" || row.image_id == null;
      return `<tr><td><span class="status ${escapeHtml(row.status)}">${escapeHtml(row.status)}</span></td><td>${escapeHtml(row.metric)}<br><small>${escapeHtml(row.level)}</small></td><td>${format(row.regression_score)}</td><td>${format(row.delta)}</td><td>${escapeHtml(row.scene_id)}</td><td>${sceneLevel ? "all" : escapeHtml(row.image_id)}</td><td><button class="inspect-button" data-scene="${escapeHtml(row.scene_id)}" ${sceneLevel ? 'data-level="scene"' : `data-image="${escapeHtml(row.image_id)}"`}>${sceneLevel ? "Inspect scene" : "Inspect"}</button></td></tr>`;
    }).join("");
    const empty = byId("regression-empty");
    empty.textContent = selectedDiagnosticRuns().length
      ? "Diagnostic-only cohorts are excluded from automatic quality regression ranking; use synchronized maps and mechanics below."
      : "No matching regression evidence.";
    empty.hidden = rows.length > 0;
    document.querySelectorAll(".inspect-button").forEach((button) => button.addEventListener("click", () => {
      if (button.dataset.level === "scene") navigateToScene(button.dataset.scene);
      else navigateToFrame(button.dataset.scene, Number(button.dataset.image));
    }));
  }

  function navigateToScene(sceneId) {
    const targetScene = model.scenes.find((item) => item.id === sceneId);
    const targetFrame = targetScene?.frames?.[0];
    if (!targetScene || !targetFrame) return;
    navigateToFrame(targetScene.id, targetFrame.image_id);
  }

  function navigateToFrame(sceneId, imageId) {
    const targetScene = model.scenes.find((item) => item.id === sceneId); const targetFrame = targetScene?.frames.find((item) => Number(item.image_id) === Number(imageId));
    if (!targetScene || !targetFrame) return;
    state.scene = targetScene.id; state.frame = targetFrame.id; state.x = null; state.y = null;
    renderSelectors(); renderCaptureProfileCoverage(); renderMechanics(); renderMaps(); renderAnnotations(); installColumnTooltips(); writeHash(); byId("maps-heading").scrollIntoView({ behavior: "smooth" });
  }

  function annotationFitAvailable(row) {
    if (!row) return false;
    const status = String(row.fit_status || "ok").toLowerCase();
    return status === "ok" || status === "available";
  }

  function annotationKindLabel(kind) {
    if (kind === "plane") return "Plane";
    if (kind === "edge") return "Line / edge";
    return kind || "Geometry";
  }

  function annotationStageLabel(stage) {
    if (stage === "post_filter") return "Post-filter";
    if (stage === "pre_filter") return "Pre-filter";
    return String(stage || "unspecified").replaceAll("_", " ");
  }

  function annotationIdentity(row, index, side) {
    const kind = String(row.annotation_kind || "annotation");
    const objectId = String(row.object_id || "");
    const chunkId = String(row.chunk_id || "");
    if (objectId || chunkId) return `${kind}\u0000${objectId}\u0000${chunkId}`;
    return `${kind}\u0000unidentified-${side}-${index}`;
  }

  function annotationMean(rows, accessor) {
    const values = rows.filter(annotationFitAvailable).map(accessor).map(finiteNumber).filter((value) => value != null);
    return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null;
  }

  function annotationDeltaClass(baseline, variant, direction) {
    const left = finiteNumber(baseline), right = finiteNumber(variant);
    if (left == null || right == null) return "unavailable";
    const delta = right - left;
    if (Math.abs(delta) <= 1e-12) return "stable";
    return (direction === "lower" ? delta < 0 : delta > 0) ? "improved" : "regressed";
  }

  function annotationMetricTile(name, baseline, variant, formatter, deltaFormatter, direction, directionLabel) {
    const left = finiteNumber(baseline), right = finiteNumber(variant);
    const delta = left == null || right == null ? null : right - left;
    const status = annotationDeltaClass(left, right, direction);
    return `<article class="annotation-metric">
      <span class="annotation-metric-name">${escapeHtml(name)}</span>
      <div class="annotation-metric-values"><strong>${escapeHtml(formatter(left))}</strong><span class="annotation-arrow">&rarr;</span><strong>${escapeHtml(formatter(right))}</strong></div>
      <span class="annotation-delta ${status}">${escapeHtml(deltaFormatter(delta))}</span>
      <span class="annotation-direction">${escapeHtml(directionLabel)}</span>
    </article>`;
  }

  function annotationCompareCell(label, baseline, variant, formatter, deltaFormatter, direction) {
    const left = finiteNumber(baseline), right = finiteNumber(variant);
    const delta = left == null || right == null ? null : right - left;
    const status = annotationDeltaClass(left, right, direction);
    return `<span class="annotation-compare-cell">
      <span class="annotation-cell-label">${escapeHtml(label)}</span>
      <strong>${escapeHtml(formatter(left))} &rarr; ${escapeHtml(formatter(right))}</strong>
      <span class="annotation-delta ${status}">${escapeHtml(deltaFormatter(delta))}</span>
    </span>`;
  }

  function annotationAsset(row, key, title) {
    const path = row?.[key];
    if (!path) return `<article class="annotation-asset"><h5>${escapeHtml(title)}</h5><div class="annotation-asset-empty">Evidence unavailable</div></article>`;
    return `<article class="annotation-asset"><h5>${escapeHtml(title)}</h5><img data-src="${escapeHtml(path)}" alt="${escapeHtml(title)}" loading="lazy"></article>`;
  }

  function annotationThresholdRows(baseline, variant) {
    return [5, 10, 20, 50].map((threshold) => {
      const rawKey = `inlier_fraction_${threshold}mm`;
      const effectiveKey = `effective_inlier_coverage_${threshold}mm`;
      const baselineRaw = finiteNumber(baseline?.[rawKey]);
      const variantRaw = finiteNumber(variant?.[rawKey]);
      const baselineEffective = finiteNumber(baseline?.[effectiveKey]);
      const variantEffective = finiteNumber(variant?.[effectiveKey]);
      const baselineBar = Math.max(0, Math.min(100, (baselineEffective ?? baselineRaw ?? 0) * 100));
      const variantBar = Math.max(0, Math.min(100, (variantEffective ?? variantRaw ?? 0) * 100));
      const deltaBase = baselineEffective ?? baselineRaw;
      const deltaVariant = variantEffective ?? variantRaw;
      const delta = deltaBase == null || deltaVariant == null ? null : deltaVariant - deltaBase;
      return `<div class="annotation-threshold-row">
        <strong>${threshold} mm</strong>
        <div class="annotation-threshold-value"><span>B: ${formatPercent(baselineRaw)} inliers / ${formatPercent(baselineEffective)} effective</span><div class="annotation-bar-track"><span class="annotation-bar-fill" style="width:${baselineBar.toFixed(2)}%"></span></div></div>
        <div class="annotation-threshold-value variant"><span>V: ${formatPercent(variantRaw)} inliers / ${formatPercent(variantEffective)} effective</span><div class="annotation-bar-track"><span class="annotation-bar-fill" style="width:${variantBar.toFixed(2)}%"></span></div></div>
        <span class="annotation-delta ${annotationDeltaClass(deltaBase, deltaVariant, "higher")}">${escapeHtml(formatPercentagePoints(delta))}</span>
      </div>`;
    }).join("");
  }

  function renderAnnotations() {
    const annotations = frame()?.annotations || [];
    const content = byId("annotation-content");
    const context = byId("annotation-context");
    const stage = state.annotationStage;
    const kind = state.annotationKind;
    context.textContent = `${state.baseline} r${state.baselineRepeat} vs ${state.variant} r${state.variantRepeat} / ${annotationStageLabel(stage)}`;
    if (selectedDiagnosticRuns().length) {
      content.innerHTML = `<p class="empty">Annotation quality comparisons exclude diagnostic-only Process&lt;true&gt; cohorts. Select production-quality runs for end metrics, or continue with the synchronized mechanics and maps.</p>`;
      return;
    }
    if (!annotations.length) { content.innerHTML = `<p class="empty">No annotation evidence is available for this frame.</p>`; return; }

    const selectedRows = (run, repeat) => annotations.filter((row) => {
      const rowRepeat = finiteNumber(row.repeat);
      return row.run === run && (rowRepeat == null || rowRepeat === Number(repeat)) &&
        (!row.stage || row.stage === stage) && (kind === "all" || row.annotation_kind === kind);
    });
    const baselineRows = selectedRows(state.baseline, state.baselineRepeat);
    const variantRows = selectedRows(state.variant, state.variantRepeat);
    const referenceRows = annotations.filter((row) => row.role === "reference" &&
      (!row.stage || row.stage === stage) && (kind === "all" || row.annotation_kind === kind));
    if (!baselineRows.length && !variantRows.length) {
      content.innerHTML = `<p class="empty">No ${escapeHtml(annotationStageLabel(stage).toLowerCase())} annotation evidence is available for the selected runs and repeats.</p>`;
      return;
    }

    const indexRows = (rows, side) => new Map(rows.map((row, index) => [annotationIdentity(row, index, side), row]));
    const baselineByIdentity = indexRows(baselineRows, "baseline");
    const variantByIdentity = indexRows(variantRows, "variant");
    const identities = [...new Set([...baselineByIdentity.keys(), ...variantByIdentity.keys()])];
    const pairs = identities.map((identity) => ({
      identity,
      baseline: baselineByIdentity.get(identity),
      variant: variantByIdentity.get(identity),
    })).sort((left, right) => {
      const leftKind = String(left.baseline?.annotation_kind || left.variant?.annotation_kind || "");
      const rightKind = String(right.baseline?.annotation_kind || right.variant?.annotation_kind || "");
      if (leftKind !== rightKind) return leftKind.localeCompare(rightKind);
      const leftBaseline = finiteNumber(left.baseline?.effective_inlier_coverage_20mm ?? left.baseline?.effective_inlier_coverage);
      const leftVariant = finiteNumber(left.variant?.effective_inlier_coverage_20mm ?? left.variant?.effective_inlier_coverage);
      const rightBaseline = finiteNumber(right.baseline?.effective_inlier_coverage_20mm ?? right.baseline?.effective_inlier_coverage);
      const rightVariant = finiteNumber(right.variant?.effective_inlier_coverage_20mm ?? right.variant?.effective_inlier_coverage);
      const leftScore = leftBaseline == null || leftVariant == null ? Number.POSITIVE_INFINITY : leftVariant - leftBaseline;
      const rightScore = rightBaseline == null || rightVariant == null ? Number.POSITIVE_INFINITY : rightVariant - rightBaseline;
      return leftScore - rightScore || left.identity.localeCompare(right.identity);
    });

    const baselineSuccessful = baselineRows.filter(annotationFitAvailable);
    const variantSuccessful = variantRows.filter(annotationFitAvailable);
    const coverageBaseline = annotationMean(baselineRows, (row) => row.coverage_fraction);
    const coverageVariant = annotationMean(variantRows, (row) => row.coverage_fraction);
    const effectiveBaseline = annotationMean(baselineRows, (row) => row.effective_inlier_coverage_20mm ?? row.effective_inlier_coverage);
    const effectiveVariant = annotationMean(variantRows, (row) => row.effective_inlier_coverage_20mm ?? row.effective_inlier_coverage);
    const aucBaseline = annotationMean(baselineRows, (row) => row.inlier_threshold_auc);
    const aucVariant = annotationMean(variantRows, (row) => row.inlier_threshold_auc);
    const residualBaseline = annotationMean(baselineRows, (row) => row.all_residual_p95_m);
    const residualVariant = annotationMean(variantRows, (row) => row.all_residual_p95_m);
    const referenceLabel = [...new Set(referenceRows.map((row) => row.run))].join(", ");
    const referenceCoverage = annotationMean(referenceRows, (row) => row.coverage_fraction);
    const referenceEffective = annotationMean(referenceRows, (row) => row.effective_inlier_coverage_20mm ?? row.effective_inlier_coverage);
    const referenceAuc = annotationMean(referenceRows, (row) => row.inlier_threshold_auc);
    const referenceResidual = annotationMean(referenceRows, (row) => row.all_residual_p95_m);
    const productReferenceHtml = referenceRows.length ? `
      <div class="annotation-product-reference">
        <h3>Archived product reference: ${escapeHtml(referenceLabel)}</h3>
        <p>Full-resolution final geometry only. The arrows below compare the selected fresh baseline to the archived product; this is a porting anchor, not a mechanics ablation or regression gate.</p>
        <div class="annotation-summary-grid">
          ${annotationMetricTile("Valid depth coverage", coverageBaseline, referenceCoverage, formatPercent, formatPercentagePoints, "higher", "Fresh baseline to archived product")}
          ${annotationMetricTile("Effective inliers @20 mm", effectiveBaseline, referenceEffective, formatPercent, formatPercentagePoints, "higher", "Fresh baseline to archived product")}
          ${annotationMetricTile("Threshold AUC", aucBaseline, referenceAuc, formatPercent, formatPercentagePoints, "higher", "Fresh baseline to archived product")}
          ${annotationMetricTile("Residual P95", residualBaseline, referenceResidual, formatMillimetres, formatMillimetreDelta, "lower", "Fresh baseline to archived product")}
        </div>
      </div>` : "";

    const problems = [];
    pairs.forEach((pair) => {
      const row = pair.baseline || pair.variant;
      const structure = `${annotationKindLabel(row?.annotation_kind)} ${String(row?.chunk_id || "unidentified").slice(0, 8)}`;
      if (!pair.baseline) problems.push(`${structure}: baseline evidence unavailable`);
      if (!pair.variant) problems.push(`${structure}: variant evidence unavailable`);
    });
    [...baselineRows, ...variantRows].filter((row) => !annotationFitAvailable(row)).forEach((row) => {
      problems.push(`${row.run} / ${annotationKindLabel(row.annotation_kind)} ${String(row.chunk_id || "unidentified").slice(0, 8)}: ${row.fit_status || "unavailable"}${row.error ? ` - ${row.error}` : ""}`);
    });
    const uniqueProblems = [...new Set(problems)];
    const geometryKinds = [...new Set([...baselineRows, ...variantRows].map((row) => annotationKindLabel(row.annotation_kind)))];
    const fitDelta = variantSuccessful.length - baselineSuccessful.length;
    const fitStatus = annotationDeltaClass(baselineSuccessful.length, variantSuccessful.length, "higher");
    const structureRows = pairs.map((pair) => {
      const row = pair.baseline || pair.variant || {};
      const chunkId = String(row.chunk_id || "unidentified");
      const objectId = String(row.object_id || "unavailable");
      const baselineStatus = pair.baseline ? String(pair.baseline.fit_status || "ok") : "missing";
      const variantStatus = pair.variant ? String(pair.variant.fit_status || "ok") : "missing";
      const baselineEffective = finiteNumber(pair.baseline?.effective_inlier_coverage_20mm ?? pair.baseline?.effective_inlier_coverage);
      const variantEffective = finiteNumber(pair.variant?.effective_inlier_coverage_20mm ?? pair.variant?.effective_inlier_coverage);
      const baselineAuc = finiteNumber(pair.baseline?.inlier_threshold_auc);
      const variantAuc = finiteNumber(pair.variant?.inlier_threshold_auc);
      const baselineP95 = finiteNumber(pair.baseline?.all_residual_p95_m);
      const variantP95 = finiteNumber(pair.variant?.all_residual_p95_m);
      return `<details class="annotation-structure" data-annotation-structure="${escapeHtml(chunkId)}">
        <summary class="annotation-structure-summary">
          <span class="annotation-identity"><strong><span class="annotation-kind">${escapeHtml(annotationKindLabel(row.annotation_kind))}</span>${escapeHtml(chunkId.slice(0, 8))}</strong><small>object ${escapeHtml(objectId.slice(0, 8))}</small></span>
          <span class="annotation-fit"><span class="status ${annotationFitAvailable(pair.baseline) ? "pass" : "fail"}">B ${escapeHtml(baselineStatus)}</span><span class="status ${annotationFitAvailable(pair.variant) ? "pass" : "fail"}">V ${escapeHtml(variantStatus)}</span><small>RANSAC fit</small></span>
          ${annotationCompareCell("Effective @20 mm", baselineEffective, variantEffective, formatPercent, formatPercentagePoints, "higher")}
          ${annotationCompareCell("Threshold AUC", baselineAuc, variantAuc, formatPercent, formatPercentagePoints, "higher")}
          ${annotationCompareCell("Residual P95", baselineP95, variantP95, formatMillimetres, formatMillimetreDelta, "lower")}
          <span class="annotation-expand">Inspect</span>
        </summary>
        <div class="annotation-evidence-panel">
          <h4>Fixed-model threshold profile</h4>
          <p class="annotation-threshold-note">One model is fitted at 20 mm and evaluated at 5/10/20/50 mm without refitting. Bars show effective inlier coverage; text also reports the valid-point inlier rate.</p>
          <div class="annotation-threshold-grid">${annotationThresholdRows(pair.baseline, pair.variant)}</div>
          <h4>Coverage and residual evidence</h4>
          <div class="annotation-assets">
            ${annotationAsset(pair.baseline, "visual_overlay_svg", `${state.baseline} r${state.baselineRepeat} coverage overlay`)}
            ${annotationAsset(pair.variant, "visual_overlay_svg", `${state.variant} r${state.variantRepeat} coverage overlay`)}
            ${annotationAsset(pair.baseline, "visual_residual_histogram_svg", `${state.baseline} r${state.baselineRepeat} residual histogram`)}
            ${annotationAsset(pair.variant, "visual_residual_histogram_svg", `${state.variant} r${state.variantRepeat} residual histogram`)}
          </div>
          <p class="annotation-provenance">Stage: ${escapeHtml(annotationStageLabel(stage))} / structure: ${escapeHtml(chunkId)} / object: ${escapeHtml(objectId)}. Green/red deltas indicate directional change only, not statistical significance.</p>
        </div>
      </details>`;
    }).join("");

    content.innerHTML = `
      <div class="annotation-scope">
        <span>${escapeHtml(geometryKinds.join(", ") || "Geometry")}</span>
        <span>${pairs.length} paired or reported structures</span>
        <span>B ${baselineSuccessful.length}/${baselineRows.length} successful fits</span>
        <span>V ${variantSuccessful.length}/${variantRows.length} successful fits</span>
        <span class="${uniqueProblems.length ? "warning" : "available"}">${uniqueProblems.length ? `${uniqueProblems.length} availability issues` : "All selected evidence available"}</span>
      </div>
      <div class="annotation-summary-grid">
        <article class="annotation-metric"><span class="annotation-metric-name">Successful geometry fits</span><div class="annotation-metric-values"><strong>${baselineSuccessful.length}/${baselineRows.length}</strong><span class="annotation-arrow">&rarr;</span><strong>${variantSuccessful.length}/${variantRows.length}</strong></div><span class="annotation-delta ${fitStatus}">${fitDelta > 0 ? "+" : ""}${fitDelta}</span><span class="annotation-direction">More successful fits are better</span></article>
        ${annotationMetricTile("Valid depth coverage", coverageBaseline, coverageVariant, formatPercent, formatPercentagePoints, "higher", "Higher is better")}
        ${annotationMetricTile("Effective inliers @20 mm", effectiveBaseline, effectiveVariant, formatPercent, formatPercentagePoints, "higher", "Higher is better; includes missing depth")}
        ${annotationMetricTile("Threshold AUC", aucBaseline, aucVariant, formatPercent, formatPercentagePoints, "higher", "Higher is better across 5-50 mm")}
        ${annotationMetricTile("Residual P95", residualBaseline, residualVariant, formatMillimetres, formatMillimetreDelta, "lower", "Lower is better")}
      </div>
      ${productReferenceHtml}
      <p class="annotation-legend"><strong>B</strong> = ${escapeHtml(state.baseline)} repeat ${state.baselineRepeat}; <strong>V</strong> = ${escapeHtml(state.variant)} repeat ${state.variantRepeat}. Delta colors show direction only; aggregate regression gates remain the source for statistical decisions.</p>
      <div class="annotation-structure-list">
        <div class="annotation-structure-head"><span>Structure</span><span>Fit status</span><span>Effective @20 mm</span><span>Threshold AUC</span><span>Residual P95</span><span>Evidence</span></div>
        ${structureRows || `<p class="empty">No structures match this selection.</p>`}
      </div>
      ${uniqueProblems.length ? `<details class="annotation-unavailable"><summary>${uniqueProblems.length} unavailable or unmatched evidence records</summary><ul class="annotation-unavailable-list">${uniqueProblems.map((problem) => `<li>${escapeHtml(problem)}</li>`).join("")}</ul></details>` : ""}`;

    content.querySelectorAll("details.annotation-structure").forEach((details) => details.addEventListener("toggle", () => {
      if (!details.open) return;
      details.querySelectorAll("img[data-src]").forEach((image) => {
        image.src = image.dataset.src;
        image.removeAttribute("data-src");
      });
    }));
  }

  function smallTable(headers, rows) {
    if (!rows.length) return "";
    return `<div class="table-wrap"><table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((value) => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function renderMechanics() {
    const mechanics = model.mechanics || {};
    const current = frame();
    const runs = new Set([state.baseline, state.variant]);
    const match = (row) => {
      const key = row.estimation_stage === "geometric_consistency" ? `geometric_consistency:${row.geometric_iteration}` : "photometric";
      const repeat = row.run === state.baseline ? state.baselineRepeat : state.variantRepeat;
      return row.scene_id === state.scene && Number(row.image_id) === Number(current?.image_id) && runs.has(row.run) && Number(row.repeat ?? 0) === Number(repeat) && key === state.captureStage && mechanicsPyramidLevelMatches(row);
    };
    const aligned = (rows) => {
      const matched = rows.filter(match);
      if (state.alignment === "same") return matched.filter((row) => Number(row.logical_iteration) === Number(state.iteration));
      const finalByRun = new Map();
      matched.forEach((row) => {
        if (row.logical_iteration == null) return;
        const key = `${row.run}:${row.repeat ?? 0}`;
        finalByRun.set(key, Math.max(finalByRun.get(key) ?? -Infinity, Number(row.logical_iteration)));
      });
      return matched.filter((row) => Number(row.logical_iteration) === finalByRun.get(`${row.run}:${row.repeat ?? 0}`));
    };
    const exactCosts = aligned(mechanics.exact_cost_evolution || []);
    const costSignals = [...new Set(exactCosts.map((row) => row.signal))];
    const costs = costSignals.map((signal) => {
      const baseline = exactCosts.find((row) => row.run === state.baseline && row.signal === signal);
      const variant = exactCosts.find((row) => row.run === state.variant && row.signal === signal);
      return [signal, format(baseline?.median), format(variant?.median), format(variant && baseline ? Number(variant.median) - Number(baseline.median) : null), format(baseline?.p90), format(variant?.p90), baseline?.measurement_quality || variant?.measurement_quality || "exact"];
    });
    const textureEvidence = aligned(mechanics.texture_stratification?.rows || []);
    const textureKeys = [...new Set(textureEvidence.map((row) => `${row.signal}\u0000${row.region}`))];
    const textureRows = textureKeys.map((key) => {
      const [signal, region] = key.split("\u0000");
      const baseline = textureEvidence.find((row) => row.run === state.baseline && row.signal === signal && row.region === region);
      const variant = textureEvidence.find((row) => row.run === state.variant && row.signal === signal && row.region === region);
      const delta = baseline?.mean == null || variant?.mean == null ? null : Number(variant.mean) - Number(baseline.mean);
      return [signal, region, baseline?.region_pixels ?? variant?.region_pixels ?? "n/a", format(baseline?.mean), format(variant?.mean), format(delta), format(baseline?.p90), format(variant?.p90)];
    });
    const hysteresisEvidence = aligned(mechanics.low_texture_update_hysteresis?.rows || []);
    const hysteresisRows = hysteresisEvidence.map((row) => [
      row.run, row.eligible_pixels,
      row.propagation_accepted, row.propagation_rejected,
      formatPercent(row.propagation_rejection_rate),
      row.refinement_accepted, row.refinement_rejected,
      formatPercent(row.refinement_rejection_rate),
      format(row.mean_required_gain, 6), format(row.mean_best_proposed_gain, 6),
    ]);
    const gainEvidence = aligned(mechanics.accepted_gain_census?.rows || []);
    const gainRows = gainEvidence.map((row) => [
      row.run, row.low_texture_pixels, row.accepted_gain_pixels,
      formatPercent(row.accepted_fraction_of_low_texture),
      format(row.gain_quantiles?.p50, 6),
      formatPercent(row.fractions_below?.["0.00025"]),
      formatPercent(row.fractions_below?.["0.0005"]),
      formatPercent(row.fractions_below?.["0.001"]),
    ]);
    const iterationRows = aligned(mechanics.exact_iterations || []);
    const updates = iterationRows.map((row) => {
      const sources = Object.entries(row).filter(([key, value]) => key.startsWith("source_") && key !== "source_csv" && Number(value) > 0).map(([key, value]) => `${key.slice(7)}=${value}`).join(", ");
      const meaning = Number(row.logical_iteration) < 0 ? "stored initialization assignments" : "sequential candidate acceptances";
      return [row.run, row.tested_candidates, row.finite_candidates, row.accepted_candidates, meaning, format(row.gap_p50), format(row.gap_p90), sources || "none"];
    });
    let viewRows = aligned(mechanics.exact_views || []);
    if (state.sourceView !== "auto") viewRows = viewRows.filter((row) => Number(row.source_view_index) === Number(state.sourceView));
    const views = viewRows.map((row) => [row.run, row.source_view_index, row.source_image_id, format(Number(row.selected_pixels) / Math.max(1, Number(row.pixels))), format(row.weight_mean), format(row.probability_mean), format(row.weighted_contribution_mean), format(row.total_cost_mean)]);
    const frameIterations = (current?.run_frames || []).flatMap((row) => row.iterations || []);
    const healthEvidence = aligned(frameIterations).filter((row) => row.view_probability_processed != null);
    const healthRows = healthEvidence.map((row) => [
      row.run,
      pyramidLevelLabel(pyramidLevelOf(row)),
      Number(row.logical_iteration) < 0 ? "initialization" : `iteration ${Number(row.logical_iteration) + 1}`,
      row.view_probability_processed,
      formatPercent(row.view_probability_healthy_ratio),
      formatPercent(row.view_probability_degenerate_ratio),
      row.view_probability_zero_mass_events,
      row.view_probability_nonfinite_component_events,
      row.view_probability_negative_component_events,
      row.view_probability_nonfinite_sum_events,
      format(row.view_probability_mean_positive_views, 2),
      row.view_probability_unassigned_draws,
      row.view_probability_legacy_last_view_collapse_events,
    ]);
    const healthStatusRows = (current?.run_frames || []).filter(match).map((row) => {
      const health = row.view_probability_health || {};
      return [
        row.run,
        health.requested ?? "n/a",
        health.available ?? "n/a",
        health.accounting_valid ?? "n/a",
        health.unavailable_reason || (health.available ? "available" : "not declared by this capture"),
      ];
    });
    const cpu = (mechanics.cpu_view_candidates || []).filter(match).sort((a, b) => Number(a.raw_rank_zero_based ?? 1e9) - Number(b.raw_rank_zero_based ?? 1e9)).slice(0, 32);
    const cpuRows = cpu.map((row) => [row.run, row.candidate_image_id, row.raw_rank_zero_based ?? "n/a", format(row.ranking_score), row.initial_decision, row.filter_decision, row.final_rank_zero_based ?? "n/a", row.accepted_after_filter]);
    const estimation = (mechanics.cpu_estimation_selection || []).filter(match).sort((a, b) => Number(a.filtered_rank_zero_based ?? 1e9) - Number(b.filtered_rank_zero_based ?? 1e9)).slice(0, 32);
    const estimationRows = estimation.map((row) => [row.run, row.admission_policy || "legacy", row.score_threshold_applied ?? "n/a", format(row.configured_effective_min_score), row.configured_score_threshold_status || "n/a", row.candidate_image_id, row.filtered_rank_zero_based ?? "n/a", format(row.ranking_score), format(row.score_ratio_to_best), row.would_pass_configured_score_threshold ?? "n/a", row.selected_rank_zero_based ?? "n/a", row.decision]);
    const filters = (mechanics.postprocess_filters || []).filter(match).sort((a, b) => Number(a.stage_index) - Number(b.stage_index) || String(a.run).localeCompare(String(b.run)));
    const filterRows = filters.map((row) => [
      row.run, row.stage_index, row.stage_name, row.artifact_status, row.enabled, row.executed,
      row.input_valid_depth_pixels ?? "n/a", row.output_valid_depth_pixels ?? "n/a",
      row.removed_pixels ?? "n/a", row.added_pixels ?? "n/a", row.depth_changed_pixels ?? "n/a",
      format(row.depth_abs_delta_mean_all_pixels), row.unavailable_reason || "-",
    ]);
    const confidence = (mechanics.confidence_adjustment || []).filter(match);
    const confidenceRows = confidence.map((row) => [
      row.run, row.method, row.artifact_status, row.enabled, row.executed, row.output_available,
      row.input_positive_confidence_pixels ?? "n/a", row.output_positive_confidence_pixels ?? "n/a",
      row.changed_pixels ?? "n/a", format(row.abs_delta_mean_all_pixels), row.final_combination || "-",
      row.unavailable_reason || "-",
    ]);
    const resources = [...(mechanics.cuda_resource_plans || []), ...(mechanics.filter_resource_plans || [])].filter(match);
    const resourceRows = resources.map((row) => [
      row.run, row.pyramid_level ?? "n/a", row.component, row.decision,
      row.trace_requested ?? "n/a", row.trace_available ?? "n/a",
      row.maps_requested, row.maps_available,
      format(Number(row.effective_device_bytes || 0) / 1048576, 2),
      format(Number(row.effective_host_bytes || 0) / 1048576, 2),
      format(Number(row.effective_storage_bytes || 0) / 1048576, 2),
      format(Number(row.current_pyramid_storage_bytes || 0) / 1048576, 2),
      format(Number(row.frame_storage_committed_before_bytes || 0) / 1048576, 2),
      format(Number(row.full_resolution_priority_reserve_bytes || 0) / 1048576, 2),
      format(Number(row.storage_frame_priority_reservation_bytes || 0) / 1048576, 2),
      row.storage_frame_priority_reservation_consumed ?? "n/a",
      row.storage_preflight_succeeded ?? "n/a", row.lease_released ?? "n/a",
      row.actual_map_count ?? "n/a", row.valid ?? "unavailable", row.reason || "-",
    ]);
    const coarseAvailabilityRows = (
      mechanics.coarse_compatibility_map_availability || []
    ).filter(match).map((row) => [
      row.run,
      row.pyramid_level,
      row.update_source_map_expected ?? "n/a",
      row.cost_map_expected ?? "n/a",
      row.cost_map_available ?? "unavailable",
      row.cost_map_unavailable_reason || "-",
      row.measurement_basis || "-",
    ]);
    const maskRows = (current?.run_frames || []).filter((row) => {
      const repeat = row.run === state.baseline ? state.baselineRepeat : state.variantRepeat;
      return runs.has(row.run) && Number(row.repeat) === Number(repeat);
    }).map((row) => {
      const mask = row.ignore_mask || {};
      const rejected = mask.status === "not_requested" ? "n/a" :
        (mask.rejection_count_available ? (mask.rejected_pixels ?? 0) : "unavailable");
      return [row.run, mask.status || "legacy_count_available", mask.requested ?? "n/a", mask.loaded ?? "n/a", rejected, mask.unavailable_reason || "-"];
    });
    const unavailable = (label, available) => available ? "" : `<p class="mechanics-unavailable">${escapeHtml(label)} unavailable for this capture. The model retains this absence explicitly.</p>`;
    const extensionBlocks = [];
    if (hysteresisRows.length) {
      extensionBlocks.push(`<div class="mechanics-block"><h3>Low-texture update hysteresis <small>(optional extension)</small></h3><p class="mechanics-note">Counts combine both checkerboards into one logical iteration. Rejection rates use only legacy-improving gate-controlled proposals; refinement can contribute multiple sequential proposals per pixel.</p>${smallTable(["Run", "Eligible pixels", "Propagation accepted", "Propagation rejected", "Propagation rejection", "Refinement accepted", "Refinement rejected", "Refinement rejection", "Mean required gain", "Mean best proposed gain"], hysteresisRows)}</div>`);
    }
    if (gainRows.length) {
      extensionBlocks.push(`<div class="mechanics-block"><h3>Low-texture accepted-gain census <small>(optional extension)</small></h3><p class="mechanics-note">Pre-change retained-winner diagnostic: positive exact incumbent-minus-winner gains below the run's explicitly configured reference-variance threshold. It does not infer coarse-prior eligibility.</p>${smallTable(["Run", "Low-texture pixels", "Accepted gain pixels", "Accepted / low texture", "Gain P50", "Below 0.00025", "Below 0.0005", "Below 0.001"], gainRows)}</div>`);
    }
    const candidateGapNote = mechanics.availability?.low_texture_update_hysteresis
      ? " Initialization reports stored assignments; iterations report sequential acceptances. The optional hysteresis extension can suppress a raw minimum; use its registered raw-order maps in those pixels."
      : " Initialization reports stored assignments; iterations report sequential acceptances.";
    byId("mechanics-content").innerHTML = [
      `<div class="mechanics-block"><h3>Exact production cost evolution</h3>${smallTable(["Signal", "Baseline median", "Variant median", "Delta", "Baseline P90", "Variant P90", "Quality"], costs) || unavailable("Exact hot-kernel cost maps", false)}</div>`,
      `<div class="mechanics-block"><h3>Texture-stratified outcomes</h3><p class="mechanics-note">Low, mid, and high regions are finite-value thirds computed independently for each matched frame and logical state.</p>${smallTable(["Signal", "Texture region", "Pixels", "Baseline mean", "Variant mean", "Delta", "Baseline P90", "Variant P90"], textureRows) || unavailable("Registered texture score or variance maps", false)}</div>`,
      ...extensionBlocks,
      `<div class="mechanics-block"><h3>Candidate update attribution and winner gap</h3><p class="mechanics-note">${candidateGapNote}</p>${smallTable(["Run", "Tested", "Finite", "Accepted / stored", "Meaning", "Gap P50", "Gap P90", "Source counts"], updates) || unavailable("Exact candidate iteration table", false)}</div>`,
      `<div class="mechanics-block"><h3>Per-view decisions and weighted contribution</h3>${smallTable(["Run", "View", "Image", "Selected", "Weight", "Probability", "Contribution", "Total cost"], views) || unavailable("Exact per-view summary", false)}</div>`,
      `<div class="mechanics-block"><h3>View probability health</h3><p class="mechanics-note">Aggregate pre-CDF observer census for the selected complete logical iteration. It does not invent unavailable per-pixel maps.</p>${smallTable(["Run", "Pyramid", "State", "Processed", "Healthy", "Degenerate", "Zero mass", "Nonfinite component", "Negative component", "Nonfinite sum", "Mean positive views", "Unassigned draws", "Legacy collapse"], healthRows) || unavailable("Per-iteration probability-health census", false)}${smallTable(["Run", "Requested", "Available", "Accounting valid", "Unavailable reason"], healthStatusRows)}</div>`,
      `<div class="mechanics-block"><h3>CPU candidate ranking</h3>${smallTable(["Run", "Candidate", "Raw rank", "Score", "Initial", "Filter", "Final rank", "Accepted"], cpuRows) || unavailable("CPU view-ranking artifacts", false)}</div>`,
      `<div class="mechanics-block"><h3>CPU estimation-source selection</h3>${smallTable(["Run", "Admission policy", "Score cutoff applied", "Configured cutoff", "Cutoff status", "Candidate", "Filtered rank", "Score", "Ratio", "Would pass configured cutoff", "Selected rank", "Decision"], estimationRows) || unavailable("CPU estimation-selection artifacts", false)}</div>`,
      `<div class="mechanics-block"><h3>Sequential postprocess filtering</h3>${smallTable(["Run", "Sequence", "Stage", "Status", "Enabled", "Executed", "Valid in", "Valid out", "Removed", "Added", "Depth changed", "Mean |depth delta|", "Unavailable reason"], filterRows) || unavailable("Sequential postprocess observations", false)}</div>`,
      `<div class="mechanics-block"><h3>Confidence adjustment</h3>${smallTable(["Run", "Method", "Status", "Enabled", "Executed", "Output", "Positive in", "Positive out", "Changed", "Mean |delta|", "Combination", "Unavailable reason"], confidenceRows) || unavailable("Confidence-adjustment observations", false)}</div>`,
      `<div class="mechanics-block"><h3>Resource admission and storage preflight</h3><p class="mechanics-note">Frame storage is cumulative across pyramid levels. Coarse levels hold the displayed full-resolution reserve until it is atomically consumed at the fine level.</p>${smallTable(["Run", "Pyramid", "Component", "Decision", "Trace requested", "Trace admitted", "Maps requested", "Maps admitted", "Device MiB", "Host MiB", "Frame MiB", "Current pyramid MiB", "Committed before MiB", "Full-res reserve MiB", "Held reserve MiB", "Reserve consumed", "Preflight", "Lease released", "Actual maps", "Valid", "Reason"], resourceRows) || unavailable("Resource-plan observations", false)}</div>`,
      `<div class="mechanics-block"><h3>Coarse compatibility-map availability</h3><p class="mechanics-note">Schema-v4 resource plans explicitly distinguish the available coarse update-source proxy from unavailable production cost/confidence maps.</p>${smallTable(["Run", "Pyramid", "Update-source expected", "Cost expected", "Cost available", "Unavailable reason", "Basis"], coarseAvailabilityRows) || unavailable("Coarse compatibility-map contracts", false)}</div>`,
      `<div class="mechanics-block"><h3>Ignore-mask availability</h3>${smallTable(["Run", "Ignore mask", "Requested", "Loaded", "Mask-rejected pixels", "Unavailable reason"], maskRows) || unavailable("Ignore-mask observations", false)}</div>`,
    ].join("");
  }

  function renderAll() {
    renderSelectors(); renderSignalPicker(); renderStatus(); renderCaptureProfileCoverage(); renderEvidenceContext(); renderAccuracyLedger(); renderMechanismImpact(); renderRegressions(); renderMechanics(); renderMaps(); renderDrilldownStatus(); renderAnnotations(); renderGuideContext(); installColumnTooltips(); writeHash();
  }

  Object.entries(controls).forEach(([key, control]) => control.addEventListener("change", () => {
    state[key] = ["baselineRepeat", "variantRepeat", "iteration", "channel"].includes(key) ? Number(control.value) : control.value;
    if (key === "scene") state.frame = scene()?.frames[0]?.id || "";
    if (key === "mechanism") state.component = "all";
    if (key === "mapPreset") applyMapPreset();
    state.x = null; state.y = null; renderAll();
  }));
  document.querySelectorAll("input[name=scale]").forEach((input) => input.addEventListener("change", () => { state.scale = input.value; renderMaps(); writeHash(); }));
  byId("clear-crosshair").addEventListener("click", () => { state.x = null; state.y = null; updateCrosshairs(); renderPixelTable(); renderDrilldownStatus(); writeHash(); });
  byId("deep-frame-request").addEventListener("click", () => exportDrilldownRequest("deep"));
  byId("trace-pixel-request").addEventListener("click", () => exportDrilldownRequest("trace"));
  byId("regression-filter").addEventListener("input", (event) => { state.regressionFilter = event.target.value; renderRegressions(); });
  document.querySelectorAll("#regression-table th button[data-sort]").forEach((button) => button.addEventListener("click", () => {
    if (state.sort === button.dataset.sort) state.sortDirection *= -1; else { state.sort = button.dataset.sort; state.sortDirection = button.dataset.sort === "regression_score" ? -1 : 1; }
    renderRegressions();
  }));
  byId("guide-open").addEventListener("click", openGuide);
  byId("guide-close").addEventListener("click", closeGuide);
  byId("guide-dialog").addEventListener("cancel", (event) => { event.preventDefault(); closeGuide(); });
  byId("guide-dialog").addEventListener("click", (event) => { if (event.target === byId("guide-dialog")) closeGuide(); });
  byId("guide-dialog").addEventListener("close", () => {
    document.body.classList.remove("guide-open");
    if (guideOpener instanceof HTMLElement) guideOpener.focus();
    guideOpener = null;
  });
  window.addEventListener("hashchange", () => { parseHash(); renderAll(); });
  window.addEventListener("resize", () => { if (activeColumnHelp) positionColumnTooltip(activeColumnHelp); });
  window.addEventListener("scroll", hideColumnTooltip, true);
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || !activeColumnHelp) return;
    activeColumnHelp.blur();
    hideColumnTooltip();
  });
  const reportName = String(model.experiment.name || "Depth-map Investigation");
  byId("report-title").textContent = reportName.replaceAll("_", " ");
  byId("report-title").title = reportName;
  byId("report-subtitle").textContent = `${model.runs.length} run captures / ${model.scenes.length} scenes / ${model.map_catalog_summary.artifacts} map artifacts`;
  initializeState(); renderGuide(); renderAll();
})();
