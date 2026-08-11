const assert = require("node:assert/strict");
const { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join, resolve } = require("node:path");
const { spawnSync } = require("node:child_process");
const test = require("node:test");

const UI_ROOT = resolve(__dirname);
const TEMPLATE = readFileSync(join(UI_ROOT, "investigation.html"), "utf8");
const CSS = readFileSync(join(UI_ROOT, "investigation.css"), "utf8");
const JAVASCRIPT = readFileSync(join(UI_ROOT, "investigation.js"), "utf8");
const PROFILE_ORDER = ["endpoint", "summary", "prefilter", "deep", "trace"];
const TEST_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WlYvWQAAAAASUVORK5CYII=";

function browserBinary() {
  const candidates = [process.env.CHROME_BIN, "/usr/bin/google-chrome", "/snap/bin/chromium"].filter(Boolean);
  return candidates.find(existsSync) || null;
}

function baseModel() {
  const profiles = PROFILE_ORDER.map((profile) => ({
    profile,
    requested: true,
    status: "complete",
    expected_units: 2,
    complete_units: 2,
    failed_units: 0,
    unavailable_units: 0,
    process_specialization: ["deep", "trace"].includes(profile) ? "Process<true>" : "Process<false>",
    quality_authority: ["deep", "trace"].includes(profile) ? "diagnostic_only" : profile === "endpoint" ? "production" : "after_endpoint_parity",
  }));
  const units = ["baseline", "candidate"].flatMap((configuredRun) => PROFILE_ORDER.map((profile) => ({
    configured_run: configuredRun,
    repeat: 0,
    scene_id: "scene-a",
    capture_profile: profile,
    status: "complete",
    reason: "",
    evidence_links: [{ label: `${profile} completion`, path: `../captures/${configuredRun}/${profile}/completion.json` }],
    frames: [{
      image_id: 17,
      status: "complete",
      evidence_links: [{ label: `${profile} frame`, path: `../captures/${configuredRun}/${profile}/frame-17.json` }],
    }],
  })));
  units.find((unit) => unit.configured_run === "candidate" && unit.capture_profile === "trace").evidence_links.push({
    label: "blocked host path",
    path: "/private/capture/trace.json",
  });
  return {
    schema_name: "openmvs.dmap.development_report",
    schema_version: 3,
    experiment: { name: "synthetic_profile_ui", source_config: "../experiment.yaml" },
    runs: [
      { label: "baseline", repeat: 0, role: "baseline", diagnostic_only: false },
      { label: "candidate", repeat: 0, role: "variant", diagnostic_only: false },
    ],
    scenes: [{
      id: "scene-a", label: "Synthetic scene", frame_count: 1,
      frames: [{ id: "frame-scene-a-17", image_id: 17, label: "frame 17", maps: [], run_frames: [], annotations: [] }],
    }],
    signals: [],
    map_catalog_summary: {
      artifacts: 0, available: 0, unavailable: 0,
      pixel_data: { selected_artifacts: 0, eligible_artifacts: 0, omitted_artifacts: 0, encoded_pixel_output_bytes: 0 },
    },
    aggregates: { accuracy_ledger: [], gates: [], mechanism_impact: [], regressions: [] },
    mechanics: { availability: {} },
    capture_validation: {},
    drilldowns: { entries: [] },
    capture_profile_coverage: {
      schema_name: "openmvs.dmap.capture_profile_coverage",
      schema_version: 1,
      requested_profiles: [...PROFILE_ORDER],
      profiles,
      units,
    },
  };
}

function patchLayout() {
  const axis = [-4, -2, 0, 2, 4];
  return {
    schema_name: "openmvs.dmap.reference_patch_layout",
    schema_version: 1,
    kind: "fixed_cartesian_grid",
    coordinate_domain: "reference_pyramid_pixels",
    sample_position: "integer_offset_from_pixel_center",
    texel_center_offset: 0.5,
    texture_address_mode_configured: "wrap",
    texture_address_mode_effective: "clamp",
    texture_address_mode_effective_basis: "cuda_runtime_unnormalized_wrap_is_clamped",
    texture_coordinates_normalized: false,
    texture_filter_mode: "linear",
    half_window_pixels: 4,
    step_pixels: 2,
    sample_count: 25,
    sample_offsets_pixels: axis.flatMap((y) => axis.map((x) => [x, y])),
    layout_provenance: "compiled_cuda_scoring_constants",
    sample_locations_captured_by_kernel: false,
    sample_values_captured_by_kernel: false,
    source_view_footprints_captured_by_kernel: false,
  };
}

function modelWithPatchEvidence() {
  const model = baseModel();
  model.signals = [{
    id: "signal-reference-rgb", name: "reference_rgb", label: "reference RGB",
    measurement_qualities: ["source"], available_artifacts: 1,
    unavailable_artifacts: 0, default: true, mechanism: "input",
    quantity: "reference_image", description: "Reference RGB image.",
  }];
  const frame = model.scenes[0].frames[0];
  frame.reference = { available: true, path: TEST_PNG, unavailable_reason: null };
  model.mechanics.reference_patch_layouts = ["baseline", "candidate"].map((run) => ({
    run, repeat: 0, scene_id: "scene-a", estimation_stage: "photometric",
    geometric_iteration: null, available: true, layout: patchLayout(),
    measurement_quality: "exact", visualization_quality: "derived_exact",
    unavailable_reason: null,
  }));
  model.mechanics.cuda_resource_plans = ["baseline", "candidate"].map((run) => ({
    run, repeat: 0, scene_id: "scene-a", image_id: 17,
    estimation_stage: "photometric", geometric_iteration: null,
    pyramid_level: null,
    grid_extent: {
      available: true, width: 8, height: 6, measurement_quality: "exact",
      measurement_basis: "resource_plan.width_height", unavailable_reason: null,
    },
  }));
  return model;
}

function renderedHtml(model) {
  const embeddedModel = JSON.stringify(model).replaceAll("<", "\\u003c");
  return TEMPLATE
    .replace("__DMAP_REPORT_CSS__", CSS)
    .replace("__DMAP_REPORT_MODEL__", embeddedModel)
    .replace("__DMAP_REPORT_JS__", JAVASCRIPT);
}

function dumpDom(model, hash = "") {
  const browser = browserBinary();
  if (!browser) return null;
  const directory = mkdtempSync(join(tmpdir(), "dmap-profile-ui-"));
  try {
    const report = join(directory, "02_investigation.html");
    writeFileSync(report, renderedHtml(model));
    const result = spawnSync(browser, [
      "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
      `--user-data-dir=${join(directory, "browser")}`, "--virtual-time-budget=1500", "--dump-dom", `file://${report}${hash}`,
    ], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
    assert.equal(result.status, 0, `headless browser failed:\n${result.stderr}`);
    return result.stdout;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function mapSection(dom) {
  return dom.match(/<section class="section-band map-section"[\s\S]*?<section class="section-band"/)?.[0] || "";
}

test("derived fixed reference grids render per arm with clamp-aware loupes", (context) => {
  if (!browserBinary()) return context.skip("Chrome or Chromium is unavailable");
  const section = mapSection(dumpDom(
    modelWithPatchEvidence(), "#x=0.01&y=0.01&signals=reference_rgb",
  ));
  assert.equal((section.match(/data-patch-sample=/g) || []).length, 50);
  assert.equal((section.match(/data-clamped="1"/g) || []).length, 32);
  assert.match(section, /data-patch-role="baseline" data-patch-state="available" data-sample-count="25"/);
  assert.match(section, /data-patch-role="variant" data-patch-state="available" data-sample-count="25"/);
  assert.equal((section.match(/data-rendered="1"/g) || []).length, 2);
  assert.equal((section.match(/data-clamped-samples="16"/g) || []).length, 2);
  assert.match(section, /descriptor requests wrap but unnormalized CUDA coordinates make clamp effective/);
  assert.match(section, /Derived from the validated fixed-layout contract plus exact resource-plan dimensions/);
  assert.match(section, /not the CUDA grayscale pyramid texture/);
  assert.match(section, /not recorded CUDA sample values or a source-view footprint/);

  const sameRunSection = mapSection(dumpDom(
    modelWithPatchEvidence(),
    "#baseline=baseline&variant=baseline&x=0.4&y=0.4&signals=reference_rgb",
  ));
  assert.equal(
    (sameRunSection.match(/class="patch-sample-marker baseline/g) || []).length,
    25,
  );
  assert.equal(
    (sameRunSection.match(/class="patch-sample-marker variant/g) || []).length,
    25,
  );
});

test("derived patch grid toggle and missing metadata remain explicit", (context) => {
  if (!browserBinary()) return context.skip("Chrome or Chromium is unavailable");
  const hidden = mapSection(dumpDom(
    modelWithPatchEvidence(),
    "#x=0.4&y=0.4&signals=reference_rgb&patchLayout=0",
  ));
  assert.doesNotMatch(hidden, /data-patch-sample=/);
  assert.match(hidden, /id="patch-inspector"[^>]*hidden/);

  const partialModel = modelWithPatchEvidence();
  partialModel.mechanics.reference_patch_layouts = partialModel.mechanics.reference_patch_layouts.slice(0, 1);
  const partial = mapSection(dumpDom(
    partialModel, "#x=0.4&y=0.4&signals=reference_rgb",
  ));
  assert.match(partial, /id="patch-inspector-quality" class="quality partial">partial/);
  assert.equal((partial.match(/data-patch-state="available"/g) || []).length, 1);
  assert.equal((partial.match(/data-patch-state="unavailable"/g) || []).length, 1);

  const missingModel = modelWithPatchEvidence();
  missingModel.mechanics.reference_patch_layouts = [];
  const missing = mapSection(dumpDom(
    missingModel, "#x=0.4&y=0.4&signals=reference_rgb",
  ));
  assert.doesNotMatch(missing, /data-patch-sample=/);
  assert.equal((missing.match(/data-patch-state="unavailable"/g) || []).length, 2);
  assert.match(missing, /id="patch-inspector-quality" class="quality unavailable">unavailable/);
  assert.match(missing, /reference patch layout was not declared/);
});

test("supported coverage renders five profiles and selected-frame evidence", (context) => {
  if (!browserBinary()) return context.skip("Chrome or Chromium is unavailable");
  const dom = dumpDom(baseModel());
  assert.match(dom, /id="capture-profile-coverage-section"[^>]*data-state="available"/);
  const section = dom.match(/<section id="capture-profile-coverage-section"[\s\S]*?<\/section>/)?.[0] || "";
  assert.equal((section.match(/class="capture-profile-column"/g) || []).length, 5);
  assert.match(section, /5\/5 profiles requested \/ 10\/10 expected units complete/);
  assert.match(section, /Process&lt;false&gt; \/ after endpoint parity/);
  assert.match(section, /Process&lt;true&gt; \/ diagnostic only/);
  assert.match(section, /Selected-frame evidence <span>scene-a \/ image 17<\/span>/);
  assert.match(section, /prefilter frame/);
  assert.match(section, /deep frame/);
  assert.match(section, /class="capture-evidence-link blocked"[^>]*>.*blocked host path/);
  assert.doesNotMatch(section, /href="\/private\/capture\/trace\.json"/);
});

test("legacy model makes missing profile coverage explicit", (context) => {
  if (!browserBinary()) return context.skip("Chrome or Chromium is unavailable");
  const model = baseModel();
  delete model.capture_profile_coverage;
  const dom = dumpDom(model);
  assert.match(dom, /id="capture-profile-coverage-section"[^>]*data-state="legacy"/);
  assert.match(dom, /Capture profile coverage is unavailable\./);
  assert.match(dom, /Do not infer capture completeness or provenance from map presence\./);
  assert.match(dom, /Capture profile coverage not recorded \(legacy model\)/);
});

test("coverage preserves failed, unavailable, not-requested, and absent unit states", (context) => {
  if (!browserBinary()) return context.skip("Chrome or Chromium is unavailable");
  const model = baseModel();
  const coverage = model.capture_profile_coverage;
  const deep = coverage.profiles.find((profile) => profile.profile === "deep");
  Object.assign(deep, { status: "failed", complete_units: 1, failed_units: 1 });
  const prefilter = coverage.profiles.find((profile) => profile.profile === "prefilter");
  Object.assign(prefilter, { status: "unavailable", complete_units: 1, unavailable_units: 1 });
  const trace = coverage.profiles.find((profile) => profile.profile === "trace");
  Object.assign(trace, { requested: false, status: "not_requested", expected_units: 0, complete_units: 0 });
  coverage.requested_profiles = coverage.requested_profiles.filter((profile) => profile !== "trace");
  coverage.units.find((unit) => unit.configured_run === "candidate" && unit.capture_profile === "deep").status = "failed";
  coverage.units.find((unit) => unit.configured_run === "candidate" && unit.capture_profile === "deep").reason = "synthetic failure";
  coverage.units.find((unit) => unit.configured_run === "candidate" && unit.capture_profile === "prefilter").status = "unavailable";
  coverage.units.find((unit) => unit.configured_run === "candidate" && unit.capture_profile === "prefilter").reason = "synthetic unavailable reason";
  coverage.units.filter((unit) => unit.capture_profile === "trace").forEach((unit) => { unit.status = "not_requested"; });
  coverage.units = coverage.units.filter((unit) => !(unit.configured_run === "baseline" && unit.capture_profile === "summary"));
  const dom = dumpDom(model);
  const section = dom.match(/<section id="capture-profile-coverage-section"[\s\S]*?<\/section>/)?.[0] || "";
  assert.match(section, /capture-state failed[^>]*>Failed/);
  assert.match(section, /capture-state unavailable[^>]*>Unavailable/);
  assert.match(section, /capture-state not-requested[^>]*>Not requested/);
  assert.match(section, /capture-state not-recorded[^>]*>Not recorded/);
  assert.match(section, /synthetic failure/);
  assert.match(section, /synthetic unavailable reason/);
});
