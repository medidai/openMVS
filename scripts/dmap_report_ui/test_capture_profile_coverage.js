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

function renderedHtml(model) {
  const embeddedModel = JSON.stringify(model).replaceAll("<", "\\u003c");
  return TEMPLATE
    .replace("__DMAP_REPORT_CSS__", CSS)
    .replace("__DMAP_REPORT_MODEL__", embeddedModel)
    .replace("__DMAP_REPORT_JS__", JAVASCRIPT);
}

function dumpDom(model) {
  const browser = browserBinary();
  if (!browser) return null;
  const directory = mkdtempSync(join(tmpdir(), "dmap-profile-ui-"));
  try {
    const report = join(directory, "02_investigation.html");
    writeFileSync(report, renderedHtml(model));
    const result = spawnSync(browser, [
      "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
      `--user-data-dir=${join(directory, "browser")}`, "--virtual-time-budget=1500", "--dump-dom", `file://${report}`,
    ], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
    assert.equal(result.status, 0, `headless browser failed:\n${result.stderr}`);
    return result.stdout;
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

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
