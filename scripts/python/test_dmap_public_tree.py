#!/usr/bin/env python3
"""Focused tests for the artifact-free public-tree policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "tools/check_dmap_observability_public_tree.py"
SPEC = importlib.util.spec_from_file_location("dmap_public_tree", CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PublicTreePolicyTests(unittest.TestCase):
	def test_observability_sources_are_allowlisted(self) -> None:
		self.assertTrue(CHECKER._is_allowed_path("libs/MVS/PatchMatchCUDA.cu"))
		self.assertTrue(CHECKER._is_allowed_path("scripts/python/test_dmap_example.py"))
		self.assertTrue(CHECKER._is_allowed_path("docs/dmap_observability/01_quickstart.md"))
		self.assertTrue(CHECKER._is_allowed_path("WORKSPACE.md"))

	def test_experiment_evidence_is_not_allowlisted(self) -> None:
		self.assertFalse(CHECKER._is_allowed_path("experiments/private/results.json"))
		self.assertFalse(CHECKER._is_allowed_path("docs/visualizations/result.png"))
		self.assertFalse(CHECKER._is_allowed_path("reports/01_development_report.md"))

	def test_result_shaped_files_inside_source_prefixes_are_not_allowlisted(self) -> None:
		for path in (
			"docs/dmap_observability/results.json",
			"docs/dmap_observability/01_concrete_report.md",
			"examples/dmap_observability/captured_metrics.yaml",
			"scripts/dmap_report_ui/experiment_result.html",
			"scripts/python/dmap_observability/captured_result.json",
		):
			with self.subTest(path=path):
				self.assertFalse(CHECKER._is_allowed_path(path))

	def test_changed_paths_include_deleted_baseline_files(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			repo = Path(temporary)
			subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
			subprocess.run(
				["git", "config", "user.email", "observer@example.org"],
				cwd=repo,
				check=True,
			)
			subprocess.run(
				["git", "config", "user.name", "Observer Test"],
				cwd=repo,
				check=True,
			)
			deleted = repo / "unrelated.txt"
			deleted.write_text("baseline\n", encoding="utf-8")
			subprocess.run(["git", "add", "unrelated.txt"], cwd=repo, check=True)
			subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
			deleted.unlink()

			self.assertIn("unrelated.txt", CHECKER._changed_paths(repo, "HEAD"))

	def test_surface_paths_include_dangling_exact_path_symlinks(self) -> None:
		with tempfile.TemporaryDirectory() as temporary:
			repo = Path(temporary)
			path = "tools/check_dmap_observability_public_tree.py"
			candidate = repo / path
			candidate.parent.mkdir(parents=True)
			candidate.symlink_to(repo / "missing-target")

			self.assertIn(path, CHECKER._surface_paths(repo))

	def test_every_prefixed_source_file_has_an_exact_manifest_entry(self) -> None:
		actual_paths = set()
		for prefix in CHECKER.SCANNED_PREFIXES:
			root = REPO_ROOT / prefix
			for candidate in root.rglob("*"):
				if not candidate.is_file() or CHECKER._is_python_cache(candidate.as_posix()):
					continue
				actual_paths.add(candidate.relative_to(REPO_ROOT).as_posix())
		self.assertEqual(actual_paths, CHECKER.PREFIX_SOURCE_PATHS)

	def test_secret_patterns_fail(self) -> None:
		errors = CHECKER._scan_text(
			"docs/dmap_observability/example.md",
			"-----BEGIN " + "PRIVATE KEY-----",
			[],
			check_concrete_paths=True,
		)
		self.assertTrue(any("private key" in error for error in errors))

	def test_concrete_machine_path_fails(self) -> None:
		errors = CHECKER._scan_text(
			"docs/dmap_observability/example.md",
			"input: /home/developer/private-dataset/scene.mvs",
			[],
			check_concrete_paths=True,
		)
		self.assertTrue(any("concrete absolute path" in error for error in errors))

	def test_test_files_are_not_blanket_exempt_from_concrete_path_scanning(self) -> None:
		path = "scripts/python/test_future_observability.py"
		self.assertNotIn(path, CHECKER.SANITIZER_FIXTURE_PATHS)
		errors = CHECKER._scan_text(
			path,
			"input: /home/developer/private-dataset/scene.mvs",
			[],
			check_concrete_paths=True,
		)
		self.assertTrue(any("concrete absolute path" in error for error in errors))

	def test_tmp_path_and_example_identity_are_portable(self) -> None:
		errors = CHECKER._scan_text(
			"docs/dmap_observability/example.md",
			"output: /tmp/dmap-demo\ncontact: developer@example.org\n",
			[],
			check_concrete_paths=True,
		)
		self.assertEqual(errors, [])

	def test_explicit_forbidden_marker_is_case_insensitive(self) -> None:
		errors = CHECKER._scan_text(
			"scripts/python/example.py",
			"DATASET = 'Internal-Corpus'",
			["internal-corpus"],
			check_concrete_paths=False,
		)
		self.assertTrue(any("forbidden marker" in error for error in errors))

	def test_disabled_source_strips_observer_branch_and_retains_else(self) -> None:
		source = """before
#ifdef _USE_DMAP_INSTRUMENTATION
observer_only();
#if OTHER_FLAG
nested_observer_only();
#endif
#else
production_only();
#endif
after
"""
		self.assertEqual(
			CHECKER._without_disabled_observer_blocks(source, "fixture.cpp"),
			["before", "production_only();", "after"],
		)

	def test_disabled_source_parity_covers_the_production_cli(self) -> None:
		self.assertIn(
			"apps/DensifyPointCloud/DensifyPointCloud.cpp",
			CHECKER.OFF_SOURCE_PARITY_PATHS,
		)

	def test_disabled_source_preserves_unrelated_conditionals(self) -> None:
		source = """#ifdef OTHER_FLAG
first();
#else
second();
#endif
"""
		self.assertEqual(
			CHECKER._without_disabled_observer_blocks(source, "fixture.cpp"),
			["#ifdef OTHER_FLAG", "first();", "#else", "second();", "#endif"],
		)

	def test_cuda_compatibility_transform_is_exact(self) -> None:
		baseline = "\n".join(
			before for before, _ in CHECKER.PATCHMATCH_CUDA_COMPATIBILITY_REPLACEMENTS
		)
		expected = "\n".join(
			after for _, after in CHECKER.PATCHMATCH_CUDA_COMPATIBILITY_REPLACEMENTS
		)
		self.assertEqual(
			CHECKER._apply_patchmatch_cuda_compatibility(baseline),
			expected,
		)

	def test_cuda_compatibility_transform_rejects_baseline_drift(self) -> None:
		with self.assertRaisesRegex(ValueError, "baseline changed"):
			CHECKER._apply_patchmatch_cuda_compatibility("unexpected source")

	def test_observer_cli_levels_have_patchmatch_runtime_contracts(self) -> None:
		densify_source = (
			REPO_ROOT / "apps/DensifyPointCloud/DensifyPointCloud.cpp"
		).read_text(encoding="utf-8")
		patchmatch_source = (
			REPO_ROOT / "libs/MVS/PatchMatchCUDA.cpp"
		).read_text(encoding="utf-8")
		accepted_levels = set(
			re.findall(r'level != _T\("([a-z]+)"\)', densify_source)
		)
		self.assertTrue(accepted_levels, "could not find the observer CLI level allowlist")
		contract_start = patchmatch_source.index('metadata["level_contract"]')
		contract_end = patchmatch_source.index(
			'metadata["disabled_contract"]', contract_start
		)
		level_contract = patchmatch_source[contract_start:contract_end]

		for level in sorted(accepted_levels):
			with self.subTest(level=level):
				self.assertTrue(
					f'{{"{level}", {{' in level_contract,
					f"CLI accepts {level!r}, but PatchMatch has no runtime level contract",
				)

	def test_private_instrumentation_guard_is_derived_only_from_public_option(self) -> None:
		cmake_source = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
		config_template = (
			REPO_ROOT / "build" / "Templates" / "ConfigLocal.h.in"
		).read_text(encoding="utf-8")
		self.assertIn(
			'SET(OPENMVS_DMAP_INSTRUMENTATION_ENABLED "${OpenMVS_DMAP_INSTRUMENTATION}")',
			cmake_source,
		)
		self.assertNotRegex(
			cmake_source,
			r"SET\s*\(\s*_USE_DMAP_INSTRUMENTATION",
		)
		self.assertIn(
			"#cmakedefine01 OPENMVS_DMAP_INSTRUMENTATION_ENABLED",
			config_template,
		)
		self.assertIn("#ifdef _USE_DMAP_INSTRUMENTATION", config_template)
		self.assertIn("#error", config_template)

	def test_cpp_publishes_the_prefilter_artifacts_required_by_python(self) -> None:
		patchmatch_source = (
			REPO_ROOT / "libs/MVS/PatchMatchCUDA.cpp"
		).read_text(encoding="utf-8")
		validator_source = (
			REPO_ROOT / "scripts/python/dmap_dev.py"
		).read_text(encoding="utf-8")

		for artifact, schema_name in (
			("prefilter_manifest.json", "openmvs.dmap.prefilter_manifest"),
			(
				"prefilter_capture_complete.json",
				"openmvs.dmap.prefilter_capture_complete",
			),
		):
			with self.subTest(artifact=artifact):
				self.assertIn(artifact, validator_source)
				self.assertIn(schema_name, validator_source)
				self.assertTrue(
					re.search(
						rf'WriteJsonFile\([^;]*_T\("{re.escape(artifact)}"\)',
						patchmatch_source,
						re.DOTALL,
					) is not None,
					f"Python requires {artifact}, but PatchMatch never publishes it",
				)
				self.assertTrue(
					schema_name in patchmatch_source,
					f"Python requires schema {schema_name}, but PatchMatch never declares it",
				)

	def test_cpp_trace_rows_declare_exact_or_proxy_source_provenance(self) -> None:
		patchmatch_source = (
			REPO_ROOT / "libs/MVS/PatchMatchCUDA.cpp"
		).read_text(encoding="utf-8")

		for contract in (
			'line["measurement_basis"] = exactHotKernelRecords ? "exact_hot_kernel" : "post_pass_proxy"',
			'line["source_quality"] = exactHotKernelRecords ? "exact" : "proxy"',
		):
			self.assertIn(contract, patchmatch_source)
		self.assertIn(
			"instrumentExtendedMaps.exactAvailable)", patchmatch_source,
			"trace provenance must be selected from the admitted exact hot-kernel capture",
		)


if __name__ == "__main__":
	unittest.main()
