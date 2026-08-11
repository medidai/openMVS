#!/usr/bin/env python3
"""Fail closed when the public DMAP observability surface is not source-only."""

from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
import re
import subprocess
import sys


EXACT_PATHS = {
	".gitignore",
	".github/instructions/dmap-observability.instructions.md",
	"AGENTS.md",
	"CMakeLists.txt",
	"README.md",
	"WORKSPACE.md",
	"apps/DensifyPointCloud/CMakeLists.txt",
	"apps/DensifyPointCloud/DensifyPointCloud.cpp",
	"build/Templates/ConfigLocal.h.in",
	"build/Templates/OpenMVSConfig.cmake.in",
	"libs/MVS/DepthMap.cpp",
	"libs/MVS/DepthMap.h",
	"libs/MVS/PatchMatchCUDA.cpp",
	"libs/MVS/PatchMatchCUDA.cu",
	"libs/MVS/PatchMatchCUDA.inl",
	"libs/MVS/Scene.cpp",
	"libs/MVS/Scene.h",
	"libs/MVS/SceneDensify.cpp",
	"libs/MVS/SceneDensify.h",
	# Build-only compatibility: top-level configuration resolves SFM dependencies
	# even when the requested product is the depth-map executable.
	"libs/SFM/CMakeLists.txt",
	"scripts/dmap_instrumentation_report.py",
	"scripts/python/dmap_dev.py",
	"scripts/python/dmap_drilldown.py",
	"scripts/python/dmap_report_model.py",
	"scripts/python/dmap_sweep.py",
	"scripts/python/generate_attested_dmap_report.py",
	"scripts/python/report_dmap_annotation_fit.py",
	"scripts/python/requirements-depth-benchmark.txt",
	"scripts/python/requirements-dmap-array-store.txt",
	"scripts/python/validate_dmap_disabled.py",
	"scripts/python/validate_dmap_instrumentation.py",
	"tools/check_dmap_observability_public_tree.py",
	"tools/dmap_observability.sh",
}

SCANNED_PREFIXES = (
	"docs/dmap_observability/",
	"examples/dmap_observability/",
	"scripts/dmap_report_ui/",
	"scripts/python/dmap_observability/",
)

# Keep traversing these directories so an unexpected addition is discovered, but
# require every publishable file to be declared here. A suffix-only prefix rule
# would also admit generated reports, metrics, and other result-shaped text.
PREFIX_SOURCE_PATHS = {
	"docs/dmap_observability/01_quickstart.md",
	"docs/dmap_observability/02_architecture.md",
	"docs/dmap_observability/03_capture_profiles.md",
	"docs/dmap_observability/04_experiment_configuration.md",
	"docs/dmap_observability/05_report_guide.md",
	"docs/dmap_observability/06_debugging_playbooks.md",
	"docs/dmap_observability/07_annotation_metrics.md",
	"docs/dmap_observability/08_schema_reference.md",
	"docs/dmap_observability/09_extension_guide.md",
	"docs/dmap_observability/10_troubleshooting.md",
	"docs/dmap_observability/11_patch_debugging.md",
	"docs/dmap_observability/agent_guide.md",
	"docs/dmap_observability/capabilities.json",
	"examples/dmap_observability/README.md",
	"examples/dmap_observability/experiment.template.yaml",
	"scripts/dmap_report_ui/investigation.css",
	"scripts/dmap_report_ui/investigation.html",
	"scripts/dmap_report_ui/investigation.js",
	"scripts/dmap_report_ui/test_capture_profile_coverage.js",
	"scripts/python/dmap_observability/__init__.py",
	"scripts/python/dmap_observability/array_store.py",
	"scripts/python/dmap_observability/array_store_workflow.py",
	"scripts/python/dmap_observability/component_registry.py",
	"scripts/python/dmap_observability/config_materialization.py",
	"scripts/python/dmap_observability/integrity.py",
	"scripts/python/dmap_observability/portable_bundle.py",
	"scripts/python/dmap_observability/predecessor_composite.py",
	"scripts/python/dmap_observability/region_metrics.py",
	"scripts/python/dmap_observability/reference_patch_layout.py",
	"scripts/python/dmap_observability/source_snapshot.py",
	"scripts/python/dmap_observability/sweep_failure_inventory.py",
	"scripts/python/dmap_observability/validated_predecessor.py",
}

ALLOWED_SUFFIXES = {
	".cpp", ".css", ".cu", ".h", ".html", ".inl", ".js", ".json",
	".in", ".md", ".py", ".sh", ".txt", ".yaml", ".yml",
}

GENERATED_DIRECTORY_NAMES = {
	"artifacts", "captures", "reports", "runs", "visualizations", "zarr",
}

OFF_SOURCE_PARITY_PATHS = (
	"apps/DensifyPointCloud/DensifyPointCloud.cpp",
	"libs/MVS/DepthMap.cpp",
	"libs/MVS/DepthMap.h",
	"libs/MVS/PatchMatchCUDA.cpp",
	"libs/MVS/PatchMatchCUDA.cu",
	"libs/MVS/PatchMatchCUDA.inl",
	"libs/MVS/Scene.cpp",
	"libs/MVS/Scene.h",
	"libs/MVS/SceneDensify.cpp",
	"libs/MVS/SceneDensify.h",
)

PATCHMATCH_CUDA_COMPATIBILITY_PATH = "libs/MVS/PatchMatchCUDA.cu"

PATCHMATCH_CUDA_COMPATIBILITY_REPLACEMENTS = (
	(
		"""// nvcc rejects `__constant__ Camera[...]` in Debug because the Eigen-backed
// Camera type is treated as needing dynamic initialization. Keep Release on
// the direct Camera array, and use aligned byte storage only for Debug.
#if defined(_DEBUG)
struct alignas(Camera) CameraConstStorage {
	unsigned char bytes[sizeof(Camera)];
};
static_assert(sizeof(CameraConstStorage) == sizeof(Camera), "Camera constant storage must preserve Camera size");
static_assert(alignof(CameraConstStorage) == alignof(Camera), "Camera constant storage must preserve Camera alignment");
#endif""",
		"""// nvcc rejects `__constant__ Camera[...]` with newer toolchains because the
// Eigen-backed Camera type is treated as needing dynamic initialization. Keep
// constant memory as aligned byte storage and reinterpret it at use sites.
struct alignas(Camera) CameraConstStorage {
	unsigned char bytes[sizeof(Camera)];
};
static_assert(sizeof(CameraConstStorage) == sizeof(Camera), "Camera constant storage must preserve Camera size");
static_assert(alignof(CameraConstStorage) == alignof(Camera), "Camera constant storage must preserve Camera alignment");""",
	),
	(
		"""#if defined(_DEBUG)
__constant__ CameraConstStorage g_cameraStorage[MAX_VIEWS + 1];
#define g_cameras reinterpret_cast<const Camera*>(g_cameraStorage)
#else
__constant__ Camera g_cameras[MAX_VIEWS + 1];
#endif""",
		"""__constant__ CameraConstStorage g_cameraStorage[MAX_VIEWS + 1];
#define g_cameras reinterpret_cast<const Camera*>(g_cameraStorage)""",
	),
	(
		"""\t#if defined(_DEBUG)
\tCUDA_CHECK(cudaMemcpyToSymbolAsync(g_cameraStorage, cameras.data(), sizeof(Camera) * n, 0, cudaMemcpyHostToDevice, cudaStream));
\t#else
\tCUDA_CHECK(cudaMemcpyToSymbolAsync(g_cameras, cameras.data(), sizeof(Camera) * n, 0, cudaMemcpyHostToDevice, cudaStream));
\t#endif""",
		"""\tCUDA_CHECK(cudaMemcpyToSymbolAsync(g_cameraStorage, cameras.data(), sizeof(Camera) * n, 0, cudaMemcpyHostToDevice, cudaStream));""",
	),
)

PREPROCESSOR_DIRECTIVE = re.compile(
	r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$"
)

SECRET_PATTERNS = {
	"private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
	"AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
	"GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
	"OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
	"Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
	"Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
	"credential-bearing URL": re.compile(
		r"\b(?:https?|ssh|ftp)://[^\s/:]+:[^\s/@]+@[^\s/]+"
	),
}

CONCRETE_PATH_PATTERN = re.compile(
	r"(?<![A-Za-z0-9])(?:/home/[^/<>\s]+/|/Users/[^/<>\s]+/|"
	r"/(?:mnt|opt|srv|data)/[^/<>\s]+/)"
)
UUID_PATTERN = re.compile(
	r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
	r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
SSH_IDENTITY_PATTERN = re.compile(
	r"(?<![\w.-])([A-Za-z_][\w.-]*)@([A-Za-z0-9][A-Za-z0-9.-]*)"
)

# These files must contain realistic private-path samples to test the sanitizer.
SANITIZER_FIXTURE_PATHS = {
	"scripts/python/dmap_observability/portable_bundle.py",
	"scripts/python/test_dmap_dev.py",
	"scripts/python/test_dmap_portable_bundle.py",
	"scripts/python/test_dmap_public_tree.py",
	"tools/check_dmap_observability_public_tree.py",
}


def _git(repo: Path, *args: str) -> list[str]:
	result = subprocess.run(
		["git", *args], cwd=repo, check=True, text=True,
		stdout=subprocess.PIPE, stderr=subprocess.PIPE,
	)
	return [line for line in result.stdout.splitlines() if line]


def _git_text(repo: Path, revision: str, path: str) -> str:
	result = subprocess.run(
		["git", "show", f"{revision}:{path}"], cwd=repo, check=True,
		stdout=subprocess.PIPE, stderr=subprocess.PIPE,
	)
	return result.stdout.decode("utf-8")


def _git_added_text(repo: Path, base: str, path: str) -> str:
	"""Return only added lines so portability checks also cover edited base files."""
	result = subprocess.run(
		["git", "diff", "--no-ext-diff", "--no-color", "--unified=0", base, "--", path],
		cwd=repo, check=True, text=True,
		stdout=subprocess.PIPE, stderr=subprocess.PIPE,
	)
	return "\n".join(
		line[1:]
		for line in result.stdout.splitlines()
		if line.startswith("+") and not line.startswith("+++")
	)


def _is_allowed_path(path: str) -> bool:
	if path in EXACT_PATHS or path in PREFIX_SOURCE_PATHS:
		return True
	if path.startswith("scripts/python/test_dmap_") and path.endswith(".py"):
		return True
	if path.startswith("scripts/python/test_validate_dmap_") and path.endswith(".py"):
		return True
	if path == "scripts/python/test_generate_attested_dmap_report.py":
		return True
	return False


def _is_python_cache(path: str) -> bool:
	return "__pycache__" in Path(path).parts or path.endswith((".pyc", ".pyo"))


def _surface_paths(repo: Path) -> set[str]:
	paths = {
		path for path in EXACT_PATHS
		if (repo / path).exists() or (repo / path).is_symlink()
	}
	for prefix in SCANNED_PREFIXES:
		root = repo / prefix
		if not root.exists():
			continue
		for candidate in root.rglob("*"):
			if (candidate.is_file() or candidate.is_symlink()) and not _is_python_cache(candidate.as_posix()):
				paths.add(candidate.relative_to(repo).as_posix())
	for candidate in (repo / "scripts/python").glob("test_dmap_*.py"):
		paths.add(candidate.relative_to(repo).as_posix())
	for pattern in ("test_validate_dmap_*.py", "test_generate_attested_dmap_report.py"):
		for candidate in (repo / "scripts/python").glob(pattern):
			paths.add(candidate.relative_to(repo).as_posix())
	return paths


def _changed_paths(repo: Path, base: str) -> set[str]:
	# Deletions are part of the publication surface too. Omitting ``D`` would let
	# an unrelated baseline file disappear without passing through the allowlist.
	changed = set(_git(repo, "diff", "--name-only", "--diff-filter=ACDMRTUXB", base, "--"))
	changed.update(
		path for path in _git(repo, "ls-files", "--others", "--exclude-standard")
		if not _is_python_cache(path)
	)
	return changed


def _scan_text(
	path: str,
	text: str,
	forbidden: list[str],
	*,
	check_concrete_paths: bool,
) -> list[str]:
	errors: list[str] = []
	for label, pattern in SECRET_PATTERNS.items():
		if pattern.search(text):
			errors.append(f"{path}: contains a possible {label}")
	for marker in forbidden:
		if marker.casefold() in text.casefold():
			errors.append(f"{path}: contains forbidden marker {marker!r}")

	if check_concrete_paths and path not in SANITIZER_FIXTURE_PATHS:
		match = CONCRETE_PATH_PATTERN.search(text)
		if match:
			errors.append(f"{path}: contains concrete absolute path {match.group(0)!r}")

	if path.startswith(("docs/", "examples/", ".github/")) or path == "AGENTS.md":
		match = UUID_PATTERN.search(text)
		if match:
			errors.append(f"{path}: contains UUID-like dataset identity {match.group(0)!r}")
		for match in SSH_IDENTITY_PATTERN.finditer(text):
			identity = match.group(0)
			if match.group(2).casefold() not in {"example.com", "example.org", "localhost"}:
				errors.append(f"{path}: contains concrete email/SSH identity {identity!r}")
	return errors


def _without_disabled_observer_blocks(text: str, path: str) -> list[str]:
	"""Return source as seen with `_USE_DMAP_INSTRUMENTATION` undefined."""
	output: list[str] = []
	# Each entry is (observer-controlled, parent-active, branch-active).
	stack: list[tuple[bool, bool, bool]] = []
	active = True
	for line_number, line in enumerate(text.splitlines(), 1):
		match = PREPROCESSOR_DIRECTIVE.match(line)
		if match is None:
			if active and line.strip():
				output.append(line.rstrip())
			continue
		directive, expression = match.group(1), match.group(2).strip()
		if directive in {"if", "ifdef", "ifndef"}:
			controlled = "_USE_DMAP_INSTRUMENTATION" in expression
			if controlled:
				if directive == "ifndef":
					branch_active = True
				elif directive == "ifdef":
					branch_active = False
				elif expression in {
					"defined(_USE_DMAP_INSTRUMENTATION)",
					"defined _USE_DMAP_INSTRUMENTATION",
				}:
					branch_active = False
				elif expression in {
					"!defined(_USE_DMAP_INSTRUMENTATION)",
					"!defined _USE_DMAP_INSTRUMENTATION",
				}:
					branch_active = True
				else:
					raise ValueError(
						f"{path}:{line_number}: unsupported observer conditional {expression!r}"
					)
				stack.append((True, active, branch_active))
				active = active and branch_active
			else:
				if active:
					output.append(line.rstrip())
				stack.append((False, active, True))
			continue
		if not stack:
			raise ValueError(f"{path}:{line_number}: unmatched #{directive}")
		controlled, parent_active, branch_active = stack[-1]
		if directive in {"else", "elif"}:
			if controlled:
				if directive == "elif":
					raise ValueError(
						f"{path}:{line_number}: observer-controlled #elif is unsupported"
					)
				branch_active = not branch_active
				stack[-1] = (controlled, parent_active, branch_active)
				active = parent_active and branch_active
			elif active:
				output.append(line.rstrip())
			continue
		controlled, parent_active, _ = stack.pop()
		if not controlled and active:
			output.append(line.rstrip())
		active = parent_active
	if stack:
		raise ValueError(f"{path}: unterminated preprocessor conditional")
	return output


def _apply_patchmatch_cuda_compatibility(text: str) -> str:
	"""Build the one reviewed non-observer CUDA compatibility delta."""
	for before, after in PATCHMATCH_CUDA_COMPATIBILITY_REPLACEMENTS:
		count = text.count(before)
		if count != 1:
			raise ValueError(
				"PatchMatch CUDA compatibility baseline changed: "
				f"expected one exact source pattern, found {count}"
			)
		text = text.replace(before, after, 1)
	return text


def _check_off_source_parity(repo: Path, base: str) -> list[str]:
	errors: list[str] = []
	for path in OFF_SOURCE_PARITY_PATHS:
		try:
			baseline_text = _git_text(repo, base, path)
			if path == PATCHMATCH_CUDA_COMPATIBILITY_PATH:
				baseline_text = _apply_patchmatch_cuda_compatibility(baseline_text)
			baseline = _without_disabled_observer_blocks(baseline_text, path)
			candidate = _without_disabled_observer_blocks((repo / path).read_text(), path)
		except (OSError, UnicodeError, subprocess.CalledProcessError, ValueError) as error:
			errors.append(f"{path}: cannot evaluate disabled-source parity: {error}")
			continue
		if baseline == candidate:
			continue
		difference = next(
			(line for line in difflib.unified_diff(baseline, candidate, n=1)
			 if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))),
			"source differs",
		)
		errors.append(
			f"{path}: source outside disabled observer guards differs from {base}: {difference}"
		)
	return errors


def check(
	repo: Path,
	base: str,
	forbidden: list[str],
	max_bytes: int,
	*,
	check_off_source_parity: bool = True,
) -> list[str]:
	errors: list[str] = []
	for path in sorted(_changed_paths(repo, base)):
		if not _is_allowed_path(path):
			errors.append(f"{path}: changed path is outside the public allowlist")

	for path in sorted(_surface_paths(repo)):
		full_path = repo / path
		if full_path.is_symlink():
			errors.append(f"{path}: symlinks are not allowed in the public surface")
			continue
		if any(part.casefold() in GENERATED_DIRECTORY_NAMES for part in Path(path).parts):
			errors.append(f"{path}: generated-evidence directory is not allowed")
		if path not in EXACT_PATHS and full_path.suffix not in ALLOWED_SUFFIXES:
			errors.append(f"{path}: file type is not source/documentation allowlisted")
		try:
			data = full_path.read_bytes()
		except OSError as error:
			errors.append(f"{path}: cannot read file: {error}")
			continue
		if len(data) > max_bytes:
			errors.append(f"{path}: {len(data)} bytes exceeds {max_bytes}-byte source limit")
		if b"\0" in data:
			errors.append(f"{path}: contains NUL bytes")
			continue
		try:
			text = data.decode("utf-8")
		except UnicodeDecodeError as error:
			errors.append(f"{path}: is not UTF-8 text: {error}")
			continue
		base_contains_path = subprocess.run(
			["git", "cat-file", "-e", f"{base}:{path}"], cwd=repo,
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
		).returncode == 0
		errors.extend(_scan_text(
			path, text, forbidden, check_concrete_paths=not base_contains_path,
		))
		if base_contains_path and path not in SANITIZER_FIXTURE_PATHS:
			added_text = _git_added_text(repo, base, path)
			match = CONCRETE_PATH_PATTERN.search(added_text)
			if match:
				errors.append(
					f"{path}: added text contains concrete absolute path {match.group(0)!r}"
				)

	wrapper = repo / "tools/dmap_observability.sh"
	if wrapper.exists() and not os.access(wrapper, os.X_OK):
		errors.append("tools/dmap_observability.sh: wrapper is not executable")
	if check_off_source_parity:
		errors.extend(_check_off_source_parity(repo, base))
	return errors


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Validate that the DMAP observability publication surface contains only portable source and documentation."
	)
	parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
	parser.add_argument("--base", default="origin/develop", help="Git base used to enforce the changed-path allowlist")
	parser.add_argument(
		"--forbid", action="append", default=[], metavar="TEXT",
		help="additional case-insensitive organization, host, user, dataset, or path marker to reject",
	)
	parser.add_argument("--max-source-bytes", type=int, default=1024 * 1024)
	parser.add_argument(
		"--skip-off-source-parity", action="store_true",
		help="skip the lexical check that observer-disabled core source matches the base",
	)
	args = parser.parse_args()

	repo = args.repo.resolve()
	errors = check(
		repo, args.base, args.forbid, args.max_source_bytes,
		check_off_source_parity=not args.skip_off_source_parity,
	)
	if errors:
		for error in errors:
			print(f"ERROR: {error}", file=sys.stderr)
		print(f"public-tree check failed with {len(errors)} issue(s)", file=sys.stderr)
		return 1
	print(f"public-tree check passed: {_git(repo, 'rev-parse', '--short', 'HEAD')[0]}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
