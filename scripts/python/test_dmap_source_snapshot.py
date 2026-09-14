#!/usr/bin/env python3
"""Tests for deterministic DMAP source provenance snapshots."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from dmap_observability import source_snapshot


class SourceSnapshotTests(unittest.TestCase):
	def _repository(self, parent: Path) -> Path:
		repository = parent / "repository"
		repository.mkdir()
		self._git(repository, "init", "--quiet")
		self._git(repository, "config", "user.email", "tests@example.invalid")
		self._git(repository, "config", "user.name", "Snapshot Tests")
		(repository / "src").mkdir()
		(repository / "src" / "base.cpp").write_text("int value = 1;\n", encoding="utf-8")
		(repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
		self._git(repository, "add", ".gitignore", "src/base.cpp")
		self._git(repository, "commit", "--quiet", "-m", "fixture")
		return repository

	@staticmethod
	def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
		return subprocess.run(
			["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
		)

	@staticmethod
	def _members(path: Path) -> dict[str, bytes]:
		if source_snapshot.zstandard is None:
			raise unittest.SkipTest("zstandard is unavailable")
		result: dict[str, bytes] = {}
		with path.open("rb") as raw:
			with source_snapshot.zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
				with tarfile.open(fileobj=decoded, mode="r|*") as archive:
					for info in archive:
						stream = archive.extractfile(info)
						if stream is not None:
								result[info.name] = stream.read()
		return result

	@classmethod
	def _rewrite_selection_policy(cls, source: Path, destination: Path, policy: str) -> None:
		members = cls._members(source)
		manifest = json.loads(members.pop("source_snapshot.json"))
		members.pop("inventory.sha256")
		manifest["selection_policy"] = policy
		files = [
			source_snapshot._SnapshotFile(path, content)
			for path, content in members.items()
		]
		files.append(source_snapshot._SnapshotFile(
			"source_snapshot.json", source_snapshot._canonical_json(manifest),
		))
		files.append(source_snapshot._SnapshotFile(
			"inventory.sha256", source_snapshot._inventory_content(files),
		))
		source_snapshot._write_archive(
			destination, files, source_date_epoch=int(manifest["source_date_epoch"]),
		)
		digest = hashlib.sha256(destination.read_bytes()).hexdigest()
		destination.with_name(destination.name + ".sha256").write_text(
			f"{digest}  {destination.name}\n", encoding="ascii",
		)

	def test_inspect_and_default_dirty_rejection_do_not_change_worktree(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			repository = self._repository(root)
			clean = source_snapshot.inspect_git_state(repository)
			self.assertFalse(clean.dirty)
			(repository / "src" / "base.cpp").write_text("int value = 2;\n", encoding="utf-8")
			before = self._git(repository, "status", "--porcelain=v1", "-z").stdout
			state = source_snapshot.inspect_git_state(repository)
			self.assertTrue(state.dirty)
			self.assertEqual([(row.status, row.path) for row in state.tracked_changes], [("M", "src/base.cpp")])
			with self.assertRaisesRegex(source_snapshot.SourceSnapshotError, "allow_dirty"):
				source_snapshot.create_source_snapshot(repository, root / "rejected.tar.zst")
			self.assertFalse((root / "rejected.tar.zst").exists())
			self.assertEqual(before, self._git(repository, "status", "--porcelain=v1", "-z").stdout)

	def test_dirty_snapshot_is_deterministic_and_excludes_artifacts_and_ignored_files(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			repository = self._repository(root)
			(repository / "src" / "base.cpp").write_text("int value = 3;\n", encoding="utf-8")
			(repository / "scripts").mkdir()
			(repository / "scripts" / "new_tool.py").write_text("print('tool')\n", encoding="utf-8")
			(repository / "docs").mkdir()
			(repository / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
			(repository / "experiments" / "trial").mkdir(parents=True)
			(repository / "experiments" / "trial" / "config.yaml").write_text("seed: 7\n", encoding="utf-8")
			(repository / "experiments" / "trial" / "02_metrics.json").write_text("{}\n", encoding="utf-8")
			(repository / "experiments" / "trial" / "03_report.md").write_text("# Result\n", encoding="utf-8")
			(repository / "experiments" / "trial" / "03_report.yaml").write_text(
				"generated: true\n", encoding="utf-8",
			)
			benchmark_tools = repository / "tools"
			benchmark_tools.mkdir()
			master_report = benchmark_tools / "dmap_observability.sh"
			master_report.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
			master_report.chmod(0o755)
			(benchmark_tools / "generate_dmap_report.py").write_text(
				"print('report tool')\n", encoding="utf-8",
			)
			(repository / "experiments" / "trial" / "reports").mkdir()
			(repository / "experiments" / "trial" / "reports" / "tool.py").write_text(
				"generated = True\n", encoding="utf-8",
			)
			(repository / "experiments" / "trial" / "runs").mkdir()
			(repository / "experiments" / "trial" / "runs" / "helper.sh").write_text(
				"#!/usr/bin/env bash\n", encoding="utf-8",
			)
			(repository / "ignored").mkdir()
			(repository / "ignored" / "secret.py").write_text("secret = True\n", encoding="utf-8")
			(repository / "data").mkdir()
			(repository / "data" / "looks_like_source.py").write_text("data = True\n", encoding="utf-8")
			before = self._git(repository, "status", "--porcelain=v1", "-z").stdout

			first = source_snapshot.create_source_snapshot(
				repository, root / "first.tar.zst", allow_dirty=True, source_date_epoch=123
			)
			second = source_snapshot.create_source_snapshot(
				repository, root / "second.tar.zst", allow_dirty=True, source_date_epoch=123
			)
			self.assertEqual((root / "first.tar.zst").read_bytes(), (root / "second.tar.zst").read_bytes())
			self.assertEqual(before, self._git(repository, "status", "--porcelain=v1", "-z").stdout)
			validation = source_snapshot.validate_source_snapshot(first.archive_path)
			self.assertTrue(validation.dirty)
			self.assertEqual(validation.untracked_file_count, 5)
			members = self._members(first.archive_path)
			self.assertEqual(
				set(members),
				{
					"inventory.sha256", "source_snapshot.json", "tracked.patch",
					"untracked/docs/guide.md", "untracked/experiments/trial/config.yaml",
					"untracked/tools/dmap_observability.sh",
					"untracked/tools/generate_dmap_report.py",
					"untracked/scripts/new_tool.py",
				},
			)
			self.assertIn(b"+int value = 3;", members["tracked.patch"])
			manifest = json.loads(members["source_snapshot.json"])
			self.assertEqual(manifest["schema_name"], source_snapshot.SNAPSHOT_SCHEMA_NAME)
			self.assertEqual(manifest["schema_version"], 1)
			self.assertEqual(
				manifest["selection_policy"], source_snapshot.SNAPSHOT_SELECTION_POLICY,
			)
			self.assertEqual(manifest["untracked_total_count"], 11)
			self.assertEqual(manifest["excluded_untracked_count"], 6)
			master_row = next(
				row for row in manifest["untracked_files"]
				if row["path"].endswith("dmap_observability.sh")
			)
			self.assertEqual(master_row["mode"], 0o755)

	def test_rejects_output_inside_repository_symlink_and_caps_without_publishing(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			repository = self._repository(root)
			with self.assertRaisesRegex(source_snapshot.SourceSnapshotError, "outside"):
				source_snapshot.create_source_snapshot(repository, repository / "snapshot.tar.zst")
			(repository / "scripts").mkdir()
			(repository / "scripts" / "linked.py").symlink_to(repository / "src" / "base.cpp")
			with self.assertRaisesRegex(source_snapshot.SourceSnapshotError, "symlink"):
				source_snapshot.create_source_snapshot(repository, root / "link.tar.zst", allow_dirty=True)
			(repository / "scripts" / "linked.py").unlink()
			(repository / "scripts" / "large.py").write_bytes(b"x" * 4096)
			with self.assertRaisesRegex(source_snapshot.SourceSnapshotError, "size cap"):
				source_snapshot.create_source_snapshot(
					repository, root / "large.tar.zst", allow_dirty=True, max_payload_bytes=128
				)
			self.assertFalse((root / "large.tar.zst").exists())
			self.assertFalse((root / "large.tar.zst.sha256").exists())

	def test_validator_accepts_frozen_v1_selection_policy(self) -> None:
		if source_snapshot.zstandard is None:
			raise unittest.SkipTest("zstandard is unavailable")
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			repository = self._repository(root)
			(repository / "scripts").mkdir()
			(repository / "scripts" / "new.py").write_text("value = 1\n", encoding="utf-8")
			created = source_snapshot.create_source_snapshot(
				repository, root / "v2.tar.zst", allow_dirty=True,
			)
			v1_path = root / "v1.tar.zst"
			self._rewrite_selection_policy(
				created.archive_path, v1_path, "nonignored_source_config_docs_v1",
			)
			validation = source_snapshot.validate_source_snapshot(v1_path)
			self.assertEqual(validation.untracked_file_count, 1)
			manifest = json.loads(self._members(v1_path)["source_snapshot.json"])
			self.assertEqual(manifest["selection_policy"], "nonignored_source_config_docs_v1")
			unsupported_path = root / "unsupported.tar.zst"
			self._rewrite_selection_policy(
				created.archive_path, unsupported_path, "nonignored_source_config_docs_v3",
			)
			with self.assertRaisesRegex(
				source_snapshot.SourceSnapshotError, "unsupported selection policy",
			):
				source_snapshot.validate_source_snapshot(unsupported_path)

	def test_build_templates_are_source_but_generated_build_files_are_not(self) -> None:
		self.assertTrue(source_snapshot.is_relevant_untracked_path("guide.md"))
		self.assertTrue(source_snapshot.is_relevant_untracked_path("build/Templates/NewConfig.cmake.in"))
		self.assertTrue(source_snapshot.is_relevant_untracked_path("build/python/helper.py"))
		self.assertTrue(source_snapshot.is_relevant_untracked_path(
			"tools/dmap_observability.sh"
		))
		self.assertTrue(source_snapshot.is_relevant_untracked_path(
			"tools/generate_dmap_report.py"
		))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("build/CMakeCache.txt"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("build/CMakeFiles/check.py"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("reports/source.py"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("experiments/run/03_report.md"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("experiments/run/03_report.yaml"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("experiments/run/02_metrics.json"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path("experiments/run/02_metrics.yaml"))
		self.assertFalse(source_snapshot.is_relevant_untracked_path(
			"experiments/run/reports/report_tool.py"
		))
		self.assertFalse(source_snapshot.is_relevant_untracked_path(
			"experiments/run/runs/helper.sh"
		))

	def test_validator_rejects_unsafe_archive_member(self) -> None:
		if source_snapshot.zstandard is None:
			raise unittest.SkipTest("zstandard is unavailable")
		with tempfile.TemporaryDirectory() as directory:
			path = Path(directory) / "unsafe.tar.zst"
			with path.open("wb") as raw:
				compressor = source_snapshot.zstandard.ZstdCompressor(level=1)
				with compressor.stream_writer(raw, closefd=False) as encoded:
					with tarfile.open(fileobj=encoded, mode="w|") as archive:
						content = b"unsafe"
						info = tarfile.TarInfo("../escape")
						info.size = len(content)
						archive.addfile(info, io.BytesIO(content))
			with self.assertRaisesRegex(source_snapshot.SourceSnapshotError, "escapes"):
				source_snapshot.validate_source_snapshot(path, verify_checksum=False)

	def test_cli_inspect_create_and_validate(self) -> None:
		with tempfile.TemporaryDirectory() as directory:
			root = Path(directory)
			repository = self._repository(root)
			(repository / "scripts").mkdir()
			(repository / "scripts" / "new.py").write_text("value = 1\n", encoding="utf-8")
			module = "dmap_observability.source_snapshot"
			inspect = subprocess.run(
				[sys.executable, "-m", module, "inspect", "--repo", str(repository)],
				cwd=SCRIPT_DIR, capture_output=True, text=True,
			)
			self.assertEqual(inspect.returncode, 0, inspect.stderr)
			self.assertTrue(json.loads(inspect.stdout)["dirty"])
			output = root / "source.tar.zst"
			create = subprocess.run(
				[
					sys.executable, "-m", module, "create", "--repo", str(repository),
					"--output", str(output), "--allow-dirty",
				],
				cwd=SCRIPT_DIR, capture_output=True, text=True,
			)
			self.assertEqual(create.returncode, 0, create.stderr)
			self.assertEqual(json.loads(create.stdout)["archive"], str(output))
			validate = subprocess.run(
				[sys.executable, "-m", module, "validate", "--snapshot", str(output)],
				cwd=SCRIPT_DIR, capture_output=True, text=True,
			)
			self.assertEqual(validate.returncode, 0, validate.stderr)
			self.assertEqual(json.loads(validate.stdout)["untracked_file_count"], 1)


if __name__ == "__main__":
	unittest.main()
