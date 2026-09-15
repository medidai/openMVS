#!/usr/bin/env python3
"""Create and validate deterministic Git source snapshots for DMAP reports.

The snapshot is intentionally a provenance attachment, not a source checkout. It
records the exact ``HEAD`` commit, a binary-capable patch for every tracked
change, and selected nonignored untracked source/configuration/documentation
files. Generated data, builds, reports, and experiment results are never copied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

try:
	import zstandard
except ImportError:  # pragma: no cover - exercised through the public error path
	zstandard = None


SNAPSHOT_SCHEMA_NAME = "openmvs.dmap_observability.source_snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_SELECTION_POLICY = "nonignored_source_config_docs_v2"
SUPPORTED_SELECTION_POLICIES = frozenset({
	"nonignored_source_config_docs_v1",
	SNAPSHOT_SELECTION_POLICY,
})
DEFAULT_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 10_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

SOURCE_ROOTS = frozenset({
	"apps", "cmake", "configs", "docker", "include", "libs", "ports", "scripts",
	"src", "test", "tests", "tools",
})
SOURCE_SUFFIXES = frozenset({
	".bash", ".c", ".cc", ".cfg", ".cmake", ".conf", ".cpp", ".css", ".cu",
	".cuh", ".cxx", ".h", ".hh", ".hpp", ".htm", ".html", ".hxx", ".in",
	".ini", ".inl", ".ipp", ".js", ".json", ".mjs", ".py", ".pyi", ".rst",
	".scss", ".sh", ".sql", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
	".zsh",
})
DOC_SUFFIXES = frozenset({
	".adoc", ".gif", ".jpeg", ".jpg", ".md", ".png", ".rst", ".svg", ".tex",
	".webp",
})
EXPERIMENT_SUFFIXES = frozenset({
	".cfg", ".ini", ".py", ".sh", ".toml", ".yaml", ".yml",
})
EXPERIMENT_CODE_SUFFIXES = frozenset({".py", ".sh"})
ROOT_SOURCE_NAMES = frozenset({
	".clang-format", ".clang-tidy", ".editorconfig", ".gitignore", ".gitmodules",
	"AGENTS.md", "CMakeLists.txt", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "LICENSE",
	"LICENSE.md", "Makefile", "README", "README.md", "WORKSPACE.md", "guide.md", "vcpkg.json",
	"vcpkg-configuration.json",
})
EXCLUDED_COMPONENTS = frozenset({
	".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__",
	"artifacts", "cache", "data", "datasets", "logs", "node_modules", "output",
	"outputs", "report", "reports", "results", "runs", "temp", "tmp", "venv",
	"visualizations",
})
EXCLUDED_ROOT_PATTERNS = (
	re.compile(r"^(?:bin|make)(?:$|[-_].*)", re.IGNORECASE),
	re.compile(r"^(?:bench|build|cmake-build)(?:[-_].*)$", re.IGNORECASE),
	re.compile(r"^texture-runs", re.IGNORECASE),
)
EXPERIMENT_ARTIFACT_RE = re.compile(
	r"(?:^|[_-])(?:metrics?|report|results?|stdout|stderr|summary)(?:[_-]|\.|$)",
	re.IGNORECASE,
)
BUILD_SOURCE_SUBTREES = frozenset({"Modules", "Templates", "python"})
SPECIAL_SOURCE_NAMES = frozenset({"Dockerfile", "Makefile"})


class SourceSnapshotError(RuntimeError):
	"""Raised when source provenance cannot be captured or validated."""


@dataclass(frozen=True)
class TrackedChange:
	status: str
	path: str


@dataclass(frozen=True)
class GitState:
	repository_root: Path
	commit: str
	branch: str | None
	tracked_changes: tuple[TrackedChange, ...]
	untracked_paths: tuple[str, ...]

	@property
	def dirty(self) -> bool:
		return bool(self.tracked_changes or self.untracked_paths)


@dataclass(frozen=True)
class SourceSnapshotBuildResult:
	archive_path: Path
	checksum_path: Path
	archive_sha256: str
	archive_bytes: int
	source_payload_bytes: int
	tracked_change_count: int
	untracked_file_count: int
	commit: str
	dirty: bool


@dataclass(frozen=True)
class SourceSnapshotValidationResult:
	archive_path: Path
	archive_sha256: str
	archive_bytes: int
	source_payload_bytes: int
	tracked_change_count: int
	untracked_file_count: int
	commit: str
	dirty: bool


@dataclass(frozen=True)
class _SnapshotFile:
	archive_path: str
	content: bytes
	mode: int = 0o644

	@property
	def sha256(self) -> str:
		return hashlib.sha256(self.content).hexdigest()


def _canonical_json(value: Any) -> bytes:
	return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _sha256_path(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as stream:
		for block in iter(lambda: stream.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _safe_relative_path(raw_path: str) -> str:
	if not raw_path or "\\" in raw_path or "\x00" in raw_path:
		raise SourceSnapshotError(f"unsafe repository path: {raw_path!r}")
	if any(ord(character) < 32 for character in raw_path):
		raise SourceSnapshotError(f"repository path contains a control character: {raw_path!r}")
	path = PurePosixPath(raw_path)
	if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
		raise SourceSnapshotError(f"repository path escapes its root: {raw_path!r}")
	if path.as_posix() != raw_path:
		raise SourceSnapshotError(f"repository path is not normalized: {raw_path!r}")
	return raw_path


def _decode_git_path(value: bytes) -> str:
	try:
		return _safe_relative_path(value.decode("utf-8", errors="strict"))
	except UnicodeDecodeError as error:
		raise SourceSnapshotError("Git path is not valid UTF-8") from error


def _git(
	repository: Path,
	arguments: Sequence[str],
	*,
	check: bool = True,
) -> bytes:
	environment = os.environ.copy()
	environment.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
	process = subprocess.run(
		["git", *arguments],
		cwd=repository,
		env=environment,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		check=False,
	)
	if check and process.returncode != 0:
		message = process.stderr.decode("utf-8", errors="replace").strip()
		raise SourceSnapshotError(f"git {' '.join(arguments)} failed: {message}")
	return process.stdout


def _repository_root(repository: Path) -> Path:
	repository = Path(repository).resolve()
	root = _git(repository, ["rev-parse", "--show-toplevel"])
	try:
		path = Path(root.decode("utf-8", errors="strict").strip()).resolve()
	except UnicodeDecodeError as error:
		raise SourceSnapshotError("repository root is not valid UTF-8") from error
	if not path.is_dir():
		raise SourceSnapshotError(f"Git repository root does not exist: {path}")
	return path


def _split_nul(value: bytes) -> list[bytes]:
	if not value:
		return []
	if not value.endswith(b"\0"):
		raise SourceSnapshotError("Git returned a malformed NUL-delimited path list")
	return value[:-1].split(b"\0")


def inspect_git_state(repository: Path = Path(".")) -> GitState:
	"""Inspect ``HEAD``, tracked changes, and all nonignored untracked paths."""
	root = _repository_root(repository)
	commit_raw = _git(root, ["rev-parse", "--verify", "HEAD"])
	commit = commit_raw.decode("ascii", errors="strict").strip().lower()
	if not SHA256_RE.fullmatch(commit) and not re.fullmatch(r"[0-9a-f]{40}", commit):
		raise SourceSnapshotError(f"Git returned an invalid commit id: {commit!r}")
	branch_raw = _git(root, ["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
	branch = branch_raw.decode("utf-8", errors="strict").strip() or None

	change_tokens = _split_nul(_git(root, [
		"diff", "--name-status", "--no-renames", "-z", "HEAD", "--",
	]))
	if len(change_tokens) % 2:
		raise SourceSnapshotError("Git returned a malformed tracked-change list")
	tracked_changes: list[TrackedChange] = []
	for index in range(0, len(change_tokens), 2):
		status = change_tokens[index].decode("ascii", errors="strict")
		if not re.fullmatch(r"[A-Z][0-9]*", status):
			raise SourceSnapshotError(f"Git returned an invalid change status: {status!r}")
		tracked_changes.append(TrackedChange(status=status, path=_decode_git_path(change_tokens[index + 1])))

	untracked_paths = tuple(sorted(
		_decode_git_path(path)
		for path in _split_nul(_git(root, [
			"ls-files", "--others", "--exclude-standard", "-z", "--",
		]))
	))
	return GitState(
		repository_root=root,
		commit=commit,
		branch=branch,
		tracked_changes=tuple(sorted(tracked_changes, key=lambda item: (item.path, item.status))),
		untracked_paths=untracked_paths,
	)


def is_relevant_untracked_path(path: str | PurePosixPath) -> bool:
	"""Return whether a nonignored untracked path belongs in source provenance."""
	raw_path = _safe_relative_path(PurePosixPath(path).as_posix())
	parts = PurePosixPath(raw_path).parts
	if not parts or any(part.lower() in EXCLUDED_COMPONENTS for part in parts):
		return False
	if parts[0].startswith(".") and parts[0] != ".github":
		return False
	if any(pattern.fullmatch(parts[0]) for pattern in EXCLUDED_ROOT_PATTERNS):
		return False
	file_name = parts[-1]
	suffix = PurePosixPath(file_name).suffix.lower()
	if len(parts) == 1:
		return file_name in ROOT_SOURCE_NAMES or suffix in {".cmake"}
	if parts[0] == "docs":
		return suffix in DOC_SUFFIXES
	if parts[0] == ".github":
		return suffix in SOURCE_SUFFIXES or suffix in DOC_SUFFIXES
	if parts[0] == "experiments":
		return (
			suffix in EXPERIMENT_CODE_SUFFIXES
			or (
				suffix in EXPERIMENT_SUFFIXES
				and not EXPERIMENT_ARTIFACT_RE.search(file_name)
			)
		)
	if parts[0] == "build":
		return (
			(len(parts) >= 3 and parts[1] in BUILD_SOURCE_SUBTREES)
			or (len(parts) == 2 and file_name == "Utils.cmake")
		) and (suffix in SOURCE_SUFFIXES or file_name in SPECIAL_SOURCE_NAMES)
	return parts[0] in SOURCE_ROOTS and (
		suffix in SOURCE_SUFFIXES or file_name in SPECIAL_SOURCE_NAMES
	)


def _tracked_patch(root: Path) -> bytes:
	return _git(root, [
		"diff", "--binary", "--full-index", "--no-color", "--no-ext-diff", "--no-textconv",
		"--no-renames", "--diff-algorithm=myers", "--src-prefix=a/", "--dst-prefix=b/",
		"--submodule=short", "-O/dev/null", "HEAD", "--",
	])


def _path_is_within(path: Path, root: Path) -> bool:
	try:
		path.resolve(strict=False).relative_to(root.resolve())
		return True
	except ValueError:
		return False


def _read_stable_file(path: Path, *, remaining_bytes: int) -> tuple[bytes, int]:
	if path.is_symlink():
		raise SourceSnapshotError(f"relevant untracked source is a symlink: {path}")
	before = path.stat()
	if not stat.S_ISREG(before.st_mode):
		raise SourceSnapshotError(f"relevant untracked source is not a regular file: {path}")
	if before.st_size > remaining_bytes:
		raise SourceSnapshotError(
			f"source snapshot payload exceeds its size cap while reading {path.name!r}"
		)
	content = path.read_bytes()
	after = path.stat()
	if (
		before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
	) != (
		after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
	) or len(content) != before.st_size:
		raise SourceSnapshotError(f"source changed while it was being captured: {path}")
	mode = 0o755 if before.st_mode & 0o111 else 0o644
	return content, mode


def _tar_info(path: str, size: int, mode: int, source_date_epoch: int) -> tarfile.TarInfo:
	info = tarfile.TarInfo(path)
	info.size = size
	info.mode = mode
	info.mtime = source_date_epoch
	info.uid = 0
	info.gid = 0
	info.uname = ""
	info.gname = ""
	return info


def _write_archive(
	path: Path,
	files: Sequence[_SnapshotFile],
	*,
	source_date_epoch: int,
) -> None:
	if zstandard is None:
		raise SourceSnapshotError("zstandard is required to create source snapshots")
	compressor = zstandard.ZstdCompressor(
		level=19,
		threads=0,
		write_checksum=True,
		write_content_size=False,
	)
	with path.open("wb") as raw_stream:
		with compressor.stream_writer(raw_stream, closefd=False) as compressed_stream:
			with tarfile.open(fileobj=compressed_stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
				for item in sorted(files, key=lambda value: value.archive_path):
					info = _tar_info(item.archive_path, len(item.content), item.mode, source_date_epoch)
					archive.addfile(info, io.BytesIO(item.content))


def _inventory_content(files: Sequence[_SnapshotFile]) -> bytes:
	return "".join(
		f"{item.sha256}  {item.archive_path}\n"
		for item in sorted(files, key=lambda value: value.archive_path)
	).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
	fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
	temp_path = Path(raw_temp)
	try:
		with os.fdopen(fd, "wb") as stream:
			stream.write(content)
			stream.flush()
			os.fsync(stream.fileno())
		os.replace(temp_path, path)
	finally:
		if temp_path.exists():
			temp_path.unlink()


def _state_identity(state: GitState) -> tuple[Any, ...]:
	return (
		state.commit,
		state.branch,
		tuple((change.status, change.path) for change in state.tracked_changes),
		state.untracked_paths,
	)


def create_source_snapshot(
	repository: Path,
	output_path: Path,
	*,
	allow_dirty: bool = False,
	max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
	max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
	max_members: int = DEFAULT_MAX_MEMBERS,
	source_date_epoch: int = 0,
	overwrite: bool = False,
) -> SourceSnapshotBuildResult:
	"""Create, self-validate, and atomically publish a source snapshot.

	Dirty repositories are rejected unless ``allow_dirty`` is explicit. The
	archive and its checksum must be written outside the repository so capturing
	provenance never changes the worktree being described.
	"""
	if max_archive_bytes <= 0 or max_payload_bytes <= 0 or max_members <= 0:
		raise SourceSnapshotError("snapshot size and member limits must be positive")
	if source_date_epoch < 0:
		raise SourceSnapshotError("source_date_epoch must be non-negative")
	state = inspect_git_state(repository)
	output_path = Path(output_path).resolve()
	checksum_path = output_path.with_name(output_path.name + ".sha256")
	if _path_is_within(output_path, state.repository_root):
		raise SourceSnapshotError("source snapshot output must be outside the Git repository")
	if not overwrite and (output_path.exists() or checksum_path.exists()):
		raise SourceSnapshotError(f"refusing to overwrite an existing source snapshot: {output_path}")
	if state.dirty and not allow_dirty:
		raise SourceSnapshotError(
			"Git worktree is dirty; pass allow_dirty=True to capture an explicit source snapshot"
		)

	patch = _tracked_patch(state.repository_root)
	if len(patch) > max_payload_bytes:
		raise SourceSnapshotError("tracked source patch exceeds the snapshot payload size cap")
	files: list[_SnapshotFile] = [_SnapshotFile("tracked.patch", patch)]
	untracked_manifest: list[dict[str, Any]] = []
	source_payload_bytes = len(patch)
	for relative in state.untracked_paths:
		if not is_relevant_untracked_path(relative):
			continue
		content, mode = _read_stable_file(
			state.repository_root / relative,
			remaining_bytes=max_payload_bytes - source_payload_bytes,
		)
		archive_path = _safe_relative_path(f"untracked/{relative}")
		item = _SnapshotFile(archive_path, content, mode)
		files.append(item)
		source_payload_bytes += len(content)
		untracked_manifest.append({
			"archive_path": archive_path,
			"mode": mode,
			"path": relative,
			"sha256": item.sha256,
			"size_bytes": len(content),
		})

	if len(files) + 2 > max_members:
		raise SourceSnapshotError(
			f"source snapshot requires {len(files) + 2} members, exceeding {max_members}"
		)
	second_state = inspect_git_state(state.repository_root)
	second_patch = _tracked_patch(state.repository_root)
	if _state_identity(second_state) != _state_identity(state) or second_patch != patch:
		raise SourceSnapshotError("Git worktree changed while source provenance was being captured")
	for row in untracked_manifest:
		if _sha256_path(state.repository_root / row["path"]) != row["sha256"]:
			raise SourceSnapshotError(
				f"untracked source changed while provenance was being captured: {row['path']}"
			)

	manifest = {
		"archive_format": "tar+zstd",
		"branch": state.branch,
		"commit": state.commit,
		"dirty": state.dirty,
		"excluded_untracked_count": len(state.untracked_paths) - len(untracked_manifest),
		"schema_name": SNAPSHOT_SCHEMA_NAME,
		"schema_version": SNAPSHOT_SCHEMA_VERSION,
		"selection_policy": SNAPSHOT_SELECTION_POLICY,
		"source_date_epoch": source_date_epoch,
		"source_payload_bytes": source_payload_bytes,
		"tracked_change_count": len(state.tracked_changes),
		"tracked_changes": [
			{"path": change.path, "status": change.status} for change in state.tracked_changes
		],
		"tracked_patch": {
			"archive_path": "tracked.patch",
			"sha256": files[0].sha256,
			"size_bytes": len(patch),
		},
		"untracked_file_count": len(untracked_manifest),
		"untracked_files": untracked_manifest,
		"untracked_total_count": len(state.untracked_paths),
	}
	manifest_file = _SnapshotFile("source_snapshot.json", _canonical_json(manifest))
	files.append(manifest_file)
	inventory_file = _SnapshotFile("inventory.sha256", _inventory_content(files))
	files.append(inventory_file)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fd, raw_temp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
	os.close(fd)
	temp_path = Path(raw_temp)
	try:
		_write_archive(temp_path, files, source_date_epoch=source_date_epoch)
		archive_bytes = temp_path.stat().st_size
		if archive_bytes > max_archive_bytes:
			raise SourceSnapshotError(
				f"compressed source snapshot requires {archive_bytes} bytes, exceeding {max_archive_bytes}"
			)
		validation = validate_source_snapshot(
			temp_path,
			verify_checksum=False,
			max_archive_bytes=max_archive_bytes,
			max_payload_bytes=max_payload_bytes,
			max_members=max_members,
		)
		archive_sha256 = validation.archive_sha256
		checksum_content = f"{archive_sha256}  {output_path.name}\n".encode("ascii")
		os.replace(temp_path, output_path)
		_atomic_write(checksum_path, checksum_content)
		return SourceSnapshotBuildResult(
			archive_path=output_path,
			checksum_path=checksum_path,
			archive_sha256=archive_sha256,
			archive_bytes=archive_bytes,
			source_payload_bytes=source_payload_bytes,
			tracked_change_count=len(state.tracked_changes),
			untracked_file_count=len(untracked_manifest),
			commit=state.commit,
			dirty=state.dirty,
		)
	finally:
		if temp_path.exists():
			temp_path.unlink()


def _read_member(stream: BinaryIO, size: int) -> bytes:
	remaining = size
	blocks: list[bytes] = []
	while remaining:
		block = stream.read(min(remaining, 1024 * 1024))
		if not block:
			raise SourceSnapshotError("source snapshot member ended unexpectedly")
		blocks.append(block)
		remaining -= len(block)
	return b"".join(blocks)


def _parse_inventory(content: bytes) -> dict[str, str]:
	try:
		lines = content.decode("utf-8", errors="strict").splitlines()
	except UnicodeDecodeError as error:
		raise SourceSnapshotError("inventory.sha256 is not valid UTF-8") from error
	entries: dict[str, str] = {}
	for line in lines:
		match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
		if not match:
			raise SourceSnapshotError("inventory.sha256 contains a malformed entry")
		path = _safe_relative_path(match.group(2))
		if path in entries:
			raise SourceSnapshotError(f"duplicate inventory path: {path}")
		entries[path] = match.group(1)
	return entries


def _validate_manifest(
	manifest: Mapping[str, Any],
	members: Mapping[str, bytes],
	member_modes: Mapping[str, int],
) -> None:
	if (
		manifest.get("schema_name") != SNAPSHOT_SCHEMA_NAME
		or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
	):
		raise SourceSnapshotError("source_snapshot.json has an unsupported schema")
	if manifest.get("selection_policy") not in SUPPORTED_SELECTION_POLICIES:
		raise SourceSnapshotError("source_snapshot.json has an unsupported selection policy")
	commit = manifest.get("commit")
	if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
		raise SourceSnapshotError("source_snapshot.json has an invalid commit id")
	for key in (
		"source_payload_bytes", "tracked_change_count", "untracked_file_count",
		"untracked_total_count", "excluded_untracked_count", "source_date_epoch",
	):
		if not isinstance(manifest.get(key), int) or manifest[key] < 0:
			raise SourceSnapshotError(f"source_snapshot.json has an invalid {key}")
	if not isinstance(manifest.get("dirty"), bool):
		raise SourceSnapshotError("source_snapshot.json has an invalid dirty flag")
	tracked_changes = manifest.get("tracked_changes")
	if not isinstance(tracked_changes, list) or len(tracked_changes) != manifest["tracked_change_count"]:
		raise SourceSnapshotError("tracked change count does not match source_snapshot.json")
	for row in tracked_changes:
		if not isinstance(row, dict) or not isinstance(row.get("status"), str):
			raise SourceSnapshotError("source_snapshot.json has an invalid tracked change")
		_safe_relative_path(str(row.get("path", "")))
	patch = manifest.get("tracked_patch")
	if not isinstance(patch, dict) or patch.get("archive_path") != "tracked.patch":
		raise SourceSnapshotError("source_snapshot.json has an invalid tracked patch entry")
	if "tracked.patch" not in members:
		raise SourceSnapshotError("source snapshot is missing tracked.patch")
	if patch.get("size_bytes") != len(members["tracked.patch"]):
		raise SourceSnapshotError("tracked patch size does not match source_snapshot.json")
	if patch.get("sha256") != hashlib.sha256(members["tracked.patch"]).hexdigest():
		raise SourceSnapshotError("tracked patch hash does not match source_snapshot.json")

	untracked = manifest.get("untracked_files")
	if not isinstance(untracked, list) or len(untracked) != manifest["untracked_file_count"]:
		raise SourceSnapshotError("untracked file count does not match source_snapshot.json")
	expected_members = {"tracked.patch", "source_snapshot.json", "inventory.sha256"}
	source_payload_bytes = len(members["tracked.patch"])
	seen_paths: set[str] = set()
	for row in untracked:
		if not isinstance(row, dict):
			raise SourceSnapshotError("source_snapshot.json has an invalid untracked file")
		source_path = _safe_relative_path(str(row.get("path", "")))
		archive_path = _safe_relative_path(str(row.get("archive_path", "")))
		if archive_path != f"untracked/{source_path}":
			raise SourceSnapshotError("untracked archive path does not match its source path")
		if source_path in seen_paths or archive_path not in members:
			raise SourceSnapshotError("source snapshot has a duplicate or missing untracked file")
		seen_paths.add(source_path)
		content = members[archive_path]
		if row.get("size_bytes") != len(content):
			raise SourceSnapshotError(f"untracked size does not match for {source_path}")
		if row.get("sha256") != hashlib.sha256(content).hexdigest():
			raise SourceSnapshotError(f"untracked hash does not match for {source_path}")
		if row.get("mode") not in (0o644, 0o755):
			raise SourceSnapshotError(f"untracked mode is invalid for {source_path}")
		if member_modes.get(archive_path) != row["mode"]:
			raise SourceSnapshotError(f"archive mode does not match for {source_path}")
		source_payload_bytes += len(content)
		expected_members.add(archive_path)
	if set(members) != expected_members:
		raise SourceSnapshotError("source snapshot contains undeclared members")
	if source_payload_bytes != manifest["source_payload_bytes"]:
		raise SourceSnapshotError("source payload size does not match source_snapshot.json")
	if manifest["untracked_total_count"] != (
		manifest["untracked_file_count"] + manifest["excluded_untracked_count"]
	):
		raise SourceSnapshotError("untracked totals do not match source_snapshot.json")
	if manifest["dirty"] != bool(
		manifest["tracked_change_count"] or manifest["untracked_total_count"]
	):
		raise SourceSnapshotError("dirty flag does not match source_snapshot.json counts")


def validate_source_snapshot(
	archive_path: Path,
	*,
	verify_checksum: bool = True,
	max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
	max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
	max_members: int = DEFAULT_MAX_MEMBERS,
) -> SourceSnapshotValidationResult:
	"""Validate archive paths, caps, inventory, manifest, payload hashes, and checksum."""
	if zstandard is None:
		raise SourceSnapshotError("zstandard is required to validate source snapshots")
	archive_path = Path(archive_path).resolve()
	if not archive_path.is_file() or archive_path.is_symlink():
		raise SourceSnapshotError(f"source snapshot is not a regular file: {archive_path}")
	archive_bytes = archive_path.stat().st_size
	if archive_bytes > max_archive_bytes:
		raise SourceSnapshotError(
			f"compressed source snapshot requires {archive_bytes} bytes, exceeding {max_archive_bytes}"
		)
	archive_sha256 = _sha256_path(archive_path)
	if verify_checksum:
		checksum_path = archive_path.with_name(archive_path.name + ".sha256")
		if not checksum_path.is_file() or checksum_path.is_symlink():
			raise SourceSnapshotError(f"source snapshot checksum is missing: {checksum_path}")
		checksum = checksum_path.read_text(encoding="ascii").strip()
		if checksum != f"{archive_sha256}  {archive_path.name}":
			raise SourceSnapshotError("source snapshot checksum does not match")

	members: dict[str, bytes] = {}
	member_modes: dict[str, int] = {}
	payload_bytes = 0
	try:
		with archive_path.open("rb") as raw_stream:
			with zstandard.ZstdDecompressor().stream_reader(raw_stream, closefd=False) as decoded:
				with tarfile.open(fileobj=decoded, mode="r|*") as archive:
					for info in archive:
						path = _safe_relative_path(info.name)
						if path in members:
							raise SourceSnapshotError(f"duplicate source snapshot member: {path}")
						if not info.isfile() or info.issym() or info.islnk():
							raise SourceSnapshotError(f"non-regular source snapshot member: {path}")
						if info.uid != 0 or info.gid != 0 or info.mode not in (0o644, 0o755):
							raise SourceSnapshotError(f"noncanonical source snapshot metadata: {path}")
						if info.size < 0 or payload_bytes + info.size > max_payload_bytes + 8 * 1024 * 1024:
							raise SourceSnapshotError("source snapshot expanded payload exceeds its size cap")
						if len(members) + 1 > max_members:
							raise SourceSnapshotError("source snapshot member count exceeds its cap")
						stream = archive.extractfile(info)
						if stream is None:
							raise SourceSnapshotError(f"cannot read source snapshot member: {path}")
						members[path] = _read_member(stream, info.size)
						member_modes[path] = info.mode
						payload_bytes += info.size
	except SourceSnapshotError:
		raise
	except (OSError, tarfile.TarError, zstandard.ZstdError) as error:
		raise SourceSnapshotError(f"cannot read source snapshot: {error}") from error

	if "inventory.sha256" not in members or "source_snapshot.json" not in members:
		raise SourceSnapshotError("source snapshot is missing its inventory or manifest")
	inventory = _parse_inventory(members["inventory.sha256"])
	expected_inventory_paths = set(members) - {"inventory.sha256"}
	if set(inventory) != expected_inventory_paths:
		raise SourceSnapshotError("source snapshot inventory paths do not match archive members")
	for path, expected_sha256 in inventory.items():
		if hashlib.sha256(members[path]).hexdigest() != expected_sha256:
			raise SourceSnapshotError(f"source snapshot inventory hash mismatch: {path}")
	try:
		manifest = json.loads(members["source_snapshot.json"].decode("utf-8", errors="strict"))
	except (UnicodeDecodeError, json.JSONDecodeError) as error:
		raise SourceSnapshotError("source_snapshot.json is not valid JSON") from error
	if not isinstance(manifest, dict):
		raise SourceSnapshotError("source_snapshot.json must contain an object")
	_validate_manifest(manifest, members, member_modes)
	if manifest["source_payload_bytes"] > max_payload_bytes:
		raise SourceSnapshotError("source snapshot source payload exceeds its size cap")
	return SourceSnapshotValidationResult(
		archive_path=archive_path,
		archive_sha256=archive_sha256,
		archive_bytes=archive_bytes,
		source_payload_bytes=manifest["source_payload_bytes"],
		tracked_change_count=manifest["tracked_change_count"],
		untracked_file_count=manifest["untracked_file_count"],
		commit=manifest["commit"],
		dirty=manifest["dirty"],
	)


def _result_json(value: GitState | SourceSnapshotBuildResult | SourceSnapshotValidationResult) -> str:
	if isinstance(value, GitState):
		result = {
			"branch": value.branch,
			"commit": value.commit,
			"dirty": value.dirty,
			"repository_root": str(value.repository_root),
			"tracked_change_count": len(value.tracked_changes),
			"untracked_count": len(value.untracked_paths),
		}
	else:
		result = {
			"archive": str(value.archive_path),
			"archive_bytes": value.archive_bytes,
			"archive_sha256": value.archive_sha256,
			"commit": value.commit,
			"dirty": value.dirty,
			"source_payload_bytes": value.source_payload_bytes,
			"tracked_change_count": value.tracked_change_count,
			"untracked_file_count": value.untracked_file_count,
		}
		if isinstance(value, SourceSnapshotBuildResult):
			result["checksum"] = str(value.checksum_path)
	return json.dumps(result, indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	inspect_parser = subparsers.add_parser("inspect", help="inspect Git source state")
	inspect_parser.add_argument("--repo", type=Path, default=Path("."))

	create_parser = subparsers.add_parser("create", help="create a deterministic source snapshot")
	create_parser.add_argument("--repo", type=Path, default=Path("."))
	create_parser.add_argument("--output", type=Path, required=True)
	create_parser.add_argument("--allow-dirty", action="store_true")
	create_parser.add_argument("--overwrite", action="store_true")
	create_parser.add_argument("--max-archive-bytes", type=int, default=DEFAULT_MAX_ARCHIVE_BYTES)
	create_parser.add_argument("--max-payload-bytes", type=int, default=DEFAULT_MAX_PAYLOAD_BYTES)
	create_parser.add_argument("--source-date-epoch", type=int, default=0)

	validate_parser = subparsers.add_parser("validate", help="validate a source snapshot")
	validate_parser.add_argument("--snapshot", type=Path, required=True)
	validate_parser.add_argument("--no-checksum", action="store_true")
	validate_parser.add_argument("--max-archive-bytes", type=int, default=DEFAULT_MAX_ARCHIVE_BYTES)
	validate_parser.add_argument("--max-payload-bytes", type=int, default=DEFAULT_MAX_PAYLOAD_BYTES)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = _build_parser().parse_args(argv)
	try:
		if args.command == "inspect":
			result = inspect_git_state(args.repo)
		elif args.command == "create":
			result = create_source_snapshot(
				args.repo,
				args.output,
				allow_dirty=args.allow_dirty,
				max_archive_bytes=args.max_archive_bytes,
				max_payload_bytes=args.max_payload_bytes,
				source_date_epoch=args.source_date_epoch,
				overwrite=args.overwrite,
			)
		else:
			result = validate_source_snapshot(
				args.snapshot,
				verify_checksum=not args.no_checksum,
				max_archive_bytes=args.max_archive_bytes,
				max_payload_bytes=args.max_payload_bytes,
			)
	except SourceSnapshotError as error:
		raise SystemExit(f"error: {error}") from error
	print(_result_json(result), end="")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
