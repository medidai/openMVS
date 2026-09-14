"""Exact artifact-closure manifests for depth-map captures and reports.

The manifests are deterministic inventories, not signatures.  They detect drift,
truncation, same-size edits, mode changes, and unexpected files between producer
and consumer boundaries.  Authenticity still requires a separately trusted
signature or content-addressed publication channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence


CAPTURE_CLOSURE_FILE = "capture_artifact_closure.json"
CAPTURE_CLOSURE_SCHEMA_NAME = "openmvs.dmap.capture_artifact_closure"
CAPTURE_CLOSURE_SCHEMA_VERSION = 1
REPORT_CLOSURE_FILE = "report_tree_closure.json"
REPORT_CLOSURE_SCHEMA_NAME = "openmvs.dmap.report_tree_closure"
REPORT_CLOSURE_SCHEMA_VERSION = 1
REPORT_POLICY_CLOSURE_SCHEMA_VERSION = 2
HASH_ALGORITHM = "sha256"

MAX_MANIFEST_BYTES = 128 * 1024 * 1024
MAX_CAPTURE_FILES = 1_000_000
MAX_CAPTURE_BYTES = 4 * 1024**4
MAX_REPORT_FILES = 250_000
MAX_REPORT_BYTES = 1024**4

CAPTURE_EXCLUSIONS = (
    {
        "path": CAPTURE_CLOSURE_FILE,
        "scope": "exact",
        "reason": "closure manifest cannot include its own digest",
    },
    {
        "path": "work/",
        "scope": "prefix",
        "reason": (
            "runtime staging and duplicate process outputs; terminal DMAPs, logs, "
            "and instrumentation are retained at the capture root"
        ),
    },
)
REPORT_EXCLUSIONS = (
    {
        "path": REPORT_CLOSURE_FILE,
        "scope": "exact",
        "reason": "closure manifest cannot include its own digest",
    },
    {
        "path": "01_development_report.validation.json",
        "scope": "exact",
        "reason": "mutable validator sidecar regenerated after closure verification",
    },
)

_COMMON_FIELDS = frozenset({
    "schema_name",
    "schema_version",
    "root",
    "hash_algorithm",
    "exclusions",
    "files",
    "file_count",
    "total_bytes",
    "files_sha256",
})
_FILE_FIELDS = frozenset({"path", "sha256", "bytes", "mode"})
_EXCLUSION_FIELDS = frozenset({"path", "scope", "reason"})


class IntegrityError(RuntimeError):
    """Raised when a tree cannot be safely inventoried or validated."""


@dataclass(frozen=True)
class ClosureValidation:
    valid: bool
    status: str
    required: bool
    reason: str
    manifest_path: str
    file_count: int | None = None
    total_bytes: int | None = None
    files_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _status_tuple(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _hash_regular_file(path: Path) -> dict[str, Any]:
    """Hash one file through a no-follow descriptor and reject concurrent drift."""

    try:
        before_path = path.lstat()
    except OSError as error:
        raise IntegrityError(f"cannot stat artifact {path}: {error}") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise IntegrityError(f"artifact is not a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrityError(f"cannot safely open artifact {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(descriptor)
        if _status_tuple(opened_before) != _status_tuple(before_path):
            raise IntegrityError(f"artifact changed while it was opened: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise IntegrityError(f"cannot hash artifact {path}: {error}") from error
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as error:
        raise IntegrityError(f"artifact disappeared while hashing: {path}") from error
    if not (
        _status_tuple(opened_before)
        == _status_tuple(opened_after)
        == _status_tuple(after_path)
    ):
        raise IntegrityError(f"artifact changed while hashing: {path}")
    return {
        "sha256": digest.hexdigest(),
        "bytes": opened_after.st_size,
        "mode": stat.S_IMODE(opened_after.st_mode),
    }


def regular_file_identity(path: Path) -> dict[str, Any]:
    """Return a race-checked identity for one regular non-symlink file."""

    candidate = path.expanduser().absolute()
    return {"path": str(candidate), **_hash_regular_file(candidate)}


def _normalized_exclusions(
    exclusions: Sequence[Mapping[str, str]],
) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    prefixes: set[str] = set()
    for row in exclusions:
        if set(row) != _EXCLUSION_FIELDS:
            raise IntegrityError("closure exclusion declaration has invalid fields")
        raw_path = row.get("path")
        scope = row.get("scope")
        reason = row.get("reason")
        if not isinstance(raw_path, str) or not raw_path or not isinstance(reason, str) or not reason:
            raise IntegrityError("closure exclusion declaration is malformed")
        normalized = raw_path[:-1] if raw_path.endswith("/") else raw_path
        candidate = Path(normalized)
        if (
            not normalized
            or candidate.is_absolute()
            or "\\" in normalized
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise IntegrityError(f"unsafe closure exclusion path: {raw_path!r}")
        if scope == "exact":
            exact.add(normalized)
        elif scope == "prefix" and raw_path.endswith("/"):
            prefixes.add(normalized)
        else:
            raise IntegrityError(f"invalid closure exclusion scope: {raw_path!r}")
    if exact & prefixes:
        raise IntegrityError("closure exclusions repeat an exact and prefix path")
    return exact, prefixes


def _scan_tree(
    root: Path,
    exclusions: Sequence[Mapping[str, str]],
    *,
    max_files: int,
    max_bytes: int,
) -> list[dict[str, Any]]:
    directory = root.expanduser().absolute()
    try:
        root_status = directory.lstat()
    except OSError as error:
        raise IntegrityError(f"closure root is unavailable: {directory}: {error}") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise IntegrityError(f"closure root must be a non-symlink directory: {directory}")
    exact, prefixes = _normalized_exclusions(exclusions)
    rows: list[dict[str, Any]] = []
    total_bytes = 0

    def visit(current: Path, relative_parent: str) -> None:
        nonlocal total_bytes
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            raise IntegrityError(f"cannot enumerate closure root {current}: {error}") from error
        for entry in entries:
            relative = f"{relative_parent}/{entry.name}" if relative_parent else entry.name
            if "\\" in relative or any(part in {"", ".", ".."} for part in Path(relative).parts):
                raise IntegrityError(f"unsafe artifact path in closure root: {relative!r}")
            try:
                status = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise IntegrityError(f"cannot stat artifact {entry.path}: {error}") from error
            if stat.S_ISLNK(status.st_mode):
                raise IntegrityError(f"closure tree contains a symlink: {entry.path}")
            prefix_excluded = any(
                relative == prefix or relative.startswith(prefix + "/")
                for prefix in prefixes
            )
            if prefix_excluded:
                if relative in prefixes and not stat.S_ISDIR(status.st_mode):
                    raise IntegrityError(
                        f"excluded closure prefix is not a directory: {entry.path}"
                    )
                continue
            if stat.S_ISDIR(status.st_mode):
                visit(Path(entry.path), relative)
                continue
            if relative in exact:
                if not stat.S_ISREG(status.st_mode):
                    raise IntegrityError(
                        f"excluded closure file is not regular: {entry.path}"
                    )
                continue
            if not stat.S_ISREG(status.st_mode):
                raise IntegrityError(f"closure tree contains a special file: {entry.path}")
            if len(rows) >= max_files:
                raise IntegrityError(
                    f"closure exceeds the {max_files} regular-file safety limit"
                )
            if total_bytes + status.st_size > max_bytes:
                raise IntegrityError(
                    f"closure exceeds the {max_bytes}-byte safety limit"
                )
            identity = _hash_regular_file(Path(entry.path))
            total_bytes += int(identity["bytes"])
            rows.append({"path": relative, **identity})

    visit(directory, "")
    rows.sort(key=lambda row: str(row["path"]))
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise IntegrityError(f"closure manifest destination is unsafe: {path}")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temp)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _manifest(
    *,
    schema_name: str,
    exclusions: Sequence[Mapping[str, str]],
    files: list[dict[str, Any]],
    capture_profile: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_name": schema_name,
        "schema_version": 1,
        "root": ".",
        "hash_algorithm": HASH_ALGORITHM,
        "exclusions": [dict(row) for row in exclusions],
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files_sha256": _stable_digest(files),
    }
    if capture_profile is not None:
        value["capture_profile"] = capture_profile
    return value


def write_capture_artifact_closure(root: Path, capture_profile: str) -> Path:
    """Write the post-close exact artifact inventory for one capture."""

    if not isinstance(capture_profile, str) or not capture_profile:
        raise IntegrityError("capture profile must be a non-empty string")
    directory = root.expanduser().absolute()
    files = _scan_tree(
        directory,
        CAPTURE_EXCLUSIONS,
        max_files=MAX_CAPTURE_FILES,
        max_bytes=MAX_CAPTURE_BYTES,
    )
    manifest = _manifest(
        schema_name=CAPTURE_CLOSURE_SCHEMA_NAME,
        exclusions=CAPTURE_EXCLUSIONS,
        files=files,
        capture_profile=capture_profile,
    )
    path = directory / CAPTURE_CLOSURE_FILE
    _atomic_write_json(path, manifest)
    return path


def write_report_tree_closure(root: Path) -> Path:
    """Write the final exact inventory for one generated report tree."""

    directory = root.expanduser().absolute()
    files = _scan_tree(
        directory,
        REPORT_EXCLUSIONS,
        max_files=MAX_REPORT_FILES,
        max_bytes=MAX_REPORT_BYTES,
    )
    manifest = _manifest(
        schema_name=REPORT_CLOSURE_SCHEMA_NAME,
        exclusions=REPORT_EXCLUSIONS,
        files=files,
    )
    path = directory / REPORT_CLOSURE_FILE
    _atomic_write_json(path, manifest)
    return path


def capture_closure_required(root: Path) -> bool:
    """Require closure for captures owned by experiment-lock schema v3+."""

    current = root.expanduser().absolute()
    for directory in (current, *current.parents):
        lock = directory / "00_experiment_lock.json"
        if not (lock.exists() or lock.is_symlink()):
            continue
        try:
            value = _read_manifest(lock)
        except IntegrityError:
            return True
        return bool(
            value.get("schema_name") == "openmvs.dmap.experiment_lock"
            and isinstance(value.get("schema_version"), int)
            and not isinstance(value.get("schema_version"), bool)
            and int(value["schema_version"]) >= 3
        )
    return False


def report_closure_required(root: Path) -> bool:
    """Require closure for report-policy schema v2+ or an explicit contract."""

    policy_path = root.expanduser().absolute() / "report_policy.json"
    if not (policy_path.exists() or policy_path.is_symlink()):
        return False
    try:
        value = _read_manifest(policy_path)
    except IntegrityError:
        return True
    version = value.get("schema_version")
    if (
        value.get("schema_name") == "openmvs.dmap.report_policy"
        and isinstance(version, int)
        and not isinstance(version, bool)
        and version >= REPORT_POLICY_CLOSURE_SCHEMA_VERSION
    ):
        return True
    contract = value.get("integrity_contract")
    return bool(
        isinstance(contract, dict)
        and contract.get("report_tree_closure_schema_version")
        == REPORT_CLOSURE_SCHEMA_VERSION
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        before_path = path.lstat()
    except OSError as error:
        raise IntegrityError(f"cannot stat closure manifest {path}: {error}") from error
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        raise IntegrityError(f"closure manifest must be a regular non-symlink file: {path}")
    if before_path.st_size > MAX_MANIFEST_BYTES:
        raise IntegrityError(f"closure manifest is too large: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            opened_before = os.fstat(descriptor)
            if _status_tuple(opened_before) != _status_tuple(before_path):
                raise IntegrityError(f"closure manifest changed while opening: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            opened_after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
        if not (
            _status_tuple(opened_before)
            == _status_tuple(opened_after)
            == _status_tuple(after_path)
        ):
            raise IntegrityError(f"closure manifest changed while reading: {path}")
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntegrityError(f"cannot read closure manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"closure manifest must be a JSON object: {path}")
    return value


def _validate_manifest_shape(
    manifest: dict[str, Any],
    *,
    schema_name: str,
    exclusions: Sequence[Mapping[str, str]],
    capture_profile: str | None,
) -> None:
    expected_fields = set(_COMMON_FIELDS)
    if schema_name == CAPTURE_CLOSURE_SCHEMA_NAME:
        expected_fields.add("capture_profile")
    if set(manifest) != expected_fields:
        raise IntegrityError("closure manifest has unexpected or missing fields")
    if not (
        manifest.get("schema_name") == schema_name
        and manifest.get("schema_version") == 1
        and not isinstance(manifest.get("schema_version"), bool)
        and manifest.get("root") == "."
        and manifest.get("hash_algorithm") == HASH_ALGORITHM
        and manifest.get("exclusions") == [dict(row) for row in exclusions]
    ):
        raise IntegrityError("closure manifest schema or exclusion contract is invalid")
    if capture_profile is not None and manifest.get("capture_profile") != capture_profile:
        raise IntegrityError("capture closure profile does not match the requested profile")
    if schema_name == CAPTURE_CLOSURE_SCHEMA_NAME and not isinstance(
        manifest.get("capture_profile"), str
    ):
        raise IntegrityError("capture closure profile is malformed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise IntegrityError("closure files must be an array")
    previous = ""
    for index, row in enumerate(files):
        if not isinstance(row, dict) or set(row) != _FILE_FIELDS:
            raise IntegrityError(f"closure file entry {index} is malformed")
        path = row.get("path")
        digest = row.get("sha256")
        size = row.get("bytes")
        mode = row.get("mode")
        if (
            not isinstance(path, str)
            or not path
            or path <= previous
            or Path(path).is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in Path(path).parts)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(mode, int)
            or isinstance(mode, bool)
            or not 0 <= mode <= 0o7777
        ):
            raise IntegrityError(f"closure file entry {index} is malformed")
        previous = path
    if (
        not isinstance(manifest.get("file_count"), int)
        or isinstance(manifest.get("file_count"), bool)
        or manifest["file_count"] != len(files)
        or not isinstance(manifest.get("total_bytes"), int)
        or isinstance(manifest.get("total_bytes"), bool)
        or manifest["total_bytes"] != sum(int(row["bytes"]) for row in files)
        or manifest.get("files_sha256") != _stable_digest(files)
    ):
        raise IntegrityError("closure aggregate fields are invalid")


def _validate_closure(
    root: Path,
    *,
    manifest_name: str,
    schema_name: str,
    exclusions: Sequence[Mapping[str, str]],
    required: bool,
    max_files: int,
    max_bytes: int,
    capture_profile: str | None = None,
) -> ClosureValidation:
    directory = root.expanduser().absolute()
    manifest_path = directory / manifest_name
    if not (manifest_path.exists() or manifest_path.is_symlink()):
        return ClosureValidation(
            valid=not required,
            status="invalid" if required else "legacy-unverified",
            required=required,
            reason=(
                f"required {manifest_name} is missing"
                if required
                else f"legacy artifact has no {manifest_name}; contents are unverified"
            ),
            manifest_path=str(manifest_path),
        )
    try:
        manifest = _read_manifest(manifest_path)
        _validate_manifest_shape(
            manifest,
            schema_name=schema_name,
            exclusions=exclusions,
            capture_profile=capture_profile,
        )
        current = _scan_tree(
            directory,
            exclusions,
            max_files=max_files,
            max_bytes=max_bytes,
        )
        if _read_manifest(manifest_path) != manifest:
            raise IntegrityError("closure manifest changed while validating its tree")
        if manifest["files"] != current:
            expected_by_path = {str(row["path"]): row for row in manifest["files"]}
            current_by_path = {str(row["path"]): row for row in current}
            missing = sorted(set(expected_by_path) - set(current_by_path))
            unexpected = sorted(set(current_by_path) - set(expected_by_path))
            changed = sorted(
                path
                for path in set(expected_by_path) & set(current_by_path)
                if expected_by_path[path] != current_by_path[path]
            )
            details = []
            if missing:
                details.append(f"missing={missing[:5]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:5]}")
            if changed:
                details.append(f"changed={changed[:5]}")
            raise IntegrityError(
                "closure tree differs from its manifest"
                + (f" ({'; '.join(details)})" if details else "")
            )
        return ClosureValidation(
            valid=True,
            status="verified",
            required=required,
            reason=(
                f"verified {manifest['file_count']} files, {manifest['total_bytes']} bytes, "
                f"aggregate {manifest['files_sha256']}"
            ),
            manifest_path=str(manifest_path),
            file_count=int(manifest["file_count"]),
            total_bytes=int(manifest["total_bytes"]),
            files_sha256=str(manifest["files_sha256"]),
        )
    except (IntegrityError, OSError, ValueError) as error:
        return ClosureValidation(
            valid=False,
            status="invalid",
            required=required,
            reason=str(error),
            manifest_path=str(manifest_path),
        )


def validate_capture_artifact_closure(
    root: Path,
    capture_profile: str | None = None,
    *,
    required: bool | None = None,
) -> ClosureValidation:
    return _validate_closure(
        root,
        manifest_name=CAPTURE_CLOSURE_FILE,
        schema_name=CAPTURE_CLOSURE_SCHEMA_NAME,
        exclusions=CAPTURE_EXCLUSIONS,
        required=capture_closure_required(root) if required is None else required,
        max_files=MAX_CAPTURE_FILES,
        max_bytes=MAX_CAPTURE_BYTES,
        capture_profile=capture_profile,
    )


def validate_report_tree_closure(
    root: Path,
    *,
    required: bool | None = None,
) -> ClosureValidation:
    return _validate_closure(
        root,
        manifest_name=REPORT_CLOSURE_FILE,
        schema_name=REPORT_CLOSURE_SCHEMA_NAME,
        exclusions=REPORT_EXCLUSIONS,
        required=report_closure_required(root) if required is None else required,
        max_files=MAX_REPORT_FILES,
        max_bytes=MAX_REPORT_BYTES,
    )


def require_report_tree_closure(
    root: Path,
    *,
    allow_legacy: bool = True,
) -> ClosureValidation:
    validation = validate_report_tree_closure(root)
    if not validation.valid or (
        validation.status == "legacy-unverified" and not allow_legacy
    ):
        raise IntegrityError(validation.reason)
    return validation


def require_capture_artifact_closure(
    root: Path,
    capture_profile: str | None = None,
    *,
    allow_legacy: bool = True,
) -> ClosureValidation:
    validation = validate_capture_artifact_closure(root, capture_profile)
    if not validation.valid or (
        validation.status == "legacy-unverified" and not allow_legacy
    ):
        raise IntegrityError(validation.reason)
    return validation
