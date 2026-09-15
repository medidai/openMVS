#!/usr/bin/env python3
"""Generate a DMAP report with immutable reporter and optional sweep provenance."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any

import tyro


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_sweep  # noqa: E402
from dmap_observability import integrity, predecessor_composite  # noqa: E402


RECOVERY_SCHEMA_NAME = "openmvs.dmap.report_recovery"
RECOVERY_SCHEMA_VERSION = 1
PUBLICATION_TEXT_SUFFIXES = frozenset({
    ".csv", ".html", ".json", ".md", ".yaml", ".yml",
})
FINALIZER_RECEIPT_FILE = "pre_publish_finalizer_receipt.json"
FINALIZER_RECEIPT_SCHEMA_NAME = "openmvs.dmap.pre_publish_finalizer_receipt"
FINALIZER_RECEIPT_SCHEMA_VERSION = 3
FINALIZER_INVENTORY_SCHEMA_NAME = "openmvs.dmap.pre_publish_finalizer_inventory"
FINALIZER_INVENTORY_SCHEMA_VERSION = 1
FINALIZER_INVENTORY_KEY = "pre_publish_finalizer"
FINALIZER_SPEC_FIELDS = frozenset({
    "path", "sha256", "bytes", "mode", "arguments", "environment",
    "approved_outputs",
    "published_report_dir", "logical_argv", "replay_command",
    "explicitly_trusted", "arbitrary_code_execution", "isolation",
    "replay_scope", "runtime_dependencies",
    "spec_sha256",
})
FINALIZER_RECEIPT_FIELDS = frozenset({
    "schema_name", "schema_version", "status", "finalizer",
    "protected_staging_tree_before", "protected_staging_tree_after",
    "published_protected_tree",
    "canonical_tree_before", "canonical_tree_after",
    "approved_output_identities", "report_source_sha256",
    "recovery_binding", "receipt_sha256",
})
TREE_IDENTITY_FIELDS = frozenset({
    "tree_sha256", "file_count", "directory_count", "total_bytes",
    "excluded_files",
})
FILE_IDENTITY_FIELDS = frozenset({"path", "sha256", "bytes", "mode"})
RUNTIME_DEPENDENCY_SCHEMA_NAME = (
    "openmvs.dmap.trusted_finalizer_runtime_dependencies"
)
RUNTIME_DEPENDENCY_SCHEMA_VERSION = 2
RUNTIME_DEPENDENCY_FIELDS = frozenset({
    "schema_name", "schema_version", "coverage", "files", "absent_paths",
    "distributions", "limitations", "manifest_sha256",
})
RUNTIME_DEPENDENCY_FILE_FIELDS = frozenset({
    "role", "owner", "path", "sha256", "bytes", "mode",
})
RUNTIME_DEPENDENCY_DISTRIBUTION_FIELDS = frozenset({"name", "version"})
RUNTIME_DEPENDENCY_FILE_ROLES = frozenset({
    "bound_input", "distribution_metadata", "external_executable",
    "python_module",
})
RUNTIME_DEPENDENCY_COVERAGES = frozenset({
    "invocation_only",
    "loaded_nonstdlib_python_modules_and_declared_files",
})
RUNTIME_DEPENDENCY_BASE_LIMITATIONS = (
    "dynamic-loader and DT_NEEDED transitive shared-library closure is not attested",
    "operating-system kernel, device drivers, firmware, and hardware are not attested",
    "Python modules imported only on unexecuted code paths are not attested",
    "Python import search-path membership and unobserved alternative candidates are not attested beyond recorded cache-file absences",
    "dependency identities are checked at transaction boundaries, not executed from held descriptors",
)
RUNTIME_DEPENDENCY_UNBOUND_LIMITATION = (
    "runtime dependency files and Python distributions are not attested"
)
RECOVERY_BINDING_FIELDS = frozenset({
    "path", "sha256", "schedule_identity_sha256", "evidence_digest",
})
COMPOSITE_ARTIFACT_BINDING_FIELDS = frozenset({
    "schema_name", "schema_version", "parent_schedule_path",
    "parent_schedule_file_sha256", "parent_schedule_identity_sha256",
    "parent_composite_sha256", "parent_evidence_digest", "completed_job_ids",
    "report_source_sha256", "manifest_matrix_sha256", "model_matrix_sha256",
    "core_artifacts", "binding_sha256",
})
FINALIZER_PROTECTED_OUTPUTS = frozenset({
    "report_policy.json", "report_manifest.json", "report_model.json",
    "report_inventory.json", "report_source_provenance.json",
    "report_source_snapshot.tar.zst", "report_source_snapshot.tar.zst.sha256",
    "sweep_report_binding.json", "report_recovery_manifest.json",
    integrity.REPORT_CLOSURE_FILE,
    predecessor_composite.ARTIFACT_BINDING_FILE,
})


@dataclass
class Arguments:
    """Generate and validate one report from existing capture evidence."""

    config: Path
    output_dir: Path
    parent_schedule: Path | None = None
    evidence_context: Path | None = None
    published_output_dir: Path | None = None


def evidence_context_identity(path: Path | None) -> dict[str, Any] | None:
    """Return the exact regular-file identity expected in report policy."""

    if path is None:
        return None
    source = path.expanduser().absolute()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(
            f"evidence context must be a regular non-symlink file: {source}"
        )
    resolved = source.resolve()
    return {
        "path": str(resolved),
        "exists": True,
        "size": resolved.stat().st_size,
        "sha256": dmap_sweep.sha256_file(resolved),
    }


def require_evidence_context_unchanged(
    path: Path | None, expected: dict[str, Any] | None,
) -> None:
    if evidence_context_identity(path) != expected:
        raise RuntimeError("evidence context changed while the report was running")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative_outputs(
    values: list[str], *, allow_framework_receipt: bool = False,
) -> list[str]:
    if not values or not all(isinstance(value, str) and value for value in values):
        raise RuntimeError("trusted finalizer requires a nonempty output allowlist")
    normalized: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
            raise RuntimeError(f"trusted finalizer output must be report-relative: {value}")
        normalized.append(path.as_posix())
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("trusted finalizer output allowlist contains duplicates")
    protected = sorted(set(normalized) & FINALIZER_PROTECTED_OUTPUTS)
    if protected:
        raise RuntimeError(
            "trusted finalizer cannot own protected report controls: "
            + ", ".join(protected)
        )
    if FINALIZER_RECEIPT_FILE in normalized and not allow_framework_receipt:
        raise RuntimeError("the framework, not the finalizer, owns its receipt")
    return sorted(normalized)


def _trusted_environment(values: dict[str, str] | None) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise RuntimeError("trusted finalizer environment must be a JSON object")
    normalized: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(name, str) or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name,
        ) is None:
            raise RuntimeError(f"invalid trusted finalizer environment name: {name!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise RuntimeError(
                f"trusted finalizer environment value must be a NUL-free string: {name}"
            )
        normalized[name] = value
    return dict(sorted(normalized.items()))


def _opened_regular_file_identity(
    descriptor: int,
) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("opened trusted-finalizer object is not a regular file")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    after = os.fstat(descriptor)
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError("opened regular file changed while being hashed")
    os.lseek(descriptor, 0, os.SEEK_SET)
    return {
        "sha256": digest.hexdigest(),
        "bytes": before.st_size,
        "mode": stat.S_IMODE(before.st_mode),
    }, identity


def _regular_file_identity(path: Path) -> dict[str, Any]:
    candidate = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RuntimeError(f"regular file cannot be opened safely: {candidate}: {exc}") from exc
    try:
        result, identity = _opened_regular_file_identity(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = candidate.lstat()
    except OSError as exc:
        raise RuntimeError(f"regular file disappeared during hashing: {candidate}") from exc
    if (
        identity != (
            current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns,
        )
        or stat.S_ISLNK(current.st_mode)
    ):
        raise RuntimeError(f"regular file changed while being hashed: {candidate}")
    return result


def runtime_dependency_file(
    role: str, owner: str, path: Path,
) -> dict[str, Any]:
    """Build one exact runtime-dependency file identity."""

    if role not in RUNTIME_DEPENDENCY_FILE_ROLES:
        raise RuntimeError(f"unsupported runtime dependency role: {role!r}")
    if not isinstance(owner, str) or not owner or "\x00" in owner:
        raise RuntimeError("runtime dependency owner must be a nonempty string")
    candidate = path.expanduser().absolute()
    return {
        "role": role, "owner": owner, "path": str(candidate),
        **_regular_file_identity(candidate),
    }


def _runtime_dependency_limitations(coverage: str) -> list[str]:
    limitations = list(RUNTIME_DEPENDENCY_BASE_LIMITATIONS)
    if coverage == "invocation_only":
        limitations.append(RUNTIME_DEPENDENCY_UNBOUND_LIMITATION)
    return limitations


def build_runtime_dependency_manifest(
    files: list[dict[str, Any]] | None = None,
    distributions: list[dict[str, str]] | None = None,
    coverage: str = "invocation_only",
    absent_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Create a canonical bounded dependency manifest for finalizer replay."""

    value: dict[str, Any] = {
        "schema_name": RUNTIME_DEPENDENCY_SCHEMA_NAME,
        "schema_version": RUNTIME_DEPENDENCY_SCHEMA_VERSION,
        "coverage": coverage,
        "files": sorted(
            list(files or []),
            key=lambda row: (
                str(row.get("role", "")), str(row.get("owner", "")),
                str(row.get("path", "")),
            ),
        ),
        "absent_paths": sorted(list(absent_paths or [])),
        "distributions": sorted(
            list(distributions or []),
            key=lambda row: (
                str(row.get("name", "")).casefold(),
                str(row.get("version", "")),
            ),
        ),
        "limitations": _runtime_dependency_limitations(coverage),
    }
    value["manifest_sha256"] = _stable_digest(value)
    return validate_runtime_dependency_manifest(value, verify_live_bindings=False)


def validate_runtime_dependency_manifest(
    value: dict[str, Any] | None,
    *,
    verify_live_bindings: bool,
) -> dict[str, Any]:
    """Validate exact schema and optionally every declared live dependency."""

    if value is None:
        value = build_runtime_dependency_manifest()
    if not isinstance(value, dict) or set(value) != RUNTIME_DEPENDENCY_FIELDS:
        raise RuntimeError("trusted finalizer runtime manifest fields are invalid")
    coverage = value.get("coverage")
    files = value.get("files")
    distributions = value.get("distributions")
    absent_paths = value.get("absent_paths")
    limitations = value.get("limitations")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not (
        value.get("schema_name") == RUNTIME_DEPENDENCY_SCHEMA_NAME
        and value.get("schema_version") == RUNTIME_DEPENDENCY_SCHEMA_VERSION
        and type(value.get("schema_version")) is int
        and coverage in RUNTIME_DEPENDENCY_COVERAGES
        and value.get("manifest_sha256") == _stable_digest(unsigned)
        and limitations == _runtime_dependency_limitations(str(coverage))
        and isinstance(files, list)
        and isinstance(absent_paths, list)
        and isinstance(distributions, list)
    ):
        raise RuntimeError("trusted finalizer runtime manifest schema/self-digest is invalid")
    if coverage == "invocation_only" and (files or absent_paths or distributions):
        raise RuntimeError("invocation-only runtime manifest must not declare dependencies")
    if coverage != "invocation_only" and not files:
        raise RuntimeError("bounded runtime manifest requires dependency files")

    file_keys: list[tuple[str, str, str]] = []
    file_paths: set[str] = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != RUNTIME_DEPENDENCY_FILE_FIELDS:
            raise RuntimeError("trusted finalizer runtime dependency file is malformed")
        role, owner, path = row.get("role"), row.get("owner"), row.get("path")
        identity = {key: row.get(key) for key in ("sha256", "bytes", "mode")}
        if not (
            role in RUNTIME_DEPENDENCY_FILE_ROLES
            and isinstance(owner, str) and owner and "\x00" not in owner
            and isinstance(path, str) and "\x00" not in path
            and Path(path).is_absolute() and ".." not in Path(path).parts
            and set(identity) == {"sha256", "bytes", "mode"}
            and isinstance(identity["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is not None
            and type(identity["bytes"]) is int and identity["bytes"] >= 0
            and type(identity["mode"]) is int and 0 <= identity["mode"] <= 0o7777
        ):
            raise RuntimeError("trusted finalizer runtime dependency file identity is invalid")
        key = (str(role), owner, path)
        file_keys.append(key)
        if path in file_paths:
            raise RuntimeError("trusted finalizer runtime manifest repeats a file path")
        file_paths.add(path)
        if verify_live_bindings and _regular_file_identity(Path(path)) != identity:
            raise RuntimeError(f"trusted finalizer runtime dependency drifted: {path}")
    if file_keys != sorted(file_keys):
        raise RuntimeError("trusted finalizer runtime dependency files are not canonical")

    if not all(isinstance(path, str) for path in absent_paths):
        raise RuntimeError("trusted finalizer absent dependency path is malformed")
    if absent_paths != sorted(absent_paths) or len(set(absent_paths)) != len(absent_paths):
        raise RuntimeError("trusted finalizer absent dependency paths are not canonical")
    for path in absent_paths:
        if not (
            isinstance(path, str) and path and "\x00" not in path
            and Path(path).is_absolute() and ".." not in Path(path).parts
            and path not in file_paths
        ):
            raise RuntimeError("trusted finalizer absent dependency path is invalid")
        candidate = Path(path)
        if verify_live_bindings and (candidate.exists() or candidate.is_symlink()):
            raise RuntimeError(
                f"trusted finalizer absent runtime dependency appeared: {path}"
            )

    distribution_keys: list[tuple[str, str]] = []
    distribution_names: set[str] = set()
    for row in distributions:
        if (
            not isinstance(row, dict)
            or set(row) != RUNTIME_DEPENDENCY_DISTRIBUTION_FIELDS
            or not isinstance(row.get("name"), str) or not row["name"]
            or not isinstance(row.get("version"), str) or not row["version"]
        ):
            raise RuntimeError("trusted finalizer runtime distribution is malformed")
        normalized_name = row["name"].casefold()
        if normalized_name in distribution_names:
            raise RuntimeError("trusted finalizer runtime manifest repeats a distribution")
        distribution_names.add(normalized_name)
        distribution_keys.append((normalized_name, row["version"]))
        if verify_live_bindings:
            try:
                current_version = importlib.metadata.version(row["name"])
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    f"trusted finalizer runtime distribution is unavailable: {row['name']}"
                ) from exc
            if current_version != row["version"]:
                raise RuntimeError(
                    f"trusted finalizer runtime distribution drifted: {row['name']}"
                )
    if distribution_keys != sorted(distribution_keys):
        raise RuntimeError("trusted finalizer runtime distributions are not canonical")
    return value


def recursive_tree_identity(
    root: Path, excluded_files: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Hash every regular file and directory entry in one non-symlink tree."""

    directory = root.expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"tree root must be a non-symlink directory: {directory}")
    excluded = {
        integrity.REPORT_CLOSURE_FILE,
    }
    excluded.update(
        set(_safe_relative_outputs(
            list(excluded_files), allow_framework_receipt=True,
        ))
        if excluded_files else set()
    )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    file_count = 0
    directory_count = 0
    for path in sorted(directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()):
        relative = path.relative_to(directory).as_posix()
        status = path.lstat()
        if stat.S_ISLNK(status.st_mode):
            raise RuntimeError(f"report tree contains a symlink: {path}")
        if stat.S_ISDIR(status.st_mode):
            entries.append({
                "path": relative, "type": "directory",
                "mode": stat.S_IMODE(status.st_mode),
            })
            directory_count += 1
            continue
        if relative in excluded:
            continue
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(f"report tree contains a special file: {path}")
        identity = _regular_file_identity(path)
        entries.append({"path": relative, "type": "file", **identity})
        file_count += 1
        total_bytes += int(identity["bytes"])
    return {
        "tree_sha256": _stable_digest(entries),
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "excluded_files": sorted(excluded),
    }


def _trusted_finalizer_spec_from_identity(
    path: Path,
    identity: dict[str, Any],
    arguments: list[str],
    approved_outputs: list[str],
    published_dir: Path,
    environment: dict[str, str] | None,
    runtime_dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) and "\x00" not in argument
        for argument in arguments
    ):
        raise RuntimeError("trusted finalizer arguments must be NUL-free strings")
    if set(identity) != {"sha256", "bytes", "mode"} or not (
        isinstance(identity.get("sha256"), str)
        and len(identity["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in identity["sha256"])
        and isinstance(identity.get("bytes"), int)
        and identity["bytes"] >= 0
        and isinstance(identity.get("mode"), int)
        and 0 <= identity["mode"] <= 0o7777
    ):
        raise RuntimeError("trusted finalizer file identity is malformed")
    finalizer = path.expanduser().absolute()
    outputs = _safe_relative_outputs(approved_outputs)
    bound_environment = _trusted_environment(environment)
    runtime_manifest = validate_runtime_dependency_manifest(
        runtime_dependencies, verify_live_bindings=False,
    )
    replay_scope = (
        "invocation_only_runtime_dependencies_unbound"
        if runtime_manifest["coverage"] == "invocation_only"
        else "invocation_and_bounded_runtime_dependencies"
    )
    published = str(published_dir.expanduser().absolute())
    logical_argv = [
        str(finalizer), *arguments,
        "--staged-report-dir", "${STAGED_REPORT_DIR}",
        "--published-report-dir", published,
    ]
    environment_argv = ["env", "-i", *[
        f"{name}={value}" for name, value in bound_environment.items()
    ]]
    replay_command = (
        f"{shlex.join(environment_argv + [str(finalizer), *arguments])} "
        f"--staged-report-dir \"${{STAGED_REPORT_DIR}}\" "
        f"--published-report-dir {shlex.quote(published)}"
    )
    spec: dict[str, Any] = {
        "path": str(finalizer),
        "sha256": identity["sha256"],
        "bytes": identity["bytes"],
        "mode": identity["mode"],
        "arguments": list(arguments),
        "environment": bound_environment,
        "approved_outputs": outputs,
        "published_report_dir": published,
        "logical_argv": logical_argv,
        "replay_command": replay_command,
        "explicitly_trusted": True,
        "arbitrary_code_execution": True,
        "isolation": "none",
        "replay_scope": replay_scope,
        "runtime_dependencies": runtime_manifest,
    }
    spec["spec_sha256"] = _stable_digest(spec)
    return spec


def trusted_finalizer_spec(
    path: Path,
    expected_sha256: str,
    arguments: list[str],
    approved_outputs: list[str],
    published_dir: Path,
    environment: dict[str, str] | None = None,
    runtime_dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not (
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError("trusted finalizer requires a lowercase SHA-256 trust pin")
    finalizer = path.expanduser().absolute()
    if finalizer.is_symlink() or not finalizer.is_file() or not os.access(finalizer, os.X_OK):
        raise RuntimeError(
            f"trusted finalizer must be an executable regular non-symlink file: {finalizer}"
        )
    identity = _regular_file_identity(finalizer)
    if identity["sha256"] != expected_sha256:
        raise RuntimeError("trusted finalizer SHA-256 does not match its explicit trust pin")
    runtime_manifest = validate_runtime_dependency_manifest(
        runtime_dependencies, verify_live_bindings=True,
    )
    return _trusted_finalizer_spec_from_identity(
        finalizer, identity, arguments, approved_outputs, published_dir,
        environment, runtime_manifest,
    )


def _receipt_digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    return _stable_digest(payload)


def _finalizer_inventory_record(
    output_identities: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_name": FINALIZER_INVENTORY_SCHEMA_NAME,
        "schema_version": FINALIZER_INVENTORY_SCHEMA_VERSION,
        "finalizer_receipt": FINALIZER_RECEIPT_FILE,
        "approved_outputs": output_identities,
    }


def publish_finalizer_inventory(
    report_dir: Path, output_identities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Append framework-owned finalizer artifacts without changing report policy."""

    inventory_path = report_dir / "report_inventory.json"
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise RuntimeError("trusted finalizer requires a regular report inventory")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"report inventory is unreadable: {exc}") from exc
    if not isinstance(inventory, dict):
        raise RuntimeError("report inventory must be a JSON object")
    if FINALIZER_INVENTORY_KEY in inventory:
        raise RuntimeError("fresh report inventory already contains finalizer artifacts")
    record = _finalizer_inventory_record(output_identities)
    inventory[FINALIZER_INVENTORY_KEY] = record
    dmap_sweep.atomic_write_json(inventory_path, inventory)
    return record


def require_finalizer_inventory(
    report_dir: Path, output_identities: list[dict[str, Any]],
) -> None:
    inventory_path = report_dir / "report_inventory.json"
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise RuntimeError("trusted finalizer report inventory is missing or unsafe")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"trusted finalizer report inventory is unreadable: {exc}") from exc
    if (
        not isinstance(inventory, dict)
        or inventory.get(FINALIZER_INVENTORY_KEY)
        != _finalizer_inventory_record(output_identities)
    ):
        raise RuntimeError("report inventory does not exactly bind finalizer artifacts")


def report_declares_finalizer_artifacts(report_dir: Path) -> bool:
    inventory_path = report_dir / "report_inventory.json"
    if not inventory_path.exists() and not inventory_path.is_symlink():
        return False
    if inventory_path.is_symlink() or not inventory_path.is_file():
        raise RuntimeError("report inventory is unsafe while checking finalizer contract")
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"report inventory is unreadable: {exc}") from exc
    if not isinstance(inventory, dict):
        raise RuntimeError("report inventory must be a JSON object")
    return FINALIZER_INVENTORY_KEY in inventory


def _require_recovery_binding_current(
    report_dir: Path, parent_schedule: Path | None, report_source: dict[str, Any],
) -> None:
    if parent_schedule is None:
        return
    parent = parent_schedule.expanduser().resolve()
    ledger = load_parent_schedule(parent)
    evidence_digest = dmap_sweep.report_evidence_digest(ledger)
    completed = sorted(
        job_id for job_id, row in (ledger.get("jobs") or {}).items()
        if row.get("status") == "complete"
    )
    binding = json.loads((report_dir / "sweep_report_binding.json").read_text(encoding="utf-8"))
    recovery = json.loads((report_dir / "report_recovery_manifest.json").read_text(encoding="utf-8"))
    checks = [
        binding.get("schema_name") == dmap_sweep.REPORT_BINDING_SCHEMA_NAME,
        binding.get("schema_version") == dmap_sweep.REPORT_BINDING_SCHEMA_VERSION,
        binding.get("schedule_identity_sha256") == ledger["identity"]["sha256"],
        binding.get("evidence_digest") == evidence_digest,
        binding.get("completed_job_ids") == completed,
        binding.get("report_source_sha256") == report_source["sha256"],
        recovery.get("schema_name") == RECOVERY_SCHEMA_NAME,
        recovery.get("schema_version") == RECOVERY_SCHEMA_VERSION,
        recovery.get("parent_schedule_identity_sha256") == ledger["identity"]["sha256"],
        recovery.get("parent_schedule_file_sha256") == dmap_sweep.sha256_file(parent),
        recovery.get("evidence_digest") == evidence_digest,
        recovery.get("completed_job_ids") == completed,
        recovery.get("report_source_sha256") == report_source["sha256"],
    ]
    if not all(checks):
        raise RuntimeError("trusted finalizer observed stale recovery/source bindings")
    composite = ledger.get("composite") or {}
    if (
        composite.get("schema_name") == predecessor_composite.COMPOSITE_SCHEMA_NAME
        and composite.get("schema_version") == predecessor_composite.COMPOSITE_SCHEMA_VERSION
    ):
        valid, reason, _evidence = predecessor_composite.validate_report_artifact_binding(
            report_dir, parent, ledger, report_source["sha256"],
        )
        if not valid:
            raise RuntimeError(f"trusted finalizer observed invalid recovery binding: {reason}")


def _refresh_composite_artifact_binding(
    report_dir: Path,
    parent_schedule: Path | None,
    report_source: dict[str, Any],
) -> None:
    """Rebind composite core artifacts after framework-owned inventory changes."""

    if parent_schedule is None:
        return
    parent = parent_schedule.expanduser().resolve()
    ledger = load_parent_schedule(parent)
    composite = ledger.get("composite") or {}
    if not (
        composite.get("schema_name") == predecessor_composite.COMPOSITE_SCHEMA_NAME
        and composite.get("schema_version")
        == predecessor_composite.COMPOSITE_SCHEMA_VERSION
    ):
        return
    binding = predecessor_composite.build_report_artifact_binding(
        report_dir, parent, ledger, report_source["sha256"],
    )
    dmap_sweep.atomic_write_json(
        report_dir / predecessor_composite.ARTIFACT_BINDING_FILE, binding,
    )
    valid, reason, _evidence = predecessor_composite.validate_report_artifact_binding(
        report_dir, parent, ledger, report_source["sha256"],
    )
    if not valid:
        raise RuntimeError(
            f"framework could not refresh composite finalizer binding: {reason}"
        )


def run_trusted_finalizer_transaction(
    staging_dir: Path,
    published_dir: Path,
    finalizer_path: Path,
    expected_sha256: str,
    arguments: list[str],
    approved_outputs: list[str],
    parent_schedule: Path | None = None,
    environment: dict[str, str] | None = None,
    runtime_dependencies: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run trusted code and detect mutations outside its declared staging outputs.

    This is detection for a compliant trusted finalizer, not containment. Arbitrary
    code can modify the canonical tree before the post-execution check and no rollback
    of such a mutation is attempted.
    """

    staging = staging_dir.expanduser().absolute()
    published = published_dir.expanduser().absolute()
    if staging == published or staging.parent.resolve() != published.parent.resolve():
        raise RuntimeError("trusted finalizer requires sibling staging/published directories")
    integrity.require_report_tree_closure(staging, allow_legacy=True)
    spec = trusted_finalizer_spec(
        finalizer_path, expected_sha256, arguments, approved_outputs, published,
        environment, runtime_dependencies,
    )
    receipt_path = staging / FINALIZER_RECEIPT_FILE
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("fresh staging report already contains a finalizer receipt")
    excluded = [*spec["approved_outputs"], FINALIZER_RECEIPT_FILE]
    staging_before = recursive_tree_identity(staging, excluded)
    canonical_before = recursive_tree_identity(published) if published.is_dir() else None
    source_valid, source_reason, report_source = dmap_sweep.validate_report_source_provenance(staging)
    if not source_valid or report_source is None:
        raise RuntimeError(f"trusted finalizer requires source attestation: {source_reason}")
    _require_recovery_binding_current(staging, parent_schedule, report_source)

    finalizer = Path(spec["path"])
    try:
        descriptor = os.open(
            finalizer,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise RuntimeError(
            f"trusted finalizer cannot be opened safely for execution: {exc}"
        ) from exc
    opened_path = f"/proc/self/fd/{descriptor}"
    try:
        opened_identity, opened_stat = _opened_regular_file_identity(descriptor)
        current_stat = finalizer.lstat()
        current_identity = (
            current_stat.st_dev, current_stat.st_ino, current_stat.st_size,
            current_stat.st_mtime_ns,
        )
        expected_identity = {
            key: spec[key] for key in ("sha256", "bytes", "mode")
        }
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or current_identity != opened_stat
            or opened_identity != expected_identity
        ):
            raise RuntimeError(
                "trusted finalizer opened for execution differs from its trust pin"
            )
        command = [
            opened_path, *spec["arguments"],
            "--staged-report-dir", str(staging),
            "--published-report-dir", str(published),
        ]
        result = subprocess.run(
            command, check=False, pass_fds=(descriptor,), env=spec["environment"],
        )
    finally:
        os.close(descriptor)

    staging_after = recursive_tree_identity(staging, excluded)
    canonical_after = recursive_tree_identity(published) if published.is_dir() else None
    if canonical_before != canonical_after:
        raise RuntimeError("trusted finalizer mutated the canonical report")
    if staging_before != staging_after:
        raise RuntimeError("trusted finalizer mutated files outside its approved output allowlist")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("trusted finalizer attempted to create the framework-owned receipt")
    if result.returncode != 0:
        raise RuntimeError(f"trusted finalizer exited with status {result.returncode}")
    if trusted_finalizer_spec(
        finalizer_path, expected_sha256, arguments, approved_outputs, published,
        environment, runtime_dependencies,
    ) != spec:
        raise RuntimeError("trusted finalizer identity changed during execution")

    output_identities = []
    for relative in spec["approved_outputs"]:
        output_identities.append({
            "path": relative, **_regular_file_identity(staging / relative),
        })
    with tempfile.TemporaryDirectory(prefix="openmvs-dmap-post-finalizer-source-") as directory:
        current_source = dmap_sweep.create_report_source_snapshot(
            Path(directory), "after-finalizer.tar.zst",
        )
    if current_source.archive_sha256 != report_source["sha256"]:
        raise RuntimeError("report-generator source changed while the trusted finalizer ran")
    _require_recovery_binding_current(staging, parent_schedule, report_source)
    publish_finalizer_inventory(staging, output_identities)
    _refresh_composite_artifact_binding(staging, parent_schedule, report_source)
    _require_recovery_binding_current(staging, parent_schedule, report_source)
    published_tree = recursive_tree_identity(staging, excluded)
    recovery_binding = None
    if parent_schedule is not None:
        parent = parent_schedule.expanduser().resolve()
        ledger = load_parent_schedule(parent)
        recovery_binding = {
            "path": str(parent),
            "sha256": dmap_sweep.sha256_file(parent),
            "schedule_identity_sha256": ledger["identity"]["sha256"],
            "evidence_digest": dmap_sweep.report_evidence_digest(ledger),
        }
    receipt: dict[str, Any] = {
        "schema_name": FINALIZER_RECEIPT_SCHEMA_NAME,
        "schema_version": FINALIZER_RECEIPT_SCHEMA_VERSION,
        "status": "complete",
        "finalizer": spec,
        "protected_staging_tree_before": staging_before,
        "protected_staging_tree_after": staging_after,
        "published_protected_tree": published_tree,
        "canonical_tree_before": canonical_before,
        "canonical_tree_after": canonical_after,
        "approved_output_identities": output_identities,
        "report_source_sha256": report_source["sha256"],
        "recovery_binding": recovery_binding,
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    dmap_sweep.atomic_write_json(receipt_path, receipt)
    integrity.write_report_tree_closure(staging)
    integrity.require_report_tree_closure(staging, allow_legacy=False)
    return receipt


def _require_recovery_binding_self_contained(
    report_dir: Path,
    recovery_binding: dict[str, Any],
    report_source: dict[str, Any],
) -> None:
    binding_path = report_dir / "sweep_report_binding.json"
    recovery_path = report_dir / "report_recovery_manifest.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (binding_path, recovery_path)
    ):
        raise RuntimeError("trusted finalizer recovery artifacts are missing or unsafe")
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"trusted finalizer recovery artifacts are unreadable: {exc}") from exc
    if not isinstance(binding, dict) or not isinstance(recovery, dict):
        raise RuntimeError("trusted finalizer recovery artifacts must be JSON objects")
    completed = binding.get("completed_job_ids")
    checks = [
        binding.get("schema_name") == dmap_sweep.REPORT_BINDING_SCHEMA_NAME,
        binding.get("schema_version") == dmap_sweep.REPORT_BINDING_SCHEMA_VERSION,
        recovery.get("schema_name") == RECOVERY_SCHEMA_NAME,
        recovery.get("schema_version") == RECOVERY_SCHEMA_VERSION,
        isinstance(completed, list),
        completed == recovery.get("completed_job_ids"),
        binding.get("schedule_identity_sha256")
        == recovery_binding.get("schedule_identity_sha256"),
        recovery.get("parent_schedule_identity_sha256")
        == recovery_binding.get("schedule_identity_sha256"),
        recovery.get("parent_schedule_file_sha256") == recovery_binding.get("sha256"),
        binding.get("evidence_digest") == recovery_binding.get("evidence_digest"),
        recovery.get("evidence_digest") == recovery_binding.get("evidence_digest"),
        binding.get("report_source_sha256") == report_source.get("sha256"),
        recovery.get("report_source_sha256") == report_source.get("sha256"),
    ]
    if not all(checks):
        raise RuntimeError("trusted finalizer self-contained recovery binding is invalid")
    composite_path = report_dir / predecessor_composite.ARTIFACT_BINDING_FILE
    if composite_path.exists() or composite_path.is_symlink():
        if composite_path.is_symlink() or not composite_path.is_file():
            raise RuntimeError("trusted finalizer composite binding is missing or unsafe")
        try:
            composite = json.loads(composite_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"trusted finalizer composite binding is unreadable: {exc}"
            ) from exc
        unsigned = dict(composite) if isinstance(composite, dict) else {}
        claimed = unsigned.pop("binding_sha256", None)
        core = composite.get("core_artifacts") if isinstance(composite, dict) else None
        composite_checks = [
            isinstance(composite, dict),
            set(composite) == COMPOSITE_ARTIFACT_BINDING_FIELDS
            if isinstance(composite, dict) else False,
            composite.get("schema_name")
            == predecessor_composite.ARTIFACT_BINDING_SCHEMA_NAME
            if isinstance(composite, dict) else False,
            composite.get("schema_version")
            == predecessor_composite.ARTIFACT_BINDING_SCHEMA_VERSION
            if isinstance(composite, dict) else False,
            claimed == _stable_digest(unsigned),
            composite.get("parent_schedule_path") == recovery_binding.get("path")
            if isinstance(composite, dict) else False,
            composite.get("parent_schedule_file_sha256")
            == recovery_binding.get("sha256")
            if isinstance(composite, dict) else False,
            composite.get("parent_schedule_identity_sha256")
            == recovery_binding.get("schedule_identity_sha256")
            if isinstance(composite, dict) else False,
            composite.get("parent_evidence_digest")
            == recovery_binding.get("evidence_digest")
            if isinstance(composite, dict) else False,
            composite.get("completed_job_ids") == completed
            if isinstance(composite, dict) else False,
            composite.get("report_source_sha256") == report_source.get("sha256")
            if isinstance(composite, dict) else False,
            isinstance(core, dict)
            and set(core) == set(predecessor_composite.CORE_REPORT_ARTIFACTS),
        ]
        if not all(composite_checks):
            raise RuntimeError("trusted finalizer self-contained composite binding is invalid")
        for name in predecessor_composite.CORE_REPORT_ARTIFACTS:
            identity = core.get(name)
            current = _regular_file_identity(report_dir / name)
            if not (
                isinstance(identity, dict)
                and set(identity) == {"bytes", "sha256"}
                and identity == {key: current[key] for key in ("bytes", "sha256")}
            ):
                raise RuntimeError(
                    f"trusted finalizer composite core artifact drifted: {name}"
                )


def validate_finalizer_receipt(
    report_dir: Path,
    published_dir: Path,
    expected_spec: dict[str, Any] | None = None,
    require_receipt: bool = False,
    verify_live_bindings: bool = True,
) -> dict[str, Any]:
    report = report_dir.expanduser().absolute()
    receipt_path = report / FINALIZER_RECEIPT_FILE
    if not receipt_path.exists():
        if require_receipt:
            raise RuntimeError("trusted finalizer receipt is required but missing")
        integrity.require_report_tree_closure(report, allow_legacy=True)
        return {"present": False, "valid": True}
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("trusted finalizer receipt must be a regular non-symlink file")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise RuntimeError("trusted finalizer receipt must be a JSON object")
    if set(receipt) != FINALIZER_RECEIPT_FIELDS:
        raise RuntimeError("trusted finalizer receipt has unexpected or missing fields")
    if not (
        receipt.get("schema_name") == FINALIZER_RECEIPT_SCHEMA_NAME
        and receipt.get("schema_version") == FINALIZER_RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "complete"
        and receipt.get("receipt_sha256") == _receipt_digest(receipt)
    ):
        raise RuntimeError("trusted finalizer receipt schema/self-digest is invalid")
    spec = receipt.get("finalizer")
    if (
        not isinstance(spec, dict)
        or set(spec) != FINALIZER_SPEC_FIELDS
        or spec.get("spec_sha256") != _stable_digest({
            key: value for key, value in spec.items() if key != "spec_sha256"
        })
    ):
        raise RuntimeError("trusted finalizer specification is invalid")
    if spec.get("published_report_dir") != str(published_dir.expanduser().absolute()):
        raise RuntimeError("trusted finalizer receipt publication binding is invalid")
    if expected_spec is not None and spec != expected_spec:
        raise RuntimeError("trusted finalizer replay specification differs from the receipt")
    if not (
        spec.get("explicitly_trusted") is True
        and spec.get("arbitrary_code_execution") is True
        and spec.get("isolation") == "none"
        and spec.get("replay_scope") in {
            "invocation_only_runtime_dependencies_unbound",
            "invocation_and_bounded_runtime_dependencies",
        }
    ):
        raise RuntimeError("trusted finalizer receipt misstates its trust/isolation model")
    arguments = spec.get("arguments")
    environment = spec.get("environment")
    approved_outputs = spec.get("approved_outputs")
    if not isinstance(arguments, list) or not all(
        isinstance(argument, str) for argument in arguments
    ):
        raise RuntimeError("trusted finalizer receipt arguments are malformed")
    if not isinstance(approved_outputs, list) or not all(
        isinstance(output, str) for output in approved_outputs
    ):
        raise RuntimeError("trusted finalizer receipt output allowlist is malformed")
    identity = {key: spec.get(key) for key in ("sha256", "bytes", "mode")}
    runtime_dependencies = validate_runtime_dependency_manifest(
        spec.get("runtime_dependencies"), verify_live_bindings=False,
    )
    self_contained_spec = _trusted_finalizer_spec_from_identity(
        Path(str(spec.get("path") or "")), identity,
        arguments, approved_outputs, published_dir, environment,
        runtime_dependencies,
    )
    if self_contained_spec != spec:
        raise RuntimeError("trusted finalizer self-contained replay specification is invalid")
    if verify_live_bindings:
        current_spec = trusted_finalizer_spec(
            Path(str(spec.get("path") or "")), str(spec.get("sha256") or ""),
            arguments, approved_outputs, published_dir, environment,
            runtime_dependencies,
        )
        if current_spec != spec:
            raise RuntimeError("trusted finalizer executable or replay command has drifted")
    outputs = receipt.get("approved_output_identities")
    if not isinstance(outputs, list) or [row.get("path") for row in outputs if isinstance(row, dict)] != spec["approved_outputs"]:
        raise RuntimeError("trusted finalizer output receipt is incomplete")
    for row in outputs:
        if not isinstance(row, dict) or set(row) != FILE_IDENTITY_FIELDS:
            raise RuntimeError("trusted finalizer output identity is malformed")
        if _regular_file_identity(report / str(row["path"])) != {
            key: row[key] for key in ("sha256", "bytes", "mode")
        }:
            raise RuntimeError(f"trusted finalizer output drifted: {row['path']}")
    if receipt.get("canonical_tree_before") != receipt.get("canonical_tree_after"):
        raise RuntimeError("trusted finalizer receipt records a canonical mutation")
    protected_before = receipt.get("protected_staging_tree_before")
    protected_after = receipt.get("protected_staging_tree_after")
    for label, tree in (
        ("protected staging before", protected_before),
        ("protected staging after", protected_after),
        ("published protected", receipt.get("published_protected_tree")),
        ("canonical before", receipt.get("canonical_tree_before")),
        ("canonical after", receipt.get("canonical_tree_after")),
    ):
        if tree is not None and (
            not isinstance(tree, dict) or set(tree) != TREE_IDENTITY_FIELDS
        ):
            raise RuntimeError(f"trusted finalizer {label} identity is malformed")
    if protected_before != protected_after:
        raise RuntimeError("trusted finalizer receipt records an unapproved staging mutation")
    current_tree = recursive_tree_identity(
        report, [*spec["approved_outputs"], FINALIZER_RECEIPT_FILE],
    )
    if current_tree != receipt.get("published_protected_tree"):
        raise RuntimeError("published report protected tree differs from its finalizer receipt")
    require_finalizer_inventory(report, outputs)
    source_valid, source_reason, source = dmap_sweep.validate_report_source_provenance(report)
    if not source_valid or source is None or source["sha256"] != receipt.get("report_source_sha256"):
        raise RuntimeError(f"trusted finalizer source binding is invalid: {source_reason}")
    recovery_binding = receipt.get("recovery_binding")
    if recovery_binding is not None:
        if (
            not isinstance(recovery_binding, dict)
            or set(recovery_binding) != RECOVERY_BINDING_FIELDS
        ):
            raise RuntimeError("trusted finalizer recovery binding is malformed")
        if not all(
            isinstance(recovery_binding.get(key), str)
            and (key == "path" or re.fullmatch(
                r"[0-9a-f]{64}", recovery_binding[key],
            ) is not None)
            for key in RECOVERY_BINDING_FIELDS
        ):
            raise RuntimeError("trusted finalizer recovery binding values are malformed")
        _require_recovery_binding_self_contained(report, recovery_binding, source)
        if verify_live_bindings:
            parent = Path(str(recovery_binding.get("path") or ""))
            ledger = load_parent_schedule(parent)
            if not (
                dmap_sweep.sha256_file(parent) == recovery_binding.get("sha256")
                and ledger["identity"]["sha256"]
                == recovery_binding.get("schedule_identity_sha256")
                and dmap_sweep.report_evidence_digest(ledger)
                == recovery_binding.get("evidence_digest")
            ):
                raise RuntimeError("trusted finalizer recovery source has drifted")
            _require_recovery_binding_current(report, parent, source)
    integrity.require_report_tree_closure(report, allow_legacy=True)
    return {"present": True, "valid": True, "receipt": receipt}


def report_text_files_referencing_path(
    report_dir: Path,
    forbidden_path: Path,
) -> list[str]:
    """List authoritative text artifacts that retain one forbidden absolute path."""

    report = report_dir.expanduser().absolute()
    needle = os.fsencode(str(forbidden_path.expanduser().absolute()))
    if not needle:
        return []
    matches: list[str] = []
    for path in sorted(report.rglob("*")):
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() not in PUBLICATION_TEXT_SUFFIXES
        ):
            continue
        overlap = b""
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                combined = overlap + chunk
                if needle in combined:
                    matches.append(path.relative_to(report).as_posix())
                    break
                retained = max(0, len(needle) - 1)
                overlap = combined[-retained:] if retained else b""
    return matches


def require_no_report_path_reference(
    report_dir: Path,
    forbidden_path: Path,
) -> None:
    matches = report_text_files_referencing_path(report_dir, forbidden_path)
    if matches:
        preview = ", ".join(matches[:8])
        suffix = "" if len(matches) <= 8 else f" (+{len(matches) - 8} more)"
        raise RuntimeError(
            "published report metadata retains its vanished staging path: "
            f"{preview}{suffix}"
        )


def _renameat2(left: Path, right: Path, flags: int, operation: str) -> None:
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise RuntimeError(
            f"atomic report promotion requires Linux renameat2({operation})"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        flags,
    )
    if result != 0:
        error = ctypes.get_errno()
        detail = os.strerror(error) if error else "unknown error"
        if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            detail += f"; filesystem/kernel lacks {operation}"
        raise OSError(error, detail, f"{left} <-> {right}")


def _atomic_exchange(left: Path, right: Path) -> None:
    """Atomically exchange two paths without an archive-first visibility gap."""

    _renameat2(left, right, 2, "RENAME_EXCHANGE")


def _atomic_rename_noreplace(left: Path, right: Path) -> None:
    """Atomically rename a path while refusing a concurrently created target."""

    _renameat2(left, right, 1, "RENAME_NOREPLACE")


def promote_staged_report(
    staging_dir: Path,
    canonical_dir: Path,
    archive_dir: Path | None,
    require_finalizer_receipt: bool = False,
) -> dict[str, Any]:
    """Atomically publish a validated sibling directory and preserve its predecessor."""

    staging = staging_dir.expanduser().absolute()
    canonical = canonical_dir.expanduser().absolute()
    archive = archive_dir.expanduser().absolute() if archive_dir is not None else None
    if staging == canonical:
        raise RuntimeError("staging and canonical report directories must differ")
    if staging.is_symlink() or not staging.is_dir():
        raise RuntimeError(f"staged report must be a regular directory: {staging}")
    if canonical.is_symlink() or (canonical.exists() and not canonical.is_dir()):
        raise RuntimeError(f"canonical report must be a non-symlink directory: {canonical}")
    if staging.parent.resolve() != canonical.parent.resolve():
        raise RuntimeError("staging and canonical report directories must be siblings")

    canonical_existed = canonical.exists()
    canonical_nonempty = (
        canonical.is_dir() and next(canonical.iterdir(), None) is not None
    )
    receipt_required = (
        require_finalizer_receipt
        or report_declares_finalizer_artifacts(staging)
    )
    finalizer_validation = validate_finalizer_receipt(
        staging, canonical,
        require_receipt=receipt_required,
    )
    require_no_report_path_reference(staging, staging)

    def require_predecessor_unchanged(path: Path = canonical) -> None:
        if not finalizer_validation["present"]:
            return
        receipt = finalizer_validation["receipt"]
        current = recursive_tree_identity(path) if path.is_dir() else None
        if current != receipt.get("canonical_tree_after"):
            raise RuntimeError(
                "canonical report drifted after trusted-finalizer execution; "
                "refusing report promotion"
            )

    def require_published_tree_valid() -> None:
        current_receipt_required = (
            receipt_required or report_declares_finalizer_artifacts(canonical)
        )
        validation = validate_finalizer_receipt(
            canonical, canonical, require_receipt=current_receipt_required,
        )
        if validation.get("present") != finalizer_validation.get("present"):
            raise RuntimeError(
                "published report finalizer contract changed during promotion"
            )
        if validation.get("present") and (
            validation.get("receipt") != finalizer_validation.get("receipt")
        ):
            raise RuntimeError(
                "published report finalizer receipt changed during promotion"
            )
        require_no_report_path_reference(canonical, staging)

    if canonical_nonempty:
        if archive is None:
            raise RuntimeError("a nonempty canonical report requires an archive path")
        if archive.parent.resolve() != canonical.parent.resolve():
            raise RuntimeError("report archive must be a sibling of the canonical report")
        if archive.exists() or archive.is_symlink():
            raise RuntimeError(f"report archive already exists: {archive}")
        require_predecessor_unchanged()
        _atomic_exchange(staging, canonical)
        try:
            require_published_tree_valid()
            require_predecessor_unchanged(staging)
            _atomic_rename_noreplace(staging, archive)
            _fsync_directory(canonical.parent)
            require_published_tree_valid()
            require_predecessor_unchanged(archive)
        except Exception as exc:
            try:
                if archive.is_dir() and not staging.exists():
                    _atomic_rename_noreplace(archive, staging)
                _atomic_exchange(staging, canonical)
                _fsync_directory(canonical.parent)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "report promotion failed and atomic rollback also failed; "
                    f"inspect canonical {canonical} and staging {staging}: {rollback_exc}"
                ) from exc
            raise RuntimeError(
                f"report promotion failed; canonical report was rolled back: {exc}"
            ) from exc
        return {
            "canonical": str(canonical),
            "archive": str(archive),
            "replaced_nonempty_report": True,
        }

    require_predecessor_unchanged()
    os.replace(staging, canonical)
    try:
        require_published_tree_valid()
        _fsync_directory(canonical.parent)
        require_published_tree_valid()
    except Exception as exc:
        try:
            os.replace(canonical, staging)
            if canonical_existed:
                canonical.mkdir()
            _fsync_directory(canonical.parent)
        except Exception as rollback_exc:
            raise RuntimeError(
                "report promotion validation failed and rollback also failed; "
                f"new report may remain at {canonical}: {rollback_exc}"
            ) from exc
        raise RuntimeError(
            f"report promotion validation failed; canonical report was rolled back: {exc}"
        ) from exc
    return {
        "canonical": str(canonical),
        "archive": None,
        "replaced_nonempty_report": False,
    }


def require_published_output_binding(
    report_dir: Path,
    published_output_dir: Path,
) -> None:
    """Require staged report metadata to name its immutable publish destination."""

    report = report_dir.expanduser().resolve()
    published = published_output_dir.expanduser().absolute()
    staged = report != published
    def load(name: str) -> dict[str, Any]:
        path = report / name
        if not path.is_file() and not staged:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"report publication binding is unreadable: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"report publication binding is not an object: {path}")
        return value

    policy = load("report_policy.json")
    manifest = load("report_manifest.json")
    expected = str(published)
    policy_binding = policy.get("published_output_dir")
    manifest_binding = manifest.get("published_output_dir")
    if staged and policy_binding != expected:
        raise RuntimeError(
            "staged report policy does not bind the requested published output directory"
        )
    if policy_binding is not None and policy_binding != expected:
        raise RuntimeError("report policy published-output binding is mismatched")
    if staged and manifest_binding != expected:
        raise RuntimeError(
            "staged report manifest does not bind the requested published output directory"
        )
    if manifest_binding is not None and manifest_binding != expected:
        raise RuntimeError("report manifest published-output binding is mismatched")


def load_parent_schedule(path: Path) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"parent schedule is unreadable: {exc}") from exc
    if (
        ledger.get("schema_name") != dmap_sweep.SCHEDULE_SCHEMA_NAME
        or ledger.get("schema_version") != dmap_sweep.SCHEDULE_SCHEMA_VERSION
        or not isinstance((ledger.get("identity") or {}).get("sha256"), str)
    ):
        raise RuntimeError("parent schedule has an unsupported schema or identity")
    if (
        ledger.get("status") not in {"complete", "incomplete", "failed", "stopped"}
        or any(
            session.get("status") == "running"
            for session in (ledger.get("sessions") or [])
        )
    ):
        raise RuntimeError("parent schedule must be finalized before report-only recovery")
    if not any(
        row.get("status") == "complete" for row in (ledger.get("jobs") or {}).values()
    ):
        raise RuntimeError("parent schedule has no completed capture evidence")
    return ledger


def write_recovery_binding(
    report_dir: Path,
    parent_schedule: Path,
    ledger: dict[str, Any],
    report_source: dict[str, Any],
    *,
    create_composite_artifact_binding: bool = True,
) -> dict[str, Any]:
    parent_sha256_before = dmap_sweep.sha256_file(parent_schedule)
    if load_parent_schedule(parent_schedule) != ledger:
        raise RuntimeError("parent schedule content changed before report binding")
    composite = ledger.get("composite") or {}
    is_composite = (
        composite.get("schema_name")
        == predecessor_composite.COMPOSITE_SCHEMA_NAME
        and composite.get("schema_version")
        == predecessor_composite.COMPOSITE_SCHEMA_VERSION
    )
    if is_composite and not create_composite_artifact_binding:
        valid, reason, _evidence = (
            predecessor_composite.validate_report_artifact_binding(
                report_dir,
                parent_schedule,
                ledger,
                report_source["sha256"],
            )
        )
        if not valid:
            raise RuntimeError(
                "reusable composite report lacks its generation-time artifact "
                f"binding: {reason}; rebuild the report"
            )
    evidence_digest = dmap_sweep.report_evidence_digest(ledger)
    dmap_sweep.write_report_binding(
        report_dir, ledger, evidence_digest, report_source
    )
    capture_source = ((ledger.get("identity") or {}).get("source_provenance") or {})
    completed_job_ids = sorted(
        job_id for job_id, row in (ledger.get("jobs") or {}).items()
        if row.get("status") == "complete"
    )
    record = {
        "schema_name": RECOVERY_SCHEMA_NAME,
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "created_at": dmap_sweep.utc_timestamp(),
        "parent_schedule_identity_sha256": ledger["identity"]["sha256"],
        "parent_schedule_file_sha256": dmap_sweep.sha256_file(parent_schedule),
        "parent_capture_source_sha256": capture_source.get("sha256"),
        "report_source_sha256": report_source["sha256"],
        "source_relation": (
            "same_snapshot"
            if capture_source.get("sha256") == report_source["sha256"]
            else "distinct_attested_reporter"
        ),
        "evidence_digest": evidence_digest,
        "completed_job_ids": completed_job_ids,
    }
    dmap_sweep.atomic_write_json(report_dir / "report_recovery_manifest.json", record)
    if is_composite and create_composite_artifact_binding:
        artifact_binding = predecessor_composite.build_report_artifact_binding(
            report_dir,
            parent_schedule,
            ledger,
            report_source["sha256"],
        )
        dmap_sweep.atomic_write_json(
            report_dir / predecessor_composite.ARTIFACT_BINDING_FILE,
            artifact_binding,
        )
    if dmap_sweep.sha256_file(parent_schedule) != parent_sha256_before:
        raise RuntimeError("parent schedule changed while writing report bindings")
    integrity.write_report_tree_closure(report_dir)
    integrity.require_report_tree_closure(report_dir, allow_legacy=False)
    if dmap_sweep.sha256_file(parent_schedule) != parent_sha256_before:
        raise RuntimeError("parent schedule changed while closing report bindings")
    return record


def run(arguments: Arguments) -> int:
    config = arguments.config.expanduser().resolve()
    raw_report_dir = arguments.output_dir.expanduser().absolute()
    if raw_report_dir.is_symlink():
        raise RuntimeError(
            f"report output directory must not be a symlink: {raw_report_dir}"
        )
    report_dir = dmap_sweep.dmap_dev.ensure_external_output_path(
        arguments.output_dir, "attested report output directory"
    )
    raw_published_output_dir = (
        arguments.published_output_dir.expanduser().absolute()
        if arguments.published_output_dir is not None else raw_report_dir
    )
    if raw_published_output_dir.is_symlink():
        raise RuntimeError(
            "published report directory must not be a symlink: "
            f"{raw_published_output_dir}"
        )
    published_output_dir = (
        dmap_sweep.dmap_dev.ensure_external_output_path(
            arguments.published_output_dir,
            "attested published report directory",
        )
        if arguments.published_output_dir is not None else report_dir
    )
    if published_output_dir.parent.resolve() != report_dir.parent.resolve():
        raise RuntimeError(
            "staged and published report directories must be siblings"
        )
    staged_publication = published_output_dir != report_dir
    context_path = (
        arguments.evidence_context.expanduser().absolute()
        if arguments.evidence_context is not None else None
    )
    context_identity = evidence_context_identity(context_path)
    if not config.is_file():
        raise RuntimeError(f"config does not exist: {config}")
    parent_path = (
        arguments.parent_schedule.expanduser().resolve()
        if arguments.parent_schedule is not None else None
    )
    parent_sha256 = (
        dmap_sweep.sha256_file(parent_path) if parent_path is not None else None
    )
    parent_ledger = load_parent_schedule(parent_path) if parent_path is not None else None

    if report_dir.is_dir() and any(report_dir.iterdir()):
        if staged_publication:
            raise RuntimeError(
                "a staged publication directory must be fresh and non-reusable"
            )
        valid, reason = dmap_sweep.valid_report(report_dir)
        if not valid:
            raise RuntimeError(
                f"nonempty report directory is not reusable: {reason}; use a new "
                "numbered directory or --rebuild-report"
            )
        source_valid, source_reason, report_source = (
            dmap_sweep.validate_report_source_provenance(report_dir)
        )
        if not source_valid or report_source is None:
            raise RuntimeError(
                f"valid legacy report is not source-attested: {source_reason}; use a "
                "new numbered directory or --rebuild-report"
            )
        policy_path = report_dir / "report_policy.json"
        try:
            policy = (
                json.loads(policy_path.read_text(encoding="utf-8"))
                if policy_path.is_file() else {}
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"report policy is unreadable: {exc}") from exc
        if context_identity is not None and not policy_path.is_file():
            raise RuntimeError(
                "reusable report has no policy binding for the requested evidence context"
            )
        if policy.get("evidence_context") != context_identity:
            raise RuntimeError(
                "reusable report evidence-context policy does not match the requested sidecar"
            )
        require_published_output_binding(report_dir, published_output_dir)
        finalizer_validation = validate_finalizer_receipt(
            report_dir, published_output_dir,
        )
        if (
            finalizer_validation["present"]
            and parent_path is not None
            and parent_ledger is not None
        ):
            bound_recovery = (
                (finalizer_validation.get("receipt") or {}).get("recovery_binding")
            )
            expected_recovery = {
                "path": str(parent_path),
                "sha256": parent_sha256,
                "schedule_identity_sha256": parent_ledger["identity"]["sha256"],
                "evidence_digest": dmap_sweep.report_evidence_digest(parent_ledger),
            }
            if bound_recovery != expected_recovery:
                raise RuntimeError(
                    "receipt-bearing report does not match the requested parent schedule"
                )
        require_evidence_context_unchanged(context_path, context_identity)
        recovery = None
        if (
            parent_path is not None
            and parent_ledger is not None
            and not finalizer_validation["present"]
        ):
            if dmap_sweep.sha256_file(parent_path) != parent_sha256:
                raise RuntimeError("parent schedule changed while validating the report")
            recovery = write_recovery_binding(
                report_dir,
                parent_path,
                parent_ledger,
                report_source,
                create_composite_artifact_binding=False,
            )
        require_evidence_context_unchanged(context_path, context_identity)
        print(json.dumps({
            "report": str(report_dir),
            "valid": True,
            "validation": reason,
            "reused": True,
            "report_source": report_source,
            "evidence_context": context_identity,
            "published_output_dir": str(published_output_dir),
            "recovery": recovery,
        }, indent=2, sort_keys=True))
        return 0

    with tempfile.TemporaryDirectory(prefix="openmvs-dmap-report-source-") as directory:
        source_dir = Path(directory)
        source_before = dmap_sweep.create_report_source_snapshot(
            source_dir, "before.tar.zst"
        )
        command = [
            sys.executable,
            str(SCRIPT_DIR / "dmap_dev.py"),
            "report",
            "--config", str(config),
            "--output-dir", str(report_dir),
        ]
        if context_path is not None:
            command.extend(["--evidence-context", str(context_path)])
        if staged_publication:
            command.extend([
                "--published-output-dir", str(published_output_dir),
            ])
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
        )
        source_after = dmap_sweep.create_report_source_snapshot(
            source_dir, "after.tar.zst"
        )
        if result.returncode != 0:
            return result.returncode
        require_evidence_context_unchanged(context_path, context_identity)
        require_published_output_binding(report_dir, published_output_dir)
        valid, reason = dmap_sweep.valid_report(report_dir)
        if not valid:
            raise RuntimeError(reason)
        report_source = dmap_sweep.publish_report_source_provenance(
            report_dir, source_before, source_after
        )

    recovery = None
    if parent_path is not None and parent_ledger is not None:
        if dmap_sweep.sha256_file(parent_path) != parent_sha256:
            raise RuntimeError("parent schedule changed while the report was running")
        recovery = write_recovery_binding(
            report_dir,
            parent_path,
            parent_ledger,
            report_source,
            create_composite_artifact_binding=True,
        )
    else:
        integrity.write_report_tree_closure(report_dir)
    valid, reason = dmap_sweep.valid_report(report_dir)
    if not valid:
        raise RuntimeError(reason)
    require_evidence_context_unchanged(context_path, context_identity)
    if staged_publication:
        require_no_report_path_reference(report_dir, report_dir)
    print(json.dumps({
        "report": str(report_dir),
        "valid": True,
        "validation": reason,
        "reused": False,
        "report_source": report_source,
        "evidence_context": context_identity,
        "published_output_dir": str(published_output_dir),
        "recovery": recovery,
    }, indent=2, sort_keys=True))
    return 0


def main() -> int:
    try:
        return run(tyro.cli(Arguments))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
