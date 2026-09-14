#!/usr/bin/env python3
"""Run resumable, budgeted depth-map capture sweeps without an interactive shell."""

from __future__ import annotations

import calendar
from collections import OrderedDict
from dataclasses import dataclass, field
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO, Iterable

import tyro


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import dmap_dev  # noqa: E402
import dmap_drilldown  # noqa: E402
import dmap_report_model  # noqa: E402
from dmap_observability import (  # noqa: E402
    config_materialization,
    integrity,
    source_snapshot,
)


SCHEDULE_SCHEMA_NAME = "openmvs.dmap.sweep_schedule"
SCHEDULE_SCHEMA_VERSION = 2
RETENTION_SCHEMA_NAME = "openmvs.dmap.retention_manifest"
COMPACTED_SCHEMA_NAME = "openmvs.dmap.compacted_completion"
RETENTION_POLICY_SCHEMA_NAME = "openmvs.dmap.retention_policy"
RETENTION_POLICY_SCHEMA_VERSION = 1
RETENTION_MANIFEST_SCHEMA_VERSION = 2
COMPACTED_COMPLETION_SCHEMA_VERSION = 2
DMAP_POLICY_COMPLETE = "complete"
DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY = "instrumented_image_ids_only"
SCENE_FAILURE_ZERO_NEIGHBOR_VIEWS = "zero_neighbor_views"
REPORT_BINDING_SCHEMA_NAME = "openmvs.dmap.sweep_report_binding"
REPORT_BINDING_SCHEMA_VERSION = 2
STAGE_REPORT_BINDING_SCHEMA_NAME = "openmvs.dmap.sweep_stage_report_binding"
STAGE_REPORT_BINDING_SCHEMA_VERSION = 1
REPORT_SOURCE_SCHEMA_NAME = "openmvs.dmap.report_source_provenance"
REPORT_SOURCE_SCHEMA_VERSION = 1
REPORT_SOURCE_SNAPSHOT_NAME = "report_source_snapshot.tar.zst"
DEFAULT_PROFILE_TIMEOUT_MINUTES = {
    "endpoint": 45.0,
    "summary": 60.0,
    "deep": 90.0,
}
CAPTURE_BUDGET_STOP_MARGIN_SECONDS = 5.0
ORPHAN_RECOVERY_MARGIN_SECONDS = 5.0
TRANSIENT_RETURN_CODES = {124, 137, -signal.SIGTERM, -signal.SIGKILL}
TRANSIENT_STDERR_PATTERNS = (
    "temporarily unavailable",
    "cuda_error_devices_unavailable",
    "cuda_error_launch_timeout",
    "cuda_error_unknown",
    "input/output error",
)
SHA256_FILE_BLOCK_BYTES = 1024 * 1024
# Full report validation revisits several thousand lineage artifacts per pass.
# Keep that working set resident while retaining a fixed process-memory bound.
SHA256_FILE_CACHE_MAX_ENTRIES = 8192
SHA256_FILE_MAX_ATTEMPTS = 3

_Sha256FileCacheKey = tuple[str, int, int, int, int, int, int]
_SHA256_FILE_CACHE: OrderedDict[_Sha256FileCacheKey, str] = OrderedDict()
_SHA256_FILE_INFLIGHT: dict[_Sha256FileCacheKey, threading.Event] = {}
_SHA256_FILE_CACHE_LOCK = threading.RLock()


class _FileChangedDuringHash(RuntimeError):
    """Internal retry signal for a path or descriptor that changed mid-hash."""


@dataclass(frozen=True)
class _StableSha256Read:
    digest: str
    bytes_read: int


def _reset_sha256_file_cache_after_fork() -> None:
    """Discard inherited locks/events because their owning threads do not survive fork."""
    global _SHA256_FILE_CACHE, _SHA256_FILE_INFLIGHT, _SHA256_FILE_CACHE_LOCK
    _SHA256_FILE_CACHE = OrderedDict()
    _SHA256_FILE_INFLIGHT = {}
    _SHA256_FILE_CACHE_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_sha256_file_cache_after_fork)


@dataclass
class Arguments:
    """Execute a resumable capture sweep and optionally generate its master report."""

    config: Path
    budget_hours: float = 12.0
    capture_budget_hours: float | None = None
    job_timeout_minutes: float | None = None
    endpoint_timeout_minutes: float | None = None
    summary_timeout_minutes: float | None = None
    deep_timeout_minutes: float | None = None
    report_timeout_minutes: float = 60.0
    term_grace_seconds: float = 120.0
    heartbeat_seconds: float = 15.0
    free_space_floor_gb: float = 300.0
    finalization_reserve_minutes: float = 60.0
    finalization_reserve_gb: float = 20.0
    default_job_output_gb: float = 24.0
    artifact_tree_cap_root: Path | None = None
    artifact_tree_cap_gb: float | None = None
    retry_transient_failures: int = 1
    run: list[str] = field(default_factory=list)
    repeat: list[int] = field(default_factory=list)
    scene: list[str] = field(default_factory=list)
    profile: list[str] = field(default_factory=list)
    ini_override: list[str] = field(default_factory=list)
    schedule_dir: Path | None = None
    report_dir: Path | None = None
    generate_report: bool = True
    compact: bool = True
    plan_only: bool = False
    require_accuracy_ledger: bool = False


@dataclass(frozen=True)
class SweepJob:
    job_id: str
    stage: str
    stage_index: int
    priority: int
    run_label: str
    run_role: str
    family: str
    always_run: bool
    repeat: int
    scene_id: str
    profile: str
    mode: str
    run_dir: Path
    timeout_seconds: float
    estimated_output_bytes: int
    argument_overrides: dict[str, str]
    run_spec: dict[str, Any]
    scene_spec: dict[str, Any]
    retention_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    return_code: int
    elapsed_seconds: float
    timed_out: bool
    terminated: bool
    killed: bool
    stop_requested: bool
    supervisor_command: list[str] | None = None
    process_identity: dict[str, Any] | None = None


def scheduler_policy(
    arguments: Arguments, *, default_report_dir: Path | None = None
) -> dict[str, Any]:
    """Return immutable policy; plan-only and process supervision tuning may change."""
    report_dir = arguments.report_dir or default_report_dir
    policy = {
        "schema_version": 1,
        "budget_hours": arguments.budget_hours,
        "job_timeout_minutes": arguments.job_timeout_minutes,
        "endpoint_timeout_minutes": arguments.endpoint_timeout_minutes,
        "summary_timeout_minutes": arguments.summary_timeout_minutes,
        "deep_timeout_minutes": arguments.deep_timeout_minutes,
        "report_timeout_minutes": arguments.report_timeout_minutes,
        "free_space_floor_gb": arguments.free_space_floor_gb,
        "finalization_reserve_minutes": arguments.finalization_reserve_minutes,
        "finalization_reserve_gb": arguments.finalization_reserve_gb,
        "default_job_output_gb": arguments.default_job_output_gb,
        "retry_transient_failures": arguments.retry_transient_failures,
        "report_dir": str(report_dir.expanduser().resolve()) if report_dir is not None else None,
        "generate_report": arguments.generate_report,
        "compact": arguments.compact,
        "require_accuracy_ledger": arguments.require_accuracy_ledger,
    }
    # Preserve the byte-for-byte policy/hash contract of schedules created
    # before the independent capture ledger existed. Introducing a cap on an
    # existing uncapped schedule remains immutable policy drift.
    if arguments.capture_budget_hours is not None:
        policy["capture_budget_hours"] = arguments.capture_budget_hours
    if arguments.artifact_tree_cap_root is not None:
        policy["artifact_tree_cap_root"] = str(
            arguments.artifact_tree_cap_root.expanduser().resolve()
        )
        policy["artifact_tree_cap_gb"] = arguments.artifact_tree_cap_gb
    policy["sha256"] = stable_digest(policy)
    return policy


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def linux_boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read Linux boot ID: {exc}") from exc
    if not value:
        raise RuntimeError("Linux boot ID is empty")
    return value


def clock_boottime_seconds() -> float:
    clock_id = getattr(time, "CLOCK_BOOTTIME", None)
    if clock_id is None:
        raise RuntimeError("CLOCK_BOOTTIME is required for orphan recovery")
    value = time.clock_gettime(clock_id)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("CLOCK_BOOTTIME returned an invalid value")
    return value


def compact_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _sha256_path_snapshot(
    path: Path, followed_stat: os.stat_result | None = None
) -> tuple[_Sha256FileCacheKey, os.stat_result]:
    """Snapshot the followed path and prove its canonical target still agrees."""
    path_stat = followed_stat if followed_stat is not None else path.stat()
    canonical = path.resolve(strict=True)
    canonical_stat = canonical.stat()
    identity = _sha256_stat_identity(path_stat)
    if _sha256_stat_identity(canonical_stat) != identity:
        raise _FileChangedDuringHash("canonical target changed while taking its snapshot")
    return (str(canonical), *identity), path_stat


def _sha256_read_uncached(path: Path) -> str:
    """Preserve the historical streaming behavior for non-cacheable targets."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(SHA256_FILE_BLOCK_BYTES), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha256_open_stable_descriptor(path: Path) -> BinaryIO:
    """Open without blocking on a target swapped to a special file after stat."""
    flags = os.O_RDONLY | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOCTTY", 0)
    descriptor = os.open(path, flags)
    try:
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _sha256_read_stable_file(
    path: Path,
    expected_key: _Sha256FileCacheKey,
    expected_stat: os.stat_result,
) -> _StableSha256Read:
    """Hash one descriptor and reject any path or descriptor identity drift."""
    hasher = hashlib.sha256()
    with _sha256_open_stable_descriptor(path) as handle:
        descriptor_stat_before = os.fstat(handle.fileno())
        if _sha256_stat_identity(descriptor_stat_before) != _sha256_stat_identity(
            expected_stat
        ):
            raise _FileChangedDuringHash("opened descriptor does not match the path snapshot")
        bytes_read = 0
        for block in iter(lambda: handle.read(SHA256_FILE_BLOCK_BYTES), b""):
            hasher.update(block)
            bytes_read += len(block)
        descriptor_stat_after = os.fstat(handle.fileno())

    if _sha256_stat_identity(descriptor_stat_after) != _sha256_stat_identity(
        descriptor_stat_before
    ):
        raise _FileChangedDuringHash("descriptor changed while it was being hashed")
    try:
        final_key, _ = _sha256_path_snapshot(path)
    except OSError as exc:
        raise _FileChangedDuringHash(
            f"path became unavailable after hashing: {exc}"
        ) from exc
    if final_key != expected_key:
        raise _FileChangedDuringHash("path target changed while it was being hashed")
    return _StableSha256Read(digest=hasher.hexdigest(), bytes_read=bytes_read)


def _clear_sha256_file_cache() -> None:
    """Clear completed entries; intended for deterministic tests and benchmarks."""
    with _SHA256_FILE_CACHE_LOCK:
        _SHA256_FILE_CACHE.clear()


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256, reusing a bounded process-local metadata cache."""
    path = Path(path)
    last_change: _FileChangedDuringHash | None = None
    changes = 0
    while changes < SHA256_FILE_MAX_ATTEMPTS:
        try:
            followed_stat = path.stat()
            if not stat.S_ISREG(followed_stat.st_mode):
                raise ValueError(f"SHA-256 artifact is not a regular file: {path}")
            if followed_stat.st_size <= 0:
                return _sha256_read_uncached(path)
            try:
                key, path_stat = _sha256_path_snapshot(path, followed_stat)
            except OSError as resolution_error:
                try:
                    current_stat = path.stat()
                except OSError:
                    raise resolution_error
                if _sha256_stat_identity(current_stat) == _sha256_stat_identity(
                    followed_stat
                ):
                    return _sha256_read_uncached(path)
                raise _FileChangedDuringHash(
                    "path target changed while resolving its canonical name"
                ) from resolution_error
        except _FileChangedDuringHash as exc:
            last_change = exc
            changes += 1
            continue

        with _SHA256_FILE_CACHE_LOCK:
            cached = _SHA256_FILE_CACHE.get(key)
            if cached is not None:
                _SHA256_FILE_CACHE.move_to_end(key)
                return cached
            pending = _SHA256_FILE_INFLIGHT.get(key)
            if pending is None:
                pending = threading.Event()
                _SHA256_FILE_INFLIGHT[key] = pending
                owns_hash = True
            else:
                owns_hash = False

        if not owns_hash:
            pending.wait()
            continue

        result: _StableSha256Read | None = None
        try:
            result = _sha256_read_stable_file(path, key, path_stat)
        except _FileChangedDuringHash as exc:
            last_change = exc
            changes += 1
        finally:
            with _SHA256_FILE_CACHE_LOCK:
                try:
                    cacheable = (
                        result is not None
                        and stat.S_ISREG(path_stat.st_mode)
                        and path_stat.st_size > 0
                        and result.bytes_read == path_stat.st_size
                    )
                    if cacheable:
                        _SHA256_FILE_CACHE[key] = result.digest
                        _SHA256_FILE_CACHE.move_to_end(key)
                        while len(_SHA256_FILE_CACHE) > SHA256_FILE_CACHE_MAX_ENTRIES:
                            _SHA256_FILE_CACHE.popitem(last=False)
                finally:
                    _SHA256_FILE_INFLIGHT.pop(key, None)
                    pending.set()
        if result is not None:
            return result.digest

    detail = f": {last_change}" if last_change is not None else ""
    raise RuntimeError(
        f"file changed while hashing after {SHA256_FILE_MAX_ATTEMPTS} attempts: {path}{detail}"
    ) from last_change


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directory_usage(path: Path) -> dict[str, int]:
    logical_bytes = 0
    allocated_bytes = 0
    files = 0
    seen_inodes: set[tuple[int, int]] = set()
    if not path.exists():
        return {"files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        stat = candidate.stat()
        files += 1
        logical_bytes += stat.st_size
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen_inodes:
            seen_inodes.add(identity)
            allocated_bytes += stat.st_blocks * 512
    return {
        "files": files,
        "logical_bytes": logical_bytes,
        "allocated_bytes": allocated_bytes,
    }


def artifact_tree_cap_check(
    arguments: Arguments, *, phase: str, job_id: str | None = None
) -> dict[str, Any] | None:
    """Measure an optional allocated-byte cap without following artifact symlinks."""

    if arguments.artifact_tree_cap_root is None:
        return None
    root = arguments.artifact_tree_cap_root.expanduser()
    if root.is_symlink():
        raise RuntimeError(f"artifact tree cap root cannot be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"artifact tree cap root is not a directory: {root}")
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*dirs, *files):
            candidate = base / name
            if candidate.is_symlink():
                raise RuntimeError(
                    f"artifact tree cap refuses symlinked content: {candidate}"
                )
    usage = directory_usage(root)
    limit_bytes = int(float(arguments.artifact_tree_cap_gb) * 1024**3)
    allocated_bytes = int(usage["allocated_bytes"])
    return {
        "schema_name": "openmvs.dmap.artifact_tree_cap_check",
        "schema_version": 1,
        "phase": phase,
        "job_id": job_id,
        "root": str(root),
        "limit_bytes": limit_bytes,
        "allocated_bytes": allocated_bytes,
        "remaining_bytes": max(0, limit_bytes - allocated_bytes),
        "files": int(usage["files"]),
        "valid": allocated_bytes <= limit_bytes,
    }


def record_artifact_tree_cap_check(
    ledger: dict[str, Any], check: dict[str, Any] | None
) -> None:
    if check is None:
        return
    record = ledger.setdefault(
        "artifact_tree_cap",
        {
            "schema_name": "openmvs.dmap.artifact_tree_cap",
            "schema_version": 1,
            "root": check["root"],
            "limit_bytes": check["limit_bytes"],
            "checks": [],
        },
    )
    if (
        record.get("root") != check["root"]
        or record.get("limit_bytes") != check["limit_bytes"]
    ):
        raise RuntimeError("artifact tree cap changed inside the scheduler ledger")
    record["checks"].append(check)
    record["latest"] = check


def artifact_tree_cap_reason(check: dict[str, Any] | None) -> str | None:
    if check is None or check.get("valid") is True:
        return None
    return (
        "artifact tree cap exceeded: "
        f"{check['allocated_bytes']} bytes allocated under {check['root']}, "
        f"limit {check['limit_bytes']} bytes"
    )


def parse_ini_overrides(values: Iterable[str]) -> dict[str, str]:
    return config_materialization.parse_ini_overrides(values)


def merge_ini_overrides(
    config: dict[str, Any], run: dict[str, Any], scene: dict[str, Any], cli_values: Iterable[str]
) -> dict[str, str]:
    return config_materialization.merge_ini_overrides(config, run, scene, cli_values)


def render_ini_override(source: Path, destination: Path, overrides: dict[str, str]) -> dict[str, Any]:
    return config_materialization.render_ini_override(source, destination, overrides)


def replace_argument_value(arguments: list[str], name: str, value: str) -> list[str]:
    result: list[str] = []
    index = 0
    replaced = False
    while index < len(arguments):
        item = arguments[index]
        if item == name:
            result.extend([name, value])
            index += 2
            replaced = True
            continue
        if item.startswith(f"{name}="):
            result.extend([name, value])
            index += 1
            replaced = True
            continue
        result.append(item)
        index += 1
    if not replaced:
        result.extend([name, value])
    return result


def argument_value(arguments: list[str], name: str, default: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(f"{name}="):
            return value.split("=", 1)[1]
    return default


def finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def profile_timeout_seconds(arguments: Arguments, config: dict[str, Any], profile: str) -> float:
    if arguments.job_timeout_minutes is not None:
        minutes = arguments.job_timeout_minutes
    else:
        explicit = {
            "endpoint": arguments.endpoint_timeout_minutes,
            "summary": arguments.summary_timeout_minutes,
            "deep": arguments.deep_timeout_minutes,
        }[profile]
        configured = ((config.get("sweep") or {}).get("profile_timeout_minutes") or {}).get(profile)
        minutes = explicit if explicit is not None else configured
        if minutes is None:
            minutes = DEFAULT_PROFILE_TIMEOUT_MINUTES[profile]
    if not finite_real(minutes) or float(minutes) <= 0:
        raise ValueError(f"timeout for {profile} must be positive")
    return float(minutes) * 60.0


def initial_output_estimate_bytes(
    arguments: Arguments, config: dict[str, Any], scene: dict[str, Any], profile: str
) -> int:
    sweep = config.get("sweep") or {}
    profile_values = sweep.get("profile_estimated_output_gb") or {}
    value = (
        (scene.get("profile_estimated_output_gb") or {}).get(profile)
        if isinstance(scene.get("profile_estimated_output_gb"), dict)
        else None
    )
    if value is None:
        value = profile_values.get(profile, sweep.get("default_job_output_gb", arguments.default_job_output_gb))
    if not finite_real(value) or float(value) < 0:
        raise ValueError(f"output estimate for {profile} must be non-negative")
    return int(float(value) * 1024**3)


def selected_values(requested: Iterable[Any], available: Iterable[Any], label: str) -> set[Any]:
    available_set = set(available)
    requested_set = set(requested)
    unknown = requested_set - available_set
    if unknown:
        raise ValueError(f"unknown {label} selector(s): {sorted(unknown)}")
    return requested_set or available_set


def sweep_stage_specs(config: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_stages = (config.get("sweep") or {}).get("stages") or []
    if raw_stages:
        if not isinstance(raw_stages, list) or not all(isinstance(stage, dict) for stage in raw_stages):
            raise ValueError("sweep.stages must be a list of mappings")
        stages = [dict(stage) for stage in raw_stages]
    else:
        grouped: dict[str, list[str]] = {}
        for run in runs:
            run_sweep = run.get("sweep") or {}
            stage_name = str(run_sweep.get("stage", "main"))
            grouped.setdefault(stage_name, []).append(str(run["label"]))
        stages = [{"name": name, "runs": labels} for name, labels in grouped.items()]
    names: list[str] = []
    for index, stage in enumerate(stages):
        name = str(stage.get("name") or f"stage_{index:02d}")
        if name in names:
            raise ValueError(f"duplicate sweep stage name: {name}")
        stage["name"] = name
        names.append(name)
    for stage in stages:
        promotion = stage.get("promotion")
        if promotion is not None:
            if not isinstance(promotion, dict):
                raise ValueError(f"promotion for stage {stage['name']} must be a mapping")
            source_stage = str(promotion.get("source_stage") or stage["name"])
            next_stage = str(promotion.get("next_stage") or "")
            if source_stage not in names:
                raise ValueError(f"unknown promotion source_stage {source_stage!r}")
            if not next_stage or next_stage not in names:
                raise ValueError(f"unknown promotion next_stage {next_stage!r}")
    return stages


def stage_repeat_indices(stage: dict[str, Any], run: dict[str, Any], repeat_count: int) -> list[int]:
    run_sweep = run.get("sweep") or {}
    raw = stage.get("repeat_indices", run_sweep.get("repeat_indices"))
    if raw is None:
        return list(range(repeat_count))
    if isinstance(raw, int):
        values = [raw]
    elif isinstance(raw, list) and all(isinstance(value, int) for value in raw):
        values = list(raw)
    else:
        raise ValueError(f"repeat_indices for {run['label']} must contain integers")
    invalid = [value for value in values if value < 0 or value >= repeat_count]
    if invalid:
        raise ValueError(f"repeat_indices out of range for {run['label']}: {invalid}")
    return list(dict.fromkeys(values))


def validated_argument_overrides(raw: Any, owner: str) -> dict[str, str]:
    return dmap_dev.validated_argument_overrides(raw, owner)


def stage_argument_overrides(stage: dict[str, Any]) -> dict[str, str]:
    return validated_argument_overrides(
        stage.get("argument_overrides"), f"stage {stage['name']}"
    )


def scene_argument_overrides(scene: dict[str, Any]) -> dict[str, str]:
    return validated_argument_overrides(
        scene.get("argument_overrides"), f"scene {scene.get('scan_id', '<unknown>')}"
    )


def retention_policy_for_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    raw = (config.get("sweep") or {}).get("retention")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("sweep.retention must be a mapping")
    unknown = set(raw) - {"schema_version", "summary_dmap_policy"}
    if unknown:
        raise ValueError(f"unknown sweep.retention option(s): {sorted(unknown)}")
    schema_version = raw.get("schema_version", RETENTION_POLICY_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("sweep.retention.schema_version must be an integer")
    if schema_version != RETENTION_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported sweep.retention.schema_version {schema_version}; "
            f"expected {RETENTION_POLICY_SCHEMA_VERSION}"
        )
    configured = str(raw.get("summary_dmap_policy", DMAP_POLICY_COMPLETE))
    allowed = {DMAP_POLICY_COMPLETE, DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY}
    if configured not in allowed:
        raise ValueError(
            f"unsupported sweep.retention.summary_dmap_policy {configured!r}; "
            f"expected one of {sorted(allowed)}"
        )
    effective = configured if profile == "summary" else DMAP_POLICY_COMPLETE
    return {
        "schema_name": RETENTION_POLICY_SCHEMA_NAME,
        "schema_version": RETENTION_POLICY_SCHEMA_VERSION,
        "profile": profile,
        "dmap_policy": effective,
        "lossy": effective == DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY,
    }


def resolved_retention_policy(job: SweepJob) -> dict[str, Any]:
    raw = job.retention_policy or {
        "schema_name": RETENTION_POLICY_SCHEMA_NAME,
        "schema_version": RETENTION_POLICY_SCHEMA_VERSION,
        "profile": job.profile,
        "dmap_policy": DMAP_POLICY_COMPLETE,
        "lossy": False,
    }
    if not isinstance(raw, dict):
        raise ValueError(f"retention policy for {job.job_id} must be a mapping")
    policy = dict(raw)
    if policy.get("schema_name") != RETENTION_POLICY_SCHEMA_NAME:
        raise ValueError(f"retention policy for {job.job_id} has an invalid schema name")
    if policy.get("schema_version") != RETENTION_POLICY_SCHEMA_VERSION:
        raise ValueError(f"retention policy for {job.job_id} has an unsupported schema version")
    if policy.get("profile") != job.profile:
        raise ValueError(f"retention policy for {job.job_id} targets a different profile")
    dmap_policy = policy.get("dmap_policy")
    allowed = {DMAP_POLICY_COMPLETE, DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY}
    if dmap_policy not in allowed:
        raise ValueError(f"retention policy for {job.job_id} has an invalid DMAP policy")
    expected_lossy = dmap_policy == DMAP_POLICY_INSTRUMENTED_IMAGE_IDS_ONLY
    if policy.get("lossy") is not expected_lossy:
        raise ValueError(f"retention policy for {job.job_id} has an inconsistent lossy flag")
    if expected_lossy and (job.profile != "summary" or job.mode != "timing"):
        raise ValueError("instrumented-image-only retention is restricted to summary/timing jobs")
    return policy


def build_jobs(arguments: Arguments, config: dict[str, Any], root: Path) -> list[SweepJob]:
    runs = [run for run in config.get("runs") or [] if not run.get("existing")]
    scenes = dmap_dev.resolve_scenes(config, dmap_dev.resolve_suite(config))
    stages = sweep_stage_specs(config, runs)
    allowed_profiles = set(dmap_dev.CAPTURE_PROFILE_MODES)
    cli_profiles = set(arguments.profile)
    unknown_profiles = cli_profiles - allowed_profiles
    if unknown_profiles:
        raise ValueError(f"unknown profile selector(s): {sorted(unknown_profiles)}")
    default_profiles = dmap_dev.capture_profiles(config, ())
    selected_runs = selected_values(arguments.run, (str(run["label"]) for run in runs), "run")
    selected_scenes = selected_values(arguments.scene, (str(scene["scan_id"]) for scene in scenes), "scene")
    available_repeats = {
        index
        for run in runs
        for index in range(int(run.get("repeats", 3 if run.get("role") == "baseline" else 1)))
    }
    selected_repeats = selected_values(arguments.repeat, available_repeats, "repeat")
    runs_by_label = {str(run["label"]): run for run in runs}
    scenes_by_id = {str(scene["scan_id"]): scene for scene in scenes}
    jobs: list[SweepJob] = []
    for stage_index, stage in enumerate(stages):
        stage_name = str(stage["name"])
        stage_overrides = stage_argument_overrides(stage)
        stage_runs = [str(value) for value in stage.get("runs") or runs_by_label]
        unknown_runs = set(stage_runs) - set(runs_by_label)
        if unknown_runs:
            raise ValueError(f"stage {stage_name} has unknown runs: {sorted(unknown_runs)}")
        for label in stage_runs:
            if label not in selected_runs:
                continue
            run = runs_by_label[label]
            run_sweep = run.get("sweep") or {}
            stage_scene_values = stage.get("scenes", run_sweep.get("scenes"))
            stage_scenes = (
                [str(value) for value in stage_scene_values]
                if stage_scene_values is not None else list(scenes_by_id)
            )
            unknown_scenes = set(stage_scenes) - set(scenes_by_id)
            if unknown_scenes:
                raise ValueError(f"stage {stage_name} has unknown scenes: {sorted(unknown_scenes)}")
            stage_profile_values = stage.get("profiles", run_sweep.get("profiles", default_profiles))
            stage_profiles = [str(value) for value in stage_profile_values]
            unknown_stage_profiles = set(stage_profiles) - allowed_profiles
            if unknown_stage_profiles:
                raise ValueError(
                    f"stage {stage_name} has unknown profiles: {sorted(unknown_stage_profiles)}"
                )
            if cli_profiles:
                stage_profiles = [profile for profile in stage_profiles if profile in cli_profiles]
            repeat_count = int(run.get("repeats", 3 if run.get("role") == "baseline" else 1))
            repeats = [
                repeat for repeat in stage_repeat_indices(stage, run, repeat_count)
                if repeat in selected_repeats
            ]
            priority = int(stage.get("priority", 0)) + int(run_sweep.get("priority", 0))
            family = str(run_sweep.get("family") or label)
            always_run = bool(run_sweep.get("always_run", run.get("role") == "baseline"))
            for profile in stage_profiles:
                mode = dmap_dev.CAPTURE_PROFILE_MODES[profile]
                for repeat in repeats:
                    for scene_id in stage_scenes:
                        if scene_id not in selected_scenes:
                            continue
                        scene = scenes_by_id[scene_id]
                        argument_overrides = {
                            **stage_overrides,
                            **scene_argument_overrides(scene),
                        }
                        run_dir = dmap_dev.contained_output_path(
                            root,
                            "runs",
                            dmap_dev.validated_output_component(label, "sweep run label"),
                            f"repeat_{repeat:02d}",
                            dmap_dev.validated_output_component(scene_id, "sweep scene_id"),
                            mode,
                            description="sweep run directory",
                        )
                        identity = f"{stage_name}|{label}|{repeat}|{scene_id}|{profile}"
                        jobs.append(SweepJob(
                            job_id=f"job-{hashlib.sha256(identity.encode()).hexdigest()[:16]}",
                            stage=stage_name,
                            stage_index=stage_index,
                            priority=priority,
                            run_label=label,
                            run_role=str(run.get("role", "variant")),
                            family=family,
                            always_run=always_run,
                            repeat=repeat,
                            scene_id=scene_id,
                            profile=profile,
                            mode=mode,
                            run_dir=run_dir,
                            timeout_seconds=profile_timeout_seconds(arguments, config, profile),
                            estimated_output_bytes=initial_output_estimate_bytes(
                                arguments, config, scene, profile
                            ),
                            argument_overrides=argument_overrides,
                            run_spec=run,
                            scene_spec=scene,
                            retention_policy=retention_policy_for_profile(config, profile),
                        ))
    jobs.sort(key=lambda job: (
        job.stage_index, job.priority, job.run_label, job.repeat, job.scene_id, job.profile
    ))
    return jobs


def ensure_source_provenance(schedule_dir: Path) -> dict[str, Any]:
    archive = schedule_dir / "source_snapshot.tar.zst"
    if archive.is_file():
        validated = source_snapshot.validate_source_snapshot(archive)
        comparison = schedule_dir / f".source_snapshot_compare_{os.getpid()}.tar.zst"
        try:
            current = source_snapshot.create_source_snapshot(
                REPO_ROOT, comparison, allow_dirty=True, source_date_epoch=0
            )
            if current.archive_sha256 != validated.archive_sha256:
                raise RuntimeError(
                    "source worktree changed after the immutable sweep snapshot was created"
                )
        finally:
            comparison.unlink(missing_ok=True)
            comparison.with_name(comparison.name + ".sha256").unlink(missing_ok=True)
        result = validated
    else:
        result = source_snapshot.create_source_snapshot(
            REPO_ROOT, archive, allow_dirty=True, source_date_epoch=0
        )
        source_snapshot.validate_source_snapshot(archive)
    return {
        "archive": str(archive.resolve()),
        "checksum": str(archive.with_name(archive.name + ".sha256").resolve()),
        "sha256": result.archive_sha256,
        "bytes": result.archive_bytes,
        "commit": result.commit,
        "dirty": result.dirty,
        "tracked_change_count": result.tracked_change_count,
        "untracked_file_count": result.untracked_file_count,
    }


def create_report_source_snapshot(directory: Path, name: str) -> Any:
    """Capture the current reporter source outside the repository."""
    archive = directory / name
    return source_snapshot.create_source_snapshot(
        REPO_ROOT, archive, allow_dirty=True, source_date_epoch=0
    )


def publish_report_source_provenance(
    report_dir: Path,
    before: Any,
    after: Any,
) -> dict[str, Any]:
    """Publish an attestation only when reporter source stayed unchanged."""
    if before.archive_sha256 != after.archive_sha256:
        raise RuntimeError("report-generator source changed while the report was running")
    report_dir.mkdir(parents=True, exist_ok=True)
    archive = report_dir / REPORT_SOURCE_SNAPSHOT_NAME
    checksum = archive.with_name(archive.name + ".sha256")
    if archive.exists() or checksum.exists():
        raise RuntimeError("report source provenance already exists in a fresh report directory")

    temporary = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
    temporary_checksum = checksum.with_name(f".{checksum.name}.tmp-{os.getpid()}")
    try:
        with before.archive_path.open("rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        if sha256_file(temporary) != before.archive_sha256:
            raise RuntimeError("copied report source snapshot failed its SHA-256 check")
        os.replace(temporary, archive)
        temporary_checksum.write_text(
            f"{before.archive_sha256}  {archive.name}\n", encoding="ascii"
        )
        os.replace(temporary_checksum, checksum)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_checksum.unlink(missing_ok=True)

    validated = source_snapshot.validate_source_snapshot(archive)
    record = {
        "schema_name": REPORT_SOURCE_SCHEMA_NAME,
        "schema_version": REPORT_SOURCE_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "archive": archive.name,
        "checksum": checksum.name,
        "sha256": validated.archive_sha256,
        "bytes": validated.archive_bytes,
        "commit": validated.commit,
        "dirty": validated.dirty,
        "tracked_change_count": validated.tracked_change_count,
        "untracked_file_count": validated.untracked_file_count,
    }
    atomic_write_json(report_dir / "report_source_provenance.json", record)
    return record


def validate_report_source_provenance(report_dir: Path) -> tuple[bool, str, dict[str, Any] | None]:
    provenance_path = report_dir / "report_source_provenance.json"
    try:
        record = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"report has no valid source provenance record: {exc}", None
    if (
        record.get("schema_name") != REPORT_SOURCE_SCHEMA_NAME
        or record.get("schema_version") != REPORT_SOURCE_SCHEMA_VERSION
        or record.get("archive") != REPORT_SOURCE_SNAPSHOT_NAME
        or record.get("checksum") != REPORT_SOURCE_SNAPSHOT_NAME + ".sha256"
    ):
        return False, "report source provenance has an unsupported schema or path", None
    try:
        validated = source_snapshot.validate_source_snapshot(
            report_dir / REPORT_SOURCE_SNAPSHOT_NAME
        )
    except Exception as exc:
        return False, f"report source snapshot failed validation: {exc}", None
    expected = {
        "sha256": validated.archive_sha256,
        "bytes": validated.archive_bytes,
        "commit": validated.commit,
        "dirty": validated.dirty,
        "tracked_change_count": validated.tracked_change_count,
        "untracked_file_count": validated.untracked_file_count,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        return False, "report source provenance does not match its snapshot", None
    return True, "validated immutable report-generator source snapshot", record


def identity_record(
    arguments: Arguments,
    config: dict[str, Any],
    jobs: list[SweepJob],
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    scenes: dict[str, dict[str, Any]] = {}
    for job in jobs:
        if job.scene_id in scenes:
            continue
        mvs = Path(str(job.scene_spec["mvs_file"])).expanduser().resolve()
        ini = mvs.parent / "Densify.ini"
        scenes[job.scene_id] = {
            "mvs": dmap_dev.file_identity(mvs),
            "densify_ini": dmap_dev.file_identity(ini),
        }
    record = {
        "config": dmap_dev.file_identity(Path(str(config["_config_path"]))),
        "production_executable": dmap_dev.file_identity(dmap_dev.densify_binary(config, instrumented=False)),
        "observer_executable": dmap_dev.file_identity(dmap_dev.densify_binary(config, instrumented=True)),
        "input_snapshot": dmap_dev.file_identity(
            dmap_dev.config_path(config, "input_snapshot_manifest", Path(str(config.get("input_snapshot_manifest"))))
        ) if config.get("input_snapshot_manifest") else None,
        "scenes": scenes,
        "selection": {
            "runs": sorted({job.run_label for job in jobs}),
            "repeats": sorted({job.repeat for job in jobs}),
            "scenes": sorted({job.scene_id for job in jobs}),
            "profiles": list(dict.fromkeys(job.profile for job in jobs)),
            "job_matrix": [
                {
                    "job_id": job.job_id,
                    "stage": job.stage,
                    "run": job.run_label,
                    "repeat": job.repeat,
                    "scene": job.scene_id,
                    "profile": job.profile,
                    "timeout_seconds": job.timeout_seconds,
                    "estimated_output_bytes": job.estimated_output_bytes,
                    "argument_overrides": job.argument_overrides,
                    "retention_policy": job.retention_policy,
                }
                for job in jobs
            ],
        },
        "ini_overrides": sorted(arguments.ini_override),
        "source_provenance": source_provenance,
    }
    phase = config.get("experiment_phase")
    if isinstance(phase, dict) and isinstance(phase.get("phase_id"), str):
        phase_root = dmap_dev.experiment_phase_evidence_dir(
            dmap_dev.experiment_root(config), str(phase["phase_id"])
        )
        record["experiment_phase"] = {
            "phase_id": phase["phase_id"],
            "lineage_sha256": phase.get("lineage_sha256"),
            "phase_lock": dmap_dev.file_identity(
                phase_root / "00_phase_lock.json"
            ),
            "resolved_phase": dmap_dev.file_identity(
                phase_root / "01_resolved_phase.yaml"
            ),
            "parent_experiment_lock": dmap_dev.file_identity(
                dmap_dev.experiment_root(config) / "00_experiment_lock.json"
            ),
            "selection_manifest": phase.get("selection_manifest"),
        }
    record["sha256"] = stable_digest(record)
    return record


class SweepLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "SweepLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another sweep owns {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_timestamp()}) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def campaign_report_lock_path(experiment_root: Path) -> Path:
    return experiment_root / "reports" / ".campaign_report.lock"


class CampaignReportLock:
    """Serialize report writers while allowing concurrent read-only consumers."""

    def __init__(self, path: Path, *, shared: bool = False):
        self.path = path
        self.shared = shared
        self.handle: Any = None

    def __enter__(self) -> "CampaignReportLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        operation = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
        try:
            fcntl.flock(self.handle.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            mode = "reader" if self.shared else "writer"
            raise RuntimeError(
                f"another campaign report {mode} conflicts with {self.path}"
            ) from exc
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def initialize_ledger(
    path: Path,
    identity: dict[str, Any],
    jobs: list[SweepJob],
    arguments: Arguments,
    *,
    default_report_dir: Path | None = None,
) -> dict[str, Any]:
    policy = scheduler_policy(arguments, default_report_dir=default_report_dir)
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("schema_name") != SCHEDULE_SCHEMA_NAME or ledger.get("schema_version") != SCHEDULE_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported sweep ledger: {path}")
        if (ledger.get("identity") or {}).get("sha256") != identity["sha256"]:
            raise RuntimeError("sweep identity changed; use a new schedule directory")
        previous_policy = ledger.get("policy") or {}
        if previous_policy.get("sha256") != policy["sha256"]:
            changed = sorted(
                key for key in set(previous_policy) | set(policy)
                if key != "sha256" and previous_policy.get(key) != policy.get(key)
            )
            raise RuntimeError(
                "sweep policy changed; use the original options or a new schedule directory "
                f"(changed: {', '.join(changed) or 'unknown'})"
            )
        return ledger
    ledger = {
        "schema_name": SCHEDULE_SCHEMA_NAME,
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "updated_at": utc_timestamp(),
        "status": "planned" if arguments.plan_only else "running",
        "identity": identity,
        "policy": policy,
        "sessions": [],
        "jobs": {
            job.job_id: {
                "job_id": job.job_id,
                "stage": job.stage,
                "stage_index": job.stage_index,
                "priority": job.priority,
                "run": job.run_label,
                "role": job.run_role,
                "family": job.family,
                "always_run": job.always_run,
                "repeat": job.repeat,
                "scene_id": job.scene_id,
                "profile": job.profile,
                "mode": job.mode,
                "run_dir": str(job.run_dir),
                "timeout_seconds": job.timeout_seconds,
                "estimated_output_bytes": job.estimated_output_bytes,
                "argument_overrides": job.argument_overrides,
                "retention_policy": job.retention_policy,
                "status": "pending",
                "attempts": [],
            }
            for job in jobs
        },
        "stages": {},
        "stage_reports": {},
        "promotions": {},
        "report": {"status": "pending" if arguments.generate_report else "not_requested"},
        "promotion": {
            "policy": "accuracy_first_report_ledger",
            "candidates_hidden": False,
            "status": "pending" if arguments.generate_report else "not_requested",
        },
    }
    refresh_capture_budget(ledger)
    atomic_write_json(path, ledger)
    return ledger


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    refresh_running_budget(ledger)
    refresh_capture_budget(ledger)
    ledger["updated_at"] = utc_timestamp()
    atomic_write_json(path, ledger)


def write_heartbeat(path: Path, **values: Any) -> None:
    atomic_write_json(path, {
        "schema_name": "openmvs.dmap.sweep_heartbeat",
        "schema_version": 1,
        "at": utc_timestamp(),
        "scheduler_pid": os.getpid(),
        **values,
    })


def timestamp_epoch(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return float(calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")))
    except (OverflowError, ValueError):
        return None


def session_elapsed_seconds(session: dict[str, Any]) -> float:
    if not isinstance(session, dict):
        raise ValueError("scheduler session must be an object")
    if "elapsed_seconds" not in session:
        if session.get("status") == "running":
            return 0.0
        raise ValueError("terminal scheduler session is missing elapsed time")
    elapsed = session.get("elapsed_seconds")
    if not finite_real(elapsed) or float(elapsed) < 0.0:
        raise ValueError("scheduler session contains invalid elapsed time")
    return float(elapsed)


def refresh_running_budget(
    ledger: dict[str, Any],
    *,
    now_epoch: float | None = None,
    now_monotonic: float | None = None,
) -> bool:
    """Refresh the durable budget snapshot for this scheduler's active session."""
    now_wall = time.time() if now_epoch is None else now_epoch
    now_steady = time.monotonic() if now_monotonic is None else now_monotonic
    updated = False
    for session in ledger.get("sessions") or []:
        if session.get("status") != "running" or session.get("pid") != os.getpid():
            continue
        elapsed = session_elapsed_seconds(session)
        try:
            started_steady = float(session.get("started_monotonic"))
        except (TypeError, ValueError):
            started_steady = math.nan
        if math.isfinite(started_steady) and now_steady >= started_steady:
            elapsed = max(elapsed, now_steady - started_steady)
        else:
            try:
                started_wall = float(session.get("started_epoch"))
            except (TypeError, ValueError):
                started_wall = math.nan
            if math.isfinite(started_wall) and now_wall >= started_wall:
                elapsed = max(elapsed, now_wall - started_wall)
        session["elapsed_seconds"] = elapsed
        updated = True
    if not updated:
        return False
    budget = ledger.get("budget")
    if isinstance(budget, dict):
        try:
            limit = float(budget.get("limit_seconds"))
        except (TypeError, ValueError):
            limit = math.nan
        if math.isfinite(limit) and limit >= 0.0:
            total = cumulative_elapsed_seconds(ledger)
            budget.update({
                "elapsed_seconds": total,
                "remaining_seconds": max(0.0, limit - total),
            })
    return True


def refresh_schedule_budget(
    path: Path,
    *,
    job_id: str | None = None,
    capture_elapsed_seconds: float | None = None,
    capture_process_identity: dict[str, Any] | None = None,
    capture_supervisor_command: list[str] | None = None,
    capture_supervisor_not_after_epoch: float | None = None,
) -> bool:
    """Persist live scheduler and capture budget heartbeats."""
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    updated = refresh_running_budget(ledger)
    if job_id is not None and capture_elapsed_seconds is not None:
        jobs = ledger.get("jobs") or {}
        job = jobs.get(job_id) if isinstance(jobs, dict) else None
        attempts = job.get("attempts") if isinstance(job, dict) else None
        attempt = attempts[-1] if isinstance(attempts, list) and attempts else None
        if isinstance(attempt, dict) and attempt.get("status") == "running":
            elapsed = float(capture_elapsed_seconds)
            if math.isfinite(elapsed) and elapsed >= 0.0:
                previous = attempt.get("elapsed_seconds", 0.0)
                if (
                    isinstance(previous, (int, float))
                    and not isinstance(previous, bool)
                    and math.isfinite(float(previous))
                    and float(previous) >= 0.0
                ):
                    attempt["elapsed_seconds"] = max(float(previous), elapsed)
                    updated = True
            if capture_process_identity is not None:
                previous_identity = attempt.get("process_identity")
                if previous_identity not in (None, capture_process_identity):
                    raise RuntimeError("capture process identity changed while running")
                attempt["process_identity"] = capture_process_identity
                updated = True
            if capture_supervisor_command is not None:
                previous_command = attempt.get("supervisor_command")
                if previous_command not in (None, capture_supervisor_command):
                    raise RuntimeError("capture supervisor command changed while running")
                attempt["supervisor_command"] = capture_supervisor_command
                attempt["supervisor_kind"] = "coreutils_timeout"
                updated = True
            if capture_supervisor_not_after_epoch is not None:
                if (
                    not finite_real(capture_supervisor_not_after_epoch)
                    or capture_supervisor_not_after_epoch <= time.time()
                ):
                    raise RuntimeError("capture supervisor lifetime bound is invalid")
                previous_not_after = attempt.get("supervisor_not_after_epoch")
                if previous_not_after not in (None, capture_supervisor_not_after_epoch):
                    raise RuntimeError("capture supervisor lifetime bound changed")
                attempt["supervisor_not_after_epoch"] = (
                    capture_supervisor_not_after_epoch
                )
                attempt["supervisor_prepared"] = True
                attempt["supervisor_prepared_boot_id"] = linux_boot_id()
                attempt["supervisor_prepared_boottime_seconds"] = (
                    clock_boottime_seconds()
                )
                updated = True
    if not updated:
        return False
    refresh_capture_budget(ledger)
    ledger["updated_at"] = utc_timestamp()
    atomic_write_json(path, ledger)
    return True


def recover_interrupted_sessions(
    ledger: dict[str, Any], heartbeat_path: Path, *, now_epoch: float | None = None
) -> list[dict[str, Any]]:
    """Close orphaned sessions using their last durable activity as accounting evidence."""
    now = time.time() if now_epoch is None else now_epoch
    heartbeat: dict[str, Any] = {}
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    recovered: list[dict[str, Any]] = []
    for session in ledger.get("sessions") or []:
        if session.get("status") != "running":
            continue
        start = session.get("started_epoch")
        try:
            start_epoch = float(start)
        except (TypeError, ValueError):
            start_epoch = timestamp_epoch(session.get("started_at"))
        if start_epoch is None or not math.isfinite(start_epoch):
            start_epoch = now
        start_epoch = min(start_epoch, now)
        durable_activity = start_epoch + session_elapsed_seconds(session)
        updated_epoch = timestamp_epoch(ledger.get("updated_at"))
        if updated_epoch is not None and start_epoch <= updated_epoch <= now:
            durable_activity = max(durable_activity, updated_epoch)
        heartbeat_epoch = timestamp_epoch(heartbeat.get("at"))
        heartbeat_matches = (
            heartbeat.get("scheduler_pid") == session.get("pid")
            and heartbeat_epoch is not None
            and start_epoch <= heartbeat_epoch <= now
        )
        if heartbeat_matches:
            durable_activity = max(durable_activity, float(heartbeat_epoch))
            recovery_grace = max(
                0.0,
                float(session.get("heartbeat_seconds", 0.0))
                + float(session.get("term_grace_seconds", 0.0)),
            )
            durable_activity = min(now, durable_activity + recovery_grace)
        elapsed = max(session_elapsed_seconds(session), durable_activity - start_epoch)
        session.update({
            "status": "interrupted",
            "finished_at": utc_timestamp(),
            "elapsed_seconds": elapsed,
            "recovered_at": utc_timestamp(),
            "recovery_source": "matching_heartbeat" if heartbeat_matches else "ledger_activity",
        })
        recovered.append(session)
    return recovered


def recover_interrupted_capture_attempts(
    ledger: dict[str, Any], heartbeat_path: Path,
    *,
    ledger_path: Path | None = None,
    current_boot_id: str | None = None,
    current_boottime_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Conservatively account orphaned capture attempts before any resume."""
    heartbeat: dict[str, Any] = {}
    try:
        heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    capture_limit = capture_budget_limit_seconds(ledger)
    recovered: list[dict[str, Any]] = []
    jobs = ledger.get("jobs") or {}
    if not isinstance(jobs, dict):
        raise ValueError("capture ledger jobs must be an object")
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            raise ValueError("capture ledger contains a non-object job")
        attempts = job.get("attempts") or []
        if not isinstance(attempts, list):
            raise ValueError("capture ledger attempts must be a list")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise ValueError("capture ledger contains a non-object attempt")
            if attempt.get("status") != "running":
                continue
            observed = attempt.get("elapsed_seconds", 0.0)
            if (
                not isinstance(observed, (int, float))
                or isinstance(observed, bool)
                or not math.isfinite(float(observed))
                or float(observed) < 0.0
            ):
                raise ValueError("running capture attempt elapsed time is invalid")
            heartbeat_elapsed = heartbeat.get("elapsed_seconds")
            if (
                heartbeat.get("job_id") == job_id
                and isinstance(heartbeat_elapsed, (int, float))
                and not isinstance(heartbeat_elapsed, bool)
                and math.isfinite(float(heartbeat_elapsed))
                and float(heartbeat_elapsed) >= 0.0
            ):
                observed = max(float(observed), float(heartbeat_elapsed))
            bound = attempt.get(
                "effective_timeout_seconds",
                attempt.get("configured_timeout_seconds", job.get("timeout_seconds")),
            )
            if (
                not isinstance(bound, (int, float))
                or isinstance(bound, bool)
                or not math.isfinite(float(bound))
                or float(bound) <= 0.0
            ):
                if capture_limit is None:
                    raise ValueError("orphaned capture attempt has no accounting bound")
                bound = capture_limit
            grace = attempt.get("effective_term_grace_seconds")
            if grace is None:
                grace_values = [
                    session.get("term_grace_seconds")
                    for session in ledger.get("sessions") or []
                    if isinstance(session, dict)
                ]
                grace = grace_values[-1] if grace_values else 0.0
            if (
                not isinstance(grace, (int, float))
                or isinstance(grace, bool)
                or not math.isfinite(float(grace))
                or float(grace) < 0.0
            ):
                if capture_limit is None:
                    raise ValueError("orphaned capture attempt has no grace bound")
                grace = 0.0
                bound = capture_limit
            supervision_bound = float(bound) + float(grace)
            process_identity = attempt.get("process_identity")
            if process_identity is None and heartbeat.get("job_id") == job_id:
                process_identity = heartbeat.get("process_identity")
            if process_identity is not None:
                if not isinstance(process_identity, dict):
                    raise RuntimeError("orphan capture process identity is malformed")
                orphan_cleanup = terminate_verified_orphan_process_group(
                    process_identity, term_grace_seconds=float(grace)
                )
            else:
                # This is the only Popen-to-first-persistence window. New
                # attempts are independently bounded by coreutils timeout, so
                # recovery may proceed only after that signed lifetime passed.
                if attempt.get("supervisor_kind") != "coreutils_timeout":
                    raise RuntimeError(
                        "orphan capture has no verifiable process identity or "
                        "independent lifetime supervisor; refusing resume"
                    )
                if attempt.get("supervisor_prepared") is not True:
                    orphan_cleanup = {"status": "not_spawned_before_preparation"}
                else:
                    recovery_boot_id = current_boot_id or linux_boot_id()
                    recovery_boottime = (
                        clock_boottime_seconds()
                        if current_boottime_seconds is None
                        else current_boottime_seconds
                    )
                    if (
                        not isinstance(recovery_boot_id, str)
                        or not recovery_boot_id
                        or not finite_real(recovery_boottime)
                        or float(recovery_boottime) < 0.0
                    ):
                        raise RuntimeError(
                            "orphan recovery boot identity is invalid"
                        )
                    wait = attempt.get("orphan_recovery_wait")
                    if wait is None:
                        safe_after = (
                            float(recovery_boottime) + supervision_bound
                            + ORPHAN_RECOVERY_MARGIN_SECONDS
                        )
                        attempt["orphan_recovery_wait"] = {
                            "schema_name": "openmvs.dmap.orphan_recovery_wait",
                            "schema_version": 1,
                            "boot_id": recovery_boot_id,
                            "started_boottime_seconds": float(recovery_boottime),
                            "safe_after_boottime_seconds": safe_after,
                            "supervision_bound_seconds": supervision_bound,
                            "margin_seconds": ORPHAN_RECOVERY_MARGIN_SECONDS,
                        }
                        refresh_capture_budget(ledger)
                        ledger["updated_at"] = utc_timestamp()
                        if ledger_path is not None:
                            atomic_write_json(ledger_path, ledger)
                        raise RuntimeError(
                            "orphan capture identity was not persisted; durable "
                            "same-boot supervision wait started, refusing resume "
                            f"until CLOCK_BOOTTIME {safe_after:.3f}"
                        )
                    if not isinstance(wait, dict):
                        raise RuntimeError("orphan recovery wait is malformed")
                    wait_boot_id = wait.get("boot_id")
                    wait_started = wait.get("started_boottime_seconds")
                    safe_after = wait.get("safe_after_boottime_seconds")
                    wait_bound = wait.get("supervision_bound_seconds")
                    wait_margin = wait.get("margin_seconds")
                    if (
                        wait.get("schema_name")
                        != "openmvs.dmap.orphan_recovery_wait"
                        or wait.get("schema_version") != 1
                        or not isinstance(wait_boot_id, str)
                        or not wait_boot_id
                        or not finite_real(wait_started)
                        or not finite_real(safe_after)
                        or not finite_real(wait_bound)
                        or not finite_real(wait_margin)
                        or float(wait_bound) != supervision_bound
                        or float(wait_margin) != ORPHAN_RECOVERY_MARGIN_SECONDS
                        or float(safe_after) != (
                            float(wait_started) + float(wait_bound)
                            + float(wait_margin)
                        )
                    ):
                        raise RuntimeError("orphan recovery wait contract is invalid")
                    if wait_boot_id != recovery_boot_id:
                        orphan_cleanup = {
                            "status": "prior_boot_process_gone",
                            "wait_boot_id": wait_boot_id,
                            "recovery_boot_id": recovery_boot_id,
                        }
                    elif float(recovery_boottime) < float(safe_after):
                        raise RuntimeError(
                            "orphan capture identity was not persisted; refusing "
                            "same-boot resume until CLOCK_BOOTTIME "
                            f"{float(safe_after):.3f}"
                        )
                    else:
                        orphan_cleanup = {
                            "status": "independent_timeout_expired",
                            "boot_id": recovery_boot_id,
                            "safe_after_boottime_seconds": float(safe_after),
                        }
            charged = max(float(observed), supervision_bound)
            uncapped_recovery = capture_limit is None
            retry_value = (ledger.get("policy") or {}).get(
                "retry_transient_failures", 0
            )
            retry_limit = (
                int(retry_value)
                if isinstance(retry_value, int)
                and not isinstance(retry_value, bool)
                and retry_value in {0, 1}
                else 0
            )
            retry_available = (
                uncapped_recovery and len(attempts) < 1 + retry_limit
            )
            attempt.update({
                "status": "interrupted",
                "finished_at": utc_timestamp(),
                "elapsed_seconds": charged,
                "observed_elapsed_seconds": float(observed),
                "capture_accounting": "conservative_supervision_bound",
                "orphan_cleanup": orphan_cleanup,
                "transient": retry_available,
                "recovery_retry_available": retry_available,
                "validation": "scheduler interruption before exact attempt finalization",
            })
            job.update({
                "status": "retry_pending" if retry_available else "failed",
                "validation": attempt["validation"],
            })
            recovered.append({
                "job_id": job_id,
                "attempt": attempt.get("attempt"),
                "charged_seconds": charged,
            })
    refresh_capture_budget(ledger)
    return recovered


def cumulative_elapsed_seconds(ledger: dict[str, Any]) -> float:
    sessions = ledger.get("sessions") or []
    if not isinstance(sessions, list):
        raise ValueError("scheduler sessions must be a list")
    return sum(session_elapsed_seconds(session) for session in sessions)


def cumulative_capture_elapsed_seconds(ledger: dict[str, Any]) -> float:
    """Return signed capture-process time from all persisted attempts."""
    total = 0.0
    for job in (ledger.get("jobs") or {}).values():
        if not isinstance(job, dict):
            raise ValueError("capture ledger contains a non-object job")
        for attempt in job.get("attempts") or []:
            if not isinstance(attempt, dict):
                raise ValueError("capture ledger contains a non-object attempt")
            if "elapsed_seconds" not in attempt:
                raise ValueError("capture ledger attempt is missing elapsed time")
            elapsed = attempt.get("elapsed_seconds")
            if (
                not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0
            ):
                raise ValueError("capture ledger contains invalid attempt elapsed time")
            total += float(elapsed)
    return total


def capture_budget_limit_seconds(ledger: dict[str, Any]) -> float | None:
    value = (ledger.get("policy") or {}).get("capture_budget_hours")
    if value is None:
        return None
    if (
        not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value)) or float(value) <= 0.0
    ):
        raise ValueError("capture budget policy is invalid")
    return float(value) * 3600.0


def refresh_capture_budget(ledger: dict[str, Any]) -> None:
    limit = capture_budget_limit_seconds(ledger)
    elapsed = cumulative_capture_elapsed_seconds(ledger)
    ledger["capture_budget"] = {
        "enforced": limit is not None,
        "limit_seconds": limit,
        "elapsed_seconds": elapsed,
        "remaining_seconds": None if limit is None else max(0.0, limit - elapsed),
        "valid": limit is None or elapsed <= limit,
    }


def remaining_capture_budget_seconds(
    arguments: Arguments, ledger: dict[str, Any]
) -> float | None:
    if arguments.capture_budget_hours is None:
        return None
    limit = float(arguments.capture_budget_hours) * 3600.0
    return max(0.0, limit - cumulative_capture_elapsed_seconds(ledger))


def remaining_campaign_budget_seconds(arguments: Arguments, ledger: dict[str, Any]) -> float:
    return max(0.0, arguments.budget_hours * 3600.0 - cumulative_elapsed_seconds(ledger))


def timeout_supervisor_command(
    command: list[str], timeout_seconds: float, term_grace_seconds: float
) -> list[str]:
    timeout_binary = shutil.which("timeout")
    if timeout_binary is None:
        raise RuntimeError("coreutils timeout is required for capture supervision")
    duration = f"{float(timeout_seconds):.9g}s"
    if term_grace_seconds > 0.0:
        return [
            timeout_binary,
            "--signal=TERM",
            f"--kill-after={float(term_grace_seconds):.9g}s",
            "--",
            duration,
            *command,
        ]
    return [timeout_binary, "--signal=KILL", "--", duration, *command]


def linux_process_identity(pid: int) -> dict[str, Any] | None:
    """Return PID-reuse-safe Linux process identity and exact argv."""
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = stat_text.rfind(")")
        if close < 0:
            raise ValueError("malformed proc stat")
        fields = stat_text[close + 2:].split()
        if len(fields) <= 19:
            raise ValueError("short proc stat")
        cmdline_bytes = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = [
            value.decode("utf-8", errors="surrogateescape")
            for value in cmdline_bytes.split(b"\0") if value
        ]
        boot_id = linux_boot_id()
        return {
            "pid": int(pid),
            "process_group_id": int(fields[2]),
            "state": fields[0],
            "start_ticks": int(fields[19]),
            "boot_id": boot_id,
            "cmdline": cmdline,
            "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
        }
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"cannot read Linux process identity for pid {pid}: {exc}") from exc


def live_process_group_members(process_group_id: int) -> list[int]:
    members: list[int] = []
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            stat_text = stat_path.read_text(encoding="utf-8")
            close = stat_text.rfind(")")
            fields = stat_text[close + 2:].split()
            if (
                close >= 0 and len(fields) > 2
                and int(fields[2]) == process_group_id
                and fields[0] not in {"Z", "X"}
            ):
                members.append(int(stat_path.parent.name))
        except (FileNotFoundError, OSError, ValueError):
            continue
    return sorted(members)


def terminate_verified_orphan_process_group(
    expected: dict[str, Any], *, term_grace_seconds: float
) -> dict[str, Any]:
    """Kill a persisted supervisor group, refusing PID reuse or uncertain cleanup."""
    pid = expected.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise RuntimeError("orphan supervisor identity has an invalid pid")
    current = linux_process_identity(pid)
    if current is None:
        return {"status": "already_exited", "pid": pid}
    identity_fields = ("pid", "process_group_id", "start_ticks", "boot_id")
    if any(current.get(field) != expected.get(field) for field in identity_fields):
        raise RuntimeError(
            f"orphan supervisor pid {pid} identity changed; refusing PID-reuse kill"
        )
    process_group_id = current["process_group_id"]
    leader_exited = current.get("state") in {"Z", "X"}
    if leader_exited:
        if not live_process_group_members(process_group_id):
            return {"status": "already_exited", "pid": pid}
    else:
        if current.get("cmdline_sha256") != expected.get("cmdline_sha256"):
            raise RuntimeError(f"orphan supervisor pid {pid} command identity changed")
        expected_command = expected.get("cmdline")
        if not isinstance(expected_command, list) or current.get("cmdline") != expected_command:
            raise RuntimeError(f"orphan supervisor pid {pid} command changed")
    if process_group_id != pid:
        raise RuntimeError("capture supervisor is not its persisted process-group leader")
    first_signal = signal.SIGTERM if term_grace_seconds > 0.0 else signal.SIGKILL
    try:
        os.killpg(process_group_id, first_signal)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.0, term_grace_seconds)
    while live_process_group_members(process_group_id) and time.monotonic() < deadline:
        time.sleep(0.05)
    members = live_process_group_members(process_group_id)
    killed = first_signal == signal.SIGKILL
    if members:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            killed = True
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 5.0
        while live_process_group_members(process_group_id) and time.monotonic() < kill_deadline:
            time.sleep(0.05)
        members = live_process_group_members(process_group_id)
    if members:
        raise RuntimeError(
            "orphan capture process group remains live after SIGKILL: "
            + ",".join(str(value) for value in members)
        )
    return {
        "status": "killed" if killed else "terminated",
        "pid": pid,
        "process_group_id": process_group_id,
    }


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float) -> tuple[bool, bool]:
    terminated = False
    killed = False
    deadline = time.monotonic() + max(0.0, grace_seconds)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            terminated = True
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(0.0, grace_seconds))
        except subprocess.TimeoutExpired:
            pass
    members = live_process_group_members(process.pid)
    while members and time.monotonic() < deadline:
        time.sleep(0.05)
        members = live_process_group_members(process.pid)
    if members:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            killed = True
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 5.0
        while live_process_group_members(process.pid) and time.monotonic() < kill_deadline:
            time.sleep(0.05)
        members = live_process_group_members(process.pid)
    if members:
        raise RuntimeError(
            "capture process group remains live after SIGKILL: "
            + ",".join(str(value) for value in members)
        )
    if process.poll() is None:
        process.wait()
    return terminated, killed


def run_process_group(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    term_grace_seconds: float,
    heartbeat_seconds: float,
    heartbeat_path: Path,
    job_id: str,
    stop_event: threading.Event,
    schedule_path: Path | None = None,
) -> ProcessResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    terminated = False
    killed = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        supervisor_command = timeout_supervisor_command(
            command, timeout_seconds, term_grace_seconds
        )
        supervisor_not_after_epoch = (
            time.time() + timeout_seconds + term_grace_seconds
            + max(60.0, heartbeat_seconds * 2.0)
        )
        if schedule_path is not None:
            persisted = refresh_schedule_budget(
                schedule_path,
                job_id=job_id,
                capture_elapsed_seconds=0.0,
                capture_supervisor_command=supervisor_command,
                capture_supervisor_not_after_epoch=supervisor_not_after_epoch,
            )
            if not persisted:
                raise RuntimeError("capture supervisor preparation was not persisted")
        process = subprocess.Popen(
            supervisor_command,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        next_heartbeat = 0.0
        supervision_failed = False
        process_identity: dict[str, Any] | None = None
        try:
            process_identity = linux_process_identity(process.pid)
            if process_identity is None and process.poll() is None:
                raise RuntimeError("capture supervisor identity disappeared after spawn")
            if schedule_path is not None:
                persisted = refresh_schedule_budget(
                    schedule_path,
                    job_id=job_id,
                    capture_elapsed_seconds=0.0,
                    capture_process_identity=process_identity,
                    capture_supervisor_command=supervisor_command,
                )
                if not persisted:
                    raise RuntimeError("capture supervisor identity was not persisted")
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= next_heartbeat:
                    write_heartbeat(
                        heartbeat_path,
                        status="running",
                        job_id=job_id,
                        pid=process.pid,
                        elapsed_seconds=elapsed,
                        timeout_seconds=timeout_seconds,
                        process_identity=process_identity,
                    )
                    if schedule_path is not None:
                        refresh_schedule_budget(
                            schedule_path,
                            job_id=job_id,
                            capture_elapsed_seconds=elapsed,
                        )
                    next_heartbeat = elapsed + max(0.05, heartbeat_seconds)
                if stop_event.is_set():
                    terminated, killed = terminate_process_group(
                        process, term_grace_seconds
                    )
                    break
                if elapsed >= timeout_seconds:
                    timed_out = True
                    terminated, killed = terminate_process_group(
                        process, term_grace_seconds
                    )
                    break
                time.sleep(max(0.05, min(heartbeat_seconds, 1.0)))
        except BaseException:
            supervision_failed = True
            raise
        finally:
            # Once Popen succeeds, no scheduler-side exception may leave the
            # child (or one of its descendants) running outside accounting.
            if process.poll() is None:
                cleanup_terminated, cleanup_killed = terminate_process_group(
                    process, term_grace_seconds
                )
                terminated = terminated or cleanup_terminated
                killed = killed or cleanup_killed
            elapsed = time.monotonic() - started
            if supervision_failed:
                try:
                    write_heartbeat(
                        heartbeat_path,
                        status="supervision_failed",
                        job_id=job_id,
                        pid=process.pid,
                        elapsed_seconds=elapsed,
                        timeout_seconds=timeout_seconds,
                        return_code=process.returncode,
                        process_identity=process_identity,
                    )
                except Exception:
                    pass
                if schedule_path is not None:
                    try:
                        refresh_schedule_budget(
                            schedule_path,
                            job_id=job_id,
                            capture_elapsed_seconds=elapsed,
                        )
                    except Exception:
                        pass
        if process.returncode == 124:
            timed_out = True
            terminated = True
        elif process.returncode == 137:
            timed_out = True
            terminated = True
            killed = True
        elif (
            process.returncode == -signal.SIGKILL
            and not stop_event.is_set()
            and elapsed >= timeout_seconds
        ):
            timed_out = True
            # Coreutils may itself surface SIGKILL instead of translating the
            # completed TERM/kill-after sequence to 137 under scheduler load.
            terminated = term_grace_seconds > 0.0
            killed = True
        final_status = (
            "timed_out" if timed_out
            else "stopped" if stop_event.is_set()
            else "complete" if process.returncode == 0
            else "failed"
        )
        write_heartbeat(
            heartbeat_path,
            status=final_status,
            job_id=job_id,
            pid=process.pid,
            elapsed_seconds=elapsed,
            timeout_seconds=timeout_seconds,
            return_code=process.returncode,
            process_identity=process_identity,
        )
        if schedule_path is not None:
            refresh_schedule_budget(
                schedule_path,
                job_id=job_id,
                capture_elapsed_seconds=elapsed,
            )
    return ProcessResult(
        return_code=int(process.returncode if process.returncode is not None else -signal.SIGKILL),
        elapsed_seconds=elapsed,
        timed_out=timed_out,
        terminated=terminated,
        killed=killed,
        stop_requested=stop_event.is_set(),
        supervisor_command=supervisor_command,
        process_identity=process_identity,
    )


def quarantine_path(path: Path, root: Path, job_id: str, reason: str) -> Path | None:
    if not path.exists() or not any(path.iterdir()):
        return None
    destination = root / "failures" / "sweep" / job_id / compact_timestamp() / path.name
    counter = 0
    while destination.exists():
        counter += 1
        destination = destination.with_name(f"{path.name}_{counter:02d}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    atomic_write_json(destination.parent / "quarantine.json", {
        "schema_name": "openmvs.dmap.sweep_quarantine",
        "schema_version": 1,
        "at": utc_timestamp(),
        "source": str(path),
        "destination": str(destination),
        "reason": reason,
    })
    return destination


def build_capture_command(
    arguments: Arguments,
    config: dict[str, Any],
    job: SweepJob,
) -> tuple[list[str], Path, Path, dict[str, Any] | None]:
    source_work = Path(str(job.scene_spec["working_folder"])).expanduser().resolve()
    source_mvs = Path(str(job.scene_spec["mvs_file"])).expanduser().resolve()
    work_dir = job.run_dir / "work"
    dmap_dev.prepare_working_folder(source_work, work_dir)
    local_mvs = dmap_dev.stage_mvs_input(source_work, source_mvs, work_dir)
    command_work_dir = dmap_dev.dmap_working_folder(local_mvs)
    scene_name = dmap_dev.validated_output_component(
        job.scene_spec.get("name", job.scene_id[:8]), "sweep scene name"
    )
    output_mvs = job.run_dir / f"{scene_name}_dense.mvs"
    densify_args = [
        *(str(value) for value in config.get("default_densify_args") or []),
        *(str(value) for value in job.run_spec.get("densify_args") or []),
    ]
    for option, value in job.argument_overrides.items():
        densify_args = replace_argument_value(densify_args, option, value)
    dmap_dev.validate_supported_densify_args(densify_args)
    ini_metadata = None
    overrides = merge_ini_overrides(config, job.run_spec, job.scene_spec, arguments.ini_override)
    if overrides:
        configured_ini = Path(argument_value(densify_args, "--dense-config-file", "Densify.ini"))
        if not configured_ini.is_absolute():
            configured_ini = command_work_dir / configured_ini
        generated_ini = job.run_dir / "generated" / "Densify.sweep.ini"
        ini_metadata = render_ini_override(configured_ini, generated_ini, overrides)
        densify_args = replace_argument_value(densify_args, "--dense-config-file", str(generated_ini))
    endpoint = job.profile == "endpoint"
    run_binary = dmap_dev.densify_binary(config, instrumented=not endpoint)
    command = [
        str(run_binary),
        "--working-folder", str(command_work_dir),
        "--input-file", str(local_mvs),
        "--output-file", str(output_mvs),
    ]
    if endpoint:
        densify_args = dmap_dev.without_value_arguments(densify_args, dmap_dev.OBSERVER_VALUE_ARGUMENTS)
    else:
        level = (
            "maps"
            if job.profile == "deep"
            else "prefilter"
            if job.profile == "prefilter"
            else "summary"
        )
        command.extend([
            "--dmap-instrumentation-dir", str(job.run_dir / "dmap_instrumentation"),
            "--dmap-instrumentation-level", level,
            "--dmap-instrumentation-sample-rate", "1",
            "--dmap-instrumentation-write-maps", "1" if job.profile == "deep" else "0",
        ])
    if not dmap_dev.has_argument(densify_args, "--fusion-mode"):
        command.extend(["--fusion-mode", "1"])
    command.extend(densify_args)
    return command, command_work_dir, run_binary, ini_metadata


def finalize_capture(
    job: SweepJob,
    command_work_dir: Path,
    run_binary: Path,
) -> tuple[bool, str]:
    depth_dir = job.run_dir / "depth_maps"
    depth_dir.mkdir(exist_ok=True)
    for dmap in sorted(command_work_dir.glob("depth*.dmap")):
        dmap_dev.hardlink_or_copy(str(dmap), str(depth_dir / dmap.name))
    if job.profile == "endpoint":
        dmap_dev.write_json(job.run_dir / "endpoint_metadata.json", {
            "schema_name": "openmvs.dmap.endpoint_capture",
            "schema_version": 1,
            "capture_profile": "endpoint",
            "instrumentation_compiled": False,
            "instrumentation_output_present": False,
            "executable": str(run_binary.resolve()),
            "executable_sha256": dmap_drilldown.file_digest(run_binary),
            "dmap_count": len(list(depth_dir.glob("depth*.dmap"))),
        })
    integrity.write_capture_artifact_closure(job.run_dir, job.profile)
    return dmap_dev.validate_completed_run_mode(job.run_dir, job.mode)


def dmap_set_identity(directory: Path) -> dict[str, Any]:
    rows = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.glob("depth*.dmap"))
    ]
    return {"count": len(rows), "sha256": stable_digest(rows), "files": rows}


def dmap_image_id(name: str) -> int | None:
    match = re.fullmatch(r"depth(\d+)\.dmap", name)
    return int(match.group(1)) if match else None


def requested_instrumentation_image_list(run_dir: Path) -> str | None:
    try:
        repro = json.loads((run_dir / "repro.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    command = repro.get("command") or []
    if not isinstance(command, list):
        return None
    return argument_value(
        [str(value) for value in command], "--dmap-instrumentation-image-list", "all"
    )


def parse_requested_instrumentation_image_ids(value: str | None) -> list[int] | None:
    if value is None:
        raise RuntimeError("cannot resolve the requested instrumentation image list")
    normalized = value.strip()
    if normalized.casefold() == "all":
        return None
    tokens = normalized.split(",")
    if not tokens or any(re.fullmatch(r"\s*\d+\s*", token) is None for token in tokens):
        raise RuntimeError(
            "instrumented-image-only retention requires 'all' or a comma-separated "
            f"numeric image list, got {value!r}"
        )
    return sorted({int(token.strip()) for token in tokens})


def instrumented_image_evidence(run_dir: Path) -> tuple[list[int], list[dict[str, Any]]]:
    instrumentation_dir = run_dir / "dmap_instrumentation"
    evidence: list[dict[str, Any]] = []
    selected: set[int] = set()
    for estimation_stage, geometric_iteration, stage_root in dmap_dev.instrumentation_stage_roots(
        instrumentation_dir
    ):
        for frame_dir in sorted((stage_root / "depthmaps").glob("*")):
            if not frame_dir.is_dir():
                continue
            summary_path = frame_dir / "summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                image_id = int(summary["image_id"])
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot resolve instrumented image ID from {summary_path}: {exc}"
                ) from exc
            if image_id < 0:
                raise RuntimeError(f"invalid instrumented image ID {image_id} in {summary_path}")
            selected.add(image_id)
            evidence.append({
                "estimation_stage": estimation_stage,
                "geometric_iteration": geometric_iteration,
                "frame": frame_dir.name,
                "image_id": image_id,
                "summary": str(summary_path.relative_to(run_dir)),
                "summary_sha256": sha256_file(summary_path),
            })
    if not selected:
        raise RuntimeError("instrumented-image-only retention found no instrumented image IDs")
    return sorted(selected), evidence


def compact_validated_run(job: SweepJob) -> dict[str, Any]:
    before_valid, before_reason = dmap_dev.validate_completed_run_mode(job.run_dir, job.mode)
    if not before_valid:
        raise RuntimeError(f"cannot compact invalid capture: {before_reason}")
    policy = resolved_retention_policy(job)
    retained_dir = job.run_dir / "depth_maps"
    retained = {path.name: path for path in retained_dir.glob("depth*.dmap")}
    complete_identity = dmap_set_identity(retained_dir)
    complete_rows = {str(row["name"]): row for row in complete_identity["files"]}
    requested_image_list = requested_instrumentation_image_list(job.run_dir)
    requested_image_ids: list[int] | None = None
    requested_image_scope = "not_enforced"
    instrumentation_evidence: list[dict[str, Any]] = []
    if policy["lossy"]:
        selected_image_ids, instrumentation_evidence = instrumented_image_evidence(job.run_dir)
        requested_image_ids = parse_requested_instrumentation_image_ids(requested_image_list)
        requested_image_scope = "all" if requested_image_ids is None else "explicit"
        if requested_image_ids is not None and requested_image_ids != selected_image_ids:
            requested_set = set(requested_image_ids)
            evidenced_set = set(selected_image_ids)
            raise RuntimeError(
                "requested instrumentation image IDs do not match evidenced IDs: "
                f"missing_evidence_for_requested_ids={sorted(requested_set - evidenced_set)}, "
                f"unexpected_evidenced_ids={sorted(evidenced_set - requested_set)}"
            )
        unparsable = sorted(name for name in retained if dmap_image_id(name) is None)
        if unparsable:
            raise RuntimeError(
                "instrumented-image-only retention cannot classify DMAP name(s): "
                f"{unparsable}"
            )
        missing = [
            image_id for image_id in selected_image_ids
            if f"depth{image_id:04d}.dmap" not in retained
        ]
        if missing:
            raise RuntimeError(
                f"instrumented-image-only retention is missing selected DMAP IDs: {missing}"
            )
    else:
        selected_image_ids = sorted({
            image_id for image_id in (dmap_image_id(name) for name in retained)
            if image_id is not None
        })

    removed: list[dict[str, Any]] = []
    usage_before = directory_usage(job.run_dir)
    staged_dir: Path | None = None
    staged_dmaps: list[tuple[Path, Path]] = []
    try:
        if policy["lossy"]:
            staged_dir = job.run_dir.parent / f".{job.run_dir.name}.dmap-compaction-{os.getpid()}"
            if staged_dir.exists():
                raise RuntimeError(f"compaction staging path already exists: {staged_dir}")
            staged_dir.mkdir(parents=True)
            selected_set = set(selected_image_ids)
            for candidate in sorted(retained.values()):
                image_id = dmap_image_id(candidate.name)
                if image_id in selected_set:
                    continue
                row = complete_rows[candidate.name]
                destination = staged_dir / candidate.name
                removed.append({
                    "path": str(candidate.relative_to(job.run_dir)),
                    "bytes": int(row["bytes"]),
                    "sha256": str(row["sha256"]),
                    "image_id": image_id,
                    "reason": "not_instrumented_image_id",
                    "lossy": True,
                })
                os.replace(candidate, destination)
                staged_dmaps.append((candidate, destination))

        integrity.write_capture_artifact_closure(job.run_dir, job.profile)
        after_valid, after_reason = dmap_dev.validate_completed_run_mode(job.run_dir, job.mode)
        if not after_valid:
            raise RuntimeError(f"compaction invalidated capture: {after_reason}")

        for candidate in sorted((job.run_dir / "work").rglob("depth*.dmap")):
            complete_row = complete_rows.get(candidate.name)
            if complete_row is None or candidate.stat().st_size != int(complete_row["bytes"]):
                continue
            candidate_hash = sha256_file(candidate)
            if candidate_hash != complete_row["sha256"]:
                continue
            removed.append({
                "path": str(candidate.relative_to(job.run_dir)),
                "bytes": candidate.stat().st_size,
                "sha256": candidate_hash,
                "reason": "hash_identical_working_copy",
                "lossy": False,
                "equivalent_pre_compaction_path": f"depth_maps/{candidate.name}",
            })
            candidate.unlink()

        if staged_dir is not None:
            shutil.rmtree(staged_dir)
            staged_dir = None
    except Exception:
        for original, staged in reversed(staged_dmaps):
            if staged.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, original)
        if staged_dir is not None and staged_dir.exists():
            staged_dir.rmdir()
        integrity.write_capture_artifact_closure(job.run_dir, job.profile)
        raise

    usage_after = directory_usage(job.run_dir)
    retained_identity = dmap_set_identity(retained_dir)
    full_dmap_set_retained = complete_identity == retained_identity
    manifest = {
        "schema_name": RETENTION_SCHEMA_NAME,
        "schema_version": RETENTION_MANIFEST_SCHEMA_VERSION,
        "complete": True,
        "created_at": utc_timestamp(),
        "profile": job.profile,
        "mode": job.mode,
        "policy": policy,
        "lossy": bool(policy["lossy"]),
        "full_dmap_set_retained": full_dmap_set_retained,
        "full_set_parity_available": full_dmap_set_retained,
        "validation_before": {"valid": before_valid, "reason": before_reason},
        "validation_after": {"valid": after_valid, "reason": after_reason},
        "requested_instrumentation_image_list": requested_image_list,
        "requested_instrumentation_image_ids": requested_image_ids,
        "requested_instrumentation_image_scope": requested_image_scope,
        "selected_image_ids": selected_image_ids,
        "selection_source": (
            "instrumentation_frame_summaries" if policy["lossy"] else "complete_dmap_set"
        ),
        "instrumentation_evidence": instrumentation_evidence,
        "complete_pre_compaction_dmap_set": complete_identity,
        "retained_dmap_set": retained_identity,
        "retained_required_paths": [
            "command.sh", "repro.json", "stdout.log", "stderr.log", "depth_maps/", "work/",
            *([] if job.profile == "endpoint" else ["dmap_instrumentation/"]),
        ],
        "removed": removed,
        "removed_logical_bytes": sum(int(row["bytes"]) for row in removed),
        "lossy_removed_logical_bytes": sum(
            int(row["bytes"]) for row in removed if bool(row["lossy"])
        ),
        "usage_before": usage_before,
        "usage_after": usage_after,
    }
    manifest_path = job.run_dir / "retention_manifest.json"
    atomic_write_json(manifest_path, manifest)
    completion = {
        "schema_name": COMPACTED_SCHEMA_NAME,
        "schema_version": COMPACTED_COMPLETION_SCHEMA_VERSION,
        "complete": True,
        "created_at": utc_timestamp(),
        "profile": job.profile,
        "mode": job.mode,
        "policy": policy,
        "lossy": bool(policy["lossy"]),
        "validation_after": {"valid": after_valid, "reason": after_reason},
        "complete_pre_compaction_dmap_count": complete_identity["count"],
        "complete_pre_compaction_dmap_set_sha256": complete_identity["sha256"],
        "retained_dmap_count": retained_identity["count"],
        "retained_dmap_set_sha256": retained_identity["sha256"],
        "retention_manifest": "retention_manifest.json",
        "retention_manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(job.run_dir / "compacted_completion.json", completion)
    integrity.write_capture_artifact_closure(job.run_dir, job.profile)
    integrity.require_capture_artifact_closure(
        job.run_dir, job.profile, allow_legacy=False,
    )
    return completion


def validate_completed_sweep_run(job: SweepJob) -> tuple[bool, str]:
    valid, reason = dmap_dev.validate_completed_run_mode(job.run_dir, job.mode)
    if not valid:
        return valid, reason
    completion_path = job.run_dir / "compacted_completion.json"
    if not completion_path.is_file():
        return valid, reason
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"compacted completion is unreadable: {exc}"
    if completion.get("schema_name") != COMPACTED_SCHEMA_NAME or not completion.get("complete"):
        return False, "compacted completion is missing or malformed"
    schema_version = completion.get("schema_version")
    if schema_version not in {1, COMPACTED_COMPLETION_SCHEMA_VERSION}:
        return False, f"unsupported compacted completion schema version: {schema_version}"
    if completion.get("profile") != job.profile or completion.get("mode") != job.mode:
        return False, "compacted completion targets a different capture profile or mode"
    manifest_path = job.run_dir / "retention_manifest.json"
    if not manifest_path.is_file():
        return False, "compacted completion has no retention manifest"
    if completion.get("retention_manifest") != manifest_path.name:
        return False, "compacted completion references an unexpected retention manifest"
    if completion.get("retention_manifest_sha256") != sha256_file(manifest_path):
        return False, "retention manifest hash does not match compacted completion"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"retention manifest is unreadable: {exc}"
    current_identity = dmap_set_identity(job.run_dir / "depth_maps")
    if (
        current_identity["count"] != completion.get("retained_dmap_count")
        or current_identity["sha256"] != completion.get("retained_dmap_set_sha256")
    ):
        return False, "current retained DMAP set does not match compacted completion"
    if schema_version == 1:
        return True, f"{reason}; validated legacy lossless compaction manifest"
    if (
        manifest.get("schema_name") != RETENTION_SCHEMA_NAME
        or manifest.get("schema_version") != RETENTION_MANIFEST_SCHEMA_VERSION
        or not manifest.get("complete")
    ):
        return False, "retention manifest is missing or malformed"
    try:
        expected_policy = resolved_retention_policy(job)
    except ValueError as exc:
        return False, str(exc)
    if completion.get("policy") != expected_policy or manifest.get("policy") != expected_policy:
        return False, "retention manifest policy does not match the scheduled job"
    if (
        manifest.get("profile") != job.profile
        or manifest.get("mode") != job.mode
        or manifest.get("lossy") is not expected_policy["lossy"]
        or completion.get("lossy") is not expected_policy["lossy"]
    ):
        return False, "retention manifest capture metadata does not match the scheduled job"
    if not (manifest.get("validation_before") or {}).get("valid") or not (
        manifest.get("validation_after") or {}
    ).get("valid"):
        return False, "retention manifest does not record successful validation before and after"
    if manifest.get("retained_dmap_set") != current_identity:
        return False, "current retained DMAP identity does not match retention manifest"
    complete_identity = manifest.get("complete_pre_compaction_dmap_set") or {}
    if (
        complete_identity.get("count") != completion.get("complete_pre_compaction_dmap_count")
        or complete_identity.get("sha256")
        != completion.get("complete_pre_compaction_dmap_set_sha256")
    ):
        return False, "pre-compaction DMAP identity does not match compacted completion"
    removed_canonical = [
        row for row in manifest.get("removed") or []
        if row.get("reason") == "not_instrumented_image_id"
    ]
    reconstructed_rows = list(current_identity["files"])
    reconstructed_rows.extend({
        "name": Path(str(row.get("path", ""))).name,
        "bytes": row.get("bytes"),
        "sha256": row.get("sha256"),
    } for row in removed_canonical)
    reconstructed_rows.sort(key=lambda row: str(row["name"]))
    if (
        len({str(row["name"]) for row in reconstructed_rows}) != len(reconstructed_rows)
        or len(reconstructed_rows) != complete_identity.get("count")
        or stable_digest(reconstructed_rows) != complete_identity.get("sha256")
    ):
        return False, "retention manifest cannot reconstruct the complete pre-compaction DMAP set"
    if expected_policy["lossy"]:
        selected_ids = manifest.get("selected_image_ids") or []
        if not selected_ids or any(not isinstance(value, int) or value < 0 for value in selected_ids):
            return False, "lossy retention manifest has invalid selected image IDs"
        try:
            persisted_requested_ids = parse_requested_instrumentation_image_ids(
                manifest.get("requested_instrumentation_image_list")
            )
            current_requested_ids = parse_requested_instrumentation_image_ids(
                requested_instrumentation_image_list(job.run_dir)
            )
        except RuntimeError as exc:
            return False, str(exc)
        if manifest.get("requested_instrumentation_image_ids") != persisted_requested_ids:
            return False, "lossy retention manifest has inconsistent normalized requested image IDs"
        expected_scope = "all" if persisted_requested_ids is None else "explicit"
        if manifest.get("requested_instrumentation_image_scope") != expected_scope:
            return False, "lossy retention manifest has an inconsistent requested image scope"
        if current_requested_ids != persisted_requested_ids:
            return False, "requested instrumentation image list changed after compaction"
        if persisted_requested_ids is not None and persisted_requested_ids != selected_ids:
            return False, "requested instrumentation image IDs do not match evidenced IDs"
        evidence = manifest.get("instrumentation_evidence") or []
        if (
            not isinstance(evidence, list)
            or not all(isinstance(row, dict) for row in evidence)
            or {row.get("image_id") for row in evidence} != set(selected_ids)
        ):
            return False, "lossy retention manifest lacks evidence for selected image IDs"
        for row in evidence:
            relative = Path(str(row.get("summary", "")))
            if relative.is_absolute() or ".." in relative.parts:
                return False, "lossy retention manifest contains an unsafe evidence path"
            summary_path = job.run_dir / relative
            if not summary_path.is_file() or sha256_file(summary_path) != row.get("summary_sha256"):
                return False, "instrumentation evidence changed after compaction"
        retained_ids = [dmap_image_id(str(row["name"])) for row in current_identity["files"]]
        if any(value is None for value in retained_ids) or sorted(retained_ids) != sorted(selected_ids):
            return False, "retained DMAPs do not match selected instrumented image IDs"
        full_set_retained = complete_identity == current_identity
        if (
            manifest.get("full_dmap_set_retained") is not full_set_retained
            or manifest.get("full_set_parity_available") is not full_set_retained
        ):
            return False, "lossy retention manifest has inconsistent full-set availability"
    elif complete_identity != current_identity:
        return False, "lossless retention manifest does not retain the complete DMAP set"
    return True, f"{reason}; validated compacted retention manifest"


def configured_reuse_contract(
    config: dict[str, Any], jobs: list[SweepJob]
) -> dict[str, Any] | None:
    """Resolve an exact confirmation reuse set from the generated config."""

    execution = config.get("execution_contract") or {}
    raw_expected = execution.get("expected_reused_jobs")
    if raw_expected is None:
        return None
    if (
        isinstance(raw_expected, bool)
        or not isinstance(raw_expected, int)
        or raw_expected < 0
    ):
        raise ValueError("execution_contract.expected_reused_jobs must be a nonnegative integer")

    confirmation = config.get("manual_confirmation") or {}
    resolution = confirmation.get("resolution") or {}
    baseline = str(confirmation.get("baseline_run") or "")
    selected = str(resolution.get("selected_run") or "")
    raw_scenes = confirmation.get("sentinel_scenes")
    if (
        not baseline
        or not selected
        or baseline == selected
        or not isinstance(raw_scenes, list)
        or not raw_scenes
    ):
        raise ValueError(
            "expected reuse requires manual_confirmation baseline, selected run, and sentinel scenes"
        )
    sentinel_scenes = {str(value) for value in raw_scenes}
    expected = [
        job
        for job in jobs
        if job.run_label in {baseline, selected} and job.scene_id in sentinel_scenes
    ]
    expected_ids = sorted(job.job_id for job in expected)
    if len(expected_ids) != raw_expected:
        raise ValueError(
            "configured confirmation reuse matrix has "
            f"{len(expected_ids)} jobs, expected {raw_expected}"
        )
    all_ids = {job.job_id for job in jobs}
    if len(all_ids) != len(jobs):
        raise ValueError("confirmation reuse contract requires unique sweep job IDs")
    record = {
        "schema_name": "openmvs.dmap.sweep_reuse_contract",
        "schema_version": 1,
        "expected_reused_jobs": raw_expected,
        "expected_new_jobs": len(jobs) - raw_expected,
        "baseline_run": baseline,
        "selected_run": selected,
        "sentinel_scenes": sorted(sentinel_scenes),
        "expected_reused_job_ids": expected_ids,
        "expected_new_job_ids": sorted(all_ids - set(expected_ids)),
    }
    record["definition_sha256"] = stable_digest(record)
    return record


def reuse_contract_preflight(
    jobs: list[SweepJob],
    definition: dict[str, Any],
    prior_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate reuse inputs before a confirmation plan or execution starts."""

    expected_ids = set(definition.get("expected_reused_job_ids") or [])
    prior_jobs = (prior_ledger or {}).get("jobs") or {}
    completed_new_ids = {
        str(job_id)
        for job_id, row in prior_jobs.items()
        if isinstance(row, dict)
        and row.get("status") == "complete"
        and row.get("reused") is not True
        and str(job_id) not in expected_ids
    }
    evidence: list[dict[str, Any]] = []
    failures: list[str] = []
    valid_expected: list[str] = []
    valid_completed_new: list[str] = []
    invalid_new_outputs: list[str] = []
    for job in jobs:
        nonempty = job.run_dir.is_dir() and any(job.run_dir.iterdir())
        valid, reason = (
            validate_completed_sweep_run(job)
            if nonempty
            else (False, "run directory is absent or empty")
        )
        expected_reuse = job.job_id in expected_ids
        allowed_completed_new = job.job_id in completed_new_ids
        if expected_reuse:
            if valid:
                valid_expected.append(job.job_id)
            else:
                failures.append(
                    f"expected reusable job {job.job_id} is invalid: {reason}"
                )
        elif valid:
            if allowed_completed_new:
                valid_completed_new.append(job.job_id)
            else:
                failures.append(
                    f"job {job.job_id} is unexpectedly reusable before confirmation"
                )
        elif nonempty:
            # Invalid partial output will be quarantined and executed as a new job.
            invalid_new_outputs.append(job.job_id)
        evidence.append({
            "job_id": job.job_id,
            "run": job.run_label,
            "scene_id": job.scene_id,
            "run_dir": str(job.run_dir),
            "expected_reuse": expected_reuse,
            "prior_completed_new": allowed_completed_new,
            "nonempty": nonempty,
            "valid": valid,
            "validation": reason,
        })
    missing_definition_ids = sorted(expected_ids - {job.job_id for job in jobs})
    if missing_definition_ids:
        failures.append(
            "reuse definition references unknown jobs: " + ", ".join(missing_definition_ids)
        )
    result = {
        "valid": not failures,
        "expected_reused_jobs": int(definition["expected_reused_jobs"]),
        "valid_expected_reused_job_ids": sorted(valid_expected),
        "valid_completed_new_job_ids": sorted(valid_completed_new),
        "invalid_partial_new_job_ids": sorted(invalid_new_outputs),
        "failed_checks": failures,
        "evidence": evidence,
    }
    result["preflight_sha256"] = stable_digest(result)
    return result


def reuse_contract_postcheck(
    ledger: dict[str, Any], definition: dict[str, Any]
) -> dict[str, Any]:
    """Require the scheduler to reuse exactly the predeclared confirmation jobs."""

    jobs = ledger.get("jobs") or {}
    expected_reused = set(definition.get("expected_reused_job_ids") or [])
    expected_new = set(definition.get("expected_new_job_ids") or [])
    actual_reused = {
        str(job_id)
        for job_id, row in jobs.items()
        if isinstance(row, dict)
        and row.get("status") == "complete"
        and row.get("reused") is True
    }
    actual_new = {
        str(job_id)
        for job_id, row in jobs.items()
        if isinstance(row, dict)
        and row.get("status") == "complete"
        and row.get("reused") is not True
    }
    missing_reused = sorted(expected_reused - actual_reused)
    unexpected_reused = sorted(actual_reused - expected_reused)
    missing_new = sorted(expected_new - actual_new)
    unexpected_new = sorted(actual_new - expected_new)
    failures: list[str] = []
    if missing_reused:
        failures.append("expected jobs were not reused: " + ", ".join(missing_reused))
    if unexpected_reused:
        failures.append("unexpected jobs were reused: " + ", ".join(unexpected_reused))
    if missing_new:
        failures.append("expected new jobs are incomplete: " + ", ".join(missing_new))
    if unexpected_new:
        failures.append("unexpected jobs executed as new: " + ", ".join(unexpected_new))
    result = {
        "valid": not failures,
        "expected_reused_jobs": len(expected_reused),
        "actual_reused_jobs": len(actual_reused),
        "expected_new_jobs": len(expected_new),
        "actual_new_jobs": len(actual_new),
        "actual_reused_job_ids": sorted(actual_reused),
        "actual_new_job_ids": sorted(actual_new),
        "failed_checks": failures,
    }
    result["postcheck_sha256"] = stable_digest(result)
    return result


def transient_failure(result: ProcessResult, stderr_path: Path) -> bool:
    if result.timed_out or result.stop_requested or result.return_code in TRANSIENT_RETURN_CODES:
        return not result.stop_requested
    try:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(pattern in stderr for pattern in TRANSIENT_STDERR_PATTERNS)


def classify_scene_capture_failure(
    job: SweepJob, result: ProcessResult, validation: str
) -> dict[str, Any] | None:
    if (
        result.return_code != 0
        or result.timed_out
        or result.terminated
        or result.killed
        or result.stop_requested
        or validation != "run_metadata.json is missing or malformed"
        or any((job.run_dir / "depth_maps").glob("depth*.dmap"))
        or (job.run_dir / "dmap_instrumentation" / "run_metadata.json").is_file()
    ):
        return None
    logs = sorted(
        (
            path for path in (job.run_dir / "work").rglob("DensifyPointCloud*.log")
            if path.is_file()
        ),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if not logs:
        return None
    log_path = logs[-1]
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    observations = [
        (int(match.group(1)), int(match.group(2)))
        for match in re.finditer(
            r"Reference image\s+(\d+)\s+paired with\s+(\d+)\s+views?", log_text
        )
    ]
    if not observations or any(view_count != 0 for _image_id, view_count in observations):
        return None
    return {
        "kind": SCENE_FAILURE_ZERO_NEIGHBOR_VIEWS,
        "scope": "scene",
        "terminal": True,
        "continuable": True,
        "promotion_eligible": False,
        "validation": validation,
        "evidence": {
            "log_relative_path": str(log_path.relative_to(job.run_dir)),
            "log_sha256": sha256_file(log_path),
            "pair_observation_count": len(observations),
            "reference_image_ids": sorted({image_id for image_id, _count in observations}),
            "maximum_neighbor_view_count": 0,
        },
    }


def recorded_scene_failure(row: dict[str, Any]) -> bool:
    failure = row.get("scene_failure") or {}
    return (
        row.get("status") == "scene_failed"
        and failure.get("scope") == "scene"
        and failure.get("terminal") is True
        and failure.get("continuable") is True
        and failure.get("promotion_eligible") is False
    )


def learned_estimate_bytes(ledger: dict[str, Any], job: SweepJob) -> int:
    observed = [
        int(row.get("artifact_usage", {}).get("allocated_bytes", 0))
        for row in ledger.get("jobs", {}).values()
        if row.get("profile") == job.profile
        and row.get("scene_id") == job.scene_id
        and row.get("status") == "complete"
    ]
    return max(job.estimated_output_bytes, int(max(observed, default=0) * 1.25))


def admission_reason(
    arguments: Arguments,
    ledger: dict[str, Any],
    job: SweepJob,
    deadline_monotonic: float,
    filesystem: Path,
) -> str | None:
    cap_check = artifact_tree_cap_check(
        arguments, phase="job_admission", job_id=job.job_id
    )
    record_artifact_tree_cap_check(ledger, cap_check)
    cap_reason = artifact_tree_cap_reason(cap_check)
    if cap_reason is not None:
        return cap_reason
    capture_remaining = remaining_capture_budget_seconds(arguments, ledger)
    if (
        capture_remaining is not None
        and capture_remaining <= CAPTURE_BUDGET_STOP_MARGIN_SECONDS
    ):
        return (
            "capture admission failed: "
            f"{capture_remaining:.1f}s remain inside the cumulative capture budget"
        )
    remaining = deadline_monotonic - time.monotonic()
    required_time = job.timeout_seconds + arguments.finalization_reserve_minutes * 60.0
    if remaining < required_time:
        return f"time admission failed: {remaining:.1f}s remain, {required_time:.1f}s required"
    free_bytes = shutil.disk_usage(filesystem).free
    required_free = int(
        (arguments.free_space_floor_gb + arguments.finalization_reserve_gb) * 1024**3
    ) + learned_estimate_bytes(ledger, job)
    if free_bytes < required_free:
        return f"storage admission failed: {free_bytes} bytes free, {required_free} required"
    return None


def write_repro(
    job: SweepJob,
    command: list[str],
    result: ProcessResult,
    ini_metadata: dict[str, Any] | None,
    *,
    effective_timeout_seconds: float,
    effective_term_grace_seconds: float,
    capture_budget_limited: bool,
) -> None:
    dmap_dev.write_json(job.run_dir / "repro.json", {
        "command": command,
        "cwd": str(REPO_ROOT),
        "return_code": result.return_code,
        "elapsed_seconds": result.elapsed_seconds,
        "dry_run": False,
        "git_commit": dmap_dev.git_hash(),
        "scheduler": {
            "schema_name": SCHEDULE_SCHEMA_NAME,
            "job_id": job.job_id,
            "timed_out": result.timed_out,
            "terminated": result.terminated,
            "killed": result.killed,
            "configured_timeout_seconds": job.timeout_seconds,
            "effective_timeout_seconds": effective_timeout_seconds,
            "effective_term_grace_seconds": effective_term_grace_seconds,
            "capture_budget_limited": capture_budget_limited,
            "supervisor_command": result.supervisor_command,
            "process_identity": result.process_identity,
            "ini_override": ini_metadata,
            "argument_overrides": job.argument_overrides,
            "retention_policy": job.retention_policy,
        },
    })


def execute_capture_attempt(
    arguments: Arguments,
    config: dict[str, Any],
    job: SweepJob,
    heartbeat_path: Path,
    stop_event: threading.Event,
    ledger_path: Path | None = None,
    *,
    timeout_seconds: float | None = None,
    term_grace_seconds: float | None = None,
    capture_budget_limited: bool = False,
) -> tuple[ProcessResult, bool, str]:
    command, command_work_dir, run_binary, ini_metadata = build_capture_command(
        arguments, config, job
    )
    job.run_dir.mkdir(parents=True, exist_ok=True)
    (job.run_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    effective_timeout = job.timeout_seconds if timeout_seconds is None else timeout_seconds
    effective_term_grace = (
        arguments.term_grace_seconds if term_grace_seconds is None else term_grace_seconds
    )
    result = run_process_group(
        command,
        cwd=REPO_ROOT,
        stdout_path=job.run_dir / "stdout.log",
        stderr_path=job.run_dir / "stderr.log",
        timeout_seconds=effective_timeout,
        term_grace_seconds=effective_term_grace,
        heartbeat_seconds=arguments.heartbeat_seconds,
        heartbeat_path=heartbeat_path,
        job_id=job.job_id,
        stop_event=stop_event,
        schedule_path=ledger_path,
    )
    write_repro(
        job, command, result, ini_metadata,
        effective_timeout_seconds=effective_timeout,
        effective_term_grace_seconds=effective_term_grace,
        capture_budget_limited=capture_budget_limited,
    )
    valid = False
    validation = "process did not exit successfully"
    if result.return_code == 0 and not result.timed_out and not result.stop_requested:
        valid, validation = finalize_capture(job, command_work_dir, run_binary)
    return result, valid, validation


def run_job(
    arguments: Arguments,
    config: dict[str, Any],
    root: Path,
    job: SweepJob,
    ledger: dict[str, Any],
    ledger_path: Path,
    heartbeat_path: Path,
    stop_event: threading.Event,
    deadline_monotonic: float | None = None,
) -> bool:
    job_row = ledger["jobs"][job.job_id]
    attempts = job_row.get("attempts") or []
    if job_row.get("status") == "scene_failed":
        return False
    if job_row.get("status") == "failed" and attempts:
        last_attempt = attempts[-1]
        if not last_attempt.get("transient") or len(attempts) >= 1 + arguments.retry_transient_failures:
            return False
    if job.run_dir.exists() and any(job.run_dir.iterdir()):
        valid, reason = validate_completed_sweep_run(job)
        if valid:
            job_row.update(status="complete", reused=True, validation=reason, artifact_usage=directory_usage(job.run_dir))
            if arguments.compact and not (job.run_dir / "compacted_completion.json").is_file():
                job_row["compaction"] = compact_validated_run(job)
            save_ledger(ledger_path, ledger)
            return True
        quarantine = quarantine_path(job.run_dir, root, job.job_id, f"resume found invalid output: {reason}")
        job_row["recovered_quarantine"] = str(quarantine) if quarantine else None

    maximum_attempts = 1 + max(0, arguments.retry_transient_failures)
    while len(job_row["attempts"]) < maximum_attempts:
        attempt_number = len(job_row["attempts"]) + 1
        if attempt_number > 1 and deadline_monotonic is not None:
            reason = admission_reason(
                arguments, ledger, job, deadline_monotonic, root.parent
            )
            if reason is not None:
                job_row.update(
                    status="deferred",
                    admission_reason=f"retry {reason}",
                )
                save_ledger(ledger_path, ledger)
                return False
        capture_remaining = remaining_capture_budget_seconds(arguments, ledger)
        effective_timeout = job.timeout_seconds
        effective_term_grace = arguments.term_grace_seconds
        capture_budget_limited = False
        if capture_remaining is not None:
            if capture_remaining <= CAPTURE_BUDGET_STOP_MARGIN_SECONDS:
                job_row.update(
                    status="deferred",
                    admission_reason=(
                        "capture admission failed: "
                        f"{capture_remaining:.1f}s remain inside the cumulative capture budget"
                    ),
                )
                save_ledger(ledger_path, ledger)
                return False
            if (
                job.timeout_seconds + arguments.term_grace_seconds
                + CAPTURE_BUDGET_STOP_MARGIN_SECONDS > capture_remaining
            ):
                effective_timeout = capture_remaining - CAPTURE_BUDGET_STOP_MARGIN_SECONDS
                effective_term_grace = 0.0
                capture_budget_limited = True
        attempt = {
            "attempt": attempt_number,
            "started_at": utc_timestamp(),
            "status": "running",
            "elapsed_seconds": 0.0,
            "configured_timeout_seconds": job.timeout_seconds,
            "effective_timeout_seconds": effective_timeout,
            "effective_term_grace_seconds": effective_term_grace,
            "supervisor_kind": "coreutils_timeout",
            "capture_budget_limited": capture_budget_limited,
            "capture_budget_remaining_before_seconds": capture_remaining,
        }
        job_row["attempts"].append(attempt)
        job_row["status"] = "running"
        save_ledger(ledger_path, ledger)
        try:
            result, valid, validation = execute_capture_attempt(
                arguments, config, job, heartbeat_path, stop_event, ledger_path,
                timeout_seconds=effective_timeout,
                term_grace_seconds=effective_term_grace,
                capture_budget_limited=capture_budget_limited,
            )
        except Exception as exc:
            validation = f"capture attempt raised {type(exc).__name__}: {exc}"
            observed_elapsed = attempt.get("elapsed_seconds", 0.0)
            try:
                durable = json.loads(ledger_path.read_text(encoding="utf-8"))
                durable_attempt = (
                    durable["jobs"][job.job_id]["attempts"][attempt_number - 1]
                )
                durable_elapsed = durable_attempt.get("elapsed_seconds", 0.0)
                if (
                    isinstance(durable_elapsed, (int, float))
                    and not isinstance(durable_elapsed, bool)
                    and math.isfinite(float(durable_elapsed))
                    and float(durable_elapsed) >= 0.0
                ):
                    observed_elapsed = max(
                        float(observed_elapsed), float(durable_elapsed)
                    )
            except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError):
                pass
            charged_elapsed = max(
                float(observed_elapsed),
                float(effective_timeout) + float(effective_term_grace),
            )
            attempt.update({
                "finished_at": utc_timestamp(),
                "status": "failed",
                "elapsed_seconds": charged_elapsed,
                "observed_elapsed_seconds": float(observed_elapsed),
                "capture_accounting": "conservative_supervision_bound",
                "transient": False,
                "validation": validation,
            })
            quarantine = quarantine_path(job.run_dir, root, job.job_id, validation)
            attempt["quarantine"] = str(quarantine) if quarantine else None
            job_row.update(status="failed", validation=validation)
            save_ledger(ledger_path, ledger)
            return False
        attempt.update({
            "finished_at": utc_timestamp(),
            "status": "complete" if valid else "failed",
            "return_code": result.return_code,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "terminated": result.terminated,
            "killed": result.killed,
            "validation": validation,
        })
        capture_limit = (
            None if arguments.capture_budget_hours is None
            else float(arguments.capture_budget_hours) * 3600.0
        )
        capture_elapsed = cumulative_capture_elapsed_seconds(ledger)
        capture_budget_exceeded = capture_limit is not None and capture_elapsed > capture_limit
        if capture_budget_exceeded:
            valid = False
            validation = (
                f"cumulative capture budget exceeded: {capture_elapsed:.6f}s > "
                f"{capture_limit:.6f}s"
            )
            attempt.update(status="failed", validation=validation, transient=False)
        if valid:
            job_row.update(
                status="complete",
                reused=False,
                validation=validation,
                artifact_usage=directory_usage(job.run_dir),
            )
            if arguments.compact:
                job_row["compaction"] = compact_validated_run(job)
                job_row["artifact_usage"] = directory_usage(job.run_dir)
            save_ledger(ledger_path, ledger)
            return True
        is_transient = (
            False if capture_budget_exceeded
            else transient_failure(result, job.run_dir / "stderr.log")
        )
        attempt["transient"] = is_transient
        scene_failure = (
            None if is_transient else classify_scene_capture_failure(job, result, validation)
        )
        quarantine = quarantine_path(job.run_dir, root, job.job_id, validation)
        attempt["quarantine"] = str(quarantine) if quarantine else None
        if scene_failure is not None:
            scene_failure["quarantine"] = str(quarantine) if quarantine else None
            attempt.update(status="scene_failed", scene_failure=scene_failure)
            job_row.update(
                status="scene_failed",
                validation=validation,
                scene_failure=scene_failure,
            )
        else:
            job_row["status"] = (
                "retry_pending"
                if is_transient and attempt_number < maximum_attempts
                else "failed"
            )
        save_ledger(ledger_path, ledger)
        if scene_failure is not None or stop_event.is_set() or not is_transient:
            return False
    return False


def valid_report(report_dir: Path) -> tuple[bool, str]:
    markdown = report_dir / "01_development_report.md"
    model_path = report_dir / "report_model.json"
    if not markdown.is_file() or not model_path.is_file():
        return False, "canonical Markdown or report model is missing"
    markdown_validation = dmap_dev.validate_report(markdown, write_sidecar=False)
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model_validation = dmap_report_model.validate_report_model(model, report_dir)
    except Exception as exc:
        return False, f"report model validation failed: {exc}"
    if not markdown_validation.get("valid") or not model_validation.get("valid"):
        return False, "Markdown or structured report validation failed"
    return True, "validated canonical Markdown and structured report model"


def accuracy_ledger_status(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "accuracy_ledger.csv"
    if not path.is_file():
        return {"available": False, "candidates_hidden": False, "reason": "accuracy-first ledger unavailable"}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        return {"available": False, "candidates_hidden": False, "reason": str(exc)}
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("candidate") or "")
        try:
            rank = int(str(row.get("accuracy_rank") or ""))
            scene_count = int(str(row.get("scene_count") or ""))
            primary_scene_metric_rows = int(
                str(row.get("primary_scene_metric_rows") or "")
            )
        except ValueError:
            continue
        if (
            not candidate
            or rank <= 0
            or scene_count <= 0
            or primary_scene_metric_rows <= 0
        ):
            continue
        normalized = dict(row)
        normalized["candidate"] = candidate
        normalized["accuracy_rank"] = rank
        normalized["scene_count"] = scene_count
        normalized["primary_scene_metric_rows"] = primary_scene_metric_rows
        normalized_rows.append(normalized)
    normalized_rows.sort(key=lambda row: (int(row["accuracy_rank"]), str(row["candidate"])))
    if not normalized_rows:
        return {
            "available": False,
            "candidates_hidden": False,
            "path": str(path),
            "sha256": sha256_file(path),
            "reason": (
                "accuracy-first ledger has no ranked candidate with a positive "
                "scene count and primary metric evidence"
            ),
        }
    return {
        "available": True,
        "candidates_hidden": False,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": normalized_rows,
    }


def report_evidence_digest(ledger: dict[str, Any]) -> str:
    evidence = {
        "identity_sha256": (ledger.get("identity") or {}).get("sha256"),
        "completed_job_ids": sorted(
            job_id for job_id, row in (ledger.get("jobs") or {}).items()
            if row.get("status") == "complete"
        ),
        "job_states": [
            {"job_id": job_id, "status": str(row.get("status") or "unknown")}
            for job_id, row in sorted((ledger.get("jobs") or {}).items())
        ],
    }
    return stable_digest(evidence)


def final_report_reuse_status(
    report_dir: Path, ledger: dict[str, Any]
) -> tuple[bool, str, str]:
    evidence_digest = report_evidence_digest(ledger)
    valid, reason = valid_report(report_dir)
    if not valid:
        return False, reason, evidence_digest
    binding_path = report_dir / "sweep_report_binding.json"
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"report has no valid sweep evidence binding: {exc}", evidence_digest
    if (
        binding.get("schema_name") != REPORT_BINDING_SCHEMA_NAME
        or binding.get("schema_version") != REPORT_BINDING_SCHEMA_VERSION
    ):
        return False, "report has an unsupported sweep evidence binding", evidence_digest
    if binding.get("evidence_digest") != evidence_digest:
        return False, "report is stale for the current completed-job evidence", evidence_digest
    if (ledger.get("report") or {}).get("evidence_digest") != evidence_digest:
        return False, "ledger does not bind the report to current completed-job evidence", evidence_digest
    source_valid, source_reason, source_record = validate_report_source_provenance(report_dir)
    if not source_valid or source_record is None:
        return False, source_reason, evidence_digest
    if binding.get("report_source_sha256") != source_record.get("sha256"):
        return False, "sweep binding does not match report-generator source", evidence_digest
    if (ledger.get("report") or {}).get("report_source_sha256") != source_record.get("sha256"):
        return False, "ledger does not bind the report-generator source", evidence_digest
    return True, reason, evidence_digest


def write_report_binding(
    report_dir: Path,
    ledger: dict[str, Any],
    evidence_digest: str,
    report_source: dict[str, Any],
) -> None:
    atomic_write_json(report_dir / "sweep_report_binding.json", {
        "schema_name": REPORT_BINDING_SCHEMA_NAME,
        "schema_version": REPORT_BINDING_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "schedule_identity_sha256": (ledger.get("identity") or {}).get("sha256"),
        "evidence_digest": evidence_digest,
        "report_source_sha256": report_source["sha256"],
        "completed_job_ids": sorted(
            job_id for job_id, row in (ledger.get("jobs") or {}).items()
            if row.get("status") == "complete"
        ),
    })


def promotion_for_target(stages: list[dict[str, Any]], target_stage: str) -> tuple[str, dict[str, Any]] | None:
    matches = []
    for stage in stages:
        promotion = stage.get("promotion")
        if isinstance(promotion, dict) and str(promotion.get("next_stage") or "") == target_stage:
            matches.append((str(promotion.get("source_stage") or stage["name"]), promotion))
    if len(matches) > 1:
        raise ValueError(f"multiple promotion policies target stage {target_stage}")
    return matches[0] if matches else None


def select_promoted_runs(
    *,
    target_stage: str,
    source_stage: str,
    policy: dict[str, Any],
    target_jobs: list[SweepJob],
    accuracy: dict[str, Any],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    candidate_jobs = [job for job in target_jobs if not job.always_run]
    planned_candidate_labels = sorted({job.run_label for job in candidate_jobs})
    always_labels = sorted({job.run_label for job in target_jobs if job.always_run})
    previously_promoted_stages = {
        str(record.get("target_stage"))
        for record in (ledger.get("promotions") or {}).values()
        if record.get("selected_runs")
    }
    configured_eligible_stages = {
        str(value) for value in policy.get("eligible_stages") or []
    }
    eligible_stages = {source_stage, *configured_eligible_stages}
    if target_stage == "finalists" and not configured_eligible_stages:
        eligible_stages.update(
            stage for stage in previously_promoted_stages if stage.endswith("_confirm")
        )
    completed_by_run = {
        str(row.get("run"))
        for row in (ledger.get("jobs") or {}).values()
        if row.get("status") == "complete" and str(row.get("stage")) in eligible_stages
    }
    scene_failed_by_run = {
        str(row.get("run"))
        for row in (ledger.get("jobs") or {}).values()
        if recorded_scene_failure(row) and str(row.get("stage")) in eligible_stages
    }
    scene_failed_candidates = sorted(
        set(planned_candidate_labels) & scene_failed_by_run
    )
    candidate_labels = sorted(
        (set(planned_candidate_labels) & completed_by_run) - scene_failed_by_run
    )
    rows_by_candidate = {
        str(row["candidate"]): row
        for row in accuracy.get("rows") or []
        if str(row.get("candidate")) in candidate_labels
    }
    ranked = sorted(
        rows_by_candidate.values(),
        key=lambda row: (int(row["accuracy_rank"]), str(row["candidate"])),
    )
    selected_candidates: list[str] = []
    if bool(policy.get("family_winners", False)):
        family_by_run = {job.run_label: job.family for job in candidate_jobs}
        seen_families: set[str] = set()
        for row in ranked:
            label = str(row["candidate"])
            family = family_by_run[label]
            if family in seen_families:
                continue
            seen_families.add(family)
            selected_candidates.append(label)
    else:
        selected_candidates = [str(row["candidate"]) for row in ranked]
    raw_top_n = policy.get("top_n")
    if raw_top_n is not None:
        top_n = int(raw_top_n)
        if top_n <= 0:
            raise ValueError(f"promotion top_n for {target_stage} must be positive")
        selected_candidates = selected_candidates[:top_n]
    required = bool(policy.get("required", True))
    if required and planned_candidate_labels and not selected_candidates:
        raise RuntimeError(
            f"required promotion from {source_stage} to {target_stage} has no ranked candidates"
        )
    selected = sorted(set(always_labels + selected_candidates))
    return {
        "source_stage": source_stage,
        "target_stage": target_stage,
        "required": required,
        "top_n": int(raw_top_n) if raw_top_n is not None else None,
        "family_winners": bool(policy.get("family_winners", False)),
        "accuracy_ledger": {
            key: value for key, value in accuracy.items() if key != "rows"
        },
        "ranked_candidates": ranked,
        "planned_candidates": planned_candidate_labels,
        "eligible_candidates": candidate_labels,
        "eligible_stages": sorted(eligible_stages),
        "ineligible_candidates": sorted(set(planned_candidate_labels) - set(candidate_labels)),
        "scene_failed_candidates": scene_failed_candidates,
        "always_run": always_labels,
        "selected_runs": selected,
        "not_promoted": sorted(set(planned_candidate_labels) - set(selected_candidates)),
        "candidates_hidden": False,
    }


def generate_final_report(
    arguments: Arguments,
    config: dict[str, Any],
    root: Path,
    report_dir: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    heartbeat_path: Path,
    stop_event: threading.Event,
) -> bool:
    with CampaignReportLock(campaign_report_lock_path(root)):
        return _generate_final_report_unlocked(
            arguments,
            config,
            root,
            report_dir,
            ledger,
            ledger_path,
            heartbeat_path,
            stop_event,
        )


def _generate_final_report_unlocked(
    arguments: Arguments,
    config: dict[str, Any],
    root: Path,
    report_dir: Path,
    ledger: dict[str, Any],
    ledger_path: Path,
    heartbeat_path: Path,
    stop_event: threading.Event,
) -> bool:
    reusable, reason, evidence_digest = final_report_reuse_status(report_dir, ledger)
    if reusable:
        _source_valid, _source_reason, report_source = validate_report_source_provenance(
            report_dir
        )
        ledger["promotion"] = {
            "policy": "accuracy_first_report_ledger",
            **accuracy_ledger_status(report_dir),
        }
        requirements_met = (
            not arguments.require_accuracy_ledger
            or bool(ledger["promotion"].get("available"))
        )
        ledger["report"].update(
            status="complete" if requirements_met else "incomplete_accuracy",
            valid=True,
            requirements_met=requirements_met,
            reused=True,
            validation=reason,
            path=str(report_dir),
            evidence_digest=evidence_digest,
            report_source_sha256=(
                report_source.get("sha256") if report_source else None
            ),
        )
        save_ledger(ledger_path, ledger)
        return requirements_met
    if report_dir.exists() and any(report_dir.iterdir()):
        quarantine = quarantine_path(report_dir, root, "master-report", reason)
        ledger["report"]["quarantine"] = str(quarantine) if quarantine else None
    command = [
        sys.executable,
        str(SCRIPT_DIR / "dmap_dev.py"),
        "report",
        "--config", str(Path(str(config["_config_path"])).resolve()),
        "--output-dir", str(report_dir),
    ]
    with tempfile.TemporaryDirectory(prefix="openmvs-dmap-report-source-") as directory:
        source_dir = Path(directory)
        source_before = create_report_source_snapshot(source_dir, "before.tar.zst")
        result = run_process_group(
            command,
            cwd=REPO_ROOT,
            stdout_path=root / "sweep" / "report.stdout.log",
            stderr_path=root / "sweep" / "report.stderr.log",
            timeout_seconds=arguments.report_timeout_minutes * 60.0,
            term_grace_seconds=arguments.term_grace_seconds,
            heartbeat_seconds=arguments.heartbeat_seconds,
            heartbeat_path=heartbeat_path,
            job_id="master-report",
            stop_event=stop_event,
            schedule_path=ledger_path,
        )
        source_after = create_report_source_snapshot(source_dir, "after.tar.zst")
        valid, reason = valid_report(report_dir)
        report_source = None
        if valid:
            try:
                report_source = publish_report_source_provenance(
                    report_dir, source_before, source_after
                )
            except Exception as exc:
                valid = False
                reason = f"report source provenance failed: {exc}"
    if valid and report_source is not None:
        try:
            write_report_binding(report_dir, ledger, evidence_digest, report_source)
            integrity.write_report_tree_closure(report_dir)
            valid, reason = valid_report(report_dir)
        except Exception as exc:
            valid = False
            reason = f"report closure finalization failed: {exc}"
    ledger["promotion"] = {
        "policy": "accuracy_first_report_ledger",
        **accuracy_ledger_status(report_dir),
    }
    requirements_met = valid and (
        not arguments.require_accuracy_ledger
        or bool(ledger["promotion"].get("available"))
    )
    ledger["report"].update({
        "status": (
            "complete" if requirements_met
            else "incomplete_accuracy" if valid
            else "failed"
        ),
        "valid": valid,
        "requirements_met": requirements_met,
        "reused": False,
        "path": str(report_dir),
        "return_code": result.return_code,
        "elapsed_seconds": result.elapsed_seconds,
        "timed_out": result.timed_out,
        "validation": reason,
        "evidence_digest": evidence_digest if valid else None,
        "report_source_sha256": report_source.get("sha256") if report_source else None,
    })
    save_ledger(ledger_path, ledger)
    return requirements_met


def stage_report_evidence(
    ledger: dict[str, Any], stage_name: str, stage_index: int
) -> dict[str, Any]:
    stage_jobs = [
        (str(job_id), row)
        for job_id, row in sorted((ledger.get("jobs") or {}).items())
        if isinstance(row, dict) and str(row.get("stage")) == stage_name
    ]
    record = {
        "schedule_identity_sha256": (ledger.get("identity") or {}).get("sha256"),
        "stage": stage_name,
        "stage_index": stage_index,
        "job_states": [
            {"job_id": job_id, "status": str(row.get("status") or "unknown")}
            for job_id, row in stage_jobs
        ],
        "completed_job_ids": [
            job_id for job_id, row in stage_jobs if row.get("status") == "complete"
        ],
    }
    record["evidence_digest"] = stable_digest(record)
    return record


def stage_report_reuse_status(
    report_dir: Path,
    ledger: dict[str, Any],
    stage_name: str,
    stage_index: int,
) -> tuple[bool, str, dict[str, Any], dict[str, Any] | None]:
    evidence = stage_report_evidence(ledger, stage_name, stage_index)
    if not evidence["job_states"]:
        return False, "stage has no selected jobs", evidence, None
    valid, reason = valid_report(report_dir)
    if not valid:
        return False, reason, evidence, None
    binding_path = report_dir / "sweep_stage_report_binding.json"
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"stage report has no valid evidence binding: {exc}", evidence, None
    if (
        binding.get("schema_name") != STAGE_REPORT_BINDING_SCHEMA_NAME
        or binding.get("schema_version") != STAGE_REPORT_BINDING_SCHEMA_VERSION
        or binding.get("schedule_identity_sha256")
        != evidence["schedule_identity_sha256"]
        or binding.get("stage") != stage_name
        or binding.get("stage_index") != stage_index
        or binding.get("evidence_digest") != evidence["evidence_digest"]
        or binding.get("completed_job_ids") != evidence["completed_job_ids"]
    ):
        return False, "stage report evidence binding is stale or mismatched", evidence, None
    source_valid, source_reason, source_record = validate_report_source_provenance(
        report_dir
    )
    if not source_valid or source_record is None:
        return False, source_reason, evidence, None
    if binding.get("report_source_sha256") != source_record.get("sha256"):
        return False, "stage report binding does not match report-generator source", evidence, None
    return True, reason, evidence, source_record


def write_stage_report_binding(
    report_dir: Path,
    evidence: dict[str, Any],
    report_source: dict[str, Any],
) -> dict[str, Any]:
    binding = {
        "schema_name": STAGE_REPORT_BINDING_SCHEMA_NAME,
        "schema_version": STAGE_REPORT_BINDING_SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "schedule_identity_sha256": evidence["schedule_identity_sha256"],
        "stage": evidence["stage"],
        "stage_index": evidence["stage_index"],
        "evidence_digest": evidence["evidence_digest"],
        "report_source_sha256": report_source["sha256"],
        "completed_job_ids": evidence["completed_job_ids"],
    }
    path = report_dir / "sweep_stage_report_binding.json"
    atomic_write_json(path, binding)
    return {**binding, "path": str(path), "sha256": sha256_file(path)}


def generate_stage_report(
    arguments: Arguments,
    config: dict[str, Any],
    root: Path,
    stage: dict[str, Any],
    stage_index: int,
    ledger: dict[str, Any],
    ledger_path: Path,
    heartbeat_path: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    with CampaignReportLock(campaign_report_lock_path(root)):
        return _generate_stage_report_unlocked(
            arguments,
            config,
            root,
            stage,
            stage_index,
            ledger,
            ledger_path,
            heartbeat_path,
            stop_event,
        )


def _generate_stage_report_unlocked(
    arguments: Arguments,
    config: dict[str, Any],
    root: Path,
    stage: dict[str, Any],
    stage_index: int,
    ledger: dict[str, Any],
    ledger_path: Path,
    heartbeat_path: Path,
    stop_event: threading.Event,
) -> dict[str, Any]:
    stage_name = str(stage["name"])
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name).strip("_") or "stage"
    report_dir = root / "reports" / "sweep_stages" / f"{stage_index:02d}_{safe_name}"
    valid, reason, evidence, report_source = stage_report_reuse_status(
        report_dir, ledger, stage_name, stage_index
    )
    binding: dict[str, Any] | None = None
    binding_path = report_dir / "sweep_stage_report_binding.json"
    if valid:
        binding = {
            "path": str(binding_path),
            "sha256": sha256_file(binding_path),
        }
    report_record: dict[str, Any] = {
        "stage": stage_name,
        "stage_index": stage_index,
        "path": str(report_dir),
        "status": "complete" if valid else "pending",
        "reused": valid,
        "validation": reason,
        "evidence_digest": evidence["evidence_digest"],
        "completed_job_ids": evidence["completed_job_ids"],
    }
    if not valid:
        if report_dir.exists() and any(report_dir.iterdir()):
            quarantine = quarantine_path(
                report_dir, root, f"stage-report-{stage_index:02d}-{safe_name}", reason
            )
            report_record["quarantine"] = str(quarantine) if quarantine else None
        command = [
            sys.executable,
            str(SCRIPT_DIR / "dmap_dev.py"),
            "report",
            "--config", str(Path(str(config["_config_path"])).resolve()),
            "--output-dir", str(report_dir),
        ]
        with tempfile.TemporaryDirectory(prefix="openmvs-dmap-report-source-") as directory:
            source_dir = Path(directory)
            source_before = create_report_source_snapshot(source_dir, "before.tar.zst")
            result = run_process_group(
                command,
                cwd=REPO_ROOT,
                stdout_path=root / "sweep" / f"stage_{stage_index:02d}.report.stdout.log",
                stderr_path=root / "sweep" / f"stage_{stage_index:02d}.report.stderr.log",
                timeout_seconds=arguments.report_timeout_minutes * 60.0,
                term_grace_seconds=arguments.term_grace_seconds,
                heartbeat_seconds=arguments.heartbeat_seconds,
                heartbeat_path=heartbeat_path,
                job_id=f"stage-report-{stage_index:02d}",
                stop_event=stop_event,
                schedule_path=ledger_path,
            )
            source_after = create_report_source_snapshot(source_dir, "after.tar.zst")
            valid, reason = valid_report(report_dir)
            if valid:
                try:
                    report_source = publish_report_source_provenance(
                        report_dir, source_before, source_after
                    )
                    binding = write_stage_report_binding(
                        report_dir, evidence, report_source
                    )
                    integrity.write_report_tree_closure(report_dir)
                    valid, reason = valid_report(report_dir)
                except Exception as exc:
                    valid = False
                    reason = f"stage report evidence binding failed: {exc}"
        report_record.update({
            "status": "complete" if valid else "failed",
            "reused": False,
            "return_code": result.return_code,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "validation": reason,
        })
    report_record["report_source_sha256"] = (
        report_source.get("sha256") if report_source else None
    )
    report_record["binding"] = binding
    report_record["accuracy_ledger"] = accuracy_ledger_status(report_dir)
    ledger.setdefault("stage_reports", {})[str(stage["name"])] = report_record
    save_ledger(ledger_path, ledger)
    return report_record


def install_stop_handlers(stop_event: threading.Event) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    return previous


def restore_stop_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def run_sweep(arguments: Arguments) -> int:
    if not finite_real(arguments.budget_hours) or arguments.budget_hours <= 0:
        raise ValueError("--budget-hours must be positive")
    if (
        not finite_real(arguments.finalization_reserve_minutes)
        or arguments.finalization_reserve_minutes < 0
    ):
        raise ValueError("--finalization-reserve-minutes must be non-negative")
    if not finite_real(arguments.term_grace_seconds) or arguments.term_grace_seconds < 0:
        raise ValueError("--term-grace-seconds must be non-negative")
    if not finite_real(arguments.heartbeat_seconds) or arguments.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive")
    timeout_values = {
        "--job-timeout-minutes": arguments.job_timeout_minutes,
        "--endpoint-timeout-minutes": arguments.endpoint_timeout_minutes,
        "--summary-timeout-minutes": arguments.summary_timeout_minutes,
        "--deep-timeout-minutes": arguments.deep_timeout_minutes,
    }
    for option, value in timeout_values.items():
        if value is not None and (not finite_real(value) or value <= 0):
            raise ValueError(f"{option} must be positive")
    if not finite_real(arguments.report_timeout_minutes) or arguments.report_timeout_minutes <= 0:
        raise ValueError("--report-timeout-minutes must be positive")
    if arguments.capture_budget_hours is not None:
        if (
            not finite_real(arguments.capture_budget_hours)
            or arguments.capture_budget_hours <= 0
        ):
            raise ValueError("--capture-budget-hours must be positive")
        required_hours = (
            arguments.capture_budget_hours
            + arguments.finalization_reserve_minutes / 60.0
        )
        if required_hours > arguments.budget_hours:
            raise ValueError(
                "--budget-hours must cover --capture-budget-hours plus the "
                "finalization reserve"
            )
    if (
        not finite_real(arguments.free_space_floor_gb)
        or not finite_real(arguments.finalization_reserve_gb)
        or not finite_real(arguments.default_job_output_gb)
        or arguments.free_space_floor_gb < 0
        or arguments.finalization_reserve_gb < 0
        or arguments.default_job_output_gb < 0
    ):
        raise ValueError("storage reserves and output estimates must be non-negative")
    if (arguments.artifact_tree_cap_root is None) != (
        arguments.artifact_tree_cap_gb is None
    ):
        raise ValueError(
            "--artifact-tree-cap-root and --artifact-tree-cap-gb must be set together"
        )
    if arguments.artifact_tree_cap_gb is not None and (
        not finite_real(arguments.artifact_tree_cap_gb)
        or arguments.artifact_tree_cap_gb <= 0
    ):
        raise ValueError("--artifact-tree-cap-gb must be positive")
    if arguments.artifact_tree_cap_root is not None:
        artifact_tree_cap_check(arguments, phase="scheduler_start")
    if (
        not isinstance(arguments.retry_transient_failures, int)
        or isinstance(arguments.retry_transient_failures, bool)
        or arguments.retry_transient_failures not in {0, 1}
    ):
        raise ValueError("--retry-transient-failures must be 0 or 1")
    config = dmap_dev.load_config(arguments.config)
    root = dmap_dev.experiment_root(config)
    root.mkdir(parents=True, exist_ok=True)
    schedule_dir = (
        dmap_dev.ensure_external_output_path(
            arguments.schedule_dir, "sweep schedule directory"
        )
        if arguments.schedule_dir is not None else root / "sweep"
    )
    if arguments.report_dir is not None:
        arguments.report_dir = dmap_dev.ensure_external_output_path(
            arguments.report_dir, "sweep report directory"
        )
    ledger_path = schedule_dir / "schedule.json"
    heartbeat_path = schedule_dir / "heartbeat.json"
    with SweepLock(schedule_dir / ".sweep.lock"):
        config, root = dmap_dev.prepare_experiment(arguments.config, False)
        jobs = build_jobs(arguments, config, root)
        if not jobs:
            raise ValueError("selectors produced no sweep jobs")
        prior_ledger = None
        if ledger_path.is_file():
            try:
                loaded_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"sweep ledger is unreadable: {exc}") from exc
            if isinstance(loaded_ledger, dict):
                prior_ledger = loaded_ledger
        reuse_definition = configured_reuse_contract(config, jobs)
        reuse_preflight = (
            reuse_contract_preflight(jobs, reuse_definition, prior_ledger)
            if reuse_definition is not None
            else None
        )
        if reuse_preflight is not None and not reuse_preflight["valid"]:
            raise RuntimeError(
                "confirmation reuse preflight failed: "
                + "; ".join(reuse_preflight["failed_checks"])
            )
        source_provenance = ensure_source_provenance(schedule_dir)
        identity = identity_record(arguments, config, jobs, source_provenance)
        ledger = initialize_ledger(
            ledger_path,
            identity,
            jobs,
            arguments,
            default_report_dir=root / "reports" / "05_production_observability",
        )
        if reuse_definition is not None and reuse_preflight is not None:
            persisted = ledger.get("reuse_contract") or {}
            persisted_definition = persisted.get("definition") or {}
            persisted_digest = persisted_definition.get("definition_sha256")
            current_digest = reuse_definition["definition_sha256"]
            if persisted_digest not in (None, current_digest):
                raise RuntimeError(
                    "confirmation reuse definition changed after schedule creation"
                )
            ledger["reuse_contract"] = {
                "definition": reuse_definition,
                "preflight": reuse_preflight,
            }
        if arguments.plan_only:
            ledger["status"] = "planned"
            save_ledger(ledger_path, ledger)
            print(json.dumps({"schedule": str(ledger_path), "jobs": len(jobs), "plan_only": True}, indent=2))
            return 0

        stop_event = threading.Event()
        previous_handlers = install_stop_handlers(stop_event)
        recovered_sessions = recover_interrupted_sessions(ledger, heartbeat_path)
        recovered_capture_attempts = recover_interrupted_capture_attempts(
            ledger, heartbeat_path, ledger_path=ledger_path
        )
        elapsed_before_session = cumulative_elapsed_seconds(ledger)
        remaining_budget = remaining_campaign_budget_seconds(arguments, ledger)
        started_monotonic = time.monotonic()
        started_epoch = time.time()
        deadline_monotonic = started_monotonic + remaining_budget
        scheduler_process_identity = linux_process_identity(os.getpid())
        if scheduler_process_identity is None:
            raise RuntimeError("cannot capture the active scheduler process identity")
        session = {
            "started_at": utc_timestamp(),
            "started_epoch": started_epoch,
            "started_monotonic": started_monotonic,
            "budget_hours": arguments.budget_hours,
            "budget_elapsed_before_seconds": elapsed_before_session,
            "budget_remaining_before_seconds": remaining_budget,
            "heartbeat_seconds": arguments.heartbeat_seconds,
            "term_grace_seconds": arguments.term_grace_seconds,
            "pid": os.getpid(),
            "process_identity": scheduler_process_identity,
            "status": "running",
        }
        ledger["sessions"].append(session)
        ledger["budget"] = {
            "limit_seconds": arguments.budget_hours * 3600.0,
            "elapsed_seconds": elapsed_before_session,
            "remaining_seconds": remaining_budget,
            "recovered_sessions": len(recovered_sessions),
            "recovered_capture_attempts": len(recovered_capture_attempts),
        }
        ledger.setdefault("stages", {})
        ledger.setdefault("stage_reports", {})
        ledger.setdefault("promotions", {})
        ledger["status"] = "running"
        save_ledger(ledger_path, ledger)
        captures_complete = True
        stages = sweep_stage_specs(config, [
            run for run in config.get("runs") or [] if not run.get("existing")
        ])
        final_report_attempted = False
        try:
            campaign_failed = False
            campaign_halted = False
            capture_reason: str | None = None
            for stage_index, stage in enumerate(stages):
                stage_name = str(stage["name"])
                stage_jobs = [job for job in jobs if job.stage == stage_name]
                stage_record = ledger["stages"].setdefault(stage_name, {
                    "stage": stage_name,
                    "stage_index": stage_index,
                    "status": "pending",
                    "job_ids": [job.job_id for job in stage_jobs],
                    "report_after": bool(stage.get("report_after", False)),
                })
                if not stage_jobs:
                    stage_record.update(
                        status="not_selected",
                        job_ids=[],
                        report_after=False,
                        finished_at=utc_timestamp(),
                    )
                    stage_record.pop("report", None)
                    ledger["stage_reports"].pop(stage_name, None)
                    save_ledger(ledger_path, ledger)
                    continue
                promotion_rule = promotion_for_target(stages, stage_name)
                if promotion_rule is not None:
                    source_stage, policy = promotion_rule
                    promotion_record = ledger["promotions"].get(stage_name)
                    source_report = (ledger.get("stage_reports") or {}).get(source_stage) or {}
                    current_accuracy = source_report.get("accuracy_ledger") or {}
                    if promotion_record is None:
                        if not current_accuracy.get("available") and bool(policy.get("required", True)):
                            raise RuntimeError(
                                f"required promotion for {stage_name} has no accuracy ledger from {source_stage}"
                            )
                        promotion_record = select_promoted_runs(
                            target_stage=stage_name,
                            source_stage=source_stage,
                            policy=policy,
                            target_jobs=stage_jobs,
                            accuracy=current_accuracy,
                            ledger=ledger,
                        )
                        ledger["promotions"][stage_name] = promotion_record
                        save_ledger(ledger_path, ledger)
                    elif (
                        current_accuracy.get("available")
                        and promotion_record.get("accuracy_ledger", {}).get("sha256")
                        != current_accuracy.get("sha256")
                    ):
                        raise RuntimeError(
                            f"accuracy ledger changed after promotion into {stage_name}"
                        )
                    selected_runs = set(promotion_record.get("selected_runs") or [])
                    for job in stage_jobs:
                        if job.run_label not in selected_runs:
                            ledger["jobs"][job.job_id].update(
                                status="not_promoted",
                                promotion_source=source_stage,
                                promotion_target=stage_name,
                            )
                    stage_jobs = [job for job in stage_jobs if job.run_label in selected_runs]

                if not stage_jobs:
                    stage_record.update(
                        status="not_selected",
                        report_after=False,
                        finished_at=utc_timestamp(),
                    )
                    stage_record.pop("report", None)
                    ledger["stage_reports"].pop(stage_name, None)
                    save_ledger(ledger_path, ledger)
                    continue

                stage_record["status"] = "running"
                save_ledger(ledger_path, ledger)
                for job in stage_jobs:
                    if stop_event.is_set():
                        captures_complete = False
                        campaign_halted = True
                        capture_reason = "stop requested"
                        break
                    job_row = ledger["jobs"][job.job_id]
                    pre_cap = artifact_tree_cap_check(
                        arguments, phase="before_job", job_id=job.job_id
                    )
                    record_artifact_tree_cap_check(ledger, pre_cap)
                    pre_cap_reason = artifact_tree_cap_reason(pre_cap)
                    if pre_cap_reason is not None:
                        job_row.update(
                            status="deferred",
                            admission_reason=pre_cap_reason,
                            artifact_tree_cap=pre_cap,
                        )
                        captures_complete = False
                        campaign_halted = True
                        capture_reason = pre_cap_reason
                        save_ledger(ledger_path, ledger)
                        break
                    if job_row.get("status") == "scene_failed":
                        if not recorded_scene_failure(job_row):
                            captures_complete = False
                            campaign_halted = True
                            campaign_failed = True
                            capture_reason = (
                                f"job {job.job_id} has a malformed terminal scene failure record"
                            )
                            break
                        captures_complete = False
                        capture_reason = (
                            f"job {job.job_id} recorded terminal scene failure "
                            f"{job_row['scene_failure']['kind']}"
                        )
                        print(json.dumps({
                            "event": "scene_failure_continue",
                            "job_id": job.job_id,
                            "stage": stage_name,
                            "run": job.run_label,
                            "scene": job.scene_id,
                            "kind": job_row["scene_failure"]["kind"],
                            "reused": True,
                        }, sort_keys=True), flush=True)
                        continue
                    if job_row.get("status") == "failed":
                        attempts = job_row.get("attempts") or []
                        if attempts and (
                            not attempts[-1].get("transient")
                            or len(attempts) >= 1 + arguments.retry_transient_failures
                        ):
                            captures_complete = False
                            campaign_halted = True
                            campaign_failed = True
                            capture_reason = (
                                f"job {job.job_id} has no retryable attempts remaining"
                            )
                            break
                    existing_nonempty = (
                        job.run_dir.is_dir() and any(job.run_dir.iterdir())
                    )
                    if job_row.get("status") == "complete" or existing_nonempty:
                        valid, reason = validate_completed_sweep_run(job)
                        if valid:
                            if job_row.get("status") != "complete":
                                job_row.update(
                                    status="complete",
                                    reused=True,
                                    validation=reason,
                                    artifact_usage=directory_usage(job.run_dir),
                                )
                                job_row.pop("admission_reason", None)
                                job_row.pop("reuse_validation_error", None)
                            if arguments.compact and not (
                                job.run_dir / "compacted_completion.json"
                            ).is_file():
                                job_row["compaction"] = compact_validated_run(job)
                                job_row["artifact_usage"] = directory_usage(job.run_dir)
                            post_cap = artifact_tree_cap_check(
                                arguments, phase="after_reused_job", job_id=job.job_id
                            )
                            record_artifact_tree_cap_check(ledger, post_cap)
                            post_cap_reason = artifact_tree_cap_reason(post_cap)
                            if post_cap_reason is not None:
                                job_row.update(
                                    status="failed",
                                    validation=post_cap_reason,
                                    artifact_tree_cap=post_cap,
                                )
                                captures_complete = False
                                campaign_halted = True
                                campaign_failed = True
                                capture_reason = post_cap_reason
                            save_ledger(ledger_path, ledger)
                            if post_cap_reason is not None:
                                break
                            continue
                        if job_row.get("status") == "complete":
                            job_row.update(status="pending", resume_validation_error=reason)
                        else:
                            job_row["reuse_validation_error"] = reason
                    reason = admission_reason(arguments, ledger, job, deadline_monotonic, root.parent)
                    if reason is not None:
                        job_row.update(status="deferred", admission_reason=reason)
                        captures_complete = False
                        campaign_halted = True
                        capture_reason = reason
                        save_ledger(ledger_path, ledger)
                        break
                    print(json.dumps({
                        "event": "job_start", "job_id": job.job_id, "stage": stage_name,
                        "run": job.run_label, "repeat": job.repeat, "scene": job.scene_id,
                        "profile": job.profile,
                    }, sort_keys=True), flush=True)
                    job_succeeded = run_job(
                        arguments, config, root, job, ledger, ledger_path, heartbeat_path,
                        stop_event, deadline_monotonic=deadline_monotonic,
                    )
                    post_cap = artifact_tree_cap_check(
                        arguments, phase="after_job", job_id=job.job_id
                    )
                    record_artifact_tree_cap_check(ledger, post_cap)
                    post_cap_reason = artifact_tree_cap_reason(post_cap)
                    if post_cap_reason is not None:
                        job_row = ledger["jobs"][job.job_id]
                        attempts = job_row.get("attempts") or []
                        if attempts:
                            attempts[-1].update(
                                status="failed",
                                validation=post_cap_reason,
                                transient=False,
                            )
                        job_row.update(
                            status="failed",
                            validation=post_cap_reason,
                            artifact_tree_cap=post_cap,
                        )
                        save_ledger(ledger_path, ledger)
                        captures_complete = False
                        campaign_halted = True
                        campaign_failed = True
                        capture_reason = post_cap_reason
                        break
                    save_ledger(ledger_path, ledger)
                    if not job_succeeded:
                        captures_complete = False
                        job_row = ledger["jobs"][job.job_id]
                        job_status = job_row.get("status")
                        capture_reason = f"job {job.job_id} ended with status {job_status}"
                        if recorded_scene_failure(job_row):
                            print(json.dumps({
                                "event": "scene_failure_continue",
                                "job_id": job.job_id,
                                "stage": stage_name,
                                "run": job.run_label,
                                "scene": job.scene_id,
                                "kind": job_row["scene_failure"]["kind"],
                                "reused": False,
                            }, sort_keys=True), flush=True)
                            continue
                        campaign_halted = True
                        if job_status == "failed":
                            campaign_failed = True
                        break
                if campaign_halted:
                    stage_record["status"] = (
                        "stopped" if stop_event.is_set()
                        else "failed" if campaign_failed
                        else "deferred"
                    )
                    save_ledger(ledger_path, ledger)
                    break
                stage_scene_failure_ids = [
                    job_id for job_id in stage_record["job_ids"]
                    if recorded_scene_failure(ledger["jobs"][job_id])
                ]
                stage_record["status"] = (
                    "complete_with_scene_failures"
                    if stage_scene_failure_ids else "complete"
                )
                stage_record["scene_failure_job_ids"] = stage_scene_failure_ids
                stage_record["finished_at"] = utc_timestamp()
                source_for_promotion = any(
                    isinstance(other.get("promotion"), dict)
                    and str(other["promotion"].get("source_stage") or other["name"]) == stage_name
                    for other in stages
                )
                if bool(stage.get("report_after", False)) or source_for_promotion:
                    remaining = deadline_monotonic - time.monotonic()
                    required = (
                        arguments.report_timeout_minutes + arguments.finalization_reserve_minutes
                    ) * 60.0
                    if remaining < required:
                        stage_record.update(
                            status="deferred_report",
                            reason=f"{remaining:.1f}s remain, {required:.1f}s required for stage report reserve",
                        )
                        captures_complete = False
                        campaign_halted = True
                        capture_reason = str(stage_record["reason"])
                        save_ledger(ledger_path, ledger)
                        break
                    report_record = generate_stage_report(
                        arguments, config, root, stage, stage_index, ledger, ledger_path,
                        heartbeat_path, stop_event,
                    )
                    stage_record["report"] = report_record["path"]
                    if report_record.get("status") != "complete":
                        stage_record["status"] = "failed_report"
                        captures_complete = False
                        campaign_halted = True
                        campaign_failed = True
                        capture_reason = (
                            f"stage report for {stage_name} failed validation"
                        )
                        save_ledger(ledger_path, ledger)
                        break
                save_ledger(ledger_path, ledger)

            if reuse_definition is not None and not campaign_halted:
                reuse_postcheck = reuse_contract_postcheck(ledger, reuse_definition)
                ledger.setdefault("reuse_contract", {})["postcheck"] = reuse_postcheck
                if not reuse_postcheck["valid"]:
                    captures_complete = False
                    campaign_failed = True
                    capture_reason = (
                        "confirmation reuse postcheck failed: "
                        + "; ".join(reuse_postcheck["failed_checks"])
                    )
                save_ledger(ledger_path, ledger)

            completed_evidence = any(
                row.get("status") == "complete" for row in ledger["jobs"].values()
            )
            report_requirements_met = not arguments.generate_report
            if arguments.generate_report and not stop_event.is_set() and completed_evidence:
                remaining = deadline_monotonic - time.monotonic()
                if remaining < arguments.report_timeout_minutes * 60.0:
                    ledger["report"].update(
                        status="deferred",
                        valid=False,
                        requirements_met=False,
                        reason="insufficient time remains for report timeout",
                    )
                    report_requirements_met = False
                else:
                    report_dir = (
                        arguments.report_dir.expanduser().resolve()
                        if arguments.report_dir is not None
                        else root / "reports" / "05_production_observability"
                    )
                    final_report_attempted = True
                    report_requirements_met = generate_final_report(
                        arguments, config, root, report_dir, ledger, ledger_path, heartbeat_path, stop_event
                    )
            elif arguments.generate_report and not stop_event.is_set():
                ledger["report"].update(
                    status="unavailable",
                    valid=False,
                    requirements_met=False,
                    reason="no completed capture evidence is available",
                )
                report_requirements_met = False

            capture_status = (
                "stopped" if stop_event.is_set()
                else "failed" if campaign_failed
                else "complete" if captures_complete
                else "incomplete"
            )
            ledger["capture"] = {
                "status": capture_status,
                "reason": capture_reason,
                "jobs_complete": sum(
                    row.get("status") == "complete" for row in ledger["jobs"].values()
                ),
                "jobs_scene_failed": sum(
                    recorded_scene_failure(row) for row in ledger["jobs"].values()
                ),
                "scene_failure_job_ids": sorted(
                    job_id for job_id, row in ledger["jobs"].items()
                    if recorded_scene_failure(row)
                ),
                "jobs_total": len(ledger["jobs"]),
            }
            ledger["status"] = (
                "stopped" if stop_event.is_set()
                else "failed" if campaign_failed
                else "complete" if captures_complete and report_requirements_met
                else "incomplete"
            )
            session["status"] = ledger["status"]
            session["finished_at"] = utc_timestamp()
            session["elapsed_seconds"] = time.monotonic() - started_monotonic
            total_elapsed = cumulative_elapsed_seconds(ledger)
            ledger["budget"].update({
                "elapsed_seconds": total_elapsed,
                "remaining_seconds": max(
                    0.0, arguments.budget_hours * 3600.0 - total_elapsed
                ),
            })
            save_ledger(ledger_path, ledger)
            write_heartbeat(
                heartbeat_path,
                status=ledger["status"],
                job_id=None,
                elapsed_seconds=session["elapsed_seconds"],
            )
            print(json.dumps({
                "schedule": str(ledger_path),
                "status": ledger["status"],
                "jobs_complete": sum(row.get("status") == "complete" for row in ledger["jobs"].values()),
                "jobs_total": len(ledger["jobs"]),
            }, indent=2), flush=True)
            return 0 if ledger["status"] == "complete" else 1
        except Exception as exc:
            completed_evidence = any(
                row.get("status") == "complete" for row in ledger["jobs"].values()
            )
            if (
                arguments.generate_report
                and completed_evidence
                and not stop_event.is_set()
                and not final_report_attempted
            ):
                remaining = deadline_monotonic - time.monotonic()
                if remaining >= arguments.report_timeout_minutes * 60.0:
                    report_dir = (
                        arguments.report_dir.expanduser().resolve()
                        if arguments.report_dir is not None
                        else root / "reports" / "05_production_observability"
                    )
                    final_report_attempted = True
                    try:
                        generate_final_report(
                            arguments, config, root, report_dir, ledger, ledger_path,
                            heartbeat_path, stop_event,
                        )
                    except Exception as report_exc:
                        ledger["report"].update({
                            "status": "failed",
                            "valid": False,
                            "requirements_met": False,
                            "error": (
                                f"final report after campaign failure raised "
                                f"{type(report_exc).__name__}: {report_exc}"
                            ),
                        })
                else:
                    ledger["report"].update({
                        "status": "deferred",
                        "valid": False,
                        "requirements_met": False,
                        "reason": "insufficient time remains for report timeout",
                    })
            # Failure-report recovery is part of this scheduler session and
            # must be charged before the terminal wall snapshot is persisted.
            elapsed_seconds = time.monotonic() - started_monotonic
            ledger["status"] = "failed"
            ledger["failure"] = {
                "at": utc_timestamp(),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            ledger["capture"] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "jobs_complete": sum(
                    row.get("status") == "complete" for row in ledger["jobs"].values()
                ),
                "jobs_scene_failed": sum(
                    recorded_scene_failure(row) for row in ledger["jobs"].values()
                ),
                "scene_failure_job_ids": sorted(
                    job_id for job_id, row in ledger["jobs"].items()
                    if recorded_scene_failure(row)
                ),
                "jobs_total": len(ledger["jobs"]),
            }
            session.update({
                "status": "failed",
                "finished_at": utc_timestamp(),
                "elapsed_seconds": elapsed_seconds,
                "error": str(exc),
            })
            total_elapsed = cumulative_elapsed_seconds(ledger)
            ledger["budget"].update({
                "elapsed_seconds": total_elapsed,
                "remaining_seconds": max(
                    0.0, arguments.budget_hours * 3600.0 - total_elapsed
                ),
            })
            save_ledger(ledger_path, ledger)
            write_heartbeat(
                heartbeat_path,
                status="failed",
                job_id=None,
                elapsed_seconds=elapsed_seconds,
            )
            raise
        finally:
            restore_stop_handlers(previous_handlers)


def main() -> int:
    try:
        return run_sweep(tyro.cli(Arguments))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
