"""Bounded sweep job failure/pending inventory for development reports."""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_NAME = "openmvs.dmap.sweep_failure_inventory"
SCHEMA_VERSION = 1
SCHEDULE_SCHEMA_NAME = "openmvs.dmap.sweep_schedule"
SUPPORTED_SCHEDULE_VERSIONS = (1, 2)
TRACKED_STATUSES = ("scene_failed", "failed", "pending")
EXPLICIT_SCHEDULE_ENV = "OPENMVS_DMAP_SWEEP_SCHEDULE"
MAX_TEXT_LENGTH = 2048
MAX_COLLECTION_ITEMS = 128
MAX_NESTING_DEPTH = 5


def _relative_path(path: Path, output_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), output_dir.resolve())).as_posix()
    except (OSError, ValueError):
        return path.resolve().as_uri()


def _bounded(value: Any, depth: int = 0) -> Any:
    """Retain useful failure evidence without embedding unbounded logs/commands."""
    if depth >= MAX_NESTING_DEPTH:
        return "<omitted: nesting limit>"
    if isinstance(value, str):
        if len(value) <= MAX_TEXT_LENGTH:
            return value
        return value[:MAX_TEXT_LENGTH] + f"... <{len(value) - MAX_TEXT_LENGTH} chars omitted>"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        result = {
            str(key): _bounded(item, depth + 1)
            for key, item in items[:MAX_COLLECTION_ITEMS]
        }
        if len(items) > MAX_COLLECTION_ITEMS:
            result["_omitted_keys"] = len(items) - MAX_COLLECTION_ITEMS
        return result
    if isinstance(value, (list, tuple)):
        result = [_bounded(item, depth + 1) for item in value[:MAX_COLLECTION_ITEMS]]
        if len(value) > MAX_COLLECTION_ITEMS:
            result.append({"_omitted_items": len(value) - MAX_COLLECTION_ITEMS})
        return result
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded(str(value), depth)


def _schedule_paths(experiment_root: Path) -> list[Path]:
    paths: set[Path] = set()
    explicit = os.environ.get(EXPLICIT_SCHEDULE_ENV)
    if explicit:
        paths.add(Path(explicit).expanduser().resolve())
    paths.update(path.resolve() for path in experiment_root.glob("sweep*/schedule.json"))
    return sorted(paths, key=str)


def _read_schedule(path: Path, output_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    diagnostic: dict[str, Any] = {
        "path": _relative_path(path, output_dir),
        "valid": False,
        "report_dir_match": False,
    }
    if not path.is_file():
        diagnostic["error"] = "schedule path does not exist"
        return None, diagnostic
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        diagnostic["error"] = f"schedule cannot be read: {exc}"
        return None, diagnostic
    if not isinstance(value, dict):
        diagnostic["error"] = "schedule root is not an object"
        return None, diagnostic
    if (
        value.get("schema_name") != SCHEDULE_SCHEMA_NAME
        or value.get("schema_version") not in SUPPORTED_SCHEDULE_VERSIONS
        or not isinstance(value.get("jobs"), dict)
    ):
        diagnostic["error"] = "unsupported or malformed sweep schedule"
        return None, diagnostic
    report_dir = (value.get("policy") or {}).get("report_dir")
    if report_dir:
        try:
            diagnostic["report_dir_match"] = (
                Path(str(report_dir)).expanduser().resolve() == output_dir.resolve()
            )
        except OSError:
            pass
    diagnostic.update({
        "valid": True,
        "status": value.get("status"),
        "updated_at": value.get("updated_at"),
        "job_count": len(value["jobs"]),
    })
    return value, diagnostic


def _job_record(job_id: str, row: dict[str, Any]) -> dict[str, Any]:
    status = str(row.get("status"))
    result = {
        "job_id": str(row.get("job_id") or job_id),
        "status": status,
        "stage": row.get("stage"),
        "stage_index": row.get("stage_index"),
        "run": row.get("run"),
        "role": row.get("role"),
        "family": row.get("family"),
        "repeat": row.get("repeat"),
        "scene_id": row.get("scene_id"),
        "profile": row.get("profile"),
        "mode": row.get("mode"),
        "validation": _bounded(row.get("validation")),
        "attempt_count": len(row.get("attempts") or []),
    }
    if isinstance(row.get("scene_failure"), dict):
        failure = row["scene_failure"]
        result["scene_failure"] = _bounded({
            key: failure.get(key)
            for key in (
                "kind", "scope", "terminal", "continuable",
                "promotion_eligible", "evidence",
            )
            if key in failure
        })
    attempts = row.get("attempts") or []
    if attempts and isinstance(attempts[-1], dict):
        attempt = attempts[-1]
        result["last_attempt"] = _bounded({
            key: attempt.get(key)
            for key in (
                "status", "return_code", "elapsed_seconds", "timed_out",
                "terminated", "killed", "stop_requested", "transient", "validation",
            )
            if key in attempt
        })
    return result


def build_inventory(experiment_root: Path, output_dir: Path) -> dict[str, Any]:
    """Load the authoritative schedule associated with a report, when available."""
    experiment_root = experiment_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    valid: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    diagnostics: list[dict[str, Any]] = []
    for path in _schedule_paths(experiment_root):
        schedule, diagnostic = _read_schedule(path, output_dir)
        diagnostics.append(diagnostic)
        if schedule is not None:
            valid.append((path, schedule, diagnostic))
    base = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "represented_statuses": list(TRACKED_STATUSES),
        "candidate_schedules": diagnostics,
        "scope_note": (
            "Only jobs in the selected sweep schedule are represented. Unscheduled scenes, "
            "input-preparation failures, and failures recorded only in another ledger are not inferred."
        ),
    }
    if not valid:
        return {
            **base,
            "available": False,
            "source_schedule": None,
            "selection_reason": None,
            "schedule_status": None,
            "schedule_updated_at": None,
            "schedule_job_count": 0,
            "tracked_job_count": 0,
            "status_counts": {status: 0 for status in TRACKED_STATUSES},
            "all_status_counts": {},
            "jobs": [],
            "unavailable_reason": (
                "no valid sweep schedule was discovered; campaign-level job failures and pending work "
                "cannot be claimed as represented"
            ),
        }
    matching = [item for item in valid if item[2].get("report_dir_match")]
    pool = matching or valid
    selected_path, selected, selected_diagnostic = max(
        pool,
        key=lambda item: (
            str(item[1].get("updated_at") or ""),
            item[0].stat().st_mtime_ns,
            str(item[0]),
        ),
    )
    all_jobs = selected["jobs"]
    all_status_counts = Counter(str(row.get("status")) for row in all_jobs.values())
    tracked = [
        _job_record(str(job_id), row)
        for job_id, row in all_jobs.items()
        if isinstance(row, dict) and str(row.get("status")) in TRACKED_STATUSES
    ]
    tracked.sort(key=lambda row: (
        TRACKED_STATUSES.index(str(row["status"])),
        int(row.get("stage_index") or 0),
        str(row.get("stage") or ""),
        str(row.get("run") or ""),
        int(row.get("repeat") or 0),
        str(row.get("scene_id") or ""),
        str(row.get("profile") or ""),
        str(row.get("job_id") or ""),
    ))
    status_counts = Counter(str(row["status"]) for row in tracked)
    return {
        **base,
        "available": True,
        "source_schedule": _relative_path(selected_path, output_dir),
        "selection_reason": (
            "most recently updated schedule whose configured report directory matches this report"
            if matching else
            "most recently updated valid schedule under the experiment root; no report-directory match was available"
        ),
        "schedule_status": selected.get("status"),
        "schedule_updated_at": selected.get("updated_at"),
        "schedule_job_count": len(all_jobs),
        "tracked_job_count": len(tracked),
        "status_counts": {status: status_counts[status] for status in TRACKED_STATUSES},
        "all_status_counts": dict(sorted(all_status_counts.items())),
        "jobs": tracked,
        "unavailable_reason": None,
        "selected_schedule_report_dir_match": bool(selected_diagnostic.get("report_dir_match")),
    }


def validate_inventory(inventory: dict[str, Any]) -> list[str]:
    """Return structural/consistency errors for a sweep failure inventory."""
    errors: list[str] = []
    if inventory.get("schema_name") != SCHEMA_NAME or inventory.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid schema identity")
    if inventory.get("represented_statuses") != list(TRACKED_STATUSES):
        errors.append("represented_statuses does not match the schema contract")
    jobs = inventory.get("jobs")
    if not isinstance(jobs, list):
        return [*errors, "jobs is not an array"]
    if any(not isinstance(row, dict) or row.get("status") not in TRACKED_STATUSES for row in jobs):
        errors.append("jobs contains an invalid row or status")
    job_ids = [str(row.get("job_id")) for row in jobs if isinstance(row, dict)]
    if len(job_ids) != len(set(job_ids)):
        errors.append("job IDs are not unique")
    if inventory.get("tracked_job_count") != len(jobs):
        errors.append("tracked_job_count does not match jobs")
    counts = Counter(str(row.get("status")) for row in jobs if isinstance(row, dict))
    expected_counts = {status: counts[status] for status in TRACKED_STATUSES}
    if inventory.get("status_counts") != expected_counts:
        errors.append("status_counts does not match jobs")
    if inventory.get("available"):
        if not inventory.get("source_schedule"):
            errors.append("available inventory has no source_schedule")
        if inventory.get("schedule_job_count", -1) < len(jobs):
            errors.append("schedule_job_count is smaller than tracked_job_count")
    elif inventory.get("source_schedule") is not None or jobs or not inventory.get("unavailable_reason"):
        errors.append("unavailable inventory must have no source/jobs and an explicit reason")
    return errors
