#!/usr/bin/env python3
"""Build and validate the finalized Experiment 57 composite parent schedule."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import yaml

from . import source_snapshot


SCHEDULE_SCHEMA_NAME = "openmvs.dmap.sweep_schedule"
SCHEDULE_SCHEMA_VERSION = 2
IDENTITY_SCHEMA_NAME = "openmvs.patchmatch_predecessor_composite_identity"
COMPOSITE_SCHEMA_NAME = "openmvs.patchmatch_predecessor_composite"
COMPOSITE_SCHEMA_VERSION = 1
REPORT_BINDING_SCHEMA_NAME = "openmvs.dmap.sweep_report_binding"
REPORT_BINDING_SCHEMA_VERSION = 2
RECOVERY_SCHEMA_NAME = "openmvs.dmap.report_recovery"
RECOVERY_SCHEMA_VERSION = 1
REPORT_SOURCE_SCHEMA_NAME = "openmvs.dmap.report_source_provenance"
REPORT_SOURCE_SCHEMA_VERSION = 1
REPORT_SOURCE_ARCHIVE = "report_source_snapshot.tar.zst"
ARTIFACT_BINDING_FILE = "composite_report_artifact_binding.json"
ARTIFACT_BINDING_SCHEMA_NAME = "openmvs.dmap.composite_report_artifact_binding"
ARTIFACT_BINDING_SCHEMA_VERSION = 1
CORE_REPORT_ARTIFACTS = (
    "report_manifest.json",
    "report_model.json",
    "01_development_report.md",
    "accuracy_ledger.csv",
    "report_inventory.json",
    "report_policy.json",
)
SOURCE_ROLES = ("screen", "view")
EXPECTED_JOB_COUNTS = {"screen": 20, "view": 8}
EXPECTED_GEOMETRIC_ITERATIONS = 4
EXPECTED_REPORT_MANIFEST_SCHEMA_VERSION = 2
EXPECTED_REPORT_MANIFEST_ROW_KEYS = {
    "label",
    "role",
    "repeat",
    "scene_id",
    "instrumentation_dir",
    "depth_map_dir",
    "diagnostic_only",
    "diagnostic_only_reason",
}


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def report_evidence_digest(ledger: dict[str, Any]) -> str:
    evidence = {
        "identity_sha256": (ledger.get("identity") or {}).get("sha256"),
        "completed_job_ids": sorted(
            job_id
            for job_id, row in (ledger.get("jobs") or {}).items()
            if row.get("status") == "complete"
        ),
        "job_states": [
            {"job_id": job_id, "status": str(row.get("status") or "unknown")}
            for job_id, row in sorted((ledger.get("jobs") or {}).items())
        ],
    }
    return stable_digest(evidence)


def _identity_valid(identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    unsigned = dict(identity)
    claimed = unsigned.pop("sha256", None)
    return isinstance(claimed, str) and claimed == stable_digest(unsigned)


def _finalized_source(path: Path, role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load_json(path)
    jobs = ledger.get("jobs") or {}
    sessions = ledger.get("sessions") or []
    expected_count = EXPECTED_JOB_COUNTS[role]
    if (
        ledger.get("schema_name") != SCHEDULE_SCHEMA_NAME
        or ledger.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or not _identity_valid(ledger.get("identity"))
        or not _identity_valid(ledger.get("policy"))
        or not isinstance(jobs, dict)
        or len(jobs) != expected_count
        or any(
            not isinstance(row, dict)
            or row.get("job_id") != job_id
            or row.get("status") != "complete"
            for job_id, row in jobs.items()
        )
        or not isinstance(sessions, list)
        or any(
            not isinstance(session, dict) or session.get("status") == "running"
            for session in sessions
        )
    ):
        raise ValueError(f"{role} schedule is not finalized with {expected_count} jobs")
    job_states = [
        {"job_id": job_id, "status": str(row.get("status"))}
        for job_id, row in sorted(jobs.items())
    ]
    completed = [row["job_id"] for row in job_states if row["status"] == "complete"]
    evidence = {
        "role": role,
        "path": str(path.expanduser().resolve()),
        "file_sha256": sha256_file(path),
        "schedule_identity_sha256": ledger["identity"]["sha256"],
        "evidence_digest": report_evidence_digest(ledger),
        "completed_job_ids": completed,
        "job_states": job_states,
        "config_identity": (ledger.get("identity") or {}).get("config"),
    }
    return ledger, evidence


def build_composite(screen_path: Path, view_path: Path) -> dict[str, Any]:
    source_paths = {"screen": screen_path, "view": view_path}
    ledgers: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for role in SOURCE_ROLES:
        ledger, evidence = _finalized_source(source_paths[role], role)
        ledgers[role] = ledger
        sources.append(evidence)
    screen_ids = set(ledgers["screen"]["jobs"])
    view_ids = set(ledgers["view"]["jobs"])
    if screen_ids & view_ids:
        raise ValueError("screen and view schedules contain overlapping job IDs")
    config_identities = [source.get("config_identity") for source in sources]
    if not config_identities[0] or config_identities[0] != config_identities[1]:
        raise ValueError("screen and view schedules do not bind the same config")

    identity = {
        "schema_name": IDENTITY_SCHEMA_NAME,
        "schema_version": COMPOSITE_SCHEMA_VERSION,
        "source_schedules": sources,
        "completed_job_ids": sorted(screen_ids | view_ids),
        "job_count": len(screen_ids | view_ids),
    }
    identity["sha256"] = stable_digest(identity)
    jobs: dict[str, dict[str, Any]] = {}
    for role in SOURCE_ROLES:
        source_identity = ledgers[role]["identity"]["sha256"]
        for job_id, row in sorted(ledgers[role]["jobs"].items()):
            jobs[job_id] = {
                "job_id": job_id,
                "status": "complete",
                "source_role": role,
                "source_schedule_identity_sha256": source_identity,
                "source_job_sha256": stable_digest(row),
                "stage": row.get("stage"),
                "stage_index": row.get("stage_index"),
                "run": row.get("run"),
                "repeat": row.get("repeat"),
                "scene_id": row.get("scene_id"),
                "profile": row.get("profile"),
                "mode": row.get("mode"),
                "run_dir": row.get("run_dir"),
            }
    policy = {
        "schema_version": 1,
        "executable": False,
        "purpose": "report_only_composite_parent",
    }
    policy["sha256"] = stable_digest(policy)
    composite = {
        "schema_name": SCHEDULE_SCHEMA_NAME,
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "policy": policy,
        "sessions": [],
        "jobs": jobs,
        "capture": {
            "status": "complete",
            "jobs_complete": len(jobs),
            "jobs_scene_failed": 0,
            "jobs_total": len(jobs),
            "scene_failure_job_ids": [],
        },
        "report": {"status": "not_requested"},
        "stages": {
            "predecessor_composite": {
                "stage": "predecessor_composite",
                "stage_index": 0,
                "status": "complete",
                "report_after": False,
                "job_ids": sorted(jobs),
            }
        },
        "composite": {
            "schema_name": COMPOSITE_SCHEMA_NAME,
            "schema_version": COMPOSITE_SCHEMA_VERSION,
            "source_roles": list(SOURCE_ROLES),
            "job_count": len(jobs),
        },
    }
    composite["composite_sha256"] = stable_digest(composite)
    return composite


def validate_composite(
    composite_path: Path, screen_path: Path, view_path: Path
) -> tuple[bool, str, dict[str, Any] | None]:
    try:
        actual = load_json(composite_path)
        unsigned = dict(actual)
        claimed = unsigned.pop("composite_sha256", None)
        if claimed != stable_digest(unsigned):
            return False, "composite schedule self-digest is invalid", None
        expected = build_composite(screen_path, view_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"composite schedule validation failed: {exc}", None
    if actual != expected:
        return False, "composite schedule does not match its screen and view sources", None
    return True, "validated 28-job predecessor composite", actual


def publish_composite(path: Path, value: dict[str, Any]) -> bool:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_file() and path.read_bytes() == payload:
                return False
            raise
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _validated_report_source(report_dir: Path) -> dict[str, Any] | None:
    record_path = report_dir / "report_source_provenance.json"
    archive = report_dir / REPORT_SOURCE_ARCHIVE
    checksum = archive.with_name(archive.name + ".sha256")
    if (
        not record_path.is_file()
        or not archive.is_file()
        or archive.is_symlink()
        or not checksum.is_file()
        or checksum.is_symlink()
    ):
        return None
    try:
        record = load_json(record_path)
        validated = source_snapshot.validate_source_snapshot(archive)
    except (OSError, ValueError, json.JSONDecodeError, source_snapshot.SourceSnapshotError):
        return None
    expected = {
        "sha256": validated.archive_sha256,
        "bytes": validated.archive_bytes,
        "commit": validated.commit,
        "dirty": validated.dirty,
        "tracked_change_count": validated.tracked_change_count,
        "untracked_file_count": validated.untracked_file_count,
    }
    if (
        record.get("schema_name") != REPORT_SOURCE_SCHEMA_NAME
        or record.get("schema_version") != REPORT_SOURCE_SCHEMA_VERSION
        or record.get("archive") != archive.name
        or record.get("checksum") != checksum.name
        or any(record.get(key) != value for key, value in expected.items())
    ):
        return None
    return record


def _exact_cli_value(arguments: Any, name: str) -> str | None:
    if not isinstance(arguments, list) or not all(
        isinstance(value, str) for value in arguments
    ):
        return None
    values: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == name:
            if index + 1 >= len(arguments):
                return None
            values.append(arguments[index + 1])
            index += 2
            continue
        if value.startswith(f"{name}="):
            values.append(value.split("=", 1)[1])
        index += 1
    return values[0] if len(values) == 1 else None


def _scheduled_geometric_iterations(
    config: Any,
    expected_jobs: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[tuple[str, int, str], int] | None:
    if not isinstance(config, dict):
        return None
    run_specs: dict[str, dict[str, Any]] = {}
    for run in config.get("runs") or []:
        if not isinstance(run, dict):
            return None
        label = run.get("label")
        if not isinstance(label, str) or not label or label in run_specs:
            return None
        run_specs[label] = run
    stage_specs: dict[str, dict[str, Any]] = {}
    for stage in ((config.get("sweep") or {}).get("stages") or []):
        if not isinstance(stage, dict):
            return None
        name = stage.get("name")
        if not isinstance(name, str) or not name or name in stage_specs:
            return None
        stage_specs[name] = stage
    scene_specs: dict[str, dict[str, Any]] = {}
    for scene in config.get("scenes") or []:
        if not isinstance(scene, dict):
            return None
        scene_id = scene.get("scan_id")
        if (
            not isinstance(scene_id, str)
            or not scene_id
            or scene_id in scene_specs
        ):
            return None
        scene_specs[scene_id] = scene
    result: dict[tuple[str, int, str], int] = {}
    for key, job in expected_jobs.items():
        run = run_specs.get(key[0])
        stage = stage_specs.get(str(job.get("stage") or ""))
        scene = scene_specs.get(key[2])
        if run is None or stage is None or scene is None:
            return None
        raw_value: Any = _exact_cli_value(
            [
                *(config.get("default_densify_args") or []),
                *(run.get("densify_args") or []),
            ],
            "--geometric-iters",
        )
        stage_overrides = stage.get("argument_overrides") or {}
        scene_overrides = scene.get("argument_overrides") or {}
        if not isinstance(stage_overrides, dict) or not isinstance(
            scene_overrides, dict
        ):
            return None
        overrides = {**stage_overrides, **scene_overrides}
        if not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in overrides.items()
        ):
            return None
        if "--geometric-iters" in overrides:
            raw_value = overrides["--geometric-iters"]
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return None
        if str(value) != str(raw_value) or value < 0:
            return None
        result[key] = value
    return result


def _configured_run_roles(
    config: Any,
    expected_jobs: dict[tuple[str, int, str], dict[str, Any]],
) -> dict[tuple[str, int, str], str] | None:
    if not isinstance(config, dict):
        return None
    roles: dict[str, str] = {}
    for run in config.get("runs") or []:
        if not isinstance(run, dict):
            return None
        label = run.get("label")
        role = run.get("role", "variant")
        if (
            not isinstance(label, str)
            or not label
            or label in roles
            or not isinstance(role, str)
            or not role
        ):
            return None
        roles[label] = role
    if not all(key[0] in roles for key in expected_jobs):
        return None
    return {key: roles[key[0]] for key in expected_jobs}


def _manifest_stage(path: Path) -> tuple[str, int | None] | None:
    if path.name == "dmap_instrumentation":
        return "photometric", None
    if path.parent.name != "geometric_iterations":
        return None
    match = re.fullmatch(r"iteration([0-9]{2})", path.name)
    if match is None:
        return None
    return "geometric_consistency", int(match.group(1))


def _normalized_manifest_row(
    row: Any,
) -> tuple[tuple[str, int, str], dict[str, Any]] | None:
    if not isinstance(row, dict) or set(row) != EXPECTED_REPORT_MANIFEST_ROW_KEYS:
        return None
    label = row.get("label")
    repeat = row.get("repeat")
    scene = row.get("scene_id")
    role = row.get("role")
    instrumentation_dir = row.get("instrumentation_dir")
    depth_map_dir = row.get("depth_map_dir")
    diagnostic_only = row.get("diagnostic_only")
    diagnostic_only_reason = row.get("diagnostic_only_reason")
    if (
        not isinstance(label, str)
        or not label
        or not isinstance(repeat, int)
        or isinstance(repeat, bool)
        or not isinstance(scene, str)
        or not scene
        or not isinstance(role, str)
        or not role
        or not isinstance(instrumentation_dir, str)
        or not instrumentation_dir
        or not isinstance(depth_map_dir, str)
        or not depth_map_dir
        or not isinstance(diagnostic_only, bool)
        or not isinstance(diagnostic_only_reason, str)
    ):
        return None
    stage = _manifest_stage(Path(instrumentation_dir))
    if stage is None:
        return None
    estimation_stage, geometric_iteration = stage
    return (label, repeat, scene), {
        "label": label,
        "repeat": repeat,
        "scene_id": scene,
        "role": role,
        "estimation_stage": estimation_stage,
        "geometric_iteration": geometric_iteration,
        "instrumentation_dir": instrumentation_dir,
        "depth_map_dir": depth_map_dir,
        "diagnostic_only": diagnostic_only,
        "diagnostic_only_reason": diagnostic_only_reason,
    }


def _manifest_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    iteration = row.get("geometric_iteration")
    return (
        row.get("label"),
        row.get("repeat"),
        row.get("scene_id"),
        -1 if iteration is None else iteration,
        row.get("instrumentation_dir"),
        row.get("depth_map_dir"),
    )


def _validate_report_inputs(
    report_dir: Path, composite: dict[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    artifact_evidence = {
        name: {
            "path": str(report_dir / name),
            "sha256": sha256_file(report_dir / name)
            if (report_dir / name).is_file()
            else None,
        }
        for name in CORE_REPORT_ARTIFACTS
    }
    checks = {
        "core_artifacts": all((report_dir / name).is_file() for name in CORE_REPORT_ARTIFACTS)
    }
    manifest_path = report_dir / "report_manifest.json"
    model_path = report_dir / "report_model.json"
    try:
        manifest = load_json(manifest_path)
        model = load_json(model_path)
    except (OSError, ValueError, json.JSONDecodeError):
        checks.update(
            manifest_schema=False,
            manifest_config=False,
            manifest_matrix=False,
            manifest_paths=False,
            manifest_row_semantics=False,
            model_matrix=False,
        )
        return checks, {"core_artifacts": artifact_evidence}
    checks["manifest_schema"] = (
        manifest.get("schema_version") == EXPECTED_REPORT_MANIFEST_SCHEMA_VERSION
    )

    expected_jobs: dict[tuple[str, int, str], dict[str, Any]] = {}
    semantic_jobs = True
    for job in (composite.get("jobs") or {}).values():
        run = job.get("run")
        repeat = job.get("repeat")
        scene = job.get("scene_id")
        run_dir = job.get("run_dir")
        if (
            not isinstance(run, str)
            or not run
            or not isinstance(repeat, int)
            or isinstance(repeat, bool)
            or not isinstance(scene, str)
            or not scene
            or not isinstance(run_dir, str)
            or not run_dir
        ):
            semantic_jobs = False
            continue
        key = (run, repeat, scene)
        if key in expected_jobs:
            semantic_jobs = False
            continue
        expected_jobs[key] = job

    configured_identity = (
        ((composite.get("identity") or {}).get("source_schedules") or [{}])[0].get(
            "config_identity"
        )
        or {}
    )
    config_path = Path(str(configured_identity.get("path") or ""))
    try:
        config_identity_valid = (
            configured_identity.get("exists") is True
            and config_path.is_file()
            and not config_path.is_symlink()
            and configured_identity.get("size") == config_path.stat().st_size
            and configured_identity.get("sha256") == sha256_file(config_path)
        )
    except OSError:
        config_identity_valid = False
    try:
        configured = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_identity_valid
            else None
        )
    except (OSError, UnicodeError, yaml.YAMLError):
        configured = None
    geometric_iterations = _scheduled_geometric_iterations(
        configured, expected_jobs
    )
    configured_roles = _configured_run_roles(configured, expected_jobs)
    scheduled_stages_valid = (
        geometric_iterations is not None
        and set(geometric_iterations) == set(expected_jobs)
        and set(geometric_iterations.values()) == {EXPECTED_GEOMETRIC_ITERATIONS}
        and configured_roles is not None
        and set(configured_roles) == set(expected_jobs)
    )
    checks["manifest_config"] = (
        config_identity_valid
        and Path(str(manifest.get("source_config") or "")) == config_path
        and scheduled_stages_valid
    )

    expected_manifest_rows: list[dict[str, Any]] = []
    if (
        scheduled_stages_valid
        and geometric_iterations is not None
        and configured_roles is not None
    ):
        for key, job in expected_jobs.items():
            run, repeat, scene = key
            role = configured_roles[key]
            root = Path(str(job["run_dir"])) / "dmap_instrumentation"
            depth_map_dir = Path(str(job["run_dir"])) / "depth_maps"
            expected_manifest_rows.append(
                {
                    "label": run,
                    "repeat": repeat,
                    "scene_id": scene,
                    "role": role,
                    "estimation_stage": "photometric",
                    "geometric_iteration": None,
                    "instrumentation_dir": str(root),
                    "depth_map_dir": str(depth_map_dir),
                    "diagnostic_only": False,
                    "diagnostic_only_reason": "",
                }
            )
            for iteration in range(geometric_iterations[key]):
                expected_manifest_rows.append(
                    {
                        "label": run,
                        "repeat": repeat,
                        "scene_id": scene,
                        "role": role,
                        "estimation_stage": "geometric_consistency",
                        "geometric_iteration": iteration,
                        "instrumentation_dir": str(
                            root
                            / "geometric_iterations"
                            / f"iteration{iteration:02d}"
                        ),
                        "depth_map_dir": str(depth_map_dir),
                        "diagnostic_only": False,
                        "diagnostic_only_reason": "",
                    }
                )

    manifest_rows = manifest.get("run_scenes") or []
    actual_manifest_rows: list[dict[str, Any]] = []
    actual_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    manifest_well_typed = isinstance(manifest_rows, list)
    if manifest_well_typed:
        for row in manifest_rows:
            normalized = _normalized_manifest_row(row)
            if normalized is None:
                manifest_well_typed = False
                continue
            key, normalized_row = normalized
            actual_manifest_rows.append(normalized_row)
            actual_groups.setdefault(key, []).append(normalized_row)
    expected_stage_inventory = {
        key: {
            ("photometric", None),
            *{
                ("geometric_consistency", iteration)
                for iteration in range(EXPECTED_GEOMETRIC_ITERATIONS)
            },
        }
        for key in expected_jobs
    }
    actual_stage_inventory = {
        key: {
            (row["estimation_stage"], row["geometric_iteration"])
            for row in rows
        }
        for key, rows in actual_groups.items()
    }
    grouped_inventory_exact = (
        set(actual_groups) == set(expected_jobs)
        and actual_stage_inventory == expected_stage_inventory
        and all(
            len(rows) == 1 + EXPECTED_GEOMETRIC_ITERATIONS
            and len(
                {
                    (row["estimation_stage"], row["geometric_iteration"])
                    for row in rows
                }
            )
            == len(rows)
            and len({row["depth_map_dir"] for row in rows}) == 1
            for rows in actual_groups.values()
        )
    )
    checks["manifest_matrix"] = (
        semantic_jobs
        and checks["manifest_schema"]
        and checks["manifest_config"]
        and manifest_well_typed
        and len(expected_jobs) == 28
        and len(expected_manifest_rows) == 140
        and len(manifest_rows) == len(actual_manifest_rows) == 140
        and grouped_inventory_exact
    )
    normalized_actual_matrix = sorted(
        actual_manifest_rows, key=_manifest_sort_key
    )
    normalized_expected_matrix = sorted(
        expected_manifest_rows, key=_manifest_sort_key
    )
    path_fields = (
        "label",
        "repeat",
        "scene_id",
        "estimation_stage",
        "geometric_iteration",
        "instrumentation_dir",
        "depth_map_dir",
    )
    checks["manifest_paths"] = (
        checks["manifest_matrix"]
        and [
            {field: row[field] for field in path_fields}
            for row in normalized_actual_matrix
        ]
        == [
            {field: row[field] for field in path_fields}
            for row in normalized_expected_matrix
        ]
    )
    checks["manifest_row_semantics"] = (
        checks["manifest_paths"]
        and normalized_actual_matrix == normalized_expected_matrix
    )

    expected_model = {}
    for run, repeat, scene in expected_jobs:
        expected_model.setdefault((run, repeat), set()).add(scene)
    model_rows = model.get("runs") or []
    actual_model: dict[tuple[str, int], set[str]] = {}
    model_unique = isinstance(model_rows, list)
    if model_unique:
        for row in model_rows:
            if not isinstance(row, dict):
                model_unique = False
                continue
            try:
                key = (str(row["label"]), int(row["repeat"]))
                scenes = row["scenes"]
            except (KeyError, TypeError, ValueError):
                model_unique = False
                continue
            if key in actual_model or not isinstance(scenes, list):
                model_unique = False
                continue
            actual_model[key] = {str(scene) for scene in scenes}
            if len(actual_model[key]) != len(scenes):
                model_unique = False
    checks["model_matrix"] = (
        model_unique
        and len(model_rows) == len(expected_model) == 7
        and actual_model == expected_model
    )
    return checks, {
        "core_artifacts": artifact_evidence,
        "expected_run_scene_count": len(expected_jobs),
        "expected_manifest_stage_count": len(expected_manifest_rows),
        "expected_stages_per_run_scene": 1 + EXPECTED_GEOMETRIC_ITERATIONS,
        "scheduled_geometric_iterations": (
            sorted(set(geometric_iterations.values()))
            if geometric_iterations is not None
            else []
        ),
        "expected_run_count": len(expected_model),
        "manifest_matrix_sha256": stable_digest(
            normalized_actual_matrix
        )
        if actual_manifest_rows
        else None,
        "expected_manifest_matrix_sha256": stable_digest(
            normalized_expected_matrix
        )
        if expected_manifest_rows
        else None,
        "model_matrix_sha256": stable_digest(
            [
                {"run": run, "repeat": repeat, "scenes": sorted(scenes)}
                for (run, repeat), scenes in sorted(actual_model.items())
            ]
        )
        if actual_model
        else None,
    }


def build_report_artifact_binding(
    report_dir: Path,
    composite_path: Path,
    composite: dict[str, Any],
    report_source_sha256: str,
) -> dict[str, Any]:
    input_checks, input_evidence = _validate_report_inputs(report_dir, composite)
    failed = sorted(name for name, passed in input_checks.items() if not passed)
    if failed:
        raise ValueError(
            "cannot bind composite report with invalid inputs: " + ", ".join(failed)
        )
    core = {}
    for name in CORE_REPORT_ARTIFACTS:
        path = report_dir / name
        core[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    value = {
        "schema_name": ARTIFACT_BINDING_SCHEMA_NAME,
        "schema_version": ARTIFACT_BINDING_SCHEMA_VERSION,
        "parent_schedule_path": str(composite_path.expanduser().resolve()),
        "parent_schedule_file_sha256": sha256_file(composite_path),
        "parent_schedule_identity_sha256": (composite.get("identity") or {}).get(
            "sha256"
        ),
        "parent_composite_sha256": composite.get("composite_sha256"),
        "parent_evidence_digest": report_evidence_digest(composite),
        "completed_job_ids": sorted(composite.get("jobs") or {}),
        "report_source_sha256": report_source_sha256,
        "manifest_matrix_sha256": input_evidence.get("manifest_matrix_sha256"),
        "model_matrix_sha256": input_evidence.get("model_matrix_sha256"),
        "core_artifacts": core,
    }
    value["binding_sha256"] = stable_digest(value)
    return value


def validate_report_artifact_binding(
    report_dir: Path,
    composite_path: Path,
    composite: dict[str, Any],
    report_source_sha256: str,
) -> tuple[bool, str, dict[str, Any]]:
    path = report_dir / ARTIFACT_BINDING_FILE
    evidence = {
        "path": str(path),
        "file_sha256": sha256_file(path) if path.is_file() else None,
    }
    try:
        actual = load_json(path)
        unsigned = dict(actual)
        claimed = unsigned.pop("binding_sha256", None)
        expected = build_report_artifact_binding(
            report_dir, composite_path, composite, report_source_sha256
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"composite report artifact binding is invalid: {exc}", evidence
    checks = {
        "schema": actual.get("schema_name") == ARTIFACT_BINDING_SCHEMA_NAME
        and actual.get("schema_version") == ARTIFACT_BINDING_SCHEMA_VERSION,
        "self_digest": claimed == stable_digest(unsigned),
        "generation_binding": actual == expected,
    }
    evidence.update(checks=checks, binding_sha256=actual.get("binding_sha256"))
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return False, "composite report artifact binding failed: " + ", ".join(failed), evidence
    return True, "validated generation-time composite report artifact binding", evidence


def validate_report_attestation(
    report_dir: Path, composite_path: Path, screen_path: Path, view_path: Path
) -> tuple[bool, str, dict[str, Any]]:
    composite_valid, reason, composite = validate_composite(
        composite_path, screen_path, view_path
    )
    evidence: dict[str, Any] = {
        "composite_path": str(composite_path),
        "composite_file_sha256": (
            sha256_file(composite_path) if composite_path.is_file() else None
        ),
    }
    if not composite_valid or composite is None:
        return False, reason, evidence
    binding_path = report_dir / "sweep_report_binding.json"
    recovery_path = report_dir / "report_recovery_manifest.json"
    try:
        binding = load_json(binding_path)
        recovery = load_json(recovery_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"composite report attestation is unreadable: {exc}", evidence
    report_source = _validated_report_source(report_dir)
    input_checks, input_evidence = _validate_report_inputs(report_dir, composite)
    expected_completed = sorted(composite["jobs"])
    expected_evidence = report_evidence_digest(composite)
    report_source_sha = (report_source or {}).get("sha256")
    artifact_valid, artifact_reason, artifact_evidence = (
        validate_report_artifact_binding(
            report_dir,
            composite_path,
            composite,
            str(report_source_sha or ""),
        )
    )
    checks = {
        "binding_schema": binding.get("schema_name") == REPORT_BINDING_SCHEMA_NAME
        and binding.get("schema_version") == REPORT_BINDING_SCHEMA_VERSION,
        "binding_identity": binding.get("schedule_identity_sha256")
        == composite["identity"]["sha256"],
        "binding_evidence": binding.get("evidence_digest") == expected_evidence,
        "binding_jobs": binding.get("completed_job_ids") == expected_completed,
        "report_source": report_source is not None
        and binding.get("report_source_sha256") == report_source_sha,
        "recovery_schema": recovery.get("schema_name") == RECOVERY_SCHEMA_NAME
        and recovery.get("schema_version") == RECOVERY_SCHEMA_VERSION,
        "recovery_parent": recovery.get("parent_schedule_identity_sha256")
        == composite["identity"]["sha256"]
        and recovery.get("parent_schedule_file_sha256") == sha256_file(composite_path),
        "recovery_evidence": recovery.get("evidence_digest") == expected_evidence
        and recovery.get("completed_job_ids") == expected_completed,
        "recovery_source": recovery.get("parent_capture_source_sha256") is None
        and recovery.get("report_source_sha256") == report_source_sha
        and recovery.get("source_relation") == "distinct_attested_reporter",
        **{f"report_input_{name}": passed for name, passed in input_checks.items()},
        "artifact_binding": artifact_valid,
    }
    evidence.update(
        composite_identity_sha256=composite["identity"]["sha256"],
        evidence_digest=expected_evidence,
        completed_job_ids=expected_completed,
        report_source_sha256=report_source_sha,
        binding_path=str(binding_path),
        binding_file_sha256=sha256_file(binding_path) if binding_path.is_file() else None,
        recovery_path=str(recovery_path),
        recovery_file_sha256=sha256_file(recovery_path) if recovery_path.is_file() else None,
        checks=checks,
        report_inputs=input_evidence,
        artifact_binding={
            "valid": artifact_valid,
            "reason": artifact_reason,
            **artifact_evidence,
        },
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        return False, f"composite report attestation failed: {', '.join(failed)}", evidence
    evidence["attestation_sha256"] = stable_digest(evidence)
    return True, "validated composite report attestation", evidence


def finalized_elapsed_seconds(schedule: dict[str, Any]) -> tuple[bool, float, str]:
    sessions = schedule.get("sessions") or []
    budget = schedule.get("budget") or {}
    policy = schedule.get("policy") or {}
    if not isinstance(sessions, list) or not sessions:
        return False, 0.0, "schedule has no accounting sessions"
    elapsed_values: list[float] = []
    for session in sessions:
        try:
            elapsed = float(session.get("elapsed_seconds"))
        except (AttributeError, TypeError, ValueError):
            return False, 0.0, "schedule contains invalid session elapsed time"
        if (
            not math.isfinite(elapsed)
            or elapsed < 0.0
            or session.get("status") == "running"
        ):
            return False, 0.0, "schedule contains a non-finalized accounting session"
        elapsed_values.append(elapsed)
    total = sum(elapsed_values)
    try:
        recorded = float(budget.get("elapsed_seconds"))
        limit = float(budget.get("limit_seconds"))
        remaining = float(budget.get("remaining_seconds"))
        policy_limit = float(policy.get("budget_hours")) * 3600.0
    except (TypeError, ValueError):
        return False, total, "schedule budget record is invalid"
    if not (
        math.isfinite(recorded)
        and math.isfinite(limit)
        and math.isfinite(remaining)
        and math.isclose(recorded, total, rel_tol=0.0, abs_tol=0.01)
        and math.isclose(limit, policy_limit, rel_tol=0.0, abs_tol=0.01)
        and math.isclose(remaining, max(0.0, limit - total), rel_tol=0.0, abs_tol=0.01)
    ):
        return False, total, "schedule budget does not match finalized sessions"
    return True, total, "validated finalized scheduler session accounting"
