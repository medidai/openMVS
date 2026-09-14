#!/usr/bin/env python3
"""Experiment-level orchestration for canonical observability array stores.

Frame stores are immutable and additive.  The experiment index is the only
mutable artifact: it is merged deterministically and replaced atomically after
all selected frame stores have converted or validated successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Protocol, Sequence

from . import array_store


SCHEMA_NAME = "openmvs.dmap.array_store_index"
SCHEMA_VERSION = 1
STORE_ROOT_RELATIVE = Path("array_stores") / "v1"
INDEX_RELATIVE = Path("array_stores") / "array_store_index.v1.json"
SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CHUNK_SIZE = array_store.DEFAULT_CHUNK_SIZE
DEFAULT_SHARD_SIZE = array_store.DEFAULT_SHARD_SIZE
DEFAULT_ZSTD_LEVEL = array_store.DEFAULT_ZSTD_LEVEL


class ArrayStoreWorkflowError(RuntimeError):
    """Raised when experiment discovery or indexing cannot complete safely."""


class RunSceneLike(Protocol):
    label: str
    role: str
    repeat: int
    scene_id: str
    instrumentation_dir: Path
    estimation_stage: str
    geometric_iteration: int | None


@dataclass(frozen=True)
class FrameSource:
    run: str
    role: str
    repeat: int
    scene_id: str
    estimation_stage: str
    geometric_iteration: int | None
    frame: str
    image_id: int | None
    frame_dir: Path
    manifest_path: Path

    @property
    def identity(self) -> tuple[str, str, int, str, str, int | None, str]:
        return (
            self.run,
            self.role,
            self.repeat,
            self.scene_id,
            self.estimation_stage,
            self.geometric_iteration,
            self.frame,
        )


@dataclass(frozen=True)
class BulkArrayStoreResult:
    index_path: Path
    selected_count: int
    created_count: int
    validated_count: int
    indexed_count: int
    artifact_count: int
    unavailable_count: int
    uncompressed_bytes: int
    stored_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": str(self.index_path),
            "selected_count": self.selected_count,
            "created_count": self.created_count,
            "validated_count": self.validated_count,
            "indexed_count": self.indexed_count,
            "artifact_count": self.artifact_count,
            "unavailable_count": self.unavailable_count,
            "uncompressed_bytes": self.uncompressed_bytes,
            "stored_bytes": self.stored_bytes,
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArrayStoreWorkflowError(f"required JSON file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArrayStoreWorkflowError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArrayStoreWorkflowError(f"JSON root must be an object: {path}")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArrayStoreWorkflowError(f"index is not canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArrayStoreWorkflowError(f"cannot checksum {path}: {exc}") from exc
    return digest.hexdigest()


def _relative_reference(path: Path, root: Path) -> str:
    """Return a non-absolute POSIX path interpreted relative to the experiment."""

    try:
        relative = os.path.relpath(path.resolve(), root.resolve())
    except OSError as exc:
        raise ArrayStoreWorkflowError(f"cannot resolve path relative to experiment: {path}") from exc
    result = Path(relative).as_posix()
    if PurePosixPath(result).is_absolute():
        raise ArrayStoreWorkflowError(f"experiment-relative path unexpectedly became absolute: {path}")
    return result


def _safe_token(value: Any) -> str:
    text = str(value)
    readable = SAFE_TOKEN_RE.sub("-", text).strip("-.")[:48] or "item"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{readable}--{digest}"


def _stage_token(source: FrameSource) -> str:
    iteration = "none" if source.geometric_iteration is None else f"{source.geometric_iteration:04d}"
    return _safe_token(f"{source.estimation_stage}:{iteration}")


def store_path_for_source(experiment_root: Path, source: FrameSource) -> Path:
    return (
        experiment_root
        / STORE_ROOT_RELATIVE
        / _safe_token(source.run)
        / f"repeat-{source.repeat:04d}"
        / _safe_token(source.scene_id)
        / _stage_token(source)
        / f"{_safe_token(source.frame)}.zarr"
    )


def _normalized_filters(values: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in (values or ()) if str(value)))


def _frame_matches(source_name: str, image_id: int | None, filters: tuple[str, ...]) -> bool:
    if not filters:
        return True
    identities = {source_name}
    if image_id is not None:
        identities.add(str(image_id))
    return any(value in identities for value in filters)


def discover_frame_sources(
    run_scenes: Iterable[RunSceneLike],
    *,
    runs: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
    frames: Sequence[str] | None = None,
) -> list[FrameSource]:
    """Discover authoritative frame manifests and enforce exact selectors."""

    run_filters = _normalized_filters(runs)
    scene_filters = _normalized_filters(scenes)
    frame_filters = _normalized_filters(frames)
    all_run_scenes = list(run_scenes)
    known_runs = {str(row.label) for row in all_run_scenes}
    known_scenes = {str(row.scene_id) for row in all_run_scenes}
    missing_runs = sorted(set(run_filters) - known_runs)
    missing_scenes = sorted(set(scene_filters) - known_scenes)
    if missing_runs:
        raise ArrayStoreWorkflowError(
            "run selector(s) did not match captured runs: " + ", ".join(missing_runs)
        )
    if missing_scenes:
        raise ArrayStoreWorkflowError(
            "scene selector(s) did not match captured scenes: " + ", ".join(missing_scenes)
        )

    selected_scenes = [
        row
        for row in all_run_scenes
        if (not run_filters or str(row.label) in run_filters)
        and (not scene_filters or str(row.scene_id) in scene_filters)
    ]
    discovered: dict[tuple[str, str, int, str, str, int | None, str], FrameSource] = {}
    matched_frames: set[str] = set()
    for run_scene in selected_scenes:
        depthmaps_root = Path(run_scene.instrumentation_dir) / "depthmaps"
        if not depthmaps_root.is_dir():
            raise ArrayStoreWorkflowError(
                "selected instrumentation capture has no depthmaps directory: "
                f"run={run_scene.label!r}, scene={run_scene.scene_id!r}, path={depthmaps_root}"
            )
        frame_dirs = sorted(path for path in depthmaps_root.iterdir() if path.is_dir())
        if not frame_dirs:
            raise ArrayStoreWorkflowError(
                "selected instrumentation capture contains no frame directories: "
                f"run={run_scene.label!r}, scene={run_scene.scene_id!r}, path={depthmaps_root}"
            )
        for frame_dir in frame_dirs:
            summary_path = frame_dir / "summary.json"
            summary = _read_json_object(summary_path) if summary_path.is_file() else {}
            raw_image_id = summary.get("image_id")
            try:
                image_id = int(raw_image_id) if raw_image_id is not None else None
            except (TypeError, ValueError) as exc:
                raise ArrayStoreWorkflowError(
                    f"summary image_id must be an integer in {summary_path}: {raw_image_id!r}"
                ) from exc
            if not _frame_matches(frame_dir.name, image_id, frame_filters):
                continue
            for requested in frame_filters:
                if requested == frame_dir.name or (image_id is not None and requested == str(image_id)):
                    matched_frames.add(requested)
            manifest_path = frame_dir / "map_manifest.json"
            if not manifest_path.is_file():
                raise ArrayStoreWorkflowError(
                    f"selected frame has no authoritative map_manifest.json: {frame_dir}"
                )
            manifest = _read_json_object(manifest_path)
            if manifest.get("schema_name") != "openmvs.dmap.map_manifest":
                raise ArrayStoreWorkflowError(
                    f"selected frame has an unsupported map manifest schema: {manifest_path}"
                )
            source = FrameSource(
                run=str(run_scene.label),
                role=str(run_scene.role),
                repeat=int(run_scene.repeat),
                scene_id=str(run_scene.scene_id),
                estimation_stage=str(run_scene.estimation_stage),
                geometric_iteration=run_scene.geometric_iteration,
                frame=frame_dir.name,
                image_id=image_id,
                frame_dir=frame_dir.resolve(),
                manifest_path=manifest_path.resolve(),
            )
            previous = discovered.get(source.identity)
            if previous is not None and previous.manifest_path != source.manifest_path:
                raise ArrayStoreWorkflowError(
                    "multiple authoritative manifests resolve to the same frame identity: "
                    f"{previous.manifest_path} and {source.manifest_path}"
                )
            discovered[source.identity] = source
    missing_frames = sorted(set(frame_filters) - matched_frames)
    if missing_frames:
        raise ArrayStoreWorkflowError(
            "frame selector(s) did not match captured frames: " + ", ".join(missing_frames)
        )
    if not discovered:
        raise ArrayStoreWorkflowError("no authoritative map manifests matched the selection")
    return [discovered[key] for key in sorted(discovered, key=_identity_sort_key)]


def _index_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str, int | None, str]:
    return (
        str(row.get("run", "")),
        str(row.get("role", "")),
        int(row.get("repeat", 0)),
        str(row.get("scene_id", "")),
        str(row.get("estimation_stage", "")),
        row.get("geometric_iteration"),
        str(row.get("frame", "")),
    )


def _identity_sort_key(identity: tuple[Any, ...]) -> tuple[Any, ...]:
    return (*identity[:5], -1 if identity[5] is None else int(identity[5]), identity[6])


def _validate_relative_store_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ArrayStoreWorkflowError("indexed store path must be a non-empty POSIX relative path")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ArrayStoreWorkflowError(f"indexed store path escapes the experiment: {raw_path!r}")
    root_parts = PurePosixPath(STORE_ROOT_RELATIVE.as_posix()).parts
    if pure.parts[: len(root_parts)] != root_parts:
        raise ArrayStoreWorkflowError(
            f"indexed store path is outside {STORE_ROOT_RELATIVE.as_posix()}: {raw_path!r}"
        )
    return pure.as_posix()


def _load_existing_index(
    index_path: Path,
    experiment_root: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not index_path.exists():
        return {}
    index = _read_json_object(index_path)
    if index.get("schema_name") != SCHEMA_NAME or index.get("schema_version") != SCHEMA_VERSION:
        raise ArrayStoreWorkflowError(f"unsupported or corrupt existing array-store index: {index_path}")
    if (
        index.get("array_store_schema_name") != array_store.SCHEMA_NAME
        or index.get("array_store_schema_version") != array_store.SCHEMA_VERSION
        or index.get("store_root") != STORE_ROOT_RELATIVE.as_posix()
    ):
        raise ArrayStoreWorkflowError(f"existing array-store index contract is inconsistent: {index_path}")
    rows = index.get("stores")
    if not isinstance(rows, list):
        raise ArrayStoreWorkflowError(f"existing array-store index has no stores array: {index_path}")
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for position, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise ArrayStoreWorkflowError(f"stores[{position}] is not an object in {index_path}")
        row = dict(raw_row)
        try:
            key = _index_key(row)
        except (TypeError, ValueError) as exc:
            raise ArrayStoreWorkflowError(
                f"stores[{position}] has an invalid frame identity in {index_path}"
            ) from exc
        if (
            any(not key[index] for index in (0, 1, 3, 4, 6))
            or key[2] < 0
            or (key[5] is not None and not isinstance(key[5], int))
        ):
            raise ArrayStoreWorkflowError(
                f"stores[{position}] has an invalid frame identity in {index_path}"
            )
        if key in indexed:
            raise ArrayStoreWorkflowError(f"duplicate frame identity in existing index: {key}")
        store_path = _validate_relative_store_path(row.get("store"))
        if store_path in seen_paths:
            raise ArrayStoreWorkflowError(f"duplicate store path in existing index: {store_path}")
        resolved_store = experiment_root / store_path
        try:
            resolved_store.resolve().relative_to(experiment_root.resolve())
        except ValueError as exc:
            raise ArrayStoreWorkflowError(
                f"indexed immutable store escapes the experiment: {resolved_store}"
            ) from exc
        if not resolved_store.is_dir() or resolved_store.is_symlink():
            raise ArrayStoreWorkflowError(
                f"indexed immutable store is missing or invalid: {resolved_store}"
            )
        symlinks = [item for item in resolved_store.rglob("*") if item.is_symlink()]
        if symlinks:
            raise ArrayStoreWorkflowError(
                f"indexed immutable store contains symlinks: {symlinks[0]}"
            )
        source_manifest = row.get("source_manifest")
        if (
            not isinstance(source_manifest, str)
            or not source_manifest
            or "\\" in source_manifest
            or PurePosixPath(source_manifest).is_absolute()
        ):
            raise ArrayStoreWorkflowError(
                f"stores[{position}].source_manifest must be experiment-relative"
            )
        resolved_source = experiment_root / source_manifest
        if not resolved_source.is_file():
            raise ArrayStoreWorkflowError(
                f"indexed source manifest is missing: {resolved_source}"
            )
        source_sha256 = str(row.get("source_manifest_sha256") or "")
        if not SHA256_RE.fullmatch(source_sha256) or _sha256(resolved_source) != source_sha256:
            raise ArrayStoreWorkflowError(
                f"indexed source manifest checksum is invalid: {resolved_source}"
            )
        for field_name in (
            "artifact_count",
            "available_count",
            "unavailable_count",
            "uncompressed_bytes",
            "stored_bytes",
        ):
            value = row.get(field_name)
            if not isinstance(value, int) or value < 0:
                raise ArrayStoreWorkflowError(
                    f"stores[{position}].{field_name} must be a nonnegative integer"
                )
        if row.get("array_store_schema_version") != array_store.SCHEMA_VERSION:
            raise ArrayStoreWorkflowError(
                f"stores[{position}].array_store_schema_version is unsupported"
            )
        if row["available_count"] + row["unavailable_count"] != row["artifact_count"]:
            raise ArrayStoreWorkflowError(
                f"stores[{position}] artifact availability counts are inconsistent"
            )
        actual_store_bytes = sum(
            item.stat().st_size for item in resolved_store.rglob("*") if item.is_file()
        )
        if actual_store_bytes != row["stored_bytes"]:
            raise ArrayStoreWorkflowError(
                f"indexed store byte count changed: {resolved_store}"
            )
        if row.get("complete") not in (True, False):
            raise ArrayStoreWorkflowError(f"stores[{position}].complete must be a boolean")
        indexed[key] = row
        seen_paths.add(store_path)
    if index.get("store_count") != len(indexed):
        raise ArrayStoreWorkflowError(f"existing index store_count is inconsistent: {index_path}")
    expected_totals = {
        "artifact_count": sum(row["artifact_count"] for row in indexed.values()),
        "unavailable_count": sum(row["unavailable_count"] for row in indexed.values()),
        "uncompressed_bytes": sum(row["uncompressed_bytes"] for row in indexed.values()),
        "stored_bytes": sum(row["stored_bytes"] for row in indexed.values()),
        "complete": all(row["complete"] is True for row in indexed.values()),
    }
    for field_name, expected in expected_totals.items():
        if index.get(field_name) != expected:
            raise ArrayStoreWorkflowError(
                f"existing index {field_name} is inconsistent: {index_path}"
            )
    return indexed


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _catalog_for_store(store_path: Path) -> dict[str, Any]:
    catalog = _read_json_object(store_path / array_store.MANIFEST_NAME)
    if catalog.get("schema_name") != array_store.SCHEMA_NAME:
        raise ArrayStoreWorkflowError(f"array store has an invalid catalog schema: {store_path}")
    return catalog


def _index_row(
    experiment_root: Path,
    source: FrameSource,
    store_path: Path,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    source_manifest = catalog.get("source_manifest") or {}
    source_sha256 = str(source_manifest.get("sha256") or "")
    if not SHA256_RE.fullmatch(source_sha256) or _sha256(source.manifest_path) != source_sha256:
        raise ArrayStoreWorkflowError(
            f"validated array-store catalog does not match its source manifest: {store_path}"
        )
    for field_name in (
        "artifact_count",
        "available_count",
        "unavailable_count",
        "uncompressed_bytes",
    ):
        value = catalog.get(field_name)
        if not isinstance(value, int) or value < 0:
            raise ArrayStoreWorkflowError(
                f"validated array-store catalog has invalid {field_name}: {store_path}"
            )
    if catalog.get("schema_version") != array_store.SCHEMA_VERSION:
        raise ArrayStoreWorkflowError(f"array-store catalog version is unsupported: {store_path}")
    if (
        int(catalog["available_count"]) + int(catalog["unavailable_count"])
        != int(catalog["artifact_count"])
    ):
        raise ArrayStoreWorkflowError(
            f"validated array-store catalog has inconsistent artifact counts: {store_path}"
        )
    return {
        "run": source.run,
        "role": source.role,
        "repeat": source.repeat,
        "scene_id": source.scene_id,
        "estimation_stage": source.estimation_stage,
        "geometric_iteration": source.geometric_iteration,
        "frame": source.frame,
        "image_id": source.image_id,
        "source_manifest": _relative_reference(source.manifest_path, experiment_root),
        "source_manifest_sha256": source_sha256,
        "store": store_path.resolve().relative_to(experiment_root.resolve()).as_posix(),
        "array_store_schema_version": catalog.get("schema_version"),
        "artifact_count": catalog.get("artifact_count"),
        "available_count": catalog.get("available_count"),
        "unavailable_count": catalog.get("unavailable_count"),
        "uncompressed_bytes": catalog.get("uncompressed_bytes"),
        "stored_bytes": sum(
            item.stat().st_size for item in store_path.rglob("*") if item.is_file()
        ),
        "complete": catalog.get("complete"),
    }


def materialize_experiment_array_stores(
    *,
    experiment_root: Path,
    source_config: Path,
    run_scenes: Iterable[RunSceneLike],
    runs: Sequence[str] | None = None,
    scenes: Sequence[str] | None = None,
    frames: Sequence[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    allow_incomplete: bool = False,
    max_uncompressed_bytes_per_store: int | None = None,
    verify_data: bool = True,
) -> BulkArrayStoreResult:
    """Convert missing selected stores, validate existing stores, and index them."""

    experiment_root = experiment_root.expanduser().resolve()
    source_config = source_config.expanduser().resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)
    sources = discover_frame_sources(
        run_scenes,
        runs=runs,
        scenes=scenes,
        frames=frames,
    )
    index_path = experiment_root / INDEX_RELATIVE
    indexed = _load_existing_index(index_path, experiment_root)
    created_count = 0
    validated_count = 0
    for source in sources:
        store_path = store_path_for_source(experiment_root, source)
        try:
            store_path.resolve(strict=False).relative_to(experiment_root)
        except ValueError as exc:
            raise ArrayStoreWorkflowError(f"array store path escapes the experiment: {store_path}") from exc
        if store_path.exists() or store_path.is_symlink():
            if not store_path.is_dir() or store_path.is_symlink():
                raise ArrayStoreWorkflowError(
                    f"immutable array-store destination is not a regular directory: {store_path}"
                )
            validation = array_store.validate_store(
                store_path,
                verify_data=verify_data,
                source_frame_dir=source.frame_dir,
            )
            if not validation.valid:
                raise ArrayStoreWorkflowError(
                    f"existing array store failed validation: {store_path}: "
                    + "; ".join(validation.errors)
                )
            validated_count += 1
        else:
            array_store.convert_map_manifest(
                source.manifest_path,
                store_path,
                frame_dir=source.frame_dir,
                chunk_size=chunk_size,
                shard_size=shard_size,
                zstd_level=zstd_level,
                allow_incomplete=allow_incomplete,
                max_uncompressed_bytes=max_uncompressed_bytes_per_store,
                validate_output=True,
            )
            validation = array_store.validate_store(
                store_path,
                verify_data=verify_data,
                source_frame_dir=source.frame_dir,
            )
            if not validation.valid:
                raise ArrayStoreWorkflowError(
                    f"new array store failed source-aware validation: {store_path}: "
                    + "; ".join(validation.errors)
                )
            created_count += 1
        catalog = _catalog_for_store(store_path)
        row = _index_row(experiment_root, source, store_path, catalog)
        previous = indexed.get(source.identity)
        if previous is not None and previous != row:
            raise ArrayStoreWorkflowError(
                "existing index conflicts with the validated immutable store for "
                f"run={source.run!r}, scene={source.scene_id!r}, frame={source.frame!r}"
            )
        indexed[source.identity] = row

    rows = [indexed[key] for key in sorted(indexed, key=_identity_sort_key)]
    index = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "array_store_schema_name": array_store.SCHEMA_NAME,
        "array_store_schema_version": array_store.SCHEMA_VERSION,
        "source_config": _relative_reference(source_config, experiment_root),
        "store_root": STORE_ROOT_RELATIVE.as_posix(),
        "store_count": len(rows),
        "artifact_count": sum(int(row.get("artifact_count") or 0) for row in rows),
        "unavailable_count": sum(int(row.get("unavailable_count") or 0) for row in rows),
        "uncompressed_bytes": sum(int(row.get("uncompressed_bytes") or 0) for row in rows),
        "stored_bytes": sum(int(row.get("stored_bytes") or 0) for row in rows),
        "complete": all(row.get("complete") is True for row in rows),
        "stores": rows,
    }
    payload = _canonical_json_bytes(index)
    if not index_path.is_file() or index_path.read_bytes() != payload:
        _atomic_write(index_path, payload)
    return BulkArrayStoreResult(
        index_path=index_path,
        selected_count=len(sources),
        created_count=created_count,
        validated_count=validated_count,
        indexed_count=len(rows),
        artifact_count=index["artifact_count"],
        unavailable_count=index["unavailable_count"],
        uncompressed_bytes=index["uncompressed_bytes"],
        stored_bytes=index["stored_bytes"],
    )
