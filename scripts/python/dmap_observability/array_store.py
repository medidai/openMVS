#!/usr/bin/env python3
"""Canonical Zarr v3 storage for depth-map observability arrays.

The converter is intentionally additive: it reads an existing frame-level
``map_manifest.json`` and creates a new immutable store without modifying or
removing any source maps.  Zarr and Pillow are imported lazily so metadata-only
tools retain clear, actionable optional-dependency failures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from . import component_registry
except ImportError:  # Allow direct invocation from the repository root.
    import component_registry  # type: ignore[no-redef]


SCHEMA_NAME = "openmvs.dmap.array_store"
SCHEMA_VERSION = 1
ZARR_FORMAT = 3
MANIFEST_NAME = "array_manifest.json"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_SHARD_SIZE = 1024
DEFAULT_ZSTD_LEVEL = 3
SUPPORTED_SOURCE_SUFFIXES = frozenset({".pfm", ".png"})
ARTIFACT_ID_RE = re.compile(r"^map_[0-9]{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DECLARED_DTYPE_RE = re.compile(r"^([A-Za-z0-9_]+?)(?:x([1-9][0-9]*))?$")


class ArrayStoreError(RuntimeError):
    """Raised when conversion, access, validation, or export cannot continue."""


class OptionalDependencyError(ArrayStoreError):
    """Raised when an explicitly optional array-store dependency is absent."""


@dataclass(frozen=True)
class ConversionResult:
    store_path: Path
    artifact_count: int
    unavailable_count: int
    uncompressed_bytes: int
    stored_bytes: int
    source_manifest_sha256: str


@dataclass(frozen=True)
class ValidationResult:
    store_path: Path
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    artifact_count: int
    available_count: int
    unavailable_count: int
    uncompressed_bytes: int
    stored_bytes: int


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    artifact_id: str
    output_format: str
    exact: bool
    transform: str
    minimum: float | None
    maximum: float | None
    invalid_pixels: int


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ArrayStoreError(f"metadata is not canonical JSON: {exc}") from exc
    return (payload + "\n").encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArrayStoreError(f"JSON file does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArrayStoreError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArrayStoreError(f"JSON root must be an object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    canonical = _canonical_array(values)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _canonical_array(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.dtype.hasobject:
        raise ArrayStoreError("object arrays are not supported")
    if data.dtype.kind not in "biuf":
        raise ArrayStoreError(f"unsupported array dtype: {data.dtype}")
    if data.ndim not in (2, 3):
        raise ArrayStoreError(f"observability arrays must be HxW or HxWxC, got {data.shape}")
    if data.shape[0] <= 0 or data.shape[1] <= 0:
        raise ArrayStoreError(f"observability array has an empty spatial dimension: {data.shape}")
    if data.ndim == 3 and data.shape[2] <= 0:
        raise ArrayStoreError(f"observability array has no channels: {data.shape}")
    dtype = data.dtype.newbyteorder("<") if data.dtype.itemsize > 1 else data.dtype
    return np.ascontiguousarray(data, dtype=dtype)


def _require_zarr() -> Any:
    if sys.version_info < (3, 11):
        raise OptionalDependencyError(
            "canonical Zarr v3 stores require Python 3.11 or newer; install "
            "scripts/python/requirements-dmap-array-store.txt in a separate environment"
        )
    try:
        zarr = importlib.import_module("zarr")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Zarr v3 support is required; install "
            "scripts/python/requirements-dmap-array-store.txt"
        ) from exc
    version = str(getattr(zarr, "__version__", "0"))
    try:
        major = int(version.split(".", 1)[0])
    except ValueError as exc:
        raise OptionalDependencyError(f"cannot determine the installed Zarr version: {version}") from exc
    if major != 3:
        raise OptionalDependencyError(
            f"Zarr >=3,<4 is required for the canonical array store; found {version}"
        )
    if not hasattr(zarr, "codecs") or not hasattr(zarr.codecs, "ZstdCodec"):
        raise OptionalDependencyError(
            f"installed Zarr {version} does not provide the required ZstdCodec"
        )
    return zarr


def _require_pillow() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Pillow is required to import or export PNG maps; install "
            "scripts/python/requirements-depth-benchmark.txt"
        ) from exc


def read_pfm(path: Path) -> np.ndarray:
    """Read standard or OpenMVS multi-channel PFM in top-to-bottom order."""

    def next_line(stream: Any) -> bytes:
        while True:
            line = stream.readline()
            if not line:
                raise ArrayStoreError(f"truncated PFM header: {path}")
            stripped = line.strip()
            if stripped and not stripped.startswith(b"#"):
                return stripped

    try:
        with path.open("rb") as stream:
            header = next_line(stream)
            if header not in (b"Pf", b"PF"):
                raise ArrayStoreError(f"invalid PFM header in {path}: {header!r}")
            dimensions = next_line(stream).split()
            if len(dimensions) != 2:
                raise ArrayStoreError(f"invalid PFM dimensions in {path}")
            width, height = (int(value) for value in dimensions)
            if width <= 0 or height <= 0:
                raise ArrayStoreError(f"invalid PFM size in {path}: {width}x{height}")
            scale = float(next_line(stream))
            if scale == 0 or not math.isfinite(scale):
                raise ArrayStoreError(f"invalid PFM scale in {path}: {scale}")
            endian = "<" if scale < 0 else ">"
            values = np.fromfile(stream, dtype=np.dtype(f"{endian}f4"))
    except (OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ArrayStoreError):
            raise
        raise ArrayStoreError(f"cannot read PFM map {path}: {exc}") from exc
    pixels = width * height
    if values.size == 0 or values.size % pixels:
        raise ArrayStoreError(
            f"invalid PFM payload size in {path}: {values.size} floats for {pixels} pixels"
        )
    channels = values.size // pixels
    if channels > 64:
        raise ArrayStoreError(f"unreasonable PFM channel count in {path}: {channels}")
    expected_channels = 3 if header == b"PF" else 1
    openmvs_channel_scale = abs(scale).is_integer() and int(abs(scale)) == channels
    if channels != expected_channels and not openmvs_channel_scale:
        raise ArrayStoreError(
            f"PFM header/payload channel mismatch in {path}: {expected_channels} vs {channels}"
        )
    shape = (height, width, channels) if channels > 1 else (height, width)
    data = values.reshape(shape)
    # OpenMVS stores the channel count in the scale magnitude for multi-channel
    # TImage payloads.  Preserve standard PFM numeric scaling otherwise.
    if abs(scale) != 1.0 and not openmvs_channel_scale:
        data = data * np.float32(abs(scale))
    return _canonical_array(np.flipud(data))


def write_pfm(path: Path, values: np.ndarray) -> None:
    """Write a scalar or three-channel array as little-endian PFM."""

    data = _canonical_array(np.asarray(values, dtype=np.float32))
    if data.ndim == 3 and data.shape[2] != 3:
        raise ArrayStoreError(f"PFM export requires one or three channels, got {data.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as stream:
            stream.write(b"PF\n" if data.ndim == 3 else b"Pf\n")
            stream.write(f"{data.shape[1]} {data.shape[0]}\n-1.0\n".encode("ascii"))
            np.flipud(data).astype("<f4", copy=False).tofile(stream)
    except OSError as exc:
        raise ArrayStoreError(f"cannot write PFM map {path}: {exc}") from exc


def _read_png(path: Path) -> np.ndarray:
    Image = _require_pillow()
    try:
        with Image.open(path) as image:
            values = np.asarray(image)
    except OSError as exc:
        raise ArrayStoreError(f"cannot read PNG map {path}: {exc}") from exc
    return _canonical_array(values)


def read_source_map(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pfm":
        return read_pfm(path)
    if suffix == ".png":
        return _read_png(path)
    raise ArrayStoreError(
        f"unsupported source map format {suffix!r} for {path}; expected PFM or PNG"
    )


def _normalized_relative_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise ArrayStoreError("map entry path must be a non-empty relative string")
    if "\\" in raw_path or "\x00" in raw_path:
        raise ArrayStoreError(f"invalid map path: {raw_path!r}")
    pure_path = PurePosixPath(raw_path)
    if pure_path.is_absolute() or any(part in ("", ".", "..") for part in pure_path.parts):
        raise ArrayStoreError(f"map path must remain within the frame directory: {raw_path!r}")
    normalized = pure_path.as_posix()
    if normalized != raw_path:
        raise ArrayStoreError(f"map path is not normalized: {raw_path!r}")
    return normalized


def _safe_relative_path(raw_path: Any, frame_dir: Path) -> tuple[str, Path]:
    normalized = _normalized_relative_path(raw_path)
    pure_path = PurePosixPath(normalized)
    path = frame_dir.joinpath(*pure_path.parts)
    try:
        path.resolve(strict=False).relative_to(frame_dir.resolve())
    except ValueError as exc:
        raise ArrayStoreError(f"map path escapes the frame directory: {raw_path!r}") from exc
    if path.is_symlink():
        raise ArrayStoreError(f"source map symlinks are not supported: {raw_path!r}")
    return normalized, path


def _declared_dtype(entry: Mapping[str, Any]) -> tuple[np.dtype[Any], int | None] | None:
    raw_dtype = entry.get("dtype")
    if raw_dtype in (None, ""):
        return None
    match = DECLARED_DTYPE_RE.fullmatch(str(raw_dtype))
    if match is None:
        raise ArrayStoreError(f"invalid declared map dtype: {raw_dtype!r}")
    try:
        dtype = np.dtype(match.group(1))
    except TypeError as exc:
        raise ArrayStoreError(f"unsupported declared map dtype: {raw_dtype!r}") from exc
    channels = int(match.group(2)) if match.group(2) else None
    return dtype, channels


def _validate_declared_array(
    entry: Mapping[str, Any],
    values: np.ndarray,
    manifest: Mapping[str, Any],
) -> None:
    declared = _declared_dtype(entry)
    if declared is not None:
        dtype, channels = declared
        if values.dtype != dtype:
            raise ArrayStoreError(
                f"map {entry.get('path')!r} declares {entry.get('dtype')}, decoded {values.dtype}"
            )
        actual_channels = values.shape[2] if values.ndim == 3 else None
        if channels != actual_channels:
            raise ArrayStoreError(
                f"map {entry.get('path')!r} declares {entry.get('dtype')}, decoded shape {values.shape}"
            )
    height = manifest.get("height")
    width = manifest.get("width")
    if isinstance(height, int) and isinstance(width, int):
        if tuple(values.shape[:2]) != (height, width):
            raise ArrayStoreError(
                f"map {entry.get('path')!r} has shape {values.shape[:2]}, expected {(height, width)}"
            )


def _normalize_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        descriptor = component_registry.descriptor_from_row(value)
        normalized = descriptor.to_dict()
        errors = component_registry.validate_registry(
            component_registry.build_registry([normalized])
        )
    except (TypeError, ValueError) as exc:
        raise ArrayStoreError(f"invalid signal descriptor: {exc}") from exc
    if errors:
        raise ArrayStoreError("invalid signal descriptor: " + "; ".join(errors))
    return normalized


def _descriptor_sources(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    registry = manifest.get("component_registry")
    rows: Iterable[Any] = []
    if isinstance(registry, Mapping):
        rows = registry.get("signals") or []
    elif isinstance(manifest.get("signal_descriptors"), list):
        rows = manifest.get("signal_descriptors") or []
    elif isinstance(manifest.get("signal_descriptors"), Mapping):
        rows = [
            {"signal_id": signal_id, **dict(raw)}
            for signal_id, raw in manifest["signal_descriptors"].items()
            if isinstance(raw, Mapping)
        ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ArrayStoreError("signal descriptor registry entries must be objects")
        normalized = _normalize_descriptor(row)
        signal_id = normalized["signal_id"]
        previous = sources.get(signal_id)
        if previous is not None and previous != normalized:
            raise ArrayStoreError(f"conflicting descriptors for signal {signal_id!r}")
        sources[signal_id] = normalized
    return sources


def _descriptor_for_entry(
    entry: Mapping[str, Any],
    declared: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    signal = str(entry.get("signal") or "").strip()
    if not signal:
        raise ArrayStoreError("map entry is missing signal")
    row: dict[str, Any] = {"signal": signal}
    if signal in declared:
        row.update(declared[signal])
    embedded = entry.get("signal_descriptor") or entry.get("descriptor")
    if embedded is not None:
        if not isinstance(embedded, Mapping):
            raise ArrayStoreError(f"embedded descriptor for {signal!r} must be an object")
        row.update(embedded)
    row.update({key: value for key, value in entry.items() if key != "path"})
    row["signal_id"] = signal
    return _normalize_descriptor(row)


def _spatial_layout(
    shape: Sequence[int],
    chunk_size: int,
    shard_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if chunk_size <= 0:
        raise ArrayStoreError("chunk size must be positive")
    if shard_size < chunk_size:
        raise ArrayStoreError("shard size must be greater than or equal to chunk size")
    chunks = [min(chunk_size, int(shape[0])), min(chunk_size, int(shape[1]))]
    shards: list[int] = []
    for dimension, chunk in zip(shape[:2], chunks):
        target = min(shard_size, math.ceil(int(dimension) / chunk) * chunk)
        shards.append(max(chunk, target))
    if len(shape) == 3:
        chunks.append(int(shape[2]))
        shards.append(int(shape[2]))
    return tuple(chunks), tuple(shards)


def _source_manifest_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "maps"}


def _acquire_output_lock(output_path: Path) -> tuple[int, Path]:
    lock_path = output_path.parent / f".{output_path.name}.conversion.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ArrayStoreError(f"array-store conversion is already active: {lock_path}") from exc
    return descriptor, lock_path


def convert_map_manifest(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    frame_dir: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    shard_size: int = DEFAULT_SHARD_SIZE,
    zstd_level: int = DEFAULT_ZSTD_LEVEL,
    allow_incomplete: bool = False,
    max_uncompressed_bytes: int | None = None,
    validate_output: bool = True,
) -> ConversionResult:
    """Convert one existing map manifest to a new immutable Zarr v3 store."""

    zarr = _require_zarr()
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    frame_dir = Path(frame_dir) if frame_dir is not None else manifest_path.parent
    if output_path.exists() or output_path.is_symlink():
        raise ArrayStoreError(f"refusing to overwrite existing output: {output_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_name") != "openmvs.dmap.map_manifest":
        raise ArrayStoreError(
            f"unsupported source schema: {manifest.get('schema_name')!r}"
        )
    maps = manifest.get("maps")
    if not isinstance(maps, list):
        raise ArrayStoreError("source map manifest is missing a maps array")
    if not allow_incomplete and manifest.get("complete") is False:
        raise ArrayStoreError("source map manifest is incomplete; pass allow_incomplete=True to catalog it")
    if not (-131072 <= zstd_level <= 22):
        raise ArrayStoreError("Zstandard compression level must be between -131072 and 22")
    if max_uncompressed_bytes is not None and max_uncompressed_bytes < 0:
        raise ArrayStoreError("max_uncompressed_bytes must be nonnegative")

    frame_dir_resolved = frame_dir.resolve()
    try:
        manifest_relative = manifest_path.resolve().relative_to(frame_dir_resolved).as_posix()
    except ValueError as exc:
        raise ArrayStoreError("map manifest must be inside the frame directory") from exc
    source_manifest_sha256 = _sha256_path(manifest_path)
    declared_descriptors = _descriptor_sources(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_descriptor, lock_path = _acquire_output_lock(output_path)
    os.close(lock_descriptor)
    temporary_path: Path | None = None
    try:
        if output_path.exists() or output_path.is_symlink():
            raise ArrayStoreError(f"refusing to overwrite existing output: {output_path}")
        temporary_path = Path(
            tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
        )
        root = zarr.open_group(str(temporary_path), mode="w", zarr_format=ZARR_FORMAT)
        arrays_group = root.require_group("arrays")
        codec = zarr.codecs.ZstdCodec(level=zstd_level, checksum=True)
        artifacts: list[dict[str, Any]] = []
        descriptors: dict[str, dict[str, Any]] = {}
        unavailable_count = 0
        uncompressed_bytes = 0

        for index, raw_entry in enumerate(maps):
            if not isinstance(raw_entry, Mapping):
                raise ArrayStoreError(f"maps[{index}] must be an object")
            entry = json.loads(_canonical_json_bytes(dict(raw_entry)))
            artifact_id = f"map_{index:06d}"
            source_relative, source_path = _safe_relative_path(entry.get("path"), frame_dir)
            descriptor = _descriptor_for_entry(entry, declared_descriptors)
            signal = descriptor["signal_id"]
            previous_descriptor = descriptors.get(signal)
            if previous_descriptor is not None and previous_descriptor != descriptor:
                raise ArrayStoreError(f"conflicting descriptors for signal {signal!r}")
            descriptors[signal] = descriptor
            common = {
                "artifact_id": artifact_id,
                "signal": signal,
                "signal_descriptor": descriptor,
                "source_entry": entry,
                "source_path": source_relative,
            }
            if not source_path.is_file():
                if not allow_incomplete:
                    raise ArrayStoreError(f"source map does not exist: {source_path}")
                artifacts.append(
                    {
                        **common,
                        "availability": "unavailable",
                        "unavailable_reason": "source map does not exist",
                    }
                )
                unavailable_count += 1
                continue
            if source_path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
                raise ArrayStoreError(
                    f"unsupported source map format for {source_relative!r}; expected PFM or PNG"
                )
            source_bytes = source_path.stat().st_size
            declared_bytes = entry.get("bytes")
            if isinstance(declared_bytes, int) and declared_bytes != source_bytes and not allow_incomplete:
                raise ArrayStoreError(
                    f"source map size mismatch for {source_relative!r}: "
                    f"manifest {declared_bytes}, file {source_bytes}"
                )
            values = read_source_map(source_path)
            _validate_declared_array(entry, values, manifest)
            next_bytes = uncompressed_bytes + int(values.nbytes)
            if max_uncompressed_bytes is not None and next_bytes > max_uncompressed_bytes:
                raise ArrayStoreError(
                    f"uncompressed array budget exceeded at {source_relative!r}: "
                    f"{next_bytes} > {max_uncompressed_bytes} bytes"
                )
            uncompressed_bytes = next_bytes
            chunks, shards = _spatial_layout(values.shape, chunk_size, shard_size)
            array_path = f"arrays/{artifact_id}"
            array_sha256 = _array_sha256(values)
            attributes = {
                "array_store_schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "signal": signal,
                "signal_descriptor": descriptor,
                "source_entry": entry,
                "source_path": source_relative,
                "source_sha256": _sha256_path(source_path),
            }
            array = arrays_group.create_array(
                artifact_id,
                shape=values.shape,
                dtype=values.dtype,
                chunks=chunks,
                shards=shards,
                compressors=codec,
                fill_value=0,
                attributes=attributes,
                overwrite=False,
            )
            array[...] = values
            artifacts.append(
                {
                    **common,
                    "availability": "available",
                    "array_path": array_path,
                    "array_sha256": array_sha256,
                    "dtype": values.dtype.name,
                    "shape": list(values.shape),
                    "chunks": list(chunks),
                    "shards": list(shards),
                    "uncompressed_bytes": int(values.nbytes),
                    "source_bytes": source_bytes,
                    "source_sha256": attributes["source_sha256"],
                    "codec": {
                        "name": "zstd",
                        "level": zstd_level,
                        "checksum": True,
                        "sharding": "indexed",
                    },
                }
            )

        registry = component_registry.build_registry(descriptors.values())
        registry_errors = component_registry.validate_registry(registry)
        if registry_errors:
            raise ArrayStoreError("invalid component registry: " + "; ".join(registry_errors))
        array_manifest = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "zarr_format": ZARR_FORMAT,
            "layout": {
                "array_group": "arrays",
                "spatial_chunk_size": chunk_size,
                "spatial_shard_size": shard_size,
                "channel_chunking": "all_channels",
                "sharding": "indexed",
                "compressor": "zstd",
                "zstd_level": zstd_level,
                "zstd_checksum": True,
            },
            "source_manifest": {
                "path": manifest_relative,
                "schema_name": manifest.get("schema_name"),
                "schema_version": manifest.get("schema_version"),
                "sha256": source_manifest_sha256,
                "metadata": _source_manifest_metadata(manifest),
            },
            "component_registry": registry,
            "artifacts": artifacts,
            "artifact_count": len(artifacts),
            "available_count": len(artifacts) - unavailable_count,
            "unavailable_count": unavailable_count,
            "uncompressed_bytes": uncompressed_bytes,
            "complete": unavailable_count == 0 and manifest.get("complete", True) is not False,
        }
        root.attrs.update(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "source_manifest_sha256": source_manifest_sha256,
                "artifact_count": len(artifacts),
            }
        )
        (temporary_path / MANIFEST_NAME).write_bytes(_canonical_json_bytes(array_manifest))
        if validate_output:
            validation = validate_store(temporary_path, verify_data=True)
            if not validation.valid:
                raise ArrayStoreError(
                    "new array store failed validation: " + "; ".join(validation.errors)
                )
        os.rename(temporary_path, output_path)
        temporary_path = None
        return ConversionResult(
            store_path=output_path,
            artifact_count=len(artifacts),
            unavailable_count=unavailable_count,
            uncompressed_bytes=uncompressed_bytes,
            stored_bytes=_tree_size(output_path),
            source_manifest_sha256=source_manifest_sha256,
        )
    finally:
        if temporary_path is not None and temporary_path.exists():
            shutil.rmtree(temporary_path)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class ArrayStore:
    """Read/query facade over one canonical frame-level array store."""

    def __init__(self, store_path: Path | str):
        self.path = Path(store_path)
        self._zarr = _require_zarr()
        self.manifest = _load_json(self.path / MANIFEST_NAME)
        if self.manifest.get("schema_name") != SCHEMA_NAME:
            raise ArrayStoreError(f"invalid array-store schema in {self.path}")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise ArrayStoreError(
                f"unsupported array-store schema version: {self.manifest.get('schema_version')}"
            )
        self.root = self._zarr.open_group(str(self.path), mode="r", zarr_format=ZARR_FORMAT)
        artifacts = self.manifest.get("artifacts") or []
        self._artifacts = {
            str(row.get("artifact_id")): row
            for row in artifacts
            if isinstance(row, Mapping) and row.get("artifact_id")
        }
        if len(self._artifacts) != len(artifacts):
            raise ArrayStoreError("array store contains missing or duplicate artifact ids")

    @property
    def artifacts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self.manifest.get("artifacts") or [])

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        try:
            return dict(self._artifacts[artifact_id])
        except KeyError as exc:
            raise ArrayStoreError(f"unknown array artifact: {artifact_id}") from exc

    def find(
        self,
        *,
        signal: str | None = None,
        availability: str | None = "available",
        **source_fields: Any,
    ) -> tuple[dict[str, Any], ...]:
        matches = []
        for row in self.manifest.get("artifacts") or []:
            if signal is not None and row.get("signal") != signal:
                continue
            if availability is not None and row.get("availability") != availability:
                continue
            source_entry = row.get("source_entry") or {}
            if any(source_entry.get(key) != value for key, value in source_fields.items()):
                continue
            matches.append(dict(row))
        return tuple(matches)

    def read(self, artifact_id: str, selection: Any = None) -> np.ndarray:
        row = self.artifact(artifact_id)
        if row.get("availability") != "available":
            raise ArrayStoreError(
                f"artifact {artifact_id} is unavailable: {row.get('unavailable_reason', 'unknown reason')}"
            )
        array_path = row.get("array_path")
        if not isinstance(array_path, str):
            raise ArrayStoreError(f"artifact {artifact_id} has no array path")
        array = self.root[array_path]
        values = array[...] if selection is None else array[selection]
        return np.asarray(values)


def validate_store(
    store_path: Path | str,
    *,
    verify_data: bool = True,
    source_frame_dir: Path | str | None = None,
) -> ValidationResult:
    """Validate schema, descriptors, layout, checksums, and optional sources."""

    store_path = Path(store_path)
    errors: list[str] = []
    warnings: list[str] = []
    artifact_count = available_count = unavailable_count = uncompressed_bytes = 0
    zarr = _require_zarr()
    try:
        manifest = _load_json(store_path / MANIFEST_NAME)
    except ArrayStoreError as exc:
        errors.append(str(exc))
        return ValidationResult(
            store_path, False, tuple(errors), tuple(warnings), 0, 0, 0, 0,
            _tree_size(store_path) if store_path.is_dir() else 0,
        )
    if manifest.get("schema_name") != SCHEMA_NAME:
        errors.append("invalid schema_name")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if manifest.get("zarr_format") != ZARR_FORMAT:
        errors.append("store does not declare Zarr format 3")
    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        errors.append("source_manifest must be an object")
        source_manifest = {}
    else:
        try:
            _normalized_relative_path(source_manifest.get("path"))
        except ArrayStoreError as exc:
            errors.append(f"source_manifest.path is invalid: {exc}")
        if not SHA256_RE.fullmatch(str(source_manifest.get("sha256") or "")):
            errors.append("source_manifest.sha256 is invalid")
        if not isinstance(source_manifest.get("metadata"), Mapping):
            errors.append("source_manifest.metadata must be an object")
    registry_errors = component_registry.validate_registry(
        manifest.get("component_registry") or {}
    )
    errors.extend(f"component_registry: {error}" for error in registry_errors)
    registry_index = component_registry.registry_index(
        manifest.get("component_registry") or {}
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("artifacts must be an array")
        artifacts = []
    artifact_count = len(artifacts)
    try:
        root = zarr.open_group(str(store_path), mode="r", zarr_format=ZARR_FORMAT)
    except Exception as exc:
        errors.append(f"cannot open Zarr v3 group: {exc}")
        root = None
    expected_array_ids: set[str] = set()
    seen_artifact_ids: set[str] = set()
    seen_array_paths: set[str] = set()
    for index, row in enumerate(artifacts):
        location = f"artifacts[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{location} must be an object")
            continue
        artifact_id = str(row.get("artifact_id") or "")
        if not ARTIFACT_ID_RE.fullmatch(artifact_id):
            errors.append(f"{location}.artifact_id is invalid")
        elif artifact_id in seen_artifact_ids:
            errors.append(f"duplicate artifact_id {artifact_id!r}")
        seen_artifact_ids.add(artifact_id)
        signal = str(row.get("signal") or "")
        descriptor = row.get("signal_descriptor")
        if not isinstance(descriptor, Mapping) or descriptor.get("signal_id") != signal:
            errors.append(f"{location}.signal_descriptor does not match signal")
        else:
            try:
                normalized_descriptor = _normalize_descriptor(descriptor)
                if normalized_descriptor != dict(descriptor):
                    errors.append(f"{location}.signal_descriptor is not canonical")
                if registry_index.get(signal) != dict(descriptor):
                    errors.append(f"{location}.signal_descriptor is absent or differs from registry")
            except ArrayStoreError as exc:
                errors.append(f"{location}.signal_descriptor is invalid: {exc}")
        source_entry = row.get("source_entry")
        if not isinstance(source_entry, Mapping):
            errors.append(f"{location}.source_entry must be an object")
            source_entry = {}
        source_path_text = row.get("source_path")
        try:
            normalized_source_path = _normalized_relative_path(source_path_text)
        except ArrayStoreError as exc:
            errors.append(f"{location}.source_path is invalid: {exc}")
            normalized_source_path = None
        if normalized_source_path is not None and source_entry.get("path") != normalized_source_path:
            errors.append(f"{location}.source_entry path does not match source_path")
        if source_entry.get("signal") != signal:
            errors.append(f"{location}.source_entry signal does not match signal")
        availability = row.get("availability")
        if availability == "unavailable":
            unavailable_count += 1
            if row.get("array_path"):
                errors.append(f"{location} is unavailable but declares array_path")
            if not row.get("unavailable_reason"):
                errors.append(f"{location} is unavailable without a reason")
            continue
        if availability != "available":
            errors.append(f"{location}.availability is invalid")
            continue
        available_count += 1
        array_path = row.get("array_path")
        expected_path = f"arrays/{artifact_id}"
        if array_path != expected_path:
            errors.append(f"{location}.array_path must be {expected_path!r}")
            continue
        if array_path in seen_array_paths:
            errors.append(f"duplicate array_path {array_path!r}")
        seen_array_paths.add(str(array_path))
        expected_array_ids.add(artifact_id)
        if root is None:
            continue
        try:
            array = root[array_path]
        except Exception as exc:
            errors.append(f"{location} cannot open array: {exc}")
            continue
        try:
            shape = tuple(int(value) for value in row.get("shape") or [])
            chunks = tuple(int(value) for value in row.get("chunks") or [])
            shards = tuple(int(value) for value in row.get("shards") or [])
        except (TypeError, ValueError):
            errors.append(f"{location} shape/chunks/shards metadata is invalid")
            continue
        if tuple(array.shape) != shape:
            errors.append(f"{location} shape mismatch: {array.shape} != {shape}")
        if np.dtype(array.dtype).name != row.get("dtype"):
            errors.append(f"{location} dtype mismatch: {array.dtype} != {row.get('dtype')}")
        if tuple(array.chunks) != chunks:
            errors.append(f"{location} chunk mismatch: {array.chunks} != {chunks}")
        if tuple(array.shards or ()) != shards:
            errors.append(f"{location} shard mismatch: {array.shards} != {shards}")
        if getattr(array.metadata, "zarr_format", None) != ZARR_FORMAT:
            errors.append(f"{location} is not a Zarr v3 array")
        if not any(type(codec).__name__ == "ZstdCodec" for codec in array.compressors):
            errors.append(f"{location} is not Zstandard-compressed")
        codec_contract = row.get("codec")
        if not isinstance(codec_contract, Mapping):
            errors.append(f"{location}.codec must be an object")
            codec_contract = {}
        expected_level = codec_contract.get("level")
        if (
            codec_contract.get("name") != "zstd"
            or codec_contract.get("sharding") != "indexed"
            or codec_contract.get("checksum") is not True
            or not isinstance(expected_level, int)
        ):
            errors.append(f"{location}.codec contract is invalid")
        metadata_codecs = array.metadata.to_dict().get("codecs") or ()
        shard_codec = metadata_codecs[0] if len(metadata_codecs) == 1 else {}
        shard_configuration = shard_codec.get("configuration") or {}
        inner_codecs = shard_configuration.get("codecs") or ()
        zstd_metadata = inner_codecs[-1] if inner_codecs else {}
        zstd_configuration = zstd_metadata.get("configuration") or {}
        if (
            shard_codec.get("name") != "sharding_indexed"
            or zstd_metadata.get("name") != "zstd"
            or zstd_configuration.get("level") != expected_level
            or zstd_configuration.get("checksum") is not True
        ):
            errors.append(f"{location} Zarr codec pipeline does not match catalog")
        attributes = dict(array.attrs)
        for key, expected in (
            ("array_store_schema_version", SCHEMA_VERSION),
            ("artifact_id", artifact_id),
            ("signal", signal),
            ("signal_descriptor", descriptor),
            ("source_entry", row.get("source_entry")),
            ("source_path", row.get("source_path")),
            ("source_sha256", row.get("source_sha256")),
        ):
            if attributes.get(key) != expected:
                errors.append(f"{location} array attribute {key!r} does not match catalog")
        if not SHA256_RE.fullmatch(str(row.get("array_sha256") or "")):
            errors.append(f"{location}.array_sha256 is invalid")
        if not SHA256_RE.fullmatch(str(row.get("source_sha256") or "")):
            errors.append(f"{location}.source_sha256 is invalid")
        declared_nbytes = row.get("uncompressed_bytes")
        if isinstance(declared_nbytes, int) and declared_nbytes >= 0:
            uncompressed_bytes += declared_nbytes
        else:
            errors.append(f"{location}.uncompressed_bytes is invalid")
        if verify_data:
            try:
                values = np.asarray(array[...])
                if int(values.nbytes) != declared_nbytes:
                    errors.append(f"{location} uncompressed byte count does not match array")
                if _array_sha256(values) != row.get("array_sha256"):
                    errors.append(f"{location} array checksum mismatch")
            except Exception as exc:
                errors.append(f"{location} cannot read array data: {exc}")
        if source_frame_dir is not None:
            try:
                _, source_path = _safe_relative_path(row.get("source_path"), Path(source_frame_dir))
                if not source_path.is_file():
                    errors.append(f"{location} source map is missing: {source_path}")
                elif _sha256_path(source_path) != row.get("source_sha256"):
                    errors.append(f"{location} source checksum mismatch")
                elif source_path.stat().st_size != row.get("source_bytes"):
                    errors.append(f"{location} source byte count mismatch")
            except ArrayStoreError as exc:
                errors.append(f"{location} invalid source: {exc}")
    if root is not None:
        try:
            actual_array_ids = set(root["arrays"].array_keys())
            missing = sorted(expected_array_ids - actual_array_ids)
            unexpected = sorted(actual_array_ids - expected_array_ids)
            if missing:
                errors.append(f"missing Zarr arrays: {', '.join(missing)}")
            if unexpected:
                errors.append(f"unindexed Zarr arrays: {', '.join(unexpected)}")
            attrs = dict(root.attrs)
            if attrs.get("schema_name") != SCHEMA_NAME:
                errors.append("root Zarr attributes have invalid schema_name")
            if attrs.get("schema_version") != SCHEMA_VERSION:
                errors.append("root Zarr attributes have invalid schema_version")
            if attrs.get("artifact_count") != artifact_count:
                errors.append("root Zarr artifact_count does not match catalog")
            if attrs.get("source_manifest_sha256") != source_manifest.get("sha256"):
                errors.append("root Zarr source_manifest_sha256 does not match catalog")
        except Exception as exc:
            errors.append(f"cannot inspect Zarr hierarchy: {exc}")
    if source_frame_dir is not None and source_manifest:
        try:
            _, source_manifest_path = _safe_relative_path(
                source_manifest.get("path"), Path(source_frame_dir)
            )
            if not source_manifest_path.is_file():
                errors.append(f"source map manifest is missing: {source_manifest_path}")
            elif _sha256_path(source_manifest_path) != source_manifest.get("sha256"):
                errors.append("source map manifest checksum mismatch")
        except ArrayStoreError as exc:
            errors.append(f"invalid source map manifest path: {exc}")
    for key, measured in (
        ("artifact_count", artifact_count),
        ("available_count", available_count),
        ("unavailable_count", unavailable_count),
        ("uncompressed_bytes", uncompressed_bytes),
    ):
        if manifest.get(key) != measured:
            errors.append(f"catalog {key} does not match artifacts")
    source_metadata = source_manifest.get("metadata") or {}
    expected_complete = (
        unavailable_count == 0
        and isinstance(source_metadata, Mapping)
        and source_metadata.get("complete", True) is not False
    )
    if manifest.get("complete") != expected_complete:
        errors.append("catalog complete flag does not match source and artifact availability")
    return ValidationResult(
        store_path=store_path,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        artifact_count=artifact_count,
        available_count=available_count,
        unavailable_count=unavailable_count,
        uncompressed_bytes=uncompressed_bytes,
        stored_bytes=_tree_size(store_path) if store_path.is_dir() else 0,
    )


def _normalized_uint8(
    values: np.ndarray,
    minimum: float | None,
    maximum: float | None,
) -> tuple[np.ndarray, float, float, int]:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    invalid = int(data.size - np.count_nonzero(finite))
    finite_values = data[finite]
    if finite_values.size == 0:
        low = 0.0 if minimum is None else float(minimum)
        high = 1.0 if maximum is None else float(maximum)
    else:
        low = float(np.min(finite_values)) if minimum is None else float(minimum)
        high = float(np.max(finite_values)) if maximum is None else float(maximum)
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ArrayStoreError(f"invalid PNG normalization range: {low}, {high}")
    normalized = np.zeros(data.shape, dtype=np.uint8)
    if high > low:
        scaled = np.clip((data[finite] - low) / (high - low), 0.0, 1.0)
        normalized[finite] = np.rint(scaled * 255.0).astype(np.uint8)
    elif finite_values.size:
        normalized[finite] = 255
    return normalized, low, high, invalid


def export_artifact(
    store_path: Path | str,
    artifact_id: str,
    output_path: Path | str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    overwrite: bool = False,
) -> ExportResult:
    """Export one artifact to PFM or PNG without mutating the canonical store."""

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise ArrayStoreError(f"refusing to overwrite existing export: {output_path}")
    store = ArrayStore(store_path)
    values = store.read(artifact_id)
    suffix = output_path.suffix.lower()
    if suffix == ".pfm":
        write_pfm(output_path, values)
        return ExportResult(output_path, artifact_id, "pfm", values.dtype == np.float32, "float32", None, None, 0)
    if suffix != ".png":
        raise ArrayStoreError("export path must end in .pfm or .png")
    Image = _require_pillow()
    if values.ndim == 3 and values.shape[2] not in (1, 3, 4):
        raise ArrayStoreError(f"PNG export requires one, three, or four channels, got {values.shape}")
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[..., 0]
    exact = values.dtype in (np.dtype(np.uint8), np.dtype(np.uint16))
    transform = "identity"
    low = high = None
    invalid = 0
    if exact:
        output_values = values
    else:
        output_values, low, high, invalid = _normalized_uint8(values, minimum, maximum)
        transform = "linear_min_max_to_uint8"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Image.fromarray(output_values).save(output_path, format="PNG", optimize=False, compress_level=9)
    except (OSError, TypeError, ValueError) as exc:
        raise ArrayStoreError(f"cannot write PNG export {output_path}: {exc}") from exc
    return ExportResult(output_path, artifact_id, "png", exact, transform, low, high, invalid)


def _result_json(value: Any) -> str:
    raw = asdict(value)
    normalized = {key: str(item) if isinstance(item, Path) else item for key, item in raw.items()}
    return _canonical_json_bytes(normalized).decode("utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert, inspect, validate, and export canonical DMAP Zarr v3 stores."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="convert a map_manifest.json without modifying it")
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--frame-dir", type=Path)
    convert.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    convert.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    convert.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL)
    convert.add_argument("--allow-incomplete", action="store_true")
    convert.add_argument("--max-uncompressed-bytes", type=int)
    validate = commands.add_parser("validate", help="validate a canonical array store")
    validate.add_argument("--store", type=Path, required=True)
    validate.add_argument("--source-frame-dir", type=Path)
    validate.add_argument("--metadata-only", action="store_true")
    listing = commands.add_parser("list", help="list catalog artifacts as canonical JSON")
    listing.add_argument("--store", type=Path, required=True)
    listing.add_argument("--signal")
    listing.add_argument("--include-unavailable", action="store_true")
    export = commands.add_parser("export", help="export one artifact as PFM or PNG")
    export.add_argument("--store", type=Path, required=True)
    export.add_argument("--artifact", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--minimum", type=float)
    export.add_argument("--maximum", type=float)
    export.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "convert":
            result = convert_map_manifest(
                args.manifest,
                args.output,
                frame_dir=args.frame_dir,
                chunk_size=args.chunk_size,
                shard_size=args.shard_size,
                zstd_level=args.zstd_level,
                allow_incomplete=args.allow_incomplete,
                max_uncompressed_bytes=args.max_uncompressed_bytes,
            )
            sys.stdout.write(_result_json(result))
            return 0
        if args.command == "validate":
            result = validate_store(
                args.store,
                verify_data=not args.metadata_only,
                source_frame_dir=args.source_frame_dir,
            )
            sys.stdout.write(_result_json(result))
            return 0 if result.valid else 1
        if args.command == "list":
            store = ArrayStore(args.store)
            availability = None if args.include_unavailable else "available"
            rows = store.find(signal=args.signal, availability=availability)
            sys.stdout.write(
                _canonical_json_bytes({"artifacts": list(rows), "count": len(rows)}).decode("utf-8")
            )
            return 0
        if args.command == "export":
            result = export_artifact(
                args.store,
                args.artifact,
                args.output,
                minimum=args.minimum,
                maximum=args.maximum,
                overwrite=args.overwrite,
            )
            sys.stdout.write(_result_json(result))
            return 0
    except ArrayStoreError as exc:
        parser.exit(2, f"error: {exc}\n")
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
