#!/usr/bin/env python3
"""Evaluate generic line/plane annotation sidecars against OpenMVS DMAPs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import sys
from typing import Any

import numpy as np


REQUIRED_RANSAC_THRESHOLDS_M = (0.005, 0.010, 0.020, 0.050)
ANNOTATION_EVALUATION_SCHEMA_VERSION = 2
ANNOTATION_SIDECAR_SCHEMA_NAME = "openmvs.dmap.annotation_sidecar"
ANNOTATION_SIDECAR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FitConfig:
    ransac_threshold_m: float
    ransac_thresholds_m: tuple[float, ...]
    edge_ribbon_px: float | None
    max_ransac_points: int
    ransac_trials: int
    seed: int
    min_confidence: float | None
    plane_grid: int
    line_bins: int
    max_visual_points: int
    edge_ribbon_source_px: float | None = None
    edge_ribbon_angle_mrad: float | None = None


def validate_fit_config(config: FitConfig) -> None:
    if not math.isfinite(config.ransac_threshold_m) or config.ransac_threshold_m <= 0.0:
        raise ValueError("RANSAC threshold must be a positive finite distance in meters")
    thresholds = tuple(sorted({float(value) for value in config.ransac_thresholds_m}))
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("RANSAC evaluation thresholds must be positive finite distances")
    missing = [
        required
        for required in REQUIRED_RANSAC_THRESHOLDS_M
        if not any(math.isclose(required, value, rel_tol=0.0, abs_tol=1e-12) for value in thresholds)
    ]
    if missing:
        missing_mm = ", ".join(f"{value * 1000.0:g}" for value in missing)
        raise ValueError(f"RANSAC evaluation thresholds must include 5/10/20/50 mm; missing {missing_mm} mm")
    if config.max_ransac_points < 3:
        raise ValueError("max_ransac_points must be at least 3")
    if config.ransac_trials <= 0:
        raise ValueError("ransac_trials must be positive")
    if config.plane_grid <= 0 or config.line_bins <= 0 or config.max_visual_points <= 0:
        raise ValueError("plane_grid, line_bins, and max_visual_points must be positive")
    ribbon_values = [
        config.edge_ribbon_px,
        config.edge_ribbon_source_px,
        config.edge_ribbon_angle_mrad,
    ]
    enabled = [value for value in ribbon_values if value is not None]
    if len(enabled) != 1:
        raise ValueError(
            "configure exactly one edge ribbon width: depth pixels, annotation/source pixels, or angular mrad"
        )
    if not math.isfinite(float(enabled[0])) or float(enabled[0]) <= 0.0:
        raise ValueError("edge ribbon width must be positive and finite")


def stable_annotation_seed(
    base_seed: int,
    *,
    scan_id: str,
    frame_id: str,
    kind: str,
    object_id: Any,
    chunk_id: Any,
) -> int:
    """Derive an order-independent per-chunk seed shared by baseline and variants."""
    identity = "\x1f".join(
        [scan_id.lower(), frame_id.lower(), kind, str(object_id or ""), str(chunk_id or "")]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "little")) % (2**32)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def find_row(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def scale_k(k: np.ndarray, sx: float, sy: float) -> np.ndarray:
    return np.array(
        [
            [k[0, 0] * sx, k[0, 1] * sx, (k[0, 2] + 0.5) * sx - 0.5],
            [0.0, k[1, 1] * sy, (k[1, 2] + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_dmap(path: Path) -> dict[str, Any]:
    def read_exact(handle: Any, size: int, label: str) -> bytes:
        payload = handle.read(size)
        if len(payload) != size:
            raise ValueError(f"truncated DMAP {label}: {path}")
        return payload

    def decode_octahedral_normals(packed: np.ndarray) -> np.ndarray:
        encoded = packed.reshape(depth_height, depth_width, 2)
        invalid = np.all(encoded == np.int16(-32768), axis=2)
        px = encoded[..., 0].astype(np.float32) * np.float32(1.0 / 32767.0)
        py = encoded[..., 1].astype(np.float32) * np.float32(1.0 / 32767.0)
        x = px.copy()
        y = py.copy()
        z = np.float32(1.0) - np.abs(px) - np.abs(py)
        folded = z < 0
        x[folded] = (np.float32(1.0) - np.abs(py[folded])) * np.where(
            px[folded] < 0, np.float32(-1.0), np.float32(1.0)
        )
        y[folded] = (np.float32(1.0) - np.abs(px[folded])) * np.where(
            py[folded] < 0, np.float32(-1.0), np.float32(1.0)
        )
        normals = np.stack((x, y, z), axis=2)
        norms = np.linalg.norm(normals, axis=2)
        valid = ~invalid & np.isfinite(norms) & (norms > 0)
        normals[valid] /= norms[valid, None]
        normals[~valid] = 0
        return normals.astype(np.float32, copy=False)

    with path.open("rb") as f:
        file_type = read_exact(f, 2, "magic").decode("ascii", errors="replace")
        content_type = int(np.frombuffer(read_exact(f, 1, "content type"), dtype=np.uint8)[0])
        depth_exp = int(np.frombuffer(read_exact(f, 1, "format byte"), dtype=np.int8)[0])
        image_width, image_height = [
            int(v) for v in np.frombuffer(read_exact(f, 8, "image size"), dtype="<u4")
        ]
        depth_width, depth_height = [
            int(v) for v in np.frombuffer(read_exact(f, 8, "depth size"), dtype="<u4")
        ]
        if (
            file_type not in {"DR", "D2"}
            or not (content_type & 1)
            or depth_width <= 0
            or depth_height <= 0
            or image_width < depth_width
            or image_height < depth_height
        ):
            raise ValueError(f"invalid DMAP header: {path}")
        depth_min, depth_max = [
            float(v) for v in np.frombuffer(read_exact(f, 8, "depth range"), dtype="<f4")
        ]
        confidence_scale = (
            float(np.frombuffer(read_exact(f, 4, "confidence scale"), dtype="<f4")[0])
            if file_type == "D2"
            else 1.0
        )
        if file_type == "D2" and (not math.isfinite(confidence_scale) or confidence_scale <= 0):
            raise ValueError(f"invalid DMAP confidence scale: {path}")
        file_name_size = int(
            np.frombuffer(read_exact(f, 2, "image-name length"), dtype="<u2")[0]
        )
        image_name = read_exact(f, file_name_size, "image name").decode("utf-8")
        view_ids_size = int(
            np.frombuffer(read_exact(f, 4, "view count"), dtype="<u4")[0]
        )
        if view_ids_size <= 0 or view_ids_size >= 256:
            raise ValueError(f"invalid DMAP view count: {path}")
        view_ids = np.frombuffer(
            read_exact(f, 4 * view_ids_size, "view IDs"), dtype="<u4"
        ).copy()
        k = np.frombuffer(read_exact(f, 72, "intrinsics"), dtype="<f8").copy().reshape(3, 3)
        r = np.frombuffer(read_exact(f, 72, "rotation"), dtype="<f8").copy().reshape(3, 3)
        c = np.frombuffer(read_exact(f, 24, "camera center"), dtype="<f8").copy()
        map_size = depth_width * depth_height
        if file_type == "D2":
            depth = np.frombuffer(
                read_exact(f, 2 * map_size, "quantized depth map"), dtype="<f2"
            ).astype(np.float32).reshape(depth_height, depth_width)
            depth *= np.float32(math.ldexp(1.0, depth_exp))
        else:
            depth = np.frombuffer(
                read_exact(f, 4 * map_size, "depth map"), dtype="<f4"
            ).copy().reshape(depth_height, depth_width)
        out: dict[str, Any] = {
            "path": str(path),
            "format": file_type,
            "content_type": content_type,
            "depth_exponent": depth_exp if file_type == "D2" else None,
            "confidence_scale": confidence_scale if file_type == "D2" else None,
            "image_width": image_width,
            "image_height": image_height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "image_name": image_name,
            "reference_view_id": int(view_ids[0]) if len(view_ids) else -1,
            "neighbor_view_ids": [int(v) for v in view_ids[1:]],
            "K": k,
            "R": r,
            "C": c,
            "depth_map": depth,
        }
        if content_type & 2:
            if file_type == "D2":
                packed_normals = np.frombuffer(
                    read_exact(f, 4 * map_size, "quantized normal map"), dtype="<i2"
                ).copy()
                out["normal_map"] = decode_octahedral_normals(packed_normals)
            else:
                out["normal_map"] = np.frombuffer(
                    read_exact(f, 4 * map_size * 3, "normal map"), dtype="<f4"
                ).copy().reshape(depth_height, depth_width, 3)
        if content_type & 4:
            if file_type == "D2":
                confidence = np.frombuffer(
                    read_exact(f, map_size, "quantized confidence map"), dtype=np.uint8
                ).astype(np.float32).reshape(depth_height, depth_width)
                out["confidence_map"] = confidence * np.float32(confidence_scale / 255.0)
            else:
                out["confidence_map"] = np.frombuffer(
                    read_exact(f, 4 * map_size, "confidence map"), dtype="<f4"
                ).copy().reshape(depth_height, depth_width)
        if content_type & 8:
            out["views_map"] = np.frombuffer(
                read_exact(f, map_size * 4, "views map"), dtype=np.uint8
            ).copy().reshape(depth_height, depth_width, 4)
        return out


def camera_matrix_from_model(camera: dict[str, Any]) -> np.ndarray:
    model = camera.get("camera_model")
    params = [float(v) for v in camera.get("camera_params", [])]
    if model == "RADIAL":
        f, cx, cy, _k1, _k2 = params
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if model == "PINHOLE":
        fx, fy, cx, cy = params
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"unsupported camera model: {model}")


def undistort_radial_points(points: np.ndarray, camera: dict[str, Any], iterations: int = 8) -> np.ndarray:
    model = camera.get("camera_model")
    params = [float(v) for v in camera.get("camera_params", [])]
    if model != "RADIAL":
        raise ValueError(f"distorted annotation mapping expects RADIAL camera, got {model}")
    f, cx, cy, k1, k2 = params
    xd = (points[:, 0] - cx) / f
    yd = (points[:, 1] - cy) / f
    xu = xd.copy()
    yu = yd.copy()
    for _ in range(iterations):
        r2 = xu * xu + yu * yu
        scale = 1.0 + k1 * r2 + k2 * r2 * r2
        safe = np.where(np.abs(scale) > 1e-12, scale, 1.0)
        xu = xd / safe
        yu = yd / safe
    return np.column_stack([xu, yu])


def annotation_points_to_depth_pixels(
    points: np.ndarray,
    dmap: dict[str, Any],
    pipeline_row: dict[str, Any],
    annotation_space: str,
) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 2)
    if annotation_space == "distorted":
        normalized = undistort_radial_points(points, pipeline_row["camera_distorted"])
        hom = np.column_stack([normalized, np.ones(len(normalized), dtype=np.float64)])
        projected = (np.asarray(dmap["K"], dtype=np.float64) @ hom.T).T
        return projected[:, :2] / projected[:, 2:3]
    if annotation_space == "final":
        final_camera = pipeline_row.get("camera_final") or pipeline_row.get("camera_undistorted")
        if not final_camera:
            raise ValueError("annotation-space=final requires camera_final or camera_undistorted")
        sx = float(dmap["depth_width"]) / float(final_camera["width"])
        sy = float(dmap["depth_height"]) / float(final_camera["height"])
        return np.column_stack([points[:, 0] * sx, points[:, 1] * sy])
    raise ValueError(f"unsupported annotation space: {annotation_space}")


def edge_ribbon_radius_depth_px(
    *,
    raw_segment: np.ndarray,
    dmap: dict[str, Any],
    pipeline_row: dict[str, Any],
    annotation_space: str,
    config: FitConfig,
) -> tuple[float, str]:
    """Resolve the configured edge half-width in the current DMAP pixel grid."""
    if config.edge_ribbon_px is not None:
        return float(config.edge_ribbon_px), "depth_pixels_legacy"
    if config.edge_ribbon_angle_mrad is not None:
        angle_rad = float(config.edge_ribbon_angle_mrad) / 1000.0
        if angle_rad >= math.pi / 2.0:
            raise ValueError("edge ribbon angular half-width must be below pi/2 radians")
        k = np.asarray(dmap["K"], dtype=np.float64)
        focal_px = math.sqrt(abs(float(k[0, 0] * k[1, 1])))
        radius = math.tan(angle_rad) * focal_px
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("edge ribbon angular width produced an invalid DMAP radius")
        return radius, "angular_mrad"
    if config.edge_ribbon_source_px is None:
        raise ValueError("edge ribbon source-pixel width is unavailable")
    if raw_segment.shape != (2, 2):
        raise ValueError("source-pixel edge ribbon conversion requires a two-point segment")
    vector = raw_segment[1] - raw_segment[0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise ValueError("source-pixel edge ribbon conversion requires a non-degenerate segment")
    perpendicular = np.array([-vector[1], vector[0]], dtype=np.float64) / length
    midpoint = raw_segment.mean(axis=0)
    offset = perpendicular * float(config.edge_ribbon_source_px)
    probes = np.vstack([midpoint - offset, midpoint + offset])
    transformed = annotation_points_to_depth_pixels(probes, dmap, pipeline_row, annotation_space)
    radius = 0.5 * float(np.linalg.norm(transformed[1] - transformed[0]))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("source-pixel edge ribbon width produced an invalid DMAP radius")
    return radius, "annotation_source_pixels"


def annotation_db_path(dataset_root: Path) -> Path:
    """Resolve the imported annotation sidecar or the legacy dataset DB."""
    imported = dataset_root / "annotations" / "db"
    return imported if imported.is_dir() else dataset_root / "db"


def annotation_image_mapping_path(dataset_root: Path, scan_id: str) -> Path:
    return dataset_root / "annotations" / "image_mappings" / f"{scan_id}.json"


def load_annotation_sidecar(path: Path, expected_scene_id: str | None = None) -> dict[str, Any]:
    """Load the public, self-contained annotation-provider interchange format."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read annotation sidecar {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("annotation sidecar must be a JSON object")
    if value.get("schema_name") != ANNOTATION_SIDECAR_SCHEMA_NAME:
        raise ValueError(
            f"annotation sidecar schema_name must be {ANNOTATION_SIDECAR_SCHEMA_NAME!r}"
        )
    if value.get("schema_version") != ANNOTATION_SIDECAR_SCHEMA_VERSION:
        raise ValueError(
            f"annotation sidecar schema_version must be {ANNOTATION_SIDECAR_SCHEMA_VERSION}"
        )
    scene_id = str(value.get("scene_id") or "")
    if not scene_id:
        raise ValueError("annotation sidecar scene_id must be non-empty")
    if expected_scene_id is not None and scene_id != expected_scene_id:
        raise ValueError(
            f"annotation sidecar scene_id {scene_id!r} does not match {expected_scene_id!r}"
        )
    mapping = value.get("image_mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("annotation sidecar image_mapping must be a non-empty object")
    frames = value.get("frames")
    annotations = value.get("annotations")
    if not isinstance(frames, list):
        raise ValueError("annotation sidecar frames must be an array")
    if not isinstance(annotations, dict):
        raise ValueError("annotation sidecar annotations must be an object")
    frame_ids: list[str] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or not str(frame.get("id") or ""):
            raise ValueError(f"annotation sidecar frames[{index}].id must be non-empty")
        frame_ids.append(str(frame["id"]).lower())
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("annotation sidecar frame IDs must be unique ignoring case")
    normalized_mapping: dict[str, str] = {}
    for raw_frame_id, raw_image_name in mapping.items():
        frame_id = str(raw_frame_id).lower()
        image_name = str(raw_image_name)
        if not frame_id or not image_name:
            raise ValueError("annotation sidecar image_mapping keys and values must be non-empty")
        if frame_id in normalized_mapping:
            raise ValueError(
                "annotation sidecar image_mapping frame IDs must be unique ignoring case"
            )
        if frame_id not in frame_ids:
            raise ValueError(
                f"annotation sidecar image_mapping references unknown frame {raw_frame_id!r}"
            )
        normalized_mapping[frame_id] = image_name
    for collection in ("controlEdges", "controlPlanes"):
        rows = annotations.get(collection, [])
        if not isinstance(rows, list):
            raise ValueError(f"annotation sidecar annotations.{collection} must be an array")
    for key in ("camera_distorted", "camera_final", "camera_undistorted"):
        if value.get(key) is not None and not isinstance(value[key], dict):
            raise ValueError(f"annotation sidecar {key} must be an object")
    pipeline = {
        "scan_id": scene_id,
        "camera_distorted": value.get("camera_distorted"),
        "camera_final": value.get("camera_final"),
        "camera_undistorted": value.get("camera_undistorted"),
    }
    if not pipeline["camera_distorted"] and not (
        pipeline["camera_final"] or pipeline["camera_undistorted"]
    ):
        raise ValueError(
            "annotation sidecar must define camera_distorted or camera_final/camera_undistorted"
        )
    return {
        "scene": {"id": scene_id},
        "review": {
            "scan_id": scene_id,
            "frames": frames,
            "annotations": annotations,
        },
        "pipeline": pipeline,
        "image_mapping": normalized_mapping,
        "source": str(path),
    }


def load_image_mapping(
    *,
    scan_id: str,
    dataset_root: Path,
    cache_root: Path,
    explicit_mapping: Path | None,
) -> tuple[dict[str, str], str]:
    candidates: list[tuple[Path, str]] = []
    if explicit_mapping is not None:
        candidates.append((explicit_mapping, "explicit"))
    candidates.append((cache_root / scan_id / "image_id_mapping.json", "cache"))
    candidates.append((annotation_image_mapping_path(dataset_root, scan_id), "dataset_annotation_sidecar"))
    for path, source in candidates:
        if path.exists():
            data = json.loads(path.read_text())
            provenance = str(data.get("provenance") or "legacy")
            return (
                {str(k).lower(): str(v) for k, v in data["uuid_to_image_name"].items()},
                f"{source}:{provenance}:{path}",
            )

    raise FileNotFoundError(f"could not find image_id_mapping.json for scan {scan_id}")


def normalize_image_name(name: str) -> str:
    return Path(name).name


def point_dict_to_array(points: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)


def chunk_points(chunk: dict[str, Any], kind: str) -> np.ndarray:
    if kind == "plane":
        return point_dict_to_array(chunk.get("points") or [])
    if kind == "edge":
        start = chunk.get("start")
        end = chunk.get("end")
        if not isinstance(start, dict) or not isinstance(end, dict):
            return np.empty((0, 2), dtype=np.float64)
        return point_dict_to_array([start, end])
    raise ValueError(f"unsupported annotation kind: {kind}")


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if len(polygon) < 3:
        return mask
    min_x = max(0, int(math.floor(float(np.min(polygon[:, 0])))))
    max_x = min(width - 1, int(math.ceil(float(np.max(polygon[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(polygon[:, 1])))))
    max_y = min(height - 1, int(math.ceil(float(np.max(polygon[:, 1])))))
    if min_x > max_x or min_y > max_y:
        return mask

    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    inside = np.zeros(grid_x.shape, dtype=bool)
    x0 = polygon[:, 0]
    y0 = polygon[:, 1]
    x1 = np.roll(x0, -1)
    y1 = np.roll(y0, -1)
    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        crosses = ((ya > grid_y) != (yb > grid_y)) & (
            grid_x < (xb - xa) * (grid_y - ya) / ((yb - ya) if abs(yb - ya) > 1e-12 else 1e-12) + xa
        )
        inside ^= crosses
    mask[min_y : max_y + 1, min_x : max_x + 1] = inside
    return mask


def edge_ribbon_mask(shape: tuple[int, int], segment: np.ndarray, radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    t_map = np.full((height, width), np.nan, dtype=np.float32)
    if len(segment) != 2:
        return mask, t_map
    a = segment[0]
    b = segment[1]
    vec = b - a
    length_sq = float(np.dot(vec, vec))
    if length_sq <= 1e-12:
        return mask, t_map
    margin = int(math.ceil(radius_px + 1.0))
    min_x = max(0, int(math.floor(min(a[0], b[0]) - margin)))
    max_x = min(width - 1, int(math.ceil(max(a[0], b[0]) + margin)))
    min_y = max(0, int(math.floor(min(a[1], b[1]) - margin)))
    max_y = min(height - 1, int(math.ceil(max(a[1], b[1]) + margin)))
    if min_x > max_x or min_y > max_y:
        return mask, t_map

    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = grid_x - a[0]
    py = grid_y - a[1]
    t = (px * vec[0] + py * vec[1]) / length_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    closest_x = a[0] + t_clamped * vec[0]
    closest_y = a[1] + t_clamped * vec[1]
    dist = np.sqrt((grid_x - closest_x) ** 2 + (grid_y - closest_y) ** 2)
    inside = (t >= 0.0) & (t <= 1.0) & (dist <= radius_px)
    mask[min_y : max_y + 1, min_x : max_x + 1] = inside
    t_sub = t_map[min_y : max_y + 1, min_x : max_x + 1]
    t_sub[inside] = t[inside].astype(np.float32)
    return mask, t_map


def grid_coverage(mask: np.ndarray, valid: np.ndarray, grid: int) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    width = max(1, xmax - xmin + 1)
    height = max(1, ymax - ymin + 1)

    def cells_for(coords_y: np.ndarray, coords_x: np.ndarray) -> set[tuple[int, int]]:
        cx = np.clip(((coords_x - xmin) * grid) // width, 0, grid - 1)
        cy = np.clip(((coords_y - ymin) * grid) // height, 0, grid - 1)
        return set(zip(cy.astype(int), cx.astype(int)))

    mask_cells = cells_for(ys, xs)
    vys, vxs = np.nonzero(mask & valid)
    if len(vxs) == 0 or not mask_cells:
        return 0.0
    valid_cells = cells_for(vys, vxs)
    return float(len(valid_cells) / len(mask_cells))


def line_bin_coverage(mask: np.ndarray, valid: np.ndarray, t_map: np.ndarray, bins: int) -> float:
    t_mask = t_map[mask]
    t_mask = t_mask[np.isfinite(t_mask)]
    if t_mask.size == 0:
        return 0.0
    mask_bins = set(np.clip((t_mask * bins).astype(int), 0, bins - 1).tolist())
    t_valid = t_map[mask & valid]
    t_valid = t_valid[np.isfinite(t_valid)]
    if t_valid.size == 0 or not mask_bins:
        return 0.0
    valid_bins = set(np.clip((t_valid * bins).astype(int), 0, bins - 1).tolist())
    return float(len(valid_bins) / len(mask_bins))


def valid_depth_mask(dmap: dict[str, Any], min_confidence: float | None) -> np.ndarray:
    depth = dmap["depth_map"]
    valid = np.isfinite(depth) & (depth > 0)
    if min_confidence is not None and "confidence_map" in dmap:
        conf = dmap["confidence_map"]
        valid &= np.isfinite(conf) & (conf >= min_confidence)
    return valid


def unproject_pixels(dmap: dict[str, Any], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    depth = dmap["depth_map"][ys, xs].astype(np.float64)
    pixels = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(len(xs), dtype=np.float64)])
    rays = (np.linalg.inv(np.asarray(dmap["K"], dtype=np.float64)) @ pixels.T).T
    x_cam = rays * depth[:, None]
    r = np.asarray(dmap["R"], dtype=np.float64)
    c = np.asarray(dmap["C"], dtype=np.float64)
    return c[None, :] + (r.T @ x_cam.T).T


def maybe_subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def plane_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    p0, p1, p2 = points[:3]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return normal / norm, p0


def line_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    p0, p1 = points[:2]
    direction = p1 - p0
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm, p0


def plane_residuals(points: np.ndarray, normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.abs((points - point[None, :]) @ normal)


def line_residuals(points: np.ndarray, direction: np.ndarray, point: np.ndarray) -> np.ndarray:
    offsets = points - point[None, :]
    return np.linalg.norm(np.cross(offsets, direction[None, :]), axis=1)


def refine_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return normal / norm, centroid


def refine_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(points) < 2:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vt[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm, centroid


def residual_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {}
    median = float(np.median(values))
    return {
        f"{prefix}_mean_m": float(np.mean(values)),
        f"{prefix}_median_m": median,
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(values * values))),
        f"{prefix}_mad_m": float(np.median(np.abs(values - median))),
        f"{prefix}_p90_m": float(np.percentile(values, 90)),
        f"{prefix}_p95_m": float(np.percentile(values, 95)),
        f"{prefix}_max_m": float(np.max(values)),
    }


def threshold_curve_stats(values: np.ndarray, thresholds_m: tuple[float, ...]) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    thresholds = sorted({float(value) for value in thresholds_m if value > 0.0})
    if clean.size == 0 or not thresholds:
        return {}
    fractions = [float(np.mean(clean <= threshold)) for threshold in thresholds]
    normalized_x = np.asarray([0.0, *thresholds], dtype=np.float64) / thresholds[-1]
    normalized_y = np.asarray([0.0, *fractions], dtype=np.float64)
    out = {
        f"inlier_fraction_{int(round(threshold * 1000.0))}mm": fraction
        for threshold, fraction in zip(thresholds, fractions)
    }
    out["inlier_threshold_auc"] = float(np.trapezoid(normalized_y, normalized_x))
    return out


def confidence_stats(dmap: dict[str, Any], selected_mask: np.ndarray) -> dict[str, float]:
    if "confidence_map" not in dmap:
        return {}
    values = dmap["confidence_map"][selected_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "confidence_mean": float(np.mean(values)),
        "confidence_median": float(np.median(values)),
        "confidence_p10": float(np.percentile(values, 10)),
        "confidence_p90": float(np.percentile(values, 90)),
    }


def escape_xml(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    values = [int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)]
    return "#" + "".join(f"{value:02x}" for value in values)


def residual_color(value: float, threshold: float) -> str:
    if not math.isfinite(value):
        return "#6b7280"
    ratio = value / max(threshold, 1e-12)
    if ratio <= 1.0:
        return lerp_color((37, 99, 235), (34, 197, 94), ratio)
    return lerp_color((250, 204, 21), (220, 38, 38), min(1.0, ratio - 1.0))


def sample_indices(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=limit, replace=False))


def svg_transform(
    x: float,
    y: float,
    scale: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float]:
    return offset_x + x * scale, offset_y + y * scale


def write_fit_overlay_svg(
    path: Path,
    *,
    image_shape: tuple[int, int],
    transformed_points: np.ndarray,
    kind: str,
    valid_xs: np.ndarray,
    valid_ys: np.ndarray,
    invalid_xs: np.ndarray,
    invalid_ys: np.ndarray,
    residuals: np.ndarray | None,
    config: FitConfig,
    title: str,
    edge_ribbon_radius_px: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_shape
    max_plot_w = 980.0
    max_plot_h = 760.0
    scale = min(max_plot_w / max(width, 1), max_plot_h / max(height, 1))
    plot_w = width * scale
    plot_h = height * scale
    pad_l, pad_t = 54.0, 72.0
    svg_w = int(plot_w + pad_l + 32)
    svg_h = int(plot_h + pad_t + 82)
    point_radius = max(0.7, min(2.2, scale * 1.2))
    invalid_limit = max(250, config.max_visual_points // 5)
    valid_indices = sample_indices(len(valid_xs), config.max_visual_points, config.seed + 17)
    invalid_indices = sample_indices(len(invalid_xs), invalid_limit, config.seed + 31)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{svg_w / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" '
        f'font-size="18" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
        f'<text x="{svg_w / 2:.1f}" y="50" text-anchor="middle" font-family="Arial" '
        f'font-size="12" fill="#4b5563">DMAP pixel space. Blue/green are below threshold; yellow/red exceed threshold; gray is masked but invalid depth.</text>',
        f'<rect x="{pad_l:.1f}" y="{pad_t:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    for i in invalid_indices:
        x, y = svg_transform(float(invalid_xs[i]) + 0.5, float(invalid_ys[i]) + 0.5, scale, pad_l, pad_t)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius:.2f}" fill="#9ca3af" fill-opacity="0.55"/>')

    for j, i in enumerate(valid_indices):
        x, y = svg_transform(float(valid_xs[i]) + 0.5, float(valid_ys[i]) + 0.5, scale, pad_l, pad_t)
        color = "#2563eb"
        if residuals is not None and len(residuals):
            color = residual_color(float(residuals[i]), config.ransac_threshold_m)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{point_radius:.2f}" fill="{color}" fill-opacity="0.74"/>')

    if len(transformed_points):
        if kind == "plane" and len(transformed_points) >= 3:
            coords = []
            for x_raw, y_raw in transformed_points:
                x, y = svg_transform(float(x_raw), float(y_raw), scale, pad_l, pad_t)
                coords.append(f"{x:.2f},{y:.2f}")
            parts.append(
                f'<polygon points="{" ".join(coords)}" fill="none" stroke="#00a6d6" stroke-width="2.2"/>'
            )
        elif kind == "edge" and len(transformed_points) >= 2:
            x1, y1 = svg_transform(float(transformed_points[0, 0]), float(transformed_points[0, 1]), scale, pad_l, pad_t)
            x2, y2 = svg_transform(float(transformed_points[1, 0]), float(transformed_points[1, 1]), scale, pad_l, pad_t)
            radius_px = edge_ribbon_radius_px
            if radius_px is None:
                radius_px = config.edge_ribbon_px
            ribbon_width = max(2.0, float(radius_px or 0.0) * scale * 2.0)
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#ff7a00" stroke-opacity="0.18" stroke-width="{ribbon_width:.2f}" stroke-linecap="round"/>'
            )
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#ff7a00" stroke-width="2.4" stroke-linecap="round"/>'
            )

    legend_y = pad_t + plot_h + 28
    legend_items = [("#2563eb", "low residual"), ("#22c55e", "inlier threshold"), ("#dc2626", "high residual"), ("#9ca3af", "invalid depth")]
    x0 = pad_l
    for color, label in legend_items:
        parts.append(f'<circle cx="{x0:.1f}" cy="{legend_y:.1f}" r="5" fill="{color}" fill-opacity="0.8"/>')
        parts.append(f'<text x="{x0 + 12:.1f}" y="{legend_y + 4:.1f}" font-family="Arial" font-size="12" fill="#374151">{escape_xml(label)}</text>')
        x0 += 132
    parts.append(
        f'<text x="{pad_l:.1f}" y="{legend_y + 28:.1f}" font-family="Arial" font-size="12" fill="#4b5563">'
        f'showing {len(valid_indices)} / {len(valid_xs)} valid points and {len(invalid_indices)} / {len(invalid_xs)} invalid pixels</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def write_residual_histogram_svg(path: Path, residuals: np.ndarray, threshold: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 860, 360
    margin_l, margin_r, margin_t, margin_b = 68, 24, 54, 58
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    clean = residuals[np.isfinite(residuals)]
    if clean.size == 0:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>'
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="16">No residuals</text></svg>\n'
        )
        return
    x_max = max(float(np.percentile(clean, 99)), threshold * 2.0, float(clean.max()), 1e-6)
    x_max = min(x_max, max(float(np.percentile(clean, 99.5)), threshold * 8.0))
    bins = np.linspace(0.0, x_max, 31)
    counts, edges = np.histogram(np.clip(clean, 0.0, x_max), bins=bins)
    y_max = max(1, int(counts.max()))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
    ]
    for i in range(6):
        val = y_max * i / 5.0
        y = margin_t + chart_h - (val / y_max) * chart_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{val:.0f}</text>')
    for count, x0, x1 in zip(counts, edges[:-1], edges[1:]):
        bar_x = margin_l + (x0 / x_max) * chart_w
        bar_w = max(1.0, ((x1 - x0) / x_max) * chart_w - 1.0)
        bar_h = (count / y_max) * chart_h
        bar_y = margin_t + chart_h - bar_h
        color = residual_color(float((x0 + x1) * 0.5), threshold)
        parts.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" fill-opacity="0.88"/>')
    threshold_x = margin_l + min(threshold / x_max, 1.0) * chart_w
    parts.append(f'<line x1="{threshold_x:.1f}" y1="{margin_t}" x2="{threshold_x:.1f}" y2="{margin_t + chart_h}" stroke="#111827" stroke-width="2" stroke-dasharray="5,4"/>')
    parts.append(f'<text x="{threshold_x + 6:.1f}" y="{margin_t + 14}" font-family="Arial" font-size="12" fill="#111827">threshold {threshold:.3f} m</text>')
    for i in range(6):
        val = x_max * i / 5.0
        x = margin_l + (val / x_max) * chart_w
        parts.append(f'<text x="{x:.1f}" y="{height - margin_b + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#4b5563">{val:.3f}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="12" fill="#374151">residual distance (m)</text>')
    parts.append(f'<text x="18" y="{height/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.1f})" font-family="Arial" font-size="12" fill="#374151">pixels</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def residuals_from_row(kind: str, points: np.ndarray, row: dict[str, Any]) -> np.ndarray | None:
    if row.get("fit_status") != "ok":
        return None
    if kind == "plane":
        normal = np.array([row["plane_normal_x"], row["plane_normal_y"], row["plane_normal_z"]], dtype=np.float64)
        point = np.array([row["model_point_x"], row["model_point_y"], row["model_point_z"]], dtype=np.float64)
        return plane_residuals(points, normal, point)
    if kind == "edge":
        direction = np.array([row["line_direction_x"], row["line_direction_y"], row["line_direction_z"]], dtype=np.float64)
        point = np.array([row["model_point_x"], row["model_point_y"], row["model_point_z"]], dtype=np.float64)
        return line_residuals(points, direction, point)
    return None


def write_overall_visual_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{int(row.get('index', i + 1)):02d}" for i, row in enumerate(rows)]
    coverage = [float(row.get("coverage_fraction") or 0.0) for row in rows]
    inliers = [float(row.get("inlier_fraction") or 0.0) for row in rows]
    p95_cm = [float(row.get("all_residual_p95_m") or 0.0) * 100.0 for row in rows]
    width, height = 980, 420
    margin_l, margin_r, margin_t, margin_b = 68, 90, 58, 72
    chart_w = width - margin_l - margin_r
    chart_h = height - margin_t - margin_b
    group_w = chart_w / max(1, len(rows))
    max_p95 = max(1.0, max(p95_cm))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700" fill="#111827">Annotation Fit Summary</text>',
    ]
    for i in range(6):
        val = i / 5.0
        y = margin_t + chart_h - val * chart_h
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#4b5563">{val:.1f}</text>')
    for i, label in enumerate(labels):
        x0 = margin_l + i * group_w + group_w * 0.14
        bar_w = max(5.0, group_w * 0.23)
        for j, (value, color) in enumerate([(coverage[i], "#2563eb"), (inliers[i], "#16a34a")]):
            h = max(0.0, min(1.0, value)) * chart_h
            y = margin_t + chart_h - h
            parts.append(f'<rect x="{x0 + j * (bar_w + 2):.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
        p95_h = min(1.0, p95_cm[i] / max_p95) * chart_h
        px = x0 + 2 * (bar_w + 2) + bar_w * 0.5
        py = margin_t + chart_h - p95_h
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#dc2626"/>')
        parts.append(f'<text x="{margin_l + i * group_w + group_w / 2:.1f}" y="{height - 46}" text-anchor="middle" font-family="Arial" font-size="11" fill="#374151">{label}</text>')
    parts.append(f'<text x="{width - margin_r + 8}" y="{margin_t + 4}" font-family="Arial" font-size="11" fill="#dc2626">red dot: P95 residual</text>')
    parts.append(f'<text x="{width - margin_r + 8}" y="{margin_t + 20}" font-family="Arial" font-size="11" fill="#dc2626">max {max_p95:.1f} cm</text>')
    legend_y = height - 20
    for x, color, label in [(margin_l, "#2563eb", "coverage"), (margin_l + 120, "#16a34a", "inlier fraction"), (margin_l + 270, "#dc2626", "P95 residual")]:
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}" font-family="Arial" font-size="12" fill="#374151">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def ransac_plane(points: np.ndarray, config: FitConfig) -> dict[str, Any]:
    if len(points) < 3:
        return {"fit_status": "skipped_too_few_points", "min_required_points": 3}
    rng = np.random.default_rng(config.seed)
    fit_points = maybe_subsample(points, config.max_ransac_points, rng)
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_error = float("inf")
    for _ in range(config.ransac_trials):
        sample = fit_points[rng.choice(len(fit_points), size=3, replace=False)]
        model = plane_from_points(sample)
        if model is None:
            continue
        normal, point = model
        residuals = plane_residuals(fit_points, normal, point)
        inliers = residuals <= config.ransac_threshold_m
        count = int(inliers.sum())
        error = float(np.median(residuals[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_count = count
            best_error = error
            best_inliers = inliers
    if best_inliers is None or best_count < 3:
        return {"fit_status": "failed_ransac", "min_required_points": 3}
    refined = refine_plane(fit_points[best_inliers])
    if refined is None:
        return {"fit_status": "failed_refine", "min_required_points": 3}
    normal, point = refined
    residuals_all = plane_residuals(points, normal, point)
    inliers_all = residuals_all <= config.ransac_threshold_m
    inlier_residuals = residuals_all[inliers_all]
    d = -float(normal @ point)
    out: dict[str, Any] = {
        "fit_status": "ok",
        "ransac_points": int(len(fit_points)),
        "model": "plane",
        "inlier_count": int(inliers_all.sum()),
        "inlier_fraction": float(inliers_all.mean()) if len(inliers_all) else 0.0,
        "plane_normal_x": float(normal[0]),
        "plane_normal_y": float(normal[1]),
        "plane_normal_z": float(normal[2]),
        "plane_d": d,
        "model_point_x": float(point[0]),
        "model_point_y": float(point[1]),
        "model_point_z": float(point[2]),
    }
    out.update(residual_stats(residuals_all, "all_residual"))
    out.update(residual_stats(inlier_residuals, "inlier_residual"))
    out.update(threshold_curve_stats(residuals_all, config.ransac_thresholds_m))
    return out


def ransac_line(points: np.ndarray, config: FitConfig) -> dict[str, Any]:
    if len(points) < 2:
        return {"fit_status": "skipped_too_few_points", "min_required_points": 2}
    rng = np.random.default_rng(config.seed)
    fit_points = maybe_subsample(points, config.max_ransac_points, rng)
    best_inliers: np.ndarray | None = None
    best_count = -1
    best_error = float("inf")
    for _ in range(config.ransac_trials):
        sample = fit_points[rng.choice(len(fit_points), size=2, replace=False)]
        model = line_from_points(sample)
        if model is None:
            continue
        direction, point = model
        residuals = line_residuals(fit_points, direction, point)
        inliers = residuals <= config.ransac_threshold_m
        count = int(inliers.sum())
        error = float(np.median(residuals[inliers])) if count else float("inf")
        if count > best_count or (count == best_count and error < best_error):
            best_count = count
            best_error = error
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return {"fit_status": "failed_ransac", "min_required_points": 2}
    refined = refine_line(fit_points[best_inliers])
    if refined is None:
        return {"fit_status": "failed_refine", "min_required_points": 2}
    direction, point = refined
    residuals_all = line_residuals(points, direction, point)
    inliers_all = residuals_all <= config.ransac_threshold_m
    inlier_residuals = residuals_all[inliers_all]
    projected = (points - point[None, :]) @ direction
    out: dict[str, Any] = {
        "fit_status": "ok",
        "ransac_points": int(len(fit_points)),
        "model": "line",
        "inlier_count": int(inliers_all.sum()),
        "inlier_fraction": float(inliers_all.mean()) if len(inliers_all) else 0.0,
        "line_direction_x": float(direction[0]),
        "line_direction_y": float(direction[1]),
        "line_direction_z": float(direction[2]),
        "model_point_x": float(point[0]),
        "model_point_y": float(point[1]),
        "model_point_z": float(point[2]),
        "line_extent_min_m": float(np.min(projected)),
        "line_extent_max_m": float(np.max(projected)),
        "line_extent_length_m": float(np.max(projected) - np.min(projected)),
    }
    out.update(residual_stats(residuals_all, "all_residual"))
    out.update(residual_stats(inlier_residuals, "inlier_residual"))
    out.update(threshold_curve_stats(residuals_all, config.ransac_thresholds_m))
    return out


def evaluate_chunk(
    *,
    dmap: dict[str, Any],
    valid_mask: np.ndarray,
    transformed_points: np.ndarray,
    kind: str,
    config: FitConfig,
    save_mask_path: Path | None,
    visual_dir: Path | None,
    visual_prefix: str,
    visual_title: str,
    edge_ribbon_radius_px: float | None = None,
    edge_ribbon_mode: str | None = None,
) -> dict[str, Any]:
    shape = dmap["depth_map"].shape
    t_map: np.ndarray | None = None
    if kind == "plane":
        mask = polygon_mask(shape, transformed_points)
        spatial_coverage = grid_coverage(mask, valid_mask, config.plane_grid)
    elif kind == "edge":
        radius_px = edge_ribbon_radius_px
        if radius_px is None:
            radius_px = config.edge_ribbon_px
        if radius_px is None:
            raise ValueError("edge ribbon radius was not resolved for this annotation")
        mask, t_map = edge_ribbon_mask(shape, transformed_points, float(radius_px))
        spatial_coverage = line_bin_coverage(mask, valid_mask, t_map, config.line_bins)
    else:
        raise ValueError(f"unsupported annotation kind: {kind}")

    selected_valid = mask & valid_mask
    ys, xs = np.nonzero(selected_valid)
    invalid_ys, invalid_xs = np.nonzero(mask & ~valid_mask)
    mask_pixels = int(mask.sum())
    valid_pixels = int(selected_valid.sum())
    row: dict[str, Any] = {
        "mask_pixels": mask_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": int(mask_pixels - valid_pixels),
        "coverage_fraction": float(valid_pixels / mask_pixels) if mask_pixels else 0.0,
        "spatial_coverage_fraction": spatial_coverage,
        "annotation_depth_min_x": float(np.min(transformed_points[:, 0])) if len(transformed_points) else None,
        "annotation_depth_max_x": float(np.max(transformed_points[:, 0])) if len(transformed_points) else None,
        "annotation_depth_min_y": float(np.min(transformed_points[:, 1])) if len(transformed_points) else None,
        "annotation_depth_max_y": float(np.max(transformed_points[:, 1])) if len(transformed_points) else None,
    }
    if kind == "edge":
        row["edge_ribbon_radius_depth_px"] = float(radius_px)
        row["edge_ribbon_width_mode"] = edge_ribbon_mode or "depth_pixels_legacy"
    row.update(confidence_stats(dmap, selected_valid))
    if save_mask_path is not None:
        save_mask_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mask": mask, "valid_mask": selected_valid}
        if t_map is not None:
            payload["t_map"] = t_map
        np.savez_compressed(save_mask_path, **payload)
        row["mask_path"] = str(save_mask_path)
    if valid_pixels == 0:
        row.update({"fit_status": "skipped_no_valid_depth", "point_count": 0})
        if visual_dir is not None:
            overlay_path = visual_dir / f"{visual_prefix}_coverage.svg"
            write_fit_overlay_svg(
                overlay_path,
                image_shape=shape,
                transformed_points=transformed_points,
                kind=kind,
                valid_xs=xs,
                valid_ys=ys,
                invalid_xs=invalid_xs,
                invalid_ys=invalid_ys,
                residuals=None,
                config=config,
                title=visual_title,
                edge_ribbon_radius_px=edge_ribbon_radius_px,
            )
            row["visual_overlay_svg"] = str(overlay_path)
        return row
    points = unproject_pixels(dmap, xs, ys)
    row["point_count"] = int(len(points))
    if kind == "plane":
        row.update(ransac_plane(points, config))
    else:
        row.update(ransac_line(points, config))
    if row.get("fit_status") == "ok":
        row["effective_inlier_coverage"] = row["coverage_fraction"] * float(row.get("inlier_fraction", 0.0))
        for threshold in config.ransac_thresholds_m:
            key = f"inlier_fraction_{int(round(threshold * 1000.0))}mm"
            if key in row:
                row[f"effective_inlier_coverage_{int(round(threshold * 1000.0))}mm"] = (
                    row["coverage_fraction"] * float(row[key])
                )
    if visual_dir is not None:
        overlay_path = visual_dir / f"{visual_prefix}_coverage.svg"
        residuals = residuals_from_row(kind, points, row)
        write_fit_overlay_svg(
            overlay_path,
            image_shape=shape,
            transformed_points=transformed_points,
            kind=kind,
            valid_xs=xs,
            valid_ys=ys,
            invalid_xs=invalid_xs,
            invalid_ys=invalid_ys,
            residuals=residuals,
            config=config,
            title=visual_title,
            edge_ribbon_radius_px=edge_ribbon_radius_px,
        )
        row["visual_overlay_svg"] = str(overlay_path)
        if residuals is not None:
            histogram_path = visual_dir / f"{visual_prefix}_residual_histogram.svg"
            write_residual_histogram_svg(histogram_path, residuals, config.ransac_threshold_m, visual_title)
            row["visual_residual_histogram_svg"] = str(histogram_path)
    return row


def rows_for_frame(review_row: dict[str, Any], frame_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    annotations = review_row.get("annotations") or {}
    for edge in annotations.get("controlEdges") or []:
        for chunk in edge.get("chunks") or []:
            if str(chunk.get("frameId", "")).lower() == frame_id.lower():
                rows.append({"kind": "edge", "object_id": edge.get("id"), "chunk": chunk})
    for plane in annotations.get("controlPlanes") or []:
        for chunk in plane.get("chunks") or []:
            if str(chunk.get("frameId", "")).lower() == frame_id.lower():
                rows.append({"kind": "plane", "object_id": plane.get("id"), "chunk": chunk})
    return rows


def ensure_annotation_row_schema(
    row: dict[str, Any],
    *,
    thresholds_m: tuple[float, ...],
    identity: dict[str, Any],
) -> dict[str, Any]:
    row.update(identity)
    defaults: dict[str, Any] = {
        "annotation_present": True,
        "failure_type": None,
        "failure_reason": None,
        "coverage_fraction": None,
        "spatial_coverage_fraction": None,
        "valid_pixels": None,
        "mask_pixels": None,
        "inlier_fraction": None,
        "effective_inlier_coverage": None,
        "inlier_threshold_auc": None,
        "all_residual_rmse_m": None,
        "all_residual_median_m": None,
        "all_residual_p95_m": None,
        "point_count": 0,
    }
    for threshold in thresholds_m:
        millimeters = int(round(threshold * 1000.0))
        defaults[f"inlier_fraction_{millimeters}mm"] = None
        defaults[f"effective_inlier_coverage_{millimeters}mm"] = None
    for key, value in defaults.items():
        row.setdefault(key, value)
    return row


def missing_annotations_row(
    *,
    scan_id: str,
    frame_id: str,
    image_name: str,
    thresholds_m: tuple[float, ...],
    identity: dict[str, Any],
) -> dict[str, Any]:
    return ensure_annotation_row_schema(
        {
            "index": 0,
            "scan_id": scan_id,
            "frame_id": frame_id,
            "image_name": image_name,
            "annotation_kind": None,
            "object_id": None,
            "chunk_id": None,
            "raw_point_count": 0,
            "annotation_present": False,
            "fit_status": "missing_annotations_for_frame",
            "failure_type": "missing_annotation",
            "failure_reason": "the matched DMAP frame has no control-edge or control-plane chunks",
        },
        thresholds_m=thresholds_m,
        identity=identity,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = [
        "run_id",
        "repeat_id",
        "stage",
        "scan_id",
        "frame_id",
        "annotation_kind",
        "object_id",
        "chunk_id",
        "annotation_present",
        "fit_status",
        "failure_type",
        "failure_reason",
        "coverage_fraction",
        "spatial_coverage_fraction",
        "valid_pixels",
        "mask_pixels",
        "inlier_fraction",
        "effective_inlier_coverage",
        "inlier_threshold_auc",
        "all_residual_rmse_m",
        "all_residual_median_m",
        "all_residual_p95_m",
        "inlier_residual_rmse_m",
        "point_count",
    ]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen and not isinstance(row[key], (dict, list)):
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = [
            "annotation_kind",
            "object_id",
            "chunk_id",
            "fit_status",
            "coverage_fraction",
            "spatial_coverage_fraction",
            "valid_pixels",
            "mask_pixels",
            "inlier_fraction",
            "all_residual_rmse_m",
            "point_count",
        ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{digits}f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    rows = result["annotations"]
    present_rows = [row for row in rows if row.get("annotation_present", True)]
    ok_rows = [row for row in present_rows if row.get("fit_status") == "ok"]

    def rel_link(item_path: str | None) -> str | None:
        if not item_path:
            return None
        try:
            return str(Path(item_path).resolve().relative_to(path.parent.resolve()))
        except ValueError:
            return item_path

    lines = [
        "# DMAP Annotation Fit Report",
        "",
        "## 1. Executive summary",
        "",
        f"- Scan ID: `{result['scan_id']}`",
        f"- DMAP: `{result['dmap']['path']}`",
        f"- Matched frame: `{result.get('frame_id')}` / `{result['dmap']['image_name']}`",
        f"- Evaluation identity: run `{result.get('run_id')}`, repeat `{result.get('repeat_id')}`, stage `{result.get('stage')}`",
        f"- Evaluated chunks: {len(present_rows)} present, {len(ok_rows)} fitted successfully",
        f"- RANSAC threshold: {result['config']['ransac_threshold_m']} m",
        "",
        "## 2. Metrics description",
        "",
        "- `coverage_fraction`: valid depth pixels inside the annotation mask divided by total mask pixels.",
        "- `spatial_coverage_fraction`: occupied grid/bin cells divided by annotation grid/bin cells.",
        "- Residual metrics are point-to-plane or point-to-line distances in meters.",
        "- `inlier_fraction` uses the configured RANSAC threshold.",
        "- `effective_inlier_coverage`: fraction of the full annotation mask that has valid depth and is a RANSAC inlier.",
        "- `inlier_threshold_auc`: normalized area under the fixed-model inlier curve over the configured thresholds.",
        "- Edge rows record the configured width mode and resolved half-width in the current DMAP pixel grid; the default is 5 annotation/source pixels.",
        "",
        "## 3. Per-annotation metrics",
        "",
        "| Type | Object | Chunk | Status | Coverage | Spatial | Inliers | Effective | AUC | RMSE m | Median m | P95 m | Points |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("annotation_kind", "")),
                    f"`{str(row.get('object_id', ''))[:8]}`",
                    f"`{str(row.get('chunk_id', ''))[:8]}`",
                    str(row.get("fit_status", "")),
                    format_float(row.get("coverage_fraction")),
                    format_float(row.get("spatial_coverage_fraction")),
                    format_float(row.get("inlier_fraction")),
                    format_float(row.get("effective_inlier_coverage")),
                    format_float(row.get("inlier_threshold_auc")),
                    format_float(row.get("all_residual_rmse_m")),
                    format_float(row.get("all_residual_median_m")),
                    format_float(row.get("all_residual_p95_m")),
                    str(row.get("point_count", 0)),
                ]
            )
            + " |"
        )
    if not rows:
        lines.append("| n/a | n/a | n/a | no_matching_annotations | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "## 4. Visualizations",
            "",
        ]
    )
    overall_visual = rel_link(result["outputs"].get("overall_visual_svg"))
    if overall_visual:
        lines.extend(
            [
                "Overall coverage, inlier fraction, and P95 residual summary:",
                "",
                f"![Overall fit summary]({overall_visual})",
                "",
            ]
        )
    if present_rows:
        for row in rows:
            if not row.get("annotation_present", True):
                continue
            overlay = rel_link(row.get("visual_overlay_svg"))
            histogram = rel_link(row.get("visual_residual_histogram_svg"))
            lines.append(
                f"### {int(row.get('index', 0)):03d}. {row.get('annotation_kind')} "
                f"`{str(row.get('chunk_id', ''))[:8]}`"
            )
            lines.append("")
            if overlay:
                lines.append(f"![Coverage and residual overlay]({overlay})")
                lines.append("")
            if histogram:
                lines.append(f"![Residual histogram]({histogram})")
                lines.append("")
    else:
        lines.append("No matching annotations were found for this DMAP frame.")
    lines.extend(
        [
            "",
            "## 5. Failure analysis",
            "",
        ]
    )
    skipped = [row for row in rows if row.get("fit_status") != "ok"]
    if skipped:
        for row in skipped:
            reason = f": {row.get('failure_reason')}" if row.get("failure_reason") else ""
            lines.append(
                f"- `{row.get('annotation_kind')}` chunk `{row.get('chunk_id')}`: {row.get('fit_status')}{reason} "
                f"({row.get('valid_pixels') or 0} valid / {row.get('mask_pixels') or 0} mask pixels)"
            )
    else:
        lines.append("No annotation chunks were skipped.")
    lines.extend(
        [
            "",
            "## 6. Reproducibility",
            "",
            "Command:",
            "",
            "```bash",
            result["command"],
            "```",
            "",
            "Outputs:",
            "",
            f"- `{result['outputs']['metrics_json']}`",
            f"- `{result['outputs']['annotation_metrics_csv']}`",
            f"- `{result['outputs']['report_md']}`",
            f"- `{result['outputs']['visualizations_dir']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def command_string(args: argparse.Namespace) -> str:
    return shlex.join([sys.executable, Path(__file__).as_posix(), *sys.argv[1:]])


def load_scan_context(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    annotation_sidecar = getattr(args, "annotation_sidecar", None)
    if annotation_sidecar is not None:
        sidecar = load_annotation_sidecar(annotation_sidecar, getattr(args, "scan_id", None))
        return sidecar["scene"], sidecar["review"], sidecar["pipeline"]
    dataset_root = args.dataset_root
    cache_root = args.cache_root
    db = annotation_db_path(dataset_root)
    scan_rows = load_jsonl(db / "scans.jsonl")
    review_rows = load_jsonl(db / "scan_reviewers.jsonl")
    pipeline_rows = load_jsonl(db / "scan_pipelines.jsonl")

    local_annotations = cache_root / args.scan_id / "annotations"
    scan_rows += load_jsonl(local_annotations / "scans.jsonl")
    review_rows += load_jsonl(local_annotations / "scan_reviewers.jsonl")
    pipeline_rows += load_jsonl(local_annotations / "scan_pipelines.jsonl")

    scan_row = find_row(scan_rows, "id", args.scan_id)
    review_row = find_row(review_rows, "scan_id", args.scan_id)
    pipeline_row = find_row(pipeline_rows, "scan_id", args.scan_id)
    missing = [
        name
        for name, row in [("scan", scan_row), ("scan_reviewers", review_row), ("scan_pipelines", pipeline_row)]
        if row is None
    ]
    if missing:
        raise ValueError(f"missing dataset rows for scan {args.scan_id}: {', '.join(missing)}")
    return scan_row, review_row, pipeline_row


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dmap = load_dmap(args.dmap)
    scan_row, review_row, pipeline_row = load_scan_context(args)
    if getattr(args, "annotation_sidecar", None) is not None:
        sidecar = load_annotation_sidecar(args.annotation_sidecar, args.scan_id)
        uuid_to_image_name = sidecar["image_mapping"]
        mapping_source = f"annotation_sidecar:{args.annotation_sidecar}"
    else:
        uuid_to_image_name, mapping_source = load_image_mapping(
            scan_id=args.scan_id,
            dataset_root=args.dataset_root,
            cache_root=args.cache_root,
            explicit_mapping=args.image_mapping,
        )
    image_to_frame = {normalize_image_name(name): frame_id for frame_id, name in uuid_to_image_name.items()}
    dmap_image_name = normalize_image_name(str(dmap["image_name"]))
    frame_id = image_to_frame.get(dmap_image_name)
    if frame_id is None:
        raise ValueError(f"DMAP image {dmap['image_name']!r} not found in image_id_mapping.json from {mapping_source}")

    edge_ribbon_px = getattr(args, "edge_ribbon_px", None)
    edge_ribbon_source_px = getattr(args, "edge_ribbon_source_px", None)
    edge_ribbon_angle_mrad = getattr(args, "edge_ribbon_angle_mrad", None)
    if edge_ribbon_px is None and edge_ribbon_source_px is None and edge_ribbon_angle_mrad is None:
        edge_ribbon_source_px = 5.0
    config = FitConfig(
        ransac_threshold_m=float(args.ransac_threshold_m),
        ransac_thresholds_m=tuple(sorted({float(value) / 1000.0 for value in args.ransac_thresholds_mm})),
        edge_ribbon_px=float(edge_ribbon_px) if edge_ribbon_px is not None else None,
        max_ransac_points=int(args.max_ransac_points),
        ransac_trials=int(args.ransac_trials),
        seed=int(args.seed),
        min_confidence=args.min_confidence,
        plane_grid=int(args.plane_grid),
        line_bins=int(args.line_bins),
        max_visual_points=int(args.max_visual_points),
        edge_ribbon_source_px=(
            float(edge_ribbon_source_px) if edge_ribbon_source_px is not None else None
        ),
        edge_ribbon_angle_mrad=(
            float(edge_ribbon_angle_mrad) if edge_ribbon_angle_mrad is not None else None
        ),
    )
    validate_fit_config(config)
    identity = {
        "run_id": getattr(args, "run_id", None),
        "repeat_id": getattr(args, "repeat_id", None),
        "stage": getattr(args, "stage", None),
    }
    valid = valid_depth_mask(dmap, config.min_confidence)
    annotation_chunks = rows_for_frame(review_row, frame_id)
    annotations: list[dict[str, Any]] = []
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(annotation_chunks, 1):
        kind = item["kind"]
        chunk = item["chunk"]
        chunk_id = str(chunk.get("id") or f"chunk_{index}")
        mask_path = None
        if args.save_masks:
            mask_path = output_dir / "masks" / f"{index:03d}_{kind}_{chunk_id}.npz"
        visual_prefix = f"{index:03d}_{kind}_{chunk_id[:8]}"
        visual_title = f"{index:03d} {kind} {chunk_id[:8]} / {dmap_image_name}"
        raw_points = chunk_points(chunk, kind)
        annotation_seed = stable_annotation_seed(
            config.seed,
            scan_id=args.scan_id,
            frame_id=frame_id,
            kind=kind,
            object_id=item["object_id"],
            chunk_id=chunk.get("id"),
        )
        chunk_config = replace(config, seed=annotation_seed)
        row_metadata = {
            "index": index,
            "scan_id": args.scan_id,
            "frame_id": frame_id,
            "image_name": dmap_image_name,
            "annotation_kind": kind,
            "object_id": item["object_id"],
            "chunk_id": chunk.get("id"),
            "raw_point_count": int(len(raw_points)),
            "ransac_seed": annotation_seed,
        }
        try:
            transformed = annotation_points_to_depth_pixels(
                raw_points, dmap, pipeline_row, args.annotation_space
            )
            edge_radius = None
            edge_mode = None
            if kind == "edge":
                edge_radius, edge_mode = edge_ribbon_radius_depth_px(
                    raw_segment=raw_points,
                    dmap=dmap,
                    pipeline_row=pipeline_row,
                    annotation_space=args.annotation_space,
                    config=chunk_config,
                )
            row = evaluate_chunk(
                dmap=dmap,
                valid_mask=valid,
                transformed_points=transformed,
                kind=kind,
                config=chunk_config,
                save_mask_path=mask_path,
                visual_dir=visual_dir,
                visual_prefix=visual_prefix,
                visual_title=visual_title,
                edge_ribbon_radius_px=edge_radius,
                edge_ribbon_mode=edge_mode,
            )
        except Exception as exc:
            row = {
                "fit_status": "failed_annotation_evaluation",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
            }
        row.update(row_metadata)
        ensure_annotation_row_schema(
            row,
            thresholds_m=config.ransac_thresholds_m,
            identity=identity,
        )
        annotations.append(row)

    if not annotations:
        annotations.append(
            missing_annotations_row(
                scan_id=args.scan_id,
                frame_id=frame_id,
                image_name=dmap_image_name,
                thresholds_m=config.ransac_thresholds_m,
                identity=identity,
            )
        )

    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "annotation_metrics.csv"
    report_path = output_dir / "report.md"
    overall_visual_path = visual_dir / "overall_fit_summary.svg"
    plotted_annotations = [row for row in annotations if row.get("annotation_present", True)]
    if plotted_annotations:
        write_overall_visual_svg(overall_visual_path, plotted_annotations)
    result: dict[str, Any] = {
        "schema_version": ANNOTATION_EVALUATION_SCHEMA_VERSION,
        "scan_id": args.scan_id,
        "frame_id": frame_id,
        **identity,
        "evaluation_identity": identity,
        "mapping_source": mapping_source,
        "dataset_root": str(args.dataset_root),
        "cache_root": str(args.cache_root),
        "dmap": {
            "path": str(args.dmap),
            "image_name": dmap["image_name"],
            "reference_view_id": dmap["reference_view_id"],
            "image_width": dmap["image_width"],
            "image_height": dmap["image_height"],
            "depth_width": dmap["depth_width"],
            "depth_height": dmap["depth_height"],
            "valid_depth_fraction": float(valid.mean()) if valid.size else 0.0,
        },
        "config": {
            "ransac_threshold_m": config.ransac_threshold_m,
            "ransac_thresholds_m": list(config.ransac_thresholds_m),
            "edge_ribbon_px": config.edge_ribbon_px,
            "max_ransac_points": config.max_ransac_points,
            "ransac_trials": config.ransac_trials,
            "seed": config.seed,
            "ransac_sampling": "numpy_pcg64_seeded_per_annotation_sha256_v1",
            "min_confidence": config.min_confidence,
            "plane_grid": config.plane_grid,
            "line_bins": config.line_bins,
            "max_visual_points": config.max_visual_points,
            "annotation_space": args.annotation_space,
            "edge_ribbon_source_px": config.edge_ribbon_source_px,
            "edge_ribbon_angle_mrad": config.edge_ribbon_angle_mrad,
        },
        "annotations": annotations,
        "outputs": {
            "metrics_json": str(metrics_path),
            "annotation_metrics_csv": str(csv_path),
            "report_md": str(report_path),
            "visualizations_dir": str(visual_dir),
            "overall_visual_svg": str(overall_visual_path) if plotted_annotations else None,
        },
        "command": command_string(args),
    }
    write_csv(csv_path, annotations)
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_report(report_path, result)
    return result


def build_corpus_annotation_manifest(dataset_root: Path, cache_root: Path) -> dict[str, Any]:
    """Audit annotation coverage using metadata and filesystem existence checks only."""
    db = annotation_db_path(dataset_root)
    scans = {str(row.get("id")): row for row in load_jsonl(db / "scans.jsonl") if row.get("id")}
    pipelines = {
        str(row.get("scan_id")): row
        for row in load_jsonl(db / "scan_pipelines.jsonl")
        if row.get("scan_id")
    }
    reviewers = load_jsonl(db / "scan_reviewers.jsonl")
    rows: list[dict[str, Any]] = []
    edge_chunks_total = 0
    plane_chunks_total = 0
    malformed_total = 0
    annotated_scan_ids: set[str] = set()

    for review in reviewers:
        scan_id = str(review.get("scan_id") or "")
        annotations = review.get("annotations") or {}
        grouped: dict[str, dict[str, Any]] = {}
        for kind, collection_name in (("edge", "controlEdges"), ("plane", "controlPlanes")):
            for annotation_object in annotations.get(collection_name) or []:
                for chunk in annotation_object.get("chunks") or []:
                    frame_id = str(chunk.get("frameId") or "")
                    group = grouped.setdefault(
                        frame_id,
                        {
                            "edge_chunk_count": 0,
                            "plane_chunk_count": 0,
                            "malformed_chunk_count": 0,
                            "object_ids": set(),
                            "chunk_ids": [],
                        },
                    )
                    group[f"{kind}_chunk_count"] += 1
                    group["object_ids"].add(str(annotation_object.get("id") or ""))
                    group["chunk_ids"].append(str(chunk.get("id") or ""))
                    minimum = 2 if kind == "edge" else 3
                    try:
                        points = chunk_points(chunk, kind)
                        malformed = len(points) < minimum or not np.all(np.isfinite(points))
                    except (KeyError, TypeError, ValueError):
                        malformed = True
                    if malformed:
                        group["malformed_chunk_count"] += 1
                        malformed_total += 1
                    if kind == "edge":
                        edge_chunks_total += 1
                    else:
                        plane_chunks_total += 1
        if not grouped:
            continue
        annotated_scan_ids.add(scan_id)
        scan = scans.get(scan_id)
        pipeline = pipelines.get(scan_id)
        frame_metadata = {
            str(frame.get("id", "")).lower(): frame for frame in review.get("frames") or []
        }
        cache_dir = cache_root / scan_id
        cached_mapping = (cache_dir / "image_id_mapping.json").is_file()
        dataset_mapping = annotation_image_mapping_path(dataset_root, scan_id).is_file()
        cached_dmap = cache_dir.exists() and any(cache_dir.rglob("depth*.dmap"))
        for frame_id, counts in sorted(grouped.items()):
            frame = frame_metadata.get(frame_id.lower())
            resolution = (frame or {}).get("imageResolution") or {}
            has_camera_metadata = bool(
                pipeline
                and pipeline.get("camera_distorted")
                and (pipeline.get("camera_final") or pipeline.get("camera_undistorted"))
            )
            metadata_ready = bool(scan and pipeline and has_camera_metadata and frame_id and frame)
            mapping_available = dataset_mapping or cached_mapping
            annotation_metadata_ready = metadata_ready and mapping_available
            evaluation_ready = annotation_metadata_ready and cached_dmap
            status = "ready_cached" if evaluation_ready else "missing_dependencies"
            if annotation_metadata_ready and not cached_dmap:
                status = "requires_densification_cache"
            if not frame_id:
                status = "missing_frame_id"
            elif counts["malformed_chunk_count"]:
                status = "contains_malformed_chunks"
            rows.append(
                {
                    "scan_id": scan_id,
                    "frame_id": frame_id or None,
                    "edge_chunk_count": counts["edge_chunk_count"],
                    "plane_chunk_count": counts["plane_chunk_count"],
                    "annotation_chunk_count": counts["edge_chunk_count"] + counts["plane_chunk_count"],
                    "annotation_object_count": len(counts["object_ids"]),
                    "malformed_chunk_count": counts["malformed_chunk_count"],
                    "source_width": resolution.get("width"),
                    "source_height": resolution.get("height"),
                    "has_scan_metadata": scan is not None,
                    "has_pipeline_metadata": pipeline is not None,
                    "has_camera_metadata": has_camera_metadata,
                    "has_frame_metadata": frame is not None,
                    "cache_dir_exists": cache_dir.exists(),
                    "cached_image_mapping_exists": cached_mapping,
                    "dataset_image_mapping_exists": dataset_mapping,
                    "cached_depth_map_exists": cached_dmap,
                    "metadata_ready": metadata_ready,
                    "mapping_available": mapping_available,
                    "annotation_metadata_ready": annotation_metadata_ready,
                    "evaluation_ready": evaluation_ready and not counts["malformed_chunk_count"],
                    "audit_status": status,
                    "chunk_ids": counts["chunk_ids"],
                }
            )

    rows.sort(key=lambda row: (str(row["scan_id"]), str(row["frame_id"] or "")))
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["audit_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "cache_root": str(cache_root),
        "summary": {
            "scan_metadata_rows": len(scans),
            "reviewer_rows": len(reviewers),
            "pipeline_rows": len(pipelines),
            "annotated_scans": len(annotated_scan_ids),
            "annotated_frames": len(rows),
            "edge_chunks": edge_chunks_total,
            "plane_chunks": plane_chunks_total,
            "malformed_chunks": malformed_total,
            "evaluation_ready_frames": sum(bool(row["evaluation_ready"]) for row in rows),
            "audit_status_counts": status_counts,
        },
        "frames": rows,
    }


def run_corpus_audit(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = build_corpus_annotation_manifest(args.dataset_root, args.cache_root)
    result["command"] = command_string(args)
    json_path = output_dir / "corpus_annotation_manifest.json"
    csv_path = output_dir / "corpus_annotation_frames.csv"
    result["outputs"] = {
        "manifest_json": str(json_path),
        "frames_csv": str(csv_path),
    }
    write_csv(csv_path, result["frames"])
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def self_test() -> None:
    rng = np.random.default_rng(123)
    xs = rng.uniform(-1.0, 1.0, 600)
    ys = rng.uniform(-1.0, 1.0, 600)
    zs = 0.4 * xs - 0.2 * ys + 1.3 + rng.normal(0.0, 0.005, 600)
    plane_points = np.column_stack([xs, ys, zs])
    plane_points[:30] += rng.normal(0.0, 0.25, (30, 3))
    config = FitConfig(0.02, (0.005, 0.01, 0.02, 0.05), 5.0, 50000, 1000, 11, None, 32, 100, 6000)
    plane = ransac_plane(plane_points, config)
    assert plane["fit_status"] == "ok", plane
    assert plane["inlier_fraction"] > 0.85, plane
    assert plane["all_residual_median_m"] < 0.02, plane
    assert 0.0 < plane["inlier_threshold_auc"] <= 1.0, plane
    assert plane["inlier_fraction_5mm"] < plane["inlier_fraction_50mm"], plane

    t = rng.uniform(-1.0, 1.0, 600)
    direction = np.array([0.2, 0.8, 0.1])
    direction /= np.linalg.norm(direction)
    base = np.array([1.0, 2.0, 3.0])
    noise = rng.normal(0.0, 0.004, (600, 3))
    line_points = base[None, :] + t[:, None] * direction[None, :] + noise
    line_points[:25] += rng.normal(0.0, 0.25, (25, 3))
    line = ransac_line(line_points, config)
    assert line["fit_status"] == "ok", line
    assert line["inlier_fraction"] > 0.85, line
    assert line["all_residual_median_m"] < 0.02, line

    poly = np.array([[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]], dtype=np.float64)
    mask = polygon_mask((12, 12), poly)
    assert int(mask.sum()) == 36, int(mask.sum())
    segment = np.array([[2.0, 2.0], [9.0, 2.0]], dtype=np.float64)
    ribbon, t_map = edge_ribbon_mask((12, 12), segment, 1.5)
    assert int(ribbon.sum()) > 0
    assert line_bin_coverage(ribbon, ribbon, t_map, 10) > 0.9
    print("self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-corpus",
        action="store_true",
        help="Write a metadata-only multi-scene annotation manifest; no DMAP is required",
    )
    parser.add_argument("--dmap", type=Path, help="Path to one OpenMVS .dmap file")
    parser.add_argument("--scan-id", help="Scene ID used to bind annotations to the DMAP")
    parser.add_argument(
        "--annotation-sidecar",
        type=Path,
        help="Self-contained openmvs.dmap.annotation_sidecar JSON file",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional legacy annotation database root (use --annotation-sidecar for new integrations)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Optional legacy scene cache root (use --annotation-sidecar for new integrations)",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for report.md, metrics.json, and CSV output")
    parser.add_argument("--image-mapping", type=Path, help="Optional explicit image_id_mapping.json")
    parser.add_argument("--annotation-space", choices=("distorted", "final"), default="distorted")
    parser.add_argument("--ransac-threshold-m", type=float, default=0.02)
    parser.add_argument("--ransac-thresholds-mm", type=float, nargs="+", default=(5.0, 10.0, 20.0, 50.0))
    ribbon = parser.add_mutually_exclusive_group()
    ribbon.add_argument(
        "--edge-ribbon-px",
        type=float,
        help="Legacy explicit edge half-width in DMAP pixels",
    )
    ribbon.add_argument(
        "--edge-ribbon-source-px",
        type=float,
        help="Edge half-width in annotation/source-image pixels (default: 5)",
    )
    ribbon.add_argument(
        "--edge-ribbon-angle-mrad",
        type=float,
        help="Resolution-independent edge angular half-width in milliradians",
    )
    parser.add_argument("--max-ransac-points", type=int, default=50000)
    parser.add_argument("--ransac-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-confidence", type=float)
    parser.add_argument("--plane-grid", type=int, default=32)
    parser.add_argument("--line-bins", type=int, default=100)
    parser.add_argument("--max-visual-points", type=int, default=6000)
    parser.add_argument("--run-id", help="Experiment run identifier recorded on every output row")
    parser.add_argument("--repeat-id", help="Repeat identifier recorded on every output row")
    parser.add_argument("--stage", help="Pipeline stage identifier recorded on every output row")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output_dir is None:
        parser.error("missing required argument: --output-dir")
    if args.audit_corpus:
        if args.dataset_root is None or args.cache_root is None:
            parser.error("--audit-corpus requires --dataset-root and --cache-root")
        result = run_corpus_audit(args)
        print(json.dumps({"manifest": result["outputs"]["manifest_json"], **result["summary"]}, indent=2))
        return 0
    if args.edge_ribbon_px is None and args.edge_ribbon_source_px is None and args.edge_ribbon_angle_mrad is None:
        args.edge_ribbon_source_px = 5.0
    if args.annotation_sidecar is not None and args.scan_id in (None, ""):
        args.scan_id = load_annotation_sidecar(args.annotation_sidecar)["scene"]["id"]
    missing = [name for name in ("dmap", "scan_id") if getattr(args, name) in (None, "")]
    if missing:
        parser.error("missing required arguments unless --self-test is used: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    if args.annotation_sidecar is None and (
        args.dataset_root is None or args.cache_root is None
    ):
        parser.error(
            "provide --annotation-sidecar, or both --dataset-root and --cache-root"
        )
    result = run_report(args)
    print(json.dumps({"report": result["outputs"]["report_md"], "chunks": len(result["annotations"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
