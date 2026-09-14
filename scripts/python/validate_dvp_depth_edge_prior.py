#!/usr/bin/env python3
"""Validate a versioned DVP depth-edge prior bundle and its artifact closure."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "openmvs_dvp_depth_edge_prior"
SCHEMA_VERSION = 2
STAGES = (
    "roberts_regions",
    "dav2_planarized",
    "eroded",
    "dilated",
    "pixel_reassigned",
)


@dataclasses.dataclass
class Args:
    bundle_dir: Path
    source_images_dir: Path | None = None
    output_json: Path | None = None


def compute_openmvs_processing_geometry(
    width: int,
    height: int,
    resolution_level: int,
    min_resolution: int,
    max_resolution: int,
) -> dict[str, int | float]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if resolution_level < 0 or resolution_level > 30:
        raise ValueError("resolution level must be in [0, 30]")
    if min_resolution < 0 or max_resolution < 0:
        raise ValueError("resolution bounds must be non-negative")
    image_size = max(width, height)
    effective_level = resolution_level
    if effective_level == 0:
        prepared_max_resolution = min(image_size, max_resolution)
    else:
        size = image_size >> effective_level
        if size < min_resolution:
            effective_level = 0
            while (image_size >> (effective_level + 1)) >= min_resolution:
                effective_level += 1
            size = image_size >> effective_level
        prepared_max_resolution = min(size, max_resolution)
    scale = (
        1.0
        if prepared_max_resolution == 0 or image_size <= prepared_max_resolution
        else prepared_max_resolution / image_size
    )
    return {
        "width": int(np.rint(width * scale)),
        "height": int(np.rint(height * scale)),
        "effective_resolution_level": effective_level,
        "prepared_max_resolution": prepared_max_resolution,
        "scale": scale,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args: Args) -> dict[str, Any]:
    errors: list[str] = []
    checks = 0

    def require(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    bundle_path = args.bundle_dir / "bundle.json"
    require(bundle_path.is_file(), f"missing bundle manifest: {bundle_path}")
    if not bundle_path.is_file():
        return {"valid": False, "checks": checks, "errors": errors, "frames": []}
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    require(bundle.get("schema") == SCHEMA, "bundle schema mismatch")
    require(bundle.get("schema_version") == SCHEMA_VERSION, "bundle schema version mismatch")
    require(bundle.get("complete") is True, "bundle is not complete")
    bundle_preparation = bundle.get("image_preparation", {})
    require(isinstance(bundle_preparation, dict), "bundle image preparation must be an object")
    frames = bundle.get("frames")
    require(isinstance(frames, list), "bundle frames must be an array")
    if not isinstance(frames, list):
        frames = []
    require(bundle.get("frame_count") == len(frames), "bundle frame count mismatch")
    frame_results: list[dict[str, Any]] = []
    seen_depth_ids: set[int] = set()
    for frame in frames:
        frame_errors_before = len(errors)
        depth_id = frame.get("depth_id")
        require(isinstance(depth_id, int) and depth_id >= 0, "invalid depth ID")
        if not isinstance(depth_id, int):
            continue
        require(depth_id not in seen_depth_ids, f"duplicate depth ID {depth_id}")
        seen_depth_ids.add(depth_id)
        relative_manifest = frame.get("manifest")
        require(isinstance(relative_manifest, str) and relative_manifest != "", f"depth{depth_id:04d}: missing manifest path")
        if not isinstance(relative_manifest, str):
            continue
        manifest_path = args.bundle_dir / relative_manifest
        require(manifest_path.is_file(), f"depth{depth_id:04d}: missing manifest")
        if not manifest_path.is_file():
            continue
        require(
            sha256_file(manifest_path) == frame.get("manifest_sha256"),
            f"depth{depth_id:04d}: manifest hash mismatch",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("schema") == SCHEMA, f"depth{depth_id:04d}: schema mismatch")
        require(manifest.get("schema_version") == SCHEMA_VERSION, f"depth{depth_id:04d}: schema version mismatch")
        require(manifest.get("complete") is True, f"depth{depth_id:04d}: incomplete manifest")
        require(manifest.get("depth_id") == depth_id, f"depth{depth_id:04d}: identity mismatch")
        require(manifest.get("direct_depth_use") == "forbidden_topology_only", f"depth{depth_id:04d}: direct-depth contract missing")
        source = manifest.get("source_image", {})
        width, height = source.get("width"), source.get("height")
        require(isinstance(width, int) and width > 0, f"depth{depth_id:04d}: invalid source width")
        require(isinstance(height, int) and height > 0, f"depth{depth_id:04d}: invalid source height")
        require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64, f"depth{depth_id:04d}: source hash missing")
        if args.source_images_dir is not None and isinstance(source.get("name"), str):
            source_path = args.source_images_dir / source["name"]
            require(source_path.is_file(), f"depth{depth_id:04d}: source image unavailable")
            if source_path.is_file():
                require(sha256_file(source_path) == source["sha256"], f"depth{depth_id:04d}: source image hash mismatch")
                source_pixels = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
                require(source_pixels is not None, f"depth{depth_id:04d}: source image unreadable")
                if source_pixels is not None:
                    require(
                        source_pixels.shape[:2] == (height, width),
                        f"depth{depth_id:04d}: raw source dimensions mismatch",
                    )
        processing = manifest.get("processing_image", {})
        processing_width = processing.get("width")
        processing_height = processing.get("height")
        require(
            isinstance(processing_width, int) and processing_width > 0,
            f"depth{depth_id:04d}: invalid processing width",
        )
        require(
            isinstance(processing_height, int) and processing_height > 0,
            f"depth{depth_id:04d}: invalid processing height",
        )
        require(
            processing.get("resize_interpolation") == "cv::INTER_AREA",
            f"depth{depth_id:04d}: processing interpolation mismatch",
        )
        require(
            processing.get("geometry_contract")
            == "OpenMVS_Image_RecomputeMaxResolution_then_ReloadImage",
            f"depth{depth_id:04d}: processing geometry contract mismatch",
        )
        preparation_keys = ("resolution_level", "min_resolution", "max_resolution")
        for key in preparation_keys:
            require(
                isinstance(bundle_preparation.get(key), int),
                f"bundle image preparation {key} is invalid",
            )
        if all(isinstance(bundle_preparation.get(key), int) for key in preparation_keys):
            requested = {
                "requested_resolution_level": bundle_preparation["resolution_level"],
                "min_resolution": bundle_preparation["min_resolution"],
                "max_resolution": bundle_preparation["max_resolution"],
            }
            for key, value in requested.items():
                require(
                    processing.get(key) == value,
                    f"depth{depth_id:04d}: processing {key} does not match bundle",
                )
            if isinstance(width, int) and isinstance(height, int):
                try:
                    expected = compute_openmvs_processing_geometry(
                        width,
                        height,
                        bundle_preparation["resolution_level"],
                        bundle_preparation["min_resolution"],
                        bundle_preparation["max_resolution"],
                    )
                except ValueError as error:
                    require(False, f"depth{depth_id:04d}: invalid processing configuration: {error}")
                else:
                    for key in ("width", "height", "effective_resolution_level", "prepared_max_resolution"):
                        require(
                            processing.get(key) == expected[key],
                            f"depth{depth_id:04d}: processing {key} does not match OpenMVS",
                        )
                    scale = processing.get("scale")
                    require(
                        isinstance(scale, (int, float))
                        and abs(float(scale) - float(expected["scale"])) <= 1e-12,
                        f"depth{depth_id:04d}: processing scale does not match OpenMVS",
                    )
        model = manifest.get("model", {})
        require(isinstance(model.get("revision"), str) and len(model["revision"]) == 40, f"depth{depth_id:04d}: model revision not pinned")
        require(isinstance(model.get("checkpoint_sha256"), str) and len(model["checkpoint_sha256"]) == 64, f"depth{depth_id:04d}: checkpoint hash not pinned")
        parameters = manifest.get("parameters", {})
        eta = parameters.get("eta")
        require(isinstance(eta, int) and eta >= 0, f"depth{depth_id:04d}: invalid eta")
        component_filtering = manifest.get("region_component_filtering", {})
        require(
            component_filtering.get("policy")
            == "components_with_area_lte_eta_become_boundary",
            f"depth{depth_id:04d}: component filtering policy mismatch",
        )
        require(
            component_filtering.get("minimum_region_size_exclusive") == eta,
            f"depth{depth_id:04d}: component filtering threshold mismatch",
        )
        raw_region_count = component_filtering.get("raw_region_count")
        retained_region_count = component_filtering.get("retained_region_count")
        discarded_region_count = component_filtering.get("discarded_region_count")
        require(
            isinstance(raw_region_count, int) and raw_region_count >= 0,
            f"depth{depth_id:04d}: invalid raw region count",
        )
        require(
            isinstance(retained_region_count, int)
            and 0 <= retained_region_count <= np.iinfo(np.uint16).max - 1,
            f"depth{depth_id:04d}: invalid retained region count",
        )
        require(
            isinstance(discarded_region_count, int) and discarded_region_count >= 0,
            f"depth{depth_id:04d}: invalid discarded region count",
        )
        if all(
            isinstance(value, int)
            for value in (raw_region_count, retained_region_count, discarded_region_count)
        ):
            require(
                retained_region_count + discarded_region_count == raw_region_count,
                f"depth{depth_id:04d}: component filtering counts do not close",
            )
        stages = manifest.get("stages", {})
        require(set(stages) == set(STAGES), f"depth{depth_id:04d}: stage set mismatch")
        for stage_name in STAGES:
            stage = stages.get(stage_name, {})
            relative_map = stage.get("label_map")
            require(isinstance(relative_map, str) and relative_map != "", f"depth{depth_id:04d}/{stage_name}: map path missing")
            if not isinstance(relative_map, str):
                continue
            map_path = manifest_path.parent / relative_map
            require(map_path.is_file(), f"depth{depth_id:04d}/{stage_name}: map missing")
            if not map_path.is_file():
                continue
            require(sha256_file(map_path) == stage.get("sha256"), f"depth{depth_id:04d}/{stage_name}: map hash mismatch")
            labels = cv2.imread(str(map_path), cv2.IMREAD_UNCHANGED)
            require(labels is not None, f"depth{depth_id:04d}/{stage_name}: map unreadable")
            if labels is None:
                continue
            require(labels.dtype == np.uint16, f"depth{depth_id:04d}/{stage_name}: map is not uint16")
            require(
                labels.shape == (processing_height, processing_width),
                f"depth{depth_id:04d}/{stage_name}: processing dimensions mismatch",
            )
            require(int(labels.max()) <= np.iinfo(np.uint16).max - 1, f"depth{depth_id:04d}/{stage_name}: reserved label used")
        raw_path = manifest_path.parent / stages.get("roberts_regions", {}).get("label_map", "")
        planar_path = manifest_path.parent / stages.get("dav2_planarized", {}).get("label_map", "")
        if raw_path.is_file() and planar_path.is_file():
            raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
            planar = cv2.imread(str(planar_path), cv2.IMREAD_UNCHANGED)
            require(np.array_equal(raw, planar), f"depth{depth_id:04d}: planarization unexpectedly changes labels")
            if raw is not None:
                retained_labels = np.unique(raw)
                retained_labels = retained_labels[retained_labels != 0]
                require(
                    retained_labels.size == retained_region_count,
                    f"depth{depth_id:04d}: retained region count does not match label map",
                )
                require(
                    np.array_equal(
                        retained_labels,
                        np.arange(1, retained_labels.size + 1, dtype=retained_labels.dtype),
                    ),
                    f"depth{depth_id:04d}: retained labels are not compact",
                )
        frame_results.append(
            {
                "depth_id": depth_id,
                "valid": len(errors) == frame_errors_before,
                "manifest": str(manifest_path),
            }
        )
    return {
        "schema": "openmvs_dvp_depth_edge_prior_validation",
        "schema_version": 1,
        "valid": not errors,
        "checks": checks,
        "error_count": len(errors),
        "errors": errors,
        "frame_count": len(frame_results),
        "frames": frame_results,
        "bundle": str(bundle_path),
    }


def main(args: Args) -> None:
    result = validate(args)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output, encoding="utf-8")
    print(output, end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        import tyro
    except ImportError as error:
        raise SystemExit("validate_dvp_depth_edge_prior.py requires tyro") from error
    main(tyro.cli(Args))
