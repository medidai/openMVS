#!/usr/bin/env python3
"""Generate versioned DVP-MVS depth-edge topology priors.

Depth Anything V2 is used only as offline topology evidence. The generated
bundle contains cumulative region-label stages consumed by CUDA PatchMatch;
the monocular depth is never loaded as an OpenMVS depth hypothesis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCHEMA = "openmvs_dvp_depth_edge_prior"
SCHEMA_VERSION = 2
STAGE_NAMES = (
    "roberts_regions",
    "dav2_planarized",
    "eroded",
    "dilated",
    "pixel_reassigned",
)
MAX_REGION_LABEL = np.iinfo(np.uint16).max - 1


@dataclasses.dataclass(frozen=True)
class PriorParameters:
    eta: int = 300
    sigma: float = 0.5
    gamma: float = 1.2
    kappa: float = 0.7
    delta: float = 0.8
    epsilon: float = 0.005
    roberts_threshold: float = 4.0
    ransac_threshold: float = 0.005
    ransac_trials: int = 128
    erosion_kernel: int = 3
    erosion_iterations: int = 1
    connectivity: int = 8
    seed: int = 0


@dataclasses.dataclass
class PlaneFit:
    normal: np.ndarray
    offset: float
    inlier_ratio: float
    mean_inlier_residual: float
    valid: bool

    @classmethod
    def invalid(cls) -> "PlaneFit":
        return cls(np.zeros(3, dtype=np.float64), 0.0, 0.0, float("inf"), False)

    def as_json(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "normal": self.normal.tolist() if self.valid else None,
            "offset": self.offset if self.valid else None,
            "inlier_ratio": self.inlier_ratio if self.valid else None,
            "mean_inlier_residual": self.mean_inlier_residual if self.valid else None,
        }


@dataclasses.dataclass
class Args:
    """Generate priors for an OpenMVS reconstruction input."""

    mapping_path: Path
    images_dir: Path
    output_dir: Path
    depth_anything_repo: Path
    checkpoint: Path
    model_revision: str
    resolution_level: int
    min_resolution: int
    max_resolution: int
    encoder: str = "vits"
    input_size: int = 518
    depth_ids: tuple[int, ...] = ()
    device: str = "cuda"
    force: bool = False
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class ProcessingGeometry:
    requested_resolution_level: int
    effective_resolution_level: int
    min_resolution: int
    max_resolution: int
    prepared_max_resolution: int
    width: int
    height: int
    scale: float

    def as_json(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "scale": self.scale,
            "requested_resolution_level": self.requested_resolution_level,
            "effective_resolution_level": self.effective_resolution_level,
            "min_resolution": self.min_resolution,
            "max_resolution": self.max_resolution,
            "prepared_max_resolution": self.prepared_max_resolution,
            "resize_interpolation": "cv::INTER_AREA",
            "geometry_contract": "OpenMVS_Image_RecomputeMaxResolution_then_ReloadImage",
        }


def compute_openmvs_processing_geometry(
    width: int,
    height: int,
    resolution_level: int,
    min_resolution: int,
    max_resolution: int,
) -> ProcessingGeometry:
    """Mirror Image::RecomputeMaxResolution and Image::ResizeImage."""
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    if resolution_level < 0 or resolution_level > 30:
        raise ValueError("resolution_level must be in [0, 30]")
    if min_resolution < 0 or max_resolution < 0:
        raise ValueError("minimum and maximum resolutions must be non-negative")

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
    # OpenCV's saturate_cast<int> uses nearest-even rounding for positive sizes.
    processing_width = int(np.rint(width * scale))
    processing_height = int(np.rint(height * scale))
    if processing_width <= 0 or processing_height <= 0:
        raise ValueError("OpenMVS image preparation produced an empty image")
    return ProcessingGeometry(
        requested_resolution_level=resolution_level,
        effective_resolution_level=effective_level,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        prepared_max_resolution=prepared_max_resolution,
        width=processing_width,
        height=processing_height,
        scale=scale,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError("Depth Anything V2 returned no finite depth")
    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    if not maximum > minimum:
        raise ValueError("Depth Anything V2 returned a constant depth map")
    result = np.zeros_like(depth, dtype=np.float32)
    result[finite] = (depth[finite] - minimum) / (maximum - minimum)
    return result


def roberts_edges(gray: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    gray_f = np.asarray(gray, dtype=np.float32)
    diagonal = gray_f[:-1, :-1] - gray_f[1:, 1:]
    cross = gray_f[1:, :-1] - gray_f[:-1, 1:]
    magnitude = np.full(gray_f.shape, 255.0, dtype=np.float32)
    magnitude[:-1, :-1] = np.sqrt(diagonal * diagonal + cross * cross)
    return magnitude > threshold, magnitude


def depth_gradient(depth: np.ndarray) -> np.ndarray:
    diagonal = depth[:-1, :-1] - depth[1:, 1:]
    cross = depth[1:, :-1] - depth[:-1, 1:]
    magnitude = np.full(depth.shape, np.inf, dtype=np.float32)
    magnitude[:-1, :-1] = np.sqrt(diagonal * diagonal + cross * cross)
    return magnitude


def connected_regions(
    edge_mask: np.ndarray,
    connectivity: int,
    minimum_region_size: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if minimum_region_size < 0:
        raise ValueError("minimum region size must be non-negative")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (~edge_mask).astype(np.uint8),
        connectivity=connectivity,
        ltype=cv2.CV_32S,
    )
    component_sizes = stats[1:, cv2.CC_STAT_AREA].astype(np.int64, copy=False)
    retained_mask = component_sizes > minimum_region_size
    retained_components = np.flatnonzero(retained_mask) + 1
    if retained_components.size > MAX_REGION_LABEL:
        raise ValueError(
            f"retained region count {retained_components.size} exceeds uint16 contract"
        )

    # DVP only planarizes regions whose area exceeds eta, and the released
    # implementation marks smaller labels ineligible for deformation. Label 0
    # is the equivalent inadmissible state in the OpenMVS anchor-filter contract.
    compact = np.zeros(count, dtype=np.uint16)
    compact[retained_components] = np.arange(
        1, retained_components.size + 1, dtype=np.uint16
    )
    output = compact[labels]
    output[edge_mask] = 0
    discarded_mask = ~retained_mask
    metadata = {
        "policy": "components_with_area_lte_eta_become_boundary",
        "minimum_region_size_exclusive": minimum_region_size,
        "raw_region_count": int(count - 1),
        "retained_region_count": int(retained_components.size),
        "discarded_region_count": int(discarded_mask.sum()),
        "retained_region_pixels": int(component_sizes[retained_mask].sum()),
        "discarded_region_pixels": int(component_sizes[discarded_mask].sum()),
    }
    return output, metadata


def relabel(labels: np.ndarray) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.uint16)
    values = np.unique(labels)
    values = values[values != 0]
    if values.size > MAX_REGION_LABEL:
        raise ValueError(f"region count {values.size} exceeds uint16 contract")
    for new_label, old_label in enumerate(values.tolist(), start=1):
        output[labels == old_label] = new_label
    return output


def normalized_points(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    return np.column_stack(
        (xs.astype(np.float64) / x_scale, ys.astype(np.float64) / y_scale, depth[ys, xs])
    )


def plane_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    normal /= norm
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, points[0]))
    return normal, offset


def fit_plane(
    depth: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    trials: int,
    seed: int,
) -> PlaneFit:
    ys, xs = np.nonzero(mask)
    if ys.size < 3:
        return PlaneFit.invalid()
    points = normalized_points(depth, ys, xs)
    # Model selection is bounded but deterministic. The final inlier ratio is
    # always evaluated over every region pixel.
    if points.shape[0] > 20000:
        indices = np.linspace(0, points.shape[0] - 1, 20000, dtype=np.int64)
        selection_points = points[indices]
    else:
        selection_points = points
    rng = np.random.default_rng(seed)
    best: tuple[int, float, tuple[int, int, int], np.ndarray, float] | None = None
    for _ in range(trials):
        sample = tuple(sorted(rng.choice(selection_points.shape[0], 3, replace=False).tolist()))
        candidate = plane_from_three(selection_points[np.asarray(sample)])
        if candidate is None:
            continue
        normal, offset = candidate
        residuals = np.abs(selection_points @ normal + offset)
        inliers = residuals <= threshold
        count = int(inliers.sum())
        mean = float(residuals[inliers].mean()) if count else float("inf")
        rank = (count, -mean, tuple(-value for value in sample))
        if best is None or rank > (best[0], -best[1], tuple(-value for value in best[2])):
            best = (count, mean, sample, normal, offset)
    if best is None or best[0] < 3:
        return PlaneFit.invalid()
    residuals = np.abs(points @ best[3] + best[4])
    inliers = residuals <= threshold
    if int(inliers.sum()) < 3:
        return PlaneFit.invalid()
    centered = points[inliers] - points[inliers].mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, points[inliers].mean(axis=0)))
    final_residuals = np.abs(points @ normal + offset)
    final_inliers = final_residuals <= threshold
    return PlaneFit(
        normal=normal,
        offset=offset,
        inlier_ratio=float(final_inliers.mean()),
        mean_inlier_residual=float(final_residuals[final_inliers].mean()),
        valid=True,
    )


def fit_region_planes(
    depth: np.ndarray, labels: np.ndarray, parameters: PriorParameters
) -> dict[int, PlaneFit]:
    planes: dict[int, PlaneFit] = {}
    for label in np.unique(labels).tolist():
        if label == 0:
            continue
        mask = labels == label
        if int(mask.sum()) <= parameters.eta:
            planes[int(label)] = PlaneFit.invalid()
            continue
        planes[int(label)] = fit_plane(
            depth,
            mask,
            parameters.ransac_threshold,
            parameters.ransac_trials,
            parameters.seed ^ (int(label) * 0x9E3779B1),
        )
    return planes


def plane_similarity(first: PlaneFit, second: PlaneFit) -> float:
    if not first.valid or not second.valid:
        return float("nan")
    # Eq. 5 is retained literally even though its offset term is
    # counter-intuitive for a quantity named similarity.
    return float(np.dot(first.normal, second.normal) + min(1.0, abs(first.offset - second.offset)))


def erode_regions(
    depth: np.ndarray,
    labels: np.ndarray,
    parent_planes: dict[int, PlaneFit],
    parameters: PriorParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = labels.astype(np.int32).copy()
    gradient = depth_gradient(depth)
    high_gradient = gradient > parameters.epsilon
    kernel = np.ones((parameters.erosion_kernel, parameters.erosion_kernel), dtype=np.uint8)
    next_label = int(output.max()) + 1
    decisions: list[dict[str, Any]] = []
    for label in np.unique(labels).tolist():
        if label == 0:
            continue
        region = labels == label
        if int(region.sum()) <= parameters.eta:
            continue
        eroded = cv2.erode(region.astype(np.uint8), kernel, iterations=parameters.erosion_iterations)
        eroded[high_gradient] = 0
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            eroded, connectivity=parameters.connectivity
        )
        component_ids = sorted(
            range(1, component_count), key=lambda component: (-int(stats[component, cv2.CC_STAT_AREA]), component)
        )
        if len(component_ids) < 2:
            continue
        component_ids = component_ids[:2]
        child_planes = [
            fit_plane(
                depth,
                components == component,
                parameters.ransac_threshold,
                parameters.ransac_trials,
                parameters.seed ^ (int(label) * 0x85EBCA77) ^ component,
            )
            for component in component_ids
        ]
        parent = parent_planes.get(int(label), PlaneFit.invalid())
        similarity = plane_similarity(child_planes[0], child_planes[1])
        ratio_gain = (
            (child_planes[0].inlier_ratio + child_planes[1].inlier_ratio)
            / (2.0 * parent.inlier_ratio)
            if parent.valid and parent.inlier_ratio > 0
            else float("nan")
        )
        accepted = bool(
            all(plane.valid for plane in child_planes)
            and np.isfinite(similarity)
            and np.isfinite(ratio_gain)
            and similarity <= parameters.sigma
            and ratio_gain >= parameters.gamma
        )
        decisions.append(
            {
                "parent_region": int(label),
                "accepted": accepted,
                "similarity_eq5": similarity if np.isfinite(similarity) else None,
                "inlier_ratio_gain_eq4": ratio_gain if np.isfinite(ratio_gain) else None,
                "child_sizes": [int(stats[component, cv2.CC_STAT_AREA]) for component in component_ids],
            }
        )
        if not accepted:
            continue
        output[region] = 0
        for component in component_ids:
            if next_label > MAX_REGION_LABEL:
                raise ValueError("erosion split exceeds uint16 region-label contract")
            output[components == component] = next_label
            next_label += 1
    return relabel(output), {"split_decisions": decisions}


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        self.parent[high] = low


def adjacent_region_pairs(labels: np.ndarray) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0), (2, 2), (2, -2)):
        y0a, y1a = max(0, -dy), labels.shape[0] - max(0, dy)
        x0a, x1a = max(0, -dx), labels.shape[1] - max(0, dx)
        first = labels[y0a:y1a, x0a:x1a]
        second = labels[y0a + dy : y1a + dy, x0a + dx : x1a + dx]
        mask = (first != 0) & (second != 0) & (first != second)
        if not mask.any():
            continue
        for left, right in np.unique(np.column_stack((first[mask], second[mask])), axis=0).tolist():
            pairs.add(tuple(sorted((int(left), int(right)))))
    return sorted(pairs)


def dilate_regions(
    labels: np.ndarray,
    planes: dict[int, PlaneFit],
    parameters: PriorParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    region_ids = [int(value) for value in np.unique(labels).tolist() if value != 0]
    union = UnionFind(region_ids)
    decisions: list[dict[str, Any]] = []
    for first, second in adjacent_region_pairs(labels):
        first_plane = planes.get(first, PlaneFit.invalid())
        second_plane = planes.get(second, PlaneFit.invalid())
        similarity = plane_similarity(first_plane, second_plane)
        accepted = bool(
            first_plane.valid
            and second_plane.valid
            and first_plane.inlier_ratio >= parameters.kappa
            and second_plane.inlier_ratio >= parameters.kappa
            and np.isfinite(similarity)
            and similarity >= parameters.sigma
        )
        decisions.append(
            {
                "regions": [first, second],
                "accepted": accepted,
                "similarity_eq5": similarity if np.isfinite(similarity) else None,
                "inlier_ratios": [first_plane.inlier_ratio, second_plane.inlier_ratio]
                if first_plane.valid and second_plane.valid
                else None,
            }
        )
        if accepted:
            union.union(first, second)
    output = labels.astype(np.int32).copy()
    for region in region_ids:
        output[labels == region] = union.find(region)
    return relabel(output), {"merge_decisions": decisions}


def reassign_boundary_pixels(
    depth: np.ndarray,
    labels: np.ndarray,
    planes: dict[int, PlaneFit],
    parameters: PriorParameters,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = labels.copy()
    boundary = labels == 0
    assigned = 0
    height, width = labels.shape
    for region in [int(value) for value in np.unique(labels).tolist() if value != 0]:
        plane = planes.get(region, PlaneFit.invalid())
        if not plane.valid or plane.inlier_ratio < parameters.kappa:
            continue
        neighbor = cv2.dilate((labels == region).astype(np.uint8), np.ones((3, 3), np.uint8)) != 0
        ys, xs = np.nonzero(boundary & neighbor & (output == 0))
        if ys.size == 0:
            continue
        points = normalized_points(depth, ys, xs)
        distances = np.abs(points @ plane.normal + plane.offset)
        accepted = distances <= parameters.delta
        output[ys[accepted], xs[accepted]] = region
        assigned += int(accepted.sum())
    return output, {"assigned_boundary_pixels": assigned, "remaining_boundary_pixels": int((output == 0).sum())}


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    values = labels.astype(np.uint32)
    red = ((values * 37 + 53) % 223 + 24).astype(np.uint8)
    green = ((values * 73 + 97) % 223 + 24).astype(np.uint8)
    blue = ((values * 109 + 19) % 223 + 24).astype(np.uint8)
    color = np.dstack((blue, green, red))
    color[values == 0] = 0
    return color


def save_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")


def stage_stats(labels: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    region_counts = counts[values != 0]
    return {
        "region_count": int((values != 0).sum()),
        "boundary_pixels": int(counts[values == 0][0]) if (values == 0).any() else 0,
        "covered_pixels": int(region_counts.sum()),
        "median_region_size": float(np.median(region_counts)) if region_counts.size else None,
        "maximum_region_size": int(region_counts.max()) if region_counts.size else None,
    }


def generate_topology(
    image_bgr: np.ndarray, depth_raw: np.ndarray, parameters: PriorParameters
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("expected a BGR uint8 image")
    depth = normalize_depth(depth_raw)
    if depth.shape != image_bgr.shape[:2]:
        depth = cv2.resize(depth, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges, edge_magnitude = roberts_edges(gray, parameters.roberts_threshold)
    raw_labels, component_filtering = connected_regions(
        edges, parameters.connectivity, parameters.eta
    )
    raw_planes = fit_region_planes(depth, raw_labels, parameters)
    planarized_labels = raw_labels.copy()
    eroded_labels, erosion_meta = erode_regions(
        depth, planarized_labels, raw_planes, parameters
    )
    eroded_planes = fit_region_planes(depth, eroded_labels, parameters)
    dilated_labels, dilation_meta = dilate_regions(eroded_labels, eroded_planes, parameters)
    dilated_planes = fit_region_planes(depth, dilated_labels, parameters)
    reassigned_labels, reassignment_meta = reassign_boundary_pixels(
        depth, dilated_labels, dilated_planes, parameters
    )
    stages = {
        "roberts_regions": raw_labels,
        "dav2_planarized": planarized_labels,
        "eroded": eroded_labels,
        "dilated": dilated_labels,
        "pixel_reassigned": reassigned_labels,
    }
    metadata = {
        "normalized_depth": depth,
        "roberts_magnitude": edge_magnitude,
        "roberts_edges": edges,
        "component_filtering": component_filtering,
        "planes": {
            "dav2_planarized": {str(key): value.as_json() for key, value in raw_planes.items()},
            "eroded": {str(key): value.as_json() for key, value in eroded_planes.items()},
            "dilated": {str(key): value.as_json() for key, value in dilated_planes.items()},
        },
        "erosion": erosion_meta,
        "dilation": dilation_meta,
        "pixel_reassignment": reassignment_meta,
    }
    return stages, metadata


def official_repo_revision(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def load_depth_anything(args: Args) -> tuple[Any, dict[str, Any]]:
    revision = official_repo_revision(args.depth_anything_repo)
    if revision != args.model_revision:
        raise ValueError(f"Depth Anything V2 repo is {revision}, expected {args.model_revision}")
    if args.encoder not in {"vits", "vitb", "vitl", "vitg"}:
        raise ValueError(f"unsupported encoder {args.encoder!r}")
    import torch

    sys.path.insert(0, str(args.depth_anything_repo))
    from depth_anything_v2.dpt import DepthAnythingV2

    configurations = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**configurations[args.encoder])
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(args.device).eval()
    provenance = {
        "repository": "https://github.com/DepthAnything/Depth-Anything-V2",
        "revision": revision,
        "encoder": args.encoder,
        "checkpoint_name": args.checkpoint.name,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": args.device,
        "input_size": args.input_size,
        "preprocessing": "official_DepthAnythingV2.infer_image",
        "output_normalization": "per_frame_finite_minmax_to_[0,1]",
    }
    return model, provenance


def parse_depth_id(value: str) -> int:
    stem = Path(value).stem
    if not stem.startswith("depth") or not stem[5:].isdigit():
        raise ValueError(f"invalid depth-map mapping value {value!r}")
    return int(stem[5:])


def generate_frame(
    image_path: Path,
    depth_id: int,
    output_dir: Path,
    model: Any,
    model_provenance: dict[str, Any],
    parameters: PriorParameters,
    resolution_level: int,
    min_resolution: int,
    max_resolution: int,
    input_size: int,
    force: bool,
) -> dict[str, Any]:
    frame_dir = output_dir / "frames" / f"depth{depth_id:04d}"
    manifest_path = frame_dir / "manifest.json"
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_image is None:
        raise OSError(f"failed to load {image_path}")
    raw_height, raw_width = raw_image.shape[:2]
    source_sha256 = sha256_file(image_path)
    processing = compute_openmvs_processing_geometry(
        raw_width,
        raw_height,
        resolution_level,
        min_resolution,
        max_resolution,
    )
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete") is True
            and manifest.get("schema") == SCHEMA
            and manifest.get("schema_version") == SCHEMA_VERSION
            and manifest.get("source_image", {}).get("sha256") == source_sha256
            and manifest.get("processing_image") == processing.as_json()
        ):
            return manifest
        raise FileExistsError(
            f"prior at {frame_dir} does not match this source/preparation; "
            "use a fresh output directory or pass --force"
        )
    frame_dir.mkdir(parents=True, exist_ok=True)
    image = raw_image
    if (processing.width, processing.height) != (raw_width, raw_height):
        image = cv2.resize(
            raw_image,
            (processing.width, processing.height),
            interpolation=cv2.INTER_AREA,
        )
    depth_raw = model.infer_image(image, input_size)
    if depth_raw.shape[:2] != image.shape[:2]:
        raise ValueError(
            "Depth Anything V2 output geometry does not match its prepared input: "
            f"{depth_raw.shape[:2]} != {image.shape[:2]}"
        )
    stages, metadata = generate_topology(image, depth_raw, parameters)

    np.save(frame_dir / "dav2_depth_raw.npy", np.asarray(depth_raw, dtype=np.float32), allow_pickle=False)
    np.save(frame_dir / "dav2_depth_normalized.npy", metadata["normalized_depth"], allow_pickle=False)
    save_png(frame_dir / "dav2_depth_preview.png", np.round(metadata["normalized_depth"] * 65535).astype(np.uint16))
    save_png(frame_dir / "roberts_magnitude.png", np.clip(metadata["roberts_magnitude"], 0, 255).astype(np.uint8))
    save_png(frame_dir / "roberts_edges.png", metadata["roberts_edges"].astype(np.uint8) * 255)

    stage_manifest: dict[str, Any] = {}
    for stage_name, labels in stages.items():
        label_name = f"regions_{stage_name}.png"
        preview_name = f"regions_{stage_name}_preview.png"
        overlay_name = f"regions_{stage_name}_overlay.jpg"
        save_png(frame_dir / label_name, labels.astype(np.uint16))
        color = colorize_labels(labels)
        save_png(frame_dir / preview_name, color)
        overlay = cv2.addWeighted(image, 0.55, color, 0.45, 0.0)
        overlay[labels == 0] = (0, 0, 0)
        save_png(frame_dir / overlay_name, overlay)
        stage_manifest[stage_name] = {
            "label_map": label_name,
            "sha256": sha256_file(frame_dir / label_name),
            "preview": preview_name,
            "overlay": overlay_name,
            "stats": stage_stats(labels),
        }
    stable_json(
        frame_dir / "topology_diagnostics.json",
        {
            "component_filtering": metadata["component_filtering"],
            "planes": metadata["planes"],
            "erosion": metadata["erosion"],
            "dilation": metadata["dilation"],
            "pixel_reassignment": metadata["pixel_reassignment"],
        },
    )
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "claim": "paper_mechanics_complete_openmvs",
        "depth_id": depth_id,
        "source_image": {
            "name": image_path.name,
            "width": raw_width,
            "height": raw_height,
            "sha256": source_sha256,
        },
        "processing_image": processing.as_json(),
        "model": model_provenance,
        "parameters": dataclasses.asdict(parameters),
        "coordinate_contract": "processing_image_normalized_(x/(w-1),y/(h-1),per_frame_DAV2_depth)",
        "paper_defaults": {"eta": 300, "sigma": 0.5, "gamma": 1.2, "kappa": 0.7, "delta": 0.8, "epsilon": 0.005},
        "region_component_filtering": metadata["component_filtering"],
        "compatibility_behavior": [
            "official_DAV2_vits_selected_because_paper_does_not_name_encoder_or_weights",
            "Roberts_is_applied_at_OpenMVS_prepared_resolution_with_released_code_threshold_4",
            "components_with_area_lte_eta_are_encoded_as_boundary_matching_the_released_positive_label_gate",
            "RANSAC_threshold_0.005_normalized_units_is_not_specified_by_paper",
            "erosion_uses_3x3_one_iteration_and_two_largest_components",
            "Eq5_similarity_is_retained_literally_despite_counter_intuitive_offset_term",
            "dilation_uses_one_pixel_region_adjacency_and_deterministic_union",
            "pixel_reassignment_is_one_deterministic_pass",
        ],
        "direct_depth_use": "forbidden_topology_only",
        "stages": stage_manifest,
        "diagnostics": "topology_diagnostics.json",
    }
    stable_json(manifest_path, manifest)
    return manifest


def main(args: Args) -> None:
    np.random.seed(args.seed)
    if not args.mapping_path.is_file():
        raise FileNotFoundError(args.mapping_path)
    if not args.images_dir.is_dir():
        raise NotADirectoryError(args.images_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.mapping_path.read_text(encoding="utf-8"))
    requested = set(args.depth_ids)
    entries = sorted(
        ((parse_depth_id(depth_name), image_name) for image_name, depth_name in mapping.items()),
        key=lambda item: item[0],
    )
    if requested:
        entries = [entry for entry in entries if entry[0] in requested]
        missing = requested - {entry[0] for entry in entries}
        if missing:
            raise ValueError(f"depth IDs absent from mapping: {sorted(missing)}")
    model, model_provenance = load_depth_anything(args)
    parameters = PriorParameters(seed=args.seed)
    frames: list[dict[str, Any]] = []
    for index, (depth_id, image_name) in enumerate(entries, start=1):
        print(f"[{index}/{len(entries)}] depth{depth_id:04d} <- {image_name}", flush=True)
        manifest = generate_frame(
            args.images_dir / image_name,
            depth_id,
            args.output_dir,
            model,
            model_provenance,
            parameters,
            args.resolution_level,
            args.min_resolution,
            args.max_resolution,
            args.input_size,
            args.force,
        )
        frames.append(
            {
                "depth_id": depth_id,
                "manifest": f"frames/depth{depth_id:04d}/manifest.json",
                "manifest_sha256": sha256_file(args.output_dir / f"frames/depth{depth_id:04d}/manifest.json"),
                "source_sha256": manifest["source_image"]["sha256"],
            }
        )
    stable_json(
        args.output_dir / "bundle.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "mapping_path": str(args.mapping_path),
            "mapping_sha256": sha256_file(args.mapping_path),
            "model": model_provenance,
            "parameters": dataclasses.asdict(parameters),
            "image_preparation": {
                "resolution_level": args.resolution_level,
                "min_resolution": args.min_resolution,
                "max_resolution": args.max_resolution,
            },
            "frame_count": len(frames),
            "frames": frames,
        },
    )


if __name__ == "__main__":
    try:
        import tyro
    except ImportError as error:
        raise SystemExit("generate_dvp_depth_edge_prior.py requires tyro") from error
    main(tyro.cli(Args))
